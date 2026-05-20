#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/SCAlign
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com

PY=/root/miniconda3/bin/python
LOGDIR=outputs/parser_robustness/internvl2_5_4b_flux60/logs
mkdir -p "$LOGDIR"

while pgrep -f "remote_t2i_compbench_pipeline.sh" >/dev/null; do
  echo "$(date +%F_%T) waiting for T2I pipeline" >> "$LOGDIR/run_after_t2i.log"
  sleep 60
done
while pgrep -f "remote_internvl_download.py" >/dev/null; do
  echo "$(date +%F_%T) waiting for InternVL download" >> "$LOGDIR/run_after_t2i.log"
  sleep 60
done

echo "$(date +%F_%T) starting InternVL parser" >> "$LOGDIR/run_after_t2i.log"
"$PY" scripts/run_internvl_parse_generic.py \
  --image-dir outputs/flux/self300_stratified_60/images \
  --out-dir outputs/parser_robustness/internvl2_5_4b_flux60/perception_graphs \
  --model-id OpenGVLab/InternVL2_5-4B \
  --parser-name InternVL2.5-4B \
  --max-new-tokens 1024 \
  > "$LOGDIR/parse.log" 2>&1
"$PY" scripts/run_csr_score_generic.py \
  --contract-file data/scselect_complex_300_contracts.jsonl \
  --prompt-file data/scselect_complex_300.jsonl \
  --perception-dir outputs/parser_robustness/internvl2_5_4b_flux60/perception_graphs \
  --image-dir outputs/flux/self300_stratified_60/images \
  --out-dir outputs/parser_robustness/internvl2_5_4b_flux60/csr_scores \
  > "$LOGDIR/csr_score.log" 2>&1
"$PY" scripts/07_run_baselines.py \
  --scores outputs/parser_robustness/internvl2_5_4b_flux60/csr_scores/candidate_scores.csv \
  --out-dir outputs/selections/parser_robustness_internvl_flux60_20260514 \
  > "$LOGDIR/baselines.log" 2>&1
echo "$(date +%F_%T) InternVL parser robustness complete" >> "$LOGDIR/run_after_t2i.log"
