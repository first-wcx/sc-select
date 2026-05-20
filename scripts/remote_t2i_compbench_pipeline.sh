#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/SCAlign
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com

PY=/root/miniconda3/bin/python
LOGDIR=outputs/t2i_compbench_sdxl_60/logs
mkdir -p "$LOGDIR"

echo "$(date +%F_%T) starting T2I-CompBench SDXL subset" >> "$LOGDIR/pipeline.log"
"$PY" scripts/run_sdxl_generic.py \
  --prompt-file data/benchmarks/t2i_compbench_subset_60_prompts.jsonl \
  --output-dir outputs/t2i_compbench_sdxl_60/images \
  --log-file "$LOGDIR/generation.log"
"$PY" scripts/run_qwen_parse_generic.py \
  --image-dir outputs/t2i_compbench_sdxl_60/images \
  --out-dir outputs/t2i_compbench_sdxl_60/perception_graphs \
  > "$LOGDIR/qwen_parse.log" 2>&1
"$PY" scripts/run_csr_score_generic.py \
  --contract-file data/benchmarks/t2i_compbench_subset_60_contracts.jsonl \
  --prompt-file data/benchmarks/t2i_compbench_subset_60_prompts.jsonl \
  --perception-dir outputs/t2i_compbench_sdxl_60/perception_graphs \
  --image-dir outputs/t2i_compbench_sdxl_60/images \
  --out-dir outputs/t2i_compbench_sdxl_60/csr_scores \
  > "$LOGDIR/csr_score.log" 2>&1
"$PY" scripts/run_clipscore_candidates.py \
  --scores outputs/t2i_compbench_sdxl_60/csr_scores/candidate_scores.csv \
  --output outputs/t2i_compbench_sdxl_60/csr_scores/candidate_scores_clipscore.csv \
  --project-root . \
  > "$LOGDIR/clipscore.log" 2>&1
"$PY" scripts/run_reward_candidates.py \
  --scores outputs/t2i_compbench_sdxl_60/csr_scores/candidate_scores_clipscore.csv \
  --output outputs/t2i_compbench_sdxl_60/csr_scores/candidate_scores_pickscore.csv \
  --project-root . \
  --metrics pickscore \
  > "$LOGDIR/pickscore.log" 2>&1
"$PY" scripts/run_reward_candidates.py \
  --scores outputs/t2i_compbench_sdxl_60/csr_scores/candidate_scores_pickscore.csv \
  --output outputs/t2i_compbench_sdxl_60/csr_scores/candidate_scores_hpsv2.csv \
  --project-root . \
  --metrics hpsv2 \
  > "$LOGDIR/hpsv2.log" 2>&1
"$PY" scripts/run_reward_candidates.py \
  --scores outputs/t2i_compbench_sdxl_60/csr_scores/candidate_scores_hpsv2.csv \
  --output outputs/t2i_compbench_sdxl_60/csr_scores/candidate_scores_all_rewards.csv \
  --project-root . \
  --metrics image_reward \
  > "$LOGDIR/image_reward.log" 2>&1
"$PY" scripts/07_run_baselines.py \
  --scores outputs/t2i_compbench_sdxl_60/csr_scores/candidate_scores_all_rewards.csv \
  --out-dir outputs/selections/t2i_compbench_sdxl_60_all_rewards_20260514 \
  > "$LOGDIR/baselines.log" 2>&1
echo "$(date +%F_%T) T2I pipeline complete" >> "$LOGDIR/pipeline.log"
