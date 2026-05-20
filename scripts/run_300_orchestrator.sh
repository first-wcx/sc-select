#!/bin/bash
# Master orchestrator for 300-prompt SC-Select experiment
# Runs all steps sequentially on server_004
# Usage: bash run_300_orchestrator.sh

set -e
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=0
PYTHON=/root/miniconda3/bin/python
PROJECT=/root/autodl-tmp/projects/SCAlign
cd $PROJECT

echo "=== SC-Select 300-prompt Pipeline ==="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Step 1: SDXL Generation (already running or complete)
echo "=== Step 1: SDXL Generation ==="
IMG_COUNT=$(find outputs/images_300 -name '*.png' 2>/dev/null | wc -l)
if [ "$IMG_COUNT" -ge 1200 ]; then
    echo "Already complete: $IMG_COUNT images"
else
    echo "Images so far: $IMG_COUNT / 1200"
    echo "Checking if generation is still running..."
    if pgrep -f "run_sdxl_300" > /dev/null; then
        echo "Generation still running. Waiting..."
        while pgrep -f "run_sdxl_300" > /dev/null; do
            sleep 60
            COUNT=$(find outputs/images_300 -name '*.png' 2>/dev/null | wc -l)
            echo "  $(date '+%H:%M:%S') - $COUNT images"
        done
    fi
    echo "Generation complete: $(find outputs/images_300 -name '*.png' | wc -l) images"
fi

# Step 2: Qwen Parsing
echo ""
echo "=== Step 2: Qwen2.5-VL Perception Graph Extraction ==="
PG_COUNT=$(find outputs/perception_graphs_300 -name '*.json' 2>/dev/null | wc -l)
if [ "$PG_COUNT" -ge 1200 ]; then
    echo "Already complete: $PG_COUNT perception graphs"
else
    echo "Running Qwen parsing..."
    $PYTHON scripts/run_qwen_parse_300.py
fi

# Step 3: CSR Scoring
echo ""
echo "=== Step 3: CSR Scoring ==="
if [ -f "outputs/csr_scores_300/csr_results.jsonl" ]; then
    CSR_COUNT=$(wc -l < outputs/csr_scores_300/csr_results.jsonl)
    echo "Already complete: $CSR_COUNT results"
else
    echo "Running CSR scoring..."
    $PYTHON scripts/run_csr_score_300.py
fi

# Step 4: CLIPScore Baseline
echo ""
echo "=== Step 4: CLIPScore Baseline ==="
echo "Running CLIPScore..."
$PYTHON scripts/10_reward_scoring.py --dataset self300 --metrics clipscore --project-root $PROJECT

# Step 5: Reward Model Baselines (if available)
echo ""
echo "=== Step 5: Reward Models ==="
$PYTHON scripts/10_reward_scoring.py --dataset self300 --metrics hpsv2 --project-root $PROJECT 2>/dev/null || echo "HPSv2 failed, skipping"

# Step 6: Summary
echo ""
echo "=== Pipeline Complete ==="
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results:"
echo "  Images: $(find outputs/images_300 -name '*.png' 2>/dev/null | wc -l)"
echo "  Perception graphs: $(find outputs/perception_graphs_300 -name '*.json' 2>/dev/null | wc -l)"
echo "  CSR results: $(wc -l < outputs/csr_scores_300/csr_results.jsonl 2>/dev/null || echo 0)"
