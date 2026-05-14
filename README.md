# SC-Select

SC-Select is a semantic-constraint-guided best-of-N selection framework for text-to-image generation. It parses candidate images into perception graphs, scores them against prompt-derived semantic contracts, and selects the candidate that best satisfies object, attribute, relation, counting, and text constraints.

This repository contains the public code, schemas, prompt/contract files, and lightweight result summaries used for the accompanying small-paper experiments. It intentionally does not include generated image corpora, model weights, server credentials, or local cache files.

## Repository Layout

```text
configs/        JSON schemas for semantic contracts and perception graphs
data/           Public prompt and contract JSONL files
docs/           Experiment summaries and registry
paper/tables/   CSV tables used by the paper draft
requirements/   Base and GPU dependency notes
results/        Lightweight result tables, stats, and human-eval summaries
scripts/        Data construction, scoring, selection, evaluation, and GPU run scripts
src/            Reusable CSR scoring and baseline selector modules
```

## Core Workflow

1. Generate multiple images per prompt with a text-to-image model.
2. Parse each image with a VLM parser such as Qwen2.5-VL or InternVL2.5.
3. Score each perception graph against the semantic contract.
4. Select the best candidate per prompt with CSR-Select.
5. Compare against Fixed Seed-0, Random Select, CLIPScore, ImageReward, PickScore, and HPSv2 where scores are available.

## Quick Commands

Run baseline selectors on an existing candidate score CSV:

```bash
python scripts/07_run_baselines.py \
  --scores path/to/candidate_scores_all_rewards.csv \
  --out-dir outputs/selections/example
```

Run paired bootstrap statistics from selected JSONL files:

```bash
python scripts/08_bootstrap_stats.py \
  --selected-a outputs/selections/example/csr_select_selected.jsonl \
  --selected-b outputs/selections/example/clipscore_select_selected.jsonl \
  --output outputs/stats/example.csv \
  --dataset example \
  --method-a csr_select \
  --method-b clipscore_select
```

Analyze a filled pairwise human-evaluation sheet:

```bash
python scripts/15_analyze_pairwise_human_eval.py \
  --answers outputs/human_eval/annotation_response_filled.csv \
  --key outputs/human_eval/pairwise_answer_key.csv \
  --output outputs/human_eval/human_eval_summary.csv
```

## Data

The main reviewed self-built prompt set is:

- `data/prompts/scselect_complex_300.jsonl`
- `data/contracts/scselect_complex_300_contracts.jsonl`

The FLUX stratified subset is:

- `data/prompts/scselect_complex_stratified_60.jsonl`
- `data/contracts/scselect_complex_stratified_60_contracts.jsonl`

The auxiliary T2I-CompBench subset is:

- `data/benchmarks/t2i_compbench_subset_60_prompts.jsonl`
- `data/benchmarks/t2i_compbench_subset_60_contracts.jsonl`

The T2I-CompBench contracts are automatically derived from formulaic public prompts and should be treated as an auxiliary stress-test subset rather than a manually curated benchmark.

## Models

Model weights are not stored in this repository. The experiments use public models loaded through their normal package or Hugging Face interfaces, including SDXL, FLUX.1-schnell, Qwen2.5-VL, InternVL2.5, CLIP, ImageReward, PickScore, and HPSv2.

## Results

Lightweight CSV summaries are under `results/` and `paper/tables/`. The full generated image pools and model caches are excluded because they are large and reproducible from the provided scripts and prompts.

Key summary documents:

- `docs/RESULTS_SUMMARY.md`
- `docs/experiment_registry.csv`
- `docs/human_eval_results_20260514.md`
- `docs/t2i_compbench_extension_status_20260514.md`
- `docs/parser_robustness_status_20260514.md`

## Safety and Privacy

This public release excludes:

- SSH credentials and server inventory secrets
- DPAPI-encrypted local secret files
- generated image folders
- model checkpoints and cache directories
- unpublished source documents and local session logs

