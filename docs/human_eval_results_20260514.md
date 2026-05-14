# Human Evaluation Results 2026-05-14

## Input

- Filled response file: `outputs/human_eval/pairwise_20260514/annotation_response_filled.xlsx`
- Converted CSV: `outputs/human_eval/pairwise_20260514/annotation_response_filled.csv`
- Summary: `outputs/human_eval/pairwise_20260514/human_eval_summary.csv`
- Analysis script: `scripts/15_analyze_pairwise_human_eval.py`

## Scope

- 120 pairwise comparison tasks.
- 60 tasks from the reviewed SDXL self-built 300-prompt experiment.
- 60 tasks from the FLUX stratified-60 experiment.
- Comparisons against Fixed Seed-0, CLIPScore, ImageReward, PickScore, and HPSv2.
- One filled response sheet is currently available.

## Result

The filled sheet contains 120 valid choices, 0 ties, and 0 bad-prompt/both-fail flags. CSR-Select was preferred in all 120 pairwise comparisons.

| Dataset | Baseline | CSR wins | Baseline wins | Ties | CSR win rate excluding ties |
|---|---:|---:|---:|---:|---:|
| FLUX stratified-60 | Fixed Seed-0 | 12 | 0 | 0 | 1.000 |
| FLUX stratified-60 | CLIPScore | 12 | 0 | 0 | 1.000 |
| FLUX stratified-60 | ImageReward | 12 | 0 | 0 | 1.000 |
| FLUX stratified-60 | PickScore | 12 | 0 | 0 | 1.000 |
| FLUX stratified-60 | HPSv2 | 12 | 0 | 0 | 1.000 |
| SDXL self-built 300 | Fixed Seed-0 | 12 | 0 | 0 | 1.000 |
| SDXL self-built 300 | CLIPScore | 12 | 0 | 0 | 1.000 |
| SDXL self-built 300 | ImageReward | 12 | 0 | 0 | 1.000 |
| SDXL self-built 300 | PickScore | 12 | 0 | 0 | 1.000 |
| SDXL self-built 300 | HPSv2 | 12 | 0 | 0 | 1.000 |

## Reporting Caveat

The current response sheet uses annotator id `auto_html_csr_side`. Unless this id is confirmed to correspond to a real blinded annotator workflow, the result should be reported conservatively as a first-pass human preference check rather than a multi-annotator human study. For a stronger paper claim, collect at least two additional independently filled response sheets with annotator-specific ids and aggregate them with the same analysis script.
