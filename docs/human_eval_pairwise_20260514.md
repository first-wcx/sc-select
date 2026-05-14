# Human Evaluation Pairwise Package 2026-05-14

## Status

One filled response sheet has been received and analyzed. See `docs/human_eval_results_20260514.md`.

## Package

- Directory: `outputs/human_eval/pairwise_20260514`
- Viewer: `annotation_viewer.html`
- Response sheet: `annotation_response_template.csv`
- Blinded task metadata: `pairwise_tasks_blinded.csv`
- Hidden answer key: `pairwise_answer_key.csv`
- Analysis entry point: `scripts/15_analyze_pairwise_human_eval.py`

## Design

- 120 pairwise tasks.
- 60 tasks from the reviewed 300-prompt SDXL experiment.
- 60 tasks from the FLUX stratified-60 experiment.
- Five baseline comparisons per dataset: Fixed Seed-0, CLIPScore, ImageReward, PickScore, and HPSv2.
- 12 prompts per baseline comparison.
- 215 local image copies were fetched for the blinded HTML viewer.

## Annotator Instructions

Annotators should choose `left`, `right`, or `tie` for each task. The decision criterion is prompt faithfulness: object presence, attribute binding, relations, counting, and text rendering. Visual appeal should not be the primary criterion.

## Analysis Command

```bash
python scripts/15_analyze_pairwise_human_eval.py \
  --answers outputs/human_eval/pairwise_20260514/annotation_response_filled.csv \
  --key outputs/human_eval/pairwise_20260514/pairwise_answer_key.csv \
  --output outputs/human_eval/pairwise_20260514/human_eval_summary.csv
```
