# SC-Select

Semantic Contract-Guided Selection for Instruction-Adherent Text-to-Image Generation.

This repository contains the public code, schemas, prompt/contract data, summarized results, and manuscript materials for the SC-Select paper. SC-Select is a training-free post-generation selection method: it decomposes a text prompt into a semantic contract, parses each generated candidate image into a perception graph, scores candidates with Contract Satisfaction Rate (CSR), and selects the candidate with the highest CSR-All.

## Repository Contents

- `src/`: core CSR scoring and selection baselines.
- `scripts/`: experiment, scoring, baseline, statistics, human-evaluation, and table/figure generation scripts.
- `configs/`: JSON schemas for semantic contracts and perception graphs.
- `data/`: reviewed prompts, semantic contracts, and public benchmark subset metadata.
- `results/tables/`: summarized result tables used by the manuscript.
- `results/stats/`: bootstrap/sign-test outputs.
- `results/selections/`: selected-output metadata and baseline summaries.
- `paper/`: manuscript Markdown, revised sections, references, and paper tables.
- `paper/figures/`: final manuscript figures.
- `results/final_tables/`: additional final tables and ablations.
- `provenance/`: result provenance and audit notes.

Generated images, model weights, private server logs, credentials, raw remote audit folders, submission-system files, and large intermediate artifacts are intentionally excluded.

## Main Result Snapshot

On the reviewed 300-prompt SDXL benchmark with 1200 candidates, SC-Select improves CSR-All to `0.5122`, compared with `0.3819` for Fixed Seed-0, `0.3779` for CLIPScore-Select, `0.3686` for ImageReward-Select, `0.3695` for PickScore-Select, and `0.3808` for HPSv2-Select.

See `RESULT_PROVENANCE_TABLE.md` and `provenance/` for the source files and scripts behind the reported numbers.

## Environment

CPU utilities:

```powershell
python -m pip install -r requirements/requirements_base.txt
```

GPU/model-dependent experiments require separate environments for generation, VLM parsing, and reward-model scoring. See `requirements/environment_notes.md` and `requirements/requirements_gpu.txt`.

## Minimal Usage

Run selection baselines from an existing candidate-score file. Full candidate-score files for large image pools are not included in this lightweight release, but the command below shows the expected interface:

```powershell
python scripts/07_run_baselines.py `
  --scores path/to/candidate_scores_all_rewards.csv `
  --out-dir results/selections/example
```

Run table generation from available summarized artifacts:

```powershell
python scripts/12_make_tables.py
```

Run figure generation:

```powershell
python scripts/13_make_figures.py
```

Some scripts expect generated images, perception graphs, candidate-score files, model caches, or remote GPU paths that are not included in this public repository.

## Data Notes

The included `data/` files provide reviewed prompt and contract metadata. Large generated candidate images and raw model outputs are not included because of size, licensing, and reproducibility constraints. The public repository instead includes schemas, scripts, seeds/metadata where available, summarized score tables, and selected-output metadata.

## Manuscript

The main Markdown manuscript is in `paper/manuscript_sc_select_20260514.md`. The final submission PDF/DOCX files are not included in the public repository; they remain local submission artifacts.

## License

License is not finalized in this prepared public snapshot. Add the intended open-source license before making the GitHub repository public.
