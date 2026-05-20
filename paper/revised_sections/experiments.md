# Experiments

## Current audited experiments

The currently usable results come from audited AutoDL / SeetaCloud server artifacts:

- self-built 50-prompt SDXL experiment on `server_004`;
- self-built stratified-60 FLUX.1-schnell subset on `server_004`;
- GenEval SDXL CSR and official detector results on `server_002` and `server_004`;
- GenEval FLUX.1-schnell CSR results on `server_001`.

## Datasets

The self-built resources include the original 50-prompt set, a 300-prompt reviewed set in `data/prompts/scselect_complex_300.jsonl`, and a stratified-60 subset with 10 prompts from each category. The 300-prompt set covers attribute binding, spatial/action relations, counting, text rendering, and complex composition; all prompt/contract rows are marked `checked:true` after manual review.

GenEval is used as the current public benchmark. It provides object-focused prompts and official detector-based evaluation categories. An auxiliary T2I-CompBench subset is also used as a public compositional stress test. The subset contains 60 formulaic prompts: 20 color-binding prompts, 20 spatial-relation prompts, and 20 numeracy prompts. Semantic contracts for this subset are derived automatically from the public prompt templates and are therefore reported separately from the manually reviewed self-built benchmark.

## Baselines

Baselines completed from existing score files:

- Fixed Seed-0;
- Random Select with five repeated seeds;
- CLIPScore-Select for the self-built 50-prompt SDXL rerun, the GenEval SDXL candidate pool, and the FLUX self-built stratified-60 subset;
- ImageReward-Select for the self-built 300-prompt reviewed set and the FLUX self-built stratified-60 subset;
- PickScore-Select for the self-built 300-prompt reviewed set and the FLUX self-built stratified-60 subset;
- HPSv2-Select for the self-built 300-prompt reviewed set and the FLUX self-built stratified-60 subset;
- CSR-Select.

Baseline not included in the completed result tables:

- VLM-Direct-Select.

VLM-Direct-Select is left for future work because this branch was not run in the completed audited experiments.

For parser robustness, the FLUX stratified-60 candidate pool is re-parsed with InternVL2.5-4B and rescored with CSR. This branch isolates the perception parser and therefore reports Fixed Seed-0, Random Select, and InternVL-based CSR-Select rather than reward-model baselines.

## Human preference check

We also construct a 120-task pairwise human evaluation package. Each task shows a prompt and two anonymized images, one selected by CSR-Select and one selected by a baseline. The package includes 60 comparisons from the reviewed SDXL self-built experiment and 60 comparisons from the FLUX stratified-60 experiment, balanced across Fixed Seed-0, CLIPScore, ImageReward, PickScore, and HPSv2. The current analysis uses one filled response sheet; multi-annotator agreement is left for the full human study.

## Metrics

The current metrics are CSR-Obj, CSR-Attr, CSR-Rel, CSR-Count, CSR-Text, CSR-All, GenEval official detector scores, reward-model-selected CSR-All, and pairwise human preference counts from the single filled response sheet.
