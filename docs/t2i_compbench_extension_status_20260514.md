# T2I-CompBench Extension Status 2026-05-14

## Source

T2I-CompBench official repository files were used from `Karine-Huang/T2I-CompBench`, under `examples/dataset`.

## Prepared Subset

- Raw files: `data/benchmarks/t2i_compbench_raw`
- Prompt file: `data/benchmarks/t2i_compbench_subset_60_prompts.jsonl`
- Contract file: `data/benchmarks/t2i_compbench_subset_60_contracts.jsonl`
- Categories: 20 color, 20 spatial, 20 numeracy.

Contracts are automatically derived from the formulaic benchmark prompts and currently marked `checked:false`. They should be described as auto-derived unless manually audited.

## Completed Run

Completed on `server_004` under:

- `outputs/t2i_compbench_sdxl_60`
- local audit mirror: `server_audits/server_004_t2i_compbench_sdxl_60_20260514`

The pipeline waits for the InternVL parser robustness job to finish, then runs:

1. SDXL generation, 4 seeds per prompt.
2. Qwen2.5-VL parsing.
3. CSR scoring.
4. CLIPScore.
5. PickScore.
6. HPSv2.
7. ImageReward.
8. Baseline selection summary.

## Results

CSR-All selected-output means on the 60-prompt subset:

- Fixed Seed-0: 0.3833
- Random Select mean: 0.3858
- CLIPScore-Select: 0.4333
- ImageReward-Select: 0.4375
- PickScore-Select: 0.3875
- HPSv2-Select: 0.4208
- CSR-Select: 0.5625

Paired bootstrap checks:

- CSR-Select vs Fixed Seed-0: +0.1792, 95% CI [0.1125, 0.2542].
- CSR-Select vs CLIPScore-Select: +0.1292, 95% CI [0.0708, 0.1958].
- CSR-Select vs ImageReward-Select: +0.1250, 95% CI [0.0750, 0.1750].

The contracts remain automatically derived from formulaic prompts and are still marked `checked:false`; report this as an auxiliary public benchmark subset rather than a fully curated benchmark.
