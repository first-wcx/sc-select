# Results summary

## Experiments actually run in this local workspace

No new GPU image generation, VLM parsing, or detector inference was run locally in this pass.

The following CPU-only consolidation scripts were run on pulled artifacts:

| Task | Input | Output |
|---|---|---|
| Baseline selection summary, self-built 50 | `server_audits/server_004/outputs/csr_scores/csr_results.jsonl` | `outputs/selections/self_built_50/baseline_summary.csv` |
| Baseline selection summary, FLUX GenEval | `server_audits/server_001/csr_scores/csr_results.jsonl` | `outputs/selections/geneval_flux/baseline_summary.csv` |
| Paper result tables | audited `paper_assets/tables/*.md` | `outputs/tables/*.csv`, `paper/tables/*.csv` |
| Paper SVG figures | generated CSV tables | `outputs/figures/*.svg`, `paper/figures/*.svg` |
| Paired bootstrap/sign test | selected JSONL files | `outputs/tables/statistical_significance_*.csv` |
| GPU smoke test on server_004 | 5 prompts, 20 SDXL images, Qwen parsing, CSR, CLIPScore | `server_audits/server_004_smoke_5_20260513_1228` |
| GPU self-built 50 rerun on server_004 | 50 prompts, 200 SDXL images, Qwen parsing, CSR, CLIPScore, Best-of-N | `server_audits/server_004_self50_rerun_20260513_1239` |
| GPU CLIPScore baseline on existing GenEval SDXL candidates | 553 prompts, 2212 existing SDXL images, existing Qwen/CSR scores, CLIPScore | `server_audits/server_004_geneval_clipscore_20260513_1317` |
| GenEval official detector for CLIPScore-selected SDXL images | 553 CLIPScore-selected images, official GenEval detector, CPU fallback | `server_audits/server_004_geneval_clipscore_20260513_1317/outputs/geneval_official/clipscore_select` |
| GPU self-built 300 reviewed-set run on server_004 | 300 manually reviewed prompts, 1200 SDXL images, Qwen parsing, CSR scoring, CLIPScore, ImageReward, PickScore, and HPSv2 baselines | `server_audits/server_004_self300_20260513` |
| GPU FLUX self-built stratified-60 run on server_004 | 60 reviewed prompts, 240 FLUX.1-schnell images, Qwen parsing, CSR scoring, CLIPScore, ImageReward, PickScore, and HPSv2 baselines | `server_audits/server_004_flux_self300_stratified_60_20260514` |

## Existing experiments with audited artifact evidence

| Experiment | Server | Evidence | Result |
|---|---|---|---|
| Self-built 50 SDXL CSR-Select | server_004 | `server_audits/server_004/paper_assets/tables/main_result.md` | CSR-All 0.5865 -> 0.7456 |
| Self-built 50 Fixed Seed-0 vs CSR-Select | server_004 + local selection summary | `outputs/selections/self_built_50/baseline_summary.csv` | Fixed Seed-0 0.5499 -> CSR-Select 0.7456 |
| GenEval SDXL CSR-Select | server_004 | `server_audits/server_004/paper_assets/tables/geneval_csr_result.md` | CSR-All 0.6363 -> 0.7821 |
| GenEval official evaluator | server_002 / server_004 | `server_audits/server_002/official_results/*/summary.txt` | Overall 0.53885 -> 0.62232 |
| GenEval official evaluator, CLIPScore baseline | server_004 | `server_audits/server_004_geneval_clipscore_20260513_1317/outputs/geneval_official/clipscore_select/summary.txt` | CLIPScore-Select overall 0.57826 |
| FLUX.1-schnell GenEval CSR-Select | server_001 | `server_audits/server_001/outputs/flux/geneval/sc_select_summary.json` | CSR-All 0.6749547920 -> 0.7879746835 |
| FLUX.1-schnell self-built stratified-60 | server_004 | `outputs/tables/flux_self300_stratified_60_results.csv` | Fixed 0.5853, CLIPScore 0.5603, ImageReward 0.5725, PickScore 0.5642, HPSv2 0.5900, CSR-Select 0.6667 |
| VQA-Select baseline | server_004 | `server_audits/server_004/paper_assets/tables/vqa_select_result.md` | VQA-Select 0.6835 vs CSR-Select 0.7821 |

## New GPU smoke test completed on server_004

| Dataset | Generator | Parser | Candidates | Method | CSR-All |
|---|---|---|---:|---|---:|
| self-built smoke-5 | SDXL | Qwen2.5-VL-7B | 20 images total | All candidates mean | 0.6250 |
| self-built smoke-5 | SDXL | Qwen2.5-VL-7B | 4 per prompt | Fixed Seed-0 | 0.7333 |
| self-built smoke-5 | SDXL | Qwen2.5-VL-7B | 4 per prompt | CLIPScore-Select | 0.6000 |
| self-built smoke-5 | SDXL | Qwen2.5-VL-7B | 4 per prompt | CSR-Select | 0.8333 |

This result verifies that the SDXL -> Qwen perception graph -> CSR scoring -> CSR selection -> CLIPScore baseline pipeline runs end to end on `server_004`. Because it uses only 5 prompts, it should be reported as a smoke test, not as a paper-level main result.

## New self-built 50 rerun completed on server_004

| Method | CSR-All |
|---|---:|
| All candidates mean | 0.5865 |
| Fixed Seed-0 | 0.5499 |
| Random Select mean | 0.5940 |
| CLIPScore-Select Best-of-4 | 0.6201 |
| CSR-Select Best-of-4 | 0.7456 |

Key gains:

- CSR-Select vs all-candidates mean: +0.1592
- CSR-Select vs Fixed Seed-0: +0.1957
- CSR-Select vs CLIPScore-Select: +0.1256

Best-of-N CSR-All:

- Best-of-1: 0.5499
- Best-of-2: 0.6962
- Best-of-3: 0.7212
- Best-of-4: 0.7456

Paired checks:

- CSR-Select - Fixed Seed-0: mean diff 0.1957, 95% CI [0.1389, 0.2596], p=7.45e-09
- CSR-Select - CLIPScore-Select: mean diff 0.1256, 95% CI [0.0736, 0.1872], p=1.91e-06

## Statistical checks run from existing selected outputs

| Dataset | Comparison | Mean CSR-All diff | 95% CI | p-value |
|---|---|---:|---|---:|
| self_built_50 | CSR-Select - Fixed Seed-0 | 0.1957 | [0.1389, 0.2596] | 7.45e-09 |
| geneval_flux | CSR-Select - Fixed Seed-0 | 0.1230 | [0.1044, 0.1410] | 5.74e-42 |

## New GenEval SDXL CLIPScore baseline completed on server_004

This run reused the existing GenEval SDXL images and CSR scores on server_004. No image generation or Qwen parsing was rerun.

| Method | CSR-All |
|---|---:|
| All candidates mean | 0.6363 |
| Fixed Seed-0 | 0.6297 |
| Random Select mean | 0.6366 |
| CLIPScore-Select Best-of-4 | 0.6528 |
| CSR-Select Best-of-4 | 0.7821 |

Key gains:

- CSR-Select vs all-candidates mean: +0.1458
- CSR-Select vs Fixed Seed-0: +0.1524
- CSR-Select vs CLIPScore-Select: +0.1293

Paired checks:

- CSR-Select - Fixed Seed-0: mean diff 0.1524, 95% CI [0.1307, 0.1722], p=1.67e-52
- CSR-Select - CLIPScore-Select: mean diff 0.1293, 95% CI [0.1090, 0.1505], p=7.17e-43

Official GenEval detector results for the same CLIPScore-selected images:

| Method | Official overall |
|---|---:|
| Fixed Seed-0 | 0.53885 |
| CLIPScore-Select | 0.57826 |
| SC-Select | 0.62232 |

SC-Select gains +0.04406 official overall over CLIPScore-Select and +0.08347 over Fixed Seed-0.

## New self-built 300 reviewed-set run completed on server_004

This run completed the full pipeline on the 300-prompt self-built set: 1200 SDXL images, 1200 Qwen perception graphs, 1200 CSR scores, plus CLIPScore, ImageReward, PickScore, and HPSv2 baselines. The prompt/contract file has now been manually reviewed in nine batches and all 300 rows are marked `checked:true`.

| Method | CSR-All |
|---|---:|
| Fixed Seed-0 | 0.3819 |
| Random Select mean | 0.3437 |
| CLIPScore-Select Best-of-4 | 0.3779 |
| ImageReward-Select Best-of-4 | 0.3686 |
| PickScore-Select Best-of-4 | 0.3695 |
| HPSv2-Select Best-of-4 | 0.3808 |
| CSR-Select Best-of-4 | 0.5122 |

Key gains:

- CSR-Select vs Fixed Seed-0: +0.1303
- CSR-Select vs CLIPScore-Select: +0.1343
- CSR-Select vs ImageReward-Select: +0.1436
- CSR-Select vs PickScore-Select: +0.1427
- CSR-Select vs HPSv2-Select: +0.1314

Best-of-N CSR-All:

- Best-of-1: 0.3819
- Best-of-2: 0.4476
- Best-of-3: 0.4879
- Best-of-4: 0.5122

Paired checks:

- CSR-Select - Fixed Seed-0: mean diff 0.1303, 95% CI [0.1116, 0.1499], p=1.43e-42
- CSR-Select - CLIPScore-Select: mean diff 0.1343, 95% CI [0.1145, 0.1557], p=7.35e-40
- CSR-Select - ImageReward-Select: mean diff 0.1436, 95% CI [0.1233, 0.1638], p=5.61e-45
- CSR-Select - PickScore-Select: mean diff 0.1427, 95% CI [0.1210, 0.1648], p=1.84e-40
- CSR-Select - HPSv2-Select: mean diff 0.1314, 95% CI [0.1101, 0.1548], p=1.20e-35

HPSv2 was completed after fixing the scoring script for the installed API. ImageReward was completed using the existing cached checkpoint and runtime compatibility shims for the installed `transformers` API. PickScore was completed after downloading the model cache through `hf-mirror.com` and patching the local scoring path for the installed `transformers` API. VLM-Direct remains not available for this run.

## T2I-CompBench SDXL 60-prompt auxiliary subset

This run used 20 color, 20 spatial, and 20 numeracy prompts from the public T2I-CompBench prompt files. Contracts were automatically derived from formulaic prompts and remain `checked:false`, so this is an auxiliary public-benchmark stress test rather than a fully curated benchmark.

| Method | CSR-All |
|---|---:|
| Fixed Seed-0 | 0.3833 |
| Random Select mean | 0.3858 |
| CLIPScore-Select Best-of-4 | 0.4333 |
| ImageReward-Select Best-of-4 | 0.4375 |
| PickScore-Select Best-of-4 | 0.3875 |
| HPSv2-Select Best-of-4 | 0.4208 |
| CSR-Select Best-of-4 | 0.5625 |

Paired checks:

- CSR-Select - Fixed Seed-0: mean diff 0.1792, 95% CI [0.1125, 0.2542], p=9.54e-07
- CSR-Select - CLIPScore-Select: mean diff 0.1292, 95% CI [0.0708, 0.1958], p=3.05e-05
- CSR-Select - ImageReward-Select: mean diff 0.1250, 95% CI [0.0750, 0.1750], p=3.81e-06

## Parser robustness with InternVL2.5-4B

The FLUX self-built stratified-60 candidate pool was re-parsed with InternVL2.5-4B and rescored with CSR. This parser-only branch reports Fixed Seed-0, Random Select, and InternVL-based CSR-Select.

| Method | CSR-All |
|---|---:|
| Fixed Seed-0 | 0.5278 |
| Random Select mean | 0.5288 |
| InternVL-CSR-Select Best-of-4 | 0.7031 |

Paired check:

- InternVL-CSR-Select - Fixed Seed-0: mean diff 0.1753, 95% CI [0.1108, 0.2469], p=5.96e-08

## Human evaluation

One filled pairwise response sheet has been analyzed for the 120-task blind comparison package. It covers 60 SDXL self-built reviewed-set comparisons and 60 FLUX stratified-60 comparisons against Fixed Seed-0, CLIPScore, ImageReward, PickScore, and HPSv2. CSR-Select was preferred in 120/120 comparisons, with no ties and no bad-prompt/both-fail flags.

This should currently be reported as a single-response human preference check. The response sheet annotator id is `auto_html_csr_side`; collect additional independently filled sheets before claiming a multi-annotator human study.

## Not yet available

- CLIPScore-Select is now available for the self-built 50 rerun and for GenEval SDXL candidates.
- ImageReward-Select, PickScore-Select, and HPSv2-Select are now available for the self-built 300 reviewed-set run.
- VLM-Direct scores are not present in the pulled candidate JSONL files.

These items must remain `Pending` in the paper until real outputs are generated.
