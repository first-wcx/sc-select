# GenEval analysis

## Data provenance

- CSR results are from audited server artifacts in `server_001` and `server_004`.
- Official GenEval detector results are from `server_002` and mirrored in `server_004`.
- A new CLIPScore baseline was run on server_004 on 2026-05-13 using existing GenEval SDXL images and existing CSR scores. No new image generation or Qwen parsing was performed for this baseline.

## CSR result

On GenEval with four candidates per prompt, the audited SDXL CSR table reports:

- All-candidates mean CSR-All: 0.6363
- SC-Select Best-of-4 CSR-All: 0.7821
- Gain: +0.1458

The audited FLUX.1-schnell GenEval table reports:

- All-candidates mean CSR-All: 0.6750
- SC-Select Best-of-4 CSR-All: 0.7880
- Gain: +0.1130

These results support the limited claim that the contract-guided selection signal transfers beyond the self-built 50-prompt set and remains positive on a public compositional benchmark.

## CLIPScore baseline on SDXL GenEval

The server_004 CLIPScore run over the 553-prompt GenEval SDXL candidate pool reports:

- All-candidates mean CSR-All: 0.6363
- Fixed Seed-0 CSR-All: 0.6297
- Random Select mean CSR-All: 0.6366
- CLIPScore-Select Best-of-4 CSR-All: 0.6528
- SC-Select Best-of-4 CSR-All: 0.7821

SC-Select improves over CLIPScore-Select by +0.1293 CSR-All. The paired bootstrap/sign-test table reports a 95% CI of [0.1090, 0.1505] with p=7.17e-43.

## Official GenEval result

For SDXL GenEval folders evaluated by the official detector:

- Fixed Seed-0 overall: 0.53885
- CLIPScore-Select overall: 0.57826
- SC-Select overall: 0.62232
- Overall gain: +0.08347
- Correct images: 52.44% to 60.58%, +8.14 percentage points

Against Fixed Seed-0, the largest official-category gain is on counting (+22.50 percentage points), followed by two-object prompts (+14.14 percentage points). Against CLIPScore-Select, SC-Select improves official overall by +0.04406, mainly from counting (+18.75 percentage points), position (+5.00 percentage points), and color attribution (+7.00 percentage points). CLIPScore-Select is stronger on the colors category, where it scores 90.43% versus 85.11% for SC-Select.

## Limitations

- Official detector results have been audited for Fixed Seed-0, CLIPScore-Select, and SC-Select.
- CLIPScore is now available for the SDXL GenEval candidate pool.
- ImageReward, PickScore, HPSv2, and VLM-direct baselines are not available in the pulled GenEval result files.
- The paper should not claim that reward/preference baselines other than CLIPScore have been beaten on GenEval until those scores are actually generated.
