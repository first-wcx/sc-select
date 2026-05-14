# Cross-generator analysis

## Current evidence

Audited GenEval CSR results currently cover:

| Generator | Dataset | Candidates | SC-Select CSR-All | Gain vs all-candidate mean |
|---|---|---:|---:|---:|
| SDXL | GenEval | 4 | 0.7821 | +0.1458 |
| FLUX.1-schnell | GenEval | 4 | 0.7880 | +0.1130 |

This is enough to state a preliminary cross-generator observation: SC-Select improves CSR-All for both audited SDXL and FLUX GenEval result sets.

An additional balanced self-built subset was run on `server_004` on 2026-05-14. It samples 60 reviewed prompts, 10 from each self-built category, and generates four FLUX.1-schnell candidates per prompt.

| Generator | Dataset | Candidates | Fixed Seed-0 | CLIPScore | ImageReward | PickScore | HPSv2 | SC-Select |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FLUX.1-schnell | self-built stratified-60 | 4 | 0.5853 | 0.5603 | 0.5725 | 0.5642 | 0.5900 | 0.6667 |

On this stratified subset, SC-Select improves CSR-All by +0.0814 over Fixed Seed-0, +0.1064 over CLIPScore-Select, +0.0942 over ImageReward-Select, +0.1025 over PickScore-Select, and +0.0767 over HPSv2-Select.

## What cannot be claimed yet

- This is not yet a broad multi-generator study.
- SD1.5, SD2.1, PixArt, Playground, and additional FLUX variants have not been run in the current local consolidation step.
- ImageReward, PickScore, and HPSv2 have been computed for the FLUX self-built stratified-60 subset, but remain pending for GenEval FLUX and other generators.

## Recommended next step

Next, either extend the FLUX self-built subset to more prompts or run a second generator such as SD1.5, SD2.1, or PixArt if the required model cache and disk budget are confirmed.
