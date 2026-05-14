# Parser Robustness Status 2026-05-14

## Status

Completed on `server_004`.

- Parser: InternVL2.5-4B
- Model ID: `OpenGVLab/InternVL2_5-4B`
- Target images: FLUX self-built stratified-60 candidate pool, 240 images.
- Output directory: `outputs/parser_robustness/internvl2_5_4b_flux60`
- Local audit mirror: `server_audits/server_004_parser_robustness_internvl_flux60_20260514`

## Purpose

This run tests whether SC-Select remains effective when the perception graph parser changes from Qwen2.5-VL-7B to InternVL2.5-4B. The intended analysis is to rerun CSR scoring and selection using InternVL-produced graphs, then compare selected-image CSR behavior with the Qwen-based FLUX stratified-60 results.

## Results

- InternVL perception graphs: 240/240.
- InternVL CSR candidate rows: 240.
- Fixed Seed-0 selected CSR-All: 0.5278.
- Random Select mean CSR-All: 0.5288.
- InternVL-CSR-Select CSR-All: 0.7031.
- Paired bootstrap vs Fixed Seed-0: +0.1753, 95% CI [0.1108, 0.2469].

This result isolates parser robustness. CLIPScore/ImageReward/PickScore/HPSv2 are not rerun in this parser-only branch because their scores do not depend on the perception parser.
