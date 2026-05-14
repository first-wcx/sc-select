# Environment Notes

Use a clean environment for GPU experiments. Do not mix SDXL generation, VLM parsing, and reward-model scoring unless dependency versions are confirmed compatible.

Recommended staged environments:

1. `scselect-base`: CPU utilities, schema validation, CSR scoring, table generation.
2. `scselect-gen`: SDXL / SD1.5 / PixArt generation.
3. `scselect-vlm`: Qwen2.5-VL / InternVL / LLaVA-OneVision parsing.
4. `scselect-reward`: CLIPScore / ImageReward / PickScore / HPSv2 scoring.

