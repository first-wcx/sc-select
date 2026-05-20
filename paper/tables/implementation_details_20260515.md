| Item | Setting |
| --- | --- |
| Main generator | `stabilityai/stable-diffusion-xl-base-1.0` |
| Main dataset | 300 manually checked prompts in `data/prompts/scselect_complex_300.jsonl` |
| Candidate pool | 4 images per prompt, seeds 0, 1, 2, and 3 |
| SDXL sampling | 30 inference steps, CFG scale 7.5, default SDXL resolution |
| FLUX generator | `black-forest-labs/FLUX.1-schnell` |
| FLUX sampling | 1024 x 1024, 4 steps, guidance scale 0.0, seeds 0, 1, 2, and 3 |
| Main parser | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Parser decoding | Greedy generation, `max_new_tokens=2048` |
| Parser output schema | JSON object with `visible_objects`, `relations`, `count`, `ocr_text`, and `uncertain` |
| JSON recovery | Direct JSON parsing first; if it fails, extract the first JSON-like block; otherwise store `raw_response` and `parse_error` |
| Contract source | Template-assisted prompt/contract construction followed by manual review; all 300 rows marked checked |
| Object matching | Lowercase exact object-name matching in the completed 300-prompt scorer |
| Attribute matching | Same object name plus substring match between expected and parsed attribute type/value |
| Relation matching | Same subject and object names plus alias-normalized relation matching |
| Count matching | Exact integer match for the expected object count |
| OCR matching | Case-insensitive substring match unless `case_sensitive` is set |
| Selection rule | Select the candidate with maximum CSR-All; missing constraint families are excluded from the average |
| Main hardware | RTX 5090 32GB server for the 300-prompt SDXL run |
