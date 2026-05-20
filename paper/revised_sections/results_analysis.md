# Results and analysis

## Self-built 50-prompt set

On the audited self-built 50-prompt result files, CSR-Select improves CSR-All from 0.5499 for Fixed Seed-0 to 0.7456. Compared with the all-candidate mean reported by the original experiment table, SC-Select Best-of-4 improves CSR-All from 0.5865 to 0.7456.

The largest category-level gains in the original table are on text rendering (+0.3393), attribute binding (+0.1923), and relation constraints (+0.1859). This is consistent with the method's design: selection is driven by explicit fine-grained constraint checks rather than a single global image-text similarity score.

## Self-built 300-prompt reviewed set

On the 300-prompt reviewed self-built set, the full SDXL -> Qwen -> CSR pipeline completed with 1200 candidates. CSR-Select improves CSR-All from 0.3819 for Fixed Seed-0, 0.3779 for CLIPScore-Select, 0.3686 for ImageReward-Select, 0.3695 for PickScore-Select, and 0.3808 for HPSv2-Select to 0.5122. The paired mean difference is +0.1303 against Fixed Seed-0, +0.1343 against CLIPScore-Select, +0.1436 against ImageReward-Select, +0.1427 against PickScore-Select, and +0.1314 against HPSv2-Select.

The prompt/contract file was manually reviewed in nine batches after the initial draft construction, and all 300 rows are now marked as checked.

## GenEval

On audited SDXL GenEval CSR results, SC-Select Best-of-4 improves CSR-All from 0.6363 to 0.7821. A newly run CLIPScore baseline on the same 553-prompt, 2212-image candidate pool gives a CLIPScore-Select CSR-All of 0.6528, while SC-Select reaches 0.7821. On audited FLUX.1-schnell GenEval results, CSR-All improves from 0.6750 to 0.7880.

On a balanced self-built stratified-60 subset generated with FLUX.1-schnell, CSR-Select improves CSR-All from 0.5853 for Fixed Seed-0, 0.5603 for CLIPScore-Select, 0.5725 for ImageReward-Select, 0.5642 for PickScore-Select, and 0.5900 for HPSv2-Select to 0.6667. This provides an additional cross-generator check on reviewed complex prompts, but it is still a subset experiment rather than a full multi-generator benchmark.

On an auxiliary 60-prompt T2I-CompBench subset generated with SDXL, CSR-Select improves CSR-All from 0.3833 for Fixed Seed-0, 0.4333 for CLIPScore-Select, 0.4375 for ImageReward-Select, 0.3875 for PickScore-Select, and 0.4208 for HPSv2-Select to 0.5625. The subset uses automatically derived contracts from formulaic public prompts, so these numbers are treated as an auxiliary public benchmark check rather than as a replacement for the manually reviewed benchmark.

For parser robustness, re-parsing the FLUX stratified-60 candidate pool with InternVL2.5-4B gives an InternVL-based CSR-Select CSR-All of 0.7031, compared with 0.5278 for Fixed Seed-0 and 0.5288 for Random Select. This suggests that the selection behavior is not tied exclusively to Qwen2.5-VL, although absolute CSR values should be interpreted within each parser's output distribution.

In the filled 120-task pairwise human preference sheet, 106 tasks have valid non-tie choices and 14 tasks are marked as bad-prompt/both-fail. Among the 106 valid choices, CSR-Select is preferred in 69 comparisons and the baseline is preferred in 37 comparisons against Fixed Seed-0, CLIPScore, ImageReward, PickScore, and HPSv2 selections. Because this is currently a single response sheet, we report it as a human preference check rather than a multi-annotator study.

The official GenEval detector evaluation shows an overall improvement from 0.53885 for Fixed Seed-0 and 0.57826 for CLIPScore-Select to 0.62232 for SC-Select. The largest detector-category gain over Fixed Seed-0 is counting, where the score increases from 41.25% to 63.75%. Compared with CLIPScore-Select, SC-Select improves counting from 45.00% to 63.75%, although CLIPScore-Select is stronger on the colors category.

## Statistical test

Paired bootstrap and sign-test results using selected outputs show positive CSR-All differences for CSR-Select over Fixed Seed-0 on self-built 50, SDXL GenEval, and FLUX GenEval. On SDXL GenEval, SC-Select also improves over CLIPScore-Select by +0.1293 CSR-All with a 95% confidence interval of [0.1090, 0.1505]. On the FLUX self-built stratified-60 subset, CSR-Select improves over CLIPScore-Select by +0.1064, ImageReward-Select by +0.0942, PickScore-Select by +0.1025, and HPSv2-Select by +0.0767 CSR-All. On the 300-prompt reviewed set, CSR-Select improves over ImageReward-Select by +0.1436, PickScore-Select by +0.1427, and HPSv2-Select by +0.1314 CSR-All. On the T2I-CompBench subset, CSR-Select improves over CLIPScore-Select by +0.1292 and over ImageReward-Select by +0.1250. The InternVL parser-robustness branch improves over Fixed Seed-0 by +0.1753. These tests support the CSR metric result, but do not replace human evaluation.

## Remaining scope

The current evidence is sufficient for the paper's main claim: explicit semantic contract scoring improves instruction-adherence selection in a multi-candidate setting. Reward-model baselines are complete for the reviewed 300-prompt SDXL set, the FLUX stratified-60 subset, and the auxiliary T2I-CompBench subset. GenEval is primarily reported with its official detector, so additional reward-model selection on GenEval is treated as optional rather than required for the central claim.

Human preference results are available for one filled 120-task response sheet. Additional independent annotators would strengthen the human-study section, but the current manuscript reports this result conservatively as a single-response preference check.

Parser robustness results are available for InternVL2.5-4B on the FLUX stratified-60 subset. Because parser output distributions differ, the manuscript compares selection trends within each parser branch rather than mixing absolute CSR values across parsers.
