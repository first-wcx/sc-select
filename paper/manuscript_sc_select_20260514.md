# SC-Select: Semantic Contract-Guided Selection for Instruction-Adherent Text-to-Image Generation

## Abstract

Text-to-image generators often struggle with fine-grained instruction following when a prompt contains multiple objects, bound attributes, spatial relations, counts, or rendered text. This paper studies SC-Select, a training-free post-generation selection strategy for improving the final output in a multi-candidate generation setting. SC-Select decomposes a prompt into a structured semantic contract, extracts a perception graph from each candidate image using a vision-language model, and selects the candidate with the highest Contract Satisfaction Rate (CSR). On a manually reviewed 300-prompt set with 1200 SDXL candidates, SC-Select improves CSR-All from 0.3779 for CLIPScore-Select, 0.3686 for ImageReward-Select, 0.3695 for PickScore-Select, and 0.3808 for HPSv2-Select to 0.5122. On GenEval, SC-Select improves the official detector overall score from 0.53885 for Fixed Seed-0 and 0.57826 for CLIPScore-Select to 0.62232. Additional experiments show gains on a FLUX.1-schnell stratified subset, an auxiliary T2I-CompBench subset, and an InternVL2.5-4B parser-robustness branch. A 120-task single-response human preference check contains 106 valid non-tie judgments, where SC-Select is preferred in 69 cases and the compared baseline is preferred in 37 cases. The method does not modify the generator and remains bounded by candidate diversity and parser reliability, but provides an interpretable selection signal for instruction-adherent text-to-image generation.

**Keywords:** text-to-image generation; instruction following; semantic evaluation; best-of-N selection; vision-language models; compositional generation

## 1. Introduction

Modern text-to-image generators can produce visually compelling images, but complex instruction following remains fragile. A prompt such as "a red robot holding a blue umbrella beside a yellow car" requires the model to satisfy object existence, attribute binding, spatial relation, and sometimes counting or text-rendering constraints at the same time. A generated image may be plausible overall while still violating one or more fine-grained requirements.

This paper frames the problem as post-generation selection. Given a prompt and a pool of candidate images, the goal is not to train a stronger generator, but to select the candidate that best satisfies the prompt's explicit semantic constraints. SC-Select first decomposes the prompt into a semantic contract, then uses a vision-language model to extract a perception graph from each candidate image, computes contract satisfaction rates over several constraint families, and selects the image with the highest CSR-All score.

The main claim is conservative: in multi-candidate settings, explicit contract-guided selection improves measured instruction adherence over fixed single-candidate selection, global image-text similarity selection, and preference-model selection on the completed experiments. SC-Select should not be interpreted as improving the generator itself or as directly optimizing image realism. It is a selection signal over an existing candidate pool, and its practical value depends on candidate diversity, parser reliability, and the cost of generating and evaluating multiple images.

The contributions are as follows.

1. We propose SC-Select, a training-free semantic contract-guided selection method for text-to-image generation.
2. We define Contract Satisfaction Rate across objects, attributes, relations, counts, and rendered text, making selection decisions traceable to explicit prompt constraints.
3. We construct and manually review a 300-prompt complex instruction benchmark and release the code, schemas, prompts, and summarized evaluation artifacts.
4. We evaluate SC-Select against fixed selection, random selection, CLIPScore, ImageReward, PickScore, and HPSv2, and further validate the method on GenEval, FLUX.1-schnell, T2I-CompBench, an InternVL parser branch, and a single-response human preference check.

## 2. Related Work

### 2.1 Text-to-image generation

Text-to-image generation has progressed from early generative adversarial networks to diffusion-based models. Denoising diffusion probabilistic models (DDPM) [ho2020ddpm] established a foundation for high-quality image synthesis. Latent diffusion models [rombach2022latent] moved the diffusion process to a compressed latent space, enabling efficient high-resolution generation and forming the backbone of Stable Diffusion. DALL-E 2 [ramesh2022hierarchical] and Imagen [saharia2022imagen] demonstrated that scaling text encoders with CLIP and T5 can improve text-image alignment. More recently, SDXL [podell2024sdxl] introduced architectural improvements including dual text encoders and a refinement model. Transformer-based diffusion architectures such as PixArt-alpha [chen2023pixart] and PixArt-Sigma [chen2024pixartsigma] further improved efficient text-to-image modeling, while Stable Diffusion 3 [esser2024sd3] and FLUX.1 adopted rectified flow or diffusion-transformer designs. SC-Select is orthogonal to these generator architectures because it operates after generation.

### 2.2 Compositional evaluation

Benchmarks such as GenEval [gokaslan2023geneval] and T2I-CompBench [huang2023t2icompbench] evaluate whether generated images satisfy object, attribute, relation, and counting requirements. GenEval decomposes prompts into object-level criteria and uses detector-based verification. T2I-CompBench covers attribute binding, spatial relations, and non-spatial relations with automated and human evaluation metrics. VQAScore [lin2024vqascore] and VIEScore [ku2023viescore] reflect a broader movement toward instruction-aware evaluation with vision-language models. These works motivate SC-Select's core design: complex prompts should be evaluated at the level of fine-grained constraints rather than only by global caption-image similarity.

### 2.3 Preference and reward models

ImageReward [li2023imagereward], HPSv2 [wu2023hpsv2], and PickScore [kirstain2023pickapic] learn human preference signals and are useful for ranking generated images. CLIPScore [hessel2021clipscore] provides reference-free image-text similarity using CLIP [radford2021clip]. These scores can favor visually appealing or globally aligned images, but they do not necessarily diagnose which prompt constraints are satisfied or violated. SC-Select is complementary: it targets interpretable constraint satisfaction and can be compared directly against these reward-model selection baselines.

### 2.4 Vision-language models as evaluators

Large vision-language models make it practical to parse generated images into structured descriptions. LLaVA [liu2023llava], LLaVA-1.5 [liu2023llava15], LLaVA-OneVision [li2024llavaonevision], Qwen2.5-VL [qwen2025qwen25vl], InternVL [chen2024internvl], and CogVLM [wang2024cogvlm] demonstrate increasingly strong image understanding, OCR, and grounding capabilities. SC-Select relies on this capability to build perception graphs. This reliance is also a limitation: parser errors can propagate into CSR scores.

## 3. Method

### 3.1 Problem formulation

Given a text prompt \(p\), a generator \(G\), and a candidate count \(N\), the generator produces a candidate pool \(\{I_1,\ldots,I_N\}\). SC-Select chooses one image from this pool without updating the generator parameters:

\[
I^* = \arg\max_{I_i} CSR\text{-}All(I_i,p).
\]

### 3.2 Semantic contract

The prompt is represented as a semantic contract containing five constraint families:

- **Objects:** entities that should appear in the image.
- **Attributes:** properties bound to specific objects.
- **Relations:** spatial or action relations between objects.
- **Counts:** expected object quantities.
- **Texts:** short strings expected to appear in the image.

The reproducible schema is defined in `configs/contract_schema.json`. The contract is designed to be explicit rather than exhaustive: it captures the prompt requirements that can be checked consistently by the parser and scorer.

### 3.3 Perception graph extraction

Each candidate image is parsed into a perception graph using a vision-language model. The graph records visible objects, attributes, relations, counts, OCR text, parser identity, and raw response. The reproducible schema is defined in `configs/perception_graph_schema.json`. The main completed experiments use Qwen2.5-VL for parsing, and a robustness branch re-parses the FLUX stratified-60 candidate pool with InternVL2.5-4B.

### 3.4 Contract Satisfaction Rate

For each candidate, SC-Select computes five partial scores: CSR-Obj, CSR-Attr, CSR-Rel, CSR-Count, and CSR-Text. CSR-All averages the available constraint-family scores. Missing constraint types are ignored for prompts where that type is not specified:

\[
CSR\text{-}All(I,p)=\frac{1}{|\mathcal{K}(p)|}\sum_{k\in \mathcal{K}(p)}CSR_k(I,p),
\]

where \(\mathcal{K}(p)\) is the set of constraint families present in the prompt contract. This design makes the selection criterion interpretable because every selected image can be traced back to satisfied and violated contract items.

### 3.5 Candidate pool effect

If a single candidate satisfies a constraint with probability \(p\), and candidates are sampled independently, the probability that at least one of \(N\) candidates satisfies the constraint is:

\[
P_N=1-(1-p)^N.
\]

This explains why best-of-N selection can improve prompt adherence without changing the generator. The gain is not unlimited: marginal improvement decreases as \(N\) grows, and generation plus parsing cost grows approximately linearly with the candidate count.

### 3.6 Optional quality-aware extension

The main method uses CSR alone so that the selected image is traceable to explicit prompt constraints. A future quality-aware variant can combine CSR with an image quality or preference score:

\[
Score(I)=\alpha CSR(I)+(1-\alpha)Q(I).
\]

This extension is not used for the main reported results. It is included only to clarify how the selection signal could be combined with preference models in future work.

## 4. Experiments

### 4.1 Datasets and generators

The self-built resources include an original 50-prompt pilot set, a 300-prompt reviewed set in `data/prompts/scselect_complex_300.jsonl`, and a stratified-60 subset with 10 prompts from each category. The 300-prompt set covers attribute binding, spatial/action relations, counting, text rendering, and complex composition. All prompt/contract rows are marked as manually checked after review.

GenEval is used as the main public benchmark. It provides object-focused prompts and official detector-based evaluation categories. An auxiliary T2I-CompBench subset is used as a public compositional stress test. The subset contains 60 formulaic prompts: 20 color-binding prompts, 20 spatial-relation prompts, and 20 numeracy prompts. Semantic contracts for this subset are derived automatically from public prompt templates and are therefore reported separately from the manually reviewed self-built benchmark.

The completed generator evidence covers SDXL and FLUX.1-schnell. The main SDXL experiment generates four candidates per prompt for the 300-prompt reviewed set, yielding 1200 candidates.

### 4.2 Baselines

The completed baselines include Fixed Seed-0, Random Select with repeated seeds, CLIPScore-Select, ImageReward-Select, PickScore-Select, HPSv2-Select, and CSR-Select. VLM-Direct-Select is not included in the completed result tables and is left for future work.

For parser robustness, the FLUX stratified-60 candidate pool is re-parsed with InternVL2.5-4B and rescored with CSR. This branch isolates the perception parser and therefore reports Fixed Seed-0, Random Select, and InternVL-based CSR-Select rather than reward-model baselines.

### 4.3 Metrics

The main metric is CSR-All, supported by CSR-Obj, CSR-Attr, CSR-Rel, CSR-Count, and CSR-Text. Public benchmark validation uses GenEval official detector scores. Additional comparisons use CSR-All after reward-model-based selection. Human preference is reported as pairwise counts from one filled 120-task response sheet; because it is a single-response sheet, it is treated as a preference check rather than a multi-annotator study.

## 5. Results

### 5.1 Reviewed 300-prompt SDXL benchmark

On the reviewed 300-prompt SDXL set, the full SDXL -> Qwen2.5-VL -> CSR pipeline completed with 1200 candidates. SC-Select improves CSR-All from 0.3819 for Fixed Seed-0, 0.3779 for CLIPScore-Select, 0.3686 for ImageReward-Select, 0.3695 for PickScore-Select, and 0.3808 for HPSv2-Select to 0.5122.

| Method | CSR-All |
| --- | ---: |
| Fixed Seed-0 | 0.3819 |
| Random Select | 0.3437 |
| CLIPScore-Select | 0.3779 |
| ImageReward-Select | 0.3686 |
| PickScore-Select | 0.3695 |
| HPSv2-Select | 0.3808 |
| **SC-Select** | **0.5122** |

The paired mean difference is +0.1303 against Fixed Seed-0, +0.1343 against CLIPScore-Select, +0.1436 against ImageReward-Select, +0.1427 against PickScore-Select, and +0.1314 against HPSv2-Select. These gains support the central claim that explicit semantic contract scoring selects more instruction-adherent outputs than global similarity or preference-model ranking on this benchmark.

### 5.2 GenEval validation

On audited SDXL GenEval CSR results, SC-Select improves CSR-All from 0.6363 for the all-candidate mean and 0.6528 for CLIPScore-Select to 0.7821. On the official GenEval detector, SC-Select improves the overall score from 0.53885 for Fixed Seed-0 and 0.57826 for CLIPScore-Select to 0.62232.

| GenEval SDXL metric | Fixed / mean baseline | CLIPScore-Select | SC-Select |
| --- | ---: | ---: | ---: |
| CSR-All | 0.6363 | 0.6528 | **0.7821** |
| Official detector overall | 0.53885 | 0.57826 | **0.62232** |

The largest official detector-category gain over Fixed Seed-0 is counting, where the score increases from 41.25% to 63.75%. Compared with CLIPScore-Select, SC-Select improves counting from 45.00% to 63.75%, although CLIPScore-Select is stronger on the colors category. This mixed category behavior is expected because SC-Select optimizes explicit contract satisfaction rather than the detector's category-specific heuristics.

### 5.3 Cross-generator and public auxiliary checks

On audited FLUX.1-schnell GenEval results, CSR-All improves from 0.6750 to 0.7880. On a balanced self-built stratified-60 subset generated with FLUX.1-schnell, SC-Select improves CSR-All from 0.5853 for Fixed Seed-0, 0.5603 for CLIPScore-Select, 0.5725 for ImageReward-Select, 0.5642 for PickScore-Select, and 0.5900 for HPSv2-Select to 0.6667.

On the auxiliary 60-prompt T2I-CompBench subset generated with SDXL, SC-Select improves CSR-All from 0.3833 for Fixed Seed-0, 0.4333 for CLIPScore-Select, 0.4375 for ImageReward-Select, 0.3875 for PickScore-Select, and 0.4208 for HPSv2-Select to 0.5625.

| Experiment | Fixed | CLIPScore | ImageReward | PickScore | HPSv2 | SC-Select |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FLUX stratified-60 | 0.5853 | 0.5603 | 0.5725 | 0.5642 | 0.5900 | **0.6667** |
| T2I-CompBench SDXL-60 | 0.3833 | 0.4333 | 0.4375 | 0.3875 | 0.4208 | **0.5625** |

The T2I-CompBench subset uses automatically derived contracts from formulaic public prompts, so these numbers are treated as an auxiliary benchmark check rather than as a replacement for the manually reviewed 300-prompt benchmark.

### 5.4 Parser robustness

To test whether the selection behavior depends only on Qwen2.5-VL, the FLUX stratified-60 candidate pool is re-parsed with InternVL2.5-4B. In this branch, InternVL-based CSR-Select reaches 0.7031 CSR-All, compared with 0.5278 for Fixed Seed-0 and 0.5288 for Random Select.

| Parser branch | Fixed Seed-0 | Random Select | CSR-Select |
| --- | ---: | ---: | ---: |
| InternVL2.5-4B on FLUX stratified-60 | 0.5278 | 0.5288 | **0.7031** |

Absolute CSR values should be interpreted within each parser's output distribution. The important observation is the consistent selection trend under a second parser.

### 5.5 Human preference check

The filled pairwise preference sheet contains 120 tasks. Each task shows a prompt and two anonymized images, one selected by SC-Select and one selected by a baseline. The package includes 60 comparisons from the reviewed SDXL self-built experiment and 60 comparisons from the FLUX stratified-60 experiment, balanced across Fixed Seed-0, CLIPScore, ImageReward, PickScore, and HPSv2. In the current filled response sheet, 106 tasks have valid non-tie choices and 14 tasks are marked as bad-prompt/both-fail. Among the 106 valid choices, SC-Select is preferred in 69 comparisons and the baseline is preferred in 37 comparisons, giving a 65.1% CSR win rate excluding flagged tasks and ties.

| Dataset | Baseline | CSR wins | Baseline wins | Flagged/missing | CSR win rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| FLUX stratified-60 | CLIPScore | 9 | 3 | 0 | 0.750 |
| FLUX stratified-60 | Fixed Seed-0 | 10 | 2 | 0 | 0.833 |
| FLUX stratified-60 | HPSv2 | 7 | 5 | 0 | 0.583 |
| FLUX stratified-60 | ImageReward | 6 | 6 | 0 | 0.500 |
| FLUX stratified-60 | PickScore | 10 | 2 | 0 | 0.833 |
| SDXL self-built 300 | CLIPScore | 8 | 4 | 0 | 0.667 |
| SDXL self-built 300 | Fixed Seed-0 | 8 | 2 | 2 | 0.800 |
| SDXL self-built 300 | HPSv2 | 3 | 5 | 4 | 0.375 |
| SDXL self-built 300 | ImageReward | 5 | 4 | 3 | 0.556 |
| SDXL self-built 300 | PickScore | 3 | 4 | 5 | 0.429 |

This result is useful as a sanity check that the metric gains are visible to a human reviewer, but it is not presented as a formal multi-annotator study. A stronger human evaluation should add independent annotators and report agreement.

### 5.6 Statistical analysis

Paired bootstrap and sign-test results show positive CSR-All differences for SC-Select over the completed baselines. On SDXL GenEval, SC-Select improves over CLIPScore-Select by +0.1293 CSR-All with a 95% confidence interval of [0.1090, 0.1505]. On the reviewed 300-prompt SDXL set, SC-Select improves over ImageReward-Select by +0.1436, PickScore-Select by +0.1427, and HPSv2-Select by +0.1314 CSR-All. On the FLUX self-built stratified-60 subset, SC-Select improves over CLIPScore-Select by +0.1064, ImageReward-Select by +0.0942, PickScore-Select by +0.1025, and HPSv2-Select by +0.0767 CSR-All. On the T2I-CompBench subset, SC-Select improves over CLIPScore-Select by +0.1292 and over ImageReward-Select by +0.1250. The InternVL parser-robustness branch improves over Fixed Seed-0 by +0.1753.

These tests support the CSR metric result, but they do not replace human evaluation. The manuscript therefore separates metric-based evidence, official detector evidence, parser robustness, and human preference evidence.

## 6. Discussion

SC-Select works because many complex-prompt failures are local. An image can be globally similar to a prompt while missing a count, binding the wrong color to an object, or placing objects in the wrong relation. Global similarity and preference models can reward plausibility and aesthetics, but they often do not expose these local failures. By decomposing prompts into checkable contract items, SC-Select selects candidates based on the same semantic units that define instruction adherence.

The method is especially suitable when users already generate multiple candidates and need a principled final selection rule. It is less suitable when candidate generation is expensive, when all candidates fail the same required constraint, or when the perception parser is unreliable for the prompt domain. The results should therefore be read as evidence for selection under candidate diversity, not as evidence that SC-Select repairs the generator.

The reward-model baselines are important because they test whether human-preference rankers already solve the selection problem. The completed experiments indicate that preference rankers do not consistently select the most constraint-satisfying image under CSR. This does not mean that SC-Select is always visually superior; it means that explicit contract satisfaction and visual preference are different ranking signals. A future quality-aware extension can combine them.

## 7. Limitations

SC-Select is a post-generation selection method, not a generator training method. It cannot create semantic content that is absent from all candidates in the pool.

The method depends on the accuracy of the perception graph extractor. If the vision-language parser misses an object, hallucinates an attribute, or misreads text, CSR may reward or penalize the wrong candidate. The InternVL branch reduces but does not eliminate this concern.

CSR is a task-specific metric. Current results show improvements on CSR-All, GenEval official scores, reward-model comparisons, and a single-response human preference check, but a stronger multi-annotator study is still needed before making broad claims about human preference.

Generating multiple candidates increases cost. Best-of-4 can improve instruction adherence, but it requires more generation, parsing, and scoring work than a single output.

The 300-prompt self-built dataset has been manually reviewed, but it is still partly template-derived. Future work should diversify prompt wording, add independent annotators, and evaluate additional generator families beyond the current SDXL and FLUX evidence.

## 8. Conclusion

SC-Select provides a training-free, interpretable selection signal for complex text-to-image prompts. By decomposing prompts into semantic contracts and matching them against perception graphs extracted from candidate images, the method targets fine-grained instruction adherence more explicitly than global image-text similarity or preference-model scores.

Audited results show positive CSR-All gains on the reviewed 300-prompt SDXL set, GenEval, a FLUX stratified subset, and an auxiliary T2I-CompBench subset. SC-Select also improves GenEval official detector scores and remains effective when the FLUX stratified-60 candidate pool is re-parsed with InternVL2.5-4B. The strongest defensible conclusion is that semantic contract-guided selection can improve final-output instruction adherence in multi-candidate settings when suitable candidates exist.

## Code and Data Availability

Code, schemas, prompt files, summarized result tables, and reproducibility documentation are available at:

https://github.com/first-wcx/sc-select

The public repository excludes generated images, model weights, server credentials, and other large or sensitive artifacts. Private experiment images and raw intermediate outputs are retained locally for audit and manuscript preparation.

## References

The BibTeX database is maintained in `paper/references/references.bib`. Citation keys used in this manuscript include: [chen2023pixart], [chen2024internvl], [chen2024pixartsigma], [esser2024sd3], [gokaslan2023geneval], [hessel2021clipscore], [heusel2017fid], [ho2020ddpm], [huang2023t2icompbench], [kirstain2023pickapic], [ku2023viescore], [li2023imagereward], [li2024llavaonevision], [lin2024vqascore], [liu2023llava], [liu2023llava15], [manakul2023selfcheckgpt], [podell2024sdxl], [qwen2025qwen25vl], [radford2021clip], [ramesh2022hierarchical], [rombach2022latent], [saharia2022imagen], [vaswani2017transformer], [wang2024cogvlm], and [wu2023hpsv2].
