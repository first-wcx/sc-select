# Conclusion

SC-Select provides a training-free, interpretable selection signal for complex text-to-image prompts. By decomposing prompts into semantic contracts and matching them against perception graphs extracted from candidate images, the method targets fine-grained instruction adherence in a way that is more explicit than global image-text similarity.

Audited results show positive CSR-All gains on the reviewed 300-prompt SDXL set, GenEval, the FLUX stratified-60 subset, and an auxiliary T2I-CompBench subset. SC-Select also improves GenEval official detector scores and remains effective when the FLUX stratified-60 candidate pool is re-parsed with InternVL2.5-4B. The strongest defensible conclusion is that semantic contract-guided selection can improve final-output instruction adherence in multi-candidate settings when suitable candidates exist.

Future work should extend reward-model baselines to more public benchmarks, collect multi-annotator human judgments, and broaden the generator set beyond the current SDXL and FLUX evidence before making stronger claims about generality.
