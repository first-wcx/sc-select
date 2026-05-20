# Limitations

SC-Select is a post-generation selection method, not a generator training method. It cannot create semantic content that is absent from all candidates in the pool.

The method depends on the accuracy of the perception graph extractor. If the vision-language parser misses an object, hallucinates an attribute, or misreads text, CSR may reward or penalize the wrong candidate.

CSR is a task-specific metric. Current results show improvements on CSR-All, GenEval official scores, reward-model comparisons, and a single-response human preference check, but a stronger multi-annotator study is still needed before making broad claims about human preference.

Generating multiple candidates increases cost. Best-of-4 can improve instruction adherence, but it requires more generation, parsing, and scoring work than a single output.

The 300-prompt self-built dataset has been manually reviewed, but it is still template-derived. Future work should diversify prompt wording and include additional independent annotators for a stronger benchmark-quality release.
