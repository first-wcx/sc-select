# Introduction

Modern text-to-image generators can produce visually compelling images, but complex instruction following remains fragile. A prompt such as "a red robot holding a blue umbrella beside a yellow car" requires the model to satisfy object existence, attribute binding, relation, and sometimes counting or text-rendering constraints at the same time. A single generated image may be plausible overall while still violating one or more fine-grained constraints.

SC-Select frames this issue as a post-generation selection problem. Given a prompt and a pool of candidate images, the goal is not to train a stronger generator, but to select the candidate that best satisfies the prompt's explicit semantic constraints. The proposed pipeline decomposes the prompt into a semantic contract, uses a vision-language model to extract a perception graph from each candidate, computes contract satisfaction rates for several constraint types, and selects the candidate with the highest CSR-All score.

The current audited evidence supports a conservative claim: in multi-candidate settings, explicit contract-guided selection improves measured instruction adherence over fixed single-candidate selection on the available result files. On the self-built 50-prompt set, CSR-All improves from 0.5499 for Fixed Seed-0 to 0.7456 for CSR-Select. On audited FLUX GenEval outputs, CSR-All improves from 0.6650 to 0.7880. On GenEval official detector scores, the overall score improves from 0.53885 to 0.62232.

This paper should avoid claiming that SC-Select improves image realism or generator capability. The method is a selection signal over an existing candidate pool. Its practical value depends on candidate diversity, parser reliability, and the cost of generating and evaluating multiple candidates.

