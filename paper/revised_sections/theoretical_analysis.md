# Theoretical analysis

## Candidate pool effect

If a single candidate satisfies a constraint with probability \(p\), and candidates are sampled independently, the probability that at least one of \(N\) candidates satisfies the constraint is:

\[
P_N = 1-(1-p)^N
\]

This explains why Best-of-N selection can improve prompt adherence without changing the generator. Increasing \(N\) increases the chance that the pool contains at least one candidate satisfying the target constraint.

The gain is not unlimited. The marginal improvement from increasing \(N\) decreases as \(N\) grows, and the cost of generation and evaluation grows approximately linearly with the candidate count.

## Why contract-guided selection differs from global scores

Global image-text similarity and preference scores can reward images that are semantically plausible or aesthetically strong while still missing a fine-grained requirement. A contract-guided score instead decomposes the prompt into checkable items and evaluates each item explicitly.

This design is useful for complex prompts because object presence, attribute binding, relations, counts, and text rendering often fail independently. The selected image is therefore chosen by local constraint satisfaction rather than by a single holistic score.

