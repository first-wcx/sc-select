# Method

## Problem formulation

Given a text prompt \(p\), a generator \(G\), and a candidate count \(N\), the generator produces a candidate pool \(\{I_1,\ldots,I_N\}\). SC-Select chooses one image from this pool without updating the generator parameters.

## Semantic contract

The prompt is represented as a semantic contract containing five constraint families:

- objects: entities that should appear in the image;
- attributes: properties bound to specific objects;
- relations: spatial or action relations between objects;
- counts: expected object quantities;
- texts: short strings expected to appear in the image.

The reproducible schema is defined in `configs/contract_schema.json`.

## Perception graph extraction

Each candidate image is parsed into a perception graph using a vision-language model. The graph records visible objects, attributes, relations, counts, OCR text, parser identity, and raw response. The reproducible schema is defined in `configs/perception_graph_schema.json`.

## Contract Satisfaction Rate

For each candidate, SC-Select computes five partial scores: CSR-Obj, CSR-Attr, CSR-Rel, CSR-Count, and CSR-Text. CSR-All averages the available constraint-family scores. Missing constraint types are ignored for prompts where that type is not specified.

## Selection

The selected image is:

\[
I^* = \arg\max_{I_i} CSR\text{-}All(I_i, p)
\]

This design makes the selection criterion interpretable: every selected image can be traced back to satisfied and violated contract items.

## Optional quality-aware extension

The main method uses CSR alone so that the selected image is traceable to explicit prompt constraints. A future quality-aware variant can combine CSR with an image quality or preference score:

\[
Score(I)=\alpha CSR(I)+(1-\alpha)Q(I)
\]

This extension is not used for the main reported results. It is included only to clarify how the selection signal could be combined with preference models in future work.
