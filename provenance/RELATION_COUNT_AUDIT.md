# Relation and Count Audit

The low relation/count numbers are retained as measured results, not edited. The audit points to three likely causes: strict matching rules, limited parser reliability for spatial/count extraction, and genuinely difficult relation/count prompt subsets.

| Prompt family | Candidates | Mean CSR-All | Mean CSR-Rel | Mean CSR-Count |
| --- | ---: | ---: | ---: | ---: |
| attr | 200 | 0.4917 | N/A | N/A |
| combo | 200 | 0.2427 | 0.0200 | 0.0250 |
| count | 200 | 0.2487 | N/A | 0.0225 |
| obj | 200 | 0.4214 | N/A | N/A |
| rel | 200 | 0.3187 | 0.0600 | N/A |
| text | 200 | 0.3717 | 0.0000 | N/A |

Interpretation:
- The scorer requires exact or alias-normalized matches between contract items and parsed graph fields; this is intentionally conservative.
- Relations and counts are more fragile than object presence under VLM parsing. A correct-looking image may still receive no relation/count credit if the parser omits the relation or count.
- The paper should frame these low sub-scores as a limitation and as evidence that SC-Select is most reliable when parser-visible constraints are available.
- No additional GPU experiment is required for this audit; it uses completed candidate score CSV files.
