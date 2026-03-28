# Regression Subset vs Full Confirmation

The accepted frame-backbone cosine recipe improved the full 20-complex panel overall, but the six-complex hard-case subset remained a clear regression pocket.

## Aggregate Comparison

| Scope | Mean Raw RMSE Delta | Mean Aligned RMSD Delta |
| --- | ---: | ---: |
| Full 20-complex confirmation | `-0.042488` | `-0.064899` |
| Hard-case 6-complex subset | `+0.122789` | `+0.135943` |

Negative deltas are better because they mean the accepted `longer_training` recipe outperformed the prior frame-backbone cosine control. The hard-case subset flips sign on both metrics, which is why this follow-on sweep focused only on these complexes.

## Prior Confirmation Deltas for the Hard Cases

| Complex | Raw RMSE Delta | Aligned RMSD Delta |
| --- | ---: | ---: |
| `13gs` | `+0.016223` | `+0.071660` |
| `184l` | `+0.335937` | `+0.189154` |
| `186l` | `+0.607104` | `+0.207361` |
| `187l` | `-0.005057` | `+0.154335` |
| `188l` | `-0.259971` | `+0.140134` |
| `1a28` | `+0.042498` | `+0.053016` |

These six rows are the justification for the regression-focused sweep in this directory.
