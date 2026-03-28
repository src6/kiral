# Crop-Cutoff Gate Decision

The six-hard-case crop ablation on `184l`, `186l`, `187l`, `188l`, `13gs`, and `1a28` did not clear the promotion gate.

## Aggregate Summary

| Crop Cutoff | Mean Raw RMSE | Mean Aligned RMSD | Mean Training Seconds |
| --- | ---: | ---: | ---: |
| `6.0` | `1.350086` | `0.952654` | `2.524` |
| `8.0` | `1.314866` | `0.980977` | `4.674` |
| `10.0` | `1.325092` | `0.957343` | `8.541` |

Compared with the `8.0A` control:

- `6.0A` improved mean aligned RMSD by `-0.028323`, but worsened mean raw RMSE by `+0.035220`, which breaks the raw-RMSE guardrail.
- `10.0A` improved mean aligned RMSD by `-0.023634` and improved `5 / 6` hard cases, but still missed the required `-0.08` aligned-RMSD gate.

## Per-Complex Best-vs-Control Aligned RMSD

| Complex | `8.0A` Control | Best Crop | Best Aligned RMSD | Delta vs `8.0A` | Raw RMSE Delta |
| --- | ---: | --- | ---: | ---: | ---: |
| `13gs` | `1.126411` | `10.0A` | `1.078237` | `-0.048174` | `-0.108538` |
| `184l` | `0.943305` | `8.0A` | `0.943305` | `+0.000000` | `+0.000000` |
| `186l` | `0.934297` | `6.0A` | `0.807781` | `-0.126517` | `-0.000525` |
| `187l` | `0.948775` | `10.0A` | `0.882090` | `-0.066685` | `+0.133927` |
| `188l` | `0.914342` | `10.0A` | `0.857307` | `-0.057035` | `-0.012065` |
| `1a28` | `1.018732` | `10.0A` | `1.005505` | `-0.013227` | `+0.025419` |

## Decision

- No crop candidate is promoted.
- No full-panel crop confirmation is run.
- `10.0A` becomes the active control for the attention comparison because it improves `5 / 6` hard cases while keeping mean raw RMSE within the guardrail.
