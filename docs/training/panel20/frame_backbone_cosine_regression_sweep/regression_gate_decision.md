# Regression Gate Decision

No candidate cleared the hard-case promotion gate. The current recommended frame-backbone cosine recipe remains unchanged:

- `--device cpu --steps 200 --sample-steps 25 --noise-schedule cosine --frame-hetero-backbone --ligand-bond-weight 0.1`

## Gate

- mean aligned RMSD delta must be at most `-0.08`
- at least `4/6` hard cases must improve on aligned RMSD
- mean raw RMSE delta must be at most `+0.03`

## Candidate Results

| Candidate | Mean Raw RMSE Delta | Mean Aligned RMSD Delta | Improved Complexes | Gate Result |
| --- | ---: | ---: | ---: | --- |
| `more_sampling` | `+0.181688` | `+0.163432` | `0 / 6` | `fail` |
| `more_training` | `-0.036009` | `+0.049634` | `3 / 6` | `fail` |
| `more_both` | `-0.006968` | `+0.053804` | `2 / 6` | `fail` |
| `lower_lr` | `+0.133569` | `+0.150010` | `0 / 6` | `fail` |

## Decision

- No candidate improved the hard-case subset on the primary metric.
- `more_training` and `more_both` both improved mean raw RMSE, but both still worsened mean aligned RMSD and missed the `4/6` improvement requirement.
- Full 20-complex confirmation was not run because no candidate qualified for promotion.
