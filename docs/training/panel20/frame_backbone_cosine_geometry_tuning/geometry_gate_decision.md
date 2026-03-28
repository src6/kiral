# Geometry Gate Decision

No geometry-aware training variant cleared the hard-case promotion gate. The current recommended frame-backbone cosine recipe remains unchanged:

- `--device cpu --steps 200 --sample-steps 25 --noise-schedule cosine --frame-hetero-backbone --ligand-bond-weight 0.1`

## Gate

- mean aligned RMSD delta must be at most `-0.08`
- at least `4/6` hard cases must improve on aligned RMSD
- mean raw RMSE delta must not be worse than `+0.03`

## Candidate Results

| Candidate | Mean Raw RMSE Delta | Mean Aligned RMSD Delta | Improved Complexes | Gate Result |
| --- | ---: | ---: | ---: | --- |
| `shape_0.02` | `-0.002438` | `-0.003337` | `4 / 6` | `fail` |
| `shape_0.05` | `-0.000385` | `-0.000617` | `4 / 6` | `fail` |
| `shape_0.10` | `+0.000661` | `+0.001431` | `4 / 6` | `fail` |

## Decision

- `shape_0.02` was the strongest geometry candidate, but its aggregate aligned-RMSD gain was only `-0.003337 A`, far below the required `-0.08 A`.
- The geometry loss changed the hard-case outcomes only marginally, and increasing the weight to `0.10` turned the aggregate result slightly negative.
- No full 20-complex geometry confirmation was run because no candidate justified promotion.
