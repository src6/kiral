# Protein-Node Gating Hard-Case Gate Decision

Control recipe:

- `--device cpu --steps 200 --sample-steps 25 --noise-schedule cosine --frame-hetero-backbone --ligand-bond-weight 0.1`

Hard-case subset:

- `184l`, `186l`, `187l`, `188l`, `13gs`, `1a28`
- seeds `42`, `43`, `44`

Candidates compared against fixed `8.0A`:

| Candidate | Mean Raw RMSE | Mean Aligned RMSD | Aligned Delta vs 8.0A | Raw Delta vs 8.0A | Aligned Delta vs Adaptive | Raw Delta vs Adaptive | Improved Complexes vs 8.0A | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fixed `8.0A` | `1.319712` | `1.002507` | `0.000000` | `0.000000` | `+0.072066` | `+0.116099` | `0/6` | control |
| adaptive | `1.203613` | `0.930441` | `-0.072066` | `-0.116099` | `0.000000` | `0.000000` | `6/6` | fail |
| gated `K=192` | `1.207976` | `0.926776` | `-0.075731` | `-0.111736` | `-0.003665` | `+0.004363` | `4/6` | fail |
| gated `K=256` | `1.308936` | `0.992060` | `-0.010446` | `-0.010777` | `+0.061619` | `+0.105322` | `5/6` | fail |
| gated `K=320` | `1.316610` | `0.998954` | `-0.003553` | `-0.003102` | `+0.068513` | `+0.112997` | `4/6` | fail |

Promotion gate:

- mean aligned RMSD delta `<= -0.08`
- at least `4/6` hard cases improve
- mean raw RMSE delta `<= +0.03`

Decision:

- No gated candidate promoted to the canonical recommendation.
- `K=192` is the strongest new node-gating result.
- `K=192` slightly improves aligned RMSD over the adaptive control, but only by `0.003665 A`, and still misses the fixed-`8.0A` aligned-RMSD threshold by `0.004269 A`.
- `K=256` and `K=320` are materially worse than adaptive, which indicates the useful direction, if any, is more aggressive pruning rather than looser gating.
- No 20-complex confirmation was run because the hard-case promotion gate was not cleared.
