# Adaptive Context Hard-Case Gate Decision

Control recipe:

- `--device cpu --steps 200 --sample-steps 25 --noise-schedule cosine --frame-hetero-backbone --ligand-bond-weight 0.1`

Hard-case subset:

- `184l`, `186l`, `187l`, `188l`, `13gs`, `1a28`
- seeds `42`, `43`, `44`

Candidates compared against fixed `8.0A`:

| Candidate | Mean Raw RMSE | Mean Aligned RMSD | Aligned Delta vs 8.0A | Raw Delta vs 8.0A | Improved Complexes | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed `8.0A` | `1.319712` | `1.002507` | `0.000000` | `0.000000` | `0/6` | control |
| fixed `10.0A` | `1.295312` | `0.962506` | `-0.040001` | `-0.024401` | `6/6` | fail |
| adaptive | `1.203613` | `0.930441` | `-0.072066` | `-0.116099` | `6/6` | fail |

Promotion gate:

- mean aligned RMSD delta `<= -0.08`
- at least `4/6` hard cases improve
- mean raw RMSE delta `<= +0.03`

Decision:

- No context candidate promoted to the canonical recommendation.
- `adaptive` is the strongest Stage A setting and becomes the Stage B control.
- No 20-complex confirmation was run, because the hard-case promotion gate was not cleared.
