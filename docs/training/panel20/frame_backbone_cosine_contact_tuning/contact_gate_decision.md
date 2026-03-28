# Contact Prior Hard-Case Gate Decision

Stage B control:

- context setting: `adaptive`
- control tag: `context_hardcase_adaptive_v2`

Hard-case subset:

- `184l`, `186l`, `187l`, `188l`, `13gs`, `1a28`
- seeds `42`, `43`, `44`

Contact-prior results against the adaptive control:

| Candidate | Mean Raw RMSE | Mean Aligned RMSD | Aligned Delta | Raw Delta | Improved Complexes | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| control (`0.00`) | `1.203613` | `0.930441` | `0.000000` | `0.000000` | control | control |
| `0.01` | `1.202368` | `0.928874` | `-0.001567` | `-0.001245` | `4/6` | fail |
| `0.02` | `1.193114` | `0.918585` | `-0.011856` | `-0.010499` | `6/6` | fail |
| `0.05` | `1.200573` | `0.929878` | `-0.000563` | `-0.003040` | `3/6` | fail |

Promotion gate:

- mean aligned RMSD delta `<= -0.08`
- at least `4/6` hard cases improve
- mean raw RMSE delta `<= +0.03`

Decision:

- No contact-weight candidate promoted.
- `0.02` is the best Stage B result, but the gain is too small to justify escalation.
- Stage C combined physics run was skipped because Stage B did not show a materially strong signal.
- No 20-complex confirmation was run.
