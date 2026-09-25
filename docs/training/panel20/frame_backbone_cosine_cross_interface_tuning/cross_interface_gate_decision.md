# Cross-Interface Block Hard-Case Gate Decision

Stage B control:

- context setting: `adaptive`
- control tag: `context_hardcase_adaptive_v2`

Hard-case subset:

- `184l`, `186l`, `187l`, `188l`, `13gs`, `1a28`
- seeds `42`, `43`, `44`

Cross-interface results against the adaptive control:

| Candidate | Mean Raw RMSE | Mean Aligned RMSD | Aligned Delta | Raw Delta | Improved Complexes | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| control (`xmsg=0`) | `1.203613` | `0.930441` | `0.000000` | `0.000000` | control | control |
| `xmsg=1` | `1.407141` | `1.014673` | `+0.084232` | `+0.203528` | `3/6` | fail |

Per-complex aligned RMSD deltas versus the adaptive control:

| Complex | Control | Cross-Interface | Delta |
| --- | ---: | ---: | ---: |
| `13gs` | `1.096622` | `1.010226` | `-0.086396` |
| `184l` | `0.811168` | `0.962429` | `+0.151261` |
| `186l` | `0.831676` | `1.047598` | `+0.215921` |
| `187l` | `0.952977` | `0.919800` | `-0.033177` |
| `188l` | `0.882999` | `1.162508` | `+0.279510` |
| `1a28` | `1.007204` | `0.985479` | `-0.021725` |

Promotion gate:

- mean aligned RMSD delta `<= -0.08`
- at least `4/6` hard cases improve
- mean raw RMSE delta `<= +0.03`

Decision:

- The v1 cross-interface block is not promoted.
- The candidate regressed the adaptive control on both aggregate metrics and failed the improvement-count threshold.
- The largest regressions were on `184l`, `186l`, and `188l`, which outweigh the gains on `13gs`, `187l`, and `1a28`.
- No 20-complex confirmation was run because the hard-case promotion gate failed decisively.
