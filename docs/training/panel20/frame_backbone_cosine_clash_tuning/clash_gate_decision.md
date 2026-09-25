# Clash-Prior Gate Decision

The hard-case clash-prior sweep on `184l`, `186l`, `187l`, `188l`, `13gs`, and `1a28` did not clear the promotion gate.

## Aggregate Summary

| Clash Weight | Mean Raw RMSE | Mean Aligned RMSD | Mean Training Seconds |
| --- | ---: | ---: | ---: |
| `0.00` | `1.294509` | `0.958256` | `6.237` |
| `0.01` | `1.297632` | `0.962931` | `6.269` |
| `0.02` | `1.294660` | `0.959002` | `6.272` |
| `0.05` | `1.291970` | `0.956771` | `6.262` |

The best aggregate aligned RMSD came from `--ligand-protein-clash-weight 0.05`, improving the hard-case mean from `0.958256` to `0.956771`. That delta is only `-0.001485`, far short of the required `-0.08` gate.

## Per-Complex Best-vs-Control Aligned RMSD

| Complex | Control | Best Clash Weight | Best Aligned RMSD | Delta vs Control |
| --- | ---: | --- | ---: | ---: |
| `13gs` | `1.104628` | `0.02` | `1.082812` | `-0.021816` |
| `184l` | `0.952836` | `0.00` | `0.952836` | `+0.000000` |
| `186l` | `0.928331` | `0.02` | `0.919736` | `-0.008595` |
| `187l` | `0.906259` | `0.02` | `0.902592` | `-0.003667` |
| `188l` | `0.845316` | `0.00` | `0.845316` | `+0.000000` |
| `1a28` | `1.012165` | `0.05` | `1.002107` | `-0.010058` |

Only four complexes improved at all, and the total improvement was too small to justify promotion.

## Decision

- No clash-prior candidate is promoted.
- No 20-complex clash-prior confirmation is run.
- The next track should pivot to context selection and message-level attention rather than further clash-weight sweeps on this branch.
