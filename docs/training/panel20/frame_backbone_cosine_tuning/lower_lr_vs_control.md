# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_tuning/control/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_tuning/lower_lr/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `15`
- Compared complexes: `5`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `+0.007640` |
| Mean Final Loss Delta | `-` | `-` | `+0.073998` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `+0.067208` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `+0.053382` |
| Mean Training Seconds Delta | `-` | `-` | `+0.227` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `11gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.734251` | `1.712416` | `-0.021835` | `1.604558` | `1.583669` | `-0.020890` | `3.261` | `3.598` | `+0.338` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.353741` | `1.339298` | `-0.014442` | `0.716895` | `0.685221` | `-0.031674` | `3.916` | `4.085` | `+0.169` |
| `1a09` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.518716` | `1.648028` | `+0.129312` | `1.452755` | `1.590750` | `+0.137995` | `3.086` | `3.211` | `+0.125` |
| `1a0t` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.112861` | `1.201452` | `+0.088591` | `1.007047` | `1.091447` | `+0.084400` | `2.949` | `3.189` | `+0.239` |
| `1a1c` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.468280` | `1.622696` | `+0.154416` | `1.385959` | `1.483036` | `+0.097078` | `2.742` | `3.003` | `+0.261` |

## Missing Pairs

No missing pairs.
