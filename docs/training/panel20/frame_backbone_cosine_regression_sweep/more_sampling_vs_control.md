# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_regression_sweep/control/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_regression_sweep/more_sampling/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `18`
- Compared complexes: `6`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `-0.005834` |
| Mean Final Loss Delta | `-` | `-` | `-0.007199` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `+0.181688` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `+0.163432` |
| Mean Training Seconds Delta | `-` | `-` | `+3.295` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.184201` | `1.308972` | `+0.124771` | `1.033504` | `1.179118` | `+0.145614` | `2.998` | `5.984` | `+2.986` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.078402` | `1.329853` | `+0.251451` | `0.801418` | `0.993338` | `+0.191920` | `3.944` | `8.081` | `+4.136` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `0.968042` | `1.559778` | `+0.591736` | `0.784783` | `1.023902` | `+0.239119` | `4.048` | `8.327` | `+4.279` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.454817` | `1.415476` | `-0.039342` | `0.863725` | `1.015042` | `+0.151317` | `3.985` | `8.191` | `+4.206` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.351718` | `1.456769` | `+0.105051` | `0.703451` | `0.839666` | `+0.136215` | `3.845` | `4.388` | `+0.542` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.113274` | `1.169733` | `+0.056460` | `0.972127` | `1.088533` | `+0.116406` | `3.585` | `7.204` | `+3.619` |

## Missing Pairs

No missing pairs.
