# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_tuning/control/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_tuning/longer_both/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `15`
- Compared complexes: `5`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `-0.013200` |
| Mean Final Loss Delta | `-` | `-` | `-0.167213` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `-0.229795` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `-0.147132` |
| Mean Training Seconds Delta | `-` | `-` | `+3.747` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `11gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.734251` | `1.346063` | `-0.388188` | `1.604558` | `1.213630` | `-0.390928` | `3.261` | `7.124` | `+3.863` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.353741` | `0.967435` | `-0.386306` | `0.716895` | `0.807074` | `+0.090179` | `3.916` | `8.678` | `+4.762` |
| `1a09` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.518716` | `1.425817` | `-0.092899` | `1.452755` | `1.325300` | `-0.127455` | `3.086` | `6.720` | `+3.634` |
| `1a0t` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.112861` | `0.973712` | `-0.139149` | `1.007047` | `0.883546` | `-0.123501` | `2.949` | `6.289` | `+3.339` |
| `1a1c` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.468280` | `1.325845` | `-0.142435` | `1.385959` | `1.202004` | `-0.183955` | `2.742` | `5.880` | `+3.137` |

## Missing Pairs

No missing pairs.
