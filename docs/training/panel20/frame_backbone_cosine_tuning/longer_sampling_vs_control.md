# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_tuning/control/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_tuning/longer_sampling/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `15`
- Compared complexes: `5`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `-0.000076` |
| Mean Final Loss Delta | `-` | `-` | `+0.000959` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `+0.021932` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `+0.027151` |
| Mean Training Seconds Delta | `-` | `-` | `+0.300` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `11gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.734251` | `1.754289` | `+0.020038` | `1.604558` | `1.640198` | `+0.035639` | `3.261` | `3.501` | `+0.241` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.353741` | `1.456769` | `+0.103028` | `0.716895` | `0.839666` | `+0.122771` | `3.916` | `4.388` | `+0.472` |
| `1a09` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.518716` | `1.478795` | `-0.039922` | `1.452755` | `1.413951` | `-0.038804` | `3.086` | `3.421` | `+0.335` |
| `1a0t` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.112861` | `1.184949` | `+0.072088` | `1.007047` | `1.060039` | `+0.052992` | `2.949` | `3.124` | `+0.174` |
| `1a1c` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.468280` | `1.422706` | `-0.045574` | `1.385959` | `1.349114` | `-0.036844` | `2.742` | `3.019` | `+0.276` |

## Missing Pairs

No missing pairs.
