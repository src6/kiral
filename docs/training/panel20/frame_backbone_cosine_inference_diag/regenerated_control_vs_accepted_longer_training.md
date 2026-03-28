# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_confirm/longer_training/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_inference_diag/control/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `18`
- Compared complexes: `6`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `-0.000113` |
| Mean Final Loss Delta | `-` | `-` | `+0.004286` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `+0.014386` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `+0.015198` |
| Mean Training Seconds Delta | `-` | `-` | `-0.370` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.200424` | `1.242786` | `+0.042362` | `1.105163` | `1.140373` | `+0.035210` | `6.493` | `6.092` | `-0.401` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.414339` | `1.422884` | `+0.008545` | `0.990572` | `1.001180` | `+0.010608` | `8.243` | `7.952` | `-0.291` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.575146` | `1.570165` | `-0.004981` | `0.992144` | `0.987369` | `-0.004775` | `8.605` | `8.199` | `-0.406` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.449760` | `1.447831` | `-0.001929` | `1.018061` | `1.022675` | `+0.004614` | `8.336` | `8.166` | `-0.171` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.091747` | `1.131247` | `+0.039501` | `0.843585` | `0.887060` | `+0.043476` | `8.344` | `7.977` | `-0.368` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.155772` | `1.158591` | `+0.002819` | `1.025143` | `1.027200` | `+0.002057` | `7.880` | `7.297` | `-0.583` |

## Missing Pairs

No missing pairs.
