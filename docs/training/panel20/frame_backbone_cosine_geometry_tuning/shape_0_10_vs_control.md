# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_confirm/longer_training/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_geometry_tuning/shape_0_10/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `18`
- Compared complexes: `6`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `-0.000078` |
| Mean Final Loss Delta | `-` | `-` | `+0.001141` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `+0.000661` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `+0.001431` |
| Mean Training Seconds Delta | `-` | `-` | `-0.377` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.200424` | `1.189350` | `-0.011074` | `1.105163` | `1.091977` | `-0.013186` | `6.493` | `6.266` | `-0.227` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.414339` | `1.410233` | `-0.004106` | `0.990572` | `0.988159` | `-0.002414` | `8.243` | `7.897` | `-0.346` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.575146` | `1.573436` | `-0.001710` | `0.992144` | `0.989754` | `-0.002390` | `8.605` | `8.079` | `-0.526` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.449760` | `1.447828` | `-0.001932` | `1.018061` | `1.018682` | `+0.000621` | `8.336` | `8.077` | `-0.260` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.091747` | `1.120053` | `+0.028307` | `0.843585` | `0.874837` | `+0.031252` | `8.344` | `7.940` | `-0.405` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.155772` | `1.150252` | `-0.005520` | `1.025143` | `1.019845` | `-0.005298` | `7.880` | `7.378` | `-0.502` |

## Missing Pairs

No missing pairs.
