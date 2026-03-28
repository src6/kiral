# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_confirm/longer_training/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_geometry_tuning/shape_0_05/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `18`
- Compared complexes: `6`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `+0.000192` |
| Mean Final Loss Delta | `-` | `-` | `+0.002753` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `-0.000385` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `-0.000617` |
| Mean Training Seconds Delta | `-` | `-` | `-0.399` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.200424` | `1.205694` | `+0.005270` | `1.105163` | `1.111104` | `+0.005941` | `6.493` | `6.094` | `-0.399` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.414339` | `1.406368` | `-0.007971` | `0.990572` | `0.981688` | `-0.008885` | `8.243` | `7.763` | `-0.480` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.575146` | `1.566380` | `-0.008766` | `0.992144` | `0.980724` | `-0.011420` | `8.605` | `8.368` | `-0.237` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.449760` | `1.441912` | `-0.007848` | `1.018061` | `1.012290` | `-0.005771` | `8.336` | `8.142` | `-0.194` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.091747` | `1.110251` | `+0.018504` | `0.843585` | `0.860518` | `+0.016933` | `8.344` | `7.711` | `-0.633` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.155772` | `1.154272` | `-0.001500` | `1.025143` | `1.024642` | `-0.000501` | `7.880` | `7.429` | `-0.451` |

## Missing Pairs

No missing pairs.
