# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_confirm/longer_training/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_geometry_tuning/shape_0_02/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `18`
- Compared complexes: `6`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `+0.000079` |
| Mean Final Loss Delta | `-` | `-` | `-0.002648` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `-0.002438` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `-0.003337` |
| Mean Training Seconds Delta | `-` | `-` | `-0.401` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.200424` | `1.184095` | `-0.016329` | `1.105163` | `1.083348` | `-0.021816` | `6.493` | `6.277` | `-0.216` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.414339` | `1.400448` | `-0.013892` | `0.990572` | `0.974778` | `-0.015794` | `8.243` | `7.812` | `-0.431` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.575146` | `1.575954` | `+0.000808` | `0.992144` | `0.994781` | `+0.002637` | `8.605` | `8.098` | `-0.507` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.449760` | `1.443219` | `-0.006541` | `1.018061` | `1.015951` | `-0.002109` | `8.336` | `7.989` | `-0.348` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.091747` | `1.118906` | `+0.027159` | `0.843585` | `0.868050` | `+0.024465` | `8.344` | `7.973` | `-0.371` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.155772` | `1.149937` | `-0.005835` | `1.025143` | `1.017740` | `-0.007403` | `7.880` | `7.347` | `-0.534` |

## Missing Pairs

No missing pairs.
