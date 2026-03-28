# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_regression_sweep/control/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_regression_sweep/more_both/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `18`
- Compared complexes: `6`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `-0.008729` |
| Mean Final Loss Delta | `-` | `-` | `-0.003396` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `-0.006968` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `+0.053804` |
| Mean Training Seconds Delta | `-` | `-` | `+7.123` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.184201` | `1.193146` | `+0.008945` | `1.033504` | `1.050469` | `+0.016966` | `2.998` | `8.915` | `+5.917` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.078402` | `1.372333` | `+0.293931` | `0.801418` | `1.068009` | `+0.266591` | `3.944` | `11.880` | `+7.936` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `0.968042` | `1.222526` | `+0.254485` | `0.784783` | `0.940974` | `+0.156190` | `4.048` | `12.656` | `+8.608` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.454817` | `1.170896` | `-0.283922` | `0.863725` | `0.673660` | `-0.190065` | `3.985` | `12.285` | `+8.300` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.351718` | `0.967435` | `-0.384283` | `0.703451` | `0.807074` | `+0.103623` | `3.845` | `8.678` | `+4.833` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.113274` | `1.182307` | `+0.069034` | `0.972127` | `0.941649` | `-0.030479` | `3.585` | `10.727` | `+7.142` |

## Missing Pairs

No missing pairs.
