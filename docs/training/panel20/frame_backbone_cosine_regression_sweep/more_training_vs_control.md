# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_regression_sweep/control/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_regression_sweep/more_training/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `18`
- Compared complexes: `6`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `-0.008793` |
| Mean Final Loss Delta | `-` | `-` | `-0.005555` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `-0.036009` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `+0.049634` |
| Mean Training Seconds Delta | `-` | `-` | `+7.125` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.184201` | `1.159580` | `-0.024621` | `1.033504` | `1.011157` | `-0.022347` | `2.998` | `8.964` | `+5.966` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.078402` | `1.144979` | `+0.066576` | `0.801418` | `0.959871` | `+0.158452` | `3.944` | `11.904` | `+7.960` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `0.968042` | `1.214645` | `+0.246603` | `0.784783` | `1.029911` | `+0.245128` | `4.048` | `12.673` | `+8.625` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.454817` | `1.177723` | `-0.277095` | `0.863725` | `0.727673` | `-0.136052` | `3.985` | `12.446` | `+8.462` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.351718` | `1.077274` | `-0.274444` | `0.703451` | `0.822547` | `+0.119096` | `3.845` | `8.394` | `+4.549` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.113274` | `1.160197` | `+0.046924` | `0.972127` | `0.905653` | `-0.066474` | `3.585` | `10.773` | `+7.187` |

## Missing Pairs

No missing pairs.
