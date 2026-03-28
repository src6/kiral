# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_regression_sweep/control/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_regression_sweep/lower_lr/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `18`
- Compared complexes: `6`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `-0.006120` |
| Mean Final Loss Delta | `-` | `-` | `-0.016547` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `+0.133569` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `+0.150010` |
| Mean Training Seconds Delta | `-` | `-` | `+3.860` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.184201` | `1.231594` | `+0.047393` | `1.033504` | `1.142688` | `+0.109185` | `2.998` | `5.948` | `+2.950` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.078402` | `1.437871` | `+0.359469` | `0.801418` | `1.020598` | `+0.219180` | `3.944` | `7.963` | `+4.018` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `0.968042` | `1.569498` | `+0.601456` | `0.784783` | `0.982099` | `+0.197315` | `4.048` | `8.526` | `+4.478` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.454817` | `1.433943` | `-0.020875` | `0.863725` | `1.009925` | `+0.146200` | `3.985` | `7.996` | `+4.011` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.351718` | `1.097829` | `-0.253889` | `0.703451` | `0.850993` | `+0.147543` | `3.845` | `7.936` | `+4.090` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.113274` | `1.181134` | `+0.067861` | `0.972127` | `1.052766` | `+0.080638` | `3.585` | `7.195` | `+3.610` |

## Missing Pairs

No missing pairs.
