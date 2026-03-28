# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_sampler_diag/control/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_sampler_diag/sample_time_power_2p0/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `18`
- Compared complexes: `6`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `+0.000000` |
| Mean Final Loss Delta | `-` | `-` | `+0.000000` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `-0.002888` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `-0.008700` |
| Mean Training Seconds Delta | `-` | `-` | `+0.000` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.242786` | `1.249348` | `+0.006562` | `1.140373` | `1.151677` | `+0.011304` | `6.092` | `6.092` | `+0.000` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.422884` | `1.394923` | `-0.027961` | `1.001180` | `1.001227` | `+0.000047` | `7.952` | `7.952` | `+0.000` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.570165` | `1.572682` | `+0.002517` | `0.987369` | `0.898006` | `-0.089363` | `8.199` | `8.199` | `+0.000` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.447831` | `1.336194` | `-0.111637` | `1.022675` | `0.913601` | `-0.109074` | `8.166` | `8.166` | `+0.000` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.131247` | `1.200554` | `+0.069307` | `0.887060` | `0.984707` | `+0.097647` | `7.977` | `7.977` | `+0.000` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.158591` | `1.202473` | `+0.043882` | `1.027200` | `1.064443` | `+0.037243` | `7.297` | `7.297` | `+0.000` |

## Missing Pairs

No missing pairs.
