# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_sampler_diag/control/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_sampler_diag/sample_time_power_1p5/*_log.md`
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
| Mean Raw Ligand RMSE Delta | `-` | `-` | `-0.002136` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `-0.003455` |
| Mean Training Seconds Delta | `-` | `-` | `+0.000` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.242786` | `1.244149` | `+0.001363` | `1.140373` | `1.143297` | `+0.002923` | `6.092` | `6.092` | `+0.000` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.422884` | `1.408043` | `-0.014842` | `1.001180` | `1.012097` | `+0.010917` | `7.952` | `7.952` | `+0.000` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.570165` | `1.563222` | `-0.006943` | `0.987369` | `0.923961` | `-0.063408` | `8.199` | `8.199` | `+0.000` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.447831` | `1.392809` | `-0.055021` | `1.022675` | `0.971511` | `-0.051164` | `8.166` | `8.166` | `+0.000` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.131247` | `1.166457` | `+0.035210` | `0.887060` | `0.940853` | `+0.053793` | `7.977` | `7.977` | `+0.000` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.158591` | `1.186008` | `+0.027416` | `1.027200` | `1.053408` | `+0.026208` | `7.297` | `7.297` | `+0.000` |

## Missing Pairs

No missing pairs.
