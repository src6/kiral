# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_inference_diag/control/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_inference_diag/sample_steps_100/*_log.md`
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
| Mean Raw Ligand RMSE Delta | `-` | `-` | `-0.011227` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `+0.017131` |
| Mean Training Seconds Delta | `-` | `-` | `+0.000` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.242786` | `1.352852` | `+0.110066` | `1.140373` | `1.192003` | `+0.051629` | `6.092` | `6.092` | `+0.000` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.422884` | `1.333661` | `-0.089223` | `1.001180` | `1.004196` | `+0.003016` | `7.952` | `7.952` | `+0.000` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.570165` | `1.479657` | `-0.090508` | `0.987369` | `0.959207` | `-0.028162` | `8.199` | `8.199` | `+0.000` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.447831` | `1.342292` | `-0.105539` | `1.022675` | `0.906255` | `-0.116420` | `8.166` | `8.166` | `+0.000` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.131247` | `1.227502` | `+0.096255` | `0.887060` | `1.018516` | `+0.131456` | `7.977` | `7.977` | `+0.000` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.158591` | `1.170181` | `+0.011590` | `1.027200` | `1.088466` | `+0.061266` | `7.297` | `7.297` | `+0.000` |

## Missing Pairs

No missing pairs.
