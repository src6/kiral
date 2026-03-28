# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_inference_diag/control/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_inference_diag/sample_steps_50/*_log.md`
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
| Mean Raw Ligand RMSE Delta | `-` | `-` | `-0.034748` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `+0.008291` |
| Mean Training Seconds Delta | `-` | `-` | `+0.000` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.242786` | `1.314834` | `+0.072048` | `1.140373` | `1.186262` | `+0.045889` | `6.092` | `6.092` | `+0.000` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.422884` | `1.351567` | `-0.071318` | `1.001180` | `1.014146` | `+0.012966` | `7.952` | `7.952` | `+0.000` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.570165` | `1.540036` | `-0.030129` | `0.987369` | `1.001021` | `+0.013652` | `8.199` | `8.199` | `+0.000` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.447831` | `1.414332` | `-0.033499` | `1.022675` | `1.015211` | `-0.007464` | `8.166` | `8.166` | `+0.000` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.131247` | `0.983268` | `-0.147980` | `0.887060` | `0.820396` | `-0.066664` | `7.977` | `7.977` | `+0.000` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.158591` | `1.160978` | `+0.002387` | `1.027200` | `1.078564` | `+0.051364` | `7.297` | `7.297` | `+0.000` |

## Missing Pairs

No missing pairs.
