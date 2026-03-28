# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_tuning/control/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_tuning/longer_training/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `15`
- Compared complexes: `5`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `-0.013030` |
| Mean Final Loss Delta | `-` | `-` | `-0.188745` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `-0.213176` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `-0.149431` |
| Mean Training Seconds Delta | `-` | `-` | `+3.772` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `11gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.734251` | `1.218514` | `-0.515737` | `1.604558` | `1.114998` | `-0.489560` | `3.261` | `7.053` | `+3.792` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.353741` | `1.077274` | `-0.276466` | `0.716895` | `0.822547` | `+0.105652` | `3.916` | `8.394` | `+4.478` |
| `1a09` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.518716` | `1.428651` | `-0.090066` | `1.452755` | `1.341124` | `-0.111631` | `3.086` | `6.629` | `+3.543` |
| `1a0t` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.112861` | `1.065910` | `-0.046951` | `1.007047` | `0.941297` | `-0.065751` | `2.949` | `6.369` | `+3.420` |
| `1a1c` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.468280` | `1.331621` | `-0.136659` | `1.385959` | `1.200096` | `-0.185863` | `2.742` | `6.369` | `+3.626` |

## Missing Pairs

No missing pairs.
