# Complex Diagnostics

## Hard-Case Rows

| Complex | Heavy Atoms | Bonds | Mean Degree | Max Degree | Cyclomatic | Ligand Radius | Max Span | Cropped Protein Nodes | Contacts <=4.5A | Contacts <=8.0A | Centroid Distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `184l` | `10` | `10` | `2.000` | `3` | `1` | `3.556` | `6.368` | `423` | `223` | `2034` | `28.024` |
| `186l` | `10` | `10` | `2.000` | `3` | `1` | `3.519` | `6.082` | `436` | `239` | `2124` | `28.222` |
| `187l` | `8` | `8` | `2.000` | `3` | `1` | `2.901` | `5.786` | `439` | `168` | `1735` | `28.249` |
| `188l` | `8` | `8` | `2.000` | `3` | `1` | `2.379` | `4.330` | `432` | `182` | `1687` | `28.230` |
| `13gs` | `28` | `30` | `2.143` | `4` | `3` | `7.922` | `14.478` | `267` | `322` | `2506` | `29.263` |
| `1a28` | `23` | `26` | `2.261` | `4` | `4` | `6.186` | `11.949` | `406` | `316` | `3774` | `65.379` |

## Cohort Contrast

| Metric | Hard Cases Mean | Comparison Mean |
| --- | ---: | ---: |
| Heavy atoms | `14.500` | `33.357` |
| Mean degree | `2.067` | `2.042` |
| Cyclomatic number | `1.833` | `1.571` |
| Ligand radius | `4.410` | `7.783` |
| Cropped protein nodes | `400.500` | `257.714` |
| Contacts <=4.5A | `241.667` | `357.786` |
| Centroid distance | `34.561` | `41.136` |

## Diagnostic Note

hard cases trend toward smaller ligands; hard cases trend toward smaller ligand spatial radius; hard cases trend toward denser cropped protein neighborhoods.
