# Subset Gate Decision

This note records the 5-complex frame-backbone cosine sweep against the reused CPU control set.

| Candidate | Mean Raw RMSE Delta | Mean Aligned RMSD Delta | Success@2A Delta | Passes Gate |
| --- | ---: | ---: | ---: | --- |
| longer_training | `-0.213176` | `-0.149431` | `+0.0%` | yes |
| longer_sampling | `+0.021932` | `+0.027151` | `+0.0%` | no |
| longer_both | `-0.229795` | `-0.147132` | `+0.0%` | yes |
| lower_lr | `+0.067208` | `+0.053382` | `+0.0%` | no |

Promoted candidate: `longer_training`.
Reason: it achieved the largest aligned-RMSD improvement among the configs that preserved Success@2A and did not worsen raw RMSE.
