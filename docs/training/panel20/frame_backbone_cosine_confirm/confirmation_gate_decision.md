# Full Panel Confirmation

This note compares the promoted `longer_training` frame-backbone cosine config against the CPU control panel on the full 20-complex manifest with seeds `42`, `43`, and `44`.

- Mean raw RMSE delta: `-0.042488`
- Mean aligned RMSD delta: `-0.064899`
- Success@2A delta: `+0.0%`

Acceptance decision: accept.
The promoted config clears the full-panel gate because it improves mean aligned RMSD by at least `0.02 A` with no drop in Success@2A.
