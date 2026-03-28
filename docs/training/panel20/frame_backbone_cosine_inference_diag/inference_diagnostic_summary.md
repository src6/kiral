# Hard-Case Inference Diagnostic Summary

This directory isolates inference-time changes on the six hard cases from the accepted frame-backbone cosine recipe.

## Main Findings

- Regenerating the accepted recipe from checkpoints introduced only small drift versus the historical accepted logs.
  - See `regenerated_control_vs_accepted_longer_training.md`
  - Mean raw RMSE drift: `+0.014386`
  - Mean aligned RMSD drift: `+0.015198`
- Only `sample_steps` materially changed the hard-case metrics.
  - `sample_steps_50` improved mean raw RMSE by `-0.034748`, but still worsened mean aligned RMSD by `+0.008291`
  - `sample_steps_100` improved only two complexes on aligned RMSD and worsened the subset mean aligned RMSD by `+0.017131`
- Score and position clipping were mostly inactive on this subset.
  - `score_clip_20`, `position_clip_25`, and `position_clip_100` reproduced the control exactly
  - `score_clip_5` changed only one complex slightly and did not improve the subset

## Practical Interpretation

- The hard-case regressions are not fixed by simple reverse-diffusion knob changes alone.
- The only inference lever with visible effect was `sample_steps`, which suggests the main sensitivity is the integration path rather than clipping saturation.
- Because the best inference-only candidate still failed the gate, the current frame-backbone cosine recommendation remains unchanged.
