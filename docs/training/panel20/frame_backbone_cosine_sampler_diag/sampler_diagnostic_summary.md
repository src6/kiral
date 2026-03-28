# Sampler Diagnostic Summary

The hard-case sampler-redesign stage did not produce a promotable inference-only override for the accepted frame-backbone cosine recipe.

## Main Findings

- The strongest variant was `sample_time_power_3p0`, with mean raw RMSE delta `-0.035016`, mean aligned RMSD delta `-0.031689`, and `4 / 6` hard-case improvements on aligned RMSD.
- `sample_time_power_3p0` still failed the promotion gate because the mean aligned RMSD improvement did not reach the required `-0.05 A`.
- `sample_steps_50` improved mean raw RMSE by `-0.034748`, but mean aligned RMSD worsened by `+0.008291`, so it failed on the primary metric.
- `sample_steps_100` also reduced raw RMSE slightly while worsening aligned RMSD.
- `score_clip_20`, `position_clip_25`, and `position_clip_100` were exact no-ops at the aggregate level on this subset.

## Interpretation

- Hard-case sensitivity is present in the reverse-time schedule, not in the clip thresholds.
- Power-respacing produced the only meaningful aligned-RMSD gains, but those gains were not large enough to justify a full-panel override.
- The accepted recommendation therefore stays training-driven rather than inference-driven.

## Reference Reports

- `docs/training/panel20/frame_backbone_cosine_sampler_diag/inference_gate_decision.md`
- `docs/training/panel20/frame_backbone_cosine_sampler_diag/sample_time_power_3p0_vs_control.md`
- `docs/training/panel20/frame_backbone_cosine_sampler_diag/sample_time_power_3p0_sampler_internal_vs_control.md`
