# Inference Gate Decision

No inference-only variant cleared the hard-case promotion gate. The current recommended frame-backbone cosine recipe remains unchanged:

- `--device cpu --steps 200 --sample-steps 25 --noise-schedule cosine --frame-hetero-backbone --ligand-bond-weight 0.1`

## Gate

- mean aligned RMSD delta must be at most `-0.05`
- at least `4/6` hard cases must improve on aligned RMSD
- mean raw RMSE delta must not be worse than `+0.03`

## Candidate Results

| Candidate | Mean Raw RMSE Delta | Mean Aligned RMSD Delta | Improved Complexes | Gate Result |
| --- | ---: | ---: | ---: | --- |
| `sample_time_power_1p5` | `-0.002136` | `-0.003455` | `2 / 6` | `fail` |
| `sample_time_power_2p0` | `-0.002888` | `-0.008700` | `2 / 6` | `fail` |
| `sample_time_power_3p0` | `-0.035016` | `-0.031689` | `4 / 6` | `fail` |
| `sample_steps_50` | `-0.034748` | `+0.008291` | `2 / 6` | `fail` |
| `sample_steps_100` | `-0.011227` | `+0.017131` | `2 / 6` | `fail` |
| `score_clip_5` | `+0.005043` | `+0.005122` | `1 / 6` | `fail` |
| `score_clip_20` | `+0.000000` | `+0.000000` | `0 / 6` | `fail` |
| `position_clip_25` | `+0.000000` | `+0.000000` | `0 / 6` | `fail` |
| `position_clip_100` | `+0.000000` | `+0.000000` | `0 / 6` | `fail` |

## Decision

- No variant improved the hard cases on the primary metric strongly enough to justify promotion.
- `sample_time_power_3p0` was the strongest sampler-redesign candidate because it improved `4 / 6` complexes and reduced mean aligned RMSD by `0.031689 A`, but it still missed the required aggregate aligned-RMSD threshold of `-0.05 A`.
- The score-clip and position-clip variants were inert or harmful on this subset, so the remaining leverage appears to be in the reverse-time schedule rather than clipping thresholds.
- Full 20-complex confirmation was not run because no inference-only change qualified for promotion.
