# Hard-Case Sampler Gate Decision

## Scope

- Baseline for this stage: regenerated checkpoint control using the accepted frame-backbone cosine training recipe
  - `--device cpu --steps 200 --sample-steps 25 --noise-schedule cosine --frame-hetero-backbone --ligand-bond-weight 0.1`
- Hard-case subset:
  - `184l`, `186l`, `187l`, `188l`, `13gs`, `1a28`
- Seeds:
  - `42`, `43`, `44`
- Inference-only candidates:
  - `sample_time_power=1.5`
  - `sample_time_power=2.0`
  - `sample_time_power=3.0`
  - `sample_steps=50`
  - `sample_steps=100`
  - `sample_score_clip=5.0`
  - `sample_score_clip=20.0`
  - `sample_position_clip=25.0`
  - `sample_position_clip=100.0`

## Promotion Gate

A candidate must satisfy all of:

- mean aligned RMSD delta `<= -0.05`
- at least `4/6` hard cases improve
- mean raw RMSE delta `<= +0.03`

## Result

No inference-only candidate cleared the hard-case gate.

| Candidate | Mean Raw RMSE Delta | Mean Aligned RMSD Delta | Improved Complexes | Gate Result |
| --- | ---: | ---: | ---: | --- |
| `sample_time_power=1.5` | `-0.002136` | `-0.003455` | `2/6` | fail |
| `sample_time_power=2.0` | `-0.002888` | `-0.008700` | `2/6` | fail |
| `sample_time_power=3.0` | `-0.035016` | `-0.031689` | `4/6` | fail |
| `sample_steps=50` | `-0.034749` | `+0.008290` | `2/6` | fail |
| `sample_steps=100` | `-0.011227` | `+0.017131` | `2/6` | fail |
| `sample_score_clip=5.0` | `+0.005043` | `+0.005122` | `1/6` | fail |
| `sample_score_clip=20.0` | `+0.000000` | `+0.000000` | `0/6` | fail |
| `sample_position_clip=25.0` | `+0.000000` | `+0.000000` | `0/6` | fail |
| `sample_position_clip=100.0` | `+0.000000` | `+0.000000` | `0/6` | fail |

## Interpretation

- `sample_time_power=3.0` was the strongest new sampler variant.
- It improved `4/6` hard cases and reduced mean late-step score norms, but the mean aligned RMSD gain was only `-0.031689`, which is below the required `-0.05`.
- The clip changes remained inert at the hard-case aggregate level, matching the earlier checkpoint-resample sweep.
- The legacy `sample_steps` knobs again reduced raw RMSE without improving aligned RMSD.

## Decision

The accepted recommendation is unchanged:

- `--device cpu --steps 200 --sample-steps 25 --noise-schedule cosine --frame-hetero-backbone --ligand-bond-weight 0.1`

No full 20-complex inference confirmation was run from this stage.
