# Sampler Audit and Corrected Panel Re-run (post-submission)

Post-submission audit of the reverse diffusion sampler, plus a full 20-complex panel re-run with
the corrected sampler and **identical trained weights**.

The submitted repository, the submitted panel tables (`docs/training/panel20/architecture`,
`docs/training/panel20/schedule`) and the submitted headline numbers are unchanged. The material
here is post-submission work, included so the corrected numbers can be checked against raw per-run
data rather than taken on trust.

## Findings (verified against the submitted code)

1. **Convention flip.** Report Eq. 2 defines `alpha_t = cos^2(...)` as variance retention
   (Nichol-Dhariwal lineage). The submitted `train.py:457` computes `sigma_t = sqrt(1 - alpha_t**2)`,
   treating `cos^2` as a standard-deviation coefficient, so the model trained against an effective
   retention of `cos^4(phi)`. At `t = 0.5` effective `alpha_bar` is 0.51 against a design 0.71.
2. **Factor of 2 in beta.** The derivation `beta = 2 tan(phi) phi'` is hardcoded as `4.0 *` at
   `schedules.py:52` -- exactly 2x for every `t` (4.22 against 2.11 at `t = 0.5`).
3. **Clamp-induced train/sampler SNR divergence.** Training noising is analytic; the reverse solver
   integrates `beta` clamped to `[0.1, 2.0]`. Implied `x_0` coefficient: sampler 0.466 against
   training ~0 at `t = 1`; 0.597 against 0.266 at `t = 0.75`.
4. **Linear-mode integral shortcut** (affects the canonical table too). `train.py:464` uses
   `beta(t) * t` instead of `int_0^t beta(t) dt`, over-noising training. The distortion is common to
   every architecture row, so the rankings are robust and the submitted absolutes are floors.
5. **Why the test suite missed it.** `test_schedules.py` checks positivity and monotonicity only;
   no test pins the identity `exp(-1/2 int beta) == training marginal`.

## Protocol

- The legacy harness reproduces the stored submitted panel to `1e-6` before any corrected run.
- Corrected runs use exact `alpha_bar` noising plus the exact ancestral posterior reverse step --
  in this repository, `--snr-mode sampler|train|full`.
- Coverage: 20 complexes x 2 models x 2 schedules x 2 seeds. The attribution ablation runs the
  train-side fix and the sampler-side fix separately, full 160-run protocol each.

## Corrected panel (mean aligned ligand RMSD, Angstrom)

| Model | Schedule | Submitted | Corrected | Submitted Succ@2A | Corrected Succ@2A | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EGNN | linear | 2.46 | **1.92** | 40% | 42.5% | 19/20 |
| EGNN | cosine | 2.68 | 4.45 | 30% | **52.5%** | 12/20 (two tail divergences) |
| frame backbone | linear | 1.46 | **0.50** | 75% | **100%** | **20/20** |
| frame backbone | cosine | 1.20 | **0.35** | 100% | 100% | **20/20** |

## Attribution: which change does the work? (mean aligned RMSD, Angstrom)

| Model / schedule | Submitted | Train-fix | Sampler-fix | Both |
| --- | ---: | ---: | ---: | ---: |
| EGNN / linear | 2.460 | 2.298 | 2.641 | **1.922** |
| EGNN / cosine | 2.682 | 2.415 | 4.424 | 4.445 |
| frame / linear | 1.459 | 1.605 | **0.361** | 0.498 |
| frame / cosine | 1.199 | 1.307 | **0.344** | 0.348 |

Paired at the complex level against the submitted panel (seeds averaged, n = 20, exact two-sided
binomial sign test):

| Cell | Complexes improved | p | Mean delta | Median delta |
| --- | ---: | ---: | ---: | ---: |
| frame + linear | 20/20 | 1.9e-06 | +0.96 A | +0.98 A |
| frame + cosine | 20/20 | 1.9e-06 | +0.85 A | +0.84 A |
| EGNN + linear | 19/20 | 4.0e-05 | +0.54 A | +0.41 A |

## Reading

- The frame backbone's 3-4x gain is **entirely the sampler**: same trained weights, same noise, only
  the reverse-step update changed. The submitted reverse process was the binding constraint on pose
  quality, not the architecture.
- Train-side correction alone is neutral to harmful (frame + linear 1.46 -> 1.61): the legacy noising
  shortcut produced a marginal distribution that the clamped sampler happened to track better.
- EGNN + cosine tail instability survives the sampler fix (worst case ~17.6 A, 2-3 of 20 diverge).
  The submitted claim survives in modified form: aggressive cosine harms the unanchored baseline by
  occasional catastrophic failure rather than uniform degradation -- Succ@2A rises from 30% to 52.5%
  while the mean worsens.

## Files

| File | Contents |
| --- | --- |
| `rerun_results.csv` | 160 corrected runs (complex, model, schedule, seed, mode, raw, aligned, train_s) |
| `rerun_ablation.csv` | 320 attribution runs, train-fix and sampler-fix arms separately |
| `rerun_snr_panel.py` | harness that produced both CSVs |
| `schedule_fix.patch` | the noising and reverse-step change |
| `comparison_summary.txt` | console summary of the comparison |
