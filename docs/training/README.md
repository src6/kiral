# Training Artifacts Guide

This directory keeps the curated experiment evidence that supports the final report. The committed material is intentionally lightweight:

- Markdown summary notes
- CSV summary tables
- LaTeX-ready comparison tables
- A very small curated `showcase/` of representative plots that makes the experimentation visible at a glance

Full exploratory histories, raw per-run logs, bulk loss traces, sample structures, and checkpoints remain local reproducible artifacts and are not versioned in this final repository snapshot. If you are reviewing the project submission, start with the canonical material below.

## Canonical Dissertation Evidence

### Architecture panel

- Directory: `docs/training/panel20/architecture/`
- Purpose: fixed 20-complex CPU comparison across the baseline EGNN, three intermediate EGNN ablations, and the heterogeneous frame-based backbone
- Main summary:
  - `docs/training/panel20/architecture/cross_complex_comparison_all.md`
  - `docs/training/panel20/architecture/cross_complex_comparison_all.csv`
  - `docs/training/panel20/architecture/cross_complex_comparison_all.tex`

### Schedule panel

- Directory: `docs/training/panel20/schedule/`
- Purpose: matched linear-versus-cosine comparison on the same 20-complex panel for the two main models
- Main summary:
  - `docs/training/panel20/schedule/cross_complex_comparison_schedule.md`
  - `docs/training/panel20/schedule/cross_complex_comparison_schedule.csv`
  - `docs/training/panel20/schedule/cross_complex_comparison_schedule.tex`

### Supplementary seed-variation panel

- Directory: `docs/training/panel20/variance_mps/`
- Purpose: supplementary MPS reruns with seeds `43` and `44` for the baseline and heterogeneous frame-based backbone
- Main summary:
  - `docs/training/panel20/variance_mps/cross_complex_comparison_variance.md`
  - `docs/training/panel20/variance_mps/cross_complex_comparison_variance.csv`
  - `docs/training/panel20/variance_mps/cross_complex_comparison_variance.tex`

## Supporting Engineering Evidence

- `docs/training/checkpoint_resume.md`
  - checkpoint/resume support for longer training runs
- `config/evaluation/dissertation_panel20.txt`
  - the fixed complex manifest used by the canonical summary commands
- `docs/training/panel20/frame_backbone_cosine_tuning/`
  - exploratory CPU sweep reports used to select a better cosine frame-backbone configuration on a 5-complex subset
- `docs/training/panel20/frame_backbone_cosine_confirm/`
  - full-panel confirmation reports for the promoted frame-backbone cosine configuration against the reused CPU control panel
- `docs/training/panel20/frame_backbone_cosine_regression_sweep/`
  - targeted hard-case follow-up on the six regression complexes from the accepted confirmation; no candidate cleared the promotion gate
- `docs/training/panel20/frame_backbone_cosine_regression_confirm/`
  - explicit note that a second full-panel confirmation was not run because the hard-case gate failed
- `docs/training/panel20/frame_backbone_cosine_inference_diag/`
  - inference-only hard-case follow-up using regenerated checkpoints from the accepted frame-backbone cosine recipe; no inference variant cleared the promotion gate
- `docs/training/panel20/frame_backbone_cosine_inference_confirm/`
  - explicit note that no inference-driven full-panel confirmation was run because the inference gate failed
- `docs/training/panel20/frame_backbone_cosine_structural_diag/`
  - structural descriptor comparison for the six hard-case regressions versus the other panel complexes
- `docs/training/panel20/frame_backbone_cosine_sampler_diag/`
  - sampler-redesign and sampler-internal diagnostics for the six hard cases; no inference-only override cleared the gate
- `docs/training/panel20/frame_backbone_cosine_geometry_tuning/`
  - hard-case geometry-aware training sweep using the new ligand-shape loss term; no candidate cleared the gate
- `docs/training/panel20/frame_backbone_cosine_geometry_confirm/`
  - explicit note that no geometry-driven full-panel confirmation was run because the geometry gate failed
- `docs/training/panel20/frame_backbone_cosine_evidence/`
  - explicit note that no extra-seed evidence pass was run because no later-stage candidate displaced the accepted recommendation
- `docs/training/panel20/frame_backbone_cosine_clash_tuning/`
  - hard-case ligand-protein clash-prior sweep on the six regression complexes; no clash-weight candidate cleared the promotion gate
- `docs/training/panel20/frame_backbone_cosine_cross_interface_tuning/`
  - hard-case frame-backbone cross-interface architecture sweep against the adaptive-context control; the v1 ligand-only cross-message block regressed the aggregate hard-case metrics and did not justify a full-panel confirmation

These are useful supporting artifacts, but they are not the headline evaluation story in the final report.

## Recommended Frame-Backbone Cosine Recipe

For future frame-backbone cosine reruns, prefer:

- `--device cpu --steps 200 --sample-steps 25 --noise-schedule cosine --frame-hetero-backbone --ligand-bond-weight 0.1`

This recommendation is based on:

- `docs/training/panel20/frame_backbone_cosine_tuning/subset_gate_decision.md`
  - 5-complex CPU sweep that promoted `longer_training`
- `docs/training/panel20/frame_backbone_cosine_confirm/confirmation_gate_decision.md`
  - full 20-complex confirmation showing mean raw RMSE `-0.042488`, mean aligned RMSD `-0.064899`, and no Success@2A drop

The historical `docs/training/panel20/schedule/` table remains the dissertation-era canonical comparison. It is preserved as historical evidence and is not retroactively rewritten by this follow-on result.

The later hard-case regression sweep in `docs/training/panel20/frame_backbone_cosine_regression_sweep/` also did not replace this recommendation, so the recipe above remains the current best validated setting in this repository.

The subsequent hard-case inference diagnostics in `docs/training/panel20/frame_backbone_cosine_inference_diag/` likewise did not improve the recommendation: only `sample_steps` materially changed the hard-case behaviour, and no inference-only variant cleared the promotion gate.

The later structural diagnostics in `docs/training/panel20/frame_backbone_cosine_structural_diag/` indicate that the six regressions cluster around smaller ligands and denser cropped protein neighborhoods.

The sampler-redesign follow-up in `docs/training/panel20/frame_backbone_cosine_sampler_diag/` found that reverse-time power-respacing was the only meaningful inference lever, with `sample_time_power=3.0` improving `4 / 6` hard cases but still missing the aggregate promotion threshold.

The geometry-aware follow-up in `docs/training/panel20/frame_backbone_cosine_geometry_tuning/` also failed to replace the recommendation: `--ligand-shape-weight 0.02` was the strongest candidate, but its hard-case aligned-RMSD gain was only `-0.003337 A`, far below the promotion threshold.

The later clash-prior follow-up in `docs/training/panel20/frame_backbone_cosine_clash_tuning/` likewise did not replace the recommendation: `--ligand-protein-clash-weight 0.05` produced only a marginal hard-case aligned-RMSD improvement (`0.958256 -> 0.956771`) and did not justify a full-panel confirmation.

The later cross-interface architecture follow-up in `docs/training/panel20/frame_backbone_cosine_cross_interface_tuning/` also failed to replace the recommendation: the v1 ligand-only protein-to-ligand cross-message block regressed the adaptive-context hard-case control on both aligned RMSD and raw RMSE and did not justify a full-panel confirmation.

No later-stage candidate displaced the accepted recipe, so the extra-seed evidence stage in `docs/training/panel20/frame_backbone_cosine_evidence/` was not run.

## Curated Experimental Showcase

- Directory: `docs/training/showcase/`
- Purpose: keep a handful of representative exploratory visuals without versioning the full set of generated plots and trajectories
- Current examples:
  - an early `10gs` baseline smoke-run loss curve
  - a checkpoint/resume loss plot showing continuation from `150` to `300` steps
  - a qualitative panel20 pose-comparison figure used in the report
  - a `10gs` schedule-comparison plot contrasting linear and cosine baseline behaviour

These files are not the canonical evaluation outputs, but they help demonstrate that the project went through real iteration rather than only one final polished panel.
