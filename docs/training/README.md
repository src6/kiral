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

## Curated Experimental Showcase

- Directory: `docs/training/showcase/`
- Purpose: keep a handful of representative exploratory visuals without versioning the full set of generated plots and trajectories
- Current examples:
  - an early `10gs` baseline smoke-run loss curve
  - a checkpoint/resume loss plot showing continuation from `150` to `300` steps
  - a qualitative panel20 pose-comparison figure used in the report
  - a `10gs` schedule-comparison plot contrasting linear and cosine baseline behaviour

These files are not the canonical evaluation outputs, but they help demonstrate that the project went through real iteration rather than only one final polished panel.
