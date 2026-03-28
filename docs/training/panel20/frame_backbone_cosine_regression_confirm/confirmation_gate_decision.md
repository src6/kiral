# Regression Confirmation Decision

No full 20-complex regression confirmation was run.

The regression-focused sweep in `docs/training/panel20/frame_backbone_cosine_regression_sweep/` did not produce a candidate that cleared the hard-case promotion gate, so there was nothing to promote into a confirmation rerun.

Current recommendation remains:

- `--device cpu --steps 200 --sample-steps 25 --noise-schedule cosine --frame-hetero-backbone --ligand-bond-weight 0.1`
