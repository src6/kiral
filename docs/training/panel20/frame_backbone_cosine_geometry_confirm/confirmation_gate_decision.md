# Full-Panel Geometry Confirmation Decision

No full 20-complex geometry-aware confirmation was run.

Reason:

- no hard-case geometry candidate cleared the promotion gate in
  - `docs/training/panel20/frame_backbone_cosine_geometry_tuning/geometry_gate_decision.md`

The active recommendation therefore remains the accepted frame-backbone cosine training recipe without any geometry-loss override:

- `--device cpu --steps 200 --sample-steps 25 --noise-schedule cosine --frame-hetero-backbone --ligand-bond-weight 0.1`
