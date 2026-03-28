# Full-Panel Inference Confirmation Decision

No full 20-complex inference-only confirmation was run for the sampler-redesign stage.

Reason:

- no hard-case inference candidate cleared the promotion gate in
  - `docs/training/panel20/frame_backbone_cosine_sampler_diag/inference_gate_decision.md`

The active recommendation therefore remains the accepted frame-backbone cosine training recipe without any inference-only override:

- `--device cpu --steps 200 --sample-steps 25 --noise-schedule cosine --frame-hetero-backbone --ligand-bond-weight 0.1`
