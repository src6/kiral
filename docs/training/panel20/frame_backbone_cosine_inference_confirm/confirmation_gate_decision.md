# Inference Confirmation Decision

No full 20-complex inference confirmation was run.

The hard-case inference diagnostics in `docs/training/panel20/frame_backbone_cosine_inference_diag/` did not produce any variant that cleared the promotion gate, so there was nothing to confirm on the full panel.

Current recommendation remains:

- `--device cpu --steps 200 --sample-steps 25 --noise-schedule cosine --frame-hetero-backbone --ligand-bond-weight 0.1`
