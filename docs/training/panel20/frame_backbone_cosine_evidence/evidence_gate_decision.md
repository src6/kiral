# Evidence Stage Decision

No extra-seed full-panel evidence run was executed for a new candidate.

Reason:

- the sampler-redesign stage did not promote an inference-only override
- the geometry-aware training stage did not promote a new training recipe

The accepted recommendation therefore remains:

- `--device cpu --steps 200 --sample-steps 25 --noise-schedule cosine --frame-hetero-backbone --ligand-bond-weight 0.1`

The existing accepted confirmation and significance analysis remain the highest-strength evidence in this repository:

- `docs/training/panel20/frame_backbone_cosine_confirm/confirmation_gate_decision.md`
- `docs/training/panel20/frame_backbone_cosine_confirm/longer_training_significance.md`
