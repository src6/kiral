# Checkpoint and Resume Run

## Purpose

This note records the first full checkpoint-and-resume training run added after introducing `--checkpoint-path`, `--checkpoint-every`, and `--resume-from` in `src/equidock_diff/train.py`.

## Pre-Run Sanity Check

```bash
.venv/bin/python -m equidock_diff.sanity_check --trials 8 --device cpu
```

Observed result:

- mean equivariance error: `2.211397e-06`
- max equivariance error: `1.928834e-05`

## Stage 1: Train and Save Checkpoints

```bash
.venv/bin/python -m equidock_diff.train \
  --device cpu \
  --steps 150 \
  --learning-rate 3e-4 \
  --ligand-bond-weight 0.1 \
  --hetgnn-backbone \
  --sample-steps 25 \
  --protein-path data/pdbbind_v2020/protein_ligand_general_minus_refined/1981-2000/10gs/10gs_protein.pdb \
  --ligand-path data/pdbbind_v2020/protein_ligand_general_minus_refined/1981-2000/10gs/10gs_ligand.sdf \
  --crop-cutoff 8.0 \
  --edge-cutoff 4.5 \
  --sample-score-clip 10.0 \
  --sample-position-clip 50.0 \
  --output docs/training/10gs_full_stage1_sample.pdb \
  --trajectory-output docs/training/10gs_full_stage1_traj.pdb \
  --ligand-output docs/training/10gs_full_stage1_ligand_sample.pdb \
  --ligand-trajectory-output docs/training/10gs_full_stage1_ligand_traj.pdb \
  --loss-csv docs/training/10gs_full_stage1_loss.csv \
  --plot-output docs/training/10gs_full_stage1_plot.png \
  --experiment-log docs/training/10gs_full_stage1_log.md \
  --checkpoint-path docs/training/10gs_full_hetgnn_checkpoint.pt \
  --checkpoint-every 50
```

Observed result:

- checkpoint saved at steps `50`, `100`, and `150`
- training seconds: `4.945`
- raw ligand RMSE: `1.413929`
- aligned ligand RMSD: `1.389536`

## Stage 2: Resume to a Longer Run

```bash
.venv/bin/python -m equidock_diff.train \
  --device cpu \
  --steps 300 \
  --learning-rate 3e-4 \
  --ligand-bond-weight 0.1 \
  --hetgnn-backbone \
  --sample-steps 25 \
  --protein-path data/pdbbind_v2020/protein_ligand_general_minus_refined/1981-2000/10gs/10gs_protein.pdb \
  --ligand-path data/pdbbind_v2020/protein_ligand_general_minus_refined/1981-2000/10gs/10gs_ligand.sdf \
  --crop-cutoff 8.0 \
  --edge-cutoff 4.5 \
  --sample-score-clip 10.0 \
  --sample-position-clip 50.0 \
  --output docs/training/10gs_full_run_sample.pdb \
  --trajectory-output docs/training/10gs_full_run_traj.pdb \
  --ligand-output docs/training/10gs_full_run_ligand_sample.pdb \
  --ligand-trajectory-output docs/training/10gs_full_run_ligand_traj.pdb \
  --loss-csv docs/training/10gs_full_run_loss.csv \
  --plot-output docs/training/10gs_full_run_plot.png \
  --experiment-log docs/training/10gs_full_run_log.md \
  --resume-from docs/training/10gs_full_hetgnn_checkpoint.pt \
  --checkpoint-every 50
```

Observed result:

- resume detected step `150` and continued to step `300`
- final loss: `0.037062`
- best loss: `0.012857`
- total training seconds: `9.837`
- raw ligand RMSE: `1.314676`
- aligned ligand RMSD: `1.278009`
- loss CSV contains `301` lines including the header, confirming a full `300`-step history

## Produced Artifacts

Versioned evidence kept in the repository:

- this note
- curated visual summary: `docs/training/showcase/10gs_checkpoint_resume_loss.png`

Locally reproducible but not versioned by default:

- final log: `docs/training/10gs_full_run_log.md`
- final loss trace: `docs/training/10gs_full_run_loss.csv`
- stage-1 log: `docs/training/10gs_full_stage1_log.md`
- stage-1 loss trace: `docs/training/10gs_full_stage1_loss.csv`
- checkpoint: `docs/training/10gs_full_hetgnn_checkpoint.pt`
- final protein-ligand sample and trajectory PDB files
- ligand-only sample and trajectory PDB files

## Interpretation

- The resume path restored the model and optimizer state cleanly enough to continue optimization instead of restarting.
- Extending the run from `150` to `300` steps improved both reported ligand metrics on `10gs`.
- This is still a single-complex case study, so it is evidence of practical trainability rather than benchmark-level generalization.
