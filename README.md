# Equidock-Diff

Equidock-Diff is an SE(3)-equivariant diffusion prototype for protein-ligand docking. The repository focuses on a compact geometric pipeline: graph construction for protein-ligand pairs, an EGNN-style score model, a VP-SDE sampler, and explicit equivariance sanity checks.

## Repository Layout

- `src/equidock_diff/data/`: protein-ligand graph construction and centering utilities
- `src/equidock_diff/models/`: EGNN baseline and opt-in heterogeneous frame backbone
- `src/equidock_diff/diffusion/`: VP-SDE schedule and forward/reverse diffusion steps
- `src/equidock_diff/train.py`: training and sampling entry point
- `src/equidock_diff/sanity_check.py`: SE(3) rotation/translation sanity check
- `src/equidock_diff/evaluation_summary.py`: aggregate experiment logs into report-ready tables
- `tests/`: unit tests for chemistry, data, diffusion, equivariance, training, and evaluation-summary helpers
- `docs/design/`: design notes and SE(3) assumptions
- `docs/perf/`: MPS/CPU performance notes and fallbacks
- `docs/training/`: experiment logs, artifacts, and evaluation summaries
- `docs/report.tex`: report source

## Quick Start

```bash
uv run python -m pytest
uv run python -m equidock_diff.sanity_check --trials 8 --device cpu
uv run python -m equidock_diff.train --device cpu --steps 20 --sample-steps 10
uv run python -m equidock_diff.evaluation_summary
```
