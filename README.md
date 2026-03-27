# Equidock-Diff

Equidock-Diff is a compact equivariant diffusion system for protein-ligand docking. The repository centers on graph construction for protein-ligand pairs, an E(3)-equivariant EGNN baseline, orientation-sensitive frame-based variants, a VP-SDE sampler, explicit rotation/translation sanity checks, and a fixed 20-complex evaluation panel for controlled architectural comparisons.

Required local tools: Python `3.13+` and `uv`. Project Python dependencies such as PyTorch, Torch Geometric, RDKit, and Matplotlib are declared in `pyproject.toml`; the optional test dependency set can be included with `uv sync --extra test`.

## Repository Layout

- `src/equidock_diff/data/`: protein-ligand graph construction and centering utilities
- `src/equidock_diff/models/`: EGNN baseline and opt-in heterogeneous frame backbone
- `src/equidock_diff/diffusion/`: VP-SDE schedule and forward/reverse diffusion steps
- `src/equidock_diff/train.py`: training and sampling entry point
- `src/equidock_diff/sanity_check.py`: rotation/translation sanity check
- `src/equidock_diff/evaluation_summary.py`: aggregate experiment logs into report-ready tables
- `tests/`: unit tests for chemistry, data, diffusion, equivariance, training, and evaluation-summary helpers
- `docs/design/`: design notes and symmetry assumptions
- `docs/perf/`: MPS/CPU performance notes and fallbacks
- `docs/training/`: experiment logs, artifacts, and the canonical `panel20/` evaluation summaries
- `docs/report.tex`: report source

## Quick Start

```bash
uv sync
uv run --extra test python -m pytest
uv run python -m equidock_diff.sanity_check --trials 8 --device cpu
uv run python -m equidock_diff.train --device cpu --steps 20 --sample-steps 10
uv run python -m equidock_diff.evaluation_summary \
  --log-glob 'docs/training/panel20/**/*_log.md' \
  --manifest config/evaluation/dissertation_panel20.txt
```
