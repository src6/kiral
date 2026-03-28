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
- `docs/training/`: curated experiment evidence, including canonical `panel20/` summaries, the checkpoint/resume note, and a small `showcase/` of representative plots

## Dataset Setup

The real-complex runs and dataset mode expect a local PDBbind/PDBbind+ download obtained from the [PDBbind+ download page](https://www.pdbbind-plus.org.cn/download). The dataset is not included in Git and must be downloaded separately.

The default repo-local path is:

- `data/pdbbind_v2020`

The extracted dataset root should contain at least:

- `index/README`
- `protein_ligand_general_minus_refined/`
- `protein_ligand_refined/`

The loader scans those protein-ligand directories for files named:

- `*_protein.pdb`
- `*_ligand.sdf`

If your local dataset lives elsewhere, link it into the repo with:

```bash
mkdir -p data
ln -s /absolute/path/to/pdbbind_v2020 data/pdbbind_v2020
```

Dataset mode can also use `--dataset-root /absolute/path/to/pdbbind_v2020`, but the default path used by the code is `data/pdbbind_v2020`. Dataset graph caches are written to `data/.cache/equidock_diff_graphs`.

You can verify the layout without running training:

```bash
test -f data/pdbbind_v2020/index/README
find data/pdbbind_v2020/protein_ligand_general_minus_refined -name '*_protein.pdb' | head
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from equidock_diff.data.io import load_paths, load_pdbbind_split_paths
root = Path("data/pdbbind_v2020")
print("dataset_pairs", len(load_paths(root)))
print("panel20_pairs", len(load_pdbbind_split_paths(root, Path("config/evaluation/dissertation_panel20.txt"))))
PY
```

## Quick Start

```bash
uv sync
uv run --extra test python -m pytest
uv run python -m equidock_diff.sanity_check --trials 8 --device cpu
uv run python -m equidock_diff.train --device cpu --steps 20 --sample-steps 10
```

The quick-start training command above uses the synthetic path. Real-complex runs and dataset mode require the dataset setup above.

Tracked evaluation summaries are kept under `docs/training/panel20/`, with representative qualitative figures in `docs/training/showcase/`.

## Fast Research Iteration

For small exploratory matrices, use the local research runner instead of hand-writing many `train` and `resample_from_checkpoint` commands:

```bash
uv run python -m equidock_diff.research_runner \
  --complex-id 10gs \
  --model frame_backbone \
  --noise-schedule cosine \
  --seed 42 \
  --steps 200 \
  --sample-steps 25 \
  --sample-steps 50 \
  --tag quick_cosine_probe
```

Research-run behavior:

- scratch runs default to auto-detected `mps` when available and fall back to `cpu`
- canonical/report-quality runs should still use CPU-oriented workflows outside the runner
- local outputs go under `runs/research/<tag>/` and are not committed
- inference-only variants reuse checkpoints through `equidock_diff.resample_from_checkpoint`
- pose, trajectory, and plot artifacts are skipped by default; add `--save-artifacts` when you need them

Use `--dry-run` to inspect the expanded run matrix before spending compute, and `--compare-against <prior-tag>` to generate a diff against an earlier local research tag.
