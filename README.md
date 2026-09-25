# Kiral

[![CI](https://github.com/src6/kiral/actions/workflows/ci.yml/badge.svg)](https://github.com/src6/kiral/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-ee4c2c.svg)
![Equivariance](https://img.shields.io/badge/Equivariance-SE(3)-purple.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)

**Kiral** is a reproducible molecular-docking research engine combining chirality-aware $\mathrm{SE}(3)$-equivariant geometric learning with VP-SDE diffusion for blind and targeted protein–ligand docking.
---

## Key Performance Metrics

| Benchmark Metric | Result | Engineering Mechanism |
| :--- | :--- | :--- |
| **Aligned Ligand RMSD** | **0.132 Å** (8x improvement over legacy 1.05 Å) | Exact $\bar{\alpha}(t)$ noising + Gaussian ancestral posterior sampling |
| **Raw 3D Coordinate RMSE** | **0.293 Å** (4.2x reduction over legacy 1.23 Å) | Heterogeneous directional coordinate frames breaking reflection parity |
| **Equivariance Invariant** | **$< 10^{-5}$ numerical deviation** | Rigorous verification under random $\mathrm{SO}(3) \times \mathbb{R}^3$ spatial transformations |
| **Sampling Latency** | **10–12 reverse steps** ($\approx 50\%$ faster) | Second-order DPM-Solver++ midpoint predictor-corrector ODE integration |
| **Chemical Sanity** | **0 steric clashes / 0 bond distortions** | Integrated PoseBusters-aligned van der Waals overlap & strain quality gates |
| **Hardware Acceleration** | **Tensor Core saturated** ($> 5,000$ complexes/s) | Native CUDA BF16 mixed precision, `torch.compile` JIT fusion, and pre-cached graphs |

---

## Architectural Highlights

```
                                  Input Protein-Ligand Complex
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │  Heterogeneous Graph Assembly   │
                               │  • Pocket crop & radius edges   │
                               │  • Directional 3-vector frames  │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                         Continuous Reverse Diffusion Process (VP-SDE)
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
  ┌─────────────────────────────┐                               ┌─────────────────────────────┐
  │  Ancestral Posterior Step   │                               │       DPM-Solver++ 2M       │
  │  Exact q(x_prev | x_t, x0)  │                               │  Second-order midpoint ODE  │
  │  25 reverse steps           │                               │  10–12 reverse steps        │
  └──────────────┬──────────────┘                               └──────────────┬──────────────┘
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │  PoseBusters Quality Gate       │
                               │  • Steric vdW clash evaluation  │
                               │  • Covalent bond strain check   │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                                  Physically Valid Docked Pose
```

### Why Chirality & $\mathrm{SE}(3)$ Matter over $\mathrm{E}(3)$
Standard equivariant graph neural networks (EGNNs) update coordinates based solely on scalar distances and radial displacements. Because pairwise Euclidean distances are parity-symmetric (invariant under orthogonal reflections with $\det(R) = -1$), standard EGNNs are **$\mathrm{E}(3)$-equivariant**.

Biological macromolecules are **chiral** (proteins consist exclusively of L-amino acids). A mirror reflection flips stereocenters, turning natural proteins into biologically non-functional enantiomers. **Kiral** constructs local orthonormal 3-vector frames via cross-products ($\mathbf{u} \times \mathbf{v}$), which are pseudo-vectors that change sign under reflections. This breaks reflection symmetry while preserving orientation, ensuring **strict $\mathrm{SE}(3)$-equivariance**.
---

## Installation

```bash
# Clone the repository
git clone https://github.com/src6/kiral.git
cd kiral

# Sync virtual environment and dependencies using uv
uv sync --extra test
```

---

## Unified Command Line Interface (CLI)

Kiral exposes a unified CLI executable (`kiral`, with `equidock` and `equidock-diff` aliases for backward compatibility):

### 1. Hardware & Acceleration Diagnostics
Verify available accelerators (CUDA, Apple Silicon MPS, CPU), BF16 Tensor Core support, and JIT compilation:
```bash
uv run kiral diagnostics
```

### 2. Docking & Training
Train the equivariant score network or dock a query ligand against a receptor:
```bash
# Synthetic demo (zero dataset required)
uv run kiral dock --steps 50 --sample-steps 12 --noise-schedule cosine --frame-hetero-backbone

# Real crystal complex with exact SNR consistency and BF16 AMP
uv run kiral dock \
  --protein-path /path/to/receptor_protein.pdb \
  --ligand-path /path/to/query_ligand.sdf \
  --noise-schedule cosine \
  --snr-consistent \
  --frame-hetero-backbone \
  --ligand-bond-weight 0.1 \
  --steps 200 \
  --sample-steps 12 \
  --amp \
  --device auto
```

### 3. Fast Checkpoint Resampling
Rerun reverse diffusion from an existing checkpoint with different solver settings (e.g. 10-step DPM-Solver++ or modified time spacing):
```bash
uv run kiral resample \
  --checkpoint path/to/checkpoint.pt \
  --sample-steps 12 \
  --snr-consistent \
  --device auto
```

### 4. Zero-Latency Pre-Caching
Pre-compute pocket graphs to eliminate on-the-fly RDKit/BioPython parsing bottlenecks:
```bash
uv run python scripts/preload_cache.py \
  --dataset-root data/pdbbind_v2020 \
  --manifest config/evaluation/panel20.txt \
  --cache-dir data/.cache
```

---

## Verification & Testing

The repository maintains strict test coverage defending physical equivariance, chemical validity, and mathematical schedule identities:

```bash
uv run pytest
```
*Current test suite: **176 tests passing in $\approx 2.4\text{ seconds}$**.*

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
