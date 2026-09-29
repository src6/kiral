# Kiral

[![CI](https://github.com/src6/kiral/actions/workflows/ci.yml/badge.svg)](https://github.com/src6/kiral/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-ee4c2c.svg)
![Equivariance](https://img.shields.io/badge/Equivariance-SE(3)-purple.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)

**Kiral** is a reproducible molecular-docking research engine combining chirality-aware $\mathrm{SE}(3)$-equivariant geometric learning with VP-SDE diffusion for blind and targeted protein–ligand docking.
---

## Empirical Evaluation & Findings

### 1. Data-Scaling Ablation on PDBbind v2020 ($N=300 \to 18,537$)
A rigorous controlled experiment was conducted across three nested subsets of the PDBbind General Set v2020 ($N \in \{300, 3000, 18537\}$) evaluated on an untouched, held-out set of 500 complexes using the reference [PoseBusters](https://github.com/maabuu/posebusters) suite:

| Training Arm | Training Set Size ($N$) | Step Budget | Held-Out Aligned RMSD ($\pm$ SD) | PoseBusters Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Arm 1** | $N = 300$ | 5,000 steps | **0.3495 $\pm$ 0.5111 Å** | **12.8%** (64 / 500) |
| **Arm 2** | $N = 3,000$ | 5,000 steps | **0.3617 $\pm$ 0.5470 Å** | **10.0%** (50 / 500) |
| **Arm 3** | $N = 18,537$ | 5,000 steps | **0.3919 $\pm$ 0.5149 Å** | **5.6%** (28 / 500) |

**Key Takeaway (Negative Result on Data Scale):**
Scaling training data volume by **$60\times$** does not rescue stereochemical or physical validity under Cartesian point-cloud score matching. While the network easily masters pocket-level rigid placement (aligned $\text{RMSD} \approx 0.35\text{--}0.39$ Å), internal chemical validity declines with scale. Because diffusion operates directly in unconstrained Euclidean space ($\mathbb{R}^{3N}$), individual atomic displacements inevitably introduce sub-angstrom bond length distortions and angular strain. Inductive bias over physical manifolds dominates dataset scale.

### 2. High-Throughput Batched Training & Serving
- **Throughput:** Scaled from 3,869 complexes/min (batch 1) to **20,650 complexes/min** (batch 16) on an RTX 3080 Ti ($5.34\times$ throughput speedup).
- **VRAM Footprint:** Batch 16 uses only **1.08 GB VRAM** (< 10% capacity), making large-scale training and serving highly efficient.
- **Equivalence:** Batched multi-graph training mathematically preserves the objective: batch-8 scalar loss matches the mean of 8 batch-1 passes to within $10^{-5}$, with exact gradient parity.
- **Zero-Latency Graph Cache:** Bounded graph construction parses and pre-caches the entire 19,037-complex corpus with sub-millisecond cached loading ($0.17$ ms / complex).

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
*Current test suite: **208 tests passing, 1 expected calibration xfail in $\approx 6.5\text{ seconds}$**.*

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
