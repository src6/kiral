# AGENTS.md

This file is for future coding agents working in this repository. It is intended to reduce repeated setup work, avoid re-deriving established workflow decisions, and keep context/token usage low.

## Repo Purpose

`equidock-diff` is a protein-ligand docking research repo built around:

- graph construction for protein-ligand pairs
- EGNN baseline and heterogeneous frame-backbone variants
- VP-SDE diffusion training and sampling
- curated `panel20` evaluation artifacts under `docs/training/`

Primary code lives under:

- `src/equidock_diff/`
- `tests/`
- curated evidence under `docs/training/`

## Active Research Policy

Current accepted frame-backbone cosine recipe:

```bash
--device cpu \
--steps 200 \
--sample-steps 25 \
--noise-schedule cosine \
--frame-hetero-backbone \
--ligand-bond-weight 0.1
```

Do not treat scratch-run results as canonical evidence. Canonical/report-quality comparisons remain CPU-based and curated under `docs/training/`.

Muon is intentionally out of scope for now. Do not propose optimizer experiments unless there is a materially different architecture to test.

## Fast Iteration Workflow

Use:

- `equidock_diff.research_runner` for local scratch experiment matrices
- `equidock_diff.remote_research_runner` for remote scratch runs on the Mac mini
- `equidock_diff.dual_host_runner` for mixed local+remote scheduling under one tag
- `equidock_diff.resample_from_checkpoint` when only inference-time parameters change

Runner controls now available:

- `--max-parallel N`
- `--stagger-seconds S`
- `--keep-going`
- per-tag `status.csv`

Inference-only changes that should reuse checkpoints:

- `sample_steps`
- `sample_time_power`
- `sample_score_clip`
- `sample_position_clip`

Training-affecting changes require fresh training, for example:

- `steps`
- model choice
- schedule
- loss weights such as `ligand_shape_weight` or `ligand_protein_clash_weight`

Scratch outputs belong in:

- `runs/research/<tag>/`
- `runs/remote/<tag>/`
- `runs/dual/<tag>/`

These are local-only and should not be committed.

Preferred scratch workflow:

- use `status.csv` and `run_index.csv` first
- do not read raw logs unless debugging a specific failed run
- for the Mac mini, start with `--max-parallel 2`
- keep remote scratch runs on `cpu` unless MPS is being explicitly re-benchmarked
- for mixed scheduling, use `dual_host_runner` with:
  - Mac mini `cpu`
  - laptop `cpu`
  - `--routing-policy explicit` when you want showcase probes on laptop and sweeps on the mini in the same tag

## Remote Mac Mini Workflow

The Mac mini is the default scratch box for larger exploratory runs.

Transport:

- standard SSH over Tailscale
- not Tailscale SSH

Expected SSH alias on the laptop:

```sshconfig
Host macmini-tailscale
  HostName 10.0.0.1
  User sadik
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

Expected remote repo:

- `/Users/runner/work/kiral/kiral`

Expected remote dataset target:

- `/tmp/pdbbind_v2020`

Expected repo-local symlink on the Mac mini:

- `/Users/runner/work/kiral/kiral/data/pdbbind_v2020 -> /tmp/pdbbind_v2020`

Remote runtime:

- prefer `uv`
- current working setup uses user-local `uv` at `~/.local/bin/uv`
- Homebrew was not available due lack of admin access, so do not assume `brew` exists or should be installed

`remote_research_runner` already supports:

- `uv` on `PATH`
- `~/.local/bin/uv`
- fallback to `.venv/bin/python`

Remote repo sync:

- prefer `rsync` for active local changes
- prefer `git` only when the local tree is clean and synced
- keep `docs/training/` excluded from rsync because it is large and not needed for scratch runs

## Security Notes

Current secure-enough setup:

- SSH key auth over Tailscale
- `IdentitiesOnly yes`
- remote `authorized_keys` permissions correct
- remote login enabled on the mini

Recommended hardening:

- once all required clients are verified, disable password SSH on the Mac mini
- do not expose SSH outside Tailscale

Do not make machine-level security changes unless explicitly asked.

## Benchmark Findings

Controlled benchmark already run on `10gs`, frame-backbone, cosine, sample steps `25` and `50`.

Findings:

- local M1 Pro CPU training time: about `4.862s`
- remote M4 CPU training time: about `2.174s`
- remote M4 CPU is about `2.24x` faster for this workload
- remote M4 MPS was substantially worse than remote CPU for this workload
- a targeted Kabsch/MPS fix removed one unsupported-op warning path, but did not materially improve runtime

Important constraint:

- local MPS was not available inside the Codex process during benchmarking
- remote MPS is available on the Mac mini, but performance is still poor even after fixing the aligned-RMSD SVD fallback path
- therefore: use Mac mini `cpu` for serious scratch runs unless the MPS path is revalidated after kernel/operator changes

Operational conclusion:

- laptop: dry-runs, smoke tests, fast code iteration
- Mac mini CPU: larger scratch sweeps
- avoid Mac mini MPS for this workflow unless rebenchmarked

## Current Model/Research State

Implemented relevant tooling and features include:

- `evaluation_diff`
- `evaluation_stats`
- `resample_from_checkpoint`
- `complex_diagnostics`
- `sampler_diagnostics`
- `research_runner`
- `remote_research_runner`
- `dual_host_runner`
- `max_parallel`
- `stagger_seconds`
- `keep_going`
- `status.csv`
- `sample_time_power`
- `sampler_diagnostics_json`
- `ligand_shape_weight`
- `ligand_protein_clash_weight`

Recent research conclusion:

- accepted frame-backbone cosine improvement is statistically significant, but narrowly
- later sampler and geometry follow-ups did not beat the accepted recipe
- next active experimental direction is the ligand-protein clash prior

## Token-Saving Guidance

When working in this repo, do not repeatedly re-scan the entire evidence history unless the task requires it.

Prefer these assumptions unless contradicted by the task:

- canonical evidence lives in `docs/training/`
- scratch outputs live in `runs/research/` and `runs/remote/`
- remote scratch machine is the Mac mini
- remote scratch device should default to `cpu` for this workflow
- remote scratch runs should prefer `--max-parallel 2`
- `docs/training/` should not be rsynced to the mini for scratch runs
- Muon is out of scope

Avoid unnecessary context expansion:

- do not open large `docs/training/` trees by default
- do not inspect all experiment logs unless the task is specifically about evidence review
- use targeted file reads and `rg`
- reuse existing summaries before rerunning experiments

When summarizing progress, prefer:

- current branch
- files changed
- exact commands run
- benchmark or metric deltas
- blockers and assumptions

Do not restate broad repo background unless the task is onboarding-oriented.

## Practical Commands

Local scratch dry-run:

```bash
uv run python -m equidock_diff.research_runner \
  --complex-id 10gs \
  --model frame_backbone \
  --noise-schedule cosine \
  --tag smoke_local \
  --dry-run
```

Remote scratch dry-run:

```bash
uv run python -m equidock_diff.remote_research_runner \
  --remote-host macmini-tailscale \
  --remote-repo /Users/runner/work/kiral/kiral \
  --remote-dataset-target /tmp/pdbbind_v2020 \
  --complex-id 10gs \
  --model frame_backbone \
  --noise-schedule cosine \
  --tag smoke_remote \
  --dry-run \
  --sync-mode rsync
```

Remote real scratch probe:

```bash
uv run python -m equidock_diff.remote_research_runner \
  --remote-host macmini-tailscale \
  --remote-repo /Users/runner/work/kiral/kiral \
  --remote-dataset-target /tmp/pdbbind_v2020 \
  --complex-id 10gs \
  --model frame_backbone \
  --noise-schedule cosine \
  --sample-steps 25 \
  --sample-steps 50 \
  --device-policy explicit \
  --device cpu \
  --tag probe_remote_cpu \
  --sync-mode rsync
```

Remote parallel scratch probe:

```bash
uv run python -m equidock_diff.remote_research_runner \
  --remote-host macmini-tailscale \
  --remote-repo /Users/runner/work/kiral/kiral \
  --remote-dataset-target /tmp/pdbbind_v2020 \
  --complex-id 10gs \
  --model frame_backbone \
  --noise-schedule cosine \
  --sample-steps 25 \
  --sample-steps 50 \
  --device-policy explicit \
  --device cpu \
  --max-parallel 2 \
  --stagger-seconds 2 \
  --tag probe_remote_parallel \
  --sync-mode rsync
```

## Editing Policy

When making repo edits:

- use `apply_patch`
- keep changes minimal and local
- do not overwrite curated canonical evidence under `docs/training/` unless explicitly requested
- do not commit scratch outputs

If remote execution fails, first check:

1. SSH connectivity to `macmini-tailscale`
2. remote dataset symlink target exists
3. `~/.local/bin/uv` exists on the mini
4. rsync exclusions are not pulling huge unnecessary trees
