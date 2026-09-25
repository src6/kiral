"""Rerun the schedule panel under legacy vs --snr-consistent modes.

Canonical protocol from the report: 100 train steps, 25 sample steps, lr 3e-4,
bond weight 0.1, crop 8.0, edge 4.5, score clip 10.0, position clip 50.0, CPU.
Legacy mode should reproduce docs/training/panel20/schedule/* exactly.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
DATA = Path.home() / "data" / "pdbbind_v2020"
DEVICE = "cuda"
PANEL = ROOT / "config" / "evaluation" / "dissertation_panel20.txt"

METRIC_RE = {
    "raw": re.compile(r"raw_ligand_rmse=([0-9.eE+-]+)"),
    "aligned": re.compile(r"aligned_ligand_rmsd=([0-9.eE+-]+)"),
    "train_s": re.compile(r"training_seconds=([0-9.]+)"),
    "best_loss": re.compile(r"best_loss=([0-9.eE+-]+)"),
}


def complex_paths(complex_id: str) -> tuple[Path, Path]:
    matches = sorted(DATA.rglob(f"*{complex_id}*/*_protein.pdb"))
    match = next(
        p for p in matches
        if p.parent.name == complex_id and p.with_name(f"{complex_id}_ligand.sdf").exists()
    )
    return match, match.with_name(f"{complex_id}_ligand.sdf")


def run_one(complex_id: str, model: str, schedule: str, seed: int, mode: str, out_dir: Path) -> dict:
    protein, ligand = complex_paths(complex_id)
    run_tag = f"{complex_id}_{model}_{schedule}_s{seed}_{mode}"
    run_dir = out_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "python", "-m", "equidock_diff.train",
        "--device", DEVICE,
        "--seed", str(seed),
        "--steps", "100",
        "--sample-steps", "25",
        "--learning-rate", "3e-4",
        "--ligand-bond-weight", "0.1",
        "--crop-cutoff", "8.0",
        "--edge-cutoff", "4.5",
        "--noise-schedule", schedule,
        "--protein-path", str(protein),
        "--ligand-path", str(ligand),
        "--output", str(run_dir / "sample.pdb"),
        "--trajectory-output", str(run_dir / "trajectory.pdb"),
        "--loss-csv", str(run_dir / "loss.csv"),
        "--experiment-log", str(run_dir / "run.log"),
    ]
    if model == "frame":
        cmd += ["--frame-hetero-backbone"]
    if mode == "snr":
        cmd += ["--snr-consistent"]
    elif mode in ("train", "sampler"):
        cmd += ["--snr-mode", mode]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    row = {"complex": complex_id, "model": model, "schedule": schedule, "seed": seed, "mode": mode}
    if proc.returncode != 0:
        row["error"] = (proc.stderr or proc.stdout)[-400:]
        return row
    for key, rx in METRIC_RE.items():
        m = rx.search(proc.stdout)
        row[key] = float(m.group(1)) if m else ""
    return row


def main() -> int:
    global DEVICE, ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="", help="comma list: legacy,snr,train,sampler")
    parser.add_argument("--quick", action="store_true", help="only 10gs, seed 42")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--out", type=Path, default=ROOT / "rerun_results.csv")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    DEVICE = args.device
    ROOT = args.repo_root.resolve()

    complexes = ["10gs"] if args.quick else [
        line.strip() for line in PANEL.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    seeds = [42] if args.quick else [int(s) for s in args.seeds.split(",")]
    rows = []
    out_file = args.out
    write_header = not out_file.exists()
    with out_file.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["complex", "model", "schedule", "seed", "mode", "raw", "aligned", "train_s", "error"])
        if write_header:
            writer.writeheader()
        modes = args.modes.split(",") if args.modes else ["snr"]
        if args.quick and not args.modes:
            modes = ["legacy", "snr"]
        for complex_id in complexes:
            for model in ("egnn", "frame"):
                for schedule in ("linear", "cosine"):
                    for seed in seeds:
                        for mode in modes:
                            row = run_one(complex_id, model, schedule, seed, mode, out_file.parent / "runs")
                            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
                            fh.flush()
                            print(row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
