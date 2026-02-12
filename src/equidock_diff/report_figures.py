"""Generate report-ready figures and summaries from training artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate report-ready figures from Equidock-Diff artifacts"
    )
    parser.add_argument(
        "--loss-csv",
        type=Path,
        default=Path("docs/training/loss_trace.csv"),
        help="Path to the training loss CSV",
    )
    parser.add_argument(
        "--trajectory-pdb",
        type=Path,
        default=Path("docs/training/synthetic_trajectory.pdb"),
        help="Path to the trajectory PDB with MODEL frames",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("docs/training/report_figures.png"),
        help="Path for the combined PNG figure",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("docs/training/report_figure_summary.md"),
        help="Path for the text summary",
    )
    return parser


def load_loss_rows(path: Path) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["step"]), float(row["loss"]), float(row["beta_t"])))
    if not rows:
        raise ValueError(f"No loss rows found in {path}.")
    return rows


def load_trajectory(path: Path) -> list[list[tuple[float, float, float]]]:
    frames: list[list[tuple[float, float, float]]] = []
    current: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                current = []
            elif line.startswith("ATOM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                current.append((x, y, z))
            elif line.startswith("ENDMDL") and current:
                frames.append(current)
                current = []
    if not frames:
        raise ValueError(f"No MODEL frames found in {path}.")
    return frames


def write_summary(
    path: Path,
    loss_rows: list[tuple[int, float, float]],
    frames: list[list[tuple[float, float, float]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first_loss = loss_rows[0][1]
    best_loss = min(row[1] for row in loss_rows)
    last_loss = loss_rows[-1][1]
    frame_count = len(frames)
    atom_count = len(frames[0])
    path.write_text(
        "\n".join(
            [
                "# Report Figure Summary",
                "",
                f"- Loss rows: {len(loss_rows)}",
                f"- Initial loss: {first_loss:.6f}",
                f"- Best loss: {best_loss:.6f}",
                f"- Final loss: {last_loss:.6f}",
                f"- Trajectory frames: {frame_count}",
                f"- Atoms per frame: {atom_count}",
                "",
                "Suggested report interpretation:",
                "The short-run loss decreases rapidly from its initial value,",
                "showing that the MVP optimization path is wired correctly.",
                "The multi-frame trajectory provides a visual record of the",
                "reverse diffusion process rather than a claim of docking quality.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def maybe_write_plot(
    path: Path,
    loss_rows: list[tuple[int, float, float]],
    frames: list[list[tuple[float, float, float]]],
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    import numpy as np  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0], dtype=float)
    mid = np.asarray(frames[len(frames) // 2], dtype=float)
    last = np.asarray(frames[-1], dtype=float)

    fig = plt.figure(figsize=(11, 4))
    ax_loss = fig.add_subplot(1, 2, 1)
    ax_loss.plot([row[0] for row in loss_rows], [row[1] for row in loss_rows], marker="o")
    ax_loss.set_title("Synthetic Training Loss")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)

    ax_traj = fig.add_subplot(1, 2, 2, projection="3d")
    ax_traj.scatter(first[:, 0], first[:, 1], first[:, 2], label="start", alpha=0.45)
    ax_traj.scatter(mid[:, 0], mid[:, 1], mid[:, 2], label="mid", alpha=0.6)
    ax_traj.scatter(last[:, 0], last[:, 1], last[:, 2], label="end", alpha=0.8)
    ax_traj.set_title("Reverse Diffusion Frames")
    ax_traj.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def main() -> int:
    args = build_parser().parse_args()
    loss_rows = load_loss_rows(args.loss_csv)
    frames = load_trajectory(args.trajectory_pdb)
    write_summary(args.summary_output, loss_rows, frames)
    print(f"summary_path={args.summary_output}")
    if maybe_write_plot(args.plot_output, loss_rows, frames):
        print(f"plot_path={args.plot_output}")
    else:
        print("plot_path=not_written (matplotlib not available)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
