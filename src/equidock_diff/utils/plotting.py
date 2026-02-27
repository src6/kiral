from __future__ import annotations

from pathlib import Path

import torch


def maybe_write_plot(
    path: Path,
    trajectory: list[torch.Tensor],
    losses: list[tuple[int, float, float]],
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 4))
    ax_traj = fig.add_subplot(1, 2, 1, projection="3d")
    first = trajectory[0].numpy()
    last = trajectory[-1].numpy()
    ax_traj.scatter(first[:, 0], first[:, 1], first[:, 2], label="start", alpha=0.6)
    ax_traj.scatter(last[:, 0], last[:, 1], last[:, 2], label="end", alpha=0.8)
    ax_traj.set_title("Reverse Diffusion Trajectory")
    ax_traj.legend()

    ax_loss = fig.add_subplot(1, 2, 2)
    ax_loss.plot([row[0] for row in losses], [row[1] for row in losses], marker="o")
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True
