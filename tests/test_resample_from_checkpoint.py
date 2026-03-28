from __future__ import annotations

from pathlib import Path

import torch

from equidock_diff.resample_from_checkpoint import main
from equidock_diff.train import make_model, save_checkpoint


class _Args:
    device = "cpu"
    seed = 42
    dry_run = False
    steps = 5
    batch_size = 1
    num_nodes = 8
    protein_path = None
    ligand_path = None
    crop_cutoff = 10.0
    edge_cutoff = 4.5
    hidden_dim = 32
    num_layers = 2
    ligand_global_node = False
    complete_frame = False
    hetero_edges = False
    frame_hetero_backbone = False
    learning_rate = 1e-3
    ligand_bond_weight = 0.0
    ligand_shape_weight = 0.0
    beta_min = 0.1
    beta_max = 2.0
    noise_schedule = "linear"
    cosine_offset = 0.008
    cosine_nu = 1.5
    sample_steps = 8
    sample_score_clip = 10.0
    sample_position_clip = 50.0
    sample_time_power = 1.0
    output = Path("docs/training/synthetic_sample.pdb")
    trajectory_output = Path("docs/training/synthetic_trajectory.pdb")
    ligand_output = None
    ligand_trajectory_output = None
    loss_csv = Path("docs/training/loss_trace.csv")
    plot_output = Path("docs/training/trajectory_plot.png")
    experiment_log = None
    sampler_diagnostics_json = None
    checkpoint_path = None
    checkpoint_every = 0
    resume_from = None
    dataset_root = None
    dataset_split = None
    dataset_limit = 0
    dataset_cache_dir = Path("data/.cache/equidock_diff_graphs")


def test_resample_from_checkpoint_writes_sampling_artifacts(tmp_path: Path) -> None:
    device = torch.device("cpu")
    args = _Args()
    model = make_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    checkpoint_path = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        args=args,
        completed_steps=5,
        training_seconds=1.0,
        loss_rows=[(1, 1.0, 0.2), (5, 0.5, 0.8)],
        graph_source="synthetic",
        node_feature_dim=4,
    )

    sample_path = tmp_path / "sample.pdb"
    traj_path = tmp_path / "traj.pdb"
    loss_csv = tmp_path / "loss.csv"
    plot_path = tmp_path / "plot.png"
    log_path = tmp_path / "log.md"
    diag_path = tmp_path / "sampler.json"

    result = main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--device",
            "cpu",
            "--sample-steps",
            "6",
            "--sample-time-power",
            "1.5",
            "--output",
            str(sample_path),
            "--trajectory-output",
            str(traj_path),
            "--loss-csv",
            str(loss_csv),
            "--plot-output",
            str(plot_path),
            "--experiment-log",
            str(log_path),
            "--sampler-diagnostics-json",
            str(diag_path),
        ]
    )

    assert result == 0
    assert sample_path.exists()
    assert traj_path.exists()
    assert loss_csv.exists()
    assert log_path.exists()
    assert diag_path.exists()
