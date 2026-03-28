"""Run sampling-only inference from a saved training checkpoint."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import torch

from equidock_diff.train import (
    SamplerDiagnostics,
    build_dataset_examples,
    build_parser as build_train_parser,
    dataset_example_for_step,
    dataset_mode_enabled,
    load_checkpoint,
    load_dataset_example,
    load_graph_inputs,
    make_model_for_node_dim,
    resolve_device,
    sample_positions,
    saved_args_to_namespace,
    write_sampler_diagnostics,
)
from equidock_diff.utils.artifacts import (
    write_experiment_log,
    write_ligand_artifacts,
    write_loss_csv,
    write_pdb,
    write_trajectory_pdb,
)
from equidock_diff.utils.geometry import aligned_rmsd
from equidock_diff.utils.plotting import maybe_write_plot


PATH_OVERRIDE_NAMES = (
    "output",
    "trajectory_output",
    "ligand_output",
    "ligand_trajectory_output",
    "loss_csv",
    "plot_output",
    "experiment_log",
    "sampler_diagnostics_json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a checkpoint and rerun reverse diffusion without training"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to resample from")
    parser.add_argument("--device", default=None, help="Optional override device")
    parser.add_argument("--sample-steps", type=int, default=None, help="Override reverse diffusion steps")
    parser.add_argument(
        "--sample-score-clip",
        type=float,
        default=None,
        help="Override score clipping during reverse diffusion",
    )
    parser.add_argument(
        "--sample-position-clip",
        type=float,
        default=None,
        help="Override coordinate clipping during reverse diffusion",
    )
    parser.add_argument(
        "--sample-time-power",
        type=float,
        default=None,
        help="Optional timestep power-respacing override",
    )
    parser.add_argument(
        "--skip-pose-artifacts",
        action="store_true",
        help="Skip writing sample, trajectory, and ligand-only pose artifacts",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip writing the optional plot artifact",
    )
    train_defaults = build_train_parser().parse_args([])
    for name in PATH_OVERRIDE_NAMES:
        default_value = getattr(train_defaults, name)
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default_value)
    return parser


def _load_checkpoint_saved_args(path: Path, device: torch.device) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    saved_args = payload.get("saved_args")
    if not isinstance(saved_args, dict):
        raise ValueError(f"Checkpoint {path} does not contain saved arguments.")
    return dict(saved_args)


def _apply_overrides(args: argparse.Namespace, cli_args: argparse.Namespace) -> None:
    if cli_args.sample_steps is not None:
        args.sample_steps = cli_args.sample_steps
    if cli_args.sample_score_clip is not None:
        args.sample_score_clip = cli_args.sample_score_clip
    if cli_args.sample_position_clip is not None:
        args.sample_position_clip = cli_args.sample_position_clip
    if cli_args.sample_time_power is not None:
        args.sample_time_power = cli_args.sample_time_power
    if cli_args.device is not None:
        args.device = cli_args.device
    if cli_args.skip_pose_artifacts:
        args.skip_pose_artifacts = True
    if cli_args.skip_plot:
        args.skip_plot = True
    for name in PATH_OVERRIDE_NAMES:
        setattr(args, name, getattr(cli_args, name))


def _load_sampling_graph(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, str]:
    if dataset_mode_enabled(args):
        examples = build_dataset_examples(args)
        sample_example = dataset_example_for_step(examples, args.steps)
        node_features, positions, edge_index, ligand_bond_index = load_dataset_example(
            sample_example,
            args,
            device,
        )
        return node_features, positions, edge_index, ligand_bond_index, sample_example.complex_id
    node_features, positions, edge_index, ligand_bond_index = load_graph_inputs(args, device)
    complex_id = None
    if args.protein_path is not None:
        complex_id = args.protein_path.name.replace("_protein.pdb", "")
    return node_features, positions, edge_index, ligand_bond_index, complex_id or "synthetic"


def main(argv: list[str] | None = None) -> int:
    cli_args = build_parser().parse_args(argv)
    command_argv = sys.argv[1:] if argv is None else argv
    device_name = cli_args.device or "cpu"
    device = resolve_device(device_name)
    saved_args = _load_checkpoint_saved_args(cli_args.checkpoint, device)
    args = saved_args_to_namespace(saved_args)
    _apply_overrides(args, cli_args)

    if args.sample_steps <= 0:
        raise ValueError("--sample-steps must be positive.")
    if args.sample_time_power <= 0.0:
        raise ValueError("--sample-time-power must be positive.")

    torch.manual_seed(int(args.seed))
    node_features, positions, edge_index, _ligand_bond_index, complex_id = _load_sampling_graph(args, device)
    model = make_model_for_node_dim(args, device, node_dim=node_features.size(-1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    checkpoint_state = load_checkpoint(
        cli_args.checkpoint,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    sampler_context = None
    if args.sampler_diagnostics_json is not None:
        sampler_context = SamplerDiagnostics(
            complex_id=complex_id,
            seed=args.seed,
            sample_steps=args.sample_steps,
            sample_time_power=args.sample_time_power,
            sample_score_clip=args.sample_score_clip,
            sample_position_clip=args.sample_position_clip,
            anchor_protein=args.frame_hetero_backbone,
            experiment_log=str(args.experiment_log) if args.experiment_log is not None else None,
            step_metrics=[],
        )

    with torch.no_grad():
        sampled_positions, trajectory, sampler_diagnostics = sample_positions(
            model,
            node_features,
            edge_index,
            positions.size(0),
            device,
            args.sample_steps,
            args.beta_min,
            args.beta_max,
            args.sample_score_clip,
            args.sample_position_clip,
            sample_time_power=args.sample_time_power,
            noise_schedule=args.noise_schedule,
            cosine_offset=args.cosine_offset,
            cosine_nu=args.cosine_nu,
            reference_positions=positions,
            anchor_protein=args.frame_hetero_backbone,
            sampler_diagnostics=sampler_context,
        )

    sample_path: Path | None = None
    trajectory_path: Path | None = None
    if args.skip_pose_artifacts:
        print("sample_artifacts=skipped")
    else:
        write_pdb(args.output, sampled_positions)
        write_trajectory_pdb(args.trajectory_output, trajectory)
        sample_path = args.output
        trajectory_path = args.trajectory_output
        print(f"sample_path={args.output}")
        print(f"trajectory_path={args.trajectory_output}")
        ligand_sample_path, ligand_trajectory_path = write_ligand_artifacts(
            node_features=node_features,
            sampled_positions=sampled_positions,
            trajectory=trajectory,
            ligand_output=args.ligand_output,
            ligand_trajectory_output=args.ligand_trajectory_output,
        )
        if ligand_sample_path is not None:
            print(f"ligand_sample_path={ligand_sample_path}")
        if ligand_trajectory_path is not None:
            print(f"ligand_trajectory_path={ligand_trajectory_path}")

    write_loss_csv(args.loss_csv, checkpoint_state.loss_rows)
    print(f"loss_csv={args.loss_csv}")
    plot_written = False
    if args.skip_plot:
        print("plot_path=skipped")
    else:
        plot_written = maybe_write_plot(args.plot_output, trajectory, checkpoint_state.loss_rows)
        if plot_written:
            print(f"plot_path={args.plot_output}")
        else:
            print("plot_path=not_written (matplotlib not available)")

    extra_metrics: dict[str, float] = {}
    ligand_mask = node_features[:, -1] > 0.5
    if bool(ligand_mask.any()):
        reference_ligand = positions[ligand_mask]
        sampled_ligand = sampled_positions[ligand_mask]
        raw_ligand_rmse = torch.sqrt(torch.mean((sampled_ligand - reference_ligand) ** 2)).item()
        aligned_ligand = aligned_rmsd(sampled_ligand, reference_ligand).item()
        extra_metrics["raw_ligand_rmse"] = raw_ligand_rmse
        extra_metrics["aligned_ligand_rmsd"] = aligned_ligand
        print(f"raw_ligand_rmse={raw_ligand_rmse:.6f}")
        print(f"aligned_ligand_rmsd={aligned_ligand:.6f}")

    if args.sampler_diagnostics_json is not None and sampler_diagnostics is not None:
        write_sampler_diagnostics(args.sampler_diagnostics_json, sampler_diagnostics)
        print(f"sampler_diagnostics_json={args.sampler_diagnostics_json}")
    if args.experiment_log is not None:
        write_experiment_log(
            args.experiment_log,
            command=f"uv run python -m equidock_diff.resample_from_checkpoint {shlex.join(command_argv)}",
            device=device,
            graph_source="dataset" if dataset_mode_enabled(args) else ("real_pair" if args.protein_path else "synthetic"),
            args=args,
            loss_rows=checkpoint_state.loss_rows,
            training_seconds=checkpoint_state.training_seconds,
            node_count=positions.size(0),
            edge_count=edge_index.size(1),
            sample_path=sample_path,
            trajectory_path=trajectory_path,
            loss_csv_path=args.loss_csv,
            plot_path=args.plot_output if plot_written else None,
            extra_metrics=extra_metrics or None,
        )
        print(f"experiment_log={args.experiment_log}")

    print(f"checkpoint={cli_args.checkpoint}")
    print(f"resampled_seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
