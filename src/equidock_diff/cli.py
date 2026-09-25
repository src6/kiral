"""Unified CLI entry point for Equidock-Diff."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import torch

from equidock_diff import __version__
from equidock_diff.models.amp_utils import get_device_benchmark_info
from equidock_diff.train import main as train_main, resolve_device
from equidock_diff.resample_from_checkpoint import main as resample_main


def _run_diagnostics(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="equidock diagnostics", description="Check hardware and ML environment")
    parser.add_argument("--device", default="auto", help="Device name to check (auto, cuda, mps, cpu)")
    args = parser.parse_args(argv)

    resolved = resolve_device(args.device)
    info = get_device_benchmark_info(resolved)

    print("=== Equidock-Diff Environment Diagnostics ===")
    print(f"Version:              {__version__}")
    print(f"PyTorch Version:      {torch.__version__}")
    print(f"Resolved Device:      {resolved}")
    print(f"CUDA Available:       {info.get('cuda_available', False)}")
    print(f"Device Name:          {info.get('device_name', 'Unknown')}")
    print(f"BF16 Supported:       {info.get('bf16_supported', False)}")
    print(f"torch.compile:        {info.get('compile_supported', False)}")
    if "vram_total_mb" in info:
        print(f"VRAM Total:           {info['vram_total_mb']:.1f} MB")
        print(f"VRAM Allocated:       {info['vram_allocated_mb']:.1f} MB")
    print("=============================================")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="equidock",
        description="Equidock-Diff: High-throughput SE(3)-equivariant molecular docking engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # dock subcommand
    subparsers.add_parser(
        "dock",
        help="Train score network or predict docked poses on protein-ligand pairs",
        add_help=False,
    )

    # resample subcommand
    subparsers.add_parser(
        "resample",
        help="Rerun reverse diffusion from an existing checkpoint with custom sampler settings",
        add_help=False,
    )

    # diagnostics subcommand
    subparsers.add_parser(
        "diagnostics",
        help="Inspect GPU, Tensor Core, and acceleration status",
        add_help=False,
    )

    if not argv:
        parser.print_help()
        return 0

    if argv[0] in ("-v", "--version"):
        print(f"equidock {__version__}")
        return 0

    subcommand = argv[0]
    remaining = argv[1:]

    if subcommand in ("-h", "--help"):
        parser.print_help()
        return 0

    if subcommand == "dock":
        return train_main(remaining)
    elif subcommand == "resample":
        return resample_main(remaining)
    elif subcommand == "diagnostics":
        return _run_diagnostics(remaining)
    else:
        # Fallback to train_main for backward compatibility with direct flag invocation
        return train_main(argv)


if __name__ == "__main__":
    sys.exit(main())
