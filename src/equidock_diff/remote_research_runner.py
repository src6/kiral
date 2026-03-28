"""Launch research_runner remotely over Tailscale SSH."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from equidock_diff.data.io import load_split_complex_ids


DEFAULT_LOCAL_REMOTE_ROOT = Path("runs/remote")
DEFAULT_REMOTE_RESEARCH_ROOT = PurePosixPath("runs/research")
IGNORED_LOCAL_STATUS_PATHS = (".mplconfig/",)
DEFAULT_REMOTE_PING_ATTEMPTS = 3
DEFAULT_REMOTE_PING_DELAY_SECONDS = 5.0
DEFAULT_REMOTE_CAFFEINATE_SECONDS = 6 * 60 * 60


@dataclass(frozen=True)
class GitSyncState:
    branch: str | None
    upstream: str | None
    clean: bool
    ahead_count: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run equidock_diff.research_runner remotely over Tailscale SSH"
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--complex-id",
        dest="complex_ids",
        action="append",
        default=None,
        help="Complex id to run; may be passed multiple times",
    )
    scope.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest listing one complex id per line; expanded locally before remote execution",
    )
    parser.add_argument(
        "--model",
        choices=("baseline", "frame_backbone"),
        required=True,
        help="Research model family to run remotely",
    )
    parser.add_argument(
        "--noise-schedule",
        choices=("linear", "cosine"),
        required=True,
        help="Noise schedule shared by the matrix",
    )
    parser.add_argument("--seed", dest="seeds", action="append", type=int, default=None)
    parser.add_argument("--steps", dest="steps_values", action="append", type=int, default=None)
    parser.add_argument(
        "--context-policy",
        dest="context_policies",
        action="append",
        choices=("fixed", "adaptive", "gated"),
        default=None,
    )
    parser.add_argument(
        "--protein-node-budget",
        dest="protein_node_budgets",
        action="append",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--crop-cutoff",
        dest="crop_cutoffs",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--sample-steps",
        dest="sample_steps_values",
        action="append",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--sample-time-power",
        dest="sample_time_powers",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ligand-shape-weight",
        dest="ligand_shape_weights",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ligand-protein-clash-weight",
        dest="ligand_protein_clash_weights",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ligand-protein-contact-weight",
        dest="ligand_protein_contact_weights",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--use-edge-attention",
        action="store_true",
        help="Enable lightweight incoming-edge attention inside frame-backbone runs",
    )
    parser.add_argument(
        "--sample-score-clip",
        dest="sample_score_clips",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--sample-position-clip",
        dest="sample_position_clips",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--device-policy",
        choices=("scratch", "canonical", "explicit"),
        default="scratch",
        help="Device policy for the remote research runner",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device required when --device-policy explicit is used",
    )
    parser.add_argument("--tag", required=True, help="Tag name under the remote research output root")
    parser.add_argument(
        "--compare-against",
        default=None,
        help="Optional prior remote research tag to diff against automatically",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Fetch heavy pose/trajectory outputs too, not just summaries",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of training-signature groups to execute concurrently on the remote host",
    )
    parser.add_argument(
        "--stagger-seconds",
        type=float,
        default=0.0,
        help="Optional delay between launching parallel groups on the remote host",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue remote groups after a group failure instead of failing fast",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the remote research runner in dry-run mode",
    )
    parser.add_argument("--remote-host", required=True, help="Tailscale SSH host or IP for the Mac mini")
    parser.add_argument(
        "--remote-repo",
        type=PurePosixPath,
        required=True,
        help="Absolute path to the dedicated repo clone on the remote host",
    )
    parser.add_argument(
        "--remote-output-root",
        type=PurePosixPath,
        default=None,
        help="Optional remote output root; defaults to <remote-repo>/runs/research",
    )
    parser.add_argument(
        "--remote-dataset-target",
        type=PurePosixPath,
        default=None,
        help="Actual dataset root on the remote host used to populate data/pdbbind_v2020",
    )
    parser.add_argument(
        "--sync-mode",
        choices=("git", "rsync", "auto"),
        default="auto",
        help="How to update the remote dedicated clone before running",
    )
    parser.add_argument(
        "--fetch-full-results",
        action="store_true",
        help="Fetch the whole remote tag directory instead of only compact summaries",
    )
    parser.add_argument(
        "--remote-ping-attempts",
        type=int,
        default=DEFAULT_REMOTE_PING_ATTEMPTS,
        help="Number of SSH responsiveness checks before remote sync/run steps",
    )
    parser.add_argument(
        "--remote-ping-delay-seconds",
        type=float,
        default=DEFAULT_REMOTE_PING_DELAY_SECONDS,
        help="Delay between remote responsiveness checks",
    )
    parser.add_argument(
        "--allow-remote-sleep",
        action="store_true",
        help="Do not use caffeinate on the remote host during sync/setup/run steps",
    )
    parser.add_argument(
        "--remote-caffeinate-seconds",
        type=int,
        default=DEFAULT_REMOTE_CAFFEINATE_SECONDS,
        help="How long to keep the remote host awake when caffeinate is enabled",
    )
    return parser


def _defaulted(values: list[object] | None) -> list[object]:
    return [] if not values else list(values)


def _complex_ids_from_args(args: argparse.Namespace) -> list[str]:
    if args.manifest is not None:
        return load_split_complex_ids(args.manifest)
    assert args.complex_ids is not None
    return list(args.complex_ids)


def _run_local_git(
    repo_root: Path,
    *argv: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *argv],
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
    )


def local_git_sync_state(repo_root: Path) -> GitSyncState:
    branch = None
    upstream = None
    ahead_count = None
    try:
        branch = _run_local_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    except subprocess.CalledProcessError:
        branch = None
    try:
        upstream = _run_local_git(
            repo_root,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{u}",
        ).stdout.strip()
    except subprocess.CalledProcessError:
        upstream = None
    if upstream is not None:
        try:
            ahead_output = _run_local_git(repo_root, "rev-list", "--count", "@{u}..HEAD").stdout.strip()
            ahead_count = int(ahead_output)
        except (subprocess.CalledProcessError, ValueError):
            ahead_count = None

    porcelain = _run_local_git(repo_root, "status", "--porcelain", check=False).stdout.splitlines()
    filtered = [line for line in porcelain if not any(line[3:].startswith(prefix) for prefix in IGNORED_LOCAL_STATUS_PATHS)]
    return GitSyncState(
        branch=branch,
        upstream=upstream,
        clean=len(filtered) == 0,
        ahead_count=ahead_count,
    )


def resolve_sync_mode(requested_mode: str, state: GitSyncState) -> str:
    if requested_mode != "auto":
        return requested_mode
    if state.clean and state.upstream is not None and state.ahead_count == 0:
        return "git"
    return "rsync"


def remote_research_root(args: argparse.Namespace) -> PurePosixPath:
    if args.remote_output_root is not None:
        return args.remote_output_root
    return args.remote_repo / DEFAULT_REMOTE_RESEARCH_ROOT


def local_summary_root(args: argparse.Namespace) -> Path:
    return DEFAULT_LOCAL_REMOTE_ROOT / args.tag


def build_remote_dataset_setup_command(
    remote_repo: PurePosixPath,
    remote_dataset_target: PurePosixPath | None,
) -> str | None:
    link_path = remote_repo / "data" / "pdbbind_v2020"
    if remote_dataset_target is None:
        return None
    return (
        f"mkdir -p {shlex.quote(str(remote_repo / 'data'))} && "
        f"(test ! -e {shlex.quote(str(link_path))} || test -L {shlex.quote(str(link_path))}) && "
        f"rm -f {shlex.quote(str(link_path))} && "
        f"ln -s {shlex.quote(str(remote_dataset_target))} {shlex.quote(str(link_path))} && "
        f"test -f {shlex.quote(str(link_path / 'index' / 'README'))}"
    )


def build_remote_git_update_command(
    remote_repo: PurePosixPath,
    *,
    branch: str,
    upstream: str,
) -> str:
    quoted_repo = shlex.quote(str(remote_repo))
    quoted_branch = shlex.quote(branch)
    quoted_upstream = shlex.quote(upstream)
    return (
        f"test -d {quoted_repo}/.git && "
        f"git -C {quoted_repo} fetch origin --prune && "
        f"(git -C {quoted_repo} switch {quoted_branch} "
        f"|| git -C {quoted_repo} switch -c {quoted_branch} --track {quoted_upstream}) && "
        f"git -C {quoted_repo} pull --ff-only"
    )


def build_remote_runner_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    for complex_id in _complex_ids_from_args(args):
        argv.extend(["--complex-id", complex_id])
    argv.extend(["--model", args.model, "--noise-schedule", args.noise_schedule, "--tag", args.tag])
    argv.extend(["--device-policy", args.device_policy])
    if args.device_policy == "explicit":
        if args.device is None:
            raise ValueError("--device is required when --device-policy explicit is used.")
        argv.extend(["--device", args.device])
    if args.compare_against is not None:
        argv.extend(["--compare-against", args.compare_against])
    if args.save_artifacts:
        argv.append("--save-artifacts")
    if args.max_parallel != 1:
        argv.extend(["--max-parallel", str(args.max_parallel)])
    if args.stagger_seconds != 0.0:
        argv.extend(["--stagger-seconds", str(args.stagger_seconds)])
    if args.keep_going:
        argv.append("--keep-going")
    if args.dry_run:
        argv.append("--dry-run")
    for seed in _defaulted(args.seeds):
        argv.extend(["--seed", str(seed)])
    for steps in _defaulted(args.steps_values):
        argv.extend(["--steps", str(steps)])
    for context_policy in _defaulted(args.context_policies):
        argv.extend(["--context-policy", str(context_policy)])
    for protein_node_budget in _defaulted(args.protein_node_budgets):
        argv.extend(["--protein-node-budget", str(protein_node_budget)])
    for crop_cutoff in _defaulted(args.crop_cutoffs):
        argv.extend(["--crop-cutoff", str(crop_cutoff)])
    for sample_steps in _defaulted(args.sample_steps_values):
        argv.extend(["--sample-steps", str(sample_steps)])
    for value in _defaulted(args.sample_time_powers):
        argv.extend(["--sample-time-power", str(value)])
    for value in _defaulted(args.ligand_shape_weights):
        argv.extend(["--ligand-shape-weight", str(value)])
    for value in _defaulted(args.ligand_protein_clash_weights):
        argv.extend(["--ligand-protein-clash-weight", str(value)])
    for value in _defaulted(args.ligand_protein_contact_weights):
        argv.extend(["--ligand-protein-contact-weight", str(value)])
    for value in _defaulted(args.sample_score_clips):
        argv.extend(["--sample-score-clip", str(value)])
    for value in _defaulted(args.sample_position_clips):
        argv.extend(["--sample-position-clip", str(value)])
    if args.use_edge_attention:
        argv.append("--use-edge-attention")
    argv.extend(["--output-root", str(remote_research_root(args))])
    return argv


def build_remote_runner_command(args: argparse.Namespace) -> str:
    runner_argv = build_remote_runner_argv(args)
    quoted_repo = shlex.quote(str(args.remote_repo))
    joined_argv = shlex.join(runner_argv)
    return (
        f"cd {quoted_repo} && "
        f"if command -v uv >/dev/null 2>&1; then "
        f"UV_CACHE_DIR=.uv-cache uv run python -m equidock_diff.research_runner {joined_argv}; "
        f"elif [ -x \"$HOME/.local/bin/uv\" ]; then "
        f"UV_CACHE_DIR=.uv-cache \"$HOME/.local/bin/uv\" run python -m equidock_diff.research_runner {joined_argv}; "
        f"elif [ -x .venv/bin/python ]; then "
        f".venv/bin/python -m equidock_diff.research_runner {joined_argv}; "
        f"else "
        f"echo 'Neither uv, ~/.local/bin/uv, nor .venv/bin/python is available on the remote host.' >&2; exit 127; "
        f"fi"
    )


def _run_subprocess(
    argv: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        details: list[str] = [f"Command failed with exit code {completed.returncode}: {' '.join(argv)}"]
        if stdout:
            details.append(f"stdout:\n{stdout}")
        if stderr:
            details.append(f"stderr:\n{stderr}")
        raise subprocess.CalledProcessError(
            completed.returncode,
            argv,
            output=completed.stdout,
            stderr=completed.stderr,
        ) from RuntimeError("\n\n".join(details))
    return completed


def run_remote_shell(remote_host: str, command: str) -> subprocess.CompletedProcess[str]:
    return _run_subprocess(["ssh", remote_host, command])


def maybe_start_remote_caffeinate(args: argparse.Namespace) -> None:
    if args.allow_remote_sleep:
        return
    run_remote_shell(
        args.remote_host,
        (
            "nohup caffeinate -dimsu -t "
            f"{int(args.remote_caffeinate_seconds)} >/tmp/equidock_diff_caffeinate.log 2>&1 </dev/null &"
        ),
    )


def ensure_remote_host_responsive(
    remote_host: str,
    *,
    attempts: int,
    delay_seconds: float,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            run_remote_shell(remote_host, "printf ready")
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(max(0.0, delay_seconds))
    if last_error is not None:
        raise RuntimeError(
            f"Remote host {remote_host} did not become responsive after {attempts} attempts."
        ) from last_error


def ensure_remote_repo_exists(remote_host: str, remote_repo: PurePosixPath) -> None:
    run_remote_shell(remote_host, f"mkdir -p {shlex.quote(str(remote_repo))}")


def sync_remote_repo(
    repo_root: Path,
    args: argparse.Namespace,
    *,
    mode: str,
    git_state: GitSyncState,
) -> None:
    ensure_remote_repo_exists(args.remote_host, args.remote_repo)
    maybe_start_remote_caffeinate(args)
    if mode == "git":
        if git_state.branch is None or git_state.upstream is None:
            raise ValueError("git sync requires a local branch with an upstream.")
        run_remote_shell(
            args.remote_host,
            build_remote_git_update_command(
                args.remote_repo,
                branch=git_state.branch,
                upstream=git_state.upstream,
            ),
        )
        return

    rsync_target = f"{args.remote_host}:{args.remote_repo}/"
    _run_subprocess(
        [
            "rsync",
            "-az",
            "--delete",
            "--exclude",
            ".git",
            "--exclude",
            "docs/training",
            "--exclude",
            "docs/training/**",
            "--exclude",
            "runs/research",
            "--exclude",
            "runs/remote",
            "--exclude",
            ".uv-cache",
            "--exclude",
            ".venv",
            "--exclude",
            ".pytest_cache",
            "--exclude",
            "__pycache__",
            "--exclude",
            ".mplconfig",
            f"{repo_root}/",
            rsync_target,
        ]
    )


def fetch_remote_results(args: argparse.Namespace) -> Path:
    local_root = local_summary_root(args)
    local_root.mkdir(parents=True, exist_ok=True)
    remote_tag_dir = remote_research_root(args) / args.tag
    if args.fetch_full_results:
        _run_subprocess(["rsync", "-az", f"{args.remote_host}:{remote_tag_dir}/", f"{local_root}/"])
        return local_root

    _run_subprocess(
        [
            "rsync",
            "-az",
            f"{args.remote_host}:{remote_tag_dir / 'run_index.csv'}",
            str(local_root / "run_index.csv"),
        ]
    )
    _run_subprocess(
        [
            "rsync",
            "-az",
            f"{args.remote_host}:{remote_tag_dir / 'run_index.md'}",
            str(local_root / "run_index.md"),
        ]
    )
    _run_subprocess(
        [
            "rsync",
            "-az",
            f"{args.remote_host}:{remote_tag_dir / 'status.csv'}",
            str(local_root / "status.csv"),
        ]
    )
    if args.compare_against is not None:
        (local_root / "comparisons").mkdir(parents=True, exist_ok=True)
        _run_subprocess(
            [
                "rsync",
                "-az",
                "--include",
                "*/",
                "--include",
                "*.md",
                "--include",
                "*.csv",
                "--exclude",
                "*",
                f"{args.remote_host}:{remote_tag_dir / 'comparisons'}/",
                f"{local_root / 'comparisons'}/",
            ]
        )
    return local_root


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path.cwd()
    git_state = local_git_sync_state(repo_root)
    sync_mode = resolve_sync_mode(args.sync_mode, git_state)

    ensure_remote_host_responsive(
        args.remote_host,
        attempts=args.remote_ping_attempts,
        delay_seconds=args.remote_ping_delay_seconds,
    )
    sync_remote_repo(repo_root, args, mode=sync_mode, git_state=git_state)
    dataset_command = build_remote_dataset_setup_command(args.remote_repo, args.remote_dataset_target)
    if dataset_command is not None:
        ensure_remote_host_responsive(
            args.remote_host,
            attempts=args.remote_ping_attempts,
            delay_seconds=args.remote_ping_delay_seconds,
        )
        maybe_start_remote_caffeinate(args)
        run_remote_shell(args.remote_host, dataset_command)
    ensure_remote_host_responsive(
        args.remote_host,
        attempts=args.remote_ping_attempts,
        delay_seconds=args.remote_ping_delay_seconds,
    )
    maybe_start_remote_caffeinate(args)
    run_remote_shell(args.remote_host, build_remote_runner_command(args))
    ensure_remote_host_responsive(
        args.remote_host,
        attempts=args.remote_ping_attempts,
        delay_seconds=args.remote_ping_delay_seconds,
    )
    fetched_root = fetch_remote_results(args)

    print(f"remote_sync_mode={sync_mode}")
    print(f"remote_host={args.remote_host}")
    print(f"remote_repo={args.remote_repo}")
    print(f"remote_tag_root={remote_research_root(args) / args.tag}")
    print(f"local_summary_root={fetched_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
