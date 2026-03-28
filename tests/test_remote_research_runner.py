from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from equidock_diff.remote_research_runner import (
    GitSyncState,
    build_parser,
    build_remote_dataset_setup_command,
    build_remote_git_update_command,
    build_remote_runner_command,
    ensure_remote_host_responsive,
    local_summary_root,
    maybe_start_remote_caffeinate,
    remote_research_root,
    resolve_sync_mode,
)


def test_resolve_sync_mode_auto_prefers_git_for_clean_synced_branch() -> None:
    mode = resolve_sync_mode(
        "auto",
        GitSyncState(branch="topic", upstream="origin/topic", clean=True, ahead_count=0),
    )

    assert mode == "git"


def test_resolve_sync_mode_auto_falls_back_to_rsync_when_dirty() -> None:
    mode = resolve_sync_mode(
        "auto",
        GitSyncState(branch="topic", upstream="origin/topic", clean=False, ahead_count=0),
    )

    assert mode == "rsync"


def test_build_remote_dataset_setup_command_links_repo_local_dataset() -> None:
    command = build_remote_dataset_setup_command(
        PurePosixPath("/Users/sadik/Projects/equidock-diff"),
        PurePosixPath("/Volumes/Data/pdbbind_v2020"),
    )

    assert command is not None
    assert "ln -sfn" in command
    assert "data/pdbbind_v2020" in command
    assert "/Volumes/Data/pdbbind_v2020" in command


def test_build_remote_git_update_command_uses_dedicated_clone_branch_update() -> None:
    command = build_remote_git_update_command(
        PurePosixPath("/Users/sadik/Projects/equidock-diff"),
        branch="codex/fast-research-iteration",
        upstream="origin/codex/fast-research-iteration",
    )

    assert "git -C" in command
    assert "fetch origin --prune" in command
    assert "switch" in command
    assert "pull --ff-only" in command


def test_build_remote_runner_command_includes_research_args() -> None:
    args = build_parser().parse_args(
        [
            "--complex-id",
            "10gs",
            "--model",
            "frame_backbone",
            "--noise-schedule",
            "cosine",
            "--seed",
            "42",
            "--crop-cutoff",
            "8.0",
            "--ligand-protein-clash-weight",
            "0.02",
            "--sample-steps",
            "50",
            "--use-edge-attention",
            "--max-parallel",
            "2",
            "--stagger-seconds",
            "1.5",
            "--keep-going",
            "--device-policy",
            "scratch",
            "--tag",
            "remote_probe",
            "--remote-host",
            "mini.tailnet.ts.net",
            "--remote-repo",
            "/Users/sadik/Projects/equidock-diff",
        ]
    )

    command = build_remote_runner_command(args)

    assert "uv run python -m equidock_diff.research_runner" in command
    assert '"$HOME/.local/bin/uv" run python -m equidock_diff.research_runner' in command
    assert ".venv/bin/python -m equidock_diff.research_runner" in command
    assert "--complex-id 10gs" in command
    assert "--model frame_backbone" in command
    assert "--noise-schedule cosine" in command
    assert "--crop-cutoff 8.0" in command
    assert "--ligand-protein-clash-weight 0.02" in command
    assert "--use-edge-attention" in command
    assert "--max-parallel 2" in command
    assert "--stagger-seconds 1.5" in command
    assert "--keep-going" in command
    assert "--output-root /Users/sadik/Projects/equidock-diff/runs/research" in command


def test_remote_and_local_output_roots_are_derived_from_tag() -> None:
    args = build_parser().parse_args(
        [
            "--complex-id",
            "10gs",
            "--model",
            "baseline",
            "--noise-schedule",
            "linear",
            "--tag",
            "probe",
            "--remote-host",
            "mini.tailnet.ts.net",
            "--remote-repo",
            "/Users/sadik/Projects/equidock-diff",
        ]
    )

    assert remote_research_root(args) == PurePosixPath("/Users/sadik/Projects/equidock-diff/runs/research")
    assert local_summary_root(args) == Path("runs/remote/probe")


def test_ensure_remote_host_responsive_retries_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[str] = []

    def fake_run_remote_shell(remote_host: str, command: str) -> None:
        attempts.append(f"{remote_host}:{command}")
        if len(attempts) < 3:
            raise subprocess.CalledProcessError(255, ["ssh", remote_host, command], stderr="timeout")

    import subprocess

    monkeypatch.setattr(
        "equidock_diff.remote_research_runner.run_remote_shell",
        fake_run_remote_shell,
    )
    monkeypatch.setattr("equidock_diff.remote_research_runner.time.sleep", lambda _seconds: None)

    ensure_remote_host_responsive("macmini-tailscale", attempts=3, delay_seconds=0.1)

    assert len(attempts) == 3


def test_ensure_remote_host_responsive_raises_after_all_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    monkeypatch.setattr(
        "equidock_diff.remote_research_runner.run_remote_shell",
        lambda remote_host, command: (_ for _ in ()).throw(
            subprocess.CalledProcessError(255, ["ssh", remote_host, command], stderr="timeout")
        ),
    )
    monkeypatch.setattr("equidock_diff.remote_research_runner.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="did not become responsive"):
        ensure_remote_host_responsive("macmini-tailscale", attempts=2, delay_seconds=0.1)


def test_maybe_start_remote_caffeinate_runs_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "equidock_diff.remote_research_runner.run_remote_shell",
        lambda host, command: commands.append((host, command)),
    )
    args = build_parser().parse_args(
        [
            "--complex-id",
            "10gs",
            "--model",
            "frame_backbone",
            "--noise-schedule",
            "cosine",
            "--tag",
            "probe",
            "--remote-host",
            "macmini-tailscale",
            "--remote-repo",
            "/Users/sadik/Projects/equidock-diff",
        ]
    )

    maybe_start_remote_caffeinate(args)

    assert commands == [
        (
            "macmini-tailscale",
            "nohup caffeinate -dimsu -t 21600 >/tmp/equidock_diff_caffeinate.log 2>&1 </dev/null &",
        )
    ]


def test_maybe_start_remote_caffeinate_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "equidock_diff.remote_research_runner.run_remote_shell",
        lambda host, command: commands.append((host, command)),
    )
    args = build_parser().parse_args(
        [
            "--complex-id",
            "10gs",
            "--model",
            "frame_backbone",
            "--noise-schedule",
            "cosine",
            "--tag",
            "probe",
            "--remote-host",
            "macmini-tailscale",
            "--remote-repo",
            "/Users/sadik/Projects/equidock-diff",
            "--allow-remote-sleep",
        ]
    )

    maybe_start_remote_caffeinate(args)

    assert commands == []
