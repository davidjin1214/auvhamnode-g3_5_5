#!/usr/bin/env python3
"""Utilities for the Phase-1A OC v4-lite formal workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"
DEFAULT_LOCAL_PROXY_ROOT = Path("/content") / "_proxy_suites"

PHASE_PREFIXES = ("smoke1", "smoke3", "decision")
PROTOCOL_TAGS = ("clean", "iid", "v4lite")
EXPORT_ARTIFACTS = (
    "runs.tsv",
    "suite_config.txt",
    "sweep_summary.json",
    "sweep_summary.txt",
    "sweep_seed_metrics.csv",
    "sweep_model_metrics.csv",
    "experiment_report.md",
    "phase1a_matrix.json",
    "phase1a_summary.csv",
    "phase1a_by_seed.csv",
    "phase1a_by_scenario.csv",
    "phase1a_by_horizon.csv",
    "phase1a_degradation.csv",
    "phase1a_protocol_delta.csv",
    "phase1a_train_audit.csv",
    "phase1a_v4_protocol_validation.json",
    "phase1a_decision_brief.md",
    "phase1a_run_config.json",
    "phase1a_environment.json",
)

WORKFLOW_ENV_KEYS = (
    "RUN_TAG",
    "DATASET",
    "NOISE_REFERENCE",
    "PHASE1A_MODELS",
    "SMOKE1_MODELS",
    "SMOKE_SEEDS",
    "DECISION_SEEDS",
    "SMOKE_EVAL_NUM_TRAJ_PER_SCENARIO",
    "DECISION_EVAL_NUM_TRAJ_PER_SCENARIO",
    "EVAL_TIMES",
    "EVAL_SCENARIOS",
    "EVAL_BASE_SEED",
    "EVAL_NOISE_SEED",
    "EVAL_PROGRESS_EVERY",
    "EVAL_NUM_DIAGNOSTIC_PLOTS",
    "IID_EVAL_PROFILES",
    "V4_EVAL_PROFILES",
    "STRICT_ZERO_NOISE_AUDIT",
    "SOFT_MIN_EPOCH_SCALE",
    "PYTHON_BIN",
    "DEVICE",
    "LOCAL_PROXY_ROOT",
    "PHASE1A_LOG_DIR",
    "PHASE1A_METADATA_DIR",
)


def _suite_name(prefix: str, protocol_tag: str, run_tag: str) -> str:
    return f"sweep_oc_phase1a_{prefix}_{protocol_tag}_{run_tag}"


def _proxy_name(prefix: str, run_tag: str) -> str:
    return f"sweep_oc_phase1a_{prefix}_proxy_{run_tag}"


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _git_output(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _read_runs(suite_dir: Path) -> pd.DataFrame:
    path = suite_dir / "runs.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing suite manifest: {path}")
    return pd.read_csv(path, sep="\t")


def _write_runs(suite_dir: Path, runs: pd.DataFrame) -> None:
    runs.to_csv(suite_dir / "runs.tsv", sep="\t", index=False)


def _resolve_run_dir(suite_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (suite_dir / path).resolve()


def _state_fingerprint(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _epoch_scale_at_best(
    resolved_protocol: str,
    resolved_profile: str,
    best_epoch,
    warmup_epochs: int,
    ramp_epochs: int,
    noise_scale: float,
) -> float:
    if resolved_protocol == "clean" or resolved_profile == "clean":
        return 0.0
    if best_epoch is None:
        return float("nan")
    if int(best_epoch) <= int(warmup_epochs):
        return 0.0
    progress = (float(best_epoch) - float(warmup_epochs)) / max(float(ramp_epochs), 1.0)
    progress = min(max(progress, 0.0), 1.0)
    return progress * float(noise_scale)


def _audit_suite(suite_dir: Path) -> pd.DataFrame:
    rows = []
    runs = _read_runs(suite_dir)
    for _, row in runs.iterrows():
        run_dir = _resolve_run_dir(suite_dir, row["run_dir"])
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
        best_epoch = checkpoint.get("epoch")
        scale_at_best = _epoch_scale_at_best(
            config.get("resolved_noise_protocol", "clean"),
            config.get("resolved_noise_profile", "clean"),
            best_epoch,
            int(config.get("noise_warmup_epochs", 20)),
            int(config.get("noise_ramp_epochs", 80)),
            float(config.get("noise_scale", 1.0)),
        )
        rows.append(
            {
                "suite_name": suite_dir.name,
                "group": row["group"],
                "model_type": row["model_type"],
                "seed": int(row["seed"]),
                "run_name": row["run_name"],
                "train_noise_profile": config.get(
                    "resolved_noise_profile", config.get("noise_profile")
                ),
                "train_noise_protocol": config.get(
                    "resolved_noise_protocol", config.get("noise_protocol")
                ),
                "best_epoch": int(best_epoch) if best_epoch is not None else None,
                "best_loss": float(checkpoint.get("loss"))
                if checkpoint.get("loss") is not None
                else float("nan"),
                "noise_warmup_epochs": int(config.get("noise_warmup_epochs", 20)),
                "noise_ramp_epochs": int(config.get("noise_ramp_epochs", 80)),
                "noise_mix_ratio": float(config.get("noise_mix_ratio", 0.5)),
                "epoch_scale_at_best": float(scale_at_best),
                "is_effectively_clean_at_best": bool(scale_at_best == 0.0),
                "state_fingerprint": _state_fingerprint(checkpoint["model_state_dict"]),
                "run_dir": str(run_dir),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["group", "model_type", "seed"]).reset_index(drop=True)
    frame.to_csv(suite_dir / "phase1a_train_audit.csv", index=False)
    return frame


def cmd_preflight(args: argparse.Namespace) -> int:
    checkpoint_root = args.checkpoint_root.resolve()
    local_proxy_root = args.local_proxy_root.resolve()
    conflicts = []

    for prefix in PHASE_PREFIXES:
        for protocol_tag in PROTOCOL_TAGS:
            path = checkpoint_root / _suite_name(prefix, protocol_tag, args.run_tag)
            if path.exists():
                conflicts.append(path)
        proxy_path = local_proxy_root / _proxy_name(prefix, args.run_tag)
        if proxy_path.exists():
            conflicts.append(proxy_path)

    exported_decision_proxy = checkpoint_root / _proxy_name("decision", args.run_tag)
    if exported_decision_proxy.exists():
        conflicts.append(exported_decision_proxy)

    if conflicts:
        print("Phase-1A clean execution requires fresh target paths.", file=sys.stderr)
        print("Change RUN_TAG or remove these directories first:", file=sys.stderr)
        for path in conflicts:
            print(f"- {path}", file=sys.stderr)
        return 2

    print("Preflight passed: all Phase-1A target suite/proxy directories are absent.")
    return 0


def cmd_write_metadata(args: argparse.Namespace) -> int:
    checkpoint_root = args.checkpoint_root.resolve()
    local_proxy_root = args.local_proxy_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow_env = {key: os.environ.get(key, "") for key in WORKFLOW_ENV_KEYS}
    run_config = {
        "run_tag": args.run_tag,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "checkpoint_root": str(checkpoint_root),
        "local_proxy_root": str(local_proxy_root),
        "phase_suites": {
            prefix: {
                protocol: str(checkpoint_root / _suite_name(prefix, protocol, args.run_tag))
                for protocol in PROTOCOL_TAGS
            }
            for prefix in PHASE_PREFIXES
        },
        "proxy_suites": {
            prefix: str(local_proxy_root / _proxy_name(prefix, args.run_tag))
            for prefix in PHASE_PREFIXES
        },
        "exported_decision_proxy": str(checkpoint_root / _proxy_name("decision", args.run_tag)),
        "workflow_env": workflow_env,
    }

    git_status = _git_output(["status", "--short"])
    environment = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "torch": {
            "version": getattr(torch, "__version__", ""),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available() else None,
        },
        "git": {
            "branch": _git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _git_output(["rev-parse", "HEAD"]),
            "dirty": bool(git_status),
            "status_short": git_status,
        },
    }

    (output_dir / "phase1a_run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "phase1a_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[metadata] {output_dir}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    frames = []
    for suite_name in args.suite_name:
        suite_dir = args.checkpoint_root.resolve() / suite_name
        print(f"[audit] {suite_dir}", flush=True)
        frames.append(_audit_suite(suite_dir))

    audit = pd.concat(frames, ignore_index=True)
    audit = audit.sort_values(["group", "model_type", "suite_name", "seed"]).reset_index(drop=True)

    display_cols = [
        "suite_name",
        "group",
        "model_type",
        "seed",
        "train_noise_protocol",
        "best_epoch",
        "best_loss",
        "noise_warmup_epochs",
        "noise_ramp_epochs",
        "noise_mix_ratio",
        "epoch_scale_at_best",
        "is_effectively_clean_at_best",
    ]
    print(audit[display_cols].to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.output, index=False)
        print(f"[audit] combined CSV: {args.output}")

    hard_flags = audit[
        (audit["train_noise_protocol"] != "clean")
        & (audit["epoch_scale_at_best"] <= 0.0)
    ]
    soft_flags = audit[
        (audit["train_noise_protocol"] != "clean")
        & (audit["epoch_scale_at_best"] < float(args.soft_min_epoch_scale))
    ]

    if not hard_flags.empty:
        print("[audit] noisy runs with best checkpoint in fully clean phase:", file=sys.stderr)
        print(hard_flags[display_cols].to_string(index=False), file=sys.stderr)
        if args.strict_zero_noise:
            return 3

    if not soft_flags.empty:
        print("[audit] warning: noisy runs with small epoch_scale_at_best:")
        print(soft_flags[display_cols].to_string(index=False))

    return 0


def _first_run_dir(suite_dir: Path) -> Path:
    runs = _read_runs(suite_dir)
    if runs.empty:
        raise ValueError(f"No runs in manifest: {suite_dir / 'runs.tsv'}")
    return _resolve_run_dir(suite_dir, runs.iloc[0]["run_dir"])


def cmd_validate(args: argparse.Namespace) -> int:
    suite_dir = args.checkpoint_root.resolve() / args.suite_name
    run_dir = _first_run_dir(suite_dir)
    _run(
        [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "validate_v4_lite_protocol.py"),
            "--run-dir",
            str(run_dir),
        ]
    )
    output_path = run_dir / "v4_lite_protocol_validation.json"
    if not output_path.exists():
        raise FileNotFoundError(f"Missing validation output: {output_path}")
    print(f"[validate] {output_path}")
    return 0


def _phase1a_protocol_tag_from_suite_name(suite_name: str) -> str | None:
    name = Path(suite_name).name.lower()
    for phase_prefix in PHASE_PREFIXES:
        marker = f"sweep_oc_phase1a_{phase_prefix}_"
        if not name.startswith(marker):
            continue
        protocol_tag = name[len(marker):].split("_", 1)[0]
        if protocol_tag in PROTOCOL_TAGS:
            return protocol_tag
    return None


def _proxy_prefix(suite_name: str) -> str:
    protocol_tag = _phase1a_protocol_tag_from_suite_name(suite_name)
    if protocol_tag:
        return protocol_tag
    lowered = Path(suite_name).name.lower()
    return lowered.replace("sweep_oc_", "").replace("/", "_")


def _copy_if_exists(src: Path, dst: Path, copied: list[str]) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(dst.name)


def _copy_metadata_files(metadata_dir: Path | None, suite_dir: Path) -> None:
    if metadata_dir is None:
        return
    for name in ("phase1a_run_config.json", "phase1a_environment.json"):
        _copy_if_exists(metadata_dir / name, suite_dir / name, [])


def _copy_log_dir(log_dir: Path | None, suite_dir: Path) -> None:
    if log_dir is None or not log_dir.exists():
        return
    shutil.copytree(log_dir, suite_dir / "phase1a_logs", dirs_exist_ok=True)


def _rewrite_proxy_manifest_paths(suite_dir: Path) -> None:
    runs = _read_runs(suite_dir)
    for idx, row in runs.iterrows():
        proxy_run_dir = suite_dir / str(row["run_name"])
        runs.at[idx, "run_dir"] = str(proxy_run_dir)
        runs.at[idx, "checkpoint"] = str(proxy_run_dir / "best_model.pt")
    _write_runs(suite_dir, runs)


def _validation_path_from_v4_suite(checkpoint_root: Path, suite_names: list[str]) -> Path | None:
    v4_suites = [
        name for name in suite_names
        if _phase1a_protocol_tag_from_suite_name(name) == "v4lite"
    ]
    if not v4_suites:
        return None
    run_dir = _first_run_dir(checkpoint_root / v4_suites[0])
    path = run_dir / "v4_lite_protocol_validation.json"
    return path if path.exists() else None


def cmd_register_proxy(args: argparse.Namespace) -> int:
    checkpoint_root = args.checkpoint_root.resolve()
    local_proxy_root = args.local_proxy_root.resolve()
    local_proxy_dir = local_proxy_root / args.proxy_suite_name
    export_dir = checkpoint_root / args.proxy_suite_name

    if local_proxy_dir.exists():
        raise FileExistsError(f"Local proxy already exists: {local_proxy_dir}")
    if args.export and export_dir.exists():
        raise FileExistsError(f"Export directory already exists: {export_dir}")

    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "register_existing_runs_as_suite.py"),
        "--suite-dir",
        str(local_proxy_dir),
    ]
    for suite_name in args.suite_name:
        suite_dir = checkpoint_root / suite_name
        runs = _read_runs(suite_dir)
        prefix = _proxy_prefix(suite_name)
        for _, row in runs.iterrows():
            source_run_dir = _resolve_run_dir(suite_dir, row["run_dir"])
            cmd.extend(
                [
                    "--run",
                    str(row["group"]),
                    str(row["model_type"]),
                    str(int(row["seed"])),
                    f"{prefix}__{row['run_name']}",
                    str(source_run_dir),
                ]
            )

    _run(cmd)
    _run([args.python_bin, str(REPO_ROOT / "scripts" / "summarize_sweep.py"), "--suite-dir", str(local_proxy_dir)])
    _run([args.python_bin, str(REPO_ROOT / "scripts" / "build_experiment_report.py"), "--suite-dir", str(local_proxy_dir)])

    validation_path = args.validation_path or _validation_path_from_v4_suite(
        checkpoint_root, args.suite_name
    )
    if args.audit_path:
        _copy_if_exists(args.audit_path, local_proxy_dir / "phase1a_train_audit.csv", [])
    if validation_path:
        _copy_if_exists(validation_path, local_proxy_dir / "phase1a_v4_protocol_validation.json", [])
    _copy_metadata_files(args.metadata_dir, local_proxy_dir)
    _copy_log_dir(args.log_dir, local_proxy_dir)

    if args.export:
        shutil.copytree(local_proxy_dir, export_dir, symlinks=True)
        _rewrite_proxy_manifest_paths(export_dir)
        _run([args.python_bin, str(REPO_ROOT / "scripts" / "summarize_sweep.py"), "--suite-dir", str(export_dir)])
        _run([args.python_bin, str(REPO_ROOT / "scripts" / "build_experiment_report.py"), "--suite-dir", str(export_dir)])
        if args.audit_path:
            _copy_if_exists(args.audit_path, export_dir / "phase1a_train_audit.csv", [])
        if validation_path:
            _copy_if_exists(validation_path, export_dir / "phase1a_v4_protocol_validation.json", [])
        _copy_metadata_files(args.metadata_dir, export_dir)
        _copy_log_dir(args.log_dir, export_dir)
        exported_artifacts = [name for name in EXPORT_ARTIFACTS if (export_dir / name).exists()]
        if (export_dir / "phase1a_logs").exists():
            exported_artifacts.append("phase1a_logs/")
        lines = [
            f"Local proxy suite: {local_proxy_dir}",
            f"Exported proxy suite: {export_dir}",
            "",
            "Source suites:",
            *[f"- {name}" for name in args.suite_name],
            "",
            "Exported artifacts:",
            *[f"- {name}" for name in exported_artifacts],
        ]
        (export_dir / "phase1a_export_info.txt").write_text(
            "\n".join(lines).rstrip() + "\n",
            encoding="utf-8",
        )
        print(f"[proxy] exported: {export_dir}")
    else:
        print(f"[proxy] local: {local_proxy_dir}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--local-proxy-root", type=Path, default=DEFAULT_LOCAL_PROXY_ROOT)

    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--run-tag", required=True)
    preflight.set_defaults(func=cmd_preflight)

    metadata = subparsers.add_parser("write-metadata")
    metadata.add_argument("--run-tag", required=True)
    metadata.add_argument("--output-dir", required=True, type=Path)
    metadata.set_defaults(func=cmd_write_metadata)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--suite-name", action="append", required=True)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--soft-min-epoch-scale", type=float, default=0.05)
    audit.add_argument("--strict-zero-noise", action="store_true")
    audit.set_defaults(func=cmd_audit)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--suite-name", required=True)
    validate.set_defaults(func=cmd_validate)

    register = subparsers.add_parser("register-proxy")
    register.add_argument("--proxy-suite-name", required=True)
    register.add_argument("--suite-name", action="append", required=True)
    register.add_argument("--audit-path", type=Path)
    register.add_argument("--validation-path", type=Path)
    register.add_argument("--metadata-dir", type=Path)
    register.add_argument("--log-dir", type=Path)
    register.add_argument("--export", action="store_true")
    register.set_defaults(func=cmd_register_proxy)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
