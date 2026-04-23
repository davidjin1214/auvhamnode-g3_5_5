#!/usr/bin/env python3
"""Register existing run directories as a local suite proxy.

This creates a suite directory that matches the repository's `runs.tsv`
contract without copying large checkpoint artifacts. Each suite run directory
contains symlinks back to the original run outputs, and optional rollout roots
can be mounted under `<run>/rollout_benchmark/<rollout_run_id>`.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-dir",
        type=Path,
        required=True,
        help="Target suite directory. The suite name should start with 'sweep_oc_'.",
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=5,
        metavar=("GROUP", "MODEL_TYPE", "SEED", "RUN_NAME", "RUN_DIR"),
        required=True,
        help="Run entry to register. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--rollout",
        action="append",
        nargs=2,
        metavar=("RUN_NAME", "ROLLOUT_ROOT"),
        default=[],
        help=(
            "Optional rollout root to mount under "
            "<suite>/<run_name>/rollout_benchmark/<basename(ROLLOUT_ROOT)>."
        ),
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Optional line to append to suite_config.txt.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite conflicting proxy files inside the target suite directory.",
    )
    return parser.parse_args()


def _remove_existing(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _ensure_symlink(target: Path, link_path: Path, *, force: bool) -> None:
    target = target.resolve()
    if link_path.is_symlink():
        if link_path.resolve() == target:
            return
        if not force:
            raise FileExistsError(f"Symlink already exists with different target: {link_path}")
        link_path.unlink()
    elif link_path.exists():
        if not force:
            raise FileExistsError(f"Path already exists: {link_path}")
        _remove_existing(link_path)

    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target, target_is_directory=target.is_dir())


def _write_runs_tsv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["group", "model_type", "seed", "run_name", "run_dir", "checkpoint"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)

    suite_name = suite_dir.name
    if not suite_name.startswith("sweep_oc_"):
        raise ValueError(
            f"Suite directory name must start with 'sweep_oc_', got {suite_name!r}."
        )

    rollout_map = {
        run_name: Path(rollout_root).resolve()
        for run_name, rollout_root in args.rollout
    }

    run_rows = []
    source_lines = []
    for group, model_type, seed_text, run_name, run_dir_text in args.run:
        run_dir = Path(run_dir_text).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

        proxy_run_dir = suite_dir / run_name
        proxy_run_dir.mkdir(parents=True, exist_ok=True)

        for child in sorted(run_dir.iterdir()):
            if child.name == "rollout_benchmark" and run_name in rollout_map:
                continue
            _ensure_symlink(child, proxy_run_dir / child.name, force=args.force)

        rollout_root = rollout_map.get(run_name)
        if rollout_root is not None:
            if not rollout_root.exists():
                raise FileNotFoundError(f"Rollout root does not exist: {rollout_root}")
            rollout_link = proxy_run_dir / "rollout_benchmark" / rollout_root.name
            _ensure_symlink(rollout_root, rollout_link, force=args.force)

        proxy_checkpoint = proxy_run_dir / "best_model.pt"
        run_rows.append(
            {
                "group": group,
                "model_type": model_type,
                "seed": str(int(seed_text)),
                "run_name": run_name,
                "run_dir": str(proxy_run_dir),
                "checkpoint": str(proxy_checkpoint),
            }
        )
        source_lines.append(f"{run_name}: {run_dir}")

    _write_runs_tsv(suite_dir / "runs.tsv", run_rows)

    lines = [
        f"Suite: {suite_name}",
        "Type: proxy suite registered from existing run directories",
        "",
        "Source runs:",
        *source_lines,
    ]
    if args.note:
        lines.extend(["", "Notes:", *args.note])
    (suite_dir / "suite_config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Registered suite: {suite_dir}")
    print(f"Runs: {len(run_rows)}")
    print(f"Manifest: {suite_dir / 'runs.tsv'}")


if __name__ == "__main__":
    main()
