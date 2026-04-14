#!/usr/bin/env python3
"""Build the stage-1 OC experiment data catalog."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent

SUITE_FILE_SPECS = [
    ("suite_runs_tsv", "runs.tsv"),
    ("suite_config_txt", "suite_config.txt"),
    ("suite_sweep_summary_json", "sweep_summary.json"),
    ("suite_experiment_report_md", "experiment_report.md"),
]

RUN_FILE_SPECS = [
    ("config_json", "config.json"),
    ("noise_budgets_json", "noise_budgets.json"),
    ("training_history_pkl", "training_history.pkl"),
    ("training_log", "training.log"),
    ("block_eval_json", "block_evaluation.json"),
    ("heldout_eval_json", "heldout_evaluation.json"),
    ("best_model_pt", "best_model.pt"),
]

EVAL_PROFILE_KEYS = {
    "clean",
    "nominal_eval",
    "degraded_eval",
    "heading_biased_eval",
    "current_bias_eval",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints",
        help="Root directory that contains checkpoint suites.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis" / "oc_data_catalog",
        help="Directory for generated catalog tables.",
    )
    parser.add_argument(
        "--include-unused",
        action="store_true",
        help="Include suites under checkpoints/unused.",
    )
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_runs_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def iter_suite_dirs(checkpoints_dir: Path, include_unused: bool) -> Iterable[Path]:
    for runs_path in sorted(checkpoints_dir.rglob("runs.tsv")):
        suite_dir = runs_path.parent
        suite_name = suite_dir.name
        if not suite_name.startswith("sweep_oc_"):
            continue
        if not include_unused and "unused" in suite_dir.parts:
            continue
        yield suite_dir


def file_row(
    scope: str,
    suite_name: str,
    suite_family: str,
    path: Path,
    file_type: str,
    run_name: str = "",
    run_uid: str = "",
    rollout_run_id: str = "",
    eval_profile: str = "",
) -> dict:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "scope": scope,
        "suite_family": suite_family,
        "suite_name": suite_name,
        "run_name": run_name,
        "run_uid": run_uid,
        "file_type": file_type,
        "rollout_run_id": rollout_run_id,
        "eval_profile": eval_profile,
        "path": repo_relative(path),
        "exists": int(exists),
        "size_bytes": stat.st_size if stat else "",
        "mtime_epoch": int(stat.st_mtime) if stat else "",
    }


def derive_train_type(noise_profile: str | None) -> str:
    if noise_profile:
        return "noisy_train"
    return "clean_train"


def derive_status(core_flags: dict[str, bool]) -> str:
    if not core_flags["has_config_json"]:
        return "missing_config"
    required = [
        "has_training_history_pkl",
        "has_block_eval_json",
        "has_heldout_eval_json",
    ]
    if all(core_flags[name] for name in required):
        return "ok"
    return "partial"


def collect_rollout_summary_rows(
    suite_name: str,
    suite_family: str,
    run_name: str,
    run_uid: str,
    run_dir: Path,
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    rollout_root = run_dir / "rollout_benchmark"
    if not rollout_root.exists():
        return rows, 0

    summaries = sorted(rollout_root.rglob("summary.json"))
    for summary_path in summaries:
        rel = summary_path.relative_to(rollout_root)
        rollout_run_id = rel.parts[0] if rel.parts else ""
        eval_profile = ""
        if len(rel.parts) >= 3:
            eval_profile = rel.parts[1]
        elif len(rel.parts) == 2:
            eval_profile = "default"
        rows.append(
            file_row(
                scope="run",
                suite_name=suite_name,
                suite_family=suite_family,
                run_name=run_name,
                run_uid=run_uid,
                path=summary_path,
                file_type="rollout_summary_json",
                rollout_run_id=rollout_run_id,
                eval_profile=eval_profile,
            )
        )
    return rows, len(summaries)


def build_catalog(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    run_rows: list[dict] = []
    file_rows: list[dict] = []

    for suite_dir in iter_suite_dirs(args.checkpoints_dir, args.include_unused):
        suite_name = suite_dir.name
        suite_family = suite_dir.parent.name
        runs_path = suite_dir / "runs.tsv"

        for file_type, filename in SUITE_FILE_SPECS:
            file_rows.append(
                file_row(
                    scope="suite",
                    suite_name=suite_name,
                    suite_family=suite_family,
                    path=suite_dir / filename,
                    file_type=file_type,
                )
            )

        for row in read_runs_tsv(runs_path):
            run_name = row["run_name"]
            run_uid = f"{suite_name}/{run_name}"
            run_dir = suite_dir / run_name
            config_path = run_dir / "config.json"
            config = load_json(config_path) if config_path.exists() else {}

            noise_profile = config.get("noise_profile")
            noise_reference = config.get("noise_reference") or config.get(
                "noise_reference_model"
            )
            core_flags = {
                "has_config_json": config_path.exists(),
                "has_noise_budgets_json": (run_dir / "noise_budgets.json").exists(),
                "has_training_history_pkl": (run_dir / "training_history.pkl").exists(),
                "has_training_log": (run_dir / "training.log").exists(),
                "has_block_eval_json": (run_dir / "block_evaluation.json").exists(),
                "has_heldout_eval_json": (run_dir / "heldout_evaluation.json").exists(),
                "has_best_model_pt": (run_dir / "best_model.pt").exists(),
            }

            rollout_file_rows, rollout_summary_count = collect_rollout_summary_rows(
                suite_name=suite_name,
                suite_family=suite_family,
                run_name=run_name,
                run_uid=run_uid,
                run_dir=run_dir,
            )
            file_rows.extend(rollout_file_rows)

            run_rows.append(
                {
                    "suite_family": suite_family,
                    "suite_name": suite_name,
                    "suite_dir": repo_relative(suite_dir),
                    "source_runs_tsv": repo_relative(runs_path),
                    "group": row["group"],
                    "model_type": row["model_type"],
                    "seed": int(row["seed"]),
                    "run_name": run_name,
                    "run_uid": run_uid,
                    "run_dir": repo_relative(run_dir),
                    "source_run_dir_raw": row.get("run_dir", ""),
                    "source_checkpoint_raw": row.get("checkpoint", ""),
                    "dataset_id": config.get("dataset_id", ""),
                    "dataset_path": config.get("dataset_path", ""),
                    "dataset_description": config.get("dataset_description", ""),
                    "train_type": derive_train_type(noise_profile),
                    "noise_profile_train": noise_profile or "clean",
                    "noise_reference": noise_reference or "",
                    "ocean_current": config.get("ocean_current", ""),
                    "device": config.get("device", ""),
                    "num_epochs": config.get("num_epochs", ""),
                    "has_config_json": int(core_flags["has_config_json"]),
                    "has_noise_budgets_json": int(core_flags["has_noise_budgets_json"]),
                    "has_training_history_pkl": int(core_flags["has_training_history_pkl"]),
                    "has_training_log": int(core_flags["has_training_log"]),
                    "has_block_eval_json": int(core_flags["has_block_eval_json"]),
                    "has_heldout_eval_json": int(core_flags["has_heldout_eval_json"]),
                    "has_best_model_pt": int(core_flags["has_best_model_pt"]),
                    "rollout_summary_count": rollout_summary_count,
                    "status": derive_status(core_flags),
                }
            )

            for file_type, filename in RUN_FILE_SPECS:
                file_rows.append(
                    file_row(
                        scope="run",
                        suite_name=suite_name,
                        suite_family=suite_family,
                        run_name=run_name,
                        run_uid=run_uid,
                        path=run_dir / filename,
                        file_type=file_type,
                    )
                )

    run_rows.sort(key=lambda row: (row["suite_name"], row["run_name"]))
    file_rows.sort(
        key=lambda row: (
            row["suite_name"],
            row["run_name"],
            row["file_type"],
            row["rollout_run_id"],
            row["eval_profile"],
            row["path"],
        )
    )
    return run_rows, file_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_history_metric_key(metric_key: str) -> tuple[str, str]:
    if metric_key.startswith("train_"):
        return "train", metric_key[len("train_") :]
    if metric_key.startswith("test_"):
        return "test", metric_key[len("test_") :]
    return "meta", metric_key


def write_training_history_long(path: Path, run_rows: list[dict]) -> None:
    fieldnames = [
        "suite_family",
        "suite_name",
        "group",
        "model_type",
        "seed",
        "run_name",
        "run_uid",
        "train_type",
        "noise_profile_train",
        "dataset_id",
        "epoch",
        "global_step",
        "split",
        "metric_key",
        "metric_name",
        "metric_value",
        "source_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for run_row in run_rows:
            history_path = REPO_ROOT / run_row["run_dir"] / "training_history.pkl"
            if not history_path.exists():
                continue

            with history_path.open("rb") as source:
                history = pickle.load(source)

            epochs = history.get("epoch")
            if not isinstance(epochs, list) or not epochs:
                continue

            global_steps = history.get("global_step", [None] * len(epochs))
            if not isinstance(global_steps, list) or len(global_steps) != len(epochs):
                global_steps = [None] * len(epochs)

            for metric_key in sorted(history.keys()):
                if metric_key in {"epoch", "global_step"}:
                    continue

                values = history[metric_key]
                if not isinstance(values, list) or len(values) != len(epochs):
                    continue

                split, metric_name = parse_history_metric_key(metric_key)
                for epoch, global_step, metric_value in zip(epochs, global_steps, values):
                    writer.writerow(
                        {
                            "suite_family": run_row["suite_family"],
                            "suite_name": run_row["suite_name"],
                            "group": run_row["group"],
                            "model_type": run_row["model_type"],
                            "seed": run_row["seed"],
                            "run_name": run_row["run_name"],
                            "run_uid": run_row["run_uid"],
                            "train_type": run_row["train_type"],
                            "noise_profile_train": run_row["noise_profile_train"],
                            "dataset_id": run_row["dataset_id"],
                            "epoch": epoch,
                            "global_step": global_step,
                            "split": split,
                            "metric_key": metric_key,
                            "metric_name": metric_name,
                            "metric_value": metric_value,
                            "source_file": repo_relative(history_path),
                        }
                    )


def scalar_type(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if value is None:
        return "null"
    return type(value).__name__


def scalar_numeric(value: object) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def flatten_leaf_items(
    value: object,
    prefix: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], object]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_leaf_items(child, prefix + (str(key),))
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_leaf_items(child, prefix + (str(index),))
        return
    yield prefix, value


def profile_eval_dicts(payload: dict, default_profile: str) -> list[tuple[str, dict]]:
    if payload and set(payload.keys()).issubset(EVAL_PROFILE_KEYS):
        return [(profile, value) for profile, value in payload.items() if isinstance(value, dict)]
    return [(default_profile, payload)]


def write_generic_eval_long(
    path: Path,
    run_rows: list[dict],
    filename: str,
    default_profile: str,
) -> None:
    fieldnames = [
        "suite_family",
        "suite_name",
        "group",
        "model_type",
        "seed",
        "run_name",
        "run_uid",
        "train_type",
        "noise_profile_train",
        "dataset_id",
        "eval_profile",
        "metric_path",
        "value_type",
        "value_text",
        "value_numeric",
        "source_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for run_row in run_rows:
            eval_path = REPO_ROOT / run_row["run_dir"] / filename
            if not eval_path.exists():
                continue

            payload = load_json(eval_path)
            for eval_profile, profile_payload in profile_eval_dicts(payload, default_profile):
                for key_path, value in flatten_leaf_items(profile_payload):
                    writer.writerow(
                        {
                            "suite_family": run_row["suite_family"],
                            "suite_name": run_row["suite_name"],
                            "group": run_row["group"],
                            "model_type": run_row["model_type"],
                            "seed": run_row["seed"],
                            "run_name": run_row["run_name"],
                            "run_uid": run_row["run_uid"],
                            "train_type": run_row["train_type"],
                            "noise_profile_train": run_row["noise_profile_train"],
                            "dataset_id": run_row["dataset_id"],
                            "eval_profile": eval_profile,
                            "metric_path": ".".join(key_path),
                            "value_type": scalar_type(value),
                            "value_text": "" if value is None else str(value),
                            "value_numeric": scalar_numeric(value),
                            "source_file": repo_relative(eval_path),
                        }
                    )


def rollout_summary_paths(run_dir: Path) -> list[tuple[Path, str, str]]:
    rollout_root = run_dir / "rollout_benchmark"
    if not rollout_root.exists():
        return []

    rows: list[tuple[Path, str, str]] = []
    for summary_path in sorted(rollout_root.rglob("summary.json")):
        rel = summary_path.relative_to(rollout_root)
        rollout_run_id = rel.parts[0] if rel.parts else ""
        if len(rel.parts) >= 3:
            eval_profile = rel.parts[1]
        else:
            eval_profile = "clean"
        rows.append((summary_path, rollout_run_id, eval_profile))
    return rows


def parse_horizon_value(raw: str) -> str:
    try:
        return str(float(raw))
    except (TypeError, ValueError):
        return raw


def write_rollout_summary_long(path: Path, run_rows: list[dict]) -> None:
    fieldnames = [
        "suite_family",
        "suite_name",
        "group",
        "model_type",
        "seed",
        "run_name",
        "run_uid",
        "train_type",
        "noise_profile_train",
        "dataset_id",
        "rollout_run_id",
        "eval_profile",
        "scope",
        "scenario",
        "horizon_s",
        "category",
        "metric_name",
        "stat_name",
        "value_type",
        "value_text",
        "value_numeric",
        "source_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for run_row in run_rows:
            run_dir = REPO_ROOT / run_row["run_dir"]
            for summary_path, rollout_run_id, eval_profile in rollout_summary_paths(run_dir):
                payload = load_json(summary_path)

                for scope_name in ("overall", "by_scenario"):
                    scope_payload = payload.get(scope_name, {})
                    if not isinstance(scope_payload, dict):
                        continue

                    if scope_name == "overall":
                        scenario_items = [("", scope_payload)]
                    else:
                        scenario_items = [
                            (scenario, scenario_payload)
                            for scenario, scenario_payload in scope_payload.items()
                            if isinstance(scenario_payload, dict)
                        ]

                    for scenario, horizons_payload in scenario_items:
                        for horizon_key, horizon_payload in horizons_payload.items():
                            if not isinstance(horizon_payload, dict):
                                continue

                            horizon_s = parse_horizon_value(horizon_key)

                            n_traj = horizon_payload.get("n_trajectories")
                            if n_traj is not None:
                                writer.writerow(
                                    {
                                        "suite_family": run_row["suite_family"],
                                        "suite_name": run_row["suite_name"],
                                        "group": run_row["group"],
                                        "model_type": run_row["model_type"],
                                        "seed": run_row["seed"],
                                        "run_name": run_row["run_name"],
                                        "run_uid": run_row["run_uid"],
                                        "train_type": run_row["train_type"],
                                        "noise_profile_train": run_row["noise_profile_train"],
                                        "dataset_id": run_row["dataset_id"],
                                        "rollout_run_id": rollout_run_id,
                                        "eval_profile": eval_profile,
                                        "scope": scope_name,
                                        "scenario": scenario,
                                        "horizon_s": horizon_s,
                                        "category": "meta",
                                        "metric_name": "n_trajectories",
                                        "stat_name": "",
                                        "value_type": scalar_type(n_traj),
                                        "value_text": str(n_traj),
                                        "value_numeric": scalar_numeric(n_traj),
                                        "source_file": repo_relative(summary_path),
                                    }
                                )

                            metrics_payload = horizon_payload.get("metrics", {})
                            for metric_name, metric_stats in metrics_payload.items():
                                if not isinstance(metric_stats, dict):
                                    continue
                                for stat_name, value in metric_stats.items():
                                    writer.writerow(
                                        {
                                            "suite_family": run_row["suite_family"],
                                            "suite_name": run_row["suite_name"],
                                            "group": run_row["group"],
                                            "model_type": run_row["model_type"],
                                            "seed": run_row["seed"],
                                            "run_name": run_row["run_name"],
                                            "run_uid": run_row["run_uid"],
                                            "train_type": run_row["train_type"],
                                            "noise_profile_train": run_row["noise_profile_train"],
                                            "dataset_id": run_row["dataset_id"],
                                            "rollout_run_id": rollout_run_id,
                                            "eval_profile": eval_profile,
                                            "scope": scope_name,
                                            "scenario": scenario,
                                            "horizon_s": horizon_s,
                                            "category": "metrics",
                                            "metric_name": metric_name,
                                            "stat_name": stat_name,
                                            "value_type": scalar_type(value),
                                            "value_text": str(value),
                                            "value_numeric": scalar_numeric(value),
                                            "source_file": repo_relative(summary_path),
                                        }
                                    )

                            rates_payload = horizon_payload.get("rates", {})
                            for metric_name, value in rates_payload.items():
                                writer.writerow(
                                    {
                                        "suite_family": run_row["suite_family"],
                                        "suite_name": run_row["suite_name"],
                                        "group": run_row["group"],
                                        "model_type": run_row["model_type"],
                                        "seed": run_row["seed"],
                                        "run_name": run_row["run_name"],
                                        "run_uid": run_row["run_uid"],
                                        "train_type": run_row["train_type"],
                                        "noise_profile_train": run_row["noise_profile_train"],
                                        "dataset_id": run_row["dataset_id"],
                                        "rollout_run_id": rollout_run_id,
                                        "eval_profile": eval_profile,
                                        "scope": scope_name,
                                        "scenario": scenario,
                                        "horizon_s": horizon_s,
                                        "category": "rates",
                                        "metric_name": metric_name,
                                        "stat_name": "",
                                        "value_type": scalar_type(value),
                                        "value_text": str(value),
                                        "value_numeric": scalar_numeric(value),
                                        "source_file": repo_relative(summary_path),
                                    }
                                )


def write_rollout_outcomes_long(path: Path, run_rows: list[dict]) -> None:
    fieldnames = [
        "suite_family",
        "suite_name",
        "group",
        "model_type",
        "seed",
        "run_name",
        "run_uid",
        "train_type",
        "noise_profile_train",
        "dataset_id",
        "rollout_run_id",
        "eval_profile",
        "scope",
        "scenario",
        "measure_group",
        "metric_name",
        "value_type",
        "value_text",
        "value_numeric",
        "source_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for run_row in run_rows:
            run_dir = REPO_ROOT / run_row["run_dir"]
            for summary_path, rollout_run_id, eval_profile in rollout_summary_paths(run_dir):
                payload = load_json(summary_path)
                outcome_payload = payload.get("rollout_outcomes", {})
                if not isinstance(outcome_payload, dict):
                    continue

                for scope_name in ("overall", "by_scenario"):
                    scope_data = outcome_payload.get(scope_name, {})
                    if not isinstance(scope_data, dict):
                        continue

                    if scope_name == "overall":
                        scenario_items = [("", scope_data)]
                    else:
                        scenario_items = [
                            (scenario, scenario_payload)
                            for scenario, scenario_payload in scope_data.items()
                            if isinstance(scenario_payload, dict)
                        ]

                    for scenario, scenario_payload in scenario_items:
                        for measure_group, measure_data in scenario_payload.items():
                            if isinstance(measure_data, dict):
                                for metric_name, value in measure_data.items():
                                    writer.writerow(
                                        {
                                            "suite_family": run_row["suite_family"],
                                            "suite_name": run_row["suite_name"],
                                            "group": run_row["group"],
                                            "model_type": run_row["model_type"],
                                            "seed": run_row["seed"],
                                            "run_name": run_row["run_name"],
                                            "run_uid": run_row["run_uid"],
                                            "train_type": run_row["train_type"],
                                            "noise_profile_train": run_row["noise_profile_train"],
                                            "dataset_id": run_row["dataset_id"],
                                            "rollout_run_id": rollout_run_id,
                                            "eval_profile": eval_profile,
                                            "scope": scope_name,
                                            "scenario": scenario,
                                            "measure_group": measure_group,
                                            "metric_name": metric_name,
                                            "value_type": scalar_type(value),
                                            "value_text": str(value),
                                            "value_numeric": scalar_numeric(value),
                                            "source_file": repo_relative(summary_path),
                                        }
                                    )
                            else:
                                writer.writerow(
                                    {
                                        "suite_family": run_row["suite_family"],
                                        "suite_name": run_row["suite_name"],
                                        "group": run_row["group"],
                                        "model_type": run_row["model_type"],
                                        "seed": run_row["seed"],
                                        "run_name": run_row["run_name"],
                                        "run_uid": run_row["run_uid"],
                                        "train_type": run_row["train_type"],
                                        "noise_profile_train": run_row["noise_profile_train"],
                                        "dataset_id": run_row["dataset_id"],
                                        "rollout_run_id": rollout_run_id,
                                        "eval_profile": eval_profile,
                                        "scope": scope_name,
                                        "scenario": scenario,
                                        "measure_group": "meta",
                                        "metric_name": measure_group,
                                        "value_type": scalar_type(measure_data),
                                        "value_text": str(measure_data),
                                        "value_numeric": scalar_numeric(measure_data),
                                        "source_file": repo_relative(summary_path),
                                    }
                                )


def derive_experiment_bucket(run_row: dict) -> str:
    suite_name = run_row["suite_name"]
    if suite_name == "sweep_oc_main_noise_seed42_smoke":
        return "smoke"
    if suite_name.startswith("sweep_oc_main_noise_nominal_train_remus100_dr_extra_"):
        return "p1_1_main_extra"
    if suite_name.startswith("sweep_oc_key_ablation_noise_nominal_train_remus100_dr_extra_"):
        return "p1_1_ablation_extra"
    if suite_name.startswith("sweep_oc_core_default_"):
        return "clean_core"
    if suite_name.startswith("sweep_oc_ablation_default_"):
        return "clean_ablation"
    if suite_name.startswith("sweep_oc_phnode_focus_extra3_"):
        return "clean_phnode_focus"
    if suite_name.startswith("sweep_oc_main_noise_nominal_train_remus100_dr_"):
        return "noisy_main"
    if suite_name.startswith("sweep_oc_baseline_noise_nominal_train_remus100_dr_"):
        return "noisy_baseline"
    if suite_name.startswith("sweep_oc_ablation_noise_nominal_train_remus100_dr_"):
        return "noisy_ablation"
    return "other"


def build_run_annotations(run_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    primary_buckets = {
        "clean_core",
        "clean_ablation",
        "clean_phnode_focus",
        "noisy_main",
        "noisy_baseline",
        "noisy_ablation",
    }
    followup_buckets = {"p1_1_main_extra", "p1_1_ablation_extra"}

    for run_row in run_rows:
        bucket = derive_experiment_bucket(run_row)
        train_data_type = "noisy" if run_row["train_type"] == "noisy_train" else "clean"
        is_smoke = bucket == "smoke"
        is_followup = bucket in followup_buckets
        is_primary_experiment = bucket in primary_buckets

        notes: list[str] = []
        if is_smoke:
            notes.append("Smoke validation run; exclude from default result views.")
        if is_followup:
            notes.append("Follow-up run added after the original sweep.")
        if bucket == "clean_phnode_focus":
            notes.append("Clean PHNODE-focused seed extension suite.")

        rows.append(
            {
                "run_uid": run_row["run_uid"],
                "suite_family": run_row["suite_family"],
                "suite_name": run_row["suite_name"],
                "group": run_row["group"],
                "model_type": run_row["model_type"],
                "seed": run_row["seed"],
                "train_type": run_row["train_type"],
                "train_data_type": train_data_type,
                "experiment_bucket": bucket,
                "is_primary_experiment": int(is_primary_experiment),
                "is_followup": int(is_followup),
                "is_smoke": int(is_smoke),
                "notes": " ".join(notes),
            }
        )

    rows.sort(key=lambda row: row["run_uid"])
    return rows


def rollout_run_timestamp_key(rollout_run_id: str) -> str:
    match = re.search(r"(\d{8}_\d{6})$", rollout_run_id)
    if match:
        return match.group(1)
    return rollout_run_id


def derive_rollout_purpose(run_row: dict, rollout_run_id: str) -> tuple[str, int, int, str]:
    bucket = derive_experiment_bucket(run_row)
    notes = []
    if bucket == "smoke":
        return "smoke", 10, 0, "Smoke suite rollout."
    if "probe" in rollout_run_id:
        return "probe", 20, 0, "Probe rollout; keep for diagnostics only."
    if rollout_run_id.startswith("heldout_"):
        return "legacy_heldout", 30, 0, "Legacy heldout rollout."
    if rollout_run_id.startswith("p12_matched_"):
        return "matched_followup", 80, 1, "Matched follow-up rollout from P1-2."
    return "primary", 100, 1, "Primary rollout benchmark."


def build_rollout_run_registry(run_rows: list[dict]) -> list[dict]:
    registry_rows: list[dict] = []

    for run_row in run_rows:
        run_dir = REPO_ROOT / run_row["run_dir"]
        for summary_path, rollout_run_id, eval_profile in rollout_summary_paths(run_dir):
            purpose, priority, eligible, notes = derive_rollout_purpose(run_row, rollout_run_id)
            registry_rows.append(
                {
                    "run_uid": run_row["run_uid"],
                    "run_name": run_row["run_name"],
                    "suite_name": run_row["suite_name"],
                    "model_type": run_row["model_type"],
                    "seed": run_row["seed"],
                    "train_type": run_row["train_type"],
                    "rollout_run_id": rollout_run_id,
                    "eval_profile": eval_profile,
                    "rollout_purpose": purpose,
                    "selection_priority": priority,
                    "is_selection_eligible": eligible,
                    "is_canonical": 0,
                    "source_file": repo_relative(summary_path),
                    "notes": notes,
                }
            )

    chosen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(registry_rows):
        if not row["is_selection_eligible"]:
            continue
        key = (row["run_uid"], row["eval_profile"])
        current = chosen.get(key)
        if current is None:
            chosen[key] = index
            continue

        candidate = registry_rows[current]
        better = (row["selection_priority"], rollout_run_timestamp_key(row["rollout_run_id"]))
        incumbent = (
            candidate["selection_priority"],
            rollout_run_timestamp_key(candidate["rollout_run_id"]),
        )
        if better > incumbent:
            chosen[key] = index

    for index in chosen.values():
        registry_rows[index]["is_canonical"] = 1

    registry_rows.sort(
        key=lambda row: (
            row["run_uid"],
            row["eval_profile"],
            -int(row["selection_priority"]),
            row["rollout_run_id"],
        )
    )
    return registry_rows


def canonical_rollout_keys(rollout_registry: list[dict]) -> set[tuple[str, str, str]]:
    keys = set()
    for row in rollout_registry:
        if int(row["is_canonical"]):
            keys.add((row["run_uid"], row["rollout_run_id"], row["eval_profile"]))
    return keys


def write_canonical_rollout_subset(
    source_path: Path,
    target_path: Path,
    canonical_keys: set[tuple[str, str, str]],
) -> int:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with source_path.open("r", encoding="utf-8", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        fieldnames = reader.fieldnames or []
        with target_path.open("w", encoding="utf-8", newline="") as target_handle:
            writer = csv.DictWriter(target_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                key = (row["run_uid"], row["rollout_run_id"], row["eval_profile"])
                if key not in canonical_keys:
                    continue
                writer.writerow(row)
                written += 1
    return written


def build_canonical_run_inventory(
    run_rows: list[dict],
    canonical_keys: set[tuple[str, str, str]],
) -> list[dict]:
    canonical_run_uids = {run_uid for run_uid, _, _ in canonical_keys}
    rows = [row for row in run_rows if row["run_uid"] in canonical_run_uids]
    rows.sort(key=lambda row: row["run_uid"])
    return rows


def write_catalog_qc_report(
    path: Path,
    run_rows: list[dict],
    run_annotations: list[dict],
    rollout_registry: list[dict],
    canonical_summary_rows: int,
    canonical_outcome_rows: int,
) -> None:
    run_bucket_counts: dict[str, int] = {}
    for row in run_annotations:
        run_bucket_counts[row["experiment_bucket"]] = (
            run_bucket_counts.get(row["experiment_bucket"], 0) + 1
        )

    rollout_purpose_counts: dict[str, int] = {}
    canonical_profile_counts: dict[str, int] = {}
    multi_rollout_counts: dict[str, int] = {}

    by_run_uid: dict[str, set[str]] = {}
    for row in rollout_registry:
        rollout_purpose_counts[row["rollout_purpose"]] = (
            rollout_purpose_counts.get(row["rollout_purpose"], 0) + 1
        )
        if int(row["is_canonical"]):
            canonical_profile_counts[row["eval_profile"]] = (
                canonical_profile_counts.get(row["eval_profile"], 0) + 1
            )
        by_run_uid.setdefault(row["run_uid"], set()).add(row["rollout_run_id"])

    for rollout_ids in by_run_uid.values():
        count = len(rollout_ids)
        multi_rollout_counts[str(count)] = multi_rollout_counts.get(str(count), 0) + 1

    lines = [
        "# OC Data Catalog QC Report",
        "",
        "## Summary",
        "",
        f"- Total runs: `{len(run_rows)}`",
        f"- Total annotated runs: `{len(run_annotations)}`",
        f"- Total rollout registry rows: `{len(rollout_registry)}`",
        f"- Canonical rollout summary rows: `{canonical_summary_rows}`",
        f"- Canonical rollout outcome rows: `{canonical_outcome_rows}`",
        "",
        "## Run Buckets",
        "",
    ]

    for bucket, count in sorted(run_bucket_counts.items()):
        lines.append(f"- `{bucket}`: `{count}`")

    lines.extend(["", "## Rollout Purposes", ""])
    for purpose, count in sorted(rollout_purpose_counts.items()):
        lines.append(f"- `{purpose}`: `{count}`")

    lines.extend(["", "## Canonical Profiles", ""])
    for profile, count in sorted(canonical_profile_counts.items()):
        lines.append(f"- `{profile}`: `{count}`")

    lines.extend(["", "## Distinct Rollout Runs Per Run UID", ""])
    for count_key, count in sorted(multi_rollout_counts.items(), key=lambda item: int(item[0])):
        lines.append(f"- `{count_key}` rollout runs: `{count}` run_uids")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_rows, file_rows = build_catalog(args)
    run_annotations = build_run_annotations(run_rows)
    rollout_registry = build_rollout_run_registry(run_rows)
    canonical_keys = canonical_rollout_keys(rollout_registry)
    canonical_run_rows = build_canonical_run_inventory(run_rows, canonical_keys)

    write_csv(args.output_dir / "run_inventory.csv", run_rows)
    write_csv(args.output_dir / "file_inventory.csv", file_rows)
    write_csv(args.output_dir / "run_annotations.csv", run_annotations)
    write_csv(args.output_dir / "rollout_run_registry.csv", rollout_registry)
    write_csv(args.output_dir / "canonical_run_inventory.csv", canonical_run_rows)
    write_training_history_long(args.output_dir / "training_history_long.csv", run_rows)
    write_generic_eval_long(
        args.output_dir / "block_eval_long.csv",
        run_rows,
        filename="block_evaluation.json",
        default_profile="clean",
    )
    write_generic_eval_long(
        args.output_dir / "heldout_eval_long.csv",
        run_rows,
        filename="heldout_evaluation.json",
        default_profile="clean",
    )
    write_rollout_summary_long(args.output_dir / "rollout_summary_long.csv", run_rows)
    write_rollout_outcomes_long(args.output_dir / "rollout_outcomes_long.csv", run_rows)
    canonical_summary_rows = write_canonical_rollout_subset(
        args.output_dir / "rollout_summary_long.csv",
        args.output_dir / "canonical_rollout_summary_long.csv",
        canonical_keys,
    )
    canonical_outcome_rows = write_canonical_rollout_subset(
        args.output_dir / "rollout_outcomes_long.csv",
        args.output_dir / "canonical_rollout_outcomes_long.csv",
        canonical_keys,
    )
    write_catalog_qc_report(
        args.output_dir / "catalog_qc_report.md",
        run_rows,
        run_annotations,
        rollout_registry,
        canonical_summary_rows,
        canonical_outcome_rows,
    )

    print(f"Generated {len(run_rows)} run rows")
    print(f"Generated {len(file_rows)} file rows")
    print(f"Generated {len(run_annotations)} run annotation rows")
    print(f"Generated {len(rollout_registry)} rollout registry rows")
    print(f"Generated {len(canonical_run_rows)} canonical run rows")
    print("Generated training history long table")
    print("Generated block / heldout / rollout long tables")
    print(f"Generated {canonical_summary_rows} canonical rollout summary rows")
    print(f"Generated {canonical_outcome_rows} canonical rollout outcome rows")
    print(f"Output directory: {repo_relative(args.output_dir)}")


if __name__ == "__main__":
    main()
