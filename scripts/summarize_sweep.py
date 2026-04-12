#!/usr/bin/env python3
"""Summarize a training/evaluation sweep into seed-level and model-level tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List


def _read_json(path: Path) -> Dict:
    with open(path) as handle:
        return json.load(handle)


def _safe_get(payload: Dict, *keys, default=float("nan")):
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _stats(values: Iterable[float]) -> Dict[str, float]:
    finite = [float(v) for v in values if _is_finite(v)]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = sum(finite) / len(finite)
    var = sum((v - mean) ** 2 for v in finite) / len(finite)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(finite),
        "max": max(finite),
    }


def _write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_profile_payload(payload: Dict, profile: str | None) -> Dict:
    if not isinstance(payload, dict):
        return {}
    if "overall" in payload or "position_rmse" in payload:
        return payload
    if profile and isinstance(payload.get(profile), dict):
        return payload[profile]
    for candidate in (
        "nominal_eval",
        "clean",
        "degraded_eval",
        "heading_biased_eval",
        "current_bias_eval",
    ):
        if isinstance(payload.get(candidate), dict):
            return payload[candidate]
    for value in payload.values():
        if isinstance(value, dict):
            return value
    return {}


def _find_latest_rollout_summary(run_dir: Path, profile: str | None = None) -> Path | None:
    rollout_root = run_dir / "rollout_benchmark"
    if not rollout_root.exists():
        return None
    if profile:
        patterns = [
            f"*/{profile}/summary.json",
            f"*/*/{profile}/summary.json",
        ]
    else:
        patterns = [
            "*/summary.json",
            "*/*/summary.json",
        ]
    summaries = sorted(
        [
            path
            for pattern in patterns
            for path in rollout_root.glob(pattern)
        ],
        key=lambda path: path.stat().st_mtime,
    )
    return summaries[-1] if summaries else None


def _resolve_local_run_dir(suite_dir: Path, run_dir: str) -> Path:
    candidate = Path(run_dir).expanduser()
    if candidate.exists():
        return candidate.resolve()

    fallback = suite_dir / candidate.name
    if fallback.exists():
        return fallback.resolve()

    return candidate


def _resolve_local_checkpoint_path(run_dir: Path, checkpoint_path: str) -> Path:
    candidate = Path(checkpoint_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    fallback = run_dir / candidate.name
    if fallback.exists():
        return fallback.resolve()

    return candidate


def _load_runs_from_tsv(path: Path) -> List[Dict]:
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "group": row["group"],
                    "model_type": row["model_type"],
                    "seed": int(row["seed"]),
                    "run_name": row["run_name"],
                    "run_dir": row["run_dir"],
                    "checkpoint": row["checkpoint"],
                }
            )
    return rows


def _load_runs_from_legacy_manifest(path: Path) -> List[Dict]:
    payload = _read_json(path)
    rows = []
    for entry in payload.get("entries", []):
        rows.append(
            {
                "group": entry["group"],
                "model_type": entry["model_type"],
                "seed": int(entry["seed"]),
                "run_name": entry["run_name"],
                "run_dir": entry["run_dir"],
                "checkpoint": entry["checkpoint"],
            }
        )
    return rows


def load_runs(suite_dir: Path) -> List[Dict]:
    tsv_path = suite_dir / "runs.tsv"
    if tsv_path.exists():
        return _load_runs_from_tsv(tsv_path)

    legacy_manifests = sorted(suite_dir.glob("*train_manifest.json"))
    if len(legacy_manifests) == 1:
        return _load_runs_from_legacy_manifest(legacy_manifests[0])

    raise FileNotFoundError(
        f"No runs.tsv found in {suite_dir}, and no unique legacy train_manifest.json was available."
    )


def build_seed_row(
    run: Dict,
    suite_dir: Path,
    horizon_s: float,
    block_profile: str | None = None,
    heldout_profile: str | None = None,
    rollout_profile: str | None = None,
) -> Dict:
    run_dir = _resolve_local_run_dir(suite_dir, run["run_dir"])
    checkpoint_path = _resolve_local_checkpoint_path(run_dir, run["checkpoint"])
    config_path = run_dir / "config.json"
    block_eval_path = run_dir / "block_evaluation.json"
    heldout_eval_path = run_dir / "heldout_evaluation.json"
    rollout_summary_path = _find_latest_rollout_summary(run_dir, profile=rollout_profile)

    row = {
        "group": run["group"],
        "model_type": run["model_type"],
        "seed": run["seed"],
        "run_name": run["run_name"],
        "run_dir": str(run_dir),
        "checkpoint_exists": int(checkpoint_path.exists()),
        "config_exists": int(config_path.exists()),
        "block_eval_exists": int(block_eval_path.exists()),
        "heldout_eval_exists": int(heldout_eval_path.exists()),
        "rollout_summary_exists": int(rollout_summary_path is not None),
        "dataset_path": "",
        "best_epoch": float("nan"),
        "best_test_loss": float("nan"),
        "block_position_rmse_mean": float("nan"),
        "block_rotation_geodesic_mean": float("nan"),
        "block_velocity_rmse_mean": float("nan"),
        "block_angular_rmse_mean": float("nan"),
        "heldout_success_rate": float("nan"),
        "heldout_position_rmse_mean": float("nan"),
        "heldout_rotation_geodesic_mean": float("nan"),
        "rollout_horizon_s": horizon_s,
        "rollout_completed_rate": float("nan"),
        "rollout_model_failed_rate": float("nan"),
        "rollout_gt_failed_rate": float("nan"),
        "rollout_final_position_error_median": float("nan"),
        "rollout_final_position_error_p95": float("nan"),
        "rollout_final_rotation_geodesic_median": float("nan"),
        "rollout_final_total_linear_velocity_error_median": float("nan"),
    }

    if config_path.exists():
        config = _read_json(config_path)
        row["dataset_path"] = config.get("dataset_path", "")

    history_path = run_dir / "training_history.pkl"
    if history_path.exists():
        try:
            import pickle

            with open(history_path, "rb") as handle:
                history = pickle.load(handle)
            test_total = history.get("test_total", [])
            epochs = history.get("epoch", [])
            if test_total:
                best_idx = min(range(len(test_total)), key=lambda idx: test_total[idx])
                row["best_test_loss"] = _safe_float(test_total[best_idx])
                if best_idx < len(epochs):
                    row["best_epoch"] = _safe_float(epochs[best_idx])
        except Exception:
            pass

    if block_eval_path.exists():
        block_eval = _select_profile_payload(_read_json(block_eval_path), block_profile)
        row["block_position_rmse_mean"] = _safe_float(_safe_get(block_eval, "position_rmse", "mean"))
        row["block_rotation_geodesic_mean"] = _safe_float(
            _safe_get(block_eval, "rotation_geodesic", "mean")
        )
        row["block_velocity_rmse_mean"] = _safe_float(_safe_get(block_eval, "velocity_rmse", "mean"))
        row["block_angular_rmse_mean"] = _safe_float(_safe_get(block_eval, "angular_rmse", "mean"))

    if heldout_eval_path.exists():
        heldout_eval = _select_profile_payload(_read_json(heldout_eval_path), heldout_profile)
        row["heldout_success_rate"] = _safe_float(_safe_get(heldout_eval, "overall", "success_rate"))
        row["heldout_position_rmse_mean"] = _safe_float(
            _safe_get(heldout_eval, "overall", "position_rmse", "mean")
        )
        row["heldout_rotation_geodesic_mean"] = _safe_float(
            _safe_get(heldout_eval, "overall", "rotation_geodesic", "mean")
        )

    if rollout_summary_path is not None:
        rollout_summary = _read_json(rollout_summary_path)
        horizon_key = str(horizon_s)
        overall_h = _safe_get(rollout_summary, "overall", horizon_key, default={})
        row["rollout_completed_rate"] = _safe_float(
            _safe_get(overall_h, "rates", "completed_to_h")
        )
        row["rollout_model_failed_rate"] = _safe_float(
            _safe_get(overall_h, "rates", "model_failed_by_h")
        )
        row["rollout_gt_failed_rate"] = _safe_float(
            _safe_get(overall_h, "rates", "gt_failed_by_h")
        )
        row["rollout_final_position_error_median"] = _safe_float(
            _safe_get(overall_h, "metrics", "final_position_error", "median")
        )
        row["rollout_final_position_error_p95"] = _safe_float(
            _safe_get(overall_h, "metrics", "final_position_error", "p95")
        )
        row["rollout_final_rotation_geodesic_median"] = _safe_float(
            _safe_get(overall_h, "metrics", "final_rotation_geodesic", "median")
        )
        row["rollout_final_total_linear_velocity_error_median"] = _safe_float(
            _safe_get(overall_h, "metrics", "final_total_linear_velocity_error", "median")
        )

    return row


def aggregate_model_rows(seed_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = {}
    for row in seed_rows:
        grouped.setdefault((row["group"], row["model_type"]), []).append(row)

    metric_keys = [
        "best_test_loss",
        "block_position_rmse_mean",
        "block_rotation_geodesic_mean",
        "block_velocity_rmse_mean",
        "block_angular_rmse_mean",
        "heldout_success_rate",
        "heldout_position_rmse_mean",
        "heldout_rotation_geodesic_mean",
        "rollout_completed_rate",
        "rollout_model_failed_rate",
        "rollout_gt_failed_rate",
        "rollout_final_position_error_median",
        "rollout_final_position_error_p95",
        "rollout_final_rotation_geodesic_median",
        "rollout_final_total_linear_velocity_error_median",
    ]

    rows = []
    for (group, model_type), items in sorted(grouped.items()):
        row = {
            "group": group,
            "model_type": model_type,
            "n_seeds": len(items),
            "seeds": ",".join(str(item["seed"]) for item in sorted(items, key=lambda r: r["seed"])),
        }
        for key in metric_keys:
            stats = _stats(item[key] for item in items)
            row[f"{key}_mean"] = stats["mean"]
            row[f"{key}_std"] = stats["std"]
            row[f"{key}_min"] = stats["min"]
            row[f"{key}_max"] = stats["max"]
        rows.append(row)
    return rows


def write_text_summary(path: Path, suite_dir: Path, seed_rows: List[Dict], model_rows: List[Dict], horizon_s: float):
    lines = []
    lines.append("Sweep Summary")
    lines.append("=" * 80)
    lines.append(f"Suite: {suite_dir}")
    lines.append(f"Runs: {len(seed_rows)}")
    lines.append(f"Aggregation horizon: {horizon_s:.1f}s")
    lines.append("")

    lines.append("By Model")
    lines.append("-" * 80)
    for row in model_rows:
        lines.append(
            f"{row['group']}/{row['model_type']}"
            f" | seeds={row['seeds']}"
            f" | heldout pos={row['heldout_position_rmse_mean_mean']:.4f}"
            f" +- {row['heldout_position_rmse_mean_std']:.4f}"
            f" | rollout pos@H median={row['rollout_final_position_error_median_mean']:.4f}"
            f" +- {row['rollout_final_position_error_median_std']:.4f}"
            f" | completion@H={row['rollout_completed_rate_mean']:.3f}"
            f" +- {row['rollout_completed_rate_std']:.3f}"
        )
    lines.append("")

    lines.append("Seed Rows")
    lines.append("-" * 80)
    for row in sorted(seed_rows, key=lambda item: (item["group"], item["model_type"], item["seed"])):
        lines.append(
            f"{row['group']}/{row['model_type']} seed={row['seed']}"
            f" | heldout pos={row['heldout_position_rmse_mean']:.4f}"
            f" | rollout pos@H median={row['rollout_final_position_error_median']:.4f}"
            f" | completion@H={row['rollout_completed_rate']:.3f}"
        )

    with open(path, "w") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=str, required=True, help="Sweep directory under checkpoints/")
    parser.add_argument(
        "--block-profile",
        type=str,
        default=None,
        help="Noise profile to read from block_evaluation.json when it stores multiple profiles.",
    )
    parser.add_argument(
        "--heldout-profile",
        type=str,
        default=None,
        help="Noise profile to read from heldout_evaluation.json when it stores multiple profiles.",
    )
    parser.add_argument(
        "--rollout-profile",
        type=str,
        default=None,
        help="Noise profile directory to read under rollout_benchmark when multiple profiles exist.",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=60.0,
        help="Rollout horizon in seconds to aggregate from summary.json. Default: 60",
    )
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    runs = load_runs(suite_dir)
    seed_rows = [
        build_seed_row(
            run,
            suite_dir=suite_dir,
            horizon_s=float(args.horizon),
            block_profile=args.block_profile,
            heldout_profile=args.heldout_profile,
            rollout_profile=args.rollout_profile,
        )
        for run in runs
    ]
    model_rows = aggregate_model_rows(seed_rows)

    _write_csv(suite_dir / "sweep_seed_metrics.csv", seed_rows)
    _write_csv(suite_dir / "sweep_model_metrics.csv", model_rows)

    summary_json = {
        "suite_dir": str(suite_dir),
        "horizon_s": float(args.horizon),
        "n_runs": len(seed_rows),
        "seed_rows": seed_rows,
        "model_rows": model_rows,
    }
    with open(suite_dir / "sweep_summary.json", "w") as handle:
        json.dump(summary_json, handle, indent=2)
    write_text_summary(
        suite_dir / "sweep_summary.txt",
        suite_dir=suite_dir,
        seed_rows=seed_rows,
        model_rows=model_rows,
        horizon_s=float(args.horizon),
    )


if __name__ == "__main__":
    main()
