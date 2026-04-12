#!/usr/bin/env python3
"""Build a single Markdown report for a sweep directory."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from summarize_sweep import (
    _extract_overall_payload,
    _find_latest_rollout_summary,
    _resolve_local_run_dir,
    _select_profile_payload,
    load_runs,
)


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


def _mean(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if _is_finite(v)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _std(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if _is_finite(v)]
    if not finite:
        return float("nan")
    mu = sum(finite) / len(finite)
    return math.sqrt(sum((v - mu) ** 2 for v in finite) / len(finite))


def _fmt(value, digits=4, nan="NA") -> str:
    if not _is_finite(value):
        return nan
    return f"{float(value):.{digits}f}"


def _fmt_pct(value, digits=1, nan="NA") -> str:
    if not _is_finite(value):
        return nan
    return f"{100.0 * float(value):.{digits}f}%"


def _collect_run_record(
    run: Dict,
    suite_dir: Path,
    horizon_s: float,
    block_profile: str | None = None,
    heldout_profile: str | None = None,
    rollout_profile: str | None = None,
) -> Dict:
    run_dir = _resolve_local_run_dir(suite_dir, run["run_dir"])
    config = _read_json(run_dir / "config.json")
    block_eval = _select_profile_payload(_read_json(run_dir / "block_evaluation.json"), block_profile)
    heldout_eval = _select_profile_payload(_read_json(run_dir / "heldout_evaluation.json"), heldout_profile)
    heldout_overall = _extract_overall_payload(heldout_eval)
    rollout_summary_path = _find_latest_rollout_summary(run_dir, profile=rollout_profile)
    rollout_summary = _read_json(rollout_summary_path) if rollout_summary_path else {}
    horizon_key = str(float(horizon_s))

    record = {
        "group": run["group"],
        "model_type": run["model_type"],
        "seed": int(run["seed"]),
        "run_name": run["run_name"],
        "run_dir": str(run_dir),
        "dataset_path": config.get("dataset_path", ""),
        "best_epoch": float("nan"),
        "best_test_loss": float("nan"),
        "block_position_rmse_mean": _safe_float(_safe_get(block_eval, "position_rmse", "mean")),
        "block_rotation_geodesic_mean": _safe_float(
            _safe_get(block_eval, "rotation_geodesic", "mean")
        ),
        "block_velocity_rmse_mean": _safe_float(_safe_get(block_eval, "velocity_rmse", "mean")),
        "heldout_success_rate": _safe_float(_safe_get(heldout_overall, "success_rate")),
        "heldout_position_rmse_mean": _safe_float(
            _safe_get(heldout_overall, "position_rmse", "mean")
        ),
        "heldout_rotation_geodesic_mean": _safe_float(
            _safe_get(heldout_overall, "rotation_geodesic", "mean")
        ),
        "heldout_n_trajectories": int(_safe_get(heldout_overall, "n_trajectories", default=0)),
        "resampled_completion_rate": _safe_float(
            _safe_get(rollout_summary, "overall", horizon_key, "rates", "completed_to_h")
        ),
        "resampled_model_failed_rate": _safe_float(
            _safe_get(rollout_summary, "overall", horizon_key, "rates", "model_failed_by_h")
        ),
        "resampled_position_median": _safe_float(
            _safe_get(rollout_summary, "overall", horizon_key, "metrics", "final_position_error", "median")
        ),
        "resampled_position_p95": _safe_float(
            _safe_get(rollout_summary, "overall", horizon_key, "metrics", "final_position_error", "p95")
        ),
        "resampled_rotation_median": _safe_float(
            _safe_get(
                rollout_summary,
                "overall",
                horizon_key,
                "metrics",
                "final_rotation_geodesic",
                "median",
            )
        ),
        "resampled_total_velocity_median": _safe_float(
            _safe_get(
                rollout_summary,
                "overall",
                horizon_key,
                "metrics",
                "final_total_linear_velocity_error",
                "median",
            )
        ),
        "heldout_by_scenario": {},
        "resampled_by_scenario": {},
        "resampled_summary_path": str(rollout_summary_path) if rollout_summary_path else "",
    }

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
                record["best_test_loss"] = _safe_float(test_total[best_idx])
                if best_idx < len(epochs):
                    record["best_epoch"] = _safe_float(epochs[best_idx])
        except Exception:
            pass

    for scenario_name, payload in heldout_eval.get("by_scenario", {}).items():
        record["heldout_by_scenario"][scenario_name] = {
            "n": int(_safe_get(payload, "n_trajectories", default=0)),
            "success_rate": _safe_float(_safe_get(payload, "success_rate")),
            "position_rmse_mean": _safe_float(_safe_get(payload, "position_rmse", "mean")),
            "rotation_geodesic_mean": _safe_float(_safe_get(payload, "rotation_geodesic", "mean")),
        }

    for scenario_name, payload in rollout_summary.get("by_scenario", {}).items():
        horizon_payload = payload.get(horizon_key, {})
        record["resampled_by_scenario"][scenario_name] = {
            "n": int(_safe_get(payload, horizon_key, "n_trajectories", default=0)),
            "completion_rate": _safe_float(_safe_get(horizon_payload, "rates", "completed_to_h")),
            "model_failed_rate": _safe_float(_safe_get(horizon_payload, "rates", "model_failed_by_h")),
            "position_median": _safe_float(
                _safe_get(horizon_payload, "metrics", "final_position_error", "median")
            ),
            "position_p95": _safe_float(
                _safe_get(horizon_payload, "metrics", "final_position_error", "p95")
            ),
            "rotation_median": _safe_float(
                _safe_get(horizon_payload, "metrics", "final_rotation_geodesic", "median")
            ),
        }

    return record


def _aggregate_model_records(run_records: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for record in run_records:
        grouped[(record["group"], record["model_type"])].append(record)

    model_records = []
    for (group, model_type), items in sorted(grouped.items()):
        model_record = {
            "group": group,
            "model_type": model_type,
            "n_seeds": len(items),
            "seeds": ",".join(str(item["seed"]) for item in sorted(items, key=lambda row: row["seed"])),
            "best_test_loss_mean": _mean(item["best_test_loss"] for item in items),
            "best_test_loss_std": _std(item["best_test_loss"] for item in items),
            "best_epoch_mean": _mean(item["best_epoch"] for item in items),
            "block_position_rmse_mean": _mean(item["block_position_rmse_mean"] for item in items),
            "block_rotation_geodesic_mean": _mean(
                item["block_rotation_geodesic_mean"] for item in items
            ),
            "block_velocity_rmse_mean": _mean(item["block_velocity_rmse_mean"] for item in items),
            "heldout_success_rate_mean": _mean(item["heldout_success_rate"] for item in items),
            "heldout_position_rmse_mean": _mean(item["heldout_position_rmse_mean"] for item in items),
            "heldout_rotation_geodesic_mean": _mean(
                item["heldout_rotation_geodesic_mean"] for item in items
            ),
            "resampled_completion_rate_mean": _mean(
                item["resampled_completion_rate"] for item in items
            ),
            "resampled_model_failed_rate_mean": _mean(
                item["resampled_model_failed_rate"] for item in items
            ),
            "resampled_position_median_mean": _mean(
                item["resampled_position_median"] for item in items
            ),
            "resampled_position_p95_mean": _mean(item["resampled_position_p95"] for item in items),
            "resampled_rotation_median_mean": _mean(
                item["resampled_rotation_median"] for item in items
            ),
            "resampled_total_velocity_median_mean": _mean(
                item["resampled_total_velocity_median"] for item in items
            ),
            "heldout_by_scenario": {},
            "resampled_by_scenario": {},
        }

        scenario_names = sorted(
            {
                *{
                    name
                    for item in items
                    for name in item["heldout_by_scenario"].keys()
                },
                *{
                    name
                    for item in items
                    for name in item["resampled_by_scenario"].keys()
                },
            }
        )
        for scenario_name in scenario_names:
            heldout_items = [
                item["heldout_by_scenario"].get(scenario_name, {}) for item in items
            ]
            resampled_items = [
                item["resampled_by_scenario"].get(scenario_name, {}) for item in items
            ]
            model_record["heldout_by_scenario"][scenario_name] = {
                "success_rate_mean": _mean(item.get("success_rate") for item in heldout_items),
                "position_rmse_mean": _mean(item.get("position_rmse_mean") for item in heldout_items),
                "rotation_geodesic_mean": _mean(
                    item.get("rotation_geodesic_mean") for item in heldout_items
                ),
            }
            model_record["resampled_by_scenario"][scenario_name] = {
                "completion_rate_mean": _mean(
                    item.get("completion_rate") for item in resampled_items
                ),
                "model_failed_rate_mean": _mean(
                    item.get("model_failed_rate") for item in resampled_items
                ),
                "position_median_mean": _mean(
                    item.get("position_median") for item in resampled_items
                ),
                "position_p95_mean": _mean(item.get("position_p95") for item in resampled_items),
                "rotation_median_mean": _mean(
                    item.get("rotation_median") for item in resampled_items
                ),
            }

        model_records.append(model_record)

    model_records.sort(
        key=lambda row: (
            math.inf
            if not _is_finite(row["resampled_position_median_mean"])
            else row["resampled_position_median_mean"]
        )
    )
    return model_records


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report_text(
    suite_dir: Path,
    run_records: List[Dict],
    model_records: List[Dict],
    horizon_s: float,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_paths = sorted({record["dataset_path"] for record in run_records if record["dataset_path"]})
    finite_resampled_runs = [
        record
        for record in run_records
        if _is_finite(record["resampled_position_median"])
    ]
    best_run = (
        min(finite_resampled_runs, key=lambda record: record["resampled_position_median"])
        if finite_resampled_runs else None
    )
    best_model = model_records[0] if model_records else None
    main_runs = [record for record in run_records if record["model_type"] == "phnode_full"]
    main_range = float("nan")
    if main_runs:
        pos_values = [
            record["resampled_position_median"]
            for record in main_runs
            if _is_finite(record["resampled_position_median"])
        ]
        if pos_values:
            main_range = max(pos_values) - min(pos_values)
    family_ranges = []
    family_records: Dict[tuple, List[Dict]] = defaultdict(list)
    for record in run_records:
        family_records[(record["group"], record["model_type"])].append(record)
    for (group, model_type), items in family_records.items():
        pos_values = [
            item["resampled_position_median"]
            for item in items
            if _is_finite(item["resampled_position_median"])
        ]
        if len(pos_values) < 2:
            continue
        family_ranges.append(
            {
                "group": group,
                "model_type": model_type,
                "range": max(pos_values) - min(pos_values),
            }
        )
    family_ranges.sort(key=lambda row: row["range"], reverse=True)
    unstable_runs = [
        record
        for record in sorted(run_records, key=lambda row: row["resampled_completion_rate"])
        if record["resampled_completion_rate"] < 0.95
    ]

    model_table_rows = []
    for record in model_records:
        model_table_rows.append(
            [
                f"{record['group']}/{record['model_type']}",
                record["seeds"],
                _fmt(record["best_test_loss_mean"], 4),
                _fmt(record["block_position_rmse_mean"], 5),
                _fmt(record["heldout_position_rmse_mean"], 5),
                _fmt_pct(record["resampled_completion_rate_mean"]),
                _fmt_pct(record["resampled_model_failed_rate_mean"]),
                _fmt(record["resampled_position_median_mean"], 4),
                _fmt(record["resampled_position_p95_mean"], 4),
                _fmt(record["resampled_rotation_median_mean"], 4),
            ]
        )

    seed_table_rows = []
    for record in sorted(
        run_records,
        key=lambda row: (
            math.inf if not _is_finite(row["resampled_position_median"]) else row["resampled_position_median"],
            -row["resampled_completion_rate"],
        ),
    ):
        seed_table_rows.append(
            [
                record["run_name"],
                f"{record['group']}/{record['model_type']}",
                str(record["seed"]),
                _fmt(record["best_test_loss"], 4),
                _fmt(record["block_position_rmse_mean"], 5),
                _fmt(record["heldout_position_rmse_mean"], 5),
                _fmt_pct(record["resampled_completion_rate"]),
                _fmt_pct(record["resampled_model_failed_rate"]),
                _fmt(record["resampled_position_median"], 4),
                _fmt(record["resampled_position_p95"], 4),
                _fmt(record["resampled_rotation_median"], 4),
            ]
        )

    heldout_scenario_rows = []
    resampled_scenario_rows = []
    for record in model_records:
        for scenario_name in sorted(record["heldout_by_scenario"].keys()):
            heldout = record["heldout_by_scenario"][scenario_name]
            resampled = record["resampled_by_scenario"][scenario_name]
            heldout_scenario_rows.append(
                [
                    f"{record['group']}/{record['model_type']}",
                    scenario_name,
                    _fmt_pct(heldout["success_rate_mean"]),
                    _fmt(heldout["position_rmse_mean"], 5),
                    _fmt(heldout["rotation_geodesic_mean"], 5),
                ]
            )
            resampled_scenario_rows.append(
                [
                    f"{record['group']}/{record['model_type']}",
                    scenario_name,
                    _fmt_pct(resampled["completion_rate_mean"]),
                    _fmt_pct(resampled["model_failed_rate_mean"]),
                    _fmt(resampled["position_median_mean"], 4),
                    _fmt(resampled["position_p95_mean"], 4),
                    _fmt(resampled["rotation_median_mean"], 4),
                ]
            )

    lines = [
        "# Experiment Report",
        "",
        f"- Suite: `{suite_dir}`",
        f"- Generated: `{generated_at}`",
        f"- Runs: `{len(run_records)}`",
        f"- Resampled horizon: `{horizon_s:.1f}s`",
        f"- Dataset: `{dataset_paths[0] if dataset_paths else 'NA'}`",
        "",
        "## Key Findings",
        "",
    ]
    if best_run is not None:
        lines.append(
            f"- Best single run: `{best_run['run_name']}` with resampled@{horizon_s:.0f}s "
            f"`pos median={_fmt(best_run['resampled_position_median'], 4)} m`, "
            f"`p95={_fmt(best_run['resampled_position_p95'], 4)} m`, "
            f"`completion={_fmt_pct(best_run['resampled_completion_rate'])}`."
        )
    else:
        lines.append(
            f"- Best single run: `NA`. No run has a finite resampled position median at `{horizon_s:.1f}s`."
        )
    if best_model is not None:
        lines.append(
            f"- Best model family by resampled mean is `{best_model['group']}/{best_model['model_type']}` "
            f"with `pos median={_fmt(best_model['resampled_position_median_mean'], 4)} m` and "
            f"`completion={_fmt_pct(best_model['resampled_completion_rate_mean'])}`."
        )
    else:
        lines.append("- Best model family by resampled mean: `NA`.")
    if main_runs and _is_finite(main_range):
        lines.append(
            f"- Main model `main/phnode_full` is competitive at its best seeds, but has noticeable seed sensitivity: "
            f"resampled position median range across seeds is `{_fmt(main_range, 4)} m`."
        )
    elif family_ranges:
        most_sensitive = family_ranges[0]
        lines.append(
            f"- Largest seed sensitivity in this sweep is `{most_sensitive['group']}/{most_sensitive['model_type']}` "
            f"with a resampled position median range of `{_fmt(most_sensitive['range'], 4)} m`."
        )
    if unstable_runs:
        lines.append(
            f"- The clearly unstable family is `{unstable_runs[0]['group']}/{unstable_runs[0]['model_type']}`; "
            f"worst completion is `{_fmt_pct(unstable_runs[0]['resampled_completion_rate'])}`."
        )

    lines.extend(
        [
            "",
            "## Model-Level Summary",
            "",
            _markdown_table(
                [
                    "Model",
                    "Seeds",
                    "Best Test Loss",
                    "Block Pos RMSE",
                    "Heldout Pos RMSE",
                    f"Resampled Completion @{horizon_s:.0f}s",
                    "Resampled Model Fail",
                    "Resampled Pos Median",
                    "Resampled Pos P95",
                    "Resampled Rot Median",
                ],
                model_table_rows,
            ),
            "",
            "## Seed-Level Summary",
            "",
            _markdown_table(
                [
                    "Run",
                    "Model",
                    "Seed",
                    "Best Test Loss",
                    "Block Pos RMSE",
                    "Heldout Pos RMSE",
                    f"Resampled Completion @{horizon_s:.0f}s",
                    "Resampled Model Fail",
                    "Resampled Pos Median",
                    "Resampled Pos P95",
                    "Resampled Rot Median",
                ],
                seed_table_rows,
            ),
            "",
            "## Heldout By Scenario",
            "",
            _markdown_table(
                [
                    "Model",
                    "Scenario",
                    "Success Rate",
                    "Pos RMSE Mean",
                    "Rot Geodesic Mean",
                ],
                heldout_scenario_rows,
            ),
            "",
            f"## Resampled By Scenario @{horizon_s:.0f}s",
            "",
            _markdown_table(
                [
                    "Model",
                    "Scenario",
                    "Completion",
                    "Model Fail",
                    "Pos Median",
                    "Pos P95",
                    "Rot Median",
                ],
                resampled_scenario_rows,
            ),
            "",
            "## Notes",
            "",
            "- `Block` metrics are short-horizon block prediction errors from `block_evaluation.json`.",
            "- `Heldout` metrics come from `heldout_evaluation.json` and use replayed held-out trajectories.",
            f"- `Resampled` metrics come from the selected `rollout_benchmark/*/summary.json` at `{horizon_s:.1f}s`.",
            "- For runs that never complete the resampled horizon, some terminal error statistics are `NA` because there is no completed trajectory to condition on.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True, type=str, help="Sweep directory under checkpoints/")
    parser.add_argument(
        "--block-profile",
        type=str,
        default=None,
        help="Noise profile to read from block_evaluation.json when multiple profiles exist.",
    )
    parser.add_argument(
        "--heldout-profile",
        type=str,
        default=None,
        help="Noise profile to read from heldout_evaluation.json when multiple profiles exist.",
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
        help="Resampled rollout horizon in seconds. Default: 60",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_report.md",
        help="Output Markdown file name relative to suite dir",
    )
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    runs = load_runs(suite_dir)
    run_records = [
        _collect_run_record(
            run,
            suite_dir=suite_dir,
            horizon_s=float(args.horizon),
            block_profile=args.block_profile,
            heldout_profile=args.heldout_profile,
            rollout_profile=args.rollout_profile,
        )
        for run in runs
    ]
    model_records = _aggregate_model_records(run_records)

    report_text = build_report_text(
        suite_dir=suite_dir,
        run_records=run_records,
        model_records=model_records,
        horizon_s=float(args.horizon),
    )
    output_path = suite_dir / args.output
    output_path.write_text(report_text)
    print(output_path)


if __name__ == "__main__":
    main()
