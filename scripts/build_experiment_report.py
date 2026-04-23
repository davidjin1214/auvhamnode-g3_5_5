#!/usr/bin/env python3
"""Build a Markdown report for a sweep directory using the Phase-1 summary contract."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from summarize_sweep import (
    DEFAULT_HORIZONS,
    DEFAULT_PROFILE_PREFERENCE,
    build_phase1_bundle,
    load_runs,
)


def _is_finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if _is_finite(v)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _fmt(value, digits=4, nan="NA") -> str:
    if not _is_finite(value):
        return nan
    return f"{float(value):.{digits}f}"


def _fmt_pct(value, digits=1, nan="NA") -> str:
    if not _is_finite(value):
        return nan
    return f"{100.0 * float(value):.{digits}f}%"


def _fmt_signed_pct(value, digits=1, nan="NA") -> str:
    if not _is_finite(value):
        return nan
    return f"{float(value):+.{digits}f}%"


def _same_horizon(value, target) -> bool:
    if not _is_finite(target):
        return not _is_finite(value)
    return _is_finite(value) and abs(float(value) - float(target)) < 1e-9


def _horizon_group_key(value) -> float | None:
    return float(value) if _is_finite(value) else None


def _train_display(row: Dict) -> str:
    if row["train_protocol_label"] == "clean":
        return "clean"
    return f"{row['train_protocol_label']}/{row['train_noise_profile']}"


def _sort_profiles(profiles: Iterable[str]) -> List[str]:
    unique = sorted({str(profile) for profile in profiles})
    order = {name: idx for idx, name in enumerate(DEFAULT_PROFILE_PREFERENCE)}
    return sorted(unique, key=lambda name: (order.get(name, len(order)), name))


def _pick_primary_profile(rows: List[Dict], *, source: str, requested: str | None) -> str | None:
    candidates = _sort_profiles(row["eval_profile"] for row in rows if row["source"] == source)
    if requested and requested in candidates:
        return requested
    return candidates[0] if candidates else None


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _aggregate_scenario_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["group"],
                row["model_type"],
                row["train_noise_profile"],
                row["train_noise_protocol"],
                row["train_protocol_label"],
                row["scenario"],
            )
        ].append(row)

    aggregates = []
    for key, items in sorted(grouped.items()):
        (
            group,
            model_type,
            train_noise_profile,
            train_noise_protocol,
            train_protocol_label,
            scenario,
        ) = key
        aggregates.append(
            {
                "group": group,
                "model_type": model_type,
                "train_noise_profile": train_noise_profile,
                "train_noise_protocol": train_noise_protocol,
                "train_protocol_label": train_protocol_label,
                "scenario": scenario,
                "rollout_completion_rate": _mean(item["rollout_completion_rate"] for item in items),
                "rollout_model_failed_rate": _mean(item["rollout_model_failed_rate"] for item in items),
                "rollout_final_position_error_median": _mean(
                    item["rollout_final_position_error_median"] for item in items
                ),
                "rollout_final_position_error_p95": _mean(
                    item["rollout_final_position_error_p95"] for item in items
                ),
                "rollout_final_rotation_geodesic_median": _mean(
                    item["rollout_final_rotation_geodesic_median"] for item in items
                ),
            }
        )
    return aggregates


def _aggregate_degradation_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["comparison_kind"],
                row["group"],
                row["model_type"],
                row["train_noise_profile"],
                row["train_noise_protocol"],
                row["train_protocol_label"],
                row["source"],
                row["eval_profile"],
                row["metric_name"],
                _horizon_group_key(row["horizon_s"]),
            )
        ].append(row)

    aggregates = []
    for key, items in sorted(grouped.items()):
        (
            comparison_kind,
            group,
            model_type,
            train_noise_profile,
            train_noise_protocol,
            train_protocol_label,
            source,
            eval_profile,
            metric_name,
            horizon_key,
        ) = key
        aggregates.append(
            {
                "comparison_kind": comparison_kind,
                "group": group,
                "model_type": model_type,
                "train_noise_profile": train_noise_profile,
                "train_noise_protocol": train_noise_protocol,
                "train_protocol_label": train_protocol_label,
                "source": source,
                "eval_profile": eval_profile,
                "metric_name": metric_name,
                "horizon_s": float("nan") if horizon_key is None else horizon_key,
                "ratio_to_clean_mean": _mean(item["ratio_to_clean"] for item in items),
                "degradation_pct_mean": _mean(item["degradation_pct"] for item in items),
                "absolute_delta_mean": _mean(item["absolute_delta"] for item in items),
            }
        )
    return aggregates


def _metric_lookup(rows: List[Dict]) -> Dict[tuple, Dict]:
    return {
        (
            row["group"],
            row["model_type"],
            row["train_protocol_label"],
            row["metric_name"],
            row["source"],
            row["eval_profile"],
            _horizon_group_key(row["horizon_s"]),
        ): row
        for row in rows
    }


def build_report_text(
    *,
    suite_dir: Path,
    bundle: Dict,
    primary_horizon: float,
    horizons: List[float],
    primary_rollout_profile: str | None,
    primary_heldout_profile: str | None,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    artifacts = bundle["artifacts"]
    by_seed_rows = bundle["by_seed_rows"]
    model_rows = bundle["model_rows"]
    by_scenario_rows = bundle["by_scenario_rows"]
    degradation_rows = bundle["degradation_rows"]

    dataset_paths = sorted({artifact["dataset_path"] for artifact in artifacts if artifact["dataset_path"]})
    rollout_model_rows = [
        row
        for row in model_rows
        if row["source"] == "rollout"
        and row["eval_profile"] == primary_rollout_profile
        and _same_horizon(row["horizon_s"], primary_horizon)
    ]
    rollout_model_rows.sort(
        key=lambda row: (
            math.inf
            if not _is_finite(row["rollout_final_position_error_median_mean"])
            else row["rollout_final_position_error_median_mean"]
        )
    )
    heldout_model_rows = [
        row
        for row in model_rows
        if row["source"] == "heldout" and row["eval_profile"] == primary_heldout_profile
    ]
    heldout_model_rows.sort(
        key=lambda row: (
            math.inf
            if not _is_finite(row["heldout_position_rmse_mean_mean"])
            else row["heldout_position_rmse_mean_mean"]
        )
    )

    rollout_seed_rows = [
        row
        for row in by_seed_rows
        if row["source"] == "rollout"
        and row["eval_profile"] == primary_rollout_profile
        and _same_horizon(row["horizon_s"], primary_horizon)
    ]
    rollout_seed_rows.sort(
        key=lambda row: (
            math.inf
            if not _is_finite(row["rollout_final_position_error_median"])
            else row["rollout_final_position_error_median"],
            -row["rollout_completion_rate"],
        )
    )

    best_model = rollout_model_rows[0] if rollout_model_rows else None
    main_rows = [
        row
        for row in rollout_seed_rows
        if row["model_type"] == "phnode_full"
    ]
    main_range = float("nan")
    if main_rows:
        values = [
            row["rollout_final_position_error_median"]
            for row in main_rows
            if _is_finite(row["rollout_final_position_error_median"])
        ]
        if values:
            main_range = max(values) - min(values)

    horizon_table_rows = []
    for row in rollout_model_rows:
        horizon_lookup = {
            horizon: next(
                (
                    candidate
                    for candidate in model_rows
                    if candidate["group"] == row["group"]
                    and candidate["model_type"] == row["model_type"]
                    and candidate["train_protocol_label"] == row["train_protocol_label"]
                    and candidate["source"] == "rollout"
                    and candidate["eval_profile"] == primary_rollout_profile
                    and _same_horizon(candidate["horizon_s"], horizon)
                ),
                None,
            )
            for horizon in horizons
        }
        horizon_row = [
            f"{row['group']}/{row['model_type']}",
            _train_display(row),
        ]
        for horizon in horizons:
            candidate = horizon_lookup[horizon]
            horizon_row.append(
                _fmt(
                    candidate["rollout_final_position_error_median_mean"]
                    if candidate is not None else float("nan"),
                    4,
                )
            )
        horizon_row.append(_fmt_pct(row["rollout_completion_rate_mean"]))
        horizon_table_rows.append(horizon_row)

    scenario_summary_rows = _aggregate_scenario_rows(
        [
            row
            for row in by_scenario_rows
            if row["source"] == "rollout"
            and row["eval_profile"] == primary_rollout_profile
            and _same_horizon(row["horizon_s"], primary_horizon)
        ]
    )
    scenario_table_rows = [
        [
            f"{row['group']}/{row['model_type']}",
            _train_display(row),
            row["scenario"],
            _fmt_pct(row["rollout_completion_rate"]),
            _fmt_pct(row["rollout_model_failed_rate"]),
            _fmt(row["rollout_final_position_error_median"], 4),
            _fmt(row["rollout_final_position_error_p95"], 4),
            _fmt(row["rollout_final_rotation_geodesic_median"], 4),
        ]
        for row in scenario_summary_rows
    ]

    degradation_summary = _aggregate_degradation_rows(degradation_rows)
    degradation_lookup = _metric_lookup(degradation_summary)

    rollout_degradation_rows = []
    for row in rollout_model_rows:
        pos_key = (
            row["group"],
            row["model_type"],
            row["train_protocol_label"],
            "rollout_final_position_error_median",
            "rollout",
            primary_rollout_profile,
            primary_horizon,
        )
        completion_key = (
            row["group"],
            row["model_type"],
            row["train_protocol_label"],
            "rollout_completion_rate",
            "rollout",
            primary_rollout_profile,
            primary_horizon,
        )
        pos_entry = degradation_lookup.get(pos_key)
        completion_entry = degradation_lookup.get(completion_key)
        if not pos_entry and not completion_entry:
            continue
        rollout_degradation_rows.append(
            [
                f"{row['group']}/{row['model_type']}",
                _train_display(row),
                _fmt(pos_entry["ratio_to_clean_mean"] if pos_entry else float("nan"), 3),
                _fmt_pct(
                    (completion_entry["degradation_pct_mean"] / 100.0)
                    if completion_entry is not None else float("nan"),
                    1,
                ),
                _fmt_signed_pct(
                    pos_entry["degradation_pct_mean"] if pos_entry else float("nan"),
                    1,
                    nan="NA",
                ),
            ]
        )

    clean_replay_rows = []
    clean_replay_candidates = [
        row
        for row in degradation_summary
        if row["comparison_kind"] == "clean_replay_cost"
    ]
    clean_replay_lookup = _metric_lookup(clean_replay_candidates)
    nonclean_rollout_rows = [row for row in rollout_model_rows if row["train_protocol_label"] != "clean"]
    for row in nonclean_rollout_rows:
        heldout_key = (
            row["group"],
            row["model_type"],
            row["train_protocol_label"],
            "heldout_position_rmse_mean",
            "heldout",
            "clean",
            None,
        )
        rollout_key = (
            row["group"],
            row["model_type"],
            row["train_protocol_label"],
            "rollout_final_position_error_median",
            "rollout",
            "clean",
            _horizon_group_key(primary_horizon),
        )
        heldout_entry = clean_replay_lookup.get(heldout_key)
        rollout_entry = clean_replay_lookup.get(rollout_key)
        if not heldout_entry and not rollout_entry:
            continue
        clean_replay_rows.append(
            [
                f"{row['group']}/{row['model_type']}",
                _train_display(row),
                _fmt(heldout_entry["ratio_to_clean_mean"] if heldout_entry else float("nan"), 3),
                _fmt(rollout_entry["ratio_to_clean_mean"] if rollout_entry else float("nan"), 3),
                _fmt_signed_pct(
                    rollout_entry["degradation_pct_mean"] if rollout_entry else float("nan"),
                    1,
                    nan="NA",
                ),
            ]
        )

    lines = [
        "# Experiment Report",
        "",
        f"- Suite: `{suite_dir}`",
        f"- Generated: `{generated_at}`",
        f"- Runs: `{len(artifacts)}`",
        f"- Horizons: `{', '.join(f'{h:.0f}s' for h in horizons)}`",
        f"- Primary rollout profile: `{primary_rollout_profile or 'NA'}`",
        f"- Primary heldout profile: `{primary_heldout_profile or 'NA'}`",
        f"- Dataset: `{dataset_paths[0] if dataset_paths else 'NA'}`",
        "",
        "## Key Findings",
        "",
    ]

    if best_model is not None:
        lines.append(
            f"- Best rollout aggregate at `{primary_horizon:.0f}s` / `{primary_rollout_profile}` is "
            f"`{best_model['group']}/{best_model['model_type']}` under train protocol "
            f"`{_train_display(best_model)}`, with "
            f"`pos median={_fmt(best_model['rollout_final_position_error_median_mean'], 4)} m` and "
            f"`completion={_fmt_pct(best_model['rollout_completion_rate_mean'])}`."
        )
    else:
        lines.append("- No finite rollout aggregate is available for the selected primary profile.")

    if _is_finite(main_range):
        lines.append(
            f"- `phnode_full` still shows seed sensitivity at `{primary_horizon:.0f}s`: "
            f"position-median range across matching rows is `{_fmt(main_range, 4)} m`."
        )

    if clean_replay_rows:
        lines.append(
            f"- Clean replay cost rows are available for `{len(clean_replay_rows)}` non-clean training groups, "
            "so the report can distinguish robustness gains from clean-performance regressions."
        )
    else:
        lines.append(
            "- Clean replay cost could not be computed from this suite because matched clean-trained baselines were not present."
        )

    rollout_table_rows = [
        [
            f"{row['group']}/{row['model_type']}",
            _train_display(row),
            row["seeds"],
            _fmt(row["rollout_final_position_error_median_mean"], 4),
            _fmt(row["rollout_final_position_error_p95_mean"], 4),
            _fmt_pct(row["rollout_completion_rate_mean"]),
            _fmt_pct(row["rollout_model_failed_rate_mean"]),
        ]
        for row in rollout_model_rows
    ]

    heldout_table_rows = [
        [
            f"{row['group']}/{row['model_type']}",
            _train_display(row),
            row["seeds"],
            _fmt(row["heldout_position_rmse_mean_mean"], 5),
            _fmt(row["heldout_rotation_geodesic_mean_mean"], 5),
            _fmt_pct(row["heldout_success_rate_mean"]),
        ]
        for row in heldout_model_rows
    ]

    seed_table_rows = [
        [
            row["run_name"],
            f"{row['group']}/{row['model_type']}",
            _train_display(row),
            str(row["seed"]),
            _fmt(row["rollout_final_position_error_median"], 4),
            _fmt(row["rollout_final_position_error_p95"], 4),
            _fmt_pct(row["rollout_completion_rate"]),
            _fmt_pct(row["rollout_model_failed_rate"]),
        ]
        for row in rollout_seed_rows
    ]

    horizon_headers = ["Model", "Train", *[f"Pos @{h:.0f}s" for h in horizons], f"Completion @{primary_horizon:.0f}s"]

    lines.extend(
        [
            "",
            f"## Rollout Summary @{primary_horizon:.0f}s / {primary_rollout_profile}",
            "",
            _markdown_table(
                [
                    "Model",
                    "Train",
                    "Seeds",
                    "Pos Median",
                    "Pos P95",
                    "Completion",
                    "Model Fail",
                ],
                rollout_table_rows,
            ),
            "",
            f"## Heldout Summary / {primary_heldout_profile}",
            "",
            _markdown_table(
                [
                    "Model",
                    "Train",
                    "Seeds",
                    "Pos RMSE Mean",
                    "Rot Geo Mean",
                    "Success",
                ],
                heldout_table_rows,
            ),
            "",
            f"## Rollout By Horizon / {primary_rollout_profile}",
            "",
            _markdown_table(horizon_headers, horizon_table_rows),
            "",
            f"## Rollout By Scenario @{primary_horizon:.0f}s / {primary_rollout_profile}",
            "",
            _markdown_table(
                [
                    "Model",
                    "Train",
                    "Scenario",
                    "Completion",
                    "Model Fail",
                    "Pos Median",
                    "Pos P95",
                    "Rot Median",
                ],
                scenario_table_rows,
            ),
            "",
            f"## Seed-Level Rollout @{primary_horizon:.0f}s / {primary_rollout_profile}",
            "",
            _markdown_table(
                [
                    "Run",
                    "Model",
                    "Train",
                    "Seed",
                    "Pos Median",
                    "Pos P95",
                    "Completion",
                    "Model Fail",
                ],
                seed_table_rows,
            ),
        ]
    )

    if rollout_degradation_rows:
        lines.extend(
            [
                "",
                f"## Clean To {primary_rollout_profile} Degradation @{primary_horizon:.0f}s",
                "",
                _markdown_table(
                    [
                        "Model",
                        "Train",
                        "Pos/Clean Ratio",
                        "Completion Drop",
                        "Pos Degradation",
                    ],
                    rollout_degradation_rows,
                ),
            ]
        )

    if clean_replay_rows:
        lines.extend(
            [
                "",
                f"## Clean Replay Cost @{primary_horizon:.0f}s",
                "",
                _markdown_table(
                    [
                        "Model",
                        "Train",
                        "Heldout Clean Ratio",
                        "Rollout Clean Ratio",
                        "Rollout Clean Degradation",
                    ],
                    clean_replay_rows,
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Train` uses `clean` or `protocol/profile` so Phase-1 can distinguish clean, iid noisy-IC, and `v4-lite` training runs.",
            "- Rollout sections use the selected primary eval profile for headline tables and keep `10s/30s/60s` in the horizon table.",
            "- `Clean To ... Degradation` compares noisy eval against the same run's clean eval.",
            "- `Clean Replay Cost` compares a noisy-trained run's clean eval against the matched clean-trained baseline with the same model and seed when available.",
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
        help="Preferred heldout profile for headline tables. Defaults to the best available Phase-1 profile.",
    )
    parser.add_argument(
        "--rollout-profile",
        type=str,
        default=None,
        help="Preferred rollout profile for headline tables. Defaults to the best available Phase-1 profile.",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=60.0,
        help="Primary rollout horizon in seconds for headline tables. Default: 60",
    )
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Rollout horizons to include in the by-horizon section. Default: 10 30 60",
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
    horizons = sorted({float(args.horizon), *(float(horizon) for horizon in args.horizons)})
    bundle = build_phase1_bundle(
        suite_dir=suite_dir,
        runs=runs,
        horizons=horizons,
        block_profile=args.block_profile,
        heldout_profile=args.heldout_profile,
        rollout_profile=args.rollout_profile,
    )

    primary_rollout_profile = _pick_primary_profile(
        bundle["model_rows"],
        source="rollout",
        requested=args.rollout_profile,
    )
    primary_heldout_profile = _pick_primary_profile(
        bundle["model_rows"],
        source="heldout",
        requested=args.heldout_profile,
    )

    report_text = build_report_text(
        suite_dir=suite_dir,
        bundle=bundle,
        primary_horizon=float(args.horizon),
        horizons=horizons,
        primary_rollout_profile=primary_rollout_profile,
        primary_heldout_profile=primary_heldout_profile,
    )
    output_path = suite_dir / args.output
    output_path.write_text(report_text)
    print(output_path)


if __name__ == "__main__":
    main()
