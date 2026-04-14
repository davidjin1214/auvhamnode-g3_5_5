#!/usr/bin/env python3
"""Small query helpers for the OC data catalog."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = REPO_ROOT / "analysis" / "oc_data_catalog"


def add_common_catalog_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=DEFAULT_BASE,
        help="Path to analysis/oc_data_catalog.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to stdout.",
    )


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows: list[dict], output_path: Path | None) -> None:
    if not rows:
        fieldnames: list[str] = []
    else:
        fieldnames = list(rows[0].keys())

    if output_path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cmd_training_curve(args: argparse.Namespace) -> int:
    table_path = args.catalog_dir / "training_history_long.csv"
    rows = []
    for row in read_csv_rows(table_path):
        if row["run_uid"] != args.run_uid:
            continue
        if row["metric_key"] != args.metric_key:
            continue
        if args.split and row["split"] != args.split:
            continue
        rows.append(
            {
                "run_uid": row["run_uid"],
                "epoch": row["epoch"],
                "global_step": row["global_step"],
                "split": row["split"],
                "metric_key": row["metric_key"],
                "metric_value": row["metric_value"],
            }
        )

    rows.sort(key=lambda row: int(row["epoch"]))
    write_rows(rows, args.output)
    return 0


def cmd_rollout_metric(args: argparse.Namespace) -> int:
    table_name = (
        "canonical_rollout_summary_long.csv"
        if args.canonical
        else "rollout_summary_long.csv"
    )
    table_path = args.catalog_dir / table_name
    rows = []
    for row in read_csv_rows(table_path):
        if args.run_uid and row["run_uid"] != args.run_uid:
            continue
        if args.train_type and row["train_type"] != args.train_type:
            continue
        if args.model_type and row["model_type"] != args.model_type:
            continue
        if row["eval_profile"] != args.eval_profile:
            continue
        if row["scope"] != args.scope:
            continue
        if row["scenario"] != args.scenario:
            continue
        if row["horizon_s"] != args.horizon_s:
            continue
        if row["category"] != args.category:
            continue
        if row["metric_name"] != args.metric_name:
            continue
        if row["stat_name"] != args.stat_name:
            continue
        rows.append(
            {
                "run_uid": row["run_uid"],
                "model_type": row["model_type"],
                "seed": row["seed"],
                "train_type": row["train_type"],
                "rollout_run_id": row["rollout_run_id"],
                "eval_profile": row["eval_profile"],
                "scope": row["scope"],
                "scenario": row["scenario"],
                "horizon_s": row["horizon_s"],
                "category": row["category"],
                "metric_name": row["metric_name"],
                "stat_name": row["stat_name"],
                "value_numeric": row["value_numeric"],
                "source_file": row["source_file"],
            }
        )

    rows.sort(
        key=lambda row: (
            row["model_type"],
            int(row["seed"]),
            row["run_uid"],
            row["rollout_run_id"],
        )
    )
    write_rows(rows, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    training_parser = subparsers.add_parser(
        "training-curve",
        help="Extract one training curve from training_history_long.csv.",
    )
    add_common_catalog_args(training_parser)
    training_parser.add_argument("--run-uid", required=True, help="Exact run_uid.")
    training_parser.add_argument(
        "--metric-key",
        required=True,
        help="Exact metric key, e.g. train_total, test_total, lr.",
    )
    training_parser.add_argument(
        "--split",
        default=None,
        help="Optional split filter: train, test, or meta.",
    )
    training_parser.set_defaults(func=cmd_training_curve)

    rollout_parser = subparsers.add_parser(
        "rollout-metric",
        help="Extract one rollout metric slice from rollout summary tables.",
    )
    add_common_catalog_args(rollout_parser)
    rollout_parser.add_argument(
        "--canonical",
        action="store_true",
        help="Read from canonical_rollout_summary_long.csv instead of the raw table.",
    )
    rollout_parser.add_argument("--run-uid", default=None, help="Optional exact run_uid.")
    rollout_parser.add_argument("--train-type", default=None, help="Optional train_type.")
    rollout_parser.add_argument("--model-type", default=None, help="Optional model_type.")
    rollout_parser.add_argument("--eval-profile", required=True, help="Exact eval_profile.")
    rollout_parser.add_argument(
        "--scope",
        default="overall",
        help="Scope filter, usually overall or by_scenario.",
    )
    rollout_parser.add_argument(
        "--scenario",
        default="",
        help="Scenario filter. Use empty string for overall rows.",
    )
    rollout_parser.add_argument("--horizon-s", required=True, help="Exact horizon_s, e.g. 60.0")
    rollout_parser.add_argument(
        "--category",
        default="metrics",
        help="Category filter, e.g. metrics, rates, meta.",
    )
    rollout_parser.add_argument(
        "--metric-name",
        required=True,
        help="Exact rollout metric name, e.g. final_position_error.",
    )
    rollout_parser.add_argument(
        "--stat-name",
        default="median",
        help="Exact stat_name. Use empty string for rates/meta rows.",
    )
    rollout_parser.set_defaults(func=cmd_rollout_metric)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
