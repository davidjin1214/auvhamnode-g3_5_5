#!/usr/bin/env python3
"""Reusable plotting/export templates for the OC data catalog."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_DIR = REPO_ROOT / "analysis" / "oc_data_catalog"


def add_common_catalog_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=DEFAULT_CATALOG_DIR,
        help="Path to analysis/oc_data_catalog.",
    )


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def csv_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    parsed: list[str] = []
    for value in values:
        parsed.extend([item.strip() for item in value.split(",") if item.strip()])
    return parsed


def set_mpl_config_dir(catalog_dir: Path) -> None:
    mpl_dir = catalog_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def cmd_plot_training_curves(args: argparse.Namespace) -> int:
    set_mpl_config_dir(args.catalog_dir)
    import matplotlib.pyplot as plt

    metric_keys = csv_list(args.metric_key)
    run_uids = csv_list(args.run_uid)
    if not metric_keys or not run_uids:
        raise SystemExit("--run-uid and --metric-key must be provided.")

    rows = read_csv_rows(args.catalog_dir / "training_history_long.csv")
    filtered_rows: list[dict] = []
    series: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)

    for row in rows:
        if row["run_uid"] not in run_uids:
            continue
        if row["metric_key"] not in metric_keys:
            continue
        if args.split and row["split"] != args.split:
            continue

        x_value = float(row[args.x_axis])
        y_value = float(row["metric_value"])
        series[(row["run_uid"], row["metric_key"])].append((x_value, y_value))
        filtered_rows.append(
            {
                "run_uid": row["run_uid"],
                "epoch": row["epoch"],
                "global_step": row["global_step"],
                "split": row["split"],
                "metric_key": row["metric_key"],
                "metric_name": row["metric_name"],
                "metric_value": row["metric_value"],
            }
        )

    if not series:
        raise SystemExit("No rows matched the requested training curve filters.")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for (run_uid, metric_key), points in sorted(series.items()):
        points.sort(key=lambda item: item[0])
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        label = f"{metric_key} | {run_uid}"
        ax.plot(xs, ys, linewidth=1.8, label=label)

    ax.set_xlabel(args.x_axis)
    ax.set_ylabel("metric_value")
    ax.set_title(args.title or "OC Training Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    if args.csv_output:
        filtered_rows.sort(
            key=lambda row: (
                row["run_uid"],
                row["metric_key"],
                float(row[args.x_axis]),
            )
        )
        write_csv(
            args.csv_output,
            filtered_rows,
            [
                "run_uid",
                "epoch",
                "global_step",
                "split",
                "metric_key",
                "metric_name",
                "metric_value",
            ],
        )

    print(f"Saved training curve plot to {args.output}")
    if args.csv_output:
        print(f"Saved filtered training curve data to {args.csv_output}")
    return 0


def load_run_metadata(catalog_dir: Path, canonical: bool) -> dict[str, dict]:
    inventory_name = "canonical_run_inventory.csv" if canonical else "run_inventory.csv"
    inventory_rows = {
        row["run_uid"]: row for row in read_csv_rows(catalog_dir / inventory_name)
    }
    annotation_rows = {
        row["run_uid"]: row for row in read_csv_rows(catalog_dir / "run_annotations.csv")
    }

    metadata: dict[str, dict] = {}
    for run_uid, inventory_row in inventory_rows.items():
        merged = dict(inventory_row)
        merged.update(annotation_rows.get(run_uid, {}))
        metadata[run_uid] = merged
    return metadata


def cmd_export_rollout_table(args: argparse.Namespace) -> int:
    table_name = (
        "canonical_rollout_summary_long.csv"
        if args.canonical
        else "rollout_summary_long.csv"
    )
    rows = read_csv_rows(args.catalog_dir / table_name)
    metadata_map = load_run_metadata(args.catalog_dir, args.canonical)

    eval_profiles = csv_list(args.eval_profile)
    if not eval_profiles:
        raise SystemExit("--eval-profile is required.")

    filtered: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["eval_profile"] not in eval_profiles:
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

        meta = metadata_map.get(row["run_uid"])
        if meta is None:
            continue
        if args.train_type and meta["train_type"] != args.train_type:
            continue
        if args.model_type and meta["model_type"] not in csv_list(args.model_type):
            continue
        if args.group and meta["group"] not in csv_list(args.group):
            continue
        if args.experiment_bucket and meta.get("experiment_bucket") not in csv_list(
            args.experiment_bucket
        ):
            continue

        filtered[row["run_uid"]][row["eval_profile"]] = row

    export_rows: list[dict] = []
    for run_uid, profile_map in filtered.items():
        meta = metadata_map[run_uid]
        export_row = {
            "run_uid": run_uid,
            "suite_name": meta["suite_name"],
            "group": meta["group"],
            "model_type": meta["model_type"],
            "seed": meta["seed"],
            "train_type": meta["train_type"],
            "noise_profile_train": meta["noise_profile_train"],
            "experiment_bucket": meta.get("experiment_bucket", ""),
            "is_primary_experiment": meta.get("is_primary_experiment", ""),
        }
        for eval_profile in eval_profiles:
            matched = profile_map.get(eval_profile)
            export_row[f"{eval_profile}__value_numeric"] = (
                matched["value_numeric"] if matched else ""
            )
            export_row[f"{eval_profile}__rollout_run_id"] = (
                matched["rollout_run_id"] if matched else ""
            )
        export_rows.append(export_row)

    export_rows.sort(
        key=lambda row: (
            row["train_type"],
            row["group"],
            row["model_type"],
            int(row["seed"]),
            row["run_uid"],
        )
    )

    fieldnames = [
        "run_uid",
        "suite_name",
        "group",
        "model_type",
        "seed",
        "train_type",
        "noise_profile_train",
        "experiment_bucket",
        "is_primary_experiment",
    ]
    for eval_profile in eval_profiles:
        fieldnames.extend(
            [
                f"{eval_profile}__value_numeric",
                f"{eval_profile}__rollout_run_id",
            ]
        )

    write_csv(args.output, export_rows, fieldnames)
    print(f"Saved rollout table to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plot_parser = subparsers.add_parser(
        "plot-training-curves",
        help="Plot one or more training curves from training_history_long.csv.",
    )
    add_common_catalog_args(plot_parser)
    plot_parser.add_argument(
        "--run-uid",
        action="append",
        required=True,
        help="Exact run_uid. Repeat or pass comma-separated values.",
    )
    plot_parser.add_argument(
        "--metric-key",
        action="append",
        required=True,
        help="Exact metric_key. Repeat or pass comma-separated values.",
    )
    plot_parser.add_argument(
        "--split",
        default=None,
        help="Optional split filter: train, test, or meta.",
    )
    plot_parser.add_argument(
        "--x-axis",
        choices=["epoch", "global_step"],
        default="epoch",
        help="Horizontal axis for the plot.",
    )
    plot_parser.add_argument("--title", default=None, help="Optional plot title.")
    plot_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="PNG output path for the figure.",
    )
    plot_parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV output path for the filtered curve data.",
    )
    plot_parser.set_defaults(func=cmd_plot_training_curves)

    table_parser = subparsers.add_parser(
        "export-rollout-table",
        help="Export a wide rollout metric table from canonical or raw summary data.",
    )
    add_common_catalog_args(table_parser)
    table_parser.add_argument(
        "--canonical",
        action="store_true",
        help="Read from canonical_rollout_summary_long.csv.",
    )
    table_parser.add_argument(
        "--eval-profile",
        action="append",
        required=True,
        help="Eval profile(s). Repeat or pass comma-separated values.",
    )
    table_parser.add_argument("--train-type", default=None, help="Optional train_type filter.")
    table_parser.add_argument(
        "--model-type",
        action="append",
        default=None,
        help="Optional model_type filter. Repeat or pass comma-separated values.",
    )
    table_parser.add_argument(
        "--group",
        action="append",
        default=None,
        help="Optional group filter. Repeat or pass comma-separated values.",
    )
    table_parser.add_argument(
        "--experiment-bucket",
        action="append",
        default=None,
        help="Optional experiment_bucket filter. Repeat or pass comma-separated values.",
    )
    table_parser.add_argument("--scope", default="overall", help="Scope filter.")
    table_parser.add_argument(
        "--scenario",
        default="",
        help="Scenario filter. Use empty string for overall rows.",
    )
    table_parser.add_argument(
        "--horizon-s",
        default="60.0",
        help="Horizon filter, e.g. 60.0.",
    )
    table_parser.add_argument(
        "--category",
        default="metrics",
        help="Category filter, e.g. metrics, rates, meta.",
    )
    table_parser.add_argument(
        "--metric-name",
        default="final_position_error",
        help="Metric name filter.",
    )
    table_parser.add_argument(
        "--stat-name",
        default="median",
        help="Stat name filter. Use empty string for rate/meta rows.",
    )
    table_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV output path.",
    )
    table_parser.set_defaults(func=cmd_export_rollout_table)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
