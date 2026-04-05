#!/usr/bin/env python3
"""Build cross-sweep seed audits and model summary reports."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path


SEED_PATTERN = re.compile(r"(.+)_seed(\d+)$")
BEST_LINE_PATTERN = re.compile(
    r"test_loss=([0-9.eE+-]+)\s+\(epoch\s+(\d+)\)"
)
SCENARIOS = ("PRBS", "CHIRP", "OU")
HORIZONS = ("10.0", "30.0", "60.0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-core", required=True)
    parser.add_argument("--suite-ablation", required=True)
    parser.add_argument("--suite-focus", required=True)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--prefix", default="all_model")
    return parser.parse_args()


def safe_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def is_finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def lower_half_anchor(values: list[float]) -> float:
    finite = sorted(v for v in values if is_finite(v))
    if not finite:
        return math.nan
    cutoff = max(1, math.ceil(len(finite) / 2))
    return statistics.median(finite[:cutoff])


def ratio_to_anchor(value: float, anchor: float) -> float:
    if not is_finite(value):
        return math.inf
    if not is_finite(anchor) or anchor <= 0.0:
        return math.inf
    return value / anchor


def mean(values: list[float]) -> float:
    finite = [v for v in values if is_finite(v)]
    return statistics.fmean(finite) if finite else math.nan


def stdev(values: list[float]) -> float:
    finite = [v for v in values if is_finite(v)]
    if len(finite) <= 1:
        return 0.0 if finite else math.nan
    return statistics.stdev(finite)


def median(values: list[float]) -> float:
    finite = [v for v in values if is_finite(v)]
    return statistics.median(finite) if finite else math.nan


def format_float(value: float, digits: int = 4) -> str:
    if not is_finite(value):
        return "nan"
    return f"{value:.{digits}f}"


def format_pct(value: float, digits: int = 1) -> str:
    if not is_finite(value):
        return "nan"
    return f"{100.0 * value:.{digits}f}%"


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def parse_best_validation(training_log: Path) -> tuple[float, int | None]:
    best_line = None
    with training_log.open(errors="ignore") as handle:
        for line in handle:
            if "Done. Best validation score:" in line:
                best_line = line.strip()
    if not best_line:
        return math.nan, None
    match = BEST_LINE_PATTERN.search(best_line)
    if not match:
        return math.nan, None
    return float(match.group(1)), int(match.group(2))


def resampled_summary_path(run_dir: Path) -> Path:
    matches = sorted(
        glob.glob(
            str(run_dir / "rollout_benchmark" / "resampled_traj30_seed42_*" / "summary.json")
        )
    )
    if not matches:
        raise FileNotFoundError(f"Missing resampled rollout summary under {run_dir}")
    return Path(matches[-1])


def extract_row(run_dir: Path) -> dict:
    run_name = run_dir.name
    match = SEED_PATTERN.match(run_name)
    if not match:
        raise ValueError(f"Unexpected run directory name: {run_name}")
    model_stub, seed_text = match.groups()
    group_name, model_type = model_stub.split("_", 1)
    seed = int(seed_text)

    config = read_json(run_dir / "config.json")
    heldout = read_json(run_dir / "heldout_evaluation.json")
    block = read_json(run_dir / "block_evaluation.json")
    resampled = read_json(resampled_summary_path(run_dir))
    best_test_loss, best_epoch = parse_best_validation(run_dir / "training.log")

    row = {
        "run": run_name,
        "group": group_name,
        "model_type": model_type,
        "model_label": f"{group_name}/{model_type}",
        "seed": seed,
        "suite_dir": str(run_dir.parent),
        "run_dir": str(run_dir),
        "best_test_loss": best_test_loss,
        "best_epoch": best_epoch,
        "heldout_success": safe_float(heldout["overall"]["success_rate"]),
        "heldout_pos_median": safe_float(heldout["overall"]["position_rmse"]["median"]),
        "heldout_pos_p95": safe_float(heldout["overall"]["position_rmse"]["p95"]),
        "block_pos_mean": safe_float(block["position_rmse"]["mean"]),
        "block_rot_mean": safe_float(block["rotation_geodesic"]["mean"]),
        "block_solver_failed_batches": int(block.get("solver_failed_batches", 0)),
        "block_invalid_prediction_batches": int(block.get("invalid_prediction_batches", 0)),
    }

    for scenario in SCENARIOS:
        scen_heldout = heldout["by_scenario"][scenario]
        prefix = scenario.lower()
        row[f"{prefix}_heldout_success"] = safe_float(scen_heldout["success_rate"])
        row[f"{prefix}_heldout_pos_median"] = safe_float(
            scen_heldout["position_rmse"]["median"]
        )

    for horizon in HORIZONS:
        horizon_key = f"h{int(float(horizon))}"
        overall = resampled["overall"][horizon]
        row[f"{horizon_key}_final_pos_median"] = safe_float(
            overall["metrics"]["final_position_error"]["median"]
        )
        row[f"{horizon_key}_final_pos_p95"] = safe_float(
            overall["metrics"]["final_position_error"]["p95"]
        )
        row[f"{horizon_key}_mean_pos_median"] = safe_float(
            overall["metrics"]["mean_position_error"]["median"]
        )
        row[f"{horizon_key}_completion"] = safe_float(overall["rates"]["completed_to_h"])
        row[f"{horizon_key}_model_fail"] = safe_float(
            overall["rates"]["model_failed_by_h"]
        )
        row[f"{horizon_key}_pred_divergence"] = safe_float(
            overall["rates"]["pred_divergence_by_h"]
        )
        row[f"{horizon_key}_so3_violation"] = safe_float(
            overall["rates"]["any_so3_violation_up_to_h"]
        )
        for scenario in SCENARIOS:
            scenario_block = resampled["by_scenario"][scenario][horizon]
            prefix = scenario.lower()
            row[f"{prefix}_{horizon_key}_final_pos_median"] = safe_float(
                scenario_block["metrics"]["final_position_error"]["median"]
            )
            row[f"{prefix}_{horizon_key}_completion"] = safe_float(
                scenario_block["rates"]["completed_to_h"]
            )
            row[f"{prefix}_{horizon_key}_model_fail"] = safe_float(
                scenario_block["rates"]["model_failed_by_h"]
            )
    return row


def classify_model(rows: list[dict]) -> dict:
    n_seeds = len(rows)
    h60_values = [row["h60_final_pos_median"] for row in rows]
    h30_values = [row["h30_final_pos_median"] for row in rows]
    h10_values = [row["h10_final_pos_median"] for row in rows]
    train_values = [row["best_test_loss"] for row in rows]
    heldout_values = [row["heldout_pos_median"] for row in rows]
    block_values = [row["block_pos_mean"] for row in rows]

    anchors = {
        "h60": lower_half_anchor(h60_values),
        "h30": lower_half_anchor(h30_values),
        "h10": lower_half_anchor(h10_values),
        "train": lower_half_anchor(train_values),
        "heldout": lower_half_anchor(heldout_values),
        "block": lower_half_anchor(block_values),
    }

    problematic = []
    seed_notes = {}
    for row in rows:
        reasons = []
        if not is_finite(row["h60_final_pos_median"]):
            if row["h60_completion"] < 0.9 or row["h60_model_fail"] > 0.1:
                reasons.append("non-finite 60s error with catastrophic rollout failure")
        else:
            primary_ratio = ratio_to_anchor(row["h60_final_pos_median"], anchors["h60"])
            support_count = 0
            if primary_ratio >= 3.0:
                reasons.append(f"60s error {primary_ratio:.1f}x stable-cluster anchor")
                if ratio_to_anchor(row["h30_final_pos_median"], anchors["h30"]) >= 3.0:
                    support_count += 1
                    reasons.append("30s error also exceeds 3x anchor")
                if ratio_to_anchor(row["h10_final_pos_median"], anchors["h10"]) >= 3.0:
                    support_count += 1
                    reasons.append("10s error also exceeds 3x anchor")
                if ratio_to_anchor(row["best_test_loss"], anchors["train"]) >= 3.0:
                    support_count += 1
                    reasons.append("training loss exceeds 3x anchor")
                if ratio_to_anchor(row["heldout_pos_median"], anchors["heldout"]) >= 3.0:
                    support_count += 1
                    reasons.append("heldout replay error exceeds 3x anchor")
                if ratio_to_anchor(row["block_pos_mean"], anchors["block"]) >= 3.0:
                    support_count += 1
                    reasons.append("block RMSE exceeds 3x anchor")
                if row["h60_pred_divergence"] >= 0.01:
                    support_count += 1
                    reasons.append("non-zero pred divergence in rollout")
                if row["h60_model_fail"] >= 0.05:
                    support_count += 1
                    reasons.append("material rollout failure rate")
                if support_count == 0:
                    reasons.clear()

        if reasons:
            problematic.append(row["seed"])
            seed_notes[row["seed"]] = "; ".join(reasons)

    finite_h60 = [value for value in h60_values if is_finite(value)]
    mean_h60 = mean(finite_h60)
    median_h60 = median(finite_h60)
    completion_mean = mean([row["h60_completion"] for row in rows])

    if problematic:
        if len(problematic) / max(1, n_seeds) > (1.0 / 3.0):
            status = "family_unstable"
            model_note = (
                f"{len(problematic)}/{n_seeds} seeds meet the bad-outlier criterion; "
                "treat this as family-level instability rather than something to prune away."
            )
        else:
            status = "prunable_bad_outliers"
            model_note = (
                f"Bad-outlier seeds are non-representative relative to the stable cluster: "
                f"{','.join(str(seed) for seed in problematic)}."
            )
    elif (is_finite(median_h60) and median_h60 >= 10.0) or (
        is_finite(completion_mean) and completion_mean < 0.95
    ):
        status = "structurally_poor"
        model_note = (
            "Most seeds are already poor, so there is no single non-representative seed "
            "worth pruning."
        )
    else:
        status = "stable"
        model_note = "No problematic bad seed under the current criterion."

    return {
        "status": status,
        "problematic_seeds": problematic,
        "seed_notes": seed_notes,
        "model_note": model_note,
        "anchor_h60": anchors["h60"],
        "anchor_h30": anchors["h30"],
        "anchor_train": anchors["train"],
    }


def aggregate_rows(rows: list[dict], label: str, audit: dict, variant_type: str) -> dict:
    summary = {
        "model_label": label,
        "variant_type": variant_type,
        "n_seeds": len(rows),
        "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda item: item["seed"])),
        "problematic_seeds": ",".join(str(seed) for seed in audit["problematic_seeds"]),
        "audit_status": audit["status"],
        "audit_note": audit["model_note"],
        "best_test_loss_mean": mean([row["best_test_loss"] for row in rows]),
        "best_test_loss_std": stdev([row["best_test_loss"] for row in rows]),
        "heldout_pos_median_mean": mean([row["heldout_pos_median"] for row in rows]),
        "heldout_pos_median_std": stdev([row["heldout_pos_median"] for row in rows]),
        "block_pos_mean": mean([row["block_pos_mean"] for row in rows]),
        "block_pos_std": stdev([row["block_pos_mean"] for row in rows]),
    }
    for scenario in SCENARIOS:
        prefix = scenario.lower()
        summary[f"{prefix}_heldout_pos_mean"] = mean(
            [row[f"{prefix}_heldout_pos_median"] for row in rows]
        )
        summary[f"{prefix}_heldout_success_mean"] = mean(
            [row[f"{prefix}_heldout_success"] for row in rows]
        )
    for horizon in ("10", "30", "60"):
        summary[f"h{horizon}_final_pos_mean"] = mean(
            [row[f"h{horizon}_final_pos_median"] for row in rows]
        )
        summary[f"h{horizon}_final_pos_std"] = stdev(
            [row[f"h{horizon}_final_pos_median"] for row in rows]
        )
        summary[f"h{horizon}_completion_mean"] = mean(
            [row[f"h{horizon}_completion"] for row in rows]
        )
        summary[f"h{horizon}_model_fail_mean"] = mean(
            [row[f"h{horizon}_model_fail"] for row in rows]
        )
        summary[f"h{horizon}_p95_mean"] = mean(
            [row[f"h{horizon}_final_pos_p95"] for row in rows]
        )
        for scenario in SCENARIOS:
            prefix = scenario.lower()
            summary[f"{prefix}_h{horizon}_final_pos_mean"] = mean(
                [row[f"{prefix}_h{horizon}_final_pos_median"] for row in rows]
            )
            summary[f"{prefix}_h{horizon}_completion_mean"] = mean(
                [row[f"{prefix}_h{horizon}_completion"] for row in rows]
            )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_seed_audit_md(seed_rows: list[dict], audits: dict[str, dict]) -> str:
    sections = ["# All-Model Seed Audit", ""]
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[row["model_label"]].append(row)
    for model_label in sorted(grouped):
        audit = audits[model_label]
        sections.append(f"## {model_label}")
        sections.append(
            f"- Audit status: `{audit['status']}` | Problematic seeds: "
            f"`{','.join(str(seed) for seed in audit['problematic_seeds']) or 'none'}`"
        )
        sections.append(f"- Note: {audit['model_note']}")
        sections.append("")
        table_rows = []
        for row in sorted(grouped[model_label], key=lambda item: item["seed"]):
            seed_note = audit["seed_notes"].get(row["seed"], "")
            table_rows.append(
                [
                    str(row["seed"]),
                    format_float(row["best_test_loss"], 4),
                    format_float(row["heldout_pos_median"], 4),
                    format_float(row["block_pos_mean"], 5),
                    format_float(row["h10_final_pos_median"], 4),
                    format_float(row["h30_final_pos_median"], 4),
                    format_float(row["h60_final_pos_median"], 4),
                    format_pct(row["h60_completion"]),
                    format_pct(row["h60_model_fail"]),
                    seed_note or "normal under current rule",
                ]
            )
        sections.append(
            markdown_table(
                [
                    "Seed",
                    "Train Loss",
                    "Heldout",
                    "Block",
                    "10s",
                    "30s",
                    "60s",
                    "Comp60",
                    "Fail60",
                    "Judgement",
                ],
                table_rows,
            )
        )
        sections.append("")
    return "\n".join(sections).strip() + "\n"


def build_compact_md(summary_rows: list[dict]) -> str:
    sections = ["# All-Model Compact Summary", ""]
    sections.append(
        "This table is for quick ranking. Keep the all-seed rows as the primary evidence. "
        "Diagnostic pruned rows appear only when the seed audit says a stable cluster exists."
    )
    sections.append("")
    table_rows = []
    ranked = sorted(
        summary_rows,
        key=lambda row: (
            row["variant_type"] != "all_seeds",
            row["h60_final_pos_mean"],
            row["model_label"],
        ),
    )
    for row in ranked:
        label = row["model_label"]
        if row["variant_type"] != "all_seeds":
            label = f"{label} [{row['variant_type']}]"
        table_rows.append(
            [
                label,
                row["seeds"],
                row["problematic_seeds"] or "none",
                row["audit_status"],
                format_float(row["best_test_loss_mean"], 4),
                format_float(row["heldout_pos_median_mean"], 4),
                format_float(row["h10_final_pos_mean"], 4),
                format_float(row["h30_final_pos_mean"], 4),
                format_float(row["h60_final_pos_mean"], 4),
                format_pct(row["h60_completion_mean"]),
                format_float(row["prbs_h60_final_pos_mean"], 4),
                format_float(row["chirp_h60_final_pos_mean"], 4),
                format_float(row["ou_h60_final_pos_mean"], 4),
                row["audit_note"],
            ]
        )
    sections.append(
        markdown_table(
            [
                "Model",
                "Seeds",
                "Problematic",
                "Audit",
                "Train",
                "Heldout",
                "10s",
                "30s",
                "60s",
                "Comp60",
                "PRBS60",
                "CHIRP60",
                "OU60",
                "Note",
            ],
            table_rows,
        )
    )
    sections.append("")
    return "\n".join(sections)


def build_detailed_md(summary_rows: list[dict]) -> str:
    sections = ["# All-Model Detailed Summary", ""]
    primary_rows = [row for row in summary_rows if row["variant_type"] == "all_seeds"]
    diagnostic_rows = [row for row in summary_rows if row["variant_type"] != "all_seeds"]
    ranked = sorted(primary_rows, key=lambda row: (row["h60_final_pos_mean"], row["model_label"]))

    sections.append("## Training-Side Summary")
    sections.append("")
    sections.append(
        markdown_table(
            [
                "Model",
                "Seeds",
                "Audit",
                "Train Mean",
                "Train Std",
                "Heldout Mean",
                "Heldout Std",
                "Block Mean",
                "Block Std",
            ],
            [
                [
                    row["model_label"],
                    row["seeds"],
                    row["audit_status"],
                    format_float(row["best_test_loss_mean"], 4),
                    format_float(row["best_test_loss_std"], 4),
                    format_float(row["heldout_pos_median_mean"], 4),
                    format_float(row["heldout_pos_median_std"], 4),
                    format_float(row["block_pos_mean"], 5),
                    format_float(row["block_pos_std"], 5),
                ]
                for row in ranked
            ],
        )
    )
    sections.append("")

    sections.append("## Heldout Scenario Summary")
    sections.append("")
    sections.append(
        markdown_table(
            [
                "Model",
                "PRBS Heldout",
                "PRBS Success",
                "CHIRP Heldout",
                "CHIRP Success",
                "OU Heldout",
                "OU Success",
            ],
            [
                [
                    row["model_label"],
                    format_float(row["prbs_heldout_pos_mean"], 4),
                    format_pct(row["prbs_heldout_success_mean"]),
                    format_float(row["chirp_heldout_pos_mean"], 4),
                    format_pct(row["chirp_heldout_success_mean"]),
                    format_float(row["ou_heldout_pos_mean"], 4),
                    format_pct(row["ou_heldout_success_mean"]),
                ]
                for row in ranked
            ],
        )
    )
    sections.append("")

    sections.append("## Overall Rollout Summary")
    sections.append("")
    sections.append(
        markdown_table(
            [
                "Model",
                "10s Mean",
                "10s Std",
                "10s Comp",
                "30s Mean",
                "30s Std",
                "30s Comp",
                "60s Mean",
                "60s Std",
                "60s P95",
                "60s Comp",
                "60s Fail",
            ],
            [
                [
                    row["model_label"],
                    format_float(row["h10_final_pos_mean"], 4),
                    format_float(row["h10_final_pos_std"], 4),
                    format_pct(row["h10_completion_mean"]),
                    format_float(row["h30_final_pos_mean"], 4),
                    format_float(row["h30_final_pos_std"], 4),
                    format_pct(row["h30_completion_mean"]),
                    format_float(row["h60_final_pos_mean"], 4),
                    format_float(row["h60_final_pos_std"], 4),
                    format_float(row["h60_p95_mean"], 4),
                    format_pct(row["h60_completion_mean"]),
                    format_pct(row["h60_model_fail_mean"]),
                ]
                for row in ranked
            ],
        )
    )
    sections.append("")

    for horizon in ("10", "30", "60"):
        sections.append(f"## Scenario Split @ {horizon}s")
        sections.append("")
        sections.append(
            markdown_table(
                [
                    "Model",
                    "PRBS",
                    "PRBS Comp",
                    "CHIRP",
                    "CHIRP Comp",
                    "OU",
                    "OU Comp",
                    "Problematic Seeds",
                ],
                [
                    [
                        row["model_label"],
                        format_float(row[f"prbs_h{horizon}_final_pos_mean"], 4),
                        format_pct(row[f"prbs_h{horizon}_completion_mean"]),
                        format_float(row[f"chirp_h{horizon}_final_pos_mean"], 4),
                        format_pct(row[f"chirp_h{horizon}_completion_mean"]),
                        format_float(row[f"ou_h{horizon}_final_pos_mean"], 4),
                        format_pct(row[f"ou_h{horizon}_completion_mean"]),
                        row["problematic_seeds"] or "none",
                    ]
                    for row in ranked
                ],
            )
        )
        sections.append("")
    if diagnostic_rows:
        sections.append("## Diagnostic Pruned Views")
        sections.append("")
        sections.append(
            markdown_table(
                [
                    "Model",
                    "Variant",
                    "Kept Seeds",
                    "Problematic Seeds",
                    "60s Mean",
                    "60s Std",
                    "60s Comp",
                    "Train Mean",
                    "Heldout Mean",
                    "Note",
                ],
                [
                    [
                        row["model_label"],
                        row["variant_type"],
                        row["seeds"],
                        row["problematic_seeds"] or "none",
                        format_float(row["h60_final_pos_mean"], 4),
                        format_float(row["h60_final_pos_std"], 4),
                        format_pct(row["h60_completion_mean"]),
                        format_float(row["best_test_loss_mean"], 4),
                        format_float(row["heldout_pos_median_mean"], 4),
                        row["audit_note"],
                    ]
                    for row in sorted(
                        diagnostic_rows,
                        key=lambda row: (row["h60_final_pos_mean"], row["model_label"]),
                    )
                ],
            )
        )
        sections.append("")
    return "\n".join(sections)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for suite_dir in (args.suite_core, args.suite_ablation, args.suite_focus):
        for run_dir in sorted(Path(suite_dir).iterdir()):
            if run_dir.is_dir() and SEED_PATTERN.match(run_dir.name):
                run_rows.append(extract_row(run_dir))

    grouped = defaultdict(list)
    for row in run_rows:
        grouped[row["model_label"]].append(row)

    audits = {
        model_label: classify_model(sorted(rows, key=lambda item: item["seed"]))
        for model_label, rows in grouped.items()
    }

    seed_audit_rows = []
    summary_rows = []
    for model_label, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: item["seed"])
        audit = audits[model_label]
        for row in rows:
            record = dict(row)
            record["audit_status"] = audit["status"]
            record["problematic_seed"] = row["seed"] in audit["problematic_seeds"]
            record["seed_note"] = audit["seed_notes"].get(row["seed"], "")
            record["model_note"] = audit["model_note"]
            seed_audit_rows.append(record)

        summary_rows.append(aggregate_rows(rows, model_label, audit, "all_seeds"))
        if audit["status"] == "prunable_bad_outliers":
            kept_rows = [row for row in rows if row["seed"] not in audit["problematic_seeds"]]
            variant_label = model_label
            summary_rows.append(
                aggregate_rows(
                    kept_rows,
                    variant_label,
                    audit,
                    f"drop_{'-'.join(str(seed) for seed in audit['problematic_seeds'])}",
                )
            )

    seed_audit_csv = output_dir / f"{args.prefix}_seed_audit.csv"
    seed_audit_md = output_dir / f"{args.prefix}_seed_audit.md"
    compact_csv = output_dir / f"{args.prefix}_summary_compact.csv"
    compact_md = output_dir / f"{args.prefix}_summary_compact.md"
    detailed_csv = output_dir / f"{args.prefix}_summary_detailed.csv"
    detailed_md = output_dir / f"{args.prefix}_summary_detailed.md"

    write_csv(seed_audit_csv, seed_audit_rows)
    seed_audit_md.write_text(build_seed_audit_md(seed_audit_rows, audits))
    write_csv(compact_csv, summary_rows)
    compact_md.write_text(build_compact_md(summary_rows))
    write_csv(detailed_csv, summary_rows)
    detailed_md.write_text(build_detailed_md(summary_rows))

    print(f"Wrote {seed_audit_csv}")
    print(f"Wrote {seed_audit_md}")
    print(f"Wrote {compact_csv}")
    print(f"Wrote {compact_md}")
    print(f"Wrote {detailed_csv}")
    print(f"Wrote {detailed_md}")


if __name__ == "__main__":
    main()
