from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from rollout_benchmark_engine import (
    aggregate_horizon_rows,
    build_heldout_trajectory_specs,
    build_resampled_trajectory_specs,
    build_sim_config,
    build_model,
    ensure_runtime_imports,
    evaluate_rollout,
    get_torch,
    load_dataset_artifact,
    parse_scenarios,
    rollout_trajectory_batch,
    resolve_dataset_path,
    resolve_generation_config_payload,
    summarize_rollout_outcomes,
    summarize_time_series,
)
from rollout_benchmark_reporting import (
    export_diagnostic_rollout_plots,
    format_wall_time,
    plot_error_growth,
    plot_example_rollouts,
    plot_terminal_error_boxplots,
    progress_print,
    resolve_output_dir,
    write_csv,
    write_metric_contract,
    write_summary_report,
)


# ---------------------------------------------------------------------------
# Diagnostic case selection (lightweight filtering on rollout KPI rows)
# ---------------------------------------------------------------------------

def build_diagnostic_cases(rows, max_horizon, top_k=5):
    max_rows = [row for row in rows if np.isclose(float(row["horizon_s"]), float(max_horizon))]
    if not max_rows:
        return {
            "all_cases": [],
            "earliest_failures": [],
            "largest_terminal_errors": [],
            "velocity_violations": [],
        }

    cases = []
    for row in max_rows:
        pos_err = float(row["final_position_error"])
        rot_err = float(row["final_rotation_geodesic"])
        total_vel_violation = int(
            np.isfinite(float(row["any_total_velocity_violation_up_to_h"]))
            and float(row["any_total_velocity_violation_up_to_h"]) > 0.5
        )
        case = {
            "scenario": row["scenario"],
            "trajectory_id": row["trajectory_id"],
            "seed": int(row["seed"]),
            "horizon_s": float(row["horizon_s"]),
            "failure_reason": row["failure_reason"],
            "completed_time": float(row["completed_time"]),
            "completed_to_h": int(row["completed_to_h"]),
            "failed_by_h": int(row["failed_by_h"]),
            "gt_failed_by_h": int(row["gt_failed_by_h"]),
            "model_failed_by_h": int(row["model_failed_by_h"]),
            "final_position_error": pos_err,
            "final_rotation_geodesic": rot_err,
            "any_depth_violation_up_to_h": int(
                np.isfinite(float(row["any_depth_violation_up_to_h"]))
                and float(row["any_depth_violation_up_to_h"]) > 0.5
            ),
            "any_total_velocity_violation_up_to_h": total_vel_violation,
            "any_relative_velocity_violation_up_to_h": int(
                np.isfinite(float(row["any_relative_velocity_violation_up_to_h"]))
                and float(row["any_relative_velocity_violation_up_to_h"]) > 0.5
            ),
            "any_so3_violation_up_to_h": int(
                np.isfinite(float(row["any_so3_violation_up_to_h"]))
                and float(row["any_so3_violation_up_to_h"]) > 0.5
            ),
        }
        if row["failure_reason"] != "completed":
            case["diagnostic_label"] = "rollout_failure"
            case["severity_score"] = float(max_horizon) - float(row["completed_time"])
        elif total_vel_violation:
            # Prioritize violations in total velocity, which is the observable-
            # space velocity contract used for main benchmark reporting.
            case["diagnostic_label"] = "velocity_violation"
            case["severity_score"] = pos_err if np.isfinite(pos_err) else float("-inf")
        else:
            case["diagnostic_label"] = "high_terminal_error"
            case["severity_score"] = pos_err if np.isfinite(pos_err) else float("-inf")
        cases.append(case)

    failures = sorted(
        [case for case in cases if case["failure_reason"] != "completed"],
        key=lambda case: (case["completed_time"], case["scenario"], case["seed"]),
    )[:top_k]
    terminal_errors = sorted(
        [
            case for case in cases
            if case["completed_to_h"] == 1 and np.isfinite(case["final_position_error"])
        ],
        key=lambda case: (-case["final_position_error"], case["scenario"], case["seed"]),
    )[:top_k]
    velocity_cases = sorted(
        [
            case for case in cases
            if case["completed_to_h"] == 1
            and case["any_total_velocity_violation_up_to_h"] == 1
        ],
        key=lambda case: (-case["final_position_error"], case["scenario"], case["seed"]),
    )[:top_k]
    all_cases = sorted(
        cases,
        key=lambda case: (
            case["failure_reason"] == "completed",
            -case["any_total_velocity_violation_up_to_h"],
            -case["severity_score"],
            case["scenario"],
            case["seed"],
        ),
    )
    return {
        "all_cases": all_cases,
        "earliest_failures": failures,
        "largest_terminal_errors": terminal_errors,
        "velocity_violations": velocity_cases,
    }


def select_example_rollouts(evaluations, max_horizon):
    examples = []
    scenario_names = sorted({evaluation.rollout.scenario for evaluation in evaluations})
    for scenario_name in scenario_names:
        subset = [
            evaluation
            for evaluation in evaluations
            if evaluation.rollout.scenario == scenario_name
            and evaluation.rollout.completed_time >= max_horizon
        ]
        if not subset:
            subset = [
                evaluation
                for evaluation in evaluations
                if evaluation.rollout.scenario == scenario_name
                and np.isfinite(evaluation.analysis.terminal_position_error)
            ]
        if not subset:
            continue

        terminal_errors = np.array(
            [evaluation.analysis.terminal_position_error for evaluation in subset]
        )
        median_error = float(np.median(terminal_errors))
        best_idx = int(np.argmin(np.abs(terminal_errors - median_error)))
        examples.append(subset[best_idx])
    return examples


def select_diagnostic_plot_cases(diagnostic_cases, max_plots):
    ordered = []
    seen = set()

    for key in ["earliest_failures", "velocity_violations", "largest_terminal_errors", "all_cases"]:
        for case in diagnostic_cases.get(key, []):
            trajectory_id = case["trajectory_id"]
            if trajectory_id in seen:
                continue
            seen.add(trajectory_id)
            ordered.append(case)
            if len(ordered) >= max_plots:
                return ordered

    return ordered


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def print_scenario_progress_summary(
    scenario_name,
    horizon_record,
    outcome_record,
    max_horizon,
    elapsed_seconds,
    quiet,
):
    if quiet:
        return

    metrics = horizon_record.get("metrics", {})
    rates = horizon_record.get("rates", {})
    pos_stats = metrics.get("final_position_error", {})
    rot_stats = metrics.get("final_rotation_geodesic", {})
    counts = outcome_record.get("counts", {})
    total_n = int(outcome_record.get("n_trajectories", 0))
    cond_n = int(pos_stats.get("count", 0))
    model_fail_count = (
        counts.get("pred_divergence", 0)
        + counts.get("solver_failure", 0)
        + counts.get("nan_or_inf", 0)
    )

    progress_print(
        (
            f"[scenario done] {scenario_name}"
            f" | completed={counts.get('completed', 0)}/{total_n}"
            f" | gt_div={counts.get('gt_divergence', 0)}"
            f" | model_fail={model_fail_count}"
            f" | H={max_horizon:.1f}s cond_n={cond_n}/{total_n}"
            f" | pos_med={pos_stats.get('median', float('nan')):.4f} m"
            f" | pos_p95={pos_stats.get('p95', float('nan')):.4f} m"
            f" | rot_med={rot_stats.get('median', float('nan')):.4f} rad"
            f" | completion={rates.get('completed_to_h', float('nan')):.3f}"
            f" | elapsed={format_wall_time(elapsed_seconds)}"
        ),
        quiet,
    )


def build_rollout_outcome_rows(rollout_outcomes, scenarios):
    scenario_names = [item.name if hasattr(item, "name") else str(item) for item in scenarios]

    return [
        {
            "scenario": "ALL",
            "n_trajectories": rollout_outcomes["overall"]["n_trajectories"],
            **rollout_outcomes["overall"]["counts"],
            **{
                f"{key}_rate": value
                for key, value in rollout_outcomes["overall"]["rates"].items()
            },
        },
        *[
            {
                "scenario": scenario_name,
                "n_trajectories": rollout_outcomes["by_scenario"][scenario_name]["n_trajectories"],
                **rollout_outcomes["by_scenario"][scenario_name]["counts"],
                **{
                    f"{key}_rate": value
                    for key, value in rollout_outcomes["by_scenario"][scenario_name]["rates"].items()
                },
            }
            for scenario_name in scenario_names
        ],
    ]


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset path; defaults to the path stored in the checkpoint config",
    )
    parser.add_argument(
        "--num_traj_per_scenario",
        type=int,
        default=30,
        help="Number of rollout trajectories per scenario",
    )
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[10.0, 30.0, 60.0],
        help="Evaluation horizons in seconds",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["PRBS", "CHIRP", "OU"],
        help="Scenario names to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", type=str, default=None, help="torch device")
    parser.add_argument(
        "--mode",
        type=str,
        default="heldout",
        choices=["heldout", "resampled"],
        help="Use held-out test initial conditions or resampled benchmark trajectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="rollout_benchmark_results",
        help="Base directory for benchmark result folders",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional name for the result subdirectory",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-trajectory progress output",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=5,
        help="Print progress for every N trajectories per scenario",
    )
    parser.add_argument(
        "--num_diagnostic_plots",
        type=int,
        default=6,
        help="Maximum number of per-case diagnostic plots to export",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    horizons = sorted({float(value) for value in args.times})
    max_rollout_time = max(horizons)
    scenarios = parse_scenarios(args.scenarios)

    try:
        ensure_runtime_imports()
    except ModuleNotFoundError as exc:
        parser.exit(
            1,
            "Missing runtime dependency for evaluation: "
            f"{exc}. Use the Python environment with the training/eval packages installed.\n",
        )

    torch = get_torch()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = resolve_output_dir(
        base_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        run_name=args.run_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_cache_dir = Path(args.output_dir) / "_gt_cache"

    model, train_cfg = build_model(args.checkpoint, device)
    dataset_path = resolve_dataset_path(train_cfg, args.dataset)
    dataset = load_dataset_artifact(dataset_path) if dataset_path and dataset_path.exists() else None
    generation_config, generation_config_source = resolve_generation_config_payload(
        train_cfg=train_cfg,
        dataset=dataset,
        dataset_path=dataset_path,
    )
    if generation_config is None:
        parser.exit(
            1,
            "Unable to resolve the dataset generation config. Re-train with the updated "
            "pipeline or supply a dataset that stores generation_config metadata.\n",
        )

    sim_cfg = build_sim_config(
        max_rollout_time,
        generation_config=generation_config,
        ocean_current=getattr(train_cfg, 'ocean_current', False),
    )

    if args.mode == "heldout":
        if dataset is None:
            parser.exit(
                1,
                "Heldout benchmark mode requires a dataset path or a checkpoint with dataset_path.\n",
            )
        trajectory_specs = build_heldout_trajectory_specs(
            dataset=dataset,
            scenarios=scenarios,
            num_traj_per_scenario=args.num_traj_per_scenario,
        )
    else:
        trajectory_specs = build_resampled_trajectory_specs(
            scenarios=scenarios,
            num_traj_per_scenario=args.num_traj_per_scenario,
            seed=args.seed,
            sim_cfg=sim_cfg,
        )

    evaluations = []
    horizon_rows = []
    specs_by_scenario = {
        scenario.name: [spec for spec in trajectory_specs if spec.scenario_type == scenario]
        for scenario in scenarios
    }
    total_traj = len(trajectory_specs)
    completed_traj = 0
    eval_start = time.perf_counter()

    progress_print(
        (
            f"Starting rollout benchmark | checkpoint={args.checkpoint} | device={device}"
            f" | mode={args.mode}"
            f" | scenarios={','.join(s.name for s in scenarios)}"
            f" | trajectories={total_traj} | max_horizon={max_rollout_time:.1f}s"
            f" | output_dir={output_dir}"
        ),
        args.quiet,
    )

    for scenario_idx, scenario_type in enumerate(scenarios):
        scenario_horizon_rows = []
        scenario_specs = specs_by_scenario[scenario_type.name]
        progress_print(
            f"[scenario {scenario_idx + 1}/{len(scenarios)}] {scenario_type.name} | "
            f"{len(scenario_specs)} trajectories",
            args.quiet,
        )
        scenario_rollouts = rollout_trajectory_batch(
            model=model,
            train_cfg=train_cfg,
            sim_cfg=sim_cfg,
            specs=scenario_specs,
            device=device,
            ground_truth_cache_dir=ground_truth_cache_dir if args.mode == "resampled" else None,
        )
        for traj_idx, (spec, rollout) in enumerate(zip(scenario_specs, scenario_rollouts)):
            init_seed = int(spec.seed)
            evaluation = evaluate_rollout(rollout, horizons, sim_cfg)

            horizon_rows.extend(evaluation.horizon_rows)
            scenario_horizon_rows.extend(evaluation.horizon_rows)
            evaluations.append(evaluation)
            completed_traj += 1

            should_log = (
                not args.quiet
                and (
                    args.progress_every <= 1
                    or traj_idx == 0
                    or traj_idx == len(scenario_specs) - 1
                    or (traj_idx + 1) % args.progress_every == 0
                )
            )
            if should_log:
                elapsed = time.perf_counter() - eval_start
                avg_elapsed = elapsed / completed_traj
                eta_seconds = avg_elapsed * (total_traj - completed_traj)
                progress_print(
                    (
                        f"[{completed_traj}/{total_traj}] {scenario_type.name} seed={init_seed} "
                        f"done | reason={rollout.failure_reason} | sim_time={rollout.completed_time:.2f}s"
                        f" | eta={format_wall_time(eta_seconds)}"
                    ),
                    args.quiet,
                )

        _, scenario_summary_json = aggregate_horizon_rows(scenario_horizon_rows)
        scenario_outcomes = summarize_rollout_outcomes(
            scenario_horizon_rows,
            [scenario_type],
            max_horizon=max_rollout_time,
        )
        print_scenario_progress_summary(
            scenario_name=scenario_type.name,
            horizon_record=scenario_summary_json["overall"].get(str(max_rollout_time), {}),
            outcome_record=scenario_outcomes["overall"],
            max_horizon=max_rollout_time,
            elapsed_seconds=time.perf_counter() - eval_start,
            quiet=args.quiet,
        )

    summary_rows, summary_json = aggregate_horizon_rows(horizon_rows)
    rollout_outcomes = summarize_rollout_outcomes(
        horizon_rows,
        scenarios,
        max_horizon=max_rollout_time,
    )
    diagnostic_cases = build_diagnostic_cases(
        horizon_rows,
        max_horizon=max_rollout_time,
        top_k=5,
    )
    diagnostic_plot_records = export_diagnostic_rollout_plots(
        evaluations=evaluations,
        diagnostic_cases=diagnostic_cases,
        config=sim_cfg,
        output_dir=output_dir / "diagnostic_plots",
        max_plots=max(0, int(args.num_diagnostic_plots)),
        select_diagnostic_plot_cases=select_diagnostic_plot_cases,
    )
    time_series_rows = summarize_time_series(
        evaluations,
        scenarios,
        max_time=max_rollout_time,
        dt_state=sim_cfg.dt_state,
    )
    rollout_outcome_rows = build_rollout_outcome_rows(rollout_outcomes, scenarios)

    summary_json["config"] = {
        "checkpoint": args.checkpoint,
        "dataset": str(dataset_path) if dataset_path else None,
        "device": str(device),
        "mode": args.mode,
        "generation_config_source": generation_config_source,
        "num_traj_per_scenario": args.num_traj_per_scenario,
        "horizons_s": horizons,
        "scenarios": [scenario.name for scenario in scenarios],
        "seed": args.seed,
        "max_horizon_s": max_rollout_time,
        "output_dir": str(output_dir),
        "run_name": args.run_name,
    }
    summary_json["rollout_outcomes"] = rollout_outcomes
    summary_json["metric_contract_file"] = "metric_contract.csv"
    summary_json["diagnostic_cases"] = {
        "earliest_failures": diagnostic_cases["earliest_failures"],
        "largest_terminal_errors": diagnostic_cases["largest_terminal_errors"],
        "velocity_violations": diagnostic_cases["velocity_violations"],
    }
    summary_json["diagnostic_plots"] = diagnostic_plot_records

    write_csv(output_dir / "trajectory_metrics.csv", horizon_rows)
    write_csv(output_dir / "horizon_metrics.csv", summary_rows)
    write_csv(output_dir / "time_series_metrics.csv", time_series_rows)
    write_csv(output_dir / "diagnostic_cases.csv", diagnostic_cases["all_cases"])
    write_csv(output_dir / "diagnostic_plots.csv", diagnostic_plot_records)
    write_csv(output_dir / "rollout_outcomes.csv", rollout_outcome_rows)
    write_metric_contract(
        output_dir / "metric_contract.csv",
        {
            "trajectory_metrics.csv": horizon_rows,
            "horizon_metrics.csv": summary_rows,
            "time_series_metrics.csv": time_series_rows,
            "diagnostic_cases.csv": diagnostic_cases["all_cases"],
            "diagnostic_plots.csv": diagnostic_plot_records,
            "rollout_outcomes.csv": rollout_outcome_rows,
        },
    )

    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary_json, handle, indent=2)
    write_summary_report(output_dir / "summary.txt", summary_json)

    plot_error_growth(time_series_rows, scenarios, output_dir / "error_growth.png")
    plot_terminal_error_boxplots(
        horizon_rows,
        scenarios,
        horizons,
        output_dir / "terminal_error_boxplots.png",
    )
    examples = select_example_rollouts(evaluations, max_horizon=max_rollout_time)
    plot_example_rollouts(examples, output_dir / "example_rollouts.png")

    summary_path = output_dir / "summary.txt"
    if summary_path.exists():
        progress_print(
            f"Benchmark finished in {format_wall_time(time.perf_counter() - eval_start)}",
            args.quiet,
        )
        progress_print(f"Results written to {summary_path.parent}", args.quiet)
        print(summary_path.read_text(), end="")


if __name__ == "__main__":
    main()
