from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from rollout_benchmark_engine import get_plot_module


def format_wall_time(seconds):
    seconds = max(0.0, float(seconds))
    minutes, seconds = divmod(seconds, 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1.0:
        return f"{int(hours)}h{int(minutes):02d}m{seconds:04.1f}s"
    if minutes >= 1.0:
        return f"{int(minutes)}m{seconds:04.1f}s"
    return f"{seconds:.1f}s"


def progress_print(message, quiet):
    if not quiet:
        print(message, flush=True)


def sanitize_name(text):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def resolve_output_dir(base_dir, checkpoint_path, run_name=None):
    base_dir = Path(base_dir)
    checkpoint_path = Path(checkpoint_path)

    if run_name:
        label = sanitize_name(run_name)
    else:
        parts = []
        parent_name = checkpoint_path.parent.name.strip()
        stem_name = checkpoint_path.stem.strip()
        if parent_name and parent_name not in (".", ".."):
            parts.append(parent_name)
        if stem_name:
            parts.append(stem_name)
        label = sanitize_name("_".join(parts) if parts else "benchmark")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"{label}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{label}_{timestamp}_{suffix:02d}"
        suffix += 1
    return candidate


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_optional(value, fmt):
    return format(value, fmt) if np.isfinite(value) else "nan"


def _field_contract(field):
    metadata_fields = {
        "scenario",
        "trajectory_id",
        "seed",
        "horizon_s",
        "eta0_z",
        "nu0_u",
        "n_trajectories",
        "time_s",
        "path",
    }
    rollout_status_fields = {
        "completed_time",
        "failure_reason",
        "failed_by_h",
        "fatal_failure",
        "gt_failed_by_h",
        "model_failed_by_h",
        "gt_divergence_by_h",
        "pred_divergence_by_h",
        "solver_failure_by_h",
        "nan_or_inf_by_h",
        "gt_rollout_failure",
        "model_rollout_failure",
        "completed_to_h",
        "n_alive",
        "survival_rate",
        "completed",
        "gt_divergence",
        "pred_divergence",
        "solver_failure",
        "nan_or_inf",
        "other_failure",
        "diagnostic_label",
    }

    def spec(group, scope, description):
        return {
            "group": group,
            "scope": scope,
            "description": description,
        }

    if field in metadata_fields:
        descriptions = {
            "scenario": "Scenario label for the rollout case or aggregate.",
            "trajectory_id": "Stable trajectory identifier built from scenario and seed.",
            "seed": "Random seed or trajectory seed.",
            "horizon_s": "Evaluation horizon in seconds.",
            "eta0_z": "Initial depth component used to initialize the rollout.",
            "nu0_u": "Initial surge velocity used to initialize the rollout.",
            "n_trajectories": "Number of trajectories in the aggregate.",
            "time_s": "Time coordinate for time-series aggregates.",
            "path": "Output path for an exported artifact.",
        }
        return spec("metadata", "n/a", descriptions[field])

    if (
        field in rollout_status_fields
        or field.endswith("_rate")
        or "completed_time" in field
        or field.endswith("_failure")
    ):
        return spec(
            "rollout_status",
            "n/a",
            "Rollout completion, failure, or constraint-rate summary field.",
        )

    if "relative_" in field:
        return spec(
            "model_space_diagnostic",
            "model_space",
            "Diagnostic error or violation computed in relative-velocity coordinates nu_r.",
        )

    if "energy" in field or "so3_" in field or "any_so3_violation" in field:
        return spec(
            "model_internal_diagnostic",
            "internal",
            "Model-internal consistency or stability diagnostic.",
        )

    if "rotation_geodesic" in field:
        return spec(
            "observable_space_kpi",
            "shared_state",
            "Primary orientation error metric on the shared rotation state.",
        )

    if "position" in field or "depth" in field or "total_" in field:
        return spec(
            "observable_space_kpi",
            "observable_space",
            "Primary rollout KPI in observable/data space.",
        )

    return spec(
        "unclassified",
        "n/a",
        "Field not matched by the benchmark metric contract rules.",
    )


def write_metric_contract(path, table_rows_map):
    rows = []
    for table_name, table_rows in table_rows_map.items():
        if not table_rows:
            continue
        for field in table_rows[0].keys():
            contract = _field_contract(field)
            rows.append(
                {
                    "table": table_name,
                    "field": field,
                    "group": contract["group"],
                    "scope": contract["scope"],
                    "description": contract["description"],
                }
            )

    if not rows:
        return

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["table", "field", "group", "scope", "description"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_report(path, summary_json):
    def horizon_line(prefix, horizon, record):
        metrics = record.get("metrics", {})
        rates = record.get("rates", {})
        pos_stats = metrics.get("final_position_error", {})
        rot_stats = metrics.get("final_rotation_geodesic", {})
        total_vel_stats = metrics.get("final_total_linear_velocity_error", {})
        rel_vel_stats = metrics.get("final_relative_linear_velocity_error", {})
        valid_count = int(pos_stats.get("count", 0))
        total_count = int(record.get("n_trajectories", 0))
        return (
            f"{prefix}H={horizon:.1f}s | pos median={pos_stats.get('median', float('nan')):.4f} m"
            f" | pos p95={pos_stats.get('p95', float('nan')):.4f} m"
            f" | rot median={rot_stats.get('median', float('nan')):.4f} rad"
            f" | total vel median={total_vel_stats.get('median', float('nan')):.4f} m/s"
            f" | rel vel median={rel_vel_stats.get('median', float('nan')):.4f} m/s"
            f" | cond_n={valid_count}/{total_count}"
            f" | completion={rates.get('completed_to_h', float('nan')):.3f}"
            f" | gt_fail_by_h={rates.get('gt_failed_by_h', float('nan')):.3f}"
            f" | model_fail_by_h={rates.get('model_failed_by_h', float('nan')):.3f}"
        )

    def outcome_line(prefix, record):
        counts = record.get("counts", {})
        rates = record.get("rates", {})
        return (
            f"{prefix}n={record.get('n_trajectories', 0)}"
            f" | completed={counts.get('completed', 0)} ({rates.get('completed', float('nan')):.3f})"
            f" | gt_divergence={counts.get('gt_divergence', 0)} ({rates.get('gt_divergence', float('nan')):.3f})"
            f" | pred_divergence={counts.get('pred_divergence', 0)} ({rates.get('pred_divergence', float('nan')):.3f})"
            f" | solver_failure={counts.get('solver_failure', 0)} ({rates.get('solver_failure', float('nan')):.3f})"
            f" | nan_or_inf={counts.get('nan_or_inf', 0)} ({rates.get('nan_or_inf', float('nan')):.3f})"
        )

    def diagnostic_line(prefix, case):
        return (
            f"{prefix}{case['scenario']} seed={case['seed']}"
            f" | reason={case['failure_reason']}"
            f" | sim_time={case['completed_time']:.2f}s"
            f" | pos={format_optional(case['final_position_error'], '.4f')} m"
            f" | rot={format_optional(case['final_rotation_geodesic'], '.4f')} rad"
            f" | total_vel_violation={int(case['any_total_velocity_violation_up_to_h'])}"
        )

    lines = []
    config = summary_json["config"]
    lines.append("Rollout Benchmark Summary")
    lines.append("=" * 80)
    lines.append(f"Checkpoint: {config['checkpoint']}")
    lines.append(f"Dataset: {config.get('dataset') or 'n/a'}")
    lines.append(f"Device: {config['device']}")
    lines.append(f"Mode: {config.get('mode', 'n/a')}")
    lines.append(f"Generation config source: {config.get('generation_config_source', 'n/a')}")
    lines.append(f"Scenarios: {', '.join(config['scenarios'])}")
    lines.append(f"Trajectories per scenario: {config['num_traj_per_scenario']}")
    lines.append(f"Horizons: {', '.join(str(value) for value in config['horizons_s'])}")
    lines.append("")

    lines.append("Overall")
    lines.append("-" * 80)
    lines.append("Conditional error statistics use only trajectories that completed to the horizon.")
    lines.append(
        "Primary KPIs are observable-space fields: position, depth, rotation, and total velocity."
    )
    lines.append(
        "Model-space diagnostics use relative velocity; internal diagnostics cover energy and raw SO(3) quality."
    )
    lines.append("Per-column field grouping is exported in metric_contract.csv.")
    for horizon in config["horizons_s"]:
        record = summary_json["overall"].get(str(horizon), {})
        lines.append(horizon_line("", horizon, record))
    lines.append("")

    lines.append("Rollout Outcomes To Max Horizon")
    lines.append("-" * 80)
    lines.append(outcome_line("", summary_json["rollout_outcomes"]["overall"]))
    lines.append("")

    lines.append("By Scenario")
    lines.append("-" * 80)
    for scenario in config["scenarios"]:
        lines.append(scenario)
        lines.append(outcome_line("  ", summary_json["rollout_outcomes"]["by_scenario"][scenario]))
        for horizon in config["horizons_s"]:
            record = summary_json["by_scenario"].get(scenario, {}).get(str(horizon), {})
            lines.append(horizon_line("  ", horizon, record))
        lines.append("")

    diagnostics = summary_json.get("diagnostic_cases", {})
    if diagnostics:
        lines.append("Diagnostic Cases To Max Horizon")
        lines.append("-" * 80)
        if diagnostics.get("earliest_failures"):
            lines.append("Earliest rollout failures")
            for case in diagnostics["earliest_failures"]:
                lines.append(diagnostic_line("  ", case))
            lines.append("")
        if diagnostics.get("largest_terminal_errors"):
            lines.append("Largest completed terminal errors")
            for case in diagnostics["largest_terminal_errors"]:
                lines.append(diagnostic_line("  ", case))
            lines.append("")
        if diagnostics.get("velocity_violations"):
            lines.append("Completed rollouts with velocity violations")
            for case in diagnostics["velocity_violations"]:
                lines.append(diagnostic_line("  ", case))
            lines.append("")

    diagnostic_plots = summary_json.get("diagnostic_plots", [])
    if diagnostic_plots:
        lines.append("Diagnostic Plot Files")
        lines.append("-" * 80)
        for plot in diagnostic_plots:
            lines.append(
                f"{plot['scenario']} seed={plot['seed']} | label={plot['diagnostic_label']} | {plot['path']}"
            )
        lines.append("")

    with open(path, "w") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_error_growth(rows, scenarios, output_path):
    plt = get_plot_module()
    if plt is None or not rows:
        return

    scenario_names = [scenario.name for scenario in scenarios]
    fig, axes = plt.subplots(
        2,
        len(scenario_names),
        figsize=(6 * len(scenario_names), 8),
        squeeze=False,
    )

    for col, scenario_name in enumerate(scenario_names):
        subset = [row for row in rows if row["scenario"] == scenario_name]
        if not subset:
            continue

        time = np.array([row["time_s"] for row in subset])

        ax_pos = axes[0, col]
        pos_med = np.array([row["position_error_median"] for row in subset])
        pos_p90 = np.array([row["position_error_p90"] for row in subset])
        ax_pos.plot(time, pos_med, color="tab:red", linewidth=2)
        ax_pos.fill_between(time, pos_med, pos_p90, color="tab:red", alpha=0.2)
        ax_pos.set_title(f"{scenario_name} Position Error")
        ax_pos.set_ylabel("m")
        ax_pos.grid(True)

        ax_rot = axes[1, col]
        rot_med = np.array([row["rotation_geodesic_median"] for row in subset])
        rot_p90 = np.array([row["rotation_geodesic_p90"] for row in subset])
        ax_rot.plot(time, rot_med, color="tab:blue", linewidth=2)
        ax_rot.fill_between(time, rot_med, rot_p90, color="tab:blue", alpha=0.2)
        ax_rot.set_title(f"{scenario_name} Rotation Error")
        ax_rot.set_xlabel("Time (s)")
        ax_rot.set_ylabel("rad")
        ax_rot.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_example_rollouts(evaluations, output_path):
    plt = get_plot_module()
    if plt is None or not evaluations:
        return

    fig = plt.figure(figsize=(6 * len(evaluations), 5))
    for idx, evaluation in enumerate(evaluations, start=1):
        rollout = evaluation.rollout
        ax = fig.add_subplot(1, len(evaluations), idx, projection="3d")
        ax.plot(
            rollout.gt_pos[:, 0],
            rollout.gt_pos[:, 1],
            -rollout.gt_pos[:, 2],
            "k-",
            linewidth=2,
            label="Ground Truth",
        )
        ax.plot(
            rollout.pred_pos[:, 0],
            rollout.pred_pos[:, 1],
            -rollout.pred_pos[:, 2],
            "r--",
            linewidth=2,
            label="Prediction",
        )
        ax.set_title(rollout.scenario)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("-Z")
        if idx == 1:
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def find_evaluation_by_case(evaluations, case):
    for evaluation in evaluations:
        rollout = evaluation.rollout
        if rollout.scenario == case["scenario"] and int(rollout.seed) == int(case["seed"]):
            return evaluation
    return None


def plot_diagnostic_rollout(evaluation, case, config, output_path):
    plt = get_plot_module()
    if plt is None:
        return False

    rollout = evaluation.rollout
    analysis = evaluation.analysis

    fig = plt.figure(figsize=(12, 8))
    title = (
        f"{case['scenario']} seed={case['seed']} | label={case['diagnostic_label']}"
        f" | reason={case['failure_reason']} | sim_time={case['completed_time']:.2f}s"
    )
    fig.suptitle(title)

    ax_traj = fig.add_subplot(2, 2, 1, projection="3d")
    ax_pos = fig.add_subplot(2, 2, 2)
    ax_rot = fig.add_subplot(2, 2, 3)
    ax_vel = fig.add_subplot(2, 2, 4)

    if len(rollout.time) > 0:
        ax_traj.plot(
            rollout.gt_pos[:, 0],
            rollout.gt_pos[:, 1],
            -rollout.gt_pos[:, 2],
            "k-",
            linewidth=2,
            label="Ground Truth",
        )
        ax_traj.plot(
            rollout.pred_pos[:, 0],
            rollout.pred_pos[:, 1],
            -rollout.pred_pos[:, 2],
            "r--",
            linewidth=2,
            label="Prediction",
        )
        ax_traj.set_xlabel("X")
        ax_traj.set_ylabel("Y")
        ax_traj.set_zlabel("-Z")
        ax_traj.legend()

        ax_pos.plot(
            rollout.time,
            analysis.position_error,
            color="tab:red",
            linewidth=2,
            label="Position",
        )
        ax_pos.plot(
            rollout.time,
            analysis.depth_error,
            color="tab:orange",
            linewidth=1.5,
            label="Depth",
        )
        ax_pos.set_title("Position Errors")
        ax_pos.set_xlabel("Time (s)")
        ax_pos.set_ylabel("m")
        ax_pos.grid(True)
        ax_pos.legend()

        ax_rot.plot(
            rollout.time,
            analysis.rotation_geodesic,
            color="tab:blue",
            linewidth=2,
        )
        ax_rot.set_title("Rotation Geodesic")
        ax_rot.set_xlabel("Time (s)")
        ax_rot.set_ylabel("rad")
        ax_rot.grid(True)

        ax_vel.plot(
            rollout.time,
            analysis.relative_linear_velocity_error,
            color="tab:green",
            linewidth=2,
            label="Relative linear vel",
        )
        ax_vel.plot(
            rollout.time,
            analysis.relative_angular_velocity_error,
            color="tab:purple",
            linewidth=1.5,
            label="Relative angular vel",
        )
        if np.any(analysis.total_velocity_violation):
            violation_times = rollout.time[analysis.total_velocity_violation]
            ax_vel.scatter(
                violation_times,
                analysis.total_linear_velocity_error[analysis.total_velocity_violation],
                color="tab:red",
                s=12,
                label="Total velocity violation",
                zorder=3,
            )
        ax_vel.set_title("Relative Velocity Errors")
        ax_vel.set_xlabel("Time (s)")
        ax_vel.set_ylabel("norm")
        ax_vel.grid(True)
        ax_vel.legend()
    else:
        ax_traj.axis("off")
        ax_pos.axis("off")
        ax_rot.axis("off")
        ax_vel.axis("off")
        fig.text(
            0.5,
            0.5,
            "No rollout states recorded.\nTrajectory failed before any evaluation samples were saved.",
            ha="center",
            va="center",
            fontsize=12,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return True


def export_diagnostic_rollout_plots(evaluations, diagnostic_cases, config, output_dir, max_plots, select_diagnostic_plot_cases):
    selected_cases = select_diagnostic_plot_cases(diagnostic_cases, max_plots=max_plots)
    if not selected_cases:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_records = []
    for case in selected_cases:
        evaluation = find_evaluation_by_case(evaluations, case)
        if evaluation is None:
            continue
        filename = (
            f"{sanitize_name(case['diagnostic_label'])}_"
            f"{sanitize_name(case['scenario'])}_seed{case['seed']}.png"
        )
        output_path = output_dir / filename
        if plot_diagnostic_rollout(evaluation, case, config, output_path):
            plot_records.append(
                {
                    "scenario": case["scenario"],
                    "seed": case["seed"],
                    "trajectory_id": case["trajectory_id"],
                    "diagnostic_label": case["diagnostic_label"],
                    "failure_reason": case["failure_reason"],
                    "path": str(output_path),
                }
            )
    return plot_records


def plot_terminal_error_boxplots(rows, scenarios, horizons, output_path):
    plt = get_plot_module()
    if plt is None or not rows:
        return

    filtered = [
        row for row in rows
        if row["completed_to_h"] == 1 and np.isfinite(row["final_position_error"])
    ]
    if not filtered:
        return

    scenario_names = [scenario.name for scenario in scenarios]
    fig, axes = plt.subplots(
        1,
        len(horizons),
        figsize=(5 * len(horizons), 5),
        squeeze=False,
    )

    for col, horizon in enumerate(horizons):
        ax = axes[0, col]
        data = []
        labels = []
        for scenario_name in scenario_names:
            values = [
                row["final_position_error"]
                for row in filtered
                if row["scenario"] == scenario_name and row["horizon_s"] == float(horizon)
            ]
            if values:
                data.append(values)
                labels.append(scenario_name)

        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"Final Position Error @ {horizon:.1f}s")
        ax.set_ylabel("m")
        ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
