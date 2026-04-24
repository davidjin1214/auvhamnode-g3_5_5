#!/usr/bin/env python3
"""Summarize a training/evaluation sweep into legacy and Phase-1 reporting tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

DEFAULT_PROFILE_PREFERENCE = (
    "nominal_eval",
    "clean",
    "degraded_eval",
    "heading_biased_eval",
    "current_bias_eval",
)
DEFAULT_HORIZONS = (10.0, 30.0, 60.0)

LEGACY_METRIC_KEYS = (
    "best_test_loss",
    "block_position_rmse_mean",
    "block_rotation_geodesic_mean",
    "block_velocity_rmse_mean",
    "block_angular_rmse_mean",
    "heldout_success_rate",
    "heldout_position_rmse_mean",
    "heldout_rotation_geodesic_mean",
    "heldout_velocity_rmse_mean",
    "heldout_angular_rmse_mean",
    "rollout_completion_rate",
    "rollout_model_failed_rate",
    "rollout_gt_failed_rate",
    "rollout_final_position_error_median",
    "rollout_final_position_error_p95",
    "rollout_final_rotation_geodesic_median",
    "rollout_final_total_linear_velocity_error_median",
)

PHASE1_NUMERIC_FIELDS = (
    *LEGACY_METRIC_KEYS,
    "horizon_s",
)

METRIC_SPECS_BY_SOURCE = {
    "block": {
        "block_position_rmse_mean": False,
        "block_rotation_geodesic_mean": False,
        "block_velocity_rmse_mean": False,
        "block_angular_rmse_mean": False,
    },
    "heldout": {
        "heldout_success_rate": True,
        "heldout_position_rmse_mean": False,
        "heldout_rotation_geodesic_mean": False,
        "heldout_velocity_rmse_mean": False,
        "heldout_angular_rmse_mean": False,
    },
    "rollout": {
        "rollout_completion_rate": True,
        "rollout_model_failed_rate": False,
        "rollout_gt_failed_rate": False,
        "rollout_final_position_error_median": False,
        "rollout_final_position_error_p95": False,
        "rollout_final_rotation_geodesic_median": False,
        "rollout_final_total_linear_velocity_error_median": False,
    },
}


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


def _horizon_group_key(value) -> float | None:
    return float(value) if _is_finite(value) else None


def _write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _base_noise_profile(profile: str) -> str:
    return {
        "heading_biased_eval": "nominal_eval",
        "current_bias_eval": "nominal_eval",
    }.get(profile, profile)


def resolve_noise_profile_from_config(config: Dict) -> str:
    resolved = config.get("resolved_noise_profile")
    if resolved:
        return str(resolved)
    profile = config.get("noise_profile")
    if profile:
        return str(profile)
    legacy_map = {
        0: "clean",
        1: "nominal_train",
        2: "nominal_eval",
        3: "degraded_eval",
    }
    raw_level = config.get("noise_level", 0)
    try:
        level = int(raw_level)
    except (TypeError, ValueError):
        level = 0
    return legacy_map.get(level, "clean")


def resolve_noise_protocol_for_profile(protocol: str | None, *, profile: str) -> str:
    profile_key = str(profile or "clean").strip().lower()
    if profile_key == "clean":
        return "clean"

    alias_map = {
        "": "iid_noisy_ic",
        "auto": "iid_noisy_ic",
        "default": "iid_noisy_ic",
        "iid": "iid_noisy_ic",
        "iid_noisy": "iid_noisy_ic",
        "iid_noisy_ic": "iid_noisy_ic",
        "block_iid": "iid_noisy_ic",
        "v4_lite": "v4_lite",
        "v4-lite": "v4_lite",
        "traj_consistent": "v4_lite",
        "trajectory_consistent": "v4_lite",
    }
    key = str(protocol or "").strip().lower()
    resolved = alias_map.get(key, key)
    if resolved not in {"iid_noisy_ic", "v4_lite"}:
        return "iid_noisy_ic"
    return resolved


def _train_protocol_from_config(config: Dict) -> str:
    resolved = config.get("resolved_noise_protocol")
    if resolved:
        return str(resolved)
    profile = resolve_noise_profile_from_config(config)
    return resolve_noise_protocol_for_profile(config.get("noise_protocol"), profile=profile)


def _protocol_label(protocol: str, profile: str) -> str:
    return "clean" if profile == "clean" or protocol == "clean" else protocol


def _sorted_profiles(profiles: Iterable[str]) -> List[str]:
    unique = sorted({str(profile) for profile in profiles})
    order = {name: idx for idx, name in enumerate(DEFAULT_PROFILE_PREFERENCE)}
    return sorted(unique, key=lambda name: (order.get(name, len(order)), name))


def _select_profile_payload(payload: Dict, profile: str | None) -> Dict:
    if not isinstance(payload, dict):
        return {}
    if "overall" in payload or "position_rmse" in payload:
        return payload
    if profile and isinstance(payload.get(profile), dict):
        return payload[profile]
    for candidate in DEFAULT_PROFILE_PREFERENCE:
        if isinstance(payload.get(candidate), dict):
            return payload[candidate]
    for value in payload.values():
        if isinstance(value, dict):
            return value
    return {}


def _profile_payloads(payload: Dict, selected_profile: str | None) -> Dict[str, Dict]:
    if not isinstance(payload, dict):
        return {}
    if "overall" in payload or "position_rmse" in payload:
        label = selected_profile or "clean"
        return {label: payload}
    if selected_profile:
        selected = payload.get(selected_profile)
        return {selected_profile: selected} if isinstance(selected, dict) else {}
    return {
        profile: profile_payload
        for profile, profile_payload in payload.items()
        if isinstance(profile_payload, dict)
    }


def _collect_rollout_profile_summaries(
    run_dir: Path,
    selected_profile: str | None = None,
) -> Dict[str, Dict]:
    rollout_root = run_dir / "rollout_benchmark"
    if not rollout_root.exists():
        return {}

    roots = sorted(
        [path for path in rollout_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    roots.append(rollout_root)

    profile_summaries: Dict[str, Dict] = {}

    for root in roots:
        summary_path = root / "summary.json"
        if summary_path.exists():
            summary = _read_json(summary_path)
            profile = _rollout_profile_from_summary(selected_profile or "clean", summary)
            if selected_profile and profile != selected_profile:
                continue
            protocol = _rollout_protocol_from_summary_payload(profile, summary)
            key = _rollout_summary_key(profile, protocol)
            profile_summaries.setdefault(
                key,
                {
                    "summary_path": str(summary_path),
                    "summary": summary,
                    "eval_profile": profile,
                    "eval_protocol": protocol,
                },
            )
        for child in sorted(root.iterdir()) if root.exists() else []:
            if not child.is_dir():
                continue
            child_summary = child / "summary.json"
            if not child_summary.exists():
                continue
            summary = _read_json(child_summary)
            profile = _rollout_profile_from_summary(child.name, summary)
            if selected_profile and profile != selected_profile:
                continue
            protocol = _rollout_protocol_from_summary_payload(profile, summary)
            key = _rollout_summary_key(profile, protocol)
            profile_summaries.setdefault(
                key,
                {
                    "summary_path": str(child_summary),
                    "summary": summary,
                    "eval_profile": profile,
                    "eval_protocol": protocol,
                },
            )

    profile_order = {name: idx for idx, name in enumerate(DEFAULT_PROFILE_PREFERENCE)}
    return {
        key: profile_summaries[key]
        for key in sorted(
            profile_summaries,
            key=lambda item: (
                profile_order.get(
                    profile_summaries[item]["eval_profile"],
                    len(profile_order),
                ),
                profile_summaries[item]["eval_profile"],
                profile_summaries[item]["eval_protocol"],
            ),
        )
    }


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


def _history_best_metrics(run_dir: Path) -> tuple[float, float]:
    history_path = run_dir / "training_history.pkl"
    if not history_path.exists():
        return float("nan"), float("nan")

    try:
        import pickle

        with open(history_path, "rb") as handle:
            history = pickle.load(handle)
        test_total = history.get("test_total", [])
        epochs = history.get("epoch", [])
        if not test_total:
            return float("nan"), float("nan")
        best_idx = min(range(len(test_total)), key=lambda idx: test_total[idx])
        best_test_loss = _safe_float(test_total[best_idx])
        best_epoch = _safe_float(epochs[best_idx]) if best_idx < len(epochs) else float("nan")
        return best_epoch, best_test_loss
    except Exception:
        return float("nan"), float("nan")


def collect_run_artifact(
    run: Dict,
    *,
    suite_dir: Path,
    block_profile: str | None = None,
    heldout_profile: str | None = None,
    rollout_profile: str | None = None,
) -> Dict:
    run_dir = _resolve_local_run_dir(suite_dir, run["run_dir"])
    checkpoint_path = _resolve_local_checkpoint_path(run_dir, run["checkpoint"])
    config_path = run_dir / "config.json"
    block_eval_path = run_dir / "block_evaluation.json"
    heldout_eval_path = run_dir / "heldout_evaluation.json"

    config = _read_json(config_path) if config_path.exists() else {}
    train_noise_profile = resolve_noise_profile_from_config(config)
    train_noise_protocol = _train_protocol_from_config(config)
    best_epoch, best_test_loss = _history_best_metrics(run_dir)

    block_profiles = (
        _profile_payloads(_read_json(block_eval_path), block_profile)
        if block_eval_path.exists() else {}
    )
    heldout_profiles = (
        _profile_payloads(_read_json(heldout_eval_path), heldout_profile)
        if heldout_eval_path.exists() else {}
    )
    rollout_profiles = _collect_rollout_profile_summaries(run_dir, selected_profile=rollout_profile)

    return {
        "suite_name": suite_dir.name,
        "group": run["group"],
        "model_type": run["model_type"],
        "seed": int(run["seed"]),
        "run_name": run["run_name"],
        "run_dir": str(run_dir),
        "checkpoint_exists": int(checkpoint_path.exists()),
        "config_exists": int(config_path.exists()),
        "block_eval_exists": int(block_eval_path.exists()),
        "heldout_eval_exists": int(heldout_eval_path.exists()),
        "rollout_summary_exists": int(bool(rollout_profiles)),
        "dataset_path": config.get("dataset_path", ""),
        "dataset_id": config.get("dataset_id", ""),
        "noise_reference": config.get("noise_reference", ""),
        "best_epoch": best_epoch,
        "best_test_loss": best_test_loss,
        "train_noise_profile": train_noise_profile,
        "train_noise_protocol": train_noise_protocol,
        "train_protocol_label": _protocol_label(train_noise_protocol, train_noise_profile),
        "block_profiles": block_profiles,
        "heldout_profiles": heldout_profiles,
        "rollout_profiles": rollout_profiles,
    }


def _common_row_fields(artifact: Dict) -> Dict:
    return {
        "suite_name": artifact["suite_name"],
        "group": artifact["group"],
        "model_type": artifact["model_type"],
        "seed": artifact["seed"],
        "run_name": artifact["run_name"],
        "run_dir": artifact["run_dir"],
        "dataset_path": artifact["dataset_path"],
        "dataset_id": artifact["dataset_id"],
        "noise_reference": artifact["noise_reference"],
        "best_epoch": artifact["best_epoch"],
        "best_test_loss": artifact["best_test_loss"],
        "train_noise_profile": artifact["train_noise_profile"],
        "train_noise_protocol": artifact["train_noise_protocol"],
        "train_protocol_label": artifact["train_protocol_label"],
        "block_position_rmse_mean": float("nan"),
        "block_rotation_geodesic_mean": float("nan"),
        "block_velocity_rmse_mean": float("nan"),
        "block_angular_rmse_mean": float("nan"),
        "heldout_success_rate": float("nan"),
        "heldout_position_rmse_mean": float("nan"),
        "heldout_rotation_geodesic_mean": float("nan"),
        "heldout_velocity_rmse_mean": float("nan"),
        "heldout_angular_rmse_mean": float("nan"),
        "rollout_completion_rate": float("nan"),
        "rollout_model_failed_rate": float("nan"),
        "rollout_gt_failed_rate": float("nan"),
        "rollout_final_position_error_median": float("nan"),
        "rollout_final_position_error_p95": float("nan"),
        "rollout_final_rotation_geodesic_median": float("nan"),
        "rollout_final_total_linear_velocity_error_median": float("nan"),
    }


def _profile_protocol_from_payload(profile: str, payload: Dict) -> str:
    if isinstance(payload, dict) and payload.get("eval_protocol"):
        return str(payload["eval_protocol"])
    noise_budget = payload.get("noise_budget") if isinstance(payload, dict) else None
    if isinstance(noise_budget, dict) and noise_budget.get("protocol"):
        return str(noise_budget["protocol"])
    return resolve_noise_protocol_for_profile(None, profile=profile)


def _rollout_profile_from_summary(fallback_profile: str, summary: Dict) -> str:
    profile = _safe_get(summary, "config", "noise_profile", default=None)
    return str(profile or fallback_profile)


def _rollout_protocol_from_summary_payload(profile: str, summary: Dict) -> str:
    protocol = _safe_get(summary, "config", "noise_protocol", default=None)
    if isinstance(protocol, str) and protocol:
        return protocol
    return resolve_noise_protocol_for_profile(None, profile=profile)


def _rollout_summary_key(profile: str, protocol: str) -> str:
    label = _protocol_label(protocol, profile)
    return "clean" if label == "clean" else f"{label}:{profile}"


def _eval_profile_from_payload(profile: str, payload: Dict) -> str:
    if isinstance(payload, dict) and payload.get("eval_profile"):
        return str(payload["eval_profile"])
    if ":" in str(profile):
        return str(profile).split(":", 1)[1]
    return str(profile)


def _rollout_protocol_from_summary(profile: str, payload: Dict) -> str:
    if isinstance(payload, dict) and payload.get("eval_protocol"):
        return str(payload["eval_protocol"])
    summary = payload.get("summary", {})
    protocol = _safe_get(summary, "config", "noise_protocol", default=None)
    if isinstance(protocol, str) and protocol:
        return protocol
    return resolve_noise_protocol_for_profile(None, profile=profile)


def build_phase1_by_seed_rows(
    artifacts: List[Dict],
    *,
    horizons: List[float],
) -> List[Dict]:
    rows: List[Dict] = []
    horizon_keys = [str(float(horizon)) for horizon in horizons]

    for artifact in artifacts:
        for profile, payload in artifact["block_profiles"].items():
            row = _common_row_fields(artifact)
            row.update(
                {
                    "source": "block",
                    "eval_profile": profile,
                    "eval_protocol": _profile_protocol_from_payload(profile, payload),
                    "eval_protocol_label": _protocol_label(
                        _profile_protocol_from_payload(profile, payload),
                        profile,
                    ),
                    "horizon_s": float("nan"),
                    "summary_path": str(Path(artifact["run_dir"]) / "block_evaluation.json"),
                }
            )
            row["block_position_rmse_mean"] = _safe_float(_safe_get(payload, "position_rmse", "mean"))
            row["block_rotation_geodesic_mean"] = _safe_float(
                _safe_get(payload, "rotation_geodesic", "mean")
            )
            row["block_velocity_rmse_mean"] = _safe_float(_safe_get(payload, "velocity_rmse", "mean"))
            row["block_angular_rmse_mean"] = _safe_float(_safe_get(payload, "angular_rmse", "mean"))
            rows.append(row)

        for profile, payload in artifact["heldout_profiles"].items():
            overall = payload.get("overall") if isinstance(payload, dict) else None
            overall = overall if isinstance(overall, dict) else payload
            protocol = _profile_protocol_from_payload(profile, payload)
            row = _common_row_fields(artifact)
            row.update(
                {
                    "source": "heldout",
                    "eval_profile": profile,
                    "eval_protocol": protocol,
                    "eval_protocol_label": _protocol_label(protocol, profile),
                    "horizon_s": float("nan"),
                    "summary_path": str(Path(artifact["run_dir"]) / "heldout_evaluation.json"),
                }
            )
            row["heldout_success_rate"] = _safe_float(_safe_get(overall, "success_rate"))
            row["heldout_position_rmse_mean"] = _safe_float(
                _safe_get(overall, "position_rmse", "mean")
            )
            row["heldout_rotation_geodesic_mean"] = _safe_float(
                _safe_get(overall, "rotation_geodesic", "mean")
            )
            row["heldout_velocity_rmse_mean"] = _safe_float(
                _safe_get(overall, "velocity_rmse", "mean")
            )
            row["heldout_angular_rmse_mean"] = _safe_float(
                _safe_get(overall, "angular_rmse", "mean")
            )
            rows.append(row)

        for profile_key, payload in artifact["rollout_profiles"].items():
            summary = payload["summary"]
            profile = _eval_profile_from_payload(profile_key, payload)
            protocol = _rollout_protocol_from_summary(profile, payload)
            for horizon, horizon_key in zip(horizons, horizon_keys):
                overall = _safe_get(summary, "overall", horizon_key, default={})
                row = _common_row_fields(artifact)
                row.update(
                    {
                        "source": "rollout",
                        "eval_profile": profile,
                        "eval_protocol": protocol,
                        "eval_protocol_label": _protocol_label(protocol, profile),
                        "horizon_s": float(horizon),
                        "summary_path": payload["summary_path"],
                    }
                )
                row["rollout_completion_rate"] = _safe_float(
                    _safe_get(overall, "rates", "completed_to_h")
                )
                row["rollout_model_failed_rate"] = _safe_float(
                    _safe_get(overall, "rates", "model_failed_by_h")
                )
                row["rollout_gt_failed_rate"] = _safe_float(
                    _safe_get(overall, "rates", "gt_failed_by_h")
                )
                row["rollout_final_position_error_median"] = _safe_float(
                    _safe_get(overall, "metrics", "final_position_error", "median")
                )
                row["rollout_final_position_error_p95"] = _safe_float(
                    _safe_get(overall, "metrics", "final_position_error", "p95")
                )
                row["rollout_final_rotation_geodesic_median"] = _safe_float(
                    _safe_get(overall, "metrics", "final_rotation_geodesic", "median")
                )
                row["rollout_final_total_linear_velocity_error_median"] = _safe_float(
                    _safe_get(
                        overall,
                        "metrics",
                        "final_total_linear_velocity_error",
                        "median",
                    )
                )
                rows.append(row)

    rows.sort(
        key=lambda row: (
            row["group"],
            row["model_type"],
            row["train_protocol_label"],
            row["seed"],
            row["source"],
            row["eval_profile"],
            math.inf if not _is_finite(row["horizon_s"]) else row["horizon_s"],
        )
    )
    return rows


def aggregate_phase1_model_rows(by_seed_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in by_seed_rows:
        grouped[
            (
                row["group"],
                row["model_type"],
                row["train_noise_profile"],
                row["train_noise_protocol"],
                row["train_protocol_label"],
                row["source"],
                row["eval_profile"],
                row["eval_protocol"],
                row["eval_protocol_label"],
                _horizon_group_key(row["horizon_s"]),
            )
        ].append(row)

    model_rows = []
    for key, items in sorted(grouped.items()):
        (
            group,
            model_type,
            train_noise_profile,
            train_noise_protocol,
            train_protocol_label,
            source,
            eval_profile,
            eval_protocol,
            eval_protocol_label,
            horizon_key,
        ) = key
        row = {
            "group": group,
            "model_type": model_type,
            "train_noise_profile": train_noise_profile,
            "train_noise_protocol": train_noise_protocol,
            "train_protocol_label": train_protocol_label,
            "source": source,
            "eval_profile": eval_profile,
            "eval_protocol": eval_protocol,
            "eval_protocol_label": eval_protocol_label,
            "horizon_s": float("nan") if horizon_key is None else horizon_key,
            "n_seeds": len(items),
            "seeds": ",".join(str(item["seed"]) for item in sorted(items, key=lambda row: row["seed"])),
        }
        for metric_name in LEGACY_METRIC_KEYS:
            stats = _stats(item[metric_name] for item in items)
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]
            row[f"{metric_name}_min"] = stats["min"]
            row[f"{metric_name}_max"] = stats["max"]
        model_rows.append(row)

    return model_rows


def build_phase1_by_scenario_rows(
    artifacts: List[Dict],
    *,
    horizons: List[float],
) -> List[Dict]:
    rows: List[Dict] = []
    horizon_keys = [str(float(horizon)) for horizon in horizons]

    for artifact in artifacts:
        common = _common_row_fields(artifact)
        for profile, payload in artifact["heldout_profiles"].items():
            protocol = _profile_protocol_from_payload(profile, payload)
            for scenario, scenario_payload in sorted(payload.get("by_scenario", {}).items()):
                row = dict(common)
                row.update(
                    {
                        "source": "heldout",
                        "scenario": scenario,
                        "eval_profile": profile,
                        "eval_protocol": protocol,
                        "eval_protocol_label": _protocol_label(protocol, profile),
                        "horizon_s": float("nan"),
                        "summary_path": str(Path(artifact["run_dir"]) / "heldout_evaluation.json"),
                    }
                )
                row["heldout_success_rate"] = _safe_float(_safe_get(scenario_payload, "success_rate"))
                row["heldout_position_rmse_mean"] = _safe_float(
                    _safe_get(scenario_payload, "position_rmse", "mean")
                )
                row["heldout_rotation_geodesic_mean"] = _safe_float(
                    _safe_get(scenario_payload, "rotation_geodesic", "mean")
                )
                row["heldout_velocity_rmse_mean"] = _safe_float(
                    _safe_get(scenario_payload, "velocity_rmse", "mean")
                )
                row["heldout_angular_rmse_mean"] = _safe_float(
                    _safe_get(scenario_payload, "angular_rmse", "mean")
                )
                rows.append(row)

        for profile_key, payload in artifact["rollout_profiles"].items():
            summary = payload["summary"]
            profile = _eval_profile_from_payload(profile_key, payload)
            protocol = _rollout_protocol_from_summary(profile, payload)
            by_scenario = summary.get("by_scenario", {})
            for scenario, scenario_payload in sorted(by_scenario.items()):
                for horizon, horizon_key in zip(horizons, horizon_keys):
                    overall = _safe_get(scenario_payload, horizon_key, default={})
                    row = dict(common)
                    row.update(
                        {
                            "source": "rollout",
                            "scenario": scenario,
                            "eval_profile": profile,
                            "eval_protocol": protocol,
                            "eval_protocol_label": _protocol_label(protocol, profile),
                            "horizon_s": float(horizon),
                            "summary_path": payload["summary_path"],
                        }
                    )
                    row["rollout_completion_rate"] = _safe_float(
                        _safe_get(overall, "rates", "completed_to_h")
                    )
                    row["rollout_model_failed_rate"] = _safe_float(
                        _safe_get(overall, "rates", "model_failed_by_h")
                    )
                    row["rollout_gt_failed_rate"] = _safe_float(
                        _safe_get(overall, "rates", "gt_failed_by_h")
                    )
                    row["rollout_final_position_error_median"] = _safe_float(
                        _safe_get(overall, "metrics", "final_position_error", "median")
                    )
                    row["rollout_final_position_error_p95"] = _safe_float(
                        _safe_get(overall, "metrics", "final_position_error", "p95")
                    )
                    row["rollout_final_rotation_geodesic_median"] = _safe_float(
                        _safe_get(overall, "metrics", "final_rotation_geodesic", "median")
                    )
                    row["rollout_final_total_linear_velocity_error_median"] = _safe_float(
                        _safe_get(
                            overall,
                            "metrics",
                            "final_total_linear_velocity_error",
                            "median",
                        )
                    )
                    rows.append(row)

    rows.sort(
        key=lambda row: (
            row["group"],
            row["model_type"],
            row["train_protocol_label"],
            row["seed"],
            row["source"],
            row["scenario"],
            row["eval_profile"],
            math.inf if not _is_finite(row["horizon_s"]) else row["horizon_s"],
        )
    )
    return rows


def _comparison_entry(value: float, clean_value: float, *, higher_is_better: bool = False) -> Dict:
    value = float(value)
    clean_value = float(clean_value)
    if not _is_finite(value) or not _is_finite(clean_value):
        return {
            "value": value,
            "clean_value": clean_value,
            "absolute_delta": float("nan"),
            "ratio_to_clean": float("nan"),
            "degradation_pct": float("nan"),
        }

    absolute_delta = value - clean_value
    if abs(clean_value) < 1e-12:
        ratio_to_clean = 1.0 if abs(value) < 1e-12 else float("inf")
        degradation_pct = 0.0 if abs(value) < 1e-12 else float("inf")
    else:
        ratio_to_clean = value / clean_value
        if higher_is_better:
            degradation_pct = (clean_value - value) / abs(clean_value) * 100.0
        else:
            degradation_pct = (value - clean_value) / abs(clean_value) * 100.0

    return {
        "value": value,
        "clean_value": clean_value,
        "absolute_delta": float(absolute_delta),
        "ratio_to_clean": float(ratio_to_clean),
        "degradation_pct": float(degradation_pct),
    }


def build_phase1_degradation_rows(by_seed_rows: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []

    by_run_eval = {
        (
            row["run_dir"],
            row["source"],
            row["eval_profile"],
            _horizon_group_key(row["horizon_s"]),
        ): row
        for row in by_seed_rows
    }

    for row in by_seed_rows:
        if row["eval_profile"] == "clean":
            continue
        baseline = by_run_eval.get(
            (
                row["run_dir"],
                row["source"],
                "clean",
                _horizon_group_key(row["horizon_s"]),
            )
        )
        if baseline is None:
            continue
        for metric_name, higher_is_better in METRIC_SPECS_BY_SOURCE[row["source"]].items():
            entry = _comparison_entry(
                row[metric_name],
                baseline[metric_name],
                higher_is_better=higher_is_better,
            )
            rows.append(
                {
                    "comparison_kind": "clean_to_noisy_eval",
                    "suite_name": row["suite_name"],
                    "group": row["group"],
                    "model_type": row["model_type"],
                    "seed": row["seed"],
                    "run_name": row["run_name"],
                    "run_dir": row["run_dir"],
                    "source": row["source"],
                    "metric_name": metric_name,
                    "horizon_s": row["horizon_s"],
                    "train_noise_profile": row["train_noise_profile"],
                    "train_noise_protocol": row["train_noise_protocol"],
                    "train_protocol_label": row["train_protocol_label"],
                    "baseline_train_protocol_label": row["train_protocol_label"],
                    "eval_profile": row["eval_profile"],
                    "baseline_eval_profile": "clean",
                    "eval_protocol": row["eval_protocol"],
                    "baseline_eval_protocol": baseline["eval_protocol"],
                    **entry,
                }
            )

    clean_training_rows = {
        (
            row["group"],
            row["model_type"],
            row["seed"],
            row["dataset_path"],
            row["source"],
            _horizon_group_key(row["horizon_s"]),
        ): row
        for row in by_seed_rows
        if row["eval_profile"] == "clean" and row["train_protocol_label"] == "clean"
    }

    for row in by_seed_rows:
        if row["eval_profile"] != "clean" or row["train_protocol_label"] == "clean":
            continue
        baseline = clean_training_rows.get(
            (
                row["group"],
                row["model_type"],
                row["seed"],
                row["dataset_path"],
                row["source"],
                _horizon_group_key(row["horizon_s"]),
            )
        )
        if baseline is None:
            continue
        for metric_name, higher_is_better in METRIC_SPECS_BY_SOURCE[row["source"]].items():
            entry = _comparison_entry(
                row[metric_name],
                baseline[metric_name],
                higher_is_better=higher_is_better,
            )
            rows.append(
                {
                    "comparison_kind": "clean_replay_cost",
                    "suite_name": row["suite_name"],
                    "group": row["group"],
                    "model_type": row["model_type"],
                    "seed": row["seed"],
                    "run_name": row["run_name"],
                    "run_dir": row["run_dir"],
                    "source": row["source"],
                    "metric_name": metric_name,
                    "horizon_s": row["horizon_s"],
                    "train_noise_profile": row["train_noise_profile"],
                    "train_noise_protocol": row["train_noise_protocol"],
                    "train_protocol_label": row["train_protocol_label"],
                    "baseline_train_protocol_label": baseline["train_protocol_label"],
                    "eval_profile": "clean",
                    "baseline_eval_profile": "clean",
                    "eval_protocol": row["eval_protocol"],
                    "baseline_eval_protocol": baseline["eval_protocol"],
                    **entry,
                }
            )

    rows.sort(
        key=lambda row: (
            row["comparison_kind"],
            row["group"],
            row["model_type"],
            row["train_protocol_label"],
            row["seed"],
            row["source"],
            row["metric_name"],
            math.inf if not _is_finite(row["horizon_s"]) else row["horizon_s"],
            row["eval_profile"],
        )
    )
    return rows


def build_phase1_bundle(
    *,
    suite_dir: Path,
    runs: List[Dict],
    horizons: List[float],
    block_profile: str | None = None,
    heldout_profile: str | None = None,
    rollout_profile: str | None = None,
) -> Dict:
    artifacts = [
        collect_run_artifact(
            run,
            suite_dir=suite_dir,
            block_profile=block_profile,
            heldout_profile=heldout_profile,
            rollout_profile=rollout_profile,
        )
        for run in runs
    ]
    by_seed_rows = build_phase1_by_seed_rows(artifacts, horizons=horizons)
    model_rows = aggregate_phase1_model_rows(by_seed_rows)
    by_scenario_rows = build_phase1_by_scenario_rows(artifacts, horizons=horizons)
    degradation_rows = build_phase1_degradation_rows(by_seed_rows)
    return {
        "artifacts": artifacts,
        "by_seed_rows": by_seed_rows,
        "model_rows": model_rows,
        "by_scenario_rows": by_scenario_rows,
        "degradation_rows": degradation_rows,
    }


def _primary_profile_name(profiles: Dict[str, Dict]) -> str | None:
    if not profiles:
        return None
    return _sorted_profiles(profiles.keys())[0]


def build_seed_row(
    run: Dict,
    suite_dir: Path,
    horizon_s: float,
    block_profile: str | None = None,
    heldout_profile: str | None = None,
    rollout_profile: str | None = None,
) -> Dict:
    artifact = collect_run_artifact(
        run,
        suite_dir=suite_dir,
        block_profile=block_profile,
        heldout_profile=heldout_profile,
        rollout_profile=rollout_profile,
    )
    row = {
        "group": artifact["group"],
        "model_type": artifact["model_type"],
        "seed": artifact["seed"],
        "run_name": artifact["run_name"],
        "run_dir": artifact["run_dir"],
        "checkpoint_exists": artifact["checkpoint_exists"],
        "config_exists": artifact["config_exists"],
        "block_eval_exists": artifact["block_eval_exists"],
        "heldout_eval_exists": artifact["heldout_eval_exists"],
        "rollout_summary_exists": artifact["rollout_summary_exists"],
        "dataset_path": artifact["dataset_path"],
        "best_epoch": artifact["best_epoch"],
        "best_test_loss": artifact["best_test_loss"],
        "train_noise_profile": artifact["train_noise_profile"],
        "train_noise_protocol": artifact["train_noise_protocol"],
        "train_protocol_label": artifact["train_protocol_label"],
        "block_position_rmse_mean": float("nan"),
        "block_rotation_geodesic_mean": float("nan"),
        "block_velocity_rmse_mean": float("nan"),
        "block_angular_rmse_mean": float("nan"),
        "heldout_success_rate": float("nan"),
        "heldout_position_rmse_mean": float("nan"),
        "heldout_rotation_geodesic_mean": float("nan"),
        "heldout_velocity_rmse_mean": float("nan"),
        "heldout_angular_rmse_mean": float("nan"),
        "rollout_horizon_s": horizon_s,
        "rollout_completion_rate": float("nan"),
        "rollout_model_failed_rate": float("nan"),
        "rollout_gt_failed_rate": float("nan"),
        "rollout_final_position_error_median": float("nan"),
        "rollout_final_position_error_p95": float("nan"),
        "rollout_final_rotation_geodesic_median": float("nan"),
        "rollout_final_total_linear_velocity_error_median": float("nan"),
    }

    block_name = _primary_profile_name(artifact["block_profiles"])
    if block_name is not None:
        payload = artifact["block_profiles"][block_name]
        row["block_position_rmse_mean"] = _safe_float(_safe_get(payload, "position_rmse", "mean"))
        row["block_rotation_geodesic_mean"] = _safe_float(
            _safe_get(payload, "rotation_geodesic", "mean")
        )
        row["block_velocity_rmse_mean"] = _safe_float(_safe_get(payload, "velocity_rmse", "mean"))
        row["block_angular_rmse_mean"] = _safe_float(_safe_get(payload, "angular_rmse", "mean"))

    heldout_name = _primary_profile_name(artifact["heldout_profiles"])
    if heldout_name is not None:
        payload = artifact["heldout_profiles"][heldout_name]
        overall = payload.get("overall") if isinstance(payload, dict) else None
        overall = overall if isinstance(overall, dict) else payload
        row["heldout_success_rate"] = _safe_float(_safe_get(overall, "success_rate"))
        row["heldout_position_rmse_mean"] = _safe_float(
            _safe_get(overall, "position_rmse", "mean")
        )
        row["heldout_rotation_geodesic_mean"] = _safe_float(
            _safe_get(overall, "rotation_geodesic", "mean")
        )
        row["heldout_velocity_rmse_mean"] = _safe_float(
            _safe_get(overall, "velocity_rmse", "mean")
        )
        row["heldout_angular_rmse_mean"] = _safe_float(
            _safe_get(overall, "angular_rmse", "mean")
        )

    rollout_name = _primary_profile_name(artifact["rollout_profiles"])
    if rollout_name is not None:
        payload = artifact["rollout_profiles"][rollout_name]
        overall = _safe_get(payload["summary"], "overall", str(float(horizon_s)), default={})
        row["rollout_completion_rate"] = _safe_float(_safe_get(overall, "rates", "completed_to_h"))
        row["rollout_model_failed_rate"] = _safe_float(
            _safe_get(overall, "rates", "model_failed_by_h")
        )
        row["rollout_gt_failed_rate"] = _safe_float(_safe_get(overall, "rates", "gt_failed_by_h"))
        row["rollout_final_position_error_median"] = _safe_float(
            _safe_get(overall, "metrics", "final_position_error", "median")
        )
        row["rollout_final_position_error_p95"] = _safe_float(
            _safe_get(overall, "metrics", "final_position_error", "p95")
        )
        row["rollout_final_rotation_geodesic_median"] = _safe_float(
            _safe_get(overall, "metrics", "final_rotation_geodesic", "median")
        )
        row["rollout_final_total_linear_velocity_error_median"] = _safe_float(
            _safe_get(overall, "metrics", "final_total_linear_velocity_error", "median")
        )

    return row


def aggregate_model_rows(seed_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in seed_rows:
        grouped[
            (
                row["group"],
                row["model_type"],
                row.get("train_noise_profile", "clean"),
                row.get("train_noise_protocol", "clean"),
                row.get("train_protocol_label", "clean"),
            )
        ].append(row)

    rows = []
    for key, items in sorted(grouped.items()):
        group, model_type, train_noise_profile, train_noise_protocol, train_protocol_label = key
        row = {
            "group": group,
            "model_type": model_type,
            "train_noise_profile": train_noise_profile,
            "train_noise_protocol": train_noise_protocol,
            "train_protocol_label": train_protocol_label,
            "n_seeds": len(items),
            "seeds": ",".join(str(item["seed"]) for item in sorted(items, key=lambda r: r["seed"])),
        }
        for key_name in LEGACY_METRIC_KEYS:
            stats = _stats(item[key_name] for item in items)
            row[f"{key_name}_mean"] = stats["mean"]
            row[f"{key_name}_std"] = stats["std"]
            row[f"{key_name}_min"] = stats["min"]
            row[f"{key_name}_max"] = stats["max"]
        rows.append(row)
    return rows


def write_text_summary(
    path: Path,
    suite_dir: Path,
    seed_rows: List[Dict],
    model_rows: List[Dict],
    horizon_s: float,
):
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
            f" | train={row['train_protocol_label']}"
            f" | seeds={row['seeds']}"
            f" | heldout pos={row['heldout_position_rmse_mean_mean']:.4f}"
            f" +- {row['heldout_position_rmse_mean_std']:.4f}"
            f" | rollout pos@H median={row['rollout_final_position_error_median_mean']:.4f}"
            f" +- {row['rollout_final_position_error_median_std']:.4f}"
            f" | completion@H={row['rollout_completion_rate_mean']:.3f}"
            f" +- {row['rollout_completion_rate_std']:.3f}"
        )
    lines.append("")

    lines.append("Seed Rows")
    lines.append("-" * 80)
    for row in sorted(
        seed_rows,
        key=lambda item: (
            item["group"],
            item["model_type"],
            item.get("train_protocol_label", "clean"),
            item["seed"],
        ),
    ):
        lines.append(
            f"{row['group']}/{row['model_type']} seed={row['seed']}"
            f" | train={row.get('train_protocol_label', 'clean')}"
            f" | heldout pos={row['heldout_position_rmse_mean']:.4f}"
            f" | rollout pos@H median={row['rollout_final_position_error_median']:.4f}"
            f" | completion@H={row['rollout_completion_rate']:.3f}"
        )

    with open(path, "w") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def _phase1_matrix_payload(
    *,
    suite_dir: Path,
    horizons: List[float],
    artifacts: List[Dict],
    block_profile: str | None,
    heldout_profile: str | None,
    rollout_profile: str | None,
) -> Dict:
    return {
        "suite_dir": str(suite_dir),
        "suite_name": suite_dir.name,
        "horizons_s": [float(horizon) for horizon in horizons],
        "requested_profiles": {
            "block_profile": block_profile,
            "heldout_profile": heldout_profile,
            "rollout_profile": rollout_profile,
        },
        "train_protocol_labels": _sorted_profiles(
            artifact["train_protocol_label"] for artifact in artifacts
        ),
        "train_noise_profiles": _sorted_profiles(
            artifact["train_noise_profile"] for artifact in artifacts
        ),
        "available_eval_profiles": {
            "block": _sorted_profiles(
                profile
                for artifact in artifacts
                for profile in artifact["block_profiles"].keys()
            ),
            "heldout": _sorted_profiles(
                profile
                for artifact in artifacts
                for profile in artifact["heldout_profiles"].keys()
            ),
            "rollout": _sorted_profiles(
                (
                    profile
                    if profile == "clean" else
                    f"{payload.get('eval_protocol', 'unknown')}:{payload.get('eval_profile', profile)}"
                )
                for artifact in artifacts
                for profile, payload in artifact["rollout_profiles"].items()
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=str, required=True, help="Sweep directory under checkpoints/")
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
        help="Legacy rollout horizon in seconds used by sweep_seed_metrics.csv. Default: 60",
    )
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Phase-1 rollout horizons to export. Default: 10 30 60",
    )
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    runs = load_runs(suite_dir)
    horizons = [float(horizon) for horizon in args.horizons]
    bundle = build_phase1_bundle(
        suite_dir=suite_dir,
        runs=runs,
        horizons=horizons,
        block_profile=args.block_profile,
        heldout_profile=args.heldout_profile,
        rollout_profile=args.rollout_profile,
    )

    legacy_seed_rows = [
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
    legacy_model_rows = aggregate_model_rows(legacy_seed_rows)

    _write_csv(suite_dir / "sweep_seed_metrics.csv", legacy_seed_rows)
    _write_csv(suite_dir / "sweep_model_metrics.csv", legacy_model_rows)
    _write_csv(suite_dir / "phase1_by_seed.csv", bundle["by_seed_rows"])
    _write_csv(suite_dir / "phase1_summary.csv", bundle["model_rows"])
    _write_csv(suite_dir / "phase1_by_scenario.csv", bundle["by_scenario_rows"])
    _write_csv(suite_dir / "phase1_degradation.csv", bundle["degradation_rows"])

    summary_json = {
        "suite_dir": str(suite_dir),
        "legacy_horizon_s": float(args.horizon),
        "phase1_horizons_s": horizons,
        "n_runs": len(legacy_seed_rows),
        "seed_rows": legacy_seed_rows,
        "model_rows": legacy_model_rows,
        "phase1_files": {
            "matrix": "phase1_matrix.json",
            "summary": "phase1_summary.csv",
            "by_seed": "phase1_by_seed.csv",
            "by_scenario": "phase1_by_scenario.csv",
            "degradation": "phase1_degradation.csv",
        },
    }
    with open(suite_dir / "sweep_summary.json", "w") as handle:
        json.dump(summary_json, handle, indent=2)
    with open(suite_dir / "phase1_matrix.json", "w") as handle:
        json.dump(
            _phase1_matrix_payload(
                suite_dir=suite_dir,
                horizons=horizons,
                artifacts=bundle["artifacts"],
                block_profile=args.block_profile,
                heldout_profile=args.heldout_profile,
                rollout_profile=args.rollout_profile,
            ),
            handle,
            indent=2,
        )
    write_text_summary(
        suite_dir / "sweep_summary.txt",
        suite_dir=suite_dir,
        seed_rows=legacy_seed_rows,
        model_rows=legacy_model_rows,
        horizon_s=float(args.horizon),
    )


if __name__ == "__main__":
    main()
