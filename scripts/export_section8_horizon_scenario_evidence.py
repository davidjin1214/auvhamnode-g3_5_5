#!/usr/bin/env python3
"""Export §8 Tier-2 evidence: horizon-resolved (10/30/60 s) and per-scenario
(PRBS/CHIRP/OU) final-position-error tables, under the SAME provenance and
selection gate as ``export_section8_t2_evidence.py``.

Why a sibling script: the canonical ``aggregate.csv`` / ``per_seed_long.csv``
that the §8 main tables cite are 60 s-overall only. The underlying
``summary.json`` files already carry ``overall.{10,30,60}`` and
``by_scenario.{PRBS,CHIRP,OU}.{10,30,60}`` buckets, so the time-growth and
scenario views can be extracted from the IDENTICAL run set with the IDENTICAL
seed-survival gate -- no catalog reconciliation, no stale-mirror risk.

Selection gate (reused verbatim from the canonical exporter):
  * B1 training anomaly: a seed with the ``no successful training batches``
    signature (nbad > 0) is excluded (clean ``ablate_no_lift`` seed43).
  * Rollout divergence: a seed whose **60 s overall** position-error median is
    missing / NaN / > ROLLOUT_COLLAPSE_THRESHOLD_M is excluded. The 60 s gate
    flag is applied to ALL horizons/scenarios of that seed, so the surviving
    seed set is identical across horizons and the 60 s-overall aggregate
    reproduces the canonical ``aggregate.csv`` exactly (provenance check).

A model that diverges on every seed (``blackbox_fullstate`` clean) reports NO
finite median at any horizon -- recorded as a long-horizon stability failure.

Outputs (written next to the canonical tables):
  * ``horizon_scenario_per_seed.csv``  -- one row per (model, train_protocol,
    seed, eval_profile, eval_protocol, scope, horizon_s).
  * ``horizon_scenario_aggregate.csv`` -- cross-seed aggregate per
    (model, train_protocol, eval_profile, eval_protocol, scope, horizon_s).
``scope`` is ``overall`` or one of ``PRBS`` / ``CHIRP`` / ``OU``.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the canonical exporter's discovery + gate primitives verbatim so the
# surviving-seed set is guaranteed identical to the §8 main tables.
import json  # noqa: E402

from export_section8_t2_evidence import (  # noqa: E402
    MODELS,
    ROLLOUT_COLLAPSE_THRESHOLD_M,
    TRAIN_PROTOCOLS,
    count_no_successful_batches,
    discover_summaries,
    extract_seed,
    is_diverged,
    run_dir_from_summary,
)

DEFAULT_OUT = REPO_ROOT / "analysis" / "section8_current_evidence"
HORIZONS = ("10.0", "30.0", "60.0")
GATE_HORIZON = "60.0"  # seed survival is decided on the 60 s overall median
SCENARIOS = ("PRBS", "CHIRP", "OU")


def _node(summary: dict, scope: str, horizon: str) -> dict:
    """Return the metrics bucket for a (scope, horizon) pair, or {}."""
    if scope == "overall":
        return summary.get("overall", {}).get(horizon, {})
    return summary.get("by_scenario", {}).get(scope, {}).get(horizon, {})


def _metric(node: dict, name: str, stat: str) -> float | None:
    m = node.get("metrics", {}).get(name)
    if not m:
        return None
    return m.get(stat)


def _completed(node: dict) -> float | None:
    return node.get("rates", {}).get("completed")


def _gate_pos_med(summary: dict) -> float | None:
    """60 s overall position-error median -- the value the divergence gate uses."""
    return _metric(_node(summary, "overall", GATE_HORIZON), "final_position_error", "median")


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for model in MODELS:
        for proto in TRAIN_PROTOCOLS:
            for path in discover_summaries(model, proto):
                summary = json.load(path.open())
                cfg = summary.get("config", {})
                eval_profile = cfg.get("noise_profile", "")
                eval_protocol = cfg.get("noise_protocol", "")
                seed = extract_seed(path)
                nbad = count_no_successful_batches(run_dir_from_summary(path))
                gate_pos = _gate_pos_med(summary)
                diverged = int(is_diverged(gate_pos))
                for scope in ("overall",) + SCENARIOS:
                    for horizon in HORIZONS:
                        node = _node(summary, scope, horizon)
                        rows.append(
                            {
                                "model_type": model,
                                "train_protocol": proto,
                                "seed": seed,
                                "eval_profile": eval_profile,
                                "eval_protocol": eval_protocol,
                                "scope": scope,
                                "horizon_s": horizon,
                                "pos_err_median": _metric(node, "final_position_error", "median"),
                                "pos_err_p95": _metric(node, "final_position_error", "p95"),
                                "rot_geodesic_median": _metric(
                                    node, "final_rotation_geodesic", "median"
                                ),
                                "completion": _completed(node),
                                "n_traj": node.get("n_trajectories"),
                                "gate_pos_med_60s": gate_pos,
                                "train_nbad": nbad,
                                "train_anomaly": int(nbad > 0),
                                "rollout_diverged": diverged,
                                "source": str(path.relative_to(REPO_ROOT)),
                            }
                        )
    rows.sort(
        key=lambda r: (
            r["model_type"],
            r["train_protocol"],
            r["eval_protocol"],
            r["eval_profile"],
            r["scope"],
            float(r["horizon_s"]),
            r["seed"],
        )
    )
    return rows


def _finite(values: list) -> list[float]:
    return [
        v
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (
            r["model_type"],
            r["train_protocol"],
            r["eval_profile"],
            r["eval_protocol"],
            r["scope"],
            r["horizon_s"],
        )
        groups.setdefault(key, []).append(r)

    agg: list[dict] = []
    for key, members in groups.items():
        train_excluded = [m for m in members if m["train_anomaly"]]
        survivors = [m for m in members if not m["train_anomaly"]]
        diverged = [m for m in survivors if m["rollout_diverged"]]
        used = [m for m in survivors if not m["rollout_diverged"]]

        row = {
            "model_type": key[0],
            "train_protocol": key[1],
            "eval_profile": key[2],
            "eval_protocol": key[3],
            "scope": key[4],
            "horizon_s": key[5],
            "n_seeds_total": len(members),
            "n_anomaly_excluded": len(train_excluded),
            "n_rollout_diverged": len(diverged),
            "n_seeds_used": len(used),
        }
        medians = _finite([m["pos_err_median"] for m in used])
        p95s = _finite([m["pos_err_p95"] for m in used])
        completions = _finite([m["completion"] for m in used])
        row.update(
            {
                "posmed_mean_of_seed_medians": round(statistics.mean(medians), 4)
                if medians
                else None,
                "posmed_median_of_seed_medians": round(statistics.median(medians), 4)
                if medians
                else None,
                "posmed_min": round(min(medians), 4) if medians else None,
                "posmed_max": round(max(medians), 4) if medians else None,
                "posp95_mean_of_seed_p95s": round(statistics.mean(p95s), 4) if p95s else None,
                "posp95_median_of_seed_p95s": round(statistics.median(p95s), 4) if p95s else None,
                "posp95_max": round(max(p95s), 4) if p95s else None,
                "completion_mean": round(statistics.mean(completions), 4)
                if completions
                else None,
                "excluded_seeds": ";".join(str(m["seed"]) for m in train_excluded),
                "diverged_seeds": ";".join(str(m["seed"]) for m in diverged),
            }
        )
        agg.append(row)
    agg.sort(
        key=lambda r: (
            r["model_type"],
            r["train_protocol"],
            r["eval_protocol"],
            r["eval_profile"],
            r["scope"],
            float(r["horizon_s"]),
        )
    )
    return agg


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = collect_rows()
    agg = aggregate(rows)
    write_csv(args.output_dir / "horizon_scenario_per_seed.csv", rows)
    write_csv(args.output_dir / "horizon_scenario_aggregate.csv", agg)
    print(f"per-seed rows: {len(rows)}  |  aggregate rows: {len(agg)}")
    print(f"written to {args.output_dir.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
