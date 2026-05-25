#!/usr/bin/env python3
"""Export §8 current-evidence tables from the T2 WP-Frag decision suites.

Reads ONLY ``checkpoints/sweep_oc_phase1a_decision_{clean,iid,v4lite}_t2_wpfrag_*``
and explicitly excludes the ``smoke1`` flow-validation suites, so the single-seed
smoke runs never leak into the result tables.

For each (model, train_protocol, seed, eval_condition) it pulls the 60 s rollout
KPIs from the per-profile ``summary.json`` and the training ``best_loss`` from
``phase1a_train_audit.csv``, then aggregates across seeds.

Cross-seed aggregation matches the convention used in
``docs/oc_followup_results_p1_p2.md`` §3.2: the per-seed value is the median over
trajectories at the 60 s horizon, and the cross-seed value is the MEAN of those
per-seed medians.

Selection policy (B1): a seed whose training collapsed with the catastrophic
``no successful training batches`` signature (nbad > 0 in training.log) is treated
as a flagged training failure and EXCLUDED from the quantitative aggregate. This is
the same uniform anomaly criterion the per-model notebook anomaly scan applies
across all four models; in the current T2 matrix it flags exactly one run
(clean ``ablate_no_lift`` seed43, nbad=276, same pred-divergence class as the
confirmed environment-drift artifacts). The excluded seed is never silently
dropped: ``aggregate.csv`` records ``n_seeds_total``, ``n_anomaly_excluded``,
``excluded_seeds`` and the excluded per-seed value (``excluded_seed_posmed``) so the
collapse stays visible as a transparency note rather than a quantified fragility
claim. ``per_seed_long.csv`` keeps every seed with its ``train_nbad`` /
``train_anomaly`` flag.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = REPO_ROOT / "checkpoints"
DEFAULT_OUT = REPO_ROOT / "analysis" / "section8_current_evidence"

TRAIN_PROTOCOLS = ("clean", "iid", "v4lite")
MODELS = ("phnode_full", "phnode_qforce", "ablate_no_lift", "ablate_no_mass_prior")
HORIZON = "60.0"
ROLLOUT_COLLAPSE_THRESHOLD_M = 10.0


def discover_summaries(model: str, train_protocol: str) -> list[Path]:
    suite = CHECKPOINTS / f"sweep_oc_phase1a_decision_{train_protocol}_t2_wpfrag_{model}"
    return sorted(suite.glob("*/rollout_benchmark/phase1a_*/**/summary.json"))


def load_audit_best_loss(model: str, train_protocol: str) -> dict[int, float]:
    suite = CHECKPOINTS / f"sweep_oc_phase1a_decision_{train_protocol}_t2_wpfrag_{model}"
    audit = suite / "phase1a_train_audit.csv"
    out: dict[int, float] = {}
    if not audit.exists():
        return out
    for row in csv.DictReader(audit.open()):
        out[int(row["seed"])] = float(row["best_loss"])
    return out


def extract_seed(path: Path) -> int:
    # run dir is two levels up from rollout_benchmark/phase1a_*/.../summary.json
    for part in path.parts:
        if part.endswith(tuple(f"seed{s}" for s in range(40, 60))):
            return int(part.split("seed")[-1])
    raise ValueError(f"cannot parse seed from {path}")


_NBAD_CACHE: dict[Path, int] = {}


def run_dir_from_summary(path: Path) -> Path:
    """Locate the training run directory (the parent of ``rollout_benchmark``)."""
    for parent in path.parents:
        if parent.name == "rollout_benchmark":
            return parent.parent
    raise ValueError(f"cannot locate run dir from {path}")


def count_no_successful_batches(run_dir: Path) -> int:
    """Count the catastrophic ``no successful training batches`` signature.

    Mirrors the per-model notebook anomaly scan: a positive count marks a
    catastrophic-gradient training failure (the seed46/seed43 pred-divergence mode).
    """
    if run_dir in _NBAD_CACHE:
        return _NBAD_CACHE[run_dir]
    log = run_dir / "training.log"
    nbad = (
        log.read_text(errors="ignore").count("no successful training batches")
        if log.exists()
        else 0
    )
    _NBAD_CACHE[run_dir] = nbad
    return nbad


def metric(summary: dict, name: str, stat: str) -> float | None:
    bucket = summary.get("overall", {}).get(HORIZON, {})
    m = bucket.get("metrics", {}).get(name)
    if not m:
        return None
    return m.get(stat)


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for model in MODELS:
        for proto in TRAIN_PROTOCOLS:
            best_loss = load_audit_best_loss(model, proto)
            for path in discover_summaries(model, proto):
                summary = json.load(path.open())
                cfg = summary.get("config", {})
                eval_profile = cfg.get("noise_profile", "")
                eval_protocol = cfg.get("noise_protocol", "")
                seed = extract_seed(path)
                nbad = count_no_successful_batches(run_dir_from_summary(path))
                ro = summary.get("rollout_outcomes", {}).get("overall", {})
                pos_med = metric(summary, "final_position_error", "median")
                rows.append(
                    {
                        "model_type": model,
                        "train_protocol": proto,
                        "seed": seed,
                        "eval_profile": eval_profile,
                        "eval_protocol": eval_protocol,
                        "pos_err_median_60s": pos_med,
                        "pos_err_p95_60s": metric(summary, "final_position_error", "p95"),
                        "rot_geodesic_median_60s": metric(
                            summary, "final_rotation_geodesic", "median"
                        ),
                        "max_so3_orth_error": metric(summary, "max_so3_orth_error", "max"),
                        "energy_span_median_60s": metric(summary, "energy_span", "median"),
                        "max_abs_energy_delta_median_60s": metric(
                            summary, "max_abs_energy_delta", "median"
                        ),
                        "completion_60s": ro.get("rates", {}).get("completed"),
                        "n_traj": ro.get("n_trajectories"),
                        "n_completed": ro.get("counts", {}).get("completed"),
                        "train_best_loss": best_loss.get(seed),
                        "train_nbad": nbad,
                        "train_anomaly": int(nbad > 0),
                        "rollout_collapse": int(
                            pos_med is not None and pos_med > ROLLOUT_COLLAPSE_THRESHOLD_M
                        ),
                        "source": str(path.relative_to(REPO_ROOT)),
                    }
                )
    rows.sort(
        key=lambda r: (
            r["model_type"],
            r["train_protocol"],
            r["eval_protocol"],
            r["eval_profile"],
            r["seed"],
        )
    )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["model_type"], r["train_protocol"], r["eval_profile"], r["eval_protocol"])
        groups.setdefault(key, []).append(r)

    agg: list[dict] = []
    for key, members in groups.items():
        members = [m for m in members if m["pos_err_median_60s"] is not None]
        if not members:
            continue

        # B1 selection policy: exclude flagged training failures (nbad > 0) from the
        # quantitative aggregate, applying the same uniform anomaly criterion to every
        # model. The excluded seeds are surfaced in dedicated columns, never hidden.
        excluded = [m for m in members if m["train_anomaly"]]
        used = [m for m in members if not m["train_anomaly"]]
        if not used:  # degenerate guard: every seed flagged -> report raw, exclude none
            used, excluded = members, []

        medians = [m["pos_err_median_60s"] for m in used]
        completions = [m["completion_60s"] for m in used if m["completion_60s"] is not None]
        worst = max(used, key=lambda m: m["pos_err_median_60s"])
        agg.append(
            {
                "model_type": key[0],
                "train_protocol": key[1],
                "eval_profile": key[2],
                "eval_protocol": key[3],
                "n_seeds_total": len(members),
                "n_anomaly_excluded": len(excluded),
                "n_seeds_used": len(used),
                "posmed_mean_of_seed_medians": round(statistics.mean(medians), 4),
                "posmed_median_of_seed_medians": round(statistics.median(medians), 4),
                "posmed_min": round(min(medians), 4),
                "posmed_max": round(max(medians), 4),
                "worst_seed": worst["seed"],
                "n_rollout_collapsed_all_seeds": sum(m["rollout_collapse"] for m in members),
                "excluded_seeds": ";".join(str(m["seed"]) for m in excluded),
                "excluded_seed_posmed": ";".join(
                    f"{m['pos_err_median_60s']:.4f}" for m in excluded
                ),
                "completion_mean": round(statistics.mean(completions), 4) if completions else None,
            }
        )
    agg.sort(
        key=lambda r: (r["model_type"], r["train_protocol"], r["eval_protocol"], r["eval_profile"])
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

    write_csv(args.output_dir / "per_seed_long.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", agg)

    print(f"per-seed rows: {len(rows)}  |  aggregate rows: {len(agg)}")
    print(f"written to {args.output_dir.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
