#!/usr/bin/env python3
"""Export the per-trajectory 60 s terminal-error distribution for §1.8 figure 4,
from the clean-eval ``trajectory_metrics.csv`` of each B-region run.

Per (model, seed): rollout_benchmark/phase1a_iideval_traj30_*/clean/
trajectory_metrics.csv holds 30 trajectories x 3 scenarios x 3 horizons. We keep
the 60 s rows (90 per run) and emit one row per trajectory, so the figure can show
WHY the median-vs-P95 ranking flips (structured models concentrate near the
median; the worst structured ablation and the black boxes carry heavy tails).

Two honest annotations the figure can pull from this CSV:
  - ``ablate_no_lift`` seed43 is the genuine training collapse (its trajectories
    sit ~44 m); it is emitted, not dropped, and flagged via ``is_anomaly_seed``.
  - ``blackbox_fullstate`` trajectories diverge (failure_reason=pred_divergence,
    ~80-90 m); emitted with failure_reason so the figure can clip/annotate.

Output (analysis/section8_current_evidence/figure_data/):
  trajectory_distribution_long.csv
  columns: model_type, seed, scenario, trajectory_id, horizon_s,
           final_position_error, failure_reason, is_anomaly_seed
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_TMPL = "checkpoints/sweep_oc_phase1a_decision_clean_t2_wpfrag_{model}"
DEFAULT_OUT = REPO_ROOT / "analysis" / "section8_current_evidence" / "figure_data"

MODELS = [
    "phnode_full", "ablate_no_lift", "ablate_no_mass_prior",
    "se3_momentum_blackbox", "se3_accel_blackbox", "phnode_qforce",
    "blackbox_fullstate",
]
SEEDS = [42, 43, 44, 45, 46]
TARGET_HORIZON = "60.0"
# Genuine reproducible training collapse (not env drift); surfaced, not dropped.
ANOMALY_SEEDS = {("ablate_no_lift", 43)}


def find_metrics_csv(model: str, seed: int) -> Path | None:
    model_dir = REPO_ROOT / SWEEP_TMPL.format(model=model)
    runs = sorted(model_dir.glob(f"*_{model}_seed{seed}"))
    if not runs:
        return None
    hits = sorted(runs[0].glob("rollout_benchmark/phase1a_iideval_traj30_*/clean/trajectory_metrics.csv"))
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_path = args.out_dir / "trajectory_distribution_long.csv"
    cols = ["model_type", "seed", "scenario", "trajectory_id", "horizon_s",
            "final_position_error", "failure_reason", "is_anomaly_seed"]
    n_rows = 0
    per_seed_check: dict[str, list[float]] = {}
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for model in MODELS:
            for seed in SEEDS:
                csv_path = find_metrics_csv(model, seed)
                if csv_path is None:
                    print(f"  MISSING {model} seed{seed}")
                    continue
                anom = "1" if (model, seed) in ANOMALY_SEEDS else "0"
                finite_meds: list[float] = []
                with csv_path.open(newline="") as src:
                    for r in csv.DictReader(src):
                        if r.get("horizon_s") != TARGET_HORIZON:
                            continue
                        fpe = r.get("final_position_error", "")
                        w.writerow({
                            "model_type": model, "seed": seed,
                            "scenario": r.get("scenario", ""),
                            "trajectory_id": r.get("trajectory_id", ""),
                            "horizon_s": TARGET_HORIZON,
                            "final_position_error": fpe,
                            "failure_reason": r.get("failure_reason", ""),
                            "is_anomaly_seed": anom,
                        })
                        n_rows += 1
                        try:
                            finite_meds.append(float(fpe))
                        except (TypeError, ValueError):
                            pass
                if finite_meds:
                    per_seed_check.setdefault(model, []).append(st.median(finite_meds))

    print(f"wrote {out_path} ({n_rows} rows)")
    # provenance sanity: phnode_full clean per-seed medians -> mean ~0.68, median ~0.61
    pf = per_seed_check.get("phnode_full", [])
    if pf:
        print(f"  phnode_full per-seed medians = {[round(x, 3) for x in sorted(pf)]} "
              f"(mean {st.mean(pf):.3f}, median {st.median(pf):.3f}; table: 0.677 / 0.611)")


if __name__ == "__main__":
    main()
