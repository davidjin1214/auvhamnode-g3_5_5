#!/usr/bin/env python3
"""Export §8 catalog-supplement evidence rows from the A-region (catalog / g3_5_5).

Companion to ``scripts/export_section8_t2_evidence.py`` (which covers ONLY the
B-region Phase-1A t2_wpfrag decision suites). This script pulls the cells that
the B-region never trained but the A-region (catalog) already did under the SAME
noise protocol (``iid_noisy_ic``) and the SAME eval convention (60 s,
``final_position_error`` median, ``scope=overall``, PRBS+CHIRP+OU x30 traj,
profiles clean / nominal_eval / degraded_eval / heading_biased_eval).

Supplemented cells (see docs/section8_evidence_merge_plan.md A.3):
  - L2 ablation ladder (clean-train): ``ablate_diag_damping``, ``ablate_bu_only``
  - L1 geometry-stability under the iid noise line (noisy-train):
    ``blackbox_fullstate``, ``se3_momentum_blackbox``, ``se3_accel_blackbox``
  - within-mirror baseline: ``phnode_full`` clean (A-region), env-drift seeds excluded

Cross-seed aggregation matches export_section8_t2_evidence.py:
  per-seed value = median over trajectories at 60 s; the cross-seed central
  tendency is reported BOTH as the MEAN and the MEDIAN of the per-seed medians
  (mean is sensitive to a bad seed, median is robust -- both are emitted so the
  paper can disclose, e.g., the no_lift seed43 genuine-fragility case honestly).

SELECTION (critical): per (model, train_type, seed, eval_profile) the catalog can
hold several rollout_run_ids (primary resampled_traj30, p12 matched-followup,
legacy heldout, and traj8 *_iideval_* probes). We de-duplicate to ONE rollout via
``rollout_run_registry.csv``: require ``is_selection_eligible == 1`` and prefer the
``resampled_traj30`` rollout, then any ``primary``. Skipping this de-dup silently
mixes in the traj8 iideval probes and corrupts the numbers.

ANOMALY TAXONOMY (docs/section8_evidence_merge_plan.md A.2), emitted per seed in
``anomaly_class``:
  - ``env_drift``           : historical g3_5_5 environment-drift artifact that does
                              NOT reproduce on the current environment (provenance
                              audit). Excluded from the aggregate. Currently the
                              hard-coded phnode_full clean seed42/seed46.
  - ``rollout_diverged``    : 60 s rollout median is non-finite OR exceeds
                              ROLLOUT_DIVERGENCE_THRESHOLD_M. Excluded from the
                              finite aggregate; if EVERY seed of a cell diverges the
                              cell is reported as a long-horizon stability failure
                              with no finite median (the diverged range is retained).
  - ``ok``                  : finite, within threshold; enters the aggregate.
  (The third taxonomy class, ``genuine_fragility`` -- reproducible model fragility
  such as B-region no_lift seed43 -- lives in the B-region export, not here. It is
  surfaced for the paper by ``--nolift-disclosure`` below.)

Outputs (under analysis/section8_current_evidence/ by default):
  - catalog_supplement_per_seed.csv
  - catalog_supplement_aggregate.csv   (mirror=g3_5_5, with within-mirror multiplier)
  - nolift_seed43_disclosure.csv        (derived from the B-region per_seed_long.csv:
                                         no_lift clean reported as all-5 vs N=4 cluster,
                                         both mean and median, per the honest-reporting
                                         decision for the genuine seed43 fragility)
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = REPO_ROOT / "analysis" / "oc_data_catalog"
DEFAULT_OUT = REPO_ROOT / "analysis" / "section8_current_evidence"
B_REGION_PER_SEED = DEFAULT_OUT / "per_seed_long.csv"

# ---- eval convention (must match export_section8_t2_evidence.py) ----
HORIZON_S = "60.0"
SCOPE = "overall"
METRIC = "final_position_error"
EVAL_PROFILES = ["clean", "nominal_eval", "degraded_eval", "heading_biased_eval"]
ROLLOUT_DIVERGENCE_THRESHOLD_M = 10.0

MIRROR = "g3_5_5"

# Cells this supplement is responsible for: (model_type, train_type)
SUPPLEMENT_CELLS = [
    ("ablate_diag_damping", "clean_train"),   # L2 ladder: coupled->diagonal damping
    ("ablate_bu_only", "clean_train"),        # L2 ladder: actuation conditioning
    ("phnode_merged_force", "clean_train"),   # L2 ladder: force-decomposition axis (keep energy core, merge D/J/B)
    ("blackbox_fullstate", "noisy_train"),    # L1 stability under iid noise line
    ("se3_momentum_blackbox", "noisy_train"),
    ("se3_accel_blackbox", "noisy_train"),
]
# baseline cell for the within-mirror multiplier
BASELINE_CELL = ("phnode_full", "clean_train", "clean")

# documented environment-drift artifacts (provenance audit; evidence_status_overrides.csv)
ENV_DRIFT = {
    ("phnode_full", "clean_train", "42"),
    ("phnode_full", "clean_train", "46"),
}


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v


def load_registry(catalog_dir: Path):
    """(model, train, seed, profile) -> chosen rollout_run_id (selection-eligible, de-duped)."""
    by_key = {}
    path = catalog_dir / "rollout_run_registry.csv"
    with path.open() as fh:
        for r in csv.DictReader(fh):
            key = (r["model_type"], r["train_type"], r["seed"], r["eval_profile"])
            by_key.setdefault(key, []).append(
                (r.get("is_selection_eligible") == "1", r.get("rollout_purpose", ""), r["rollout_run_id"])
            )
    chosen = {}
    for key, cands in by_key.items():
        elig = [c for c in cands if c[0]] or cands
        pick = None
        for c in elig:
            if "resampled_traj30" in c[2]:
                pick = c[2]
                break
        if pick is None:
            for c in elig:
                if c[1] == "primary":
                    pick = c[2]
                    break
        if pick is None and elig:
            pick = elig[0][2]
        chosen[key] = pick
    return chosen


def load_metric(catalog_dir: Path):
    """(model, train, seed, profile, rollout_id, stat) -> 60s overall final_position_error value."""
    out = {}
    path = catalog_dir / "canonical_rollout_summary_long.csv"
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if (
                r["metric_name"] == METRIC
                and r["horizon_s"] == HORIZON_S
                and r["scope"] == SCOPE
                and r["stat_name"] in ("median", "p95")
            ):
                v = _f(r["value_numeric"])
                if v is None:
                    continue
                out[(r["model_type"], r["train_type"], r["seed"], r["eval_profile"], r["rollout_run_id"], r["stat_name"])] = v
    return out


def seeds_for(registry, model, train, profile):
    return sorted({k[2] for k in registry if k[0] == model and k[1] == train and k[3] == profile})


def per_seed_rows(registry, metric, model, train, profile):
    """Return list of dicts: seed, median, p95, anomaly_class."""
    rows = []
    for s in seeds_for(registry, model, train, profile):
        rid = registry.get((model, train, s, profile))
        med = metric.get((model, train, s, profile, rid, "median")) if rid else None
        p95 = metric.get((model, train, s, profile, rid, "p95")) if rid else None
        if (model, train, s) in ENV_DRIFT:
            cls = "env_drift"
        elif med is None or med != med or med > ROLLOUT_DIVERGENCE_THRESHOLD_M:
            cls = "rollout_diverged"
        else:
            cls = "ok"
        rows.append({"seed": s, "median": med, "p95": p95, "anomaly_class": cls, "rollout_run_id": rid})
    return rows


def aggregate(rows):
    ok = [r for r in rows if r["anomaly_class"] == "ok" and r["median"] is not None]
    diverged = [r for r in rows if r["anomaly_class"] == "rollout_diverged"]
    drift = [r for r in rows if r["anomaly_class"] == "env_drift"]
    meds = [r["median"] for r in ok]
    p95s = [r["p95"] for r in ok if r["p95"] is not None]

    def _round(x, n=4):
        return round(x, n) if x is not None else None

    agg = {
        "n_seeds_total": len(rows),
        "n_ok": len(ok),
        "n_rollout_diverged": len(diverged),
        "n_env_drift_excluded": len(drift),
        "diverged_seeds": ";".join(r["seed"] for r in diverged),
        "diverged_seed_values": ";".join(
            ("nan" if (r["median"] is None or r["median"] != r["median"]) else f"{r['median']:.2f}") for r in diverged
        ),
        "env_drift_seeds": ";".join(r["seed"] for r in drift),
        "posmed_mean_of_seed_medians": _round(st.mean(meds)) if meds else None,
        "posmed_median_of_seed_medians": _round(st.median(meds)) if meds else None,
        "posmed_min": _round(min(meds)) if meds else None,
        "posmed_max": _round(max(meds)) if meds else None,
        "posp95_mean_of_seed_p95s": _round(st.mean(p95s)) if p95s else None,
        "posp95_median_of_seed_p95s": _round(st.median(p95s)) if p95s else None,
        # cell-level structural failure when no seed survives the threshold
        "cell_status": "stability_failure" if (len(ok) == 0 and rows) else "ok",
    }
    return agg


def compute_baseline(registry, metric):
    rows = per_seed_rows(registry, metric, *BASELINE_CELL)
    ok = [r["median"] for r in rows if r["anomaly_class"] == "ok" and r["median"] is not None]
    return round(st.mean(ok), 4) if ok else None, rows


def write_per_seed(out_dir, all_rows):
    path = out_dir / "catalog_supplement_per_seed.csv"
    cols = ["mirror", "model_type", "train_type", "eval_profile", "seed",
            "pos_err_median_60s", "pos_err_p95_60s", "anomaly_class", "rollout_run_id"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    return path


def write_aggregate(out_dir, agg_rows):
    path = out_dir / "catalog_supplement_aggregate.csv"
    cols = ["mirror", "model_type", "train_type", "eval_profile", "cell_status",
            "n_seeds_total", "n_ok", "n_rollout_diverged", "n_env_drift_excluded",
            "posmed_mean_of_seed_medians", "posmed_median_of_seed_medians",
            "posmed_min", "posmed_max",
            "posp95_mean_of_seed_p95s", "posp95_median_of_seed_p95s",
            "within_mirror_baseline_m", "degradation_multiplier_mean",
            "diverged_seeds", "diverged_seed_values", "env_drift_seeds"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)
    return path


def nolift_disclosure(b_per_seed: Path, out_dir: Path):
    """Honest dual (mean & median) reporting of no_lift clean, all-5 vs N=4 cluster.

    Reads the B-region per_seed_long.csv. seed43 is a GENUINE reproducible fragility
    (not env drift), so it is reported -- not silently dropped. We emit both the
    all-5-seeds central tendency (mean inflated by seed43; median robust) and the
    N=4 stable-cluster central tendency.
    """
    if not b_per_seed.exists():
        return None
    vals = {}
    with b_per_seed.open() as fh:
        for r in csv.DictReader(fh):
            if (r["model_type"] == "ablate_no_lift" and r["train_protocol"] == "clean"
                    and r["eval_profile"] == "clean" and r["eval_protocol"] == "clean"):
                v = _f(r["pos_err_median_60s"])
                if v is not None:
                    vals[r["seed"]] = (v, r.get("train_anomaly", ""), r.get("train_nbad", ""))
    if not vals:
        return None
    all5 = sorted(vals.items())
    seed43 = vals.get("43")
    cluster = {s: v for s, (v, *_rest) in vals.items() if s != "43"}
    all_meds = [v for _s, (v, *_r) in all5]
    clus_meds = list(cluster.values())

    path = out_dir / "nolift_seed43_disclosure.csv"
    cols = ["scope", "n", "includes_seed43", "mean_of_seed_medians",
            "median_of_seed_medians", "min", "max", "seed43_value", "seed43_train_nbad"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerow({
            "scope": "all_5_seeds", "n": len(all_meds), "includes_seed43": "yes",
            "mean_of_seed_medians": round(st.mean(all_meds), 4),
            "median_of_seed_medians": round(st.median(all_meds), 4),
            "min": round(min(all_meds), 4), "max": round(max(all_meds), 4),
            "seed43_value": round(seed43[0], 4) if seed43 else "",
            "seed43_train_nbad": seed43[2] if seed43 else "",
        })
        w.writerow({
            "scope": "stable_cluster_excl_seed43", "n": len(clus_meds), "includes_seed43": "no",
            "mean_of_seed_medians": round(st.mean(clus_meds), 4),
            "median_of_seed_medians": round(st.median(clus_meds), 4),
            "min": round(min(clus_meds), 4), "max": round(max(clus_meds), 4),
            "seed43_value": "", "seed43_train_nbad": "",
        })
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog-dir", type=Path, default=CATALOG_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--b-region-per-seed", type=Path, default=B_REGION_PER_SEED)
    ap.add_argument("--no-nolift-disclosure", action="store_true",
                    help="skip the B-region no_lift seed43 dual-reporting CSV")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry(args.catalog_dir)
    metric = load_metric(args.catalog_dir)

    baseline, base_rows = compute_baseline(registry, metric)
    if baseline is None:
        raise SystemExit("could not compute within-mirror baseline (phnode_full clean)")

    per_seed_out, agg_out = [], []
    for model, train in SUPPLEMENT_CELLS:
        for profile in EVAL_PROFILES:
            rows = per_seed_rows(registry, metric, model, train, profile)
            if not rows:
                continue
            for r in rows:
                per_seed_out.append({
                    "mirror": MIRROR, "model_type": model, "train_type": train,
                    "eval_profile": profile, "seed": r["seed"],
                    "pos_err_median_60s": ("" if r["median"] is None else r["median"]),
                    "pos_err_p95_60s": ("" if r["p95"] is None else r["p95"]),
                    "anomaly_class": r["anomaly_class"], "rollout_run_id": r["rollout_run_id"] or "",
                })
            agg = aggregate(rows)
            # The within-mirror degradation multiplier (vs clean phnode_full baseline) is an
            # ABLATION-degradation only for clean-trained ablations evaluated under clean eval.
            # For noise-trained baselines it would be a cross-condition ratio, not a degradation
            # -> leave it blank to avoid misreading.
            mult = None
            if (train == "clean_train" and profile == "clean"
                    and agg["posmed_mean_of_seed_medians"] is not None):
                mult = round(agg["posmed_mean_of_seed_medians"] / baseline, 2)
            agg_out.append({
                "mirror": MIRROR, "model_type": model, "train_type": train, "eval_profile": profile,
                "within_mirror_baseline_m": baseline, "degradation_multiplier_mean": mult, **agg,
            })

    p1 = write_per_seed(args.out_dir, per_seed_out)
    p2 = write_aggregate(args.out_dir, agg_out)
    p3 = None
    if not args.no_nolift_disclosure:
        p3 = nolift_disclosure(args.b_region_per_seed, args.out_dir)

    # ---- stdout summary ----
    print(f"within-mirror baseline (phnode_full clean, A-region, env-drift excluded) = {baseline} m")
    drift_seeds = ";".join(r["seed"] for r in base_rows if r["anomaly_class"] == "env_drift")
    print(f"  baseline ok seeds: {[r['seed'] for r in base_rows if r['anomaly_class']=='ok']}; env_drift excluded: {drift_seeds}")
    print()
    print(f"{'model':<22}{'train':<12}{'profile':<20}{'status':<18}{'mean':>8}{'median':>8}{'mult':>7}  diverged")
    for a in agg_out:
        mean = "" if a["posmed_mean_of_seed_medians"] is None else f"{a['posmed_mean_of_seed_medians']:.3f}"
        med = "" if a["posmed_median_of_seed_medians"] is None else f"{a['posmed_median_of_seed_medians']:.3f}"
        mult = "" if a["degradation_multiplier_mean"] is None else f"{a['degradation_multiplier_mean']:.1f}x"
        dv = f"{a['n_rollout_diverged']}/{a['n_seeds_total']}" + (f" [{a['diverged_seed_values']}]" if a["diverged_seeds"] else "")
        print(f"{a['model_type']:<22}{a['train_type']:<12}{a['eval_profile']:<20}{a['cell_status']:<18}{mean:>8}{med:>8}{mult:>7}  {dv}")
    print()
    print(f"wrote: {p1}")
    print(f"wrote: {p2}")
    if p3:
        print(f"wrote: {p3}")


if __name__ == "__main__":
    main()
