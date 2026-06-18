#!/usr/bin/env python3
"""Export the error-growth (figure 2) and perturbation-gradient (figure 3)
intermediate tables for §1.8, from ``horizon_scenario_aggregate.csv``.

CRITICAL -- eval_protocol disambiguation. The aggregate carries an
``eval_protocol`` column (clean / iid_noisy_ic / v4_lite). For ``nominal_eval``
there are TWO rows per cell (iid_noisy_ic and v4_lite); a naive group-by that
filters only on (train_protocol, eval_profile, scope) silently AVERAGES the two
and yields the wrong number (e.g. phnode_full nominal 0.90 instead of the table
value 0.96). The §1.8 main tables use:
    clean profile          -> eval_protocol == 'clean'
    nominal/degraded/head.  -> eval_protocol == 'iid_noisy_ic'
This script applies that rule and ASSERTS reproduction of the published table
cells before writing, so the figures cannot drift from the tables.

Outputs (analysis/section8_current_evidence/figure_data/):
  - horizon_growth.csv        figure 2: clean-train / clean-eval, scope=overall,
                              one row per (model, horizon_s).
  - perturbation_gradient.csv figure 3: clean-train, horizon=60, scope=overall,
                              one row per (model, eval_profile).
columns include posmed_mean_of_seed_medians (primary, table caliber),
posmed_median_of_seed_medians, posp95_mean_of_seed_p95s, n_seeds_used,
n_rollout_diverged, cell_status.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVID = REPO_ROOT / "analysis" / "section8_current_evidence"
DEFAULT_OUT = EVID / "figure_data"

MODELS = [
    "phnode_full", "ablate_no_lift", "ablate_no_mass_prior",
    "se3_momentum_blackbox", "se3_accel_blackbox", "phnode_qforce",
    "blackbox_fullstate",
]
PROFILES = ["clean", "nominal_eval", "degraded_eval", "heading_biased_eval"]
HORIZONS = ["10.0", "30.0", "60.0"]
PRIMARY = "posmed_mean_of_seed_medians"
KEEP = [
    "posmed_mean_of_seed_medians", "posmed_median_of_seed_medians",
    "posp95_mean_of_seed_p95s", "n_seeds_used", "n_rollout_diverged",
    "n_seeds_total",
]


def eval_protocol_for(profile: str) -> str:
    return "clean" if profile == "clean" else "iid_noisy_ic"


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def pick(rows, *, model, profile, horizon):
    ep = eval_protocol_for(profile)
    hits = [
        r for r in rows
        if r["model_type"] == model and r["train_protocol"] == "clean"
        and r["eval_profile"] == profile and r["eval_protocol"] == ep
        and r["scope"] == "overall" and r["horizon_s"] == horizon
    ]
    if len(hits) > 1:
        raise SystemExit(f"non-unique cell after eval_protocol filter: {model}/{profile}/{horizon} -> {len(hits)} rows")
    return hits[0] if hits else None


def cell_status(row) -> str:
    if row is None:
        return "missing"
    v = row.get(PRIMARY, "")
    if v in ("", None) or (isinstance(v, str) and v.strip() == ""):
        return "stability_failure"
    return "ok"


def emit(rows, out_path: Path, key_name: str, keys: list[str], *, fixed) -> None:
    cols = ["model_type", key_name, *KEEP, "cell_status"]
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for model in MODELS:
            for k in keys:
                row = pick(rows, model=model, profile=(k if key_name == "eval_profile" else fixed["profile"]),
                           horizon=(k if key_name == "horizon_s" else fixed["horizon"]))
                rec = {"model_type": model, key_name: k, "cell_status": cell_status(row)}
                for c in KEEP:
                    rec[c] = (row.get(c, "") if row else "")
                w.writerow(rec)


def approx(a: str, b: float, tol=1e-3) -> bool:
    try:
        return abs(float(a) - b) <= tol
    except (TypeError, ValueError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(EVID / "horizon_scenario_aggregate.csv")

    # --- provenance assertions (must reproduce the published table cells) ---
    f60 = pick(rows, model="phnode_full", profile="clean", horizon="60.0")
    f30 = pick(rows, model="phnode_full", profile="clean", horizon="30.0")
    f10 = pick(rows, model="phnode_full", profile="clean", horizon="10.0")
    assert f60 and approx(f60[PRIMARY], 0.6767), f"fig2 60s {f60 and f60[PRIMARY]} != 0.6767"
    assert approx(f30[PRIMARY], 0.2716) and approx(f10[PRIMARY], 0.0831), "fig2 10/30s mismatch"
    nom = pick(rows, model="phnode_full", profile="nominal_eval", horizon="60.0")
    deg = pick(rows, model="phnode_full", profile="degraded_eval", horizon="60.0")
    hed = pick(rows, model="phnode_full", profile="heading_biased_eval", horizon="60.0")
    assert approx(nom[PRIMARY], 0.9604), f"fig3 nominal {nom[PRIMARY]} != 0.9604 (eval_protocol leak?)"
    assert approx(deg[PRIMARY], 1.7574) and approx(hed[PRIMARY], 3.1257), "fig3 degraded/heading mismatch"
    print("provenance assertions PASS (0.0831/0.2716/0.6767 ; nominal 0.9604 / degraded 1.7574 / heading 3.1257)")

    # --- figure 2: horizon growth (clean eval) ---
    fig2 = args.out_dir / "horizon_growth.csv"
    emit(rows, fig2, "horizon_s", HORIZONS, fixed={"profile": "clean"})
    # --- figure 3: perturbation gradient (60 s) ---
    fig3 = args.out_dir / "perturbation_gradient.csv"
    emit(rows, fig3, "eval_profile", PROFILES, fixed={"horizon": "60.0"})

    print(f"wrote {fig2}")
    print(f"wrote {fig3}")


if __name__ == "__main__":
    main()
