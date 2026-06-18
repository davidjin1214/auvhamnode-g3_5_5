#!/usr/bin/env python3
"""Export the internal-diagnostics intermediate table for §1.8 figure 5, from the
B-region ``per_seed_long.csv`` (clean train / clean eval / eval_protocol=clean).

Two diagnostics, matching table ``tab:s8-diag``:
  - SO(3) orthogonality error: reported as the MAX over seeds (worst-case drift
    off the rotation group). All seven models carry it; the geometry-preserving
    models sit at ~1.3e-5, the energy-core ablation (config force) is ~3.4e-4.
  - mechanical energy span: reported as the MEDIAN over seeds, DEFINED ONLY for
    the scalar-potential models (full model, no_lift). For no_lift the genuine
    seed43 training collapse is excluded from the energy-span median (N=4).

Outputs (analysis/section8_current_evidence/figure_data/):
  - diagnostics_per_seed.csv  one row per (model, seed)
  - diagnostics_summary.csv   one row per model (so3 max, energy-span median)
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVID = REPO_ROOT / "analysis" / "section8_current_evidence"
DEFAULT_OUT = EVID / "figure_data"

MODELS = [
    "phnode_full", "ablate_no_lift", "ablate_no_mass_prior",
    "se3_momentum_blackbox", "se3_accel_blackbox", "phnode_qforce",
    "blackbox_fullstate",
]
# scalar-potential models -- mechanical energy span is defined here (these keep the
# scalar potential V(q); only the energy-core ablation / black boxes drop it).
ENERGY_DEFINED = {"phnode_full", "ablate_no_lift", "ablate_no_mass_prior"}
# genuine training collapse excluded from the energy-span median
ENERGY_EXCLUDE = {("ablate_no_lift", "43")}


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_model: dict[str, list[dict]] = {m: [] for m in MODELS}
    with (EVID / "per_seed_long.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if (r["train_protocol"] == "clean" and r["eval_profile"] == "clean"
                    and r["eval_protocol"] == "clean" and r["model_type"] in by_model):
                by_model[r["model_type"]].append(r)

    per_seed_path = args.out_dir / "diagnostics_per_seed.csv"
    with per_seed_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model_type", "seed", "max_so3_orth_error", "energy_span_median_60s",
                    "energy_defined", "energy_excluded"])
        for m in MODELS:
            for r in sorted(by_model[m], key=lambda x: int(x["seed"])):
                defd = m in ENERGY_DEFINED
                excl = (m, r["seed"]) in ENERGY_EXCLUDE
                w.writerow([m, r["seed"], r.get("max_so3_orth_error", ""),
                            (r.get("energy_span_median_60s", "") if defd else ""),
                            int(defd), int(excl)])

    summary_path = args.out_dir / "diagnostics_summary.csv"
    summ = {}
    with summary_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model_type", "so3_orth_max", "energy_span_median", "energy_defined", "n_energy_seeds"])
        for m in MODELS:
            so3 = [v for v in (_f(r.get("max_so3_orth_error")) for r in by_model[m]) if v is not None]
            so3_max = max(so3) if so3 else None
            if m in ENERGY_DEFINED:
                es = [_f(r.get("energy_span_median_60s")) for r in by_model[m]
                      if (m, r["seed"]) not in ENERGY_EXCLUDE]
                es = [v for v in es if v is not None]
                es_med = st.median(es) if es else None
                n_es = len(es)
            else:
                es_med, n_es = None, 0
            summ[m] = (so3_max, es_med)
            w.writerow([m, (f"{so3_max:.3e}" if so3_max is not None else ""),
                        (f"{es_med:.4f}" if es_med is not None else ""),
                        int(m in ENERGY_DEFINED), n_es])

    # provenance assertions vs tab:s8-diag
    def approx(a, b, rel=0.02):
        return a is not None and abs(a - b) <= rel * b
    assert approx(summ["phnode_full"][1], 17.8, 0.02), summ["phnode_full"]
    assert approx(summ["ablate_no_lift"][1], 18.7, 0.02), summ["ablate_no_lift"]
    assert approx(summ["ablate_no_mass_prior"][1], 1.9, 0.05), summ["ablate_no_mass_prior"]
    assert approx(summ["phnode_qforce"][0], 3.4e-4, 0.05), summ["phnode_qforce"]
    assert approx(summ["phnode_full"][0], 1.4e-5, 0.1), summ["phnode_full"]
    print("provenance assertions PASS (energy span 17.8/18.7/1.9 ; so3 full 1.4e-5 / qforce 3.4e-4)")
    print(f"wrote {per_seed_path}")
    print(f"wrote {summary_path}")
    for m in MODELS:
        s = summ[m]
        print(f"  {m:<22} so3_max={s[0]:.3e}" + (f"  energy_span_median={s[1]:.3f}" if s[1] is not None else "  energy_span=undef"))


if __name__ == "__main__":
    main()
