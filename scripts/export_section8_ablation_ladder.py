#!/usr/bin/env python3
"""Build the structural-ablation ladder for §1.8 figure (remade from the old
two-level bar chart): the remove-one marginal effect of each structural prior on
60 s clean accuracy.

Merges two evidence sources into ONE ladder, all degradation multipliers taken
against the SINGLE full-model baseline (phnode_full clean = 0.6767 m), so the
ladder reads on one scale regardless of provenance:
  - the seven main-table models (aggregate.csv, clean/clean);
  - the broader structural ablations (catalog_supplement_aggregate.csv):
    diagonal-damping, narrow-actuation, merged-force.

Each ablation is tagged with the structural axis it probes (NOT shown as a data
source in the paper -- the axis is the scientific grouping):
  - inertial_energy : the {energy-core, mass-prior, lift} sub-prior group that
    degrades monotonically (config-force 5.5x > mass-prior 1.9x > lift 1.2x);
  - damping         : coupled -> diagonal damping (diag_damping);
  - actuation       : actuation conditioning too narrow (bu_only, diverges);
  - force_decomp    : merge the structured D/J/B into one learned force
    (merged_force), energy core kept.

Output: analysis/section8_current_evidence/figure_data/ablation_ladder.csv
columns: model_type, axis_group, clean_median, clean_p95, multiplier_vs_full,
         status, n_used
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVID = REPO_ROOT / "analysis" / "section8_current_evidence"
DEFAULT_OUT = EVID / "figure_data"

BASELINE_MODEL = "phnode_full"

# ladder membership and axis grouping; order = display order (full -> worst)
LADDER = [
    ("phnode_full", "reference"),
    ("ablate_no_lift", "inertial_energy"),
    ("ablate_no_mass_prior", "inertial_energy"),
    ("phnode_merged_force", "force_decomp"),
    ("phnode_qforce", "inertial_energy"),
    ("ablate_diag_damping", "damping"),
    ("ablate_bu_only", "actuation"),
]


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def load_main(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["train_protocol"] == "clean" and r["eval_profile"] == "clean":
                out[r["model_type"]] = r
    return out


def load_supplement(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["train_type"] == "clean_train" and r["eval_profile"] == "clean":
                out[r["model_type"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    main_rows = load_main(EVID / "aggregate.csv")
    supp_rows = load_supplement(EVID / "catalog_supplement_aggregate.csv")

    baseline = _f(main_rows[BASELINE_MODEL]["posmed_mean_of_seed_medians"])
    assert baseline is not None and abs(baseline - 0.6767) < 1e-3, f"baseline {baseline} != 0.6767"

    out_path = args.out_dir / "ablation_ladder.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model_type", "axis_group", "clean_median", "clean_p95",
                    "multiplier_vs_full", "status", "n_used"])
        for model, axis in LADDER:
            if model in main_rows:
                r = main_rows[model]
                med = _f(r["posmed_mean_of_seed_medians"])
                p95 = _f(r["posp95_mean_of_seed_p95s"])
                n = r.get("n_seeds_used", "")
                status = "ok" if med is not None else "stability_failure"
            elif model in supp_rows:
                r = supp_rows[model]
                med = _f(r["posmed_mean_of_seed_medians"])
                p95 = _f(r["posp95_mean_of_seed_p95s"])
                n = r.get("n_ok", "")
                status = "ok" if med is not None else r.get("cell_status", "stability_failure")
            else:
                print(f"  MISSING {model}")
                continue
            mult = round(med / baseline, 2) if med is not None else ""
            w.writerow([model, axis,
                        (f"{med:.4f}" if med is not None else ""),
                        (f"{p95:.4f}" if p95 is not None else ""),
                        mult, status, n])

    print(f"wrote {out_path} (baseline phnode_full = {baseline:.4f} m)")
    with out_path.open() as fh:
        for line in fh:
            print("  " + line.rstrip())


if __name__ == "__main__":
    main()
