#!/usr/bin/env python3
"""Export per-epoch training/validation curves for the §1.8 training-convergence
figure (figure 1), from the B-region Phase-1A t2_wpfrag decision suites.

The figure tells an honest three-part story (all read from the SAME run set that
backs the §1.8 main table):
  1. structured models train cleanly  -- ``phnode_full`` train/val total loss
     converge and the SO(3) orthogonality penalty stays at ~1e-7.
  2. the unstructured baseline ALSO converges on the task loss but never on
     geometry -- ``blackbox_fullstate`` train/val total ~0.01 yet
     ``train_so3_orth`` stays ~0.17 (the latent geometric drift that only
     surfaces as rollout divergence later).
  3. a single training collapse exists -- ``ablate_no_lift`` seed43 suffers a
     catastrophic gradient blow-up (train_total -> ~1e3, success rate -> ~0).

Output (one row per model, seed, epoch):
  analysis/section8_current_evidence/figure_data/training_curves_long.csv
columns: model_type, seed, epoch, train_total, test_total, train_so3_orth,
         train_success_rate
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_TMPL = "checkpoints/sweep_oc_phase1a_decision_clean_t2_wpfrag_{model}"
DEFAULT_OUT = REPO_ROOT / "analysis" / "section8_current_evidence" / "figure_data"

# The honest-three-elements cast for figure 1.
MODELS = ["phnode_full", "blackbox_fullstate", "ablate_no_lift"]
SEEDS = [42, 43, 44, 45, 46]
FIELDS = ["train_total", "test_total", "train_so3_orth", "train_success_rate"]


def find_run_dir(model: str, seed: int) -> Path | None:
    model_dir = REPO_ROOT / SWEEP_TMPL.format(model=model)
    if not model_dir.is_dir():
        return None
    hits = sorted(model_dir.glob(f"*_{model}_seed{seed}"))
    return hits[0] if hits else None


def load_history(run_dir: Path) -> dict | None:
    pkl = run_dir / "training_history.pkl"
    if not pkl.exists():
        return None
    with pkl.open("rb") as fh:
        return pickle.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_path = args.out_dir / "training_curves_long.csv"
    n_rows = 0
    summary: list[str] = []
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model_type", "seed", "epoch", *FIELDS])
        for model in MODELS:
            for seed in SEEDS:
                run = find_run_dir(model, seed)
                if run is None:
                    summary.append(f"  MISSING {model} seed{seed}")
                    continue
                hist = load_history(run)
                if hist is None:
                    summary.append(f"  no history {model} seed{seed}")
                    continue
                epochs = hist.get("epoch") or list(range(len(hist["train_total"])))
                series = {f: hist.get(f, []) for f in FIELDS}
                n = len(series["train_total"])
                for i in range(n):
                    ep = epochs[i] if i < len(epochs) else i
                    writer.writerow([
                        model, seed, ep,
                        *[f"{series[f][i]:.6g}" if i < len(series[f]) and series[f][i] is not None else "" for f in FIELDS],
                    ])
                n_rows += n
                last_tot = series["train_total"][-1] if series["train_total"] else float("nan")
                last_orth = series["train_so3_orth"][-1] if series["train_so3_orth"] else float("nan")
                mx_tot = max(series["train_total"]) if series["train_total"] else float("nan")
                summary.append(
                    f"  {model:<20} seed{seed}: epochs={n} final train_total={last_tot:.4g} "
                    f"max train_total={mx_tot:.4g} final so3_orth={last_orth:.4g}"
                )

    print(f"wrote {out_path} ({n_rows} rows)")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
