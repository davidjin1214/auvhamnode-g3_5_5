#!/usr/bin/env python3
"""Figure 1 (§1.8): training convergence -- task loss vs geometry.

Panel a -- both models drive the task (total) loss down: AUVHamNODE and the
full-state black box converge (~4e-3 vs ~1e-2).
Panel b -- only the structured model keeps the rotation on the group during
training: the SO(3) orthogonality penalty stays at ~1e-7 for AUVHamNODE but
~0.17 for the black box -- the latent geometric drift that only surfaces later
as rollout divergence.

Curves are the per-epoch median across repeated runs (descriptive register; no
single run is singled out -- the one unstable ablation training run is handled
in the main-table footnote, not here).

Reads analysis/section8_current_evidence/figure_data/training_curves_long.csv
(produced by scripts/export_section8_training_curves.py).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import _section8_style as S

import matplotlib.pyplot as plt

WIDTH_MM = 150
HEIGHT_MM = 62
OUT_BASE = Path(__file__).resolve().parent / "section8_training_curves"
DATA = S.EVIDENCE_DIR / "figure_data" / "training_curves_long.csv"


def load():
    """model -> {seed -> {epoch: {field: value}}}."""
    data: dict = defaultdict(lambda: defaultdict(dict))
    with DATA.open(newline="") as fh:
        for r in csv.DictReader(fh):
            rec = {}
            for f in ("train_total", "test_total", "train_so3_orth"):
                try:
                    rec[f] = float(r[f]) if r[f] != "" else None
                except ValueError:
                    rec[f] = None
            data[r["model_type"]][int(r["seed"])][int(r["epoch"])] = rec
    return data


def median_curve(seed_map, field, *, seeds=None):
    """Per-epoch median over seeds of `field`."""
    seeds = seeds or list(seed_map.keys())
    epochs = sorted({e for s in seeds for e in seed_map[s]})
    xs, ys = [], []
    for e in epochs:
        vals = [seed_map[s][e][field] for s in seeds
                if e in seed_map[s] and seed_map[s][e].get(field) is not None]
        if vals:
            xs.append(e)
            ys.append(median(vals))
    return xs, ys


def draw():
    S.setup_style()
    data = load()
    pf, bb = data["phnode_full"], data["blackbox_fullstate"]
    pf_seeds = [42, 43, 44, 45, 46]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(WIDTH_MM * S.MM_TO_IN, HEIGHT_MM * S.MM_TO_IN), facecolor="white"
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.175, top=0.90, wspace=0.255)

    # ---- Panel a: task (total) loss -- both models converge ----
    x, y = median_curve(pf, "train_total", seeds=pf_seeds)
    axA.plot(x, y, color=S.PALETTE["geometry"], lw=1.4, label="AUVHamNODE", zorder=4)
    x, y = median_curve(bb, "train_total")
    axA.plot(x, y, color=S.PALETTE["risk"], lw=1.4, label="Full-state black box", zorder=3)
    axA.set_yscale("log")
    axA.set_xlim(0, 250)
    axA.set_ylim(2e-3, 2.0)
    axA.set_xlabel("training epoch", fontsize=6.8)
    axA.set_ylabel("total loss", fontsize=6.8)
    axA.grid(True, which="major", color=S.PALETTE["rule"], lw=0.5)
    axA.set_axisbelow(True)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)
    axA.tick_params(labelsize=6.0)
    axA.legend(loc="upper right", fontsize=5.8, frameon=False, handlelength=1.6, borderaxespad=0.2)
    S.panel_label(axA, "a", dx=-0.135, dy=0.03)

    # ---- Panel b: SO(3) orthogonality penalty -- only the structured model converges ----
    x, y = median_curve(pf, "train_so3_orth", seeds=pf_seeds)
    axB.plot(x, y, color=S.PALETTE["geometry"], lw=1.4, label="AUVHamNODE", zorder=4)
    x, y = median_curve(bb, "train_so3_orth")
    axB.plot(x, y, color=S.PALETTE["risk"], lw=1.4, label="Full-state black box", zorder=3)
    axB.set_yscale("log")
    axB.set_xlim(0, 250)
    axB.set_ylim(3e-8, 3.0)
    axB.set_xlabel("training epoch", fontsize=6.8)
    axB.set_ylabel(r"$\mathrm{SO}(3)$ orthogonality loss", fontsize=6.8)
    axB.grid(True, which="major", color=S.PALETTE["rule"], lw=0.5)
    axB.set_axisbelow(True)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)
    axB.tick_params(labelsize=6.0)
    axB.legend(loc="center right", fontsize=5.8, frameon=False, handlelength=1.6, borderaxespad=0.4)
    S.panel_label(axB, "b", dx=-0.155, dy=0.03)

    S.save_fig(fig, OUT_BASE)
    plt.close(fig)
    print(f"wrote {OUT_BASE}.pdf / .png / .svg")


if __name__ == "__main__":
    draw()
