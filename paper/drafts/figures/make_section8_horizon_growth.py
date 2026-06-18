#!/usr/bin/env python3
"""Figure 2 (§1.8): terminal position error vs prediction horizon (10/30/60 s),
clean train / clean eval. Short-horizon errors are close across models; the gap
opens at the 60 s horizon. The full-state black box diverges at every horizon and
carries no finite line (annotated).

Reads figure_data/horizon_growth.csv (scripts/export_section8_horizon_curves.py).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import _section8_style as S

import matplotlib.pyplot as plt

WIDTH_MM = 90
HEIGHT_MM = 72
OUT_BASE = Path(__file__).resolve().parent / "section8_horizon_growth"
DATA = S.EVIDENCE_DIR / "figure_data" / "horizon_growth.csv"

FINITE_MODELS = [
    "phnode_full", "ablate_no_lift", "ablate_no_mass_prior",
    "se3_momentum_blackbox", "se3_accel_blackbox", "phnode_qforce",
]
HORIZONS = [10.0, 30.0, 60.0]


def load():
    d: dict = defaultdict(dict)
    status: dict = {}
    with DATA.open(newline="") as fh:
        for r in csv.DictReader(fh):
            status[r["model_type"]] = r["cell_status"]
            v = r["posmed_mean_of_seed_medians"]
            if v != "":
                d[r["model_type"]][float(r["horizon_s"])] = float(v)
    return d, status


def draw():
    S.setup_style()
    data, status = load()
    fig, ax = plt.subplots(figsize=(WIDTH_MM * S.MM_TO_IN, HEIGHT_MM * S.MM_TO_IN), facecolor="white")
    fig.subplots_adjust(left=0.135, right=0.975, bottom=0.135, top=0.94)

    for m in FINITE_MODELS:
        ys = [data[m].get(h) for h in HORIZONS]
        lw = 1.7 if m == "phnode_full" else 1.0
        ax.plot(HORIZONS, ys, marker="o", ms=3.0, lw=lw, color=S.MODEL_COLOR[m],
                label=S.DISPLAY[m], zorder=5 if m == "phnode_full" else 3)
        ax.text(60.8, ys[-1], f"{ys[-1]:.2f}", fontsize=5.4, color=S.MODEL_COLOR[m],
                va="center", ha="left")

    ax.set_xticks(HORIZONS)
    ax.set_xlim(6, 78)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("prediction horizon / s", fontsize=6.8)
    ax.set_ylabel("60 s terminal position error / m", fontsize=6.8)
    ax.grid(True, color=S.PALETTE["rule"], lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=6.0)
    ax.legend(loc="upper left", fontsize=5.8, frameon=False, handlelength=1.7, labelspacing=0.32)
    # the full-state black box diverges at every horizon (no finite curve) -> caption.

    S.save_fig(fig, OUT_BASE)
    plt.close(fig)
    print(f"wrote {OUT_BASE}.pdf / .png / .svg")


if __name__ == "__main__":
    draw()
