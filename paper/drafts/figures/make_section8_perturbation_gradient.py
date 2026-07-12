#!/usr/bin/env python3
"""Figure 3 (§1.8): 60 s terminal position error across the initial-condition
perturbation profiles (clean -> nominal -> degraded -> heading-biased, ordered by
induced error rather than a single intensity axis -- heading-biased is a
systematic yaw-offset profile), clean training. The structured accuracy lead is
clearest under clean / mild perturbation and narrows under the highest-error
profile, where the structured models and the SE(3) momentum black box converge.
Geometry stability holds throughout; the full-state black box diverges under
every profile.

Reads figure_data/perturbation_gradient.csv (export_section8_horizon_curves.py).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import _section8_style as S

import matplotlib.pyplot as plt

WIDTH_MM = 90
HEIGHT_MM = 72
OUT_BASE = Path(__file__).resolve().parent / "section8_perturbation_gradient"
DATA = S.EVIDENCE_DIR / "figure_data" / "perturbation_gradient.csv"

FINITE_MODELS = [
    "phnode_full", "ablate_no_lift", "ablate_no_mass_prior",
    "se3_momentum_blackbox", "se3_accel_blackbox", "phnode_qforce",
]
PROFILES = ["clean", "nominal_eval", "degraded_eval", "heading_biased_eval"]
PROFILE_LABEL = {
    "clean": "clean", "nominal_eval": "nominal",
    "degraded_eval": "degraded", "heading_biased_eval": "heading\nbiased",
}


def load():
    d: dict = defaultdict(dict)
    with DATA.open(newline="") as fh:
        for r in csv.DictReader(fh):
            v = r["posmed_mean_of_seed_medians"]
            if v != "":
                d[r["model_type"]][r["eval_profile"]] = float(v)
    return d


def draw():
    S.setup_style()
    data = load()
    xs = list(range(len(PROFILES)))
    fig, ax = plt.subplots(figsize=(WIDTH_MM * S.MM_TO_IN, HEIGHT_MM * S.MM_TO_IN), facecolor="white")
    fig.subplots_adjust(left=0.135, right=0.975, bottom=0.155, top=0.94)

    for m in FINITE_MODELS:
        ys = [data[m].get(p) for p in PROFILES]
        lw = 1.7 if m == "phnode_full" else 1.0
        ax.plot(xs, ys, marker="o", ms=3.0, lw=lw, color=S.MODEL_COLOR[m],
                label=S.DISPLAY[m], zorder=5 if m == "phnode_full" else 3)

    ax.set_xticks(xs)
    ax.set_xticklabels([PROFILE_LABEL[p] for p in PROFILES], fontsize=6.0)
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(0, 5.0)
    ax.set_xlabel("initial-condition perturbation profile", fontsize=6.8)
    ax.set_ylabel("60 s terminal position error / m", fontsize=6.8)
    ax.grid(True, color=S.PALETTE["rule"], lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=6.0)
    # legend in the empty lower-right (curves rise to the right, so that corner is clear)
    ax.legend(loc="lower right", fontsize=5.8, frameon=False, handlelength=1.7, labelspacing=0.34)
    # accuracy lead narrows under the strongest perturbation; the full-state black box
    # diverges under every profile (no finite curve) -> both stated in the caption.

    S.save_fig(fig, OUT_BASE)
    plt.close(fig)
    print(f"wrote {OUT_BASE}.pdf / .png / .svg")


if __name__ == "__main__":
    draw()
