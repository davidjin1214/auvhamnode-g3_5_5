#!/usr/bin/env python3
"""Figure 4 (§1.8): per-trajectory 60 s terminal-error distribution, clean eval.

Explains why the median-vs-P95 ranking flips: structured models concentrate near
a low median but some carry heavier tails, so a low-median model can still be
beaten on the P95. Horizontal box-and-whisker per model on a log error axis (box
= IQR, whisker = P5-P95, line = median). The genuine no-lift seed43 training
collapse (~44 m) is shown as a flagged outlier, NOT folded into the box (the box
uses the four stable seeds, matching the table). The full-state black box
diverges (~80-90 m) and is annotated rather than ranked.

Reads figure_data/trajectory_distribution_long.csv
(scripts/export_section8_trajectory_distribution.py).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import _section8_style as S

import matplotlib.pyplot as plt
import numpy as np

WIDTH_MM = 130
HEIGHT_MM = 74
OUT_BASE = Path(__file__).resolve().parent / "section8_error_distribution"
DATA = S.EVIDENCE_DIR / "figure_data" / "trajectory_distribution_long.csv"

# top (best median) -> bottom; black box handled separately
ORDER = [
    "phnode_full", "ablate_no_lift", "ablate_no_mass_prior",
    "se3_momentum_blackbox", "se3_accel_blackbox", "phnode_qforce",
]


def load():
    vals: dict = defaultdict(list)          # model -> finite fpe (stable seeds)
    anomaly: dict = defaultdict(list)       # model -> seed43-style fpe
    diverged_models: dict = defaultdict(int)
    with DATA.open(newline="") as fh:
        for r in csv.DictReader(fh):
            m = r["model_type"]
            try:
                v = float(r["final_position_error"])
            except (TypeError, ValueError):
                v = math.nan
            is_anom = r.get("is_anomaly_seed") == "1"
            if not math.isfinite(v) or v > 60.0:
                diverged_models[m] += 1
                if is_anom and math.isfinite(v):
                    anomaly[m].append(v)
                continue
            if is_anom:
                anomaly[m].append(v)
            else:
                vals[m].append(v)
    return vals, anomaly, diverged_models


def draw():
    S.setup_style()
    vals, anomaly, diverged = load()
    fig, ax = plt.subplots(figsize=(WIDTH_MM * S.MM_TO_IN, HEIGHT_MM * S.MM_TO_IN), facecolor="white")
    fig.subplots_adjust(left=0.165, right=0.975, bottom=0.135, top=0.94)

    positions = list(range(len(ORDER), 0, -1))
    box_data = [vals[m] for m in ORDER]
    colors = [S.MODEL_COLOR[m] for m in ORDER]

    bp = ax.boxplot(
        box_data, positions=positions, vert=False, widths=0.56,
        whis=(5, 95), showfliers=False, patch_artist=True,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.30)
        patch.set_edgecolor(c)
        patch.set_linewidth(0.9)
    for med, c in zip(bp["medians"], colors):
        med.set_color(c)
        med.set_linewidth(1.6)
    for key in ("whiskers", "caps"):
        for art, c in zip(bp[key], [c for c in colors for _ in (0, 1)]):
            art.set_color(c)
            art.set_linewidth(0.8)

    # faint per-trajectory jitter to show concentration vs spread
    rng = np.random.default_rng(0)
    for m, pos in zip(ORDER, positions):
        xs = vals[m]
        ys = pos + (rng.random(len(xs)) - 0.5) * 0.30
        ax.scatter(xs, ys, s=1.1, color=S.MODEL_COLOR[m], alpha=0.18, linewidths=0, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(0.03, 130)
    ax.set_ylim(0.3, len(ORDER) + 0.7)
    ax.set_yticks(positions)
    ax.set_yticklabels([S.DISPLAY[m] for m in ORDER], fontsize=6.4)
    ax.set_xlabel("60 s terminal position error / m  (log scale)", fontsize=6.8)
    ax.grid(True, axis="x", color=S.PALETTE["rule"], lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=6.0)

    # full-state black box: diverged band at the right edge (model name only; the
    # divergence range and the median-vs-P95 reordering are explained in the caption)
    ax.axvspan(80, 130, color=S.PALETTE["risk_pale"], zorder=0)
    ax.text(102, (len(ORDER) + 1) / 2, "Full-state\nblack box",
            fontsize=5.6, color=S.PALETTE["risk"], ha="center", va="center", linespacing=1.2)

    S.save_fig(fig, OUT_BASE)
    plt.close(fig)
    print(f"wrote {OUT_BASE}.pdf / .png / .svg")
    for m in ORDER:
        xs = vals[m]
        if xs:
            print(f"  {m:<22} n={len(xs):3d} median={np.median(xs):.3f} p95={np.percentile(xs,95):.3f} diverged_traj={diverged.get(m,0)}")


if __name__ == "__main__":
    draw()
