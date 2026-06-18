"""§1.8 structural-ablation ladder (replaces the old two-level bar chart): the
remove-one marginal effect of each structural prior on 60 s clean accuracy, all
degradation multipliers against the single full-model baseline (0.68 m).

The ladder separates three structural stories so the headline stays exact:
  - within the inertial-energy sub-prior group the energy core dominates and the
    effect decreases monotonically (config force 5.5x > mass prior 1.9x >
    lift 1.2x);
  - the merged-force baseline isolates the value of decomposing the
    non-conservative forces (energy core kept) -- a separate axis (~2.2x);
  - coupled damping and actuation conditioning are two further independent
    structural necessities: diagonal-only damping degrades ~6x, and over-narrow
    actuation conditioning diverges outright.

Reads figure_data/ablation_ladder.csv (scripts/export_section8_ablation_ladder.py).
"""

from __future__ import annotations

import csv
from pathlib import Path

import _section8_style as S

import matplotlib.pyplot as plt

WIDTH_MM = 150
HEIGHT_MM = 70
OUT_BASE = Path(__file__).resolve().parent / "section8_ablation_ladder"
DATA = S.EVIDENCE_DIR / "figure_data" / "ablation_ladder.csv"

AXIS_TAG = {
    "reference": "full model",
    "inertial_energy": "inertial-energy prior",
    "force_decomp": "force decomposition",
    "damping": "damping structure",
    "actuation": "actuation conditioning",
}


def load():
    rows = []
    with DATA.open(newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def draw():
    S.setup_style()
    rows = load()
    # display order: ascending degradation, divergent ablation last
    def sort_key(r):
        try:
            return (0, float(r["multiplier_vs_full"]))
        except ValueError:
            return (1, 1e9)
    rows = sorted(rows, key=sort_key)

    fig, ax = plt.subplots(figsize=(WIDTH_MM * S.MM_TO_IN, HEIGHT_MM * S.MM_TO_IN), facecolor="white")
    fig.subplots_adjust(left=0.265, right=0.975, bottom=0.135, top=0.975)

    y = list(range(len(rows), 0, -1))
    DIVERGE_X = 8.2  # placeholder bar length for the divergent ablation
    for yy, r in zip(y, rows):
        m = r["model_type"]
        color = S.MODEL_COLOR[m]
        diverged = r["status"] != "ok"
        med = DIVERGE_X if diverged else float(r["clean_median"])
        ax.barh(yy, med, height=0.60, color=color, alpha=0.32 if diverged else 0.9,
                edgecolor=color, linewidth=0.9, hatch="////" if diverged else None, zorder=3)
        if not diverged:
            p95 = float(r["clean_p95"])
            ax.plot([med, p95], [yy, yy], color=S.PALETTE["ink"], lw=0.7, solid_capstyle="round", zorder=4)
            ax.plot([p95, p95], [yy - 0.16, yy + 0.16], color=S.PALETTE["ink"], lw=0.7, zorder=4)
            mult = r["multiplier_vs_full"]
            tag = "baseline" if m == "phnode_full" else f"{float(mult):.1f}x"
            ax.text(p95 + 0.18, yy, tag, va="center", ha="left", fontsize=5.8,
                    color=color, fontweight="bold")
        else:
            ax.text(DIVERGE_X / 2, yy, "diverges", va="center", ha="center", fontsize=6.0,
                    color=color, fontweight="bold", style="italic")

    # left-margin labels: model name (bold) over structural-axis tag (small, muted)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    tr = ax.get_yaxis_transform()
    for yy, r in zip(y, rows):
        ax.text(-0.018, yy + 0.17, S.DISPLAY[r["model_type"]], transform=tr,
                fontsize=6.5, color=S.PALETTE["ink"], ha="right", va="center", fontweight="bold")
        ax.text(-0.018, yy - 0.21, AXIS_TAG[r["axis_group"]], transform=tr,
                fontsize=5.0, color=S.PALETTE["muted"], ha="right", va="center")

    ax.set_xlim(0, 13)
    ax.set_ylim(0.4, len(rows) + 0.6)
    ax.set_xlabel("60 s terminal position error / m  (bar = median, whisker to P95)", fontsize=6.6)
    ax.grid(True, axis="x", color=S.PALETTE["rule"], lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=6.0)
    ax.axvline(float(rows[0]["clean_median"]), color=S.PALETTE["geometry"], lw=0.7,
               ls=(0, (3, 2)), alpha=0.7, zorder=1)

    S.save_fig(fig, OUT_BASE)
    plt.close(fig)
    print(f"wrote {OUT_BASE}.pdf / .png / .svg")


if __name__ == "__main__":
    draw()
