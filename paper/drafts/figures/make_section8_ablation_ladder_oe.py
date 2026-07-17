"""Ocean Engineering manuscript variant of the structural-ablation ladder.

The OE submission (mymanu1.tex, Figure 5) uses a variant of
``make_section8_ablation_ladder.py`` that was adapted ad hoc on 2026-07-13 and
never preserved; this script reconstructs it from the shipped PDF so the
manuscript figure has a tracked source. Differences against the thesis-chapter
generator, matching the manuscript's own terminology and multiplier convention:

  - multiplier annotations format the pre-rounded ``multiplier_vs_full`` column
    (config force prints 5.5x, matching the manuscript text), instead of
    recomputing from the exact medians (which prints 5.6x);
  - the divergent ablation is display-named "Force Conditioning" with axis tag
    "force interface" (manuscript contrast A4), not "Narrow Actuation";
  - the divergent row is a full-axis-width pale hatched band (alpha 0.12) with
    the centred annotation "no finite 60 s run" -- the manuscript's run-level
    accounting phrase -- instead of the arrow-through-axis "diverges" marker;
  - fix applied 2026-07-17 (review item A-2): the "No Lift" (A3) axis sub-label
    reads "coupling structure", consistent with the manuscript's grouping of A3
    under dissipation and zero-power coupling; the thesis chapter keeps A3 in
    its inertial-energy sub-prior group, so the shared generator is unchanged.

Reads figure_data/ablation_ladder.csv (scripts/export_section8_ablation_ladder.py).
Writes section8_ablation_ladder_oe.* (copied into the manuscript repository as
section8_ablation_ladder.pdf).
"""

from __future__ import annotations

import csv
from pathlib import Path

import _section8_style as S

import matplotlib.pyplot as plt

WIDTH_MM = 150
HEIGHT_MM = 70
OUT_BASE = Path(__file__).resolve().parent / "section8_ablation_ladder_oe"
DATA = S.EVIDENCE_DIR / "figure_data" / "ablation_ladder.csv"

AXIS_TAG = {
    "reference": "full model",
    "inertial_energy": "inertial-energy prior",
    "force_decomp": "force decomposition",
    "damping": "damping structure",
    "actuation": "force interface",
}

# Manuscript-side overrides against the shared thesis-chapter naming.
DISPLAY_OVERRIDE = {"ablate_bu_only": "Force Conditioning"}
# A3 probes the zero-power coupling branch (skew J_theta); the manuscript groups
# it with A2 on the dissipation / zero-power coupling axis (review item A-2).
AXIS_TAG_OVERRIDE = {"ablate_no_lift": "coupling structure"}


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
    for yy, r in zip(y, rows):
        m = r["model_type"]
        color = S.MODEL_COLOR[m]
        if r["status"] != "ok":
            # Full-width pale band: the run-level accounting yields no finite
            # 60 s result for any repetition of this contrast.
            ax.barh(yy, 13.0, height=0.60, color=color, alpha=0.12,
                    edgecolor=color, linewidth=0.7, hatch="////", zorder=2)
            ax.text(6.5, yy, "no finite 60 s run", va="center", ha="center",
                    fontsize=5.8, color=color, fontweight="bold", style="italic")
            continue
        med = float(r["clean_median"])
        p95 = float(r["clean_p95"])
        ax.barh(yy, med, height=0.60, color=color, alpha=0.9,
                edgecolor=color, linewidth=0.9, zorder=3)
        ax.plot([med, p95], [yy, yy], color=S.PALETTE["ink"], lw=0.7, solid_capstyle="round", zorder=4)
        ax.plot([p95, p95], [yy - 0.16, yy + 0.16], color=S.PALETTE["ink"], lw=0.7, zorder=4)
        mult = r["multiplier_vs_full"]
        tag = "baseline" if m == "phnode_full" else f"{float(mult):.1f}x"
        ax.text(p95 + 0.18, yy, tag, va="center", ha="left", fontsize=5.8,
                color=color, fontweight="bold")

    # left-margin labels: model name (bold) over structural-axis tag (small, muted)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    tr = ax.get_yaxis_transform()
    for yy, r in zip(y, rows):
        name = DISPLAY_OVERRIDE.get(r["model_type"], S.DISPLAY[r["model_type"]])
        tag = AXIS_TAG_OVERRIDE.get(r["model_type"], AXIS_TAG[r["axis_group"]])
        ax.text(-0.018, yy + 0.17, name, transform=tr,
                fontsize=6.5, color=S.PALETTE["ink"], ha="right", va="center", fontweight="bold")
        ax.text(-0.018, yy - 0.21, tag, transform=tr,
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
