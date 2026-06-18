#!/usr/bin/env python3
"""Figure 5 (§1.8): internal structural diagnostics, clean config, 60 s.

Panel A -- SO(3) orthogonality error (worst-case over seeds, log scale). The
geometry-preserving models hold rotations on the group to ~1.3e-5; the energy-
core ablation (config force) is ~3.4e-4, an order of magnitude higher, in line
with its accuracy degradation. The errors are small but non-zero (ordinary ODE
solver), so orthogonality is a diagnostic, not a strict invariant. The full-state
black box leaves the group entirely and is omitted.
Panel B -- mechanical energy span (median over seeds). Defined only for the
scalar-potential models; removing the potential (config force) or using a black
box leaves the mechanical energy undefined by construction.

Reads figure_data/diagnostics_summary.csv (scripts/export_section8_diagnostics.py).
"""

from __future__ import annotations

import csv
from pathlib import Path

import _section8_style as S

import matplotlib.pyplot as plt

WIDTH_MM = 150
HEIGHT_MM = 64
OUT_BASE = Path(__file__).resolve().parent / "section8_diagnostics"
DATA = S.EVIDENCE_DIR / "figure_data" / "diagnostics_summary.csv"

# Panel A: SO(3), all models except the fully diverged full-state black box.
SO3_ORDER = [
    "phnode_full", "ablate_no_lift", "ablate_no_mass_prior",
    "se3_momentum_blackbox", "se3_accel_blackbox", "phnode_qforce",
]
# Panel B: energy span, scalar-potential models only.
ENERGY_ORDER = ["phnode_full", "ablate_no_lift", "ablate_no_mass_prior"]


def load():
    rows = {}
    with DATA.open(newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["model_type"]] = r
    return rows


def draw():
    S.setup_style()
    rows = load()
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(WIDTH_MM * S.MM_TO_IN, HEIGHT_MM * S.MM_TO_IN),
        facecolor="white", gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.235, top=0.86, wspace=0.34)

    # ---- Panel A: SO(3) orthogonality (log) ----
    xa = list(range(len(SO3_ORDER)))
    ya = [float(rows[m]["so3_orth_max"]) for m in SO3_ORDER]
    ca = [S.PALETTE["bar_qforce"] if m == "phnode_qforce" else
          (S.PALETTE["geometry"] if m == "phnode_full" else S.PALETTE["bar_energy"])
          for m in SO3_ORDER]
    axA.bar(xa, ya, width=0.66, color=ca, alpha=0.9, edgecolor="none")
    axA.set_yscale("log")
    axA.set_ylim(5e-6, 1e-3)
    axA.set_xticks(xa)
    axA.set_xticklabels([S.AXIS_DISPLAY[m] for m in SO3_ORDER], fontsize=5.6)
    axA.set_ylabel(r"$\mathrm{SO}(3)$ orthogonality error (max)", fontsize=6.4)
    axA.grid(True, axis="y", color=S.PALETTE["rule"], lw=0.5)
    axA.set_axisbelow(True)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)
    axA.tick_params(axis="y", labelsize=5.8)
    axA.tick_params(axis="x", length=0)
    S.panel_label(axA, "a", dx=-0.115, dy=0.04)

    # ---- Panel B: mechanical energy span ----
    xb = list(range(len(ENERGY_ORDER)))
    yb = [float(rows[m]["energy_span_median"]) for m in ENERGY_ORDER]
    cb = [S.PALETTE["geometry"] if m == "phnode_full" else S.PALETTE["bar_energy"] for m in ENERGY_ORDER]
    axB.bar(xb, yb, width=0.62, color=cb, alpha=0.9, edgecolor="none")
    for x, y in zip(xb, yb):
        axB.text(x, y + 0.6, f"{y:.1f}", ha="center", va="bottom", fontsize=5.8, color=S.PALETTE["ink"])
    axB.set_xticks(xb)
    axB.set_xticklabels([S.AXIS_DISPLAY[m] for m in ENERGY_ORDER], fontsize=5.6)
    axB.set_ylim(0, 22)
    axB.set_ylabel("mechanical energy span (median)", fontsize=6.4)
    axB.grid(True, axis="y", color=S.PALETTE["rule"], lw=0.5)
    axB.set_axisbelow(True)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)
    axB.tick_params(axis="y", labelsize=5.8)
    axB.tick_params(axis="x", length=0)
    S.panel_label(axB, "b", dx=-0.165, dy=0.04)

    S.save_fig(fig, OUT_BASE)
    plt.close(fig)
    print(f"wrote {OUT_BASE}.pdf / .png / .svg")


if __name__ == "__main__":
    draw()
