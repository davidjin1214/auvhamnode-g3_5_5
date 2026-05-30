#!/usr/bin/env python3
"""Draw the two-level section 8 evidence figure for the AUVHamNODE chapter."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/auvhamnode_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/auvhamnode_xdg_cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch


WIDTH_MM = 154
HEIGHT_MM = 92
MM_TO_IN = 1 / 25.4

OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[2]
DATA_PATH = REPO_ROOT / "analysis" / "section8_current_evidence" / "aggregate.csv"
OUT_BASE = OUT_DIR / "section8_two_level_evidence"

PALETTE = {
    "ink": "#252525",
    "muted": "#62666D",
    "rule": "#D8DDE3",
    "paper": "#FFFFFF",
    "geometry": "#2E6EBA",
    "geometry_pale": "#EEF5FC",
    "energy": "#2F8E86",
    "energy_pale": "#EDF7F4",
    "risk": "#A95B3A",
    "risk_pale": "#F8EFEA",
    "accent": "#C78F2D",
    "accent_pale": "#FBF4E5",
    "bar": "#8BA9C9",
    "bar_dark": "#2E6EBA",
    "bar_energy": "#6AAEA7",
    "bar_qforce": "#C78F2D",
}


DISPLAY = {
    "phnode_full": "AUVHamNODE",
    "ablate_no_lift": "No Lift",
    "ablate_no_mass_prior": "No Mass Prior",
    "se3_momentum_blackbox": "SE(3) Momentum BB",
    "se3_accel_blackbox": "SE(3) Accel BB",
    "phnode_qforce": "pH q-Force",
    "blackbox_fullstate": "Full-State BB",
}


def pick_font() -> str:
    candidates = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Arial Unicode MS",
        "PingFang SC",
        "Hiragino Sans GB",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


def setup_style() -> None:
    font_name = pick_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 6.7,
            "axes.linewidth": 0.65,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )


def read_clean_rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with DATA_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["train_protocol"] == "clean" and row["eval_profile"] == "clean":
                rows[row["model_type"]] = row
    return rows


def as_float(value: str) -> float:
    if value == "":
        return math.nan
    return float(value)


def add_text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 6.5,
    weight: str = "regular",
    color: str = PALETTE["ink"],
    ha: str = "center",
    va: str = "center",
    linespacing: float = 1.12,
    transform=None,
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color,
        linespacing=linespacing,
        transform=transform if transform is not None else ax.transData,
    )


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str,
    edge: str,
    radius: float = 0.035,
    lw: float = 0.85,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        mutation_aspect=1,
    )
    ax.add_patch(patch)


def draw_geometry_panel(ax, rows: dict[str, dict[str, str]]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_text(ax, 0.020, 0.965, "Layer 1: geometry and rollout stability", size=7.2, weight="bold", color=PALETTE["geometry"], ha="left")
    add_text(
        ax,
        0.020,
        0.900,
        "clean train / clean eval, 60 s horizon",
        size=5.6,
        color=PALETTE["muted"],
        ha="left",
    )

    fs = rows["blackbox_fullstate"]
    se3 = rows["se3_momentum_blackbox"]
    boxes = [
        (
            0.070,
            0.600,
            0.860,
            0.185,
            PALETTE["risk_pale"],
            PALETTE["risk"],
            DISPLAY["blackbox_fullstate"],
            "no pose geometry",
            f"{fs['n_rollout_diverged']}/5 diverged",
            "no finite median",
        ),
        (
            0.070,
            0.310,
            0.860,
            0.185,
            PALETTE["geometry_pale"],
            PALETTE["geometry"],
            DISPLAY["se3_momentum_blackbox"],
            "SE(3)+mass",
            f"{as_float(se3['posmed_mean_of_seed_medians']):.2f} m",
            f"median; P95 {as_float(se3['posp95_mean_of_seed_p95s']):.2f} m; N={se3['n_seeds_used']}",
        ),
    ]
    for x, y, w, h, face, edge, title, subtitle, value, note in boxes:
        rounded_box(ax, x, y, w, h, face=face, edge=edge)
        add_text(ax, x + 0.035, y + h - 0.045, title, size=6.7, weight="bold", color=edge, ha="left")
        add_text(ax, x + 0.035, y + 0.060, subtitle, size=5.6, color=PALETTE["muted"], ha="left")
        add_text(ax, x + w - 0.065, y + h * 0.58, value, size=7.2, weight="bold", color=edge, ha="right")
        add_text(ax, x + w - 0.065, y + 0.060, note, size=5.4, color=PALETTE["muted"], ha="right")

    ax.annotate(
        "",
        xy=(0.500, 0.515),
        xytext=(0.500, 0.585),
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color=PALETTE["geometry"], mutation_scale=9),
    )
    add_text(ax, 0.540, 0.550, "retain manifold structure", size=5.6, color=PALETTE["geometry"], ha="left")

    rounded_box(ax, 0.090, 0.120, 0.820, 0.100, face=PALETTE["paper"], edge=PALETTE["rule"], radius=0.025, lw=0.75)
    add_text(
        ax,
        0.500,
        0.170,
        "Evidence claim: geometry is a key stability factor\nwithin this model family.",
        size=5.4,
        color=PALETTE["ink"],
    )


def draw_precision_panel(ax, rows: dict[str, dict[str, str]]) -> None:
    models = [
        "phnode_full",
        "ablate_no_lift",
        "ablate_no_mass_prior",
        "se3_momentum_blackbox",
        "se3_accel_blackbox",
        "phnode_qforce",
    ]
    labels = []
    medians = []
    p95s = []
    colors = []
    for model in models:
        row = rows[model]
        labels.append(f"{DISPLAY[model]}\nN={row['n_seeds_used']}")
        medians.append(as_float(row["posmed_mean_of_seed_medians"]))
        p95s.append(as_float(row["posp95_mean_of_seed_p95s"]))
        if model == "phnode_full":
            colors.append(PALETTE["bar_dark"])
        elif model == "phnode_qforce":
            colors.append(PALETTE["bar_qforce"])
        elif "blackbox" in model:
            colors.append("#A8AFBA")
        else:
            colors.append(PALETTE["bar_energy"])

    y = list(reversed(range(len(models))))
    ax.barh(y, medians, height=0.56, color=colors, alpha=0.92, edgecolor="none")
    for yy, median, p95 in zip(y, medians, p95s):
        ax.plot([median, p95], [yy, yy], color=PALETTE["ink"], lw=0.85, solid_capstyle="round")
        ax.plot([p95, p95], [yy - 0.16, yy + 0.16], color=PALETTE["ink"], lw=0.85)
        ax.text(median + 0.08, yy, f"{median:.2f}", va="center", ha="left", fontsize=5.7, color=PALETTE["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.8)
    ax.set_xlim(0, 12.2)
    ax.set_xlabel("60 s final position error (m)", fontsize=6.2, labelpad=6)
    ax.set_title("Layer 2: energy core and precision", loc="left", fontsize=7.2, fontweight="bold", color=PALETTE["energy"], pad=8)
    ax.grid(axis="x", color=PALETTE["rule"], lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=5.7)
    ax.axvline(medians[0], color=PALETTE["bar_dark"], lw=0.75, ls=(0, (3, 2)), alpha=0.8)
    ax.text(
        medians[0] + 0.06,
        len(models) - 0.38,
        "full model",
        fontsize=5.4,
        color=PALETTE["bar_dark"],
        va="top",
        ha="left",
    )

def draw() -> None:
    setup_style()
    rows = read_clean_rows()
    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN), facecolor="white")
    gs = GridSpec(1, 2, width_ratios=[0.90, 1.45], left=0.040, right=0.985, bottom=0.190, top=0.875, wspace=0.260)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    fig.text(
        0.035,
        0.955,
        "Two-level evidence under the current protocol",
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.985,
        0.955,
        "current data / model family only",
        ha="right",
        va="top",
        fontsize=5.8,
        color=PALETTE["muted"],
    )

    draw_geometry_panel(ax_left, rows)
    draw_precision_panel(ax_right, rows)
    fig.text(
        0.500,
        0.050,
        "Bars show seed-mean terminal medians; whiskers show seed-mean P95. No Lift excludes seed 43 by the training-anomaly rule.",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color=PALETTE["muted"],
    )

    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
    ]:
        fig.savefig(OUT_BASE.with_suffix(suffix), bbox_inches="tight", pad_inches=0.01, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    draw()
