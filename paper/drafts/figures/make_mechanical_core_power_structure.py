#!/usr/bin/env python3
"""Draw a compact mechanical-core power schematic for the thesis chapter."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/auvhamnode_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/auvhamnode_xdg_cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


WIDTH_MM = 138
HEIGHT_MM = 60
MM_TO_IN = 1 / 25.4

OUT_DIR = Path(__file__).resolve().parent
OUT_BASE = OUT_DIR / "mechanical_core_power_structure"

PALETTE = {
    "ink": "#24272B",
    "muted": "#60646B",
    "rule": "#D7DDE4",
    "paper": "#FFFFFF",
    "state": "#2E5E8C",
    "state_pale": "#F4F8FB",
    "power": "#9A7418",
    "power_pale": "#FBF7E8",
    "aux_pale": "#F6F7F8",
}

TITLE_SIZE = 6.9
BODY_SIZE = 5.9
MATH_SIZE = 7.2
LABEL_SIZE = 5.9


def pick_font() -> str:
    candidates = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Arial Unicode MS",
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
            "mathtext.fontset": "dejavusans",
            "font.size": BODY_SIZE,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )


def add_text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = BODY_SIZE,
    weight: str = "regular",
    color: str = PALETTE["ink"],
    ha: str = "center",
    va: str = "center",
    linespacing: float = 1.12,
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
    radius: float = 0.010,
    lw: float = 0.85,
    linestyle: str = "-",
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=lw,
        linestyle=linestyle,
        edgecolor=edge,
        facecolor=face,
        mutation_aspect=1,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = PALETTE["muted"],
    lw: float = 0.85,
    scale: float = 8.0,
    linestyle: str = "-",
    zorder: int = 4,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def labeled_box(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    face: str,
    edge: str,
    title_size: float = TITLE_SIZE,
    body_size: float = BODY_SIZE,
    title_y: float = 0.62,
) -> None:
    rounded_box(ax, x, y, w, h, face=face, edge=edge, lw=0.85)
    add_text(
        ax,
        x + w / 2,
        y + h * title_y,
        title,
        size=title_size,
        weight="bold",
        color=PALETTE["ink"],
    )
    if body:
        add_text(ax, x + w / 2, y + h * 0.31, body, size=body_size, color=PALETTE["ink"])


def draw() -> None:
    setup_style()
    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    core_x, core_y, core_w, core_h = 0.055, 0.130, 0.675, 0.760
    rounded_box(
        ax,
        core_x,
        core_y,
        core_w,
        core_h,
        face=PALETTE["paper"],
        edge=PALETTE["rule"],
        radius=0.018,
        lw=0.95,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    add_text(
        ax,
        core_x + 0.024,
        core_y + core_h - 0.050,
        "Open six-DOF mechanical core",
        size=TITLE_SIZE,
        weight="bold",
        color=PALETTE["ink"],
        ha="left",
    )

    labeled_box(
        ax,
        x=0.105,
        y=0.660,
        w=0.215,
        h=0.135,
        title="Core state",
        body="configuration\nmomentum",
        face=PALETTE["state_pale"],
        edge=PALETTE["state"],
        body_size=BODY_SIZE,
        title_y=0.68,
    )
    labeled_box(
        ax,
        x=0.425,
        y=0.660,
        w=0.235,
        h=0.135,
        title="Storage",
        body="mechanical energy",
        face=PALETTE["paper"],
        edge=PALETTE["rule"],
        body_size=BODY_SIZE,
    )

    labeled_box(
        ax,
        x=0.105,
        y=0.455,
        w=0.215,
        h=0.120,
        title="Zero-power coupling",
        body="coadjoint\nskew branch",
        face=PALETTE["paper"],
        edge=PALETTE["rule"],
        title_size=TITLE_SIZE,
        body_size=BODY_SIZE,
        title_y=0.70,
    )
    labeled_box(
        ax,
        x=0.425,
        y=0.455,
        w=0.235,
        h=0.120,
        title="Dissipation",
        body="energy extraction",
        face=PALETTE["power_pale"],
        edge=PALETTE["power"],
        title_size=TITLE_SIZE,
        body_size=BODY_SIZE,
    )

    rounded_box(ax, 0.165, 0.215, 0.430, 0.125, face=PALETTE["power_pale"], edge=PALETTE["power"], lw=0.90)
    add_text(
        ax,
        0.380,
        0.286,
        "Energy balance",
        size=TITLE_SIZE,
        weight="bold",
        color=PALETTE["ink"],
    )
    add_text(ax, 0.380, 0.240, r"$\dot H_\theta=-P_D+P_\tau$", size=MATH_SIZE, color=PALETTE["ink"])

    labeled_box(
        ax,
        x=0.790,
        y=0.610,
        w=0.165,
        h=0.195,
        title="Carried context",
        body="actuator state\ncurrent, depth",
        face=PALETTE["aux_pale"],
        edge=PALETTE["muted"],
        title_size=TITLE_SIZE,
        body_size=BODY_SIZE,
        title_y=0.70,
    )
    labeled_box(
        ax,
        x=0.790,
        y=0.330,
        w=0.165,
        h=0.145,
        title="Force port",
        body="external\ngeneralized force",
        face=PALETTE["power_pale"],
        edge=PALETTE["power"],
        title_size=TITLE_SIZE,
        body_size=BODY_SIZE,
        title_y=0.70,
    )

    arrow(ax, (0.320, 0.725), (0.425, 0.725), color=PALETTE["state"])
    arrow(ax, (0.215, 0.660), (0.215, 0.575), color=PALETTE["muted"])
    arrow(ax, (0.540, 0.660), (0.540, 0.575), color=PALETTE["muted"])
    arrow(ax, (0.215, 0.455), (0.270, 0.340), color=PALETTE["muted"])
    arrow(ax, (0.540, 0.455), (0.490, 0.340), color=PALETTE["power"])
    arrow(ax, (0.790, 0.400), (0.595, 0.265), color=PALETTE["power"])
    arrow(ax, (0.870, 0.610), (0.870, 0.475), color=PALETTE["muted"], linestyle=(0, (3, 2)))
    arrow(ax, (0.790, 0.700), (0.730, 0.540), color=PALETTE["muted"], linestyle=(0, (3, 2)))

    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
    ]:
        fig.savefig(OUT_BASE.with_suffix(suffix), bbox_inches="tight", pad_inches=0.02, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    draw()
