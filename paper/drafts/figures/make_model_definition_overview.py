#!/usr/bin/env python3
"""Draw a compact AUVHamNODE model-definition overview."""

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
HEIGHT_MM = 78
MM_TO_IN = 1 / 25.4

OUT_DIR = Path(__file__).resolve().parent
OUT_BASE = OUT_DIR / "model_definition_overview"

PALETTE = {
    "ink": "#24272B",
    "muted": "#60646B",
    "rule": "#D7DDE4",
    "paper": "#FFFFFF",
    "blue": "#2E6EBA",
    "blue_pale": "#EEF5FC",
    "teal": "#2F8E86",
    "teal_pale": "#EDF7F4",
    "gold": "#B98524",
    "gold_pale": "#FBF3DF",
    "gray_pale": "#F4F6F8",
}


def pick_font() -> str:
    candidates = ["Arial", "Helvetica", "DejaVu Sans", "Arial Unicode MS"]
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
            "font.size": 6.7,
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
    size: float = 6.6,
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
    radius: float = 0.012,
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
    body: str,
    face: str,
    edge: str,
    title_size: float = 6.9,
    body_size: float = 6.3,
    title_y: float = 0.66,
    body_y: float = 0.34,
) -> None:
    rounded_box(ax, x, y, w, h, face=face, edge=edge, lw=0.85)
    add_text(ax, x + w / 2, y + h * title_y, title, size=title_size, weight="bold", color=edge)
    add_text(ax, x + w / 2, y + h * body_y, body, size=body_size, color=PALETTE["ink"])


def draw() -> None:
    setup_style()
    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    add_text(ax, 0.055, 0.955, "AUVHamNODE model definition", size=8.2, weight="bold", ha="left")
    add_text(
        ax,
        0.945,
        0.955,
        "single integration window",
        size=6.2,
        color=PALETTE["muted"],
        ha="right",
    )

    # Data/model state boundary.
    labeled_box(
        ax,
        x=0.055,
        y=0.725,
        w=0.195,
        h=0.120,
        title="Data state",
        body=r"$s$",
        face=PALETTE["blue_pale"],
        edge=PALETTE["blue"],
        body_size=7.2,
    )
    labeled_box(
        ax,
        x=0.325,
        y=0.725,
        w=0.270,
        h=0.120,
        title="Model state",
        body=r"$y$",
        face=PALETTE["teal_pale"],
        edge=PALETTE["teal"],
        body_size=7.2,
    )
    labeled_box(
        ax,
        x=0.745,
        y=0.725,
        w=0.200,
        h=0.120,
        title="Output state",
        body=r"$\hat{s}$",
        face=PALETTE["blue_pale"],
        edge=PALETTE["blue"],
        body_size=7.2,
    )

    arrow(ax, (0.252, 0.785), (0.322, 0.785), color=PALETTE["blue"])
    add_text(ax, 0.287, 0.822, r"$\mathcal{T}_{d\to m}$", size=6.3, color=PALETTE["blue"])
    arrow(ax, (0.597, 0.785), (0.742, 0.785), color=PALETTE["blue"])
    add_text(ax, 0.670, 0.822, r"$\mathcal{T}_{m\to d}$", size=6.3, color=PALETTE["blue"])

    # Vector-field container.
    rounded_box(
        ax,
        0.060,
        0.175,
        0.885,
        0.465,
        face=PALETTE["paper"],
        edge=PALETTE["rule"],
        radius=0.018,
        lw=0.9,
        zorder=0,
    )
    add_text(ax, 0.082, 0.606, "continuous-time vector field", size=7.4, weight="bold", ha="left")
    add_text(ax, 0.925, 0.606, r"$\dot{y}=F_\theta(y)$", size=7.4, weight="bold", color=PALETTE["muted"], ha="right")

    labeled_box(
        ax,
        x=0.095,
        y=0.435,
        w=0.185,
        h=0.105,
        title="SE(3) kinematics",
        body="pose flow",
        face=PALETTE["blue_pale"],
        edge=PALETTE["blue"],
    )
    labeled_box(
        ax,
        x=0.330,
        y=0.435,
        w=0.185,
        h=0.105,
        title="Mechanical storage",
        body="mass + storage",
        face=PALETTE["teal_pale"],
        edge=PALETTE["teal"],
    )
    labeled_box(
        ax,
        x=0.565,
        y=0.435,
        w=0.185,
        h=0.105,
        title="Force branches",
        body="potential, damping\nzero-power, input",
        face=PALETTE["gold_pale"],
        edge=PALETTE["gold"],
        body_size=5.4,
        title_y=0.72,
        body_y=0.30,
    )

    labeled_box(
        ax,
        x=0.095,
        y=0.245,
        w=0.185,
        h=0.105,
        title="Actuator lag",
        body="command to state",
        face=PALETTE["gray_pale"],
        edge=PALETTE["muted"],
        body_size=5.9,
    )
    labeled_box(
        ax,
        x=0.330,
        y=0.245,
        w=0.185,
        h=0.105,
        title="Carried context",
        body="context $c$\ncurrent, depth",
        face=PALETTE["gray_pale"],
        edge=PALETTE["muted"],
        body_size=5.4,
        title_y=0.72,
        body_y=0.30,
    )
    labeled_box(
        ax,
        x=0.565,
        y=0.245,
        w=0.185,
        h=0.105,
        title="Momentum flow",
        body="momentum to velocity",
        face=PALETTE["teal_pale"],
        edge=PALETTE["teal"],
        body_size=5.9,
    )

    labeled_box(
        ax,
        x=0.790,
        y=0.315,
        w=0.115,
        h=0.150,
        title="ODE flow",
        body="model trajectory\n$\\hat{y}(t)$",
        face=PALETTE["paper"],
        edge=PALETTE["ink"],
        title_size=6.8,
        body_size=5.9,
        title_y=0.68,
        body_y=0.31,
    )

    # Internal module arrows. Dashed arrows indicate conditioning inputs.
    arrow(ax, (0.280, 0.488), (0.327, 0.488), color=PALETTE["muted"])
    arrow(ax, (0.515, 0.488), (0.562, 0.488), color=PALETTE["muted"])
    arrow(ax, (0.705, 0.435), (0.705, 0.353), color=PALETTE["muted"])
    arrow(ax, (0.750, 0.298), (0.788, 0.360), color=PALETTE["muted"])
    arrow(ax, (0.280, 0.350), (0.562, 0.465), color=PALETTE["muted"], linestyle=(0, (3, 3)))
    arrow(ax, (0.515, 0.350), (0.625, 0.437), color=PALETTE["muted"], linestyle=(0, (3, 3)))

    # State to vector field and vector field to output.
    arrow(ax, (0.448, 0.724), (0.448, 0.642), color=PALETTE["teal"])
    arrow(ax, (0.848, 0.464), (0.848, 0.722), color=PALETTE["blue"])

    for suffix in ("pdf", "svg", "png"):
        out = OUT_BASE.with_suffix(f".{suffix}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(out, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    draw()
