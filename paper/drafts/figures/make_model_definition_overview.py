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
    "state": "#2E5E8C",
    "state_pale": "#F4F8FB",
    "power": "#9A7418",
    "power_pale": "#FBF7E8",
    "aux_pale": "#F6F7F8",
}

TITLE_SIZE = 6.9
BODY_SIZE = 5.9
MATH_SIZE = 7.0
LABEL_SIZE = 5.9


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
    title_size: float = TITLE_SIZE,
    body_size: float = BODY_SIZE,
    title_y: float = 0.66,
    body_y: float = 0.34,
) -> None:
    rounded_box(ax, x, y, w, h, face=face, edge=edge, lw=0.85)
    add_text(ax, x + w / 2, y + h * title_y, title, size=title_size, weight="bold")
    add_text(ax, x + w / 2, y + h * body_y, body, size=body_size, color=PALETTE["ink"])


def draw() -> None:
    setup_style()
    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Data/model state boundary.
    labeled_box(
        ax,
        x=0.055,
        y=0.775,
        w=0.195,
        h=0.120,
        title="Data state",
        body=r"$s$",
        face=PALETTE["state_pale"],
        edge=PALETTE["state"],
        body_size=MATH_SIZE,
    )
    labeled_box(
        ax,
        x=0.325,
        y=0.775,
        w=0.270,
        h=0.120,
        title="Model state",
        body=r"$y$",
        face=PALETTE["state_pale"],
        edge=PALETTE["state"],
        body_size=MATH_SIZE,
    )
    labeled_box(
        ax,
        x=0.745,
        y=0.775,
        w=0.200,
        h=0.120,
        title="Output state",
        body=r"$\hat{s}$",
        face=PALETTE["state_pale"],
        edge=PALETTE["state"],
        body_size=MATH_SIZE,
    )

    arrow(ax, (0.252, 0.835), (0.322, 0.835), color=PALETTE["state"])
    add_text(ax, 0.287, 0.872, r"$\mathcal{T}_{d\to m}$", size=LABEL_SIZE, color=PALETTE["muted"])
    arrow(ax, (0.597, 0.835), (0.742, 0.835), color=PALETTE["state"])
    add_text(ax, 0.670, 0.872, r"$\mathcal{T}_{m\to d}$", size=LABEL_SIZE, color=PALETTE["muted"])

    # Vector-field container.
    rounded_box(
        ax,
        0.060,
        0.140,
        0.885,
        0.535,
        face=PALETTE["paper"],
        edge=PALETTE["rule"],
        radius=0.018,
        lw=0.9,
        zorder=0,
    )
    add_text(ax, 0.082, 0.637, "continuous-time vector field", size=TITLE_SIZE, weight="bold", ha="left")
    add_text(ax, 0.925, 0.637, r"$\dot{y}=F_\theta(y)$", size=MATH_SIZE, color=PALETTE["muted"], ha="right")

    labeled_box(
        ax,
        x=0.095,
        y=0.435,
        w=0.185,
        h=0.105,
        title="SE(3) kinematics",
        body="pose flow",
        face=PALETTE["paper"],
        edge=PALETTE["rule"],
    )
    labeled_box(
        ax,
        x=0.330,
        y=0.435,
        w=0.185,
        h=0.105,
        title="Mechanical storage",
        body="mass + storage",
        face=PALETTE["paper"],
        edge=PALETTE["rule"],
    )
    labeled_box(
        ax,
        x=0.565,
        y=0.435,
        w=0.185,
        h=0.105,
        title="Force branches",
        body="potential, damping\nzero-power, input",
        face=PALETTE["power_pale"],
        edge=PALETTE["power"],
        body_size=BODY_SIZE,
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
        face=PALETTE["aux_pale"],
        edge=PALETTE["muted"],
        body_size=BODY_SIZE,
    )
    labeled_box(
        ax,
        x=0.330,
        y=0.245,
        w=0.185,
        h=0.105,
        title="Carried context",
        body="context $c$\ncurrent, depth",
        face=PALETTE["aux_pale"],
        edge=PALETTE["muted"],
        body_size=BODY_SIZE,
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
        face=PALETTE["paper"],
        edge=PALETTE["rule"],
        body_size=BODY_SIZE,
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
        title_size=TITLE_SIZE,
        body_size=BODY_SIZE,
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
    arrow(ax, (0.448, 0.774), (0.448, 0.677), color=PALETTE["state"])
    arrow(ax, (0.848, 0.464), (0.848, 0.772), color=PALETTE["state"])

    for suffix in ("pdf", "svg", "png"):
        out = OUT_BASE.with_suffix(f".{suffix}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(out, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    draw()
