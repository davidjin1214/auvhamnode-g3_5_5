#!/usr/bin/env python3
"""Draw the velocity-state contract schematic for the AUVHamNODE chapter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/auvhamnode_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/auvhamnode_xdg_cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


WIDTH_MM = 138
HEIGHT_MM = 54
MM_TO_IN = 1 / 25.4

OUT_DIR = Path(__file__).resolve().parent
OUT_BASE = OUT_DIR / "velocity_state_contract"

PALETTE = {
    "ink": "#252525",
    "muted": "#60646B",
    "rule": "#D8DDE3",
    "paper": "#FFFFFF",
    "blue": "#2E6EBA",
    "blue_pale": "#EEF5FC",
    "teal": "#2F8E86",
    "teal_pale": "#EDF7F4",
    "current": "#C78F2D",
}


@dataclass(frozen=True)
class StateStyle:
    face: str
    edge: str


def pick_font() -> str:
    """Prefer fonts with reliable math rendering in vector exports."""
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
            "axes.linewidth": 0.6,
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
    linespacing: float = 1.15,
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
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
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
    color: str,
    lw: float = 0.9,
    scale: float = 8.5,
    zorder: int = 2,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def state_box(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    style: StateStyle,
    title: str,
    equation: str,
    role: str,
    velocity: str,
) -> None:
    rounded_box(ax, x, y, w, h, face=style.face, edge=style.edge, lw=0.95)
    add_text(ax, x + 0.030, y + h - 0.044, title, size=7.0, weight="bold", color=style.edge, ha="left")
    add_text(ax, x + w * 0.48, y + h * 0.52, equation, size=7.7)
    add_text(ax, x + 0.030, y + 0.040, role, size=5.8, color=PALETTE["muted"], ha="left")

    pill_w, pill_h = 0.130, 0.055
    pill_x = x + w - pill_w - 0.028
    pill_y = y + h * 0.50 - pill_h / 2
    rounded_box(
        ax,
        pill_x,
        pill_y,
        pill_w,
        pill_h,
        face=PALETTE["paper"],
        edge=style.edge,
        radius=0.018,
        lw=0.75,
        zorder=3,
    )
    add_text(ax, pill_x + pill_w / 2, pill_y + pill_h / 2, velocity, size=7.1, weight="bold", color=style.edge)


def conversion_box(ax, *, x: float, y: float, w: float, h: float) -> None:
    rounded_box(ax, x, y, w, h, face=PALETTE["paper"], edge=PALETTE["rule"], radius=0.012, lw=0.75, zorder=3)

    add_text(ax, x + 0.035, y + h * 0.70, r"$\mathcal{T}_{d\to m}$", size=6.9, weight="bold", color=PALETTE["blue"], ha="left")
    add_text(ax, x + 0.155, y + h * 0.70, r"$\nu_r=\nu_b-\Delta_c$", size=7.1, ha="left")
    add_text(ax, x + 0.035, y + h * 0.34, r"$\mathcal{T}_{m\to d}$", size=6.9, weight="bold", color=PALETTE["teal"], ha="left")
    add_text(ax, x + 0.155, y + h * 0.34, r"$\nu_b=\nu_r+\Delta_c$", size=7.1, ha="left")

    add_text(
        ax,
        x + w - 0.035,
        y + h * 0.52,
        r"$\Delta_c(R,v_c^n)=[R^\top v_c^n;\,0]$",
        size=6.6,
        color=PALETTE["current"],
        ha="right",
    )


def draw() -> None:
    setup_style()
    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    left_x, box_w = 0.070, 0.860
    box_h = 0.245
    data_y, model_y = 0.675, 0.080
    conv_x, conv_y, conv_w, conv_h = 0.190, 0.390, 0.620, 0.180

    state_box(
        ax,
        x=left_x,
        y=data_y,
        w=box_w,
        h=box_h,
        style=StateStyle(PALETTE["blue_pale"], PALETTE["blue"]),
        title="Data / evaluation space",
        equation=r"$s=[x,R,\nu_b,u_a,u_c,v_c^n,z_{\rm ref}]$",
        role=r"pose kinematics, rollout metrics",
        velocity=r"$\nu_b$",
    )

    conversion_box(ax, x=conv_x, y=conv_y, w=conv_w, h=conv_h)

    state_box(
        ax,
        x=left_x,
        y=model_y,
        w=box_w,
        h=box_h,
        style=StateStyle(PALETTE["teal_pale"], PALETTE["teal"]),
        title="ODE model space",
        equation=r"$y=[x,R,\nu_r,u_a,u_c,v_c^n,z_{\rm ref}]$",
        role=r"hydrodynamic branch, velocity loss",
        velocity=r"$\nu_r$",
    )

    x_mid = 0.500
    arrow(ax, (x_mid - 0.040, data_y), (x_mid - 0.040, conv_y + conv_h), color=PALETTE["blue"])
    arrow(ax, (x_mid - 0.040, conv_y), (x_mid - 0.040, model_y + box_h), color=PALETTE["blue"])
    arrow(ax, (x_mid + 0.040, model_y + box_h), (x_mid + 0.040, conv_y), color=PALETTE["teal"])
    arrow(ax, (x_mid + 0.040, conv_y + conv_h), (x_mid + 0.040, data_y), color=PALETTE["teal"])

    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
        (".tiff", {"dpi": 600}),
    ]:
        fig.savefig(OUT_BASE.with_suffix(suffix), bbox_inches="tight", pad_inches=0.01, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    draw()
