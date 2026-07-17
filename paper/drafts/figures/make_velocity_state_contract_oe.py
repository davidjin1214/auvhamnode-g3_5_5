#!/usr/bin/env python3
"""Ocean Engineering manuscript variant of the velocity-state contract schematic.

Copy of ``make_velocity_state_contract.py`` with the state-vector expansions
written in the manuscript's notation (fix 2026-07-17, review item B-8.3): the
first element is the inertial position $p^n$, matching manuscript Eqs. (7)/(8),
whereas the thesis chapter writes the same element as $x$. The .tiff export is
dropped (the manuscript embeds the vector PDF only). Writes
velocity_state_contract_oe.* (copied into the manuscript repository as
velocity_state_contract.pdf).
"""

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
OUT_BASE = OUT_DIR / "velocity_state_contract_oe"

PALETTE = {
    "ink": "#252525",
    "muted": "#60646B",
    "rule": "#D8DDE3",
    "paper": "#FFFFFF",
    "state": "#2E5E8C",
    "state_pale": "#F4F8FB",
}

TITLE_SIZE = 6.9
BODY_SIZE = 5.9
MATH_SIZE = 7.0
LABEL_SIZE = 5.9


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
        "font.size": BODY_SIZE,
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
    size: float = BODY_SIZE,
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
    rounded_box(ax, x, y, w, h, face=style.face, edge=style.edge, lw=0.85)
    add_text(ax, x + 0.030, y + h - 0.044, title, size=TITLE_SIZE, weight="bold", ha="left")
    add_text(ax, x + w * 0.48, y + h * 0.52, equation, size=MATH_SIZE)
    add_text(ax, x + 0.030, y + 0.040, role, size=LABEL_SIZE, color=PALETTE["muted"], ha="left")

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
        lw=0.70,
        zorder=3,
    )
    add_text(ax, pill_x + pill_w / 2, pill_y + pill_h / 2, velocity, size=MATH_SIZE, weight="bold", color=style.edge)


def conversion_box(ax, *, x: float, y: float, w: float, h: float) -> None:
    rounded_box(ax, x, y, w, h, face=PALETTE["paper"], edge=PALETTE["rule"], radius=0.012, lw=0.75, zorder=3)

    add_text(ax, x + 0.035, y + h * 0.70, r"$\mathcal{T}_{d\to m}$", size=MATH_SIZE, weight="bold", ha="left")
    add_text(ax, x + 0.155, y + h * 0.70, r"$\nu_r=\nu_b-\Delta_c$", size=MATH_SIZE, ha="left")
    add_text(ax, x + 0.035, y + h * 0.34, r"$\mathcal{T}_{m\to d}$", size=MATH_SIZE, weight="bold", ha="left")
    add_text(ax, x + 0.155, y + h * 0.34, r"$\nu_b=\nu_r+\Delta_c$", size=MATH_SIZE, ha="left")

    add_text(
        ax,
        x + w - 0.035,
        y + h * 0.52,
        r"$\Delta_c(R,v_c^n)=[R^\top v_c^n;\,0]$",
        size=LABEL_SIZE,
        color=PALETTE["muted"],
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
        style=StateStyle(PALETTE["state_pale"], PALETTE["state"]),
        title="Data / evaluation space",
        equation=r"$s=[p^n,R,\nu_b,u_a,u_c,v_c^n,z_{\rm ref}]$",
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
        style=StateStyle(PALETTE["state_pale"], PALETTE["state"]),
        title="ODE model space",
        equation=r"$y=[p^n,R,\nu_r,u_a,u_c,v_c^n,z_{\rm ref}]$",
        role=r"hydrodynamic branch, velocity loss",
        velocity=r"$\nu_r$",
    )

    x_mid = 0.500
    arrow(ax, (x_mid - 0.040, data_y), (x_mid - 0.040, conv_y + conv_h), color=PALETTE["state"])
    arrow(ax, (x_mid - 0.040, conv_y), (x_mid - 0.040, model_y + box_h), color=PALETTE["state"])
    arrow(ax, (x_mid + 0.040, model_y + box_h), (x_mid + 0.040, conv_y), color=PALETTE["state"])
    arrow(ax, (x_mid + 0.040, conv_y + conv_h), (x_mid + 0.040, data_y), color=PALETTE["state"])

    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
    ]:
        fig.savefig(OUT_BASE.with_suffix(suffix), bbox_inches="tight", pad_inches=0.01, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    draw()
