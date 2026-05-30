#!/usr/bin/env python3
"""Draw the mechanical-core power schematic for the AUVHamNODE chapter."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/auvhamnode_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/auvhamnode_xdg_cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


WIDTH_MM = 145
HEIGHT_MM = 84
MM_TO_IN = 1 / 25.4

OUT_DIR = Path(__file__).resolve().parent
OUT_BASE = OUT_DIR / "mechanical_core_power_structure"

PALETTE = {
    "ink": "#252525",
    "muted": "#62666D",
    "rule": "#D8DDE3",
    "paper": "#FFFFFF",
    "core": "#2E6EBA",
    "core_pale": "#EEF5FC",
    "store": "#2F8E86",
    "store_pale": "#EDF7F4",
    "diss": "#A95B3A",
    "diss_pale": "#F8EFEA",
    "zero": "#6B6F7A",
    "zero_pale": "#F3F4F6",
    "port": "#C78F2D",
    "port_pale": "#FBF4E5",
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
    size: float = 6.5,
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
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
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
    scale: float = 8.5,
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
    title_color: str | None = None,
    title_size: float = 6.7,
    body_size: float = 6.3,
) -> None:
    rounded_box(ax, x, y, w, h, face=face, edge=edge, lw=0.9)
    add_text(
        ax,
        x + 0.018,
        y + h - 0.030,
        title,
        size=title_size,
        weight="bold",
        color=title_color or edge,
        ha="left",
    )
    add_text(ax, x + w / 2, y + h * 0.34, body, size=body_size)


def draw() -> None:
    setup_style()
    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    core_x, core_y, core_w, core_h = 0.055, 0.140, 0.650, 0.780
    rounded_box(
        ax,
        core_x,
        core_y,
        core_w,
        core_h,
        face=PALETTE["paper"],
        edge=PALETTE["core"],
        radius=0.018,
        lw=1.0,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    add_text(
        ax,
        core_x + 0.022,
        core_y + core_h - 0.035,
        "Six-DOF mechanical core (open subsystem)",
        size=7.3,
        weight="bold",
        color=PALETTE["core"],
        ha="left",
    )

    labeled_box(
        ax,
        x=0.090,
        y=0.690,
        w=0.250,
        h=0.140,
        title="State pairing",
        body=r"$q=(x,R)$" + "\n" + r"$p_r=M_\theta\nu_r$",
        face=PALETTE["core_pale"],
        edge=PALETTE["core"],
        body_size=7.0,
    )
    labeled_box(
        ax,
        x=0.375,
        y=0.690,
        w=0.285,
        h=0.140,
        title="Storage",
        body=r"$H_\theta=K_\theta+V_\theta$" + "\n" + r"$\partial H_\theta/\partial p_r=\nu_r$",
        face=PALETTE["store_pale"],
        edge=PALETTE["store"],
        body_size=6.8,
    )

    role_specs = [
        (
            0.105,
            0.500,
            0.250,
            0.125,
            "Coordinate coupling",
            r"$\operatorname{ad}^{*}_{\nu_r}p_r$" + "\n" + r"$\nu_r^\top(\cdot)=0$",
            PALETTE["zero_pale"],
            PALETTE["zero"],
        ),
        (
            0.395,
            0.500,
            0.250,
            0.125,
            "Conservative force",
            r"$f_\theta^{V}(q)$" + "\n" + r"$\nu_r^\top f_\theta^V=-\dot V_\theta$",
            PALETTE["store_pale"],
            PALETTE["store"],
        ),
        (
            0.105,
            0.340,
            0.250,
            0.125,
            "Dissipation",
            r"$-D_\theta(\xi)\nu_r$" + "\n" + r"$-\nu_r^\top D_\theta\nu_r\leq0$",
            PALETTE["diss_pale"],
            PALETTE["diss"],
        ),
        (
            0.395,
            0.340,
            0.250,
            0.125,
            "Zero-power lift",
            r"$J_\theta(\xi)\nu_r$" + "\n" + r"$J_\theta=-J_\theta^\top$",
            PALETTE["zero_pale"],
            PALETTE["zero"],
        ),
    ]
    for spec in role_specs:
        labeled_box(
            ax,
            x=spec[0],
            y=spec[1],
            w=spec[2],
            h=spec[3],
            title=spec[4],
            body=spec[5],
            face=spec[6],
            edge=spec[7],
            title_size=6.1,
            body_size=5.6,
        )

    rounded_box(ax, 0.135, 0.190, 0.485, 0.105, face=PALETTE["port_pale"], edge=PALETTE["port"], lw=0.95)
    add_text(
        ax,
        0.377,
        0.247,
        r"$\dot H_\theta=-\nu_r^\top D_\theta(\xi)\nu_r+\nu_r^\top\tau_\theta$",
        size=7.1,
        weight="bold",
        color=PALETTE["ink"],
    )
    add_text(ax, 0.377, 0.208, "hydrostatic continuous-time power balance", size=5.7, color=PALETTE["muted"])

    labeled_box(
        ax,
        x=0.770,
        y=0.690,
        w=0.180,
        h=0.135,
        title="Carried context",
        body=r"$v_c^n,\ z_{\rm ref}$" + "\n" + "no stored energy",
        face=PALETTE["zero_pale"],
        edge=PALETTE["zero"],
        title_size=6.2,
        body_size=5.8,
    )
    labeled_box(
        ax,
        x=0.770,
        y=0.500,
        w=0.180,
        h=0.135,
        title="Actuator lag",
        body=r"$u_c\rightarrow u_a$" + "\n" + r"$\dot u_a=T_\theta^{-1}(u_c-u_a)$",
        face=PALETTE["core_pale"],
        edge=PALETTE["core"],
        title_size=6.2,
        body_size=5.7,
    )
    labeled_box(
        ax,
        x=0.760,
        y=0.285,
        w=0.200,
        h=0.140,
        title="Power port",
        body=r"$\tau_\theta(\nu_r,u_a,c_\tau)$" + "\n" + r"$\nu_r^\top\tau_\theta$",
        face=PALETTE["port_pale"],
        edge=PALETTE["port"],
        title_size=6.2,
        body_size=6.1,
    )
    rounded_box(ax, 0.755, 0.095, 0.205, 0.095, face=PALETTE["paper"], edge=PALETTE["rule"], lw=0.75)
    add_text(
        ax,
        0.858,
        0.145,
        "outside the core:\nconditioning, not closure",
        size=5.8,
        color=PALETTE["muted"],
    )

    arrow(ax, (0.340, 0.760), (0.375, 0.760), color=PALETTE["core"])
    arrow(ax, (0.250, 0.690), (0.250, 0.625), color=PALETTE["muted"])
    arrow(ax, (0.520, 0.690), (0.520, 0.625), color=PALETTE["store"])
    arrow(ax, (0.355, 0.570), (0.395, 0.570), color=PALETTE["muted"])
    arrow(ax, (0.355, 0.415), (0.395, 0.415), color=PALETTE["muted"])
    arrow(ax, (0.250, 0.340), (0.305, 0.295), color=PALETTE["diss"])
    arrow(ax, (0.520, 0.340), (0.460, 0.295), color=PALETTE["zero"])
    arrow(ax, (0.770, 0.568), (0.705, 0.410), color=PALETTE["core"], linestyle=(0, (3, 2)))
    arrow(ax, (0.770, 0.755), (0.705, 0.430), color=PALETTE["zero"], linestyle=(0, (3, 2)))
    arrow(ax, (0.860, 0.500), (0.860, 0.425), color=PALETTE["core"])
    arrow(ax, (0.760, 0.355), (0.620, 0.245), color=PALETTE["port"])

    add_text(ax, 0.705, 0.460, "features", size=5.6, color=PALETTE["muted"], ha="right")
    add_text(ax, 0.685, 0.275, "generalized force", size=5.6, color=PALETTE["port"], ha="left")

    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
    ]:
        fig.savefig(OUT_BASE.with_suffix(suffix), bbox_inches="tight", pad_inches=0.01, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    draw()
