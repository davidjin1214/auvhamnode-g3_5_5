#!/usr/bin/env python3
"""Draw the velocity-state contract schematic for the AUVHamNODE chapter."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


WIDTH_MM = 138
HEIGHT_MM = 82
MM_TO_IN = 1 / 25.4

OUT_DIR = Path(__file__).resolve().parent
OUT_BASE = OUT_DIR / "velocity_state_contract"

COLORS = {
    "ink": "#202124",
    "muted": "#5F6368",
    "data": "#EAF2FB",
    "data_edge": "#6F93B8",
    "model": "#E8F4EF",
    "model_edge": "#5B9B84",
    "output": "#F6ECEB",
    "output_edge": "#B87973",
    "formula": "#F7F7F4",
    "formula_edge": "#9B9A92",
    "accent": "#0F4D92",
}


def pick_font() -> str:
    candidates = [
        "Arial Unicode MS",
        "Hiragino Sans GB",
        "PingFang SC",
        "Heiti TC",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


def setup_style() -> str:
    font_name = pick_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.2,
            "axes.linewidth": 0.6,
            "axes.unicode_minus": False,
        }
    )
    return font_name


def rounded_box(ax, xy, wh, face, edge, radius=0.035, lw=0.9):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=7.2, weight="regular", color=None, ha="center", va="center"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS["ink"],
        linespacing=1.28,
    )


def math_label(ax, x, y, text, size=7.0, color=None, ha="center", va="center"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        color=color or COLORS["ink"],
        linespacing=1.2,
    )


def arrow(ax, start, end, color=None, lw=1.0, rad=0.0, scale=9):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        color=color or COLORS["accent"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(patch)
    return patch


def draw():
    setup_style()
    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Panel label and title.
    label(ax, 0.030, 0.955, "a", size=8.3, weight="bold", ha="left")
    label(ax, 0.075, 0.955, "海流条件下的数据态—模型态—输出态速度契约", size=8.2, weight="bold", ha="left")

    # Layer bands.
    layer_x, layer_w = 0.075, 0.84
    layers = [
        ("数据空间  $\\mathcal{S}_d$", 0.705, COLORS["data"], COLORS["data_edge"]),
        ("模型空间  $\\mathcal{S}_m$", 0.405, COLORS["model"], COLORS["model_edge"]),
        ("输出与评估空间", 0.105, COLORS["output"], COLORS["output_edge"]),
    ]
    for title, y, face, edge in layers:
        rounded_box(ax, (layer_x, y), (layer_w, 0.205), face, edge, radius=0.018, lw=0.8)
        label(ax, layer_x + 0.018, y + 0.178, title, size=7.0, weight="bold", color=edge, ha="left")

    # Data layer.
    rounded_box(ax, (0.115, 0.745), (0.250, 0.095), "white", COLORS["data_edge"], radius=0.018)
    math_label(ax, 0.240, 0.803, r"$s=[x,R,\nu_b,u_a,u_c,v_c^n,z_{\rm ref}]$", size=6.8)
    label(ax, 0.240, 0.772, "块内相对位移  |  总体速度口径", size=6.5, color=COLORS["muted"])

    rounded_box(ax, (0.645, 0.745), (0.225, 0.095), "white", COLORS["data_edge"], radius=0.018)
    math_label(ax, 0.758, 0.804, r"$\nu_b=[v_b,\omega]$", size=7.0)
    label(ax, 0.758, 0.772, "数据保存的体坐标总体速度", size=6.5, color=COLORS["muted"])

    # Transform from data to model.
    rounded_box(ax, (0.405, 0.618), (0.190, 0.085), COLORS["formula"], COLORS["formula_edge"], radius=0.018)
    math_label(ax, 0.500, 0.675, r"$\mathcal{T}_{d\to m}$", size=7.4, color=COLORS["accent"])
    math_label(ax, 0.500, 0.644, r"$\nu_r=\nu_b-[R^\top v_c^n;0]$", size=6.6)
    arrow(ax, (0.365, 0.786), (0.405, 0.663), rad=-0.12)
    arrow(ax, (0.645, 0.786), (0.595, 0.663), rad=0.12)

    # Model layer.
    rounded_box(ax, (0.115, 0.445), (0.250, 0.095), "white", COLORS["model_edge"], radius=0.018)
    math_label(ax, 0.240, 0.504, r"$y=[x,R,\nu_r,u_a,u_c,v_c^n,z_{\rm ref}]$", size=6.8)
    label(ax, 0.240, 0.473, "增强状态携带命令与外源上下文", size=6.5, color=COLORS["muted"])

    rounded_box(ax, (0.415, 0.445), (0.190, 0.095), "white", COLORS["model_edge"], radius=0.018)
    math_label(ax, 0.510, 0.504, r"$\hat y(t)=\Phi_\theta(t;t_0,y_0)$", size=6.8)
    label(ax, 0.510, 0.473, "ODE 在模型空间积分", size=6.5, color=COLORS["muted"])

    rounded_box(ax, (0.655, 0.445), (0.205, 0.095), "white", COLORS["model_edge"], radius=0.018)
    math_label(ax, 0.758, 0.504, r"$f_\theta^{\rm nc}(\cdot)\leftarrow\nu_r$", size=7.0)
    label(ax, 0.758, 0.473, "水动力广义力使用相对水速度", size=6.5, color=COLORS["muted"])

    arrow(ax, (0.500, 0.618), (0.240, 0.540), rad=0.0)
    arrow(ax, (0.365, 0.493), (0.415, 0.493), scale=8)
    arrow(ax, (0.605, 0.493), (0.655, 0.493), scale=8)

    # Transform from model to output.
    rounded_box(ax, (0.405, 0.318), (0.190, 0.085), COLORS["formula"], COLORS["formula_edge"], radius=0.018)
    math_label(ax, 0.500, 0.375, r"$\mathcal{T}_{m\to d}$", size=7.4, color=COLORS["accent"])
    math_label(ax, 0.500, 0.344, r"$\nu_b=\nu_r+[R^\top v_c^n;0]$", size=6.6)
    arrow(ax, (0.510, 0.445), (0.500, 0.403), rad=0.0)

    # Output layer.
    rounded_box(ax, (0.115, 0.145), (0.250, 0.095), "white", COLORS["output_edge"], radius=0.018)
    math_label(ax, 0.240, 0.204, r"$\hat s=[\hat x,\hat R,\hat\nu_b,\hat u_a,\ldots]$", size=6.8)
    label(ax, 0.240, 0.173, "输出恢复为数据空间口径", size=6.5, color=COLORS["muted"])

    rounded_box(ax, (0.415, 0.145), (0.190, 0.095), "white", COLORS["output_edge"], radius=0.018)
    label(ax, 0.510, 0.204, "外部报告指标", size=6.8, weight="bold")
    label(ax, 0.510, 0.174, "位置、姿态、总体速度", size=6.5, color=COLORS["muted"])

    rounded_box(ax, (0.655, 0.145), (0.205, 0.095), "white", COLORS["output_edge"], radius=0.018)
    label(ax, 0.758, 0.204, "模型空间诊断", size=6.8, weight="bold")
    math_label(ax, 0.758, 0.174, r"$\nu_r$ 误差与能量 / $\mathrm{SO}(3)$ 诊断", size=6.3, color=COLORS["muted"])

    arrow(ax, (0.500, 0.318), (0.240, 0.240), rad=0.0)
    arrow(ax, (0.365, 0.193), (0.415, 0.193), scale=8, color=COLORS["output_edge"])
    arrow(ax, (0.605, 0.193), (0.655, 0.193), scale=8, color=COLORS["output_edge"])

    # Right-side physical meaning callout.
    rounded_box(ax, (0.075, 0.020), (0.840, 0.055), "#FFFFFF", "#D6D6D0", radius=0.016, lw=0.7)
    math_label(
        ax,
        0.495,
        0.047,
        r"$\dot x=Rv_b=Rv_r+v_c^n$: 位姿由总体速度推进；"
        r"  $f_\theta^{\rm nc}$: 水动力相关广义力由相对水速度参数化。",
        size=6.7,
    )

    fig.savefig(OUT_BASE.with_suffix(".pdf"))
    fig.savefig(OUT_BASE.with_suffix(".svg"))
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=600)


if __name__ == "__main__":
    draw()
