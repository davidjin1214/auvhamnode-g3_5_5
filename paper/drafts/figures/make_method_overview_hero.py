#!/usr/bin/env python3
"""AUVHamNODE core-method hero figure for §1.5 (merged model overview).

Composition: top contract loop (s -> y -> [field] -> y_hat(t) -> s_hat),
center structured vector field with the open six-DOF mechanical subsystem
(7 modules, 3-class colour coding) and exogenous variables inside the field
container but outside the dashed mechanical-subsystem boundary.
Vertical spacing enlarged so the in-core assembly arrows are clearly visible.
"""

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
HEIGHT_MM = 124
MM_TO_IN = 1 / 25.4

OUT_BASE = Path(__file__).resolve().parent / "method_overview_hero"

PALETTE = {
    "ink": "#24272B", "muted": "#60646B", "rule": "#D7DDE4", "paper": "#FFFFFF",
    "state": "#2E5E8C", "state_pale": "#F4F8FB",
    "power": "#9A7418", "power_pale": "#FBF7E8", "aux_pale": "#F6F7F8",
}

TITLE_SIZE = 6.9
BODY_SIZE = 5.9
MATH_SIZE = 7.0
LABEL_SIZE = 5.9
SMALL_SIZE = 5.2

CORE_DASH = (0, (4, 3))
COND_DASH = (0, (3, 2))


def pick_font() -> str:
    candidates = ["Arial", "Helvetica", "DejaVu Sans", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [pick_font(), "Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans", "font.size": BODY_SIZE,
        "axes.unicode_minus": False, "savefig.facecolor": "white",
    })


def add_text(ax, x, y, text, *, size=BODY_SIZE, weight="regular",
             color=PALETTE["ink"], ha="center", va="center", linespacing=1.12):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, fontweight=weight,
            color=color, linespacing=linespacing)


def rounded_box(ax, x, y, w, h, *, face, edge, radius=0.010, lw=0.85,
                linestyle="-", zorder=1):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.006,rounding_size={radius}",
                       linewidth=lw, linestyle=linestyle, edgecolor=edge,
                       facecolor=face, mutation_aspect=1, zorder=zorder)
    ax.add_patch(p)
    return p


def arrow(ax, start, end, *, color=PALETTE["muted"], lw=0.85, scale=8.0,
          linestyle="-", connectionstyle=None, zorder=4):
    p = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=scale,
                        linewidth=lw, color=color, linestyle=linestyle,
                        shrinkA=2, shrinkB=2, connectionstyle=connectionstyle,
                        zorder=zorder)
    ax.add_patch(p)
    return p


def module_box(ax, *, x, y, w, h, title, tag, learned, subtitle=""):
    rounded_box(ax, x, y, w, h, face=PALETTE["state_pale"], edge=PALETTE["state"],
                lw=0.85, zorder=2)
    ty = 0.70 if subtitle else 0.64
    add_text(ax, x + w / 2, y + h * ty, title, size=BODY_SIZE, weight="bold",
             color=PALETTE["ink"])
    if subtitle:
        add_text(ax, x + w / 2, y + h * 0.46, subtitle, size=SMALL_SIZE,
                 color=PALETTE["muted"])
    tag_w, tag_h = w * 0.90, h * 0.30
    tag_x = x + (w - tag_w) / 2
    tag_y = y + h * 0.10
    if learned:
        rounded_box(ax, tag_x, tag_y, tag_w, tag_h, face=PALETTE["power_pale"],
                    edge=PALETTE["power"], radius=0.020, lw=0.70, zorder=3)
        add_text(ax, x + w / 2, tag_y + tag_h / 2, tag, size=SMALL_SIZE,
                 weight="bold", color=PALETTE["power"])
    else:
        add_text(ax, x + w / 2, tag_y + tag_h / 2, tag, size=SMALL_SIZE,
                 color=PALETTE["muted"])


# ---- geometry -------------------------------------------------------------
BAND_Y, BAND_H = 0.790, 0.082
FIELD_X, FIELD_Y, FIELD_W, FIELD_H = 0.060, 0.185, 0.880, 0.485
FIELD_TOP = FIELD_Y + FIELD_H            # 0.670
ENTRY_X = 0.430
EXIT_X = 0.860


def draw_contract_band(ax):
    def box(x, w, title, sym, note):
        rounded_box(ax, x, BAND_Y, w, BAND_H, face=PALETTE["state_pale"],
                    edge=PALETTE["state"], lw=0.85, zorder=2)
        add_text(ax, x + w / 2, BAND_Y + BAND_H * 0.74, title, size=BODY_SIZE,
                 weight="bold", color=PALETTE["ink"])
        add_text(ax, x + w / 2, BAND_Y + BAND_H * 0.45, sym, size=MATH_SIZE - 0.4,
                 color=PALETTE["ink"])
        add_text(ax, x + w / 2, BAND_Y + BAND_H * 0.16, note,
                 size=SMALL_SIZE - 0.3, color=PALETTE["muted"],
                 linespacing=0.92)

    box(0.030, 0.215, "Data space", r"$s$",
        "absolute generalized\n" + r"velocity $\nu_b$")
    box(0.300, 0.235, "Augmented state", r"$y$",
        "water-relative generalized\n" + r"velocity $\nu_r$")
    box(0.755, 0.215, "Output", r"$\hat s$",
        "absolute generalized\n" + r"velocity $\nu_b$")

    arrow(ax, (0.245, BAND_Y + BAND_H * 0.5), (0.300, BAND_Y + BAND_H * 0.5),
          color=PALETTE["state"])
    add_text(ax, 0.272, BAND_Y + BAND_H + 0.016, r"$\mathcal{T}_{d\to m}$",
             size=LABEL_SIZE, color=PALETTE["muted"])

    arrow(ax, (ENTRY_X, BAND_Y), (ENTRY_X, FIELD_TOP), color=PALETTE["state"],
          lw=1.4, scale=9.0)

    arrow(ax, (EXIT_X, FIELD_TOP), (EXIT_X, BAND_Y), color=PALETTE["state"],
          lw=1.2, scale=8.5)
    add_text(ax, EXIT_X + 0.014, FIELD_TOP + 0.038, r"$\int\!\rightarrow\hat y(t)$",
             size=SMALL_SIZE, color=PALETTE["muted"], ha="left")
    add_text(ax, EXIT_X + 0.014, BAND_Y - 0.026, r"$\mathcal{T}_{m\to d}$",
             size=LABEL_SIZE, color=PALETTE["muted"], ha="left")
    add_text(ax, EXIT_X - 0.012, (FIELD_TOP + BAND_Y) / 2, "integrate\nover window",
             size=SMALL_SIZE - 0.6, color=PALETTE["muted"], ha="right")


def draw_vector_field(ax):
    rounded_box(ax, FIELD_X, FIELD_Y, FIELD_W, FIELD_H, face=PALETTE["paper"],
                edge=PALETTE["rule"], radius=0.014, lw=0.95, zorder=0)
    add_text(ax, FIELD_X + 0.022, FIELD_TOP - 0.020,
             "Structured continuous-time vector field", size=TITLE_SIZE,
             weight="bold", ha="left")
    add_text(ax, FIELD_X + FIELD_W - 0.018, FIELD_TOP - 0.020,
             r"$\dot y=F_\theta(y)$", size=MATH_SIZE, color=PALETTE["muted"],
             ha="right")

    mx, my, mw, mh = 0.085, 0.300, 0.830, 0.330
    rounded_box(ax, mx, my, mw, mh, face=PALETTE["paper"], edge=PALETTE["ink"],
                radius=0.012, lw=1.0, linestyle=CORE_DASH, zorder=0)
    add_text(ax, mx + 0.018, my + mh - 0.020,
             "Open six-DOF mechanical subsystem",
             size=BODY_SIZE, weight="bold", color=PALETTE["ink"], ha="left")

    bw, bh = 0.165, 0.075
    col_x = (0.105, 0.288, 0.471)
    row_top, row_bot = 0.505, 0.380

    module_box(ax, x=col_x[0], y=row_top, w=bw, h=bh, title="SE(3) kinematics",
               tag="pose flow", learned=False)
    module_box(ax, x=col_x[1], y=row_top, w=bw, h=bh, title="Inverse mass",
               tag=r"$M_\theta^{-1}\!\succ\!0$", learned=True)
    module_box(ax, x=col_x[2], y=row_top, w=bw, h=bh, title="Coadjoint coupling",
               tag="zero-power", learned=False)
    module_box(ax, x=col_x[0], y=row_bot, w=bw, h=bh, title="Conservative force",
               tag=r"$V_\theta\!\to\!f_\theta^{V}$", learned=True)
    module_box(ax, x=col_x[1], y=row_bot, w=bw, h=bh, title="Dissipation",
               tag=r"$D_\theta\!\succ\!0$", learned=True)
    module_box(ax, x=col_x[2], y=row_bot, w=bw, h=bh, title="Skew coupling",
               tag=r"$J_\theta\!=\!-J_\theta^{\top}$", learned=True)

    fp_x, fp_y = 0.665, row_bot
    fp_w, fp_h = 0.250, (row_top + bh) - row_bot
    rounded_box(ax, fp_x, fp_y, fp_w, fp_h, face=PALETTE["state_pale"],
                edge=PALETTE["state"], lw=1.5, zorder=2)
    add_text(ax, fp_x + fp_w / 2, fp_y + fp_h * 0.72,
             "External generalized force",
             size=BODY_SIZE, weight="bold", color=PALETTE["ink"])
    add_text(ax, fp_x + fp_w / 2, fp_y + fp_h * 0.50,
             r"mechanical power $\nu_r^\top\tau_\theta$",
             size=SMALL_SIZE, color=PALETTE["muted"])
    tw, th = fp_w * 0.80, 0.044
    tx = fp_x + (fp_w - tw) / 2
    ty2 = fp_y + fp_h * 0.13
    rounded_box(ax, tx, ty2, tw, th, face=PALETTE["power_pale"],
                edge=PALETTE["power"], radius=0.016, lw=0.70, zorder=3)
    add_text(ax, fp_x + fp_w / 2, ty2 + th / 2, r"$\tau_\theta\!\neq\!G(q)u$",
             size=SMALL_SIZE, weight="bold", color=PALETTE["power"])

    # in-core assembly flow (now clearly visible)
    flow = dict(color=PALETTE["state"], lw=0.85, scale=6.5, zorder=4)
    coll_y = 0.345
    for cx in col_x:
        arrow(ax, (cx + bw / 2, row_top), (cx + bw / 2, row_bot + bh), **flow)
        arrow(ax, (cx + bw / 2, row_bot), (cx + bw / 2, coll_y), **flow)
    arrow(ax, (fp_x + fp_w / 2, fp_y), (fp_x + fp_w / 2, coll_y), **flow)

    # Exogenous-variable band inside the field container but outside the
    # mechanical-subsystem boundary.
    band_y, band_h = 0.205, 0.060
    ports = [
        (0.105, 0.165, "Actuator lag", r"first-order  $u_c\!\to\!u_a$"),
        (0.380, 0.235, r"Ocean current $v_c^n$", "piecewise-constant"),
        (0.700, 0.215, r"Depth context $z_{\mathrm{ref}}$", "piecewise-constant"),
    ]
    centres = []
    for px, pw, title, note in ports:
        rounded_box(ax, px, band_y, pw, band_h, face=PALETTE["aux_pale"],
                    edge=PALETTE["muted"], lw=0.85, zorder=2)
        add_text(ax, px + pw / 2, band_y + band_h * 0.64, title, size=BODY_SIZE,
                 weight="bold", color=PALETTE["ink"])
        add_text(ax, px + pw / 2, band_y + band_h * 0.26, note, size=SMALL_SIZE,
                 color=PALETTE["muted"])
        centres.append(px + pw / 2)

    cond = dict(color=PALETTE["muted"], lw=0.9, linestyle=COND_DASH, scale=7.0,
                zorder=5)
    for cx in centres:
        arrow(ax, (cx, band_y + band_h), (cx, my + 0.010), **cond)


def draw_legend(ax):
    rule_y = 0.150
    ax.plot([0.030, 0.970], [rule_y, rule_y], color=PALETTE["rule"], lw=0.7,
            zorder=0)
    sw = 0.022
    y1 = 0.108
    items1 = [
        (0.030, PALETTE["state_pale"], PALETTE["state"],
         "structural prior (construction-guaranteed)"),
        (0.400, PALETTE["power_pale"], PALETTE["power"],
         "learned shape (structure-preserving)"),
        (0.730, PALETTE["aux_pale"], PALETTE["muted"],
         "exogenous variable or state"),
    ]
    for x, face, edge, label in items1:
        rounded_box(ax, x, y1 - sw / 2, sw, sw, face=face, edge=edge,
                    radius=0.004, lw=0.85, zorder=2)
        add_text(ax, x + sw + 0.010, y1, label, size=SMALL_SIZE,
                 color=PALETTE["ink"], ha="left")

    y2 = 0.052
    rounded_box(ax, 0.030, y2 - sw / 2, sw, sw, face=PALETTE["paper"],
                edge=PALETTE["ink"], radius=0.004, lw=0.9, linestyle=CORE_DASH,
                zorder=2)
    add_text(ax, 0.030 + sw + 0.010, y2,
             "dashed box = open six-DOF mechanical subsystem",
             size=SMALL_SIZE, color=PALETTE["ink"], ha="left")
    arrow(ax, (0.520, y2), (0.560, y2), color=PALETTE["muted"], lw=0.9,
          linestyle=COND_DASH, scale=7.0, zorder=2)
    add_text(ax, 0.572, y2, "dashed arrow = context input", size=SMALL_SIZE,
             color=PALETTE["ink"], ha="left")


def draw():
    setup_style()
    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN),
                     facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    draw_contract_band(ax)
    draw_vector_field(ax)
    draw_legend(ax)
    for suffix, kw in [(".png", {"dpi": 400}), (".pdf", {}), (".svg", {})]:
        fig.savefig(OUT_BASE.with_suffix(suffix), bbox_inches="tight",
                    pad_inches=0.02, **kw)
    plt.close(fig)


if __name__ == "__main__":
    draw()
