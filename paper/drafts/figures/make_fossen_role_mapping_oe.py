#!/usr/bin/env python3
"""Ocean Engineering manuscript variant of the Fossen-role mapping (Fig. S1).

Copy of ``make_fossen_role_mapping.py`` with two notation fixes for the OE
submission (2026-07-17, review items A-1/B-8.4): the restoring term and the
explicit force sum print $g(q)$ -- the manuscript's configuration symbol used
in Eqs. (14)/(17) and Supplement Eq. (S1) -- instead of Fossen's $g(\\eta)$,
which the manuscript never defines (the thesis chapter keeps $g(\\eta)$, so the
shared generator is unchanged); and the power-pairing note reads
"(water-relative)" in the manuscript's standard word order. Writes
fossen_role_mapping_oe.* (copied into the manuscript repository as
fossen_role_mapping.pdf).

Original description follows.

A two-column mapping by power properties: classical phenomenological Fossen
terms on the left map to geometric, structure-preserving AUVHamNODE components.
This is a red-line-safe remake that corrects three
errors of an earlier "Fossen -> port-Hamiltonian" sketch:

  1. Lift is NOT conservative hydrodynamics: it is carried by the zero-power
     skew-symmetric coupling J_theta (non-conservative, non-dissipative).
  2. Actuation is NOT a fixed input matrix G(q)u / B u: tau_theta is an
     external generalized-force term, paired with nu_r to define mechanical
     power, with the explicit assertion tau_theta != G(q)u.
  3. No closed port-Hamiltonian form x_dot = (J - R) grad H + G u appears. The
     right column uses the open structured form, and a footnote states that the
     full vehicle-actuator-environment system is open, not a closed pH system.

Two-axis colour semantics match the method-architecture overview: blue edge =
construction-guaranteed structural prior; gold tag = data-learned shape inside a
structure-preserving class. Full applicability boundaries stay in the caption.
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
HEIGHT_MM = 130
MM_TO_IN = 1 / 25.4

OUT_DIR = Path(__file__).resolve().parent
OUT_BASE = OUT_DIR / "fossen_role_mapping_oe"

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
SMALL_SIZE = 5.2

COND_DASH = (0, (3, 2))


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
    radius: float = 0.010,
    lw: float = 0.85,
    linestyle="-",
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
    linestyle="-",
    connectionstyle: str | None = None,
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
        connectionstyle=connectionstyle,
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
    note: str = "",
    face: str,
    edge: str,
    title_size: float = TITLE_SIZE,
    body_size: float = BODY_SIZE,
    note_size: float = SMALL_SIZE,
    title_y: float = 0.66,
    body_y: float = 0.34,
    note_y: float = 0.13,
    lw: float = 0.85,
    linestyle="-",
    zorder: int = 1,
) -> None:
    rounded_box(ax, x, y, w, h, face=face, edge=edge, lw=lw, linestyle=linestyle, zorder=zorder)
    add_text(ax, x + w / 2, y + h * title_y, title, size=title_size, weight="bold", color=PALETTE["ink"])
    if body:
        add_text(ax, x + w / 2, y + h * body_y, body, size=body_size, color=PALETTE["ink"])
    if note:
        add_text(ax, x + w / 2, y + h * note_y, note, size=note_size, color=PALETTE["muted"])


# --- Layout constants ---------------------------------------------------------
LEFT_X, LEFT_W = 0.035, 0.300       # left (Fossen) column boxes
RIGHT_X, RIGHT_W = 0.560, 0.310     # right (component) column boxes
ROW_H = 0.082                       # standard mapping-box height (room for tag pill)
# The right column carries six slots (Coriolis forks into coadjoint + skew);
# the left column has five boxes, with the Coriolis box centred against its two
# right-hand targets. Centres are spaced to leave a clean gap between every box.
RY_MASS, RY_COADJ, RY_SKEW, RY_DISS, RY_POT, RY_PORT = (
    0.852, 0.756, 0.660, 0.564, 0.468, 0.372,
)
LY_INERTIA, LY_CORIOLIS, LY_DAMPING, LY_RESTORING, LY_EXTERNAL = (
    0.852, 0.708, 0.564, 0.468, 0.372,
)


def left_box(ax, y_centre, title, sym, *, note=""):
    """Grey phenomenological Fossen-role box: role name + symbol (+ opt. note)."""
    y = y_centre - ROW_H / 2
    rounded_box(ax, LEFT_X, y, LEFT_W, ROW_H, face=PALETTE["aux_pale"], edge=PALETTE["muted"], lw=0.85, zorder=2)
    if note:
        add_text(ax, LEFT_X + LEFT_W / 2, y + ROW_H * 0.72, title, size=BODY_SIZE, weight="bold", color=PALETTE["ink"])
        add_text(ax, LEFT_X + LEFT_W / 2, y + ROW_H * 0.44, sym, size=MATH_SIZE - 0.5, color=PALETTE["ink"])
        add_text(ax, LEFT_X + LEFT_W / 2, y + ROW_H * 0.18, note, size=SMALL_SIZE, color=PALETTE["muted"])
    else:
        add_text(ax, LEFT_X + LEFT_W / 2, y + ROW_H * 0.66, title, size=BODY_SIZE, weight="bold", color=PALETTE["ink"])
        add_text(ax, LEFT_X + LEFT_W / 2, y + ROW_H * 0.32, sym, size=MATH_SIZE - 0.5, color=PALETTE["ink"])


def component_box(ax, x, y_centre, w, title, *, tag="", learned=False, port=False, note=""):
    """Right structure-preserving component box: name + <=1 short tag.

    Blue edge = construction-guaranteed structural prior. A gold tag (learned or
    boundary term) marks the data-learned shape or external generalized force.
    The force term uses a thicker boundary to distinguish it from an interior
    module.
    """
    y = y_centre - ROW_H / 2
    edge = PALETTE["state"]
    lw = 1.15 if port else 0.95
    rounded_box(ax, x, y, w, ROW_H, face=PALETTE["state_pale"], edge=edge, lw=lw, zorder=2)
    if note:
        add_text(ax, x + w / 2, y + ROW_H * 0.80, title, size=BODY_SIZE, weight="bold", color=PALETTE["ink"])
        add_text(ax, x + w / 2, y + ROW_H * 0.55, note, size=SMALL_SIZE, color=PALETTE["muted"])
        tag_cy = y + ROW_H * 0.20
    else:
        add_text(ax, x + w / 2, y + ROW_H * 0.66, title, size=BODY_SIZE, weight="bold", color=PALETTE["ink"])
        tag_cy = y + ROW_H * 0.26
    if not tag:
        return
    tag_w, tag_h = min(w * 0.80, 0.165), 0.026
    tag_x = x + (w - tag_w) / 2
    if learned or port:
        rounded_box(ax, tag_x, tag_cy - tag_h / 2, tag_w, tag_h, face=PALETTE["power_pale"], edge=PALETTE["power"],
                    radius=0.016, lw=0.70, zorder=3)
        add_text(ax, x + w / 2, tag_cy, tag, size=SMALL_SIZE, weight="bold", color=PALETTE["power"])
    else:
        add_text(ax, x + w / 2, tag_cy, tag, size=SMALL_SIZE, color=PALETTE["muted"])


def map_arrow(ax, y_centre, *, x0=None, x1=None):
    x0 = LEFT_X + LEFT_W if x0 is None else x0
    x1 = RIGHT_X if x1 is None else x1
    arrow(ax, (x0, y_centre), (x1, y_centre), color=PALETTE["state"], lw=0.9, scale=7.5)


def draw_headers(ax) -> None:
    head_y = 0.905
    # Left column header (grey).
    rounded_box(ax, LEFT_X, head_y, LEFT_W, 0.058, face=PALETTE["aux_pale"], edge=PALETTE["muted"], lw=0.85, zorder=2)
    add_text(ax, LEFT_X + LEFT_W / 2, head_y + 0.040, "Classical Fossen roles", size=BODY_SIZE, weight="bold", color=PALETTE["ink"])
    add_text(ax, LEFT_X + LEFT_W / 2, head_y + 0.016, "phenomenological", size=SMALL_SIZE, color=PALETTE["muted"])

    # Right column header (structure-preserving).
    rounded_box(ax, RIGHT_X, head_y, RIGHT_W, 0.058, face=PALETTE["state_pale"], edge=PALETTE["state"], lw=0.85, zorder=2)
    add_text(ax, RIGHT_X + RIGHT_W / 2, head_y + 0.040, "Structure-preserving components", size=BODY_SIZE, weight="bold", color=PALETTE["ink"])
    add_text(ax, RIGHT_X + RIGHT_W / 2, head_y + 0.016, "geometric · learned in constrained classes", size=SMALL_SIZE, color=PALETTE["muted"])

    # Middle transfer label.
    mid_x = (LEFT_X + LEFT_W + RIGHT_X) / 2
    add_text(ax, mid_x, head_y + 0.030, "mapping by", size=SMALL_SIZE - 0.3,
             color=PALETTE["muted"])
    add_text(ax, mid_x, head_y + 0.012, "power properties",
             size=SMALL_SIZE - 0.3, color=PALETTE["muted"])


def draw_rows(ax) -> None:
    # Row 1: Inertia -> inverse mass.
    left_box(ax, LY_INERTIA, "Inertia", r"$M=M_{RB}+M_A$")
    component_box(ax, RIGHT_X, RY_MASS, RIGHT_W, "Inverse mass", tag=r"$M_\theta^{-1}\!\succ\!0$", learned=True)
    map_arrow(ax, LY_INERTIA)

    # Row 2: Coriolis -> fork into coadjoint (structure) + skew coupling (learned).
    left_box(ax, LY_CORIOLIS, "Coriolis and centripetal",
             r"$C(\nu)\nu$", note="zero-power")
    # Upper right box: coadjoint (pure structure, blue, plain grey tag).
    component_box(ax, RIGHT_X, RY_COADJ, RIGHT_W, "Coadjoint coupling", note=r"$\mathrm{ad}^{*}_{\nu_r}p_r$",
                  tag="fixed · zero-power", learned=False)
    # Lower right box: skew coupling (learned, gold tag); carries lift / lateral.
    component_box(ax, RIGHT_X, RY_SKEW, RIGHT_W, "Skew coupling", note="lift / lateral",
                  tag=r"$J_\theta\!=\!-J_\theta^{\top}$", learned=True)
    # Single stem from the left box, forking into two non-crossing short arrows.
    stem_x = LEFT_X + LEFT_W
    fork_x = (stem_x + RIGHT_X) / 2
    ax.plot([stem_x, fork_x], [LY_CORIOLIS, LY_CORIOLIS], color=PALETTE["state"], lw=0.9, zorder=4, solid_capstyle="round")
    arrow(ax, (fork_x, LY_CORIOLIS), (RIGHT_X, RY_COADJ), color=PALETTE["state"], lw=0.9, scale=7.0,
          connectionstyle="arc3,rad=-0.18")
    arrow(ax, (fork_x, LY_CORIOLIS), (RIGHT_X, RY_SKEW), color=PALETTE["state"], lw=0.9, scale=7.0,
          connectionstyle="arc3,rad=0.18")

    # Row 3: Damping -> dissipation.
    left_box(ax, LY_DAMPING, "Damping", r"$D(\nu)\nu$")
    component_box(ax, RIGHT_X, RY_DISS, RIGHT_W, "Dissipation",
                  tag=r"$D_\theta\!\succ\!0$", learned=True)
    map_arrow(ax, LY_DAMPING)

    # Row 4: Restoring -> potential.
    left_box(ax, LY_RESTORING, "Restoring", r"$g(q)$")
    component_box(ax, RIGHT_X, RY_POT, RIGHT_W, "Potential", tag=r"$V_\theta\!\to\!f_\theta^{V}$", learned=True)
    map_arrow(ax, LY_RESTORING)

    # Row 5: External force -> learned generalized-force term.
    left_box(ax, LY_EXTERNAL, "External force", r"$\tau$")
    component_box(ax, RIGHT_X, RY_PORT, RIGHT_W,
                  "External generalized force",
                  note=r"power $\nu_r^\top\tau_\theta$",
                  tag=r"$\tau_\theta\!\neq\!G(q)u$", port=True)
    map_arrow(ax, LY_EXTERNAL)


def draw_anchors_and_footnote(ax) -> None:
    # Divider above the before/after anchors.
    ax.plot([0.035, 0.965], [0.318, 0.318], color=PALETTE["rule"], lw=0.7, zorder=0)

    # Before: explicit force sum (classical Fossen).
    add_text(ax, 0.035, 0.288, "explicit force sum", size=SMALL_SIZE, weight="bold", color=PALETTE["muted"], ha="left")
    add_text(ax, 0.035, 0.262, r"$M\dot\nu+C(\nu)\nu+D(\nu)\nu+g(q)=\tau$", size=MATH_SIZE - 0.6, color=PALETTE["ink"], ha="left")

    # Arrow between the two anchored forms.
    arrow(ax, (0.482, 0.270), (0.530, 0.270), color=PALETTE["state"], lw=0.9, scale=7.0)

    # After: open structured form (NO closed port-Hamiltonian form).
    add_text(ax, 0.560, 0.288, "open structured form", size=SMALL_SIZE, weight="bold", color=PALETTE["muted"], ha="left")
    add_text(
        ax,
        0.560,
        0.262,
        r"$\dot p_r=\mathrm{ad}^{*}_{\nu_r}p_r+f_\theta^{V}-D_\theta(\xi)\nu_r+J_\theta(\xi)\nu_r+\tau_\theta$",
        size=SMALL_SIZE + 0.2,
        color=PALETTE["ink"],
        ha="left",
    )
    add_text(ax, 0.905, 0.236, "external force term",
             size=SMALL_SIZE - 0.6, color=PALETTE["power"], ha="center")

    # Power-pairing velocity note between/below the two forms.
    add_text(
        ax,
        0.500,
        0.214,
        r"power-pairing velocity $\nu\to\nu_r$ (water-relative)",
        size=SMALL_SIZE,
        color=PALETTE["muted"],
        ha="center",
    )

    # Mini colour legend (bottom-right).
    add_text(
        ax,
        0.965,
        0.176,
        "blue = structural prior · gold = data-learned shape",
        size=SMALL_SIZE - 0.2,
        color=PALETTE["muted"],
        ha="right",
    )

    # Red-line footnote (single grey line; open-system / not closed pH).
    ax.plot([0.035, 0.965], [0.150, 0.150], color=PALETTE["rule"], lw=0.7, zorder=0)
    foot = (
        "Mechanical-subsystem power properties; current / actuator / depth are exogenous. "
        "The full vehicle–actuator–environment system is open — not a closed\n"
        r"port-Hamiltonian system, and $\tau_\theta\neq G(q)u$."
    )
    add_text(ax, 0.035, 0.105, foot, size=SMALL_SIZE - 0.2, color=PALETTE["muted"], ha="left", va="center", linespacing=1.35)


def draw() -> None:
    setup_style()
    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    draw_headers(ax)
    draw_rows(ax)
    draw_anchors_and_footnote(ax)

    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 400}),
    ]:
        fig.savefig(OUT_BASE.with_suffix(suffix), bbox_inches="tight", pad_inches=0.02, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    draw()
