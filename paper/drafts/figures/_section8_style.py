#!/usr/bin/env python3
"""Shared plotting style for the §1.8 (Section 8) result figures.

Single source of truth for palette, model display names, font selection,
rcParams and multi-format export, so the six §1.8 figures stay visually
uniform. Mirrors the conventions of the pre-existing
``make_section8_two_level_evidence.py`` / ``make_velocity_state_contract.py``
generators (physical-mm sizing, sans-serif, embedded fonts, white facecolor).

Import this module BEFORE importing ``matplotlib.pyplot`` in a figure script:
the cache-isolation env vars must be set prior to the first matplotlib import.
"""

from __future__ import annotations

import os
from pathlib import Path

# Cache isolation -- must precede the first matplotlib import (hence at module top).
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/auvhamnode_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/auvhamnode_xdg_cache")

import matplotlib as mpl  # noqa: E402
from matplotlib import font_manager  # noqa: E402

MM_TO_IN = 1 / 25.4

# Repo root, given this file lives at paper/drafts/figures/_section8_style.py
REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_DIR = REPO_ROOT / "analysis" / "section8_current_evidence"

# Semantic palette (shared with make_section8_two_level_evidence.py).
PALETTE = {
    "ink": "#252525",
    "muted": "#62666D",
    "rule": "#D8DDE3",
    "paper": "#FFFFFF",
    "geometry": "#2E6EBA",      # SE(3) geometry / stability axis -- blue
    "geometry_pale": "#EEF5FC",
    "energy": "#2F8E86",        # mechanical-energy / accuracy axis -- green
    "energy_pale": "#EDF7F4",
    "risk": "#9A5B3F",          # divergence / black-box risk -- brown
    "risk_pale": "#F7EFEA",
    "accent": "#B8862B",        # emphasis -- gold
    "accent_pale": "#FBF4E5",
    "bar": "#B7C0CB",
    "bar_dark": "#2E6EBA",
    "bar_energy": "#77B7AF",
    "bar_qforce": "#C99A3C",
    "blackbox": "#A8AFBA",      # generic black-box grey
}

# One-line display names (legends, annotations).
DISPLAY = {
    "phnode_full": "AUVHamNODE",
    "ablate_no_lift": "No Lift",
    "ablate_no_mass_prior": "No Mass Prior",
    "ablate_diag_damping": "Diagonal Damping",
    "ablate_bu_only": "Narrow Actuation",
    "phnode_merged_force": "Merged Force",
    "phnode_qforce": "Config Force",
    "se3_momentum_blackbox": "SE(3) mom.",
    "se3_accel_blackbox": "SE(3) accel",
    "blackbox_fullstate": "Full-state",
}

# Multi-line names for tick labels.
AXIS_DISPLAY = {
    "phnode_full": "AUVHamNODE",
    "ablate_no_lift": "No Lift",
    "ablate_no_mass_prior": "No Mass\nPrior",
    "ablate_diag_damping": "Diagonal\nDamping",
    "ablate_bu_only": "Narrow\nActuation",
    "phnode_merged_force": "Merged\nForce",
    "phnode_qforce": "Config\nForce",
    "se3_momentum_blackbox": "SE(3)\nMomentum",
    "se3_accel_blackbox": "SE(3)\nAccel",
    "blackbox_fullstate": "Full-state",
}

# Per-model line/marker colour by structural family (figures 2/3).
MODEL_COLOR = {
    "phnode_full": PALETTE["geometry"],
    "ablate_no_lift": PALETTE["energy"],
    "ablate_no_mass_prior": PALETTE["bar_energy"],
    "ablate_diag_damping": "#6E5BA6",       # damping structure -- violet
    "ablate_bu_only": PALETTE["risk"],       # actuation conditioning -- brown (diverges)
    "phnode_merged_force": PALETTE["accent"],
    "phnode_qforce": PALETTE["bar_qforce"],
    "se3_momentum_blackbox": PALETTE["muted"],
    "se3_accel_blackbox": "#9AA0A8",
    "blackbox_fullstate": PALETTE["risk"],
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
            "font.size": 7.0,
            "axes.linewidth": 0.65,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax, letter: str, *, dx: float = -0.13, dy: float = 0.04) -> None:
    """Nature-style bold lower-case subpanel label at the top-left of `ax`.

    dx/dy are axes-fraction offsets from the top-left corner; tune per figure so
    the label clears the y-axis label without clipping.
    """
    ax.text(dx, 1.0 + dy, letter, transform=ax.transAxes, fontsize=9.0,
            fontweight="bold", va="bottom", ha="left", color=PALETTE["ink"])


def save_fig(fig, out_base: Path, *, png_dpi: int = 600) -> None:
    """Write .svg / .pdf / .png(600 dpi) next to each other, tight bbox."""
    for suffix, kwargs in [(".svg", {}), (".pdf", {}), (".png", {"dpi": png_dpi})]:
        fig.savefig(out_base.with_suffix(suffix), bbox_inches="tight", pad_inches=0.02, **kwargs)
