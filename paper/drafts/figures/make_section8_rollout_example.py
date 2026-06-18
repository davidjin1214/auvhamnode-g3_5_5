#!/usr/bin/env python3
"""Figure (§1.8) qualitative rollout example: one clean CHIRP trajectory, same
initial condition, AUVHamNODE vs the full-state black box.

Panel A -- 3D trajectory overlay: ground truth and the AUVHamNODE prediction
stay together over the full 60 s horizon, while the full-state black box leaves
the trajectory and runs away.
Panel B -- position error vs time (log): the structured prediction holds at the
sub-metre level; the black box accumulates an unrecoverable error of ~100 m.

Reads the read-only dumps written by evaluate_rollout_benchmark.py
--dump_example_trajectories (copied to figure_data/rollout_example/):
  phnode_full_CHIRP_seed42.npz, blackbox_fullstate_CHIRP_seed42.npz
"""

from __future__ import annotations

from pathlib import Path

import _section8_style as S

import matplotlib.pyplot as plt
import numpy as np

WIDTH_MM = 150
HEIGHT_MM = 66
OUT_BASE = Path(__file__).resolve().parent / "section8_rollout_example"
DATA_DIR = S.EVIDENCE_DIR / "figure_data" / "rollout_example"


def load(tag):
    d = np.load(DATA_DIR / f"{tag}_CHIRP_seed42.npz", allow_pickle=True)
    return d["time"], d["gt_pos"], d["pred_pos"]


def draw():
    S.setup_style()
    t, gt, pf_pred = load("phnode_full")
    _, _, bb_pred = load("blackbox_fullstate")

    fig = plt.figure(figsize=(WIDTH_MM * S.MM_TO_IN, HEIGHT_MM * S.MM_TO_IN), facecolor="white")

    # ---- Panel A: top-down (x-y) trajectory overlay ----
    # depth varies only ~2.5 m over the run, so the top view captures the motion.
    axA = fig.add_subplot(1, 2, 1)
    axA.plot(gt[:, 0], gt[:, 1], color=S.PALETTE["ink"], lw=1.7, label="Ground truth", zorder=5)
    axA.plot(pf_pred[:, 0], pf_pred[:, 1], color=S.PALETTE["geometry"],
             lw=1.3, ls=(0, (4, 1.5)), label="AUVHamNODE", zorder=6)
    axA.plot(bb_pred[:, 0], bb_pred[:, 1], color=S.PALETTE["risk"],
             lw=1.3, ls=(0, (4, 1.5)), label="Full-state black box", zorder=4)
    axA.scatter([gt[0, 0]], [gt[0, 1]], color=S.PALETTE["ink"], s=14, zorder=7)
    axA.text(gt[0, 0] + 2, gt[0, 1] + 2, "start", fontsize=5.4, color=S.PALETTE["ink"])
    axA.set_aspect("equal", adjustable="datalim")
    axA.set_xlabel("x / m", fontsize=6.6)
    axA.set_ylabel("y / m", fontsize=6.6)
    axA.grid(True, color=S.PALETTE["rule"], lw=0.5)
    axA.set_axisbelow(True)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)
    axA.tick_params(labelsize=6.0)
    axA.legend(loc="lower left", fontsize=5.4, frameon=False, handlelength=1.8)
    S.panel_label(axA, "a", dx=-0.16, dy=0.03)

    # ---- Panel B: position error vs time (log) ----
    axB = fig.add_subplot(1, 2, 2)
    err_pf = np.linalg.norm(pf_pred - gt, axis=1)
    err_bb = np.linalg.norm(bb_pred - gt, axis=1)
    axB.plot(t, np.maximum(err_pf, 1e-3), color=S.PALETTE["geometry"], lw=1.4, label="AUVHamNODE")
    axB.plot(t, np.maximum(err_bb, 1e-3), color=S.PALETTE["risk"], lw=1.4, label="Full-state black box")
    axB.set_yscale("log")
    axB.set_xlim(0, 60)
    axB.set_ylim(1e-2, 3e2)
    axB.set_xlabel("time / s", fontsize=6.6)
    axB.set_ylabel("position error / m", fontsize=6.6)
    axB.grid(True, which="major", color=S.PALETTE["rule"], lw=0.5)
    axB.set_axisbelow(True)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)
    axB.tick_params(labelsize=6.0)
    axB.legend(loc="lower right", fontsize=5.6, frameon=False, handlelength=1.8, borderaxespad=0.4)
    axB.text(58, err_pf[-1] * 2.1, f"{err_pf[-1]:.2f} m", fontsize=5.8,
             color=S.PALETTE["geometry"], ha="right", va="bottom", fontweight="bold")
    axB.text(58, err_bb[-1] * 0.60, f"{err_bb[-1]:.0f} m", fontsize=5.8,
             color=S.PALETTE["risk"], ha="right", va="top", fontweight="bold")
    S.panel_label(axB, "b", dx=-0.165, dy=0.03)

    fig.subplots_adjust(left=0.09, right=0.975, bottom=0.155, top=0.91, wspace=0.30)
    S.save_fig(fig, OUT_BASE)
    plt.close(fig)
    print(f"wrote {OUT_BASE}.pdf / .png / .svg  (final err: AUVHamNODE {err_pf[-1]:.2f} m, black box {err_bb[-1]:.1f} m)")


if __name__ == "__main__":
    draw()
