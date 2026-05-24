#!/usr/bin/env python3
"""Generate the T2-supplementary `ablate_no_lift` clean-train seed-scan Colab notebook.

Purpose. The T2 WP-Frag rerun produced a reproducible catastrophic clean-train
collapse at `ablate_no_lift` seed43 (60 s position median 44 m, best_loss 0.217)
on the current-main g3_5_7 mirror, even though the catalog-era seed43 was healthy
(0.66 m). This notebook disambiguates whether that is (a) an environment-sensitive
artifact in the same class as phnode_full seed46, or (b) a genuine, rare clean-train
fragility introduced by removing the lift submodule.

Design. clean protocol ONLY (the collapse is clean-train specific; iid/v4lite
already regularise it away), single model `ablate_no_lift`, seeds {43,47,48,49,50,51}
— rerun seed43 for reproducibility plus five fresh seeds for the collapse base-rate.
It calls the proven `scripts/train_all_models_noise_profile.sh` and
`scripts/batch_eval_models.sh` directly (the same entry points the phase1a driver
wraps), so the training/eval contract is identical to T2 without modifying the
shared driver and without wasting compute on the iid/v4lite protocols.

Run locally:  python notebook/make_t2supp_nolift_notebook.py
Output:       notebook/t2supp_nolift_seedscan.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

DATASET = "data/auv_oc_traj1000_blk150_s23_d0be9434.pkl"
PROJECT_DEFAULT = "/content/drive/MyDrive/Colab Notebooks/auvhamnode/g3_5_7"
MODEL = "ablate_no_lift"
SUITE = "sweep_oc_phase1a_decision_clean_t2supp_nolift"
# seed43 is dropped: its clean-train collapse (60s 44.5m, best_loss 0.217 @epoch19) already
# reproduced deterministically (T2 + the protocol='clean' attempt), so re-running it adds
# nothing. These five fresh seeds measure the no-lift clean-train collapse base-rate.
SEEDS = "47 48 49 50 51"
EVAL_RUN_NAME = "phase1a_iideval_traj30_seed42"

_uid = 0


def _next_id() -> str:
    global _uid
    _uid += 1
    return f"t2supp{_uid:03d}"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


CONFIG_CELL = f'''import os
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("AUV_PROJECT_DIR", "{PROJECT_DEFAULT}"))
assert PROJECT_DIR.exists(), f"Project directory not found: {{PROJECT_DIR}}"

# clean-only seed-scan identity
MODEL   = "{MODEL}"
SUITE   = "{SUITE}"
SEEDS   = "{SEEDS}"        # fresh seeds for no-lift clean-train collapse base-rate (seed43 already reproduced)
DATASET = "{DATASET}"
DEVICE  = "cuda"
NOISE_REFERENCE = "remus100_dr"
EVAL_RUN_NAME = "{EVAL_RUN_NAME}"

SUITE_DIR = PROJECT_DIR / "checkpoints" / SUITE
print("MODEL   =", MODEL)
print("SUITE   =", SUITE)
print("SEEDS   =", SEEDS)
print("SUITE_DIR exists?", SUITE_DIR.exists())'''


PREFLIGHT_CELL = '''# Refuse to overwrite an existing suite (mirrors the phase1a require_absent gate).
# If a previous attempt left a partial suite (e.g. the protocol='clean' crash that only
# trained seed43), remove it first — run, in a cell:  !rm -rf "$SUITE_DIR"
import shutil  # noqa: F401  (handy if you choose to clean up programmatically)
assert not SUITE_DIR.exists(), (
    f"Target suite already exists: {SUITE_DIR}\\n"
    f'Remove the partial/old suite before re-running:  !rm -rf "{SUITE_DIR}"'
)
print("preflight OK — target suite is absent:", SUITE_DIR)'''


TRAIN_CELL = '''# Clean-train ablate_no_lift across the seed scan. Identical contract to the T2
# clean suite: same dataset, same train_all_models_noise_profile.sh entry point,
# repo-default noise schedule (warmup=20, ramp=80, mix_ratio=0.5; inert under clean).
#
# NOTE: --noise-protocol must be `auto`, NOT `clean`. With profile=clean the trainer
# itself produces a clean-trained model, but its post-training detailed evaluation
# also builds noise configs for the non-clean eval profiles (nominal/degraded/
# heading_biased); resolve_noise_protocol() rejects protocol='clean' for those and
# crashes the whole sweep. `auto` resolves to clean for the clean train profile and
# to iid_noisy_ic for the eval profiles. This mirrors the phase1a driver (`clean auto`).
%cd $PROJECT_DIR
!bash scripts/train_all_models_noise_profile.sh \\
  --profile oc \\
  --models "$MODEL" \\
  --dataset "$DATASET" \\
  --seeds "$SEEDS" \\
  --suite-name "$SUITE" \\
  --noise-profile clean \\
  --noise-protocol auto \\
  --noise-reference "$NOISE_REFERENCE" \\
  --device "$DEVICE"'''


EVAL_CELL = '''# Rollout-evaluate the clean-trained suite across the 4 robustness profiles
# (iid eval protocol), matching the T2 eval contract so numbers are comparable.
%cd $PROJECT_DIR
!bash scripts/batch_eval_models.sh \\
  --suite-dir "checkpoints/$SUITE" \\
  --mode resampled \\
  --num-traj-per-scenario 30 \\
  --times "10 30 60" \\
  --scenarios "PRBS CHIRP OU" \\
  --seed 42 \\
  --progress-every 5 \\
  --num-diagnostic-plots 6 \\
  --device "$DEVICE" \\
  --extra-eval-arg --run_name --extra-eval-arg "$EVAL_RUN_NAME" \\
  --extra-eval-arg --noise_protocol --extra-eval-arg iid_noisy_ic \\
  --extra-eval-arg --noise_reference --extra-eval-arg "$NOISE_REFERENCE" \\
  --extra-eval-arg --noise_seed --extra-eval-arg 2024 \\
  --extra-eval-arg --noise_profiles \\
  --extra-eval-arg clean \\
  --extra-eval-arg nominal_eval \\
  --extra-eval-arg degraded_eval \\
  --extra-eval-arg heading_biased_eval'''


PROVENANCE_CELL = '''# Per-run provenance so these seeds are distinguishable from catalog-era drift.
import sys, subprocess, datetime
import torch

def _sh(args):
    try:
        return subprocess.run(args, cwd=PROJECT_DIR, capture_output=True, text=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {exc}>"

head = _sh(["git", "rev-parse", "HEAD"])
diffstat = _sh(["git", "diff", "HEAD", "--stat"])
env_txt = "\\n".join([
    f"python={sys.version.split()[0]}",
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
    f"cudnn={torch.backends.cudnn.version()}",
    f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
    f"captured_at={datetime.datetime.now().isoformat()}",
])

n = 0
for cfg in SUITE_DIR.rglob("config.json"):
    meta = cfg.parent / "_audit_meta"
    meta.mkdir(exist_ok=True)
    (meta / "code_revision.txt").write_text(f"git_head={head}\\n\\n{diffstat}\\n")
    (meta / "environment.txt").write_text(env_txt + "\\n")
    n += 1
print(f"provenance written for {n} run dirs; git_head={head!r}")'''


RESULTS_CELL = '''# Immediate verdict: per-seed clean-eval 60 s position median + completion + best_loss.
# seed43 (44.5 m) is already settled; this measures the collapse base-rate on fresh seeds.
# Decision rule:
#   * >=1 of 47-51 collapses (>10 m) -> no-lift has a genuine (recurring) clean-train fragility
#   * none of 47-51 collapses        -> seed43 is an isolated environment x seed artifact (seed46-class)
import json, glob, pickle

def best_loss(run_dir):
    hist = run_dir / "training_history.pkl"
    if not hist.exists():
        return None
    try:
        h = pickle.load(open(hist, "rb"))
        for key in ("best_loss", "best_test_loss"):
            if key in h:
                return float(h[key])
        tl = h.get("test_loss") or h.get("test") or []
        return float(min(tl)) if tl else None
    except Exception:
        return None

print(f"{'seed':>5} {'pos_med60':>10} {'pos_p95_60':>11} {'compl':>7} {'best_loss':>11}  flag")
seeds = [int(s) for s in SEEDS.split()]
for s in sorted(seeds):
    run_dir = SUITE_DIR / f"ablation_{MODEL}_seed{s}"
    cand = glob.glob(str(run_dir / "rollout_benchmark" / "phase1a_iideval_*" / "clean" / "summary.json"))
    if not cand:
        print(f"{s:>5}  <no clean summary found>")
        continue
    d = json.load(open(cand[0]))
    m = d["overall"]["60.0"]["metrics"]["final_position_error"]
    compl = d["rollout_outcomes"]["overall"]["rates"]["completed"]
    bl = best_loss(run_dir)
    flag = "  <-- COLLAPSE" if m["median"] > 10.0 else ""
    bl_s = f"{bl:.4f}" if bl is not None else "?"
    print(f"{s:>5} {m['median']:>10.4f} {m['p95']:>11.4f} {compl:>7.3f} {bl_s:>11}{flag}")
print("\\nSync checkpoints/ back, then extend scripts/export_section8_t2_evidence.py "
      "or analyse this suite directly.")'''


def build_notebook() -> dict:
    cells = [
        md(
            "# T2-supplementary — `ablate_no_lift` clean-train seed scan\n"
            "\n"
            "**Why.** The T2 rerun produced a *deterministically reproducible* catastrophic clean-train\n"
            "collapse at `ablate_no_lift` seed43 (60 s position median **44.5 m**, best_loss 0.217 @epoch19,\n"
            "signature `no successful training batches`/pred-divergence) on the current-main `g3_5_7`\n"
            "mirror, although the catalog-era seed43 was healthy (0.66 m). seed43 itself is settled and\n"
            "not re-run here. This notebook measures the **collapse base-rate** on fresh seeds to decide\n"
            "whether that collapse is a genuine (rare) no-lift clean-train fragility or an isolated\n"
            "environment x seed artifact in the same class as phnode_full seed46.\n"
            "\n"
            "**Matrix.** `ablate_no_lift` x **clean protocol only** x seeds {47,48,49,50,51}\n"
            "= 5 fresh training runs. iid/v4lite are intentionally skipped: the collapse is clean-train\n"
            "specific and a little training noise already regularises it away.\n"
            "\n"
            "**Contract.** Calls `train_all_models_noise_profile.sh` + `batch_eval_models.sh`\n"
            "directly — the same entry points the phase1a driver wraps — so training/eval are\n"
            "identical to the T2 clean suite, with no driver change and no wasted compute.\n"
            "\n"
            "**After it finishes:** sync `checkpoints/` back; the results cell prints the per-seed\n"
            "verdict, and the suite name (`sweep_oc_phase1a_decision_clean_t2supp_nolift`) matches\n"
            "the existing `sweep_oc_phase1a_decision_clean_*` convention for downstream analysis."
        ),
        code("!nvidia-smi"),
        code(
            "import torch\n"
            'print(f"PyTorch: {torch.__version__}")\n'
            'print(f"CUDA available: {torch.cuda.is_available()}")\n'
            'print(f"CUDA: {torch.version.cuda}")\n'
            'print(f"cuDNN: {torch.backends.cudnn.version()}")'
        ),
        code("from google.colab import drive\ndrive.mount('/content/drive')"),
        md("## 0. Configuration"),
        code(CONFIG_CELL),
        code("%cd $PROJECT_DIR\n%pip install -q torchdiffeq pandas"),
        md("## 1. Preflight\nRefuses to overwrite an existing suite."),
        code(PREFLIGHT_CELL),
        md(
            "## 2. Clean-train seed scan (6 runs)\n"
            "Rerun seed43 + fresh seeds 47-51, clean protocol only."
        ),
        code(TRAIN_CELL),
        md(
            "## 3. Rollout eval (clean-trained suite x 4 profiles)\n"
            "`clean nominal_eval degraded_eval heading_biased_eval` under the iid eval protocol."
        ),
        code(EVAL_CELL),
        md("## 4. Provenance injection (per-run `_audit_meta/`)"),
        code(PROVENANCE_CELL),
        md(
            "## 5. Verdict — per-seed clean 60 s median (collapse base-rate)\n"
            ">=1 of seeds 47-51 above 10 m = genuine recurring no-lift clean-train fragility; "
            "none above 10 m = seed43 is an isolated environment x seed artifact (seed46-class)."
        ),
        code(RESULTS_CELL),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out = Path(__file__).resolve().parent / "t2supp_nolift_seedscan.ipynb"
    nb = build_notebook()
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n")
    print(f"wrote {out}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
