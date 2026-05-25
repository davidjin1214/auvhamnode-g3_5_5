#!/usr/bin/env python3
"""Generate the black-box current-evidence Colab notebooks (Path B).

The four-model T2 rerun (`make_t2_notebooks.py`) deliberately omitted the
black-box / semi-structured baselines. The §8 "structured >> black-box" claim
therefore had no clean-mirror anchor: the only black-box numbers lived in the
2026-04 `sweep_oc_core_default_*` suite, trained in the old environment that
catastrophically collapsed `phnode_full` on 3/6 clean seeds and is provably a
different regime from T2 (e.g. `phnode_qforce` clean flips 0.57 old <-> 3.76 T2).
Old-regime numbers cannot be cross-compared with T2, so we regenerate the
black-box rows in the T2 regime instead.

Each notebook reruns ONE black-box model **clean-only** across seeds 42-46 on the
current-main `g3_5_7` mirror, then evaluates the clean-trained suite across the
four robustness eval profiles (clean / nominal_eval / degraded_eval /
heading_biased_eval) under both the iid_noisy_ic and v4_lite eval protocols --
exactly the eval contract the T2 clean suites already received, so the rows drop
straight into `analysis/section8_current_evidence/aggregate.csv`.

Clean-only is enabled via the backward-compatible `PHASE1A_PROTOCOLS=clean` knob
added to `scripts/run_phase1a_oc_v4lite.sh` (default still runs the full triple).
RUN_TAG is `t2_wpfrag_<model>` so `scripts/export_section8_t2_evidence.py`
discovers the suites with zero extra wiring.

Run locally:  python notebook/make_t2_blackbox_notebook.py
Outputs:      notebook/t2_wpfrag_<model>.ipynb  (x3, black-box)
"""
from __future__ import annotations

import json
from pathlib import Path

MODELS = [
    "blackbox_fullstate",
    "se3_momentum_blackbox",
    "se3_accel_blackbox",
]

DATASET = "data/auv_oc_traj1000_blk150_s23_d0be9434.pkl"
PROJECT_DEFAULT = "/content/drive/MyDrive/Colab Notebooks/auvhamnode/g3_5_7"

_uid = 0


def _next_id() -> str:
    global _uid
    _uid += 1
    return f"bbxcell{_uid:03d}"


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


def config_cell(model: str, run_tag: str) -> str:
    return f'''import os

# Runtime
os.environ["PYTHON_BIN"] = "python"
os.environ["DEVICE"] = "cuda"
os.environ["LOCAL_PROXY_ROOT"] = "/content/_proxy_suites"

# ---- identity (one black-box model per notebook) ----
os.environ["RUN_TAG"] = "{run_tag}"
os.environ["DATASET"] = "{DATASET}"
os.environ["NOISE_REFERENCE"] = "remus100_dr"
os.environ["PHASE1A_LOG_DIR"] = str(PROJECT_DIR / "checkpoints" / "phase1a_logs" / os.environ["RUN_TAG"])
os.environ["PHASE1A_METADATA_DIR"] = str(PROJECT_DIR / "checkpoints" / f"phase1a_metadata_{{os.environ['RUN_TAG']}}")

# ---- matrix: single black-box model x CLEAN-ONLY x 5 seeds ----
os.environ["PHASE1A_MODELS"] = "{model}"
os.environ["SMOKE1_MODELS"] = "{model}"
os.environ["PHASE1A_PROTOCOLS"] = "clean"        # clean-only knob (driver default is "clean iid v4lite")
os.environ["SMOKE_SEEDS"] = "42"                 # cheap protocol-correctness gate (1 seed)
os.environ["DECISION_SEEDS"] = "42 43 44 45 46"  # full current-evidence seed set (matches T2)

# ---- Evaluation contract (identical to T2 clean suite) ----
os.environ["SMOKE_EVAL_NUM_TRAJ_PER_SCENARIO"] = "6"
os.environ["DECISION_EVAL_NUM_TRAJ_PER_SCENARIO"] = "30"
os.environ["EVAL_TIMES"] = "10 30 60"
os.environ["EVAL_SCENARIOS"] = "PRBS CHIRP OU"
os.environ["EVAL_BASE_SEED"] = "42"
os.environ["EVAL_NOISE_SEED"] = "2024"
os.environ["EVAL_PROGRESS_EVERY"] = "5"
os.environ["EVAL_NUM_DIAGNOSTIC_PLOTS"] = "6"
# Full robustness set: evaluate the clean-trained suite across all 4 profiles (caveat B).
os.environ["IID_EVAL_PROFILES"] = "clean nominal_eval degraded_eval heading_biased_eval"
os.environ["V4_EVAL_PROFILES"] = "nominal_eval"

# ---- Audit gate ----
os.environ["STRICT_ZERO_NOISE_AUDIT"] = "1"
os.environ["SOFT_MIN_EPOCH_SCALE"] = "0.05"

print("RUN_TAG          =", os.environ["RUN_TAG"])
print("PHASE1A_MODELS   =", os.environ["PHASE1A_MODELS"])
print("PHASE1A_PROTOCOLS=", os.environ["PHASE1A_PROTOCOLS"])
print("DECISION_SEEDS   =", os.environ["DECISION_SEEDS"])
print("IID_EVAL_PROFILES=", os.environ["IID_EVAL_PROFILES"])'''


PROVENANCE_CELL = '''# Inject per-run provenance. The training flow does NOT write these automatically,
# so for paper-grade evidence we record them here (see docs/provenance_audit_phnode_full_clean.md sec 5.2).
import os, sys, subprocess, datetime
from pathlib import Path
import torch

RUN_TAG = os.environ["RUN_TAG"]
ckpt = PROJECT_DIR / "checkpoints"
# clean-only sweep: only the clean suite exists, but we scan the full triple defensively.
suites = [ckpt / f"sweep_oc_phase1a_decision_{p}_{RUN_TAG}" for p in ("clean", "iid", "v4lite")]

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
for suite in suites:
    if not suite.exists():
        continue
    for cfg in suite.rglob("config.json"):
        run_dir = cfg.parent
        meta = run_dir / "_audit_meta"
        meta.mkdir(exist_ok=True)
        (meta / "code_revision.txt").write_text(f"git_head={head}\\n\\n{diffstat}\\n")
        (meta / "environment.txt").write_text(env_txt + "\\n")
        n += 1
print(f"provenance written for {n} run dirs")
print("git_head =", head)'''


ANOMALY_CELL = '''# Quick anomaly scan: catch seed46/seed43-style catastrophic training before trusting numbers.
# Black-box models are collapse-prone (blackbox_fullstate collapsed in the 2026-04 catalog:
# s44=86.8m, s42/s43=nan), so a clean-mirror collapse here is a LEGITIMATE black-box instability
# finding, not an artifact -- still flag it and handle per the B1 policy (exclude from the
# quantitative aggregate, annotate transparently).
import os, re
from pathlib import Path

RUN_TAG = os.environ["RUN_TAG"]
ckpt = PROJECT_DIR / "checkpoints"
suites = [ckpt / f"sweep_oc_phase1a_decision_{p}_{RUN_TAG}" for p in ("clean", "iid", "v4lite")]

print(f"{'suite':<48}{'run':<32}{'best_epoch':>10}{'best_loss':>14}{'nbad':>6}  flag")
for suite in suites:
    if not suite.exists():
        continue
    for cfg in sorted(suite.rglob("config.json")):
        run_dir = cfg.parent
        log = run_dir / "training.log"
        text = log.read_text() if log.exists() else ""
        nbad = text.count("no successful training batches")
        best_epoch, best_loss = "?", "?"
        m = re.findall(r"[Bb]est.*?epoch[^0-9]*([0-9]+).*?([0-9]+\\.[0-9eE+-]+)", text)
        if m:
            best_epoch, best_loss = m[-1]
        flag = "  <-- CHECK (possible artifact / instability)" if nbad > 0 else ""
        print(f"{suite.name[:46]:<48}{run_dir.name[:30]:<32}{str(best_epoch):>10}{str(best_loss):>14}{nbad:>6}{flag}")
print("\\nFlagged seeds: handle per B1 (exclude from aggregate via train_anomaly, annotate).")'''


def build_notebook(model: str) -> dict:
    run_tag = f"t2_wpfrag_{model}"
    cells = [
        md(
            f"# Black-box current-evidence rerun (clean-only) — `{model}`\n"
            "\n"
            "**Goal.** Regenerate a black-box / semi-structured baseline on the current-main\n"
            "(`g3_5_7` mirror) regime so the chapter-section-8 \"structured >> black-box\" claim has a\n"
            "clean-mirror anchor directly comparable to the four T2 models. Catalog-era black-box numbers\n"
            "come from the 2026-04 `sweep_oc_core_default_*` suite — a different regime that collapsed\n"
            "`phnode_full` on 3/6 clean seeds and flipped `phnode_qforce` clean 0.57<->3.76, so they\n"
            "cannot be cross-compared with T2.\n"
            "\n"
            "**Matrix (this notebook).** `" + model + "` x {clean} x seeds {42,43,44,45,46} = 5 training\n"
            "runs (clean-only via `PHASE1A_PROTOCOLS=clean`), then the clean suite is evaluated across\n"
            "{clean, nominal_eval, degraded_eval, heading_biased_eval} under iid_noisy_ic + v4_lite eval\n"
            "protocols — identical to the eval the T2 clean suites received.\n"
            "\n"
            "**Parallelism.** Run the three black-box notebooks (`blackbox_fullstate`,\n"
            "`se3_momentum_blackbox`, `se3_accel_blackbox`) in separate Colab sessions. Distinct\n"
            "`RUN_TAG`s, no collision.\n"
            "\n"
            "**After all three finish:** sync `checkpoints/sweep_oc_phase1a_decision_clean_t2_wpfrag_*`\n"
            "back, then locally rerun `scripts/export_section8_t2_evidence.py` — the black-box models are\n"
            "already in its MODELS list, so the clean rows append to `aggregate.csv` automatically.\n"
            "\n"
            "> Watch list: black-box models are collapse-prone (`blackbox_fullstate` was nan/86.8m in the\n"
            "> old catalog). The anomaly-scan cell flags any clean-mirror collapse; a real collapse is a\n"
            "> legitimate instability finding (handle per B1: exclude from aggregate, annotate)."
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
        code(
            "import os\n"
            "from pathlib import Path\n"
            "\n"
            "PROJECT_DIR = Path(os.environ.get(\n"
            '    "AUV_PROJECT_DIR",\n'
            f'    "{PROJECT_DEFAULT}",\n'
            "))\n"
            'assert PROJECT_DIR.exists(), f"Project directory not found: {PROJECT_DIR}"\n'
            "%cd $PROJECT_DIR"
        ),
        code("%pip install -q torchdiffeq pandas"),
        md("## 0. Configuration"),
        code(config_cell(model, run_tag)),
        md(
            "## 1. Preflight\n"
            "Confirms target suite/proxy dirs are absent and saves suite-level run config + environment "
            "metadata. If it fails, change `RUN_TAG` or remove the target dirs."
        ),
        code('os.environ["MODE"] = "preflight"\n!bash scripts/run_phase1a_oc_v4lite.sh'),
        md(
            "## 2. Smoke gate (seed 42, clean only)\n"
            "Cheap protocol-correctness check for the clean path + strict zero-noise audit. Smoke results "
            "are flow-validation only — never cited."
        ),
        code('os.environ["MODE"] = "smoke1_train"\n!bash scripts/run_phase1a_oc_v4lite.sh'),
        code('os.environ["MODE"] = "smoke1_eval"\n!bash scripts/run_phase1a_oc_v4lite.sh'),
        md(
            "## 3. Decision train (clean x 5 seeds)\n"
            "The 5 evidence-bearing clean training runs for this black-box model."
        ),
        code('os.environ["MODE"] = "decision_train"\n!bash scripts/run_phase1a_oc_v4lite.sh'),
        md(
            "## 4. Decision rollout eval (clean suite x 4 robustness profiles)\n"
            "Evaluates the clean-trained suite across `clean nominal_eval degraded_eval "
            "heading_biased_eval` (iid eval protocol) plus the `v4_lite` eval at `nominal_eval`."
        ),
        code('os.environ["MODE"] = "decision_eval"\n!bash scripts/run_phase1a_oc_v4lite.sh'),
        md(
            "## 5. Provenance injection (per-run `_audit_meta/`)\n"
            "Records git HEAD + environment fingerprint into every decision run dir so these runs are "
            "distinguishable from catalog-era drift."
        ),
        code(PROVENANCE_CELL),
        md(
            "## 6. Anomaly scan\n"
            "Flags any run with the `no successful training batches` catastrophic-gradient signature. For "
            "black-box models a clean-mirror collapse is a legitimate instability result — flag and handle "
            "per B1 (exclude from the quantitative aggregate, annotate)."
        ),
        code(ANOMALY_CELL),
        md(
            "## 7. Next step (after all three black-box notebooks finish)\n"
            "1. Ensure `checkpoints/sweep_oc_phase1a_decision_clean_t2_wpfrag_{blackbox_fullstate,se3_momentum_blackbox,se3_accel_blackbox}` are synced back.\n"
            "2. Locally rerun `python scripts/export_section8_t2_evidence.py` (black-box models are already in MODELS).\n"
            "3. The clean rows append to `analysis/section8_current_evidence/aggregate.csv` for the §8 structured-vs-black-box table.\n"
            "4. No shared-catalog rebuild is needed — the export reads the decision suites directly."
        ),
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
    out_dir = Path(__file__).resolve().parent
    for model in MODELS:
        global _uid
        _uid = 0
        nb = build_notebook(model)
        path = out_dir / f"t2_wpfrag_{model}.ipynb"
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n")
        print(f"wrote {path}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
