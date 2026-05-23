#!/usr/bin/env python3
"""Generate the four per-model T2 (WP-Frag current-evidence rerun) Colab notebooks.

Each notebook reruns ONE model across the three training protocols
(clean / iid_noisy_ic@nominal_train / v4_lite@nominal_train) and five seeds
(42-46) on the current-main g3_5_7 cloud mirror, then evaluates every trained
suite across the four robustness eval profiles
(clean / nominal_eval / degraded_eval / heading_biased_eval).

The notebooks reuse the proven `scripts/run_phase1a_oc_v4lite.sh` driver and only
change per-model environment. They additionally:
  * expand IID_EVAL_PROFILES to the full 4-profile robustness set (caveat B), and
  * inject per-run `_audit_meta/{code_revision,environment}.txt` provenance,
    which the training flow does NOT write automatically.

Run locally:  python notebook/make_t2_notebooks.py
Outputs:      notebook/t2_wpfrag_<model>.ipynb  (x4)
"""
from __future__ import annotations

import json
from pathlib import Path

MODELS = [
    "phnode_full",
    "phnode_qforce",
    "ablate_no_lift",
    "ablate_no_mass_prior",
]

DATASET = "data/auv_oc_traj1000_blk150_s23_d0be9434.pkl"
PROJECT_DEFAULT = "/content/drive/MyDrive/Colab Notebooks/auvhamnode/g3_5_7"

_uid = 0


def _next_id() -> str:
    global _uid
    _uid += 1
    return f"t2cell{_uid:03d}"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _next_id(),
        "metadata": {},
        "source": text,
    }


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

# ---- T2 identity (one model per notebook) ----
os.environ["RUN_TAG"] = "{run_tag}"
os.environ["DATASET"] = "{DATASET}"
os.environ["NOISE_REFERENCE"] = "remus100_dr"
os.environ["PHASE1A_LOG_DIR"] = str(PROJECT_DIR / "checkpoints" / "phase1a_logs" / os.environ["RUN_TAG"])
os.environ["PHASE1A_METADATA_DIR"] = str(PROJECT_DIR / "checkpoints" / f"phase1a_metadata_{{os.environ['RUN_TAG']}}")

# ---- T2 matrix: single model x 3 protocols x 5 seeds ----
os.environ["PHASE1A_MODELS"] = "{model}"
os.environ["SMOKE1_MODELS"] = "{model}"
os.environ["SMOKE_SEEDS"] = "42"                 # cheap protocol-correctness gate (1 seed)
os.environ["DECISION_SEEDS"] = "42 43 44 45 46"  # full current-evidence seed set

# ---- Evaluation contract ----
os.environ["SMOKE_EVAL_NUM_TRAJ_PER_SCENARIO"] = "6"
os.environ["DECISION_EVAL_NUM_TRAJ_PER_SCENARIO"] = "30"
os.environ["EVAL_TIMES"] = "10 30 60"
os.environ["EVAL_SCENARIOS"] = "PRBS CHIRP OU"
os.environ["EVAL_BASE_SEED"] = "42"
os.environ["EVAL_NOISE_SEED"] = "2024"
os.environ["EVAL_PROGRESS_EVERY"] = "5"
os.environ["EVAL_NUM_DIAGNOSTIC_PLOTS"] = "6"
# Full robustness set: evaluates clean/iid/v4lite suites across all 4 profiles (caveat B).
os.environ["IID_EVAL_PROFILES"] = "clean nominal_eval degraded_eval heading_biased_eval"
os.environ["V4_EVAL_PROFILES"] = "nominal_eval"

# ---- Audit gate ----
os.environ["STRICT_ZERO_NOISE_AUDIT"] = "1"
os.environ["SOFT_MIN_EPOCH_SCALE"] = "0.05"

print("RUN_TAG       =", os.environ["RUN_TAG"])
print("PHASE1A_MODELS=", os.environ["PHASE1A_MODELS"])
print("DECISION_SEEDS=", os.environ["DECISION_SEEDS"])
print("IID_EVAL_PROFILES=", os.environ["IID_EVAL_PROFILES"])'''


PROVENANCE_CELL = '''# Inject per-run provenance. The training flow does NOT write these automatically,
# so for paper-grade evidence we record them here (see docs/provenance_audit_phnode_full_clean.md sec 5.2).
import os, sys, subprocess, datetime
from pathlib import Path
import torch

RUN_TAG = os.environ["RUN_TAG"]
ckpt = PROJECT_DIR / "checkpoints"
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
# Flags any run whose training.log contains the "no successful training batches" failure mode.
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
        flag = "  <-- CHECK (possible artifact)" if nbad > 0 else ""
        print(f"{suite.name[:46]:<48}{run_dir.name[:30]:<32}{str(best_epoch):>10}{str(best_loss):>14}{nbad:>6}{flag}")
print("\\nIf any run is flagged, treat that seed like seed46/seed43: do NOT cite it; rerun or annotate.")'''


def build_notebook(model: str) -> dict:
    run_tag = f"t2_wpfrag_{model}"
    cells = [
        md(
            f"# T2 WP-Frag current-evidence rerun — `{model}`\n"
            "\n"
            "**Goal.** Reproduce current-main (`g3_5_7` mirror) evidence for this one model so the\n"
            "paper Results section (chapter section 8) can be cited from a single, machine-reproducible\n"
            "suite instead of catalog-era cloud numbers contaminated by environment drift.\n"
            "\n"
            "**Matrix (this notebook).** `" + model + "` x {clean, iid_noisy_ic@nominal_train, "
            "v4_lite@nominal_train} x seeds {42,43,44,45,46} = 15 training runs, then every trained\n"
            "suite is evaluated across {clean, nominal_eval, degraded_eval, heading_biased_eval}.\n"
            "\n"
            "**Parallelism.** Run the four `t2_wpfrag_*` notebooks (one per model:\n"
            "`phnode_full`, `phnode_qforce`, `ablate_no_lift`, `ablate_no_mass_prior`) in separate\n"
            "Colab sessions. They use distinct `RUN_TAG`s and do not collide.\n"
            "\n"
            "**Why these knobs differ from phase1a:** (1) `IID_EVAL_PROFILES` is the full 4-profile\n"
            "robustness set so clean-vs-noisy matched comparison is available; (2) a provenance cell\n"
            "writes per-run `_audit_meta/` (the trainer does not). noise schedule defaults\n"
            "(warmup=20, ramp=80, mix_ratio=0.5) match the repo `nominal_train` contract.\n"
            "\n"
            "**After all four finish:** sync `checkpoints/` back, then locally rebuild the catalog\n"
            "(`scripts/build_oc_data_catalog.py`) and export the section-8 current-evidence tables.\n"
            "\n"
            "> Watch list while running: `phnode_full` seeds 42/46, `ablate_no_lift` seeds 43/44 —\n"
            "> these were catalog-era anomalies. The anomaly-scan cell flags any recurrence."
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
            "Confirms all target suite/proxy dirs are absent and saves suite-level run config + "
            "environment metadata. If it fails, change `RUN_TAG` or remove the target dirs."
        ),
        code('os.environ["MODE"] = "preflight"\n!bash scripts/run_phase1a_oc_v4lite.sh'),
        md(
            "## 2. Smoke gate (seed 42, all 3 protocols)\n"
            "Cheap protocol-correctness check (especially the `v4_lite` trajectory-consistent IC path "
            "and the strict zero-noise audit for `clean`). Important for `phnode_qforce`, which was not "
            "in the phase1a smoke set. Smoke results are flow-validation only — never cited."
        ),
        code('os.environ["MODE"] = "smoke1_train"\n!bash scripts/run_phase1a_oc_v4lite.sh'),
        code('os.environ["MODE"] = "smoke1_eval"\n!bash scripts/run_phase1a_oc_v4lite.sh'),
        md(
            "## 3. Decision train (3 protocols x 5 seeds)\n"
            "The 15 evidence-bearing training runs for this model."
        ),
        code('os.environ["MODE"] = "decision_train"\n!bash scripts/run_phase1a_oc_v4lite.sh'),
        md(
            "## 4. Decision rollout eval (each trained suite x 4 robustness profiles)\n"
            "Evaluates the clean / iid / v4lite trained suites across "
            "`clean nominal_eval degraded_eval heading_biased_eval` (iid eval protocol) plus the "
            "`v4_lite` eval at `nominal_eval`."
        ),
        code('os.environ["MODE"] = "decision_eval"\n!bash scripts/run_phase1a_oc_v4lite.sh'),
        md(
            "## 5. Provenance injection (per-run `_audit_meta/`)\n"
            "Records git HEAD + environment fingerprint into every decision run dir so these runs are "
            "distinguishable from catalog-era drift when the catalog is rebuilt."
        ),
        code(PROVENANCE_CELL),
        md(
            "## 6. Anomaly scan\n"
            "Flags any run with the `no successful training batches` catastrophic-gradient signature "
            "(the seed46 failure mode). A flagged seed must NOT be cited — rerun or annotate."
        ),
        code(ANOMALY_CELL),
        md(
            "## 7. Next step (after all 4 notebooks finish)\n"
            "1. Ensure `checkpoints/sweep_oc_phase1a_decision_{clean,iid,v4lite}_t2_wpfrag_*` are synced back to the repo.\n"
            "2. Locally: `conda run -n mytorch1 python scripts/build_oc_data_catalog.py` to ingest the new runs.\n"
            "3. Export the section-8 current-evidence tables from the rebuilt canonical views.\n"
            "4. Re-verify the `ablate_no_lift` noisy-degradation claim is not driven by a seed44 artifact."
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
