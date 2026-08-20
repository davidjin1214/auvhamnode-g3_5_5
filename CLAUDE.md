# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project Overview

Research codebase for AUV dynamics modeling using structured port-Hamiltonian Neural ODEs on SE(3). Compares `phnode_full` against ablations and black-box baselines via long-horizon rollout accuracy under clean and noisy initial conditions, with ocean current (`oc`) as the primary environment. Paper writeup: `paper/drafts/auvhamnode_thesis_chapter_zh.tex` (consumes `analysis/oc_data_catalog/` and `analysis/section8_current_evidence/`).

Orientation: `AGENTS.md` (full workflow), `EXPERIMENT_PROGRESS_TRACKER.md` (timeline), `docs/repo_structure_audit.md` (delete candidates).

## Working copies

`D:\Codes\g3_5_5` (outside OneDrive) is the working copy and the only place work
happens: writing, LaTeX, code changes and all git operations. `origin` (GitHub) is the
only push target.

The OneDrive path is a read-only backup plus the source of the heavy run artifacts
(`checkpoints/`, `data/`, `analysis/oc_data_catalog/`) that are not in version control.
Since 2026-08-20 it is no longer pushed to, so its git state is stale by design — never
read it to judge what changed recently. Never compile LaTeX under the OneDrive path
either — that is where the `*-A Mac mini.aux` conflict copies came from.
Details in `docs/repo_structure_audit.md` §6.2.

## Commands

All commands run under `conda activate mytorch1`.

```bash
# Data
python data_collection.py --num_traj 500 --blocks 150 --seed 42 --save_dir ./data/oc --workers 4 --ocean_current --current_speed_max 0.5

# Single training (clean / noisy IC)
python train_auv_hamnode.py --dataset ./data/oc/<dataset>.pkl --model_type phnode_full --save_dir ./checkpoints
python train_auv_hamnode.py --dataset ./data/oc/<dataset>.pkl --model_type phnode_full --save_dir ./checkpoints \
  --noise_profile nominal_train --noise_warmup_epochs 20 --noise_ramp 80 --noise_mix_ratio 0.5

# Rollout eval
python evaluate_rollout_benchmark.py --checkpoint ./checkpoints/<run>/best_model.pt \
  --mode resampled --noise_profiles clean nominal_eval degraded_eval heading_biased_eval \
  --output_dir ./checkpoints/<run>/rollout_benchmark

# Preferred sweep workflow
bash scripts/train_all_models_noise_profile.sh --profile oc --group core --noise-profile nominal_train
bash scripts/eval_all_models_noise_profile.sh --suite-dir ./checkpoints/<suite>

# Analysis
python scripts/summarize_sweep.py         --suite-dir ./checkpoints/<suite>
python scripts/build_experiment_report.py --suite-dir ./checkpoints/<suite>
python scripts/build_oc_data_catalog.py
```

Deprecated: `scripts/{train,eval}_all_models_noise.sh` and `--noise_level`.

No `tests/` dir — validate via the smallest affected workflow (one training job / one rollout / one catalog rebuild).

## Architecture

Pipeline: **Data → Training → Rollout Eval → Catalog/Analysis**

- `remus100_core.py` — REMUS 100 physics simulator (Euler + quaternion).
- `data_collection.py` — dataset generator; `oc`/`noc` in filename drives trainer defaults.
- `AUVHamNODE.py` — main `phnode_full` SE(3) model (augmented ODE state; learnable M⁻¹, V, D, J, B, actuator τ).
- `auv_baselines.py` — 10 variants: 3 `phnode_*`, 2 `se3_*_blackbox`, 1 `blackbox_fullstate`, 4 `ablate_*`.
- `auv_model_registry.py` — name→class map; add new models here.
- `train_auv_hamnode.py` / `train_utils.py` — training entrypoint and the largest support file (config, SE(3) loss, profile noise injection, eval helpers).
- `evaluate_rollout_benchmark.py` + `rollout_benchmark_{engine,reporting}.py` — rollout eval stack.
- `scripts/build_oc_data_catalog.py` — rebuilds the normalized catalog CSVs under `analysis/oc_data_catalog/`.
- `scripts/{oc_catalog_templates,query_oc_catalog_examples,export_section8_t2_evidence,register_existing_runs_as_suite}.py` — catalog helpers.
- `analysis/provenance_audit/` — investigation notes for the active `phnode_full` audit (do not delete).

## Conventions

- **Noise:** IC-only, profile-based. Profiles: `clean`, `nominal_train`, `nominal_eval`, `degraded_eval`, `heading_biased_eval`, `current_bias_eval` (OC checkpoints only; auto-selected only under the `remus100_ins` reference, not the default `remus100_dr`). Budgets are Remus100 DR/inertial-grounded — see `docs/noise_model_design.md`.
- **Run artifacts:** every training run writes `config.json`, `training_history.pkl` (prefer over `.log`), `best_model.pt`, `block_evaluation.json`, `heldout_evaluation.json`. New evidence-bearing runs must also write `_audit_meta/{code_revision,environment}.txt`. Rollout outputs go under `rollout_benchmark/` in the same run dir.
- **Catalog:** CSVs are generated — never hand-edit. The only hand-editable sidecar is `analysis/oc_data_catalog/evidence_status_overrides.csv`. Default to `canonical_rollout_*` tables; touch raw `rollout_*` only when all variants are needed.
- **Evidence gate:** before citing any catalog row, check `evidence_status`. `phnode_full clean seed42/46` are `stale_environment_drift` — use the cleanrun v1 baseline instead (`docs/phase1a_oc_v4lite_cleanrun_v1_report.md`).
- **Catalog deep-dives:** field defs `docs/oc_data_catalog_dictionary.md`; selection rules `docs/oc_result_selection_policy.md`; active audit `docs/provenance_audit_phnode_full_clean.md`.
- **Checkpoint retention:** per-run classes in `docs/checkpoints_retention_manifest.csv` (regenerate with `scripts/build_checkpoints_retention_manifest.py`). Check `retention_class` before deleting or citing a run dir; storage/OneDrive facts in `docs/repo_structure_audit.md` §6.

## Off-limits / Stale

- `original/bf3n/` — legacy reference, not active code.
- `checkpoints/unused/` — old noise design, invalid for current evidence.
- `smoke1` / `probe` / `smoke_v4lite` checkpoint dirs — flow-validation only, not paper evidence.
  Do **not** generalize this to every dir with `smoke` in the name: `sweep_oc_phase1a_smoke3_*_cleanrun_v1`
  holds 27 of the 45 cleanrun-v1 decision runs and is evidence-bearing. Go by `retention_class`, not by name.

## Project Claude tooling (`.claude/`)

Repo-local skills that wrap the workflows above (prefer over reconstructing the steps by hand):
- `/catalog-refresh [suite_dir]` — runs the 3-step catalog rebuild (catalog → summary → report) as one step; prevents downstream tables drifting when a step is skipped. Use instead of the three Analysis commands above.
- `/provenance-snapshot <run_dir>` — writes the reproducibility snapshot (git SHA, dataset checksum, env, config hash) to `<run_dir>/provenance/`.

The `provenance-auditor` and `catalog-consistency-reviewer` subagents (auto-listed in the agent registry) cover single-run and catalog consistency audits.
