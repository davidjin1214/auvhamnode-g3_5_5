# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for AUV (Autonomous Underwater Vehicle) dynamics modeling using structured port-Hamiltonian Neural ODEs on SE(3). The main focus is comparing `phnode_full` against ablations and black-box baselines, evaluated by long-horizon rollout accuracy under clean and noisy initial conditions, with ocean current (`oc`) as the primary environment.

> See `AGENTS.md` for the full orientation checklist, workflow preferences, coding conventions, and commit guidelines.

## Environment

All commands require:
```bash
conda activate mytorch1
```

## Common Commands

**Data generation:**
```bash
# oc dataset
python data_collection.py --num_traj 500 --blocks 150 --seed 42 --save_dir ./data/oc --workers 4 --ocean_current --current_speed_max 0.5
```

**Single training run:**
```bash
# Clean
python train_auv_hamnode.py --dataset ./data/oc/<dataset>.pkl --model_type phnode_full --save_dir ./checkpoints

# Noisy IC
python train_auv_hamnode.py --dataset ./data/oc/<dataset>.pkl --model_type phnode_full --save_dir ./checkpoints \
  --noise_profile nominal_train --noise_warmup_epochs 20 --noise_ramp 80 --noise_mix_ratio 0.5
```

**Rollout evaluation:**
```bash
python evaluate_rollout_benchmark.py --checkpoint ./checkpoints/<run>/best_model.pt \
  --mode resampled --noise_profiles clean nominal_eval degraded_eval heading_biased_eval \
  --output_dir ./checkpoints/<run>/rollout_benchmark
```

**Sweep workflows (preferred for multi-model runs):**
```bash
bash scripts/train_all_models_noise_profile.sh --profile oc --group core --noise-profile nominal_train
bash scripts/eval_all_models_noise_profile.sh --suite-dir ./checkpoints/<suite>
```

**Analysis:**
```bash
python scripts/summarize_sweep.py --suite-dir ./checkpoints/<suite>
python scripts/build_experiment_report.py --suite-dir ./checkpoints/<suite>
python scripts/build_oc_data_catalog.py    # rebuild all catalog CSVs
```

**Validation (no tests dir — use smallest affected workflow):**
- Model/trainer change → run one small training job
- Evaluation change → run one rollout benchmark
- Catalog change → rebuild `analysis/oc_data_catalog/`

## Architecture

The pipeline is: **Data Generation → Training → Rollout Evaluation → Catalog/Analysis**

### Data & Physics Layer
- `remus100_core.py` — REMUS 100 AUV physics simulator (Euler angles + quaternions)
- `data_collection.py` — wraps the simulator to produce `.pkl` trajectory datasets; `oc`/`noc` in the filename is used by the trainer to infer defaults

### Model Layer
- `AUVHamNODE.py` — main `phnode_full` model; learnable submodules: inverse mass M⁻¹, potential energy V(q), damping D(ν_r), lift J(ν_r), actuator force B(u), and actuator time constants; augmented ODE state encodes position (3), rotation matrix (9), relative body velocity (6), actuator states, commands, and optionally ocean current (3) + depth (1)
- `auv_baselines.py` — all 10 model variants: 3 structured (`phnode_*`), 2 semi-structured (`se3_*_blackbox`), 1 fully black-box, 4 ablations (`ablate_*`)
- `auv_model_registry.py` — central name → class mapping used by all entry points; add new models here

### Training Layer
- `train_auv_hamnode.py` — CLI entrypoint; delegates config parsing and checkpoint I/O to `train_utils.py`
- `train_utils.py` — config dataclass, SE(3) trajectory loss, profile-based noise injection (warmup→ramp→steady mix), logging, evaluation helpers; this is the largest and most complex file

### Evaluation Layer
- `evaluate_rollout_benchmark.py` — CLI entrypoint for rollout benchmarks
- `rollout_benchmark_engine.py` — executes multi-profile rollouts from a checkpoint; handles resampled vs fixed-IC modes
- `rollout_benchmark_reporting.py` — aggregates per-rollout results into `summary.json` and structured outputs

### Analysis Layer
- `scripts/build_oc_data_catalog.py` — ingests all `checkpoints/` artifacts and writes normalized CSVs to `analysis/oc_data_catalog/`
- `scripts/oc_catalog_templates.py` — plotting and export helpers that read from the catalog CSVs

### Key Data Flow Invariants
- A training run writes: `config.json`, `training_history.pkl` (preferred over `.log`), `best_model.pt`, `block_evaluation.json`, `heldout_evaluation.json`
- Rollout evaluation appends results under `rollout_benchmark/` inside the same run directory
- Catalog CSVs are generated artifacts — never hand-edit; always regenerate via `build_oc_data_catalog.py`
- For figures/tables, default to `canonical_rollout_*` tables; use raw `rollout_*` tables only when all rollout variants are needed

## Noise System

Profile-based IC-only noise (not force noise). Profiles: `clean`, `nominal_train`, `nominal_eval`, `degraded_eval`, `heading_biased_eval`. Hardware-grounded budgets based on Remus100 dead-reckoning inertial reference. See `docs/noise_model_design.md` for design rationale.

## Result Catalog

Normalized tables under `analysis/oc_data_catalog/`. Key files:
- `run_inventory.csv` — all trained runs
- `canonical_rollout_summary_long.csv` / `canonical_rollout_outcomes_long.csv` — default citation view
- `rollout_run_registry.csv` — maps rollout benchmark folders to runs

See `docs/oc_data_catalog_dictionary.md` for field definitions and `docs/oc_result_selection_policy.md` for canonical selection rules.

## Reference Material

- `original/bf3n/` — reference only, not the active implementation
- `checkpoints/unused/` — inactive experiments, ignore by default
