# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for modeling AUV (Autonomous Underwater Vehicle) dynamics using a **port-Hamiltonian Neural ODE on SE(3)**. The main model (`phnode_full`) decomposes dynamics into physically interpretable components: learned mass matrix M^{-1}, potential energy V(q), velocity-dependent damping D(nu_r), lift J(nu_r), actuator mapping B, and first-order actuator lag. The ODE state is augmented with exogenous channels (actuator commands, ocean current, depth reference).

## Environment

```bash
conda activate mytorch1
```

Dependencies: `torch`, `torchdiffeq`, `numpy`, `matplotlib`. No package manager config (no requirements.txt/pyproject.toml) — use the existing conda env.

## Commands

### Data generation
```bash
python data_collection.py --num_traj 500 --blocks 150 --seed 42 --save_dir ./data/oc --workers 4 --ocean_current --current_speed_max 0.5
```

### Training (single run)
```bash
# Clean training
python train_auv_hamnode.py --dataset ./data/oc/<dataset>.pkl --model_type phnode_full --save_dir ./checkpoints

# Noisy IC training
python train_auv_hamnode.py --dataset ./data/oc/<dataset>.pkl --model_type phnode_full --save_dir ./checkpoints --noise_profile nominal_train
```

### Evaluation
```bash
python evaluate_rollout_benchmark.py --checkpoint ./checkpoints/<run>/best_model.pt --mode heldout --noise_profiles clean nominal_eval degraded_eval
```

### Batch sweeps (preferred for experiments)
```bash
bash scripts/train_all_models_noise_profile.sh --profile oc --group core --noise-profile nominal_train
bash scripts/eval_all_models_noise_profile.sh --suite-dir ./checkpoints/<suite>
python scripts/summarize_sweep.py --suite-dir ./checkpoints/<suite>
```

## Architecture

**Flat layout** — all core modules live at repo root. `original/bf3n/` is frozen reference material.

### Data flow

1. `data_collection.py` → generates trajectory `.pkl` datasets using `remus100_core.py` (REMUS-100 simulator)
2. `train_auv_hamnode.py` (`AUVHamNODETrainer`) → loads dataset, trains model, runs post-training block + heldout evaluation
3. `evaluate_rollout_benchmark.py` → standalone rollout benchmarks from trained checkpoints

### Key modules

| File | Role |
|------|------|
| `AUVHamNODE.py` | Model definition. `StateLayout` (frozen dataclass) defines ODE state slicing. `AUVHamNODE.forward()` computes the pH ODE RHS. `to_ode_state()`/`to_data_state()` convert between data convention (nu_total) and ODE convention (nu_r) via `_shift_linear_velocity`. |
| `train_utils.py` | All training infrastructure: `NoiseConfig`, `TrainConfig`, `StateNormalizer`, SE(3) loss, data loading, noise injection (`build_noisy_initial_condition`), evaluation metrics. This is the largest file (~1600 lines). |
| `auv_model_registry.py` | Central model name → builder mapping. `ModelSpec` metadata, `instantiate_model()`. |
| `auv_baselines.py` | Baseline model implementations (blackbox, merged force, qforce, ablations). |
| `rollout_benchmark_engine.py` | Benchmark execution engine: trajectory rollout, metric computation, reporting. |
| `rollout_benchmark_reporting.py` | Report formatting, summary tables, diagnostic plots. |
| `remus100_core.py` | REMUS-100 dynamics model and simulator (ground truth). |

### State conventions

The model operates in two state conventions:
- **Data-space**: uses `nu_total` (total body velocity = v_r + R^T v_c^n)
- **ODE-space**: uses `nu_r` (velocity relative to water)

`model.to_ode_state()` converts data→ODE; `model.to_data_state()` converts ODE→data. The shift only affects linear velocity channels [0:3] when `ocean_current=True`.

### Noise system (IC-only)

Noise is injected **only at the initial condition** of each training rollout (not throughout the trajectory). Key design:
- Profile-based: `clean`, `nominal_train`, `nominal_eval`, `degraded_eval`
- Curriculum learning: `warmup_epochs` → `ramp_epochs` → steady-state `mix_ratio`
- Deterministic seeding via `_sample_scaled_noise()` with `sample_ids` + `base_seed` + `stream`
- Rotation perturbation uses SO(3) exponential map
- Noise design documents: `docs/noise_model_design.md`, `docs/noise_parameter_revision_sensor_grounded.md`

### Model types

Core: `phnode_full`, `phnode_merged_force`, `phnode_qforce`, `se3_momentum_blackbox`, `se3_accel_blackbox`, `blackbox_fullstate`

Ablations: `ablate_no_mass_prior`, `ablate_diag_damping`, `ablate_no_lift`, `ablate_bu_only`

### Dataset kinds

Training hyperparameters auto-adapt based on dataset filename:
- `noc` — no ocean current (simpler dynamics)
- `oc` — with ocean current (more complex, carries `v_c^n` and uses current-aware features)

## Coding Conventions

- 4-space indent, `snake_case` functions, `PascalCase` classes
- No formatter/linter configured — follow PEP 8 and match surrounding code
- Comments in English
- Commit messages may be in Chinese or English, short and task-focused
- Generated artifacts (`data/`, `checkpoints/`, `*.pkl`, `*.pt`) are gitignored

## Validation

No test suite. Validate changes by running the smallest affected workflow:
- Model/trainer changes → single-run training
- Noise changes → training with `--noise_profile nominal_train`
- Benchmark changes → `evaluate_rollout_benchmark.py` on existing checkpoint
