# AUV port-Hamiltonian Neural ODE on `SE(3)`

This repository studies **underwater vehicle dynamics modeling** with structured Neural ODEs, with a focus on:

- learning AUV dynamics on `SE(3)`
- comparing structured port-Hamiltonian models with black-box baselines
- evaluating long-horizon rollout behavior
- studying robustness under **initial-condition noise** in ocean-current (`oc`) settings

The codebase is a flat Python research repo. It is usable end-to-end today for:

- dataset generation
- single-run training
- clean and noisy sweep training
- rollout benchmark evaluation
- sweep-level summary/report generation
- experiment result cataloging and canonical result export

## Repository Status

The current repo state already includes:

- clean-data and noisy-data `oc` sweeps under `checkpoints/`
- follow-up experiments for noisy robustness
- summary/report scripts for model sweeps
- a structured result catalog under `analysis/oc_data_catalog/`

If you are new to the repo, the most useful current documents are:

- [docs/noise_model_design.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/noise_model_design.md)
  Current noisy-IC design background
- [docs/oc_experiments_comprehensive_report.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/oc_experiments_comprehensive_report.md)
  Main experiment summary
- [docs/oc_followup_results_p1_p2.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/oc_followup_results_p1_p2.md)
  Follow-up results that update parts of the main summary
- [docs/oc_data_catalog_plan.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/oc_data_catalog_plan.md)
  Data catalog design and current organization
- [docs/oc_data_catalog_dictionary.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/oc_data_catalog_dictionary.md)
  Field dictionary for the catalog tables
- [docs/oc_result_selection_policy.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/oc_result_selection_policy.md)
  Canonical rollout selection rules
- [docs/oc_catalog_template_usage.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/oc_catalog_template_usage.md)
  Ready-to-use plotting/export templates

## Quick Start

All local commands should be run in the Conda environment `mytorch1`.

```bash
conda activate mytorch1
```

### 1. Generate a dataset

`noc` dataset:

```bash
python data_collection.py \
  --num_traj 500 \
  --blocks 150 \
  --seed 42 \
  --save_dir ./data/noc \
  --workers 4
```

`oc` dataset:

```bash
python data_collection.py \
  --num_traj 500 \
  --blocks 150 \
  --seed 42 \
  --save_dir ./data/oc \
  --workers 4 \
  --ocean_current \
  --current_speed_max 0.5
```

### 2. Train a single model

Clean training:

```bash
python train_auv_hamnode.py \
  --dataset ./data/oc/<dataset>.pkl \
  --model_type phnode_full \
  --save_dir ./checkpoints
```

Noisy IC training:

```bash
python train_auv_hamnode.py \
  --dataset ./data/oc/<dataset>.pkl \
  --model_type phnode_full \
  --save_dir ./checkpoints \
  --noise_profile nominal_train \
  --noise_warmup_epochs 20 \
  --noise_ramp 80 \
  --noise_mix_ratio 0.5
```

### 3. Run rollout evaluation

```bash
python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run>/best_model.pt \
  --mode resampled \
  --noise_profiles clean nominal_eval degraded_eval heading_biased_eval \
  --output_dir ./checkpoints/<run>/rollout_benchmark
```

## Recommended Entry Points

For most work, you do **not** need to manually orchestrate every step.

### Dataset generation

- `data_collection.py`

### Single-run training

- `train_auv_hamnode.py`

### Rollout benchmark

- `evaluate_rollout_benchmark.py`

### Current recommended sweep wrappers

For the current profile-based noisy-IC workflow, prefer:

- `scripts/train_all_models_noise_profile.sh`
- `scripts/eval_all_models_noise_profile.sh`

Example:

```bash
bash scripts/train_all_models_noise_profile.sh \
  --profile oc \
  --group core \
  --noise-profile nominal_train

bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/<suite_name>
```

### Sweep summary/report

- `scripts/summarize_sweep.py`
- `scripts/build_experiment_report.py`

### Result catalog and template exports

- `scripts/build_oc_data_catalog.py`
- `scripts/query_oc_catalog_examples.py`
- `scripts/oc_catalog_templates.py`

## Project Layout

Top-level files worth knowing:

- `AUVHamNODE.py`
  Main structured port-Hamiltonian model
- `auv_baselines.py`
  Baseline and ablation models
- `auv_model_registry.py`
  Model registry and name mapping
- `train_auv_hamnode.py`
  Main training entrypoint
- `train_utils.py`
  Training config, logging, persistence, evaluation helpers
- `data_collection.py`
  Dataset generation
- `evaluate_rollout_benchmark.py`
  Rollout benchmark entrypoint
- `rollout_benchmark_engine.py`
  Benchmark execution
- `rollout_benchmark_reporting.py`
  Benchmark summary/report generation

Important directories:

- `scripts/`
  Sweep wrappers, summary scripts, catalog utilities, template exporters
- `docs/`
  Experiment reports, noise design notes, catalog documentation
- `data/`
  Generated datasets
- `checkpoints/`
  Trained runs and sweep suites
- `analysis/oc_data_catalog/`
  Cataloged experiment tables and canonical views
- `original/bf3n/`
  Reference material only, not the active implementation

## Main Experimental Axes

The repo is organized around two practical axes:

### 1. Environment type

- `noc`: no ocean current
- `oc`: ocean current included in the state/simulation

### 2. Training regime

- clean training
- noisy IC training with profile-based noise

The current research emphasis is on `oc` experiments, especially **clean vs noisy IC training** and long-horizon rollout robustness.

## Supported Model Families

Main structured model:

- `phnode_full`

Other core models:

- `phnode_merged_force`
- `phnode_qforce`
- `se3_momentum_blackbox`
- `se3_accel_blackbox`
- `blackbox_fullstate`

Current ablations:

- `ablate_no_mass_prior`
- `ablate_diag_damping`
- `ablate_no_lift`
- `ablate_bu_only`

## Noise Workflow

The current implementation uses **profile-based IC-only noise**.

Important profiles:

- `clean`
- `nominal_train`
- `nominal_eval`
- `degraded_eval`
- `heading_biased_eval`

Recommended training profile:

- `nominal_train`

Recommended benchmark profiles for `oc`:

- `clean nominal_eval degraded_eval heading_biased_eval`

For details, see:

- [docs/noise_model_design.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/noise_model_design.md)
- [docs/noise_cli_parameter_reference.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/noise_cli_parameter_reference.md)
- [docs/noise_cli_command_templates.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/noise_cli_command_templates.md)

## Training Outputs

A typical run directory under `checkpoints/` contains:

- `config.json`
- `training.log`
- `training_history.pkl`
- `best_model.pt`
- `block_evaluation.json`
- `heldout_evaluation.json`
- optional checkpoint snapshots
- rollout results under `rollout_benchmark/`

`training_history.pkl` is the preferred structured source for training curves.

## Result Catalog

The repo now includes a structured result catalog for `oc` experiments:

- [analysis/oc_data_catalog/run_inventory.csv](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/analysis/oc_data_catalog/run_inventory.csv)
- [analysis/oc_data_catalog/file_inventory.csv](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/analysis/oc_data_catalog/file_inventory.csv)
- [analysis/oc_data_catalog/training_history_long.csv](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/analysis/oc_data_catalog/training_history_long.csv)
- [analysis/oc_data_catalog/block_eval_long.csv](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/analysis/oc_data_catalog/block_eval_long.csv)
- [analysis/oc_data_catalog/heldout_eval_long.csv](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/analysis/oc_data_catalog/heldout_eval_long.csv)
- [analysis/oc_data_catalog/rollout_summary_long.csv](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/analysis/oc_data_catalog/rollout_summary_long.csv)
- [analysis/oc_data_catalog/rollout_outcomes_long.csv](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/analysis/oc_data_catalog/rollout_outcomes_long.csv)
- [analysis/oc_data_catalog/rollout_run_registry.csv](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/analysis/oc_data_catalog/rollout_run_registry.csv)
- [analysis/oc_data_catalog/canonical_rollout_summary_long.csv](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/analysis/oc_data_catalog/canonical_rollout_summary_long.csv)
- [analysis/oc_data_catalog/canonical_rollout_outcomes_long.csv](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/analysis/oc_data_catalog/canonical_rollout_outcomes_long.csv)

Use these rules:

- use raw tables when you want **all recorded results**
- use canonical tables when you want the repo’s **default citation/plotting view**

Rebuild the catalog with:

```bash
conda run -n mytorch1 python scripts/build_oc_data_catalog.py
```

## Plotting and Export Templates

Minimal query/export helpers:

- `scripts/query_oc_catalog_examples.py`
- `scripts/oc_catalog_templates.py`

Example: plot `train_total` and `test_total` for one run:

```bash
conda run -n mytorch1 python scripts/oc_catalog_templates.py \
  plot-training-curves \
  --run-uid sweep_oc_core_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_115414/main_phnode_full_seed42 \
  --metric-key train_total \
  --metric-key test_total \
  --output analysis/oc_data_catalog/examples/main_phnode_full_seed42_total_loss.png
```

Example: export canonical `60s final_position_error median` table for noisy runs:

```bash
conda run -n mytorch1 python scripts/oc_catalog_templates.py \
  export-rollout-table \
  --canonical \
  --train-type noisy_train \
  --eval-profile clean \
  --eval-profile nominal_eval \
  --eval-profile degraded_eval \
  --eval-profile heading_biased_eval \
  --output analysis/oc_data_catalog/examples/noisy_train_60s_final_position_error_median.csv
```

See:

- [docs/oc_catalog_template_usage.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/oc_catalog_template_usage.md)

## Suggested First Reads

If you only want a fast orientation, read in this order:

1. `README.md`
2. [docs/noise_model_design.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/noise_model_design.md)
3. [docs/oc_experiments_comprehensive_report.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/oc_experiments_comprehensive_report.md)
4. [docs/oc_followup_results_p1_p2.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/oc_followup_results_p1_p2.md)
5. [docs/oc_data_catalog_dictionary.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/oc_data_catalog_dictionary.md)

## Practical Notes

- Keep `oc` or `noc` in dataset filenames. The code uses it to infer defaults.
- Prefer `scripts/train_all_models_noise_profile.sh` and `scripts/eval_all_models_noise_profile.sh` for new noisy experiments.
- Prefer `training_history.pkl` over `training.log` for plotting.
- Prefer canonical rollout tables over raw rollout tables when making default figures.
- `checkpoints/unused/` should not be treated as active experiment results.

## Validation and Testing

There is no separate `tests/` directory yet.

When modifying the repo, validate the smallest affected workflow:

- dataset generation change: run `data_collection.py`
- trainer/model change: run one small training job
- evaluation/report change: run one rollout benchmark or one summary script
- catalog change: rebuild `analysis/oc_data_catalog/`

## Notes for Contributors

- Generated artifacts under `data/`, `checkpoints/`, and `analysis/oc_data_catalog/` are working outputs, not source code.
- Do not hand-edit catalog CSV files; regenerate them from scripts.
- The repo contains many historical experiment files. When in doubt, treat `docs/` and the catalog tables as the authoritative orientation layer, not random checkpoint subdirectories.
