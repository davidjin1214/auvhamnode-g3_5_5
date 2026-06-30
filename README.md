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
- a Phase-1A `v4-lite` cleanrun decision package
- a provenance audit for catalog-era `phnode_full clean` seed fragility
- summary/report scripts for model sweeps
- a structured result catalog under `analysis/oc_data_catalog/`
- a human-readable experiment progress tracker
- a `paper/` writing workspace for the AUVHamNODE thesis chapter, including current rewrite notes and the active LaTeX draft (complete 10-section chapter, PDF compiled, under final-pass revision)

If you are new to the repo, see the **[Documentation map](#documentation-map)** near the end for an ordered reading path and the full doc index.

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

### 4. Run a multi-model sweep (preferred)

For profile-based noisy-IC work, prefer these wrappers over orchestrating runs by hand:

```bash
bash scripts/train_all_models_noise_profile.sh \
  --profile oc \
  --group core \
  --noise-profile nominal_train

bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/<suite_name>
```

### 5. Summarize, report, and catalog

```bash
python scripts/summarize_sweep.py         --suite-dir ./checkpoints/<suite>
python scripts/build_experiment_report.py --suite-dir ./checkpoints/<suite>
python scripts/build_oc_data_catalog.py
```

Catalog query/plot helpers (`scripts/query_oc_catalog_examples.py`, `scripts/oc_catalog_templates.py`) are covered under [Plotting and Export Templates](#plotting-and-export-templates).

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
- `paper/`
  Thesis-chapter writing materials, review notes, active LaTeX draft (complete 10-section chapter), and deprecated intermediate drafts
- `original/bf3n/`
  Delete-candidate legacy reference material; not the active implementation

For the current keep/deprecate/delete-candidate boundary, see:

- [docs/repo_structure_audit.md](docs/repo_structure_audit.md)

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

- [docs/noise_model_design.md](docs/noise_model_design.md)
- [docs/noise_cli_parameter_reference.md](docs/noise_cli_parameter_reference.md)
- [docs/noise_cli_command_templates.md](docs/noise_cli_command_templates.md)

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

For new evidence-bearing training runs, also record provenance under `_audit_meta/`:

- `_audit_meta/code_revision.txt`
- `_audit_meta/environment.txt`

These files are needed to distinguish code and environment drift when comparing runs across local and cloud mirrors.

## Result Catalog

The repo includes a structured result catalog for `oc` experiments under [`analysis/oc_data_catalog/`](analysis/oc_data_catalog/), with generated CSVs in four groups:

- **inventories** — `run_inventory.csv`, `file_inventory.csv`, `rollout_run_registry.csv`
- **per-metric long tables** — `training_history_long.csv`, `block_eval_long.csv`, `heldout_eval_long.csv`
- **raw rollout** — `rollout_summary_long.csv`, `rollout_outcomes_long.csv`
- **canonical rollout** — `canonical_rollout_summary_long.csv`, `canonical_rollout_outcomes_long.csv`

Field definitions: [docs/oc_data_catalog_dictionary.md](docs/oc_data_catalog_dictionary.md).

Use these rules:

- use raw tables when you want **all recorded results**
- use canonical tables when you want the repo’s **default citation/plotting view**
- check `analysis/oc_data_catalog/evidence_status_overrides.csv` before treating a canonical row as current evidence

Important citation caveat:

- catalog-era `phnode_full clean seed42/46` results are marked `stale_environment_drift` by the provenance audit
- do not use the old ~11 m `phnode_full clean` 5-seed mean as model-fragility evidence
- the aligned cleanrun v1 / current-main baseline for `phnode_full clean` is 0.6767 m for 60s clean `pos_err_median` 5-seed mean

Rebuild the catalog with:

```bash
conda run -n mytorch1 python scripts/build_oc_data_catalog.py
```

## Thesis Chapter Drafting

The `paper/` directory is the current workspace for writing the AUVHamNODE method as a Chinese doctoral thesis chapter. Start from:

- [paper/README.md](paper/README.md)
  Writing entrypoint, document roles, progress board, and current chapter structure
- [paper/drafts/auvhamnode_thesis_chapter_review_notes_zh.md](paper/drafts/auvhamnode_thesis_chapter_review_notes_zh.md)
  Strict review of the deprecated intermediate draft and constraints for rewriting
- [paper/drafts/auvhamnode_thesis_chapter_zh.tex](paper/drafts/auvhamnode_thesis_chapter_zh.tex)
  Active LaTeX draft of the formal thesis chapter — complete 10-section draft (PDF compiled), under final-pass revision

The previous draft has been downgraded to `paper/drafts/deprecated/auvhamnode_thesis_chapter_zh_intermediate_20260519.tex`. Treat it as a source of formulas and reusable material only, not as the main text to polish line by line. The active thesis chapter should use a formal method definition before introducing `AUVHamNODE` as a shorthand, avoid internal writing-plan language, and keep experiment conclusions gated on current evidence status.

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

- [docs/oc_catalog_template_usage.md](docs/oc_catalog_template_usage.md)

## Documentation map

The full doc index, in suggested reading order:

1. `README.md`
2. [EXPERIMENT_PROGRESS_TRACKER.md](EXPERIMENT_PROGRESS_TRACKER.md)
3. [docs/repo_structure_audit.md](docs/repo_structure_audit.md)
4. [docs/experiment_stages_overview.md](docs/experiment_stages_overview.md)
5. [docs/provenance_audit_phnode_full_clean.md](docs/provenance_audit_phnode_full_clean.md)
6. [docs/phase1a_oc_v4lite_cleanrun_v1_report.md](docs/phase1a_oc_v4lite_cleanrun_v1_report.md)
7. [docs/phnode_realistic_validation_plan.md](docs/phnode_realistic_validation_plan.md)
8. [docs/phnode_realistic_validation_execution_plan.md](docs/phnode_realistic_validation_execution_plan.md)
9. [docs/phase1_realistic_validation_plan.md](docs/phase1_realistic_validation_plan.md)
10. [docs/noise_model_design.md](docs/noise_model_design.md)
11. [docs/oc_data_catalog_dictionary.md](docs/oc_data_catalog_dictionary.md)
12. [docs/oc_result_selection_policy.md](docs/oc_result_selection_policy.md)
13. [paper/README.md](paper/README.md)

### Deeper references

- [docs/oc_experiments_comprehensive_report.md](docs/oc_experiments_comprehensive_report.md) — main experiment summary
- [docs/oc_followup_results_p1_p2.md](docs/oc_followup_results_p1_p2.md) — follow-up results that update parts of the main summary
- [docs/oc_data_catalog_plan.md](docs/oc_data_catalog_plan.md) — catalog design and organization

## Validation and Testing

There is no separate `tests/` directory yet.

When modifying the repo, validate the smallest affected workflow:

- dataset generation change: run `data_collection.py`
- trainer/model change: run one small training job
- evaluation/report change: run one rollout benchmark or one summary script
- catalog change: rebuild `analysis/oc_data_catalog/`

## Notes for Contributors

- Generated artifacts under `data/`, `checkpoints/`, and `analysis/oc_data_catalog/` are working outputs, not source code.
- Do not hand-edit generated catalog CSV files; regenerate them from scripts. The explicit sidecar exception is `analysis/oc_data_catalog/evidence_status_overrides.csv`.
- The repo contains many historical experiment files. When in doubt, treat `docs/` and the catalog tables as the authoritative orientation layer, not random checkpoint subdirectories.
- Keep `oc` or `noc` in dataset filenames — the code infers trainer defaults from it.
- `scripts/train_all_models_noise.sh`, `scripts/eval_all_models_noise.sh`, and the `--noise_level` interface are deprecated compatibility paths; prefer the `*_noise_profile.sh` wrappers.
- `checkpoints/unused/` is not active evidence (older incorrect noise design), and smoke/probe checkpoint dirs are flow-validation only — neither belongs in headline conclusions.
