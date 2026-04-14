# Repository Guidelines

## Purpose
This repository is an active research codebase for **AUV dynamics modeling on `SE(3)`**, centered on:

- structured port-Hamiltonian Neural ODE models
- ocean-current (`oc`) vs no-current (`noc`) datasets
- clean vs noisy initial-condition training
- long-horizon rollout benchmark evaluation

The repo already contains substantial experiment artifacts and a structured result catalog. When orienting yourself, prefer the documented workflow and catalog tables over ad hoc inspection of old checkpoint folders.

## Current Orientation Layer
Before diving into random files, start from these:

- `README.md`
- `docs/noise_model_design.md`
- `docs/oc_experiments_comprehensive_report.md`
- `docs/oc_followup_results_p1_p2.md`
- `docs/oc_data_catalog_plan.md`
- `docs/oc_data_catalog_dictionary.md`
- `docs/oc_result_selection_policy.md`

For current result lookup, prefer:

- `analysis/oc_data_catalog/run_inventory.csv`
- `analysis/oc_data_catalog/rollout_run_registry.csv`
- `analysis/oc_data_catalog/canonical_rollout_summary_long.csv`
- `analysis/oc_data_catalog/canonical_rollout_outcomes_long.csv`

## Project Structure & Module Organization
This is a flat Python repo. Important active files:

- `AUVHamNODE.py`: main structured model
- `auv_baselines.py`: baselines and ablations
- `auv_model_registry.py`: model registry
- `train_auv_hamnode.py`: training entrypoint
- `train_utils.py`: config parsing, logging, checkpoint I/O, evaluation helpers
- `data_collection.py`: dataset generation
- `evaluate_rollout_benchmark.py`: rollout benchmark entrypoint
- `rollout_benchmark_engine.py`: benchmark execution
- `rollout_benchmark_reporting.py`: benchmark summaries

Important directories:

- `scripts/`: batch workflows, summaries, catalog tools, export templates
- `docs/`: experiment notes, runbooks, reports, catalog docs
- `data/`: generated datasets
- `checkpoints/`: trained runs and sweep suites
- `analysis/oc_data_catalog/`: normalized experiment tables and canonical views
- `original/bf3n/`: reference material only, not the active implementation

## Preferred Commands
Run all local commands in the Conda environment `mytorch1`.

```bash
conda activate mytorch1
```

Single-run workflow:

```bash
python data_collection.py --num_traj 500 --blocks 150 --save_dir ./data/oc --workers 4 --ocean_current
python train_auv_hamnode.py --dataset ./data/oc/<dataset>.pkl --model_type phnode_full --save_dir ./checkpoints
python train_auv_hamnode.py --dataset ./data/oc/<dataset>.pkl --model_type phnode_full --save_dir ./checkpoints --noise_profile nominal_train --noise_warmup_epochs 20 --noise_ramp 80 --noise_mix_ratio 0.5
python evaluate_rollout_benchmark.py --checkpoint ./checkpoints/<run>/best_model.pt --output_dir ./checkpoints/<run>/rollout_benchmark --mode resampled --noise_profiles clean nominal_eval degraded_eval heading_biased_eval
```

Current recommended sweep workflow:

```bash
bash scripts/train_all_models_noise_profile.sh --profile oc --group core --noise-profile nominal_train
bash scripts/eval_all_models_noise_profile.sh --suite-dir ./checkpoints/<suite>
```

Summaries and catalog:

```bash
python scripts/summarize_sweep.py --suite-dir ./checkpoints/<suite>
python scripts/build_experiment_report.py --suite-dir ./checkpoints/<suite>
python scripts/build_oc_data_catalog.py
```

Template utilities:

```bash
python scripts/query_oc_catalog_examples.py ...
python scripts/oc_catalog_templates.py ...
```

## Workflow Preferences
For new work:

- Prefer `scripts/train_all_models_noise_profile.sh` and `scripts/eval_all_models_noise_profile.sh` for noisy `oc` experiments.
- Prefer `training_history.pkl` over `training.log` for training curves.
- Prefer `canonical_rollout_summary_long.csv` and `canonical_rollout_outcomes_long.csv` for default plotting and reporting.
- Use raw rollout tables only when you intentionally need all rollout variants.
- Treat `checkpoints/unused/` as non-active unless explicitly needed.

## Coding Style & Naming Conventions
Follow current Python style:

- 4-space indentation
- `snake_case` for functions and variables
- `PascalCase` for classes
- concise module docstrings where useful

Keep code modular. Prefer small helper functions over long inline logic. Comments must be in English and only added when the code would otherwise be hard to parse.

No formatter or linter is enforced in the repo, so stay PEP 8-aligned and consistent with surrounding files.

## Editing and Generated Files
Do not hand-edit generated outputs such as:

- `data/*`
- `checkpoints/*`
- `analysis/oc_data_catalog/*.csv`

If you need catalog changes, modify:

- `scripts/build_oc_data_catalog.py`
- `scripts/query_oc_catalog_examples.py`
- `scripts/oc_catalog_templates.py`

then regenerate.

## Testing and Validation
There is no dedicated `tests/` directory yet. Validate changes with the smallest affected workflow:

- data logic: run `data_collection.py`
- model/trainer logic: run a single training job
- benchmark/report logic: run a single evaluation or sweep summary
- catalog logic: rebuild `analysis/oc_data_catalog/`

When changing experiment logic, record:

- the command used
- the relevant output path
- key structured outputs such as `config.json`, `heldout_evaluation.json`, or rollout `summary.json`

## Commit & PR Guidelines
Keep commits short and task-focused. Recent history often uses concise Chinese subjects; concise English imperative messages are also fine.

Examples:

- `Refine OC catalog canonical export`
- `更新 noisy sweep 汇总逻辑`

PRs should explain:

- the motivation
- the modified entrypoints
- the validation commands actually run
- any changed reports, plots, or catalog outputs
