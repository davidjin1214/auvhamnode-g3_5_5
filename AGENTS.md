# Repository Guidelines

## Project Structure & Module Organization
This repository is a flat Python research codebase for AUV dynamics modeling on `SE(3)`. Core training and model code lives at the top level: `AUVHamNODE.py`, `auv_baselines.py`, `auv_model_registry.py`, `train_auv_hamnode.py`, and `train_utils.py`. Data generation starts from `data_collection.py`. Rollout evaluation is handled by `evaluate_rollout_benchmark.py`, `rollout_benchmark_engine.py`, and `rollout_benchmark_reporting.py`. Batch workflows live in `scripts/`. Treat `original/bf3n/` as reference material, not the active implementation. Generated artifacts belong under `data/` and `checkpoints/` and should stay out of commits.

## Build, Test, and Development Commands
Run all local commands in the Conda environment `mytorch1`.

```bash
conda activate mytorch1
python data_collection.py --num_traj 500 --blocks 150 --save_dir ./data/noc --workers 4
python train_auv_hamnode.py --dataset ./data/noc/<dataset>.pkl --model_type phnode_full --save_dir ./checkpoints
python evaluate_rollout_benchmark.py --checkpoint ./checkpoints/<run>/best_model.pt --output_dir ./checkpoints/<run>/rollout_benchmark
scripts/run_all.sh --profile oc --group all
```

Use `scripts/batch_train_models.sh` for sweep training, `scripts/batch_eval_models.sh` for sweep evaluation, and `scripts/summarize_sweep.py` or `scripts/build_experiment_report.py` for result summaries.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and short module-level docstrings where useful. Keep code modular and prefer small helper functions over long inline blocks. Comments must be in English and only added when the intent is not obvious from the code. Match current dependency style; no formatter or linter is configured in this repo, so keep changes PEP 8-aligned and consistent with surrounding files.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes by running the smallest affected workflow: dataset generation for `data_collection.py`, single-run training for model or trainer changes, and rollout evaluation for benchmark/reporting changes. When touching experiment logic, include the exact command used and summarize key outputs such as `config.json`, `heldout_evaluation.json`, or rollout `summary.txt`.

## Commit & Pull Request Guidelines
Recent history uses short, task-focused commit subjects, often in Chinese. Keep commit messages concise and imperative, for example: `Refine ocean-current noise defaults` or `调整 baseline sweep 配置`. PRs should explain the experimental motivation, list modified entrypoints, include the validation commands you ran, and attach plots or report snippets when metrics or visual diagnostics changed. Do not commit ignored artifacts such as `data/`, `checkpoints/`, `*.pkl`, or model weights.
