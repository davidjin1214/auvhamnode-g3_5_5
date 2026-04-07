# AUVHamNODE Workflow

This repository studies AUV dynamics modeling with a port-Hamiltonian Neural ODE on `SE(3)`.

Core files:

- `AUVHamNODE.py`: main structured model
- `train_auv_hamnode.py`: training entrypoint
- `evaluate_rollout_benchmark.py`: rollout benchmark entrypoint
- `data_collection.py`: dataset generation entrypoint
- `remus100_core.py`: REMUS-100 dynamics and simulator core

## Environment

Use a Python environment with the required packages installed, especially:

- `torch`
- `torchdiffeq`
- `numpy`
- `matplotlib`

## Recommended Workflow

For each experiment:

1. Generate a dataset.
2. Train a model on that dataset.
3. Run rollout benchmark evaluation from `best_model.pt`.

The main experiment axes are:

- ocean current: `noc` vs `oc`
- training noise: clean vs noisy

## 1. Generate Data

### CLI

```bash
python data_collection.py \
  --num_traj 500 \
  --blocks 150 \
  --seed 42 \
  --save_dir ./data \
  --workers 4
```

Optional switches:

- `--ocean_current`: generate an ocean-current dataset
- `--absolute_depth_context`: append block-start absolute depth `z_ref`
- `--current_speed_max 0.5`: max inertial current speed for `oc`
- `--no_filter`: disable trajectory filtering
- `--no_figures`: skip dataset figures

### Recommended commands

No current:

```bash
python data_collection.py \
  --num_traj 500 \
  --blocks 150 \
  --seed 42 \
  --save_dir ./data/noc \
  --workers 4
```

With current:

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

### Output files

The dataset filename is generated automatically and looks like:

```text
auv_noc_traj500_blk150_s42_<dataset_id>.pkl
auv_oc_traj500_blk150_s42_<dataset_id>.pkl
```

Alongside the `.pkl`, the generator also writes:

- `.meta.json`
- `.stats.json`
- `.stats.npz`
- `.summary.txt`
- `<dataset_stem>_figures/`

## 2. Train Models

### Important behavior

- The documentation uses canonical model names only.
- Training defaults depend on dataset kind inferred from the dataset filename.
- Keep `noc` or `oc` in the dataset filename. This matters because the script uses that name to choose default hyperparameters.
- The dataset metadata is authoritative for `ocean_current`, `u_dim`, and related state layout settings.

### Main model

The main structured model is:

```text
--model_type phnode_full
```

Other supported model types:

Core models:

- `phnode_full`
- `phnode_merged_force`
- `phnode_qforce`
- `se3_momentum_blackbox`
- `se3_accel_blackbox`
- `blackbox_fullstate`

Ablations:

- `ablate_no_mass_prior`
- `ablate_diag_damping`
- `ablate_no_lift`
- `ablate_bu_only`

### Clean training

No current, no noise:

```bash
python train_auv_hamnode.py \
  --dataset ./data/noc/auv_noc_traj500_blk150_s42_<dataset_id>.pkl \
  --model_type phnode_full \
  --save_dir ./checkpoints
```

With current, no noise:

```bash
python train_auv_hamnode.py \
  --dataset ./data/oc/auv_oc_traj500_blk150_s42_<dataset_id>.pkl \
  --model_type phnode_full \
  --save_dir ./checkpoints
```

### Noisy training

The current implementation uses **IC-only noise regularization**. Noise is
applied to the rollout initial condition only, using profile-based settings.

Recommended profile for training:

- `--noise_profile nominal_train`

No current, with training noise:

```bash
python train_auv_hamnode.py \
  --dataset ./data/noc/auv_noc_traj500_blk150_s42_<dataset_id>.pkl \
  --model_type phnode_full \
  --save_dir ./checkpoints \
  --noise_profile nominal_train
```

With current, with training noise:

```bash
python train_auv_hamnode.py \
  --dataset ./data/oc/auv_oc_traj500_blk150_s42_<dataset_id>.pkl \
  --model_type phnode_full \
  --save_dir ./checkpoints \
  --noise_profile nominal_train \
  --block_eval_noise_profiles clean nominal_eval \
  --heldout_eval_noise_profiles clean nominal_eval degraded_eval
```

Useful noise-related overrides:

```bash
--noise_warmup_epochs 20
--noise_ramp 80
--noise_mix_ratio 0.5
--noise_scale 1.0
--block_eval_noise_profiles clean nominal_eval
--heldout_eval_noise_profiles clean nominal_eval degraded_eval
```

Available profiles:

- `clean`: disable noisy IC
- `nominal_train`: recommended mild training regularization
- `nominal_eval`: evaluation profile for normal navigation uncertainty
- `degraded_eval`: stronger evaluation-only stress profile

Legacy `--noise_level {0,1,2,3}` is still accepted for backward compatibility,
but `--noise_profile` is the preferred interface.

Post-training automatic evaluation is also profile-aware:

- block evaluation defaults to `clean nominal_eval`
- held-out evaluation defaults to `clean nominal_eval degraded_eval`

You can override them, for example:

```bash
--block_eval_noise_profiles clean degraded_eval
--heldout_eval_noise_profiles all
```

Pass `none` to skip a phase entirely:

```bash
--block_eval_noise_profiles none
--heldout_eval_noise_profiles none
```

### Current-aware feature options

These are mainly relevant for `oc` experiments:

- `--dj_current_feature {none,current_body,total_velocity}`
- `--actuation_current_feature {none,current_body,total_velocity}`

Recommended starting point for `oc`:

```bash
--dj_current_feature current_body \
--actuation_current_feature current_body
```

In practice, the script already provides dataset-aware defaults, so you can usually omit these unless you are running an ablation.

### Useful explicit overrides

```bash
--batch_size 4096
--epochs 300
--total_steps 5000
--lr 6e-3
--min_lr 1e-4
--warmup_steps 400
--hidden_dim 128
```

### Training outputs

Each run creates a directory under `./checkpoints/`, containing at least:

- `config.json`
- `training.log`
- `training_history.pkl`
- `best_model.pt`
- `checkpoint_<epoch>.pt` at save intervals
- `block_evaluation.json`
- `heldout_evaluation.json`
- `evaluation_results.pkl`
- `heldout_evaluation.pkl`

### Batch sweep scripts

For repeated benchmark sweeps, use the scripts in [scripts/](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/scripts).

Train all core models and ablations on the default `oc` dataset with seeds `42 43 44`:

```bash
./scripts/batch_train_models.sh
```

Run the matching rollout benchmark for every trained run in that sweep:

```bash
./scripts/batch_eval_models.sh \
  --suite-dir ./checkpoints/<suite_name>
```

Run both stages end-to-end:

```bash
./scripts/run_all.sh
```

This wrapper now runs training, rollout evaluation, and final sweep summarization in one pass.

### Noise-profile sweep scripts

For the current profile-based noisy-IC workflow, prefer these wrappers:

Train a noisy sweep:

```bash
bash ./scripts/train_all_models_noise_profile.sh \
  --profile oc \
  --group core \
  --noise-profile nominal_train \
  --noise-warmup-epochs 20 \
  --noise-ramp 80 \
  --noise-mix-ratio 0.5 \
  --noise-scale 1.0
```

Run the matching benchmark, sweep summary, and experiment report:

```bash
bash ./scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/<suite_name> \
  --mode heldout \
  --noise-profiles "clean nominal_eval degraded_eval"
```

These wrappers sit on top of `batch_train_models.sh` and `batch_eval_models.sh`,
but expose the current noise controls directly. They are the recommended entry
point for new noisy-training experiments.

Summarize a completed sweep across seeds and models:

```bash
python ./scripts/summarize_sweep.py \
  --suite-dir ./checkpoints/<suite_name>
```

Useful sweep controls:

```bash
--profile oc
--group core
--group ablation
--group all
--seeds "42 43 44"
--prefix oc_core_default
--device cuda:0
```

Useful noisy-sweep controls:

```bash
--noise-profile nominal_train
--noise-scale 1.0
--noise-warmup-epochs 20
--noise-ramp 80
--noise-mix-ratio 0.5
--block-eval-noise-profiles "clean nominal_eval"
--heldout-eval-noise-profiles "clean nominal_eval degraded_eval"
```

Each sweep creates a suite directory under `./checkpoints/`, for example:

```text
checkpoints/sweep_oc_all_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_<timestamp>/
```

Inside that suite, every run gets its own training folder, and each rollout benchmark is written back into the corresponding run directory under `rollout_benchmark/`.

## 3. Evaluate Rollout Benchmark

Evaluation is run from a trained checkpoint, usually `best_model.pt`.

### Held-out benchmark

This uses held-out test trajectories from the dataset.

```bash
python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run_name>/best_model.pt \
  --mode heldout \
  --output_dir ./rollout_benchmark_results \
  --noise_profiles clean nominal_eval degraded_eval
```

If the checkpoint already stores the dataset path, `--dataset` is optional.

### Resampled benchmark

This regenerates benchmark trajectories from the stored dataset generation config.

```bash
python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run_name>/best_model.pt \
  --mode resampled \
  --output_dir ./rollout_benchmark_results \
  --noise_profiles clean nominal_eval degraded_eval
```

### Useful evaluation options

```bash
--dataset ./data/oc/auv_oc_traj500_blk150_s42_<dataset_id>.pkl
--num_traj_per_scenario 30
--times 10 30 60
--scenarios PRBS CHIRP OU
--noise_profiles clean nominal_eval degraded_eval
--noise_seed 2024
--device cuda
--run_name oc_noisy_eval
--num_diagnostic_plots 6
```

`--noise_profiles` accepts one, several, or `all`. For example:

```bash
--noise_profiles clean
--noise_profiles nominal_eval degraded_eval
--noise_profiles all
```

If you pass multiple noise profiles, benchmark outputs are written into
profile-specific subdirectories under the resolved run directory, for example:

- `.../clean/summary.txt`
- `.../nominal_eval/summary.txt`
- `.../degraded_eval/summary.txt`

### Evaluation outputs

Each benchmark run creates a timestamped directory under `./rollout_benchmark_results/`, containing:

- `summary.txt`
- `summary.json`
- `trajectory_metrics.csv`
- `horizon_metrics.csv`
- `time_series_metrics.csv`
- `rollout_outcomes.csv`
- `diagnostic_cases.csv`
- `diagnostic_plots.csv`
- `metric_contract.csv`
- `error_growth.png`
- `terminal_error_boxplots.png`
- `example_rollouts.png`
- `diagnostic_plots/`

If you request multiple noise profiles, the benchmark creates one subdirectory
per profile, for example:

```text
rollout_benchmark_results/<run_name>/clean/
rollout_benchmark_results/<run_name>/nominal_eval/
rollout_benchmark_results/<run_name>/degraded_eval/
```

## Experiment Matrix

Recommended minimum experiment set for the main model:

### A. No current, clean

Dataset:

```bash
python data_collection.py --save_dir ./data/noc
```

Training:

```bash
python train_auv_hamnode.py \
  --dataset ./data/noc/auv_noc_traj500_blk150_s42_<dataset_id>.pkl \
  --model_type phnode_full
```

Evaluation:

```bash
python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run_name>/best_model.pt \
  --mode heldout
```

### B. No current, noisy

Training:

```bash
python train_auv_hamnode.py \
  --dataset ./data/noc/auv_noc_traj500_blk150_s42_<dataset_id>.pkl \
  --model_type phnode_full \
  --noise_profile nominal_train
```

### C. Current, clean

Dataset:

```bash
python data_collection.py \
  --save_dir ./data/oc \
  --ocean_current \
  --current_speed_max 0.5
```

Training:

```bash
python train_auv_hamnode.py \
  --dataset ./data/oc/auv_oc_traj500_blk150_s42_<dataset_id>.pkl \
  --model_type phnode_full
```

### D. Current, noisy

Training:

```bash
python train_auv_hamnode.py \
  --dataset ./data/oc/auv_oc_traj500_blk150_s42_<dataset_id>.pkl \
  --model_type phnode_full \
  --noise_profile nominal_train
```

This is the most realistic and most important setting if your goal is a convincing real-world dynamics story.

## Practical Notes

### 1. `oc` noise behavior

The current root version uses an `oc`-aware **IC-consistent** noise scheme:

- noise is defined around the initial navigation-state uncertainty, not a full noisy sequence
- OC runs perturb the model-space relative velocity budget and then reconstruct a consistent data-space state
- the implementation preserves consistency between `nu_r`, `nu_total`, `R`, and `v_c^n`

This is intentionally simpler than the older trajectory-level noise design. If
you need the detailed rationale, see
[docs/noise_robustness_experiment_design_codex.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/noise_robustness_experiment_design_codex.md).

### 2. When to use `heldout` vs `resampled`

- Use `heldout` first. It is the most direct comparison against the test split.
- Use `resampled` when you want a more benchmark-style stress test using regenerated trajectories from the stored simulation config.

### 3. Depth context

Only use `--absolute_depth_context` in data generation if you want models that need block-start absolute depth context. This is mainly relevant if you later plan to enable depth-conditioned potential or force variants.

### 4. Figures during data generation

Dataset figure export is useful for inspection but can slow generation. For quick iterations:

```bash
--no_figures
```

### 5. GPU selection

Training and evaluation both accept:

```bash
--device cuda
```

or

```bash
--device cpu
```

If omitted, the scripts choose CUDA when available.

## Suggested First Runs

If you want a compact first pass, run these two experiments first:

1. `noc + clean`
2. `oc + noisy`

That gives you one controlled baseline and one realistic target setting.

## Example End-to-End

You can also use shell variables to avoid editing the same path repeatedly.

```bash
DATASET=./data/oc/auv_oc_traj500_blk150_s42_<dataset_id>.pkl
RUN_DIR=./checkpoints/<run_name>
CKPT=$RUN_DIR/best_model.pt
```

Generate `oc` data:

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

Train `phnode_full` with noise:

```bash
python train_auv_hamnode.py \
  --dataset "$DATASET" \
  --model_type phnode_full \
  --save_dir ./checkpoints \
  --noise_profile nominal_train
```

Evaluate rollout benchmark:

```bash
python evaluate_rollout_benchmark.py \
  --checkpoint "$CKPT" \
  --mode heldout \
  --output_dir ./rollout_benchmark_results \
  --times 10 30 60 \
  --scenarios PRBS CHIRP OU
```

Or run the same workflow as a sweep:

```bash
bash ./scripts/train_all_models_noise_profile.sh \
  --profile oc \
  --group core

bash ./scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/<suite_name> \
  --mode heldout
```

## File Lookup Tips

Find generated datasets:

```bash
find ./data -name "*.pkl"
```

Find trained checkpoints:

```bash
find ./checkpoints -name "best_model.pt"
```

Find benchmark summaries:

```bash
find ./rollout_benchmark_results -name "summary.txt"
```
