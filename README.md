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

- The training script uses canonical model names only.
- Training defaults depend on dataset kind inferred from the dataset filename.
- Keep `noc` or `oc` in the dataset filename. This matters because the script uses that name to choose default hyperparameters.
- The dataset metadata is authoritative for `ocean_current`, `u_dim`, and related state layout settings.

### Main model

The main structured model is:

```text
--model_type ph_se3_full
```

Other supported model types:

- `ph_se3_nomassinit`
- `ph_se3_diagd`
- `ph_se3_noj`
- `ph_se3_buonly`
- `ph_se3_mergednc`
- `ph_se3_qforce`
- `mom_se3_unstruct`
- `ham_se3_unstruct`
- `se3_unstruct`
- `bb_free_unstruct`

### Clean training

No current, no noise:

```bash
python train_auv_hamnode.py \
  --dataset ./data/noc/auv_noc_traj500_blk150_s42_<dataset_id>.pkl \
  --model_type ph_se3_full \
  --save_dir ./checkpoints
```

With current, no noise:

```bash
python train_auv_hamnode.py \
  --dataset ./data/oc/auv_oc_traj500_blk150_s42_<dataset_id>.pkl \
  --model_type ph_se3_full \
  --save_dir ./checkpoints
```

### Noisy training

No current, with training noise:

```bash
python train_auv_hamnode.py \
  --dataset ./data/noc/auv_noc_traj500_blk150_s42_<dataset_id>.pkl \
  --model_type ph_se3_full \
  --save_dir ./checkpoints \
  --init_state_noise \
  --observation_noise
```

With current, with training noise:

```bash
python train_auv_hamnode.py \
  --dataset ./data/oc/auv_oc_traj500_blk150_s42_<dataset_id>.pkl \
  --model_type ph_se3_full \
  --save_dir ./checkpoints \
  --init_state_noise \
  --observation_noise
```

If you want persistent sensor-style bias as well:

```bash
--observation_bias
```

If you want to change the noise ramp:

```bash
--noise_ramp_epochs 100
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

## 3. Evaluate Rollout Benchmark

Evaluation is run from a trained checkpoint, usually `best_model.pt`.

### Held-out benchmark

This uses held-out test trajectories from the dataset.

```bash
python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run_name>/best_model.pt \
  --mode heldout \
  --output_dir ./rollout_benchmark_results
```

If the checkpoint already stores the dataset path, `--dataset` is optional.

### Resampled benchmark

This regenerates benchmark trajectories from the stored dataset generation config.

```bash
python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run_name>/best_model.pt \
  --mode resampled \
  --output_dir ./rollout_benchmark_results
```

### Useful evaluation options

```bash
--dataset ./data/oc/auv_oc_traj500_blk150_s42_<dataset_id>.pkl
--num_traj_per_scenario 30
--times 10 30 60
--scenarios PRBS CHIRP OU
--device cuda
--run_name oc_noisy_eval
--num_diagnostic_plots 6
```

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
  --model_type ph_se3_full
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
  --model_type ph_se3_full \
  --init_state_noise \
  --observation_noise
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
  --model_type ph_se3_full
```

### D. Current, noisy

Training:

```bash
python train_auv_hamnode.py \
  --dataset ./data/oc/auv_oc_traj500_blk150_s42_<dataset_id>.pkl \
  --model_type ph_se3_full \
  --init_state_noise \
  --observation_noise
```

This is the most realistic and most important setting if your goal is a convincing real-world dynamics story.

## Practical Notes

### 1. `oc` noise behavior

The current root version uses a simple `oc`-aware noise scheme:

- initial-condition noise perturbs both velocity and ocean-current channels
- observation noise perturbs both body velocity and inertial current channels
- the implementation preserves consistency between `nu_r`, `nu_total`, and `v_c^n`

This is intentionally simpler than the heavier `original/bf3n` unified noise constructor.

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

Train `ph_se3_full` with noise:

```bash
python train_auv_hamnode.py \
  --dataset "$DATASET" \
  --model_type ph_se3_full \
  --save_dir ./checkpoints \
  --init_state_noise \
  --observation_noise
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
