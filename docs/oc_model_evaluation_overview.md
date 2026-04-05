# Port-Hamiltonian Neural ODE AUV Models: Design, Evaluation Protocol, and Results

## Purpose

This document is the recommended entry point for understanding the ocean-current (`oc`) experiment family in this repository. It consolidates:

- what each model variant is designed to test
- how training-side and rollout-side metrics are defined
- how seed quality is judged
- what the current results support

It is intended to replace ad hoc reading across many files under `checkpoints/`.

Primary raw references:
- [all-model seed audit](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/all_model_review_seed_audit.md)
- [all-model compact summary](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/all_model_review_summary_compact.md)
- [all-model detailed summary](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/all_model_review_summary_detailed.md)
- [core vs ablation historical summary](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/oc_core_vs_oc_ablation_summary.md)
- [PHNODE 6-seed focus summary](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/phnode_focus_6seed_summary.md)

## Experiment Scope

- Dataset: `data/auv_oc_traj1000_blk150_s23_d0be9434.pkl`
- Core sweep: seeds `42,43,44`
- Ablation sweep: seeds `42,43,44`
- PHNODE supplementary sweep: seeds `45,46,47`
- Combined all-model view:
  - core baselines use `42,43,44`
  - PHNODE family uses `42,43,44,45,46,47` where available

The main deployment-oriented metric is:
- `resampled @ 60s final position error median`

The reason is simple: short replay metrics can look excellent even when long-horizon rollout is poor.

## Model Families

Model metadata comes from [auv_model_registry.py](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/auv_model_registry.py).

### Main PHNODE

| Model | Registry Description | Experimental Role |
| --- | --- | --- |
| `main/phnode_full` | Exact `SE(3)` + mass + scalar potential + split `D/J/B` | Main structured port-Hamiltonian model |

### Core Baselines

| Model | Registry Description | Experimental Role |
| --- | --- | --- |
| `baseline/phnode_merged_force` | pH core with a single merged non-conservative force branch | Tests a weaker structured pH decomposition |
| `baseline/phnode_qforce` | Structured pH model with a generic configuration-dependent force | Strongest current baseline; keeps useful structure without the full split design |
| `baseline/se3_momentum_blackbox` | Exact `SE(3)` + constant mass matrix, with black-box momentum dynamics | Tests whether exact geometry plus simpler dynamics is enough |
| `baseline/se3_accel_blackbox` | Exact `SE(3)` kinematics with black-box acceleration dynamics | Weaker physical prior than momentum-space baseline |
| `baseline/blackbox_fullstate` | Fully unstructured state-derivative model | Fully black-box reference |

### Ablations

| Model | Registry Description | Experimental Role |
| --- | --- | --- |
| `ablation/ablate_no_mass_prior` | AUVHamNODE without physics-based mass initialization | Tests the value of the mass prior |
| `ablation/ablate_diag_damping` | AUVHamNODE with diagonal damping only | Tests the value of coupled damping structure |
| `ablation/ablate_no_lift` | AUVHamNODE without the learned skew-symmetric lift term | Tests whether the lift term helps or mainly harms optimization |
| `ablation/ablate_bu_only` | AUVHamNODE with actuation conditioned only on actuator states | Tests whether the removed actuation-conditioning structure is essential |

## Evaluation Protocol

### Training-Side Metrics

Each run produces:

- `best_test_loss`
  Source: `training.log`
  Meaning: best validation/test loss reached during training under the existing training loop
- `block_position_rmse_mean`
  Source: `block_evaluation.json`
  Meaning: local replay quality on sampled short blocks
- `heldout_position_rmse_median`
  Source: `heldout_evaluation.json`
  Meaning: trajectory replay quality on held-out trajectories

These metrics are important, but they are not sufficient for model selection in this project.

### Rollout Benchmark

The rollout benchmark is run with a fixed evaluation protocol:

- mode: `resampled`
- trajectories per scenario: `30`
- scenarios: `PRBS`, `CHIRP`, `OU`
- horizons: `10s`, `30s`, `60s`
- rollout seed: `42`

For each horizon we record:

- final position error median
- final position error p95
- completion rate
- model failure rate
- scenario-wise breakdown for `PRBS`, `CHIRP`, `OU`

### Why `60s resampled` Is The Main Metric

Several poor models still look acceptable on:

- block replay
- held-out replay
- short-horizon rollout

But they degrade strongly on resampled long-horizon rollout. This is especially clear for:

- `ablation/ablate_bu_only`
- `baseline/blackbox_fullstate`
- bad-outlier seeds of `main/phnode_full`

Therefore the most reliable headline metric is:
- `resampled @ 60s final position error median`

## Seed-Audit Rule

This project now uses a strict separation between:

- structurally poor families
- stable families
- families with a stable cluster plus bad-outlier seeds

### Rule

A seed is marked as problematic only if it is a bad outlier relative to a stable cluster of sibling seeds. In practice, a seed is considered problematic when:

- its `60s` rollout error is far outside the stable cluster
- and that abnormality is supported by additional evidence such as:
  - abnormal `10s` and `30s` rollout errors
  - abnormal training loss
  - abnormal heldout or block replay error
  - non-zero rollout divergence or other failure signs

Additional interpretation rules:

- If most seeds are already poor, do not create a pruned variant.
- If more than about one third of seeds meet the bad-outlier criterion, treat the whole family as unstable rather than rescued by pruning.
- All-seed results are always primary.
- Pruned views are diagnostic only.

### Current Audit Outcome

| Model | Audit Status | Problematic Seeds | Interpretation |
| --- | --- | --- | --- |
| `baseline/phnode_qforce` | `stable` | none | Strong and consistent |
| `ablation/ablate_no_lift` | `stable` | none | Best current PHNODE family |
| `baseline/se3_momentum_blackbox` | `stable` | none | Consistent baseline |
| `ablation/ablate_no_mass_prior` | `stable` | none | Strong and stable ablation |
| `baseline/phnode_merged_force` | `stable` | none | Solid but weaker than q-force |
| `baseline/se3_accel_blackbox` | `stable` | none | Weaker and more variable, but not a bad-outlier case |
| `ablation/ablate_diag_damping` | `stable` | none | Consistently degraded, but not because of isolated bad seeds |
| `main/phnode_full` | `prunable_bad_outliers` | `42,46` | Strong stable cluster plus a real bad-outlier failure mode |
| `ablation/ablate_bu_only` | `structurally_poor` | none | Broadly poor across seeds; do not prune |
| `baseline/blackbox_fullstate` | `family_unstable` | `42,43` | Family-level instability; pruning is not a meaningful rescue |

## Compact Results Table

This is the quickest useful ranking view.

| Model | Seeds | Audit | Train | Heldout | 10s | 30s | 60s | Comp60 | PRBS60 | CHIRP60 | OU60 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline/phnode_qforce` | `42,43,44` | stable | 0.0039 | 0.0001 | 0.0663 | 0.2131 | 0.5708 | 98.1% | 0.3942 | 0.6194 | 0.6517 |
| `ablation/ablate_no_lift` | `42,43,44,45,46,47` | stable | 0.0045 | 0.0001 | 0.1003 | 0.3736 | 1.0022 | 99.3% | 0.9148 | 1.0748 | 1.0874 |
| `baseline/se3_momentum_blackbox` | `42,43,44` | stable | 0.0058 | 0.0002 | 0.1423 | 0.4663 | 1.1647 | 98.5% | 1.1046 | 1.0101 | 1.4964 |
| `ablation/ablate_no_mass_prior` | `42,43,44,45,46,47` | stable | 0.0049 | 0.0001 | 0.1453 | 0.5449 | 1.4409 | 98.5% | 1.5670 | 1.1105 | 1.7641 |
| `baseline/phnode_merged_force` | `42,43,44` | stable | 0.0063 | 0.0002 | 0.1346 | 0.5080 | 1.4991 | 97.8% | 1.1903 | 1.4909 | 1.7316 |
| `baseline/se3_accel_blackbox` | `42,43,44` | stable | 0.0389 | 0.0002 | 0.3277 | 1.2606 | 3.8874 | 98.1% | 3.0709 | 4.4931 | 6.6170 |
| `ablation/ablate_diag_damping` | `42,43,44,45,46,47` | stable | 0.0211 | 0.0014 | 0.6473 | 2.0768 | 4.1659 | 98.3% | 4.9850 | 2.9249 | 5.9399 |
| `main/phnode_full` | `42,43,44,45,46,47` | prunable_bad_outliers | 0.0510 | 0.0004 | 0.6620 | 3.1903 | 9.0863 | 98.5% | 9.4134 | 8.7915 | 8.6274 |
| `ablation/ablate_bu_only` | `42,43,44,45,46,47` | structurally_poor | 0.0589 | 0.0007 | 2.2007 | 11.2395 | 27.7967 | 94.1% | 39.8126 | 20.9290 | 31.2086 |
| `baseline/blackbox_fullstate` | `42,43,44` | family_unstable | 0.0115 | 0.0039 | 16.8398 | 52.6307 | 86.7755 | 18.9% | 72.9683 | 94.1638 | 77.6991 |

## Main Findings

### 1. The strongest overall model is still `baseline/phnode_qforce`

This model remains the best all-seed result on the main `60s` rollout metric.

### 2. The strongest PHNODE family is `ablation/ablate_no_lift`

This is the most important PHNODE-level conclusion right now. It is:

- the best PHNODE family on all-seed rollout accuracy
- the most reliable PHNODE family
- clearly better than `main/phnode_full` in all-seed deployment-style evaluation

### 3. `main/phnode_full` has both high potential and a real robustness defect

The stable-cluster seeds `43,44,45,47` are strong. Their diagnostic-only cluster summary is:

| Model Variant | Kept Seeds | 10s | 30s | 60s | Comp60 |
| --- | --- | --- | --- | --- | --- |
| `main/phnode_full [drop_42-46]` | `43,44,45,47` | 0.0860 | 0.2604 | 0.6098 | 98.1% |

This means:

- the model family has a strong ceiling
- but all-seed usage is currently unreliable because of bad-outlier seeds `42,46`

So the correct reading is not “slightly seed-sensitive.” It is:
- strong stable cluster plus a real bad-outlier failure mode

### 4. The `bu_only`-related structure is essential

`ablation/ablate_bu_only` is poor across the whole family:

- 10s already degrades
- 30s degrades strongly
- 60s degrades severely
- completion also drops

This is not a single unlucky seed. It is structural evidence that this removed component matters.

### 5. Coupled damping matters

`ablation/ablate_diag_damping` is consistently weaker than the best PHNODE variants and much worse than the stable cluster of `main/phnode_full`.

This supports keeping the richer damping structure.

### 6. The current mass prior is not yet justified as essential

`ablation/ablate_no_mass_prior` remains both:

- strong
- stable

Under this training setup, there is still no evidence that the current mass prior is necessary for the best rollout behavior.

### 7. Fully black-box modeling is clearly inferior here

`baseline/blackbox_fullstate` is catastrophically poor. It is not just low-performing; it is family-level unstable under this evaluation protocol.

## Historical Interpretation Changes

Some earlier intermediate summaries used a weaker pruning rule. Those should now be read as superseded.

The key corrections are:

- `main/phnode_full`
  Old reading: only `seed46` is a catastrophic outlier
  Current reading: `seed42` and `seed46` are both problematic bad outliers

- `ablation/ablate_bu_only`
  Old reading: can be shown with a `[drop 45]` diagnostic variant
  Current reading: should be treated as structurally poor, with no pruned variant

- `baseline/blackbox_fullstate`
  Old reading: can be partially described by dropping `42,43`
  Current reading: family-level unstable; the pruned view is not a meaningful headline result

## Recommended Reporting Practice

When writing papers, slides, or notes:

- use all-seed results for headline claims
- keep diagnostic pruned views clearly separated
- use `60s resampled final position median` as the main selection metric
- use `10s/30s/60s` and `PRBS/CHIRP/OU` as supporting evidence

A good concise statement of the current evidence is:

1. `baseline/phnode_qforce` is the best overall model.
2. `ablation/ablate_no_lift` is the best PHNODE family under current training.
3. `main/phnode_full` has a strong stable cluster but a real bad-outlier seed failure mode.
4. The `bu_only`-related component and coupled damping are both important.
5. The current mass prior is lower priority than robustness and lift-related optimization.

## Reproduction And Maintenance

The current consolidated summaries are generated by:

- [build_cross_model_summary.py](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/scripts/build_cross_model_summary.py)

If new seeds or new sweeps are added, regenerate the summaries first and then update this document if the conclusions change.

## Figure And Report Index

### Recommended Reading Order

1. [Chinese results section](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/oc_results_section_zh.md)
2. [All-model compact summary](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/all_model_review_summary_compact.md)
3. [All-model detailed summary](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/all_model_review_summary_detailed.md)
4. [All-model seed audit](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/all_model_review_seed_audit.md)

### Sweep Reports

| Sweep | Report | Metrics Table |
| --- | --- | --- |
| Core sweep | [experiment_report.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_core_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_115414/experiment_report.md) | [sweep_model_metrics.csv](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_core_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_115414/sweep_model_metrics.csv) |
| Ablation sweep | [experiment_report.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_ablation_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_143830/experiment_report.md) | [sweep_model_metrics.csv](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_ablation_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_143830/sweep_model_metrics.csv) |
| PHNODE supplementary sweep | [experiment_report.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47/experiment_report.md) | [sweep_model_metrics.csv](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47/sweep_model_metrics.csv) |

### Consolidated Reports

| Purpose | File |
| --- | --- |
| Full seed-level judgement | [all_model_review_seed_audit.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/all_model_review_seed_audit.md) |
| Quick ranking | [all_model_review_summary_compact.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/all_model_review_summary_compact.md) |
| Detailed metrics | [all_model_review_summary_detailed.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/all_model_review_summary_detailed.md) |
| Historical compatibility summary | [all_model_performance_with_pruned_variants.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/all_model_performance_with_pruned_variants.md) |
| Historical core-vs-ablation view | [oc_core_vs_oc_ablation_summary.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/oc_core_vs_oc_ablation_summary.md) |
| PHNODE-only six-seed view | [phnode_focus_6seed_summary.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/phnode_focus_6seed_summary.md) |

### Representative Diagnostic Figures

These are useful when explaining why the updated seed-audit rule marks `main/phnode_full` seeds `42` and `46` as problematic, and why `ablation/ablate_no_lift` is currently the most reliable PHNODE family.

| Use Case | File |
| --- | --- |
| `main/phnode_full` bad-outlier seed `42`: rollout growth | [error_growth.png](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_core_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_115414/main_phnode_full_seed42/rollout_benchmark/resampled_traj30_seed42_20260405_004246/error_growth.png) |
| `main/phnode_full` bad-outlier seed `42`: example rollouts | [example_rollouts.png](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_core_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_115414/main_phnode_full_seed42/rollout_benchmark/resampled_traj30_seed42_20260405_004246/example_rollouts.png) |
| `main/phnode_full` catastrophic seed `46`: rollout growth | [error_growth.png](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47/main_phnode_full_seed46/rollout_benchmark/resampled_traj30_seed42_20260405_111554/error_growth.png) |
| `main/phnode_full` catastrophic seed `46`: terminal error spread | [terminal_error_boxplots.png](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47/main_phnode_full_seed46/rollout_benchmark/resampled_traj30_seed42_20260405_111554/terminal_error_boxplots.png) |
| `main/phnode_full` catastrophic seed `46`: example rollouts | [example_rollouts.png](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47/main_phnode_full_seed46/rollout_benchmark/resampled_traj30_seed42_20260405_111554/example_rollouts.png) |
| Best current PHNODE seed `ablate_no_lift_seed45`: rollout growth | [error_growth.png](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47/ablation_ablate_no_lift_seed45/rollout_benchmark/resampled_traj30_seed42_20260405_113743/error_growth.png) |
| Best current PHNODE seed `ablate_no_lift_seed45`: terminal error spread | [terminal_error_boxplots.png](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47/ablation_ablate_no_lift_seed45/rollout_benchmark/resampled_traj30_seed42_20260405_113743/terminal_error_boxplots.png) |
| Best current PHNODE seed `ablate_no_lift_seed45`: example rollouts | [example_rollouts.png](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47/ablation_ablate_no_lift_seed45/rollout_benchmark/resampled_traj30_seed42_20260405_113743/example_rollouts.png) |
