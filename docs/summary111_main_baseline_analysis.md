# `summary111.txt` Main vs Baseline Analysis

Source file: [`checkpoints/summary111.txt`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/summary111.txt)

Detailed lookup: [`docs/summary111_detailed_lookup.md`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/summary111_detailed_lookup.md)

Scenario-horizon appendix: [`docs/summary111_scenario_horizon_lookup.md`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/summary111_scenario_horizon_lookup.md)

## Scope

- Dataset/context: `oc`, `nominal_train`
- Models:
  - `main/phnode_full`
  - `baseline/phnode_merged_force`
  - `baseline/se3_momentum_blackbox`
  - `baseline/phnode_qforce`
  - `baseline/se3_accel_blackbox`
  - `baseline/blackbox_fullstate`
- Seeds: `42, 43, 44`
- Eval noise profiles: `clean`, `nominal_eval`, `degraded_eval`
- Primary comparison focus: overall `H=60s` rollout metrics averaged over 3 seeds

## H=60 Overall Ranking

### `clean`

| Model | Pos Median (m) | Pos P95 (m) | Rot Median (rad) | Completion | Pred Div |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline/phnode_merged_force` | 2.535 | 8.233 | 0.063 | 0.982 | 0.018 |
| `baseline/se3_momentum_blackbox` | 2.937 | 8.296 | 0.063 | 0.985 | 0.015 |
| `main/phnode_full` | 3.377 | 6.037 | 0.074 | 0.982 | 0.018 |
| `baseline/phnode_qforce` | 4.595 | 10.793 | 0.103 | 0.985 | 0.015 |
| `baseline/se3_accel_blackbox` | 5.372 | 14.878 | 0.137 | 0.985 | 0.015 |
| `baseline/blackbox_fullstate` | 104.361 | 198.403 | 2.953 | 0.337 | 0.663 |

Observation:
- If only看 `pos median`，`merged_force` and `momentum_blackbox` are ahead of `main/phnode_full`.
- If看 tail error，`main/phnode_full` has the best `pos p95` by a clear margin.

### `nominal_eval`

| Model | Pos Median (m) | Pos P95 (m) | Rot Median (rad) | Completion | Pred Div |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline/phnode_merged_force` | 2.844 | 8.245 | 0.063 | 0.982 | 0.018 |
| `main/phnode_full` | 3.291 | 7.371 | 0.072 | 0.982 | 0.018 |
| `baseline/se3_momentum_blackbox` | 3.312 | 8.891 | 0.068 | 0.985 | 0.015 |
| `baseline/phnode_qforce` | 4.788 | 10.722 | 0.104 | 0.985 | 0.015 |
| `baseline/se3_accel_blackbox` | 5.331 | 15.526 | 0.141 | 0.985 | 0.015 |
| `baseline/blackbox_fullstate` | 105.824 | 198.387 | 2.952 | 0.337 | 0.663 |

Observation:
- Under moderate noise, `main/phnode_full` moves to rank 2 by `pos median`.
- `main/phnode_full` still has the best `pos p95`.

### `degraded_eval`

| Model | Pos Median (m) | Pos P95 (m) | Rot Median (rad) | Completion | Pred Div |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline/phnode_merged_force` | 4.328 | 14.464 | 0.075 | 0.982 | 0.018 |
| `baseline/se3_momentum_blackbox` | 4.724 | 13.000 | 0.078 | 0.985 | 0.015 |
| `main/phnode_full` | 4.771 | 12.064 | 0.081 | 0.982 | 0.018 |
| `baseline/phnode_qforce` | 5.967 | 14.463 | 0.109 | 0.985 | 0.015 |
| `baseline/se3_accel_blackbox` | 6.220 | 18.715 | 0.149 | 0.985 | 0.015 |
| `baseline/blackbox_fullstate` | 106.044 | 198.322 | 2.948 | 0.337 | 0.663 |

Observation:
- At heavy noise, `main/phnode_full` is not best in `pos median`, but it keeps the best `pos p95`.
- `merged_force` wins the center statistic; `main/phnode_full` wins the tail statistic.

## Noise Robustness Relative To Clean

Computed from overall `H=60s` `pos median`.

| Model | Clean | Nominal | Nominal vs Clean | Degraded | Degraded vs Clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `main/phnode_full` | 3.377 | 3.291 | -2.5% | 4.771 | +41.3% |
| `baseline/phnode_merged_force` | 2.535 | 2.844 | +12.2% | 4.328 | +70.7% |
| `baseline/se3_momentum_blackbox` | 2.937 | 3.312 | +12.8% | 4.724 | +60.8% |
| `baseline/phnode_qforce` | 4.595 | 4.788 | +4.2% | 5.967 | +29.8% |
| `baseline/se3_accel_blackbox` | 5.372 | 5.331 | -0.8% | 6.220 | +15.8% |
| `baseline/blackbox_fullstate` | 104.361 | 105.824 | +1.4% | 106.044 | +1.6% |

Important interpretation:
- Relative degradation cannot be read alone.
- `qforce` and `se3_accel_blackbox` have smaller percentage degradation than `main/phnode_full`, but their absolute errors are still clearly worse.
- `blackbox_fullstate` looks “stable” only because it is already collapsed in the clean case.

## Scenario Breakdown At `degraded_eval`, `H=60s`

| Model | PRBS Pos | CHIRP Pos | OU Pos | PRBS Comp | CHIRP Comp | OU Comp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `main/phnode_full` | 5.009 | 4.283 | 4.635 | 1.000 | 1.000 | 0.944 |
| `baseline/phnode_merged_force` | 4.209 | 4.354 | 4.855 | 1.000 | 1.000 | 0.944 |
| `baseline/se3_momentum_blackbox` | 5.048 | 4.268 | 5.202 | 1.000 | 1.000 | 0.955 |
| `baseline/phnode_qforce` | 6.915 | 5.441 | 5.858 | 1.000 | 1.000 | 0.956 |
| `baseline/se3_accel_blackbox` | 6.672 | 5.420 | 6.556 | 1.000 | 1.000 | 0.956 |
| `baseline/blackbox_fullstate` | 120.355 | 103.919 | 92.860 | 0.311 | 0.355 | 0.344 |

Interpretation:
- `main/phnode_full` is strongest on `OU`.
- `merged_force` is strongest on `PRBS`.
- `main/phnode_full` and `momentum_blackbox` are essentially tied on `CHIRP`.
- For the top 3 models, the hardest scenario remains `OU`, mainly through larger long-horizon error rather than a dramatic completion collapse.

## Seed Stability

Standard deviation of overall `H=60s` `pos median` across seeds:

| Model | Clean Std | Nominal Std | Degraded Std |
| --- | ---: | ---: | ---: |
| `main/phnode_full` | 1.938 | 1.512 | 1.168 |
| `baseline/phnode_merged_force` | 0.507 | 0.551 | 0.485 |
| `baseline/se3_momentum_blackbox` | 0.409 | 0.342 | 0.216 |
| `baseline/phnode_qforce` | 2.166 | 2.103 | 1.007 |
| `baseline/se3_accel_blackbox` | 2.792 | 2.542 | 1.759 |

Interpretation:
- `main/phnode_full` is not the most seed-stable model.
- `merged_force` and especially `momentum_blackbox` are more consistent across seeds.
- `qforce` and `se3_accel_blackbox` have clear seed sensitivity.

## Failure Pattern Notes

- `baseline/blackbox_fullstate` is effectively unusable in this setting.
- One seed collapses completely at `H=60s` with `completion=0.000` and `pred_div=1.000`; the other seeds are also poor.
- For the other five models, completion and divergence rates change very little across `clean / nominal_eval / degraded_eval`.
- That means the current noise range mainly affects long-horizon error growth, not rollout survivability.

## Conclusions

1. `main/phnode_full` is not the best model if the only criterion is `H=60s pos median` under `clean`.
2. `main/phnode_full` is the strongest model in tail robustness: it is best in `pos p95` under `clean`, `nominal_eval`, and `degraded_eval`.
3. `baseline/phnode_merged_force` is the strongest competitor on central accuracy and wins `clean` and `degraded_eval` `pos median`.
4. `baseline/se3_momentum_blackbox` is the most stable competitor across seeds and remains very competitive under noise.
5. If your paper emphasis is “robust long-horizon deployment behavior under noisy filtered initial state”, `main/phnode_full` has a defensible advantage, but the argument should be built around tail-risk control and `OU` robustness, not around clean median error alone.
