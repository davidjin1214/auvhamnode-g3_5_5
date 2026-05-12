# Phase 2 — Aggregation 口径对齐

## Phase 2.1 — 两侧 5-seed × 60s × pos_err 数据

来源：
- catalog: `analysis/oc_data_catalog/canonical_rollout_summary_long.csv`
- cleanrun v1: `checkpoints/sweep_oc_phase1a_decision_proxy_phase1a_oc_v4lite_cleanrun_v1/phase1a_by_seed.csv`

### catalog phnode_full clean_train + clean eval @ 60s

每个 seed 在 catalog 里有 4 个 rollout_run（不同时间、不同 source_file 重跑），下表给出每 seed 的 4-rollout 平均：

| seed | per-rollout pos_err_**mean** (4 rollouts) | per-seed mean | per-rollout pos_err_**median** (4 rollouts) | per-seed mean |
| --- | --- | --- | --- | --- |
| 42 | 2.68 / 5.18 / 5.27 / 7.68 | 5.20 | 2.50 / 4.21 / 4.97 / 6.26 | 4.49 |
| 43 | 0.46 / 0.55 / 0.71 / 1.14 | 0.71 | 0.31 / 0.49 / 0.60 / 0.93 | 0.58 |
| 44 | 0.51 / 0.58 / 0.60 / 0.63 | 0.58 | 0.42 / 0.45 / 0.46 / 0.47 | 0.45 |
| 45 | 0.70 / 0.95 / 1.02 / 1.43 | 1.02 | 0.58 / 0.78 / 0.84 / 0.99 | 0.80 |
| 46 | 41.14 / 47.69 / 49.22 / 52.70 | **47.69** | 42.18 / 47.86 / 48.42 / 49.09 | **46.89** |

5-seed mean (mean stat):   (5.20 + 0.71 + 0.58 + 1.02 + 47.69) / 5 = **11.04 m**
5-seed mean (median stat): (4.49 + 0.58 + 0.45 + 0.80 + 46.89) / 5 = **10.64 m**

### cleanrun v1 phnode_full clean_train + clean eval @ 60s

每个 seed 在 cleanrun v1 里只有 1 个 rollout：

| seed | best_loss | pos_err_median (60s, clean eval) | pos_err_p95 (60s, clean eval) |
| --- | --- | --- | --- |
| 42 | 4.02e-03 | 0.6110 | 1.554 |
| 43 | 4.15e-03 | 0.6793 | 1.384 |
| 44 | 4.21e-03 | 1.2059 | 3.067 |
| 45 | 4.02e-03 | 0.4316 | 1.353 |
| 46 | 4.05e-03 | **0.4558** | 1.514 |

5-seed mean (median stat) = (0.611 + 0.679 + 1.206 + 0.432 + 0.456) / 5 = **0.6767 m**

→ 完美匹配报告 §11 给的 `clean_value=0.6767`。

注意：cleanrun v1 by_seed 表**没有 mean 列**，只有 median / p95。`phase1a_summary.csv` 在 60s clean 这一行的 `rollout_final_position_error_median_mean = 0.6767`。

## Phase 2.2 — 报告 §12 0.9604 m 的真实口径

`phase1a_summary.csv` 查询：

| eval_profile | horizon_s | rollout_final_position_error_median_mean | rollout_final_position_error_p95_mean |
| --- | --- | --- | --- |
| clean | 60.0 | 0.6767 | (省略) |
| nominal_eval | 60.0 | **0.9604** | (省略) |
| nominal_eval | 60.0 | 0.8438 | (省略) |

→ 报告 §12 的 `0.9604 m` 实际是 cleanrun v1 phnode_full clean_train + **nominal_eval** + 60s + 5-seed mean of pos_err_median；**不是 clean+clean**。

报告 §12 的 catalog `10.9557 m` 是 catalog phnode_full clean_train + clean eval + 60s + 5-seed mean of `final_position_error.mean`。

**§12 表里的 11 m vs 0.96 m 包含三层口径不一致**：
1. catalog 用 clean eval，cleanrun v1 用 nominal_eval
2. catalog 用 mean stat，cleanrun v1 用 median stat
3. catalog 一个 seed 多个 rollout 取均值，cleanrun v1 一个 seed 一个 rollout

## Phase 2.3 — 同口径对齐 (clean+clean, 60s, 5-seed mean of pos_err_median)

| 来源 | seed42 | seed43 | seed44 | seed45 | seed46 | 5-seed mean | 相对 cleanrun v1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| catalog (4-rollout per-seed mean of median) | 4.49 | 0.58 | 0.45 | 0.80 | 46.89 | **10.64** | **15.7×** |
| cleanrun v1 (single rollout median) | 0.61 | 0.68 | 1.21 | 0.43 | 0.46 | **0.6767** | 1× |
| per-seed ratio (catalog / cleanrun v1) | **7.3×** | 0.86× | 0.37× | 1.85× | **103×** | 15.7× | — |

→ seed44 在 catalog 上 0.45，**比 cleanrun v1 还低**（cleanrun v1 = 1.21）。
→ seed46 是 gap 的绝对主因 (103×)，seed42 次之 (7.3×)。
→ seed43/44/45 在 catalog 上反而表现相近或更好 — 那些 seed 训练正常收敛。

## Phase 2 结论

**聚合口径只能解释 §12 表面 11 m / 0.96 m 比值里很小的一部分** ——`mean vs median` 和 `clean vs nominal_eval` 切换合起来把 cleanrun v1 一侧从 0.6767 → 0.9604（增量 ≈ 0.28 m）。**对齐后 catalog vs cleanrun v1 仍然有 ~16× 的真实 gap**，且这个 gap **完全由 catalog 时代 seed46（103×）+ seed42（7.3×）的训练异常驱动**，与 Phase 1 的训练 log 证据完全一致：

- catalog seed46 训练发散 (epoch 26+ ODE solver 全 fail, best_loss=0.27) → 60s rollout 47 m
- catalog seed42 训练收敛但 best_loss=0.02 (vs cleanrun v1 0.004) → 60s rollout 4-5 m
- cleanrun v1 全部 5 seeds 收敛到 best_loss ≈ 0.004 → 60s rollout < 1.5 m，无灾难

## 唯一未解释的关键问题

**为什么 cleanrun v1 seed46 训练收敛，catalog seed46 训练发散？**

两侧 dataset / 显式超参 / wrapper / noise profile / seed 数字本身完全一致，差异**必由非超参代码或环境差异引起**。这就是 Phase 3 Setup A 要回答的问题。

由于 fragility 在 cleanrun v1 一侧已经确凿消失（直接证据：seed46 60s pos_err_median = 0.4558 m，best_loss = 0.004），Phase 3 缩小为：

→ **在当前 main 重跑 seed46 clean 一个 < 1 min 的 run，观察训练是否仍在 epoch 26 附近发散。** 这一个 run 就能立即判定 fragility 在当前 main 上是否仍可复现，进而决定是否需要 git bisect。
