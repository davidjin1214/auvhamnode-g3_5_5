# Phase 1.1 — Canonical Run Lock

锁定 catalog 中 `phnode_full clean seed42/46` 的真实 run（A42、A46），并锁定 cleanrun v1 对应训练 run 的来源。

## A42 — catalog seed42 clean_train

| Field | Value |
| --- | --- |
| suite_family | `sweep_oc_all` |
| suite_name | `sweep_oc_core_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_115414` |
| run_uid | `sweep_oc_core_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_115414/main_phnode_full_seed42` |
| 磁盘路径 | `checkpoints/sweep_oc_all/sweep_oc_core_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_115414/main_phnode_full_seed42` |
| dataset_id | `d0be9434` |
| dataset_path | `data/auv_oc_traj1000_blk150_s23_d0be9434.pkl` |
| train_type | `clean_train` |
| noise_profile_train | `clean` |
| noise_protocol_train | `clean` |
| num_epochs (config) | 300 |
| best_epoch (training.log) | 250 |
| best_loss (training.log, test_loss) | 2.098e-02 |
| training warnings | 0 (无 solver 失败、无 NaN) |
| training 完成 | 是 |
| status | `ok` |

## A46 — catalog seed46 clean_train

| Field | Value |
| --- | --- |
| suite_family | `sweep_oc_all` |
| suite_name | `sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47` |
| run_uid | `sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47/main_phnode_full_seed46` |
| 磁盘路径 | `checkpoints/sweep_oc_all/sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47/main_phnode_full_seed46` |
| dataset_id | `d0be9434` (同 A42) |
| dataset_path | `data/auv_oc_traj1000_blk150_s23_d0be9434.pkl` (同 A42) |
| train_type | `clean_train` |
| noise_profile_train | `clean` |
| noise_protocol_train | `clean` |
| num_epochs (config) | 300 |
| best_epoch (training.log) | **21** |
| best_loss (training.log, test_loss) | **2.6881e-01** |
| training warnings | **275 行 "no successful training batches"，自 epoch 26 起每个 epoch 全部 batch ODE 求解失败** |
| training 完成 | 形式上完成 300 epoch，但 epoch 26 起所有训练 batch 全部失败 (solver=0/20, pred=20/20, grad=0/20) |
| status | `ok` (catalog 没有捕获训练发散) |

## C42–C46 — cleanrun v1 phnode_full clean (5 seeds)

来源：`checkpoints/sweep_oc_phase1a_decision_proxy_phase1a_oc_v4lite_cleanrun_v1/phase1a_train_audit.csv`

实际训练产物（云端 `g3_5_7` 仓库，本地未保留 ckpt 原文件）：
- seeds 42/44/46 在 `sweep_oc_phase1a_smoke3_clean_phase1a_oc_v4lite_cleanrun_v1/main_phnode_full_seed{42,44,46}`
- seeds 43/45 在 `sweep_oc_phase1a_decision_extra43-45_clean_phase1a_oc_v4lite_cleanrun_v1/main_phnode_full_seed{43,45}`

| seed | best_epoch | best_loss | warmup | ramp | mix_ratio | source phase |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 249 | 4.021e-03 | 20 | 80 | 0.5 | smoke3 |
| 43 | 247 | 4.155e-03 | 20 | 80 | 0.5 | extra43-45 |
| 44 | 242 | 4.207e-03 | 20 | 80 | 0.5 | smoke3 |
| 45 | 247 | 4.025e-03 | 20 | 80 | 0.5 | extra43-45 |
| 46 | 250 | 4.047e-03 | 20 | 80 | 0.5 | smoke3 |

## 与 catalog 直接对照（phnode_full clean）

| seed | catalog best_epoch | catalog best_loss | cleanrun v1 best_epoch | cleanrun v1 best_loss | best_loss ratio (catalog / v1) |
| --- | --- | --- | --- | --- | --- |
| 42 | 250 | 2.098e-02 | 249 | 4.021e-03 | **5.2×** |
| 46 | **21** | **2.688e-01** | 250 | 4.047e-03 | **66×** |

两个 seed 都收敛到比 cleanrun v1 显著更差的 loss；其中 seed46 是训练发散，seed42 是收敛到次优 basin。

## Phase 1.1 数值验证

催化 §12 表格的 catalog `phnode_full clean_train clean` 5-seed 60s mean = 10.9557 m 由以下完整产生：

```
seed | rollout per-seed mean (clean profile, 60s) | 备注
42  | 5.20  | catalog rollout (4 个 rollout_run × 同 run，取平均)
43  | 0.71  | seed43 是 ablate_no_lift 异常 seed，本表只看 phnode_full
44  | 0.58  |
45  | 1.02  |
46  | 47.69 | catastrophic
mean = (5.20 + 0.71 + 0.58 + 1.02 + 47.69)/5 = 11.04 m  ≈ §12 给的 10.96 m
```

11 m 这个数字**完全**由 seed46 ~48 m 灾难驱动。seed42 偏高（5.2 m）也是次因。其它 seed 全部 < 1.5 m。

而 cleanrun v1 报告 §12 给的 `phnode_full clean_train clean` 5-seed mean = 0.9604 m → 必然 seed46 的 ~48 m 灾难已经消失。

## 静态对齐三大不变量（A42/A46 vs C42–C46 完全相同）

- dataset: `data/auv_oc_traj1000_blk150_s23_d0be9434.pkl` — 同一 dataset_id `d0be9434`
- train_type 输入: `clean_train` / `noise_profile=clean` / `noise_protocol=clean`
- num_epochs: 300

所以 Phase 1 阶段可以**排除** dataset / noise profile / epochs 数作为 gap 的解释。
