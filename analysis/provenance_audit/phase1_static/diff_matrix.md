# Phase 1.3 — Static Diff Matrix

对比 catalog A42/A46 与 cleanrun v1 C42–C46（同为 `phnode_full clean_train clean`）。

每一行注明：catalog 来源、cleanrun v1 来源、是否一致、对 gap 的解释力。

## 1. Dataset

| Field | catalog A42 / A46 | cleanrun v1 C42–C46 | match | 解释力 |
| --- | --- | --- | --- | --- |
| dataset filename | `auv_oc_traj1000_blk150_s23_d0be9434.pkl` | `auv_oc_traj1000_blk150_s23_d0be9434.pkl` | ✅ | n/a |
| dataset_id | `d0be9434` | `d0be9434` | ✅ | n/a |
| dataset generation seed | 23 | 23（同一文件） | ✅ | n/a |
| num_trajectories | 1000 | 1000 | ✅ | n/a |
| blocks_per_trajectory | 150 | 150 | ✅ | n/a |
| train/test split ratio | 0.8 | 0.8 | ✅ | n/a |
| train_traj count (训练时切分) | 559 (A46 log) / 559 (A42 log) | 同 wrapper 默认 0.8 split | ✅ | n/a |
| dataset_velocity_convention | `body_total` | `body_total` | ✅ | n/a |
| dataset_state_dim | 27 | 27 (state_fingerprint 不同但 layout 相同) | ✅ layout | n/a |
| ocean_current | True | True | ✅ | n/a |

→ **Dataset 完全一致**，不是 gap 来源。

## 2. Training hyperparameters

源：A42 config.json (catalog) / 当前 main `train_utils.py` `DATASET_TRAINING_DEFAULTS["oc"]` (cleanrun v1 wrapper 链路用的也是默认值)。

| Field | catalog A42 | 当前 main oc 默认 (= cleanrun v1) | match | 解释力 |
| --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | ✅ | n/a |
| num_epochs | 300 | 300 | ✅ | n/a |
| learning_rate | 0.006 | 0.006 | ✅ | n/a |
| min_learning_rate | 0.0001 | 0.0001 | ✅ | n/a |
| warmup_steps | 400 | 400 | ✅ | n/a |
| total_steps | 5000 | 5000 | ✅ | n/a |
| weight_decay | 0.0001 | (未变 — train_utils.py @dataclass ConfigSchema) | ✅ 默认 | n/a |
| ode_solver | `rk4` | `rk4` | ✅ | n/a |
| hidden_dim | 128 | 128 | ✅ | n/a |
| so3_regularization_weight | 0.001 | 0.001 | ✅ | n/a |
| actuator_loss_weight | 0.2 | 0.2 | ✅ | n/a |
| mass_init | `remus` | `remus` | ✅ | n/a |
| t_actuator_init | [0.1, 0.1, 1.0] | [0.1, 0.1, 1.0] | ✅ | n/a |
| u_act_scale | [1.0, 1.0, 0.001] | [1.0, 1.0, 0.001] | ✅ | n/a |
| dj_current_feature | `current_body` | `current_body` | ✅ | n/a |
| actuation_current_feature | `current_body` | `current_body` | ✅ | n/a |

→ **训练超参完全一致**，不是 gap 来源。

## 3. Noise configuration

| Field | catalog | cleanrun v1 | match | 解释力 |
| --- | --- | --- | --- | --- |
| noise_profile_train | `clean` | `clean` | ✅ | n/a |
| noise_protocol_train | `clean` | `clean` | ✅ | n/a |
| noise_warmup_epochs | 20 (audit) | 20 | ✅ | n/a |
| noise_ramp_epochs | 80 (audit) | 80 | ✅ | n/a |
| noise_mix_ratio | 0.5 (audit) | 0.5 | ✅ | n/a |
| noise reference | remus100_dr | remus100_dr | ✅ | n/a |

注：clean 训练下 `is_active=False`，`epoch_scale(epoch)=0` 恒成立，所以 warmup/ramp/mix 仅作为静态记录。

→ **Noise 配置完全一致**，且对 clean 训练无生效。不是 gap 来源。

## 4. Training outcome（决定性差异）

| Field | catalog A42 (seed42) | catalog A46 (seed46) | cleanrun v1 C42 | cleanrun v1 C46 | 解释力 |
| --- | --- | --- | --- | --- | --- |
| best_epoch | 250 | **21** | 249 | 250 | C |
| best_loss (test) | 2.098e-02 | **2.688e-01** | 4.021e-03 | 4.047e-03 | C |
| 训练 warnings | 0 | **275 行 solver=0 fail (epoch 26→end)** | 0 (assumed) | 0 (audit 无 flag) | C |
| 60s clean rollout mean | 5.20 m | **47.69 m** | < 1 m (5-seed mean = 0.96 m) | < 1 m (同上) | C |

C = critical — 这是 gap 唯一确凿不同。

**A46 catastrophic 的物理机制：从 epoch 26 起到训练结束，每个 epoch 全部 20 个 batch 的 ODE solver 都返回失败 (solver=0/20, pred=20/20, grad=0/20)。**这意味着模型权重在 epoch 21-25 间被毁坏，之后 forward 永远产生 NaN/Inf，optimizer 无法更新。这是经典的训练发散，不是模型表达能力问题。

**A42 次优收敛：**训练全程无 warning，但 best_loss=0.02 比 cleanrun v1 的 0.004 高 5×。即模型收敛到了一个比 cleanrun v1 更差的局部 basin。

## 5. Invocation 链路

| 维度 | catalog 时代 | cleanrun v1 | match |
| --- | --- | --- | --- |
| 训练 wrapper | `train_all_models_noise_profile.sh` (推测，与 cleanrun 同链) | `train_all_models_noise_profile.sh` | ✅ |
| 是否传 --num_epochs / --lr / --batch_size | 否（依赖 dataset_defaults） | 否（依赖 dataset_defaults） | ✅ |
| dataset_kind 推导 | 来自文件名 `auv_oc_*` → `oc` | 来自文件名 `auv_oc_*` → `oc` | ✅ |

→ **wrapper 与 invocation 完全一致**。

## 6. git commit at training time

| 维度 | catalog | cleanrun v1 |
| --- | --- | --- |
| training timestamp | 2026-04-04 | 2026-04 / 2026-05（具体 ipynb 不带时间戳） |
| code_revision / git_hash | **catalog 不记录此字段** | 也未直接落盘，需要从云端 g3_5_7 仓库历史推断 |
| run_dir 来源仓库 | `g3_5_5` / `g3_5_5n2cx_v3` | **`g3_5_7`** |

→ catalog `run_inventory.csv` 没有 `code_revision`，无法精确对齐 commit。这是 Phase 4 文档落盘必须补的一条 evidence schema 缺陷。

## 7. 结论（Phase 1 静态对齐）

可以**排除**的 gap 来源：
- dataset（含 generation 参数与切分）
- 显式训练超参（batch_size / epochs / lr / warmup / total_steps / hidden_dim / ode_solver / so3_reg / actuator_loss / mass_init / t_actuator_init / u_act_scale / dj_current_feature）
- noise profile / protocol / 调度（clean 训练下全部 no-op）
- wrapper / sh 调用链与参数传递

**剩余 gap 来源候选**（按可能性排序）：
1. **训练稳定性代码**（NaN guard、optimizer state、ODE solver fallback、autocast/AMP 配置等非超参代码）在 catalog 与 cleanrun v1 之间发生变化 → 解释 A46 训练发散
2. **DataLoader / shuffle seed 语义** 变化 → 解释 A42 与 cleanrun v1 收敛 basin 不同（同 seed 但 batch 顺序变了）
3. **环境差异**：cleanrun v1 训练在云端 `g3_5_7` 仓库，PyTorch 版本/CUDA 版本/numpy 版本与 catalog 时代不同 → 浮点 / 非确定性
4. **catalog `run_inventory` 缺 `code_revision`** 让以上 3 个 hypotheses 无法直接 git diff 验证

## 8. 直接的下一步建议

→ Phase 2 **聚合口径对齐**（不消耗算力）必须先做：cleanrun v1 报告 §12 给的 5-seed mean = 0.9604 m 是 `mean` 还是 `median`？是否做了离群 seed 剔除？因为 cleanrun v1 audit 显示 seed46 best_loss=0.004 → rollout 必然小，但如果 seed42 在 cleanrun v1 时代仍然 ~5 m，5-seed mean 也不该 < 1 m。需要从 cleanrun proxy 的 `phase1a_summary.csv` / `sweep_seed_metrics.csv` 直接取每 seed 的 60s rollout 值，去验证。

→ 同时建议 Phase 3 Setup A 缩小为**只重跑 seed46 clean** 一个 run（< 1 min），观察当前 main 是否仍发散到 epoch 26。这一个 run 就能立刻判定 fragility 的可重复性。
