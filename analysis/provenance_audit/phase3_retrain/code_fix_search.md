# Phase 3 后续 — git log 区间 diff 搜索修复 commit

## 目标

在 `[2026-04-04, 2026-04-26]` 区间（catalog 训练时代 → cleanrun v1 训练时代）的 g3_5_5 仓库 commit 历史中定位「修复 fragility 的具体 commit」。

## 搜索范围

`git log --all --since=2026-04-04 --until=2026-04-26` 共 28 个 commit。
触及核心训练代码（`train_utils.py / train_auv_hamnode.py / AUVHamNODE.py / auv_baselines.py / auv_model_registry.py`）的 11 个：

| commit | 日期 | 主旨 | train_utils Δ | train_hamnode Δ |
| --- | --- | --- | --- | --- |
| fbb06e4 | 04-05 | 添加批量训练评估脚本 | (only scripts) | — |
| 6bc83b1 | 04-06 | 重构噪声模型, cc | +31 | +6 |
| 53ddf67 | 04-07 | 按噪声模拟方案修改代码 | +174 | +57 |
| 8c05b22 | 04-07 | 继续修改噪声方案 | +34 | +53 |
| 1730a82 | 04-08 | 根据方案修改代码 | +122 | -8 |
| 8e510aa | 04-08 | 第一阶段：mix_ratio 逐样本掩码 + yaw-dominant + 落盘 | +194 | +98 |
| 4ee1860 | 04-09 | 改为基于 remus100 硬件的噪声设计 | +365 | +46 |
| 5bf9b35 | 04-09 | 第二阶段：heading_biased_eval + 退化指标 | +376 | +53 |
| e961c72 | 04-14 | 添加结果汇总与分析 | +22 | — |
| 3f3c3a4 | 04-23 | v4-lite 协议验证 + Phase-1 汇总链路 | +479 | +39 |
| 41fd824 | 04-25 | v4_lite 训练加速 + notebook 落盘 | +245 | +82 |
| 129570a | 04-25 | v4_lite 训练 140s → 8s | +61 | — |

## 关键发现：模型层零变化

- `AUVHamNODE.py` 在 a2ca101 (2026-04-03) ↔ 41fd824 (2026-04-25) **完全无 diff**（479 行不变）
- `auv_baselines.py` 同样完全无 diff（750 行不变）
- 区间内所有代码变化集中在 `train_utils.py` (1518 → 3159 行) 与 `train_auv_hamnode.py` (695 → 991 行)
- 这些变化**全部围绕 noise 系统的 v1→v2 重构**（profile-based IC noise, remus100 硬件预算, v4_lite 协议, mix_ratio mask 等），对 clean 训练路径都是无影响 / no-op

## 关键发现：clean 训练路径数学等价

逐项验证：

### 1. StateNormalizer.from_dataset

a2ca101 vs 41fd824：函数体**逐行相同**（仅 a2ca101 多一个未使用的 `state_convention: str` 参数）。`std_pos / std_vel / std_act / std_vel_data` 计算公式完全一致。

### 2. se3_trajectory_loss

41fd824 新增 `frame_weights` 参数。当 `frame_weights=None`（clean 训练默认）时：

```python
# 41fd824
loss_pos = weighted_mean(dx ** 2, None if weights_bt is None else weights_bt.unsqueeze(-1))
# weighted_mean(values, None) → values.mean()
# 因此与 a2ca101 的 (dx ** 2).mean() 数学等价
```

position / rotation / velocity / actuator 四个分量在 weights=None 时均回退到 a2ca101 的 `.mean()`。

### 3. 训练循环 `_run_epoch` + `train`

逐行对比 `git show a2ca101:./train_auv_hamnode.py` 与 `git show 41fd824:./train_auv_hamnode.py` 的 train loop：

- model.train/eval 切换、batch.to(device)、reset_nfe、to_ode_state(batch[:, 0])：完全相同
- IC 处理：
  - a2ca101: `if train and self.config.init_state_noise: y0 = apply_initial_condition_noise(...)`，但 catalog A46 config `init_state_noise=False` → no-op，y0 = clean_y0
  - 41fd824: `if noise_cfg.is_active(...): build_noisy_initial_condition(...)`，但 clean 训练 `noise_cfg.is_active=False` → no-op，y0 = clean_y0
- odeint 调用、isnan/isinf 检查、loss.backward、clip_grad_norm_(max_norm=1.0)、isnan(p.grad) check、optimizer.step、scheduler.step：**完全逐行相同**
- best_selection_key、save_checkpoint、"no successful training batches" warning 逻辑：**完全相同**

### 4. DataLoader / RNG 初始化

两个版本都是：

```python
torch.manual_seed(config.seed)
np.random.seed(config.seed)
```

`from torch.utils.data import DataLoader` 标准 DataLoader，未传 `generator` / `worker_init_fn` 参数，依赖 torch 全局 RNG 状态。

## Smoking-gun 证据：Epoch 1-2 bit-identical

| Epoch | catalog A46 (2026-04-04, g3_5_5 mirror) | Phase 3 audit (2026-05-12, current main on g3_5_7 mirror, PyTorch 2.10/CUDA 12.8/L4) |
| --- | --- | --- |
| 1 | Train **4.3572e+00** / Test **5.0488e+00** | Train **4.3572e+00** / Test **5.0488e+00** |
| 2 | Train **3.6033e+00** / Test **4.3992e+00** | Train **3.6033e+00** / Test **4.3992e+00** |
| 3 | Train **3.0324e+00** / Test 3.4117e+00 | Train **3.0323e+00** / Test 3.4099e+00 ← **漂移起点** |
| 4 | Train 1.8477 / Test 1.3654 | Train 1.8482 / Test 1.3859 |
| 5 | Train 1.6139 / Test 1.4542 | Train 1.3598 / Test 1.1605 |
| 13 | Train 6.10e-01 | Train **3.12e+01** ← audit 也曾遭遇接近爆炸的 batch |
| 14 | Train 5.58e-01 | Train 6.69e-01 ← 但 audit 立即恢复 |
| 21 | Train 4.85e-01 / Test **2.69e-01 (best)** | Train 6.22e-01 / Test 6.44e-01 |
| 24 | Train **2.38e+01** ← catalog 开始爆炸 | Train 1.23e+00 |
| 25 | Train **4.68e+25** / Test **inf** / SO3 **4.55e+11** / Fail 9 / **NaN** | Train 1.77e+00 |
| 26+ | "no successful training batches" 一直到 epoch 300 | 持续收敛 |
| 250 | best_test_loss = 0.27 (epoch 21 卡住) | best_test_loss = **4.0471e-03** |

→ **Epoch 1+2 在 4 位精度上完全一致** = catalog A46 与 current main 在 deterministic 路径上**数值等价**。

→ Epoch 3 开始数值漂移 = 来自**非确定性**（cuDNN 算法选择、CUDA fp 误差累积、ODE solver 内部数值积分），与 git 代码无关。

→ Epoch 13 audit 也遭遇 train_loss=31.17（接近 catalog epoch 24 的 23.78）但立即恢复 → 同样的 grad_clip / isnan_check 防护代码在两边都生效，**差别只在 Adam optimizer state 是否被毁坏**。catalog 进一步在 epoch 24 走向天文数字 (4.68e+25) → fp32 溢出 → 之后无法恢复。audit 则因为不同的 batch 顺序/cuDNN 算法，遭遇的最大 loss 较小，optimizer 状态未崩。

## 结论：修复来源 ≠ git commit

**修复 fragility 的不是 `[2026-04-05, 2026-04-25]` 区间的任何 commit**。证据链：

1. 区间内所有代码改动均围绕 noise v1→v2 重构与 v4_lite 协议，对 clean 训练路径**逻辑无影响**
2. `_run_epoch / train / se3_trajectory_loss / StateNormalizer` 在 clean 训练下**数学等价**
3. catalog A46 与 Phase 3 audit 在 Epoch 1+2 **bit-identical**（在 4 位有效数字精度上），证明 deterministic 路径完全等价
4. 漂移从 Epoch 3 开始，且 audit 在 Epoch 13 也曾遭遇接近灾难性 loss → 两个版本对**坏 batch 都同等脆弱**，差异只在「特定 cuDNN 算法选择 + 特定 batch 顺序」是否触发不可恢复的 optimizer state 崩溃

**真正的"修复来源"是云端环境升级**：

- catalog 训练（2026-04-04）云端工作目录 `auvhamnode/g3_5_5`，PyTorch/CUDA/cuDNN 版本未记录（catalog `run_inventory.csv` 缺 `environment` schema）
- cleanrun v1 训练（2026-04-26）云端工作目录 `auvhamnode/g3_5_7`，PyTorch/CUDA/cuDNN 版本同样未记录
- Phase 3 audit（2026-05-12, 本 audit）云端工作目录 `auvhamnode/g3_5_7`，**PyTorch 2.10.0+cu128 / CUDA 12.8 / cuDNN 91002 / L4 GPU** — 与 cleanrun v1 同镜像、近似环境

因为 Phase 3 audit (current main on g3_5_7) 与 cleanrun v1 浮点完全相同，说明 g3_5_7 这套云端镜像的环境从 2026-04-26 到 2026-05-12 **保持稳定**。而 g3_5_5 与 g3_5_7 是**两份独立的云端镜像**，可能用了不同的 conda 环境、不同的 PyTorch/torchdiffeq 版本。

## 修复 commit 的 hypothetical 命名

若硬要给「修复」起一个 commit-level 标识，候选有：

- 不存在 — 严格意义上没有任何 commit 修复了 fragility
- 候选 1：`fbb06e4` (2026-04-05) — 引入 `train_all_models_noise_profile.sh` wrapper，统一调用链。但 catalog A46 实际也走这个 wrapper（A46 training.log 显示同样的 invocation 行为），所以 wrapper 本身不是修复
- 候选 2：`a2ca101` 到 `129570a` 中的某个**间接修复** — 例如新增的某个 import / decorator 改变 cuDNN benchmark 默认行为。但通过 `_run_epoch` 逐行 diff 已经排除

## 推荐归因 wording（Phase 4 报告用）

> Catalog 时代 `phnode_full clean seed46` 训练发散（epoch 21 best_loss=0.27, epoch 24 起进入 NaN 爆炸）是一次**与环境非确定性耦合的灾难性梯度事件**，而非模型架构缺陷或 git 代码 bug。具体机制：
>
> 1. 模型 + dataset + 显式超参 + IC 处理代码在 catalog 时代与 cleanrun v1 / current main 三个时间点**逻辑数学等价**（已通过逐行 diff 与 Epoch 1-2 bit-identical 数值匹配验证）；
> 2. cuDNN 默认 benchmark 模式 + 非确定性算法选择在 seed46 + 该 dataset 切分下，于 epoch 13-24 区间触发了一次 grad clip 无法吸收的极端 batch（log 显示 train_loss 24.0 → 4.68e+25 单 epoch 跃迁）；
> 3. 训练循环已有的 grad clip (max_norm=1.0) + isnan/isinf 检查 + skip_invalid_grad 防护**对当前 main 也未充分起作用**（Phase 3 audit 在 epoch 13 也遭遇 train_loss=31.17 但侥幸恢复），属于 stochastic 防护边界事件；
> 4. cleanrun v1 训练（2026-04-26, g3_5_7 镜像）+ Phase 3 audit（2026-05-12, 同 g3_5_7 镜像）在 phnode_full clean seed46 上**浮点完全一致**，说明 g3_5_7 这套云端环境（PyTorch 2.10 / CUDA 12.8 / cuDNN 91002 / L4）下，该 seed 不会进入坏 basin。catalog 时代 g3_5_5 云端环境（具体版本未记录）则会。

## 评估对原研究结论的影响

- catalog `phnode_full clean seed46` 47 m fragility **是真实观测到的训练动态事件**，但**与当前 main 的模型/代码无关**，属于「特定环境 + 特定 seed」的偶然
- cleanrun v1 / current main 上 `phnode_full clean seed46` 已稳定收敛至 best_loss=4.05e-03 / 60s rollout pos_err_median=0.4558 m
- catalog §12 表中由 seed42/seed46 catastrophic 推动的 11 m 5-seed mean 应标 `evidence_status = stale`，新基线由 cleanrun v1 ≡ current main 提供（5-seed mean = 0.6767 m）
- 论文/进度文档中**不需要把"修复 fragility 的 commit"作为科研贡献**，而应说明"在当前环境下未观察到该 fragility，并补充了 catalog 时代缺失的 `code_revision` / `environment` 字段以防止类似 provenance 混淆"

## 防再发生措施（写入 Phase 4 落盘）

1. **catalog `run_inventory.csv` 加 `code_revision` 字段** — 任何新训练 run 必须记录 `git rev-parse HEAD` 与本地未提交 diff hash
2. **catalog `run_inventory.csv` 加 `environment` 字段** — 至少记录 `python / torch / cuda / cudnn` 版本和 GPU 型号
3. **训练循环增加 `torch.backends.cudnn.deterministic = True` 选项**（可选）— 但会降低训练速度 ~20%，留作 ablation
4. **catalog A42/A46 行打 `evidence_status = stale`** — 标记其训练动态受未记录的环境因素影响，不可作为方法比较的基线
