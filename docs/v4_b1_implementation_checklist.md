# v4-B1 实施清单

本文档给出 `v4-B1` 的最小可实现方案与分阶段实施清单。

`v4-B1` 的目标不是直接做长时 multi-block rollout，而是先回答一个更干净的问题：

```text
给定一段 noisy navigation history，
能否恢复当前 block 的 clean 初始状态，
并据此预测当前 block 的 clean trajectory？
```

这一定义对应：

```text
history-aware clean block prediction
```

它是 `v4-B` 的首个可执行版本，也是后续 `v4-B2` 的基础。

---

## 1. 范围冻结

第一版请明确只做以下范围：

1. 只做 `v4-B1`
2. 只预测**当前 block** 的 clean trajectory
3. 输入使用 `history blocks + current block initial observation`
4. target 始终保持 clean truth
5. 主干动力学模型先只支持 `phnode_full`
6. 保持现有 `train_auv_hamnode.py` 不被破坏

第一版**不要**做以下事情：

1. 不做 multi-block rollout
2. 不做 posterior-to-posterior 监督
3. 不把 `v4-B1` 直接扩展到所有 baselines
4. 不把 `v4-B1` 直接塞进现有 model registry 的主路径
5. 不先碰数据生成器 `data_collection.py`

---

## 2. 成功标准

实现完成后，至少要满足以下五条：

1. 能直接读取现有数据集里的 `train_trajectories / test_trajectories`
2. 能生成 trajectory-level 一致的 noisy history observation
3. 能训练 `history encoder + state estimator + existing ODE backbone`
4. 能在 held-out trajectory 上输出 clean-target 指标
5. 不影响现有 `v3 / v4-A` 的训练与评估路径

---

## 3. 推荐架构

推荐采用：

```text
History encoder + current-state estimator + existing ODE rollout
```

而不是：

```text
全新一体化 dynamics model
```

原因：

1. 现有 `AUVHamNODE` 与 baselines 已经能完成 block rollout
2. `v4-B1` 的新增问题本质上是“history 是否能帮助恢复当前 clean state”
3. 把历史建模和动力学 rollout 解耦，最容易定位问题

---

## 4. 文件级改动建议

### 4.1 新增文件

建议新增以下文件：

1. `train_auv_history_model.py`
   新的训练入口；不要在第一版直接改造 `train_auv_hamnode.py`

2. `auv_history_models.py`
   存放 `BlockEncoder / HistoryEncoder / HistoryStateEstimator`

### 4.2 修改文件

建议修改以下文件：

1. `train_utils.py`
   增加 trajectory-aware dataset、history-aware noise synthesis、history-aware evaluation

2. `docs/noise_design_v4_dr_ekf_output.md`
   可补一个到本清单的链接；不是必须，但建议做

### 4.3 第一版不建议修改的文件

1. `AUVHamNODE.py`
2. `auv_baselines.py`
3. `data_collection.py`
4. `evaluate_rollout_benchmark.py`
5. `rollout_benchmark_engine.py`

原因：

- `v4-B1` 还不是新的 ODE 主干
- 数据集已经有 trajectory 结构
- benchmark 主线先不要被 history-aware 分支污染

---

## 5. 设计决策

### 5.1 新入口，不要硬改旧 trainer

第一版最重要的工程决策：

```text
新增 train_auv_history_model.py
而不是把 v4-B1 硬塞进 train_auv_hamnode.py
```

这样做的好处：

1. 避免把 block-only 训练器改得过于复杂
2. 保留现有 `v3 / v4-A` 训练路径不变
3. 更容易比较 `pure dynamics` 与 `history-aware` 两条线

### 5.2 第一版只支持 `phnode_full`

推荐第一版只支持：

```text
--backbone_model_type phnode_full
```

不要一开始就为所有 baseline 都接 history encoder。

原因：

1. `v4-B1` 的目标是先验证方法可行
2. 主干过多会显著放大调试成本
3. 如果第一版就泛化到所有模型，很难判断失败来自哪一层

等 `phnode_full` 跑通后，再决定是否扩到：

- `ablate_no_mass_prior`
- `ablate_no_lift`
- `phnode_qforce`

### 5.3 history 只做输入，不做 target

第一版必须坚持：

```text
input  = noisy history + noisy current initial observation
target = clean current block
```

绝不允许：

```text
target = noisy posterior block
```

---

## 6. 数据层实施清单

### 6.1 新增 trajectory getter

在 `train_utils.py` 中补齐：

1. `get_train_trajectories(dataset)`
2. `get_test_trajectories(dataset)` 已存在，可直接复用

推荐行为：

- 若数据集有 `train_trajectories`，直接返回
- 否则显式报错，不做 block-to-trajectory 的隐式猜测

### 6.2 新增 `AUVTrajectoryWindowDataset`

建议新增 dataset：

```python
class AUVTrajectoryWindowDataset(Dataset):
    ...
```

每个样本至少返回：

```text
history_blocks      [L_ctx, T_blk, state_dim]
current_block       [T_blk, state_dim]
traj_idx            scalar
block_idx           scalar
sample_id           scalar
```

其中：

- `history_blocks = trajectory[b-L_ctx : b]`
- `current_block  = trajectory[b]`
- `b >= L_ctx`

注意：

1. dataset 返回的必须是 **clean** 数据
2. noisy observation 不要在 `__getitem__()` 里生成
3. 这样每个 epoch 才能做可控的重采样

### 6.3 新增 dataloader builder

建议新增：

```python
create_trajectory_window_dataloaders_from_dataset(
    dataset,
    batch_size,
    history_blocks,
    ...
)
```

返回：

```text
train_loader, test_loader, t_eval, data_cfg
```

与现有 `create_dataloaders_from_dataset()` 对齐。

### 6.4 推荐默认历史长度

首版默认：

```text
history_blocks = 10
```

理由：

- 数据集 `dt_ctrl = 0.2 s`
- `10` blocks 对应 `2.0 s` 历史
- 足以覆盖 actuator lag、短时姿态/速度演化、慢噪声相关性

---

## 7. 噪声生成实施清单

### 7.1 不在 Dataset 中直接采样噪声

不要把 trajectory noise 写进 dataset 的 `__getitem__()`。

原因：

1. DataLoader worker 下随机性难控
2. 很难保证同一 trajectory 在同一 epoch 内共享 realization
3. eval 时也不方便复用

正确做法是：

```text
Dataset 只返回 clean history/current block
Trainer 在 epoch 级别构造 noisy observation cache
```

### 7.2 新增 trajectory observation cache

建议在 `train_utils.py` 中新增一类或一组函数：

```python
class TrajectoryObservationCache:
    ...
```

职责：

1. 对每条 trajectory 在每个 epoch 采样一条 OU realization
2. 生成整条 trajectory 的 noisy observation
3. 支持通过 `traj_idx` 取出任意 history window / current block

推荐接口：

```python
cache = build_noisy_trajectory_observation_cache(
    trajectories_clean,
    noise_cfg,
    base_seed,
    epoch,
    ...
)
```

### 7.3 为什么要做 cache

这是 `v4-B1` 的关键实现点。

因为 OU 是 trajectory-level 时间相关过程，若你按 sample 独立采样：

- 同一 trajectory 的不同 window 会不一致
- history 与 current 无法共享同一 realization
- mission-level 物理语义会被破坏

而 trajectory cache 恰好能解决这个问题。

### 7.4 cache 的推荐存储方式

建议：

- 用 CPU `torch.Tensor` 或 `numpy.ndarray` 缓存整条 trajectory 的 noisy observation
- 在 batch 组装时只切出所需窗口

对当前数据规模，这样做是可接受的：

- 约 `1000 trajectories * 150 blocks * 5 points * 27 dims`
- 内存量级约几十到一百多 MB

第一版完全可以接受。

### 7.5 cache 的 deterministic 规则

建议：

```text
base_seed_this_epoch = hash(global_seed, epoch)
```

每条 trajectory 的 realization 再由：

```text
hash(base_seed_this_epoch, traj_idx, stream_id)
```

派生。

这样能保证：

1. 同一 epoch 内，同一 trajectory 的 history/current 一致
2. 不同 epoch 会重采样
3. eval 可以用固定 seed 重复

---

## 8. 模型层实施清单

### 8.1 新增 `auv_history_models.py`

建议新建文件：

```text
auv_history_models.py
```

首版建议放三类模块：

1. `BlockEncoder`
2. `HistoryEncoder`
3. `HistoryStateEstimator`

### 8.2 `BlockEncoder`

作用：

```text
把一个 block 内的 [T_blk, state_dim] 局部轨迹编码成一个向量
```

推荐最简实现：

- flatten 后进 2-layer MLP

输入：

```text
[B, T_blk, state_dim]
```

输出：

```text
[B, d_block]
```

### 8.3 `HistoryEncoder`

作用：

```text
把 L_ctx 个 block embedding 聚合成 history summary
```

推荐首版只做：

```text
GRU
```

不要一开始就上 Transformer。

理由：

1. 参数更小
2. 调试更容易
3. 2 秒历史并不长

### 8.4 `HistoryStateEstimator`

推荐形式：

```text
y0_est = y0_noisy + Delta(history, y0_noisy)
```

而不是：

```text
y0_est = g(history)
```

原因：

1. residual correction 更稳定
2. 与“导航状态 refinement”语义一致
3. 更容易限制修正幅度

### 8.5 输入空间建议

这里要做一个清晰决定：

1. history encoder 的输入保留 **data convention**
2. current initial observation 先转成 **ODE state**
3. state estimator 输出的是 **ODE-state correction**

这样做的理由：

- history 本身就是“导航系统输出”
- backbone rollout 需要 ODE state
- `nu_r` 预算仍能在 ODE space 中控制

### 8.6 第一版不要改 backbone 结构

第一版中：

- `AUVHamNODE.py` 不改
- `auv_baselines.py` 不改

history-aware 部分只负责估计 `y0_est`，
然后继续调用现有 backbone 做：

```text
odeint(backbone, y0_est, t_eval)
```

---

## 9. Trainer 实施清单

### 9.1 新增训练入口 `train_auv_history_model.py`

建议新增新入口，而不是在旧 trainer 上分支。

建议新增类：

```python
class AUVHistoryTrainer:
    ...
```

### 9.2 batch 契约

DataLoader batch 建议是字典，至少包含：

```text
history_blocks
current_block
traj_idx
block_idx
sample_id
```

Trainer 在每个 batch 中完成：

1. 取 clean batch
2. 从 epoch cache 中切出 noisy history/current
3. 构造 `y0_noisy`
4. 用 history encoder 得到 `y0_est`
5. 用 backbone rollout 当前 block
6. 对 clean target 计算 loss

### 9.3 训练主路径

推荐伪代码：

```python
history_noisy, current_noisy = cache.slice(batch)
current_clean = batch["current_block"]

y0_noisy = backbone.to_ode_state(current_noisy[:, 0])
y0_clean = backbone.to_ode_state(current_clean[:, 0])

y0_est = estimator(history_noisy, current_noisy[:, 0], backbone)
pred = odeint(backbone, y0_est, t_eval, method=cfg.ode_solver).permute(1, 0, 2)

target_ode = backbone.to_ode_state(current_clean)
loss_main, comp = se3_trajectory_loss(target_ode, pred, normalizer, ...)
loss_est = mse(y0_est, y0_clean)
loss = loss_main + lambda_est * loss_est
```

### 9.4 是否冻结 backbone

第一版推荐支持两种模式：

1. `freeze_backbone_epochs = 0`
2. `freeze_backbone_epochs = 10`

更推荐先试第二种。

理由：

- 先让 history estimator 学会“状态修正”
- 再允许 backbone 联合适配
- 可减少一开始 joint training 的不稳定性

### 9.5 是否从 checkpoint warm start

建议支持：

```text
--backbone_checkpoint <path>
```

如果提供：

- 加载现有 `v3 / v4-A` backbone 权重
- 初始化 history estimator 为小残差修正

这通常比从零开始更稳。

---

## 10. Loss 与约束

### 10.1 主损失

主损失仍使用现有：

```python
se3_trajectory_loss(...)
```

target 必须是 clean block。

### 10.2 辅助状态估计损失

推荐新增轻量辅助项：

```text
loss_est = ||y0_est - y0_clean||^2
```

首版建议权重：

```text
lambda_est = 0.05
```

### 10.3 可选修正幅度正则

如果发现 estimator 修正过大，可新增：

```text
loss_delta = ||Delta(history, y0_noisy)||^2
```

但首版不是必须项。

### 10.4 不要做的事

1. 不要把 history block 本身做去噪重建作为主任务
2. 不要把 noisy target 混进主损失
3. 不要引入未来信息泄漏

---

## 11. 评估实施清单

### 11.1 新增 history-aware held-out eval

建议在 `train_utils.py` 中新增：

```python
evaluate_heldout_trajectories_with_history(...)
```

流程：

1. 遍历 `test_trajectories`
2. 对每个 `b >= history_blocks`
3. 生成该 trajectory 的 fixed noisy observation
4. 用 history 估计 `y0_est`
5. 预测当前 clean block
6. 聚合指标

### 11.2 首版指标

首版只报告：

1. `position_rmse`
2. `rotation_geodesic`
3. `velocity_rmse`
4. `angular_rmse`
5. 每个 DOF 的 RMSE

与现有 held-out 报告格式尽量保持一致。

### 11.3 对照组

至少保留三个对照：

1. `clean current_init`
2. `noisy current_init without history`
3. `noisy current_init with history`

这三组能回答：

- 噪声到底带来多大退化
- history 是否真的有帮助
- 帮助来自状态恢复还是偶然正则效应

---

## 12. CLI 与配置建议

建议在新入口中新增以下参数：

```text
--backbone_model_type
--history_blocks
--history_hidden_dim
--history_encoder {gru}
--history_est_loss_weight
--backbone_checkpoint
--freeze_backbone_epochs
--history_eval_max_trajectories
```

首版推荐默认值：

```text
--backbone_model_type phnode_full
--history_blocks 10
--history_hidden_dim 128
--history_encoder gru
--history_est_loss_weight 0.05
--freeze_backbone_epochs 10
```

---

## 13. 分阶段实施步骤

### 阶段 1：数据与接口打通

完成条件：

1. 新增 `get_train_trajectories()`
2. 新增 `AUVTrajectoryWindowDataset`
3. 新增 `create_trajectory_window_dataloaders_from_dataset()`
4. 能打印出：
   - number of windows
   - history length
   - per-split trajectory count

### 阶段 2：trajectory noise cache

完成条件：

1. 能为整条 trajectory 生成 noisy observation
2. 同一 epoch 内，同一 trajectory 的不同 window 一致
3. 不同 epoch 的 realization 不同
4. eval 时 fixed seed 可复现

### 阶段 3：history model

完成条件：

1. `BlockEncoder` 跑通
2. `HistoryEncoder` 跑通
3. `HistoryStateEstimator` 能输出 `y0_est`
4. `y0_est` shape、layout 与 backbone 完全兼容

### 阶段 4：训练器

完成条件：

1. 新入口可以训练一个 epoch
2. loss 正常下降
3. 没有 `NaN / Inf / solver failure` 爆炸
4. checkpoint 能保存与恢复

### 阶段 5：history-aware eval

完成条件：

1. 能在 held-out trajectory 上跑完整评估
2. 能输出 clean-target metrics
3. 能与 `no-history noisy current_init` 做 head-to-head 对照

### 阶段 6：最小实验

最小实验建议：

1. `phnode_full`
2. 一组 `nominal_train`
3. `history_blocks = 10`
4. `max_trajectories = 32` 先 smoke test
5. 再扩到 full held-out

---

## 14. 建议的验证命令

以下是建议的第一批命令模板。

### 14.1 smoke test

```bash
python train_auv_history_model.py ^
  --dataset ./data/oc/<dataset>.pkl ^
  --backbone_model_type phnode_full ^
  --history_blocks 10 ^
  --history_hidden_dim 128 ^
  --epochs 5 ^
  --total_steps 200 ^
  --batch_size 64 ^
  --noise_profile nominal_train
```

### 14.2 带 warm start 的主实验

```bash
python train_auv_history_model.py ^
  --dataset ./data/oc/<dataset>.pkl ^
  --backbone_model_type phnode_full ^
  --backbone_checkpoint ./checkpoints/<run>/best_model.pt ^
  --history_blocks 10 ^
  --history_hidden_dim 128 ^
  --freeze_backbone_epochs 10 ^
  --noise_profile nominal_train
```

---

## 15. 风险清单

实施时最容易出问题的点有五个：

1. **trajectory cache 与 sample window 不一致**
   这会直接破坏 mission-level 噪声语义

2. **history 输入和 current init 的 state convention 混乱**
   data-space 与 ODE-space 必须严格区分

3. **block-relative `Δp` 被误当成跨 block 连续绝对位置**
   这是错误的

4. **history estimator 改动过大，直接覆盖 backbone 能学到的动力学结构**
   所以建议 residual correction + 小权重辅助项

5. **评估时偷偷给了未来信息**
   例如 history window 不小心包含当前 block 的未来点

---

## 16. 推荐的第一版终态

如果按最小闭环来定义，`v4-B1` 第一版完成的标志应是：

```text
在不修改现有 backbone ODE 结构的前提下，
用 noisy history 改善当前 clean block prediction，
并能在 held-out trajectories 上稳定评估。
```

做到这一点，就已经足够支撑下一步：

```text
v4-B2：history-aware multi-block rollout
```

在此之前，不建议提前把任务扩大。
