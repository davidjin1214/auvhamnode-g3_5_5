# 轨迹噪声设计 v4（修订版：noisy input -> clean target）

本文档重写 v4 方案的任务定义与训练协议。

核心修正只有一条：

```text
v4 的目标是“从导航样式的带噪观测学习真实动力学”，
不是“从 posterior 轨迹拟合 posterior 轨迹”。
```

因此，v4 的正确训练语义必须是：

```text
noisy observation input  ->  dynamics model  ->  clean future trajectory
```

而不是：

```text
noisy observation input  ->  dynamics model  ->  noisy navigation trajectory
```

---

## 1. 文档定位

- [docs/noise_design_v3_remus100_reference_grounded.md](noise_design_v3_remus100_reference_grounded.md) 负责给出各通道的误差幅值预算 `σ`
- 本文档负责给出这些误差在时间上的物理语义、如何生成 noisy observation，以及它们应当如何进入训练
- 本文档不再把“整条 noisy posterior 轨迹”定义为训练监督目标

v4 的研究问题是：

```text
当部署时只能拿到 DR / EKF 导航估计状态，而不是仿真真值时，
模型能否仍然学到并预测真实 AUV 动力学？
```

这一定义与当前项目主线一致：研究的是 **dynamics robustness to navigation-state uncertainty**，
而不是导航滤波器输出建模。

---

## 2. 任务定义

### 2.1 主任务

主任务必须写成：

```text
给定带噪导航状态观测 x_hat，
学习真实动力学 f_true，
并预测 clean future trajectory x_true(1:T)。
```

当前仓库中，最严格、最清晰的任务形式是：

```text
y0_noisy -> ODE rollout -> pred_{1:T}
target   = clean future trajectory
```

其中 `y0_noisy` 是导航系统后处理输出的近似观测，
`target` 始终是仿真真值。

### 2.2 明确不做的任务

本方案**不做**以下事情：

1. 不把 noisy posterior trajectory 作为主监督目标
2. 不让 loss 奖励模型去复现滤波器残差、bias 漂移或 phase lag
3. 不把 raw sensor stream 建模为训练对象
4. 不在“纯动力学模型”名义下偷偷加入未说明的 observer / denoiser / smoothing block

### 2.3 两种可接受的研究线

v4 允许两条研究线，但必须严格区分：

1. **纯动力学线**
   `noisy input -> clean target`
   这是本文档的默认主线

2. **动力学 + observer 线**
   输入是 noisy sequence，前端显式使用短历史进行状态恢复，再预测 clean target
   这条线是后续扩展，不属于当前仓库的无记忆 open-loop trainer

如果未来真的引入 context encoder / observer front-end，必须在论文和结果表里单独命名，
不能继续与“纯 dynamics model”混写。

---

## 3. 当前代码事实与 v4 的边界

### 3.1 当前 trainer 真正消费什么

当前训练循环中，模型真正接收的是 `t=0` 的初始状态：

```text
clean_y0 or noisy_y0 -> open-loop rollout -> clean target_{1:T}
```

对应实现见：

- [train_auv_hamnode.py](../train_auv_hamnode.py)
- [`build_noisy_initial_condition`](../train_utils.py)
- [`se3_trajectory_loss`](../train_utils.py)

这意味着：

- 当前主训练器**并不消费整段 noisy observation sequence**
- 如果不改 trainer，仅在 collate 阶段生成整条 OU 轨迹，并不会自动变成“sequence-noisy learning”
- 因此，v4 首版必须区分：
  - **v4-A：当前 trainer 兼容版**
  - **v4-B：需要 trainer 升级的 sequence-input 版**

### 3.2 为什么这一步必须说清

如果不先把这个边界说清，就会出现两个常见误判：

1. 以为“生成了整条 noisy trajectory”就等于“模型学会了从 noisy sequence 学动力学”
2. 以为“用 posterior-like 轨迹做 target，但 τ 很大”就不会引入滤波器行为

这两个判断都不成立。

### 3.3 `v3` / `v4-A` / `v4-B` 的关系

这三个名字容易被误读成三代完全不同的方法。实际并不是。

从**当前仓库可执行的训练行为**看：

| 方案 | 输入 | target | 是否需要改 trainer | 方法本质 |
|---|---|---|---|---|
| `v3` | noisy IC | clean truth | 否 | profile-based noisy IC regularization |
| `v4-A` | noisy IC | clean truth | 否 | 用 mission-level OU 语义重写 v3 的 noisy IC |
| `v4-B` | noisy observation history | clean truth | **是** | dynamics + observation-history / observer |

因此：

1. **`v3` 与 `v4-A` 的差别主要是物理语义和扩展约束，不是主训练接口改变**
2. **`v4-B` 才是实质性新增的方法轴**

更尖锐地说：

```text
v4-A ≈ v3 的规范化、任务澄清版
v4-B 才是新的训练问题
```

这也是本文档为什么把 `v4-A` 作为当前主线，
而把 `v4-B` 单列为“需要 trainer 升级”的原因。

---

## 4. 状态语义与一致性约束

### 4.1 数据状态与模型状态

数据集存储的状态是：

```text
[Δp(3), R(9), nu_total(6), u_act(3), u_cmd(3), v_c^n(3)]
```

其中：

- `Δp`：block-relative 位置
- `R`：body-to-inertial 旋转矩阵
- `nu_total`：总机体系速度
- `u_act`：执行器状态
- `u_cmd`：控制命令
- `v_c^n`：惯性系海流速度

模型内部真正积分的是：

```text
nu_r = nu_total - R^T v_c^n
```

因此在 OC 场景下，噪声设计必须首先控制 **模型真正消费的 `nu_r`**，
而不能把 `nu_total` 与 `v_c^n` 当作互不相关的观测量随意独立拍噪。

### 4.2 ODE-space-first 原则

所有 observation noise 的主构造都应先在 ODE 语义里定义，再映射回 data convention：

1. 从 clean state 得到 clean ODE state
2. 在 `R / nu_r / u_act / v_c` 上生成物理可解释的误差
3. 用同一组 noisy state 重建 data-space 观测

这条原则延续 v3 的 ODE-space-consistent 思路，不改变。

### 4.3 block-relative 位置的约束

当前数据的 `Δp` 是 block-relative 位置，每个 block 起点约定：

```text
Δp(t0) = 0
```

因此：

- 不对 `Δp(t0)` 单独加噪
- 若需要生成 noisy trajectory input，则 `Δp(t)` 必须由同一组 noisy kinematics 积分得到
- 不能把位置当作一个独立的“随手叠加噪声”的通道

---

## 5. 物理对象：DR / EKF posterior，而非 raw sensor

### 5.1 v4 模拟的是什么

v4 模拟的是：

```text
导航系统后处理输出的状态估计轨迹
```

而不是：

```text
DVL beam / gyro / accel / compass 的原始传感器流
```

因此，v4 噪声模型应解释为：

- bias 估计残差
- 慢变漂移
- 由速度误差积分形成的位置误差
- 必要时的固定方向 bias 压力测试

而不是原始传感器带宽上的高频 white noise。

### 5.2 时间结构

延续旧版 v4 的思路，时间相关结构仍采用单尺度 OU 过程作为首选父模型：

```text
dξ_i/dt = -ξ_i / τ_i + sqrt(2 / τ_i) * w_i(t)
η_i(t)  = σ_i(t) * ξ_i(t)
```

其中：

- `ξ_i(t)`：单位方差 OU 状态
- `σ_i(t)`：由 v3 profile 给出的幅值预算
- `η_i(t)`：观测误差

默认工程选择仍可取：

```text
τ = 900 s
```

但需要明确：

- 在 **v4-A（仅 noisy IC）** 里，训练器只消费 `η(0)` 的稳态边缘分布，几乎不直接利用 `τ`
- 在 **v4-B（sequence input）** 里，`τ` 才真正进入训练样本的时间结构

也就是说，OU 是 v4 的**父观测模型**；是否显式用到其时间相关性，取决于 trainer 是否消费整段 noisy sequence。

---

## 6. 各通道的修订设计

本节保持 v3 的 `σ` 数值预算不变，重点修正它们在 v4 中的使用方式。

### 6.1 `delta_nu_r`

`nu_r` 是最核心的噪声通道。

仍采用 v3 的状态相关结构：

```text
σ_lin_i(t) = sqrt(floor_i^2 + (ratio_i * |nu_r_i(t)|)^2)
σ_ang_i(t) = constant
```

profile 数值直接继承 v3：

- `nominal_train`
- `nominal_eval`
- `degraded_eval`

`heading_biased_eval` 的随机部分沿用 `nominal_eval`。

### 6.2 `delta_theta`

姿态误差仍在 SO(3) 上作用：

```text
R_tilde(t) = Exp(delta_theta(t)) * R(t)
```

profile 数值也直接继承 v3。

但这里要补上一条旧稿中没有说清的关键约束：

```text
姿态误差对位置误差是一阶耦合，不可在 noisy trajectory 生成时忽略。
```

如果 `R_tilde` 变了，而位置仍用 clean `R` 去积分速度，
就会生成一个不满足同一套运动学约束的伪 observation。

### 6.3 `delta_p`

位置误差不再单独定义为“独立位置噪声通道”。

若需要合成 noisy trajectory input，则必须用 noisy kinematics 一致生成：

```text
nu_total_tilde[:, :3] = nu_r_tilde[:, :3] + R_tilde^T v_c_tilde^n
nu_total_tilde[:, 3:] = nu_r_tilde[:, 3:]

d(Δp_tilde)/dt = R_tilde * nu_total_tilde[:, :3]
Δp_tilde(t0)   = 0
```

离散实现时，位置应沿当前 block 的时间栅格积分，和数据原始 `timestamps` 对齐。

这里必须强调三点：

1. `Δp(t0)` 仍固定为 0
2. 位置误差来自 noisy 运动学的积分结果
3. 不能再写成“只用 clean `R` 积分线速度噪声，姿态对位置是二阶项所以忽略”

该近似对本项目不成立。

### 6.4 `delta_u_act`

执行器观测不属于 DR / EKF 主导航链路。

因此建议：

- **IC 噪声**：允许保留 v3 的小幅一次性扰动
- **trajectory noise**：默认 `σ = 0`

只有当未来明确模拟 servo readback 漂移，才单独加新通道；
在 v4 首版里不把它塞进导航 posterior 误差模型。

### 6.5 `delta_v_c`

这里需要比旧稿更严谨地区分三种场景：

1. **`REMUS100-DR-surrogate`**
   当前仓库 `oc` 状态布局仍显式携带 `v_c`，若主实验继续沿用 clean `v_c`，
   应明确把它称为“known-current surrogate”，而不是“完整现实 DR”

2. **`REMUS100-INS`**
   若把 `v_c` 视为增强配置下的已估计状态，则可按 v3-INS 表注入 OU / bias

3. **`DR-hidden-current`**
   若要把海流完全视为未观测扰动，需要改模型接口与状态定义，不属于 v4 首版

因此，本文建议：

- 当前默认实现仍可与现有 `remus100_dr` 命名兼容
- 但文档语义上必须明确：它是 **DR kinematics + known-current surrogate**

这样才能避免把 clean `v_c` 误写成“典型 DR 部署现实”。

---

## 7. v4 的两阶段方案

### 7.1 v4-A：当前 trainer 兼容版（推荐主线）

#### 定义

```text
mission-consistent noisy initial condition
with clean future trajectory supervision
```

#### 训练接口

```text
y0_noisy -> ODE rollout -> pred_{1:T}
target   = clean trajectory
```

#### 实现方式

1. 仍从 clean `y0` 出发
2. 在 ODE-space 采样与 mission-level OU 父模型一致的 `η(0)`
3. 只把 `η(0)` 用于构造 noisy IC
4. target 保持 clean
5. 复用现有 `warmup / ramp / mix_ratio`

#### 重要说明

v4-A 不是“整段 trajectory-noisy training”。

它只是：

- 用更清晰的 mission-level 物理语义解释 noisy IC
- 为未来 sequence-input 扩展提供统一父模型
- 保持当前纯 dynamics trainer 的科学闭合性

从科学叙事上看，v4-A 仍然是：

```text
noisy-to-truth dynamics learning
```

这也是当前仓库最稳妥、最合适的主线版本。

### 7.2 v4-B：sequence-input 扩展版（需要 trainer 升级）

#### 何时才成立

只有当训练器真正消费 noisy observation history 时，v4-B 才成立。

可接受的升级方式包括：

1. receding-horizon training
2. k-step teacher forcing
3. 显式 observer front-end + clean-target rollout

#### 正确接口

```text
noisy observation window/history -> model -> clean future trajectory
```

这里 noisy trajectory 只能作为 **input**，
不能偷偷变成 **target**。

#### 需要的额外约束

若采用 observer / context encoder，则必须在结果中单列为：

```text
dynamics + observer
```

而不能继续写成“纯 dynamics model”。

#### v4-B 的意义

它允许你研究一个更接近现实的问题：

```text
模型在持续接收带噪导航状态时，能否保持对真实未来动力学的预测能力？
```

但这是后续问题，不应和 v4-A 混在一起。

#### 7.2.1 `v4-B` 与 `v3 / v4-A` 的本质区别

`v4-B` 的新增内容不是“噪声更复杂”，而是**模型可用信息变了**。

`v3 / v4-A` 中，模型只能看到：

```text
t = 0 的一个 noisy initial state
```

因此它回答的是：

```text
当初始状态有误差时，open-loop dynamics rollout 是否仍能预测真轨迹？
```

而 `v4-B` 中，模型持续看到：

```text
一段 noisy observation history
```

因此它回答的是：

```text
当模型能利用时间历史时，是否能一边恢复 latent clean state，
一边学习并预测真实动力学？
```

所以 `v4-B` 实际上是：

```text
dynamics + state recovery / observer
```

而不再是纯粹的 noisy-IC robustness。

#### 7.2.2 推荐的最小可实现方案

对当前代码库，最小、最稳妥、并且真正能跑起来的 `v4-B`，
我建议采用：

```text
history encoder + current-state estimator + existing ODE rollout
```

而不是一上来就把整个 trainer 改成复杂的全序列 teacher forcing。

实施层面的分阶段清单见：

- [docs/v4_b1_implementation_checklist.md](v4_b1_implementation_checklist.md)

#### A. 数据接口

当前数据集已经保存：

- `train_trajectories`
- `test_trajectories`
- `train_meta`
- `test_meta`

因此 `v4-B` **不需要重生成数据集**。

需要新增的是 trajectory-aware dataset，而不是 block-only dataset。

推荐新增：

```text
AUVTrajectoryWindowDataset
```

每个样本定义为：

- 一条 trajectory `traj_idx`
- 一个当前 block 索引 `b`
- 一个历史窗口长度 `L_ctx`

样本内容建议为：

```text
history_blocks = trajectory[b-L_ctx : b]      # 完整历史 blocks
current_init   = trajectory[b, 0]             # 当前 block 起点观测
target_block   = clean trajectory[b]          # 当前 block 的 clean 5-point target
```

其中 `b >= L_ctx`。

默认建议：

- `L_ctx = 10` blocks = `2.0 s` 历史

这个长度通常足以覆盖：

- actuator lag 的影响
- 一小段姿态/速度演化
- 观测噪声与真实动态响应之间的时间差异

#### B. 为什么用“历史 blocks + 当前初值”

当前数据的 `Δp` 是 block-relative，跨 block 会重置。

因此最自然的做法不是把整条 trajectory 当成一个连续绝对位置序列去喂模型，
而是：

1. 用过去 `L_ctx` 个 block 的局部轨迹作为 history
2. 用当前 block 的起点状态作为要估计的“当前状态”
3. 让模型预测当前 block 的 clean future

这样可以最大限度复用现有数据契约，而不需要先重构绝对位置状态。

#### C. 噪声生成

`v4-B` 仍使用 mission-level OU 父模型，但用法与 `v4-A` 不同。

对每条 trajectory、每个 epoch：

1. 采样一条 trajectory-level OU realization
2. 用同一 realization 给 `history_blocks` 和 `current_init` 生成 noisy observation
3. `target_block` 保持 clean

也就是说：

```text
input  = noisy history + noisy current initial observation
target = clean current/future block
```

历史与当前共享同一条 realization，这一点非常重要。

#### D. 模型结构

最小推荐结构分三段：

1. **Block encoder**
   把一个 block 内的 `5 x state_dim` 局部轨迹编码成一个 embedding

2. **History encoder**
   对 `L_ctx` 个 block embedding 做时序聚合
   可用 GRU / small Transformer / temporal CNN

3. **State head + existing dynamics model**
   用 history encoder 的输出修正当前 noisy initial state，
   得到 `y0_est`
   再把 `y0_est` 送入现有 `AUVHamNODE` 或 baseline rollout

建议采用的最小形式是：

```text
y0_est = y0_noisy + Delta(history)
```

而不是让网络从零生成整个状态。

这种 residual correction 更稳定，也更符合“navigation-state refinement”的物理语义。

#### E. 训练目标

主损失保持：

```text
pred_block = ODE(y0_est, u_cmd_current)
loss_main  = se3_trajectory_loss(pred_block, clean_target_block)
```

可选地再加一个很轻的辅助项：

```text
loss_est = ||y0_est - y0_clean||^2
```

但辅助项必须是次要的，避免模型退化成单纯状态去噪器。

推荐总损失：

```text
loss = loss_main + lambda_est * loss_est
```

其中：

- `lambda_est = 0.05` 可作为首试量级

#### F. 为什么先预测“当前 block”

这是最小可实现方案里最重要的降复杂策略。

先让 `v4-B` 做：

```text
history -> estimate current clean state -> predict current clean block
```

而不是一上来做多 block 长时 rollout，原因有三点：

1. 这与当前 block-level loss 最兼容
2. 不需要先解决未来多 block 的 `u_cmd` 调度与状态传递问题
3. 能先验证“history 是否真的帮助恢复 clean state”

等这一步成立后，再升级到：

```text
history -> y0_est -> multi-block rollout
```

#### G. 推荐实现顺序

我建议按以下顺序落地：

1. 新增 `AUVTrajectoryWindowDataset`
2. 新增 trajectory-level noisy observation synthesis
3. 新增 `HistoryEncoder`
4. 训练 `HistoryEncoder + existing dynamics model`
5. 首先只预测当前 block
6. 验证有效后再扩展到 `H > 1` blocks

#### H. 评估接口

`v4-B` 不能继续直接复用当前只看独立 block 的 held-out evaluation。

应新增一条 trajectory-aware 评估路径，例如：

```text
evaluate_heldout_trajectories_with_history()
```

其流程为：

1. 遍历 `test_trajectories`
2. 对每个 block `b >= L_ctx` 提供 noisy history
3. 预测当前 clean block
4. 聚合 position / rotation / velocity 指标

首版不要求一开始就做全 30s 链式 rollout；
先把“history-aware block prediction”验证清楚更重要。

#### 7.2.3 一期与二期目标

为了避免 `v4-B` 一上来工程量失控，建议拆成两期：

#### `v4-B1`

```text
history-aware clean block prediction
```

特点：

- 只预测当前 block
- 用 history 改善当前 clean state 估计
- 最大限度复用现有 dynamics rollout 和 loss

#### `v4-B2`

```text
history-aware multi-block rollout
```

特点：

- 从 history 估计 `y0_est`
- 继续向后 rollout 多个 block
- 需要更完整的 future command scheduling 与评估协议

我建议你把真正的 `v4-B` 首次落地定义为 `v4-B1`。
这是最可能成功、也最容易解释的版本。

---

## 8. 为什么不能把 noisy trajectory 当监督目标

设真实动力学轨迹为 `x(t)`，导航 posterior 为：

```text
x_hat(t) = x(t) + eta(t)
```

如果训练目标写成：

```text
loss = ||x_pred(t) - x_hat(t)||^2
```

那么最优解会被奖励去复现 `eta(t)` 中保留下来的：

- bias
- 慢漂移
- lag
- 几何不一致

无论 `eta(t)` 来自 raw sensor 还是 EKF posterior，这一点都不会改变。

`τ >> T` 只能减轻模型去拟合快速衰减项的诱因，
但不能把 noisy target 重新变成 clean dynamics target。

因此，为了避免模型“隐式学习滤波器行为”，最硬的约束不是“把 posterior 做得更平滑”，
而是：

```text
target 必须保持 clean truth
```

---

## 9. 训练调度

当前 `warmup / ramp / mix_ratio` 机制可继续沿用。

推荐解释如下：

- `warmup`：先让结构先验在 clean setting 下锁定动力学主模态
- `ramp`：逐步抬高 noisy observation 的误差幅值
- `mix_ratio`：保持 clean / noisy 样本混合，避免模型完全围绕 noisy input 重排参数

### v4-A 建议

沿用当前默认：

```bash
--noise_profile nominal_train \
--noise_warmup_epochs 20 \
--noise_ramp 80 \
--noise_mix_ratio 0.5
```

若后续发现主模型依然存在结构相关的退化，再按模型单独调 schedule，
不要先把问题归咎于噪声物理模型。

---

## 10. 评估协议

评估协议延续当前 canonical benchmark 的主口径，不改。

### 10.1 真值口径

主评估始终对 clean truth 打分：

- 不在 target 上注入轨迹噪声
- 不用 noisy posterior 作为主 metric 基准

### 10.2 v4-A 的评估

继续沿用当前 noisy IC benchmark：

- `clean`
- `nominal_eval`
- `degraded_eval`
- `heading_biased_eval`

若是 `oc` 场景，可按当前实验设计保留：

- `current_bias_eval`

但要在报告中说明它对应的是哪一种 `v_c` 语义。

### 10.3 v4-B 的评估

若未来升级成 sequence-input trainer，则评估也应给模型喂 noisy observation history，
但评估指标仍是 clean truth 上的：

- position error
- rotation geodesic
- velocity / angular error
- rollout stability

必要时可以把“对 noisy posterior 的跟踪误差”作为辅助诊断单列，
但不能替代 clean-target 指标。

---

## 11. 实验轴建议

### 11.1 当前阶段

当前最值得保留的对比是：

1. `clean`
2. `ic_only_v3`
3. `v4-A`：mission-consistent noisy IC with clean target

这三档已经足以回答：

- 纯 clean 训练上限是多少
- 现有 v3 IC-only 正则是否有效
- 在更清晰的 mission-level 物理语义下，noisy-to-truth 学习是否仍成立

### 11.2 不建议保留的旧写法

旧稿中的：

```text
trajectory_v4 = clean IC + noisy trajectory target
```

不再建议作为主实验轴。

它可以作为非常特殊的对照存在，但不能再被写成物理主场景。

### 11.3 后续阶段

当 trainer 升级后，再增加：

4. `v4-B`：sequence noisy input + clean target

这时才有资格研究：

- mission-level temporal correlation
- observation history 对预测的帮助
- τ 的敏感性

---

## 12. 诊断建议

主指标仍是 canonical rollout benchmark。

在此之外，建议保留两类辅助诊断：

### 12.1 扰动响应诊断

固定 clean truth，
对两个不同 noisy IC 做 rollout，
比较它们误差差分 `delta(t)` 的衰减行为。

若模型学到的是合理动力学，
其衰减应与物理阻尼模态一致；
若出现异常慢尺度，则说明模型可能仍在吸收观测误差结构。

### 12.2 noisy-input vs clean-input gap

同一模型、同一 trajectory，分别输入：

- clean initial state
- noisy initial state / noisy observation history

比较两者对 clean truth 的误差差分。

这个 gap 应被解释为对观测不确定性的敏感性，
而不是 target 本身变脏后的“拟合改进”。

---

## 13. 对实现者的直接约束

### 13.1 允许的最小改动

对当前代码，最小且正确的 v4 落地是：

1. 保持 `target = clean`
2. 保持 `loss` 不对 noisy target 打分
3. 保持 `build_noisy_initial_condition()` 作为主注入入口
4. 将 mission-level OU 明确写成父模型语义，而不是立即强行把整条 noisy sequence 接入 trainer

### 13.2 v4-B 才需要的改动

只有要做 sequence-input 时，才需要新增：

1. trajectory-aware dataset / collate
2. noisy observation sequence synthesis
3. 明确的 observation-input interface
4. 与 clean-target 对齐的 sequence trainer

### 13.3 不允许的实现

以下做法都应明确禁止：

1. 对 noisy posterior trajectory 直接做主监督
2. 对 target 做滑动平均后再训练
3. 用 clean `R` + noisy `nu` 生成位置，同时另外给 `R` 加姿态噪声
4. 把显式 observer 前端混进主模型却仍宣称“纯 dynamics”

---

## 14. 最终拍板建议

### D1. 主任务定义

采用：

```text
noisy observation input -> clean future trajectory
```

### D2. 当前主线版本

采用：

```text
v4-A：mission-consistent noisy IC
```

### D3. 时间相关噪声的地位

OU 仍保留，但作为父观测模型：

- 当前用于解释 `η(0)` 的来源
- 后续用于 sequence-input 扩展

### D4. 位置生成规则

若未来构造 noisy trajectory input，
位置必须由同一套 noisy kinematics 积分生成。

### D5. `v_c` 语义

当前 `remus100_dr` 在 `oc` 状态布局下应明确写为：

```text
DR kinematics + known-current surrogate
```

不要把它直接写成“完整现实 DR”。

---

## 15. 一句话总结

v4 的正确方向不是“把 noisy posterior 当成新的监督目标”，
而是：

```text
用物理一致的导航样式噪声去污染模型输入，
同时始终让模型为 clean truth 负责。
```

这才是“从噪声数据学习真实动力学”的自洽定义。
