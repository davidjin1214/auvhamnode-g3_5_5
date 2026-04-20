# `v4-lite` 协议规格

本文档给出 `v4-lite` 的**协议规格**，用于指导后续实现。

它与
[noise_design_v4_lite_traj_consistent_ic.md](noise_design_v4_lite_traj_consistent_ic.md)
的关系是：

- 后者负责解释研究动机、价值与实验意义
- 本文档负责定义协议合同、实现边界、验收标准

本文档不讨论 `v4-B`。`v4-lite` 仍属于 pure dynamics 主线。

---

## 1. 协议目标

`v4-lite` 要回答的问题是：

```text
如果把 noisy initial condition 从“block 独立采样”
升级为“来自同一条 trajectory-consistent noisy observation”，
pure dynamics 模型的训练与评估结论是否会改变？
```

因此，`v4-lite` 的科学定位是：

```text
trajectory-consistent noisy IC protocol
for pure dynamics models
```

它不是：

- 新模型
- history-aware encoder
- observer-augmented dynamics
- multi-block rollout framework

---

## 2. 协议范围

### 2.1 `v4-lite` 必须保持不变的部分

以下内容与当前 block rollout 主线保持一致：

- 模型 backbone 不变
- 训练目标仍是 clean truth
- 输入仍是当前 block 的 initial state
- 输出仍是当前 block 的 rollout
- 不输入 history
- 不新增 state recovery 模块

### 2.2 `v4-lite` 唯一允许改变的部分

唯一改变的是：

```text
当前 block 所使用的 noisy initial state 的来源
```

即：

- 旧协议：对每个 block 的 clean `y0` 独立采样 noisy IC
- `v4-lite`：先生成 trajectory-level noisy observation，再从当前 block 起点读取 noisy `y0`

---

## 3. 正式任务定义

对每个 block，`v4-lite` 的训练和评估任务都是：

```text
input:
  y0_noisy_from_trajectory_observation

target:
  clean current block trajectory
```

这意味着：

- noisy 的只是真正给模型的当前初值
- target 始终是 clean
- block 的 future trajectory 仍来自 clean truth

---

## 4. 核心协议要求

### 4.1 trajectory consistency

对于同一条 trajectory，在同一次训练 epoch 或同一次评估 run 中：

- 所有 block 必须共享同一条 noisy observation trajectory
- 不允许每个 block 重新独立采样一个新的 noisy realization

这条是 `v4-lite` 最核心的协议要求。

### 4.2 block input contract

对于 trajectory `j` 的第 `b` 个 block：

- clean block 起点状态记为 `x_clean[j, b, 0]`
- noisy observation trajectory 在对应时刻的状态记为 `x_obs[j, t_b]`
- 模型实际使用的初值应来自 `x_obs[j, t_b]`

其中 `t_b` 是该 block 的起始时间索引。

### 4.3 target contract

无论训练还是评估：

- target 始终是 clean current block trajectory
- 不得把 noisy observation trajectory 当作监督目标

### 4.4 backbone contract

`v4-lite` 第一版不得要求：

- 修改 `AUVHamNODE` 主体结构
- 为 baseline 添加新的 history path
- 改写成 multi-block trainer

否则就不再是 pure dynamics 主线的协议对比。

---

## 5. 状态语义要求

### 5.1 ODE-space-first

`v4-lite` 必须延续当前噪声设计的主原则：

1. 从 clean data-state 出发
2. 转换到 model-consumed ODE semantics
3. 在 ODE-space-consistent 变量上施加误差
4. 再映射回 noisy observation contract

尤其在 `oc` 情况下，必须优先控制模型真正消费的：

```text
nu_r = nu_total - R^T v_c^n
```

### 5.2 `R / nu_r / v_c / u_act`

`v4-lite` 第一版至少应保证以下变量的 trajectory-level 一致性语义：

- `R`
- `nu_r`
- `v_c`（若该任务包含 ocean current）
- `u_act`（若保留 actuator state uncertainty）

### 5.3 `Δp` 的处理

block-relative 位置 `Δp` 不应被当作独立噪声通道直接拍值。

协议要求是：

- `Δp(t0)` 不单独加噪
- 若生成完整 noisy observation trajectory，则 block 内 `Δp` 应由 noisy kinematics 一致导出

第一版如果只在训练器内部需要 block 起点 `y0_noisy`，可以不把完整 `Δp` 作为训练输入显式暴露，但不能在语义上把它重新退回 block-iid 噪声。

---

## 6. 时间相关性要求

### 6.1 最小要求

`v4-lite` 的最小要求不是必须完全复现某种连续时间导航后验，而是：

- 同一 trajectory 内的噪声 realization 相互一致
- 不同 block 之间能反映“来自同一条 noisy observation”这一事实

### 6.2 推荐实现

推荐采用 mission-level latent noise process，再映射到 state channels。

无论底层选用：

- OU-like 过程
- bias + drift
- 离散 AR 过程

只要满足 trajectory consistency，即可视为协议合格。

### 6.3 第一版不强求的事情

`v4-lite` 第一版不强求：

- 完整导航滤波器后验协方差重建
- sensor-level sampling fidelity
- history-aware posterior correction

这些都超出了 `v4-lite` 的职责边界。

---

## 7. 训练协议

### 7.1 训练样本定义

训练器仍以 block 为基本 supervision 单位。

但 noise generation 的基本单位变为：

```text
trajectory per epoch
```

即：

- 每个 epoch 为每条 trajectory 生成一条 noisy observation trajectory
- 该 trajectory 中所有 block 都从这同一 realization 中读取起点 noisy state

### 7.2 resampling 规则

推荐：

- 不同 epoch 允许重新采样 trajectory-level noisy observation
- 同一 epoch 内同一 trajectory 不得重复重采样

### 7.3 mix 规则

如果保留 `noise_mix_ratio`，则应明确其作用层级。

推荐定义是：

- mix 的对象仍是 block 样本
- 但一旦某个 block 进入 noisy 路径，其 noisy initial state 必须来自该 epoch 下对应 trajectory 的唯一 noisy observation

不允许出现：

```text
同一条 trajectory 里，
两个 noisy block 来自两个互不相关的 noisy observation realization
```

---

## 8. 评估协议

### 8.1 `v4-lite` eval 的定义

对 held-out 或 rollout benchmark 中的每条 trajectory：

- 先生成一条 trajectory-consistent noisy observation
- 再按 block 起点读取 noisy initial state
- 在相同 noisy observation 来源下完成全部 block 或 rollout 评估

### 8.2 与 iid noisy eval 的关系

`v4-lite` eval 必须能和当前 iid noisy eval 做 matched comparison。

两者唯一差别应是：

- initial noisy state 的来源机制不同

不允许在比较中同时改：

- targets
- horizons
- seeds
- model budget
- reporting logic

### 8.3 bias-type profile 叠加规则

`heading bias` 这类 profile 在 `v4-lite` 路径下应被解释为：

- 叠加在 trajectory-consistent noisy observation 之上的额外 bias

而不是重新退回 block-iid 噪声。

---

## 9. 随机种子与可复现性

### 9.1 层级

建议采用如下 seed 层级：

- global seed
- epoch seed
- trajectory seed
- channel stream id

### 9.2 必须保证的复现性

给定：

- 相同 global seed
- 相同 epoch
- 相同 trajectory id
- 相同 profile

生成的 noisy observation 应一致。

### 9.3 不同对象的独立性

以下对象应彼此解耦：

- trajectory 间的 realization
- 误差通道间的随机流
- 训练与评估中的随机流

---

## 10. 与现有代码路径的关系

### 10.1 推荐实现方式

`v4-lite` 推荐作为当前 noisy-IC 接口的一个新协议层接入，而不是新建一套平行训练框架。

也就是说，首选是：

- 扩展当前 noise synthesis 路径
- 保持 `train_auv_hamnode.py` 的主训练接口不变
- 在 `train_utils.py` 中新增 trajectory-consistent noise builder

### 10.2 第一版不建议做的事情

- 不要把 `v4-lite` 与 `v4-B1` 混写
- 不要新增 history encoder
- 不要重写 model registry
- 不要把 rollout benchmark 逻辑改成 observer benchmark

---

## 11. 协议产物

为便于后续审计，`v4-lite` 路径建议额外记录：

- protocol name
- protocol version
- noise reference
- trajectory-level sampling contract
- bias 叠加规则
- epoch / eval seed 规则

推荐在 run 产物中写出：

- `noise_protocol.json`
- `noise_budget_summary.txt`

至少让后续能区分：

- `iid noisy-IC`
- `v4-lite`

避免两者在 catalog 里混成同类 run。

---

## 12. 验收标准

`v4-lite` 被视为“协议实现完成”，至少要满足以下条件。

### 12.1 协议正确性

1. 同一 trajectory 内 block 使用同一 noisy observation realization
2. `v4-lite` 与 iid noisy-IC 仅在 noise source 上不同
3. target 始终为 clean truth

### 12.2 工程兼容性

1. 不破坏当前 clean / iid noisy-IC 路径
2. 能在 `noc` 与 `oc + known-current surrogate` 下运行
3. 能进入现有 held-out 与 rollout benchmark 流程

### 12.3 结果可解释性

1. summary 中能区分 `iid` 与 `v4-lite`
2. 能输出 by-seed、by-scenario、by-horizon 比较
3. 能计算 clean replay 代价

---

## 13. 第一版不回答的问题

为防止协议膨胀，`v4-lite` 第一版明确不回答：

- history 是否有助于 state recovery
- observer-augmented dynamics 是否更强
- multi-block trajectory prediction 是否更优
- sensor-level realism 是否足够

这些问题应留给后续 `v4-B` 路线，而不是混进 `v4-lite`。

---

## 14. 一句话规格摘要

`v4-lite` 的正式定义可以压缩为：

```text
在保持 pure dynamics trainer、clean target 和 block rollout 不变的前提下，
把 noisy initial state 的来源从 block-iid sampling
升级为 trajectory-consistent noisy observation sampling。
```
