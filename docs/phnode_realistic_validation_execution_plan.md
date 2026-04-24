# ph-NODE 现实导向验证执行方案

本文档是 [phnode_realistic_validation_plan.md](phnode_realistic_validation_plan.md) 的配套执行方案。

研究计划回答“要证明什么”；本文回答“按什么顺序推进，做到什么算完成”。

本文档的根本目标不是证明某个 noise protocol 更好，也不是构建完整 AUV 真实系统，而是验证：

```text
在给定相同状态来源、未来控制和扰动表示的离线条件下，
PHNODE / structured dynamics 是否能作为真实 AUV 运动建模与轨迹预报系统的 dynamics core。
```

因此，`v4-lite` 在本文中只是协议敏感性检查工具：它用于判断 PHNODE 的现实导向结论是否依赖 block-iid noisy IC 这一简化假设，而不是新的研究主角。

---

## 1. 执行原则

### 1.1 先验证模型命题，再决定协议扩展

当前最大风险不是“实验不够多”，而是过早展开过大的现实化矩阵，导致研究重心从动力学模型本体滑向协议、observer 或系统级 benchmark。

因此 Phase-1 不再默认执行完整矩阵，而是拆成：

- `Phase-1A`：轻量协议敏感性检查，判断 PHNODE 结论是否依赖 iid noisy-IC 简化
- `Phase-1B`：条件扩展，只有在 `Phase-1A` 显示结论改变或仍需澄清时才执行

### 1.2 先证明可归因差异，再增加 realism

推进顺序仍然遵守：

1. matched-information dynamics comparison
2. trajectory-consistent noisy-state protocol sensitivity check
3. 必要的 heading bias / degraded eval 状态误差扩展
4. 模型本体优先的 OOD / current-representation / parameter-regime 扩展
5. 真实日志 replay

不得一开始把多个 realism 轴混在同一组主结果里。

### 1.3 all-seed 是主证据，by-seed 是判读底线

所有关键结论都必须保留：

- matched-seed all-seed aggregate
- per-seed delta
- by-scenario breakdown
- clean replay cost

如果某个收益主要来自修复单个 catastrophic seed，应在阶段结论中直接写明。

---

## 2. 总体阶段划分

新版执行路线如下。

| 阶段 | 目标 | 产出 |
|---|---|---|
| `P0` | 冻结轻量 Phase-1 方案与判断规则 | Phase-1A/1B plan |
| `P1` | 用 `v4-lite` 检查模型结论对 noisy-state 协议的敏感性 | model-evidence decision brief |
| `P2` | 条件执行 Phase-1B，补强 family-level 对照 | extended matched comparison |
| `P3` | 扩展模型本体相关 realism 轴 | model-centric robustness package |
| `P4` | 视资源补真实日志 replay | offline real-log package |

当前主线是 `P0-P1`。`P2` 不是默认必做项，而是由 `P1` 的结果触发。

---

## 3. 阶段 `P0`：冻结轻量 Phase-1 方案

### 3.1 目标

`P0` 的目标是把 Phase-1 从旧版完整矩阵改为轻量决策流程，并冻结以下内容：

- `Phase-1A` 的数据层、模型集、seed 策略和评估协议
- `Phase-1B` 的触发条件
- `v4-lite` 是否改变模型结论的判断规则
- Phase-1 的输出合同

### 3.2 权威文档

`P0` 的权威文档是：

- [phase1_realistic_validation_plan.md](phase1_realistic_validation_plan.md)

以下旧文档已归档到 `docs/unused/`，不再作为当前入口：

- [unused/phase1_comparison_matrix_legacy.md](unused/phase1_comparison_matrix_legacy.md)
- [unused/phase1_implementation_checklist_legacy.md](unused/phase1_implementation_checklist_legacy.md)

### 3.3 Phase-1A 冻结对象

#### 数据层

`Phase-1A` 只做：

- `oc + known-current surrogate`

`noc` 只允许作为代码 sanity check，不进入 `Phase-1A` 主决策表。

#### 模型集

`Phase-1A` 固定三个模型：

- `phnode_full`
- `ablate_no_mass_prior`
- `ablate_no_lift`

选择逻辑：

- `phnode_full`：主模型与 seed-fragility 诊断对象
- `ablate_no_mass_prior`：当前 noisy training 下最稳定受益的结构模型
- `ablate_no_lift`：clean 下强且稳、noisy 下出现新异常的关键对照

#### seed 集

先做 smoke：

```text
42, 44, 46
```

再做决策：

```text
42, 43, 44, 45, 46
```

#### 训练组

`Phase-1A` 比较：

- `clean`
- `iid_noisy_ic`
- `v4_lite`

原则是复用现有 clean / iid noisy-IC checkpoints，新增计算集中在 `v4_lite`。

#### 评估组

`Phase-1A` 必做：

- `clean eval`
- `iid noisy eval`
- `v4-lite noisy eval`

`heading bias` 与 `degraded_eval` 不再是 `Phase-1A` 的硬前提。

### 3.4 停机条件

如果仍无法接受轻量 Phase-1 的范围，就不应进入实现。否则很容易重新滑回“先跑完整矩阵再解释”的低效路径。

---

## 4. 阶段 `P1`：`v4-lite` 实现与 Phase-1A 协议敏感性检查

### 4.1 目标

`P1` 是当前最重要的执行阶段。它要回答：

```text
在不改变模型 backbone、target 与 rollout 任务的前提下，
把 noisy IC 从 block-iid 改为 trajectory-consistent，
是否实质改变 PHNODE / structured dynamics 的建模结论？
```

换句话说，`P1` 不以 `v4-lite` 本身为研究对象。它只检查当前关于 PHNODE 真实场景建模能力的证据，是否对 noisy-state 协议选择敏感。

### 4.2 不做什么

`P1` 不做：

- history encoder
- observer-augmented dynamics
- multi-block state recovery
- current-representation uncertainty
- actuator mismatch
- mass / damping mismatch
- OOD maneuver 主结论
- 真实日志 replay

### 4.3 工作包

#### WP1. 协议实现

实现 `v4-lite` 训练与评估路径。要求：

- 同一 trajectory 内 block 使用同一 noisy observation realization
- `v4-lite` 与 iid noisy-IC 只在 noise source 上不同
- target 始终为 clean truth
- `AUVHamNODE.py` 与 baseline backbone 不因 `v4-lite` 增加新结构路径

#### WP2. 协议验证

至少验证：

- 固定 seed、epoch、trajectory id 和 profile 时 noisy observation 可复现
- 同一 epoch 内同一 trajectory 不重复采样互不相关的 realization
- 不同 trajectory 的 realization 相互独立
- clean / iid noisy-IC 路径没有被破坏

#### WP3. Smoke

先用 `phnode_full` 和 smoke seeds 跑通：

- `clean`
- `iid_noisy_ic`
- `v4_lite`

然后扩到三模型 smoke：

- `phnode_full`
- `ablate_no_mass_prior`
- `ablate_no_lift`

Smoke 只检查协议、训练、评估和 summary，不写研究结论。

#### WP4. 五 seed 决策实验

对三模型完成五 seed matched comparison：

```text
42, 43, 44, 45, 46
```

不再默认纳入 `47`。`42/44/46` 覆盖已知诊断 seed，`43/45` 提供普通稳定簇参照；若这五个 seeds 无法支撑判断，额外普通 seed 也不太可能改变结论。

主输出：

- all-seed aggregate
- by-seed delta
- by-scenario breakdown
- by-horizon summary
- clean replay cost
- clean-to-noisy degradation

#### WP5. 决策简报

`P1` 结束时必须写出简短结论，判断 noisy-state protocol 变化对模型结论的影响属于以下哪类：

1. 不改变 PHNODE 结论，说明当前 iid noisy-IC 证据已足够稳健
2. 改变模型排序或退化规律，说明 PHNODE 现实导向结论对 noisy-state protocol 敏感
3. 主要修复个别坏 seed，应表述为 seed-stabilization，而不是普适 realism 优势
4. clean replay cost 或整体退化过大，应暂停协议扩展并检查 noise budget / schedule

### 4.4 进入 `P2` 的条件

只有满足以下任一条件，才进入 `P2`：

1. `v4-lite` 显著改变模型排序或 clean-to-noisy 退化规律
2. `v4-lite` 暴露出与 iid noisy-IC 不同的 seed failure / stabilization 模式
3. 结果存在必须用额外 baseline 或 bias profile 澄清的模型归因歧义

如果 `v4-lite` 与 iid noisy-IC 接近，且没有改变结论，则不进入 `P2`。

---

## 5. 阶段 `P2`：Phase-1B 条件扩展

### 5.1 目标

`P2` 只在 `P1` 触发后执行。它的目标不是重新打开完整旧矩阵，也不是继续追求更复杂协议，而是对 `P1` 中影响模型结论的关键发现做最小扩展验证。

### 5.2 可扩展模型

按优先级加入：

1. `phnode_qforce`
2. `se3_accel_blackbox`
3. `se3_momentum_blackbox`

解释：

- `phnode_qforce` 用于检查 clean winner 是否仍压制 `v4-lite` 下的 full/ablation 结构
- `se3_accel_blackbox` 用于补足非结构化神经动力学 baseline
- `se3_momentum_blackbox` 用于区分几何/动量语义与完整 PH 结构

`blackbox_fullstate` 默认不进入 `P2` 主表，除非论文需要展示纯黑箱失稳上界。

### 5.3 可扩展评估

按优先级加入：

1. `heading bias`
2. `degraded_eval`

仍然不加入：

- current-representation uncertainty
- actuator mismatch
- parameter mismatch
- OOD maneuver

这些留给 `P3`。

### 5.4 可扩展数据层

`noc` 只有在以下情况下加入：

- 需要验证 `v4-lite` 接口不依赖 ocean-current 状态
- 论文叙事需要无海流 sanity baseline

否则 `P2` 仍以 `oc + known-current surrogate` 为主。

### 5.5 `P2` 退出条件

`P2` 完成后，应能回答：

1. `v4-lite` 的收益是否仍存在于新增模型或新增评估 profile
2. structured family 结论是否可以从三模型诊断扩展到更广模型集
3. `phnode_full` 是否仍不能作为当前最佳实现
4. 后续是否值得进入模型本体相关的 Level B2 扩展

---

## 6. 阶段 `P3`：模型本体相关 Level B2 扩展

### 6.1 目标

`P3` 只在 Phase-1 已经形成清晰结论后执行。它研究更强 realism 轴，但仍必须优先服务于动力学模型本体的验证，而不是转向完整系统建模。

### 6.2 建议顺序

推荐顺序：

1. `control / maneuver OOD`
2. `current-representation uncertainty`
3. `vehicle-parameter regime shift`
4. `actuator mismatch`

这个顺序的理由是：

- control / maneuver OOD 最直接检验动力学泛化，仍属于模型本体问题
- current uncertainty 应写成 current representation 的误差，而不是完整 current estimator 问题
- vehicle-parameter shift 检验物理结构先验是否带来跨参数泛化
- actuator mismatch 容易引入执行器/输入通道建模问题，应放在后面

### 6.3 每个轴的进入要求

每增加一个 realism 轴，都必须先写清：

- 它要回答的模型本体问题是什么
- 与前一层 realism 的差别是什么
- 主指标是什么
- 若结果为正/负，各自意味着什么

如果做不到，就不应把该轴并入主结果。

---

## 7. 阶段 `P4`：真实日志离线 replay

### 7.1 目标

为“更适合作为真实 dynamics core”提供外部证据。它仍然是离线 dynamics replay，不是完整系统部署验证。

### 7.2 进入条件

只有在以下条件同时满足时才值得推进：

1. Phase-1 已形成清晰结论
2. Level B 至少一个 realism 轴表现出稳定规律
3. 可获得一致的离线状态与未来控制日志

### 7.3 输出目标

真实日志 replay 的价值不是重新做完整 benchmark，而是验证：

```text
前面在仿真和现实导向离线条件下观察到的相对优势，
是否在真实日志中仍然有迹可循。
```

---

## 8. 文档依赖关系

当前文档关系如下：

- [phnode_realistic_validation_plan.md](phnode_realistic_validation_plan.md)
  定义研究问题、证据层级和结论强度
- [phase1_realistic_validation_plan.md](phase1_realistic_validation_plan.md)
  定义新版轻量 Phase-1 的矩阵、实施工作包和决策规则
- [v4_lite_protocol_spec.md](v4_lite_protocol_spec.md)
  定义 `v4-lite` 的协议合同、边界和验收标准
- [noise_design_v4_lite_traj_consistent_ic.md](noise_design_v4_lite_traj_consistent_ic.md)
  定义 trajectory-consistent noisy IC 的研究动机和噪声语义

旧版归档：

- [unused/phase1_comparison_matrix_legacy.md](unused/phase1_comparison_matrix_legacy.md)
- [unused/phase1_implementation_checklist_legacy.md](unused/phase1_implementation_checklist_legacy.md)

---

## 9. 当前立即下一步

建议下一步按以下顺序执行：

1. 按 [phase1_realistic_validation_plan.md](phase1_realistic_validation_plan.md) 固化 `Phase-1A`
2. 完成 `v4-lite` 协议规格复核
3. 实现 protocol label、trajectory-consistent noise builder 和最小 reporting contract
4. 先跑 `phnode_full` smoke
5. 再跑三模型 smoke
6. 最后跑三模型五 seed decision run

不要跳过 smoke 直接进入多模型全量 sweep。

---

## 10. 一句话执行摘要

新版执行路线是：

```text
先用 OC-only、三模型、关键 seed 到五 seed 的轻量 Phase-1A，
检查 PHNODE 结论是否对 noisy-state protocol 敏感；
只有当模型结论被改变或归因仍不清楚时，才进入 Phase-1B，
之后再考虑模型本体相关 OOD、current representation、参数漂移与真实日志 replay。
```
