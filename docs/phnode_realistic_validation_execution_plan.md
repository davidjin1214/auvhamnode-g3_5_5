# ph-NODE 现实导向验证执行方案

本文档是
[phnode_realistic_validation_plan.md](phnode_realistic_validation_plan.md)
的配套执行方案。

前者回答“研究要证明什么”；本文回答“下一步按什么顺序推进，做到什么算完成”。

本文档仍然不是逐命令 runbook。它的作用是：

- 把研究计划翻译成阶段、依赖与里程碑
- 约束每一阶段的范围，避免一次性铺开过多 realism 轴
- 明确每个阶段的输入、输出、判定规则与停机条件

---

## 1. 执行原则

本项目的执行顺序必须遵守以下三条原则。

### 1.1 先证明可归因的差异，再增加 realism

推进顺序应是：

1. 先建立 matched-information 的基线比较
2. 再引入 realistic noisy-state
3. 再引入更强的 mismatch / OOD
4. 最后才考虑真实日志 replay

不得一开始就把所有 realism 轴混在一起。

### 1.2 先实现协议，再做大规模 sweep

像 `v4-lite` 这类协议，必须先完成：

- 协议定义
- 最小实现
- smoke validation
- 单模型 sanity check

之后才能扩到多模型、多 seed sweep。

### 1.3 all-seed 是主证据，诊断性结果不能替代主结果

后续所有执行阶段都要保留：

- matched-seed all-seed aggregate
- per-seed delta
- by-scenario breakdown

如果某阶段的主要收益只来自修复个别坏 seed，应在该阶段结束时直接写明，不得等到论文写作时再补。

---

## 2. 总体阶段划分

本执行方案建议分成五个阶段。

| 阶段 | 目标 | 产出 |
|---|---|---|
| `P0` | 冻结比较对象与主证据口径 | frozen comparison matrix |
| `P1` | 跑通 Level A 基线与 Phase-1 主实验外壳 | clean baseline package |
| `P2` | 实现并验证 `v4-lite`，完成 Level B1 | realistic noisy-state package |
| `P3` | 扩展 Level B2：mismatch / OOD | robustness extension package |
| `P4` | 视资源补真实日志 replay | offline real-log package |

其中：

- `P0-P2` 是当前主线
- `P3` 是增强证据
- `P4` 是外部证据增强，不是当前硬前提

---

## 3. 阶段 `P0`：冻结研究比较框架

### 3.1 目标

在写代码前先冻结：

- 主模型集
- 主数据层
- 主 seed 集
- 主指标
- 主结论口径

### 3.2 必须冻结的对象

#### 模型集

第一阶段主模型集建议固定为：

- `phnode_full`
- `phnode_qforce`
- `ablate_no_mass_prior`
- `se3_momentum_blackbox`
- `se3_accel_blackbox`

如资源允许，可补：

- `ablate_no_lift`
- `blackbox_fullstate`

#### 数据层

Phase-1 只覆盖：

- `noc`
- `oc + known-current surrogate`

不在这个阶段把 `current-estimation stress` 混进主结果。

#### seed 集

主实验固定使用：

```text
42, 43, 44, 45, 46, 47
```

#### 主训练组

Phase-1 只比较：

- `clean`
- `iid noisy-IC`
- `v4-lite`

#### 主评估组

Phase-1 只比较：

- `clean`
- `iid noisy-state eval`
- `v4-lite noisy-state eval`
- `heading bias`

### 3.3 交付物

`P0` 结束时应至少产出一份 frozen matrix 文档，建议固定为
[phase1_comparison_matrix.md](phase1_comparison_matrix.md)，明确：

- 哪些模型属于主结果
- 哪些模型只是补充结果
- 哪些 realism 轴纳入 Phase-1
- 哪些 realism 轴延后到 `P3`

### 3.4 停机条件

如果这一步还无法就“Phase-1 比较矩阵”达成统一，就不应进入实现阶段。

---

## 4. 阶段 `P1`：Level A 与 Phase-1 外壳

### 4.1 目标

`P1` 的目标不是做新协议，而是建立一个稳定、可重复的比较外壳，为 `v4-lite` 留出干净插槽。

### 4.2 工作内容

1. 复核当前 `clean` 与 `iid noisy-IC` 的 baseline 路径
2. 明确 `noc` 与 `oc + known-current surrogate` 的默认比较配置
3. 明确 Phase-1 的输出目录、结果文件与 summary 格式
4. 补齐 Phase-1 所需的 summary / reporting contract

### 4.3 关键问题

这一步要回答：

```text
如果未来引入 `v4-lite`，
我们将用哪一套完全相同的矩阵去和当前 iid noisy-IC 做 matched comparison？
```

### 4.4 交付物

`P1` 结束时应有：

- frozen Phase-1 matrix
- 可重复的 clean / iid noisy-IC baseline runs
- Phase-1 summary 模板
- 主结果表的字段定义

### 4.5 进入下一阶段的条件

只有在以下条件同时满足时，才进入 `P2`：

1. clean baseline 路径可重复
2. 当前 iid noisy-IC 路径的结果口径已冻结
3. summary / report 输出能支持 by-seed 和 by-scenario 对比

---

## 5. 阶段 `P2`：实现并验证 `v4-lite`

### 5.1 目标

`P2` 是当前执行方案的核心阶段。

它要回答的不是“`v4-lite` 最终是否赢”，而是更基本的问题：

```text
能否以最小、可控、可解释的方式，
把 trajectory-consistent noisy-state 协议接入当前 pure dynamics 主线？
```

### 5.2 这一阶段只做什么

`P2` 只覆盖 Level B1：

- realistic noisy-state protocol
- heading bias
- clean-to-noisy degradation analysis

`P2` 不做：

- current mismatch 主结论
- actuator mismatch 主结论
- mass / damping mismatch
- OOD maneuver 扩展
- 真实日志 replay

### 5.3 `P2` 的工作包

#### WP1. 协议实现

实现 `v4-lite` 的训练与评估协议。

#### WP2. 协议验证

验证以下事实：

- 同一 trajectory 内 block 使用同一 noisy observation realization
- `v4-lite` 与 iid noisy-IC 的区别只在 noise source，不在 target 与 backbone
- 同一 seed、同一 profile 下结果可重复

#### WP3. 单模型 smoke

先用 `phnode_full` 跑通：

- clean
- iid noisy-IC
- `v4-lite`

并完成协议级 sanity check。

#### WP4. 小规模 matched comparison

对 `phnode_full` 与一个强结构对照完成 matched-seed comparison。

#### WP5. 扩展到主模型集

在 smoke 和小规模 comparison 稳定后，再扩到 Phase-1 主模型集。

### 5.4 `P2` 的必答问题

`P2` 完成后，必须能够回答：

1. `v4-lite` 是否相对 iid noisy-IC 改变了结果排序
2. 这种变化是否稳定出现在多 seed / 多 scenario 上
3. 观察到的改善是 family-level 现象，还是只在修复单个 catastrophic seed
4. `v4-lite` 的 clean replay 代价是否可接受

### 5.5 `P2` 的交付物

至少应包括：

- `v4-lite` 协议实现
- 协议验证报告
- Phase-1 matched comparison 表
- by-seed / by-scenario / by-horizon 汇总
- 结论级简报：`v4-lite` 值不值得进入主线

### 5.6 退出判断

`P2` 结束时，只允许出现以下三种结论之一：

#### 结论 A：`v4-lite` 明显优于 iid noisy-IC

说明：

- 现实导向的 noise protocol 本身有研究价值
- 后续主线可转向 `v4-lite`

#### 结论 B：`v4-lite` 与 iid noisy-IC 接近

说明：

- 协议 realism 提升没有带来实质区别
- 后续可保留为补充协议，不必升级成主线

#### 结论 C：`v4-lite` 只修复个别坏 seed

说明：

- 它更像训练稳定化工具
- 后续叙事应从“普适鲁棒优势”收紧为“修复特定 failure mode”

---

## 6. 阶段 `P3`：扩展 Level B2

### 6.1 目标

在 `P2` 已经跑通的前提下，逐项扩展更强的 realism 轴。

### 6.2 建议顺序

不建议并行展开全部 Level B2。推荐顺序：

1. `current-estimation stress`
2. `actuator mismatch`
3. `vehicle-parameter mismatch`
4. `OOD maneuver / OOD disturbance`

### 6.3 原因

这个顺序与当前研究主线更一致：

- `oc` 问题最自然会先碰到 current uncertainty
- actuator 与 parameter mismatch 会更强地引入系统级解释变量
- OOD 定义若不够清楚，最容易变成“描述性压力测试”

### 6.4 `P3` 的要求

每增加一个 realism 轴，都必须单独回答：

- 它的科学问题是什么
- 它与前一层 realism 的差别是什么
- 它的主指标是什么
- 若结果为正/负，各自意味着什么

如果做不到这一点，就不应把该 realism 轴并入主结果。

---

## 7. 阶段 `P4`：真实日志离线 replay

### 7.1 目标

为“更适合作为真实 dynamics core”提供外部证据。

### 7.2 进入条件

只有在以下条件满足时才值得推进：

1. Level A 与 Phase-1 主线已经形成清晰结论
2. Level B 至少有一个 realism 轴已经表现出稳定规律
3. 可获得一致的离线状态与未来控制日志

### 7.3 输出目标

这一步的价值不是重新做一套完整 benchmark，而是验证：

```text
前面在仿真和现实导向离线条件下观察到的相对优势，
是否在真实日志中仍然有迹可循。
```

---

## 8. 文档依赖关系

本执行方案依赖以下文档：

- [phnode_realistic_validation_plan.md](phnode_realistic_validation_plan.md)
- [phase1_comparison_matrix.md](phase1_comparison_matrix.md)
- [v4_lite_protocol_spec.md](v4_lite_protocol_spec.md)
- [phase1_implementation_checklist.md](phase1_implementation_checklist.md)
- [noise_design_v4_lite_traj_consistent_ic.md](noise_design_v4_lite_traj_consistent_ic.md)

它们之间的关系是：

- 研究计划定义“为何做”
- frozen matrix 定义“Phase-1 具体比较什么”
- 本文档定义“何时做、先做什么”
- `v4_lite_protocol_spec` 定义“协议必须长什么样”
- `phase1_implementation_checklist` 定义“代码层面具体怎么拆任务”

---

## 9. 当前建议的立即下一步

在文档准备阶段结束后，建议立即进入以下顺序：

1. 冻结 Phase-1 comparison matrix
2. 完成 `v4-lite` 协议规格评审
3. 按 checklist 拆实现任务
4. 先做 protocol-level smoke，再做多模型 sweep

不要跳过 smoke 直接全量跑。

---

## 10. 一句话执行摘要

当前最合理的推进路线不是“直接做完整现实 benchmark”，而是：

```text
先冻结 Phase-1 比较矩阵，
再用最小、可解释的方式把 `v4-lite` 接入当前 pure dynamics 主线，
先完成 Level A + Level B1 的稳定比较，
之后再决定是否值得扩展到更强的 mismatch、OOD 与真实日志 replay。
```
