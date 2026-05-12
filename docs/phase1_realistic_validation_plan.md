# Phase-1 轻量现实导向验证方案

本文档是当前现实导向验证主线的 **Phase-1 统一方案**。

它合并并替代以下旧文档的主内容：

- [unused/phase1_comparison_matrix_legacy.md](unused/phase1_comparison_matrix_legacy.md)
- [unused/phase1_implementation_checklist_legacy.md](unused/phase1_implementation_checklist_legacy.md)

旧文档已移入 `docs/unused/` 作为归档说明。后续 Phase-1 的比较矩阵、实施顺序和完成标准以本文为准。

---

## 1. 修订动机

早期 Phase-1 设计把 `noc / oc`、五个主模型、三个训练协议、四个评估协议和完整候选 seed 集全部放入同一个 frozen matrix。这个设计严谨，但对当前阶段偏重。

补充实验 [oc_followup_results_p1_p2.md](oc_followup_results_p1_p2.md) 已经说明：

1. noisy training 不再支持把 `phnode_full` 写成 noisy all-seed winner。
2. `phnode_full` 的 noisy 收益主要来自修复 `seed46`，不是普遍降低所有 seeds 的误差。
3. `ablate_no_mass_prior` 更像稳定受益于 noisy training 的结构模型。
4. `ablate_no_lift` 与 `phnode_qforce` 在当前 noisy schedule 下没有稳定收益。

因此，Phase-1 的下一步不应直接扩成完整现实 benchmark，而应先回答一个更窄、更关键的问题：

```text
PHNODE / structured dynamics 的现实导向建模结论，
是否依赖当前 block-iid noisy IC 这一简化协议？
```

`v4-lite` 只是回答这个问题的最小协议工具。如果答案是否定的，就不应为 `v4-lite` 付出全量 sweep 成本。

---

## 2. Phase-1 的新版定位

Phase-1 现在定义为一个 **lightweight decision package**，而不是完整主实验包。

它的目标是：

1. 以最小实现接入 `v4-lite`
2. 验证协议本身正确
3. 用少量但有诊断力的模型和 seeds 判断 PHNODE 结论是否对 noisy-state protocol 敏感
4. 决定是否进入后续扩展，而不是默认进入全量矩阵

Phase-1 不直接承担以下任务：

- 证明完整 structured family 在所有 realism 轴上优于黑箱
- 证明 `phnode_full` 是当前最优实现
- 覆盖 current-representation uncertainty / actuator mismatch / parameter-regime shift
- 做 OOD maneuver 主结论
- 做真实日志 replay

---

## 3. 证据分层

Phase-1 分成两个层级。

| 层级 | 状态 | 目标 | 是否必做 |
|---|---|---|---|
| `Phase-1A` | 协议敏感性检查 | 用最小矩阵判断 PHNODE 结论是否依赖 iid noisy-IC 简化 | 必做 |
| `Phase-1B` | 条件扩展 | 只有在 `Phase-1A` 改变模型结论或归因不明确时，扩展模型、评估轴或数据层 | 条件执行 |

这一区分很重要：`Phase-1A` 的结果只用于判断当前模型结论是否被协议选择影响；`Phase-1B` 才开始接近论文主结果矩阵。

### 3.1 Phase-1A 决策的环境-provenance 前提

**`Phase-1A` 决策的有效性依赖一个底层前提：seed 之间的差异反映模型 / 协议差异，而非环境-训练动态的偶然耦合**。

2026-05-12 完成的 provenance audit（详见 [docs/provenance_audit_phnode_full_clean.md](provenance_audit_phnode_full_clean.md)）确认：catalog 时代 `main/phnode_full clean seed42/46` 的训练发散（best_loss 卡在 0.27 / epoch 21，275 行 "no successful training batches"）实际上是**云端 g3_5_5 镜像 cuDNN 算法选择 + epoch 24 极端 batch 的偶然耦合事件**，与模型架构、git 代码、dataset、显式超参全部无关。同一份代码 + 同一份 dataset + 同一个 seed 在 g3_5_7 镜像（PyTorch 2.10 / CUDA 12.8 / cuDNN 91002 / L4 GPU）下重跑收敛到 best_loss=4.05e-03，与 cleanrun v1 浮点 bit-identical。

这对 Phase-1A 的影响：

1. **5-seed 协议矩阵中任意 seed 的灾难性 outlier，必须先以 environment provenance 验证再作为协议决策证据**。「同 dataset / 同 seed / 同代码但显著不同结果」是云端环境差异的红旗，不是协议比较的有效信号
2. **Phase-1A 启动前必须把 catalog 时代受影响的 run 标 `evidence_status = stale_environment_drift`**（已 sidecar 落盘于 `analysis/oc_data_catalog/evidence_status_overrides.csv`），并以 cleanrun v1 ≡ current main 为新基线
3. **Phase-1A 的每个 run 都必须落盘 `_audit_meta/code_revision.txt` 和 `_audit_meta/environment.txt`**（参见 `notebook/phase3_provenance_audit_seed46_replay.ipynb` 的示范），catalog build 时合并到 `run_inventory.csv` 的新 schema 字段中
4. **若 Phase-1A 任一 seed 仍出现训练发散，必须先排查是否为云端环境偶然（不同 GPU、不同 PyTorch 版本、cuDNN benchmark 算法切换），再做协议归因**。重训一次（在 different GPU 或 different PyTorch build 下）可以快速判定

由于 catalog 时代的 phnode_full clean seed42/46 fragility 已确认为环境偶然、不复现于 current main，**§7.3 / §7.5 (EXPERIMENT_PROGRESS_TRACKER) 不再可作为 Phase-1A 决策的「待解释 fragility」**。Phase-1A 决策矩阵应以 cleanrun v1 5-seed 收敛基线（5-seed mean of 60s clean pos_err_median = 0.6767 m）为出发点。

---

## 4. Phase-1A：最小决策矩阵

### 4.1 数据层

`Phase-1A` 只使用：

- `oc + known-current surrogate`

理由：

- 当前主要证据缺口出现在 ocean-current 场景
- 已有 clean/noisy follow-up 结果都集中在 `oc`
- `v4-lite` 的关键价值是检查模型结论对 noisy-state protocol 的敏感性，而不是重新证明 `noc` 基线

`noc` 在 `Phase-1A` 中最多作为代码 sanity check，不进入决策主表。

### 4.2 模型集

`Phase-1A` 固定三个模型：

| 模型 | 角色 | 选择理由 |
|---|---|---|
| `phnode_full` | 主模型与 seed-fragility 诊断对象 | 需要检查协议变化是否只修复 `seed42/46` 这类坏 seed |
| `ablate_no_mass_prior` | 稳定受益结构对照 | 当前 noisy follow-up 下最像稳定 regularization 受益者 |
| `ablate_no_lift` | clean 强结构对照 | clean 下强且稳，但 noisy 下有 `seed44` 异常，可检验 `v4-lite` 是否伤害已有强结构 |

`phnode_qforce`、`se3_momentum_blackbox`、`se3_accel_blackbox` 不进入 `Phase-1A`。它们只在 `Phase-1B` 条件扩展中加入。

### 4.3 训练协议

`Phase-1A` 比较三类训练协议：

| 训练协议 | 执行方式 | 说明 |
|---|---|---|
| `clean` | 复用现有 clean checkpoints | 不因 `v4-lite` 重训，除非发现缺口 |
| `iid_noisy_ic` | 复用现有 noisy checkpoints | 当前主线 noisy-IC 对照 |
| `v4_lite` | 新增训练 | 只新增这一条训练协议 |

原则是：已有 clean 和 iid noisy-IC 结果能复用就复用，Phase-1 的新增计算主要花在 `v4-lite` 上。

### 4.4 评估协议

`Phase-1A` 的必做评估为：

- `clean eval`
- `iid noisy eval`
- `v4-lite noisy eval`

`heading bias` 与 `degraded_eval` 暂不作为决策门控必需项。若实现成本很低，可以作为诊断输出，但不能因为缺少它们而阻塞 `Phase-1A` 结论。

### 4.5 seed 策略

`Phase-1A` 分两步跑。

#### Smoke seeds

先用：

```text
42, 44, 46
```

理由：

- `42`：`phnode_full` 在 clean/noisy 下都暴露核心困难
- `44`：`ablate_no_lift` 在 noisy 下有明显异常
- `46`：`phnode_full` 被 iid noisy training 显著修复的典型 seed

Smoke 只回答协议是否能正确跑通，不直接写研究结论。

#### Decision seeds

协议 smoke 通过后，扩到：

```text
42, 43, 44, 45, 46
```

这五个 seeds 才能作为 `Phase-1A` 的决策证据。

不再默认加入 `47`。理由是：

- `42` 和 `46` 已覆盖 `phnode_full` 最关键的 catastrophic failure / 修复模式
- `44` 覆盖 `ablate_no_lift` 在 noisy training 下的新异常
- `43` 和 `45` 提供普通稳定簇参照
- 如果这五个 seeds 仍无法说明协议变化是否影响模型结论，加入 `47` 也很难根本改变判断，只会增加计算成本

### 4.6 scenario 与 horizon

保持现有 benchmark 合同：

- scenario: `PRBS`, `CHIRP`, `OU`
- horizon: `10s`, `30s`, `60s`

headline 仍然使用：

- `60s final position error median`
- `completion@60s`

### 4.7 必报切片

`Phase-1A` 至少输出：

- all-seed aggregate
- by-seed delta
- by-scenario breakdown
- by-horizon summary
- clean replay cost
- clean-to-noisy degradation

缺少 by-seed delta 时，不允许判断协议变化是否只是修复单个坏 seed。

---

## 5. Phase-1A 决策规则

`Phase-1A` 结束时，只允许得出以下四类结论。

### 5.1 采用为后续主评估协议

只有同时满足以下条件时，才把 `v4-lite` 作为后续主评估协议之一：

1. 相对 iid noisy-IC 有稳定 aggregate 改善，或显著改变模型排序
2. 改善不是主要来自单个 catastrophic seed
3. clean replay cost 可接受
4. 至少在两个模型或多个 scenario 上可见一致趋势

### 5.2 保留为补充协议

如果 `v4-lite` 与 iid noisy-IC 接近，且没有改变模型排序，则结论应是：

```text
trajectory-consistent noisy IC 增强了协议真实性，
但没有实质改变 PHNODE / structured dynamics 的当前实验结论。
```

此时不做全量 Phase-1B，`v4-lite` 保留为 appendix 或诊断协议。

### 5.3 仅视为 seed-stabilization 工具

如果收益主要来自修复 `seed42`、`seed46` 或其他单个坏 seed，则结论应收紧为：

```text
协议变化主要缓解特定训练 failure mode，
尚不能写成普适性的 realism/robustness 优势。
```

### 5.4 暂停协议扩展

如果 `v4-lite` 明显增加 clean replay cost，或对三个核心模型普遍退化，则应暂停全量扩展。此时优先检查：

- noise budget 是否过重
- warmup/ramp/mix schedule 是否与 trajectory-consistent noise 不匹配
- ODE-space 状态语义是否实现错误

---

## 6. Phase-1B：条件扩展矩阵

只有在 `Phase-1A` 通过决策门控，或结果具有科学上必须澄清的歧义时，才执行 `Phase-1B`。

### 6.1 可加入的模型

按优先级加入：

1. `phnode_qforce`
   - 用于判断协议变化是否改变 clean winner 的相对地位
2. `se3_accel_blackbox`
   - 用作非结构化神经动力学强 baseline
3. `se3_momentum_blackbox`
   - 用作弱结构/动量语义 baseline

不建议在 `Phase-1B` 一开始加入 `blackbox_fullstate`。它在既有结果中已经非常弱，更适合 appendix。

### 6.2 可加入的评估轴

按优先级加入：

1. `heading bias`
2. `degraded_eval`

仍不加入：

- current-representation uncertainty
- actuator mismatch
- mass / damping mismatch
- OOD maneuver

这些属于后续 `P3`。

### 6.3 可加入的数据层

`noc` 只在以下情况下进入：

- 需要证明 `v4-lite` 接口不依赖 ocean-current 特定状态
- 论文需要一个无海流 sanity baseline

否则 `Phase-1B` 仍以 `oc + known-current surrogate` 为主。

---

## 7. 实施工作包

### WP1. 协议标识与配置

需要让 run config 和 summary 明确区分：

- `clean`
- `iid_noisy_ic`
- `v4_lite`

不得把 `iid noisy-IC` 与 `v4-lite` 混写为同一种 noisy training。

### WP2. Trajectory-consistent noise builder

实现要求：

- 以 trajectory 为噪声生成单位
- 同一 epoch 内，同一 trajectory 的所有 block 共享同一 noisy observation realization
- 不同 epoch 可以重新采样
- target 始终为 clean truth
- backbone 不变，不引入 history encoder 或 observer

### WP3. 训练路径接入

训练器需要支持：

- 选择 `v4_lite` 作为 noise protocol
- 与现有 clean / iid noisy-IC 路径共存
- 在调试模式下检查同一 trajectory 内 block 的 noise realization 一致性

### WP4. 评估路径接入

评估需要支持：

- held-out evaluation 的 `v4-lite noisy eval`
- rollout benchmark 的 `v4-lite noisy eval`
- 输出目录和 summary 中保留 protocol 标识

`heading bias` 与 `v4-lite` 的叠加可以延后到 `Phase-1B`。

### WP5. Reporting contract

summary/report 至少输出：

- protocol label
- model
- seed
- scenario
- horizon
- final position error
- completion
- clean replay cost
- clean-to-noisy delta

推荐产物：

- `phase1a_decision_summary.csv`
- `phase1a_by_seed.csv`
- `phase1a_by_scenario.csv`
- `phase1a_degradation.csv`

文件名可按实现调整，但字段含义必须稳定。

### WP6. Smoke 与决策实验

执行顺序：

1. 单模型 `phnode_full` smoke
2. 三模型、三 seed smoke
3. 三模型、五 seed decision run
4. 写出 `Phase-1A` 决策简报
5. 决定是否进入 `Phase-1B`

禁止跳过 smoke 直接跑全量。

---

## 8. 完成标准

`Phase-1A` 完成需要同时满足：

1. `v4-lite` 协议实现通过一致性验证
2. 三个核心模型完成五 seed matched comparison
3. clean / iid noisy / `v4-lite` 三类评估可统一汇总
4. by-seed、by-scenario、by-horizon 结果齐全
5. 能明确判断 PHNODE 结论是否对 noisy-state protocol 敏感，以及是否需要进入 `Phase-1B`

`Phase-1B` 不再是 Phase-1 完成的硬前提。

---

## 9. 论文表述边界

如果只完成 `Phase-1A`，可以写：

```text
我们验证了 trajectory-consistent noisy IC protocol
是否相对 block-iid noisy IC 改变 PHNODE / structured dynamics 的 pure dynamics benchmark 结论。
```

不能直接写：

```text
结构化模型在所有现实导向条件下优于非结构化模型。
```

只有当 `Phase-1B` 加入非结构化 baseline 并通过 matched comparison 后，才可以把结论升级到 family-level robustness。

---

## 10. 一句话摘要

新版 Phase-1 的核心策略是：

```text
先用 OC-only、三模型、关键 seed 到五 seed 的轻量决策矩阵，
检查 PHNODE 结论是否对 noisy-state protocol 敏感；
只有模型结论被改变或归因仍不清楚时，才扩展到更多模型、heading bias、degraded eval 或 NOC。
```
