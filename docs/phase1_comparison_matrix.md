# Phase-1 Frozen Comparison Matrix

本文档冻结当前现实导向验证主线的 **Phase-1 比较矩阵**。

它的作用只有一个：

```text
在进入实现和大规模 sweep 之前，
先把 Phase-1 到底比较什么、不比较什么 固定下来。
```

本文档一旦冻结，后续实现不应再临时改动：

- 主模型名单
- 主训练组
- 主评估组
- 主数据层
- 主 seed 集
- 主指标与主表口径

若未来确需修改，应在文档中显式记录版本变更，而不是在脚本或命令层面临时漂移。

---

## 1. 文档定位

本文档是以下文档的落地补充：

- [phnode_realistic_validation_plan.md](phnode_realistic_validation_plan.md)
- [phnode_realistic_validation_execution_plan.md](phnode_realistic_validation_execution_plan.md)
- [phase1_implementation_checklist.md](phase1_implementation_checklist.md)

它回答的问题是：

```text
Phase-1 的正式比较矩阵到底是什么？
```

---

## 2. Phase-1 的目标

Phase-1 不是完整现实 benchmark。

Phase-1 的目标是构造一个**最小但足够有说服力**的执行包，用来回答以下问题：

1. 在 matched-information 条件下，structured family 是否优于非结构化模型
2. 当输入状态从 clean truth 变成 noisy navigation-like state 时，这种优势是否仍然存在
3. `v4-lite` 相对当前 iid noisy-IC 是否改变结论
4. 观察到的变化是 family-level 现象，还是只是在修复个别坏 seed

因此，Phase-1 只覆盖：

- Level A
- Level B1

不覆盖：

- current-estimation stress
- actuator mismatch
- parameter mismatch
- OOD maneuver 主结论
- 真实日志 replay

---

## 3. 冻结的比较轴

### 3.1 数据层

Phase-1 只允许以下两个数据层进入主结果：

#### `Tier-1`

- `noc`

定位：

- clean vehicle dynamics baseline

#### `Tier-2`

- `oc + known-current surrogate`

定位：

- realistic-but-controlled ocean-current setting

### 3.2 明确不进入 Phase-1 主结果的数据层

- current-estimation stress
- current bias as a standalone Tier
- 真实日志 replay

这些只能进入后续 `P3/P4`。

---

## 4. 冻结的模型层

### 4.1 主结果模型集

以下模型进入 Phase-1 主结果表：

| 模型 | 角色 | 进入主结果的原因 |
|---|---|---|
| `phnode_full` | 主模型 | 当前 full PHNODE 主叙事对象 |
| `phnode_qforce` | 强结构对照 | 检验 full pH 结构是否真的带来额外收益 |
| `ablate_no_mass_prior` | 关键消融 | 当前 evidence 中最像稳定受益型结构模型 |
| `se3_momentum_blackbox` | 弱结构对照 | 用于区分几何/动量语义与完整 pH 结构 |
| `se3_accel_blackbox` | 非结构化强 baseline | 最接近“常规神经动力学”对照 |

### 4.2 补充模型集

以下模型允许在 Phase-1 中运行，但默认不进入 headline 主结果表：

| 模型 | 角色 | 用途 |
|---|---|---|
| `ablate_no_lift` | 补充强结构对照 | 用于诊断 full 结构中 lift 相关路径是否是关键困难源 |
| `blackbox_fullstate` | 弱稳定性黑箱 | 用于补充展示纯黑箱的失稳上界 |

### 4.3 当前不纳入 Phase-1 的模型

以下模型当前不进入 Phase-1 matrix：

- `phnode_merged_force`
- `ablate_diag_damping`
- `ablate_bu_only`

原因不是它们没有研究价值，而是：

- 它们对当前主问题不是最小必要集合
- Phase-1 需要优先控制复杂度

后续如需扩展，应在 `P3` 之后再考虑。

---

## 5. 冻结的 seed 集

Phase-1 固定使用以下 seed 集：

```text
42, 43, 44, 45, 46, 47
```

选择理由：

1. 已经是当前仓库结论最充分的一组 seed
2. 能直接对照现有 clean/noisy follow-up 结果
3. 能较好暴露 catastrophic seed failure 与 regularization 是否只是修复单点异常

### 5.1 seed 规则

Phase-1 必须遵守：

- 主比较使用 matched seeds
- all-seed aggregate 是主证据
- per-seed delta 必报
- pruned / problematic-seed 结果只用于诊断

---

## 6. 冻结的训练组

Phase-1 训练组固定为三类：

| 训练组 | 协议定位 | 说明 |
|---|---|---|
| `clean` | 无输入噪声 | Level A baseline |
| `iid noisy-IC` | 当前主线对照 | block-level 独立 noisy initial condition |
| `v4-lite` | 新协议主角 | trajectory-consistent noisy initial condition |

### 6.1 训练组命名要求

后续 run config 和 report 中必须明确区分：

- `clean`
- `iid_noisy_ic`
- `v4_lite`

不得把 `iid noisy-IC` 与 `v4-lite` 混写为同一类 noisy training。

### 6.2 当前不进入 Phase-1 的训练组

- `v4-B*`
- history-aware 训练
- observer-augmented 训练

这些超出 Phase-1 的 pure dynamics 比较边界。

---

## 7. 冻结的评估组

Phase-1 评估组固定为四类：

| 评估组 | 作用 | 定位 |
|---|---|---|
| `clean eval` | 纯动力学能力基线 | Level A |
| `iid noisy eval` | 当前现实导向对照 | Level B1 baseline |
| `v4-lite noisy eval` | 新协议评估 | Level B1 candidate |
| `heading bias` | bias-type 压力测试 | Level B1 extension |

### 7.1 评估组的逻辑关系

Phase-1 的关键不是单看某个 noisy profile 下谁最好，而是看以下 paired comparison：

1. `clean train` vs `iid noisy-IC train`
2. `iid noisy-IC train` vs `v4-lite train`
3. `iid noisy eval` vs `v4-lite noisy eval`
4. `clean -> noisy` 的退化是否变小

### 7.2 当前不进入 Phase-1 的评估组

- `current mismatch`
- `current bias` 作为主线评估
- `actuator mismatch`
- `mass / damping mismatch`
- `OOD maneuver` 主结论评估

这些评估只允许在 Phase-1 之后追加。

---

## 8. 冻结的 scenario 与 horizon

### 8.1 scenario

Phase-1 默认保留当前 benchmark 的三类 scenario：

- `PRBS`
- `CHIRP`
- `OU`

原因：

- 这是当前仓库已有的稳定 benchmark 合同
- 它们已经足以支撑 by-scenario 分解

### 8.2 horizon

Phase-1 主报告固定输出以下 horizon：

- `10s`
- `30s`
- `60s`

### 8.3 可选附加 horizon

如实现方便，可在内部保留：

- `1s`
- `5s`

但它们不是 Phase-1 headline 指标。

---

## 9. 冻结的主指标与主表

### 9.1 headline 指标

Phase-1 headline 指标固定为：

- `60s final position error`
- `completion@60s`

### 9.2 必报指标

除 headline 外，Phase-1 报告至少还要包含：

- `10s / 30s / 60s final position error`
- trajectory position RMSE
- rotation geodesic error
- velocity RMSE
- clean replay cost
- clean-to-noisy degradation

### 9.3 必报切片

任何 Phase-1 主结果都必须同时提供：

- by-horizon
- by-scenario
- by-seed

若缺少其中任一切片，则该结果不能进入主结论段落。

---

## 10. 主结果表与补充结果表

### 10.1 主结果表

Phase-1 主结果表应只覆盖：

- 主结果模型集
- 两个数据层
- 三个训练组
- 四个评估组

### 10.2 补充结果表

以下内容放入补充结果表或 appendix 级摘要：

- 补充模型集
- 额外 horizon
- 单独的失败案例图
- pruned seed 诊断视角

### 10.3 禁止事项

不得把以下结果拿来替代主结果：

- 只在单一 seed 上更好的结果
- 只在单一 scenario 上更好的结果
- 去掉 problematic seed 后的 pruned aggregate

---

## 11. 输出文件合同

Phase-1 的统一输出应至少包括：

- matrix 配置记录
- run-level config
- noise protocol 标识
- by-horizon summary
- by-scenario summary
- by-seed summary
- paired degradation summary

推荐新增或固定以下逻辑产物：

- `phase1_matrix.json`
- `phase1_summary.csv`
- `phase1_by_seed.csv`
- `phase1_by_scenario.csv`
- `phase1_degradation.csv`

这里只冻结产物类型，不强行冻结最终文件名实现方式。

---

## 12. 结果解释规则

### 12.1 family-level 结论成立的条件

若要写：

```text
structured family 在现实导向条件下更可靠
```

至少应满足：

1. 主结果模型集中，structured 模型相对非结构化 baseline 有稳定优势
2. 该优势不只来自单个坏 seed
3. 该优势不只存在于单一 scenario

### 12.2 `phnode_full` 主模型结论成立的条件

若要写：

```text
`phnode_full` 是当前最优实现
```

至少应满足：

1. `phnode_full` 在主结果模型集中稳定占优
2. 不被 `phnode_qforce`、`ablate_no_mass_prior` 等强结构对照持续压制
3. 优势不是仅靠修复个别 catastrophic seed

如果这些条件不满足，结论必须收缩为：

```text
structured family 的方向成立，
但 `phnode_full` 还不是当前最优实现。
```

### 12.3 `v4-lite` 进入主线的条件

若要把 `v4-lite` 升级为后续主线协议，至少应满足：

1. 相对 iid noisy-IC 带来稳定收益，或显著改变结论排序
2. 收益不是只来自修复单个坏 seed
3. clean replay 代价可接受

否则 `v4-lite` 只能保留为补充协议。

---

## 13. 当前明确排除的内容

以下内容在 Phase-1 中被显式排除：

- 真实日志 replay 主结果
- observer/history-aware 路径
- multi-block state recovery 问题
- `current mismatch` 主结果
- actuator / mass / damping mismatch 主结果
- OOD maneuver 主结果

这些都不是 Phase-1 失败或成功的判定条件。

---

## 14. 版本与变更规则

本文档当前版本可视为：

```text
Phase-1 Matrix v1
```

如未来需要调整，应遵守：

1. 记录版本号
2. 明确变更了哪一个轴
3. 说明变更原因
4. 说明该变更是否会影响前后结果可比性

没有版本说明的矩阵变动视为不允许。

---

## 15. 一句话摘要

Phase-1 Frozen Matrix 的核心定义是：

```text
用固定的主模型集、固定的两个数据层、固定的 six-seed 集合，
比较 clean、当前 iid noisy-IC 与 `v4-lite` 三种训练协议在 clean、noisy 和 heading-bias 条件下的 matched-information 轨迹预报表现，
并以 all-seed、by-seed、by-scenario、by-horizon 的统一口径作为主证据。
```
