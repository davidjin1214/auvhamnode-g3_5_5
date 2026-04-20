# Phase-1 实施清单

本文档给出当前研究计划下的 **Phase-1 实施清单**。

Phase-1 的目标不是完成全部 realism 轴，而是交付一个稳定、可比较、可解释的首个执行包：

- Level A 基线
- Level B1 主线
- `v4-lite` 协议接入
- by-seed / by-scenario / by-horizon 的标准输出

本文档是 checklist，不是研究论证。

---

## 1. Phase-1 范围冻结

### 1.1 数据层

只做：

- `noc`
- `oc + known-current surrogate`

不做：

- `current-estimation stress`
- 真实日志 replay

### 1.2 模型层

主结果模型集：

- `phnode_full`
- `phnode_qforce`
- `ablate_no_mass_prior`
- `se3_momentum_blackbox`
- `se3_accel_blackbox`

补充模型集：

- `ablate_no_lift`
- `blackbox_fullstate`

### 1.3 seed

固定：

```text
42, 43, 44, 45, 46, 47
```

### 1.4 训练组

- `clean`
- `iid noisy-IC`
- `v4-lite`

### 1.5 评估组

- `clean`
- `iid noisy eval`
- `v4-lite noisy eval`
- `heading bias`

---

## 2. 文件级工作包

### 2.1 需要新增的文档

- [phnode_realistic_validation_execution_plan.md](phnode_realistic_validation_execution_plan.md)
- [phase1_comparison_matrix.md](phase1_comparison_matrix.md)
- [v4_lite_protocol_spec.md](v4_lite_protocol_spec.md)
- 本文档

### 2.2 优先修改的代码文件

- `train_utils.py`
- `train_auv_hamnode.py`
- `evaluate_rollout_benchmark.py`
- `rollout_benchmark_engine.py`
- `scripts/train_all_models_noise_profile.sh`
- `scripts/eval_all_models_noise_profile.sh`
- `scripts/summarize_sweep.py`
- `scripts/build_experiment_report.py`

### 2.3 如需新增代码文件，建议优先考虑

- `noise_protocols.py`
  放 trajectory-consistent noise synthesis

如果不单独拆文件，则至少保证 `train_utils.py` 中协议相关代码有清晰边界。

---

## 3. 工作流 A：比较矩阵冻结

### A1. 冻结 Phase-1 comparison matrix

需要确认并固化：

- 主模型集
- 补充模型集
- 数据层
- 训练组
- 评估组
- 主指标

验收标准：

- 存在一个明确的 matrix 文档或常量定义，推荐固定为
  [phase1_comparison_matrix.md](phase1_comparison_matrix.md)
- 后续 sweep 不再临时改模型名单

### A2. 冻结主指标与主表

Phase-1 主表至少应固定输出：

- `10s / 30s / 60s final position error`
- `completion`
- clean-to-noisy degradation
- by-seed delta

验收标准：

- 报告脚本能稳定产出统一表头

---

## 4. 工作流 B：`v4-lite` 协议实现

### B1. 新增协议标识

需要支持至少三类 noise protocol：

- `clean`
- `iid_noisy_ic`
- `v4_lite`

验收标准：

- run config 中可明确区分协议类型
- summary / report 不会把 `iid` 与 `v4-lite` 混写

### B2. 实现 trajectory-consistent noise builder

要求：

- 以 trajectory 为基本噪声生成单位
- 同一 trajectory 内全部 block 共享同一 noisy observation realization
- 训练与评估都可调用

验收标准：

- 可对同一 trajectory 的多个 block 做一致性检查
- 给定固定 seed 时结果可重复

### B3. 保持 clean target 不变

要求：

- 无论 `iid` 还是 `v4-lite`
- target 一律是 clean truth

验收标准：

- 训练路径和评估路径中均无 noisy target 分支

### B4. 保持 pure dynamics backbone 不变

要求：

- 不引入 history encoder
- 不引入 state estimator
- 不把训练改写成 multi-block rollout

验收标准：

- `AUVHamNODE.py` 与 baseline 主干无需为 `v4-lite` 新增结构路径

---

## 5. 工作流 C：训练路径接入

### C1. 训练阶段支持 `v4-lite`

要求：

- 当前 trainer 能选择 `v4-lite` 作为 noise protocol
- 不破坏 clean 与 iid noisy-IC 路径

验收标准：

- 单模型 smoke 能跑通三种协议

### C2. epoch 内 trajectory consistency

要求：

- 同一 epoch 内，同一 trajectory 的 noisy observation 不重复采样

验收标准：

- 有调试模式或日志可检查这一点

### C3. epoch 间可重采样

要求：

- 不同 epoch 可重新生成 trajectory-level realization

验收标准：

- 文档和代码中都明确这一约定

---

## 6. 工作流 D：评估路径接入

### D1. held-out evaluation 支持 `v4-lite`

要求：

- held-out block/trajectory 路径可使用 `v4-lite`

验收标准：

- `clean / iid / v4-lite / heading bias` 可在 held-out 上输出

### D2. rollout benchmark 支持 `v4-lite`

要求：

- rollout benchmark 能区分 `iid` 与 `v4-lite`

验收标准：

- result 目录与 summary 中可识别协议类型

### D3. `heading bias` 与 `v4-lite` 可叠加

要求：

- `heading bias` 在 `v4-lite` 路径下被解释为叠加 bias，而不是重新退回 block-iid

验收标准：

- 协议层级在代码和文档中一致

---

## 7. 工作流 E：结果与报告

### E1. summary 结果必须区分协议

至少区分：

- clean
- iid noisy-IC
- `v4-lite`

验收标准：

- `summarize_sweep.py` 与 `build_experiment_report.py` 输出中可直接识别协议

### E2. 必报切片

Phase-1 报告必须至少包含：

- by-horizon
- by-scenario
- by-seed
- clean replay cost

验收标准：

- 缺少任一切片时，Phase-1 不算完成

### E3. degradation 结果

必须能回答：

- clean 到 perturbed 的退化是否更小
- `v4-lite` 相对 iid 是否改变模型排序

验收标准：

- summary 中存在 paired degradation 表

---

## 8. 工作流 F：smoke 与验收实验

### F1. 单模型 smoke

建议顺序：

1. `phnode_full`
2. `ablate_no_mass_prior`

每个模型先跑：

- clean
- iid noisy-IC
- `v4-lite`

验收标准：

- 三类协议都能训练、评估、汇总

### F2. 小规模 matched comparison

建议在两个模型上先跑：

- `phnode_full`
- 一个强结构对照

使用完整 seed 集中的小子集进行验证。

验收标准：

- 能稳定输出 by-seed delta
- 能判断收益是否来自单个坏 seed

### F3. 全量 Phase-1 sweep

只有在 smoke 与小规模 comparison 均通过后，才进入全量 sweep。

验收标准：

- 主模型集全部完成
- 结果可被统一汇总

---

## 9. Phase-1 结果解释规则

Phase-1 结束后，必须按以下规则写结论。

### 9.1 如果 `v4-lite` 明显优于 iid noisy-IC

可写：

```text
trajectory-consistent noisy-state protocol 改变了 pure dynamics benchmark 的结论，
因此它值得成为后续主线协议。
```

### 9.2 如果 `v4-lite` 与 iid noisy-IC 接近

可写：

```text
协议 realism 的提升没有实质改变当前结论，
`v4-lite` 可保留为补充协议，而不是必须替代主线。
```

### 9.3 如果收益主要来自修复单个坏 seed

必须写：

```text
当前观察到的收益更像 seed-specific stabilization，
而不是普适性的 family-level robustness gain。
```

### 9.4 如果 `phnode_full` 仍不是 structured family 最优

必须写：

```text
Phase-1 支持 structured family 的方向，
但不支持把 `phnode_full` 写成当前最优实现。
```

---

## 10. Phase-1 完成标准

只有同时满足以下条件，Phase-1 才算完成：

1. comparison matrix 已冻结
2. `v4-lite` 协议已实现并通过一致性验证
3. clean / iid / `v4-lite` 三条路径均可训练与评估
4. 主模型集完成 matched-seed comparison
5. summary 能输出 by-horizon、by-scenario、by-seed、clean replay cost
6. 已能明确判断 `v4-lite` 是否值得进入主线

如果其中任一项缺失，Phase-1 只能算部分完成。

---

## 11. 当前建议的实际顺序

推荐按以下顺序执行：

1. 冻结 comparison matrix
2. 接入协议标识与 config plumbing
3. 实现 trajectory-consistent noise builder
4. 打通训练路径
5. 打通 held-out 与 rollout evaluation
6. 打通 summary / report
7. 先做 smoke，再做小规模 comparison，再做全量 sweep

这个顺序的目标是：

```text
先验证协议，
再验证训练，
最后才验证结论。
```
