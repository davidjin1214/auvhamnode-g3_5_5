# OC 结果选择策略

本文档定义 `analysis/oc_data_catalog/` 第二阶段中的“默认引用规则”。

目标不是删除原始结果，也不是提前做模型分析，而是明确：

- 原始底表全部保留
- 哪些 rollout 记录默认可用于后续查阅、画图和论文表格
- 哪些 rollout 只应保留作诊断或补充用途

---

## 1. 基本原则

### 1.1 原始数据永远全部保留

以下表中的原始记录不做删除：

- `run_inventory.csv`
- `file_inventory.csv`
- `training_history_long.csv`
- `block_eval_long.csv`
- `heldout_eval_long.csv`
- `rollout_summary_long.csv`
- `rollout_outcomes_long.csv`

第二阶段只是在这些原始表之上增加：

- `run_annotations.csv`
- `rollout_run_registry.csv`

因此，“默认引用”不等于“唯一保留”。

### 1.2 默认引用规则按 `(run_uid, eval_profile)` 生效

同一个 `run_uid` 下可能存在多个 rollout run，例如：

- 原始 `resampled_*`
- `p12_matched_*`
- `heldout_*`
- `probe*`

默认选择时，不在整个 run 上只选一个 rollout，而是在每个 `eval_profile` 下分别选择。

---

## 2. rollout 用途分类

`rollout_run_registry.csv` 使用以下 `rollout_purpose`：

### 2.1 `primary`

表示正式 benchmark rollout。

典型来源：

- `resampled_*`
- 来自正式 clean / noisy suite 的标准 rollout
- 来自 `P1-1` follow-up suite 的正式 rollout

这类记录默认允许参与后续正式引用。

### 2.2 `matched_followup`

表示为 clean-vs-noisy 严格对照而补跑的 matched rollout。

当前主要对应：

- `p12_matched_*`

这类记录默认允许参与正式引用，但优先级低于 `primary`。

### 2.3 `legacy_heldout`

表示历史 `heldout_traj*` rollout。

这类记录保留用于回溯和兼容，但不作为主结果默认来源。

### 2.4 `probe`

表示 probe / 临时诊断 rollout。

这类记录保留用于排查问题，但不进入默认引用。

### 2.5 `smoke`

表示 smoke validation suite 中的 rollout。

这类记录只用于流程验证，不进入默认图表和论文表格。

---

## 3. 默认优先级

`rollout_run_registry.csv` 中的 `selection_priority` 当前规则为：

- `primary`: `100`
- `matched_followup`: `80`
- `legacy_heldout`: `30`
- `probe`: `20`
- `smoke`: `10`

同时，只有以下类型进入 `is_selection_eligible = 1`：

- `primary`
- `matched_followup`

以下类型始终为 `is_selection_eligible = 0`：

- `legacy_heldout`
- `probe`
- `smoke`

---

## 4. canonical 选择规则

对每个 `(run_uid, eval_profile)`，按以下规则设置 `is_canonical`：

1. 只在 `is_selection_eligible = 1` 的记录里选。
2. 优先选择更高的 `selection_priority`。
3. 若优先级相同，则选择时间戳更新的 `rollout_run_id`。

这意味着：

- 正式 benchmark 会优先压过 matched follow-up。
- 如果某个 `eval_profile` 没有正式 benchmark，只存在 matched follow-up，那么 matched 结果会成为该 profile 的 canonical 记录。
- `smoke`、`probe`、`legacy_heldout` 永远不会变成 canonical。

---

## 5. 当前预期行为

### 5.1 clean 训练的原始 sweep

对于只做过原始 clean rollout 的 run：

- `clean` profile 通常来自原始 `resampled_*`
- 若后续补了 `p12_matched_*`，则：
  - `clean` profile 仍优先使用原始 `resampled_*`
  - `nominal_eval / degraded_eval / heading_biased_eval` 由 `p12_matched_*` 提供 canonical 记录

### 5.2 noisy 训练的正式 sweep

对于 noisy 正式 sweep：

- `clean / nominal_eval / degraded_eval / heading_biased_eval` 通常都由正式 `resampled_*` 提供 canonical 记录

### 5.3 smoke suite

`sweep_oc_main_noise_seed42_smoke` 中的 rollout：

- 保留在原始底表中
- 出现在 `rollout_run_registry.csv`
- 但不会进入 canonical 选择

### 5.4 历史 `heldout_traj*`

这类结果保留用于回溯：

- 可以查
- 可以人工引用
- 但不会进入默认图表或默认统计

---

## 6. 使用建议

如果后续任务的目标是：

- 排查实验是否存在
- 检查所有原始数据
- 手工做特殊对照

优先使用第一阶段原始长表。

如果后续任务的目标是：

- 画默认图表
- 生成论文表格
- 做一致性的对比分析

应先基于 `rollout_run_registry.csv` 过滤到 `is_canonical = 1`，再去关联 `rollout_summary_long.csv` 或 `rollout_outcomes_long.csv`。

---

## 7. 当前派生视图

当前这个策略已经落地并导出了：

- `canonical_rollout_summary_long.csv`
- `canonical_rollout_outcomes_long.csv`

另外还导出了：

- `canonical_run_inventory.csv`

这些表都只是策略落地后的派生视图，不替代原始长表。
