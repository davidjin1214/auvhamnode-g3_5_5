# OC 实验数据汇总方案

本文档描述 `oc` 实验结果目录的两阶段建设方案。

- 第一阶段：建立**完整、可追溯的原始数据目录**
- 第二阶段：在不丢失原始数据的前提下，建立**规范化的默认引用层**

这份方案的目标不是直接做模型分析，而是先把数据组织问题解决干净，确保后续查阅、绘图、论文写作和补实验都有稳定的数据基础。

---

## 1. 总体设计

整个方案分成两层：

1. 原始数据层  
   负责完整收集训练、block、heldout、rollout 等结构化结果，并保留来源路径。
2. 规范化引用层  
   负责区分主实验、follow-up、smoke、probe 等用途，并定义默认应引用哪一次 rollout。

两层必须同时存在：

- 没有原始数据层，后续无法追溯。
- 没有规范化引用层，后续很容易把不同用途的结果混在一起。

---

## 2. 第一阶段目标

第一阶段只做一件事：把当前 `oc` 相关实验的结构化原始结果统一整理到一个长期可维护的数据目录中。

第一阶段**不做**：

- 模型排序
- 指标聚合
- 论文主结论归纳
- best model 判断

第一阶段需要满足四个要求：

1. 能快速回答“有哪些 run、哪些文件、哪些文件缺失”。
2. 能保留训练过程信息，尤其是 loss、lr、failure rate 等曲线数据。
3. 能统一承载 block、heldout、rollout 三类评估结果。
4. 每一条汇总数据都能追溯到具体文件路径。

---

## 3. 第一阶段数据源

### 3.1 配置与索引类

用于说明一个 run 是什么、从哪来、属于哪类实验：

- `config.json`
- `noise_budgets.json`
- suite 级 `runs.tsv`
- suite 级 `suite_config.txt`

主要用途：

- 识别 `model_type`、`seed`、`train_type`
- 识别训练噪声配置和评估 profile
- 追踪 suite 与 run 的隶属关系

### 3.2 训练过程类

用于保留训练曲线和训练稳定性：

- `training_history.pkl`
- `training.log`

其中：

- `training_history.pkl` 是**主数据源**
- `training.log` 是**兜底和人工核查源**

当前已确认 `training_history.pkl` 中包含：

- `epoch`
- `global_step`
- `lr`
- `epoch_time`
- `train_total` / `test_total`
- `train_position` / `test_position`
- `train_rotation` / `test_rotation`
- `train_velocity` / `test_velocity`
- `train_u/v/w/p/q/r`
- `test_u/v/w/p/q/r`
- `train_actuator` / `test_actuator`
- `train_so3_reg` / `test_so3_reg`
- `train_so3_orth` / `test_so3_orth`
- `train_so3_det` / `test_so3_det`
- `train_failure_rate` / `test_failure_rate`
- `train_skipped_batches` / `test_skipped_batches`

### 3.3 短时评估类

用于保存 block 和 heldout 结果：

- `block_evaluation.json`
- `heldout_evaluation.json`

对于 noisy-profile 工作流，这两个 JSON 可能同时包含：

- `clean`
- `nominal_eval`
- `degraded_eval`
- `heading_biased_eval`

因此在汇总时必须保留 `eval_profile` 维度。

### 3.4 rollout 评估类

用于保存长时 rollout benchmark 结果：

- `rollout_benchmark/*/summary.json`
- `rollout_benchmark/*/horizon_metrics.csv`
- `rollout_benchmark/*/trajectory_metrics.csv`
- `rollout_benchmark/*/rollout_outcomes.csv`

其中：

- `summary.json` 是**主汇总源**
- 其余 csv 用于补充分析或问题排查

---

## 4. 第一阶段主源与兜底源

### 4.1 主源

- 训练过程：`training_history.pkl`
- 运行配置：`config.json`
- 噪声说明：`noise_budgets.json`
- block 评估：`block_evaluation.json`
- heldout 评估：`heldout_evaluation.json`
- rollout 汇总：`rollout_benchmark/*/summary.json`
- suite 索引：`runs.tsv`、`suite_config.txt`

### 4.2 兜底源

- 训练过程：`training.log`
- rollout 补充：`rollout_outcomes.csv`、`trajectory_metrics.csv`

### 4.3 不建议作为正式数据源的文件

以下文件适合阅读，但不应作为长期数据底表的正式来源：

- `experiment_report.md`
- `sweep_summary.txt`
- `heldout_evaluation.txt`
- `block_evaluation.txt`
- 各种 png 图像

原因是这些文件偏展示层，格式容易变化，不适合做长期稳定的数据接口。

---

## 5. 第一阶段目录与表结构

建议结果目录为：

`analysis/oc_data_catalog/`

第一阶段的核心产物如下：

```text
analysis/oc_data_catalog/
  run_inventory.csv
  file_inventory.csv
  training_history_long.csv
  block_eval_long.csv
  heldout_eval_long.csv
  rollout_summary_long.csv
  rollout_outcomes_long.csv
```

如果后续数据量显著增大，再考虑从 CSV 迁移到 Parquet。当前阶段优先使用 CSV，便于人工核对。

### 5.1 `run_inventory.csv`

作用：一行一个 run，作为总索引表。

主要回答：

- 哪些 run 属于 clean training
- 哪些 run 属于 noisy training
- 哪些模型、哪些 seeds 已经存在
- 哪些 run 的关键文件缺失

### 5.2 `file_inventory.csv`

作用：一行一个文件，记录文件存在性和追溯路径。

主要回答：

- 每个 run 下有哪些关键文件
- 哪些文件缺失
- 某条结果记录的源文件路径是什么

### 5.3 `training_history_long.csv`

作用：训练曲线主表。

一行表示：

- 一个 run
- 一个 epoch
- 一个训练过程指标

它是后续绘制：

- loss 曲线
- lr 曲线
- failure rate 曲线
- success rate 曲线

的标准底表。

### 5.4 `block_eval_long.csv`

作用：统一保存 block 评估指标。

要求：

- 兼容 clean 老格式
- 兼容 noisy profile-aware 格式
- 保留 `eval_profile`

### 5.5 `heldout_eval_long.csv`

作用：统一保存 heldout 评估指标。

要求与 `block_eval_long.csv` 一致，并允许保留 `relative_to_clean` 这类衍生路径。

### 5.6 `rollout_summary_long.csv`

作用：统一保存 rollout benchmark 的 summary 指标。

要求显式保留：

- `run_uid`
- `rollout_run_id`
- `eval_profile`
- `scope`
- `scenario`
- `horizon_s`

因为同一个 checkpoint 下可能存在多次 rollout 运行。

### 5.7 `rollout_outcomes_long.csv`

作用：保存 rollout 的离散 outcome 计数与比例。

之所以单独成表，是因为它与连续 summary 指标不是同一种结构。

---

## 6. 第一阶段边界

第一阶段只做：

- 结构化数据清点
- 结构化数据抽平
- 结构化数据归档

第一阶段不做：

- 模型排序
- 均值/方差聚合
- all-seed vs matched-seed 比较
- 主结论归纳
- 从 markdown/txt 报告中抄数字
- 从图像中反推结果

---

## 7. 第二阶段目标

第一阶段完成后，虽然已经有完整原始底表，但仍然缺少一层“可安全引用”的规范化组织。

原因是：

- 同一个 `run_uid` 下可能有多次 rollout 运行
- 这些 rollout 的用途并不相同
- 有些是正式 benchmark
- 有些是 `P1-2` matched follow-up
- 有些是 smoke / probe
- 有些是历史 `heldout_traj*`

如果直接从第一阶段长表出发画图或做统计，很容易把不同用途的数据混在一起。

因此第二阶段的目标不是继续分析模型，而是增加一层**结果选择与引用规范层**。

第二阶段主要回答三件事：

1. 哪些 run 是主实验，哪些是 follow-up，哪些是 smoke。
2. 同一个 `run_uid` 下有多个 rollout run 时，默认该引用哪一个。
3. 如何在保留全部原始结果的同时，为后续图表和论文表格提供稳定的默认视图。

---

## 8. 第二阶段产物

第二阶段建议增加以下产物：

```text
analysis/oc_data_catalog/
  run_annotations.csv
  rollout_run_registry.csv
```

并增加一份规则文档：

- `docs/oc_result_selection_policy.md`

### 8.1 `run_annotations.csv`

作用：给每个 `run_uid` 增加实验语义标签。

建议字段包括：

- `run_uid`
- `suite_name`
- `model_type`
- `seed`
- `train_type`
- `experiment_bucket`
- `train_data_type`
- `is_primary_experiment`
- `is_followup`
- `is_smoke`
- `notes`

### 8.2 `rollout_run_registry.csv`

作用：给每个 `(run_uid, rollout_run_id, eval_profile)` 建立 rollout 索引，并记录用途标签和默认选择优先级。

建议字段包括：

- `run_uid`
- `run_name`
- `suite_name`
- `rollout_run_id`
- `eval_profile`
- `rollout_purpose`
- `selection_priority`
- `is_selection_eligible`
- `is_canonical`
- `source_file`
- `notes`

### 8.3 `oc_result_selection_policy.md`

作用：把默认引用规则写成显式文档，而不是隐含在脚本中。

至少要写清楚：

- 原始底表全部保留
- `smoke` 和 `probe` 不进入默认图表
- `legacy_heldout` 不作为主结果默认来源
- 正式 benchmark 优先于 matched follow-up
- 若某个 `eval_profile` 没有正式 benchmark，则 matched follow-up 可成为默认来源

### 8.4 可选 canonical 视图表

如果第二阶段规则稳定，再考虑导出：

- `canonical_run_inventory.csv`
- `canonical_rollout_summary_long.csv`
- `canonical_rollout_outcomes_long.csv`

这些表只是按规则筛过的默认视图，不替代原始长表。

---

## 9. 第二阶段边界

第二阶段只做：

- 语义标签
- rollout 用途标注
- 默认引用规则
- 规范化视图准备

第二阶段仍然不做：

- 模型排名
- 指标聚合分析
- 论文主结论
- 预设哪类 rollout 必然代表“最好结果”

---

## 10. 推荐实施顺序

建议按下面顺序推进：

1. 建立第一阶段的原始目录
2. 检查第一阶段表结构是否覆盖 clean / noisy / follow-up / smoke
3. 增加 `run_annotations.csv`
4. 增加 `rollout_run_registry.csv`
5. 写出 `oc_result_selection_policy.md`
6. 规则稳定后，再考虑导出 canonical 视图表

这样可以保证：

- 先解决“数据是否完整”
- 再解决“默认引用哪条结果”
- 最后才进入图表和分析

---

## 11. 当前判断

对于这个项目，真正重要的不是尽快做图，而是先把数据目录建设成：

- 统一
- 可追溯
- 可重建
- 不混淆不同用途实验

需要特别坚持的四点是：

- `training_history.pkl` 必须进入正式汇总体系
- `training.log` 只作为兜底和核查来源
- profile-aware 的 block / heldout / rollout 必须保留 `eval_profile`
- 所有结果表都必须保留 `source_file` 与稳定 join key

只要这一步做扎实，后面的训练曲线、箱线图、seed strip plot、论文表格都会简单很多，而且不需要再回 checkpoint 目录反复翻文件。
