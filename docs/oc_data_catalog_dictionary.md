# OC Data Catalog 字段字典

本文档说明 `analysis/oc_data_catalog/` 中各张表的用途、主键、常用 join 键以及字段含义。

目标不是重复解释实验结论，而是让后续查阅、画图、脚本开发都能明确：

- 每张表负责什么
- 该从哪张表读什么
- 表之间应如何关联

---

## 1. 总体分层

当前 `oc_data_catalog` 分为三层：

### 1.1 原始底表

完整保留所有结构化结果，不做选择：

- `run_inventory.csv`
- `file_inventory.csv`
- `training_history_long.csv`
- `block_eval_long.csv`
- `heldout_eval_long.csv`
- `rollout_summary_long.csv`
- `rollout_outcomes_long.csv`

### 1.2 规则与标签层

定义实验语义和 rollout 选择规则：

- `run_annotations.csv`
- `rollout_run_registry.csv`

### 1.3 canonical 视图层

按默认选择规则筛过的结果视图：

- `canonical_run_inventory.csv`
- `canonical_rollout_summary_long.csv`
- `canonical_rollout_outcomes_long.csv`

---

## 2. 全局主键与 join 键

### 2.1 `run_uid`

格式：

- `suite_name/run_name`

作用：

- 当前 catalog 中最重要的 run 级唯一键
- 跨表关联时优先使用

适用表：

- 所有 run 级、training 级、evaluation 级和 rollout 级表

### 2.2 `rollout_run_id`

作用：

- 区分同一个 `run_uid` 下的不同 rollout 运行

典型值：

- `resampled_traj30_seed42_...`
- `p12_matched_resampled_traj30_seed42_dr_...`
- `heldout_traj28_seed42_...`
- `p12_probe2_resampled_traj30_seed42_dr_...`

### 2.3 canonical rollout 键

对 rollout 相关表，最稳妥的唯一定位键是：

- `(run_uid, rollout_run_id, eval_profile)`

canonical 视图也是按这个键从原始 rollout 长表中过滤得到。

---

## 3. 表说明

### 3.1 `run_inventory.csv`

作用：

- run 总索引
- 汇总配置、路径和核心文件存在性

建议主键：

- `run_uid`

重要字段：

- `suite_family`
  - checkpoint 顶层目录，例如 `sweep_oc_all`、`sweep_oc_all_noise`
- `suite_name`
  - 具体 suite 名
- `group`
  - `main` / `baseline` / `ablation`
- `model_type`
  - 具体模型名
- `seed`
  - 随机种子
- `run_name`
  - suite 内 run 名
- `run_uid`
  - 全局唯一 run 键
- `dataset_id`
  - 数据集短 ID
- `dataset_path`
  - 原始 checkpoint 配置里记录的数据路径
- `train_type`
  - `clean_train` / `noisy_train`
- `noise_profile_train`
  - 训练时使用的 noise profile，clean run 记为 `clean`
- `noise_reference`
  - 噪声参考模型，例如 `remus100_dr`
- `ocean_current`
  - 是否启用 ocean current 状态
- `rollout_summary_count`
  - 该 run 目前登记到的 `summary.json` 数量
- `status`
  - 当前 catalog 视角下的完整性状态

常见用途：

- 查看有哪些 run
- 按模型、seed、train_type 做筛选
- 检查文件链路是否完整

### 3.2 `file_inventory.csv`

作用：

- 文件粒度台账
- 用于检查缺失、追踪来源

建议主键：

- `(run_uid, file_type, rollout_run_id, eval_profile, path)`

重要字段：

- `scope`
  - `suite` / `run`
- `file_type`
  - 文件类别，例如 `config_json`、`training_history_pkl`、`rollout_summary_json`
- `path`
  - 仓库相对路径
- `exists`
  - `1` / `0`
- `size_bytes`
  - 文件大小
- `mtime_epoch`
  - 修改时间戳

常见用途：

- 为什么某个 run 没进某张长表
- 某个 `summary.json` 或 `training_history.pkl` 从哪来

### 3.3 `run_annotations.csv`

作用：

- 为每个 run 提供实验语义标签

建议主键：

- `run_uid`

重要字段：

- `train_data_type`
  - `clean` / `noisy`
- `experiment_bucket`
  - 例如 `clean_core`、`noisy_ablation`、`p1_1_main_extra`、`smoke`
- `is_primary_experiment`
  - 是否属于正式主实验
- `is_followup`
  - 是否属于后续补实验
- `is_smoke`
  - 是否是 smoke suite
- `notes`
  - 简短人工说明

常见用途：

- 把主实验、follow-up、smoke 明确分开

### 3.4 `rollout_run_registry.csv`

作用：

- rollout 运行索引与默认选择规则表

建议主键：

- `(run_uid, rollout_run_id, eval_profile)`

重要字段：

- `rollout_purpose`
  - `primary` / `matched_followup` / `legacy_heldout` / `probe` / `smoke`
- `selection_priority`
  - 默认引用优先级
- `is_selection_eligible`
  - 是否允许参与 canonical 选择
- `is_canonical`
  - 对当前 `(run_uid, eval_profile)` 是否是默认结果
- `source_file`
  - 对应 `summary.json`

常见用途：

- 在 raw rollout 长表上做 canonical 过滤
- 解释为什么某个 rollout 没成为默认结果

### 3.5 `training_history_long.csv`

作用：

- 训练过程曲线主表

建议主键：

- `(run_uid, epoch, metric_key)`

重要字段：

- `epoch`
- `global_step`
- `split`
  - `train` / `test` / `meta`
- `metric_key`
  - 原始完整指标名，例如 `train_total`、`test_failure_rate`、`lr`
- `metric_name`
  - 去掉 `train_` / `test_` 前缀后的名称
- `metric_value`

常见用途：

- 训练 loss 曲线
- 学习率曲线
- failure rate / success rate 曲线

注意：

- `metric_key` 比 `metric_name` 更稳，脚本过滤时优先用 `metric_key`

### 3.6 `block_eval_long.csv`

作用：

- block evaluation 的通用长表

建议主键：

- `(run_uid, eval_profile, metric_path)`

重要字段：

- `eval_profile`
  - `clean` 或 noisy-profile 名
- `metric_path`
  - 递归展开后的叶子路径，例如 `position_rmse.mean`
- `value_type`
  - `float` / `int` / `bool` / `str` / `null`
- `value_text`
  - 原始文本表示
- `value_numeric`
  - 若可数值化则填入

常见用途：

- 提取 block RMSE、SO(3) violation、noise budget 中的数值字段

### 3.7 `heldout_eval_long.csv`

作用：

- heldout evaluation 的通用长表

建议主键：

- `(run_uid, eval_profile, metric_path)`

结构和 `block_eval_long.csv` 相同。

常见用途：

- 提取 `overall.*`、`by_scenario.*`、`relative_to_clean.*`

### 3.8 `rollout_summary_long.csv`

作用：

- 所有 rollout `summary.json` 的主结果长表

建议主键：

- `(run_uid, rollout_run_id, eval_profile, scope, scenario, horizon_s, category, metric_name, stat_name)`

重要字段：

- `rollout_run_id`
- `eval_profile`
- `scope`
  - `overall` / `by_scenario`
- `scenario`
  - overall 行为空字符串；scenario 级通常为 `PRBS` / `CHIRP` / `OU`
- `horizon_s`
  - 例如 `10.0`、`30.0`、`60.0`
- `category`
  - `meta` / `metrics` / `rates`
- `metric_name`
  - 例如 `final_position_error`、`completed_to_h`
- `stat_name`
  - 对 `metrics` 行通常为 `mean` / `median` / `p95` / `max`；对 `rates` 行为空

常见用途：

- 读取 60s 主指标
- 读取 completion / failure rates
- 读取 scenario 分层结果

### 3.9 `rollout_outcomes_long.csv`

作用：

- 单独保存 rollout outcome 计数和比例

建议主键：

- `(run_uid, rollout_run_id, eval_profile, scope, scenario, measure_group, metric_name)`

重要字段：

- `measure_group`
  - `counts` / `rates` / `meta`
- `metric_name`
  - 例如 `completed`、`pred_divergence`

常见用途：

- 查离散失败类型分布
- 查按 scenario 分层的 rollout outcome

### 3.10 canonical 视图表

#### `canonical_run_inventory.csv`

作用：

- 仅保留存在 canonical rollout 结果的 run

结构：

- 与 `run_inventory.csv` 相同

注意：

- 这不是新的实验结果，只是去掉了没有默认 rollout 结果的 run，例如 smoke-only run

#### `canonical_rollout_summary_long.csv`

作用：

- 从 `rollout_summary_long.csv` 中筛出 `is_canonical = 1` 对应的 rollout 结果

结构：

- 与 `rollout_summary_long.csv` 相同

#### `canonical_rollout_outcomes_long.csv`

作用：

- 从 `rollout_outcomes_long.csv` 中筛出 `is_canonical = 1` 对应的 rollout 结果

结构：

- 与 `rollout_outcomes_long.csv` 相同

---

## 4. 推荐读取顺序

### 4.1 训练曲线

直接使用：

- `training_history_long.csv`

常见过滤键：

- `run_uid`
- `metric_key`
- `split`

### 4.2 默认 rollout 图表或论文结果

优先使用：

- `canonical_rollout_summary_long.csv`
- `canonical_rollout_outcomes_long.csv`

只有在需要排查或对比多个 rollout run 时，再回到：

- `rollout_run_registry.csv`
- `rollout_summary_long.csv`
- `rollout_outcomes_long.csv`

### 4.3 block / heldout 结果

直接使用：

- `block_eval_long.csv`
- `heldout_eval_long.csv`

通常按：

- `run_uid`
- `eval_profile`
- `metric_path`

过滤。

---

## 5. 一致性建议

为了避免混乱，后续脚本建议遵守三条规则：

1. run 级 join 一律优先使用 `run_uid`
2. rollout 级 join 一律优先使用 `(run_uid, rollout_run_id, eval_profile)`
3. 默认图表优先从 canonical 表读取，而不是在脚本里临时重写选择规则
