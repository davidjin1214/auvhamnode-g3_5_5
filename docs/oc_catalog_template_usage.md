# OC Catalog 模板使用说明

本文档给出 `scripts/oc_catalog_templates.py` 的最小使用方式。

这个脚本的定位不是做复杂分析，而是提供两类高频模板：

- 从 `training_history_long.csv` 直接画训练曲线
- 从 `canonical_rollout_summary_long.csv` 直接导出论文表格底稿

---

## 1. 训练曲线模板

### 1.1 单个 run 的 total loss 曲线

```bash
conda run -n mytorch1 python scripts/oc_catalog_templates.py \
  plot-training-curves \
  --run-uid sweep_oc_core_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_115414/main_phnode_full_seed42 \
  --metric-key train_total \
  --metric-key test_total \
  --output analysis/oc_data_catalog/examples/main_phnode_full_seed42_total_loss.png \
  --csv-output analysis/oc_data_catalog/examples/main_phnode_full_seed42_total_loss.csv
```

作用：

- 输出一张 PNG 曲线图
- 同时导出筛过的曲线数据 CSV

### 1.2 用 `global_step` 作为横轴

```bash
conda run -n mytorch1 python scripts/oc_catalog_templates.py \
  plot-training-curves \
  --run-uid sweep_oc_main_noise_nominal_train_remus100_dr_auv_oc_traj1000_blk150_s23_d0be9434_s43-44-45_20260409_101923/main_phnode_full_seed43 \
  --metric-key train_total \
  --metric-key test_total \
  --x-axis global_step \
  --output analysis/oc_data_catalog/examples/main_phnode_full_seed43_total_loss_global_step.png
```

### 1.3 画训练稳定性相关指标

例如 failure rate：

```bash
conda run -n mytorch1 python scripts/oc_catalog_templates.py \
  plot-training-curves \
  --run-uid sweep_oc_main_noise_nominal_train_remus100_dr_auv_oc_traj1000_blk150_s23_d0be9434_s43-44-45_20260409_101923/main_phnode_full_seed43 \
  --metric-key train_failure_rate \
  --metric-key test_failure_rate \
  --output analysis/oc_data_catalog/examples/main_phnode_full_seed43_failure_rate.png
```

---

## 2. canonical rollout 表格模板

### 2.1 导出 noisy-train 的 60s 主指标宽表

默认导出的指标是：

- `scope=overall`
- `scenario=""`
- `horizon_s=60.0`
- `category=metrics`
- `metric_name=final_position_error`
- `stat_name=median`

示例：

```bash
conda run -n mytorch1 python scripts/oc_catalog_templates.py \
  export-rollout-table \
  --canonical \
  --train-type noisy_train \
  --eval-profile clean \
  --eval-profile nominal_eval \
  --eval-profile degraded_eval \
  --eval-profile heading_biased_eval \
  --output analysis/oc_data_catalog/examples/noisy_train_60s_final_position_error_median.csv
```

导出结果每行一个 `run_uid`，并带有：

- 基本元信息
- 每个 profile 对应的指标值
- 每个 profile 对应的 `rollout_run_id`

### 2.2 只导出某一类模型

例如只导出 `phnode_full`：

```bash
conda run -n mytorch1 python scripts/oc_catalog_templates.py \
  export-rollout-table \
  --canonical \
  --train-type noisy_train \
  --model-type phnode_full \
  --eval-profile nominal_eval \
  --eval-profile degraded_eval \
  --eval-profile heading_biased_eval \
  --output analysis/oc_data_catalog/examples/noisy_phnode_full_60s_table.csv
```

### 2.3 导出 completion rate 之类的 rate 指标

例如 `completed_to_h`：

```bash
conda run -n mytorch1 python scripts/oc_catalog_templates.py \
  export-rollout-table \
  --canonical \
  --train-type noisy_train \
  --eval-profile nominal_eval \
  --horizon-s 60.0 \
  --category rates \
  --metric-name completed_to_h \
  --stat-name "" \
  --output analysis/oc_data_catalog/examples/noisy_train_completed_to_h.csv
```

注意：

- 对 `rates` 或 `meta` 类别，`stat_name` 一般为空字符串。

---

## 3. 何时使用 canonical 表

如果目标是：

- 画默认论文图
- 生成默认论文表格
- 做主结果对比

优先使用：

- `canonical_rollout_summary_long.csv`
- `canonical_rollout_outcomes_long.csv`

如果目标是：

- 排查为什么某个 run 有多个 rollout
- 手工比较原始 benchmark 与 matched follow-up

先查：

- `rollout_run_registry.csv`
- `rollout_summary_long.csv`
- `rollout_outcomes_long.csv`

---

## 4. 当前示例输出

当前已经实际生成过的示例文件位于：

- `analysis/oc_data_catalog/examples/main_phnode_full_seed42_total_loss.png`
- `analysis/oc_data_catalog/examples/main_phnode_full_seed42_total_loss.csv`
- `analysis/oc_data_catalog/examples/noisy_train_60s_final_position_error_median.csv`
