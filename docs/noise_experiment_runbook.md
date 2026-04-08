# 新版噪声方案实验运行手册

本文给出基于新版 noisy-IC 方案的标准实验运行清单，目标是减少临时查参数的成本。

配套文档：

- [`docs/noise_cli_parameter_reference.md`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/noise_cli_parameter_reference.md)
- [`docs/noise_cli_command_templates.md`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/noise_cli_command_templates.md)

---

## 1. 推荐实验结构

建议把新版噪声方案实验分成两条线：

- `OC` 主实验：验证 `nominal_eval`、`degraded_eval`、`heading_biased_eval`、`current_bias_eval`
- `NOC` 对照实验：验证不含海流状态时的鲁棒性趋势是否保持一致

建议执行顺序：

1. 先跑一个单模型 smoke test，确认配置、输出文件和 profile 行为正常。
2. 再跑 `core` sweep。
3. 最后跑 rollout benchmark、summary 和 experiment report。

---

## 2. OC 主实验

### 2.1 单模型 smoke test

```bash
conda run -n mytorch1 python train_auv_hamnode.py \
  --dataset ./data/auv_oc_traj1000_xxx.pkl \
  --model_type phnode_full \
  --noise_profile nominal_train \
  --noise_warmup_epochs 20 \
  --noise_ramp 80 \
  --noise_mix_ratio 0.5
```

默认自动评估 profile：

- block-level: `clean nominal_eval heading_biased_eval current_bias_eval`
- held-out: `clean nominal_eval degraded_eval heading_biased_eval current_bias_eval`

建议重点检查：

- `config.json`
- `noise_budgets.json`
- `heldout_evaluation.json`
- `heldout_evaluation.txt`

你应该确认：

- `noise_mix_ratio` 已按 sample-level 记录
- `heading_biased_eval` 的 budget 中存在 yaw bias
- `current_bias_eval` 的 budget 中存在 `v_c` bias
- `relative_to_clean` 已写入结果文件

### 2.2 单个 checkpoint 的 rollout benchmark

```bash
conda run -n mytorch1 python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run>/best_model.pt \
  --mode heldout \
  --noise_profiles clean nominal_eval degraded_eval heading_biased_eval current_bias_eval
```

建议重点检查：

- `rollout_benchmark_results/<run>/summary.json`
- `rollout_benchmark_results/<run>/summary.txt`

你应该确认：

- `config.noise_budget` 存在
- `current_bias_eval` 成功展开
- 每个 profile 都有独立输出目录

### 2.3 OC 主 sweep

```bash
bash scripts/train_all_models_noise_profile.sh \
  --profile oc \
  --group core
```

该命令默认使用：

- `noise_profile=nominal_train`
- `block_eval_noise_profiles=auto`
- `heldout_eval_noise_profiles=auto`

其中 `auto` 会展开为：

- block-level: `clean nominal_eval heading_biased_eval current_bias_eval`
- held-out: `clean nominal_eval degraded_eval heading_biased_eval current_bias_eval`

### 2.4 OC 主 sweep 的 rollout 评估

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/<suite>
```

该命令默认使用：

- `mode=resampled`
- `noise_profiles=auto`
- `summary_horizon=60`

其中 `auto` 会展开为：

- `clean nominal_eval degraded_eval heading_biased_eval current_bias_eval`

### 2.5 OC 实验产物

标准产物包括：

- `checkpoints/<run>/config.json`
- `checkpoints/<run>/noise_budgets.json`
- `checkpoints/<run>/heldout_evaluation.json`
- `checkpoints/<run>/heldout_evaluation.txt`
- `checkpoints/<suite>/sweep_model_metrics.csv`
- `checkpoints/<suite>/experiment_report.md`

---

## 3. NOC 对照实验

### 3.1 单模型 smoke test

```bash
conda run -n mytorch1 python train_auv_hamnode.py \
  --dataset ./data/auv_noc_traj1000_xxx.pkl \
  --model_type phnode_full \
  --noise_profile nominal_train \
  --noise_warmup_epochs 20 \
  --noise_ramp 80 \
  --noise_mix_ratio 0.5
```

默认自动评估 profile：

- block-level: `clean nominal_eval`
- held-out: `clean nominal_eval degraded_eval heading_biased_eval`

建议重点检查：

- `config.json`
- `noise_budgets.json`
- `heldout_evaluation.json`

你应该确认：

- 不会出现 `current_bias_eval`
- `heading_biased_eval` 仍然存在
- `relative_to_clean` 已写入结果文件

### 3.2 单个 checkpoint 的 rollout benchmark

```bash
conda run -n mytorch1 python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run>/best_model.pt \
  --mode heldout \
  --noise_profiles clean nominal_eval degraded_eval heading_biased_eval
```

### 3.3 NOC 对照 sweep

```bash
bash scripts/train_all_models_noise_profile.sh \
  --profile noc \
  --group core
```

该命令默认使用：

- block-level: `clean nominal_eval`
- held-out: `clean nominal_eval degraded_eval heading_biased_eval`

### 3.4 NOC 对照 sweep 的 rollout 评估

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/<suite>
```

若该 suite 对应 `noc`，脚本会自动解析并展开：

- `clean nominal_eval degraded_eval heading_biased_eval`

### 3.5 NOC 实验产物

标准产物包括：

- `checkpoints/<run>/config.json`
- `checkpoints/<run>/noise_budgets.json`
- `checkpoints/<run>/heldout_evaluation.json`
- `checkpoints/<suite>/sweep_model_metrics.csv`
- `checkpoints/<suite>/experiment_report.md`

---

## 4. 推荐比较方式

完成 `OC` 与 `NOC` 两条线后，建议至少比较以下内容：

- clean 基线性能
- `nominal_eval` 相对 clean 的退化比例
- `degraded_eval` 相对 clean 的退化比例
- `heading_biased_eval` 是否引起明显额外退化
- `current_bias_eval` 是否成为 `OC` 模型的主要脆弱点

优先读取的字段：

- `relative_to_clean.ratio_to_clean`
- `relative_to_clean.degradation_pct`
- `noise_budget`

---

## 5. 常见提醒

- `current_bias_eval` 只适用于 ocean-current 模型或 checkpoint。
- 直接调用 Python 脚本时，多值 profile 参数不要写成一个整体字符串。
- 调用顶层 Bash 包装脚本时，多值 profile 参数应写成一个带空格的字符串，由脚本内部拆分。
- `scripts/eval_all_models_noise_profile.sh` 的 `auto` 规则依赖 `suite_config.txt` 或 suite 目录名中的 `oc/noc` 信息。
- 如果只是检查链路是否打通，优先跑单模型 smoke test，而不是直接跑全 sweep。
