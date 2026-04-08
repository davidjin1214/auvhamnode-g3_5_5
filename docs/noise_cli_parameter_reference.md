# 新版噪声方案参数速查

本文整理以下 4 个入口文件与新版噪声方案直接相关的可传入参数：

- `train_auv_hamnode.py`
- `evaluate_rollout_benchmark.py`
- `scripts/train_all_models_noise_profile.sh`
- `scripts/eval_all_models_noise_profile.sh`

适用范围：

- `IC-only` noisy-initial-condition 训练主线
- `heading_biased_eval`
- `current_bias_eval`
- noisy/clean 比值与退化百分比的结果输出链路

如果你想直接按标准流程跑实验，请优先看：

- [`docs/noise_experiment_runbook.md`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/noise_experiment_runbook.md)

---

## 0. 多值参数传法

先记住这一条，避免最常见的传参错误。

### 0.1 直接调用 Python 脚本

对 `nargs="+"` 参数，直接写成多 token：

```bash
python train_auv_hamnode.py \
  --dataset ./data/xxx.pkl \
  --block_eval_noise_profiles clean nominal_eval heading_biased_eval
```

```bash
python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/xxx/best_model.pt \
  --noise_profiles clean nominal_eval degraded_eval heading_biased_eval
```

### 0.2 调用顶层 Bash 包装脚本

对多值参数，传成一个带空格的字符串即可，脚本内部会再拆开：

```bash
bash scripts/train_all_models_noise_profile.sh \
  --block-eval-noise-profiles "clean nominal_eval heading_biased_eval"
```

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --noise-profiles "clean nominal_eval degraded_eval heading_biased_eval"
```

---

## 1. `train_auv_hamnode.py`

用途：

- 单次训练
- 训练后自动执行 block-level evaluation
- 训练后自动执行 held-out trajectory evaluation
- 自动写出 `noise_budgets.json`、评估 JSON/TXT、relative-to-clean summary

### 1.1 训练与模型参数

| 参数 | 取值 | CLI 默认值 | 实际默认规则 | 说明 |
|---|---|---:|---|---|
| `--dataset` | 路径 | 必填 | 无 | 训练数据集 `.pkl` 路径 |
| `--model_type` | 任一注册模型名 | `phnode_full` | 无 | 模型结构 |
| `--run_name` | 字符串 | `None` | 自动生成 | 运行名 |
| `--save_dir` | 路径 | `./checkpoints` | 无 | 输出根目录 |
| `--batch_size` | 整数 | `None` | `noc=2048`, `oc=4096` | dataset-aware 默认 |
| `--epochs` | 整数 | `None` | `noc=200`, `oc=300` | 最大 epoch 数 |
| `--total_steps` | 整数 | `None` | `noc=7000`, `oc=5000` | 目标优化步数 |
| `--lr` | 浮点数 | `None` | `noc=5e-3`, `oc=6e-3` | 峰值学习率 |
| `--min_lr` | 浮点数 | `None` | `1e-4` | cosine decay 下限 |
| `--warmup_steps` | 整数 | `None` | `noc=300`, `oc=400` | 学习率 warmup 步数 |
| `--hidden_dim` | 整数 | `128` | 无 | 隐层宽度 |
| `--so3_reg` | 浮点数 | `1e-3` | 无 | SO(3) 正则权重 |
| `--actuator_loss_weight` | 浮点数 | `0.2` | 无 | 执行器状态监督权重 |
| `--include_depth_in_potential` | flag | `False` | 无 | 是否给 potential / force 模型加入深度上下文 |
| `--device` | 字符串 | `cuda` if available else `cpu` | 无 | 训练设备 |
| `--seed` | 整数 | `42` | 无 | 随机种子 |

### 1.2 训练期 noisy-IC 参数

这些参数只作用于训练主线，不控制 bias-type eval profile。

| 参数 | 取值 | 默认值 | 说明 |
|---|---|---:|---|
| `--noise_profile` | `clean`, `nominal_train`, `nominal_eval`, `degraded_eval` | `None` | 推荐接口。通常训练只用 `clean` 或 `nominal_train` |
| `--noise_level` | `0`, `1`, `2`, `3` | `0` | 旧接口兼容：`0=clean`, `1=nominal_train`, `2=nominal_eval`, `3=degraded_eval` |
| `--noise_scale` | 浮点数 | `1.0` | 全局噪声强度缩放 |
| `--noise_ramp` | 整数 | `100` | noisy IC 强度 ramp 长度 |
| `--noise_warmup_epochs` | 整数 | `20` | 纯 clean warmup epoch 数 |
| `--noise_mix_ratio` | `0~1` 浮点数 | `0.5` | noisy IC 的逐样本占比，不再是 batch-level 开关 |

### 1.3 自动评估 profile 参数

| 参数 | 取值 | CLI 默认值 | 实际默认规则 | 说明 |
|---|---|---:|---|---|
| `--block_eval_noise_profiles` | `clean`, `nominal_eval`, `degraded_eval`, `heading_biased_eval`, `current_bias_eval`, `all`, `none` | `None` | `NOC -> clean nominal_eval`; `OC -> clean nominal_eval heading_biased_eval current_bias_eval` | block-level 自动评估 profile 集合 |
| `--heldout_eval_noise_profiles` | 同上 | `None` | `NOC -> clean nominal_eval degraded_eval heading_biased_eval`; `OC -> clean nominal_eval degraded_eval heading_biased_eval current_bias_eval` | held-out 自动评估 profile 集合 |

限制：

- `current_bias_eval` 仅允许 ocean-current 任务使用
- `all` 会根据当前任务是 `OC` 还是 `NOC` 自动展开
- `none` 可用于跳过该阶段自动评估

### 1.4 Ocean-current / 先验相关参数

| 参数 | 取值 | CLI 默认值 | 实际默认规则 | 说明 |
|---|---|---:|---|---|
| `--ocean_current` | flag | `False` | 通常会被数据集 metadata 覆盖 | 是否启用 OC 模型接口 |
| `--dj_current_feature` | `none`, `current_body`, `total_velocity` | `None` | `noc=none`, `oc=current_body` | D/J 网络附加 current 特征 |
| `--actuation_current_feature` | `none`, `current_body`, `total_velocity` | `None` | `noc=current_body`, `oc=current_body` | B 网络附加 current 特征 |
| `--mass_init` | `none`, `remus`, `file` | `None` | `noc=remus`, `oc=remus` | 质量矩阵先验来源 |
| `--mass_init_path` | 路径 | `None` | 仅 `mass_init=file` 时需要 | 外部质量矩阵文件 |
| `--t_actuator_init` | 一个或多个浮点数 | `None` | `noc=None`, `oc=[0.1, 0.1, 1.0]` | 执行器时间常数先验 |
| `--u_act_scale` | 一个或多个浮点数 | `None` | `noc=None`, `oc=[1.0, 1.0, 0.001]` | B 网络使用的执行器尺度 |

### 1.5 新版噪声方案下的关键行为

- 训练期噪声仍是 `IC-only`
- `noise_mix_ratio` 已按 sample-level 生效
- 训练后自动评估支持 `heading_biased_eval`
- 训练后自动评估支持 `current_bias_eval`
- 评估结果会写出 `relative_to_clean`
- `relative_to_clean` 至少包含 `ratio_to_clean` 与 `degradation_pct`

---

## 2. `evaluate_rollout_benchmark.py`

用途：

- 对单个 checkpoint 跑 heldout / resampled rollout benchmark
- 支持 clean 与多个 noisy initialization profile
- 每个 profile 单独写目录与结果

### 2.1 参数表

| 参数 | 取值 | 默认值 | 说明 |
|---|---|---:|---|
| `--checkpoint` | 路径 | 必填 | `best_model.pt` 路径 |
| `--dataset` | 路径 | `None` | 可选；未提供时优先用 checkpoint 中记录的数据集路径 |
| `--num_traj_per_scenario` | 整数 | `30` | 每个场景评估轨迹数 |
| `--times` | 多个浮点数 | `10 30 60` | 评估 horizon，单位秒 |
| `--scenarios` | 多个字符串 | `PRBS CHIRP OU` | 场景列表 |
| `--seed` | 整数 | `42` | resampled 模式的基础 seed |
| `--noise_profiles` | `clean`, `nominal_eval`, `degraded_eval`, `heading_biased_eval`, `current_bias_eval`, `all` | `clean` | rollout 初值噪声 profile 集合 |
| `--noise_seed` | 整数 | `2024` | noisy initialization 的基础 seed |
| `--device` | 字符串 | `None` | 未提供时自动选 `cuda/cpu` |
| `--mode` | `heldout`, `resampled` | `heldout` | 评估模式 |
| `--output_dir` | 路径 | `rollout_benchmark_results` | benchmark 输出根目录 |
| `--run_name` | 字符串 | `None` | 输出子目录名 |
| `--quiet` | flag | `False` | 关闭进度输出 |
| `--progress_every` | 整数 | `5` | 每隔多少条轨迹打印一次进度 |
| `--num_diagnostic_plots` | 整数 | `6` | 最多导出多少张诊断图 |

### 2.2 profile 限制

- `current_bias_eval` 仅对 ocean-current checkpoint 可用
- `all` 会根据 checkpoint 的 `ocean_current` 属性自动展开：
  - `NOC -> clean nominal_eval degraded_eval heading_biased_eval`
  - `OC -> clean nominal_eval degraded_eval heading_biased_eval current_bias_eval`

### 2.3 新版噪声方案下的关键行为

- rollout 初值使用与训练端一致的 ODE-space noisy IC 接口
- 支持 `heading_biased_eval`
- 支持 `current_bias_eval`
- `summary.json` 的 `config.noise_budget` 会记录 profile 实际预算

---

## 3. `scripts/train_all_models_noise_profile.sh`

用途：

- 针对一个 `oc` / `noc` 数据集批量训练一组模型
- 统一附带新版 noisy-IC 训练参数
- 自动把 profile-aware 评估参数透传到 `train_auv_hamnode.py`

### 3.1 参数表

| 参数 | 取值 | 默认值 | 说明 |
|---|---|---:|---|
| `--profile` | `oc`, `noc` | `oc` | 决定默认数据集与 auto profile 规则 |
| `--group` | `main`, `baseline`, `ablation`, `core`, `all` | `all` | 模型分组 |
| `--models` | `"A B C"` | 空 | 显式模型列表，覆盖 `--group` |
| `--dataset` | 路径 | 空 | 显式数据集路径，覆盖 auto-discovery |
| `--seeds` | `"43 44 45"` 等 | `"43 44 45"` | 训练 seeds |
| `--device` | 字符串 | 空 | 透传到 `train_auv_hamnode.py` |
| `--noise-profile` | `clean`, `nominal_train`, `nominal_eval`, `degraded_eval` | `nominal_train` | 训练期 noise profile |
| `--noise-scale` | 浮点数 | `1.0` | 透传 |
| `--noise-warmup-epochs` | 整数 | `20` | 透传 |
| `--noise-ramp` | 整数 | `80` | 透传 |
| `--noise-mix-ratio` | 浮点数 | `0.5` | 透传 |
| `--block-eval-noise-profiles` | `"A B"` 或 `auto` | `auto` | block-level 自动评估 profile |
| `--heldout-eval-noise-profiles` | `"A B"` 或 `auto` | `auto` | held-out 自动评估 profile |
| `--prefix` | 字符串 | 空 | suite 名中的前缀；空时自动用 `noise_<profile>` |
| `--suite-name` | 字符串 | 空 | 显式 suite 目录名 |
| `--extra-train-arg` | 任意字符串，可重复 | 空 | 额外透传给 `train_auv_hamnode.py` |

### 3.2 `auto` 规则

`--block-eval-noise-profiles auto`

- `noc -> clean nominal_eval`
- `oc -> clean nominal_eval heading_biased_eval current_bias_eval`

`--heldout-eval-noise-profiles auto`

- `noc -> clean nominal_eval degraded_eval heading_biased_eval`
- `oc -> clean nominal_eval degraded_eval heading_biased_eval current_bias_eval`

### 3.3 新版噪声方案下的关键行为

- 顶层脚本内部会把多值 profile 正确拆开，再透传给 `train_auv_hamnode.py`
- 不再把多值 profile 当成单个字符串传下去
- 默认 auto 规则已覆盖 `heading_biased_eval`
- `oc` 默认 auto 规则已覆盖 `current_bias_eval`

---

## 4. `scripts/eval_all_models_noise_profile.sh`

用途：

- 对一个训练 sweep 统一执行 rollout benchmark
- 再生成 sweep summary 和 experiment report

### 4.1 参数表

| 参数 | 取值 | 默认值 | 说明 |
|---|---|---:|---|
| `--suite-dir` | 路径 | 空 | 显式 sweep 目录 |
| `--suite-name` | 字符串 | 空 | sweep 目录名 |
| `--mode` | `heldout`, `resampled` | `resampled` | rollout benchmark 模式 |
| `--num-traj-per-scenario` | 整数 | `30` | 每个场景轨迹数 |
| `--times` | `"10 30 60"` | `"10 30 60"` | horizon 列表 |
| `--scenarios` | `"PRBS CHIRP OU"` | `"PRBS CHIRP OU"` | 场景列表 |
| `--eval-seed` | 整数 | `42` | 透传为 rollout 基础 seed |
| `--device` | 字符串 | 空 | 透传到 benchmark |
| `--progress-every` | 整数 | `5` | 透传 |
| `--num-diagnostic-plots` | 整数 | `6` | 透传 |
| `--summary-horizon` | 整数/浮点数 | `60` | 汇总与报告默认 horizon |
| `--noise-profiles` | `"A B"` 或 `auto` | `auto` | rollout init-noise profile 集合 |
| `--noise-seed` | 整数 | `2024` | noisy initialization 的 seed |
| `--extra-eval-arg` | 任意字符串，可重复 | 空 | 额外透传给 `evaluate_rollout_benchmark.py` |

### 4.2 `auto` 规则

该脚本会优先从 `suite_config.txt` 读取 `profile=oc/noc`；如果缺失，则回退到 suite 目录名推断。

`--noise-profiles auto`

- `noc -> clean nominal_eval degraded_eval heading_biased_eval`
- `oc -> clean nominal_eval degraded_eval heading_biased_eval current_bias_eval`

### 4.3 新版噪声方案下的关键行为

- 顶层脚本内部会把多值 `noise_profiles` 正确拆开，再透传给 `evaluate_rollout_benchmark.py`
- 不再把 `"clean nominal_eval degraded_eval"` 当成单个参数传下去
- `auto` 已升级为新版推荐组合

---

## 5. 推荐用法

### 5.1 单次训练

```bash
conda run -n mytorch1 python train_auv_hamnode.py \
  --dataset ./data/auv_oc_traj1000_xxx.pkl \
  --model_type phnode_full \
  --noise_profile nominal_train \
  --noise_warmup_epochs 20 \
  --noise_ramp 80 \
  --noise_mix_ratio 0.5
```

### 5.2 单次 rollout benchmark

```bash
conda run -n mytorch1 python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run>/best_model.pt \
  --mode heldout \
  --noise_profiles clean nominal_eval degraded_eval heading_biased_eval current_bias_eval
```

### 5.3 整个训练 sweep

```bash
bash scripts/train_all_models_noise_profile.sh \
  --profile oc \
  --group core
```

### 5.4 整个评估 sweep

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/<suite>
```

---

## 6. 当前结论

就新版噪声方案而言，这 4 个文件当前已经满足以下要求：

- `heading_biased_eval` 已贯通
- `current_bias_eval` 已贯通
- OC / NOC 上下文对 `all` / `auto` 的展开已区分
- shell 包装脚本对多值 profile 的透传已修正
- 训练后评估已输出 noisy/clean 比值与退化百分比

如果后续继续扩展，优先级建议是：

1. 在结果汇总层补 `failure rate` 增量与排名稳定性统计。
2. 再决定是否加入 receding-horizon benchmark。
