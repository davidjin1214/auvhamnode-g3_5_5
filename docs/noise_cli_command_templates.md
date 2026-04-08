# 新版噪声方案常用命令模板

本文只保留最常用的命令模板，适合直接复制后按需修改路径或 profile。

配套参数说明请查：

- [`docs/noise_cli_parameter_reference.md`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/noise_cli_parameter_reference.md)
- [`docs/noise_experiment_runbook.md`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/noise_experiment_runbook.md)

---

## 1. 单次训练

### 1.1 OC 数据集，推荐训练配置

```bash
conda run -n mytorch1 python train_auv_hamnode.py \
  --dataset ./data/auv_oc_traj1000_xxx.pkl \
  --model_type phnode_full \
  --noise_profile nominal_train \
  --noise_warmup_epochs 20 \
  --noise_ramp 80 \
  --noise_mix_ratio 0.5
```

### 1.2 NOC 数据集，推荐训练配置

```bash
conda run -n mytorch1 python train_auv_hamnode.py \
  --dataset ./data/auv_noc_traj1000_xxx.pkl \
  --model_type phnode_full \
  --noise_profile nominal_train \
  --noise_warmup_epochs 20 \
  --noise_ramp 80 \
  --noise_mix_ratio 0.5
```

### 1.3 显式指定自动评估 profile

```bash
conda run -n mytorch1 python train_auv_hamnode.py \
  --dataset ./data/auv_oc_traj1000_xxx.pkl \
  --model_type phnode_full \
  --noise_profile nominal_train \
  --block_eval_noise_profiles clean nominal_eval heading_biased_eval current_bias_eval \
  --heldout_eval_noise_profiles clean nominal_eval degraded_eval heading_biased_eval current_bias_eval
```

---

## 2. 单个 checkpoint 的 rollout benchmark

### 2.1 OC checkpoint，全量新版评估 profile

```bash
conda run -n mytorch1 python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run>/best_model.pt \
  --mode heldout \
  --noise_profiles clean nominal_eval degraded_eval heading_biased_eval current_bias_eval
```

### 2.2 NOC checkpoint，全量新版评估 profile

```bash
conda run -n mytorch1 python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run>/best_model.pt \
  --mode heldout \
  --noise_profiles clean nominal_eval degraded_eval heading_biased_eval
```

### 2.3 Resampled benchmark

```bash
conda run -n mytorch1 python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run>/best_model.pt \
  --mode resampled \
  --num_traj_per_scenario 30 \
  --times 10 30 60 \
  --scenarios PRBS CHIRP OU \
  --noise_profiles clean nominal_eval degraded_eval heading_biased_eval
```

---

## 3. 整个训练 sweep

### 3.1 OC sweep，使用新版自动评估组合

```bash
bash scripts/train_all_models_noise_profile.sh \
  --profile oc \
  --group core
```

说明：

- `block_eval_noise_profiles=auto`
- `heldout_eval_noise_profiles=auto`
- 会自动展开为包含 `heading_biased_eval` 和 `current_bias_eval` 的 OC 默认组合

### 3.2 NOC sweep，使用新版自动评估组合

```bash
bash scripts/train_all_models_noise_profile.sh \
  --profile noc \
  --group core
```

### 3.3 OC sweep，手动指定评估 profile

```bash
bash scripts/train_all_models_noise_profile.sh \
  --profile oc \
  --group core \
  --block-eval-noise-profiles "clean nominal_eval heading_biased_eval current_bias_eval" \
  --heldout-eval-noise-profiles "clean nominal_eval degraded_eval heading_biased_eval current_bias_eval"
```

---

## 4. 整个评估 sweep

### 4.1 对一个 sweep 跑 rollout benchmark、汇总和报告

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/<suite>
```

说明：

- `noise_profiles=auto`
- 若该 suite 是 OC，会自动加入 `current_bias_eval`
- 若该 suite 是 NOC，则不会加入 `current_bias_eval`

### 4.2 手动指定 rollout profile

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/<suite> \
  --noise-profiles "clean nominal_eval degraded_eval heading_biased_eval current_bias_eval"
```

### 4.3 指定 resampled 模式与 horizon

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/<suite> \
  --mode resampled \
  --times "10 30 60" \
  --summary-horizon 60
```

---

## 5. 最常用的 profile 组合

### 5.1 OC 自动评估推荐组合

block-level:

```text
clean nominal_eval heading_biased_eval current_bias_eval
```

held-out / rollout:

```text
clean nominal_eval degraded_eval heading_biased_eval current_bias_eval
```

### 5.2 NOC 自动评估推荐组合

block-level:

```text
clean nominal_eval
```

held-out / rollout:

```text
clean nominal_eval degraded_eval heading_biased_eval
```

---

## 6. 常见提醒

- `current_bias_eval` 只适用于 ocean-current 模型或 checkpoint。
- 直接调用 Python 脚本时，多值参数不要加引号包成一个整体。
- 调用顶层 Bash 包装脚本时，多值参数应写成一个字符串，由脚本内部拆分。
- 如果你只想跑 clean baseline，可以把相关 profile 参数写成 `clean`。
