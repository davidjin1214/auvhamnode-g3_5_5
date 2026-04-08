# CLI 参数参考

本文档覆盖四个入口的全部参数、默认值和说明。

---

## `train_auv_hamnode.py`

```
python train_auv_hamnode.py --dataset PATH [options]
```

`--dataset` 是唯一必填参数。带 **†** 的参数默认值随数据集类型自动选择（从文件名推断
`noc` 或 `oc`，无法识别时回退到 `noc`）。

### 模型

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model_type` | `phnode_full` | 模型架构。可选值见下表 |
| `--hidden_dim` | `128` | 所有子网络的隐藏层宽度 |

**`--model_type` 可选值：**

| 值 | 说明 |
|---|---|
| `phnode_full` | 完整 port-Hamiltonian NODE（主模型） |
| `phnode_merged_force` | 合并非保守力分支的 pH 核 |
| `phnode_qforce` | 使用广义 q 力替代标量 V(q) |
| `se3_momentum_blackbox` | 精确 SE(3) + 常数质量矩阵（动量坐标） |
| `se3_accel_blackbox` | 精确 SE(3) 运动学 + 黑箱加速度 |
| `blackbox_fullstate` | 完全非结构化（含运动学学习） |
| `ablate_no_mass_prior` | 无物理质量初始化的消融 |
| `ablate_diag_damping` | 仅对角阻尼的消融 |
| `ablate_no_lift` | 无反对称 lift 的消融 |
| `ablate_bu_only` | 仅 B(u) 的消融 |

### 训练超参数

| 参数 | 默认值（noc †） | 默认值（oc †） | 说明 |
|---|---|---|---|
| `--batch_size` | `2048` | `4096` | |
| `--epochs` | `200` | `300` | 最多运行的 epoch 数（上限） |
| `--total_steps` | `7000` | `5000` | 优化器步数达到后提前停止 |
| `--lr` | `5e-3` | `6e-3` | warmup 后的峰值学习率 |
| `--min_lr` | `1e-4` | `1e-4` | cosine decay 的终止学习率 |
| `--warmup_steps` | `300` | `400` | 线性 warmup 步数 |
| `--hidden_dim` | `128` | `128` | 子网络隐藏层宽度 |
| `--so3_reg` | `1e-3` | `1e-3` | SO(3) 正交正则化权重 |
| `--actuator_loss_weight` | `0.2` | `0.2` | 执行器状态监督损失权重 |
| `--seed` | `42` | `42` | 随机种子 |
| `--device` | `cuda`/`cpu` | `cuda`/`cpu` | 有 CUDA 时自动选 `cuda` |

### 学习率调度

```
步数 0 → warmup_steps      : lr 从 min_lr 线性升至 lr（峰值）
步数 warmup_steps → total_steps : cosine decay 降至 min_lr
到达 total_steps 后提前结束，不等待 epochs 耗尽
```

### 噪声（训练时 IC 注入）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--noise_profile` | `None`（→ `clean`） | 训练噪声 profile。可选：`clean` `nominal_train` `nominal_eval` `degraded_eval` |
| `--noise_scale` | `1.0` | 噪声幅度全局倍数 |
| `--noise_warmup_epochs` | `20` | 开启噪声前的纯净 warmup epoch 数 |
| `--noise_ramp` | `100` | 噪声从 0 线性增加到满幅的 ramp epoch 数 |
| `--noise_mix_ratio` | `0.5` | noisy IC 批次占训练批次的比例 |

> 默认 `noise_profile=None` 解析为 `clean`，**训练时不注入任何噪声**。
> `noise_warmup / noise_ramp / noise_mix_ratio` 仅在 `noise_profile` 为
> `nominal_train` 时实际生效。

**Curriculum 时间线（激活噪声时）：**

```
epoch 1 → noise_warmup_epochs     : 完全干净
epoch warmup → warmup + noise_ramp : 噪声幅度线性从 0 升至 noise_scale
epoch warmup + noise_ramp → 结束   : 稳定态，mix_ratio 比例的批次使用 noisy IC
```

### 物理初始化 & 执行器

| 参数 | 默认值（noc †） | 默认值（oc †） | 说明 |
|---|---|---|---|
| `--mass_init` | `remus` | `remus` | 质量矩阵先验来源。可选：`none` `remus` `file` |
| `--mass_init_path` | `None` | `None` | `--mass_init file` 时必填的 `.npy/.npz` 路径 |
| `--t_actuator_init` | `None` | `0.1 0.1 1.0` | 执行器时间常数先验（s），长度 1 或 u_dim |
| `--u_act_scale` | `None` | `1.0 1.0 0.001` | B_net 执行器缩放，长度 1 或 u_dim |

### 海流 & 深度

| 参数 | 默认值（noc †） | 默认值（oc †） | 说明 |
|---|---|---|---|
| `--ocean_current` | `False` | `True` | 启用海流感知（也由数据集 metadata 覆盖） |
| `--dj_current_feature` | `none` | `current_body` | 附加到 D_net/J_net 的 3D 特征。可选：`none` `current_body` `total_velocity` |
| `--actuation_current_feature` | `current_body` | `current_body` | 附加到 B_net 的 3D 特征。可选：`none` `current_body` `total_velocity` |
| `--include_depth_in_potential` | `False` | `False` | 势能/力基线是否以深度为条件（flag） |

### 保存 & 输出

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--save_dir` | `./checkpoints` | checkpoint 根目录 |
| `--run_name` | `None`（自动生成） | run 子目录名，含模型类型、lr、steps、seed、时间戳、config hash |

### 训练后评估（自动执行）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--block_eval_noise_profiles` | `clean nominal_eval` | 按 block 评估的噪声条件（空格分隔，支持 `all` `none`） |
| `--heldout_eval_noise_profiles` | `clean nominal_eval degraded_eval` | 保留轨迹评估的噪声条件（空格分隔） |

---

## `evaluate_rollout_benchmark.py`

```
python evaluate_rollout_benchmark.py --checkpoint PATH [options]
```

`--checkpoint` 是唯一必填参数。

### 核心

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--checkpoint` | **必填** | `best_model.pt` 路径 |
| `--dataset` | `None` | 数据集路径；省略时从 checkpoint config 中读取 |
| `--mode` | `heldout` | 轨迹来源。`heldout`：使用数据集保留的测试轨迹；`resampled`：重新采样基准轨迹 |
| `--device` | `None`（自动） | torch 设备，省略时自动选 `cuda`/`cpu` |
| `--seed` | `42` | `resampled` 模式的基础随机种子 |

### 评估范围

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num_traj_per_scenario` | `30` | 每个场景的轨迹数 |
| `--times` | `10.0 30.0 60.0` | 评估时域（秒），空格分隔，可多个 |
| `--scenarios` | `PRBS CHIRP OU` | 场景名称，空格分隔 |

### 噪声

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--noise_profiles` | `clean` | IC 噪声条件，空格分隔。可选：`clean` `nominal_eval` `degraded_eval` `all` |
| `--noise_seed` | `2024` | noisy IC 初始化的随机种子 |

> 指定多个 profile 时（如 `--noise_profiles clean nominal_eval degraded_eval`），
> 每个 profile 结果保存在独立子目录中。

### 输出 & 进度

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--output_dir` | `rollout_benchmark_results` | 结果根目录 |
| `--run_name` | `None` | 结果子目录名（省略时自动推导） |
| `--progress_every` | `5` | 每 N 条轨迹打印一次进度 |
| `--num_diagnostic_plots` | `6` | 最多导出的诊断图数量 |
| `--quiet` | `False` | 抑制逐轨迹进度输出（flag） |

**输出文件（每个 profile 一套）：**

| 文件 | 内容 |
|---|---|
| `summary.json` / `summary.txt` | 整体指标汇总 |
| `trajectory_metrics.csv` | 逐轨迹 × 逐时域指标 |
| `horizon_metrics.csv` | 按时域聚合的统计量 |
| `time_series_metrics.csv` | 误差增长时间序列 |
| `rollout_outcomes.csv` | 完成率 / 失败率分类统计 |
| `diagnostic_cases.csv` | 最差案例列表 |
| `metric_contract.csv` | 文件完整性校验 |
| `error_growth.png` | 误差增长曲线 |
| `terminal_error_boxplots.png` | 终端误差箱线图 |
| `example_rollouts.png` | 中位误差轨迹示例 |
| `diagnostic_plots/` | 最多 `num_diagnostic_plots` 张逐案例诊断图 |

---

## `scripts/train_noise_sweep.sh`

```
scripts/train_noise_sweep.sh [options]
```

在 `batch_train_models.sh` 基础上将噪声 profile 提升为一等参数，
并将噪声标签嵌入 suite 目录名以与干净训练隔离。

### 噪声控制

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--noise-profile PROFILE` | `nominal_train` | 训练噪声 profile。可选：`clean` `nominal_train` `nominal_eval` `degraded_eval` |
| `--noise-scale FLOAT` | `1.0` | 噪声幅度全局倍数 |
| `--noise-ramp N` | `100` | curriculum ramp 长度（epochs） |
| `--noise-warmup N` | `20` | 开启噪声前的纯净 warmup epoch 数 |
| `--noise-mix FLOAT` | `0.5` | noisy IC 批次占训练批次的比例 |

### 数据集 & 模型

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--profile {oc\|noc}` | `oc` | 数据集预设，决定默认数据集路径和超参数 |
| `--dataset PATH` | 由 `--profile` 推断 | 显式指定 `.pkl` 路径，覆盖 `--profile` 默认值 |
| `--group {main\|baseline\|ablation\|core\|all}` | `all` | 要训练的模型子集 |
| `--models "A B C"` | — | 显式指定模型列表（空格分隔），覆盖 `--group` |

**`--group` 包含的模型：**

| 值 | 包含模型 |
|---|---|
| `main` | `phnode_full` |
| `baseline` | `phnode_merged_force` `phnode_qforce` `se3_momentum_blackbox` `se3_accel_blackbox` `blackbox_fullstate` |
| `ablation` | `ablate_no_mass_prior` `ablate_diag_damping` `ablate_no_lift` `ablate_bu_only` |
| `core` | `main` + `baseline` |
| `all` | `core` + `ablation` |

### 执行控制

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--seeds "N1 N2 ..."` | `"42 43 44"` | 多 seed，空格分隔，每个 seed 独立训练 |
| `--device DEVICE` | — | 转发给 `train_auv_hamnode.py`（如 `cuda:0`） |
| `--extra-train-arg ARG` | — | 转发给训练脚本的额外参数，可重复 |

### 输出控制

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--prefix NAME` | `default` | 嵌入 suite 目录名的标签 |
| `--suite-name NAME` | 自动生成 | 显式指定 suite 目录名（格式：`noise_sweep_{profile}_{profile}_{group}_{prefix}_...`） |

**自动生成 suite 目录名示例：**
```
noise_sweep_nominal_train_oc_core_default_auv_oc_..._s42-43-44_20260408_143022
```

**run 命名规则：** `{model_group}_{model_type}_{noise_profile}_seed{N}`

**已有 checkpoint 自动跳过**（幂等，可安全重跑）。

**典型用法：**
```bash
# 用 nominal_train 训练 core 模型
scripts/train_noise_sweep.sh --noise-profile nominal_train --group core

# 指定 seed 和 GPU
scripts/train_noise_sweep.sh --noise-profile degraded_eval --seeds "42 43" --device cuda:0

# 作为干净训练基线
scripts/train_noise_sweep.sh --noise-profile clean --group main --prefix baseline
```

---

## `scripts/eval_noise_sweep.sh`

```
scripts/eval_noise_sweep.sh --suite-dir PATH [options]
```

在 `batch_eval_models.sh` 基础上将噪声 profiles 提升为一等参数，
支持在一次调用中对比多个噪声条件。

### 核心

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--suite-dir PATH` | **必填** | 包含 `runs.tsv` 的训练 suite 目录 |
| `--noise-profiles "P1 P2 ..."` | `"clean nominal_eval degraded_eval"` | 评估噪声条件（空格分隔）。可选：`clean` `nominal_eval` `degraded_eval` `all` |
| `--mode {heldout\|resampled}` | `heldout` | rollout benchmark 模式 |

### 评估范围

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num-traj-per-scenario N` | `30` | 每个场景的轨迹数 |
| `--times "10 30 60"` | `"10 30 60"` | 预测时域（秒），空格分隔 |
| `--scenarios "PRBS CHIRP OU"` | `"PRBS CHIRP OU"` | 场景名称，空格分隔 |
| `--seed N` | `42` | `resampled` 模式的基础随机种子 |

### 输出 & 进度

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--device DEVICE` | — | 转发给 `evaluate_rollout_benchmark.py` |
| `--progress-every N` | `5` | 每 N 条轨迹打印一次进度 |
| `--num-diagnostic-plots N` | `6` | 最多导出的诊断图数量 |
| `--extra-eval-arg ARG` | — | 转发给评估脚本的额外参数，可重复 |

**结果目录结构：**
```
{run_dir}/rollout_benchmark/
└── {eval_name}_noise{profiles_tag}/    # 如 heldout_traj30_seed42_noiseclean-nominal_eval-degraded_eval
    ├── clean/                          # 多 profile 时各 profile 独立子目录
    ├── nominal_eval/
    └── degraded_eval/
```

**已有结果自动跳过**（基于 profile tag 匹配，幂等）。

**典型用法：**
```bash
# 默认三档噪声全评
scripts/eval_noise_sweep.sh --suite-dir ./checkpoints/noise_sweep_nominal_train_...

# 只评 clean 和 nominal_eval
scripts/eval_noise_sweep.sh --suite-dir ./checkpoints/my_sweep --noise-profiles "clean nominal_eval"

# all 简写 + resampled 模式
scripts/eval_noise_sweep.sh --suite-dir ./checkpoints/my_sweep --noise-profiles all --mode resampled
```

---

## 完整工作流

```bash
# 1. 生成数据集
python data_collection.py --num_traj 500 --blocks 150 --seed 42 \
  --save_dir ./data/oc --workers 4 --ocean_current

# 2a. 单次训练（干净）
python train_auv_hamnode.py --dataset ./data/oc/<dataset>.pkl

# 2b. 单次训练（噪声）
python train_auv_hamnode.py --dataset ./data/oc/<dataset>.pkl \
  --noise_profile nominal_train

# 3a. 批量训练 sweep（噪声）
scripts/train_noise_sweep.sh --noise-profile nominal_train --group core --profile oc

# 3b. 批量训练 sweep（干净基线）
scripts/train_noise_sweep.sh --noise-profile clean --group core --prefix clean_baseline

# 4. 批量评估（三档噪声对比）
scripts/eval_noise_sweep.sh --suite-dir ./checkpoints/noise_sweep_nominal_train_...

# 5. 单模型独立评估
python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run>/best_model.pt \
  --noise_profiles clean nominal_eval degraded_eval \
  --mode heldout

# 6. 汇总 sweep 结果
python scripts/summarize_sweep.py --suite-dir ./checkpoints/<suite>
```
