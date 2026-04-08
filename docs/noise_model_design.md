# 当前训练噪声设计说明

## 1. 文档状态

这份文档描述的是**当前代码实现**已经采用的噪声接口。

旧版文档里基于 `Level 1 / Level 2 / Level 3` 的整段 AR(1) 轨迹噪声设计，
已经不再代表当前主训练路径。当前仓库的训练器只对 rollout 初值敏感，因此主方案已经收敛为：

```text
IC-only, profile-based, ODE-space-consistent noise
```

**修订历史：**

| 版本 | 日期 | 主要变化 |
|------|------|----------|
| v1 | 2025-xx-xx | IC-only profile 方案上线；速度噪声使用 `alpha × dataset_std` |
| v2 | 2026-04-08 | 速度噪声改为传感器绝对值；旋转噪声改为各向异性（tilt/heading 分离）；执行器噪声对齐电位计精度；新增 depth_ref 噪声（条件性）。详见 [noise_parameter_revision_sensor_grounded.md](./noise_parameter_revision_sensor_grounded.md) |

更完整的设计背景和取舍，请参见：

- [docs/noise_robustness_experiment_design_codex.md](./noise_robustness_experiment_design_codex.md)
- [docs/noise_parameter_revision_sensor_grounded.md](./noise_parameter_revision_sensor_grounded.md)（v2 修订方案，含传感器规格来源）

---

## 2. 当前问题定义

训练数据来自仿真器真值，部署时模型拿到的是**导航滤波器的后验估计状态**（非原始传感器数据）。
当前训练要解决的问题是：

```text
给定带噪初始导航状态估计 y0_hat，
模型能否仍然预测真实未来轨迹？
```

因此，当前主训练接口是：

```text
y0_noisy -> ODE rollout -> pred_{1:T}
target = clean future trajectory
```

这不是"从带噪观测序列学习动力学"，而是"对带噪初始状态更鲁棒"。

噪声分布应模拟**导航 EKF 的后验估计误差**，其特征是：
- 零均值高斯（EKF 的无偏性保证）
- 量级由各传感器精度决定，而非由飞行速度范围决定

---

## 3. 状态语义

数据集存储的状态是：

```text
[Δp(3), R(9), nu_total(6), u_act(3), u_cmd(3), v_c^n(3)]
```

其中：

- `Δp` 是 block-relative 位置
- `R` 是 body-to-inertial 旋转
- `nu_total` 是总机体系速度
- `v_c^n` 是惯性系海流速度

模型内部真正使用的是：

```text
nu_r = nu_total - R^T v_c^n
```

因此 OC 场景下，噪声设计不能只在 `nu_total` 和 `v_c^n` 上各自独立拍值，
而必须控制模型真正消费的 `nu_r` 误差预算。

---

## 4. 当前实现的核心原则

### 4.1 只对初值加噪

当前训练路径中，噪声只作用于 `t=0` 的初始状态。不会再构造整段 noisy block 作为主训练输入。

### 4.2 先在 ODE 语义上采样，再回到数据语义

当前实现的逻辑是：

1. 从 clean data-state `x0_clean` 出发；
2. 转成 clean ODE-state `y0_clean`；
3. 在 ODE 空间对 `R / nu_r / u_act / v_c` 采样噪声；
4. 得到 noisy ODE initial condition `y0_noisy`；
5. 用它直接启动 ODE rollout。

这样做的目标是：

- 控制 `nu_r` 噪声预算；
- 保证 OC 场景下 `R`、`nu_r`、`v_c^n` 的语义一致；
- 避免旧方案里"data-space 看起来噪声不大，但 ODE 实际输入已经被过度污染"的问题。

### 4.3 block-relative 位置不作为独立噪声通道

当前 block 的起点位置按约定总是 0，因此不再对 `Δp(t0)` 单独加噪。

---

## 5. Profile 接口

当前推荐使用：

```bash
--noise_profile {clean,nominal_train,nominal_eval,degraded_eval}
```

含义如下：

| Profile | 用途 | 说明 |
|---|---|---|
| `clean` | 训练 / 评估 | 不加 noisy IC |
| `nominal_train` | 训练 | 推荐的轻量 IC 正则，幅度略低于 nominal_eval |
| `nominal_eval` | 评估 | 模拟典型运行条件下 EKF 后验误差（对应传感器正常工作） |
| `degraded_eval` | 评估 | 退化工况压力测试（DVL 临界、磁场干扰等） |

旧的：

```bash
--noise_level {0,1,2,3}
```

仍然保留，但只是兼容映射：

- `0 -> clean`
- `1 -> nominal_train`
- `2 -> nominal_eval`
- `3 -> degraded_eval`

---

## 6. 各通道噪声设计（v2，传感器精度基准）

> **v2 修订说明：** 下述数值基于 REMUS 100 实际传感器规格（Teledyne RDI Explorer
> 1200 kHz DVL、Honeywell HG1700 AG58 IMU、磁罗盘），替换了原先的
> `alpha × dataset_std` 相对值方案。详细推导见
> [noise_parameter_revision_sensor_grounded.md](./noise_parameter_revision_sensor_grounded.md)。

### 6.1 相对速度 `delta_nu_r`

速度噪声使用**绝对传感器精度值**，不再依赖 `dataset_std`：

| 通道 | nominal_train | nominal_eval | degraded_eval | 传感器依据 |
|------|:-------------:|:------------:|:-------------:|-----------|
| u, v, w（线速度，m/s） | 0.003 | 0.005 | 0.020 | DVL BT 精度 ±0.2%×V±1mm/s，在 1.5 m/s 时 ≈0.004 m/s |
| p, q, r（角速度，rad/s） | 0.0005 | 0.001 | 0.003 | HG1700 陀螺 ARW 0.125 deg/√hr，滤波后主导项 ≈0.5–3 mrad/s |

`floor` 值（安全下界，通常不生效）：

- 线速度：`0.001 m/s`（DVL 硬件噪声底）
- 角速度：`0.0002 rad/s`（HG1700 陀螺噪声底）

### 6.2 姿态初值误差 `delta_theta`

> **v2 关键变化：旋转噪声改为各向异性**，分离重力约束轴（横滚/俯仰）与非约束轴（航向）。

`delta_theta = [delta_roll, delta_pitch, delta_yaw]` 通过 SO(3) 指数映射作用到旋转矩阵：

| 轴 | nominal_train | nominal_eval | degraded_eval | 传感器依据 |
|----|:-------------:|:------------:|:-------------:|-----------|
| 横滚 roll（rad） | 0.003 | 0.005 | 0.015 | HG1700 加速度计 1mg，重力约束，动态残差 3–7 mrad |
| 俯仰 pitch（rad） | 0.003 | 0.005 | 0.015 | 同上 |
| **航向 yaw（rad）** | **0.009** | **0.017** | **0.052** | **磁罗盘：开阔水域 ±1°（0.017 rad），干扰环境 ±3°（0.052 rad）** |

> **为什么航向噪声远大于倾角噪声？**
> 横滚/俯仰受重力持续约束，EKF 可用加速度计持续修正；
> 航向在水下无绝对参考（无 GPS），仅靠磁罗盘辅助，精度受当地磁场异常影响，
> 典型精度比倾角差约 3–4 倍。

### 6.3 海流估计误差 `delta_v_c`

OC 场景下海流误差使用各轴独立预算（此部分 v2 未修改）：

| Profile | `v_cx, v_cy`（m/s） | `v_cz`（m/s） |
|---|---:|---:|
| `nominal_train` | 0.008 | 0.004 |
| `nominal_eval` | 0.012 | 0.006 |
| `degraded_eval` | 0.030 | 0.015 |

这部分误差会和姿态一起影响 OC 下的等效初始条件。

### 6.4 执行器反馈误差 `delta_u_act`

执行器噪声按通道设置，v2 对齐 REMUS 100 实际舵面传感器精度：

| 通道 | nominal_train | nominal_eval | degraded_eval | 传感器依据 |
|------|:-------------:|:------------:|:-------------:|-----------|
| `delta_r`（rad） | 0.004 | 0.009 | 0.017 | 电位计/LVDT 精度 ±0.5–1°（0.009–0.017 rad） |
| `delta_s`（rad） | 0.004 | 0.009 | 0.017 | 同上 |
| `rpm` | 3 | 8 | 20 | 霍尔效应传感器精度 ±5–10 RPM |

如果 `u_dim != 3`，当前实现退化为按 actuator 标准差比例缩放。

### 6.5 绝对深度参考误差 `delta_depth_ref`（条件性）

仅在 `absolute_depth_context=True` 时生效。误差来源为水体密度差异引起的压强-深度换算偏差
（深度计已表面调零，随机零均值误差由密度不均匀性主导）：

| Profile | sigma_depth_ref（m） | 说明 |
|---------|:--------------------:|------|
| `nominal_train` | 0.0（不加） | 训练时默认关闭，避免影响势能梯度学习早期阶段 |
| `nominal_eval` | 0.3 | 均匀水体，校准良好 |
| `degraded_eval` | 1.0 | 存在热跃层或密度异常 |

传感器依据：DSTO ADA604237 记录 REMUS 100 装备 Honeywell 汽车级压力传感器，
精度 ±1% FS BFSL ≈ ±1.4 m（100 m 额定深度，未修正）；调零后实际误差由密度差异决定。

---

## 7. 训练调度

当前 noisy training 不是从第一个 epoch 就全量打开，而是使用：

- `--noise_warmup_epochs`
- `--noise_ramp`
- `--noise_mix_ratio`
- `--block_eval_noise_profiles`
- `--heldout_eval_noise_profiles`

默认建议：

```bash
--noise_profile nominal_train \
--noise_warmup_epochs 20 \
--noise_ramp 80 \
--noise_mix_ratio 0.5
```

含义：

1. 前 20 个 epoch 完全 clean；
2. 后续 80 个 epoch 逐步把 noisy IC 强度从 0 拉到目标值；
3. 达到稳态后，大约一半训练样本使用 noisy IC，另一半保持 clean。

这样可以减少 noisy IC 直接把优化过程打崩的风险。

训练完成后，脚本还会自动运行 profile-aware 评估：

- block evaluation 默认：`clean nominal_eval`
- held-out trajectory evaluation 默认：`clean nominal_eval degraded_eval`

可以按需修改，例如：

```bash
--block_eval_noise_profiles clean degraded_eval
--heldout_eval_noise_profiles all
```

如果你只想保留 clean 自动评估，也可以写：

```bash
--block_eval_noise_profiles clean
--heldout_eval_noise_profiles clean
```

如果你想完全跳过某个评估阶段，使用 `none`：

```bash
--block_eval_noise_profiles none
--heldout_eval_noise_profiles none
```

### 7.1 不同实验目标下的参数取向

这组参数没有唯一"全局最优"组合，因为你可能在优化不同目标。

当前最常见的三种目标是：

1. 追求最稳训练：少炸、少 early stop、少 seed 敏感。
2. 追求最强 noisy robustness：带噪评估时退化更小。
3. 追求尽量不损失 clean 指标：clean held-out 尽量接近纯 clean training。

推荐起点如下：

| 目标 | `noise_profile` | `noise_warmup_epochs` | `noise_ramp` | `noise_mix_ratio` | `noise_scale` | 预期效果 |
|---|---|---:|---:|---:|---:|---|
| 最稳训练 | `nominal_train` | `30` | `100` | `0.3` | `0.7` | 最不容易崩，seed 波动较小，但 noisy 增益偏保守 |
| 最强 noisy robustness | `nominal_train` | `15` | `60` | `0.7` | `1.0` | noisy 退化通常更小，但 clean 指标更容易下降 |
| 尽量保 clean 指标 | `nominal_train` | `25` | `100` | `0.2` | `0.5` | clean 最容易保住，但 noisy 提升通常有限 |

使用建议：

- 如果你刚开始做新数据集或新模型，先从"最稳训练"配置开始。
- 如果 clean 结果已经很好，接下来主要想验证鲁棒性，再切到"最强 noisy robustness"。
- 如果论文主表仍然以 clean 成绩为核心，而你只想要一点点鲁棒性正则，就用"尽量保 clean 指标"。

---

## 8. 评估协议

训练完成后，当前脚本会输出多组评估结果：

- block-level: `clean` + `nominal_eval`
- held-out trajectory: `clean` + `nominal_eval` + `degraded_eval`

这些评估使用固定 noise seed，以保证不同模型之间可直接比较。

当前 rollout benchmark 也支持相同的初值噪声 profile 选择。CLI 入口为：

```bash
python evaluate_rollout_benchmark.py \
  --checkpoint ./checkpoints/<run>/best_model.pt \
  --mode heldout \
  --noise_profiles clean nominal_eval degraded_eval \
  --noise_seed 2024
```

说明：

- `--noise_profiles` 可以传一个、多个，或者 `all`
- `clean` 表示不注 noisy IC
- `nominal_eval` 和 `degraded_eval` 会在 rollout 初值上注入与训练端一致的 profile 噪声

如果传入多个 profile，benchmark 会按 profile 分目录写结果。

---

## 9. 明确不包含的内容

当前主实现**不再**把下面这些内容作为训练主路径：

- AR(1) 全轨迹相关噪声
- block 内位置漂移积分
- block 内姿态漂移积分
- random-walk bias
- DVL dropout 训练

这些内容只有在训练器升级为真正消费 noisy sequence 时才值得恢复。

---

## 10. 推荐命令

Clean training:

```bash
python train_auv_hamnode.py \
  --dataset ./data/oc/<dataset>.pkl \
  --model_type phnode_full \
  --save_dir ./checkpoints
```

Recommended noisy-IC training:

```bash
python train_auv_hamnode.py \
  --dataset ./data/oc/<dataset>.pkl \
  --model_type phnode_full \
  --save_dir ./checkpoints \
  --noise_profile nominal_train \
  --noise_warmup_epochs 20 \
  --noise_ramp 80 \
  --noise_mix_ratio 0.5
```

Evaluation-only ablation with a stronger profile:

```bash
python train_auv_hamnode.py \
  --dataset ./data/oc/<dataset>.pkl \
  --model_type phnode_full \
  --save_dir ./checkpoints \
  --noise_profile nominal_eval
```

通常不建议把 `degraded_eval` 作为默认训练 profile。
