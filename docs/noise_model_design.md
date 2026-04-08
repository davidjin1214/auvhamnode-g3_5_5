# 当前训练噪声设计说明

## 1. 文档状态

这份文档描述的是**当前代码实现**已经采用的噪声接口。

同时，这份文档也会标出当前实现与新版修订方向之间仍然存在的差距。换言之：

- 本文首先回答“代码现在实际做了什么”；
- 然后显式注明“哪些点只是当前近似实现，还不是最终推荐形态”。

旧版文档里基于 `Level 1 / Level 2 / Level 3` 的整段 AR(1) 轨迹噪声设计，
已经不再代表当前主训练路径。当前仓库的训练器只对 rollout 初值敏感，因此主方案已经收敛为：

```text
IC-only, profile-based, ODE-space-consistent noise
```

更完整的设计背景和取舍，请参见：

- [docs/noise_robustness_experiment_design_codex.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/noise_robustness_experiment_design_codex.md)

---

## 2. 当前问题定义

训练数据来自仿真器真值，部署时模型拿到的是导航系统估计状态。当前训练要解决的问题是：

```text
给定带噪初始导航状态估计 y0_hat，
模型能否仍然预测真实未来轨迹？
```

因此，当前主训练接口是：

```text
y0_noisy -> ODE rollout -> pred_{1:T}
target = clean future trajectory
```

这不是“从带噪观测序列学习动力学”，而是“对带噪初始状态更鲁棒”。

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

## 4.1 只对初值加噪

当前训练路径中，噪声只作用于 `t=0` 的初始状态。不会再构造整段 noisy block 作为主训练输入。

## 4.2 先在 ODE 语义上采样，再回到数据语义

当前实现的逻辑是：

1. 从 clean data-state `x0_clean` 出发；
2. 转成 clean ODE-state `y0_clean`；
3. 在 ODE 空间对 `R / nu_r / u_act / v_c` 采样噪声；
4. 得到 noisy ODE initial condition `y0_noisy`；
5. 用它直接启动 ODE rollout。

这样做的目标是：

- 控制 `nu_r` 噪声预算；
- 保证 OC 场景下 `R`、`nu_r`、`v_c^n` 的语义一致；
- 避免旧方案里“data-space 看起来噪声不大，但 ODE 实际输入已经被过度污染”的问题。

## 4.3 block-relative 位置不作为独立噪声通道

当前 block 的起点位置按约定总是 0，因此不再对 `Δp(t0)` 单独加噪。

## 4.4 当前噪声模型的边界

当前实现虽然已经切换到 ODE-space-consistent 的 noisy IC，但它仍然只是一个**结构化鲁棒性正则接口**，而不是完整的导航误差后验模型。

具体来说，当前实现默认的是：

- 在 `nu_r / R / u_act / v_c` 等通道上分别施加零均值扰动；
- 各通道误差预算主要通过 profile 常数和训练集统计量来确定；
- 重点是控制模型真正消费的变量误差半径，而不是复现导航滤波器的联合协方差结构。

因此，当前噪声接口更准确的定位是：

```text
filtered-state robustness regularization
```

而不是：

```text
full navigation posterior error model
```

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
| `nominal_train` | 训练 | 推荐的轻量 IC 正则 |
| `nominal_eval` | 评估 | 正常导航不确定性 |
| `degraded_eval` | 评估 | 更强的退化压力测试 |

评估侧另外支持一个 bias-type profile：

| Profile | 用途 | 说明 |
|---|---|---|
| `heading_biased_eval` | 评估 | 在 `nominal_eval` 基础上叠加固定幅值的 yaw bias |
| `current_bias_eval` | 评估 | 仅用于 OC；在 `nominal_eval` 基础上叠加固定幅值的 current bias |

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

## 6. 各通道噪声的当前设计

## 6.1 相对速度 `delta_nu_r`

速度噪声不再使用“所有轴统一标量”的设计，而是按通道自然尺度确定：

```text
sigma_i = max(floor_i, alpha * std_i)
```

其中：

- `std_i` 来自训练集统计
- `alpha` 由 profile 决定
- `floor_i` 保证噪声不会不现实地趋近于 0

这意味着当前 `nominal_*` / `degraded_*` 的一部分语义是**相对训练数据分布**的，而不是完全固定的物理规格。

因此，解释结果时应注意：

- 同一数据集内比较不同模型时，这种定义是合理的；
- 跨数据集、跨采样范围比较时，profile 的实际物理强度可能漂移；
- 推荐在运行产物中额外记录各通道最终落地的 `sigma_i`，避免 `nominal_eval` 被误读成绝对不变的噪声标准。

当前实现使用：

- 线速度 floor: `0.005 m/s`
- 角速度 floor: `0.0015 rad/s`

profile 对应的 `alpha`：

- `nominal_train`: `0.03`
- `nominal_eval`: `0.05`
- `degraded_eval`: `0.10`

## 6.2 姿态初值误差 `delta_theta`

当前姿态初值误差已经调整为 **yaw-dominant 的各向异性小角度扰动**，随后通过 SO(3) 指数映射作用到旋转矩阵。

实现原则是：

- 保持与旧版 profile 接近的总量级；
- 让 yaw 方向成为主导误差；
- 使初始姿态误差更贴近 AUV 导航中常见的 heading-dominant 特征。

- `nominal_train`: `0.0035 rad`
- `nominal_eval`: `0.0050 rad`
- `degraded_eval`: `0.0120 rad`

上面三组值仍表示 profile 的基础量级；当前代码会将它们映射为 yaw-dominant 的逐轴标准差。

评估侧新增的 `heading_biased_eval` 使用以下规则：

- 随机部分沿用 `nominal_eval` 的逐轴姿态标准差；
- 再叠加固定幅值的 yaw bias：`0.015 rad`；
- bias 的符号按样本固定，在一条 rollout 内不随时间变化。

这样可以把“带方向性的 heading 偏差”和“零均值小扰动”区分开来，同时又避免整个评估只落在单一正负号上。

## 6.3 海流估计误差 `delta_v_c`

OC 场景下海流误差使用各轴独立预算：

| Profile | `v_cx, v_cy` | `v_cz` |
|---|---:|---:|
| `nominal_train` | `0.008 m/s` | `0.004 m/s` |
| `nominal_eval` | `0.012 m/s` | `0.006 m/s` |
| `degraded_eval` | `0.030 m/s` | `0.015 m/s` |

这部分误差会和姿态一起影响 OC 下的等效初始条件。

评估侧新增的 `current_bias_eval` 使用以下规则：

- 随机部分沿用 `nominal_eval` 的 `delta_v_c` 标准差；
- 再叠加固定幅值的 current bias：`0.015 / 0.015 / 0.005 m/s`；
- bias 的符号按样本固定，在一条 rollout 内不随时间变化；
- 该 profile 仅对 ocean-current 模型开放。

## 6.4 执行器反馈误差 `delta_u_act`

执行器噪声改成了按通道设置，不再使用单一标量：

| 通道 | nominal_train | nominal_eval | degraded_eval |
|---|---:|---:|---:|
| `delta_r` | `0.002 rad` | `0.003 rad` | `0.008 rad` |
| `delta_s` | `0.002 rad` | `0.003 rad` | `0.008 rad` |
| `rpm` | `3 rpm` | `5 rpm` | `15 rpm` |

如果未来 `u_dim != 3`，当前实现会退化为按 actuator 标准差比例缩放。

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

当前实现中，`noise_mix_ratio` 已按 **sample-level noisy mask** 生效。也就是说，`noise_mix_ratio=0.5` 现在表示大约一半训练样本使用 noisy IC，而不是按 batch 粗粒度开关。

训练完成后，脚本还会自动运行 profile-aware 评估：

- NOC block evaluation 默认：`clean nominal_eval`
- NOC held-out trajectory evaluation 默认：`clean nominal_eval degraded_eval heading_biased_eval`
- OC block evaluation 默认：`clean nominal_eval heading_biased_eval current_bias_eval`
- OC held-out trajectory evaluation 默认：`clean nominal_eval degraded_eval heading_biased_eval current_bias_eval`

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

训练与评估输出现在还会额外记录各 profile 的实际噪声预算，包括：

- `delta_nu_r` 的逐通道 `sigma_i`
- 姿态扰动的逐轴标准差
- 若存在 bias-type profile，还会记录 bias 的幅值与模式
- `delta_u_act` 与 `delta_v_c` 的实际配置值

多 profile 评估结果现在还会额外给出相对 `clean` 的比较指标，包括：

- `ratio_to_clean`
- `degradation_pct`
- 以及 `success_rate` / `failure_rate` 一类稳定性指标的相对变化

这些信息会写入运行目录中的 `noise_budgets.json`，并同步写入评估 JSON / TXT 结果。

## 7.1 不同实验目标下的参数取向

这组参数没有唯一“全局最优”组合，因为你可能在优化不同目标。

当前最常见的三种目标是：

1. 追求最稳训练：少炸、少 early stop、少 seed 敏感。
2. 追求最强 noisy robustness：带噪评估时退化更小。
3. 追求尽量不损失 clean 指标：clean held-out 尽量接近纯 clean training。

它们的差别在于主优化方向不同：

- 更高的 noisy 暴露通常更有利于鲁棒性；
- 但 noisy 暴露越强，clean 上界越容易下降；
- 更保守的噪声调度通常最稳，但 noisy 增益也更有限。

推荐起点如下：

| 目标 | `noise_profile` | `noise_warmup_epochs` | `noise_ramp` | `noise_mix_ratio` | `noise_scale` | 预期效果 |
|---|---|---:|---:|---:|---:|---|
| 最稳训练 | `nominal_train` | `30` | `100` | `0.3` | `0.7` | 最不容易崩，seed 波动较小，但 noisy 增益偏保守 |
| 最强 noisy robustness | `nominal_train` | `15` | `60` | `0.7` | `1.0` | noisy 退化通常更小，但 clean 指标更容易下降 |
| 尽量保 clean 指标 | `nominal_train` | `25` | `100` | `0.2` | `0.5` | clean 最容易保住，但 noisy 提升通常有限 |

使用建议：

- 如果你刚开始做新数据集或新模型，先从“最稳训练”配置开始。
- 如果 clean 结果已经很好，接下来主要想验证鲁棒性，再切到“最强 noisy robustness”。
- 如果论文主表仍然以 clean 成绩为核心，而你只想要一点点鲁棒性正则，就用“尽量保 clean 指标”。

一个务实的顺序是：

```text
先跑最稳训练
再根据 held-out noisy 退化幅度，决定是否把 mix_ratio / scale 往上调
```

这样比一开始就上高强度 noisy 配置更稳妥。

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
  --noise_profiles clean nominal_eval degraded_eval heading_biased_eval current_bias_eval \
  --noise_seed 2024
```

说明：

- `--noise_profiles` 可以传一个、多个，或者 `all`
- `clean` 表示不注 noisy IC
- `nominal_eval` 和 `degraded_eval` 会在 rollout 初值上注入与训练端一致的 profile 噪声
- `heading_biased_eval` 会在 `nominal_eval` 的随机扰动基础上再叠加固定幅值的 yaw bias
- `current_bias_eval` 会在 `nominal_eval` 的海流扰动基础上再叠加固定幅值的 current bias，仅对 OC checkpoint 可用

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

---

## 11. 当前状态与下一步重点

第一阶段修订已经完成：

1. `noise_mix_ratio` 已改为 sample-level mask。
2. 姿态误差已改为 yaw-dominant。
3. 训练与评估产物会记录实际噪声预算。

当前更值得继续推进的是：

1. 在评估侧继续补充 bias-type profile；其中 `heading_biased_eval` 与 `current_bias_eval` 已完成。
2. 继续扩展基于 noisy / clean 比值和退化百分比的稳健性报告指标。
3. 视研究目标再决定是否引入 receding-horizon benchmark 与 `current-unobservable` 扩展。

因此，当前文档里的 profile 仍应被解释为：

- 当前研究阶段的 robustness interface；
- 而不是可直接外推到所有部署条件的真实导航误差标准。
