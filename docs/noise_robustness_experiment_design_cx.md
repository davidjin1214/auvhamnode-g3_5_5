# AUV 噪声鲁棒性实验设计文档

## 1. 文档目的

本文档给出本仓库后续噪声相关实验的统一设计，目标是回答下面两个问题，并明确其主次关系：

1. **主问题（P1）**：在真实部署可获得的带噪导航状态下，结构化模型是否比 black-box 模型退化更少？
2. **次问题（P2）**：模型是否能够通过合适的训练策略，从带噪状态估计中学习到有效动力学？

本文档的结论是：

- **主实验聚焦 P1**，采用“干净训练，带噪评估”的设计。
- **补充实验研究 P2**，仅在 P1 建立后再做，且不作为主结论。

这样做的原因是：P1 直接对应部署场景，因果最干净；P2 会把模型结构优势与训练技巧混合，不适合作为主论点。

---

## 2. 当前方案存在的核心问题

当前实现把两个命题混在了一起：

- 训练时仅对 `t=0` 的初始状态起实际作用；
- 监督目标始终是干净未来轨迹；
- 验证和 held-out 评估仍使用干净初值；
- OC 场景下噪声预算直接污染模型真正使用的 `nu_r`；
- 设计文档中的 Level 2 / Level 3 轨迹级噪声，大部分没有被训练过程真正消费。

因此，当前实现既不能干净回答“结构鲁棒性”，也不能严谨回答“带噪观测学习”。

---

## 3. 实验总原则

### 3.1 状态语义

本仓库的数据状态是导航级状态，而不是原始传感器读数：

```text
[Δp(3), R(9), nu_total(6), u_act(3), u_cmd(3), v_c^n(3)]
```

其中：

- `Δp` 是 **block-relative** 位置，不是绝对导航位置；
- `nu_total` 是融合后的总机体系速度；
- 模型内部实际使用的是 `nu_r = nu_total - R^T v_c^n`。

因此，噪声模型必须解释为“导航状态估计误差”，而不是直接套用裸 DVL / IMU 规格。

### 3.2 评估目标

部署时更接近的问题是：

```text
给定带噪导航状态估计 x_hat(t0) 与控制输入，
模型是否仍能正确预测真实未来状态 x(t0+τ)？
```

因此评估时：

- 输入应为带噪状态估计；
- target 应为干净真值轨迹；
- 所有模型必须共享同一批噪声 realizations。

### 3.3 噪声预算原则

OC 场景中模型真正看到的是 `nu_r`，因此噪声预算必须以 `nu_r` 为核心量来设计，而不是分别对 `nu_total` 与 `v_c^n` 独立拍值。

推荐原则：

1. 先确定目标复合误差 `sigma_target(nu_r)`；
2. 再在 `nu_total` 与 `v_c^n` 之间分配预算；
3. 约束其满足：

```text
Var(delta_nu_r) ≈ Var(delta_nu_total) + Var(R^T delta_v_c)
```

在一阶近似下，可按每轴独立预算处理。

---

## 4. 研究问题与实验层次

## 4.1 主实验：结构鲁棒性（P1）

### 目标

验证结构化模型在带噪初始导航状态下的退化是否小于 black-box 模型。

### 结论形式

主结论应写成：

```text
在相同训练集、相同模型容量级别、相同噪声评估协议下，
pHNODE / structured SE(3) 模型的性能退化曲线更平坦，
说明结构先验提升了对导航状态估计误差的鲁棒性。
```

### 实验设计

- 训练：全部使用干净训练数据。
- 验证：全部使用干净验证数据。
- 测试：在 held-out 上进行 clean 和 noisy 两套评估。

训练与评估形式：

```text
训练: y0_clean -> rollout -> pred   vs. target_clean
评估: y0_noisy -> rollout -> pred   vs. target_clean
```

### 为什么这是主实验

- 它直接回答部署问题；
- 它避免把“结构优势”和“噪声增强训练技巧”混合；
- 它最适合做模型家族对比；
- 它不要求先重新设计整个训练框架。

## 4.2 补充实验：带噪状态学习（P2）

### 目标

验证模型能否通过适当训练，在带噪状态估计输入下学习到更稳定的局部动力学。

### 定位

P2 仅作为补充实验，不作为主结论。它的结论应写成：

```text
在主实验之外，合适的噪声感知训练策略可以进一步提升
模型对带噪状态输入的适应性，但这不是结构鲁棒性的唯一来源。
```

### 推荐训练形式

不再沿用当前“noisy t=0 + clean rollout target”的做法，而采用混合训练：

- `50% clean rollout loss`
- `50% noisy k-step teacher forcing loss`

其中 `k` 推荐取 `2~4`，不建议只做纯 1-step。

---

## 5. 主实验的具体方案

## 5.1 参与模型

建议至少包含以下模型：

- `phnode_full`
- `phnode_merged_force`
- `se3_accel_blackbox`
- `blackbox_fullstate`

如果资源允许，可加入关键 ablation：

- `ablate_no_mass_prior`
- `ablate_diag_damping`
- `ablate_no_lift`

主表中优先展示“结构化 vs 非结构化”代表模型，ablation 放补充材料。

## 5.2 训练协议

- 所有模型使用相同数据集划分；
- 所有模型使用干净训练；
- 所有模型使用干净 validation 选 checkpoint；
- 每个模型至少跑 `3` 个 seed；
- 所有模型使用相同的 held-out noise realizations 集合。

### 禁止事项

主实验中不要：

- 不要启用当前 `noise_level > 0` 的训练；
- 不要在 validation 阶段注噪；
- 不要把不同模型暴露在不同噪声 realization 下；
- 不要只报告单个 seed。

## 5.3 评估协议

主实验采用两种部署评估模式。

### 模式 A：Single-Shot Open-Loop

对每个 block：

1. 从干净 block 取 `x0_clean`；
2. 生成 `x0_noisy`；
3. 用 `x0_noisy` 初始化 ODE；
4. rollout 整个 block；
5. 与干净 block 全段比较。

这是最直接的“带噪初值 rollout”测试。

### 模式 B：Receding-Horizon Reinitialization

对每条 held-out trajectory：

1. 在每个 block 起点用新的 noisy observation 重初始化；
2. 只预测当前 block 或一个固定短窗口；
3. 跨 block 汇总所有误差。

该模式更接近“真实导航系统每个周期都会提供新状态估计”的部署方式。

### 推荐顺序

- 主文使用模式 A；
- 模式 B 作为补充验证，说明结论在更接近部署的协议下仍成立。

---

## 6. 噪声 profile 设计

不再沿用现有 Level 1 / 2 / 3 的语义作为主实验协议，而改用更清晰的三类 profile。

## 6.1 Profile 定义

### Profile N：Nominal Navigation Noise

表示正常导航工作状态，作为主评估分布。

包含：

- 姿态初值误差；
- 各轴异方差速度误差；
- 轻量 actuator state 误差；
- OC 场景下轻量 current estimate 误差；
- 小幅 block-constant bias。

不包含：

- dropout；
- 大幅随机游走；
- 极端失真。

### Profile D：Degraded Navigation Noise

表示退化但仍可运行的导航状态。

在 Profile N 基础上增强：

- 各轴 bias；
- current estimate 误差；
- 姿态误差；
- 时间相关性。

可选加入：

- 线速度通道的冻结型 dropout。

### Profile F：Failure / Stress Test

表示明显退化或局部故障，仅用于 stress test。

包含：

- 较高比例 dropout；
- 更大 bias drift；
- 更强 current estimate error。

该 profile 不建议作为默认训练分布，也不建议作为主表唯一结论来源。

## 6.2 注噪对象

对当前数据契约，建议的注噪对象如下。

### 应注噪

- `R`：通过小角度扰动后重新投影到 `SO(3)`；
- `nu_total`：但按 `nu_r` 预算设计；
- `u_act`：小幅测量误差；
- `v_c^n`：仅 OC 场景。

### 谨慎处理

- `Δp`：由于数据是 block-relative，`t=0` 必须保持为 `0`；
- 位置误差可以在 `t>0` 的整段观测轨迹构造中出现，但不应作为主实验 noisy-IC 的直接自由扰动。

### 不应注噪

- `u_cmd`：控制器指令不是观测通道。

## 6.3 各轴异方差预算

对 6 维速度分量，建议按数据统计做相对标定。

定义：

```text
sigma_i = alpha_profile * std_i(dataset_reference)
```

其中：

- `dataset_reference` 对 NOC 用 `nu_total`；
- 对 OC 优先用 `nu_r`；
- `alpha_profile` 随 profile 改变。

推荐起点：

- Profile N: `alpha = 0.03 ~ 0.05`
- Profile D: `alpha = 0.08 ~ 0.12`
- Profile F: `alpha = 0.15 ~ 0.20`

注：这里是每轴相对比例，不是统一标量。

## 6.4 OC 场景的 current 预算

OC 下不要再直接使用较大的固定 `current_std`。应采用复合预算：

```text
sigma_target(nu_r_lin, i) = alpha_profile * std_i(nu_r_lin)
sigma(delta_nu_total, i)^2 + sigma(delta_v_c_body, i)^2 <= sigma_target(i)^2
```

推荐做法：

- 先给 `delta_nu_total` 分配 `60%~80%` 的预算；
- 再把剩余预算分配给 `delta_v_c`；
- 若 `delta_v_c` 推导值过大，取较小者并回退到 `delta_nu_total`。

推荐经验范围：

- Profile N: `sigma_vc ≈ 0.005 ~ 0.015 m/s`
- Profile D: `sigma_vc ≈ 0.015 ~ 0.03 m/s`
- Profile F: `sigma_vc ≈ 0.03 ~ 0.05 m/s`

这明显比当前固定 `0.05 m/s` 更保守，也更符合当前 block 预测任务。

## 6.5 dropout 语义修正

若引入 DVL dropout，其作用应只限于**线速度观测通道**，不应冻结角速度通道。

也就是说：

- dropout 作用于 `nu_total[:3]`；
- 不作用于 `nu_total[3:6]`；
- 不作用于 `R`；
- 不作用于 `u_act`。

对当前实现而言，这是一项必须修复的物理语义问题。

---

## 7. 指标与汇报方式

## 7.1 基础指标

对每个模型、每个 seed、每个 noise profile、每个 noise realization，汇报：

- position RMSE
- rotation geodesic error
- linear velocity RMSE
- angular velocity RMSE
- per-axis RMSE: `u, v, w, p, q, r`
- solver failure rate
- invalid prediction rate

## 7.2 鲁棒性核心指标

除了绝对误差，必须增加退化比：

```text
degradation_ratio = metric(noisy) / metric(clean)
```

主表建议至少展示：

- clean metric
- noisy metric
- degradation ratio

这样才能把“本来就强”和“受噪声影响小”分开。

## 7.3 统计汇总

每组实验至少对 `K = 10` 个噪声 realization 求统计量。

推荐报告：

- mean
- std
- median
- p90 / p95
- worst-case

每个模型至少 `3` 个训练 seed。

最终对外结论基于：

- 跨 seed 汇总；
- 跨 noise realization 汇总；
- 同一 realization 下模型间配对比较。

## 7.4 图表设计

主文建议放以下图表：

1. **退化曲线图**
   横轴：noise scale
   纵轴：RMSE 或 degradation ratio
   曲线：不同模型

2. **profile 对比表**
   行：模型
   列：clean / nominal / degraded / failure

3. **分轴退化热图**
   展示 `u, v, w, p, q, r` 的敏感性差异。

4. **solver stability 图**
   展示 failure rate 随噪声增强的变化。

---

## 8. 补充实验的推荐设计

如果在主实验之后仍要研究带噪学习，推荐如下设计。

## 8.1 训练目标

使用混合损失：

```text
L = lambda_roll * L_clean_rollout + lambda_tf * L_noisy_kstep
```

其中：

- `L_clean_rollout`：从 clean `y0` 做标准 rollout；
- `L_noisy_kstep`：从 noisy state 预测 `k` 步未来；
- 建议 `lambda_roll = lambda_tf = 0.5`；
- 推荐 `k = 2 ~ 4`。

## 8.2 补充实验要回答的问题

不是“噪声训练能不能把 clean 指标做得更好”，而是：

```text
在相同模型结构下，
是否存在一种适度的带噪训练方式，
可以进一步降低 noisy-profile 下的退化？
```

## 8.3 补充实验的评价标准

若补充训练：

- 明显提升 noisy-profile 指标；
- 对 clean 指标损伤有限；
- 不引入明显 solver instability；

则可作为正面结果；否则应保留主实验结论，不强行推广。

---

## 9. 实施顺序

## Phase 1：协议修正

1. 新增独立的 noisy-IC evaluation 接口；
2. 固定噪声 seed 与 realization 池；
3. 定义 `Nominal / Degraded / Failure` 三个 profile；
4. 将噪声实现改为各轴异方差；
5. 修正 OC 场景下的 current budget；
6. 修正 dropout 仅作用在线速度通道。

## Phase 2：主实验

1. 所有模型干净训练；
2. 运行 clean held-out evaluation；
3. 运行三类 noisy-profile held-out evaluation；
4. 汇总跨 seed 与跨 realization 统计；
5. 生成退化曲线与主表。

## Phase 3：补充实验

1. 增加混合训练目标；
2. 在最关键的 2~3 个模型上测试；
3. 只在主实验结论稳定后开展。

---

## 10. 成功标准

主实验成功的标准不是“噪声训练后更准”，而是：

1. 在 clean 指标上，模型基线合理；
2. 在 noisy profile 下，结构化模型退化更少；
3. 结论在多个 seed 与多个 noise realization 下稳定；
4. 结论在 NOC 和 OC 至少一个主场景中成立；
5. 评估协议与部署问题一致。

补充实验成功的标准是：

1. 带噪训练相对 clean training 显著改善 noisy-profile 指标；
2. clean 指标下降可接受；
3. 结论不会反转主实验的结构鲁棒性结论。

---

## 11. 明确不做的事情

为避免再次混淆实验问题，以下做法不建议继续作为主线：

- 不再把当前 `noise_level=1/2/3` 直接当作主实验协议；
- 不再使用“训练带噪、验证干净、据此选 best epoch”的设置；
- 不再使用统一标量速度噪声；
- 不再对 `nu_total` 和 `v_c^n` 独立给大噪声后再让 `nu_r` 被动承受；
- 不再把 dropout 作用到角速度通道；
- 不再用单次 noisy run 的结果推导鲁棒性结论。

---

## 12. 推荐的最终论文叙述

如果按本文档执行，建议论文主叙述如下：

1. 首先证明结构化模型在 clean 数据上达到有竞争力的精度；
2. 然后在模拟真实导航状态估计误差的 noisy profiles 下，比较不同模型的性能退化；
3. 由退化曲线证明结构先验提供了更好的鲁棒性；
4. 最后再用补充实验说明，适度的带噪训练可以进一步增强这种鲁棒性，但不是主结论的来源。

这样的叙述最清晰，也最符合当前代码基础与研究目标。
