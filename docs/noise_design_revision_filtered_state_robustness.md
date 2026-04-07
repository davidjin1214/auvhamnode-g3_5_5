# 面向滤波后状态观测鲁棒性的噪声设计审查与修订建议

## 1. 文档目的

这份文档重新梳理当前项目的噪声设计问题，目标不是讨论“如何模拟原始传感器噪声”，而是回答下面两个更具体的问题：

1. 当前方向

```text
noisy initial estimated state -> ODE rollout -> predict clean future
```

是否合理？

2. 如果它合理，它的适用边界是什么？在什么情况下它足够，什么情况下必须补充更强的评估？

本文的结论基于当前仓库的训练与评估实现，尤其是：

- `train_auv_hamnode.py`
- `train_utils.py`
- `evaluate_rollout_benchmark.py`
- `docs/noise_model_design.md`

---

## 2. 问题应如何正确定义

当前项目里的输入不是 IMU、DVL、深度计等原始传感器流，而是导航/滤波模块输出的状态估计。因此，当前噪声设计的目标应明确为：

```text
模拟真实部署时可获得的 AUV 滤波后状态观测误差，
并检验动力学模型对这种状态估计误差的鲁棒性。
```

这与下面两类问题不同：

- 原始传感器噪声建模；
- 让模型同时承担滤波/去噪和动力学学习。

如果研究目标是“比较不同动力学模型在带误差状态估计输入下的预测退化程度”，那么当前问题定义是合理且公平的。

---

## 3. 对当前方向的独立审查

## 3.1 当前方向是什么

当前训练主线可以准确概括为：

```text
给定带误差的滤波后初始状态估计 y0_hat，
用它启动 ODE rollout，
并用真实 clean future trajectory 作为监督目标。
```

也就是：

```text
noisy initial estimated state -> ODE rollout -> predict clean future
```

这不是 sequence denoising，也不是原始传感器噪声学习，而是 filtered-state robustness training。

## 3.2 这个方向为什么合理

从水下航行器运动控制和结构化动力学学习的角度，这个方向是合理的，主要原因有四点。

第一，输入语义是对的。模型实际消费的是状态估计，而不是原始观测，因此在状态空间而不是传感器空间做扰动，符合部署接口。

第二，任务边界是干净的。监督目标保持 clean future trajectory，避免把问题混成“动力学学习 + 去噪恢复”。

第三，这个目标真实对应一个工程问题：

```text
如果当前导航状态有小误差，
模型对未来轨迹预测会退化多少？
```

第四，对 port-Hamiltonian neural ODE 这类结构化连续时间模型而言，这样的训练会直接检验模型对初值误差的局部敏感性，而这正是实际预测器可用性的关键部分。

## 3.3 这个方向为什么不能被过度解释

虽然这个方向合理，但它不能被解释成“已经完整模拟了真实导航误差传播”。

原因是：

- 对于确定性动力系统，错误初值不可能严格推出真实 clean future；
- 因此这个训练目标本质上不是在拟合一个精确的生成过程，而是在施加一种鲁棒性正则；
- 它更接近“让模型对小的状态估计误差不过度敏感”，而不是“让模型从错误状态恢复真实状态”。

所以，这个方向的正确定位应是：

```text
small-error filtered-state robustness
```

而不是：

```text
complete real-world navigation error model
```

---

## 4. 对当前实现的复核结论

## 4.1 训练主线与目标基本一致

当前实现已经收敛为 `IC-only` 路线，即只对初始状态注入扰动，而不再把整段 noisy sequence 当作主训练输入。这一点与当前问题定义是一致的，应保留。

## 4.2 ODE-space-consistent 的方向是必要的

在 ocean-current 场景下，模型内部真正消费的是：

```text
nu_r = nu_total - R^T v_c^n
```

因此噪声预算必须围绕模型真实使用的变量来定义，而不能只在 data space 对 `nu_total` 和 `v_c^n` 独立加噪。

当前实现先转到 ODE 语义，再对 `R / nu_r / u_act / v_c` 扰动，然后用 noisy ODE initial condition 直接启动 rollout。这个思路是正确的，也是当前方案最关键的优点之一。

## 4.3 之前需要修正的一点

之前曾担心 `std_vel` 的统计量是基于 data-space 的 `nu_total`，从而与模型真实消费的 `nu_r` 不一致。复核后，这个担心对当前训练主线不成立，因为训练前已经先做了与模型语义一致的状态适配，再构造 normalizer。

这说明当前方案在“噪声定义”和“训练语义”之间的一致性比表面看上去更好。

## 4.4 当前实现仍有一个重要不足

`noise_mix_ratio` 的文档语义是“部分训练样本使用 noisy IC”，但当前实现更接近“整个 batch noisy 或整个 batch clean”。

这会带来两个直接问题：

- 噪声暴露粒度偏粗；
- 梯度方差更大，训练统计不够平滑。

从“训练集中有一部分滤波状态估计样本存在误差”的目标出发，更合理的实现应是逐样本生效。

---

## 5. 对当前噪声尺度的独立审查

这一节回答一个单独的问题：

```text
当前 profile 的量级是否合理？
会不会明显偏大或偏小？
是否贴近 REMUS100 这类小型巡航式 AUV 的常规配置？
```

结论先行：

- 当前噪声尺度整体没有明显失真，不属于“离谱偏大”；
- `nu_r` 和 `u_act` 两部分总体在合理区间；
- `delta_theta` 的总量级可以接受，但“各向同性”不够贴近真实导航误差结构；
- `delta_v_c` 是否合理，取决于部署里是否真的存在可用的 current estimate。

## 5.1 相对速度 `delta_nu_r`

当前设计采用：

```text
sigma_i = max(floor_i, alpha * std_i)
```

其中线速度 floor 为 `0.005 m/s`，角速度 floor 为 `0.0015 rad/s`。

结合当前数据统计，这意味着典型的速度噪声量级大致落在：

- `nominal_train`: 约 `0.5 cm/s` 到 `1.4 cm/s`
- `nominal_eval`: 约 `0.5 cm/s` 到 `2.3 cm/s`
- `degraded_eval`: 约 `1.0 cm/s` 到 `4.7 cm/s`

对 REMUS100 级别平台，这个量级并不夸张。当前项目的数据生成配置里：

- 初始 surge 范围是 `0.8` 到 `2.5 m/s`
- RPM 范围是 `400` 到 `1400`

这与 REMUS100 常见巡航量级是一致的。以此为参照，当前 `nominal_eval` 的 surge 误差大约是航速的 `1%` 到 `3%`，`degraded_eval` 也大致仍在“压力测试但不离谱”的区间。

因此，对 `nu_r` 而言，我的判断是：

- `nominal_train` 不偏大；
- `nominal_eval` 基本合理；
- `degraded_eval` 适合作为压力测试；
- 不建议再整体上调这组噪声。

## 5.2 姿态初值误差 `delta_theta`

当前姿态误差为：

- `nominal_train`: `0.0035 rad`
- `nominal_eval`: `0.0050 rad`
- `degraded_eval`: `0.0120 rad`

换算后约为：

- `0.20°`
- `0.29°`
- `0.69°`

从总量级看，这组值不算大，对带 DVL/INS 约束的导航系统也讲得通，尤其作为初始状态误差是可以接受的。

但真正的问题不在于“偏大还是偏小”，而在于“误差结构是否合理”。

对 REMUS100 这类巡航式 AUV，更贴近现实的情况通常是：

- roll / pitch 相对更稳；
- yaw / heading 更容易成为主导误差；
- 姿态误差会通过方向映射影响速度解释和 current projection。

因此，这组姿态噪声的总量级可以保留，但应优先从“各向同性”改为“yaw-dominant 的各向异性”。

## 5.3 current estimate 误差 `delta_v_c`

当前 ocean-current 场景下的 current 误差为：

- `nominal_train`: `0.008 / 0.008 / 0.004 m/s`
- `nominal_eval`: `0.012 / 0.012 / 0.006 m/s`
- `degraded_eval`: `0.030 / 0.030 / 0.015 m/s`

这里必须先明确一个判断：

```text
在当前仓库的 OC 研究设定里，v_c^n 作为模型输入是自洽的；
但对一般的 REMUS100 类实际部署，不应默认 v_c^n 是稳定可获得的状态估计。
```

原因是：

- 当前仓库的数据合同本来就显式携带 `v_c^n`，模型和训练流程围绕这一接口构建，因此在当前研究设定里把它当成状态量是成立的；
- 但对更一般的 REMUS100 级平台，current estimate 往往依赖额外观测条件、专门滤波器设计或增强配置，不应被默认当成“任何常规任务都稳定可读”的基础状态。

因此，对 `delta_v_c` 的判断必须区分两类 deployment assumption。

### 情形 A：`current-observable`

如果部署设定是：

```text
滤波器或外部模块能够提供可用的 current-related state estimate
```

那么这组值是合理到略保守的。原因是水体速度或相对流速观测本来就在 cm/s 量级，经过滤波后把 nominal 设在 `1 cm/s` 左右、degraded 设在 `3 cm/s` 左右，是可以成立的。

在这种设定下：

- 当前 `delta_v_c` 可以保留；
- 当前 `OC` 模型路径有明确物理语义；
- `current_bias_eval` 应作为重要补充评估。

### 情形 B：`current-unobservable`

如果更接近下面这种部署现实：

```text
current 更像未建模扰动，
或只能粗略推断而不能稳定估计
```

那么当前 `nominal_*` 的 current 误差反而可能偏小，因为它默认了一个比很多常规配置更强的可观测性假设。

在这种设定下，更合理的做法应是：

- 不把 `v_c^n` 作为默认“稳定可得”的状态接口；
- 将 current 误差从“状态估计误差”改写为“环境未建模影响”；
- 将当前 OC 结果明确标为增强假设下的结果，而不是一般 baseline。

因此，对 `delta_v_c` 的判断不能脱离部署接口单独做。它的关键不是数值本身，而是 `v_c^n` 这个状态语义是否在目标部署里真实存在。

## 5.4 执行器反馈误差 `delta_u_act`

当前 actuator 误差为：

- 舵面：`0.002 / 0.003 / 0.008 rad`
- 转速：`3 / 5 / 15 rpm`

项目内的 REMUS100 实现采用：

- 舵面限幅 `±15°`
- 推进器限幅 `±1525 rpm`

以此为参照，当前 actuator 噪声大致对应：

- 舵面满量程的 `0.8% / 1.1% / 3.1%`
- RPM 满量程的 `0.2% / 0.3% / 1.0%`

这组数并不大，甚至可以说略偏保守，但作为“执行器反馈状态误差”是合适的，不会喧宾夺主，也不会把问题变成 actuator fault simulation。

因此，对 `u_act` 的结论是：

- 当前尺度合理；
- 无需优先调整大小；
- 更值得后续补充的是 bias / scale mismatch 类型评估，而不是更大的零均值随机噪声。

## 5.5 尺度审查的最终结论

综合来看，当前噪声尺度最合理的判断不是“整体偏大”或“整体偏小”，而是：

- `nu_r`：合理，不建议整体上调；
- `u_act`：合理，可保持；
- `delta_theta`：总量级可接受，但应改误差结构；
- `delta_v_c`：数值本身合理，但必须绑定 `current-observable / current-unobservable` 假设一起解释。

因此，近期不建议先整体调大或调小所有 noise。更值得优先做的是：

1. 将姿态误差从 isotropic 改为 yaw-dominant。
2. 明确目标部署属于 `current-observable` 还是 `current-unobservable`。
3. 若 current 可观测性不稳定，则将 current 相关 profile 从“默认 nominal”改成更明确的条件性 profile。

---

## 6. 当前方向的适用边界

## 6.1 当前方向何时是足够的

当研究问题是下面这种形式时，当前方向通常足够：

```text
给定一个带小误差的滤波后当前状态，
模型的短中时域 open-loop 预测会退化多少？
```

更具体地说，当满足以下条件时，当前方案可以作为主训练和主评估范式：

- 输入确实是滤波后的状态估计，而不是原始传感器；
- 状态误差主要体现为当前时刻的不确定性，而不是长时间累积的系统偏差；
- 预测时域有限；
- 部署中会周期性拿到新的状态估计；
- 研究重点是比较模型对状态估计误差的敏感性，而不是比较谁更能做误差恢复。

## 6.2 当前方向何时会开始不够

只要问题开始更接近下面这些情况，当前方向就不再充分：

- 导航误差包含明显长期 bias；
- current estimate 经常存在持续偏差，而不是一次性小扰动；
- 预测时域很长，误差传播主导结果；
- 实际系统会不断接收新的滤波状态并做滚动预测；
- 研究目标开始转向“真实部署可用性”而不只是“单次初始化鲁棒性”。

在这些情况下，当前主线仍然可以保留，但必须配合更强的评估协议。

---

## 7. 修订后的实验准则

## 7.1 主线问题保持不变

建议保留以下核心问题定义：

```text
给定带误差的滤波后初始状态估计 y0_hat，
模型能否仍然较准确地预测未来真实轨迹？
```

这是当前项目最清晰、最公平、也最可解释的主线问题。

## 7.2 什么时候只用当前主线就够

如果实验只想回答下面的问题：

```text
模型对小的 filtered-state 初值误差稳不稳？
```

那么下面这一套已经足够：

- 训练使用 `clean` 或 `nominal_train`
- 评估使用 `clean`、`nominal_eval`、`degraded_eval`
- benchmark 采用单次初始化的 open-loop rollout

这套协议适合做：

- 模型对比；
- 消融研究；
- 早期 robustness 验证；
- 证明结构先验是否降低了对状态估计误差的敏感性。

## 7.3 什么时候必须补充 bias-type 评估

如果实验结论开始涉及“更贴近真实导航输出”，那么至少应增加偏差型 profile，而不能只依赖零均值小扰动。

最值得优先加入的两类是：

### `heading_biased_eval`

目的：

- 模拟以 yaw 为主的小方向偏差；
- 检验姿态误差对 body/inertial 映射和 current projection 的影响；
- 更贴近真实导航解中的方向误差。

适用条件：

- 航向保持、转弯、横向误差对任务很重要；
- ocean current 是重要场景；
- 你怀疑模型对姿态误差传播敏感。

### `current_bias_eval`

目的：

- 模拟 current estimate 的系统性偏差或更新滞后；
- 检验 `R / v_c / nu_r` 耦合误差对 rollout 的影响。

适用条件：

- 使用 `current-observable` 的 ocean-current 模型；
- 流场估计本身是状态解释的重要组成部分；
- 你希望评估更贴近部署中的 current mismatch 风险。

## 7.4 什么时候必须补充 receding-horizon 评估

如果真实使用方式不是“一次初始化后长时间自由 rollout”，而是：

```text
周期性接收新的滤波状态，
并反复执行短时预测
```

那么必须增加 receding-horizon benchmark。

它的意义不是让模型做 sequence denoising，而是更真实地模拟下面这种调用方式：

```text
predict for a short window -> receive a new filtered state -> reinitialize -> predict again
```

对于 planner、MPC 或基于滚动窗口的预测器，这种 benchmark 往往比单次初始化更接近真实部署。

---

## 8. 建议采用的实验包

如果目标是让 robustness 结论更扎实，我建议最少使用下面两组维度。

## 8.1 噪声 profile 维度

- `clean`
- `nominal_eval`
- `degraded_eval`
- `heading_biased_eval`
- `current_bias_eval`，仅 `current-observable` 的 ocean-current 任务强烈建议加入

如需进一步扩展，可再增加：

- `actuator_mismatch_eval`

但它的重要性低于 `heading_biased_eval` 和 `current_bias_eval`。

## 8.2 benchmark 维度

- `single-shot rollout`
- `receding-horizon rollout`

这两者分别回答不同问题：

- `single-shot rollout` 反映初值误差下的局部敏感性；
- `receding-horizon rollout` 反映真实使用方式下的可用性。

---

## 9. 对训练设计的修订建议

## 8.1 保持训练主线简单

训练端建议继续保持当前 `IC-only` 主线，不要过早重新引入以下复杂机制作为主路径：

- AR(1) 全轨迹噪声；
- block 内姿态/位置漂移积分；
- random-walk bias 序列；
- dropout window 序列噪声。

原因不是这些机制没有价值，而是当前 trainer 并不真正消费 noisy observation sequence。把它们塞回训练主线，只会降低问题定义的清晰度。

## 8.2 优先修改 `mix_ratio`

如果只做一项训练代码修订，我建议优先改 `mix_ratio` 的实现：

```text
从 batch-level noisy/clean switch
改为 sample-level noisy mask
```

这是当前最直接、最低风险、也最符合目标语义的改动。

## 8.3 训练 profile 建议加入轻量随机化

如果训练始终只使用固定强度的 `nominal_train`，模型容易只在一个很窄的误差半径附近变稳。

更合理的做法是：

- 保留 `nominal_train` 作为主 profile；
- 对 `noise_scale` 做小范围随机化；
- 或在后期少量混入更强的 tail cases。

这样更接近真实滤波状态误差的离散程度，也更有利于评估外推性能。

---

## 10. 对 profile 语义的修订建议

当前 profile 体系应从“噪声大小分级”进一步升级为“导航状态误差类型分级”。

建议将其理解为两类：

## 10.1 训练 profile

### `nominal_train`

用途：

- 小到中等强度的 filtered-state uncertainty regularization；
- 不追求覆盖全部部署误差；
- 重点是稳定训练并获得基础鲁棒性。

## 10.2 评估 profile

### `clean`

用途：

- clean 上界；
- 与带误差输入对照。

### `nominal_eval`

用途：

- 正常导航状态误差；
- 与训练分布相近或相邻。

### `degraded_eval`

用途：

- 压力测试；
- 观察退化幅度是否可控。

### `heading_biased_eval`

用途：

- 评估以 yaw 为主的方向偏差；
- 更贴近真实导航解中常见的姿态误差模式。

### `current_bias_eval`

用途：

- 评估 current estimate 偏差或滞后；
- 检验 `current-observable` 假设下 ocean-current 建模的耦合鲁棒性。

### 关于 current 相关 profile 的限定

建议今后在文档和结果表述中显式区分：

- `current-observable`：滤波状态接口中包含 `v_c^n`
- `current-unobservable`：`v_c^n` 不作为稳定可得状态提供

对后一种情况，不应直接复用当前的 current-state noise profile 作为默认 nominal 设定。

### `actuator_mismatch_eval`

用途：

- 评估轻微执行器反馈偏置或尺度误差；
- 作为补充现实性检查，而不是故障仿真。

---

## 11. 建议如何报告结果

当前的 clean / noisy 多 profile 评估结构是合理的，但报告时不应只给 noisy 绝对误差。

建议至少增加以下统计量：

- noisy metric / clean metric 比值；
- 相对 clean 的退化百分比；
- failure rate 增量；
- 不同 profile 下模型排名是否稳定。

如果这些指标同时显示：

- clean 不显著变差；
- nominal/bias-type profile 下退化更小；
- 不同 seed 下趋势一致；

那么“模型对滤波后状态观测误差更鲁棒”的结论才更有说服力。

---

## 12. 最终结论

对当前方向的最终判断如下。

### 12.1 可以保留的结论

```text
noisy initial estimated state -> ODE rollout -> predict clean future
```

作为主训练和主评估方向是合理的。

它适合回答的问题是：

```text
模型对小的滤波后状态估计误差是否敏感？
```

对于当前项目，这是一个干净、可解释、且公平的问题定义。

### 12.2 必须明确的边界

这个方向不能被直接解释成：

- 完整的真实导航误差模型；
- 长时部署条件下的全部可用性证明；
- 模型具备从错误状态恢复真实状态的能力证明。

它更准确的定位是：

```text
small-error filtered-state robustness
```

此外，对 ocean-current 相关结论还必须再加一层限定：

```text
只有在 current-observable 的部署接口下，
当前 OC 噪声设计才应被视为直接对应“状态估计误差”。
```

### 12.3 最务实的下一步

如果近期只推进最关键的修订，我建议顺序如下：

1. 保留当前 `IC-only` 主线。
2. 将 `mix_ratio` 改为逐样本生效。
3. 在评估中新增 `heading_biased_eval`。
4. 明确区分 `current-observable` 与 `current-unobservable` 两类部署假设。
5. 仅对 `current-observable` 任务新增 `current_bias_eval`。
6. 增加 `receding-horizon` benchmark。

这条路线能够在不破坏当前训练主线清晰度的前提下，让 robustness 结论更贴近真实部署语义。

---

## 参考依据

- 当前噪声设计文档：`docs/noise_model_design.md`
- 当前推荐量级推导：`docs/noise_robustness_experiment_design_codex.md`
- 项目内 REMUS100 执行器限幅：`remus100_core.py`
- 项目内数据生成速度与 RPM 范围：`data_collection.py`
- REMUS100 参考平台信息：Fossen Python Vehicle Simulator, WHOI REMUS 100 页面
- DVL 精度参考：Teledyne RDI Navigator / Pathfinder / Explorer 相关资料
