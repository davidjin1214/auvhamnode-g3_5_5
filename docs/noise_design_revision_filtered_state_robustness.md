# 面向滤波后状态观测鲁棒性的噪声设计审查与修订建议

## 1. 文档目的

这份文档重新梳理当前项目的噪声设计问题，并统一回答下面四个问题：

1. 当前训练目标

```text
noisy initial estimated state -> ODE rollout -> predict clean future
```

是否合理。

2. 在 ocean-current 场景下，`v_c^n`、`nu_total`、`nu_r` 分别应如何理解为“可获得状态”或“模型内部变量”。

3. 当前噪声 profile 的量级是否合理，是否贴近 REMUS100 这类小型巡航式 AUV 的常见尺度。

4. 在当前研究阶段与后续“真实部署可用性”阶段，实验协议应如何分层设计。

本文的判断基于当前仓库实现，以及外部关于 REMUS100、DVL/INS 导航和海流估计的公开资料。

---

## 2. 正确的问题定义

当前项目的输入对象不是 IMU、DVL、深度计等原始传感器流，而是导航/滤波模块输出的状态估计。因此，当前噪声设计要解决的问题应明确表述为：

```text
模拟真实部署时可获得的 AUV 滤波后状态观测误差，
并检验动力学模型对这种状态估计误差的鲁棒性。
```

这一定义有两个直接含义：

- 当前工作不是“原始传感器噪声建模”。
- 当前工作也不是“让模型同时做滤波/去噪和动力学学习”。

如果研究目标是比较不同动力学模型在带误差状态估计输入下的预测退化程度，那么这个问题定义是合理且公平的。

---

## 3. 对当前训练方向的独立审查

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

这不是 sequence denoising，也不是原始传感器级噪声学习，而是 filtered-state robustness training。

## 3.2 这个方向为什么合理

从 AUV 运动控制和结构化动力学学习的角度，这个方向是合理的，原因如下：

- 输入语义正确。模型消费的是状态估计，而不是原始观测，因此在状态空间而不是传感器空间做扰动，符合部署接口。
- 任务边界清晰。监督目标保持 clean future trajectory，不会把问题混成“动力学学习 + 去噪恢复”。
- 工程问题真实。它对应的问题是：

```text
如果当前导航状态有小误差，
模型对未来轨迹预测会退化多少？
```

- 对 port-Hamiltonian neural ODE 这类结构化连续时间模型而言，这种训练直接检验模型对初值误差的局部敏感性，而这正是实际预测器可用性的关键部分。

## 3.3 这个方向的边界

虽然这个方向合理，但它不能被解释成“已经完整模拟了真实导航误差传播”。

关键原因是：

- 对确定性动力系统而言，错误初值不可能严格推出真实 clean future。
- 因而这个训练目标本质上不是在拟合一个精确生成过程，而是在施加一种鲁棒性正则。
- 它更接近“让模型对小的状态估计误差不过度敏感”，而不是“让模型从错误状态恢复真实状态”。

因此，当前方向更准确的定位应是：

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

## 4.3 训练语义的一致性

之前一个潜在担心是：`std_vel` 会不会基于 data-space 的 `nu_total` 统计，从而与模型实际消费的 `nu_r` 不一致。

复核后，这个担心对当前训练主线不成立。训练前已经先做了与模型语义一致的状态适配，再构造 normalizer。因此当前方案在“噪声定义”和“训练语义”之间的一致性比表面看起来更好。

## 4.4 第一阶段修订的落实情况

第一阶段修订已经完成了最关键的三项：

- `noise_mix_ratio` 已改为逐样本 noisy mask；
- 姿态误差已从 isotropic 调整为 yaw-dominant；
- 训练与评估产物已开始记录各 profile 的实际噪声预算。

这意味着当前实现已经比文档初稿阶段更接近目标语义：

- 噪声暴露粒度更符合“部分样本存在状态估计误差”；
- 姿态误差结构更贴近 AUV 导航中的 heading-dominant 特征；
- profile 的实际物理强度也变得可追溯。

## 4.5 还需要再明确的一层边界

虽然当前方案已经在 ODE 语义上闭环，但它仍然不是导航滤波误差的完整统计模型。

更准确地说，当前方案做的是：

- 围绕模型真实消费的状态变量定义误差预算；
- 用 profile + 数据统计量近似地控制误差半径；
- 将这种误差注入用作 filtered-state robustness regularization。

它没有做的是：

- 复现真实导航滤波器的联合协方差结构；
- 显式建模长期时间相关 bias；
- 给出可直接外推到所有平台和工况的 vendor-level 误差规格。

因此，后续文档和实验结论里应始终避免把当前 profile 直接表述成“真实导航误差本身”，而应表述成：

```text
面向滤波后状态观测误差的结构化鲁棒性近似
```

---

## 5. 状态接口与部署假设

这一节的核心目的，是把 ocean-current 相关状态的语义一次说清。

## 5.1 `nu_total`、`v_c^n`、`nu_r` 的关系

当前数据合同中，ocean-current 版本的状态显式存储：

```text
[Δp, R, nu_total, u_act, u_cmd, v_c^n]
```

模型内部则使用：

```text
nu_r = nu_total - R^T v_c^n
```

因此：

- `nu_total` 是数据接口中的速度状态；
- `v_c^n` 是惯性系 current state；
- `nu_r` 是模型内部真正使用的相对水体速度。

## 5.2 `v_c^n` 是否应被视为稳定可得状态

我的判断是：

```text
在当前仓库的 OC 研究设定里，v_c^n 作为模型输入是自洽的；
但对一般的 REMUS100 类实际部署，不应默认 v_c^n 是稳定可获得的状态估计。
```

理由是：

- 当前仓库的数据合同本来就显式携带 `v_c^n`，模型和训练流程围绕这一接口构建，所以在当前研究设定里把它当成状态量是成立的。
- 但对更一般的 REMUS100 级平台，current estimate 往往依赖额外观测条件、专门滤波器设计或增强配置，不应被默认当成“任何常规任务都稳定可读”的基础状态。

## 5.3 `nu_r` 是否能稳定获得

一般不能。

因为：

```text
nu_r = nu_total - R^T v_c^n
```

只要 `v_c^n` 不是稳定可得的滤波状态，`nu_r` 就不可能比 `nu_total` 更容易稳定获得。

因此，在更一般的真实部署语义下：

- `nu_r` 更适合被视为模型内部变量；
- 而不是默认可测、默认稳定的导航状态。

只有在 `current-observable` 的接口设定下，才可以合理地把 `nu_r` 当成由状态接口稳定构造出来的量。

## 5.4 `nu_total` 是否更接近常规可得状态

是。

对 REMUS100 类平台和一般 AUV 导航链而言，更接近常规可得状态的是 `nu_total`，而不是 `nu_r`。这里的“更接近可得”需要按分量理解，而不是把整个 6-DOF twist 当成同等质量、同等稳定的统一量。原因包括：

- 角速度通常可直接由 IMU / INS 提供；
- 线速度在有 DVL 或相关速度辅助、底跟踪条件较好时通常更容易稳定估计；
- 公开资料里的 REMUS100 导航描述和 DVL 产品接口都更贴近“速度辅助导航”，而不是“直接输出 current-compensated `nu_r`”。

因此，更自然的部署接口判断是：

- `nu_total`：默认可得或较可得；
- `v_c^n`：条件性增强状态；
- `nu_r`：由 `nu_total` 与 `v_c^n` 进一步构造的内部变量。

## 5.5 本文建议采用的两层接口假设

为避免把不同研究阶段混在一起，本文建议明确区分两类接口假设：

### `current-observable`

含义：

- 外部状态接口包含 `nu_total`
- 外部状态接口包含 `v_c^n`
- 模型内部据此构造 `nu_r`

这是**当前默认主设定**。它适合当前研究阶段的问题：

```text
在一个包含 current estimate 的增强滤波状态接口下，
检验动力学模型对状态估计误差的鲁棒性。
```

### `current-unobservable`

含义：

- 外部状态接口不把 `v_c^n` 视为稳定可得状态；
- current 更像未建模扰动，或只能粗略推断；
- 模型和评估不能再把 `v_c^n` 的误差直接解释为“状态估计误差”。

这是**保留的可选扩展**。它不是当前主设定，但在研究目标转向“真实部署可用性”时应补充进来。

---

## 6. 对当前噪声尺度的独立审查

这一节回答的问题是：

```text
当前 profile 的量级是否合理？
会不会明显偏大或偏小？
是否贴近 REMUS100 这类小型巡航式 AUV 的常规尺度？
```

结论先行：

- 当前噪声尺度整体没有明显失真，不属于“离谱偏大”；
- `nu_r` 和 `u_act` 两部分总体在合理区间；
- `delta_theta` 的总量级可以接受，但“各向同性”不够贴近真实导航误差结构；
- `delta_v_c` 的数值本身基本合理，但解释时必须绑定 `current-observable / current-unobservable` 接口假设。

## 6.1 相对速度 `delta_nu_r`

当前设计采用：

```text
sigma_i = max(floor_i, alpha * std_i)
```

其中线速度 floor 为 `0.005 m/s`，角速度 floor 为 `0.0015 rad/s`。

这里还需要强调一点：由于 `std_i` 来自训练集统计，这组 profile 的一部分含义是**数据集相对的**，而不是完全固定的物理标尺。

这带来两个直接推论：

- 在同一数据集上比较不同模型，这样定义完全可以接受；
- 但在不同数据集或不同采样范围之间比较时，`nominal_eval` 的实际物理强度会漂移。

因此，后续实现里建议把各 profile 真实落地的逐通道 `sigma_i` 一并记录到运行产物中，而不是只保存 profile 名称。

结合当前数据统计，这意味着典型速度噪声量级大致落在：

- `nominal_train`: 约 `0.5 cm/s` 到 `1.4 cm/s`
- `nominal_eval`: 约 `0.5 cm/s` 到 `2.3 cm/s`
- `degraded_eval`: 约 `1.0 cm/s` 到 `4.7 cm/s`

以当前项目的数据生成范围为参照：

- 初始 surge 范围是 `0.8` 到 `2.5 m/s`
- RPM 范围是 `400` 到 `1400`

当前 `nominal_eval` 的 surge 误差大约是航速的 `1%` 到 `3%`，`degraded_eval` 仍大致处在“压力测试但不离谱”的区间。

因此，这里的判断应理解为：

- 在当前默认的 `current-observable` OC 设定下，这组 `nu_r` 误差预算总体合理；
- 这里评价的是模型内部消费变量的扰动尺度，而不是在宣称 `nu_r` 本身是一般部署下可直接测得的状态。

在此前提下，对 `nu_r` 的结论是：

- `nominal_train` 不偏大；
- `nominal_eval` 基本合理；
- `degraded_eval` 适合作为压力测试；
- 不建议整体上调这一组噪声。

## 6.2 姿态初值误差 `delta_theta`

当前姿态误差为：

- `nominal_train`: `0.0035 rad`
- `nominal_eval`: `0.0050 rad`
- `degraded_eval`: `0.0120 rad`

换算后约为：

- `0.20°`
- `0.29°`
- `0.69°`

从总量级看，这组值不算大，作为初始状态误差是可以接受的。

但真正的问题不在于绝对大小，而在于误差结构。

对 REMUS100 这类巡航式 AUV，更贴近现实的情况通常是：

- roll / pitch 相对更稳；
- yaw / heading 更容易成为主导误差；
- 姿态误差会通过方向映射影响速度解释和 current projection。

因此，这组姿态噪声的总量级可以保留，但应优先从“各向同性”改为“yaw-dominant 的各向异性”。

## 6.3 current estimate 误差 `delta_v_c`

当前 ocean-current 场景下的 current 误差为：

- `nominal_train`: `0.008 / 0.008 / 0.004 m/s`
- `nominal_eval`: `0.012 / 0.012 / 0.006 m/s`
- `degraded_eval`: `0.030 / 0.030 / 0.015 m/s`

在 `current-observable` 设定下，这组值可以作为合理的工程先验。更准确地说，它们适合作为“增强接口下的滤波状态误差预算”起点，而不应被解读为适用于所有平台和工况的通用 vendor-level 规格。用 nominal 约 `1 cm/s`、degraded 约 `3 cm/s` 作为 current-state 误差量级，在有专门 current estimation 支撑的场景下是可以成立的。

但在 `current-unobservable` 设定下，这组 current-state 噪声就不应再被解释为默认 nominal 的“状态估计误差”，因为那会默认一个比很多常规配置更强的可观测性假设。

因此，对 `delta_v_c` 的判断不能脱离部署接口单独做。关键不只是数值本身，而是 `v_c^n` 这个状态语义是否在目标部署里真实存在。

## 6.4 执行器反馈误差 `delta_u_act`

当前 actuator 误差为：

- 舵面：`0.002 / 0.003 / 0.008 rad`
- 转速：`3 / 5 / 15 rpm`

项目内的 REMUS100 实现采用：

- 舵面限幅 `±15°`
- 推进器限幅 `±1525 rpm`

以此为参照，当前 actuator 噪声大致对应：

- 舵面满量程的 `0.8% / 1.1% / 3.1%`
- RPM 满量程的 `0.2% / 0.3% / 1.0%`

这组数并不大，甚至略偏保守，但作为“执行器反馈状态误差”是合适的，不会把问题变成 actuator fault simulation。

因此，对 `u_act` 的结论是：

- 当前尺度合理；
- 无需优先调整大小；
- 更值得后续补充的是 bias / scale mismatch 类型评估，而不是更大的零均值随机噪声。

## 6.5 尺度审查的总结

综合来看，当前噪声尺度最合理的判断不是“整体偏大”或“整体偏小”，而是：

- `nu_r`：合理，不建议整体上调；
- `u_act`：合理，可保持；
- `delta_theta`：总量级可接受，但应改误差结构；
- `delta_v_c`：数值本身基本合理，但必须绑定接口假设解释。

因此，近期不建议先整体调大或调小所有 noise。更值得优先做的是：

1. 将姿态误差从 isotropic 改为 yaw-dominant。
2. 明确当前主实验采用 `current-observable`。
3. 将 `current-unobservable` 保留为后续扩展接口，而不是现在就并列成第二条主线。

---

## 7. 当前方向何时足够，何时不够

## 7.1 当前主线何时足够

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

对 ocean-current 任务，这里再加一条限定：

- 当前研究阶段可以先采用 `current-observable` 作为默认主设定。

## 7.2 当前主线何时开始不够

只要问题开始更接近下面这些情况，当前方向就不再充分：

- 导航误差包含明显长期 bias；
- current estimate 经常存在持续偏差，而不是一次性小扰动；
- 预测时域很长，误差传播主导结果；
- 实际系统会不断接收新的滤波状态并做滚动预测；
- 研究目标开始转向“真实部署可用性”而不只是“单次初始化鲁棒性”。

在这些情况下，当前主线仍然可以保留，但必须配合更强的评估协议。

对 ocean-current 任务，这通常意味着：

- 不能只停留在 `current-observable`；
- 还应补充 `current-unobservable`，以支撑更接近真实部署的结论。

---

## 8. 推荐的实验协议

## 8.1 主线问题保持不变

建议保留以下核心问题定义：

```text
给定带误差的滤波后初始状态估计 y0_hat，
模型能否仍然较准确地预测未来真实轨迹？
```

这是当前项目最清晰、最公平、也最可解释的主线问题。

## 8.2 当前阶段的默认实验协议

如果实验只想回答下面的问题：

```text
模型对小的 filtered-state 初值误差稳不稳？
```

那么当前阶段建议使用下面这套协议：

- 训练默认采用 `clean` 或 `nominal_train`
- 评估默认采用 `clean`、`nominal_eval`、`degraded_eval`
- ocean-current 任务默认主设定采用 `current-observable`
- benchmark 默认采用单次初始化的 open-loop rollout

这套协议适合做：

- 模型对比；
- 消融研究；
- 当前阶段的 robustness 主实验；
- 证明结构先验是否降低了对状态估计误差的敏感性。

## 8.3 应补充的评估 profile

如果实验结论开始涉及“更贴近真实导航输出”，那么至少应增加偏差型 profile，而不能只依赖零均值小扰动。

最值得优先加入的两类是：

### `heading_biased_eval`

目的：

- 模拟以 yaw 为主的小方向偏差；
- 检验姿态误差对 body/inertial 映射和 current projection 的影响；
- 更贴近真实导航解中的方向误差。

### `current_bias_eval`

目的：

- 模拟 current estimate 的系统性偏差或更新滞后；
- 检验 `R / v_c / nu_r` 耦合误差对 rollout 的影响。

适用条件：

- 使用 `current-observable` 的 ocean-current 模型；
- 流场估计本身是状态解释的重要组成部分；
- 当前研究阶段仍站在增强接口假设下。

## 8.4 何时引入 `current-unobservable`

本文建议的策略不是现在就把 `current-unobservable` 做成并列主线，而是：

- 当前默认主设定使用 `current-observable`
- 将 `current-unobservable` 保留为可选扩展

当研究目标转向下列问题时，再引入 `current-unobservable`：

```text
真实部署可用性
```

或：

```text
更一般、较弱接口条件下的模型可用性
```

## 8.5 何时引入 receding-horizon benchmark

如果真实使用方式不是“一次初始化后长时间自由 rollout”，而是：

```text
周期性接收新的滤波状态，
并反复执行短时预测
```

那么应增加 receding-horizon benchmark。

它的意义不是让模型做 sequence denoising，而是更真实地模拟：

```text
predict for a short window -> receive a new filtered state -> reinitialize -> predict again
```

---

## 9. 对训练与 profile 设计的修订建议

## 9.1 训练主线保持简单

训练端建议继续保持当前 `IC-only` 主线，不要过早把以下机制重新塞回主路径：

- AR(1) 全轨迹噪声；
- block 内姿态/位置漂移积分；
- random-walk bias 序列；
- dropout window 序列噪声。

原因不是这些机制没有价值，而是当前 trainer 并不真正消费 noisy observation sequence。过早加入这些机制，只会降低问题定义的清晰度。

## 9.2 优先修改 `mix_ratio`

如果只做一项训练代码修订，我建议优先改 `mix_ratio` 的实现：

```text
从 batch-level noisy/clean switch
改为 sample-level noisy mask
```

这是当前最直接、最低风险、也最符合目标语义的改动。

## 9.3 训练 profile 建议加入轻量随机化

如果训练始终只使用固定强度的 `nominal_train`，模型容易只在一个很窄的误差半径附近变稳。

更合理的做法是：

- 保留 `nominal_train` 作为主 profile；
- 对 `noise_scale` 做小范围随机化；
- 或在后期少量混入更强的 tail cases。

这样更接近真实滤波状态误差的离散程度，也更有利于评估外推性能。

这里建议再加一个约束：只有在**已经记录并可回溯实际噪声预算**之后，再引入这类随机化。否则训练 profile 的语义会进一步变模糊，不利于复现实验。

## 9.4 profile 语义应从“大小分级”升级为“误差类型分级”

建议今后将 profile 理解为两类：

### 训练 profile

- `nominal_train`

用途：基础 filtered-state uncertainty regularization，不追求覆盖全部部署误差。

### 评估 profile

- `clean`
- `nominal_eval`
- `degraded_eval`
- `heading_biased_eval`
- `current_bias_eval`，仅对 `current-observable` 默认主设定使用
- `actuator_mismatch_eval`，作为后续现实性补充

同时建议在文档和结果表述中显式区分：

- `current-observable`：滤波状态接口中包含 `v_c^n`
- `current-unobservable`：`v_c^n` 不作为稳定可得状态提供

对后一种情况，不应直接复用当前的 current-state noise profile 作为默认 nominal 设定。

## 9.5 建议的分阶段执行方式

为了避免文档目标和代码状态再次脱节，建议按两个阶段推进。

### 第一阶段：收紧训练语义，保持主线不变

这一阶段只做最必要、最不改变问题定义的修订。
目前这部分已经完成：

1. 保留当前 `IC-only` 主线。
2. 将 `mix_ratio` 改为 sample-level noisy mask。
3. 将姿态误差从 isotropic 改为 yaw-dominant。
4. 在训练和评估产物中记录各 profile 实际落地的噪声预算。
5. 在文档和结果说明中明确 `current-observable` 是当前默认增强接口假设。

这一阶段完成后，就可以较稳妥地回答：

```text
模型对小的 filtered-state 初始误差是否更鲁棒？
```

### 第二阶段：增强评估现实性，不急于增加训练复杂度

这一阶段优先扩评估，而不是重新把复杂噪声塞回训练端：

1. 新增 `heading_biased_eval`。
2. 仅对 `current-observable` 任务新增 `current_bias_eval`。
3. 输出 noisy / clean 比值、退化百分比、failure rate 增量等稳健性统计。
4. 增加 receding-horizon benchmark。

只有当研究问题明确转向“更一般、较弱接口下的真实部署可用性”时，再单独引入 `current-unobservable`。

---

## 10. 建议如何报告结果

当前的 clean / noisy 多 profile 评估结构是合理的，但报告时不应只给 noisy 绝对误差。

建议至少增加以下统计量：

- noisy metric / clean metric 比值；
- 相对 clean 的退化百分比；
- failure rate 增量；
- 不同 profile 下模型排名是否稳定。

如果这些指标同时显示：

- clean 不显著变差；
- nominal / bias-type profile 下退化更小；
- 不同 seed 下趋势一致；

那么“模型对滤波后状态观测误差更鲁棒”的结论才更有说服力。

此外，建议把每个 profile 对应的实际噪声预算也一并报告，至少包括：

- `delta_nu_r` 的逐通道 `sigma_i`
- 姿态扰动标准差
- `delta_v_c` 与 `delta_u_act` 的配置值

否则，`nominal_eval` / `degraded_eval` 仍然容易被读者误解为跨数据集不变的绝对规格。

---

## 11. 最终结论

对当前问题的最终判断如下。

## 11.1 可以保留的结论

```text
noisy initial estimated state -> ODE rollout -> predict clean future
```

作为主训练和主评估方向是合理的。

它适合回答的问题是：

```text
模型对小的滤波后状态估计误差是否敏感？
```

对于当前项目，这是一个干净、可解释、且公平的问题定义。

## 11.2 必须明确的边界

这个方向不能被直接解释成：

- 完整的真实导航误差模型；
- 长时部署条件下的全部可用性证明；
- 模型具备从错误状态恢复真实状态的能力证明。

它更准确的定位是：

```text
small-error filtered-state robustness
```

对 ocean-current 相关结论还必须再加一层限定：

```text
当前阶段默认采用 current-observable；
current-unobservable 作为面向真实部署可用性的可选扩展保留。
```

## 11.3 当前最务实的推进顺序

如果近期只推进最关键的修订，我建议顺序如下。
其中前五步已完成，后续重点从第六步开始：

1. 保留当前 `IC-only` 主线。
2. 将 `mix_ratio` 改为逐样本生效。
3. 将姿态误差从 isotropic 改为 yaw-dominant。
4. 在训练与评估产物中记录各 profile 的实际噪声预算。
5. 当前研究阶段默认采用 `current-observable`，并在文档中明确这是增强接口假设。
6. 新增 `heading_biased_eval`。
7. 仅对 `current-observable` 任务新增 `current_bias_eval`。
8. 增加基于退化百分比和 failure rate 增量的报告指标。
9. 将 `current-unobservable` 保留为可选扩展，并在研究目标转向真实部署可用性时补充进来。
10. 增加 receding-horizon benchmark。

这条路线能够在不破坏当前训练主线清晰度的前提下，让 robustness 结论逐步向真实部署语义靠拢。

---

## 参考信息来源

- 当前噪声设计文档：`docs/noise_model_design.md`
- 当前推荐量级推导：`docs/noise_robustness_experiment_design_codex.md`
- 项目内 REMUS100 执行器限幅：`remus100_core.py`
- 项目内数据生成速度与 RPM 范围：`data_collection.py`
- REMUS100 参考平台与仿真参数：
  - Fossen Python Vehicle Simulator: https://www.fossen.biz/pythonVehicleSim/
  - WHOI REMUS 100 overview: https://www2.whoi.edu/site/osl/vehicles/remus-100/
- DVL / 速度辅助导航资料：
  - Teledyne Pathfinder DVL Guide: https://www.teledynemarine.com/en-us/resources/Documents/Brand%20Support/RD%20INSTRUMENTS/Technical%20Resources/Manuals%20and%20Guides/Pathfinder/PathFinder%20DVL%20Guide_Apr22.pdf
  - Teledyne WorkHorse Navigator Technical Manual: https://www.teledynemarine.com/en-us/resources/Documents/Brand%20Support/RD%20INSTRUMENTS/Technical%20Resources/Manuals%20and%20Guides/WorkHorse%20Navigator/Navigator%20Technical%20Manual_Jun20.pdf?csf=1&e=ZjOB6m
  - Teledyne Explorer DVL Operation Manual: https://www.teledynemarine.com/en-us/support/SiteAssets/RDI/Manuals%20and%20Guides/Explorer%20DVL/Explorer_Operation_Manual.pdf
- INS / DVL 可观测性与 current estimation 相关研究：
  - Observability analysis of DVL/PS-aided INS for a maneuvering AUV: https://www.mdpi.com/1424-8220/15/10/26818
  - Ocean current estimation for AUV navigation using SINS/DVL: https://www.sciencedirect.com/science/article/pii/S0029801823016724
  - Model-aided INS with sea current estimation / observer-based approaches: https://journals.sagepub.com/doi/10.5772/60415
