# ph-NODE 现实导向验证研究计划

> **2026-05 状态注**：本文档保留为研究计划层入口，但部分前提已被后续 Phase-1A cleanrun v1 与 provenance audit 更新。阅读本文中关于 `phnode_full` seed fragility 或 `v4-lite` 尚属拟实施协议的表述时，应同时参考 [experiment_stages_overview.md](experiment_stages_overview.md)、[phase1a_oc_v4lite_cleanrun_v1_report.md](phase1a_oc_v4lite_cleanrun_v1_report.md) 和 [provenance_audit_phnode_full_clean.md](provenance_audit_phnode_full_clean.md)。当前可引用状态以 [../EXPERIMENT_PROGRESS_TRACKER.md](../EXPERIMENT_PROGRESS_TRACKER.md) 为准。

本文档是一份**研究计划**，不是实验执行手册。

它的职责是先回答四个更根本的问题：

1. 当前阶段到底要验证什么科学命题
2. 哪些证据足以支持该命题，哪些证据还不够
3. 研究对象到底是 `phnode_full` 本身，还是更广义的 structured dynamics family
4. 后续执行文档需要把哪些“计划中的协议”落成可运行实验

因此，本文档**不默认任何尚未实现的协议已经存在**。像 `v4-lite`、更强的模型本体相关 realism benchmark、某些 OOD 评估等，在这里都被视为**拟实施的研究协议**，而不是现成事实。

---

## 1. 文档定位

本计划服务于当前仓库的主研究方向：

- AUV 在 `SE(3)` 上的动力学建模
- structured port-Hamiltonian Neural ODE 与非结构化模型对比
- 长时 open-loop rollout 稳定性
- 在现实导向输入不确定性下的鲁棒性与泛化

它与后续文档的关系是：

- 本文档定义研究问题、证据层级、比较对象和解释规则
- 后续执行方案文档再负责给出具体协议、参数、命令、目录结构和报告格式

换句话说，本计划回答的是：

```text
我们要证明什么，
以及什么样的证据才算证明。
```

而不是：

```text
下一条命令该怎么跑。
```

---

## 2. 当前研究背景

当前仓库已经具备以下基础：

- `SE(3)` 上的结构化动力学模型
- 多个强弱不同的 baseline 与 ablation
- `noc / oc` 两类数据
- open-loop rollout benchmark
- clean 与 noisy-IC 训练路径
- 比较完整的结果汇总与 catalog

现有结果已经提示出两个事实：

1. structured model family 在该任务上是有竞争力的，纯黑箱模型并不稳定
2. `phnode_full` 本身仍存在训练鲁棒性与 seed fragility 问题，不能直接被当成“已被证明的最优实现”

这意味着下一阶段的研究不应再停留在“模型能不能在 clean 仿真里拟合”，而应转向一个更严格的问题：

```text
在相同信息条件下，
哪一类动力学模型更适合作为现实场景中的轨迹预报核心？
```

---

## 3. 研究目标

本计划的总体目标不是构建完整可部署系统，而是构建一套**现实导向但仍可归因于动力学模型本体**的验证框架，用来判断：

```text
当任务被定义为“给定当前状态与未来控制，预测未来轨迹”时，
结构化动力学模型是否比非结构化神经动力学模型更可靠，
并因此更适合作为真实 AUV 轨迹预报系统的 dynamics core。
```

这里的关键词有三个：

- `现实导向`
  评估条件应逐步接近部署中会遇到的状态不确定性、偏差和分布漂移
- `模型归因`
  必须尽量固定信息条件，避免把 observer、planner、sensor stack 的收益混进动力学模型结论
- `证据分级`
  不同强度的实验只支撑不同强度的结论，不能过度外推

---

## 4. 研究对象的层级

本计划必须明确区分两层研究对象，否则后续结论会混乱。

### 4.1 第一层：structured dynamics family

第一层问题是 family-level 的：

```text
与非结构化神经动力学模型相比，
结构化动力学 family 是否整体更适合 AUV 条件轨迹预报？
```

这里的“structured”不只指 `phnode_full`，也包括：

- `phnode_qforce`
- `ablate_no_lift`
- `ablate_no_mass_prior`
- 其他保留部分结构先验的模型

### 4.2 第二层：`phnode_full` 是否是当前最佳实现

第二层问题才是 model-level 的：

```text
在当前结构化 family 中，
`phnode_full` 是否真的是最强、最稳、最值得主推的实现？
```

这层问题很重要，但它从属于第一层。

如果未来结果显示：

- structured family 整体优于非结构化 baseline
- 但 `phnode_full` 仍不如 `phnode_qforce` 或 `ablate_no_lift`

那么正确结论应是：

```text
结构化 family 的研究方向是成立的，
但当前 full 版本还不是最优实现。
```

而不是把整个研究计划判成失败。

---

## 5. 核心研究问题

本计划围绕以下五个研究问题展开。

### RQ1. 受控条件下的动力学能力

在高质量状态、已知未来控制、固定扰动表示的条件下：

```text
哪类模型更擅长 long-horizon conditional rollout？
```

这是最基础的问题。如果连这一步都不能成立，就不应继续强化“现实导向更强”的叙事。

### RQ2. 现实导向输入不确定性下的退化规律

当输入状态从 clean truth 变成更接近导航估计的 noisy / biased state 后：

```text
谁掉得更少，谁更稳？
```

这里关注的是**退化规律**，而不仅仅是某个 perturbed profile 下的单点最好成绩。

### RQ3. 结构组件的真实贡献

如果结构模型表现更好，这个优势究竟来自哪里：

- `SE(3)` 几何表示
- 相对速度 `nu_r` 的语义
- split pH 结构
- 质量、阻尼、lift、actuation 等先验

这个问题决定研究结论应落在“哪种结构有价值”，而不是只做 winner ranking。

### RQ4. 结论是否在 seed 和 scenario 维度上稳定

当前仓库已经表明，单个 catastrophic seed 足以扭曲 aggregate 结论。因此必须问：

```text
观察到的优势是 family-level 现象，
还是只是在修复个别坏 seed？
```

### RQ5. 结论能走到多强

如果证据只来自仿真 benchmark，那么能支撑的最强表述是什么？

如果未来补上真实日志 replay，又能把结论升级到什么程度？

这个问题本质上是在控制论文叙事的强度。

---

## 6. 研究范围与边界

### 6.1 本计划明确要做的事情

- offline 条件轨迹预报
- open-loop dynamics rollout 对比
- 在相同信息条件下比较不同动力学模型
- 逐步引入更现实的输入不确定性
- 研究结构先验、鲁棒性、seed 稳定性和泛化

### 6.2 本计划明确不做的事情

- raw sensor 到 state 的端到端建模
- observer / filter / history encoder 作为主研究对象
- planner / MPC / closed-loop controller 集成
- 在线系统部署
- 把“可用仿真结果”直接写成“真实海试已证明”

### 6.3 边界控制的核心原则

本计划坚持以下原则：

```text
先把系统级复杂度固定住，
再比较动力学模型本身。
```

如果未来要研究 observer-augmented dynamics、闭环控制收益或真实系统联调，那应作为后续独立研究轴，而不是直接并入本计划主线。

---

## 7. 任务定义

本计划采用统一的任务定义：

```text
输入：
  当前状态 x_t
  未来控制序列 u_{t:t+H}
  可选扰动表示 d_t

输出：
  未来轨迹 x_{t+1:t+H}
```

这是一个 **conditional dynamics forecasting** 问题。

这个定义的好处是：

- 它直接对应轨迹预报任务
- 它便于控制信息条件
- 它让差异更容易归因到动力学模型，而不是外围模块

---

## 8. 信息条件设计

为了让比较具有可解释性，所有主结论都必须建立在 matched-information 的前提下。

### 8.1 必须被固定的信息

所有模型应共享：

- 相同的当前状态来源
- 相同的未来控制序列
- 相同的 current 表示
- 相同的数据切分
- 相同的训练预算
- 相同的 rollout horizon
- 相同的 seed 集合

### 8.2 当前状态的三种来源

本计划允许三类状态来源，但必须在文中明确区分：

1. `clean truth`
   适用于 Level A，用来检验纯动力学能力
2. `high-quality pseudo-truth`
   适用于现实导向但仍希望控制噪声来源的离线研究
3. `noisy navigation-like state`
   适用于 Level B，用来研究部署输入不确定性

### 8.3 未来控制的定位

本计划默认未来控制序列是已知条件输入。

这不是在假设真实系统一定能拿到长期完美控制，而是在刻意把问题收缩为：

```text
给定同样的条件输入，
谁的动力学预报更可靠？
```

如果未来要研究“控制也不确定”这一层，那应作为新的 realism 轴单列。

---

## 9. 研究假设

本计划采用以下四个假设，并同时规定其可能失败的含义。

| 假设 | 内容 | 若被支持 | 若被否定 |
|---|---|---|---|
| `H1` | structured family 在受控条件下优于非结构化模型 | 支持“结构先验确实帮助 AUV 轨迹预报” | 说明当前结构设计未形成基本优势 |
| `H2` | 在现实导向输入不确定性下，structured family 的退化更小 | 支持“结构优势不只存在于 clean 仿真” | 说明结构模型的现实鲁棒性不足 |
| `H3` | `phnode_full` 的完整 pH 结构带来额外收益 | 支持主模型叙事 | 若 `qforce` 或某个 ablation 更强，则只能支持 family-level 叙事 |
| `H4` | 观察到的优势在多 seed、多 scenario 下稳定 | 支持“方法优势是系统性的” | 若收益主要来自修复单个坏 seed，则更像训练稳定化，而不是普适优势 |

这四个假设中：

- `H1` 和 `H2` 决定研究方向是否成立
- `H3` 决定 `phnode_full` 能否成为论文主模型
- `H4` 决定结果是否可以被当成稳健证据

---

## 10. 证据层级

本计划采用三层证据体系。

### 10.1 Level A：受控条件动力学基准

设置：

- 当前状态质量高
- 未来控制已知
- 扰动表示固定
- 只比较动力学模型本体的 open-loop rollout

它回答的问题是：

```text
在理想但公平的信息条件下，
谁更擅长 AUV 轨迹预报？
```

Level A 是必要条件。若在这一层都不能形成优势，就不应推进更强主张。

### 10.2 Level B：现实导向鲁棒性基准

Level B 的目标不是“加更大的噪声”，而是引入更接近部署误差类型的输入不确定性。

它回答的问题是：

```text
当输入开始偏离 clean truth 时，
谁掉得更少、谁更稳？
```

为避免把所有现实复杂性一次性混在一起，Level B 再分成两类子证据。

#### Level B1：状态不确定性与 bias-type 扰动

这一层主要研究：

- realistic noisy-state protocol
- heading bias
- 其他导航型 bias

它仍然属于“pure dynamics under imperfect state input”的范畴。

#### Level B2：模型本体相关分布漂移

这一层研究：

- control / maneuver OOD
- current-representation uncertainty
- vehicle-parameter regime shift
- actuator mismatch

它更接近部署，但仍必须优先服务于动力学模型本体的验证。凡是会把问题推向 observer、actuator subsystem 或完整系统联调的轴，都只能作为后续补充，而不能混同为主线。

### 10.3 Level C：真实日志离线 replay

设置：

- 用同一套离线状态来源初始化所有模型
- 用真实日志中的未来控制序列做条件 rollout
- 比较短时到中时 horizon 的误差与退化规律

它回答的问题是：

```text
在真实 AUV 离线日志中，
这种优势是否仍然可见？
```

Level C 不是当前主线的必要前提，但它决定结论能否进一步升级。

---

## 11. 不同证据允许的结论强度

本计划要求结论口径与证据层级严格匹配。

### 11.1 只有 Level A 成立时

最多可以说：

```text
在 matched-information 的条件轨迹预报任务上，
某类模型具有更强的纯动力学 rollout 能力。
```

### 11.2 Level A + B 成立时

可以说：

```text
在现实导向的离线条件轨迹预报 benchmark 上，
该模型 family 更适合作为轨迹预报系统的 dynamics core。
```

这里的关键词是：

- `离线`
- `matched-information`
- `现实导向`
- `dynamics core`

### 11.3 只有 Level C 也成立时

才可以进一步说：

```text
这种优势在真实 AUV 离线日志 replay 中仍然可见，
因此它更有希望在真实场景中获得更稳定的轨迹预报表现。
```

### 11.4 当前明确不能写的口径

仅凭 Level A/B，不能直接写：

```text
ph-NODE 在真实场景中一定更好。
```

这不仅是措辞问题，也是研究严谨性的底线。

---

## 12. 数据与环境层级

本计划建议将数据与环境分成四层，而不是把所有 realism 轴混成一个实验包。

### 12.1 Tier 1：`noc`

目的：

- 验证无海流时的 vehicle dynamics 能力
- 给结构模型与黑箱模型提供最干净的比较场

Tier 1 的作用不是追求现实，而是建立基础对照。

### 12.2 Tier 2：`oc + known-current surrogate`

目的：

- 在保留 ocean current 相关状态语义的前提下，测试更接近真实任务的条件轨迹预报
- 但仍保持 current 信息的可控性

这一层必须明确标注为：

```text
known-current surrogate
```

而不是把它写成完整现实 DR。

### 12.3 Tier 3：current-representation uncertainty

这一层才研究：

- current 表示误差
- current bias
- current statistics 或 current 可用性变化

它和 Tier 2 不是同一个问题。

如果在 Tier 2 结果好，只能说明：

```text
模型在 known-current surrogate 条件下表现好。
```

不能自动推出：

```text
模型对 current representation uncertainty 也鲁棒。
```

### 12.4 Tier 4：真实日志离线 replay

这一层的价值在于给“更适合真实场景”增加外部证据，而不是替代前面三层。

---

## 13. realism 轴的定义原则

### 13.1 realism 不是“噪声越大越好”

本计划中的现实导向评估关注的是：

- 误差类型是否更贴近部署
- 模型退化是否更可解释
- 结论是否更能迁移到真实系统思维

而不是简单追求更高噪声强度。

### 13.2 realism 轴必须彼此区分

后续执行方案中，应把以下轴分开实例化：

- noisy-state
- bias-type errors
- current-representation errors
- actuator mismatch
- vehicle-parameter regime shift
- OOD maneuvers

如果这些轴被混在同一个“强压力测试”里，后面就很难解释：

```text
模型到底是在抵抗哪一类现实误差。
```

### 13.3 `v4-lite` 的定位

本计划认为，`v4-lite` 的研究价值在于：

- 它仍属于 pure dynamics 主线
- 它比 block-iid noisy IC 更接近 trajectory-consistent 的 navigation-like input
- 它有助于检查 PHNODE / structured dynamics 的结论是否依赖 block-iid noisy IC 这一简化假设

但在本研究计划中，`v4-lite` 被视为：

```text
拟实施的协议敏感性检查工具
```

而不是新的模型、observer 路线，或必须升级成主线的默认协议。

这点必须与后续执行文档保持一致。

---

## 14. OOD 的研究定义

`OOD maneuver / OOD disturbance` 是本计划的重要组成部分，但这里必须先给出研究定义，而不是只保留一个笼统标签。

本计划接受以下几类模型本体相关 OOD：

1. `control-family OOD`
   训练未见过的控制波形家族
2. `control-scale OOD`
   训练见过该类操纵，但幅值、频率或持续时间明显超出训练范围
3. `disturbance-regime OOD`
   扰动统计特征发生系统性变化
4. `parameter-regime OOD`
   等效质量、阻尼、执行器时常数等发生结构性偏移

但第一版研究不应同时展开所有 OOD 轴。

更合适的策略是：

- 第一篇研究只选 1 到 2 个最有解释力、最容易归因到动力学模型本体的 OOD 轴
- 后续执行方案再给出其明确的数据划分与协议

否则“OOD 更强”会退化成一个不可验证的笼统说法。

---

## 15. 模型比较设计

### 15.1 主模型

- `phnode_full`

### 15.2 必须保留的强结构对照

至少保留下列之一，最好保留两者：

- `phnode_qforce`
- `ablate_no_lift`
- `ablate_no_mass_prior`

原因不是为了丰富表格，而是为了回答一个关键问题：

```text
如果结构 family 表现好，
这个优势究竟属于 full PHNODE，
还是属于更一般的 structured model family？
```

### 15.3 弱结构与非结构化对照

建议保留：

- `se3_momentum_blackbox`
- `se3_accel_blackbox`
- `blackbox_fullstate`

它们分别代表：

- 部分几何或状态语义保留
- 更弱的结构先验
- 更接近纯黑箱的动力学拟合

### 15.4 最小可发表模型集

如果研究资源有限，本计划认为最小可发表集不应缩到只剩：

- `phnode_full`
- `se3_accel_blackbox`
- `blackbox_fullstate`

因为那样只能说明：

```text
full PHNODE 可能比部分弱黑箱更强。
```

却无法回答：

- family-level 结构优势是否存在
- `phnode_full` 是否真是最优实现

更合理的最小集应至少包含：

- `phnode_full`
- 一个强结构对照
- 一个弱结构对照
- 一个非结构化强 baseline

---

## 16. 评估维度

### 16.1 主指标

本计划建议把以下指标作为主报告对象：

- final position error
- trajectory position RMSE
- rotation geodesic error
- velocity RMSE
- angular RMSE
- completion / divergence rate

### 16.2 horizon 设计

研究计划层面建议关注三类时间尺度：

- 短时：约 `1s` 到 `5s`
- 中时：约 `10s`
- 长时：约 `30s` 到 `60s`

实际执行时可根据 benchmark 选择具体集合，但最终论文表述不应只依赖单一 horizon。

### 16.3 robustness 指标

现实导向研究不能只看 perturbed 条件下的绝对误差，还必须看：

- 从 clean 到 perturbed 的绝对退化量
- 从 clean 到 perturbed 的相对退化量
- 不同模型在相同 perturbation 下的 paired delta

其中最重要的是：

```text
同一模型在 matched seeds 下，
从 clean 到 perturbed 的退化是否更小。
```

### 16.4 必报切片

任何主结论至少要同时提供以下视角：

- by-horizon
- by-scenario
- by-seed
- clean replay 代价

如果缺少这些切片，很容易把局部现象误读成方法结论。

---

## 17. seed 与统计解释规则

在本项目里，seed 不是附属细节，而是核心科学变量。

### 17.1 为什么 seed 必须上升到主规则层

现有证据已经表明：

- 某些模型存在 catastrophic seed failure
- noisy training 可能主要是在修复个别坏 seed，而不是普遍改善

因此，如果不把 seed 设计写进研究计划，后续结论很容易失真。

### 17.2 本计划的 seed 规则

后续主实验必须遵守：

1. 主比较采用 matched seeds
2. all-seed aggregate 是主证据
3. per-seed delta 必报
4. problematic-seed 剔除结果只用于诊断，不用于替代主结论

### 17.3 结果解释的底线

如果某个方法的 aggregate 改善主要来自：

- 修复一个极坏 seed
- 其余大多数 seed 并未受益，甚至略退化

那么更准确的结论应是：

```text
该方法改善了训练稳定性中的某个 failure mode，
但还不能写成“整体更强”。
```

---

## 18. 研究判断规则

为了避免实验做完之后“想怎么解释都行”，本计划预先写明判断规则。

### 情况 A：Level A 下 structured family 不占优

解释：

- 当前结构先验尚未形成基础动力学优势
- 不应继续推进“现实导向更强”的主叙事

### 情况 B：Level A 占优，但 Level B 不稳

解释：

- 模型在 clean / ideal 信息条件下有潜力
- 但现实导向鲁棒性仍不足
- 研究重点应转向状态不确定性与训练稳定性

### 情况 C：structured family 在 Level A/B 都占优，但 `phnode_full` 不是 family 最优

解释：

- 支持 structured family 的研究方向
- 不支持把 full 版本写成“当前最佳实现”

此时论文主张应落在 family-level，而不是强行落在 `phnode_full`。

### 情况 D：优势主要来自修复单个坏 seed

解释：

- 这更像 training stabilization
- 不能直接等同于普适性的 robustness advantage

### 情况 E：优势只出现在 `oc + known-current surrogate`

解释：

- 只能说明模型在 surrogate 条件下更好
- 不能自动推出对 current-representation uncertainty 也更鲁棒

### 情况 F：Level C 也成立

解释：

- 可以将结论从“现实导向离线 benchmark 优势”升级到“真实日志离线 replay 中仍可见”

---

## 19. 论文叙事的推荐落点

如果研究按理想路径推进，本计划建议按以下顺序构造叙事。

### 第一层叙事

```text
在 matched-information 的条件轨迹预报任务上，
结构化动力学模型比非结构化模型更可靠。
```

### 第二层叙事

```text
当输入状态变得更接近导航估计而非 clean truth 时，
这种优势仍然保留，且退化更小。
```

### 第三层叙事

```text
这种优势在 seed、scenario 与 realism 轴上具有一定稳定性，
因此结构化模型更适合作为真实系统中的 dynamics core。
```

### 第四层叙事

只有在真实日志 replay 也成立时，才进一步说：

```text
这种优势在真实 AUV 离线日志中仍可见。
```

---

## 20. 与后续执行文档的接口

本文档之后的执行方案必须把每一个研究轴落实为明确协议。

当前推荐的配套文档分工如下：

- [phnode_realistic_validation_execution_plan.md](phnode_realistic_validation_execution_plan.md)
  负责阶段划分、依赖关系、推进顺序与里程碑
- [phase1_realistic_validation_plan.md](phase1_realistic_validation_plan.md)
  负责新版轻量 Phase-1 的决策矩阵、实施工作包、输出合同与进入扩展阶段的判断规则
- [v4_lite_protocol_spec.md](v4_lite_protocol_spec.md)
  负责 `v4-lite` 的正式协议合同、实现边界与验收标准
- [unused/phase1_comparison_matrix_legacy.md](unused/phase1_comparison_matrix_legacy.md) 与 [unused/phase1_implementation_checklist_legacy.md](unused/phase1_implementation_checklist_legacy.md)
  作为旧版 Phase-1 文档归档说明保留，当前方案不再引用其旧矩阵

对于每个 planned protocol，执行文档至少应补全：

- 数据来源与 split
- 当前状态来源
- 未来控制来源
- 扰动生成机制
- `seed` 集合
- horizon 集合
- 主指标与次指标
- 报告格式
- 哪些结果属于主证据，哪些只作诊断

对于像 `v4-lite` 这样的计划中协议，执行文档还必须进一步写清：

- 它与当前 iid noisy-IC 的唯一区别是什么
- 哪些变量必须 trajectory-consistent
- 它要回答的协议敏感性问题是什么
- 若结果为正/负，各自意味着什么

只有做到这一步，研究计划才真正变成可运行的实验计划。

---

## 21. 一句话总结

本计划的核心主张是：

```text
当前阶段不应把目标定义成“做出完整真实系统”，
而应构造一套现实导向、信息条件可控、证据层级清晰的条件轨迹预报研究框架，
用来判断 structured dynamics family 是否比非结构化模型更适合作为真实 AUV 轨迹预报系统的 dynamics core，
并进一步判断 `phnode_full` 是否是这个 family 中值得主推的当前实现。
```

这一定义比“直接追求完整现实部署”更适合当前仓库状态，也更容易形成清晰、可证伪、可发表的科学结论。
