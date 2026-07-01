# 新对话审查 prompt：AUVHamNODE 方法章新增两张插图 + 图注

> 用途：在**新开的对话**里独立审查 §1.3 `fig:method-architecture-overview` 与 §1.4 `fig:fossen-role-mapping` 两张图及其 caption 草稿。
> 使用方法：把下面 `===== PROMPT 开始 =====` 与 `===== PROMPT 结束 =====` 之间的整段复制到新会话。
> 设计意图：红线为硬约束；美学/范围/版面/落点等限制放宽，鼓励 AI 主动质疑并提出更大胆的改进。
> 生成日期：2026-06-07。

---

===== PROMPT 开始 =====

# 任务：严格审查 AUVHamNODE 学位论文方法章「新增的两张插图 + 图注」（放手提出改进）

## 你的角色（四重身份，全程用中文）
1) 高水平 AUV/机器人动力学建模专家（端口哈密顿、SE(3) 几何力学、Neural ODE、Fossen 六自由度水动力）；
2) 严格的中文博士学位论文审稿专家；
3) 资深中文学术论文编辑；
4) 顶刊级科研可视化专家。
基调：严格校准（不夸大、不漏报），结论以证据为准——**务必亲自看渲染出的 PNG 图像**，不要只读脚本臆测。

## 环境与路径
- 仓库根：/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5
- conda 环境 mytorch1。注意：`conda run -n mytorch1 ...` 在本机沙箱常被拒绝；如需重渲染，直接用绝对路径
  `/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin/python <脚本>`。
- 主稿：paper/drafts/auvhamnode_thesis_chapter_zh.tex（方法在 §1.1–1.7；§1.3 标签 sec:state-current-contract，§1.4 sec:fossen-to-structure，§1.5 sec:structured-continuous-model，§1.6 energy-power-properties，§1.7 training-baselines-protocol）。

## 待审查对象：本轮新增的两张图 + 图注草稿
### 图 A（拟接入 §1.3 开头，作全章方法路线图）fig:method-architecture-overview
- 脚本 paper/drafts/figures/make_method_architecture_overview.py
- 产物 paper/drafts/figures/method_architecture_overview.{pdf,png,svg}
- 设计：方案 A「流程主轴 + 三类语义着色」。左上入口 s →(速度契约 𝒯_{d→m})→ 增强态 y；中央结构化连续时间向量场 ẏ=F_θ(y)，含虚线「开放六自由度机械核心」与 7 个模块（SE(3) kinematics / Mechanical storage / Coadjoint coupling / Conservative force / Dissipation / Zero-power coupling / Force port）；核心框外下方为外源开放端口（Actuator lag / Ocean current / Depth context），竖直短虚线条件化接入；底部三类语义图例。**已刻意不画数值积分(RK4)与评估细节**（核心方法图，验证由 §1.8 结果图承载）。

### 图 B（拟接入 §1.4，一行结构映射之后、表 tab:fossen-to-auvhamnode-roles 之前）fig:fossen-role-mapping
- 脚本 paper/drafts/figures/make_fossen_role_mapping.py
- 产物 paper/drafts/figures/fossen_role_mapping.{pdf,png,svg}
- 设计：左→右双栏「经典 Fossen 角色 → 保结构组件」。5 行映射（Inertia→逆质量；Coriolis→分叉为 Coadjoint+Skew coupling；Damping→Dissipation；Restoring→Potential；External force→Force port），底部 before/after 锚（Fossen 显式合力式 → 开放结构形式 ṗ_r=…），底部红线脚注（开放系统、非闭合 PH、τ_θ≠G(q)u）。这是对一份早期文档里某张图的**红线安全重制**。

> 注：两段图注目前只是草稿（见文末），尚未写入 .tex；请把它们当作待审稿件一并评估。

## 必读（开工前，用于对照与校验）
1. 方法节 §1.3–§1.7，重点公式：eq:velocity-contract、eq:complete-augmented-vector-field、eq:hamiltonian、eq:mass-parameterization、eq:coadjoint-term、eq:nonconservative-force、eq:potential-force、eq:momentum-dynamics、eq:actuator-dynamics、eq:fossen；以及表 tab:fossen-to-auvhamnode-roles。
2. 既有 4 张图脚本（house style 基线 + 各自职责边界，新图绝不能与之重复）：
   figures/make_velocity_state_contract.py（fig:velocity-state-contract，§1.3 速度契约细节）
   figures/make_model_definition_overview.py（fig:model-definition-overview，§1.5 向量场内部接线）
   figures/make_mechanical_core_power_structure.py（fig:mechanical-core-power-structure，§1.6 机械核心功率）
   figures/make_section8_two_level_evidence.py（fig:s8-two-level-evidence，§1.8 结果）
3. 红线来源：AGENTS.md「Paper Boundaries」段 + paper/README.md。
4. 早期参考文档：/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/邱春华文件/AUVHamNODE文档.pdf（图1/图2）。**这是修正前的旧框架**：它把执行器写成 Gu/Bu 标准输入矩阵、把完整系统写成闭合端口哈密顿 (J−R)∇H+Gu、把升力归为"保守力"——这些都违反现稿红线。图 B 正是它的红线安全纠正版；请核对图 B 是否彻底纠正了这三处、并据此判断图 A 是否也无残留。

## 红线（**不可放宽**；美学/范围/版面/落点可放宽，方法学红线不可碰）
- 完整航行器–执行器–环境系统**不是**闭合/严格端口哈密顿系统；PH 语义只限开放六自由度机械核心。
- τ_θ / B_net **不是**标准输入矩阵 G(q)u；执行器是外部广义力端口。
- 海流只作分段常值外源，**非**闭合哈密顿环境子系统。
- 普通 ODE 求解器**不**严格保持 SO(3)；正交性是诊断量，不可暗示精确守恒。
- 不得宣称 v4_lite 为更优最终训练协议。
- 既有约定：虚线框=机械核心；虚线箭头=条件化输入；图内英文标签 + 中文 caption；宽度 ≤138mm。

## 审查维度（请逐图 + 逐图注覆盖）
1. **科学/方法学正确性**：图是否忠实表示模型？符号、归类、功率角色、零功率/耗散/端口语义有无细微错误或误导？图与正文（公式/表）是否一致？
2. **红线合规**：有无任何越界表述或暗示（尤其 PH 闭合性、Gu、海流、SO(3)）。
3. **图注质量**：caption 是否准确匹配图、\eqref 锚点是否正确、边界声明是否完整且不啰嗦、中文是否学术规范（无内部备忘语气）。
4. **可视化设计（顶刊判据）**：清晰度、信息密度、对齐、配色与色盲安全、留白、是否"PPT 感"。
5. **分工与去冗余**：与既有 4 张图、以及两张新图彼此之间，是否职责清晰、无重复。
6. **叙事有效性**：图 A 是否"一眼讲清核心方法"；图 B 是否清楚传达"角色转换"且突出三处纠正。
7. **接入落点**：§1.3 / §1.4 落点是否最优；如不优，请给替代方案与理由。

## 本次特别放宽 / 请发挥主观能动性
- 不要只做"挑错式"微调评审。**欢迎质疑既有决定**：构图方案、内容取舍（例如图 A 删掉积分/评估是否恰当）、版面、落点、house style 乃至配色与标签语言约定，只要有更优解就大胆提出。
- 可以提出更大胆的重构想法（含全新构图），并说明收益与代价。
- 如有助于说明，你可以动手**重渲染或产出改进 mockup** 来佐证你的建议（用上面的 python 绝对路径）。
- 输出请把发现按严重度分层：
  · **必改**（红线/科学错误/图文不符）
  · **应改**（清晰度、密度、对齐、caption 精度）
  · **可选/大胆**（更优构图或叙事的进取建议）
  每条给出"问题—证据（指向图中具体位置）—建议改法"。

## 工作流
- 先**通读 + 看图 + 出结构化审查报告**；除非我确认，不要直接覆盖已定稿的脚本/图（如要示范可另存 mockup 或在报告中给出 diff 思路）。
- 若判断需要重画，先说清"改什么、为什么、预期效果"，等我确认后再落地。

---
## 附 1：图 A 当前 caption 草稿（§1.3，尚未入 .tex，请评估）
AUVHamNODE 核心方法总览。数据态 \(s\) 经速度契约 \(\mathcal{T}_{d\to m}\)（式~\eqref{eq:velocity-contract}）转换为增强模型态 \(y\)，作为结构化连续时间向量场 \(\dot y=F_\theta(y)\)（式~\eqref{eq:complete-augmented-vector-field}）所演化的状态。配色区分三类语义：蓝色为参数化构造保证的结构先验（\(\SE(3)\) 运动学、机械存储中正定逆质量 \(M_\theta^{-1}\succ0\)（式~\eqref{eq:mass-parameterization}）、余伴随零功率耦合（式~\eqref{eq:coadjoint-term}）、半正定耗散 \(D_\theta\succeq0\) 与斜对称零功率耦合 \(J_\theta=-J_\theta^\top\)（式~\eqref{eq:nonconservative-force}）），金色标记保结构函数类内由网络学习的形状（机械存储、势能、耗散、零功率耦合与广义力端口）。虚线框为开放式六自由度机械核心；执行器（式~\eqref{eq:actuator-dynamics}）、海流 \(v_c^n\) 与深度 \(z_{\mathrm{ref}}\) 为分段常值外源开放端口，位于核心之外、经虚线条件化箭头接入，其中 \(\tau_\theta\) 是外部广义力端口而非固定输入矩阵 \(G(q)u\)。普通 ODE 求解器不严格保持 \(\SO(3)\)，正交性与能量仅作内部诊断。该图为全章方法路线图，速度契约、向量场内部模块与机械核心功率关系的细节分别见图~\ref{fig:velocity-state-contract}、图~\ref{fig:model-definition-overview} 与图~\ref{fig:mechanical-core-power-structure}。

## 附 2：图 B 当前 caption 草稿（§1.4，尚未入 .tex，请评估）
Fossen 型功率角色到 AUVHamNODE 结构化组件的角色转换。左列为经典向量（唯象）Fossen 项，右列为几何、保结构的可训练组件：广义质量 \(\to\) 正定逆质量 \(M_\theta^{-1}\succ0\)（式~\eqref{eq:mass-parameterization}）；科氏--向心项 \(\to\) 体坐标余伴随耦合 \(\mathrm{ad}^{*}_{\nu_r}p_r\)（几何诱导、固定、零功率，式~\eqref{eq:coadjoint-term}）与可学习斜对称耦合 \(J_\theta=-J_\theta^\top\)（数据诱导的升力/横向类零功率作用）；阻尼 \(\to\) 半正定耗散 \(D_\theta\succeq0\)；恢复力 \(\to\) 标量势能 \(V_\theta\) 诱导的保守广义力 \(f_\theta^V\)（式~\eqref{eq:potential-force}）；外部广义力 \(\to\) 可学习广义力端口 \(\tau_\theta\)（式~\eqref{eq:nonconservative-force}）。蓝色表示参数化构造保证的结构先验，金色标记保结构函数类内由网络学习的形状。底部对照 Fossen 显式合力式（式~\eqref{eq:fossen}）与开放结构形式 \(\dot p_r=\mathrm{ad}^{*}_{\nu_r}p_r+f_\theta^V-D_\theta\nu_r+J_\theta\nu_r+\tau_\theta\)（式~\eqref{eq:momentum-dynamics}），功率配对统一使用相对水速度 \(\nu_r\)。该角色转换只约束未知项的组织方式，不声称唯一辨识真实水动力来源；\(\tau_\theta\) 是外部广义力端口而非固定输入矩阵 \(G(q)u\)，完整航行器--执行器--环境系统作为开放系统处理，不构成闭合端口哈密顿系统。各角色的适用边界详见表~\ref{tab:fossen-to-auvhamnode-roles}。

===== PROMPT 结束 =====
