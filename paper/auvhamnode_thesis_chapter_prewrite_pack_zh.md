# AUVHamNODE 学位论文章节开写前材料包

> 版本：2026-05-21
> 文件定位：本文件服务于学位论文章节正式开写前的材料锁定。它不是第四份写作指南，而是把符号表、claim-evidence 表、理论到代码映射、实验矩阵、图表清单和正文落点集中整理成可执行底稿。  
> 使用原则：学位论文不以篇幅压缩为优先，应完整保留理论背景、问题定义、方法推导、实现映射、实验协议、结果边界和局限性。正式正文中仍需避免保留“建议如何写”“后续再处理”等内部备忘录语气。
> 2026-05-20 更新：当前正式主稿采用正文骨架 v2；本材料包中的正文落点应映射到 v2 的 8 节结构。
> 2026-05-22 更新：第 1--2 节已完成引言式和相关建模基础返工。以下材料可继续作为方法章、理论章、验证协议和结果章的素材库，但不能把后文技术部件压缩成引言中的概念清单，也不能把旧的待办语气带入正式正文。
> 2026-05-22 图表更新：第 3 节已接入三层速度变量关系图 `drafts/figures/velocity_state_contract.pdf`；后续图表制作应优先转向机械核心和能量/功率关系图。

---

## 1. 学位论文章节定位

本章建议定位为：

> **面向长期状态预测的 AUV 结构化神经动力学建模方法。**

`AUVHamNODE` 在正文中作为方法简称和实验标识使用，不作为章标题中心词。更完整的学术表述可写为“融合 \(\mathrm{SE}(3)\) 运动学、开放式端口哈密顿机械核心与海流相对速度约定的结构化连续时间模型”。

相比短论文，学位论文章节应完整展开以下内容。这里列出的是全章材料边界，不是第 1 节引言的贡献清单：

1. 为什么长期状态预测需要把 AUV 运动建模为受控动力学学习问题，而不是普通序列外推；
2. 传统参数化模型和数据驱动模型各自提供什么价值，又在长期递推场景下留下什么缺口；
3. Fossen 被动性、Hamiltonian Neural Network、port-Hamiltonian system 与本文方法之间的关系；
4. AUVHamNODE 的状态定义、海流速度变量关系、机械核心、非保守力分解和执行器动态，这些内容应在第 3--6 节逐步展开；
5. 代码实现如何对应理论对象；
6. 实验协议如何验证结构约束，而不是只比较局部拟合误差；
7. 当前证据哪些可写、哪些需重查、哪些只能作为 provenance 或协议敏感性说明。

本章的中心句应保持为：

> **AUVHamNODE 的贡献不是把 REMUS100 工程模拟器逐项改写成严格端口哈密顿系统，而是提出面向长期状态预测的 AUV 结构化神经动力学建模方法：在受控连续时间动力学学习问题中，用已知运动结构约束模型假设空间，并用可学习项表达难以显式参数化的非线性作用。**

---

## 2. 符号表

### 2.1 坐标系与配置变量

| 符号 | 维度/空间 | 含义 | 正文落点 | 实现对应 |
|---|---:|---|---|---|
| \(n\) | - | 惯性坐标系 / 导航坐标系 | 问题定义 | 数据中的位置和海流方向 |
| \(b\) | - | 体坐标系 | 问题定义 | 速度、角速度、执行器力主要在体坐标系表达 |
| \(x\) | \(\mathbb{R}^3\) | 惯性系位置或控制块内相对位置 | 问题定义、运动学 | state `[0:3]` |
| \(R\) | \(SO(3)\) | 从体坐标系到惯性系的旋转矩阵 | 问题定义、SE(3) 运动学 | state `[3:12]` row-major |
| \(q=(x,R)\) | \(\mathbb{R}^3\times SO(3)\) | AUV 配置 | 方法章 | `q = state[:, :12]` |
| \(\widehat{\omega}\) | \(\mathfrak{so}(3)\) | 角速度对应的反对称矩阵 | SE(3) 运动学 | `dR = R \hat\omega` 的等价实现 |

### 2.2 速度与海流

| 符号 | 维度/空间 | 含义 | 正文落点 | 实现对应 |
|---|---:|---|---|---|
| \(v_b\) | \(\mathbb{R}^3\) | 体坐标系总体线速度 | 速度变量关系 | 数据速度线速度槽 |
| \(\omega\) | \(\mathbb{R}^3\) | 体坐标系角速度 | 速度变量关系、运动学 | state velocity `[15:18]` |
| \(\nu_b=[v_b,\omega]\) | \(\mathbb{R}^6\) | 数据空间体坐标系总体速度 | 问题定义 | dataset velocity convention `body_total` |
| \(v_c^n\) | \(\mathbb{R}^3\) | 惯性系海流速度 | 速度变量关系、海流通道 | optional carried channel |
| \(v_c^b=R^\top v_c^n\) | \(\mathbb{R}^3\) | 体坐标系海流速度 | 速度变量关系 | `_body_current` |
| \(v_r=v_b-v_c^b\) | \(\mathbb{R}^3\) | 相对水线速度 | 速度变量关系、水动力建模 | model velocity linear slot |
| \(\nu_r=[v_r,\omega]\) | \(\mathbb{R}^6\) | 模型空间相对水速度 | 方法核心 | ODE state velocity slot |
| \(c\) | variable | 可选上下文，包括体坐标系海流、总体速度或深度上下文 | 广义力分支 | `actuation_current_feature`, `dj_current_feature`, `z_ref` |

必须在正文中固定的关系：

\[
v_c^b=R^\top v_c^n,
\qquad
\nu_r=
\nu_b-
\begin{bmatrix}
R^\top v_c^n\\0
\end{bmatrix},
\qquad
\nu_b=
\nu_r+
\begin{bmatrix}
R^\top v_c^n\\0
\end{bmatrix}.
\]

### 2.3 控制、执行器与增强状态

| 符号 | 维度/空间 | 含义 | 正文落点 | 实现对应 |
|---|---:|---|---|---|
| \(u_c\) | \(\mathbb{R}^{m}\) | 执行器命令 | 执行器滞后 | carried channel |
| \(u_a\) | \(\mathbb{R}^{m}\) | 执行器实际状态 | 执行器滞后、广义力 | `u_actual` slot |
| \(T_\theta\) | \(\mathbb{R}^{m}_{>0}\) | 执行器时间常数 | 执行器滞后 | `T_actuator` |
| \(z_{ref}\) | \(\mathbb{R}\) | 可选绝对深度上下文 | 势能/深度上下文 | optional carried channel |
| \(s\) | variable | 数据空间状态 | 训练协议 | stored dataset state |
| \(y\) | variable | ODE 模型空间增强状态 | 方法与实现 | `[x,R,nu_r,u_a,u_c,v_c^n,z_ref]` |

执行器动态建议写为：

\[
\dot u_a=T_\theta^{-1}(u_c-u_a),
\qquad
\dot u_c=0.
\]

海流和深度上下文在单个积分窗口内写为：

\[
\dot v_c^n=0,
\qquad
\dot z_{ref}=0.
\]

### 2.4 机械核心与能量

| 符号 | 维度/空间 | 含义 | 正文落点 | 实现对应 |
|---|---:|---|---|---|
| \(M_\theta\) | \(\mathbb{R}^{6\times6}\) | 正定质量矩阵 | 机械核心 | inverse of `mass_inv` |
| \(M_\theta^{-1}\) | \(\mathbb{R}^{6\times6}\) | 正定逆质量矩阵 | 参数化 | `mass_inv = L L^T` |
| \(p_r=M_\theta\nu_r\) | \(\mathbb{R}^6\) | 相对动量 | 机械核心 | `_momentum` |
| \(V_\theta(q)\) | \(\mathbb{R}\) | 标量势能 | 机械能与保守力 | `V_net`, `V_linear` |
| \(H_\theta(q,p_r)\) | \(\mathbb{R}\) | 机械存储函数 | 能量命题 | `energy` |
| \(D_\theta(\xi)\) | \(\mathbb{R}^{6\times6}\) | 正定/半正定耗散矩阵 | 非保守力分解 | `D_net`, `damping` |
| \(J_\theta(\xi)\) | \(\mathbb{R}^{6\times6}\) | 斜对称零功率耦合 | 非保守力分解 | `J_net`, `lift` |
| \(\tau_\theta(\nu_r,u_a,c)\) | \(\mathbb{R}^6\) | 可学习广义力分支 | 执行器与残差力 | `B_net` |
| \(\xi\) | variable | 阻尼和 lift 的输入特征 | 非保守力分解 | `nu_r` plus optional current/total velocity |

机械存储函数建议固定为：

\[
H_\theta(q,p_r)
=
\frac12p_r^\top M_\theta^{-1}p_r
+V_\theta(q),
\qquad
p_r=M_\theta\nu_r.
\]

非保守力分解建议固定为：

\[
f_\theta^{nc}
=
-D_\theta(\xi)\nu_r
+J_\theta(\xi)\nu_r
+\tau_\theta(\nu_r,u_a,c).
\]

---

## 3. 数据空间、模型空间和输出空间

### 3.1 数据空间

数据空间保存总体速度：

\[
s=
[\Delta x,R,\nu_b,u_a,u_c,v_c^n,z_{ref}],
\]

其中 \(\nu_b\) 是体坐标系总体速度。这个约定应作为数据契约写入问题定义章，不能把数据直接说成保存相对速度。

### 3.2 模型空间

ODE 模型空间使用相对水速度：

\[
y=
[x,R,\nu_r,u_a,u_c,v_c^n,z_{ref}].
\]

数据态到 ODE 态：

\[
\mathrm{to\_ode\_state}(s):
\quad
\nu_b\mapsto
\nu_r=\nu_b-
\begin{bmatrix}
R^\top v_c^n\\0
\end{bmatrix}.
\]

ODE 态到数据态：

\[
\mathrm{to\_data\_state}(y):
\quad
\nu_r\mapsto
\nu_b=\nu_r+
\begin{bmatrix}
R^\top v_c^n\\0
\end{bmatrix}.
\]

### 3.3 运动学和广义力使用不同速度

必须在正文中反复保持以下句子：

> **位姿运动学由总体速度推进，水动力相关广义力由相对水速度参数化。**

数学上：

\[
\dot x=R v_b=R(v_r+R^\top v_c^n)=Rv_r+v_c^n,
\qquad
\dot R=R\widehat{\omega}.
\]

这个区分是海流场景下方法合理性的基础。

---

## 4. Claim-Evidence 表

| ID | 可写结论 | 推荐正文表述 | 证据来源 | 状态 | 落点 | 边界 |
|---|---|---|---|---|---|---|
| C1 | 数据空间与模型空间速度变量关系成立 | 数据保存总体速度，模型内部转换为相对水速度，输出时恢复总体速度 | `AUVHamNODE.py`, `train_utils.py` | current | 问题定义、方法 | 只在有海流时发生速度平移；无海流为 no-op |
| C2 | SE(3) 运动学被显式编码 | 模型显式使用 \(\dot x=Rv_b\)、\(\dot R=R\hat\omega\) | `AUVHamNODE.py` | current | 方法 | 普通 ODE solver 不保证数值严格在 SO(3) 上 |
| C3 | 机械核心具有 pH 风格能量结构 | 正定质量、标量势能、正定耗散和斜对称零功率项构成开放式机械核心 | `AUVHamNODE.py` 与理论推导 | current | 方法、理论命题 | 不等于完整闭合严格 pH 系统 |
| C4 | 静水机械核心满足能量平衡 | 限定条件下 \(\dot H_\theta=-\nu_r^\top D_\theta\nu_r+\nu_r^\top\tau_\theta\) | 理论推导 | current | 命题 | 需要列出连续时间、\(R\in SO(3)\)、\(D\succeq0\)、\(J=-J^\top\) 等条件 |
| C5 | \(D/J/\tau\) 分解有物理诱导意义 | 分解区分耗散功率、零功率耦合和外部广义力 | `D_net`, `J_net`, `B_net` | current | 方法、讨论 | 不保证唯一恢复真实水动力项 |
| C6 | 执行器滞后是模型组成部分 | 控制命令先经一阶执行器状态，再参与广义力分支 | `T_actuator`, `du_act` | current | 方法 | 未把执行器能量建模为闭合 Hamiltonian 子系统 |
| C7 | 海流是外源携带通道 | 海流在短积分窗口内作为上下文携带，用于速度转换和可选特征 | `v_c^n` state slot | current | 方法、讨论 | 不写成闭合海流动力学或 Hamiltonian 环境 |
| C8 | canonical 表是默认展示口径 | 默认论文图表应优先使用 canonical rollout views | `docs/oc_result_selection_policy.md` | current | 实验协议 | canonical 不等于 current evidence |
| C9 | 旧 `phnode_full clean seed42/46` 异常不能作为脆弱性证据 | catalog 时代异常应作为环境漂移/provenance 案例，而非模型缺陷结论 | `docs/provenance_audit_phnode_full_clean.md` | stale_environment_drift | 讨论或附录 | 旧 11 m 五 seed 均值不可作为当前模型脆弱性证据 |
| C10 | current-main aligned `phnode_full clean` 基线约为 0.6767 m | 若需要同口径 clean baseline，应使用 cleanrun v1 / current-main aligned 结果 | provenance audit | current for audit scope | 实验边界 | 不自动替代所有 canonical 结果 |
| C11 | clean 下 `phnode_qforce` 是当前 all-seed 强基线 | 在 OC clean setting 下可作为强结构化基线 | `EXPERIMENT_PROGRESS_TRACKER.md` | current | 结果 | 写具体数值前需从 current evidence 表重新导出 |
| C12 | clean 下 `ablate_no_lift` 是否 PHNODE family 最稳需重查 | 可写为已有证据提示其强，但存在 seed43 异常风险 | progress tracker, phase1a report | needs_recheck | 结果、讨论 | 不写成无条件最终结论 |
| C13 | noisy training 不是普适增强 | noisy IC training 的收益与模型结构强耦合 | follow-up/progress tracker | needs_recheck/current with caveat | 结果、讨论 | 不再沿用“修复 phnode_full seed46”作为主因果链 |
| C14 | `ablate_no_mass_prior` 是当前最稳定受益于 noisy training 的结构模型 | 可作为 noisy training 结构相关性的代表证据 | progress tracker, provenance audit impact table | current | 结果 | 具体数值需重新按 current 表导出 |
| C15 | `ablate_bu_only` 退化说明 actuation-conditioning 应保留 | 移除速度/海流条件的执行器分支会损害长期性能 | progress tracker | current | 消融 | 需要配合当前结果表报告 |
| C16 | coupled damping 有价值 | 耦合阻尼优于仅对角阻尼的证据可保留 | progress tracker | current | 消融 | 具体差异应按结果表说明 |
| C17 | mass prior 尚未显示不可替代性 | 当前数据规模和训练设置下，质量先验不是唯一性能来源 | progress tracker | current | 消融、讨论 | 不等于质量先验无用 |
| C18 | `v4_lite` 是协议敏感性方向 | `v4_lite` 可写成现实化扰动协议和诊断压力 | phase1a report | current as diagnostic | 讨论、未来工作 | 不写成主训练协议胜利 |
| C19 | 尚未完成真实海试泛化验证 | 当前主证据来自 REMUS100 风格仿真和受控 benchmark | repo docs | current | 局限性 | 不声称真实海域泛化已验证 |

---

## 5. 理论对象到代码对象映射

| 理论对象 | 代码对象 | 结构保证 | 论文写法 | 避免写法 |
|---|---|---|---|---|
| \(M_\theta^{-1}\succ0\) | `Minv_L`, `mass_inv` | Cholesky 型 \(LL^\top\) 正定构造 | 正定逆质量矩阵参数化 | 任意质量矩阵 |
| \(M_\theta\) | inverse of `mass_inv` | 由正定逆质量得到 | 相对动量 \(p_r=M_\theta\nu_r\) | 逐项等同 REMUS 全质量模型 |
| \(V_\theta(q)\) | `V_net`, `V_linear` | 标量势能 | 保守力来自标量势能梯度 | 任意配置力都来自真实势能 |
| \(D_\theta(\xi)\succeq0\) | `D_net`, `damping` | lower-triangular 或 diagonal 构造正定阻尼 | 耗散分支提供非正功率 | 学到唯一真实阻尼 |
| \(J_\theta(\xi)=-J_\theta^\top\) | `J_net`, `lift` | 由上三角参数生成斜对称矩阵 | 零功率耦合或 lift 分支 | 真实 lift 项被唯一恢复 |
| \(\tau_\theta(\nu_r,u_a,c)\) | `B_net` | 可学习六维广义力 | 执行器状态和上下文驱动的可学习广义力 | 标准 pH 输入矩阵 \(G(q)u\) |
| \(T_\theta>0\) | `T_actuator_raw`, `T_actuator` | softplus 保证正时间常数 | 一阶执行器滞后 | 完整执行器能量模型 |
| \(\nu_b\to\nu_r\) | `to_ode_state`, `_shift_linear_velocity(sign=-1)` | 有海流时减去 \(R^\top v_c^n\) | 数据态到模型态转换 | 数据直接保存相对速度 |
| \(\nu_r\to\nu_b\) | `to_data_state`, `_shift_linear_velocity(sign=+1)` | 有海流时加回 \(R^\top v_c^n\) | 预测回到数据空间评估 | 只在模型空间评估全部 KPI |
| \(\dot x=Rv_b\) | `dx = R @ nu_total[:3]` | 使用总体速度推进位置 | 位姿运动学由总体速度驱动 | 位置由相对水速度直接推进 |
| \(\dot R=R\hat\omega\) | `dR` construction | 连续时间切空间相容 | SO(3) 结构化向量场 | 数值积分严格保持 SO(3) |
| 能量诊断 \(H_\theta\) | `energy` | 机械能语义用于结构模型 | 内部物理一致性诊断 | 与所有黑箱模型完全同质比较 |
| 数据契约校验 | `validate_dataset_config` | 要求 `body_total` | 数据保存总体速度 | 数据契约不明 |
| rollout 指标 | `rollout_benchmark_engine.py`, reporting | completion, failure, observable KPI, diagnostics | 长期递推 benchmark | 只报告单步 RMSE |

---

## 6. 实验矩阵

### 6.1 模型矩阵

| 模型名 | 论文显示名 | 角色 | 主要结构 | 写作注意 |
|---|---|---|---|---|
| `phnode_full` | AUVHamNODE / pH NODE Full | 主模型 | SE(3), mass, scalar V, split D/J/B, actuator, current convention | 不沿用旧 bad-seed 脆弱性叙事 |
| `phnode_merged_force` | pH NODE Merged Force | 结构化基线 | pH 机械核心，合并非保守力 | 用于验证 D/J/B 分解价值 |
| `phnode_qforce` | pH NODE q-Force | 结构化基线 | 保留部分 pH 结构，用通用配置力替代标量势能 | clean setting 强基线 |
| `se3_momentum_blackbox` | SE(3) Momentum Black-Box | 几何/动量基线 | SE(3) 和动量坐标，黑箱动量动力学 | 用于隔离能量结构贡献 |
| `se3_accel_blackbox` | SE(3) Acceleration Black-Box | 几何基线 | SE(3) 运动学，黑箱加速度 | 用于隔离几何结构贡献 |
| `blackbox_fullstate` | Full-State Black-Box | 黑箱基线 | 完全无结构状态导数 | 用于显示无结构长期递推风险 |
| `ablate_no_mass_prior` | No Mass Prior | 消融 | 去掉 REMUS 质量初始化 | 当前 noisy 受益重要模型 |
| `ablate_diag_damping` | Diagonal Damping | 消融 | 阻尼仅对角 | 验证耦合阻尼 |
| `ablate_no_lift` | No Lift | 消融 | 去掉斜对称 lift | clean 结论需重查 |
| `ablate_bu_only` | B(u) Only | 消融 | 广义力仅依赖执行器状态 | 验证 actuation-conditioning |

### 6.2 训练协议和评估协议

| 类别 | 协议/profile | 用途 | 正文状态 |
|---|---|---|---|
| clean training | `clean` | 基础训练条件 | 主结果可写，但需 evidence status |
| noisy IC training | `nominal_train` | 当前 profile-based noisy 训练 | 主线可写为结构相关性实验 |
| clean eval | `clean` | 干净初值评估 | 主评估 profile |
| nominal eval | `nominal_eval` | 标准导航不确定性 | 主评估 profile |
| degraded eval | `degraded_eval` | 更强初值扰动 | stress profile |
| heading-biased eval | `heading_biased_eval` | 航向偏差压力 | stress profile |
| current-biased eval | `current_bias_eval` | 海流估计偏差压力 | 仅有海流扩展线适用，当前主实验需谨慎 |
| v4-lite protocol | `v4_lite` | 现实化扰动协议方向 | 写成 protocol sensitivity diagnostic |

### 6.3 实验问题矩阵

| 实验编号 | 实验问题 | 最小模型集合 | 训练/eval | 关键指标 | 论文结论类型 |
|---|---|---|---|---|---|
| E1 | 模型是否能拟合局部动力学 | core models | block prediction | position, rotation, velocity, actuator loss | 局部拟合能力 |
| E2 | 结构是否改善长期递推 | blackbox, SE3, pH, full | clean/noisy, rollout | completion, failure reason, final pos/rot/vel | 主证据 |
| E3 | 速度变量关系在海流下是否合理 | oc vs noc, current feature ablations if available | clean and OC eval | total velocity, position, relative velocity diagnostic | 速度变量关系证据 |
| E4 | noisy IC training 是否鲁棒 | selected strong models | clean vs noisy train, multiple eval profiles | degradation vs clean, completion, final error | 结构相关鲁棒性 |
| E5 | D/J/B 分解是否有价值 | `phnode_full`, `phnode_merged_force`, `phnode_qforce` | matched eval | long rollout and diagnostics | 结构消融 |
| E6 | mass prior 是否关键 | `phnode_full`, `ablate_no_mass_prior` | clean/noisy | rollout, convergence, seed spread | 消融与讨论 |
| E7 | coupled damping 是否关键 | `phnode_full`, `ablate_diag_damping` | matched eval | long rollout, failure | 消融 |
| E8 | lift 是否稳定收益 | `phnode_full`, `ablate_no_lift` | clean/noisy | rollout, seed sensitivity | needs recheck |
| E9 | actuation-conditioning 是否必要 | `phnode_full`, `ablate_bu_only` | matched eval | rollout failure and terminal error | current 消融结论 |
| E10 | 内部物理诊断是否一致 | energy-semantics models | selected rollouts | energy span, \(P_D\), \(P_J\), SO(3) drift | 诊断证据 |
| E11 | v4-lite 是否改变排序 | `phnode_full`, `ablate_no_mass_prior`, `ablate_no_lift` | v4-lite decision package | 60s position, completion, clean replay cost | protocol sensitivity |

### 6.4 指标分类

| 类别 | 指标 | 是否主 KPI | 说明 |
|---|---|---:|---|
| 完成状态 | completion rate, completed-to-horizon, failure reason | 是 | 长期递推首先看是否完成 |
| 可观测空间误差 | final position error, rotation geodesic, total velocity error, depth error | 是 | 论文结果表核心 |
| 误差增长 | median/p90/p95 over time | 是 | 显示长期误差累积 |
| 模型空间诊断 | relative velocity error | 否 | 用于解释水动力建模，不替代总体速度 KPI |
| 内部物理诊断 | energy span, energy delta, \(P_D\), \(P_J\), SO(3) determinant/orthogonality | 否 | 解释结构一致性 |
| 训练诊断 | training history, heldout/block eval, solver failure | 辅助 | 用于解释异常 seed 和 provenance |

---

## 7. 图表清单

### 7.1 核心图

| 图号 | 图名 | 内容 | 目的 | 当前状态 |
|---|---|---|---|---|
| Fig. 1 | 总体框架图 | 数据态、模型态、ODE solve、输出评估 | 建立全章结构 | 可与 Fig. 2 合并或后续单独制作 |
| Fig. 2 | 速度变量关系图 | \(\nu_b\)、\(\nu_r\)、\(v_c^n\)、\(R^\top v_c^n\)、\(\dot x=Rv_b\) | 解释海流下变量契约 | 已完成：`drafts/figures/velocity_state_contract.pdf`，已接入主稿第 3 节 |
| Fig. 3 | AUVHamNODE 机械核心图 | \(M,V,D,J,\tau,u_c\to u_a\) | 解释能量流和非保守力分解 | 下一张优先制作 |
| Fig. 4 | 模型结构阶梯图 | full-state blackbox 到 pH full | 展示结构逐级增强 | 待制作 |
| Fig. 5 | 长期 rollout 误差增长图 | position, rotation, total velocity median/p90 | 展示长期稳定性 | 待 current evidence 表导出后制作 |
| Fig. 6 | failure reason 统计图 | completed, pred divergence, solver failure, NaN/Inf, velocity/depth violation | 防止只看完成样本误差 | 待 current evidence 表导出后制作 |
| Fig. 7 | 噪声初值鲁棒性图 | clean/nominal/degraded/heading-biased degradation | 展示鲁棒性 | 待 current evidence 表导出后制作 |
| Fig. 8 | 能量和 SO(3) 诊断图 | \(H(t)\), \(P_D\), \(P_J\), determinant/orthogonality | 内部物理一致性 | 待模型诊断口径确认后制作 |

### 7.2 核心表

| 表号 | 表名 | 内容 | 目的 |
|---|---|---|---|
| Table 1 | 符号表 | 本文件第 2 节核心变量 | 保证符号一致 |
| Table 2 | 理论声称范围表 | 可以写和不应写 | 降低审稿风险 |
| Table 3 | 理论到实现映射表 | 理论对象、代码对象、结构保证 | 连接方法和实现 |
| Table 4 | 模型结构对比表 | 每个模型保留哪些结构 | 支撑消融逻辑 |
| Table 5 | 实验矩阵表 | 模型、训练、评估、指标 | 保证实验可复现 |
| Table 6 | 长期 rollout 汇总表 | completion, final pos/rot/vel, failure | 主结果表 |
| Table 7 | evidence status 表 | current, stale, needs_recheck | 防止旧证据误用 |
| Table 8 | v4-lite protocol sensitivity 表 | matched protocol comparison | 作为诊断，不作主胜利 |

---

## 8. 正文写作顺序

学位论文不需要从摘要和引言开始。建议实际写作顺序如下：

1. **受控状态表示与海流速度约定**：先写 \(\nu_b,\nu_r,v_c^n,u_a,u_c,y,s\)，这是全章地基。
2. **结构化连续时间动力学模型**：依次补强 \(\mathrm{SE}(3)\) 运动学、机械核心、非保守力分解、执行器和外源通道、相对动量动力学。
3. **能量平衡与理论边界**：给出机械核心定义、命题条件、证明、完整增强系统边界和数值积分边界。
4. **训练目标与验证协议**：把理论对象映射到结构保持参数化，写清控制块损失、长期递推、结构消融和 evidence status 口径。
5. **相关建模基础**：当前主稿第 2 节已完成 v1；后续仅随方法章符号、功率边界和新增引用做一致性修订，不扩展成泛泛综述。
6. **讨论**：主动写非闭合 pH、非严格李群积分、非唯一水动力辨识、仿真证据和真实海试泛化边界。
7. **研究问题、摘要和本章小结**：最后修订，使其准确概括已经完成的正文。第 1 节应建立问题张力、研究缺口和方法切入点，不应把第 3--6 节的概念压缩为一串术语。
8. **实验结果**：按 evidence status 过滤后再填入结果；若篇幅较大，优先另起实验章或独立结果节。

---

## 9. 开写前仍需补齐或确认的材料

这些不是继续写指南，而是正式正文需要的证据准备：

1. 从 `canonical_rollout_summary_long.csv` 和 `canonical_rollout_outcomes_long.csv` 导出当前可引用的主结果表，并排除或显式标注 `evidence_status != current` 的 run。
2. 为 `phnode_full clean` 统一使用 current-main / cleanrun v1 对齐口径，避免旧 `seed42/46` 异常进入主结论。
3. 决定 `ablate_no_lift clean seed43` 是重跑、剔除说明，还是在正文中标注 needs recheck。
4. 将 `v4_lite` 结果定位为 protocol sensitivity diagnostic，不并入主 canonical 结论表，除非后续补齐 provenance 和异常修复。
5. 速度变量关系图已完成并接入第 3 节；下一步生成机械核心图，这张图仍应早于实验结果图完成。
6. 为每个结果图记录数据来源、过滤条件、eval profile、horizon 和 metric key，避免后续图表不可追溯。
7. 若计划把离散 RNN/TCN/MLP 作为额外基线，需要先确认是否已有实现和结果；没有结果时不要在正文主实验中承诺。

---

## 10. 正式正文避免项

以下内容可以留在本文件或写作指南中，但不能原样进入学位论文正文：

- “本章应……”
- “建议后续……”
- “这里可以放一张图……”
- “不要写成……”
- “这个结果需要之后再查……”
- “作为 preliminaries 的安排……”
- “案例参数应在仿真章节给出……”

对应处理方式：

1. 有学术信息的句子，转化为定义、假设、命题、协议、边界或局限性；
2. 只有写作调度功能的句子，删除；
3. 不确定证据状态的句子，写入讨论或附录，不放入主结论。

---

## 11. 可直接作为正文种子的段落

### 11.1 问题定义种子段落

六自由度 AUV 的配置由惯性系位置 \(x\in\mathbb{R}^3\) 和旋转矩阵 \(R\in SO(3)\) 给出，其中 \(R\) 将体坐标系向量映射到惯性坐标系。在有海流环境中，航行器相对于惯性系的总体速度与相对于周围水体的相对速度并不相同。本文将数据空间中的体坐标系总体速度记为 \(\nu_b=[v_b,\omega]\)，将模型空间中的相对水速度记为 \(\nu_r=[v_r,\omega]\)。若惯性系海流速度为 \(v_c^n\)，则体坐标系海流为 \(R^\top v_c^n\)，并有 \(\nu_r=\nu_b-[R^\top v_c^n;0]\)。该速度变量关系使位姿运动学和水动力建模分别使用其物理上对应的速度变量。

### 11.2 方法总述种子段落

AUVHamNODE 的构造从几何、能量和非保守力三个层面约束连续时间向量场。几何层面，模型显式使用 SE(3) 运动学推进位置和姿态；能量层面，模型通过正定质量矩阵和标量势能定义机械存储函数；非保守力层面，模型将水动力和执行器相关效应分解为正定耗散、斜对称零功率耦合和可学习广义力分支。该结构并不声称完整增强系统构成闭合严格端口哈密顿系统，而是在机械核心中保留端口哈密顿风格的能量平衡关系。

这段适合放在方法构造部分，不宜直接作为第 1 节的引言贡献段。引言中应先说明为什么长期递推需要受控动力学结构约束，再自然引出本文方法。

### 11.3 实验协议种子段落

实验评估的目标不是单纯比较短时预测误差，而是检验结构约束是否改善长期递推稳定性、噪声初值鲁棒性和物理一致性。因此，本文采用分层评估协议：首先在控制块预测任务中评估局部动力学拟合能力；随后在 held-out trajectory 上进行长期递推，报告 completion rate、failure reason 和终端位置、姿态、总体速度误差；进一步在不同初值噪声 profile 下评估鲁棒性，并通过结构消融、能量诊断和 SO(3) 诊断分析模型内部一致性。

### 11.4 局限性种子段落

本文的理论结论限定于开放式机械核心及其结构化向量场。当前模型将海流作为单个积分窗口内的外源携带通道，并未建立闭合海流场动力学；执行器通过一阶滞后状态进入可学习广义力分支，也未作为完整能量子系统建模；旋转矩阵虽然满足连续时间切空间结构，但当前数值实现并非严格李群积分器。此外，当前主证据来自 REMUS100 风格仿真与受控 rollout benchmark，真实海试数据下的泛化能力仍需进一步验证。

---

## 12. 下一步最小行动

当前正式主稿已建立并完成第 1--3 节的主要正文返工。下一步只做两件事：

1. 继续扩写 `paper/drafts/auvhamnode_thesis_chapter_zh.tex` 第 4 节“结构化连续时间动力学模型”，优先补强相对动量动力学、非保守广义力分支和执行器通道之间的论证衔接。
2. 运行或编写结果导出脚本，将 current evidence 的 rollout 主表导出为论文表格底稿。

第一件事推进方法正文，第二件事锁定证据。不要再继续扩展指南，除非正式写作中发现具体缺口。
