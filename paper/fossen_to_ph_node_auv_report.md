# 从 Fossen AUV 模型到 Port-Hamiltonian Neural ODE：结构保持水下航行器动力学建模

**文档目标**：本文系统整理标准 Fossen AUV 动力学模型、其 port-Hamiltonian（pH）表达、真实 REMUS100 工程代码中的 pH 结构，以及最终作为创新点的 PH-NODE / AUVHamNODE 模型。本文的主线不是把三类模型平行摆放，而是从经典解析模型逐步过渡到结构保持的神经动力学模型：

$$
\boxed{
\text{Fossen 模型}
\rightarrow
\text{能量结构分析}
\rightarrow
\text{pH 形式}
\rightarrow
\text{REMUS100 工程复杂性}
\rightarrow
\text{PH-NODE 创新模型}
}
$$

---

## 摘要

水下航行器（AUV）动力学具有强非线性、强耦合、欠驱动、海流扰动和复杂水动力等特点。经典 Fossen 模型以刚体动力学、水动力阻尼、重力—浮力回复力和推进器广义力为基础，是 AUV 建模与控制中的标准形式。然而，Fossen 模型本身并不仅仅是一组力和力矩平衡方程，它隐含了清晰的能量结构：惯性项定义储能，Coriolis/centripetal 项不做功，阻尼项耗散能量，重力—浮力项来自势能，推进器通过功率端口注入能量。

基于这一结构，标准 Fossen 模型在满足若干自然物理条件时，可以严格整理为 port-Hamiltonian 形式。pH 形式显式揭示了系统的 Hamiltonian、反对称互联、半正定耗散以及输入输出功率端口，从而为无源性分析、能量整形控制和结构保持学习提供统一语言。

进一步分析真实 REMUS100 Python 动力学模型可以发现，其连续时间车辆核心仍然保留 Fossen/pH 的基本结构：质量矩阵由刚体质量和附加质量组成，Coriolis 项由 skew 矩阵构造，阻尼和横流阻力具有耗散性，重浮力项具有势能来源，控制力以外部端口形式注入能量。但是完整工程仿真器还包含海流、相对速度变换、执行器一阶滞后、四元数姿态、数值积分和饱和裁剪等因素，因此不能把整个程序不加区分地称为标准连续 pH 系统。

这些工程复杂性自然引出 PH-NODE 的建模思想：与其在解析模型中手工构造所有未知水动力，不如直接将 pH 结构作为硬约束，把质量、势能、阻尼、无功 lift 项和执行器到广义力映射参数化为可学习函数。AUVHamNODE 正是在 $SE(3)$ 上构造的 open port-Hamiltonian Neural ODE。它不是简单由 Fossen 模型变换而来，而是从第一性原理和几何力学出发，将 Fossen-pH 的能量结构推广为可学习、结构保持的神经动力学模型。

本文最终观点是：

$$
\boxed{
\text{PH-NODE 不是对 Fossen 模型的黑箱替代，而是 Fossen-pH 能量结构的几何化、学习化和工程化推广。}
}
$$

---

## 目录

1. [研究动机：为什么从 Fossen 走向 PH-NODE](#1-研究动机为什么从-fossen-走向-ph-node)
2. [标准 Fossen AUV 模型](#2-标准-fossen-auv-模型)
3. [Fossen 模型的能量结构](#3-fossen-模型的能量结构)
4. [Fossen 模型到 pH 形式的严格转换条件](#4-fossen-模型到-ph-形式的严格转换条件)
5. [Fossen-to-pH 的严格推导](#5-fossen-to-ph-的严格推导)
6. [REMUS100 工程模型中的 pH 结构验证](#6-remus100-工程模型中的-ph-结构验证)
7. [传统 Fossen-pH 建模的局限](#7-传统-fossen-ph-建模的局限)
8. [从 pH 到 PH-NODE：结构保持学习思想](#8-从-ph-到-ph-node结构保持学习思想)
9. [AUVHamNODE 的 $SE(3)$ 几何 pH 模型](#9-auvhamnode-的-se3-几何-ph-模型)
10. [PH-NODE 的可学习结构与能量保证](#10-ph-node-的可学习结构与能量保证)
11. [从 Fossen-pH 到 PH-NODE 的统一视角](#11-从-fossen-ph-到-ph-node-的统一视角)
12. [对建模、辨识和控制的启示](#12-对建模辨识和控制的启示)
13. [结论](#13-结论)
14. [附录 A：主要符号表](#附录-a主要符号表)
15. [附录 B：REMUS100 代码模块与 pH 解释](#附录-bremus100-代码模块与-ph-解释)
16. [附录 C：AUVHamNODE 代码模块与数学结构对应关系](#附录-cauvhamnode-代码模块与数学结构对应关系)

---

# 1. 研究动机：为什么从 Fossen 走向 PH-NODE

AUV 动力学建模通常面临两类矛盾。

第一类矛盾是**物理解释性与建模复杂性**之间的矛盾。Fossen 模型具有清晰物理意义，每个项都可以解释为惯性、Coriolis/centripetal、阻尼、重力—浮力或控制广义力。但是在真实 REMUS100 这样的工程模型中，阻尼、升阻力、横流阻力、舵面力、螺旋桨力、海流和执行器滞后都会引入大量经验参数和非线性表达。模型越真实，解析表达越复杂。

第二类矛盾是**数据驱动能力与物理结构保持**之间的矛盾。普通 Neural ODE 或黑箱神经网络可以拟合复杂动力学，但如果不加约束，它可能学出违反能量耗散、无源性和刚体几何结构的动力学。例如，它可能在无输入情况下生成能量，或者把本应无功的 gyroscopic 项学成耗散或增能项。

因此，一个自然问题是：

$$
\boxed{
\text{能否在保留 Fossen 模型能量结构的同时，引入神经网络学习未知动力学？}
}
$$

本文采用如下递进逻辑回答该问题：

1. 从标准 Fossen AUV 模型出发，分析其隐含的能量结构；
2. 证明在自然物理条件下，Fossen 模型可以严格写成 pH 形式；
3. 用真实 REMUS100 代码说明工程模型确实保留 pH 核心结构，但同时暴露出传统解析建模的复杂性；
4. 最后引出 PH-NODE：先规定 pH / 几何力学结构，再用神经网络学习未知质量、势能、阻尼、lift 和执行器映射。

换言之，本文的核心不是“用神经网络替代物理模型”，而是：

$$
\boxed{
\text{从固定参数物理模型走向结构保持的可学习物理模型。}
}
$$

---

# 2. 标准 Fossen AUV 模型

## 2.1 状态变量

标准 6-DOF AUV 模型通常采用：

$$
\eta =
\begin{bmatrix}
 x & y & z & \phi & \theta & \psi
\end{bmatrix}^\top,
$$

$$
\nu =
\begin{bmatrix}
 u & v & w & p & q & r
\end{bmatrix}^\top.
$$

其中：

- $\eta$ 是惯性坐标系下的位置和姿态；
- $\nu$ 是体坐标系下的线速度和角速度；
- $[u,v,w]^\top$ 是 surge、sway、heave 速度；
- $[p,q,r]^\top$ 是 roll、pitch、yaw 角速度。

运动学方程为：

$$
\boxed{
\dot\eta = J(\eta)\nu.
}
$$

其中 $J(\eta)$ 是体坐标速度到惯性坐标位姿导数的映射矩阵。若使用 Euler 角，则 $J(\eta)$ 在 $\theta=\pm\pi/2$ 附近存在奇异性。

## 2.2 动力学方程

标准 Fossen 动力学写成：

$$
\boxed{
M\dot\nu + C(\nu)\nu + D(\nu)\nu + g(\eta)=\tau.
}
$$

其中：

| 符号 | 含义 |
|---|---|
| $M$ | 总质量矩阵，包括刚体惯性和附加质量 |
| $C(\nu)$ | Coriolis/centripetal 矩阵，包括刚体和附加质量贡献 |
| $D(\nu)$ | 水动力阻尼矩阵，可包含线性和非线性阻尼 |
| $g(\eta)$ | 重力—浮力回复力和力矩 |
| $\tau$ | 推进器、舵面或外部作用产生的广义力 |

该模型看起来是“力平衡”形式，但从能量角度看，每一项都有明确角色：

$$
\boxed{
M\dot\nu：\text{惯性储能变化},
\quad
C(\nu)\nu：\text{无功 gyroscopic 项},
\quad
D(\nu)\nu：\text{耗散},
\quad
g(\eta)：\text{保守回复力},
\quad
\tau：\text{端口输入}.
}
$$

这正是将 Fossen 模型转换为 pH 形式的基础。

---

# 3. Fossen 模型的能量结构

## 3.1 动能

若质量矩阵满足：

$$
M=M^\top>0,
$$

则可以定义动能：

$$
\boxed{
T(\nu)=\frac12\nu^\top M\nu.
}
$$

该条件在物理 AUV 模型中非常自然，因为 $M$ 通常由刚体质量矩阵 $M_{RB}$ 与附加质量矩阵 $M_A$ 组成：

$$
M=M_{RB}+M_A.
$$

## 3.2 势能

若重力—浮力回复力 $g(\eta)$ 来自势能 $P(\eta)$，则应存在标量函数 $P(\eta)$，使得：

$$
\boxed{
J^\top(\eta)\frac{\partial P}{\partial\eta}(\eta)=g(\eta).
}
$$

这是一个非常关键的条件。它说明体坐标下的回复力 $g(\eta)$ 是惯性坐标势能梯度经过 $J^\top(\eta)$ 映射后的结果。

于是总能量可以定义为：

$$
\boxed{
H(\eta,\nu)=\frac12\nu^\top M\nu+P(\eta).
}
$$

## 3.3 能量导数

对 $H$ 求导：

$$
\dot H
=\nu^\top M\dot\nu + P_\eta^\top\dot\eta.
$$

由 Fossen 模型：

$$
M\dot\nu=\tau-C(\nu)\nu-D(\nu)\nu-g(\eta).
$$

由运动学：

$$
\dot\eta=J(\eta)\nu.
$$

代入得：

$$
\begin{aligned}
\dot H
&=\nu^\top\tau
-\nu^\top C(\nu)\nu
-\nu^\top D(\nu)\nu
-\nu^\top g(\eta)
+P_\eta^\top J(\eta)\nu \\
&=\nu^\top\tau
-\nu^\top C(\nu)\nu
-\nu^\top D(\nu)\nu
+\nu^\top\left(J^\top(\eta)P_\eta-g(\eta)\right).
\end{aligned}
$$

如果满足：

$$
\nu^\top C(\nu)\nu=0,
$$

$$
J^\top(\eta)P_\eta=g(\eta),
$$

则：

$$
\boxed{
\dot H=\nu^\top\tau-\nu^\top D(\nu)\nu.
}
$$

若 $D(\nu)$ 不对称，则只需其对称部分：

$$
D_s(\nu)=\frac{D(\nu)+D^\top(\nu)}{2}
$$

满足：

$$
D_s(\nu)\ge 0.
$$

此时能量平衡为：

$$
\boxed{
\dot H=\nu^\top\tau-\nu^\top D_s(\nu)\nu\le \nu^\top\tau.
}
$$

这正是 AUV 模型的无源性表达。

---

# 4. Fossen 模型到 pH 形式的严格转换条件

标准 port-Hamiltonian 系统写成：

$$
\boxed{
\dot x=\left[\mathcal J(x)-\mathcal R(x)\right]\nabla H(x)+\mathcal G(x)u,
}
$$

$$
\boxed{
y=\mathcal G^\top(x)\nabla H(x).
}
$$

其中：

$$
\mathcal J(x)=-\mathcal J^\top(x),
\qquad
\mathcal R(x)=\mathcal R^\top(x)\ge 0.
$$

Fossen 模型能够严格转换为 pH 形式，需要满足以下条件。

## 4.1 质量矩阵正定

$$
\boxed{
M=M^\top>0.
}
$$

该条件保证可以定义正定动能和动量变量。

## 4.2 运动学映射局部有效

若使用 Euler 角，需要：

$$
\boxed{
\det J(\eta)\ne 0.
}
$$

这意味着 pH 表达在 Euler 角坐标图内局部成立。若希望避免姿态奇异性，应转向四元数或 $SE(3)$ 几何表达。

## 4.3 Coriolis 项不做功

需要：

$$
\boxed{
\nu^\top C(\nu)\nu=0.
}
$$

更强且常见的条件是：

$$
\boxed{
C(\nu)=-C^\top(\nu).
}
$$

这说明 Coriolis/centripetal 项只改变动量方向，不改变机械能。

## 4.4 阻尼项耗散

需要：

$$
\boxed{
D_s(\nu)=\frac{D(\nu)+D^\top(\nu)}{2}\ge 0.
}
$$

若 $D$ 有反对称部分：

$$
D_a(\nu)=\frac{D(\nu)-D^\top(\nu)}{2},
$$

由于：

$$
\nu^\top D_a(\nu)\nu=0,
$$

它不耗散能量，可以并入 pH 的反对称互联结构。

## 4.5 回复力保守

需要存在势能 $P(\eta)$，使得：

$$
\boxed{
J^\top(\eta)P_\eta(\eta)=g(\eta).
}
$$

等价地，在局部单连通区域中，向量场 $J^{-\top}(\eta)g(\eta)$ 应为某个标量函数的梯度。

## 4.6 输入输出功率共轭

若输入是广义力 $\tau$，则输出应取体速度：

$$
\boxed{
y=\nu.
}
$$

输入功率为：

$$
\boxed{
P_{in}=\tau^\top\nu.
}
$$

若实际输入是舵角、螺旋桨转速或执行器命令，则需要通过非线性映射生成广义力：

$$
\tau=\Gamma(\nu,u_{act}).
$$

此时最干净的 pH 端口仍是 $\tau$ 到 $\nu$，而执行器映射应作为外部端口映射或扩展系统处理。

## 4.7 条件汇总

| Fossen 项 | pH 条件 | 物理含义 |
|---|---|---|
| $M$ | $M=M^\top>0$ | 能定义正定动能 |
| $J(\eta)$ | 局部非奇异 | 位姿速度映射有效 |
| $C(\nu)\nu$ | $\nu^\top C(\nu)\nu=0$ | Coriolis/centripetal 项不做功 |
| $D(\nu)\nu$ | $D_s(\nu)\ge0$ | 阻尼耗散能量 |
| $g(\eta)$ | $\exists P: J^\top P_\eta=g$ | 回复力来自势能 |
| $\tau$ | $\tau^\top\nu$ 为输入功率 | 输入输出功率共轭 |

---

# 5. Fossen-to-pH 的严格推导

## 5.1 体坐标准动量

定义体坐标准动量：

$$
\boxed{
p_b=M\nu.
}
$$

则：

$$
\nu=M^{-1}p_b.
$$

取 pH 状态：

$$
\boxed{
x_b=
\begin{bmatrix}
\eta\\
p_b
\end{bmatrix}.
}
$$

Hamiltonian 取为：

$$
\boxed{
H_b(\eta,p_b)=\frac12 p_b^\top M^{-1}p_b+P(\eta).
}
$$

于是：

$$
\frac{\partial H_b}{\partial p_b}=M^{-1}p_b=\nu,
$$

$$
\frac{\partial H_b}{\partial \eta}=P_\eta.
$$

## 5.2 阻尼分解

令：

$$
D(\nu)=D_s(\nu)+D_a(\nu),
$$

其中：

$$
D_s=D_s^\top\ge0,
\qquad
D_a=-D_a^\top.
$$

## 5.3 构造 pH 矩阵

若 $C(\nu)=-C^\top(\nu)$，则可以取：

$$
\boxed{
\mathcal J_b(\eta,p_b)=
\begin{bmatrix}
0 & J(\eta)\\
-J^\top(\eta) & -C(\nu)-D_a(\nu)
\end{bmatrix},
}
$$

$$
\boxed{
\mathcal R_b(\eta,p_b)=
\begin{bmatrix}
0&0\\
0&D_s(\nu)
\end{bmatrix},
}
$$

$$
\boxed{
\mathcal G_b=
\begin{bmatrix}
0\\
I_6
\end{bmatrix}.
}
$$

其中：

$$
\nu=M^{-1}p_b.
$$

显然：

$$
\mathcal J_b=-\mathcal J_b^\top,
\qquad
\mathcal R_b=\mathcal R_b^\top\ge0.
$$

## 5.4 pH 形式

于是系统可以写成：

$$
\boxed{
\begin{bmatrix}
\dot\eta\\
\dot p_b
\end{bmatrix}
=
\left[
\mathcal J_b(\eta,p_b)-\mathcal R_b(\eta,p_b)
\right]
\begin{bmatrix}
H_\eta\\
H_{p_b}
\end{bmatrix}
+
\mathcal G_b\tau.
}
$$

输出为：

$$
\boxed{
y=\mathcal G_b^\top\nabla H_b=H_{p_b}=\nu.
}
$$

## 5.5 等价性验证

第一行给出：

$$
\dot\eta=J(\eta)H_{p_b}=J(\eta)\nu.
$$

第二行给出：

$$
\dot p_b
=-J^\top H_\eta-C(\nu)\nu-D_a(\nu)\nu-D_s(\nu)\nu+\tau.
$$

利用：

$$
J^\top P_\eta=g(\eta),
$$

并且：

$$
D=D_s+D_a,
$$

得：

$$
\dot p_b=-g(\eta)-C(\nu)\nu-D(\nu)\nu+\tau.
$$

由于 $p_b=M\nu$ 且 $M$ 为常值矩阵：

$$
\dot p_b=M\dot\nu.
$$

因此：

$$
\boxed{
M\dot\nu+C(\nu)\nu+D(\nu)\nu+g(\eta)=\tau.
}
$$

这与原始 Fossen 模型完全一致。

## 5.6 能量平衡

沿 pH 系统轨线：

$$
\begin{aligned}
\dot H_b
&=\nabla H_b^\top\dot x_b \\
&=\nabla H_b^\top\mathcal J_b\nabla H_b
-\nabla H_b^\top\mathcal R_b\nabla H_b
+\nabla H_b^\top\mathcal G_b\tau.
\end{aligned}
$$

因为 $\mathcal J_b=-\mathcal J_b^\top$：

$$
\nabla H_b^\top\mathcal J_b\nabla H_b=0.
$$

又因为 $\mathcal G_b^\top\nabla H_b=\nu$，得到：

$$
\boxed{
\dot H_b=-\nu^\top D_s(\nu)\nu+\nu^\top\tau.
}
$$

这说明 Fossen-pH 从输入 $\tau$ 到输出 $\nu$ 是无源的。

---

# 6. REMUS100 工程模型中的 pH 结构验证

本节分析用户提供的 `remus100_core.py`。该代码不是一个抽象教科书模型，而是一个包含动力学、控制器和仿真器的 REMUS100 AUV 工程模型。因此需要区分两个层面：

$$
\boxed{
\text{连续时间车辆核心动力学}
}
$$

和：

$$
\boxed{
\text{包含积分器、执行器、海流、四元数归一化和裁剪的完整仿真程序。}
}
$$

前者可以整理成 pH 结构；后者不能不加区分地称为标准连续 pH 系统。

## 6.1 质量矩阵

代码中构造了刚体质量矩阵 $M_{RB}$、附加质量矩阵 $M_A$，并定义：

$$
\boxed{
M=M_{RB}+M_A.
}
$$

这与 Fossen 模型完全一致。其物理意义是：

$$
\boxed{
\text{刚体惯性} + \text{附加质量} = \text{总惯性储能矩阵。}
}
$$

因此可以定义：

$$
T=\frac12\nu_r^\top M\nu_r.
$$

这里 $\nu_r$ 是相对水流速度，而非绝对速度。

## 6.2 相对速度核心动力学

代码核心函数 `_relative_rhs` 中，动力学结构可以概括为：

$$
\boxed{
M\dot\nu_r
=
\tau_{control}
+
\tau_{liftdrag}
+
\tau_{crossflow}
-
(C+D)\nu_r
-
g(\eta).
}
$$

其中：

- $C=C_{RB}+C_A$；
- $D$ 是经验阻尼矩阵；
- $g(\eta)$ 是重力—浮力回复力；
- $\tau_{control}$ 来自舵面和螺旋桨；
- $\tau_{liftdrag}$ 是机体升阻力；
- $\tau_{crossflow}$ 是横流阻力。

这说明真实 REMUS100 工程模型并没有脱离 Fossen/pH 框架，而是在该框架上叠加了更丰富的水动力项。

## 6.3 Coriolis 项

代码中 `_m2c` 使用 skew 矩阵构造 Coriolis 矩阵。结构上可写成：

$$
C(\nu_r)=
\begin{bmatrix}
0 & -S(p_v)\\
-S(p_v) & -S(p_\omega)
\end{bmatrix},
$$

其中 $S(\cdot)$ 是叉乘矩阵。该结构满足：

$$
\boxed{
C(\nu_r)=-C^\top(\nu_r)
}
$$

或至少满足：

$$
\boxed{
\nu_r^\top C(\nu_r)\nu_r=0.
}
$$

因此 Coriolis 项可以进入 pH 的反对称互联矩阵。

## 6.4 经验阻尼矩阵

代码中的 $D$ 是对角阻尼矩阵，其中 surge 和 sway 阻尼还包含速度相关指数因子：

$$
D_{11}\leftarrow D_{11}e^{-3U_r},
\qquad
D_{22}\leftarrow D_{22}e^{-3U_r}.
$$

只要各时间常数、阻尼比和质量矩阵对角项为正，该阻尼矩阵满足：

$$
\boxed{
D(\nu_r)=D^\top(\nu_r)>0.
}
$$

因此它可以作为 pH 中的耗散项。

## 6.5 重力—浮力回复力

代码中的 `_g_forces(phi, theta)` 给出重力—浮力回复力和力矩。其结构与标准 Fossen 重浮力项一致，可以由重力和浮力势能导出。

因此满足：

$$
\boxed{
\exists P(\eta):\quad J^\top(\eta)P_\eta(\eta)=g(\eta).
}
$$

在 REMUS100 代码参数中，重量和浮力相等：

$$
W=B,
$$

并且重心与浮心在垂向上存在偏移，因此主要产生横摇和纵摇恢复力矩。

## 6.6 lift-drag 项

`_force_lift_drag` 计算机体升力和阻力。其功率性质可以分解为：

$$
\tau_{liftdrag}
=
-D_{LD}(\nu_r)\nu_r
+
S_{LD}(\nu_r)\nu_r,
$$

其中：

$$
D_{LD}=D_{LD}^\top\ge0,
\qquad
S_{LD}=-S_{LD}^\top.
$$

阻力部分耗散能量，升力部分与速度正交、不做功。因此 lift-drag 不破坏 pH 结构，只是需要将其分解为耗散项和无功项。

## 6.7 cross-flow drag 项

`_cross_flow_drag` 对航体长度方向进行离散积分。其典型形式为：

$$
Y_{cf}=-\int k|v_r+xr|(v_r+xr)\,dx,
$$

$$
N_{cf}=-\int kx|v_r+xr|(v_r+xr)\,dx.
$$

对应功率为：

$$
\begin{aligned}
 v_rY_{cf}+rN_{cf}
&=-\int k|v_r+xr|(v_r+xr)^2\,dx \\
&\le 0.
\end{aligned}
$$

因此 cross-flow drag 可以写成非线性半正定阻尼：

$$
\boxed{
\tau_{crossflow}=-D_{cf}(\nu_r)\nu_r,
\qquad
D_{cf}=D_{cf}^\top\ge0.
}
$$

## 6.8 控制力是端口输入

代码中的 `_control_forces` 根据舵角、舵面状态和螺旋桨 RPM 计算广义力：

$$
\tau_{control}=\Gamma(\nu_r,u_{actual}).
$$

该项可以向系统注入能量，也可以从系统抽取能量。因此它不应被放入内部耗散矩阵 $\mathcal R$，而应作为端口输入：

$$
\boxed{
u_r^\top\tau_{control}}
$$

是控制端口功率。

如果直接以 $\tau_{control}$ 为输入，则输出为 $\nu_r$。如果以舵角和 RPM 为输入，则需要将 $\Gamma(\nu_r,u_{actual})$ 看作非线性执行器—力映射。

## 6.9 海流与相对速度

代码中区分总速度和相对水流速度：

$$
\nu_r=\nu-\nu_c.
$$

当存在海流时，位置运动学不再是简单的：

$$
\dot\eta=J(\eta)\nu_r,
$$

而是包含海流漂移项。此时能量平衡中会出现环境功率交换。因此海流不应被粗暴塞进内部耗散项，而应解释为外生扰动或环境端口。

## 6.10 完整仿真器不是标准连续 pH 系统

`Remus100Simulator` 使用四元数状态、RK4 或 Euler 积分、每步四元数归一化、执行器状态裁剪以及相对速度与总速度之间的转换。这些操作对工程仿真非常合理，但它们不是标准连续 pH 向量场的一部分。

因此，准确结论是：

$$
\boxed{
\text{REMUS100 的连续车辆核心动力学可以整理为 pH 结构；完整仿真程序需要额外的离散、约束或混合系统解释。}
}
$$

---

# 7. 传统 Fossen-pH 建模的局限

Fossen-to-pH 提供了非常清晰的物理和能量结构，但在真实 AUV 建模中仍面临若干局限。这些局限不是否定 Fossen-pH，而是自然引出 PH-NODE 的动机。

## 7.1 Euler 角局部坐标限制

标准 Fossen 模型常使用：

$$
\eta=[x,y,z,\phi,\theta,\psi]^\top.
$$

该表达直观，但 Euler 角在 $\theta=\pm\pi/2$ 附近存在奇异性。真实仿真器通常倾向使用四元数或旋转矩阵来避免数值问题。

这说明下一步更自然的建模空间不是局部 Euler 坐标，而是：

$$
\boxed{
SE(3)=\mathbb R^3\rtimes SO(3).
}
$$

## 7.2 水动力项高度经验化

真实 AUV 的阻尼、升阻力、横流阻力、舵面力和螺旋桨力往往依赖经验公式。不同航速、姿态、海流和工况下，这些经验公式可能存在偏差。

因此，即使 pH 结构正确，具体函数形式仍可能不准确。

## 7.3 海流和相对速度带来环境功率交换

使用相对速度 $\nu_r$ 能更好描述水动力，但海流使系统不再是封闭机械系统。此时应把海流作为外部通道，而不是强行解释为内部阻尼或保守项。

## 7.4 执行器不是理想广义力源

理论 pH 推导中最自然的输入是广义力：

$$
u^\top\tau.$$

但真实 AUV 输入通常是：

$$
[\delta_r^c,\delta_s^c,n^c]^\top,
$$

即舵角命令和螺旋桨转速命令。它们经过执行器滞后和流体动力映射后才形成广义力。因此，端口映射本身也是需要建模或学习的对象。

## 7.5 未建模效应难以完全解析表达

包括但不限于：

- 复杂三维流动；
- 舵面与艇体耦合；
- 推进器尾流影响；
- 高攻角升力非线性；
- 海流与航体姿态耦合；
- 传感器和执行器动态误差。

这些因素都提示我们：

$$
\boxed{
\text{需要一种既保持 pH 能量结构、又能从数据中学习未知动力学的模型。}
}
$$

这就是 PH-NODE 的出发点。

---

# 8. 从 pH 到 PH-NODE：结构保持学习思想

## 8.1 Fossen-to-pH 是事后结构化

Fossen-to-pH 的逻辑是：

$$
\boxed{
(M,C,D,g,\tau)
\longrightarrow
\text{检查 pH 条件}
\longrightarrow
(H,\mathcal J,\mathcal R,\mathcal G).
}
$$

也就是说，先有解析动力学，再验证它是否满足能量结构。

## 8.2 PH-NODE 是事前结构约束

PH-NODE 的逻辑相反：

$$
\boxed{
(H_\theta,\mathcal J_\theta,\mathcal R_\theta,\mathcal G_\theta)
\longrightarrow
\dot x_\theta.
}
$$

其中 $\theta$ 是神经网络参数。模型从一开始就被限制在满足 pH 结构的函数类中。

因此：

$$
\boxed{
\text{PH-NODE 的创新不在于抛弃物理，而在于把物理结构作为神经网络的硬约束。}
}
$$

## 8.3 从“验证无源性”到“保证无源性”

Fossen-to-pH 中需要检查：

$$
M>0,
\qquad
C=-C^\top,
\qquad
D_s\ge0,
\qquad
g=J^\top P_\eta.
$$

PH-NODE 中则通过参数化直接保证：

$$
M_\theta^{-1}=LL^\top>0,
$$

$$
D_\theta(\nu_r)=L_D L_D^\top\ge0,
$$

$$
S_\theta(\nu_r)=A-A^\top,
$$

$$
\text{回复力来自 }V_\theta(q)\text{ 的梯度。}
$$

这使得训练过程中模型始终保持能量结构。

---

# 9. AUVHamNODE 的 $SE(3)$ 几何 pH 模型

AUVHamNODE 是一个定义在 $SE(3)$ 上的 open port-Hamiltonian Neural ODE。它不是从 Fossen 方程逐项转换得到的，而是从第一性原理和几何力学出发构造 pH core。

## 9.1 状态空间

定义：

$$
q=(x,R)\in SE(3),
$$

其中：

- $x\in\mathbb R^3$ 是惯性坐标位置；
- $R\in SO(3)$ 是 body 到 inertial 的旋转矩阵。

相对水流速度为：

$$
\nu_r=
\begin{bmatrix}
v_r\\
\omega
\end{bmatrix}
\in\mathbb R^6.
$$

动量定义为：

$$
\boxed{
p_r=M\nu_r.
}
$$

AUVHamNODE 的增强状态包含：

$$
[x(3),R(9),\nu_r(6),u_{actual}(m),u_{cmd}(m),v_c^n(3?),z_{ref}(1?)].
$$

其中 $v_c^n$ 和 $z_{ref}$ 是可选外生通道或上下文变量。

## 9.2 Hamiltonian

AUVHamNODE 的机械储能为：

$$
\boxed{
H_\theta(q,p_r)=\frac12p_r^\top M_\theta^{-1}p_r+V_\theta(q).
}
$$

等价地，由于 $p_r=M_\theta\nu_r$：

$$
H_\theta(q,\nu_r)=\frac12\nu_r^\top M_\theta\nu_r+V_\theta(q).
$$

这与 Fossen-pH 的：

$$
H_F(\eta,p_b)=\frac12p_b^\top M^{-1}p_b+P(\eta)
$$

在结构上完全同构。

## 9.3 几何运动学

无海流时：

$$
\dot x=Rv_r,
$$

$$
\dot R=R\widehat\omega.
$$

有海流时，线速度使用总速度：

$$
 v=v_r+R^\top v_c^n,
$$

因此：

$$
\dot x=R(v_r+R^\top v_c^n)=Rv_r+v_c^n.
$$

这说明海流被视为外生环境通道，而不是内部耗散项。

## 9.4 动量方程

AUVHamNODE 的核心动量方程可写成：

$$
\boxed{
\dot p_r
=
\operatorname{ad}_{\nu_r}^*p_r
-g_{V_\theta}(q)
-D_\theta(\nu_r)\nu_r
+S_\theta(\nu_r)\nu_r
+\tau_\theta.
}
$$

其中：

$$
\operatorname{ad}_{\nu_r}^*p_r
=
\begin{bmatrix}
 p_v\times\omega\\
 p_\omega\times\omega+p_v\times v_r
\end{bmatrix}.
$$

该项就是刚体几何力学中的 coadjoint / gyroscopic 项。它对应 Fossen 模型中的 Coriolis/centripetal 结构，但不需要显式构造 $C(\nu)$ 矩阵。

势能梯度项 $g_{V_\theta}(q)$ 由 $V_\theta(q)$ 自动微分得到。因此保守性由势能函数本身保证。

## 9.5 非保守力结构

非保守项为：

$$
\boxed{
f_{nc}
=-D_\theta(\nu_r)\nu_r
+S_\theta(\nu_r)\nu_r
+\tau_\theta.
}
$$

其中：

- $D_\theta(\nu_r)$ 是正定或半正定阻尼；
- $S_\theta(\nu_r)$ 是反对称 lift / gyroscopic 矩阵；
- $\tau_\theta$ 是执行器到广义力映射网络输出。

---

# 10. PH-NODE 的可学习结构与能量保证

本节集中说明 PH-NODE 的创新点：它把 Fossen-pH 的各个结构元素变成可学习但受约束的模块。

## 10.1 学习正定质量矩阵

PH-NODE 学习的是逆质量矩阵：

$$
\boxed{
M_\theta^{-1}=LL^\top>0.
}
$$

该结构保证动能始终正定：

$$
T_\theta=\frac12p_r^\top M_\theta^{-1}p_r>0
\quad
(p_r\ne0).
$$

与 Fossen 模型相比：

| Fossen-pH | PH-NODE |
|---|---|
| $M=M_{RB}+M_A$ 由物理参数给出 | $M_\theta$ 可由物理初值初始化，再由数据修正 |
| 刚体质量和附加质量可解释性强 | 总质量结构可学习，但分解不一定唯一 |

## 10.2 学习势能函数

Fossen-pH 中，回复力 $g(\eta)$ 必须满足：

$$
J^\top P_\eta=g.
$$

PH-NODE 中直接学习：

$$
V_\theta(q).
$$

然后由梯度产生保守力。因此：

$$
\boxed{
\text{Fossen-pH 是先给回复力再验证可积性；PH-NODE 是先给势能函数，保守力自动成立。}
}
$$

AUVHamNODE 中势能网络使用 $R^\top e_3$ 作为重力方向特征，并可选加入深度上下文。这体现了重力—浮力势能的物理先验。

## 10.3 学习正定阻尼

PH-NODE 将阻尼参数化为：

$$
D_\theta(\nu_r)=L_D(\nu_r)L_D^\top(\nu_r)
$$

或对角 softplus 形式，从而保证：

$$
\boxed{
\nu_r^\top D_\theta(\nu_r)\nu_r\ge0.
}
$$

这意味着神经网络可以学习复杂速度相关阻尼，但不能学出违反耗散性的负阻尼。

## 10.4 学习反对称 lift 项

PH-NODE 将无功 lift / gyroscopic 项参数化为：

$$
\boxed{
S_\theta(\nu_r)=A_\theta(\nu_r)-A_\theta^\top(\nu_r).
}
$$

因此：

$$
\boxed{
\nu_r^\top S_\theta(\nu_r)\nu_r=0.
}
$$

这类项可以表达复杂的升力或速度耦合效应，但不会改变系统能量。

需要注意：这里的 $S_\theta$ 或 $J_\theta$ 不是 Fossen 运动学矩阵 $J(\eta)$。为了避免混淆，本文用 $S_\theta$ 表示 PH-NODE 中的反对称无功水动力项。

## 10.5 学习执行器到广义力映射

真实 AUV 的输入通常不是 $\tau$，而是执行器状态或命令，例如：

$$
u_{input}=[\delta_r,\delta_s,n]^\top.
$$

PH-NODE 使用网络：

$$
\boxed{
\tau_\theta=B_\theta(\nu_r,u_{actual},v_c).
}
$$

这相当于将 REMUS100 中的解析 `_control_forces` 替换为可学习映射，但该映射仍然作为功率端口力进入机械系统。

## 10.6 执行器一阶滞后

PH-NODE 中执行器状态满足：

$$
\boxed{
\dot u_{actual}=\frac{u_{cmd}-u_{actual}}{T_{actuator}}.
}
$$

并通过 softplus 等方式保证：

$$
T_{actuator}>0.
$$

这说明 PH-NODE 不仅学习机械核心，也显式携带执行器动态，使模型更接近真实数据采集过程。

## 10.7 能量平衡

忽略海流外部功率项，并将 $\tau_\theta$ 视为广义力端口，系统满足：

$$
\dot H_\theta
=
\nu_r^\top
\left(-D_\theta(\nu_r)\nu_r+S_\theta(\nu_r)\nu_r+\tau_\theta\right).
$$

由于：

$$
S_\theta=-S_\theta^\top,
$$

有：

$$
\nu_r^\top S_\theta(\nu_r)\nu_r=0.
$$

因此：

$$
\boxed{
\dot H_\theta
=
-\nu_r^\top D_\theta(\nu_r)\nu_r
+\nu_r^\top\tau_\theta
\le
\nu_r^\top\tau_\theta.
}
$$

这就是 PH-NODE 的结构保证：

$$
\boxed{
\text{训练过程中无论网络参数如何变化，模型都保持 pH 能量结构。}
}
$$

有海流时，应额外加入环境端口功率项，例如势能对位置的梯度与海流漂移之间的功率交换。这并不破坏 pH 思想，但说明海流应被解释为外部通道。

---

# 11. 从 Fossen-pH 到 PH-NODE 的统一视角

## 11.1 继承关系

PH-NODE 可以被理解为 Fossen-pH 的结构保持推广。

| 结构元素 | Fossen-pH | PH-NODE 推广 |
|---|---|---|
| 位姿 | $\eta=[x,y,z,\phi,\theta,\psi]^\top$ | $q=(x,R)\in SE(3)$ |
| 动量 | $p_b=M\nu$ | $p_r=M_\theta\nu_r$ |
| 质量 | 解析 $M_{RB}+M_A$ | 学习 SPD $M_\theta$ 或 $M_\theta^{-1}$ |
| 势能 | 解析 $P(\eta)$ | 学习 $V_\theta(q)$ |
| Coriolis | 显式 $C(\nu)\nu$ | $\operatorname{ad}_{\nu}^*p$ 几何结构 |
| 阻尼 | 经验 $D(\nu)$ | 学习 PSD/PD $D_\theta(\nu_r)$ |
| lift | 经验升力或拆分项 | 学习 skew $S_\theta(\nu_r)$ |
| 控制力 | 解析舵桨模型 | 学习 $B_\theta(\nu_r,u_{act},v_c)$ |
| 海流 | 外部扰动或端口 | 外生通道或条件输入 |
| 能量平衡 | $\dot H=-\nu^\top D\nu+\nu^\top\tau$ | $\dot H_\theta=-\nu_r^\top D_\theta\nu_r+\nu_r^\top\tau_\theta$ |

因此：

$$
\boxed{
\text{Fossen-pH 是 PH-NODE 模型类中的一个物理参数化特例。}
}
$$

但反过来不一定成立。PH-NODE 能表示许多满足 pH 结构的动力学，它们未必能还原为传统 Fossen 模型中的某一组固定 $M_{RB}$、$M_A$、$D$、$g$ 和舵桨经验参数。

## 11.2 从固定解析模型到结构保持学习模型

传统 Fossen 模型回答的是：

$$
\text{给定物理参数和经验公式，系统如何运动？}
$$

Fossen-pH 进一步回答：

$$
\text{这些运动方程背后的能量结构是什么？}
$$

PH-NODE 则回答：

$$
\text{能否在保持该能量结构的前提下，从数据中学习未知动力学？}
$$

因此，从 Fossen 到 PH-NODE 的逻辑并不是从物理走向黑箱，而是：

$$
\boxed{
\text{从固定物理参数走向结构保持的可学习物理模型。}
}
$$

## 11.3 PH-NODE 的创新点总结

PH-NODE 的创新可以概括为五点。

### 第一，从 Euler 坐标走向 $SE(3)$ 几何表达

它避免了 Euler 角奇异性，并与刚体几何力学更一致。

### 第二，从解析水动力走向结构约束学习

它用神经网络学习 $M^{-1}$、$V$、$D$、$S$ 和 $B_\theta$，但每个网络都受到物理结构约束。

### 第三，从验证无源性走向保证无源性

Fossen-to-pH 是训练或建模后检查结构；PH-NODE 是模型定义时就保证结构。

### 第四，从理想广义力输入走向真实执行器映射

它允许将执行器状态、命令、速度和海流特征纳入广义力映射。

### 第五，从纯解析模型走向可校正模型

PH-NODE 可以用 Fossen/REMUS100 作为初值或先验，再通过数据学习未建模残差。

---

# 12. 对建模、辨识和控制的启示

## 12.1 对建模的启示

最佳建模策略不是在 Fossen 和神经网络之间二选一，而是结合二者：

$$
\boxed{
\text{用 Fossen 提供物理结构和初值，用 PH-NODE 学习未知误差。}
}
$$

例如：

- 用 REMUS100 的 $M$ 初始化 $M_\theta$；
- 用重力—浮力势能初始化或正则化 $V_\theta$；
- 用传统阻尼模型初始化 $D_\theta$；
- 用 $S_\theta$ 学习未建模 lift 或 gyroscopic 效应；
- 用 $B_\theta$ 学习真实舵桨力映射。

## 12.2 对辨识的启示

普通黑箱辨识可能拟合短期轨迹，但容易在长时预测中出现能量漂移。PH-NODE 通过结构约束降低了不可物理模型的搜索空间，使学习结果更容易泛化到未见工况。

特别是：

$$
D_\theta\ge0
$$

保证模型不会把阻尼学成非物理增能项；

$$
S_\theta=-S_\theta^\top
$$

保证 lift / gyroscopic 项不改变机械能；

$$
V_\theta(q)
$$

保证回复力来自势能。

## 12.3 对控制的启示

Fossen-pH 和 PH-NODE 都可以服务于无源控制、能量整形、阻尼注入和基于 Hamiltonian 的稳定性分析。

若控制设计基于端口功率：

$$
P_{in}=\tau^\top\nu_r,
$$

则无论内部水动力由解析公式给出还是由 PH-NODE 学习，只要模型保持 pH 结构，控制分析就可以沿用类似的能量方法。

## 12.4 对工程实现的启示

实际系统中应明确区分：

1. 连续时间机械 pH core；
2. 执行器动态；
3. 海流外生通道；
4. 数值积分器；
5. 姿态约束或归一化；
6. 饱和和裁剪。

PH-NODE 的优势在于可以把这些模块分层组织，而不是把所有东西混成一个无结构黑箱。

---

# 13. 结论

本文从标准 Fossen AUV 模型出发，逐步推导并解释了其 port-Hamiltonian 结构，然后通过 REMUS100 工程代码说明真实 AUV 模型中的 pH 核心与工程复杂性，最后自然引出 PH-NODE 作为结构保持学习模型。

主要结论如下。

第一，标准 Fossen 模型在满足以下条件时可以严格写成 pH 形式：

$$
M=M^\top>0,
\qquad
\nu^\top C(\nu)\nu=0,
\qquad
D_s(\nu)\ge0,
\qquad
J^\top P_\eta=g,
\qquad
P_{in}=\tau^\top\nu.
$$

第二，Fossen-to-pH 的本质是结构重排。它不改变原始动力学，而是显式揭示能量、互联、耗散和端口结构。

第三，REMUS100 连续车辆核心动力学基本保留 pH 结构；但完整仿真器中的海流、执行器滞后、四元数归一化、数值积分和饱和裁剪需要作为外生端口、扩展状态、约束或离散操作来解释。

第四，传统 Fossen-pH 的主要局限不在于能量结构本身，而在于真实水动力、舵桨映射和海流耦合难以完全解析建模。

第五，PH-NODE 的创新在于：它不是先拟合一个黑箱再检查是否物理，而是从一开始就把 pH 结构作为神经网络模型类的硬约束。AUVHamNODE 在 $SE(3)$ 上构造 Hamiltonian 机械核心，并学习正定质量、势能、正定阻尼、反对称 lift 和执行器到广义力映射。

最终，本文的核心观点可以概括为：

$$
\boxed{
\text{标准 Fossen 模型揭示 AUV 动力学的物理结构；pH 形式揭示其能量结构；REMUS100 工程模型暴露解析建模复杂性；PH-NODE 则在保留能量结构的基础上，为未知水动力和执行器映射提供可学习表达。}
}
$$

更简洁地说：

$$
\boxed{
\text{PH-NODE 不是对 Fossen 的替代，而是 Fossen-pH 能量结构的几何化、学习化和工程化推广。}
}
$$

---

# 附录 A：主要符号表

| 符号 | 含义 |
|---|---|
| $\eta$ | Fossen 模型中的位姿向量 |
| $\nu$ | 体坐标速度向量 |
| $\nu_r$ | 相对水流速度 |
| $M$ | 总质量矩阵 |
| $M_{RB}$ | 刚体质量矩阵 |
| $M_A$ | 附加质量矩阵 |
| $C(\nu)$ | Coriolis/centripetal 矩阵 |
| $D(\nu)$ | 阻尼矩阵 |
| $D_s$ | 阻尼矩阵对称部分 |
| $D_a$ | 阻尼矩阵反对称部分 |
| $g(\eta)$ | 重力—浮力回复力 |
| $P(\eta)$ | Fossen 坐标下势能 |
| $H$ | Hamiltonian / 总能量 |
| $p_b$ | 体坐标准动量，$p_b=M\nu$ |
| $q=(x,R)$ | $SE(3)$ 上的位姿 |
| $p_r$ | 相对速度对应的动量，$p_r=M\nu_r$ |
| $V_\theta(q)$ | PH-NODE 学习势能 |
| $D_\theta(\nu_r)$ | PH-NODE 学习阻尼 |
| $S_\theta(\nu_r)$ | PH-NODE 学习反对称 lift / gyroscopic 项 |
| $B_\theta$ | 执行器到广义力映射网络 |
| $u_{actual}$ | 实际执行器状态 |
| $u_{cmd}$ | 执行器命令 |
| $v_c^n$ | 惯性坐标系海流速度 |

---

# 附录 B：REMUS100 代码模块与 pH 解释

| 代码模块 | 数学结构 | pH 解释 |
|---|---|---|
| `MRB`, `MA`, `M`, `Minv` | $M=M_{RB}+M_A$ | 正定惯性储能 |
| `_m2c` | $C(\nu_r)$ | 反对称 Coriolis / 无功项 |
| `_g_forces` | $g(\eta)$ | 重力—浮力势能梯度 |
| `_relative_rhs` | $M\dot\nu_r=\cdots$ | 连续车辆核心动力学 |
| 阻尼矩阵 `D` | $D(\nu_r)$ | 对称耗散项 |
| `_force_lift_drag` | lift + drag | drag 耗散，lift 无功 |
| `_cross_flow_drag` | cross-flow drag | 非线性半正定耗散 |
| `_control_forces` | $\tau_{control}$ | 外部功率端口 |
| `compute_derivatives` | $\nu=\nu_r+\nu_c$ | 海流相对速度处理 |
| `Remus100Simulator` | 四元数、RK4、归一化、裁剪 | 工程仿真层，不是单纯连续 pH core |

---

# 附录 C：AUVHamNODE 代码模块与数学结构对应关系

| AUVHamNODE 模块 | 数学表达 | 结构保证 |
|---|---|---|
| `mass_inv` | $M_\theta^{-1}=LL^\top$ | 质量逆矩阵正定 |
| `_mass_matrix` | 返回 $M_\theta^{-1}$ 与 $M_\theta$ | 动能可计算 |
| `_momentum` | $p_r=M_\theta\nu_r$ | 定义相对动量 |
| `_potential` | $V_\theta(q)$ | 势能函数 |
| `_potential_gradients` | $\nabla_q V_\theta$ | 保守力自动生成 |
| `damping` | $D_\theta(\nu_r)$ | 正定或半正定阻尼 |
| `lift` | $S_\theta=A-A^\top$ | 反对称无功项 |
| `B_net` | $\tau_\theta=B_\theta(\cdot)$ | 学习执行器到广义力映射 |
| `forward` | ODE 右端 | 几何 pH 动力学 |
| `energy` | $H_\theta=\frac12\nu_r^\top M\nu_r+V_\theta(q)$ | 机械储能 |
| `T_actuator` | $T_{actuator}>0$ | 执行器一阶滞后时间常数为正 |
| `to_ode_state`, `to_data_state` | $\nu_r=\nu-R^\top v_c^n$ | 海流相对速度转换 |

---

# 推荐使用方式

如果本文用于论文或报告，建议将第 1–7 章作为理论铺垫，第 8–10 章作为核心创新模型，第 11–13 章作为统一视角和结论。若篇幅有限，可以压缩第 5 章中 Fossen-to-pH 的推导细节，将其放入附录，而把更多篇幅留给 PH-NODE 的结构设计、训练策略和实验验证。

