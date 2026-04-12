# AUVHamNODE 噪声鲁棒性实验设计文档

**版本**：1.0
**日期**：2026-04-07
**基于数据集**：`auv_oc_traj1000_blk150_s23_d0be9434`（主）、`auv_oc_traj667_blk150_s42_9b2d7617`（交叉验证）

---

## 1. 科学主张与实验目标

### 1.1 核心主张

> **pHNODE 的 port-Hamiltonian 结构对导航状态估计误差具有内生鲁棒性**：在相同噪声量级下，其轨迹预测质量相比物理无约束的 baseline 模型退化更少。

这一主张的物理依据：
- Hamiltonian 结构保证能量守恒，误差不会通过虚假能量注入被放大
- 正定质量矩阵与阻尼矩阵为物理阻尼，对相对速度误差有自然衰减作用
- SE(3) 精确运动学保证几何一致性，旋转误差不会串扰速度预测

### 1.2 明确不做的主张

本实验**不声明**：
- "噪声训练使模型变得更好"（这需要另一套训练框架）
- "模型能从含噪观测序列中学习动力学"（这需要 teacher forcing 或状态估计前端）

当前架构是"从带噪 IC 做 open-loop 短期 rollout"，实验设计围绕这一使用场景。

### 1.3 实验问题

**主问题**：在来自真实导航系统的初始状态不确定性下，各模型的预测精度退化了多少？

**导出问题**：
- 退化率与噪声量级是否呈近线性关系？（线性 → 良好的 IC 鲁棒性）
- 哪些 DOF 最脆弱？（识别瓶颈）
- 不同控制场景（CHIRP/OU/PRBS）下的鲁棒性是否一致？

---

## 2. 问题定义

### 2.1 训练-部署接口

```
训练时：y0_noisy → ODE → pred_{1:T}  对比  target_clean_{1:T}
         ↑                                   ↑
    导航系统估计输出                      仿真真值（监督标签）

部署时：y0_nav → ODE → pred_{1:T}
         ↑
    真实导航系统输出（DVL+IMU+INS 融合后的状态估计）
```

**关键约定**：
- 监督目标始终是干净真值（仿真器生成），不对目标加噪
- 噪声仅作用于初始条件 y0，模型做 open-loop rollout，不在中间步骤注入噪声
- 评估时也从带噪 y0 出发，与训练接口一致

### 2.2 噪声语义的澄清

数据集中存储的是**导航滤波器输出**（DVL/IMU 融合后），不是原始传感器读数。因此：
- 噪声参数应解释为"导航状态估计的不确定性"，不是 DVL 硬件规格
- 合理量级参考：INS 融合后速度精度约 1–3 cm/s RMS，优于原始 DVL 规格
- OC 场景的海流估计误差约 1–3 cm/s（ADCP/模型估计精度）

---

## 3. 噪声模型设计

### 3.1 核心原则

**原则 1：以 ν_r 空间的预算为基准，而非对各状态独立设定**

模型 ODE 的核心状态是 `ν_r`（机体相对速度），由 `to_ode_state()` 计算：
```
ν_r = ν_total - R^T v_c^n
```

当前设计对 `ν_total` 和 `v_c^n` 分别施加独立噪声（`vel_lin_std=0.02`，`current_std=0.05`），
导致 ODE 实际接收的复合误差：
```
σ(δν_r) ≈ sqrt(0.02² + 0.05²) ≈ 0.054 m/s
```
海流噪声贡献了 ν_r 误差方差的 85%，参数设定时并未意识到这一点。

正确做法：先确定 `δν_r` 的目标误差，再分配各来源的贡献。

**原则 2：各轴独立参数化，以 ν_r 各轴标准差的固定比例为噪声量级**

数据集 ν_r 各 DOF 标准差（两个数据集高度一致）：

| DOF | ν_r std | 当前统一噪声 | 相对比例 | 建议（α=5%）|
|-----|---------|------------|---------|-----------|
| u（纵向）| 0.464 m/s | 0.020 m/s | **4.3%** | 0.023 m/s |
| v（侧向）| 0.091 m/s | 0.020 m/s | **22%** | 0.0046 m/s |
| w（垂向）| 0.073 m/s | 0.020 m/s | **27%** | 0.0037 m/s |
| p（横滚）| 0.025 rad/s | 0.005 rad/s | **20%** | 0.0012 rad/s |
| q（俯仰）| 0.091 rad/s | 0.005 rad/s | **5.5%** | 0.0046 rad/s |
| r（偏航）| 0.053 rad/s | 0.005 rad/s | **9.4%** | 0.0027 rad/s |

当前统一标量噪声对 v/w/p（弱激励轴）施加了 20–27% 的相对扰动，是 u 轴的 5–6 倍，
这是角速度指标系统性退化的根因。

**原则 3：DVL dropout 只作用于线速度通道**

DVL 测量线速度（[0:3]），底锁丢失不影响 IMU 陀螺仪（[3:6]）。
当前代码对所有 6 个通道做冻结，物理上不正确。

### 3.2 三档噪声 Profile

| Profile | 用途 | ν_r 噪声系数 α | 含义 |
|---------|------|-------------|------|
| **nominal** | 主训练 + 主评估 | 5% | 正常导航条件下的状态估计不确定性 |
| **degraded** | 压力测试评估 | 12% | 恶劣海况/浅水/长时间无 GPS 修正 |
| **failure** | 仅评估，不训练 | α=5% + DVL dropout | 底锁丢失故障场景 |

> **为什么 failure 不用于训练**：DVL dropout 在 0.2s 短块内对损失函数产生极端干扰。
> 已有实验数据（scale=1.0）证明此类训练完全失败：best epoch 在 ramp 完成前（epoch 51–69）
> 就停止，说明模型无法在 Level 3 噪声下收敛。

### 3.3 具体噪声参数表

#### ν_r 目标误差预算（ODE 输入层）

```
σ_nur_i = α × std_i(ν_r)    α = 0.05（nominal）/ 0.12（degraded）
```

| DOF | nominal σ | degraded σ |
|-----|-----------|-----------|
| u | 0.023 m/s | 0.056 m/s |
| v | 0.0046 m/s | 0.011 m/s |
| w | 0.0037 m/s | 0.0088 m/s |
| p | 0.0012 rad/s | 0.0030 rad/s |
| q | 0.0046 rad/s | 0.011 rad/s |
| r | 0.0027 rad/s | 0.0064 rad/s |

#### 其他状态的噪声（独立于 ν_r 预算）

| 状态 | nominal | degraded | 说明 |
|------|---------|---------|------|
| v_c^n（OC 专用）| 0.010 m/s（各轴）| 0.025 m/s | 海流估计不确定性；与上表 δν_total 联合后 δν_r 额外贡献约 0.006/0.014 m/s，仍在预算范围内 |
| δθ₀（初始姿态）| 0.005 rad（0.3°）| 0.010 rad | INS 入水对准精度 |
| u_act（执行器）| 0.003 | 0.008 | 传感器读取不确定性 |

#### 实现方式（关键）

直接对 `ν_total` 按 ν_r 预算加噪，等价于在 ν_r 空间施加目标扰动：

```python
# per-axis noise derived from nu_r budget
vel_noise = α * torch.tensor([
    0.464,  # u: nu_r std
    0.091,  # v
    0.073,  # w
    0.025,  # p
    0.091,  # q
    0.053,  # r
], dtype=dtype, device=device)

delta_nu = vel_noise * torch.randn(B, 6, dtype=dtype, device=device)
noisy[:, 0, 12:18] = clean[:, 0, 12:18] + delta_nu  # applies to nu_total

# OC: small independent current noise (not dominating nu_r budget)
delta_vc = sigma_c * torch.randn(B, 3, ...)          # sigma_c = 0.010 m/s
noisy[:, 0, 24:27] = clean[:, 0, 24:27] + delta_vc
```

#### 可废弃的噪声组件（IC-only 路径下无效）

以下组件仅影响 t>0 的状态，当前训练只消费 t=0，计算后丢弃：
- AR(1) 全轨迹构建（Level 2）
- 位置漂移积分
- 姿态漂移积分

这些组件**仅在训练框架升级为 teacher forcing 后才有意义**，
目前是无效计算，应从 IC-only 路径移除。
保留：块常偏置（bias_ratio），它是 y0 层面 Level 1 和 Level 2 的实质区别。

---

## 4. 训练策略

### 4.1 主训练配置

**推荐：干净训练为主，noisy IC 训练为辅助消融**

| 配置 | 训练 | 目的 |
|------|------|------|
| `clean` | Level 0，无噪声 | 干净 IC 性能上界；结构鲁棒性验证的基础 |
| `noisy_nominal` | nominal profile（α=5%）| IC 正则化；对比干净训练的退化量 |

**为什么主实验用干净训练**：
若 pHNODE 的结构鲁棒性来自物理约束，则应在**相同训练条件**下展示其优势，而非依赖
噪声训练技巧。干净训练 + 带噪评估控制了训练条件这一变量，结论更干净。
如果 pHNODE 在相同的干净训练条件下就比 blackbox 更鲁棒，这才是架构贡献。

**废弃**：`scale=1.0` 的 Level 2/3 噪声训练——在当前 0.2s 短块架构下无法有效训练。

### 4.2 验证协议修复

当前代码 `train_auv_hamnode.py:241` 在 `train=False` 时始终用干净 IC，导致
checkpoint 选择偏向干净 IC 性能，对 noisy IC 训练产生系统性偏见（表现为 best epoch
异常提前）。

**修复方案**：
- **方案 A（如使用 noisy 训练）**：验证也用相同 nominal noise profile 扰动 IC
- **方案 B（推荐）**：直接用干净训练，不存在偏差问题

---

## 5. 评估协议

### 5.1 评估层次

```
Layer 1  Block-level（0.2 s）  evaluate_trajectory_prediction
Layer 2  Trajectory-level（30 s，150 blocks 链式推进）  evaluate_heldout_trajectories
```

Trajectory-level 评估是**主报告指标**——更接近真实部署（AUV 从导航系统拿到初始状态，
做较长时间的预测），且对误差积累更敏感，更能区分模型间的差异。

两层评估都需要支持 noisy IC 模式。

### 5.2 Noisy IC 评估流程

```
对每条 held-out trajectory（共 140 条，按 scenario 分层）：
  对每个 noise profile in {clean, nominal, degraded, failure}：
    对每个 noise realization k = 1 … K（K=10）：
      固定随机种子 seed_k（所有模型共用）
      对 trajectory 第一个 block 的 y0 施加噪声 → y0_noisy_k
      从 y0_noisy_k 出发，链式 rollout 整条 trajectory（150 blocks）
      记录每步预测误差（位置/速度/姿态/各 DOF）

聚合：对每条 trajectory 的 K 次 realization，取 mean/median/p90/worst
最终报告：跨 140 条 trajectory 的统计分布
```

**关键设计决策**：
- Noise realization 固定（不在每次评估时重新采样）→ 跨模型比较完全公平
- K=10 次 realization → 统计稳定，同时可接受的计算开销
- 所有模型共用相同的 noise seeds → 方差来源仅来自模型差异

### 5.3 报告指标

**主指标**（per trajectory，报 mean±std 和 p90）：

| 指标 | 含义 |
|------|------|
| 位置 RMSE（m）| 预测位置 vs. 真值 |
| 线速度 RMSE（m/s）| 预测线速度 vs. 真值 |
| 角速度 RMSE（rad/s）| 预测角速度 vs. 真值 |
| 姿态测地距离（rad）| SO(3) 上的真实旋转误差 |
| 各 DOF RMSE（u/v/w/p/q/r）| 识别最脆弱轴 |

**核心衍生指标（鲁棒性证据）**：

```
退化比（degradation ratio）= noisy_RMSE / clean_RMSE
```

退化比 ≈ 1.0 → 完全鲁棒；退化比 >> 1.0 → 对 IC 噪声敏感。

**横向比较表（实验结论的核心输出）**：

```
                   | Clean pos  | Nominal    | Nominal    | Degraded   | Degraded
Model              | RMSE (m)   | RMSE (m)   | ratio      | RMSE (m)   | ratio
-------------------|------------|------------|------------|------------|----------
phnode_full        |            |            |            |            |
se3_accel_blackbox |            |            |            |            |
blackbox_fullstate |            |            |            |            |
phnode_merged_force|            |            |            |            |
```

若 pHNODE 的退化比显著低于 blackbox，即为结构鲁棒性的实证证据。

### 5.4 按场景分报

数据集覆盖三种控制场景（CHIRP/OU/PRBS），**必须分别报告**：

| 场景 | 预期特性 | 对噪声的敏感方向 |
|------|---------|---------------|
| CHIRP | 高频激励，角速度大 | 对 IC 角速率噪声（p/q/r）敏感 |
| OU | 缓变轨迹，较平滑 | 对 IC 速度 bias 和姿态误差敏感 |
| PRBS | 随机切换，动态多样 | 综合测试 |

若某场景下退化比异常高，可定向查找该场景下哪个 DOF 最脆弱。

### 5.5 不建议做的评估

- **对目标加噪后的评估**：这测的是"噪声拟合"能力，不是动力学预测鲁棒性
- **每个 block 都重新从带噪 IC 初始化（receding-horizon）**：超出当前架构语义，
  作为 future work（需配合在线状态更新逻辑）

---

## 6. 实验矩阵

### 实验 1（必做）：结构鲁棒性主实验

**目的**：在相同训练条件下，展示 pHNODE vs. blackbox 在带噪 IC 下的退化率差异。

```
训练：clean × {phnode_full, se3_accel_blackbox, blackbox_fullstate}
评估：clean IC / nominal noisy IC / degraded noisy IC
种子：42, 43, 44（报 mean±std）
数据集：auv_oc_traj1000_blk150_s23（主）
```

**工程量**：低——仅需实现 noisy IC 评估路径，不改任何训练代码。
**这是支撑核心主张的必要实验。**

---

### 实验 2（必做）：噪声参数修复验证

**目的**：确认修复后的噪声参数（各轴 α=5%）不损害训练质量，可以用于后续实验。

```
训练：noisy_nominal（α=5%）× phnode_full
评估：clean IC（验证无损失）+ nominal noisy IC（验证有收益）
对比：clean 训练的 phnode_full（相同 seed）
种子：42, 43, 44
```

关键检查点：best epoch 是否回到正常范围（≈200），跨 seed 方差是否小于 clean 训练。

---

### 实验 3（建议做）：各轴噪声各向异性的消融

**目的**：量化统一标量噪声 vs. 各轴比例噪声对各 DOF 退化的影响。

```
训练：clean × phnode_full
评估（噪声方案对比）：
  A. 统一标量（当前默认）× {scale=0.25, scale=0.5, scale=1.0}
  B. 各轴比例（α=3%, α=5%, α=12%）
报告：各 DOF 退化比分布；重点关注 v/w/p 轴
```

---

### 实验 4（可选）：failure profile 评估

**目的**：评估 DVL 底锁丢失场景下各模型的退化（不作为训练分布）。

```
训练：clean × {phnode_full, blackbox_fullstate}
评估：failure profile（nominal α=5% + 线速度通道 DVL dropout）
注意：dropout 只作用 delta_nu[:, :3]（修复 bug 后）
```

---

### 实验优先级

| 优先级 | 实验 | 核心价值 |
|--------|------|---------|
| P0 | 实验 1 | 支撑主张必须，工程量最低 |
| P0 | 噪声参数修复（代码改动）| 所有后续实验的前提 |
| P1 | 实验 2 | 验证修复后的训练稳定性 |
| P2 | 实验 3 | 诊断各轴不均衡问题 |
| P3 | 实验 4 | 扩展结论覆盖范围 |

---

## 7. 代码改动清单

### 7.1 必须修复（影响结论正确性）

**噪声参数化**（`train_utils.py:NoiseConfig`）

将 `vel_lin_std`/`vel_ang_std` 替换为各轴独立的 `vel_noise_per_dof: List[float]`（长度 6），
并预填 α=5% 下的默认值（见附录 A）。
`current_std` 默认值从 0.05 降至 0.010 m/s。

**DVL dropout 只作用线速度通道**（`train_utils.py:678`）

```python
# 修改前：对全部 6 维操作
delta_nu = _dvl_dropout_freeze(delta_nu, cfg.dropout_prob)

# 修改后：仅对线速度 [:3] 操作
delta_nu_lin = _dvl_dropout_freeze(delta_nu[:, :, :3], cfg.dropout_prob)
delta_nu = torch.cat([delta_nu_lin, delta_nu[:, :, 3:]], dim=2)
```

**移除 IC-only 路径的无效计算**（`train_utils.py:build_noisy_training_pair`）

Level 1/2 下废弃 AR(1) 全轨迹构建、位置漂移积分、姿态漂移积分。
保留：块常偏置（bias_ratio）和初始姿态扰动（rot_init_std）。

### 7.2 必须新增（评估协议）

**`evaluate_heldout_trajectories` 加 noisy IC 模式**（`train_utils.py:1078`）

新增参数：
```python
def evaluate_heldout_trajectories(
    model, dataset, t_eval, device, ode_solver="rk4",
    noise_cfg=None,          # Optional[NoiseConfig]：非 None 时扰动 y0
    noise_realizations=10,   # K 次 realization
    noise_seed_base=0,       # 固定种子确保可重复
    max_trajectories=None,
) -> Dict:
```

对每条 trajectory 的第一个 block 用 `build_noisy_ic(block, noise_cfg, seed)` 扰动 y0，
其余 blocks 按当前逻辑（从上一块末尾状态初始化）；聚合 K 次 realization 的统计量。

**跨模型 noisy IC 批量评估脚本**

在 `evaluate_rollout_benchmark.py` 加 `--noise_profile {none,nominal,degraded,failure}` 参数，
批量评估所有指定 checkpoint，输出退化比汇总表。

### 7.3 建议修复（验证协议一致性）

若使用 noisy IC 训练（实验 2），将 `train_auv_hamnode.py:241` 的 validation 路径改为
同样使用 nominal noise profile，避免 checkpoint 选择偏向干净 IC 性能。

---

## 8. 可 falsify 的预测

| 预测 | 理由 | 若不成立说明 |
|------|------|------------|
| phnode_full 退化比 < se3_accel_blackbox < blackbox_fullstate（nominal）| 物理约束越强，IC 误差扩散越受控 | pHNODE 的 D/J/V 学习可能未收敛；检查 Hamiltonian 分量 |
| v/w/p 轴退化比 > u 轴（统一噪声下）| 弱激励轴信噪比低 | 表明动力学主要由 u 主导，v/w/p 的预测已接近零 |
| nominal 退化比：phnode ≈ 1.5–4×，blackbox ≈ 5–15× | 有意义的结构优势范围 | 若 phnode 也达到 5–10×，说明当前物理先验还不够强 |
| 修复后的 noisy_nominal 训练 best epoch ≈ 200，方差 < clean | 正确参数化后不再触发异常早停 | 说明 0.2s 块内动力学信号对 α=5% 仍然太弱 |
| CHIRP 退化比 < OU 退化比（位置指标）| 高激励场景动力学信号强，噪声占比小 | 说明噪声的影响与控制策略解耦不良 |

---

## 9. Future Work 范围界定

以下问题技术上更完整，但超出当前架构：

| 问题 | 需要的变化 |
|------|-----------|
| 从带噪观测序列学习动力学 | 训练改为 teacher forcing；噪声贯穿整个推理过程 |
| 测试时在线噪声适应 | Kalman filter / UKF 作为状态估计前端 |
| 海流 v_c^n 作为块内隐变量推断 | Variational inference / meta-learning |
| Receding-horizon 评估 | 每块从最新带噪观测重新初始化 |
| 离线生成 truth+noisy nav trajectory 对 | 修改 data_collection.py；数据格式扩展 |

---

## 附录 A：ν_r 各轴噪声速查表

基于数据集 `d0be9434`（两个数据集结果高度一致，差异 < 2%）：

| DOF | ν_r std | α=5%（nominal）| α=12%（degraded）|
|-----|---------|--------------|----------------|
| u | 0.469 m/s | **0.0234 m/s** | **0.0563 m/s** |
| v | 0.092 m/s | **0.0046 m/s** | **0.0110 m/s** |
| w | 0.073 m/s | **0.0037 m/s** | **0.0088 m/s** |
| p | 0.025 rad/s | **0.0013 rad/s** | **0.0030 rad/s** |
| q | 0.092 rad/s | **0.0046 rad/s** | **0.0110 rad/s** |
| r | 0.053 rad/s | **0.0027 rad/s** | **0.0064 rad/s** |
| v_c^n（各轴）| — | **0.010 m/s** | **0.025 m/s** |
| δθ₀ | — | **0.005 rad** | **0.010 rad** |
| u_act | — | **0.003** | **0.008** |

## 附录 B：当前统一标量噪声各轴 α 等效值

说明为何不存在适合所有轴的统一 scale 值：

| DOF | vel_std=0.02, scale=1.0 对应的 α | 若目标 α=5%，所需 scale |
|-----|--------------------------------|----------------------|
| u | 4.3% | 1.16 |
| v | 22% | **0.23** ← 瓶颈 |
| w | 27% | **0.19** ← 瓶颈 |
| p | 20% | **0.25** ← 瓶颈 |
| q | 5.5% | 0.91 |
| r | 9.4% | 0.53 |

v/w/p 所需的 scale 比 u 小 5–6 倍。这解释了为什么：
- scale=1.0 → v/w/p 轴被过度扰动，训练失败
- scale=0.25 → v/w/p 噪声约 5–7%（合理），但 u 只有 ~1%（不足）
- 没有任何统一 scale 值能同时满足所有轴

---

*本文档基于代码审查（`train_utils.py:572`、`train_auv_hamnode.py:239`、`AUVHamNODE.py:369`）、
数据集统计量（`data/*.summary.txt`）及三轮独立分析（`docs/noise_scheme_review_*.md`）撰写。*
