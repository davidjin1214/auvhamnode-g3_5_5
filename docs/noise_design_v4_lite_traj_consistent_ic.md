# 轨迹噪声设计 v4-lite（trajectory-consistent noisy IC 主方案）

本文档给出一版新的实验主线方案，用于替代“直接进入 `v4-B` history-aware observer”的做法。

核心判断只有一条：

```text
在当前项目阶段，主问题仍然应当是：
模型对导航状态噪声是否鲁棒，
以及它是否能在 noisy state input 下仍然预测 clean truth。
```

因此，本方案的默认主线不是 `v4-B`，
而是一个更克制、更容易解释、也更符合当前代码结构的中间版本：

```text
v4-lite = trajectory-consistent noisy IC training
```

它保留现有纯 dynamics trainer 的问题定义：

```text
noisy y0 -> ODE rollout -> clean future trajectory
```

但把 `y0_noisy` 的生成方式，从“每个 block 独立采样的 noisy IC”
升级为“来自同一条 noisy navigation trajectory 的 block initial state”。

---

## 1. 文档定位

- [docs/noise_design_v3_remus100_reference_grounded.md](noise_design_v3_remus100_reference_grounded.md)
  负责给出各通道的误差幅值预算 `σ`
- [docs/noise_design_v4_dr_ekf_output.md](noise_design_v4_dr_ekf_output.md)
  负责给出 trajectory-noise 的物理语义、状态一致性约束，以及 `v4-A / v4-B` 的边界
- 本文档负责定义：
  - 为什么当前主线应落在 `v4-lite`
  - `v4-lite` 与 `v3 / v4-A / v4-B` 的关系
  - `v4-lite` 的实验矩阵、评价目标与推进条件

本文档不是 `v4-B` 的实施清单。
如果未来真的进入 history-aware observer 路线，应单列为后续扩展问题。

---

## 2. 研究目标

当前阶段应优先回答以下两个问题：

1. **robustness**
   当部署输入是带噪导航状态而不是真值时，模型的 clean-target rollout 是否仍然稳定
2. **noisy-to-truth dynamics learning**
   当训练输入带噪而监督仍为 clean truth 时，模型是否仍然学到可推广的真实动力学

这两个问题都属于：

```text
filtered-state robustness for pure dynamics models
```

它们还不等价于：

```text
history-aware state recovery / observer-augmented dynamics
```

后者正是 `v4-B` 实际研究的问题，不应在当前阶段混入主线。

---

## 3. 为什么当前不应以 `v4-B` 为主

从研究问题上看，`v4-B` 的复杂度明显超出当前主目标。

它不再只是“给纯动力学模型一个 noisy initial state”，而是：

```text
给模型一段 noisy observation history，
并要求它同时做 state recovery + dynamics rollout。
```

这会带来三个直接后果：

1. **问题被改写**
   `v4-B` 研究的是 `dynamics + observer`，而不是纯 noisy-IC robustness
2. **收益难以归因**
   如果结果变好，很难判断收益来自 dynamics 学得更好，还是 observer 先把状态修正了
3. **工程复杂度显著抬升**
   需要新 trainer、新 dataset、新 cache、新评估路径，且实验解释更复杂

对当前项目而言，这种复杂度和新增问题并不匹配。

更稳妥的路线应是：

1. 先把纯 dynamics 主线下的噪声协议做得更真实、更一致
2. 再判断这种协议是否已经足够回答 robustness 问题
3. 只有当单个 noisy state 明显信息不足时，才进入 `v4-B`

---

## 4. `v3` / `v4-A` / `v4-lite` / `v4-B` 的关系

从“模型真正看到什么”来区分，这四者的本质关系如下：

| 方案 | 输入 | target | 是否改 trainer | 方法本质 |
|---|---|---|---|---|
| `v3` | noisy IC | clean truth | 否 | profile-based noisy IC regularization |
| `v4-A` | noisy IC | clean truth | 否 | 用 mission-level OU 语义重写 `v3` |
| `v4-lite` | noisy IC | clean truth | 否 | trajectory-consistent noisy IC protocol |
| `v4-B` | noisy history | clean truth | 是 | dynamics + state recovery / observer |

因此：

1. `v3 / v4-A / v4-lite` 都仍属于 **纯 dynamics 主线**
2. `v4-lite` 与 `v3 / v4-A` 的主要差别在 **噪声生成协议**
3. `v4-B` 才是 **新的方法轴**

更直接地说：

```text
v4-lite 不是一个新模型，
而是一种更严格、更物理一致的 noisy-IC 实验协议。
```

---

## 5. `v4-lite` 的正式定义

### 5.1 训练任务

`v4-lite` 的训练接口保持为：

```text
y0_noisy -> ODE rollout -> pred_{1:T}
target   = clean current block trajectory
```

与当前 `v3 / v4-A` 一样：

- model input 是 **当前 block 起点** 的 noisy state
- target 始终是 **clean truth**
- trainer 仍然是现有 open-loop block rollout trainer

### 5.2 与 `v3 / v4-A` 的唯一区别

区别不在 trainer，而在 `y0_noisy` 的来源：

- `v3 / v4-A`
  直接对每个 block 的 clean `y0` 独立采样 noisy IC
- `v4-lite`
  先为整条 trajectory 生成一条 trajectory-consistent noisy navigation observation，
  再从其中取出当前 block 的 initial observation 作为 `y0_noisy`

因此，`v4-lite` 的额外语义是：

1. 同一 trajectory 内各 block 的噪声 realization 相互一致
2. `R / nu_r / v_c / Δp` 的耦合来自同一套 noisy kinematics
3. 当前 block 初值更接近“posterior-like navigation state”的边缘样本

### 5.3 明确不做的事情

`v4-lite` 第一版不做：

1. 不输入 history
2. 不训练 observer / encoder
3. 不把 noisy trajectory 当作监督目标
4. 不把任务改写成 multi-block rollout
5. 不改变现有 ODE backbone 结构

---

## 6. 状态语义与噪声构造原则

### 6.1 保持 ODE-space-first

`v4-lite` 延续当前噪声设计的主原则：

1. 从 clean data-state 得到 clean ODE-state
2. 在 `R / nu_r / u_act / v_c` 上定义误差
3. 再映射回 data convention 生成 noisy navigation observation

也就是说，OC 情况下仍必须优先控制模型真正消费的：

```text
nu_r = nu_total - R^T v_c^n
```

而不是把 `nu_total` 与 `v_c^n` 视为彼此独立的观测通道。

### 6.2 noisy trajectory 的物理对象

`v4-lite` 模拟的是：

```text
导航系统后处理输出的状态估计轨迹
```

而不是 raw sensor stream。

因此噪声应解释为：

- bias 残差
- 慢变漂移
- 姿态/速度误差经运动学耦合形成的位置偏差
- 必要时的固定方向 bias 压力测试

不应解释成传感器带宽上的高频 white noise。

### 6.3 `Δp` 不再是独立噪声通道

block-relative 位置始终满足：

```text
Δp(t0) = 0
```

因此在 `v4-lite` 中：

1. 不对 `Δp(t0)` 直接拍噪
2. block 内 `Δp(t)` 必须由 noisy kinematics 一致积分得到
3. 不能用 clean `R` 或 clean `nu_total` 去偷算 noisy 位置

离散实现上，应沿当前 block 的时间栅格积分。

### 6.4 `u_act` 与 `v_c` 的处理

建议延续以下约束：

- `u_act`
  - IC 噪声可保留小幅扰动
  - trajectory noise 默认不开启
- `v_c`
  - 当前 `oc` 主实验仍属于 **known-current surrogate**
  - 结果表与文档中应显式注明这一点
  - 不应把 clean `v_c` 写成“典型 DR 完整现实”

---

## 7. `v4-lite` 的噪声生成协议

### 7.1 父观测模型

父观测模型仍可沿用 mission-level OU / bias 语义：

```text
dξ_i/dt = -ξ_i / τ_i + sqrt(2 / τ_i) * w_i(t)
η_i(t)  = σ_i(t) * ξ_i(t)
```

其中：

- `σ_i(t)` 来自 v3 的 profile 预算
- `η_i(t)` 用于生成 trajectory-level noisy observation

需要强调的是：

```text
在 v4-lite 中，OU 的时间结构用于定义 noisy observation 的来源，
而不是让模型直接消费 sequence。
```

因此 `v4-lite` 的科学定位是：

```text
trajectory-consistent noisy-IC training
```

而不是 sequence-noisy learning。

### 7.2 单条 trajectory 的生成流程

对每条 clean trajectory，推荐按以下顺序生成 noisy observation：

1. 将 clean trajectory 转到 ODE convention
2. 在 trajectory 级别采样 `R / nu_r / v_c / optional bias` 的时间相关误差
3. 重建 noisy `nu_total`
4. 对每个 block 内的 `Δp(t)` 用 noisy kinematics 积分生成
5. 必要时保留 `u_act` 的 IC-only 小扰动，不默认加入 trajectory drift
6. 将 noisy ODE-state 转回 data convention

### 7.3 训练时模型真正使用什么

虽然整条 noisy trajectory 被生成出来，但训练时模型真正使用的仍然只是：

```text
current_block_noisy[0]
```

也就是：

```text
y0_noisy = current noisy navigation state at block start
```

clean target 则仍然是：

```text
current_block_clean[0:T]
```

---

## 8. 为什么 `v4-lite` 值得做

`v4-lite` 的价值，不在于引入了更复杂的模型，而在于它能回答一个当前尚未分离清楚的问题：

```text
当前 noisy-IC 结果，究竟反映了“合理的导航误差鲁棒性”，
还是只反映了“对 block 独立加噪的正则效果”？
```

`v4-lite` 可以帮助区分这两者。

如果 `v4-lite` 与 `v3 / v4-A` 结果近似，说明：

- 纯 noisy-IC 方案已经抓住了主要结论
- trajectory-consistent 噪声语义没有显著改变结论
- 现阶段没有必要引入 `v4-B`

如果 `v4-lite` 明显优于 `v3 / v4-A`，说明：

- 噪声生成协议本身很重要
- 当前主线应从 iid IC noise 升级为 trajectory-consistent IC noise
- 但仍然不等于必须进入 history-aware observer

---

## 9. 实验矩阵

## 9.1 训练协议

建议至少保留三组训练：

1. `clean-train`
2. `iid noisy-IC train`
   即当前 `v3 / v4-A`
3. `traj-consistent noisy-IC train`
   即 `v4-lite`

训练时应尽量保持以下条件不变：

- 同一 backbone
- 同一 optimizer / scheduler
- 同一 total steps
- 同一 warmup / ramp / mix ratio
- 同一 seed 集合

这样才能把差异归因到噪声协议，而不是训练设置变化。

## 9.2 评估协议

建议至少保留三组评估：

1. `clean current_init`
2. `iid noisy current_init`
3. `traj-consistent noisy current_init`

其中第三组是 `v4-lite` 的关键新增评估：

- noisy current_init 必须来自 fixed noisy trajectory cache
- 同一 held-out trajectory 内各 block 共享同一 realization

若资源允许，可在以上基础上补充：

4. `heading_biased_eval`
5. `degraded_eval`

但主结论应先建立在前三组上。

## 9.3 模型集合

最小可解释模型集建议是：

1. `phnode_full`
2. `ablate_no_mass_prior`

原因：

- `phnode_full` 是主模型，必须保留
- `ablate_no_mass_prior` 在现有 noisy 结果里更像稳定受益者
- 只看 `phnode_full` 容易把结构效应误判成噪声协议效应

若资源有限，不建议一开始扩到全部 baseline。

## 9.4 seed 选择

主实验建议使用 matched seeds。

对于当前 `oc` 主线，建议优先使用：

```text
42, 43, 44, 45, 46, 47
```

原因：

- 它们已经是当前结论讨论最充分的一组
- 能直接对照现有 clean/noisy follow-up 结果
- 更容易识别“是否只是修复单个 catastrophic seed”

## 9.5 推荐的最小实验表

建议先完成下面这张最小矩阵：

| 模型 | 训练 | 评估 | 目的 |
|---|---|---|---|
| `phnode_full` | clean | clean / iid noisy / traj-consistent noisy | 基线 |
| `phnode_full` | iid noisy-IC | clean / iid noisy / traj-consistent noisy | 当前主线 |
| `phnode_full` | v4-lite | clean / iid noisy / traj-consistent noisy | 比较噪声协议 |
| `ablate_no_mass_prior` | clean | clean / iid noisy / traj-consistent noisy | 对照结构 |
| `ablate_no_mass_prior` | iid noisy-IC | clean / iid noisy / traj-consistent noisy | 当前对照 |
| `ablate_no_mass_prior` | v4-lite | clean / iid noisy / traj-consistent noisy | 稳定性对照 |

---

## 10. 指标与结果解读

### 10.1 主指标

建议继续保持现有主指标体系：

- `60s final position error median`
- `completion@60s`

同时保留 block-level / held-out 误差：

- `position_rmse`
- `rotation_geodesic`
- `velocity_rmse`
- `angular_rmse`
- 每个 DOF 的 RMSE

### 10.2 必须看的视角

除 aggregate 结果外，必须至少看：

1. by-seed
2. by-scenario (`PRBS / CHIRP / OU`)
3. clean replay 代价

否则容易把“修复单个坏 seed”误读成“方法整体更强”。

### 10.3 结果解读规则

若出现以下结果，应分别这样解释：

#### A. `v4-lite ≈ v3 / v4-A`

解释：

- trajectory-consistent 噪声协议没有显著改变主结论
- 当前 `v3 / v4-A` 已足够作为主线
- 不建议继续推进 `v4-B`

#### B. `v4-lite` 稳定优于 `v3 / v4-A`

解释：

- 噪声生成协议本身对 robustness 结论有实质影响
- 后续主线应采用 `v4-lite`
- 但这仍然属于纯 dynamics 主线，不应夸大为新模型

#### C. `v4-lite` 只修复单个 catastrophic seed

解释：

- 这更像训练稳定性或 seed-specific regularization
- 不应写成“普遍更真实、更强”

#### D. `v4-lite` 对所有模型都没有帮助，且 clean replay 明显更差

解释：

- trajectory-consistent 噪声协议在当前强度下可能过重
- 应先回调 profile / schedule，而不是立即引入 history-aware observer

---

## 11. 何时才值得进入 `v4-B`

只有同时满足以下条件时，才建议把主线从 `v4-lite` 推进到 `v4-B`：

1. `v4-lite` 已经实现并稳定评估
2. `v4-lite` 相比 `v3 / v4-A` 仍然不能充分解释部署噪声下的误差模式
3. 误差分析显示“单个 noisy current state 的信息确实不够”
4. 研究问题明确升级为：
   `history 是否帮助状态恢复并进一步提升 clean-target prediction`

若不满足这些条件，`v4-B` 很可能只会增加工程复杂度和解释负担。

---

## 12. 最小实现范围

`v4-lite` 第一版建议只做以下最小改动：

1. 在 `train_utils.py` 中新增 trajectory-aware 训练索引
   目标是能拿到 `traj_idx + block_idx + current_block`
2. 新增 trajectory-level noisy observation cache
   但只用于生成每个 block 的 `y0_noisy`
3. 在现有 trainer 中增加一种 `traj_consistent_ic` 噪声模式
4. 在 held-out evaluation 中增加 trajectory-consistent noisy eval

第一版不建议做的事情：

1. 不新增 history encoder
2. 不新增新 backbone
3. 不修改 `AUVHamNODE.py`
4. 不修改 `data_collection.py`
5. 不把任务改成 multi-block rollout

---

## 13. 成功标准

如果按研究价值而不是工程复杂度来定义，`v4-lite` 的成功标准应是：

1. 能在现有数据集上稳定运行，无需重生成数据
2. 能以 trajectory-consistent 方式生成 noisy observation
3. 能与当前 `v3 / v4-A` 做 matched-seed、matched-backbone 比较
4. 能清楚回答：
   - 当前结论是否依赖于 iid noisy-IC 近似
   - trajectory-consistent 噪声协议是否改变模型排序或结论强度
5. 能为“是否值得进入 `v4-B`”提供清晰决策依据

---

## 14. 推荐结论口径

如果未来采用 `v4-lite` 作为主实验协议，论文或报告中建议使用如下口径：

```text
我们研究的是纯动力学模型在 navigation-state uncertainty 下的鲁棒性。
训练输入为 trajectory-consistent noisy initial state，
监督目标始终为 clean ground-truth trajectory。
该设定模拟的是 posterior-like navigation state，
而不是 raw sensor stream，也不是 history-aware observer。
```

这样可以同时避免两类常见误读：

1. 把结果误写成“模型学会了 posterior trajectory”
2. 把结果误写成“我们已经研究了 history-based state recovery”

---

## 15. 一句话总结

`v4-lite` 的定位应当是：

```text
在不改变纯 dynamics 问题定义的前提下，
把 noisy-IC 训练从“block 独立加噪”
升级为“trajectory-consistent posterior-like noisy IC”。
```

它是当前项目最合适的下一步主线。

在完成并验证这一步之前，不建议把 `v4-B` 作为主方案推进。
