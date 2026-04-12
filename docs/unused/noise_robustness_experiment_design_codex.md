# AUVHamNODE 噪声设计方案 V2

## 1. 结论先行

这套仓库当前最需要的，不是继续调 `Level 2 / Level 3` 的参数，而是先把**噪声语义和训练接口对齐**。

基于代码事实，我的判断是：

1. 现有训练实现本质上只是 **noisy initial condition augmentation**，不是“从连续带噪观测学习动力学”。
2. 现有 OC 噪声设计的主要问题，不是“有噪声”本身，而是 **噪声加在了错误的变量组合上**，导致模型真正使用的 `nu_r` 被过度污染。
3. 新方案应拆成两条线：
   - **A. IC-robust 方案**：立即可用，服务当前训练框架。
   - **B. Sequence-noisy 方案**：只有在 trainer 改成 `k-step teacher forcing` 或 receding-horizon 后才有意义。

如果现在就要一版正确、可落地、并且大概率不会再把性能打崩的方案，我建议先落地 **A**，并把 **B** 作为下一阶段升级。

---

## 2. 基于代码的关键诊断

下面这些结论都直接来自当前仓库实现。

### 2.1 当前训练只真正消费 `t=0`

`train_auv_hamnode.py` 在训练时调用 `build_noisy_training_pair()`，但随后只使用：

```python
y0 = self.model.to_ode_state(noisy_block[:, 0])
```

监督目标始终还是 clean trajectory，且 loss 在有噪声时只跳过 `t=0`。

这意味着：

- `AR(1)` 全轨迹噪声的大部分结构没有被训练真正消费；
- 位置漂移、姿态漂移、current drift 在 `t>0` 的构造大部分只是“看起来复杂”，但当前训练并没有真的利用它们；
- 当前所谓 `Level 2 / Level 3`，在训练接口上并不等价于“从噪声序列学习”。

### 2.2 OC 场景下，当前噪声是加在错误组合上的

数据里存的是：

```text
[Δp(3), R(9), nu_total(6), u_act(3), u_cmd(3), v_c^n(3)]
```

但模型真正积分的是：

```text
nu_r = nu_total - R^T v_c^n
```

当前代码独立地给 `nu_total` 和 `v_c^n` 注噪，然后再由 `to_ode_state()` 变成 `nu_r`。

这会带来两个后果：

1. 你在 `nu_total` 上设的噪声预算，并不等于模型看到的 `nu_r` 噪声预算。
2. `current_std=0.05 m/s` 在当前数据尺度下过强，会主导 `nu_r` 误差。

以 `data/auv_oc_traj1000_blk150_s23_d0be9434.summary.txt` 的 `nu_r` 标准差为例：

- `u`: `0.4688`
- `v`: `0.0917`
- `w`: `0.0733`
- `p`: `0.0250`
- `q`: `0.0919`
- `r`: `0.0534`

当前默认值：

- 线速度噪声 `0.02 m/s`
- 角速度噪声 `0.005 rad/s`
- current 噪声 `0.05 m/s`

只看相对尺度：

- `0.02 / std(v) ≈ 21.8%`
- `0.02 / std(w) ≈ 27.3%`
- `0.005 / std(p) ≈ 20.0%`

如果再把 `0.05 m/s` 的 current 误差独立叠加进 `nu_r`，弱激励轴会被严重过扰动。这正是“有噪声模式明显差于无噪声模式”的核心原因之一。

### 2.3 当前 dropout 物理语义不正确

`_dvl_dropout_freeze()` 现在对 `delta_nu[..., :6]` 全部冻结，等于把：

- 线速度 `[u, v, w]`
- 角速度 `[p, q, r]`

一起冻结。

这不符合 DVL/IMU 语义。DVL dropout 只能作用于**线速度通道**，不应冻结 IMU 角速度。

### 2.4 当前 actuator 噪声是标量，量纲不统一

当前 `u_act` 直接加一个标量标准差 `0.005`，但数据里 actuator 三个通道量纲不同：

- 两个舵面是 rad
- 一个 rpm 是几百到上千

所以 `0.005 rpm` 几乎等于没加噪，而 `0.005 rad` 对舵面已经是可见扰动。这个设计在量纲上不成立。

### 2.5 block-relative position 的 `t=0` 不应加噪

数据中的位置是 block-relative `Δp`，每个 block 起点都约定为 0。

因此在当前 block 训练框架下：

- `Δp(t0)` 不是一个有独立物理意义的导航观测量；
- 给 `Δp(t0)` 加噪没有帮助，反而会破坏 block 约定。

所以 `t=0` 的位置项应保持 0。

---

## 3. 新方案总结构

## 3.1 方案 A：IC-consistent Noise

这是**当前代码立刻应该采用**的方案。

目标不是模拟整段传感器噪声，而是模拟：

```text
block 起点时刻，导航状态估计 y0_hat 的不确定性
```

核心原则：

1. **只构造 initial condition 噪声**，不要再伪装成全轨迹导航噪声。
2. **先在 ODE 真正使用的变量上定义预算**，再映射回 data convention。
3. **OC 场景下保证 `nu_r` 噪声预算可控**。

## 3.2 方案 B：Sequence-noisy Learning

这是下一阶段方案，前提是训练器升级为：

- `k-step teacher forcing`，或
- receding-horizon noisy observation training

只有到那时，AR(1)、bias random walk、dropout window 这些“时间结构噪声”才是有意义的。

在此之前，不建议再把这类设计作为主训练路径。

---

## 4. 方案 A 的正确构造方式

## 4.1 不要再直接对 `nu_total` 独立加噪

正确做法是：

1. 从 clean initial data-state `x0_clean` 出发；
2. 先得到 clean ODE state `y0_clean`；
3. 在 `y0_clean` 的语义空间里采样噪声；
4. 再把 noisy ODE state 映射回 data-state。

对于 OC 场景，这一点很关键。

## 4.2 推荐的构造顺序

对每个 block 初值：

### Step 1. 计算 clean ODE 初值

```text
y0_clean = to_ode_state(x0_clean)
```

其中 ODE 状态里的速度槽是 `nu_r`，不是 `nu_total`。

### Step 2. 在 ODE 空间采样噪声

采样以下量：

- `δθ`：初始姿态误差
- `δnu_r`：相对速度误差
- `δu_act`：执行器反馈误差
- `δv_c`：海流估计误差，仅 OC

构造：

```text
R_hat     = Exp([δθ]x) R
nu_r_hat  = nu_r + δnu_r
u_act_hat = u_act + δu_act
v_c_hat   = v_c + δv_c
```

### Step 3. 再回到 data convention

用 noisy 的 `R_hat` 和 `v_c_hat` 重构：

```text
nu_total_hat[:3] = nu_r_hat[:3] + R_hat^T v_c_hat
nu_total_hat[3:] = nu_r_hat[3:]
```

然后得到 noisy data-state：

```text
x0_hat = [Δp=0, R_hat, nu_total_hat, u_act_hat, u_cmd_clean, v_c_hat]
```

这样做的好处是：

- OC 情况下 `nu_r` 预算是你显式控制的；
- 不会再出现“`nu_total` 看起来加得不大，但 `nu_r` 实际已经爆了”的问题；
- 姿态误差与 current 误差通过 `R_hat^T v_c_hat` 自动耦合，语义正确。

---

## 5. 噪声 profile 设计

## 5.1 设计原则

速度噪声不建议再用“所有轴统一标量”的方式。更合理的是：

```text
sigma_i = max(sigma_floor_i, alpha * std_i)
```

含义：

- `alpha * std_i` 反映该通道在数据中的自然尺度；
- `sigma_floor_i` 反映导航系统的最小测量不确定性；
- 这样既避免弱激励轴被统一标量过扰动，也避免把弱轴噪声压到不现实地接近 0。

## 5.2 以仓库主数据集为例的推荐参数

基于 `data/auv_oc_traj1000_blk150_s23_d0be9434.summary.txt` 的 `nu_r std by DOF`：

```text
u=0.4688, v=0.0917, w=0.0733, p=0.0250, q=0.0919, r=0.0534
```

我建议定义三个 profile：

| Profile | 用途 | `alpha` | 说明 |
|---|---|---:|---|
| `nominal_train` | 训练 | 0.03 | 当前 trainer 的主训练噪声 |
| `nominal_eval` | 评估 | 0.05 | 正常导航条件 |
| `degraded_eval` | 评估 | 0.10 | 明显退化但仍可工作 |

建议 floor：

- 线速度 floor: `0.005 m/s`
- 角速度 floor: `0.0015 rad/s`

据此可得一组可直接使用的近似值：

### `delta_nu_r` nominal_train

```text
[u, v, w, p, q, r] =
[0.014, 0.005, 0.005, 0.0015, 0.0028, 0.0016]
```

### `delta_nu_r` nominal_eval

```text
[0.023, 0.005, 0.005, 0.0015, 0.0046, 0.0027]
```

### `delta_nu_r` degraded_eval

```text
[0.047, 0.010, 0.010, 0.0030, 0.0092, 0.0053]
```

这里最重要的变化是：

- `v/w/p` 不再承受当前实现那种 20% 以上的相对扰动；
- `u/q/r` 仍保留与数据尺度匹配的真实扰动；
- OC 时 `nu_r` 预算变成显式可控。

## 5.3 其他状态的推荐噪声

### 姿态初值误差 `δθ`

| Profile | 推荐值 |
|---|---:|
| `nominal_train` | `0.0035 rad` |
| `nominal_eval` | `0.0050 rad` |
| `degraded_eval` | `0.0120 rad` |

说明：

- `0.005 rad ≈ 0.29°`，作为正常导航初始姿态误差是合理的；
- 训练值略小于评估值，更利于收敛。

### current estimate 误差 `δv_c`

建议不要再用统一的 `0.05 m/s` 作为训练默认值。

推荐：

| Profile | `v_cx, v_cy` | `v_cz` |
|---|---:|---:|
| `nominal_train` | `0.008 m/s` | `0.004 m/s` |
| `nominal_eval` | `0.012 m/s` | `0.006 m/s` |
| `degraded_eval` | `0.030 m/s` | `0.015 m/s` |

这组值不会再主导 `nu_r` 误差，同时仍保留 OC 估计不确定性。

### actuator 反馈误差 `δu_act`

必须改成按通道配置，而不是单一标量。

推荐：

| 通道 | nominal_train | nominal_eval | degraded_eval |
|---|---:|---:|---:|
| `delta_r` | `0.002 rad` | `0.003 rad` | `0.008 rad` |
| `delta_s` | `0.002 rad` | `0.003 rad` | `0.008 rad` |
| `rpm` | `3 rpm` | `5 rpm` | `15 rpm` |

如果你后续决定把 actuator 状态先从噪声建模里拿掉，也完全可以。它不是当前性能崩溃的主要矛盾。

---

## 6. 训练协议建议

## 6.1 当前 trainer 下的推荐训练方式

在当前架构下，我不建议“全量 noisy training”。

更稳妥的做法是 **mixed clean/noisy IC training**：

| 阶段 | 训练样本组成 |
|---|---|
| epoch 1-20 | 100% clean |
| epoch 21-80 | 70% clean + 30% `nominal_train` |
| epoch 81+ | 50% clean + 50% `nominal_train` |

原因：

- 干净样本保留动力学上界；
- mild noisy IC 提供鲁棒性正则；
- 避免一开始就让 optimizer 面对偏差过大的初值。

如果你更想要简单版本，也可以直接：

```text
warmup clean 20 epochs
+ thereafter 50% clean / 50% nominal_train
```

## 6.2 验证与 checkpoint 选择

如果启用 noisy training，验证不能再只看 clean。

建议至少同时报告：

- `clean_val`
- `nominal_noisy_val`

checkpoint 选择分数建议用：

```text
score = 0.5 * clean_val + 0.5 * nominal_noisy_val
```

否则模型会继续被系统性地选成“对 clean 最优、对 noisy 不稳”。

---

## 7. 评估协议建议

## 7.1 当前阶段的主评估

在 held-out 上固定一组 noise seeds，统一评估：

- `clean`
- `nominal_eval`
- `degraded_eval`

每个 seed 对所有模型共享同一批噪声 realizations。

主指标建议继续看：

- position RMSE
- rotation geodesic
- linear velocity RMSE
- angular velocity RMSE
- per-DOF RMSE

并额外报告：

```text
degradation ratio = noisy_metric / clean_metric
```

因为真正想比较的是“谁退化得更慢”。

## 7.2 DVL dropout 只作为 stress test

dropout 不建议进入当前主训练。

如果要评估 failure case，应单独加一个 stress profile：

- 仅冻结线速度 `[u, v, w]`
- 以窗口形式触发，而不是每步独立 Bernoulli 更合理
- 不冻结角速度

这部分只做 eval，不做当前主训练。

---

## 8. 方案 B：未来的序列噪声训练

如果研究目标是真正回答：

```text
模型能否从带噪观测序列中学习动力学？
```

那就必须升级 trainer。

推荐最小改法：

1. 从 noisy observation block 中取 `t_k` 的 noisy state；
2. 做 `k-step` rollout，`k=2~4`；
3. 与 clean latent future 对齐；
4. 重复滑窗训练。

只有在这一步完成后，下面这些噪声才值得重新启用：

- AR(1) correlated sequence noise
- block bias / slow drift
- random-walk bias
- dropout windows

在那之前，这些内容不应再作为当前训练方案的核心。

---

## 9. 对代码层面的直接建议

如果后续要实现，我建议按下面的结构改：

1. 新增一个 **IC-only** 噪声构造函数，只处理 `x0`。
2. 该函数内部先转 ODE 语义，再回到 data convention。
3. `delta_nu_r`、`delta_v_c`、`delta_u_act` 改成**向量化配置**。
4. 当前 `build_noisy_training_pair()` 中和 `t>0` 有关的 AR(1)、位置积分、姿态积分逻辑，不要再用于主训练路径。
5. noisy validation 和 held-out noisy evaluation 单独实现，且固定 noise seeds。
6. dropout 逻辑只作用于线速度通道，并只放在 eval stress path。

---

## 10. 最终建议

如果只选一条最重要的结论：

> **先把噪声定义在 `nu_r` 语义上，再重构回 `nu_total`，并把当前训练严格收敛成“mild noisy IC + clean target”的方案。**

这是当前仓库里最关键、也最可能立即改善结果的一步。

相比继续调现在这套 `Level 1/2/3` 参数，这个改动更根本，也更正确。
