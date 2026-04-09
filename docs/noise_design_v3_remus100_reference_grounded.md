# 新版噪声设计 v3（REMUS100 参考约束版）

本文档用于替代“按训练集统计缩放噪声”的旧思路，给出一版更贴近典型 `REMUS100` 使用场景的新版噪声设计建议。

核心原则只有两条：

1. 噪声预算应尽量绑定传感器 / 导航链路，而不是绑定训练集分布。
2. 文档必须区分“官方资料可直接支持的事实”和“基于典型传感器链路的工程推断”。

---

## 1. 这次重新核对后可以确认的事实

### 1.1 当前代码里哪些噪声依赖数据集统计

当前实现中，只有 `delta_nu_r` 明确依赖训练集统计，来源如下：

- [`StateNormalizer.from_dataset()`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L361) 从训练集计算 `std_vel`
- [`train_auv_hamnode.py`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_auv_hamnode.py#L545) 先把 `oc` 数据适配到模型空间，再统计 `std_vel`
- [`summarize_noise_budget()`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L758) 和 [`build_noisy_initial_condition()`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L944) 都使用
  `nu_r_std = scale * max(alpha * std_vel, floor)`

对应 profile 系数是固定表：

- [`_profile_alpha()`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L614): `nominal_train=0.03`, `nominal_eval=0.05`, `degraded_eval=0.10`

其余几类噪声当前是固定表，不依赖训练集统计：

- 姿态随机误差与 `heading_biased_eval`：[`train_utils.py`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L623)
- current 误差与 `current_bias_eval`：[`train_utils.py`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L658)
- 执行器误差：[`train_utils.py`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L680)

### 1.2 典型 REMUS100 的导航链路

`REMUS 100` 官方页面明确写到：

- dead reckoning 使用 `ADCP` 速度和 `compass/rate gyro` heading
- 更高配置可加 `T-16 Inertial Navigation System`
- inertial navigator 可把 DR 误差扩展到约 `5 m/hour`

来源：

- [S1] WHOI REMUS 100 page: https://www2.whoi.edu/site/osl/vehicles/remus-100/

这意味着：

- 对“典型 REMUS100”而言，标准场景更像 `ADCP/DVL + compass/rate gyro` 的 DR，而不是默认拥有高等级 INS 和稳定 current-state。
- `current-observable` 更适合写成增强配置或专门假设，不宜再写成默认部署现实。

### 1.3 速度误差更像“固定底噪 + 比例项”，而不是“按数据集 std 缩放”

Teledyne 官方 DVL/ADCP资料给出的速度误差结构是典型的：

- `accuracy = percentage of measured velocity + cm/s floor`

可直接参考的官方资料：

- [S2] Teledyne Pathfinder DVL Guide:
  https://www.teledynemarine.com/en-us/support/SiteAssets/RDI/Manuals%20and%20Guides/Pathfinder/PathFinder_DVL_Guide.pdf
- [S3] Teledyne Pathfinder DVL Guide (2022 mirror):
  https://www.teledynemarine.com/en-us/resources/Documents/Brand%20Support/RD%20INSTRUMENTS/Technical%20Resources/Manuals%20and%20Guides/Pathfinder/PathFinder%20DVL%20Guide_Apr22.pdf
- [S4] Teledyne Workhorse II Mariner ADCP:
  https://www.teledynemarine.com/brands/rdi/workhorse-mariner-adcp

其中公开规格可支持这样的判断：

- 高性能 DVL 可以做到远优于 `1 cm/s`
- 常见海洋测速产品也经常以 `0.5%~1.0% of velocity + 0.5 cm/s` 这种形式给出精度

因此，把 `nu_r` 建模成“比例项 + floor”是合理的；把它建模成“训练集标准差的固定百分比”则缺乏物理依据。

### 1.4 heading 误差应区分标准 DR 和高配 INS

可直接参考的官方资料：

- [S5] VectorNav VN-100 AHRS:
  https://www.vectornav.com/products/detail/vn-100
- [S6] VectorNav Tactical Embedded:
  https://www.vectornav.com/products/tactical-embedded
- [S7] VectorNav VN-210 GNSS/INS:
  https://www.vectornav.com/products/detail/vn-210

这些资料支持一个稳定结论：

- 标准 AHRS / compass 级 heading 通常是 `degree-level`
- 高配 tactical INS / aided INS 的 heading 可以进入明显更小的子度量级

因此，`heading` 预算不应只有一套，而应至少区分：

- `REMUS100-DR` 默认场景
- `REMUS100-INS` 增强场景

这里还要明确：

- [S5][S6][S7] 用于界定“标准 AHRS”与“高配惯导”两类精度等级
- 它们不是 `REMUS100` 官方装机清单

---

## 2. 对旧版数值准确性的再判断

### 2.1 可以保留的结论

- 执行器噪声量级总体不离谱
- yaw-dominant 姿态扰动方向是对的
- `heading_biased_eval` / `current_bias_eval` 作为专门压力测试 profile 是合理的

### 2.2 需要修正的结论

#### A. `delta_nu_r` 不应继续使用训练集统计

这是旧版最大问题。原因不是“数值一定很大或很小”，而是：

- 同一 profile 在不同数据集上的物理意义会漂移
- profile 名称不再对应固定的设备 / 导航场景
- 无法把实验结果自然解释成“典型 REMUS100 会遇到的初值误差”

#### B. 旧版 `current` 误差过于像“默认可观测状态”

旧版：

- `nominal_eval`: `0.012 / 0.012 / 0.006 m/s`
- `degraded_eval`: `0.030 / 0.030 / 0.015 m/s`

这组数值本身不一定荒谬，但如果把它写成“典型 REMUS100 默认状态估计误差”，证据不够。

更稳妥的写法应是：

- 对默认 `REMUS100-DR` 场景，不把 `v_c` 作为主状态预算
- 对增强 `REMUS100-INS` 场景，`v_c` 作为工程假设预算单独出现

#### C. 旧版 `heading_biased_eval = 0.015 rad` 偏保守

`0.015 rad` 约等于 `0.86°`。如果目标真是典型 `compass/rate gyro` 主导的 DR 初值误差，这个 bias 更像“轻微偏差”，还不像典型部署中的主要脆弱点。

---

## 3. v3 设计总原则

### 3.1 用两类部署场景取代单一 profile 含义

建议后续文档和代码都围绕两类语义组织：

1. `REMUS100-DR`
   典型 `ADCP/DVL + compass/rate gyro` dead reckoning 场景
2. `REMUS100-INS`
   带高质量 inertial navigator 的增强场景

### 3.2 默认主实验采用 `REMUS100-DR`

原因：

- 这和 [S1] 的官方表述更一致
- 更接近“典型使用场景”
- 也更适合作为 robustness 主结论

### 3.3 `delta_nu_r` 改成状态相关但与数据集无关

建议把线速度噪声改成：

```text
sigma_lin_i = sqrt(floor_i^2 + (k_i * |nu_r_i|)^2)
```

角速度先采用固定预算即可：

```text
sigma_ang_i = constant
```

这属于“传感器 / 状态幅值相关”，不是“训练集相关”。

---

## 4. v3 推荐数值

## 4.1 默认主方案：REMUS100-DR

### A. `delta_nu_r`

线速度：

- `nominal_train`
  - floor: `[0.002, 0.002, 0.003] m/s`
  - ratio: `0.002`
- `nominal_eval`
  - floor: `[0.003, 0.003, 0.004] m/s`
  - ratio: `0.003`
- `degraded_eval`
  - floor: `[0.006, 0.006, 0.008] m/s`
  - ratio: `0.006`

角速度：

- `nominal_train`: `[0.0006, 0.0006, 0.0010] rad/s`
- `nominal_eval`: `[0.0008, 0.0008, 0.0015] rad/s`
- `degraded_eval`: `[0.0015, 0.0015, 0.0030] rad/s`

设计依据：

- 采用 [S2][S3][S4] 支持的“比例项 + floor”结构
- 同时保留对 heave / yaw 略强于水平面的保守性

### B. `delta_theta`

- `nominal_train`: `[0.0015, 0.0015, 0.0060] rad`
- `nominal_eval`: `[0.0020, 0.0020, 0.0100] rad`
- `degraded_eval`: `[0.0040, 0.0040, 0.0250] rad`

换算后，`nominal_eval` 的 yaw 约为 `0.57°`，`degraded_eval` 的 yaw 约为 `1.43°`。

这组值的含义不是“单传感器原始噪声”，而是“进入 rollout 前的滤波后姿态初值误差预算”。

### C. `heading_biased_eval`

- 随机部分沿用 `nominal_eval`
- 固定 yaw bias: `0.035 rad`

即约 `2.0°`。

这比旧版 `0.015 rad` 更符合标准 DR 场景里 heading 偏差是主要风险这一判断。

### D. `delta_u_act`

- `nominal_train`: `[0.002 rad, 0.002 rad, 3 rpm]`
- `nominal_eval`: `[0.003 rad, 0.003 rad, 5 rpm]`
- `degraded_eval`: `[0.004 rad, 0.004 rad, 10 rpm]`

这部分建议只做小幅收紧，不把它当作主矛盾。

### E. `delta_v_c`

默认 `REMUS100-DR` 主方案中：

- 不把 `v_c` 作为 nominal 主状态预算
- 不把 `current_bias_eval` 纳入默认主实验

如果代码层面必须保留 `oc` 状态位，建议：

- `nominal_eval`: 不注入 `v_c` 噪声，或仅作为极小占位量
- `current_bias_eval`: 保留为单独扩展测试，而不是默认 held-out profile

## 4.2 扩展方案：REMUS100-INS

只有当实验目标明确转向“带高质量惯导 / 增强 current estimation 的 REMUS100 配置”时，才建议使用这一组。

### A. `delta_theta`

- `nominal_train`: `[0.0008, 0.0008, 0.0030] rad`
- `nominal_eval`: `[0.0010, 0.0010, 0.0040] rad`
- `degraded_eval`: `[0.0020, 0.0020, 0.0100] rad`

### B. `delta_v_c`

- `nominal_train`: `[0.020, 0.020, 0.008] m/s`
- `nominal_eval`: `[0.030, 0.030, 0.010] m/s`
- `degraded_eval`: `[0.060, 0.060, 0.020] m/s`

### C. `current_bias_eval`

- 固定 bias: `[0.050, 0.050, 0.020] m/s`

注意：

- 这一组没有找到足以把它写成“典型 REMUS100 默认规格”的官方资料
- 它应被明确标注为增强配置下的工程假设

---

## 5. 新旧方案对比

## 5.1 `delta_nu_r`

| 项目 | 旧版 | v3 建议 | 变化原因 | 来源 |
|---|---|---|---|---|
| 结构 | `max(floor, alpha * std_dataset)` | `sqrt(floor^2 + (ratio * (state))^2)` | 去掉数据集依赖，改成传感器型误差结构 | 旧版实现见 [`train_utils.py`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L944)，结构依据见 [S2][S3][S4] |
| nominal_eval 线速度 | 由训练集决定 | floor `[0.003, 0.003, 0.004]`, ratio `0.003` | 物理意义固定 | [S2][S3][S4] + 工程推断 |
| degraded_eval 线速度 | 由训练集决定 | floor `[0.006, 0.006, 0.008]`, ratio `0.006` | 压力测试仍保留物理解释 | [S2][S3][S4] + 工程推断 |
| 角速度 | 由训练集角速度 std 放大 | 固定 budget | 陀螺误差更适合独立预算 | 工程推断 |

## 5.2 `delta_theta`

| 项目 | 旧版 | v3-DR 建议 | 变化原因 | 来源 |
|---|---|---|---|---|
| nominal_eval | 基础量级 `0.005 rad`，再映射 yaw-dominant | `[0.002, 0.002, 0.010] rad` | 明确区分 roll/pitch 与 yaw | 旧版实现见 [`train_utils.py`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L623)，场景依据见 [S1][S5][S6][S7] |
| degraded_eval | 基础量级 `0.012 rad` | `[0.004, 0.004, 0.025] rad` | 对 DR 压力测试更有辨识度 | [S1][S5] + 工程推断 |
| heading bias | `0.015 rad` | `0.035 rad` | 旧版偏保守 | 旧版实现见 [`train_utils.py`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L647)，量级依据见 [S1][S5] + 工程推断 |

## 5.3 `delta_u_act`

| 项目 | 旧版 | v3 建议 | 变化原因 | 来源 |
|---|---|---|---|---|
| nominal_eval | `[0.003, 0.003, 5 rpm]` | 保持不变 | 旧版已基本合理 | 旧版实现见 [`train_utils.py`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L680)，执行器时间常数见 [`remus100_core.py`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/remus100_core.py#L153) |
| degraded_eval | `[0.008, 0.008, 15 rpm]` | `[0.004, 0.004, 10 rpm]` | 旧版压力测试略重，收紧后更像反馈偏差而非执行器故障 | 同上 + 工程推断 |

## 5.4 `delta_v_c`

| 项目 | 旧版 | v3-DR 建议 | v3-INS 建议 | 变化原因 | 来源 |
|---|---|---|---|---|---|
| nominal_eval | `[0.012, 0.012, 0.006] m/s` | 默认不作为主状态预算 | `[0.030, 0.030, 0.010] m/s` | 避免把 `current-observable` 写成默认现实 | 旧版实现见 [`train_utils.py`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L669)，场景约束见 [S1]，数值为工程推断 |
| degraded_eval | `[0.030, 0.030, 0.015] m/s` | 非默认 | `[0.060, 0.060, 0.020] m/s` | current-state 预算应只在增强配置下出现 | 同上 |
| current bias | `[0.015, 0.015, 0.005] m/s` | 非默认 | `[0.050, 0.050, 0.020] m/s` | 旧版 bias 偏轻，且不应默认启用 | 旧版实现见 [`train_utils.py`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L658)，数值为工程推断 |

---

## 6. 这份 v3 文档里哪些数据是“准确确认”，哪些是“工程预算”

### 6.1 可直接确认

- `REMUS100` 标准 DR 由 `ADCP + compass/rate gyro` 支撑，增强配置可加 `T-16 INS`，见 [S1]
- DVL / ADCP 速度误差通常采用“百分比 + floor”的规格表达，见 [S2][S3][S4]
- AHRS 与 tactical INS 的 heading 精度存在明显等级差异，见 [S5][S6][S7]
- 本仓库当前噪声实现只有 `nu_r` 仍依赖训练集统计，见 [`train_utils.py`](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/train_utils.py#L944)

### 6.2 工程推断

- `REMUS100-DR` 下 `delta_theta` 的具体数值表
- `REMUS100-DR` 下 `heading_biased_eval = 0.035 rad`
- `REMUS100-INS` 下 `delta_v_c` 与 `current_bias_eval` 的具体数值表

这些不是 vendor-level 官方规格，而是：

- 由 [S1] 给出的导航链路约束
- 由 [S2][S3][S4] 给出的速度误差结构
- 由 [S5][S6][S7] 给出的 heading 等级差异
- 再结合本项目是“rollout 初始状态误差预算”这一建模目标

共同推导出的工程预算

---

## 7. 最终建议

如果后续要真正把 v3 落到代码，我建议这样执行：

1. 彻底移除 `delta_nu_r` 对 `normalizer.std_vel` 的依赖。
2. 引入两套显式模式：
   - `remus100_dr`
   - `remus100_ins`
3. 默认主实验切到 `remus100_dr`。
4. `current_bias_eval` 不再出现在默认主实验里，只作为增强场景压力测试。
5. 文档和论文正文都明确写出：
   - 哪些量是官方资料直接支撑
   - 哪些量是工程预算，而不是设备硬规格

---

## 8. 参考来源

- [S1] WHOI, REMUS 100:
  https://www2.whoi.edu/site/osl/vehicles/remus-100/
- [S2] Teledyne Marine, Pathfinder DVL Guide:
  https://www.teledynemarine.com/en-us/support/SiteAssets/RDI/Manuals%20and%20Guides/Pathfinder/PathFinder_DVL_Guide.pdf
- [S3] Teledyne Marine, Pathfinder DVL Guide (Apr 2022):
  https://www.teledynemarine.com/en-us/resources/Documents/Brand%20Support/RD%20INSTRUMENTS/Technical%20Resources/Manuals%20and%20Guides/Pathfinder/PathFinder%20DVL%20Guide_Apr22.pdf
- [S4] Teledyne Marine, Workhorse II Mariner ADCP:
  https://www.teledynemarine.com/brands/rdi/workhorse-mariner-adcp
- [S5] VectorNav, VN-100 IMU/AHRS:
  https://www.vectornav.com/products/detail/vn-100
- [S6] VectorNav, Tactical Embedded:
  https://www.vectornav.com/products/tactical-embedded
- [S7] VectorNav, VN-210 GNSS/INS:
  https://www.vectornav.com/products/detail/vn-210
