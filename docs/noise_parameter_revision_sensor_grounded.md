# 噪声参数修订方案：基于传感器真实精度的标定

**分支：** `cc-noise`
**日期：** 2026-04-08
**状态：** 待实现

---

## 1. 修订动机

### 1.1 现有方案的根本问题

当前速度噪声公式：

```python
sigma_i = max(floor_i, alpha * dataset_std_i)
```

其中 `dataset_std_i` 来自 `StateNormalizer.from_dataset()`，即训练集上各速度分量的**标准差**（动力学变化范围）。这是对速度在不同任务轨迹间**变化幅度**的统计，与导航滤波器对当前速度状态的**估计误差**在物理概念上截然不同，数值上可能相差一个数量级。

**具体问题：** 以 REMUS 100 标准数据集为例（纵向速度在 0-2.5 m/s 范围内变化，`std_u ≈ 0.5 m/s`）：

| 通道 | 旧值 nominal_eval（`alpha=0.05`） | 实际 DVL 精度 |
|------|----------------------------------|---------------|
| `u`（纵向） | `0.05 × 0.5 = 0.025 m/s` | ≈ 0.004–0.005 m/s |
| `v`（横向） | `max(0.05×0.1, 0.005) = 0.005 m/s` | ≈ 0.004–0.005 m/s |

纵向速度噪声被**高估约 5 倍**，原因是 DVL 精度与飞行速度范围无关，而旧方案把速度范围当成精度代理量。

### 1.2 旋转噪声各向同性问题

当前三轴姿态噪声使用同一个标量 `rot_std`，但实际上：
- **横滚/俯仰**：受重力持续约束，加速度计提供参考，EKF 误差较小且有界
- **航向**：水下无绝对参考（无 GPS、磁罗盘精度有限），误差显著大于前两者

当前 `nominal_eval = 0.005 rad ≈ 0.3°` 接近横滚/俯仰精度，**低估了航向不确定性约 3–4 倍**。

---

## 2. REMUS 100 传感器规格基准

### 2.1 DVL：Teledyne RDI Explorer 1200 kHz

REMUS 100 典型配置（经 NTNU AUR-Lab、WHOI OSL 文档及 DSTO 技术报告 ADA604237 证实）：

**底部跟踪（Bottom-Track）精度：**
- 长期系统精度：**±0.2% × 速度 ± 1 mm/s**
  - 在 1.5 m/s 时：±(0.002×1.5 + 0.001) = **±0.004 m/s**
  - 在 2.0 m/s 时：±(0.002×2.0 + 0.001) = **±0.005 m/s**
- 单 ping 标准差：±3 mm/s（1200 kHz）

**BT 测量量语义：** DVL 底部跟踪测量的是 AUV 体系坐标下**相对于海底的速度**，
即总体速度 `v_total`（非相对速度 `v_r`）。在有海流场景下，`v_total = v_r + R^T v_c^n`。
由于 DVL 噪声是加性的，`nu_r = nu_total - R^T v_c^n` 为线性变换，
在 `nu_r` 上加等幅高斯噪声与在 `nu_total` 上加噪**数学等价**。
详细论证见 [noise_model_design.md](./noise_model_design.md) Section 4.2。

**水体跟踪（Water-Track）精度：** 与 BT 相同规格

**数据来源：**
> Teledyne RDI Workhorse Navigator / Explorer Series Datasheet（确认于 BODC 镜像
> `rdi_workhorse_nav_ds_lr.pdf`：BT accuracy ±0.2% ± 1 mm/s for 600/1200 kHz）
> Teledyne Marine Pathfinder DVL product page（±0.06%/±0.2% ± 0.1 cm/s，视底部距离而定）
> DSTO Technical Report ADA604237（REMUS 100 系统级 dead-reckoning 精度 ≈0.5% 航程）

### 2.2 IMU：Honeywell HG1700 AG58（RLG 战术级）

研究型 REMUS 100 的标准 IMU（确认于 NovAtel / Honeywell 产品页及 NTNU 文档）：

| 参数 | 数值 |
|------|------|
| 陀螺角度随机游走（ARW） | 0.125 deg/√hr = **3.64×10⁻⁵ rad/√s** |
| 陀螺运行偏置稳定性 | 1.0 deg/hr = **4.85×10⁻⁶ rad/s** |
| 陀螺尺度因子 | 150 ppm |
| 加速度计偏置 | 1.0 mg |
| 加速度计尺度因子 | 300 ppm |

**导航滤波器后验角速度不确定性推算：**
- ARW 贡献（1 Hz 有效带宽）：`3.64×10⁻⁵ × √1 ≈ 3.6×10⁻⁵ rad/s`（极小）
- 实际主导项为残余振动混叠与尺度因子误差，保守估计：**0.5–3 mrad/s**

**数据来源：**
> Honeywell HG1700 AG58 datasheet（NovAtel 镜像页，
> Gyro bias 1.0 deg/hr, ARW 0.125 deg/sqrt(hr), scale factor 150 ppm）

### 2.3 AHRS 姿态精度（滤波后）

滤波器输出姿态误差由两类因素决定：

**横滚/俯仰（重力约束）：**
- 加速度计偏置 1 mg → 静态倾角误差 ≈ arctan(0.001 g / g) ≈ **1 mrad**
- 动态机动 + 滤波残差：**3–7 mrad（约 0.2–0.4°）**

**航向（无绝对参考，磁罗盘辅助）：**
- 典型配置为 KVH 或同级磁罗盘，开阔水域精度 **±0.5–2.0°（0.009–0.035 rad）**
- 校准良好的中纬度开阔海域：**±1.0°（0.017 rad）** 作为名义精度
- 沿岸/港湾/钢结构附近磁场干扰：可达 **±3.0°（0.052 rad）**

> 注：军用 REMUS 100M 配备 Kearfott T-16 INS（RLG 导航级），航向精度可达 0.1–0.5°，
> 但在研究型部署中最常见的是 HG1700 + 磁罗盘组合。本方案以后者为基准。

**数据来源：**
> HG1700 加速度计偏置 1 mg（同上 Honeywell datasheet）；
> KVH 1725/1775 系列磁罗盘精度 ±0.5° RMS（KVH Industries product page）；
> 沿海磁偏差估计来自 BGS 世界磁模型说明文档

### 2.4 深度传感器

- 型号：Honeywell 汽车级压力传感器（DSTO 报告 ADA604237 明确记录）
- 规格：**±1% FS BFSL ≈ ±1.4 m（在 100 m 额定深度）**
- 实际误差：每次下潜前水面调零后，主要误差来自水体密度差异（热跃层）：
  - 均匀水体：±0.1–0.3 m；存在热跃层：±0.5–1.5 m

**数据来源：**
> DSTO Technical Report ADA604237，第 2.3 节：
> "Honeywell automotive-grade transducer… specified accuracy of 1% FS BFSL"

### 2.5 执行器反馈传感器

REMUS 100 舵面（舵角/升降舵）：
- 反馈传感器：电位计（pot）或 LVDT
- 典型角度范围：±15°（±0.26 rad）
- 精度：**±0.5–1.0°（0.009–0.017 rad）**

螺旋桨转速反馈：
- 传感器：霍尔效应传感器
- 工作转速范围：0–1500 RPM
- 精度：**±5–10 RPM**

**数据来源：**
> 电位计/LVDT 舵机反馈精度为行业通用规范（REMUS 100 具体型号未公开），
> 参考 Measurement Specialties/TE Connectivity 舵机位移传感器规格书（±0.5° 典型值）；
> 霍尔效应 RPM 传感器精度 ±5–10 RPM 为通用规范，REMUS 100 螺旋桨参数参考
> Remus100Dynamics（本仓库 `remus100_core.py`）：`D_prop=0.14 m, KT_0=0.4566`

---

## 3. 修订方案详述

### 3.1 变更 1：速度噪声替换为绝对值（最重要）

**影响文件：** `train_utils.py`

**删除：**
```python
def _profile_alpha(profile: str) -> float:
    return {"nominal_train": 0.03, "nominal_eval": 0.05, "degraded_eval": 0.10}.get(profile, 0.0)
```

以及 `build_noisy_initial_condition` 中的：
```python
vel_std = normalizer.std_vel.to(...)
alpha = _profile_alpha(cfg.profile)
nu_r_std = scale * torch.maximum(alpha * vel_std, vel_floor)
```

**新增：**
```python
def _profile_velocity_std(
    profile: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    绝对速度噪声 1-sigma [u, v, w, p, q, r]，基于 REMUS 100 传感器规格。

    线速度（m/s）——Teledyne RDI Explorer 1200 kHz DVL 底部跟踪精度：
      系统精度 ±0.2%×V ± 1 mm/s（长期）
      在 1.5 m/s 时：±0.004 m/s；在 2.0 m/s 时：±0.005 m/s
      nominal_eval 取 0.005 m/s（含 25% 余量）
      degraded: DVL 底跟踪临界（浅水面反射/近悬停低速）→ 约 4× nominal

    角速度（rad/s）——Honeywell HG1700 AG58 陀螺：
      ARW = 0.125 deg/√hr = 3.64×10⁻⁵ rad/√s（远低于 floor）
      运行偏置 = 1 deg/hr = 4.85×10⁻⁶ rad/s（可忽略）
      滤波后主导项为残余振动混叠，保守估计 0.5–3 mrad/s

    参考文献：
      [1] Teledyne RDI Workhorse Navigator/Explorer datasheet via BODC
      [2] Honeywell HG1700 AG58 datasheet (NovAtel mirror)
    """
    lin = {
        "nominal_train": 0.003,   # ~DVL 噪声底；轻度 IC 正则
        "nominal_eval":  0.005,   # DVL BT @ 1.5 m/s 实测精度（含余量）
        "degraded_eval": 0.020,   # DVL 临界工况；约 4× nominal
    }.get(profile, 0.0)
    ang = {
        "nominal_train": 0.0005,  # HG1700 级陀螺，滤波后偏紧
        "nominal_eval":  0.001,   # 1 mrad/s，保守估计
        "degraded_eval": 0.003,   # 振动/大机动时的 3 mrad/s
    }.get(profile, 0.0)
    return torch.tensor(
        [lin, lin, lin, ang, ang, ang], dtype=dtype, device=device
    )
```

**替换 `build_noisy_initial_condition` 中的速度噪声部分：**
```python
# 旧：
nu_r_std = scale * torch.maximum(alpha * vel_std, vel_floor)

# 新：
nu_r_std = scale * _profile_velocity_std(cfg.profile, dtype=dtype, device=device)
```

**`NoiseConfig` 中的 floor 值更新：**
```python
# 旧：
linear_floor_std:  float = 0.005   # m/s  （相对值方案的防呆下界，偏大）
angular_floor_std: float = 0.0015  # rad/s

# 新（绝对值方案下，floor 仅作安全下界）：
linear_floor_std:  float = 0.001   # m/s  （DVL 硬件噪声底：1 mm/s）
angular_floor_std: float = 0.0002  # rad/s（HG1700 陀螺噪声底）
```

> **备注：** 切换到绝对值后，`build_noisy_initial_condition` 中的 `torch.maximum(profile_std, floor)`
> 实际由 profile 值主导，floor 仅作保护，可去掉 `torch.maximum` 改为直接使用 profile 值。
> `normalizer` 参数保留（仍用于 loss 归一化和 `u_dim≠3` 的执行器噪声 fallback），
> 但不再用于速度噪声计算。

---

### 3.2 变更 2：旋转噪声改为各向异性

**影响文件：** `train_utils.py`

**删除旧标量版本：**
```python
def _profile_rotation_std(profile: str) -> float:
    return {
        "nominal_train": 0.0035, "nominal_eval": 0.0050, "degraded_eval": 0.0120,
    }.get(profile, 0.0)
```

**新增向量版本：**
```python
def _profile_rotation_std(
    profile: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    各向异性旋转噪声 1-sigma [sigma_roll, sigma_pitch, sigma_yaw]（rad）。

    `delta_theta` 的三个分量对应 AUV 体坐标系中绕 x/y/z 轴的小角度扰动，
    经 SO(3) 指数映射作用到旋转矩阵（x=横滚, y=俯仰, z=航向）。

    横滚/俯仰（重力约束）：
      Honeywell HG1700 AG58 加速度计偏置 1 mg
        → 静态倾角误差 ≈ arctan(0.001) ≈ 1 mrad（约 0.06°）
      动态机动 + 滤波残差：3–7 mrad（约 0.2–0.4°）
      nominal_eval 取 5 mrad（≈ 0.3°），含动态余量

    航向（无重力约束，仅磁罗盘）：
      KVH / 同级磁罗盘精度：±0.5–2.0°（0.009–0.035 rad）
      开阔水域校准良好：±1.0°（0.017 rad）作为名义值
      沿海/港湾/磁场干扰环境：±3.0°（0.052 rad）

    参考文献：
      [1] Honeywell HG1700 AG58 datasheet (NovAtel mirror)：accel bias 1 mg
      [2] KVH 1725 Fiber Optic Gyro Compass datasheet：heading accuracy ±0.5° RMS
      [3] 沿海磁偏差：BGS World Magnetic Model technical note
    """
    tilt, yaw = {
        "nominal_train": (0.003, 0.009),   # tilt ~0.2°, heading ~0.5°；轻度正则
        "nominal_eval":  (0.005, 0.017),   # tilt ~0.3°, heading ~1.0°；典型磁罗盘精度
        "degraded_eval": (0.015, 0.052),   # tilt ~0.9°, heading ~3.0°；磁场干扰场景
    }.get(profile, (0.0, 0.0))
    return torch.tensor([tilt, tilt, yaw], dtype=dtype, device=device)
```

**更新 `build_noisy_initial_condition` 中的旋转扰动部分：**
```python
# 旧：
rot_std = scale * _profile_rotation_std(cfg.profile)   # float
if rot_std > 0.0:
    delta_theta = _sample_scaled_noise(
        torch.full((3,), rot_std, dtype=dtype, device=device),
        ...
    )

# 新：
rot_std_vec = scale * _profile_rotation_std(cfg.profile, dtype=dtype, device=device)  # Tensor[3]
if torch.any(rot_std_vec > 0):
    delta_theta = _sample_scaled_noise(rot_std_vec, ...)  # _sample_scaled_noise 已支持向量 std
```

`_so3_exp_map`、`_project_to_so3` 接口不变，无需修改。

---

### 3.3 变更 3：执行器噪声对齐传感器精度

**影响文件：** `train_utils.py`，`_profile_actuator_std` 中 `u_dim == 3` 分支

**物理依据：**
- 舵面角度反馈（电位计）：精度 ±0.5–1.0°（0.009–0.017 rad）
  旧值 `nominal_eval = 0.003 rad ≈ 0.17°` 为高精度编码器**分辨率**，
  不代表 REMUS 100 实际使用的电位计**精度**
- RPM 反馈（霍尔效应）：精度 ±5–10 RPM，旧值 `nominal_eval = 5 RPM` 偏紧

**新值：**
```python
# u_dim == 3: [delta_r (rad), delta_s (rad), RPM]
table = {
    "nominal_train": [0.004, 0.004,  3.0],  # 舵角 ~0.23°，RPM 3（轻度正则）
    "nominal_eval":  [0.009, 0.009,  8.0],  # 舵角 ~0.52°（对应 ±0.5° 电位计精度）
    "degraded_eval": [0.017, 0.017, 20.0],  # 舵角 ~0.97°（老化/未校准）；RPM 20
}
```

---

### 3.4 变更 4（条件性）：depth_ref 噪声

**影响条件：** `absolute_depth_context=True`（此时状态向量包含 `layout.depth_ref`）

**物理依据：**
- Honeywell 汽车级压力传感器精度 ±1.4 m（原始）；表面调零后误差来自水体密度差异
- 均匀水体：±0.1–0.3 m；热跃层：±0.5–1.5 m

**新增辅助函数：**
```python
def _profile_depth_ref_std(profile: str) -> float:
    """
    绝对深度参考（depth_ref）的 1-sigma 不确定性（m）。
    误差来源：水体密度差异导致的压强-深度换算误差（已表面调零）。

    参考文献：
      [1] DSTO ADA604237：REMUS 100 Honeywell 压力传感器 ±1% FS BFSL
      [2] IOC UNESCO 海水状态方程（密度对深度换算的影响）
    """
    return {
        "nominal_train": 0.0,    # 训练时默认不加，避免影响势能梯度学习早期阶段
        "nominal_eval":  0.3,    # 均匀水体，标定良好
        "degraded_eval": 1.0,    # 热跃层或密度异常区域
    }.get(profile, 0.0)
```

**在 `build_noisy_initial_condition` 末尾添加：**
```python
if getattr(model, "absolute_depth_context", False):
    depth_std = scale * _profile_depth_ref_std(cfg.profile)
    if depth_std > 0.0:
        depth_noise = _sample_scaled_noise(
            torch.tensor([depth_std], dtype=dtype, device=device),
            batch_size,
            device=device,
            dtype=dtype,
            sample_ids=sample_ids,
            base_seed=base_seed,
            stream=67,
        )
        y0[:, layout.depth_ref] = y0[:, layout.depth_ref] + depth_noise
```

> **注意：** 必须使用 `_sample_scaled_noise` 而非直接 `torch.randn`，
> 以保证评估阶段通过 `sample_ids` + `base_seed` 获得确定性噪声，
> 与其他噪声通道的种子协议一致。

> **注意：** 此变更仅在 `absolute_depth_context=True` 时生效。
> 如果 `include_depth_in_potential=False`，depth_ref 不进入势能计算，
> 此噪声对 rollout 无实质影响，可以忽略。

---

## 4. 变更前后数值对比

以典型 REMUS 100 数据集（`std_u ≈ 0.5 m/s`）为参照，对比各 profile 的噪声 1-sigma：

### 4.1 速度噪声

| 通道 | 旧 nominal_eval | **新 nominal_eval** | 旧 degraded_eval | **新 degraded_eval** |
|------|----------------|---------------------|-----------------|----------------------|
| u (m/s) | 0.025 | **0.005** ↓5× | 0.050 | **0.020** ↓2.5× |
| v (m/s) | 0.005 | **0.005** ≈ | 0.010 | **0.020** ↑2× |
| w (m/s) | 0.005 | **0.005** ≈ | 0.005–0.010 | **0.020** ↑2–4× |
| p (rad/s) | 0.0015 | **0.001** ↓ | 0.002 | **0.003** ↑1.5× |
| q,r (rad/s) | 0.0025 | **0.001** ↓2.5× | 0.005 | **0.003** ↓1.7× |

> **分析：** 纵向速度（u）噪声大幅减小，因为旧方案用 dataset_std=0.5 m/s × alpha=0.05=0.025 m/s，
> 远超 DVL 实际精度 0.004–0.005 m/s。横向速度（v,w）在旧方案中因 floor 生效而与新值相近，
> 但 degraded_eval 中新方案将其上调至与 u 相同，反映 DVL 四波束均匀退化。

### 4.2 旋转噪声（旧为各向同性，新为各向异性）

| 轴 | 旧 nominal_eval | **新 nominal_eval** | 旧 degraded_eval | **新 degraded_eval** |
|----|----------------|---------------------|-----------------|----------------------|
| 横滚 roll (rad) | 0.005 | **0.005** ≈ | 0.012 | **0.015** ↑1.3× |
| 俯仰 pitch (rad) | 0.005 | **0.005** ≈ | 0.012 | **0.015** ↑1.3× |
| **航向 yaw (rad)** | **0.005** | **0.017** ↑**3.4×** | **0.012** | **0.052** ↑**4.3×** |

> **分析：** 这是最重要的定性改变。旧方案严重低估了航向不确定性：
> 磁罗盘典型精度 ±1°（0.017 rad）远大于重力约束的倾角精度 ±0.3°（0.005 rad）。
> 新方案将航向噪声上调 3–4 倍，使其与真实 AHRS+磁罗盘组合的性能一致。

### 4.3 执行器噪声

| 通道 | 旧 nominal_eval | **新 nominal_eval** | 旧 degraded_eval | **新 degraded_eval** |
|------|----------------|---------------------|-----------------|----------------------|
| delta_r (rad) | 0.003 | **0.009** ↑3× | 0.008 | **0.017** ↑2× |
| delta_s (rad) | 0.003 | **0.009** ↑3× | 0.008 | **0.017** ↑2× |
| RPM | 5 | **8** ↑1.6× | 15 | **20** ↑1.3× |

---

## 5. 不修改的部分及理由

| 现有设计 | 理由 |
|----------|------|
| 海流噪声（`_profile_current_std`）值 | `nominal_eval: [0.012, 0.012, 0.006] m/s` 合理对应 DVL BT/WT 差值法（~0.007 m/s RSS）加时空变化余量；垂向取水平一半反映海流场垂向变化较慢。不修改 |
| SO(3) 指数映射 + SO(3) 投影 | 几何正确，各向异性输入天然兼容，无需修改 |
| Curriculum 调度（warmup/ramp/mix_ratio） | 与噪声幅度标定无关，不修改 |
| `_sample_scaled_noise` | 已支持向量 std，无需修改 |
| `normalizer` 参数 | 保留（用于 loss 归一化 + `u_dim≠3` 执行器 fallback） |
| profile 名称和枚举 | 保持兼容性 |
| 各通道独立采样（无交叉相关） | 真实 EKF 后验中姿态/速度/海流误差存在交叉相关（通过 R 耦合），但相关系数高度依赖 EKF 调参和传感器可用性。独立采样覆盖更广的初始条件空间，对鲁棒性测试是更保守的简化 |
| ODE 空间（nu_r）加噪而非数据空间（nu_total） | DVL BT 噪声为加性高斯，nu_r = nu_total - R^T v_c^n 为线性变换，两者等价。姿态误差通过 R^T v_c^n 引入的二阶交叉项量级约 δR × v_c ≈ 0.017×0.2 ≈ 0.003 m/s，在 nominal_eval 层级与 DVL 噪声同量级，作为 IC 扰动可接受 |

---

## 6. 参考文献

1. **Teledyne RDI Workhorse Navigator/Explorer Datasheet**
   Bottom-track accuracy ±0.2% ± 0.1 cm/s (600/1200 kHz)
   来源：BODC镜像 `rdi_workhorse_nav_ds_lr.pdf`；Teledyne Marine Pathfinder DVL 产品页

2. **Honeywell HG1700 AG58 IMU Datasheet**
   Gyro bias 1.0 deg/hr, ARW 0.125 deg/√hr, accel bias 1.0 mg
   来源：NovAtel/Honeywell Aerospace 产品页

3. **DSTO Technical Report ADA604237**
   "Shallow Water Bathymetric Survey Using the REMUS 100 AUV"
   DVL dead-reckoning ≈ 0.5% 航程；深度传感器 ±1% FS BFSL；REMUS 100 系统配置描述

4. **NTNU AUR-Lab REMUS 100 Documentation**
   "1200 kHz Explorer R100 Doppler Velocity Log" 确认于 NTNU AUR-Lab 设备页

5. **WHOI OSL REMUS 100 Documentation**
   "1200 kHz RDI ADCP for dead-reckoning navigation"

6. **KVH 1725 Fiber Optic Gyro Compass Datasheet**
   Heading accuracy ±0.5° RMS（KVH Industries 产品页）
   用于 nominal_eval 航向噪声 0.017 rad（≈1.0°）的上限估计依据
