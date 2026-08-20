# AUV SE(3) 仓库 — 全部实验与结果完整清单

生成时间：2026-05-30  
工作区：`.`  
分支：`provenance-audit-phnode_full`

## 0. 本文档定位

本文档对当前仓库中**所有**训练实验、rollout 评估结果、数据集、报告与论文产物做一次**诚实、完整**的清点。

- 不做"哪些结果可以用于论文"的取舍判断；
- 对任何结论一律标注其来源报告（"报告称 X"），不注入本文档自身的论断；
- 覆盖范围包括被标记为 stale / unused / smoke 的实验，并如实记录其被排除/限定的理由。

口径来源：`checkpoints/` 磁盘文件、`analysis/oc_data_catalog/` 规范化 CSV、`analysis/provenance_audit/`、`docs/` 报告、`paper/` 草稿、各 run 的 `config.json`。

> 范围说明：本清单 §1–§J 覆盖当前 `g3_5_5` 仓库（全部 300 个 run 均为 `oc`）。更早的 `noc`/`oc` 实验（30 个 run，单一架构 `ph_se3_full` 的大 batch 配方扫描）属于**前代仓库 `g3_5_4`**。**2026-05-30 跨仓库复核**已把前代仓库作为新分区并入本文档：g3_5_4 的 `noc` 专节 + `oc` 专节（含统一口径 60s rollout 数值）见 **§K**，两仓库的总账对账表见 **§L**。模型命名映射（旧 11 → 新 10）与各报告散文结论的完整版仍在 [g3_5_4_legacy_noc_oc_inventory_zh.md](g3_5_4_legacy_noc_oc_inventory_zh.md)。当前论文 §8 不使用 `g3_5_4` 的任何结果（见 §L.4）。

---

## 1. 总览与对账

仓库磁盘上共有 **300 个**含 `best_model.pt` 的训练 run，可无缺口地划入三大分区：

| 分区 | run 数 | 是否进入 OC catalog | 物理位置 |
|---|---:|---|---|
| **A. Catalog 主线（旧 OC 主线）** | 98 | 是（`run_inventory.csv`） | `checkpoints/sweep_oc_all*`、`sweep_oc_*_noise_*_extra_*`、`sweep_oc_phase1_probe_*`、`sweep_oc_phase1_smoke_clean_fix_*`、`sweep_oc_smoke`、`sweep_oc_main_noise_seed42_smoke` |
| **B. Phase-1A 现实性验证（v4-lite / t2-wpfrag）** | 148 | 否 | `checkpoints/sweep_oc_phase1a_*`、`smoke_v4lite/` |
| **E. Unused 旧噪声设计** | 54 | 否 | `checkpoints/unused/` |
| 合计 | **300** | — | — |

另有不计入 300 的产物：
- **C. Provenance 审计重训**：1 个 forensic run（`analysis/provenance_audit/phase3_retrain/audit_phase3_seed46_clean_20260512_095957/`，不在 `checkpoints/` 下）。
- **manifest-only 聚合目录**（无 `best_model.pt`，仅 CSV/TSV/JSON 清单，指向云端路径）：`sweep_oc_phase1a_decision_{clean,iid,v4lite}_*_cleanrun_v1`、`sweep_oc_phase1a_decision_proxy_*`、`sweep_oc_phase1_probe_iideval_20260424_024232`、`sweep_oc_phase1_smoke_matched_20260423_173332`（phase1 聚合 CSV/JSON + `experiment_report.md` + `proxy_export_info.txt`）。
- **元数据/日志目录**（无 run）：`checkpoints/phase1a_metadata_*`（8 个，审计元数据）、`checkpoints/phase1a_logs/`（Colab 端 `.log`）、`checkpoints/` 顶层 `p1_2_clean_matched_eval_*.{log,pid}`（5 个 P1-2 后台作业痕迹）。

> 关键事实：规范化 catalog（`analysis/oc_data_catalog/`）**只覆盖 A 分区的 98 个 run**。B 分区（最近的 Phase-1A 现实性验证）和 E 分区（旧噪声）都不在 catalog 中。引用 catalog 数字时，覆盖面仅限 A 分区。

rollout 评估产物：磁盘上共 **241** 个 `rollout_benchmark/` 目录；catalog `rollout_run_registry.csv` 登记 **352** 行 rollout run（含同一 checkpoint 在多个 eval profile 下的多次评估）。

---

## 2. 模型清单（10 个变体）

来源：`auv_model_registry.py`（`MODEL_SPECS`）、`auv_baselines.py`。

| model_type | 家族/组 | 一句话描述 |
|---|---|---|
| `phnode_full` | main / core | 主 pH 模型：精确 SE(3) + 可学习 M⁻¹ + 标量势 V + 拆分 D/J/B |
| `phnode_qforce` | baseline / core | 结构化 pH，使用通用的构型相关广义力 |
| `phnode_merged_force` | baseline / core | pH 核心，单一合并的非保守力分支 |
| `se3_momentum_blackbox` | baseline / core | 精确 SE(3) + 常数质量矩阵 + 黑箱动量动力学 |
| `se3_accel_blackbox` | baseline / core | 精确 SE(3) 运动学 + 黑箱加速度动力学 |
| `blackbox_fullstate` | baseline / core | 完全无结构的状态导数模型 |
| `ablate_no_mass_prior` | ablation | 去掉物理质量先验初始化 |
| `ablate_diag_damping` | ablation | 仅对角阻尼（`coupled_damping=False`） |
| `ablate_no_lift` | ablation | 去掉可学习反对称 lift 项（`learn_lift=False`） |
| `ablate_bu_only` | ablation | 执行机构仅以 actuator 状态为条件 |

种子覆盖（A 分区 catalog，clean / noisy 训练）：

| model_type | clean 种子 | noisy 种子 |
|---|---|---|
| `phnode_full` | 42-47（6） | 42-47（6，含 P1-1 extra 42/46/47） |
| `ablate_no_mass_prior` | 42-47（6） | 42-47（6，含 P1-1 extra） |
| `ablate_no_lift` | 42-47（6） | 42-47（6，含 P1-1 extra） |
| `ablate_diag_damping` | 42-47（6） | 43,44,45（3） |
| `ablate_bu_only` | 42-47（6） | 43,44,45（3） |
| `phnode_qforce` | 42,43,44（3） | 43,44,45（3） |
| `phnode_merged_force` | 42,43,44（3） | 43,44,45（3） |
| `se3_momentum_blackbox` | 42,43,44（3） | 43,44,45（3） |
| `se3_accel_blackbox` | 42,43,44（3） | 43,44,45（3） |
| `blackbox_fullstate` | 42,43,44（3） | 43,44,45（3） |

---

## 3. 数据集清单

所有数据集 schema `auv_dataset_v3`，state_dim 27 `[Δpos(3), R(9), nu_b(6), u_actual(3), u_cmd(3), v_c^n(3)]`，actuator_dim 3，均为 **oc**（带海流），`current_speed_range [0.0, 0.5]`，`dt_state=0.05`。

| 数据集 id | 轨迹数（train/test） | blocks | gen-seed | oc/noc | current_max | .pkl 大小 | 被谁使用 |
|---|---|---:|---:|---|---:|---:|---|
| `auv_oc_traj1000_blk150_s23_d0be9434` | 1000 (559/140) | 150 | 23 | oc | 0.5 | 217 MB | **唯一规范训练集**：A 分区全部 + B 分区全部（含所有 `t2_wpfrag_*`、cleanrun v1） |
| `auv_oc_traj2000_blk150_s233_555e5dd1` | 2000 (1109/278) | 150 | 233 | oc | 0.5 | 429 MB | **未发现任何引用**（无 checkpoint/catalog/脚本引用），当前证据未使用 |
| `auv_oc_traj667_blk150_s42_9b2d7617` | 667 (372/94) | 150 | 42 | oc | 0.5 | 145 MB | **未发现 checkpoint 引用**，仅 `docs/unused/` 一处旧设计文档提及 |
| `smoke_v4lite/auv_oc_traj12_blk8_s123_86c6e0b8` | 12 (9/3) | 8 | 123 | oc | 0.5 | 1 MB | 仅 code/flow 验证（`smoke_v4lite/` 4 个 ep=3 run） |

> 重要澄清：**不存在单独的 "t2_wpfrag" 数据集**。"t2_wpfrag"（waypoint-fragment / T2）是 **suite 命名标签**，所有 `t2_wpfrag_*` run 训练在同一个 `d0be9434` 上。`make_t2_notebooks.py` 硬编码 `DATASET="data/auv_oc_traj1000_blk150_s23_d0be9434.pkl"`，每个 phase1a `config.json` 的 `dataset_path` 也都是该文件（`/content/drive/...` 前缀为 Colab 挂载路径，文件与本地 `data/` 相同）。"T2" 指决策矩阵 = 每模型 × 3 训练协议 {clean, iid_noisy_ic, v4_lite} × 5 种子 {42–46}。

---

## A. Catalog 主线实验（98 run，在册）

时间窗口约 2026-04-04 ~ 04-24，云端镜像 `auvhamnode/g3_5_5`。这是 `analysis/oc_data_catalog/` 规范化覆盖的全部范围。

### A.1 实验桶总表

桶来自 `run_annotations.csv` 的 `experiment_bucket`；rollout 覆盖来自 `rollout_run_registry.csv`。

| 桶 | suite | 模型 | 种子 | run 数 | rollout eval | 备注 |
|---|---|---|---|---:|---|---|
| clean 主（core） | `sweep_oc_core_default_..._s42-43-44` | 6 个 core（phnode_full + 5 baseline） | 42,43,44 | 18 | 有 | primary；其中 seed42 phnode_full 被标 stale |
| clean phnode_focus | `sweep_oc_phnode_focus_extra3_..._s45-46-47` | phnode_full + 4 ablation | 45,46,47 | 有 | 15 | primary；其中 seed46 phnode_full 被标 stale |
| clean ablation | `sweep_oc_ablation_default_..._s42-43-44` | 4 ablation | 42,43,44 | 12 | 有 | primary |
| noisy 主 | `sweep_oc_main_noise_nominal_train_remus100_dr_..._s43-44-45` | phnode_full | 43,44,45 | 3 | 有 | primary |
| noisy baseline | `sweep_oc_baseline_noise_..._s43-44-45` | 5 baseline | 43,44,45 | 15 | 有 | primary |
| noisy ablation | `sweep_oc_ablation_noise_..._s43-44-45` | 4 ablation | 43,44,45 | 12 | 有 | primary |
| Followup P1-1（noisy 主） | `sweep_oc_main_noise_..._extra_42-46-47` | phnode_full | 42,46,47 | 3 | 有 | is_followup |
| Followup P1-1（noisy ablation） | `sweep_oc_key_ablation_noise_..._extra_42-46-47` | ablate_no_lift, ablate_no_mass_prior | 42,46,47 | 6 | 有 | is_followup |
| Probe（phase1） | `sweep_oc_phase1_probe_{clean,iid,v4lite}_*`、`sweep_oc_phase1_smoke_clean_fix_*` | phnode_full, ablate_no_mass_prior | 43,44 | 10 | 有 | is_smoke=0、is_primary=0（中间探针） |
| Smoke | `sweep_oc_main_noise_seed42_smoke`、`sweep_oc_v4lite_protocol_smoke_*`（在 `sweep_oc_smoke/`） | phnode_full | 42 | 4 | 有（registry 标 smoke，永不 canonical） | is_smoke=1 |

合计 98。`canonical_run_inventory.csv` = 94（剔除 4 个 smoke）。

> 注：**P1-2** 不是新训练桶，而是把已有 clean checkpoint 在 noisy rollout profile 下重评的 `matched_followup` rollout（registry 84 行），不产生新 run。

### A.2 rollout 评估 profile 覆盖

profile：`clean` / `nominal_eval` / `degraded_eval` / `heading_biased_eval`。canonical 覆盖：clean 94、nominal_eval 70、degraded_eval 70、heading_biased_eval 70。noisy 桶原生带四 profile（`primary`）；clean 桶原生 `clean`，三个 noisy profile 通过 P1-2 `matched_followup` 补齐。

### A.3 evidence_status 标记

唯一在用标记 **`stale_environment_drift`**（`evidence_status_overrides.csv`，2 行，2026-05-13 设定）：

| run_uid | 理由（如实摘录） |
|---|---|
| `...core_default..._s42-43-44/main_phnode_full_seed42` | clean seed42 catalog 期 best_loss=2.10e-02，远差于 cleanrun v1 的 4.02e-03，归因于云端镜像环境漂移 |
| `...phnode_focus_extra3..._s45-46-47/main_phnode_full_seed46` | clean seed46 catalog 期训练发散（epoch 24-26 → inf → "no successful training batches"）；当前 main 复现 cleanrun v1 C46 = 4.05e-03 |

`oc_data_catalog_dictionary.md` 另提到一个**候选**第三标记（`ablate_no_lift seed43 clean`），但未独立核验、未写入 overrides。selection policy 规定：`is_canonical=1` 与 `evidence_status` 是两层——stale run 仍留在 canonical 视图供溯源，但不应作为当前 phnode_full 脆弱性证据引用。

### A.4 各报告记录的结论（标注来源，不做取舍）

**`docs/oc_model_evaluation_overview.md` / `oc_results_section_zh.md`（clean 主，60s 末位置误差中位数，all-seed）：**
- 报告称整体最强为 `phnode_qforce`（0.5708 m，98.1% 完成）；最强 PHNODE 家族为 `ablate_no_lift`（1.0022 m，99.3%）。
- 报告称 `phnode_full` all-seed=9.0863 m，被审计为 `prunable_bad_outliers`，坏种子 42/46；剔除后稳定簇（43,44,45,47）=0.6098 m。读法为"强稳定簇 + 真实坏种子失效模式"，非表达力弱。
- 结构性结论：`ablate_bu_only` 结构退化（27.80 m，执行条件必要）；`ablate_diag_damping`（4.17 m，耦合阻尼有价值）；`ablate_no_mass_prior`（1.44 m，稳定，质量先验尚未显示不可替代）；`blackbox_fullstate` 家族不稳定（86.78 m，18.9% 完成）。

**`docs/oc_experiments_comprehensive_report.md`（clean + noisy 综合）：**
- noisy `nominal_eval`（43,44,45）：报告称 `phnode_full` 排名第 1（1.2230 m），其后 `ablate_no_mass_prior`（1.3066）、`ablate_no_lift`（1.8562）；`phnode_qforce` 跌至第二梯队（2.0952，被 seed45 拖累）。
- 应力 profile：`degraded_eval` 冠军 `ablate_no_mass_prior`（1.8846，phnode_full 1.9922）；`heading_biased_eval` 冠军 `ablate_no_mass_prior`（2.9027，phnode_full 3.0362）。
- noisy 训练的 clean-replay 代价对多数结构模型很小，唯 `phnode_qforce` 达 5.00×。
- 报告显式警告（§4.2/§8）：noisy sweep 未覆盖 phnode_full 的关键坏种子 42/46，故不能宣称"noisy 训练修复了种子脆弱性"——这正是后续 P1 的动机。报告头注称结论已被 P1-1/P1-2 修订。

**`docs/oc_followup_results_p1_p2.md`（P1-1 / P1-2）：**
- **P1-1**（noisy 6 种子 42-47）：报告称 all-seed noisy 头名应从 `phnode_full` 改为 `ablate_no_mass_prior`（1.2494 m）；`ablate_no_lift` 1.4339（新坏种子 44）；`phnode_full` 1.8025（seed42 仍异常）。phnode_full 逐种子 noisy nominal：42=5.16, 43=1.09, 44=0.96, 45=1.62, 46=0.93, 47=1.05；剔除 42 后 1.1312 m。结论：noisy 训练**仅部分**修复脆弱性——seed46 修好，seed42 仍是核心难种子。
- **P1-2**（clean vs noisy matched）：报告称 phnode_full 的巨大 noisy 增益（9.27→1.80）**几乎全部**来自修复 seed46（47.85→0.93），其余 5/6 种子 noisy 下反而更差。`ablate_no_mass_prior` 是唯一"稳定获益"模型（四 profile 全好，5/6 种子改善）。`ablate_no_lift` 四 profile 轻微回退。`phnode_qforce`（43,44）noisy 下四 profile 全差。总体：noisy 训练效果**强烈依赖模型结构**，非一致鲁棒性增益。

### A.5 后续计划完成度（`docs/oc_followup_experiment_plan.md`，如实标注）

计划定义 P0（smoke）、P1-1、P1-2、P2-1（mass-prior/lift 2×2 机制，含尚不存在的 `ablate_no_mass_prior_no_lift` 模型）、P2-2（noisy schedule 扫描）、P3（`remus100_ins`、`noc` 线）。
- **已做**：P0、P1-1、P1-2（catalog 内有对应 suite/rollout）。
- **未做**：P2-1、P2-2、P3（catalog 中无 `sweep_oc_main_noise_schedule_*`、`remus100_ins`、`noc`、`ablate_no_mass_prior_no_lift`）。

---

## B. Phase-1A 现实性验证（148 run，不在册）

时间窗口约 2026-04-24 ~ 04-26，云端镜像 `auvhamnode/g3_5_7`。这是最近的主线工作，**不在 OC catalog 中**。命名空间下并存两条线：`*_t2_wpfrag_*`（每个 suite 一个模型、内含 5 种子的物理 checkpoint）与 `*_phase1a_oc_v4lite_cleanrun_v1`（smoke1/smoke3/extra43-45 持有物理 checkpoint；`decision_*`/`proxy` 仅 manifest 聚合）。

### B.1 t2_wpfrag 决策套件

| suite 前缀 `sweep_oc_phase1a_decision_*_t2_wpfrag_*` | 模型 | 种子 | 训练协议 | run 数 | rollout |
|---|---|---|---|---:|---|
| `decision_clean_t2_wpfrag_<7 模型>` | phnode_full, phnode_qforce, ablate_no_lift, ablate_no_mass_prior, blackbox_fullstate, se3_accel_blackbox, se3_momentum_blackbox | 42-46 | clean | 35 | 每 run 有 |
| `decision_iid_t2_wpfrag_<4 模型>` | phnode_full, phnode_qforce, ablate_no_lift, ablate_no_mass_prior | 42-46 | iid_noisy_ic | 20 | 有 |
| `decision_v4lite_t2_wpfrag_<4 模型>` | 同上 4 模型 | 42-46 | v4_lite | 20 | 有 |

> 不对称：clean 决策有 7 个模型（含 3 个黑箱/se3 基线），iid 与 v4lite 仅 4 个模型——3 个黑箱/se3 基线只训了 clean。

### B.2 cleanrun v1（3 模型 × 3 协议 × 5 种子 = 45 run 决策包，物理 checkpoint 分散在以下）

| suite | 模型 | 种子 | 协议 | run 数 | rollout |
|---|---|---|---|---:|---|
| `smoke3_{clean,iid,v4lite}_..._cleanrun_v1` | phnode_full, ablate_no_lift, ablate_no_mass_prior | 42,44,46 | clean/iid/v4lite | 9×3 = 27 | 有 |
| `decision_extra43-45_{clean,iid,v4lite}_..._cleanrun_v1` | 同上 3 模型 | 43,45 | clean/iid/v4lite | 6×3 = 18 | 有 |
| `smoke1_{clean,iid,v4lite}_..._cleanrun_v1` | phnode_full | 42,44,46 | clean/iid/v4lite | 3×3 = 9 | 有 |
| `decision_{clean,iid,v4lite}_..._cleanrun_v1` + `decision_proxy_*` | （清单） | 42-46 | — | **0**（manifest-only） | 无 |

> smoke3 的种子 42/44/46 与 extra43-45 的 43/45 合并 = 完整的 5 种子决策包（这是"先分批训练再合并"的有意设计，非污染）。

### B.3 t2_wpfrag / cleanrun smoke1 与 code smoke

| suite | 模型 | 种子 | 协议 | run 数 | rollout |
|---|---|---|---|---:|---|
| `smoke1_clean_t2_wpfrag_<7 模型>` | 7 个模型各 1 | 单种子 | clean | 7 | 有 |
| `smoke1_iid_t2_wpfrag_<4 模型>` | 4 个模型各 1 | 单种子 | iid | 4 | 有 |
| `smoke1_v4lite_t2_wpfrag_<4 模型>` | 4 个模型各 1 | 单种子 | v4lite | 4 | 有 |
| `smoke_v4lite/`（本地） | phnode_full | 42 | 2×clean + iid + v4lite，**ep=3** | 4 | 无（纯 code/protocol smoke） |

非 run 产物：`checkpoints/phase1a_metadata_*`（8 目录，每个含 `phase1a_run_config.json`+`phase1a_environment.json` 审计元数据）；`checkpoints/phase1a_logs/phase1a_oc_v4lite_cleanrun_v1/`（17 个 Colab 端 `.log`，Stage B 的溯源链）。

### B.4 v4-lite 协议定义（`docs/v4_lite_protocol_spec.md`）

v4-lite 是面向**纯动力学模型**的轨迹一致噪声-IC 协议。相对 iid 的唯一变化：噪声初始状态来源。iid = 每 block 独立采样噪声 IC；v4-lite = 先生成轨迹级噪声观测，再为每个 block 读取其噪声 `y0`。同一轨迹所有 block 共享同一次噪声实现（epoch 间可重采，epoch 内不可）。骨干、clean target、block rollout 不变；无 history encoder / observer / multi-block。作为 `train_utils.py` 的新噪声层实现，非并行 trainer。

### B.5 `docs/phase1a_oc_v4lite_cleanrun_v1_report.md` 记录的结论（标注来源）

- 决策代理套件 = **45 run**（3 协议 × 3 模型 × 5 种子 42-46）。主指标：60s 末位置误差中位数 + completion@60s（Pos Median = 各种子 rollout 中位数的均值）。
- 头部 60s/nominal_eval（best 优先，Pos Median /m）：phnode_full clean→v4_lite eval **0.844**（最佳）；phnode_full clean→iid 0.960；ablate_no_lift iid→iid 1.004；phnode_full iid→v4 1.032；ablate_no_mass_prior 约 1.40–1.52；**ablate_no_lift clean→{v4,iid} = 9.68 / 9.74（被污染）**。
- **报告标为 CRITICAL 的异常**：`ablate_no_lift seed43 clean` 停在 `best_epoch=19`、`best_loss=0.2169`、60s rollout ≈44 m，污染 clean ablation 排名，须重跑/排除后才可用。
- v4_lite 作为**训练**协议：仅 `ablate_no_mass_prior` 聚合获益（约 4–7%），phnode_full 持平/略差，ablate_no_lift 明显更差 → 未达采纳门槛。
- v4_lite 作为**评估**协议：利好 phnode_full（clean/iid 训练），轻微不利两个 ablation → 结构偏置诊断，非中性替换。
- **catalog 不匹配**：cleanrun v1 不复现 canonical clean 期结果（如 catalog phnode_full clean 10.96 m vs cleanrun 0.96 m，归因于重训/provenance 差异）→ 报告称暂不应并入 canonical catalog。
- **最终判定**：Phase-1A 报告契约（5 种子 × 3 模型 × 3 协议，逐种子/场景/horizon + clean replay）已完成，但**不是 v4-lite 的强正向结果**；v4-lite 保持为诊断工具而非默认训练协议；不进入 Phase-1B。

### B.6 完成度（如实标注）

- **已完成（物理 checkpoint + rollout）**：cleanrun v1 决策矩阵（smoke3 27 + extra43-45 18 = 45 run 决策包，3 模型 × 3 协议 × 5 种子）；t2_wpfrag **clean** 决策（7 模型全宽 × 5 种子 = 35 run）。
- **部分（窄于 clean）**：t2_wpfrag **iid / v4lite** 决策仅 4 模型（缺 3 个黑箱/se3 基线），各 20 run。
- **仅 manifest / proxy（无 checkpoint）**：`decision_{clean,iid,v4lite}_*_cleanrun_v1`、`decision_proxy_*`、`phase1_probe_iideval_*`（仅 seed43 的 phnode_full + ablate_no_mass_prior 汇总）。
- **仅 smoke（验证，非模型证据）**：所有 `smoke1_*`（单种子，但确为 300-epoch 训练 + rollout）；`smoke_v4lite/`（ep=3，小数据集，无 rollout）。
- **notebook**：`phase1a_oc_v4lite_formal_workflow.ipynb`（含 `_completed`）；8 个 `t2_wpfrag_<model>.ipynb`（含 `_completed`）；`t2supp_nolift_seedscan.ipynb`（+ 未提交 `_tmp`）。

---

## C. Provenance 审计（phnode_full clean seed46）

来源：`analysis/provenance_audit/`、`docs/provenance_audit_phnode_full_clean.md`。分支 `provenance-audit-phnode_full`，四阶段均已完成。

- **调查问题**：为何 catalog `phnode_full clean seed42/46` 的 60s rollout 误差（5 种子均值 ~11 m）远高于 cleanrun v1（~0.96 m，约 11×），phnode_full clean 是否仍存在 seed42/46 灾难性失效。
- **Phase 1（静态 diff）**：数据集、所有显式超参、噪声 profile（clean，no-op）、wrapper 调用链在 catalog A42/A46 与 cleanrun C42–C46 之间**逐位一致**。唯一关键差异是**训练结果本身**：catalog A46 `best_epoch=21`、`best_loss=0.27`，并有 275 行"no successful training batches"（epoch 26 起每 batch ODE 求解失败 = 经典发散）；A42 收敛但 loss 差 5×（0.02 vs 0.004）。`run_inventory.csv` 缺 `code_revision` 字段，无法做 commit 级 diff。
- **Phase 2（聚合）**：11 m vs 0.96 m 混淆了三种口径（clean vs nominal_eval；mean vs median；4 rollout/种子 vs 1）。同口径（clean+clean，60s，5 种子 pos_err_median 均值）= catalog **10.64 m** vs cleanrun **0.6767 m**，真实 **15.7×** gap，几乎全部来自 seed46（103×）与 seed42（7.3×）；43/44/45 持平或更好。cleanrun seed46 = 0.4558 m（曾 46.89 m）→ cleanrun 侧脆弱性已消失。
- **Phase 3（重训 artifact）**：在当前 main（`7643dc9`）重跑 `phnode_full × clean × seed46`（Colab L4 / PyTorch 2.10.0+cu128 / CUDA 12.8 / cuDNN 91002）。产物 `analysis/provenance_audit/phase3_retrain/audit_phase3_seed46_clean_20260512_095957/`。结果 = **"Signal B（脆弱性完全自愈）"**：`best_epoch=250`、`best_loss=4.0471e-03`、60s clean `pos_err_median=0.4558 m`，零"no successful training batches"。该 run 在 60s clean rollout 的 mean/median/p90/p95/max 上与 cleanrun C46 **IEEE-754 逐位一致**，训练在 epoch 1-2 与 catalog A46 逐位一致（epoch 3 起分叉）。
- **Phase 4（根因）**：跨 2026-04-04~04-26 的 git-log diff（28 commit，11 个动训练代码）**未发现任何 commit 修复脆弱性**——`AUVHamNODE.py`/`auv_baselines.py` 零 diff，`train_utils.py`/`train_auv_hamnode.py` 改动都是 noise-v1→v2 / v4_lite 重构，对 clean 路径为 no-op。
- **结论/根因**：catalog 期 seed46 发散是**真实训练动力学事件但非模型/代码 bug**——是（未记录的 `g3_5_5` 云镜像 PyTorch/CUDA/cuDNN 版本）+（cuDNN 在 seed46 上的非确定算法选择）+（epoch-24 一个极端 batch 梯度裁剪未吸收 → 4.68e+25 → inf）的随机耦合，当前 main 不复现。`snapshot_log.csv` 记 catalog A46 为 `status=anomaly`。
- **当前状态**：审计结束于 Phase 4，catalog seed42/46 标 `stale_environment_drift`。建议 schema 修复：为 `run_inventory.csv` 增 `code_revision`+`environment`（新 run 已写 `_audit_meta/`）。

---

## D. Smoke / Probe / Flow-validation

来源：`docs/experiment_stages_overview.md`（Stage D）。性质 `flow_validation_only`，永不作为模型证据。

- A 分区内：`sweep_oc_smoke/`（含 `sweep_oc_v4lite_protocol_smoke_*`）、`sweep_oc_main_noise_seed42_smoke`、`sweep_oc_phase1_probe_{clean,iid,v4lite}_*`、`sweep_oc_phase1_smoke_clean_fix_*`。
- B 分区内：`phase1a_smoke1_*`（单模型/单种子）、`smoke_v4lite/`（ep=3 纯 code smoke）。
- **`phase1a_smoke3_*` 不属于本节。** 它名字带 smoke，但持有 cleanrun v1 决策包 45 个 run 中的 27 个物理 checkpoint（见 §B.2 / §B.6），属 evidence-bearing，删除会直接损毁决策包。逐 run 判定以 `docs/checkpoints_retention_manifest.csv` 的 `retention_class` 为准。
- 散落日志/pid（`checkpoints/` 顶层）：`p1_2_clean_matched_eval_*` 的 5 个 `.log/.pid` 是 2026-04-13 后台 P1-2 "matched clean→noisy 评估" 批作业的痕迹——前两次为空/停滞误启动，`_live_` 那次（8612 行）为完成 run。非证据。

---

## E. Unused 旧噪声设计（54 run，已排除）

来源：`checkpoints/unused/`、`docs/repo_structure_audit.md`、`experiment_stages_overview.md`（Stage E）。

| suite（`checkpoints/unused/` 下） | run 数 | 噪声设计 |
|---|---:|---|
| `sweep_oc_core_noise_l1_..._s43-44-45_20260406_070456` | 18 | `--noise_level 1`（l1） |
| `sweep_oc_core_noise_l2_..._s43-44-45_20260406_043408` | 18 | `--noise_level 2`（l2） |
| `sweep_oc_core_noise_l2_..._s43-44-45_20260406_090347` | 18 | `--noise_level 2`（l2，重跑） |

- 每 suite 6 模型（main_phnode_full + 5 baseline）× 种子 43,44,45 = 18。
- **排除理由**（`repo_structure_audit.md` / Stage E）："使用旧版且有错误的噪声设计，不应作为当前证据"——已被弃用的 noise-v1 接口（`--noise_level l1/l2`），是 CLAUDE.md 标记 deprecated 的接口。
- `summary111.txt`（642 KB）实为另两个 `noise_sweep_nominal_train_oc_*` suite 的 rollout-eval stdout 日志，**误置于此**，与 l1/l2 套件无关。
- 另：`original/bf3n/` 为 legacy 参考，非活动代码（同属 Stage E 排除范围）。

---

## F. 实验阶段五分类（`docs/experiment_stages_overview.md`，如实摘录）

划分轴 = 云端训练镜像 + 时间窗口。

| 阶段 | 名称 | 时间窗 | 云镜像 | 本地形态 | 主要目录 | 可用性 |
|---|---|---|---|---|---|---|
| **A** | Catalog 期 | 04-04~04-21 | `auvhamnode/g3_5_5` | 完整 artifact | `sweep_oc_all*`、`sweep_oc_*_noise_*_extra_*` | catalog 全部 88 行；部分环境漂移污染；仅 2 行明确 stale，其余 needs_recheck |
| **B** | Cleanrun v1（Phase-1A 决策） | 04-24~04-26 | `auvhamnode/g3_5_7` | **仅 manifest**（本地无 `best_model.pt`，物理 checkpoint 在 smoke3/extra） | `sweep_oc_phase1a_*_cleanrun_v1` | 当前 main 可逐位复现；3 模型 × 3 协议 × 5 种子 = 45；不覆盖 baseline/qforce/blackbox |
| **C** | Provenance 审计重训 | 05-12 | `auvhamnode/g3_5_7`（forensic） | tarball + 报告 | `analysis/provenance_audit/phase3_retrain/audit_phase3_seed46_clean_*` | 仅 1 forensic run（phnode_full×clean×seed46） |
| **D** | Smoke/probe | 04-21~04-24 | g5/g7/local | 完整 artifact | `sweep_oc_smoke/`、`smoke_v4lite/`、`sweep_oc_phase1_probe_*`、`sweep_oc_phase1a_smoke{1,3}_*` | flow_validation_only，永不证据 |
| **E** | Unused/legacy | 04-06 前 | `auvhamnode/g3_5_5` | 完整 artifact | `checkpoints/unused/`、`original/bf3n/` | 弃用 noise-v1，非任何结论证据 |

§7 主张可信度表（如实摘录）：phnode_full clean 自性能与 PHNODE 家族内比较（Stage B）✅ 安全；跨家族 vs baseline/qforce/blackbox 排名（Stage A）⚠️ 受污染；"noisy 训练修复 seed46"框架 ❌ 失效（seed46 异常是环境 artifact）。

---

## G. 文档清单（`docs/`，按类别）

| 文件 | 类别 | 描述 |
|---|---|---|
| `experiment_stages_overview.md` | 结果/审计 | 实验阶段 A–E 映射与证据状态 |
| `oc_experiments_comprehensive_report.md` | 结果报告 | OC 综合实验报告 |
| `oc_model_evaluation_overview.md` | 结果/协议 | pH NODE 模型设计、评估协议与结果（英文） |
| `oc_results_section_zh.md` | 结果报告 | OC 结果章节（草稿散文） |
| `oc_followup_results_p1_p2.md` | 结果报告 | OC 后续结果（P1-1 + P1-2） |
| `phase1a_oc_v4lite_cleanrun_v1_report.md` | 结果报告 | Phase-1A v4-lite cleanrun v1 结果分析（被引基线） |
| `provenance_audit_phnode_full_clean.md` | 审计 | phnode_full clean provenance 审计 |
| `repo_structure_audit.md` | 审计 | 仓库结构审计与删除候选 |
| `oc_followup_experiment_plan.md` | 计划 | OC 后续实验计划 |
| `phnode_realistic_validation_plan.md` / `phnode_realistic_validation_execution_plan.md` / `phase1_realistic_validation_plan.md` | 计划 | 现实性验证研究/执行计划 |
| `v4_b1_implementation_checklist.md` | 计划/runbook | v4-B1 实现清单 |
| `noise_model_design.md` / `noise_design_v3_*` / `noise_design_v4_dr_ekf_output.md` / `noise_design_v4_lite_*` / `noise_design_revision_filtered_state_robustness.md` / `v4_lite_protocol_spec.md` | 设计/协议 | 噪声与协议设计系列 |
| `noise_experiment_runbook.md` / `noise_cli_parameter_reference.md` / `noise_cli_command_templates.md` | runbook | 噪声实验操作说明 |
| `oc_data_catalog_plan.md` / `oc_data_catalog_dictionary.md` / `oc_result_selection_policy.md` / `oc_catalog_template_usage.md` | catalog 系统 | catalog 计划/字典/选择规则/模板 |
| `docs/unused/`（7 个 .md） | 归档/stale | 旧噪声设计（cc/codex/cx）、旧 Phase-1 矩阵/清单 |

---

## H. 论文产物（`paper/`）

- **当前章节草稿**：`paper/drafts/auvhamnode_thesis_chapter_zh.tex`（117 KB，2026-05-30），`ctexrep` 中文学位论文章节，标题《面向长期状态预测的 AUV 结构化神经动力学建模方法》。**10 个 `\section`、33 个 `\subsection`**，已编译为 PDF（661 KB），含 `.bbl`/`.aux` 等构建产物与 `auvhamnode_refs.bib`。
  - 章节：1 研究问题与方法概述；2 相关建模基础；3 受控状态表示与海流速度约定；4 从 Fossen 能量结构到结构保持学习模型；5 结构化连续时间动力学模型；6 能量性质与功率关系；7 训练目标、基线体系与验证协议；8 实验结果与结构证据分析；9 讨论；10 本章小结。
- **弃用版本**（`paper/drafts/deprecated/`）：`*_framework_20260520`、`*_intermediate_20260519`（各含 .tex/.pdf，仅作素材库）。
- **评审笔记**（3 份）：`auvhamnode_thesis_chapter_review_notes_zh.md`、`auvhamnode_thesis_chapter_expert_review_20260524.md`（§1–§7 专家评审，未跟踪）、`auvhamnode_thesis_chapter_revision_review_20260529.md`（终稿评审跟踪）。
- **图**（`paper/drafts/figures/`，各含 Python 生成器 + PNG/PDF/SVG）：`velocity_state_contract`（接入 §3）、`mechanical_core_power_structure`（2026-05-30 重建）、`section8_two_level_evidence`（2026-05-30 重建）。
- **写作指南/伴随**（`paper/` 顶层）：`auvhamnode_paper_writing_guide_expert_revised_zh.md`、`auvhamnode_formal_writing_companion_zh.md`、`auvhamnode_expert_review_decision_notes_zh.md`、`auvhamnode_thesis_chapter_prewrite_pack_zh.md`、`fossen_to_ph_node_auv_report.md`（§4 背景，未跟踪）、`README.md`（写作索引 + 进度板）。
- **阶段**（据文件名与 README）：完整 10 节章节草稿已存在并编译为 PDF，处于**终稿级一致性评审/修订**。README 称 10 节均 `done`，§8 采用"当前证据" + B1 训练异常 + rollout 发散报告约定 + 两层（几何/能量）证据框架；唯一 `blocked` 项为真海试泛化（无主证据，作为局限/未来工作）。

---

## I. exports 发布包

`exports/phnode_full_oc_clean/` — 单个 clean 训练 `phnode_full`（**seed 45**，cleanrun v1 cohort 中 clean-clean 60s 最佳，0.4316 m）的自包含可移植发布：
- `checkpoints/seed45/`（`best_model.pt` ~772 KB + `config.json` + `provenance.json`）；溯源 `decision_extra43-45_clean_..._cleanrun_v1/main_phnode_full_seed45`（best epoch 247 / loss 4.02e-3）。
- `phnode_full_oc/`（纯 torch 可导入包：`model.py`/`load.py`/`inference.py`/`state_layout.py`）；`reference_simulator/`（逐字 `remus100_core.py` + 一致性校验）；`tests/test_smoke.py`（pytest）；`examples/`；`requirements.txt`。
- README 显式声明局限：仅 clean（无噪声-IC 鲁棒性）、已知海流假设、≤60s horizon、"Provenance audit Phase 3 unfinished"（README 早于审计收尾）、历史 seed42/46 脆弱性。

---

## J. 当前证据空白（如实标注，非建议）

以下为各计划文档列出但**当前仓库未发现对应 run/产物**者：

- A 分区后续计划：**P2-1**（mass-prior/lift 2×2 机制，含尚不存在的 `ablate_no_mass_prior_no_lift` 模型）、**P2-2**（noisy schedule 扫描）、**P3**（`remus100_ins`、`noc` 线）均未做。
- Phase-1A：**Phase-1B 未进入**（cleanrun 报告判定不进入）；t2_wpfrag 的 **iid/v4lite 缺 3 个黑箱/se3 基线**。
- 数据集 `auv_oc_traj2000_*`、`auv_oc_traj667_*` 当前证据未使用。
- Phase-1A v4-lite 的 `phase1a_v4_protocol_validation.json`、proxy 目录的逐 run config/environment 未导出（cleanrun 报告标记缺失）。

---

# K. 前代仓库 g3_5_4 分区（noc + oc）

> 本节为 **2026-05-30 跨仓库复核**新增。口径来源：`g3_5_4/checkpoints/` 磁盘文件、各 suite 的 `*_summary.csv`/`*_runs.csv`、各 run 的 `config.json` 与 `rollout_benchmark/*/horizon_metrics.csv`、`g3_5_4/checkpoints/**/*.md` findings 文档、`g3_5_4/docs/`。所有 rollout 数值经从磁盘**独立重取**核验（非沿用旧表）。模型命名映射全表与报告散文完整版见 [g3_5_4_legacy_noc_oc_inventory_zh.md](g3_5_4_legacy_noc_oc_inventory_zh.md)。

## K.0 定位与镜像/环境

- `g3_5_4` 是 `g3_5_5` 的**前一代仓库**，共 **30 个**含 `best_model.pt` 的训练 run，**全部为单一架构 `ph_se3_full`**（= 新版 `phnode_full`，同一 `AUVHamNODE` 类）。
- 这条线回答的问题是"**大 batch 默认配方该取哪个、noc 与 oc 是否需要不同配方**"——是**训练稳定性 / 精度-效率权衡 + noc-vs-oc 数据对照**研究，架构固定，只扫训练超参（batch / lr / warmup / total_steps）与数据是否带海流。**不是**多模型消融。
- **镜像/环境**：Colab，镜像路径 `auvhamnode/g3_5_4`（`dataset_path` 前缀 `/content/drive/MyDrive/Colab Notebooks/auvhamnode/g3_5_4/`），`config.json` `device=cuda`。时间窗约 **2026-03-30 ~ 04-02**。
- **溯源缺口（如实标注）**：g3_5_4 的 `config.json` **不含** `code_revision` / `git_commit` / `environment` 字段，也无 `_audit_meta/`——与 g3_5_5 A 区同样无法做 commit 级 / 环境级溯源；无 provenance 重训记录。

## K.1 模型命名映射（旧 11 → 新 10，摘要）

旧版 `ph_se3_full` ↔ 新版 `phnode_full` 为**同一 `AUVHamNODE` 类、同一结构化 pH 核心**（M⁻¹ + 标量势 V(q) + 拆分 D/J/B，默认 `learn_lift=True, coupled_damping=True, condition_on_velocity=True`），唯一差异是装饰性的已弃用构造参数。其余 9 个旧名（`ph_se3_nomassinit/diagd/noj/buonly/mergednc/qforce`、`mom_se3_unstruct`、`se3_unstruct`、`bb_free_unstruct`）一一对应新版 `ablate_*`/`phnode_*`/`se3_*_blackbox`/`blackbox_fullstate`；旧版第 11 个 `ham_se3_unstruct`（单一可学习 H(q,p) 的 SE(3) Hamiltonian 基线）在新版**已删除**（11→10）。完整映射表见 legacy doc §1。

> **关键**：`g3_5_4` 实际**只训练了 `ph_se3_full` 一个模型**（其余 10 个 baseline/ablation 仅定义、未训练）。因此前代全部 30 个 checkpoint 与新版 `phnode_full` 是 1:1 对应。

## K.2 数据集与 noc/oc 设置差异

两个数据集，schema 均为 `auv_dataset_v3`，gen-seed 均为 **42**，blocks=150，dt_state=0.05，场景比例 PRBS 0.4 / CHIRP 0.35 / OU 0.25。

| 项 | **noc**（无海流） | **oc**（带海流） |
|---|---|---|
| 文件 / id | `auv_noc_traj1000_blk150_s42_32ec4535` | `auv_oc_traj1000_blk150_s42_89c80d68` |
| `ocean_current` | **False** | **True** |
| `state_dim` | **24** `[Δpos3,R9,nu_b6,u_act3,u_cmd3]` | **27**（末 3 维加 `v_c^n`） |
| nu_model | `nu_b`（体速度=相对速度） | `nu_r = nu_b − [Rᵀv_c^n; 0]`（相对速度） |
| 海流特征 flag | 无 | `dj_current_feature=current_body`、`actuation_current_feature=current_body` |
| 训练/测试轨迹 | 558 / 140 | 559 / 140 |
| 海流均值 | 无（`current_speed_range[0,0.5]` 为默认残留、零扰动） | ‖v_c^n‖≈0.262 m/s（中位 0.251，RMS 0.304） |
| .pkl 大小 | 192 MB | 216 MB |

> **跨代不可混用**：`g3_5_4` 的 oc 数据集是 `89c80d68`（**seed 42**），与 `g3_5_5` 主线的 `d0be9434`（**seed 23**）是**不同数据集**；noc 数据集 `32ec4535` 在 `g3_5_5` 中不存在。两代结果不可直接跨代比较（见 §L.2）。

## K.3 noc 专节（10 run，单一 `ph_se3_full`）

### K.3.1 run 明细 + 统一口径 rollout

统一口径：60s 自由递推**终端位置误差中位数**（`resampled` 评估，scope=overall=场景 ALL=PRBS+CHIRP+OU 汇总）。`best_loss`=各 run `best_test_total`。所有 noc run 均含 `block_evaluation.json`+`heldout_evaluation.json`+`rollout_benchmark/`（无纯训练 run）。

| suite | run | seed | batch | lr | wu | steps | best_loss | **60s 中位/m** | 60s P95/m | comp@60s | heldout 30s 中位/m | 训练h | 标记 |
|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| run_largebatch_noc | bs2048_seed233 | 233 | 2048 | 5e-3 | 300 | 7000 | 9.99e-05 | **0.3331** | 0.9858 | 0.989 | 0.1498 | 0.342 | 正常 |
| run_largebatch_noc | bs2048_seed43 | 43 | 2048 | 5e-3 | 300 | 7000 | 1.64e-04 | **0.3338** | 1.023 | 0.989 | 0.1359 | 0.330 | 正常 |
| run_largebatch_noc | bs2048_seed44 | 44 | 2048 | 5e-3 | 300 | 7000 | 1.79e-04 | **0.3685** | 0.9836 | 0.989 | 0.1688 | 0.332 | 正常 |
| run_largebatch_noc | bs4096_seed233 | 233 | 4096 | 6e-3 | 400 | 5000 | 3.87e-04 | **0.6698** | 1.841 | 0.989 | 0.3051 | 0.259 | 正常（精度较差） |
| run_largebatch_noc | bs4096_seed43 | 43 | 4096 | 6e-3 | 400 | 5000 | 2.95e-04 | **0.4742** | 1.236 | 0.989 | 0.2040 | 0.261 | 正常（精度较差） |
| run_largebatch_noc | bs4096_seed44 | 44 | 4096 | 6e-3 | 400 | 5000 | 2.75e-04 | **0.4338** | 1.292 | 0.989 | 0.1704 | 0.266 | 正常（精度较差） |
| followup/noc_4096_recipe | diag4096_s233_lr5e-3_wu400_ts5000 | 233 | 4096 | 5e-3 | 400 | 5000 | 4.76e-04 | **0.5771** | 1.469 | 0.988 | 0.2814 | 0.293 | 正常 |
| followup/noc_4096_recipe | diag4096_s233_lr5e-3_wu400_ts7000 | 233 | 4096 | 5e-3 | 400 | 7000 | 1.42e-04 | **0.3686** | 0.9611 | 0.988 | 0.1601 | 0.399 | **followup 最佳配方** |
| followup/noc_4096_recipe | diag4096_s233_lr6e-3_wu400_ts5000 | 233 | 4096 | 6e-3 | 400 | 5000 | 3.87e-04 | **0.6885** | 1.659 | 0.988 | 0.3051 | 0.285 | 正常（≈主 bs4096_s233，见 K.6） |
| followup/noc_4096_recipe | diag4096_s233_lr6e-3_wu400_ts7000 | 233 | 4096 | 6e-3 | 400 | 7000 | **1.75e-01** | **N/A（发散）** | — | — | 17.04 | — | **发散：ep29，resampled 目录为空** |

跨种子聚合（与 legacy doc / 报告口径一致，"先轨迹内取中位、再种子间取均值"）：noc **bs2048**（233/43/44）60s 中位均值 ≈ **0.3451 m**；noc **bs4096**（233/43/44）≈ **0.5260 m**（劣化 ≈1.52×）。

> 发散 run `diag4096_…_lr6e-3_wu400_ts7000`：`best_loss=0.1748`、`ever_nonfinite_test=1.0`、仅约 738 步（step_coverage≈0.105）；其 `resampled_batch_compare_10_30_60_*/` 目录**存在但为空**（评估中止），故 60s 列为 N/A；heldout 30s 中位 = 17.04 m（严重发散）。

### K.3.2 rollout 评估覆盖（与 g3_5_5 的差异）

每个 noc run 的 `rollout_benchmark/` 含两套：`heldout_batch_compare_10_20_30_*`（horizon 10/20/30s）+ `resampled_batch_compare_10_30_60_*`（horizon 10/30/60s）；场景 PRBS / CHIRP / OU（+ 汇总 ALL）。**关键差异**：g3_5_4 **只有 clean rollout**，**无 profile 噪声评估**（`nominal/degraded/heading_biased` 这套 profile 噪声体系是 g3_5_5 才引入的）。resampled 评估集为 90 条轨迹（run_largebatch_noc）/ 84 条（followup）——同一训练在不同 suite 下因评估集大小不同，60s 中位会略有差别（见 K.6）。

### K.3.3 noc 报告结论（标注来源，不做取舍）

- **`largebatch_noc_report.md`**：报告称 "**bs=2048 是 noc 更好的默认大 batch 配方**"。bs4096 唯一优势是 wall-clock（0.784×），代价是 best val loss 劣化 2.16×、heldout 30s 位置中位 1.50×、resampled 60s 位置中位 1.52×；训练期 solver failure / invalid prediction / SO(3) 违例均为 0（**非数值不稳定，是优化质量问题**）。报告称严格说只能主张"当前 bs4096 配方劣于当前 bs2048 配方"，不过度推广。推荐默认 `bs2048, lr5e-3, min_lr1e-4, wu300, total_steps7000, epochs200`。
- **`largebatch_followup.md`（noc 部分）**：报告称 followup noc_4096 的最优（stability-first）配方为 `bs4096, lr0.005, wu400, steps7000`（best 1.424e-04）；`lr0.006, wu400, steps7000` **发散**（`reach_target=0`、`step_coverage=0.105`、`no_success_epoch_warnings=383`）。
- **`../g3_5_4/docs/experiment_command_matrix.md` §7**：noc 配方定义——bs2048：`lr5e-3/min_lr1e-4/wu300/ts7000/epochs200`；bs4096：`lr6e-3/min_lr1e-4/wu400/ts5000/epochs300`。

## K.4 oc 专节（20 run，单一 `ph_se3_full`）

### K.4.1 run 明细 + 统一口径 rollout

| suite | run | seed | batch | lr | wu | steps | best_loss | **60s 中位/m** | 60s P95/m | comp@60s | heldout 30s 中位/m | 训练h | 标记 |
|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| run_largebatch_oc | bs2048_seed233 | 233 | 2048 | 5e-3 | 300 | (停于246) | **0.4572** | **67.68** | 99.91 | 0.733 | 30.57 | 0.015 | **崩溃：ep7 Test→inf** |
| run_largebatch_oc | bs2048_seed42 | 42 | 2048 | 5e-3 | 300 | 7000 | 3.68e-03 | 0.8970 | 3.002 | 1.000 | 0.2580 | 0.367 | 正常 |
| run_largebatch_oc | bs2048_seed43 | 43 | 2048 | 5e-3 | 300 | 7000 | 3.79e-03 | 1.009 | 3.313 | 0.978 | 0.3027 | 0.365 | 正常 |
| run_largebatch_oc | bs2048_seed44 | 44 | 2048 | 5e-3 | 300 | 7000 | 3.75e-03 | 1.040 | 3.447 | 1.000 | 0.3416 | 0.359 | 正常 |
| run_largebatch_oc | bs4096_seed233 | 233 | 4096 | 6e-3 | 400 | 5000 | 3.93e-03 | 1.086 | 3.061 | 0.978 | 0.3497 | 0.281 | 正常 |
| run_largebatch_oc | bs4096_seed42 | 42 | 4096 | 6e-3 | 400 | 5000 | 3.78e-03 | 0.8152 | 3.072 | 0.989 | 0.3244 | 0.287 | 正常 |
| run_largebatch_oc | bs4096_seed43 | 43 | 4096 | 6e-3 | 400 | 5000 | 3.73e-03 | 0.9960 | 3.062 | 0.989 | 0.3445 | 0.287 | 正常 |
| run_largebatch_oc | bs4096_seed44 | 44 | 4096 | 6e-3 | 400 | 5000 | 3.91e-03 | 0.9538 | 3.418 | 0.978 | 0.3906 | 0.284 | 正常 |
| oc_aligned | aligned_bs4096_s233 | 233 | 4096 | 4.5e-3 | 300 | 7000 | 3.71e-03 | 0.7843 | 3.351 | 0.976 | 0.3129 | 0.464 | 正常 |
| oc_aligned | aligned_bs4096_s42 | 42 | 4096 | 4.5e-3 | 300 | 7000 | 3.82e-03 | 1.111 | 3.232 | 1.000 | 0.3448 | 0.460 | 正常 |
| oc_aligned | aligned_bs4096_s43 | 43 | 4096 | 4.5e-3 | 300 | 7000 | 3.68e-03 | 1.159 | 3.515 | 0.976 | 0.3982 | 0.440 | 正常 |
| oc_aligned | aligned_bs4096_s44 | 44 | 4096 | 4.5e-3 | 300 | 7000 | 3.69e-03 | 0.9491 | 3.684 | 0.976 | 0.3236 | 0.459 | 正常 |
| confirm/oc_2048_confirm | confirm2048_s233 | 233 | 2048 | 4.5e-3 | 300 | 7000 | 3.72e-03 | 0.9400 | 3.325 | 0.988 | 0.3021 | 0.353 | 正常（≡followup lr4.5/wu300，见 K.6） |
| confirm/oc_2048_confirm | confirm2048_s42 | 42 | 2048 | 4.5e-3 | 300 | 7000 | 3.81e-03 | 0.9821 | 3.030 | 1.000 | 0.3602 | 0.349 | 正常 |
| confirm/oc_2048_confirm | confirm2048_s43 | 43 | 2048 | 4.5e-3 | 300 | 7000 | 3.77e-03 | 0.9361 | 3.503 | 1.000 | 0.2681 | 0.354 | 正常 |
| confirm/oc_2048_confirm | confirm2048_s44 | 44 | 2048 | 4.5e-3 | 300 | 7000 | 3.91e-03 | 1.481 | 4.183 | 0.976 | 0.4137 | 0.346 | 正常（稳定簇中 60s 最大者） |
| followup/oc_2048_stability | diag2048_s233_lr4e-3_wu300_ts7000 | 233 | 2048 | 4e-3 | 300 | 7000 | 3.88e-03 | 1.307 | 2.853 | 0.989 | 0.4686 | 0.353 | 正常 |
| followup/oc_2048_stability | diag2048_s233_lr4e-3_wu400_ts7000 | 233 | 2048 | 4e-3 | 400 | (停于400) | **0.1322** | **34.15** | 68.72 | 1.000 | 14.75 | 0.021 | **发散：ep10 Test→inf** |
| followup/oc_2048_stability | diag2048_s233_lr4.5e-3_wu300_ts7000 | 233 | 2048 | 4.5e-3 | 300 | 7000 | 3.72e-03 | 1.014 | 3.287 | 0.989 | 0.3098 | 0.355 | 正常（≡confirm2048_s233，见 K.6） |
| followup/oc_2048_stability | diag2048_s233_lr4.5e-3_wu400_ts7000 | 233 | 2048 | 4.5e-3 | 400 | 7000 | 3.83e-03 | 1.145 | 3.426 | 0.989 | 0.3431 | 0.352 | 正常 |

> **复核更正（如实标注）**：本次实测 oc 稳定子集（seed42/43/44 及 confirm/aligned）60s 中位实际落在 **≈0.78–1.48 m** 区间，**否定了"约 0.4–0.5 m"的先验预期**。oc 任务（含海流）在 60s 上系统性难于 noc（noc 最佳配方 ≈0.33 m），但二者数据集/state_dim 不同，**非受控对照**（见 §L.2）。

### K.4.2 oc 报告结论（标注来源）

- **`largebatch_oc_report.md`**：报告称 oc **不呈现 noc 的规律**——noc 中 bs4096 一贯较差，但 oc 的主要事件是 **bs2048 seed233 单次灾难性崩溃**（ep6 正常 → ep7 `Test inf` / `Fail 34/11` → ep8 起无成功 batch → 仅 246/7000 步，best 卡在 ep4=0.4572）。报告称这是"**genuine optimization failure, not a reporting artifact**"。在稳定子集（42/43/44）上 2048 与 4096 接近：报告称要安全默认就用 bs4096（更可靠），推荐默认 `bs4096, lr6e-3, wu400, steps5000`。
- **`largebatch_oc_paired_summary.md`**（配对稳定子集 4096/2048 比值）：best_test 1.02×、resampled 60s 中位 0.939×（4096 略好）、heldout 30s 中位 1.17×（4096 较差）、train_hours 0.787×（4096 较快）。
- **`oc_confirm_vs_baselines.md`**：报告称 `confirm2048`（lr4.5e-3/wu300/ts7000）**修复了 seed233 崩溃**（全 4 seed reach_target、零 non-finite）；相对旧 4096：优化质量/局部 vel RMSE/heldout 30s 中位/completion/发散率更好，但 resampled 60s 中位 1.127×、p95 1.113× **更差**，慢 1.217×。报告称这是"**配方比较，非纯 batch-size 单变量**，尚不能宣称 4096 过时"。
- **`oc_aligned_vs_confirm.md`**：严格对齐 `aligned bs4096, lr4.5e-3/wu300/ts7000` vs confirm2048。报告称 aligned4096 四 seed 全稳定，best_test/速度 RMSE/resampled 60s 中位/p95/rot 中位更好；confirm2048 在 heldout 30s 中位/completion/发散率/效率（快约 30%）略优。结论："去掉配方失配后，4096 不再需要旧的稳定性论据；权衡变为 **60s 精度 vs 效率/发散裕度**"。
- **`largebatch_followup.md`（oc 部分）**：oc_2048_stability 最佳 `bs2048, lr4.5e-3, wu300, steps7000`（best 3.719e-03）；`lr4e-3, wu400` **发散**（ep10，best 0.1322，60s 中位 34.15）。报告称**驱动发散的是 lr/warmup 选择，而非单独的 batch size**。

### K.4.3 海流物理退化分析（`../g3_5_4/docs/current_ocean_performance_analysis.md`，单独摘录）

该文是**海流场景物理/架构退化分析**（根目录代码相对 `original/` 版本为何在海流下变差），**不是 batch 配方报告**，故不计入上面的配方数字。报告分层结论（标注来源）：

1. **推进器入流定义（最高置信度）**：报告称根目录把推进器入流从相对轴向速度改成总速度模长 `‖ν‖`，正确做法应是 `nu_r[0]`；海流下尤其危险——主体/舵面按 `nu_r`、推进器按 `‖ν‖`，构成混合闭合，横流被错误注入推进器支路。列为第一优先级修复。
2. **B_net 条件变量语义（第二层）**：根目录 B_net 偏向 `nu_r + u_act (+ v_c_body)`，original 在海流下更接近 `nu_r + u_act + v_total`；报告称影响控制力分支可学习性，但二者均非理想答案。
3. **D/J current conditioning（第三层）**：真实架构差异但不应先验视为主因，建议作为可控 ablation 开关暴露。
4. **Actuator loss（第四层）**：有建模意义（约束 u_actual 可辨识性），但更像训练目标重加权项。
- 报告称**不宜直接下结论**："推进器入流错误是唯一已确认主因"、"D/J conditioning 可完全排除"、"B_net 一定应看 v_total/v_c_body" 三者均**不下定论**。

## K.5 g3_5_4 异常分类（沿用三分类口径的诚实落点）

g3_5_4 是**单一架构的配方扫描**，与三分类口径设计所针对的"固定配方下多模型逐种子行为"语境不同，因此**三类映射并不干净**，如实说明如下：

| g3_5_4 异常实例 | 三分类归属 | 诚实说明（标注来源） |
|---|---|---|
| oc `bs2048_seed233` 崩溃（ep7→inf，67.68 m） | **不属 (1)/(3)，最接近"配方驱动脆弱"** | 报告称是 genuine optimization failure；但 `confirm2048`（同 seed233，仅 lr 5e-3→4.5e-3）**已复现稳定**——即同种子在更好配方下收敛。故**非模型固有逐种子脆弱**，是**配方（lr/warmup）条件性失败**，被换配方修复。无 provenance 重训，不能归 (1) 环境漂移。 |
| oc followup `lr4e-3_wu400_ts7000` 发散（ep10，34.15 m） | 同上，配方驱动 | 同 seed233 在 `lr4e-3_wu300` 收敛（best 3.879e-03）→ 报告归因 wu400 与 lr 的组合，非 batch size。 |
| noc followup `lr6e-3_wu400_ts7000` 发散（ep29，best 0.175） | 同上，配方驱动 | 同 seed233 在 `lr5e-3` 或 `ts5000` 均收敛 → 报告归因 lr/steps 组合。 |

> **诚实结论**：g3_5_4 的全部 3 个异常都是 **"配方驱动训练发散（lr/warmup/steps 组合）"** ——报告一致将其归因于优化器配方、并以"同种子换配方即收敛"证明可被修复；**无一**符合 (1) 环境漂移（无 provenance 重训）或 (3) 结构性全种子发散（每个都是单配方单种子、且同种子的姊妹配方收敛）。它们与 (2) 真实可复现脆弱的相似点仅在于"非随机一次性"（与特定配方绑定），但本质是**配方条件性**而非模型固有。本节据此把它们标为独立的"配方驱动"类，不强行塞进三分类。

## K.6 g3_5_4 内部 overlap 与计数诚实说明

磁盘上确为 **30 个独立 run 目录**（noc 10 + oc 20），但其中存在**配方重叠的再评估**，引用数字时需知：

- **oc**：`confirm/oc_2048_confirm/confirm2048_s233` 与 `followup/oc_2048_stability/diag2048_s233_lr4.5e-3_wu300_ts7000` 为**同配方同种子**，`best_loss` 逐位一致（0.0037186…），训练历史相同（同一训练，两份报告各自引用 / 确定性复跑）。
- **noc**：`run_largebatch_noc/bs4096_seed233` 与 `followup/noc_4096_recipe/diag4096_s233_lr6e-3_wu400_ts5000` 为**同配方**，`best_loss` 同为 3.870e-04（followup 的配方扫描把基线 recipe 也纳入作参照点）。
- **评估集大小不一致**：`run_largebatch_*` 的 resampled 用 **90** 条轨迹，`followup` 用 **84** 条——故上述"同一训练"在两 suite 下 60s 中位会略不同（如 confirm 0.9400 vs followup 1.014；noc 主 0.6698 vs followup 0.6885）。**跨 suite 比较 60s 数字时务必同评估集**。

---

# L. 跨仓库总账对账表（g3_5_5 + g3_5_4）

> 本节为 2026-05-30 跨仓库复核新增，统一两仓库的 run 计数、noc/oc 拆分、重叠、镜像/环境与论文证据归属。评估口径统一为 60s 终端位置误差中位数、scope=overall。

## L.1 run 计数对账

| 仓库 | 分区 | run 数 | 数据 | 模型 | 是否进入当前论文 §8 |
|---|---|---:|---|---|---|
| **g3_5_5** | A. Catalog 主线 | 98 | oc（`d0be9434`/s23） | 10 模型族 | 部分（补充表，剔除 2 个 stale 种子） |
| **g3_5_5** | B. Phase-1A | 148 | oc（`d0be9434`/s23） | 7/4 模型 | **是（§8 主证据）** |
| **g3_5_5** | E. Unused 旧噪声 | 54 | oc（`d0be9434`/s23） | 6 模型 | 否（弃用 noise-v1） |
| g3_5_5 小计 | | **300** | **全 oc** | | |
| **g3_5_4** | noc（run_largebatch_noc + followup/noc_4096_recipe） | 10 | **noc**（`32ec4535`/s42） | 仅 `ph_se3_full` | 否 |
| **g3_5_4** | oc（run_largebatch_oc + oc_aligned + confirm + followup/oc_2048_stability） | 20 | oc（`89c80d68`/s42） | 仅 `ph_se3_full` | 否 |
| g3_5_4 小计 | | **30** | **noc 10 + oc 20** | | |
| **两仓库合计** | | **330** | **noc 10 + oc 320** | | |

（g3_5_5 另有不计入 300 的：C. provenance 审计 forensic 重训 1 个；多个 manifest-only 聚合目录——见 §1。）

## L.2 noc/oc 拆分与重叠

- **noc 仅存在于 g3_5_4**（10 run，数据集 `32ec4535`）。g3_5_5 全部 300 run 均为 oc，**无 noc**。
- **oc 在两仓库都有，但数据集不同**：g3_5_4 oc = `89c80d68`（gen-seed 42、state_dim 27）；g3_5_5 oc = `d0be9434`（gen-seed 23、state_dim 27）。**两者不是同一数据集**。
- **跨仓库 run 重叠 = 0**：不同代码镜像、不同数据集、g3_5_4 仅训 `ph_se3_full`——没有任何 run 被两仓库共享。
- **仓库内 overlap**：g3_5_4 内有 2 对"同配方再评估"（见 K.6）；g3_5_5 内 catalog rollout 有同一 checkpoint 多 `rollout_run_id` 的去重坑（见下 L.5）。
- **可比性**：g3_5_4 noc-vs-oc 是同仓库同模型、但数据集/state_dim 不同（海流有无），属"任务难度差异"而非受控消融；两代 oc（89c80d68 vs d0be9434）gen-seed 不同，**不可直接跨代比较数值**。

## L.3 代码镜像 / 环境对账

| 分区 | 云镜像 | 时间窗 | 溯源元数据 |
|---|---|---|---|
| g3_5_5 A（catalog） | `auvhamnode/g3_5_5` | 04-04~24 | 无 `code_revision`（审计补 `_audit_meta` 仅新 run）；2 种子标 `stale_environment_drift` |
| g3_5_5 B（Phase-1A） | `auvhamnode/g3_5_7` | 04-24~26 | 带 `_audit_meta/`，当前 main 可逐位复现 |
| g3_5_5 C（forensic 重训） | `auvhamnode/g3_5_7` | 05-12 | 完整 `_audit_meta/`（PyTorch 2.10.0+cu128 / CUDA 12.8 / cuDNN 91002） |
| g3_5_5 E（unused） | `auvhamnode/g3_5_5` | 04-06 前 | 弃用 noise-v1（`--noise_level l1/l2`） |
| **g3_5_4 noc + oc** | `auvhamnode/g3_5_4` | **03-30~04-02** | **无** `code_revision`/`git_commit`/`environment`/`_audit_meta`（`device=cuda`，余不可考） |

## L.4 是否进入当前论文证据

- **进入 §8**：仅 g3_5_5 **B 区**（主证据，可逐位复现）+ g3_5_5 **A 区**可信子集（作为补充表，按异常三分类剔除 2 个 stale 种子；同镜像算倍数）。
- **不进入**：g3_5_5 E 区（弃用噪声）；g3_5_5 C（仅审计 forensic）；**g3_5_4 全部 30 run（noc + oc）**——前代不同数据集/不同代码镜像，论文 §8 不使用（与 legacy doc §5 一致）。
- g3_5_4 的价值定位：早期**训练稳定性 / 大 batch 配方 + noc-vs-oc 数据对照**研究，为 g3_5_5 收敛到"结构化 SE(3) pH + 海流主线 + profile 噪声 + 规范化 catalog"提供配方与稳定性背景，但本身不作为论文模型证据。

## L.5 评估口径统一说明

- 统一指标：rollout 自由递推 **60s 终端位置误差中位数**，scope=**overall**，先轨迹内取中位、再种子间取均值。
- **g3_5_4**：直接取各 suite `*_summary.csv`/`*_runs.csv` 的 `resampled_pos_med_60s`，等于该 run `rollout_benchmark/resampled_batch_compare_10_30_60_*/horizon_metrics.csv` 中 `scenario=ALL`、`horizon_s=60.0` 的 `final_position_error_median`（已逐位核验）。**仅 clean rollout，无 profile 噪声评估**。
- **g3_5_5（A 区 catalog）**：从 `canonical_rollout_summary_long.csv` 过滤 `metric_name=final_position_error & stat_name∈{median,p95} & horizon_s=60.0 & scope=overall`，并按 `rollout_run_registry.csv` 的 `is_selection_eligible=1`、优先 `resampled_traj30_*` 对每 (model,train,seed,profile) **唯一去重**。
  - **去重坑（复核实证）**：registry 共 352 行、`is_selection_eligible=1` 325 行；其中 40 行（phnode_full 20 + ablate_no_mass_prior 20）是 `traj8` 的 `*_iideval_*` 探针，`selection_priority=100` **不低于**正确的 `resampled_traj30` matched（priority 80）。例：phnode_full seed43 clean nominal_eval，traj8_iideval=**34.28 m** vs traj30 matched=**0.84 m**（**41×** 误差）。若按"最高 priority"简单去重会取错——必须额外按 `rollout_purpose` 排除 iideval/traj8 探针。
  - 复核交叉校验（全部成立）：phnode_full clean 非漂移种子(43,44,45,47)均值 **0.6098 m**；seed42 **4.2148**、seed46 **47.8637**（标 stale）；noisy nominal_eval phnode_full 逐种子 42/43/44/45/46/47 = 5.159/1.092/0.956/1.621/0.932/1.055（均值 1.803）；ablate_diag_damping clean **4.166**；ablate_bu_only clean **27.80**。

---

# M. 磁盘 ↔ 清单覆盖率核验（2026-05-30 复核，可复现）

> 目的：证明两仓库 `checkpoints/` 下**每个**含 `best_model.pt` 的训练目录都已 1:1 映射到本清单的某个分区，无孤儿目录；且所有 0-run（manifest / 元数据 / 日志）目录都已点名。核验方法：对每个顶层目录 `find -name best_model.pt | wc -l`，再把计数归并到分区。

## M.1 g3_5_5 覆盖矩阵（300 run）

| 顶层目录组 | run 数 | 分区 | 清单位置 |
|---|---:|---|---|
| `sweep_oc_all/` | 45 | A | §A.1 |
| `sweep_oc_all_noise/` | 30 | A | §A.1 |
| `sweep_oc_main_noise_*_extra_42-46-47/` | 3 | A（followup P1-1） | §A.1 |
| `sweep_oc_key_ablation_noise_*_extra_42-46-47/` | 6 | A（followup P1-1） | §A.1 |
| `sweep_oc_phase1_probe_{clean,iid,v4lite}_*` | 2+2+2=6 | A（probe） | §A.1 |
| `sweep_oc_phase1_smoke_clean_fix_*` | 4 | A（probe） | §A.1 |
| `sweep_oc_smoke/` | 3 | A（smoke） | §A.1 / §D |
| `sweep_oc_main_noise_seed42_smoke/` | 1 | A（smoke） | §A.1 / §D |
| **A 区小计** | **98** | | |
| `sweep_oc_phase1a_decision_clean_t2_wpfrag_*`（7 目录×5） | 35 | B | §B.1 |
| `sweep_oc_phase1a_decision_iid_t2_wpfrag_*`（4×5） | 20 | B | §B.1 |
| `sweep_oc_phase1a_decision_v4lite_t2_wpfrag_*`（4×5） | 20 | B | §B.1 |
| `sweep_oc_phase1a_decision_extra43-45_{clean,iid,v4lite}_*_cleanrun_v1`（3×6） | 18 | B | §B.2 |
| `sweep_oc_phase1a_smoke3_{clean,iid,v4lite}_*_cleanrun_v1`（3×9） | 27 | B | §B.2 |
| `sweep_oc_phase1a_smoke1_{clean,iid,v4lite}_*_cleanrun_v1`（3×3） | 9 | B（smoke） | §B.3 |
| `sweep_oc_phase1a_smoke1_clean_t2_wpfrag_*`（7×1） | 7 | B（smoke） | §B.3 |
| `sweep_oc_phase1a_smoke1_iid_t2_wpfrag_*`（4×1） | 4 | B（smoke） | §B.3 |
| `sweep_oc_phase1a_smoke1_v4lite_t2_wpfrag_*`（4×1） | 4 | B（smoke） | §B.3 |
| `smoke_v4lite/`（ep=3 code smoke） | 4 | B（smoke） | §B.3 / §D |
| **B 区小计** | **148** | | |
| `checkpoints/unused/`（l1/l2 旧噪声 3 suite） | 54 | E | §E |
| **E 区小计** | **54** | | |
| **g3_5_5 合计** | **300** | | |

**0-run 目录（不计入 300，均已点名）**：`sweep_oc_phase1a_decision_{clean,iid,v4lite}_*_cleanrun_v1`、`decision_proxy_*`、`sweep_oc_phase1_probe_iideval_20260424_024232`、`sweep_oc_phase1_smoke_matched_20260423_173332`（manifest 聚合，§1）；`phase1a_metadata_*`×8、`phase1a_logs/`、顶层 `p1_2_clean_matched_eval_*.{log,pid}`×5（元数据/日志，§1/§D）。

## M.2 g3_5_4 覆盖矩阵（30 run）

| 顶层目录 | run 数 | noc/oc | 清单位置 |
|---|---:|---|---|
| `run_largebatch_noc/` | 6 | noc | §K.3 |
| `run_largebatch_followup/`（noc_4096_recipe 4 + oc_2048_stability 4） | 8 | noc 4 + oc 4 | §K.3 / §K.4 |
| `run_largebatch_oc/` | 8 | oc | §K.4 |
| `run_largebatch_oc_aligned/` | 4 | oc | §K.4 |
| `run_largebatch_confirm/`（oc_2048_confirm） | 4 | oc | §K.4 |
| **g3_5_4 合计** | **30**（noc 10 + oc 20） | | |

仓库内无 `checkpoints/` 之外的 `best_model.pt`。

## M.3 复现命令

```bash
# 每个顶层目录的 run 数（两仓库各跑一次）
for d in checkpoints/*/; do echo "$(find "$d" -name best_model.pt 2>/dev/null | wc -l)  $d"; done
# 总 run 数
find checkpoints -name best_model.pt | wc -l   # g3_5_5 → 300；g3_5_4 → 30
```

**核验结论**：g3_5_5 300 + g3_5_4 30 = **330 个训练 run 全部映射到分区，无孤儿**；noc/oc、catalog/t2_wpfrag/phase-1A/cleanrun v1/smoke/probe/unused/provenance/manifest 各类别均已收录。
