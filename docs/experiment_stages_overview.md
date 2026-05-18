# 实验阶段总览

生成时间：2026-05-13 CST (+0800)
最后更新：2026-05-13 CST (+0800)
来源：本仓库 `checkpoints/`、`analysis/oc_data_catalog/`、`analysis/provenance_audit/`、`EXPERIMENT_PROGRESS_TRACKER.md` 与 `experiment_progress_log.csv` 的实际状态。

## 0. 文档目的

本文档把仓库里现存的训练产物按**「训练时期 + 训练环境」**做时间线分阶段梳理，回答"哪些 run 来自哪个云镜像、哪些已被 audit 标 stale、哪些只能作流程验证用"这种问题。

它**不替代**：

- [EXPERIMENT_PROGRESS_TRACKER.md](../EXPERIMENT_PROGRESS_TRACKER.md)（按计划项/状态组织的总表）
- [docs/repo_structure_audit.md](repo_structure_audit.md)（按目录结构组织的边界文档）
- [analysis/oc_data_catalog/](../analysis/oc_data_catalog/)（按 run 粒度组织的规范化目录）
- [docs/provenance_audit_phnode_full_clean.md](provenance_audit_phnode_full_clean.md)（catalog 时代 fragility 取证报告）

阶段划分的核心轴是「**云端训练镜像**」+「**时间窗口**」，因为 2026-05-13 Phase 3 audit 已确认 catalog 时代 `phnode_full` clean seed46 fragility 是 `auvhamnode/g3_5_5` 镜像的 cuDNN 耦合训练发散，而 `auvhamnode/g3_5_7` 镜像下当前 main 能比特相同地复现 cleanrun v1。

## 1. 阶段总览表

| # | 阶段名 | 时间窗口 | 云镜像 | 本地形态 | 主目录 | 数据可用性 |
|---|---|---|---|---|---|---|
| **A** | Catalog 时代 | 2026-04-04 ~ 04-21 | `auvhamnode/g3_5_5` | 完整产物 | `checkpoints/sweep_oc_all/`、`checkpoints/sweep_oc_all_noise/`、`checkpoints/sweep_oc_main_noise_*_extra_*`、`checkpoints/sweep_oc_key_ablation_noise_*_extra_*` | 部分被环境漂移污染（详见 §2） |
| **B** | Cleanrun v1（Phase-1A 决策套件） | 2026-04-24 ~ 04-26 | `auvhamnode/g3_5_7` | **manifest only**（runs.tsv） | `checkpoints/sweep_oc_phase1a_*_cleanrun_v1/`、`checkpoints/phase1a_metadata_phase1a_oc_v4lite_cleanrun_v1/` | 当前 main 上比特相同可复现 |
| **C** | Provenance audit retrain | 2026-05-12 | `auvhamnode/g3_5_7`（取证） | tarball + 报告 | `analysis/provenance_audit/phase3_retrain/audit_phase3_seed46_clean_20260512_095957/` | 取证用 1 个 run，已落 `findings.md` |
| **D** | Smoke / probe | 2026-04-21 ~ 04-24 | g5/g7/本地皆有 | 完整产物 | `checkpoints/sweep_oc_smoke/`、`checkpoints/smoke_v4lite/`、`checkpoints/sweep_oc_phase1_probe_*`、`checkpoints/sweep_oc_phase1_smoke_*`、`checkpoints/sweep_oc_phase1a_smoke{1,3}_*` | 仅 flow-validation，不用作模型证据 |
| **E** | Unused / legacy | 2026-04-06 之前 | `auvhamnode/g3_5_5` | 完整产物 | `checkpoints/unused/`、`original/bf3n/` | 已废弃 noise v1 设计 |

## 2. 阶段 A — Catalog 时代

### 2.1 范围

进 `analysis/oc_data_catalog/run_inventory.csv` 的全部 88 行均来自这一阶段（catalog 是 g3_5_5 镜像 sweeps 的规范化视图）。

- **云镜像**：`/content/drive/MyDrive/Colab Notebooks/auvhamnode/g3_5_5/`
- **数据集主体**：`auv_oc_traj1000_blk150_s23_d0be9434.pkl`（traj=1000、blocks=150、seed=23、digest=d0be9434）
- **train_type 分布**：clean 训练 46 行，noisy 训练 42 行
- **noise reference 分布**：43 行使用 `remus100_dr`
- **模型覆盖**：10 个模型变体
  - PHNODE 主线：`phnode_full`、`phnode_qforce`、`phnode_merged_force`
  - 关键消融：`ablate_no_mass_prior`、`ablate_no_lift`、`ablate_diag_damping`、`ablate_bu_only`
  - 半结构化/黑盒：`se3_momentum_blackbox`、`se3_accel_blackbox`、`blackbox_fullstate`

### 2.2 `phnode_full` 16 个 catalog 行拆解

| Sweep 目录 | 训练类型 | seed | 备注 |
|---|---|---|---|
| `sweep_oc_core_default_..._s42-43-44_20260404_115414` | clean | 42, 43, 44 | **seed42** 已标 `stale_environment_drift` |
| `sweep_oc_phnode_focus_extra3_..._s45-46-47` | clean | 45, 46, 47 | **seed46** 已标 `stale_environment_drift`（best_loss 0.27、60s rollout 47 m） |
| `sweep_oc_main_noise_nominal_train_remus100_dr_..._s43-44-45_20260409` | noisy | 43, 44, 45 | P1-1 初始 3-seed |
| `sweep_oc_main_noise_nominal_train_remus100_dr_extra_42-46-47` | noisy | 42, 46, 47 | P1-1 补 3-seed |
| `sweep_oc_main_noise_seed42_smoke` | noisy | 42 | smoke，但被 catalog 收录 |
| `sweep_oc_v4lite_protocol_smoke_..._traj12_blk8_s123_20260421` | clean | 42 × 3 行 | 12-traj 8-block 微型 smoke，**不是正式证据** |

注：catalog 里同一 (model, seed) 出现多行属于不同 sweep family，不是去重错误。

### 2.3 Audit 已标记的 evidence_status

仅 2 行被 sidecar `analysis/oc_data_catalog/evidence_status_overrides.csv` 显式标 `stale_environment_drift`：

| run_uid | 原因 |
|---|---|
| `sweep_oc_core_default_..._s42-43-44_20260404_115414/main_phnode_full_seed42` | catalog cloud mirror 环境漂移，best_loss=2.10e-02 vs cleanrun v1 4.02e-03 |
| `sweep_oc_phnode_focus_extra3_..._s45-46-47/main_phnode_full_seed46` | catalog cloud mirror cuDNN 耦合训练发散，epoch 24-26 4.68e+25→inf |

其他 catalog 行**没有被显式判 stale**，但与上述两行同属一个云镜像；理论上同一环境下可能有「轻微漂移」未被检出。EXPERIMENT_PROGRESS_TRACKER.md §7.2 与 §7.4 已被加 `needs_recheck` 标签。

### 2.4 衍生分析层

`analysis/oc_data_catalog/` 下所有 CSV / canonical 视图、`docs/oc_experiments_comprehensive_report.md`、`docs/oc_followup_results_p1_p2.md` 都基于阶段 A 数据。Phase 3 audit 之后，这些报告里跨模型对比应被视为「来自被污染环境」，引用时需附 audit 说明。

## 3. 阶段 B — Cleanrun v1（Phase-1A 决策套件）

### 3.1 关键定位

**Cleanrun v1 不是单一 phnode_full 复跑，而是一个完整的 Phase-1A 决策套件**。Phase 3 audit 用它作为「新基线」依据，是因为当前 main + g3_5_7 镜像能比特相同复现它的 C46。

### 3.2 矩阵规模

- **云镜像**：`/content/drive/MyDrive/Colab Notebooks/auvhamnode/g3_5_7/`
- **`run_tag`**：`phase1a_oc_v4lite_cleanrun_v1`
- **写入 UTC**：2026-04-24T12:11:30Z（见 `checkpoints/phase1a_metadata_phase1a_oc_v4lite_cleanrun_v1/phase1a_run_config.json`）
- **本地形态**：仅 `runs.tsv` 与 `suite_config.txt`。每个 run 目录里**没有 `best_model.pt`**，模型文件留在 Drive。
- **矩阵**：**3 模型 × 3 协议 × 5 seed = 45 个训练 run**
  - 模型：`phnode_full`（main）、`ablate_no_lift`、`ablate_no_mass_prior`（**仅 PHNODE 主线 + 两个关键消融**）
  - 协议：`clean` / `iid` / `v4lite`
  - seed：42, 43, 44, 45, 46
- **数据集**：`/content/drive/MyDrive/Colab Notebooks/auvhamnode/g3_5_7/data/auv_oc_traj1000_blk150_s23_d0be9434.pkl`（与 catalog 同 digest）

### 3.3 环境指纹

| 项 | 值 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| cuDNN | 91002 |
| 平台 | Linux-6.6.113+-x86_64-with-glibc2.35 |

### 3.4 对应本地 manifest 目录

| 类型 | 目录 |
|---|---|
| decision 主套件 | `sweep_oc_phase1a_decision_{clean,iid,v4lite}_phase1a_oc_v4lite_cleanrun_v1/` |
| decision seed 43/45 补 | `sweep_oc_phase1a_decision_extra43-45_{clean,iid,v4lite}_*` |
| smoke3 套件（seed 42/44/46 部分被 decision 引用） | `sweep_oc_phase1a_smoke3_{clean,iid,v4lite}_*` |
| smoke1（仅 phnode_full） | `sweep_oc_phase1a_smoke1_{clean,iid,v4lite}_*` |
| proxy 视图（横向拼合三个协议） | `sweep_oc_phase1a_decision_proxy_phase1a_oc_v4lite_cleanrun_v1/` |
| 元数据 | `phase1a_metadata_phase1a_oc_v4lite_cleanrun_v1/{phase1a_environment.json, phase1a_run_config.json}` |

### 3.5 评估协议（来自 phase1a_run_config.json）

- `DECISION_SEEDS=42 43 44 45 46`
- `DECISION_EVAL_NUM_TRAJ_PER_SCENARIO=30`
- `EVAL_TIMES=10 30 60`
- `EVAL_SCENARIOS=PRBS CHIRP OU`
- `IID_EVAL_PROFILES=clean nominal_eval`
- `V4_EVAL_PROFILES=nominal_eval`
- `STRICT_ZERO_NOISE_AUDIT=1`

### 3.6 局限

- **不覆盖 baseline / phnode_qforce / blackbox / se3_*_blackbox / 其他消融**——只有 PHNODE 主线 + 两个消融
- 如果论文需要 "phnode_full vs baseline" 或 "phnode_full vs blackbox" 跨家族对比，cleanrun v1 不够；需要在阶段 A 找对应数据或重跑

## 4. 阶段 C — Provenance audit retrain

### 4.1 范围

仅 1 个 run，用于取证：

- **位置**：`analysis/provenance_audit/phase3_retrain/audit_phase3_seed46_clean_20260512_095957/`
- **tarball**：`audit_phase3_seed46_clean_20260512_095957.tar.gz`（被 `.gitignore` 排除，未入仓）
- **报告**：`analysis/provenance_audit/phase3_retrain/findings.md`、`code_fix_search.md`
- **配置**：`phnode_full × seed46 × clean`，HEAD `7643dc9`（branch `provenance-audit-phnode_full`）

### 4.2 关键结论

- `best_loss=4.0471e-03`、60s clean `pos_err_median=0.45575005972892096` m
- 与 cleanrun v1 C46 **比特相同**（IEEE 754 完全一致）
- 证实 catalog 时代 fragility ≠ git/code bug，根因是 g3_5_5 镜像 cuDNN 算法选择 + epoch 24 极端 batch 的 stochastic 偶然耦合

### 4.3 限定

这一阶段**只验证 phnode_full × clean × seed46 一个 cell**。其他 seed、其他模型、其他协议**没有**用 audit 流程独立复现，只是因为环境一致被推断"也可用"。

## 5. 阶段 D — Smoke / probe

进 `experiment_progress_log.csv` 时被标 `flow_validation_only`，不可作为模型证据。

| 目录 | 用途 | 备注 |
|---|---|---|
| `checkpoints/sweep_oc_smoke/` | 早期 dataset/pipeline smoke | — |
| `checkpoints/smoke_v4lite/` | v4-lite 协议 smoke | — |
| `checkpoints/sweep_oc_phase1_probe_clean_20260423_180908/` | Phase-1 clean probe | 见 `experiment_report.md` |
| `checkpoints/sweep_oc_phase1_probe_iid_20260423_180908/` | Phase-1 iid probe | — |
| `checkpoints/sweep_oc_phase1_probe_iideval_20260424_024232/` | Phase-1 iid_eval probe | — |
| `checkpoints/sweep_oc_phase1_probe_v4lite_20260423_180908/` | Phase-1 v4-lite probe | — |
| `checkpoints/sweep_oc_phase1_smoke_clean_fix_20260423_124711/` | Phase-1 clean smoke 修复 | — |
| `checkpoints/sweep_oc_phase1_smoke_matched_20260423_173332/` | Phase-1 matched smoke | — |
| `checkpoints/sweep_oc_phase1a_smoke1_*` | Phase-1A smoke1（仅 phnode_full） | — |
| `checkpoints/sweep_oc_phase1a_smoke3_*` | Phase-1A smoke3 | seed 42/44/46 子集被 cleanrun v1 decision manifest 引用 |

特别提示：阶段 D 的 `sweep_oc_phase1a_smoke3_*` 与阶段 B 在物理目录上有交集（cleanrun v1 把 smoke3 的 seed 42/44/46 作为 decision 的一部分）。这是 cleanrun v1 设计上的「分批训练 + 后期合并」工艺，不是数据污染。

## 6. 阶段 E — Unused / legacy

- `checkpoints/unused/sweep_oc_core_noise_l1_..._20260406_*`
- `checkpoints/unused/sweep_oc_core_noise_l2_..._20260406_*`
- `original/bf3n/`

特征：使用已淘汰的 `--noise_level l1/l2` CLI 接口（旧 noise v1 设计），现接口被 profile-based 替换。**不应作为当前任何 claim 的依据**。

## 7. 现状评估（哪些 claim 站得住、哪些不站得住）

| Claim 类别 | 数据来源 | 当前状态 |
|---|---|---|
| `phnode_full` clean 自身性能（5-seed）| 阶段 B（cleanrun v1） | ✅ 安全，已验证 = 当前 main |
| `phnode_full` vs `ablate_no_lift`、`ablate_no_mass_prior`（PHNODE 家族内） | 阶段 B | ✅ 同环境同协议同 seed，可直接比较 |
| `phnode_full` vs baseline / phnode_qforce / blackbox（跨家族）| 阶段 A | ⚠️ 对手方仍 catalog 时代；如要严格 ranking，需补阶段 B 风格的同环境 sweep |
| Catalog §12 fragility 表 | 阶段 A | ⚠️ A42/A46 已标 stale；其余行为 `needs_recheck` |
| "noisy 训练修复 phnode_full seed46" framing | 阶段 A | ❌ 已破，因 seed46 clean 异常本身是环境产物 |
| `phnode_full` iid / v4lite 性能（5-seed）| 阶段 B | ✅ 安全 |

## 8. 关联文档

- 计划/总览：[EXPERIMENT_PROGRESS_TRACKER.md](../EXPERIMENT_PROGRESS_TRACKER.md)、[analysis/experiment_progress_log.csv](../analysis/experiment_progress_log.csv)
- 仓库边界：[docs/repo_structure_audit.md](repo_structure_audit.md)
- Catalog 体系：[docs/oc_data_catalog_dictionary.md](oc_data_catalog_dictionary.md)、[docs/oc_result_selection_policy.md](oc_result_selection_policy.md)
- Audit 报告：[docs/provenance_audit_phnode_full_clean.md](provenance_audit_phnode_full_clean.md)、[analysis/provenance_audit/PLAN.md](../analysis/provenance_audit/PLAN.md)
- Phase-1A 决策计划：[docs/phase1_realistic_validation_plan.md](phase1_realistic_validation_plan.md)
- 噪声协议设计：[docs/noise_model_design.md](noise_model_design.md)、[docs/noise_design_v4_lite_traj_consistent_ic.md](noise_design_v4_lite_traj_consistent_ic.md)

## 9. 更新日志

- **2026-05-13**：首次落盘。基于 Phase 3+4 audit 结果整理；阶段划分轴为「云镜像 + 时间窗口」。
