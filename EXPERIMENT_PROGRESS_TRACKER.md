# AUV SE(3) 实验进展跟踪总表

生成时间：2026-04-25 11:44:54 CST (+0800)  
工作区：`/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5`  
状态口径：基于 `README.md`、`docs/`、`analysis/oc_data_catalog/` 与 `checkpoints/` 中可见的本地文件状态。

## 1. 文档目的

本文档用于把当前仓库中的实验计划、实验报告、已完成实验、流程验证实验和未进行实验放到同一个进展视图中，方便后续继续跟踪。

本文档不替代以下权威来源：

- 当前代码和入口说明：[README.md](README.md)
- 仓库状态边界：[docs/repo_structure_audit.md](docs/repo_structure_audit.md)
- OC 结果目录：[analysis/oc_data_catalog/](analysis/oc_data_catalog/)
- OC 结果选择规则：[docs/oc_result_selection_policy.md](docs/oc_result_selection_policy.md)

## 2. 当前总览

当前仓库的正式研究主线是：

```text
REMUS100 simulation
  -> dataset generation
  -> model training
  -> rollout benchmark
  -> OC result catalog
```

当前重点是 `oc` 场景下的 clean vs noisy-IC training、长时 rollout benchmark、profile-based 初始条件噪声，以及 `analysis/oc_data_catalog/` 的规范化结果查询。

截至本次整理，catalog 中 `run_inventory.csv` 记录了 88 个 run：

- clean training：46 个
- noisy training：42 个
- `remus100_dr` noisy-reference run：43 个
- canonical rollout registry 已建立，`canonical_rollout_summary_long.csv` 与 `canonical_rollout_outcomes_long.csv` 已可用于默认论文图表和表格。

## 3. Markdown 文档分类

| 类别 | 文档 | 当前状态 |
|---|---|---|
| 仓库入口与边界 | [README.md](README.md), [docs/repo_structure_audit.md](docs/repo_structure_audit.md) | 当前权威入口 |
| 总体研究计划 | [docs/phnode_realistic_validation_plan.md](docs/phnode_realistic_validation_plan.md) | 研究计划，不是执行记录 |
| 现实导向执行方案 | [docs/phnode_realistic_validation_execution_plan.md](docs/phnode_realistic_validation_execution_plan.md), [docs/phase1_realistic_validation_plan.md](docs/phase1_realistic_validation_plan.md) | 当前 P0-P1 / Phase-1A 执行方案 |
| OC 补充实验计划 | [docs/oc_followup_experiment_plan.md](docs/oc_followup_experiment_plan.md) | P1 已执行并报告；P2/P3 尚未完成 |
| OC 正式结果报告 | [docs/oc_experiments_comprehensive_report.md](docs/oc_experiments_comprehensive_report.md), [docs/oc_followup_results_p1_p2.md](docs/oc_followup_results_p1_p2.md), [docs/oc_model_evaluation_overview.md](docs/oc_model_evaluation_overview.md), [docs/oc_results_section_zh.md](docs/oc_results_section_zh.md) | 已完成并有正式报告 |
| 噪声设计与协议 | [docs/noise_model_design.md](docs/noise_model_design.md), [docs/noise_design_v3_remus100_reference_grounded.md](docs/noise_design_v3_remus100_reference_grounded.md), [docs/noise_design_v4_dr_ekf_output.md](docs/noise_design_v4_dr_ekf_output.md), [docs/noise_design_v4_lite_traj_consistent_ic.md](docs/noise_design_v4_lite_traj_consistent_ic.md), [docs/v4_lite_protocol_spec.md](docs/v4_lite_protocol_spec.md), [docs/v4_b1_implementation_checklist.md](docs/v4_b1_implementation_checklist.md) | 设计/协议文档；部分已落代码，部分未形成正式实验结论 |
| 噪声运行手册 | [docs/noise_experiment_runbook.md](docs/noise_experiment_runbook.md), [docs/noise_cli_parameter_reference.md](docs/noise_cli_parameter_reference.md), [docs/noise_cli_command_templates.md](docs/noise_cli_command_templates.md) | 操作说明，不是结果报告 |
| Catalog 体系 | [docs/oc_data_catalog_plan.md](docs/oc_data_catalog_plan.md), [docs/oc_data_catalog_dictionary.md](docs/oc_data_catalog_dictionary.md), [docs/oc_result_selection_policy.md](docs/oc_result_selection_policy.md), [docs/oc_catalog_template_usage.md](docs/oc_catalog_template_usage.md) | 已落地 |
| 历史归档 | [docs/unused/](docs/unused/) | 旧方案/旧审查，不作为当前主线依据 |

## 4. 已完成并已有正式报告记录

### 4.1 Clean OC 主实验

状态：已完成，已报告。  
主要记录：

- [docs/oc_model_evaluation_overview.md](docs/oc_model_evaluation_overview.md)
- [docs/oc_results_section_zh.md](docs/oc_results_section_zh.md)
- [docs/oc_experiments_comprehensive_report.md](docs/oc_experiments_comprehensive_report.md)

当前正式结论：

- clean all-seed 下，整体最强模型是 `baseline/phnode_qforce`。
- clean 下最强 PHNODE family 是 `ablation/ablate_no_lift`。
- `main/phnode_full` 有强 stable cluster，但存在真实 bad-outlier seeds：`42` 与 `46`。
- `ablation/ablate_bu_only` 结构性退化。
- coupled damping 有明确价值。
- 当前 mass prior 尚未显示出不可替代性。

### 4.2 原始 noisy OC sweep

状态：已完成，已有初始报告，但部分结论已被后续 P1 follow-up 修正。  
主要记录：

- [docs/oc_experiments_comprehensive_report.md](docs/oc_experiments_comprehensive_report.md)

注意：

- 该报告中关于 noisy training 下 `phnode_full` 可作为 noisy all-seed winner 的早期判断，不应再作为最终结论。
- 最新 noisy-training 结论以 [docs/oc_followup_results_p1_p2.md](docs/oc_followup_results_p1_p2.md) 为准。

### 4.3 P1-1：补齐 noisy training 关键 seeds

状态：已完成，已报告。  
主要记录：

- [docs/oc_followup_results_p1_p2.md](docs/oc_followup_results_p1_p2.md)
- `checkpoints/sweep_oc_main_noise_nominal_train_remus100_dr_extra_42-46-47/`
- `checkpoints/sweep_oc_key_ablation_noise_nominal_train_remus100_dr_extra_42-46-47/`

结论：

- noisy six-seed all-seed 下，headline model 不应写成 `phnode_full`，而应写成 `ablate_no_mass_prior` 更稳。
- `phnode_full` noisy training 主要修复了 `seed46`，但 `seed42` 仍然异常。
- `ablate_no_mass_prior` 更像稳定受益于 noisy training 的结构模型。
- `ablate_no_lift` 在 noisy training 下出现新的 `seed44` 异常。

### 4.4 P1-2：clean-trained checkpoints 补跑 noisy rollout

状态：已完成，已报告。  
主要记录：

- [docs/oc_followup_results_p1_p2.md](docs/oc_followup_results_p1_p2.md)
- `checkpoints/p1_2_clean_matched_eval_live_20260413_124225.log`

结论：

- noisy training 的收益与模型结构强耦合，不是普适提升。
- `phnode_full` 的 aggregate 改善主要由 `seed46` catastrophic failure 修复驱动。
- `ablate_no_mass_prior` 是当前最像“稳定 regularization 受益”的模型。
- `ablate_no_lift` 与 `phnode_qforce` 暂不支持 noisy training 带来稳定收益。

### 4.5 OC data catalog

状态：已完成并可用。  
主要记录：

- [analysis/oc_data_catalog/run_inventory.csv](analysis/oc_data_catalog/run_inventory.csv)
- [analysis/oc_data_catalog/rollout_run_registry.csv](analysis/oc_data_catalog/rollout_run_registry.csv)
- [analysis/oc_data_catalog/canonical_rollout_summary_long.csv](analysis/oc_data_catalog/canonical_rollout_summary_long.csv)
- [analysis/oc_data_catalog/canonical_rollout_outcomes_long.csv](analysis/oc_data_catalog/canonical_rollout_outcomes_long.csv)

说明：

- 原始长表保留全部记录。
- canonical 表按 [docs/oc_result_selection_policy.md](docs/oc_result_selection_policy.md) 选择默认引用结果。
- smoke/probe/legacy heldout 不进入默认正式图表。

## 5. 已完成但未形成顶层正式报告记录

这些内容已经有代码、产物或 checkpoint-local 报告，但当前不应作为正式研究结论直接引用。

### 5.1 `v4-lite` 协议实现与 smoke/probe

状态：代码已接入，smoke/probe 已跑过；未形成正式 Phase-1A 五 seed 决策报告。  
证据：

- 代码已支持 `noise_protocol=v4_lite`。
- 存在 `checkpoints/smoke_v4lite/`。
- 存在 `checkpoints/sweep_oc_phase1_probe_v4lite_20260423_180908/`。
- 这些目录下有 `experiment_report.md` 与 `phase1_summary.csv`。

限制：

- 这些被 [docs/repo_structure_audit.md](docs/repo_structure_audit.md) 标为 `flow-validation-only`。
- 目前只适合作为协议和流程验证，不适合作为论文结论或模型排序依据。

### 5.2 Phase-1 smoke/probe suites

状态：已跑过若干小规模流程验证；未形成正式 Phase-1A 决策结论。  
相关目录：

- `checkpoints/sweep_oc_phase1_probe_clean_20260423_180908/`
- `checkpoints/sweep_oc_phase1_probe_iid_20260423_180908/`
- `checkpoints/sweep_oc_phase1_probe_iideval_20260424_024232/`
- `checkpoints/sweep_oc_phase1_probe_v4lite_20260423_180908/`
- `checkpoints/sweep_oc_phase1_smoke_clean_fix_20260423_124711/`
- `checkpoints/sweep_oc_phase1_smoke_matched_20260423_173332/`

限制：

- 这些目录有本地 summary/report，但不属于正式 evidence-bearing checkpoint。
- 当前 catalog 的 canonical 正式结果层没有把这些作为主证据。

### 5.3 `current_bias_eval` 与 `remus100_ins` 接口

状态：代码和 CLI 文档已贯通；未看到正式主实验结果报告。  
依据：

- [docs/noise_cli_parameter_reference.md](docs/noise_cli_parameter_reference.md)
- [docs/noise_cli_command_templates.md](docs/noise_cli_command_templates.md)
- [docs/noise_experiment_runbook.md](docs/noise_experiment_runbook.md)

限制：

- 当前默认 OC 主线是 `remus100_dr`。
- `current_bias_eval` 更适合 `OC + remus100_ins` 扩展线，目前未见正式报告闭环。

## 6. 已计划但尚未进行或尚未完成正式闭环

### 6.1 Phase-1A 正式五 seed 决策实验

状态：未完成正式闭环。  
计划来源：

- [docs/phase1_realistic_validation_plan.md](docs/phase1_realistic_validation_plan.md)
- [docs/phnode_realistic_validation_execution_plan.md](docs/phnode_realistic_validation_execution_plan.md)

要求：

- `oc + known-current surrogate`
- 模型：`phnode_full`, `ablate_no_mass_prior`, `ablate_no_lift`
- seeds：`42,43,44,45,46`
- 训练协议：`clean`, `iid_noisy_ic`, `v4_lite`
- 评估协议：clean eval, iid noisy eval, v4-lite noisy eval
- 输出：by-seed、by-scenario、by-horizon、clean replay cost、clean-to-noisy degradation

### 6.2 Phase-1B 条件扩展

状态：未执行。  
触发条件：

- 只有 Phase-1A 显示 `v4-lite` 改变模型排序、退化规律或 seed failure 模式时才执行。

候选扩展：

- `phnode_qforce`
- `se3_accel_blackbox`
- `se3_momentum_blackbox`
- `heading bias`
- `degraded_eval`

### 6.3 P2-1：mass prior 与 lift 机制实验

状态：未执行。  
计划来源：

- [docs/oc_followup_experiment_plan.md](docs/oc_followup_experiment_plan.md)

计划内容：

- 2x2 组合：`phnode_full`, `ablate_no_mass_prior`, `ablate_no_lift`, `ablate_no_mass_prior_no_lift`
- clean 与 noisy `nominal_train` 都跑
- seeds：`42-47`

当前缺口：

- `ablate_no_mass_prior_no_lift` 尚未见正式模型/结果。

### 6.4 P2-2：noisy schedule 小范围扫描

状态：未执行。  
计划内容：

- 只跑 `phnode_full`
- seeds：`42,46`
- schedule：默认、保守、激进三组

目的：

- 判断 full PHNODE 的 bad-seed 问题是否主要来自优化路径，而不是模型结构本身。

### 6.5 P3 扩展线

状态：未执行。  
计划内容：

- `remus100_ins` 扩展线
- `noc` 对照线
- current-representation uncertainty
- control / maneuver OOD
- vehicle-parameter regime shift
- actuator mismatch

### 6.6 更远期扩展

状态：计划/设计阶段。  
包括：

- `current-unobservable`
- receding-horizon benchmark
- 真实日志离线 replay
- `v4-B1` history-aware clean block prediction

这些内容当前不能写成已完成实验，也不能用于支持当前论文主结论。

## 7. 当前正式可引用结论

> **2026-05-13 修订说明**：2026-05-12 完成的 provenance audit（详见
> [docs/provenance_audit_phnode_full_clean.md](docs/provenance_audit_phnode_full_clean.md)）确认 catalog 时代
> `main/phnode_full clean seed42/46` 灾难性训练发散是**与云端环境非确定性耦合的偶然事件**，不是模型/代码缺陷。
> 在当前 main HEAD（`7643dc9`, g3_5_7 镜像环境）下，seed46 重跑收敛到 best_loss=4.05e-03 / 60s rollout
> pos_err_median=0.4558 m，与 cleanrun v1 C46 浮点 bit-identical，fragility 不复现。
> 受影响结论按下表标注：

| 状态标签 | 说明 |
| --- | --- |
| **current** | 不受 fragility audit 影响，继续有效 |
| **stale_environment_drift** | 直接依赖 catalog 时代 seed42/46 fragility 的结论，证据链断裂，不再可引用 |
| **needs_recheck** | 部分依赖 fragility，但有独立证据支持，需在 cleanrun v1 ≡ current main 基线下重新背书 |

1. 在 `oc` clean setting 下，`baseline/phnode_qforce` 是当前 all-seed 最强整体模型。 **[current]**
2. 在 PHNODE family 内，clean all-seed 下 `ablation/ablate_no_lift` 当前最稳。 **[needs_recheck]** — catalog `ablate_no_lift seed43 clean` 也存在异常（best_epoch=19, best_loss=0.22, 60s ≈ 44 m），与 seed46 fragility 同环境，应在 cleanrun v1 ≡ current main 重训后重新判断。
3. ~~`main/phnode_full` 有强 stable cluster，但存在真实 bad-outlier seeds：`42` 与 `46`。~~ **[stale_environment_drift]** — seed42/46 outlier 是 catalog 时代云端 g3_5_5 镜像环境（PyTorch/CUDA/cuDNN 版本未记录）下的偶然训练发散；在 cleanrun v1（g3_5_7 镜像）与 Phase 3 audit（current main on g3_5_7）下两 seed 均收敛正常，不应继续作为模型脆弱性论据。
4. noisy training 不是普适增强；它与模型结构强耦合。 **[needs_recheck]** — 该结论原本依赖 §7.5 「noisy training 修复 seed46」，§7.5 stale 后需以 ablate_no_lift seed44 / ablate_no_mass_prior 收益等独立证据重新背书。
5. ~~noisy training 对 `phnode_full` 的主要收益是修复 `seed46`，不是普遍降低全部 seed 的误差。~~ **[stale_environment_drift]** — 「待修复的 seed46 脆弱性」本身已被 audit 推翻；该因果链不再成立。
6. `ablate_no_mass_prior` 是当前最稳定受益于 noisy training 的结构模型。 **[current]** — 独立证据，不依赖 phnode_full clean fragility。
7. `ablate_bu_only` 的退化是结构性的，actuation-conditioning 相关结构应保留。 **[current]**
8. coupled damping 有明确价值。 **[current]**
9. 当前 mass prior 尚未显示出不可替代性。 **[current]**
10. `v4-lite` 目前只能写成已实现并通过 smoke/probe 的协议方向，不能写成已完成正式决策实验。 **[current]**

### 7.A 修订后的 fragility 表述

- catalog `phnode_full clean seed42/46` 60s rollout 11 m 5-seed mean **不应作为模型脆弱性证据**。
- 同口径锁定（clean+clean, 60s, 5-seed mean of pos_err_median）下，正确基线是 cleanrun v1 ≡ current main = **0.6767 m**。
- 若需引用 catalog 时代该数据，必须同时引用 [docs/provenance_audit_phnode_full_clean.md](docs/provenance_audit_phnode_full_clean.md) 作为环境耦合说明。
- WP-Frag（在新基线下重训 catalog §12 矩阵）作为可选工单留待用户决策启动；不启动也不影响调查闭环。

## 8. 长期跟踪办法

当前已采用两层跟踪：

1. 继续维护本文档，作为人工进展摘要。
2. 维护机器可读状态表：[analysis/experiment_progress_log.csv](analysis/experiment_progress_log.csv)。

机器可读状态表字段为：

   - `item_id`
   - `title`
   - `category`
   - `status`
   - `planned_doc`
   - `result_doc`
   - `checkpoint_or_catalog_path`
   - `last_updated_at`
   - `notes`

这样做的好处是：

- Markdown 负责解释上下文和结论；
- CSV 负责排序、筛选、生成待办清单；
- 不需要手工改 `analysis/oc_data_catalog/*.csv` 这类生成产物。

## 9. 更新记录

| 时间 | 更新 |
|---|---|
| 2026-04-25 11:44:54 CST (+0800) | 创建初版，完成 README、docs、catalog、checkpoint report 状态梳理。 |
| 2026-04-25 11:50:10 CST (+0800) | 新增机器可读进展表 `analysis/experiment_progress_log.csv`，并将长期跟踪方案落地为 Markdown + CSV 双层结构。 |
| 2026-05-13 CST (+0800) | 完成 phnode_full clean provenance audit（详见 [docs/provenance_audit_phnode_full_clean.md](docs/provenance_audit_phnode_full_clean.md)）。§7 受影响结论按 stale_environment_drift / needs_recheck / current 三档标注。catalog 时代 seed42/46 fragility 不再可作为模型脆弱性引用，新基线为 cleanrun v1 ≡ current main = 0.6767 m。 |
