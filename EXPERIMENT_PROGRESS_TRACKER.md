# AUV SE(3) 实验进展跟踪总表

生成时间：2026-04-25 11:44:54 CST (+0800)  
最近核对：2026-08-20 CST (+0800)
工作副本：`D:\Codes\g3_5_5`（论文写作、LaTeX 编译、代码改动与全部 git 操作）  
数据归档：`C:\Users\jinxiang\OneDrive\我的\Code\auv_se3node\g3_5_5`（`checkpoints/`、`data/`、`analysis/oc_data_catalog/` 等非版本控制产物）  
两副本分工见 [docs/repo_structure_audit.md](docs/repo_structure_audit.md) §6.2
状态口径：基于 `README.md`、`docs/` 与版本控制内的 `analysis/section8_current_evidence/`；涉及 `analysis/oc_data_catalog/` 与 `checkpoints/` 的条目以数据归档副本中可见的文件状态为准。

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

截至 2026-07-25 核对，catalog 中 `run_inventory.csv` 记录了 98 个 run：

- clean training：52 个
- noisy training：46 个
- `remus100_dr` noisy-reference run：53 个
- canonical rollout registry 已建立，`canonical_rollout_summary_long.csv` 与 `canonical_rollout_outcomes_long.csv` 已可用于默认论文图表和表格。

## 3. Markdown 文档分类

| 类别 | 文档 | 当前状态 |
|---|---|---|
| 仓库入口与边界 | [README.md](README.md), [docs/repo_structure_audit.md](docs/repo_structure_audit.md) | 当前权威入口 |
| 实验阶段总览 | [docs/experiment_stages_overview.md](docs/experiment_stages_overview.md) | 按云镜像+时间窗口梳理 catalog / cleanrun v1 / audit retrain / smoke-probe / unused 五阶段 |
| 总体研究计划 | [docs/phnode_realistic_validation_plan.md](docs/phnode_realistic_validation_plan.md) | 研究计划，不是执行记录 |
| 现实导向执行方案 | [docs/phnode_realistic_validation_execution_plan.md](docs/phnode_realistic_validation_execution_plan.md), [docs/phase1_realistic_validation_plan.md](docs/phase1_realistic_validation_plan.md) | 当前 P0-P1 / Phase-1A 执行方案 |
| OC 补充实验计划 | [docs/oc_followup_experiment_plan.md](docs/oc_followup_experiment_plan.md) | P1 已执行并报告；P2/P3 尚未完成 |
| OC 正式结果报告 | [docs/oc_experiments_comprehensive_report.md](docs/oc_experiments_comprehensive_report.md), [docs/oc_followup_results_p1_p2.md](docs/oc_followup_results_p1_p2.md), [docs/oc_model_evaluation_overview.md](docs/oc_model_evaluation_overview.md), [docs/oc_results_section_zh.md](docs/oc_results_section_zh.md) | 已完成并有正式报告 |
| 噪声设计与协议 | [docs/noise_model_design.md](docs/noise_model_design.md), [docs/noise_design_v3_remus100_reference_grounded.md](docs/noise_design_v3_remus100_reference_grounded.md), [docs/noise_design_v4_dr_ekf_output.md](docs/noise_design_v4_dr_ekf_output.md), [docs/noise_design_v4_lite_traj_consistent_ic.md](docs/noise_design_v4_lite_traj_consistent_ic.md), [docs/v4_lite_protocol_spec.md](docs/v4_lite_protocol_spec.md), [docs/v4_b1_implementation_checklist.md](docs/v4_b1_implementation_checklist.md) | profile-based 与 `v4_lite` 已落代码；`v4_lite` 决策实验已闭环，`v4-B1` 仍为远期设计 |
| 噪声运行手册 | [docs/noise_experiment_runbook.md](docs/noise_experiment_runbook.md), [docs/noise_cli_parameter_reference.md](docs/noise_cli_parameter_reference.md), [docs/noise_cli_command_templates.md](docs/noise_cli_command_templates.md) | 操作说明，不是结果报告 |
| Catalog 体系 | [docs/oc_data_catalog_plan.md](docs/oc_data_catalog_plan.md), [docs/oc_data_catalog_dictionary.md](docs/oc_data_catalog_dictionary.md), [docs/oc_result_selection_policy.md](docs/oc_result_selection_policy.md), [docs/oc_catalog_template_usage.md](docs/oc_catalog_template_usage.md) | 已落地 |
| 存储与保留清单 | [docs/repo_structure_audit.md](docs/repo_structure_audit.md) §6, [docs/checkpoints_retention_manifest.csv](docs/checkpoints_retention_manifest.csv), [docs/checkpoints_png_purge_manifest.csv](docs/checkpoints_png_purge_manifest.csv) | 2026-08-20 建立；生成产物，改 `checkpoints/` 前先查 `retention_class` 与 `referenced_by` |
| 论文稿件清单 | [paper/README.md](paper/README.md), [paper/drafts/INDEX.md](paper/drafts/INDEX.md) | 写作决策与进度看板在前者，`paper/drafts/` 的活稿与历史记录分类在后者 |
| 历史归档 | [docs/unused/](docs/unused/) | 旧方案/旧审查，不作为当前主线依据 |

## 4. 已完成并已有正式报告记录

### 4.1 Catalog 时代 Clean OC 主实验

状态：已完成并报告；以下排名只描述 catalog 时代结果，不作为当前论文 §8 的默认排名。
主要记录：

- [docs/oc_model_evaluation_overview.md](docs/oc_model_evaluation_overview.md)
- [docs/oc_results_section_zh.md](docs/oc_results_section_zh.md)
- [docs/oc_experiments_comprehensive_report.md](docs/oc_experiments_comprehensive_report.md)

历史 catalog 结论及其当前边界：

- catalog clean all-seed 下，`baseline/phnode_qforce` 曾是表内最强整体模型；当前论文的重叠单元改用可复现的 B 区证据，不能把该排名写成 current headline。
- catalog clean 下，`ablation/ablate_no_lift` 曾是最强 PHNODE family；当前 B 区存在可复现的 clean seed43 训练失败，不能写成“all-seed 最稳”。
- `main/phnode_full` 的 seed42/46 catalog 异常已证实为 `stale_environment_drift`，不再作为模型脆弱性证据。
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

## 5. 早期流程验证与尚未形成正式结果的接口

本节保留早期 smoke/probe 与接口状态。smoke/probe 本身仍不可作为正式研究结论；后续完成的 T2/Phase-1A 决策套件另见 §6.1 与 §7.B。

### 5.1 `v4-lite` 早期实现与 smoke/probe

状态：代码已接入，早期 smoke/probe 已跑过；这些早期目录仍为 `flow-validation-only`。正式五 seed 决策后来已由 T2/Phase-1A 套件完成，见 §6.1。
证据：

- 代码已支持 `noise_protocol=v4_lite`。
- 存在 `checkpoints/smoke_v4lite/`。
- 存在 `checkpoints/sweep_oc_phase1_probe_v4lite_20260423_180908/`。
- 这些目录下有 `experiment_report.md` 与 `phase1_summary.csv`。

早期产物的限制：

- 这些被 [docs/repo_structure_audit.md](docs/repo_structure_audit.md) 标为 `flow-validation-only`。
- 它们只适合作为协议和流程验证，不适合作为论文结论或模型排序依据；论文使用的是后续定向导出的 current evidence。

### 5.2 Phase-1 smoke/probe suites

状态：已跑过若干小规模流程验证；这些目录本身不形成正式 Phase-1A 决策结论，后续正式决策套件另见 §6.1。
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

## 6. 已闭环决策包与尚未执行的扩展

### 6.1 Phase-1A 正式五 seed 决策实验

状态：**已完成并闭环**。早期 `cleanrun_v1` 完成三模型矩阵，后续 `t2_wpfrag` 补齐 `phnode_qforce` 与三类黑箱 clean 对照；论文使用 `analysis/section8_current_evidence/` 的定向导出，不把 Phase-1A suites 并入 canonical catalog。
计划来源：

- [docs/phase1_realistic_validation_plan.md](docs/phase1_realistic_validation_plan.md)
- [docs/phnode_realistic_validation_execution_plan.md](docs/phnode_realistic_validation_execution_plan.md)

已完成矩阵：

- `oc + known-current surrogate`
- 噪声训练协议模型：`phnode_full`, `phnode_qforce`, `ablate_no_mass_prior`, `ablate_no_lift`
- clean-only 黑箱对照：`blackbox_fullstate`, `se3_accel_blackbox`, `se3_momentum_blackbox`
- seeds：`42,43,44,45,46`
- 训练协议：`clean`, `iid_noisy_ic`, `v4_lite`
- 评估协议：clean eval, iid noisy eval, v4-lite noisy eval
- 输出：by-seed、by-scenario、by-horizon、clean replay cost、clean-to-noisy degradation，以及论文 §8 使用的 375 行 `per_seed_long.csv` 与 75 行 `aggregate.csv`
- 执行证据：`notebook/t2_wpfrag_*_completed.ipynb`、`notebook/phase1a_oc_v4lite_formal_workflow_completed.ipynb`

### 6.2 Phase-1B 条件扩展

状态：**决议不进入**，不是待执行阻塞项。
原触发条件：

- 只有 Phase-1A 显示 `v4-lite` 改变模型排序、退化规律或 seed failure 模式时才执行。

核对结果：

- 对完整模型与无质量先验消融，两条噪声训练线差异不超过约 5%；无升力耦合消融与配置广义力基线是明确例外。
- 三协议下模型的**分层排序**不变：完整模型/无升力耦合消融位于头部且两者次序可互换，无质量先验消融居中，配置广义力基线居后。
- `heading_biased_eval`、`degraded_eval` 和三类黑箱 clean 对照已由 T2/Path B 覆盖；黑箱 iid/v4lite 缺口按 §7.B 决议不补。
- 两项例外已被单列为协议敏感性/高方差现象，但没有推翻稳定主力模型的结论或整体分层；按 Phase-1A 最终判定，它们不足以触发全因子扩展，Phase-1B 不再作为待办。

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
| **superseded_by_current_evidence** | 历史 catalog/P1 结论已被同协议、同种子、可复现的 T2/B 区证据替代 |

1. 在当前 B 区 `oc + clean train + clean eval` 五 seed 证据中，`phnode_full` 是有限模型里中心精度最优且 5/5 无训练异常的模型：60 s 位置误差重复间均值为 **0.6767 m**。`phnode_qforce` 的同口径结果为 3.7564 m；“qforce 当前最强”只属于历史 catalog 排名，已被 current evidence 替代。 **[current]**
2. `ablate_no_lift` 的稳定簇中心为 **0.8288 m（N=4）**，与完整模型接近；但 clean seed43 出现 44.38 m、`nbad=276` 的可复现训练失败。因此它不能再写成“clean all-seed 最稳”，正确表述是“稳定簇精度接近完整模型，同时存在一个透明披露的可复现失效重复”。 **[current]**
3. ~~`main/phnode_full` 有强 stable cluster，但存在真实 bad-outlier seeds：`42` 与 `46`。~~ **[stale_environment_drift]** — seed42/46 outlier 是 catalog 时代云端 g3_5_5 镜像环境（PyTorch/CUDA/cuDNN 版本未记录）下的偶然训练发散；在 cleanrun v1（g3_5_7 镜像）与 Phase 3 audit（current main on g3_5_7）下两 seed 均收敛正常，不应继续作为模型脆弱性论据。
4. noisy training 不是普适增强，其效果与模型结构相关。当前 T2 clean-eval 证据中，完整模型、无升力耦合消融和无质量先验消融均未显示系统性净收益；较弱的配置广义力基线则从 clean 的 3.7564 m 改善到 iid 的 2.4713 m、v4lite 的 1.3551 m。 **[current]**
5. ~~noisy training 对 `phnode_full` 的主要收益是修复 `seed46`，不是普遍降低全部 seed 的误差。~~ **[stale_environment_drift]** — 「待修复的 seed46 脆弱性」本身已被 audit 推翻；该因果链不再成立。
6. ~~`ablate_no_mass_prior` 是当前最稳定受益于 noisy training 的结构模型。~~ **[superseded_by_current_evidence]** — matched T2 中其 clean/iid/v4lite clean-eval 结果分别为 1.2966/1.3471/1.2806 m，没有稳定的净改善；该历史 P1 结论不再作为论文 current claim。
7. `ablate_bu_only` 的退化是结构性的，actuation-conditioning 相关结构应保留。 **[current]**
8. coupled damping 有明确价值。 **[current]**
9. 当前 mass prior 尚未显示出不可替代性。 **[current]**
10. ~~`v4-lite` 目前只能写成已实现并通过 smoke/probe 的协议方向，不能写成已完成正式决策实验。~~ **[decided 2026-07-01；2026-07-25 收紧表述]** — 正式决策实验已由 B 区 `t2_wpfrag` 决策套件完成（4 结构化模型 × {clean, iid_noisy_ic, v4_lite} × 5 种子，含 rollout）。对完整模型与无质量先验消融，两条噪声训练线相对差不超过约 5%；无升力耦合消融与配置广义力基线分别有 26% 与 40% 的明确差异。因此可写“稳定主力模型上协议近似等价、分层排序不变”，不能泛化为四模型普遍等价，也不升级为「v4_lite 更优」。详见 §7.B 工单 B。

### 7.A 修订后的 fragility 表述

- catalog `phnode_full clean seed42/46` 60s rollout 11 m 5-seed mean **不应作为模型脆弱性证据**。
- 同口径锁定（clean+clean, 60s, 5-seed mean of pos_err_median）下，正确基线是 cleanrun v1 ≡ current main = **0.6767 m**。
- 若需引用 catalog 时代该数据，必须同时引用 [docs/provenance_audit_phnode_full_clean.md](docs/provenance_audit_phnode_full_clean.md) 作为环境耦合说明。
- WP-Frag（在新基线下重训 catalog §12 矩阵）**决议不启动**（2026-07-01）：实质等价工作已由 B 区 `t2_wpfrag` 决策套件在受控 g3_5_7 基线下完成，R-A 合并口径已规定重叠实验取可复现的 B 区、不再引用 A 区 catalog 漂移行。详见 §7.B 工单 A。

### 7.B 可选工单决议闭环（2026-07-01）

以下两项曾长期挂为「可选工单待用户决策」。经证据核对，**实质工作均已完成**、论文所需 claim 已被现有数据覆盖，现固定决议，不再回头调查是否启动。

**工单 A — WP-Frag（在新基线下重训 catalog §12 矩阵）：决议不启动。**

- 实质等价工作已由 B 区 `t2_wpfrag` 决策套件在受控 g3_5_7 clean 基线下完成：`decision_clean_t2_wpfrag`（7 模型 × 5 种子 = 35 run）+ `decision_iid`（4 模型 × 20 run）+ `decision_v4lite`（4 模型 × 20 run），均含 rollout；执行版 notebook `notebook/t2_wpfrag_*_completed.ipynb` 已入库。
- 已提交的 R-A 合并口径（`docs/section8_evidence_merge_plan.md`）规定：重叠实验取可逐位复现的 B 区，论文不再引用 A 区 catalog 漂移行——「重训旧 catalog」在证据上已被取代。
- 唯一残留是**可选**目录簿记：把 catalog `ablate_no_lift seed43 clean` 行标 `stale_environment_drift`（`docs/oc_data_catalog_dictionary.md:478`）。其科学问题已由 `notebook/t2supp_nolift_seedscan_completed.ipynb` 解答（真实可复现脆弱、与环境无关）。此簿记不烧算力、不影响论文，留作可选。
- 重启条件：仅当论文改口径、确需「同镜像纯净 A 区 catalog 数字」时——当前论文不需要。

**工单 B — v4-lite 正式决策实验：决议视为已完成，不再补跑、不升级 claim。**

- 决策数据已存在：B 区 `t2_wpfrag` 决策套件对 4 个结构化模型（`phnode_full` / `phnode_qforce` / `ablate_no_lift` / `ablate_no_mass_prior`）完成 {clean, iid_noisy_ic, v4_lite} × 5 种子，含 rollout。论文 §1.8 协议表（`tab:s8-protocol`）即基于此。
- 对完整模型与无质量先验消融，两条噪声训练线近似等价（相对差 ≤约 5%）；无升力耦合消融与配置广义力基线分别为 26% 与 40%，是显式例外。
- 三协议下模型的**分层排序**不变：完整模型/无升力耦合消融保持头部组但精确次序可互换，无质量先验消融居中，配置广义力基线居后。论文只在这一条件化范围内把噪声初值训练作为单一鲁棒性轴处理。
- 缺的 3 个黑箱 / SE(3) 基线的 iid/v4lite run **决议不补**：协议表本就声明黑箱仅有 clean、不参与噪声线比较。
- **不**把 v4_lite 升级为「确认更优的最终协议」：数据显示等价而非更优，`AGENTS.md` 论文边界亦禁止此断言。
- 连带闭环 §6.2 Phase-1B：虽然无升力耦合消融与配置广义力基线显示协议敏感性，但没有推翻稳定主力模型的结论或整体分层；按 `docs/phase1a_oc_v4lite_cleanrun_v1_report.md` 的最终判定，这些例外不足以触发全因子扩展，故 **Phase-1B 决议不进入**。

> 注：§5.1 只描述早期 flow-validation 目录；正式决策状态以 §6.1、§6.2 与本节为准。

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
| 2026-07-01 CST (+0800) | 固定两项可选工单决议（新增 §7.B）：**WP-Frag 决议不启动**（实质已由 B 区 `t2_wpfrag` 决策套件完成、R-A 口径取代旧 catalog）；**v4-lite 正式决策实验决议视为已完成**（4 结构化模型数据已入论文 §1.8 协议表，当时概括为“协议等价”，后于 2026-07-25 收紧为条件等价）、Phase-1B 不进入。同步更新结论 10（改标 decided）、结论 2（no_lift seed43 改 current 背书、去除重训依赖）、§7.A WP-Frag 行。另：`.claude/settings.local.json` 停止跟踪并加入 `.gitignore`（本机个人配置）。 |
| 2026-08-20 CST (+0800) | 仓库治理轮次（`01a3fd2`..`0f7be4e`），不涉及任何实验结论或数字口径变动：`analysis/section8_current_evidence/` 纳入版本控制，论文 §1.8 图数据链路进入 git；新增 `docs/checkpoints_retention_manifest.csv`（300 个 run 全覆盖，四级 `retention_class`）与 `docs/checkpoints_png_purge_manifest.csv`（8,655 张 rollout 绘图中 8,625 张可删，**清单已出、删除未执行**），生成脚本为 `scripts/build_checkpoints_retention_manifest.py` 与 `scripts/build_checkpoints_png_purge_manifest.py`；已删除 `checkpoints/unused/` 下 270 个中间 epoch 检查点（0.20 GB）；修复 17 份文档的路径漂移并新增 `paper/drafts/INDEX.md`；确立工作副本与数据归档两份本地副本的分工（见文首与 `docs/repo_structure_audit.md` §6.2）。 |
| 2026-07-25 CST (+0800) | 重新以 `analysis/section8_current_evidence/` 与执行版 notebook 核对全表：catalog inventory 更新为 98 run；Phase-1A 改为已闭环、Phase-1B 改为决议不进入；current clean headline 改为 `phnode_full=0.6767 m (N=5)`，`ablate_no_lift=0.8288 m (N=4)` 并披露 seed43；废止 qforce/no_lift 的历史“当前最强/最稳”表述及 no-mass noisy 稳定获益结论；协议结论收紧为“完整模型与无质量先验消融近似等价、四模型分层排序不变但存在两项明确例外”。同步机器可读进度表与论文 README。 |
