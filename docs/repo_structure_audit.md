# 仓库结构审计与当前使用边界

本文档用于说明当前仓库中哪些内容是主线入口、哪些是生成产物、哪些只保留作流程验证或历史参考。目标是降低后续维护时的认知负担，避免误把旧实验、smoke/probe 结果或废弃接口当作当前结论依据。

本文档只定义逻辑状态，不代表已经物理删除任何文件。

> 体积、OneDrive 同步状态与 `checkpoints/` 的逐 run 保留级别见 §6 与 `docs/checkpoints_retention_manifest.csv`。

## 1. 当前主线

当前仓库的主流程是：

```text
REMUS100 simulation
  -> dataset generation
  -> model training
  -> rollout benchmark
  -> OC result catalog
```

对应入口为：

- `remus100_core.py`
- `data_collection.py`
- `train_auv_hamnode.py`
- `train_utils.py`
- `evaluate_rollout_benchmark.py`
- `rollout_benchmark_engine.py`
- `rollout_benchmark_reporting.py`
- `scripts/build_oc_data_catalog.py`

当前研究计划入口为：

- `EXPERIMENT_PROGRESS_TRACKER.md`
- `docs/experiment_stages_overview.md`
- `docs/phnode_realistic_validation_plan.md`
- `docs/phnode_realistic_validation_execution_plan.md`
- `docs/phase1_realistic_validation_plan.md`
- `docs/phase1a_oc_v4lite_cleanrun_v1_report.md`
- `docs/provenance_audit_phnode_full_clean.md`

当前 OC 结果查询入口为：

- `analysis/oc_data_catalog/canonical_run_inventory.csv`
- `analysis/oc_data_catalog/run_inventory.csv`
- `analysis/oc_data_catalog/rollout_run_registry.csv`
- `analysis/oc_data_catalog/canonical_rollout_summary_long.csv`
- `analysis/oc_data_catalog/canonical_rollout_outcomes_long.csv`
- `analysis/oc_data_catalog/evidence_status_overrides.csv`

## 2. 状态分类

### 2.1 `active`

当前仍然作为代码主线维护的文件：

- `AUVHamNODE.py`
- `auv_baselines.py`
- `auv_model_registry.py`
- `remus100_core.py`
- `data_collection.py`
- `train_auv_hamnode.py`
- `train_utils.py`
- `evaluate_rollout_benchmark.py`
- `rollout_benchmark_engine.py`
- `rollout_benchmark_reporting.py`

这些文件定义当前模型、数据、训练、评估和结果导出的实际行为。修改这些文件后，应按影响范围运行最小验证流程。

### 2.2 `active-script`

当前推荐使用的脚本：

- `scripts/train_all_models_noise_profile.sh`
- `scripts/eval_all_models_noise_profile.sh`
- `scripts/summarize_sweep.py`
- `scripts/build_experiment_report.py`
- `scripts/build_oc_data_catalog.py`
- `scripts/query_oc_catalog_examples.py`
- `scripts/oc_catalog_templates.py`

这些脚本是当前 profile-based noisy-IC 工作流和 catalog 工作流的优先入口。

### 2.3 `generated-keep`

生成产物，但当前仍需保留：

- `data/`
- `analysis/oc_data_catalog/`

`data/` 下的数据集暂时全部保留。`analysis/oc_data_catalog/` 是当前结果查询和默认绘图/表格的主要来源。不要手动编辑生成型 catalog CSV；如需变更，应修改生成脚本后重新运行 `scripts/build_oc_data_catalog.py`。人工 sidecar 例外是 `analysis/oc_data_catalog/evidence_status_overrides.csv`，用于标注 provenance audit 后的证据状态。

### 2.4 `investigation-records`

以下文件或目录是人工维护的进展、审计和证据解释层，不是随机产物：

- `EXPERIMENT_PROGRESS_TRACKER.md`
- `analysis/experiment_progress_log.csv`
- `analysis/provenance_audit/`
- `docs/experiment_stages_overview.md`
- `docs/provenance_audit_phnode_full_clean.md`
- `docs/phase1a_oc_v4lite_cleanrun_v1_report.md`

这些内容用于判断哪些结果仍可引用、哪些结论已被标记为 `stale_environment_drift` 或 `needs_recheck`。特别是 catalog 时代 `phnode_full clean seed42/46` 结果已被 provenance audit 判定为环境耦合的历史异常，不应继续作为模型脆弱性证据。

### 2.5 `evidence-bearing-checkpoints`

以下 checkpoint/sweep 目录包含当前报告、follow-up 或 catalog 仍会引用的正式实验结果：

- `checkpoints/sweep_oc_all/`
- `checkpoints/sweep_oc_all_noise/`
- `checkpoints/sweep_oc_main_noise_nominal_train_remus100_dr_extra_42-46-47/`
- `checkpoints/sweep_oc_key_ablation_noise_nominal_train_remus100_dr_extra_42-46-47/`

这些目录不应随意删除。若后续要移动，应同步更新相关文档、catalog 或明确保留 catalog 为历史快照。

Phase-1A cleanrun v1 相关的 `checkpoints/sweep_oc_phase1a_*_phase1a_oc_v4lite_cleanrun_v1/` 目录属于协议敏感性 decision package。使用时必须同时查看 `docs/phase1a_oc_v4lite_cleanrun_v1_report.md` 中关于 `ablate_no_lift seed43 clean` 和 provenance 限制的说明，不能直接当作 canonical catalog 的替代。

**2026-08-20 补充——论文正文的主证据不在上列 A 区四个 sweep 里。** 按 `docs/section8_evidence_merge_plan.md` 的 R-A 口径，重叠实验一律取可逐位复现的 B 区，A 区只补 B 区未覆盖的单元（`ablate_diag_damping`、`ablate_bu_only`、黑箱噪声训练线）。因此 evidence-bearing 目录实际分两区：

- **B 区（§8 主证据，120 run / 7.87 GB）**：`checkpoints/sweep_oc_phase1a_decision_*_t2_wpfrag_*/`（clean 7 模型 × 5 种子 + iid/v4lite 各 4 模型 × 5 种子 = 75 run）、`checkpoints/sweep_oc_phase1a_smoke3_*_phase1a_oc_v4lite_cleanrun_v1/`（27 run）、`checkpoints/sweep_oc_phase1a_decision_extra43-45_*_phase1a_oc_v4lite_cleanrun_v1/`（18 run）。
- **A 区（catalog 时代，84 run / 4.58 GB）**：上列四个 sweep，对应 catalog 98 run 中除 smoke/other 桶之外的部分。

逐 run 的保留级别见 `docs/checkpoints_retention_manifest.csv`。

### 2.6 `flow-validation-only`

以下目录只用于证明流程、协议或 smoke/probe 实验跑通，不应作为当前论文结论或模型排序依据：

- `checkpoints/smoke_v4lite/`
- `checkpoints/sweep_oc_smoke/`
- `checkpoints/sweep_oc_phase1_probe_clean_20260423_180908/`
- `checkpoints/sweep_oc_phase1_probe_iid_20260423_180908/`
- `checkpoints/sweep_oc_phase1_probe_iideval_20260424_024232/`
- `checkpoints/sweep_oc_phase1_probe_v4lite_20260423_180908/`
- `checkpoints/sweep_oc_phase1_smoke_clean_fix_20260423_124711/`
- `checkpoints/sweep_oc_phase1_smoke_matched_20260423_173332/`
- `checkpoints/sweep_oc_main_noise_seed42_smoke/`

这些原始产物可在未来物理清理时删除，但删除前应先确认是否已有最终记录文档。若删除后仍保留现有 `analysis/oc_data_catalog/`，catalog 中部分 `source_file` 会成为历史路径，而不是当前磁盘上可直接打开的文件。

**不要按名字里的 `smoke` 判断保留级别。** `checkpoints/sweep_oc_phase1a_smoke3_*_phase1a_oc_v4lite_cleanrun_v1/` 名字带 smoke，实际持有 cleanrun v1 决策包 45 个 run 中的 27 个物理 checkpoint（`docs/experiment_full_inventory_zh.md` §B.2 / §B.6），属 evidence-bearing。真正的 flow-validation 只有 `smoke1`（单模型单种子）、`smoke_v4lite/`（ep=3 纯 code smoke）与上列 probe/smoke 目录，合计 42 run / 1.27 GB。判定一律以 `docs/checkpoints_retention_manifest.csv` 的 `retention_class` 为准。

### 2.7 `deprecated`

旧接口或兼容脚本，当前不再推荐运行：

- `scripts/train_all_models_noise.sh`
- `scripts/eval_all_models_noise.sh`
- 训练 CLI 的 `--noise_level` 接口

这些内容保留是为了读旧配置和理解历史结果。新实验应使用 `--noise_profile` 和 `scripts/train_all_models_noise_profile.sh` / `scripts/eval_all_models_noise_profile.sh`。

### 2.8 `delete-candidate`

用户已确认不再使用，但本文档阶段不直接删除：

- `checkpoints/unused/`
- `original/bf3n/`

`checkpoints/unused/` 中的实验使用旧版且有错误的噪声设计，不应作为当前证据。`original/bf3n/` 已不再作为参考实现使用。若后续目标转为物理瘦身，这两处是优先清理对象。

### 2.9 `notebook`

`notebook/` 暂时保留：

- 带结果的 notebook 是历史执行记录。
- 不带结果的 notebook 可能是未执行或需要修改后执行的 workflow 草稿。

因此现阶段不归档、不删除。

### 2.10 `docs`

`docs/` 暂时全部保留。当前计划入口是：

- `docs/phnode_realistic_validation_plan.md`
- `docs/phnode_realistic_validation_execution_plan.md`
- `docs/phase1_realistic_validation_plan.md`

旧版和修订版噪声设计文档仍有研究演化记录价值。后续如需整理，优先增加索引和状态说明，而不是删除内容。

### 2.11 `not-in-git-but-unrecoverable`

体积很小但无法重建、因此必须进版本控制的产物：

- `analysis/section8_current_evidence/`（16 文件 / 2.9 MB）——论文正文每个数字与每张图的来源；`paper/drafts/figures/make_section8_*.py` 直接读其中的 `figure_data/*.csv`。**2026-08-20 起已纳入 git**（`.gitignore` 中以 `!analysis/section8_current_evidence/` 放行）。此前它被 `analysis/*` 规则挡在版本控制外，而重建它需要多 GB 的 `checkpoints/sweep_oc_phase1a_decision_*`，实际不可重建。

以下仍在版本控制之外，属待定决策项：

- `exports/phnode_full_oc_clean/`——自包含的 `phnode_full` OC clean（seed 45）可移植导出包，含 README / requirements / tests。被 `.gitignore` 的 `exports/` 规则忽略，目前只存在于 OneDrive 一处。若打算对外分发或作为论文补充材料，应单独入库或独立成仓库。

## 3. 推荐阅读顺序

如果目标是快速理解当前项目：

1. `README.md`
2. `EXPERIMENT_PROGRESS_TRACKER.md`
3. `docs/repo_structure_audit.md`
4. `docs/experiment_stages_overview.md`
5. `docs/provenance_audit_phnode_full_clean.md`
6. `docs/phase1a_oc_v4lite_cleanrun_v1_report.md`
7. `docs/phnode_realistic_validation_plan.md`
8. `docs/phnode_realistic_validation_execution_plan.md`
9. `docs/phase1_realistic_validation_plan.md`
10. `docs/noise_model_design.md`
11. `docs/oc_data_catalog_dictionary.md`
12. `docs/oc_result_selection_policy.md`

如果目标是查当前 OC 结果：

1. `analysis/oc_data_catalog/canonical_run_inventory.csv`
2. `analysis/oc_data_catalog/rollout_run_registry.csv`
3. `analysis/oc_data_catalog/canonical_rollout_summary_long.csv`
4. `analysis/oc_data_catalog/canonical_rollout_outcomes_long.csv`
5. `analysis/oc_data_catalog/evidence_status_overrides.csv`

## 4. 维护规则

- 不要从随机 checkpoint 目录开始理解实验结果；先看 catalog 和文档。
- 不要忽略 `evidence_status`；canonical 只说明 rollout 选择优先级，不自动证明该行仍是 current evidence。
- 不要把 catalog 时代 `phnode_full clean seed42/46` 的 stale run 写成当前模型脆弱性证据。
- 不要把 `flow-validation-only` 的 smoke/probe 结果写进正式结论。
- 不要把 `checkpoints/unused/` 里的旧噪声实验作为当前证据。
- 不要手动编辑 `data/`、`checkpoints/` 或生成型 `analysis/oc_data_catalog/*.csv`；`evidence_status_overrides.csv` 是人工 sidecar 例外。
- 新实验优先使用 profile-based noisy-IC 脚本。
- 新的 evidence-bearing run 应落盘 `_audit_meta/code_revision.txt` 与 `_audit_meta/environment.txt`。
- 物理删除大目录前，先确认是否需要同步重建 catalog 或保留 catalog 为历史快照。

## 5. 后续整理建议

低风险优先级：

1. 在入口文档中引用本文档，降低新读者误入旧产物的概率。
2. 给 deprecated 脚本加明显提示，但暂不移除。
3. 给 smoke/probe 目录保留最终说明文档后，再决定是否物理删除原始产物。
4. 若需要代码结构重构，优先拆分 `train_utils.py`，并抽出共享的 state layout / state conversion 工具。

## 6. 存储层现状（2026-08-20 实测）

`du` 在本仓库会严重低估体积：OneDrive 按需下载的文件在本地占 0 块，但逻辑大小照算。实测 43,725 个文件、16.35 GB，其中 42,977 个文件 / 14.87 GB 是**未下载的云端占位符**，本地实际下载的只有 1.48 GB（且其中 1.45 GB 是 `analysis/oc_data_catalog/` 的三个大 CSV）。

| 目录 | 逻辑体积 | 文件数 | 说明 |
|---|---:|---:|---|
| `checkpoints/` | 14.0 GB | 43,120 | 见下表 |
| `data/` | 795 MB | 47 | 3 个 `.pkl` 数据集，全部为云端占位符 |
| `analysis/oc_data_catalog/` | 1.45 GB | 18 | 生成产物；`training_history_long.csv` 550 MB、`rollout_summary_long.csv` 478 MB、`canonical_rollout_summary_long.csv` 425 MB |
| `paper/` | 26 MB | — | 含一个 16 MB 的 `velocity_state_contract.tiff`（已被 `.gitignore` 排除） |
| tracked 工作面 | ~25 MB | 240 | 代码 + 文档 + 论文 + notebook，`git ls-files` 全集 |

`checkpoints/` 内部构成（按扩展名）：

| 类型 | 文件数 | 体积 | 性质 |
|---|---:|---:|---|
| `.png` | 8,655 | 4.85 GB | rollout 评估绘图，**可从同目录 CSV 再生** |
| `.pkl` | 20,020 | 4.72 GB | rollout 逐轨迹转储 + `training_history.pkl` |
| `.csv` | 8,262 | 3.30 GB | rollout 指标表，绘图的数据源 |
| `.pt` | 1,690 | 1.20 GB | `best_model.pt`（无中间 epoch 检查点） |
| 其余 | 4,493 | 0.15 GB | `config.json` / 评估 JSON / 日志 / `runs.tsv` |

按保留级别（`docs/checkpoints_retention_manifest.csv`，300 个 run 目录全覆盖）：

| retention_class | run 数 | 体积 |
|---|---:|---:|
| `evidence-bearing-B` | 120 | 7.87 GB |
| `evidence-bearing-A` | 84 | 4.58 GB |
| `flow-validation-only` | 42 | 1.27 GB |
| `delete-candidate` | 54 | 0.25 GB |

清单由 `scripts/build_checkpoints_retention_manifest.py` 生成，可重跑；`referenced_by` 列记录哪些 tracked 文档提到了该 run，删除前应先看这一列。

### 6.1 已识别的可清理项

**rollout 绘图（4.84 GB，已列清单，未执行）。** `checkpoints/**/rollout_benchmark/**/*.png` 共 8,655 张，其中 8,625 张可删——它们由同一 profile 目录下的 `.csv` / `summary.json` 画出，数据本身不动；全仓库没有任何代码读取 PNG；论文插图走的是 `analysis/section8_current_evidence/figure_data/` → `paper/drafts/figures/make_*.py` 这条独立链路。逐文件清单见 `docs/checkpoints_png_purge_manifest.csv`（`scripts/build_checkpoints_png_purge_manifest.py` 生成）。保留的 30 张分三类：不在 `rollout_benchmark/` 下的旧式布局、被 `docs/oc_model_evaluation_overview.md` 链接的、以及所在 profile 目录没有任何 metrics 文件（图是仅存产物）的。

**中间 epoch 检查点。** `checkpoint_{50,100,150,200,250}.pt` 原有 1,390 个 / 0.97 GB。这类文件不可再生但 `best_model.pt` 已单独存在，只有续训或训练动力学分析才会用到。**2026-08-20 已删除 `checkpoints/unused/` 下的 270 个（0.20 GB）**——该目录整体为 `delete-candidate`；其余 1,120 个（约 0.77 GB，主要在 `sweep_oc_all`、`sweep_oc_all_noise`、`smoke3_*`）保持原状。

**同步层注意事项**：

- 重建 `analysis/oc_data_catalog/` 需要把 `checkpoints/` 全量拉回本地（`scripts/build_oc_data_catalog.py` 走 `rglob("runs.tsv")` 扫描），代价是 14 GB 的按需下载，非必要不要触发。
- 删除 OneDrive 占位符是真删除，会同步到云端，没有本地回收站兜底。
- 不要在 OneDrive 路径下编译 LaTeX：`paper/drafts/` 曾出现 `auvhamnode_thesis_chapter_zh-A Mac mini.{aux,out,toc}` 这类设备名冲突副本，来源是两台机器同时写 LaTeX 中间产物。
