# 仓库结构审计与当前使用边界

本文档用于说明当前仓库中哪些内容是主线入口、哪些是生成产物、哪些只保留作流程验证或历史参考。目标是降低后续维护时的认知负担，避免误把旧实验、smoke/probe 结果或废弃接口当作当前结论依据。

本文档只定义逻辑状态，不代表已经物理删除任何文件。

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

- `docs/phnode_realistic_validation_plan.md`
- `docs/phnode_realistic_validation_execution_plan.md`
- `docs/phase1_realistic_validation_plan.md`

当前 OC 结果查询入口为：

- `analysis/oc_data_catalog/run_inventory.csv`
- `analysis/oc_data_catalog/rollout_run_registry.csv`
- `analysis/oc_data_catalog/canonical_rollout_summary_long.csv`
- `analysis/oc_data_catalog/canonical_rollout_outcomes_long.csv`

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

`data/` 下的数据集暂时全部保留。`analysis/oc_data_catalog/` 是当前结果查询和默认绘图/表格的主要来源。不要手动编辑 catalog CSV；如需变更，应修改生成脚本后重新运行 `scripts/build_oc_data_catalog.py`。

### 2.4 `evidence-bearing-checkpoints`

以下 checkpoint/sweep 目录包含当前报告、follow-up 或 catalog 仍会引用的正式实验结果：

- `checkpoints/sweep_oc_all/`
- `checkpoints/sweep_oc_all_noise/`
- `checkpoints/sweep_oc_main_noise_nominal_train_remus100_dr_extra_42-46-47/`
- `checkpoints/sweep_oc_key_ablation_noise_nominal_train_remus100_dr_extra_42-46-47/`

这些目录不应随意删除。若后续要移动，应同步更新相关文档、catalog 或明确保留 catalog 为历史快照。

### 2.5 `flow-validation-only`

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

### 2.6 `deprecated`

旧接口或兼容脚本，当前不再推荐运行：

- `scripts/train_all_models_noise.sh`
- `scripts/eval_all_models_noise.sh`
- 训练 CLI 的 `--noise_level` 接口

这些内容保留是为了读旧配置和理解历史结果。新实验应使用 `--noise_profile` 和 `scripts/train_all_models_noise_profile.sh` / `scripts/eval_all_models_noise_profile.sh`。

### 2.7 `delete-candidate`

用户已确认不再使用，但本文档阶段不直接删除：

- `checkpoints/unused/`
- `original/bf3n/`

`checkpoints/unused/` 中的实验使用旧版且有错误的噪声设计，不应作为当前证据。`original/bf3n/` 已不再作为参考实现使用。若后续目标转为物理瘦身，这两处是优先清理对象。

### 2.8 `notebook`

`notebook/` 暂时保留：

- 带结果的 notebook 是历史执行记录。
- 不带结果的 notebook 可能是未执行或需要修改后执行的 workflow 草稿。

因此现阶段不归档、不删除。

### 2.9 `docs`

`docs/` 暂时全部保留。当前计划入口是：

- `docs/phnode_realistic_validation_plan.md`
- `docs/phnode_realistic_validation_execution_plan.md`
- `docs/phase1_realistic_validation_plan.md`

旧版和修订版噪声设计文档仍有研究演化记录价值。后续如需整理，优先增加索引和状态说明，而不是删除内容。

## 3. 推荐阅读顺序

如果目标是快速理解当前项目：

1. `README.md`
2. `docs/repo_structure_audit.md`
3. `docs/phnode_realistic_validation_plan.md`
4. `docs/phnode_realistic_validation_execution_plan.md`
5. `docs/phase1_realistic_validation_plan.md`
6. `docs/noise_model_design.md`
7. `docs/oc_data_catalog_dictionary.md`
8. `docs/oc_result_selection_policy.md`

如果目标是查当前 OC 结果：

1. `analysis/oc_data_catalog/canonical_run_inventory.csv`
2. `analysis/oc_data_catalog/rollout_run_registry.csv`
3. `analysis/oc_data_catalog/canonical_rollout_summary_long.csv`
4. `analysis/oc_data_catalog/canonical_rollout_outcomes_long.csv`

## 4. 维护规则

- 不要从随机 checkpoint 目录开始理解实验结果；先看 catalog 和文档。
- 不要把 `flow-validation-only` 的 smoke/probe 结果写进正式结论。
- 不要把 `checkpoints/unused/` 里的旧噪声实验作为当前证据。
- 不要手动编辑 `data/`、`checkpoints/` 或 `analysis/oc_data_catalog/*.csv`。
- 新实验优先使用 profile-based noisy-IC 脚本。
- 物理删除大目录前，先确认是否需要同步重建 catalog 或保留 catalog 为历史快照。

## 5. 后续整理建议

低风险优先级：

1. 在入口文档中引用本文档，降低新读者误入旧产物的概率。
2. 给 deprecated 脚本加明显提示，但暂不移除。
3. 给 smoke/probe 目录保留最终说明文档后，再决定是否物理删除原始产物。
4. 若需要代码结构重构，优先拆分 `train_utils.py`，并抽出共享的 state layout / state conversion 工具。
