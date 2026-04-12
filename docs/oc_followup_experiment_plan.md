# OC 补充实验方案与执行清单

## 1. 目的

本文档把当前 `oc` 实验分析中最关键的未决问题，整理成一套可直接执行的补充实验方案。目标不是盲目扩大 sweep，而是用尽量少的新增计算，显著提高结论可信度。

当前最需要补强的结论有三类：

1. `main/phnode_full` 在 noisy training 下是否真的缓解了 clean sweep 中暴露的 bad-seed 模式。
2. clean-trained 与 noisy-trained 模型在 noisy rollout benchmark 下到底谁更强，是否存在严格 matched 的训练方式收益。
3. `mass prior` 与 `lift` 的现象究竟是结构问题，还是训练机制问题。

---

## 2. 总体原则

补充实验遵循三条原则：

1. 先跑会改变核心论文结论的实验。
2. 先补 strict comparison，再补机制解释。
3. 尽量复用现有脚本、现有数据集、现有评估口径，不重新引入新的变量。

统一约定：

- 数据集：`data/auv_oc_traj1000_blk150_s23_d0be9434.pkl`
- 训练噪声：`--noise_profile nominal_train`
- 噪声参考：`--noise_reference remus100_dr`
- rollout 模式：`resampled`
- rollout profile：`clean nominal_eval degraded_eval heading_biased_eval`
- 主比较指标：`60s final position error median`

---

## 3. 优先级划分

### P0：链路校验

先做单模型 smoke test，确认 noisy training 和 profile-aware evaluation 链路工作正常。

### P1：必须补齐

这是最重要的一层：

1. noisy sweep 补齐 `main/phnode_full` 的关键缺失 seeds
2. 对 clean-trained checkpoints 补跑 noisy rollout benchmark，形成 matched comparison

### P2：机制实验

在 P1 完成后再做：

1. `mass prior` 与 `lift` 的组合机制实验
2. noisy schedule 小范围扫描

### P3：扩展实验

只有在主结论已经稳定后再考虑：

1. `remus100_ins` 扩展线
2. `noc` 对照线

---

## 4. P0：单模型 smoke test

### 4.1 目的

确认以下事项：

- noisy training 参数已正确写入 `config.json`
- `noise_budgets.json` 正常生成
- `heldout_evaluation.json` 中包含 `clean / nominal_eval / degraded_eval / heading_biased_eval`
- rollout benchmark 输出中包含各 profile 独立 summary

### 4.2 推荐命令

训练：

```bash
bash scripts/train_all_models_noise_profile.sh \
  --profile oc \
  --group main \
  --seeds "42" \
  --suite-name sweep_oc_main_noise_seed42_smoke \
  --noise-reference remus100_dr
```

评估：

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/sweep_oc_main_noise_seed42_smoke
```

### 4.3 必查文件

- `checkpoints/sweep_oc_main_noise_seed42_smoke/main_phnode_full_seed42/config.json`
- `checkpoints/sweep_oc_main_noise_seed42_smoke/main_phnode_full_seed42/noise_budgets.json`
- `checkpoints/sweep_oc_main_noise_seed42_smoke/main_phnode_full_seed42/heldout_evaluation.json`
- `checkpoints/sweep_oc_main_noise_seed42_smoke/main_phnode_full_seed42/rollout_benchmark_results/.../summary.json`

### 4.4 通过标准

- `noise_profile=nominal_train`
- `noise_reference=remus100_dr`
- `relative_to_clean` 字段存在
- rollout summary 中的 `config.noise_budget` 存在

如果 smoke test 不通过，后续所有批量实验先暂停。

---

## 5. P1-1：补齐 noisy sweep 的关键 seeds

### 5.1 研究问题

当前 noisy sweep 只覆盖 `43/44/45`，但 clean sweep 中 `main/phnode_full` 的关键坏 seed 是 `42` 与 `46`。因此，当前还不能严谨地说 noisy training 修复了 full PHNODE 的 seed fragility。

这一组实验的目标是直接回答：

- noisy training 下，`seed42` 和 `seed46` 还会不会是 bad outlier？
- 如果不再是 bad outlier，那么这种改善只发生在 `phnode_full`，还是 `ablate_no_mass_prior` / `ablate_no_lift` 也有类似现象？

### 5.2 最小必要模型集合

建议只补三类模型：

- `phnode_full`
- `ablate_no_mass_prior`
- `ablate_no_lift`

原因：

- `phnode_full` 是核心问题本体
- `ablate_no_mass_prior` 是 noisy stress-test 最强对照
- `ablate_no_lift` 是 clean PHNODE 家族最强对照

### 5.3 目标 seeds

- `42 46 47`

原因：

- `42` 与 `46` 是 clean sweep 中 full PHNODE 的关键坏 seed
- `47` 用来与 clean PHNODE 的稳定簇对齐，补齐对照

### 5.4 推荐 suite 命名

- `sweep_oc_main_noise_nominal_train_remus100_dr_extra_42-46-47`
- `sweep_oc_key_ablation_noise_nominal_train_remus100_dr_extra_42-46-47`

### 5.5 推荐命令

先跑主模型：

```bash
bash scripts/train_all_models_noise_profile.sh \
  --profile oc \
  --models "phnode_full" \
  --seeds "42 46 47" \
  --suite-name sweep_oc_main_noise_nominal_train_remus100_dr_extra_42-46-47 \
  --noise-reference remus100_dr
```

再跑关键消融：

```bash
bash scripts/train_all_models_noise_profile.sh \
  --profile oc \
  --models "ablate_no_mass_prior ablate_no_lift" \
  --seeds "42 46 47" \
  --suite-name sweep_oc_key_ablation_noise_nominal_train_remus100_dr_extra_42-46-47 \
  --noise-reference remus100_dr
```

然后统一跑 rollout benchmark：

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/sweep_oc_main_noise_nominal_train_remus100_dr_extra_42-46-47
```

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/sweep_oc_key_ablation_noise_nominal_train_remus100_dr_extra_42-46-47
```

### 5.6 主要检查项

对 `main/phnode_full`：

- `seed42` 是否仍为 problematic seed
- `seed46` 是否仍为 problematic seed
- 扩展到 `42/43/44/45/46/47` 后，all-seed `nominal_eval` 排名是否仍保持第一梯队

对两个消融：

- `ablate_no_mass_prior` 是否继续保持 stress-test 优势
- `ablate_no_lift` 是否在 noisy 下依然明显弱于 full model

### 5.7 预期输出

完成后应额外生成一张 seed audit 表：

| Model | Train type | Seeds | Problematic seeds | Audit status | 60s nominal |
| --- | --- | --- | --- | --- | ---: |
| `main/phnode_full` | noisy | `42~47` | ... | ... | ... |
| `ablate_no_mass_prior` | noisy | `42~47` | ... | ... | ... |
| `ablate_no_lift` | noisy | `42~47` | ... | ... | ... |

---

## 6. P1-2：对 clean-trained checkpoints 补跑 noisy rollout benchmark

### 6.1 研究问题

当前报告中关于 clean vs noisy 的比较，大多是“跨 sweep 趋势判断”，不是严格 matched comparison。要把结论写得更硬，需要对 clean-trained 模型补跑 noisy rollout benchmark。

这一组实验要回答：

- noisy training 相对 clean training，是否在 `nominal_eval` 下稳定获益？
- 这种获益是否以牺牲 clean performance 为代价？
- 哪些模型从 noisy training 中真正受益？

### 6.2 优先模型集合

建议优先只补以下四类模型：

- `phnode_full`
- `ablate_no_mass_prior`
- `ablate_no_lift`
- `phnode_qforce`

原因：

- 它们已经构成当前论文主叙事中的核心竞争集合

### 6.3 对齐 seed 范围

严格 matched comparison 采用重叠 seeds：

- `phnode_full`: `43 44 45`
- `ablate_no_mass_prior`: `43 44 45`
- `ablate_no_lift`: `43 44 45`
- `phnode_qforce`: `43 44`

### 6.4 推荐命令

对 clean core suite 补跑 noisy rollout：

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/sweep_oc_core_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_115414 \
  --noise-profiles "clean nominal_eval degraded_eval heading_biased_eval"
```

对 clean ablation suite 补跑 noisy rollout：

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/sweep_oc_ablation_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_143830 \
  --noise-profiles "clean nominal_eval degraded_eval heading_biased_eval"
```

对 clean PHNODE extra suite 补跑 noisy rollout：

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47 \
  --noise-profiles "clean nominal_eval degraded_eval heading_biased_eval"
```

### 6.5 主汇总表

建议汇总成以下表格：

| Model | Seeds | Train type | Clean | Nominal | Degraded | Heading-biased |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `main/phnode_full` | `43,44,45` | clean | ... | ... | ... | ... |
| `main/phnode_full` | `43,44,45` | noisy | ... | ... | ... | ... |
| `ablate_no_mass_prior` | `43,44,45` | clean | ... | ... | ... | ... |
| `ablate_no_mass_prior` | `43,44,45` | noisy | ... | ... | ... | ... |

### 6.6 核心判据

- 对每个模型，比较 `nominal_eval` 下 noisy-train 是否优于 clean-train
- 比较 `degraded_eval` 下增益是否更明显
- 比较 `clean` profile 下是否存在明显代价

### 6.7 建议写法

如果这组实验支持 noisy training，则论文里可写为：

> Under matched seeds and identical rollout-noise profiles, noisy-IC training improves the long-horizon deployment benchmark for the full PHNODE family, while incurring only a small penalty on clean replay.

如果不支持，则可以改写为：

> The apparent robustness gain of noisy training is model-dependent and should not be interpreted as a uniform benefit across architectures.

---

## 7. P2-1：`mass prior` 与 `lift` 的机制实验

### 7.1 研究问题

当前最值得进一步澄清的结构现象有两个：

- clean 下 `ablate_no_lift` 强于 `phnode_full`
- noisy stress 下 `ablate_no_mass_prior` 反而最稳

这提示 `mass prior` 与 `lift` 可能不是简单的“有/无效果”，而是与训练机制存在交互。

### 7.2 建议设计

理想设计是一个 2x2 组合：

- `phnode_full`
- `ablate_no_mass_prior`
- `ablate_no_lift`
- `ablate_no_mass_prior_no_lift`

最后一个组合模型当前仓库里没有现成 CLI 名称，建议单独加一个临时 model spec，再做这一组实验。

### 7.3 训练设置

建议两种训练方式都跑：

- clean
- noisy `nominal_train`

seeds 建议：

- `42 43 44 45 46 47`

### 7.4 主要问题

这组实验要回答：

- lift 的负面作用是否只出现在 clean training
- mass prior 的问题是否主要出现在 stress robustness
- noisy training 是否本质上是在缓解 lift 对优化造成的困难

### 7.5 推荐在 P1 完成后再做

在没有 P1 的 seed-complete noisy 结论之前，不建议先做这个机制实验，否则很容易把“seed 覆盖差异”误判成“结构机制差异”。

---

## 8. P2-2：noisy schedule 小范围扫描

### 8.1 研究问题

如果 `main/phnode_full` 的问题主要是训练鲁棒性，那么 noisy schedule 本身应当影响坏 seed 是否出现。

### 8.2 只跑单模型

只跑：

- `phnode_full`

只补最关键 seeds：

- `42 46`

### 8.3 推荐 3 组 schedule

1. 默认：
   - `warmup=20`
   - `ramp=80`
   - `mix=0.5`
2. 更保守：
   - `warmup=40`
   - `ramp=120`
   - `mix=0.3`
3. 更激进：
   - `warmup=0`
   - `ramp=40`
   - `mix=0.7`

### 8.4 推荐命令示例

保守设置：

```bash
bash scripts/train_all_models_noise_profile.sh \
  --profile oc \
  --models "phnode_full" \
  --seeds "42 46" \
  --suite-name sweep_oc_main_noise_schedule_conservative \
  --noise-reference remus100_dr \
  --noise-warmup-epochs 40 \
  --noise-ramp 120 \
  --noise-mix-ratio 0.3
```

激进设置：

```bash
bash scripts/train_all_models_noise_profile.sh \
  --profile oc \
  --models "phnode_full" \
  --seeds "42 46" \
  --suite-name sweep_oc_main_noise_schedule_aggressive \
  --noise-reference remus100_dr \
  --noise-warmup-epochs 0 \
  --noise-ramp 40 \
  --noise-mix-ratio 0.7
```

训练后都要跑：

```bash
bash scripts/eval_all_models_noise_profile.sh \
  --suite-dir ./checkpoints/<suite_name>
```

### 8.5 核心判据

- `seed42` / `seed46` 是否从 bad outlier 变成正常 seed
- nominal profile 下是否仍保持竞争力
- stress profile 下是否进一步变稳

这组实验的价值在于：如果它有效，说明 full PHNODE 的主要问题更可能来自优化路径，而不是结构本身。

---

## 9. P3：扩展实验

### 9.1 `remus100_ins` 扩展线

这条线主要用于评估：

- `current_bias_eval`
- 更强导航估计假设下的鲁棒性排名是否变化

建议只跑 top-3：

- `phnode_full`
- `ablate_no_mass_prior`
- `ablate_no_lift`

前提是 P1 已经做完。

### 9.2 `noc` 对照线

这条线主要用于回答：

- 当前 `oc` 下观察到的 noisy robustness 排名变化，是否依赖 current-state coupling

建议只跑：

- `phnode_full`
- `ablate_no_mass_prior`
- `phnode_qforce`

这不是当前主论文结论必须的一组，因此优先级低于 P1 和 P2。

---

## 10. 推荐执行顺序

建议严格按以下顺序执行：

1. `P0`：single-model smoke test
2. `P1-1`：补齐 noisy `42/46/47`
3. `P1-2`：对 clean-trained checkpoints 补跑 noisy rollout benchmark
4. 更新综合实验报告
5. `P2-1`：`mass prior` / `lift` 机制实验
6. `P2-2`：schedule 扫描
7. `P3`：扩展线

不建议一开始直接开 `all-model all-seed all-profile` 大 sweep，因为当前最关键的不确定点非常集中，先集中打掉这些不确定点更有效。

---

## 11. 最小闭环版本

如果只希望用最少算力，拿到最大信息增益，建议只做两件事：

### 方案 A

补跑 noisy：

- `phnode_full`
- `ablate_no_mass_prior`
- `ablate_no_lift`

seeds：

- `42 46 47`

### 方案 B

对现有 clean-trained checkpoints 补跑 noisy rollout：

- `phnode_full`
- `ablate_no_mass_prior`
- `ablate_no_lift`
- `phnode_qforce`

profile：

- `clean nominal_eval degraded_eval heading_biased_eval`

这两步完成后，当前报告中最关键的薄弱点就会被补齐。

---

## 12. 最终建议输出表

补充实验完成后，建议在论文或内部报告中固定输出两张表。

### 表 A：Matched Train-Type Comparison

| Model | Seeds | Train type | Clean | Nominal | Degraded | Heading-biased |
| --- | --- | --- | ---: | ---: | ---: | ---: |

用途：

- 直接比较 clean-trained vs noisy-trained 的真实收益

### 表 B：Seed Robustness Audit

| Model | Train type | Seeds | Problematic seeds | Audit status | 60s median |
| --- | --- | --- | --- | --- | ---: |

用途：

- 明确哪些结论来自平均性能，哪些结论来自 seed 稳定性提升

---

## 13. 一句话总结

当前最值钱的补充实验，不是继续扩大模型数量，而是：

- 用缺失的关键 seeds 验证 full PHNODE 的 noisy 稳定性
- 用 matched noisy benchmark 验证 noisy training 是否真的优于 clean training

这两件事做完之后，论文中的核心实验结论才会从“趋势上可信”升级为“证据上扎实”。
