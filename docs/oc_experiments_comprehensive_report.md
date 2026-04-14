# 海流场景下 AUV 动力学建模综合实验报告

> 更新说明：本文档主要总结最初的 clean/noisy 两组核心 sweep。补充实验 `P1-1` 与 `P1-2` 已进一步修正了 noisy-training 相关结论，尤其是否把 `phnode_full` 视为 noisy all-seed winner。更新后的结论见 [docs/oc_followup_results_p1_p2.md](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/docs/oc_followup_results_p1_p2.md)。

## 1. 实验目标与报告范围

本文档综合分析本仓库中 ocean-current (`oc`) 场景的两组核心实验结果：

- `checkpoints/sweep_oc_all/`：干净数据训练与 clean evaluation。
- `checkpoints/sweep_oc_all_noise/`：带噪初值正则训练，训练 profile 为 `nominal_train`，主评估 profile 为 `nominal_eval`，并补充 `degraded_eval` 和 `heading_biased_eval`。

从研究问题上看，这两组实验共同回答三个问题：

1. 在海流扰动存在时，结构化 port-Hamiltonian Neural ODE 是否优于弱结构或纯黑箱模型？
2. 哪些结构组件真正贡献了长时 rollout 稳定性？
3. 加入 IC-only 噪声训练后，模型排名与鲁棒性结论会发生什么变化？

本文尽量采用论文 experiments 章节的写法，但表达上保持直接，不刻意追求过度正式。

---

## 2. 实验设置与评价口径

### 2.1 数据与模型范围

两组实验都基于同一个海流数据集：

- dataset: `auv_oc_traj1000_blk150_s23_d0be9434.pkl`

比较对象包括：

- 主模型：`main/phnode_full`
- 结构化基线：`baseline/phnode_qforce`, `baseline/phnode_merged_force`
- 弱结构基线：`baseline/se3_momentum_blackbox`, `baseline/se3_accel_blackbox`
- 全黑箱基线：`baseline/blackbox_fullstate`
- 关键消融：`ablation/ablate_no_mass_prior`, `ablation/ablate_diag_damping`, `ablation/ablate_no_lift`, `ablation/ablate_bu_only`

### 2.2 主指标

本项目最应该看的指标是：

- `resampled @ 60s final position error median`

原因很明确：block replay 和 heldout replay 虽然能反映局部拟合质量，但不足以区分长时 rollout 是否稳定。多个模型在短时 replay 上都可以做到很小误差，但在 `60s` rollout 上会明显分层。因此，本文所有 headline claim 都以 `60s` 末端位置误差中位数为准。

同时辅助参考以下指标：

- `best_test_loss`
- `block_position_rmse_mean`
- `heldout_position_rmse_median`
- `10s / 30s / 60s` rollout 误差
- `completion rate` 与 `model failure rate`
- `PRBS / CHIRP / OU` 三类场景分解结果

### 2.3 关于 seed 审查

本项目已经采用较严格的 seed 审查规则：

- 只有当某个 seed 相对同模型稳定主簇表现出明显异常，并且这种异常同时被训练损失、heldout/block 指标或 rollout failure 支持时，才标记为 problematic seed。
- all-seed 统计始终是主证据。
- pruned 结果只能用来做诊断解释，不能替代主结论。

这一点对 `main/phnode_full` 与 `baseline/phnode_qforce` 的解释尤其重要。

---

## 3. 干净数据训练结果

### 3.1 总体表现

干净数据训练的 compact ranking 如下：

| 模型 | Seeds | Audit | 60s 误差/m | 完成率@60s |
| --- | --- | --- | ---: | ---: |
| `baseline/phnode_qforce` | `42,43,44` | stable | **0.5708** | 98.1% |
| `ablation/ablate_no_lift` | `42,43,44,45,46,47` | stable | 1.0022 | 99.3% |
| `baseline/se3_momentum_blackbox` | `42,43,44` | stable | 1.1647 | 98.5% |
| `ablation/ablate_no_mass_prior` | `42,43,44,45,46,47` | stable | 1.4409 | 98.5% |
| `baseline/phnode_merged_force` | `42,43,44` | stable | 1.4991 | 97.8% |
| `baseline/se3_accel_blackbox` | `42,43,44` | stable | 3.8874 | 98.1% |
| `ablation/ablate_diag_damping` | `42,43,44,45,46,47` | stable | 4.1659 | 98.3% |
| `main/phnode_full` | `42,43,44,45,46,47` | prunable bad outliers | 9.0863 | 98.5% |
| `ablation/ablate_bu_only` | `42,43,44,45,46,47` | structurally poor | 27.7967 | 94.1% |
| `baseline/blackbox_fullstate` | `42,43,44` | family unstable | 86.7755 | 18.9% |

从 all-seed 结果看，clean setting 下最强模型仍然是 `baseline/phnode_qforce`。在 PHNODE 家族内部，表现最好的不是主模型 `main/phnode_full`，而是 `ablation/ablate_no_lift`。这说明在干净训练条件下，当前 full PHNODE 的结构潜力尚未被稳定释放。

### 3.2 对主模型 `main/phnode_full` 的正确解读

如果只看 clean sweep 的全 seed 结果，`main/phnode_full` 会显得很差；但这并不是一个简单的“模型设计失败”结论。更准确的解释是：

- `seed43,44,45,47` 形成一个强稳定主簇；
- `seed42,46` 是真实的 bad outlier；
- 坏 seed 不只是 60s 指标差，而且训练损失、heldout/block 指标和 rollout 失稳征象都同步异常。

诊断性地去掉 `42,46` 后，`main/phnode_full` 的稳定主簇结果是：

- `60s median = 0.6098 m`

这个数值已经非常接近 `baseline/phnode_qforce` 的 `0.5708 m`。因此，clean sweep 对 full PHNODE 最合理的结论不是“性能弱”，而是“上限很强，但训练鲁棒性存在明显问题”。

### 3.3 结构性结论

clean sweep 中有几条结构结论非常稳定：

1. `ablate_bu_only` 全面退化，说明 actuation-conditioning 结构是长时稳定性的关键组成部分。
2. `ablate_diag_damping` 明显弱于最优 PHNODE 变体，说明 richer damping structure 不是装饰，而是在实际承担稳定化作用。
3. `ablate_no_mass_prior` 仍然强且稳，说明当前质量先验至少还不是主要收益来源。
4. `ablate_no_lift` 成为 clean 条件下最优 PHNODE 家族，提示 lift 项目前更像是在提高优化难度，而不是稳定提升最终性能。
5. `blackbox_fullstate` 几乎完全失效，说明这个任务不能依赖纯黑箱动力学拟合。

---

## 4. 带噪初值训练结果

### 4.1 `nominal_eval` 下的总体排名

noisy sweep 的主评估 profile 是 `nominal_eval`。其 compact ranking 如下：

| 模型 | Seeds | Audit | 60s 误差/m | 完成率@60s |
| --- | --- | --- | ---: | ---: |
| `main/phnode_full` | `43,44,45` | stable | **1.2230** | 98.1% |
| `ablation/ablate_no_mass_prior` | `43,44,45` | stable | 1.3066 | 98.1% |
| `ablation/ablate_no_lift` | `43,44,45` | stable | 1.8562 | 98.5% |
| `baseline/se3_momentum_blackbox` | `43,44,45` | stable | 2.0076 | 98.5% |
| `baseline/phnode_qforce` | `43,44,45` | prunable bad outlier | 2.0952 | 98.5% |
| `baseline/phnode_merged_force` | `43,44,45` | stable | 2.1464 | 98.1% |

这一组结果和 clean sweep 有明显不同。最关键的变化有两点：

- `main/phnode_full` 成为 noisy `nominal_eval` 下的第一名。
- `baseline/phnode_qforce` 不再是 headline model，反而因为 `seed45` 的明显异常被拉低。

这说明噪声 IC 训练不仅改变了绝对误差水平，更重新排列了模型家族之间的相对优势。

### 4.2 主模型在 noisy sweep 中的表现

在 noisy sweep 已覆盖的 seeds `43,44,45` 上，`main/phnode_full` 没有出现 clean sweep 中那种明显 bad-outlier 行为。其各 seed 的 `60s` 指标分别为：

- seed43: `1.0918 m`
- seed44: `0.9557 m`
- seed45: `1.6214 m`

从这三个 seed 看，它的 family-level stability 明显好于 clean sweep 的 all-seed 表现。不过这里要严格说明一个限制：

- clean sweep 中 full PHNODE 的已知坏 seed 是 `42` 和 `46`
- noisy sweep 并没有覆盖 `42` 和 `46`

因此，现阶段最多只能说：

- noisy training 下，`main/phnode_full` 在 `43/44/45` 上表现稳定并取得了最好的 all-seed 结果

但还不能过度外推成：

- noisy training 已经彻底解决了 full PHNODE 的 seed fragility

这个差别在论文表述里必须保留。

### 4.3 stress profile 结果

noisy sweep 还补充了两个更强压力测试：

- `degraded_eval`
- `heading_biased_eval`

top-3 结果如下：

| Profile | 第1名 | 60s/m | 第2名 | 60s/m | 第3名 | 60s/m |
| --- | --- | ---: | --- | ---: | --- | ---: |
| `nominal_eval` | `main/phnode_full` | 1.2230 | `ablate_no_mass_prior` | 1.3066 | `ablate_no_lift` | 1.8562 |
| `degraded_eval` | `ablate_no_mass_prior` | 1.8846 | `main/phnode_full` | 1.9922 | `ablate_no_lift` | 2.3852 |
| `heading_biased_eval` | `ablate_no_mass_prior` | 2.9027 | `main/phnode_full` | 3.0362 | `baseline/phnode_merged_force` | 3.2208 |

这说明 noisy sweep 顶层竞争的真实格局其实是：

- `main/phnode_full` 最擅长标准噪声设定
- `ablate_no_mass_prior` 在更强 stress 下最稳

也就是说，top-2 竞争已经从 clean sweep 中的 `qforce vs no_lift`，转移到了 noisy sweep 中的 `full PHNODE vs no-mass-prior`。

### 4.4 场景分解

从 `PRBS / CHIRP / OU` 三类场景分解来看，强结构模型在 noisy `nominal_eval` 下通常都是：

- `OU` 更难
- `completion rate` 仍然较高
- 真正分化的是 final error 而不是 completion

例如：

- `main/phnode_full`: `PRBS 1.1997`, `CHIRP 1.0933`, `OU 1.3985`
- `ablate_no_mass_prior`: `PRBS 1.0979`, `CHIRP 1.2490`, `OU 1.7860`
- `ablate_no_lift`: `PRBS 1.8262`, `CHIRP 1.5539`, `OU 2.6756`

因此，对 noisy benchmark 来说，`OU` 更像是真正能拉开顶层模型差距的场景。

---

## 5. clean 与 noisy 两组实验如何联合解释

### 5.1 模型排名发生了实质性变化

两组实验最重要的综合结论是：噪声 IC 训练并不是一个“对所有模型均匀加分”的技巧，它会和模型结构发生强相互作用。

clean setting 下：

- 最强是 `baseline/phnode_qforce`
- 最稳的 PHNODE 家族是 `ablate_no_lift`
- `main/phnode_full` 有强上限，但被 bad seed 拖垮

noisy `nominal_eval` 下：

- 最强变成 `main/phnode_full`
- `ablate_no_mass_prior` 紧随其后
- `phnode_qforce` 退到第二梯队

因此，噪声训练并不是简单地“整体提高鲁棒性”，而是改变了不同结构假设之间的优劣关系。

### 5.2 noisy training 对 clean replay 的代价并不大

虽然两组 sweep 的 rollout setting 不能直接一一对应，但 noisy run 中保留了 `clean` profile 的 heldout replay 结果，可以用来估计它对 clean 拟合的代价。

对若干关键模型，clean heldout position median 的变化如下：

| 模型 | clean-trained | noisy-trained `clean` replay | 相对变化 |
| --- | ---: | ---: | ---: |
| `main/phnode_full` | `9.55e-05` | `1.13e-04` | `1.19x` |
| `ablate_no_mass_prior` | `1.32e-04` | `1.31e-04` | `0.99x` |
| `ablate_no_lift` | `1.20e-04` | `1.48e-04` | `1.23x` |
| `baseline/se3_momentum_blackbox` | `1.92e-04` | `2.08e-04` | `1.08x` |
| `baseline/phnode_merged_force` | `1.54e-04` | `1.99e-04` | `1.29x` |
| `baseline/phnode_qforce` | `7.66e-05` | `3.83e-04` | `5.00x` |

这里有两个值得注意的点：

1. 对大部分强结构模型，noisy training 对 clean replay 的绝对损失很小，量级仍然非常低。
2. `baseline/phnode_qforce` 是明显例外，它的 clean replay 退化相对更大，这和它在 noisy nominal rollout 下的掉队是相互一致的。

因此，更合适的说法是：

- noisy training 的主要影响不在短时拟合，而在长时 rollout 排名和 seed 稳定性

### 5.3 跨 sweep 稳定不变的结构结论

有三类结论在两组实验中都非常稳定：

1. `ablate_bu_only` 持续很差，说明 actuation-conditioning 结构是刚需。
2. `ablate_diag_damping` 持续偏弱，说明 richer damping structure 应保留。
3. `blackbox_fullstate` 始终失效，说明该任务必须依赖几何与物理结构先验。

这些结论的可信度很高，因为它们并不依赖单个 seed，也不依赖某一个 evaluation profile。

### 5.4 当前最值得重新审视的两个设计点

#### mass prior

`ablate_no_mass_prior` 在 clean sweep 中已经不弱，在 noisy sweep 中更是成为 stress-test 最强模型。这说明：

- 当前实现里的质量先验要么没有提供足够强的归纳偏置收益
- 要么它在优化上引入了额外约束，抵消了其物理意义带来的潜在好处

因此，质量先验目前不能再被当作“默认必要结构”。

#### lift term

`lift` 的现象更微妙：

- clean 条件下，去掉 lift 反而更强更稳
- noisy 条件下，带 lift 的 full PHNODE 又回到第一

这提示 lift 项并不是简单“有害”或“无用”，而更像是：

- 它可能提高了模型上限
- 但只有在更合适的训练机制下，这个上限才会被稳定释放

这比直接下结论“lift 不该保留”要更符合当前证据。

---

## 6. 当前最合理的实验结论

综合 clean 与 noisy 两组结果，当前最稳妥的结论可以概括为：

1. 在 clean evaluation 下，最强模型仍是 `baseline/phnode_qforce`。
2. 在 noisy `nominal_eval` 下，最强 all-seed 模型是 `main/phnode_full`。
3. 在更强噪声和 heading bias 压力测试下，`ablate_no_mass_prior` 最稳。
4. actuation-conditioning 与 coupled damping 是明确有效的结构，应当保留。
5. 当前质量先验并未显示出不可替代性。
6. full PHNODE 仍然是最有潜力的主模型，但 clean sweep 暴露出真实的 bad-seed failure mode；noisy sweep 说明它在部分 seed 上可以显著更稳，但尚不能宣称问题已被完全解决。
7. 完全黑箱模型在该任务下没有竞争力。

---

## 7. 对论文写作的建议表述

如果把这部分内容写进论文 experiments 章节，我建议主叙事采用下面这条线：

首先，在 ocean-current 场景下，结构化模型整体显著优于纯黑箱模型，说明几何和物理先验对长时 rollout 至关重要。其次，在干净数据训练下，最佳 all-seed 结果来自结构化基线 `phnode_qforce`，而 full PHNODE 虽然有很强的稳定主簇，但会受到少数 bad-outlier seeds 的明显影响。然后，当引入 IC-only 噪声训练后，模型排名发生重排：`main/phnode_full` 在标准导航不确定性下成为最佳模型，而 `ablate_no_mass_prior` 在更强 stress profile 下最稳。这表明噪声鲁棒训练不是简单的统一增益，而会与模型结构发生实质性交互。

再往下一层，ablation 结果支持两个比较坚实的结构性结论：actuation-conditioning 和 richer damping structure 都是长时稳定性的重要来源；相反，当前质量先验尚未体现出必须保留的价值。至于 lift 项，现有证据更支持“它提高了优化难度，但可能也提高了上限”，而不是简单地把它判定为无效设计。

---

## 8. 局限性与下一步实验

当前结论仍有三点需要谨慎：

1. clean sweep 与 noisy sweep 的 seed 覆盖并不完全一致。
2. clean-trained 模型尚未系统补跑 `nominal_eval / degraded_eval / heading_biased_eval` rollout，因此目前跨 sweep 还不是严格 matched benchmark。
3. noisy sweep 尚未覆盖 full PHNODE 在 clean sweep 中最关键的坏 seed `42` 和 `46`。

因此，下一步最有价值的补充实验是：

1. 为 noisy sweep 补跑 `seed42/46/47`，验证 noisy training 是否真的消除了 full PHNODE 的 bad-outlier 模式。
2. 用同一套 evaluation profile 给 clean-trained checkpoints 系统补跑 noisy benchmark，形成严格的 clean-vs-noisy 对照。
3. 针对 `mass prior` 和 `lift` 设计更聚焦的机制实验，而不是继续做无差别结构消融。

---

## 9. 一句话总结

如果只用一句话概括当前实验结论，那么最准确的说法是：

> 在海流场景下，结构化动力学建模远优于纯黑箱；而引入噪声初值鲁棒训练后，full PHNODE 从一个“高上限但 seed 脆弱”的模型，转向了当前最强的标准噪声设定模型，但其鲁棒性是否已经被彻底修复，还需要补全关键 seeds 后才能最终确认。
