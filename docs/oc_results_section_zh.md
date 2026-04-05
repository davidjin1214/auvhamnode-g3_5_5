# 海流场景下 AUV 动力学建模实验结果小节

本节总结基于 port-Hamiltonian neural ODE 的海流场景 (`oc`) AUV 动力学建模实验结果。本文比较了一个主模型、多个结构化基线以及若干关键消融版本。所有主结论均以全 seed 结果为准，诊断性 seed 剔除结果仅用于解释训练鲁棒性，不作为主排名依据。

## 实验设置与评价口径

所有模型均在同一数据集 `auv_oc_traj1000_blk150_s23_d0be9434.pkl` 上训练与评估。核心比较指标为 `resampled @ 60s` 的最终位置误差中位数，因为短时 replay 指标往往无法充分反映长时 rollout 稳定性。除主指标外，还同时统计：

- 训练侧指标：`best_test_loss`
- 短块 replay 指标：`block_position_rmse_mean`
- held-out 轨迹 replay 指标：`heldout_position_rmse_median`
- rollout 指标：`10s / 30s / 60s` 的最终位置误差中位数、`p95`、完成率与失败率
- 场景分解指标：`PRBS`、`CHIRP`、`OU`

在 seed 审查上，本文采用统一标准：只有当某个 seed 相对于同模型的稳定主簇表现出明显异常，且该异常同时得到训练损失、heldout/block 误差或 rollout 失稳征象支持时，才将其标记为 problematic seed。若一个模型家族的大部分 seed 本身都表现较差，则不构造 seed 剔除版。

## 总体结果

从所有模型的全 seed 汇总结果看，整体最优模型仍为 `baseline/phnode_qforce`，其 `60s` 最终位置误差中位数均值为 `0.5708 m`。在 PHNODE 家族内部，表现最好的不是主模型 `main/phnode_full`，而是 `ablation/ablate_no_lift`，其 `60s` 指标为 `1.0022 m`，且完成率达到 `99.3%`。相比之下，`main/phnode_full` 在全 seed 统计下的 `60s` 指标为 `9.0863 m`，明显受异常 seed 拖累。

从结构化对比角度看，几个结论非常明确。第一，`ablation/ablate_bu_only` 表现显著劣化，`60s` 误差达到 `27.7967 m`，完成率下降到 `94.1%`，说明被移除的 actuation-conditioning 结构是长时稳定性的重要组成部分。第二，`ablation/ablate_diag_damping` 明显差于最优 PHNODE 变体，说明耦合阻尼结构提供了有效的稳定化作用。第三，`ablation/ablate_no_mass_prior` 仍保持较强性能与稳定性，说明当前质量先验在这一训练设定下尚未表现出不可替代的收益。第四，`ablation/ablate_no_lift` 在 PHNODE 家族中给出了最好的全 seed 表现，表明当前 lift 相关设计更可能是在提升优化难度，而不是稳定地提升最终性能。

## 关于 `main/phnode_full` 的修正性结论

早期三 seed 结果容易让人将 `main/phnode_full` 理解为“有一定 seed sensitivity，但上限较高”。加入 `45/46/47` 三个 seed 并统一审查后，这一说法需要修正。当前证据表明，`main/phnode_full` 同时具备两个事实：

- 它存在一个强的稳定主簇：`seed43,44,45,47`
- 它也存在真实的 bad-outlier failure mode：`seed42,46`

按照统一 seed 审查标准，`seed42` 与 `seed46` 都应被标记为 problematic。二者不仅在 `60s` rollout 上显著偏离稳定主簇，而且在训练损失、heldout/block 指标以及 rollout 失稳征象上也表现出一致异常。因此，对 `main/phnode_full` 更准确的描述不是“略微不稳定”，而是“具备强稳定主簇，但当前训练方案下存在真实的重尾失稳模式”。

需要强调的是，诊断性地去除 `seed42,46` 后，`main/phnode_full` 的稳定主簇结果仍然很强，`60s` 指标可达到 `0.6098 m`，已经接近 `baseline/phnode_qforce`，也优于大部分其它模型。这说明主模型并非表达能力不足，而是训练鲁棒性仍未解决。

## 关于各消融结论的解释

`ablation/ablate_bu_only` 的退化不是由个别异常 seed 引起，而是一个结构性现象。该模型在 `10s`、`30s`、`60s` 各尺度上均显著差于其它 PHNODE 变体，并在 `PRBS`、`CHIRP`、`OU` 三类场景中普遍表现不佳，因此可以较为明确地判断，被移除的相关结构对长时动态建模是必要的。

`ablation/ablate_diag_damping` 的结果说明，对角阻尼近似不足以保持与原模型相同的稳定性。虽然该模型并未像 `ablate_bu_only` 那样全面崩坏，但其长时 rollout 误差与短时 replay 指标都明显劣化，这说明更丰富的阻尼结构确实承担了物理正则化作用。

`ablation/ablate_no_mass_prior` 的持续强势结果意味着，当前质量先验至少不是主要性能来源。更谨慎的表述应是：在当前数据规模与训练配置下，质量先验的重要性低于 actuation-conditioning、阻尼结构以及训练鲁棒性问题。

`ablation/ablate_no_lift` 则给出了一个非常有信息量的结果。它不仅在全 seed 排名中成为最佳 PHNODE 家族，而且没有表现出类似 `main/phnode_full` 的 bad-outlier seed。这提示当前 lift 项可能提升了模型的潜在上限，但同时显著放大了优化难度与 seed 敏感性。

## 当前最合理的结论

综合来看，当前实验支持以下判断：

1. 如果目标是整体最强且最稳的模型，当前最佳结果仍是 `baseline/phnode_qforce`。
2. 如果目标是选择当前最可靠的 PHNODE 变体，`ablation/ablate_no_lift` 是最好的候选。
3. `main/phnode_full` 仍有很强潜力，但在现有训练方案下不能被视为鲁棒可复现的最优模型。
4. `bu_only` 相关结构和耦合阻尼结构都应保留。
5. 当前质量先验不是最优先需要继续打磨的部分。

因此，下一阶段更值得投入的方向，不是继续做更多无差别结构消融，而是专门提升 `main/phnode_full` 的训练鲁棒性，尤其是围绕 lift 相关路径与 bad-outlier seed 失稳模式展开机制分析与优化。
