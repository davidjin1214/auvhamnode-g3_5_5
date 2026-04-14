# OC 补充实验结果记录（P1-1 + P1-2）

这份文档记录两组最关键的后续实验结果：

- `P1-1`：补齐 noisy training 的关键 seeds `42/46/47`
- `P1-2`：对 clean-trained checkpoints 补跑 noisy-profile rollout，形成 matched train-type comparison

主指标仍然统一使用 `60s final position error median`，并辅以 `completion@60s`。

## 1. 数据范围

### 1.1 P1-1 noisy extra seeds

- [checkpoints/sweep_oc_main_noise_nominal_train_remus100_dr_extra_42-46-47](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_main_noise_nominal_train_remus100_dr_extra_42-46-47)
- [checkpoints/sweep_oc_key_ablation_noise_nominal_train_remus100_dr_extra_42-46-47](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_key_ablation_noise_nominal_train_remus100_dr_extra_42-46-47)

### 1.2 P1-2 matched clean-train evaluation

这一步对以下 clean-trained suites 补跑了 `clean / nominal_eval / degraded_eval / heading_biased_eval`：

- [sweep_oc_core_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_115414](</Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_all/sweep_oc_core_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_115414>)
- [sweep_oc_ablation_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_143830](</Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_all/sweep_oc_ablation_default_auv_oc_traj1000_blk150_s23_d0be9434_s42-43-44_20260404_143830>)
- [sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47](</Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/sweep_oc_all/sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47>)

批量运行日志：

- [checkpoints/p1_2_clean_matched_eval_live_20260413_124225.log](/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5/checkpoints/p1_2_clean_matched_eval_live_20260413_124225.log)

## 2. P1-1：补齐 noisy `42/46/47` 后，原结论被改写

### 2.1 `nominal_eval @ 60s` 六 seed 汇总

| Model | Seeds | 60s error / m | Completion@60s | 备注 |
|---|---|---:|---:|---|
| `ablate_no_mass_prior` | 42-47 | **1.2494** | 0.9833 | 最稳 |
| `ablate_no_lift` | 42-47 | 1.4339 | 0.9833 | `seed44` 是明显 outlier |
| `phnode_full` | 42-47 | 1.8025 | 0.9815 | `seed42` 仍然异常 |

### 2.2 对 `phnode_full` 的更准确说法

补齐关键 seeds 之后，`phnode_full` 在 noisy training 下的结论应改写为：

- noisy training 只**部分修复**了 seed fragility
- `seed46` 在 noisy 下被明显修复
- `seed47` 正常
- `seed42` 仍然是 full PHNODE 的核心困难 seed

六个 seeds 的 `nominal_eval @ 60s` 分别为：

| Seed | 60s error / m |
|---|---:|
| 42 | 5.1592 |
| 43 | 1.0918 |
| 44 | 0.9557 |
| 45 | 1.6214 |
| 46 | 0.9324 |
| 47 | 1.0547 |

如果去掉 `seed42`，`phnode_full` 的稳定簇均值是 `1.1312 m`。因此它仍然有很强的 stable cluster，但已经不能再写成“noisy setting 下的 robust all-seed winner”。

### 2.3 P1-1 的直接结论

- noisy six-seed all-seed headline model 应从 `phnode_full` 改成 `ablate_no_mass_prior`
- `ablate_no_lift` 依然很强，但它在 noisy 下暴露了新的坏 seed：`44`
- noisy training 对 `phnode_full` 的收益是真实存在的，但不是普遍、均匀的修复

## 3. P1-2：matched clean-train vs noisy-train 对照

### 3.1 seed 覆盖

| Model | clean-train seeds | noisy-train seeds | matched seeds |
|---|---|---|---|
| `phnode_full` | 42-47 | 42-47 | 42-47 |
| `ablate_no_mass_prior` | 42-47 | 42-47 | 42-47 |
| `ablate_no_lift` | 42-47 | 42-47 | 42-47 |
| `phnode_qforce` | 42-44 | 43-45 | 43-44 |

`phnode_qforce` 只能做 `43/44` 的 matched comparison，因此它的结论只能作为局部参考，不能与前三个模型的 six-seed 结果等量齐观。

### 3.2 matched aggregate：`60s final position error median`

| Model | Matched seeds | Clean-train / clean | Clean-train / nominal | Clean-train / degraded | Clean-train / heading | Noisy-train / clean | Noisy-train / nominal | Noisy-train / degraded | Noisy-train / heading |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `phnode_full` | 42-47 | 9.0863 | 9.2746 | 9.8434 | 10.9715 | 1.5623 | 1.8025 | 2.4513 | 3.5482 |
| `ablate_no_mass_prior` | 42-47 | 1.4409 | 1.4662 | 2.0159 | 3.0516 | 1.0754 | 1.2494 | 1.8544 | 2.9566 |
| `ablate_no_lift` | 42-47 | **1.0022** | **1.2020** | **1.9095** | **2.9287** | 1.2941 | 1.4339 | 2.0623 | 3.3422 |
| `phnode_qforce` | 43-44 | **0.5692** | **0.8034** | **1.5482** | 3.0390 | 0.6970 | 0.9332 | 1.7151 | 3.1852 |

### 3.3 matched aggregate：`completion@60s`

| Model | Matched seeds | Clean-train | Noisy-train |
|---|---|---:|---:|
| `phnode_full` | 42-47 | 0.9852 | 0.9815 |
| `ablate_no_mass_prior` | 42-47 | 0.9852 | 0.9833 |
| `ablate_no_lift` | 42-47 | 0.9926 | 0.9833 |
| `phnode_qforce` | 43-44 | 0.9833 | 0.9833 |

completion 变化不大，真正决定模型排序的仍然是长时位置误差。

## 4. P1-2 的解释

### 4.1 `phnode_full`：收益巨大，但几乎全部来自修复 `seed46`

`nominal_eval` 下，`phnode_full` 从 `9.2746 m` 降到 `1.8025 m`，看起来是非常大的收益。但按 seed 拆开后，会发现这几乎完全是由 `seed46` 的灾难性修复驱动的：

| Seed | Clean-train nominal | Noisy-train nominal | Delta |
|---|---:|---:|---:|
| 42 | 4.1961 | 5.1592 | +0.9632 |
| 43 | 0.8403 | 1.0918 | +0.2515 |
| 44 | 0.7306 | 0.9557 | +0.2251 |
| 45 | 1.1585 | 1.6214 | +0.4629 |
| 46 | 47.8529 | 0.9324 | **-46.9205** |
| 47 | 0.8689 | 1.0547 | +0.1857 |

这里最重要的事实是：

- 六个 matched seeds 中，`phnode_full` 只有 `1/6` 个 seed 受益
- 其余 `5/6` 个 seeds 都是 noisy training 更差
- noisy training 的总体收益是真实的，但它并不是“普遍提升 full PHNODE”，而是“显著修复某个极坏 seed”

因此，对 `phnode_full` 的最稳妥表述应是：

> noisy training 对 full PHNODE 的主要作用，是缓解 catastrophic seed failure，而不是稳定地降低所有 seeds 的 rollout error。

### 4.2 `ablate_no_mass_prior`：这是最像“稳定受益”的模型

`ablate_no_mass_prior` 在 matched six-seed aggregate 上，四个评估 profile 都优于 clean-train：

- `clean`: `1.4409 -> 1.0754`
- `nominal_eval`: `1.4662 -> 1.2494`
- `degraded_eval`: `2.0159 -> 1.8544`
- `heading_biased_eval`: `3.0516 -> 2.9566`

按 seed 看，`nominal_eval` 下它是 `5/6` 个 seeds 受益，只有 `seed42` 略差：

| Seed | Clean-train nominal | Noisy-train nominal | Delta |
|---|---:|---:|---:|
| 42 | 1.3611 | 1.4042 | +0.0431 |
| 43 | 1.5599 | 1.4779 | -0.0820 |
| 44 | 1.3150 | 1.1805 | -0.1345 |
| 45 | 1.9764 | 1.2615 | -0.7149 |
| 46 | 1.1913 | 1.1030 | -0.0883 |
| 47 | 1.3938 | 1.0692 | -0.3246 |

这说明 noisy training 对 `ablate_no_mass_prior` 的收益更像是真正的 regularization，而不是单个 outlier 修复。

### 4.3 `ablate_no_lift`：收益并不稳定，整体反而略退化

`ablate_no_lift` 在 clean-train 下本来就是最稳的 PHNODE 家族。matched comparison 显示 noisy training 对它并没有稳定收益，反而四个 profile 都略有退化：

- `clean`: `1.0022 -> 1.2941`
- `nominal_eval`: `1.2020 -> 1.4339`
- `degraded_eval`: `1.9095 -> 2.0623`
- `heading_biased_eval`: `2.9287 -> 3.3422`

按 seed 看，它是 `3` 个 seeds 改善、`3` 个 seeds 恶化；其中最大的恶化来自 `seed44`：

| Seed | Clean-train nominal | Noisy-train nominal | Delta |
|---|---:|---:|---:|
| 42 | 0.8456 | 0.7891 | -0.0565 |
| 43 | 0.8586 | 0.9485 | +0.0899 |
| 44 | 2.0345 | 3.7214 | +1.6870 |
| 45 | 0.9641 | 0.8986 | -0.0655 |
| 46 | 1.0990 | 1.2042 | +0.1052 |
| 47 | 1.4103 | 1.0416 | -0.3688 |

对这个模型来说，noisy training 至少在当前 schedule 下不是默认应采用的设置。

### 4.4 `phnode_qforce`：当前证据不支持 noisy training 对它有收益

虽然 `phnode_qforce` 只有 `43/44` 的 matched seeds，但这两个 seeds 在四个 profile 上都是 noisy-train 更差：

- `clean`: `0.5692 -> 0.6970`
- `nominal_eval`: `0.8034 -> 0.9332`
- `degraded_eval`: `1.5482 -> 1.7151`
- `heading_biased_eval`: `3.0390 -> 3.1852`

因此，至少在现有证据下，noisy training 并不是对所有强模型都“白赚”的改进。

## 5. 综合结论

### 5.1 当前最稳妥的主结论

1. `P1-1` 之后，noisy six-seed all-seed 下最稳妥的 headline model 是 `ablate_no_mass_prior`，不是 `phnode_full`。
2. `phnode_full` 在 noisy training 下确实获得了巨大收益，但该收益主要来自修复 `seed46` 这种 catastrophic failure，而不是普遍降低所有 seeds 的误差。
3. `ablate_no_mass_prior` 是当前最像“稳定受益于 noisy training”的模型，四个 profile 上都优于 clean-train。
4. `ablate_no_lift` 和 `phnode_qforce` 都没有表现出稳定的 noisy-training 收益，当前 schedule 下甚至有轻微退化。
5. 因此，noisy training 的作用应表述为“与模型结构强耦合”，而不是“普适性的鲁棒增强”。

### 5.2 对论文叙事的影响

补充实验之后，experiments 章节里关于 noisy training 的叙事建议改成：

- noisy training 不会统一提升所有结构模型。
- 它对不同结构的作用方式不同：
  - 对 `phnode_full`，主要是修复坏 seed；
  - 对 `ablate_no_mass_prior`，更像稳定 regularization；
  - 对 `ablate_no_lift` 和 `phnode_qforce`，当前设置下没有明显收益。
- 因此，noisy training 的收益不能脱离结构设计来讨论。

### 5.3 下一步最值得做的事

如果还要继续做补充实验，优先级我会调整为：

1. 扫 `phnode_full` 的 noisy schedule，验证能否在保留 `seed46` 修复效果的同时，减少 `42/43/44/45/47` 的退化。
2. 为 `ablate_no_mass_prior` 做更细的机制分析，因为它是目前最稳定受益的模型。
3. 再决定是否值得把 noisy training 作为默认训练设置，而不是直接在论文里把它描述成普适改进。
