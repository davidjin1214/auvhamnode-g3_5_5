# Phase-1A OC v4-lite cleanrun v1 结果分析报告

本文记录并分析 `phase1a_oc_v4lite_cleanrun_v1` 这次 Phase-1A OC v4-lite 实验。

说明：notebook 中出现的 `g3_5_7` 是云端运行工作区；按本次实验说明，它对应本地仓库 `g3_5_5`。本文中的路径均按本地仓库相对路径书写；导出的 `runs.tsv` 里仍保留云端绝对路径。

## 1. 数据来源

本报告基于以下文件：

- `notebook/phase1a_oc_v4lite_formal_workflow_completed.ipynb`
- `docs/phase1_realistic_validation_plan.md`
- `docs/phnode_realistic_validation_execution_plan.md`
- `docs/oc_followup_results_p1_p2.md`
- `analysis/oc_data_catalog/canonical_rollout_summary_long.csv`
- `checkpoints/sweep_oc_phase1a_decision_proxy_phase1a_oc_v4lite_cleanrun_v1/`

本次结果目录包含以下关键产物：

- `phase1a_summary.csv`
- `phase1a_by_seed.csv`
- `phase1a_by_scenario.csv`
- `phase1a_by_horizon.csv`
- `phase1a_degradation.csv`
- `phase1a_protocol_delta.csv`
- `phase1a_train_audit.csv`
- `phase1a_decision_brief.md`
- `sweep_summary.json`
- `sweep_summary.txt`
- `runs.tsv`

其中 `phase1a_summary.csv` 是主要聚合表，`phase1a_by_seed.csv`、`phase1a_by_scenario.csv` 和 `phase1a_by_horizon.csv` 用于判断 Phase-1A 要求的 seed / scenario / horizon 分解。本文主指标沿用计划文档中的口径：

```text
60s final position error median
completion@60s
```

表中 `Pos Median` 是 `phase1a_summary.csv` 中对 seed-level rollout median 的均值，不是把所有轨迹重新池化后的全局 median。

## 2. 实验矩阵

本次 decision proxy suite 共 `45` 个 run：

- train protocol: `clean`, `iid_noisy_ic`, `v4_lite`
- models: `phnode_full`, `ablate_no_mass_prior`, `ablate_no_lift`
- seeds: `42, 43, 44, 45, 46`

主 rollout evaluation 设置：

- scenarios: `PRBS`, `CHIRP`, `OU`
- horizons: `10s`, `30s`, `60s`
- primary noisy eval profile: `nominal_eval`
- rollout eval protocols: `iid_noisy_ic`, `v4_lite`
- clean replay eval: `clean`

`phase1a_matrix.json` 显示，本次 rollout 层只导出了：

- `clean`
- `iid_noisy_ic:nominal_eval`
- `v4_lite:nominal_eval`

`heading_biased_eval` 与 `degraded_eval` 在本次 Phase-1A 中没有进入 rollout 决策表；它们不是本次结论的依据。

## 3. Notebook 执行记录摘要

notebook 的主要流程为：

1. 设置 `RUN_TAG=phase1a_oc_v4lite_cleanrun_v1`。
2. 配置 Phase-1A 矩阵：三模型、smoke seeds `42/44/46`、decision seeds `42/43/44/45/46`。
3. 运行 preflight。
4. 运行 smoke-1、smoke-3 training / evaluation。
5. 补训 decision extra seeds `43/45`。
6. 将 smoke3 的 `42/44/46` 与 extra `43/45` 合并注册为五 seed decision suite。
7. 对 decision suite 运行 audit / validation / rollout eval。
8. 构造 manifest-only proxy suite 并运行 summary/report 脚本。

本地导出的 proxy suite 是一个 manifest-only 汇总目录：

```text
checkpoints/sweep_oc_phase1a_decision_proxy_phase1a_oc_v4lite_cleanrun_v1
```

其 `suite_config.txt` 标注：

```text
Type: manifest-only combined decision suite; no symlinks
Sources: decision_clean + decision_iid + decision_v4lite
```

这意味着本地目录中保存了汇总表，但 `runs.tsv` 中的原始 `run_dir` / `checkpoint` 字段仍指向云端路径。本文分析以导出的 CSV 和 Markdown brief 为准。

## 4. 训练审计

`phase1a_train_audit.csv` 中大多数 run 的 `best_epoch` 接近训练末期，`best_loss` 处于正常范围。唯一明显异常为：

| group | model_type | seed | train_noise_protocol | best_epoch | best_loss | run_dir |
|:--|:--|--:|:--|--:|--:|:--|
| ablation | ablate_no_lift | 43 | clean | 19 | 0.2169 | `/content/drive/MyDrive/Colab Notebooks/auvhamnode/g3_5_7/checkpoints/sweep_oc_phase1a_decision_extra43-45_clean_phase1a_oc_v4lite_cleanrun_v1/ablation_ablate_no_lift_seed43` |

这个异常非常重要。该 run 的 clean 训练在 `best_epoch=19` 就停止在很高 loss 上，随后 60s rollout error 达到约 `44 m`。因此：

- 当前 `ablate_no_lift + clean train` 的 all-seed 结果不能作为 clean baseline 的可靠证据。
- 所有依赖 `ablate_no_lift clean` 的 clean replay cost 或 clean ranking 都应先排除或重跑该 seed。
- 该异常不直接影响 `iid_noisy_ic train` 与 `v4_lite train` 的同协议比较，但会污染 clean-vs-noisy 的解释。

本次 proxy 目录中没有导出 `phase1a_v4_protocol_validation.json`。notebook 中执行过 validate 命令，但本地可查的最终 artifact 只包含 summary/audit CSV 和 brief。若后续要将该结果写入论文证据链，应补保存 protocol validation JSON、run config 和 environment metadata。

## 5. Headline 结果

下表为 `60s / nominal_eval` 的聚合结果，按 `Pos Median` 从小到大排序。

| model | train | eval | seeds | Pos Median / m | Pos P95 / m | Completion |
|:--|:--|:--|:--|--:|--:|--:|
| `main/phnode_full` | `clean` | `v4_lite` | 42,43,44,45,46 | 0.8439 | 2.1544 | 0.9911 |
| `main/phnode_full` | `clean` | `iid_noisy_ic` | 42,43,44,45,46 | 0.9604 | 2.8640 | 0.9911 |
| `ablation/ablate_no_lift` | `iid_noisy_ic` | `iid_noisy_ic` | 42,43,44,45,46 | 1.0043 | 3.4515 | 0.9822 |
| `main/phnode_full` | `iid_noisy_ic` | `v4_lite` | 42,43,44,45,46 | 1.0317 | 2.9404 | 0.9844 |
| `ablation/ablate_no_lift` | `iid_noisy_ic` | `v4_lite` | 42,43,44,45,46 | 1.0551 | 3.0847 | 0.9822 |
| `main/phnode_full` | `iid_noisy_ic` | `iid_noisy_ic` | 42,43,44,45,46 | 1.0934 | 3.2639 | 0.9822 |
| `main/phnode_full` | `v4_lite` | `iid_noisy_ic` | 42,43,44,45,46 | 1.1116 | 3.3398 | 0.9822 |
| `main/phnode_full` | `v4_lite` | `v4_lite` | 42,43,44,45,46 | 1.1148 | 2.9198 | 0.9822 |
| `ablation/ablate_no_lift` | `v4_lite` | `iid_noisy_ic` | 42,43,44,45,46 | 1.2696 | 3.6817 | 0.9844 |
| `ablation/ablate_no_lift` | `v4_lite` | `v4_lite` | 42,43,44,45,46 | 1.3078 | 3.5389 | 0.9844 |
| `ablation/ablate_no_mass_prior` | `v4_lite` | `iid_noisy_ic` | 42,43,44,45,46 | 1.4021 | 4.5957 | 0.9822 |
| `ablation/ablate_no_mass_prior` | `v4_lite` | `v4_lite` | 42,43,44,45,46 | 1.4075 | 4.0668 | 0.9844 |
| `ablation/ablate_no_mass_prior` | `clean` | `iid_noisy_ic` | 42,43,44,45,46 | 1.4163 | 4.6889 | 0.9867 |
| `ablation/ablate_no_mass_prior` | `iid_noisy_ic` | `iid_noisy_ic` | 42,43,44,45,46 | 1.4613 | 4.3018 | 0.9822 |
| `ablation/ablate_no_mass_prior` | `clean` | `v4_lite` | 42,43,44,45,46 | 1.4622 | 4.2905 | 0.9867 |
| `ablation/ablate_no_mass_prior` | `iid_noisy_ic` | `v4_lite` | 42,43,44,45,46 | 1.5156 | 3.8697 | 0.9822 |
| `ablation/ablate_no_lift` | `clean` | `v4_lite` | 42,43,44,45,46 | 9.6808 | 17.1002 | 0.9889 |
| `ablation/ablate_no_lift` | `clean` | `iid_noisy_ic` | 42,43,44,45,46 | 9.7397 | 17.4667 | 0.9889 |

直接观察：

- 最好的 headline 行是 `phnode_full + clean train + v4_lite eval`，但这不能解释成 `v4_lite training` 的胜利。
- 在 noisy training 的匹配比较中，`v4_lite train` 没有成为任何模型的稳定全局 winner。
- `ablate_no_lift clean` 的末尾两行被 `seed43` 异常严重污染，不应作为 clean baseline 结论。

## 6. v4-lite 作为训练协议的效果

为了判断 `v4_lite` 是否应该成为后续主训练协议，最直接的比较是固定模型和 eval protocol，对比 `v4_lite train` 与 `iid_noisy_ic train`。

| model | eval_protocol | iid_train | v4_train | delta v4-iid | ratio |
|:--|:--|--:|--:|--:|--:|
| `ablation/ablate_no_lift` | `iid_noisy_ic` | 1.0043 | 1.2696 | +0.2653 | 1.2642 |
| `ablation/ablate_no_lift` | `v4_lite` | 1.0551 | 1.3078 | +0.2527 | 1.2395 |
| `ablation/ablate_no_mass_prior` | `iid_noisy_ic` | 1.4613 | 1.4021 | -0.0592 | 0.9595 |
| `ablation/ablate_no_mass_prior` | `v4_lite` | 1.5156 | 1.4075 | -0.1081 | 0.9287 |
| `main/phnode_full` | `iid_noisy_ic` | 1.0934 | 1.1116 | +0.0182 | 1.0167 |
| `main/phnode_full` | `v4_lite` | 1.0317 | 1.1148 | +0.0831 | 1.0806 |

结论：

- `ablate_no_mass_prior` 是唯一在 aggregate 上受益于 `v4_lite train` 的模型，收益约 `4.0%` 到 `7.1%`。
- `phnode_full` 基本持平或小幅退化。
- `ablate_no_lift` 明显退化，尤其在 60s 长时 rollout 上。

这不满足 Phase-1A “采用为后续主评估协议”的条件，因为改善没有出现在至少两个模型上，也没有形成跨模型稳定趋势。

## 7. v4-lite 作为评估协议的效果

固定训练协议，对比 `v4_lite eval` 与 `iid_noisy_ic eval`：

| model | train_protocol | iid_eval | v4_eval | delta v4-iid | ratio |
|:--|:--|--:|--:|--:|--:|
| `ablation/ablate_no_lift` | `clean` | 9.7397 | 9.6808 | -0.0589 | 0.9940 |
| `ablation/ablate_no_lift` | `iid_noisy_ic` | 1.0043 | 1.0551 | +0.0508 | 1.0506 |
| `ablation/ablate_no_lift` | `v4_lite` | 1.2696 | 1.3078 | +0.0382 | 1.0301 |
| `ablation/ablate_no_mass_prior` | `clean` | 1.4163 | 1.4622 | +0.0459 | 1.0324 |
| `ablation/ablate_no_mass_prior` | `iid_noisy_ic` | 1.4613 | 1.5156 | +0.0542 | 1.0371 |
| `ablation/ablate_no_mass_prior` | `v4_lite` | 1.4021 | 1.4075 | +0.0054 | 1.0038 |
| `main/phnode_full` | `clean` | 0.9604 | 0.8439 | -0.1165 | 0.8786 |
| `main/phnode_full` | `iid_noisy_ic` | 1.0934 | 1.0317 | -0.0616 | 0.9436 |
| `main/phnode_full` | `v4_lite` | 1.1116 | 1.1148 | +0.0032 | 1.0029 |

结论：

- `v4_lite eval` 对 `phnode_full` 有利，尤其是 clean-trained 和 iid-trained 行。
- `v4_lite eval` 对两个 ablation 通常略差。
- 因此，`v4_lite eval` 不是一个中性替换；它会改变不同结构的相对压力方式。它可以保留为诊断协议，但不能单独作为 headline 评价口径。

## 8. Seed-level 分解

下表固定 `eval_protocol=iid_noisy_ic`，比较 `v4_lite train` 与 `iid_noisy_ic train`。负数表示 `v4_lite train` 更好。

| model | seed | iid_noisy_ic | v4_lite | v4-iid |
|:--|--:|--:|--:|--:|
| `ablation/ablate_no_lift` | 42 | 1.0938 | 0.9641 | -0.1297 |
| `ablation/ablate_no_lift` | 43 | 0.9870 | 1.0314 | +0.0444 |
| `ablation/ablate_no_lift` | 44 | 0.8266 | 1.1309 | +0.3042 |
| `ablation/ablate_no_lift` | 45 | 1.0093 | 0.8990 | -0.1104 |
| `ablation/ablate_no_lift` | 46 | 1.1047 | 2.3225 | +1.2178 |
| `ablation/ablate_no_mass_prior` | 42 | 0.9981 | 0.8923 | -0.1057 |
| `ablation/ablate_no_mass_prior` | 43 | 1.6696 | 1.6385 | -0.0311 |
| `ablation/ablate_no_mass_prior` | 44 | 1.6001 | 1.4175 | -0.1826 |
| `ablation/ablate_no_mass_prior` | 45 | 1.8488 | 1.4910 | -0.3579 |
| `ablation/ablate_no_mass_prior` | 46 | 1.1901 | 1.5712 | +0.3812 |
| `main/phnode_full` | 42 | 1.0081 | 1.0064 | -0.0017 |
| `main/phnode_full` | 43 | 0.9545 | 0.9868 | +0.0322 |
| `main/phnode_full` | 44 | 1.1830 | 1.0800 | -0.1030 |
| `main/phnode_full` | 45 | 1.0639 | 1.3948 | +0.3309 |
| `main/phnode_full` | 46 | 1.2572 | 1.0899 | -0.1673 |

判读：

- `ablate_no_mass_prior` 的 `v4_lite train` 是 `4/5` seeds 改善，唯一明显退化是 `seed46`。
- `phnode_full` 是 `3/5` seeds 小幅改善，但 `seed45` 退化较大，aggregate 几乎持平。
- `ablate_no_lift` 是 `2/5` seeds 改善，`seed46` 被明显破坏，导致 aggregate 退化。

因此，`v4_lite train` 的收益并非单纯来自一个 catastrophic seed 修复，但也没有跨模型稳定性。

## 9. Scenario 分解

下表固定 `eval_protocol=iid_noisy_ic`，比较 `iid_noisy_ic train` 与 `v4_lite train` 在三个 scenario 的 `60s / nominal_eval` 表现。

| model | train | scenario | Pos Median / m | Pos P95 / m | Completion |
|:--|:--|:--|--:|--:|--:|
| `ablation/ablate_no_lift` | `iid_noisy_ic` | CHIRP | 0.8016 | 2.3218 | 1.0000 |
| `ablation/ablate_no_lift` | `iid_noisy_ic` | OU | 1.4331 | 4.2132 | 0.9467 |
| `ablation/ablate_no_lift` | `iid_noisy_ic` | PRBS | 0.8955 | 2.9182 | 1.0000 |
| `ablation/ablate_no_lift` | `v4_lite` | CHIRP | 1.4289 | 2.8832 | 1.0000 |
| `ablation/ablate_no_lift` | `v4_lite` | OU | 1.4524 | 3.8966 | 0.9533 |
| `ablation/ablate_no_lift` | `v4_lite` | PRBS | 1.0962 | 3.2329 | 1.0000 |
| `ablation/ablate_no_mass_prior` | `iid_noisy_ic` | CHIRP | 1.2244 | 3.0779 | 1.0000 |
| `ablation/ablate_no_mass_prior` | `iid_noisy_ic` | OU | 1.6851 | 4.3770 | 0.9467 |
| `ablation/ablate_no_mass_prior` | `iid_noisy_ic` | PRBS | 1.6672 | 4.4856 | 1.0000 |
| `ablation/ablate_no_mass_prior` | `v4_lite` | CHIRP | 1.0343 | 2.7278 | 1.0000 |
| `ablation/ablate_no_mass_prior` | `v4_lite` | OU | 1.9713 | 5.9156 | 0.9467 |
| `ablation/ablate_no_mass_prior` | `v4_lite` | PRBS | 1.3482 | 3.5786 | 1.0000 |
| `main/phnode_full` | `iid_noisy_ic` | CHIRP | 0.9667 | 2.3579 | 1.0000 |
| `main/phnode_full` | `iid_noisy_ic` | OU | 1.2296 | 3.5567 | 0.9467 |
| `main/phnode_full` | `iid_noisy_ic` | PRBS | 1.1550 | 3.3646 | 1.0000 |
| `main/phnode_full` | `v4_lite` | CHIRP | 1.0175 | 2.7701 | 1.0000 |
| `main/phnode_full` | `v4_lite` | OU | 1.2898 | 3.8120 | 0.9467 |
| `main/phnode_full` | `v4_lite` | PRBS | 1.0187 | 2.8829 | 1.0000 |

判读：

- `ablate_no_mass_prior` 在 CHIRP 和 PRBS 上受益，但 OU 变差。
- `phnode_full` 在 PRBS 上小幅受益，但 CHIRP / OU 变差。
- `ablate_no_lift` 三个 scenario 中两个变差，CHIRP 退化尤其明显。
- OU 仍然是更容易拉开模型差距的 scenario，completion 也主要在 OU 低于 100%。

这不满足“至少在两个模型或多个 scenario 上有一致趋势”的采用条件。

## 10. Horizon 分解

下表固定 `eval_protocol=iid_noisy_ic`，比较不同 horizon 下的 `Pos Median`。

| model | train | horizon / s | Pos Median / m | Completion |
|:--|:--|--:|--:|--:|
| `ablation/ablate_no_lift` | `iid_noisy_ic` | 10 | 0.1669 | 1.0000 |
| `ablation/ablate_no_lift` | `iid_noisy_ic` | 30 | 0.4582 | 1.0000 |
| `ablation/ablate_no_lift` | `iid_noisy_ic` | 60 | 1.0043 | 0.9822 |
| `ablation/ablate_no_lift` | `v4_lite` | 10 | 0.1703 | 1.0000 |
| `ablation/ablate_no_lift` | `v4_lite` | 30 | 0.4929 | 1.0000 |
| `ablation/ablate_no_lift` | `v4_lite` | 60 | 1.2696 | 0.9844 |
| `ablation/ablate_no_mass_prior` | `iid_noisy_ic` | 10 | 0.1872 | 1.0000 |
| `ablation/ablate_no_mass_prior` | `iid_noisy_ic` | 30 | 0.5760 | 1.0000 |
| `ablation/ablate_no_mass_prior` | `iid_noisy_ic` | 60 | 1.4613 | 0.9822 |
| `ablation/ablate_no_mass_prior` | `v4_lite` | 10 | 0.1900 | 1.0000 |
| `ablation/ablate_no_mass_prior` | `v4_lite` | 30 | 0.6120 | 1.0000 |
| `ablation/ablate_no_mass_prior` | `v4_lite` | 60 | 1.4021 | 0.9822 |
| `main/phnode_full` | `iid_noisy_ic` | 10 | 0.1697 | 1.0000 |
| `main/phnode_full` | `iid_noisy_ic` | 30 | 0.4716 | 1.0000 |
| `main/phnode_full` | `iid_noisy_ic` | 60 | 1.0934 | 0.9822 |
| `main/phnode_full` | `v4_lite` | 10 | 0.1620 | 1.0000 |
| `main/phnode_full` | `v4_lite` | 30 | 0.4521 | 1.0000 |
| `main/phnode_full` | `v4_lite` | 60 | 1.1116 | 0.9822 |

短 horizon 上的差异较小，真正决定判断的是 60s。`v4_lite train` 对 `ablate_no_mass_prior` 的收益主要体现在 60s；对 `ablate_no_lift` 的伤害也主要在 60s 放大。

## 11. Clean replay cost

下表来自 `phase1a_degradation.csv`，固定 `metric=rollout_final_position_error_median`、`horizon=60s`。注意：`ratio` 是 per-seed ratio 的均值，不一定等于 `value / clean_value`。

| model | train | value | clean_value | delta | ratio | degradation |
|:--|:--|--:|--:|--:|--:|--:|
| `ablation/ablate_no_lift` | `iid_noisy_ic` | 0.8291 | 9.5391 | -8.7100 | 0.9545 | -4.5460% |
| `ablation/ablate_no_lift` | `v4_lite` | 1.1606 | 9.5391 | -8.3785 | 1.1346 | +13.4562% |
| `ablation/ablate_no_mass_prior` | `iid_noisy_ic` | 1.3471 | 1.2966 | +0.0505 | 1.0320 | +3.1958% |
| `ablation/ablate_no_mass_prior` | `v4_lite` | 1.2806 | 1.2966 | -0.0161 | 1.0090 | +0.9007% |
| `main/phnode_full` | `iid_noisy_ic` | 0.8914 | 0.6767 | +0.2147 | 1.4692 | +46.9168% |
| `main/phnode_full` | `v4_lite` | 0.9566 | 0.6767 | +0.2799 | 1.6767 | +67.6724% |

判读：

- `ablate_no_mass_prior` 的 clean replay cost 很小。
- `phnode_full` 的 noisy training clean replay cost 明显更大，`v4_lite train` 比 `iid_noisy_ic train` 更高。
- `ablate_no_lift` 的 clean replay cost 表被异常 clean seed43 污染，不能直接解释。

## 12. 与既有 canonical catalog 的关系

将本次 cleanrun v1 的 `iid_noisy_ic eval / nominal_eval / 60s` 与既有 canonical catalog 中同模型、同 seeds `42-46` 的结果对齐，会看到明显差异：

| model | train_type | protocol | catalog mean / m | cleanrun mean / m | seeds |
|:--|:--|:--|--:|--:|:--|
| `ablation/ablate_no_lift` | `clean_train` | `clean` | 1.1604 | 9.7397 | 42,43,44,45,46 |
| `ablation/ablate_no_lift` | `noisy_train` | `iid_noisy_ic` | 1.5124 | 1.0043 | 42,43,44,45,46 |
| `ablation/ablate_no_mass_prior` | `clean_train` | `clean` | 1.4807 | 1.4163 | 42,43,44,45,46 |
| `ablation/ablate_no_mass_prior` | `noisy_train` | `iid_noisy_ic` | 1.2854 | 1.4613 | 42,43,44,45,46 |
| `main/phnode_full` | `clean_train` | `clean` | 10.9557 | 0.9604 | 42,43,44,45,46 |
| `main/phnode_full` | `noisy_train` | `iid_noisy_ic` | 1.9521 | 1.0934 | 42,43,44,45,46 |

这说明本次 cleanrun v1 不能简单替代既有 P1-2/canonical 证据。主要原因有三点：

1. 本次 notebook 实际上重新训练并组合了 smoke3 与 extra seed suite，而不是完全复用既有 clean / iid noisy checkpoints。
2. `ablate_no_lift seed43 clean` 明显异常，直接造成 clean ablation 排名失真。
3. `phnode_full` 在本次 cleanrun v1 中没有复现既有 canonical clean 结果里的 seed42/seed46 catastrophic failure；这可能是真实的重训差异，也可能与执行路径、代码版本、checkpoint 选择或训练中断/跳过逻辑有关。

因此，本次实验更适合作为 `v4_lite` protocol sensitivity 的一次独立 decision package，而不是马上并入当前 canonical catalog。

## 13. Phase-1A 决策

根据 `docs/phase1_realistic_validation_plan.md`，`v4_lite` 只有在同时满足以下条件时才应升级为后续主评估协议之一：

1. 相对 `iid_noisy_ic` 有稳定 aggregate 改善，或显著改变模型排序。
2. 改善不是主要来自单个 catastrophic seed。
3. clean replay cost 可接受。
4. 至少在两个模型或多个 scenario 上可见一致趋势。

本次结果不满足这些条件。

具体判断：

- `v4_lite train` 只对 `ablate_no_mass_prior` 有 aggregate 收益。
- `phnode_full` 基本持平或小幅退化。
- `ablate_no_lift` 明显退化。
- scenario 分解没有形成一致趋势。
- clean replay cost 对 `phnode_full` 更高，对 `ablate_no_lift` 又被异常 clean baseline 污染。
- `v4_lite eval` 改善 `phnode_full`，但通常略伤两个 ablation，说明它是有结构偏向的诊断压力，不是中性替换。

因此，本次 Phase-1A 更接近以下结论：

```text
trajectory-consistent noisy IC 增强了协议真实性，
但 cleanrun v1 没有证明它应替代 block-iid noisy IC 成为默认训练协议；
它更适合作为补充/诊断协议保留。
```

同时，当前结果不支持直接进入 Phase-1B 的全量扩展。Phase-1B 只有在 protocol 改变模型结论或归因仍不清楚时才应触发；本次更大的不确定性来自 run provenance 和异常 seed，而不是来自清晰的 protocol 效应。

## 14. 建议

### 14.1 立即修正

1. 重跑或替换 `ablate_no_lift seed43 clean`。
2. 重新生成：
   - `phase1a_train_audit.csv`
   - `phase1a_summary.csv`
   - `phase1a_by_seed.csv`
   - `phase1a_by_scenario.csv`
   - `phase1a_by_horizon.csv`
   - `phase1a_degradation.csv`
   - `phase1a_decision_brief.md`
3. 补保存：
   - `phase1a_v4_protocol_validation.json`
   - `phase1a_run_config.json`
   - `phase1a_environment.json`

### 14.2 结果使用边界

当前 cleanrun v1 可以用于说明：

- Phase-1A workflow 能跑完整三模型五 seed decision matrix。
- `v4_lite train` 没有在本次矩阵中形成跨模型稳定收益。
- `ablate_no_mass_prior` 仍是最值得继续观察的 v4-lite 受益模型。
- `v4_lite eval` 对不同结构的压力不完全相同。

当前 cleanrun v1 不应直接用于说明：

- `ablate_no_lift clean` 的真实性能。
- `phnode_full` 已经彻底消除既有 clean seed fragility。
- `v4_lite` 应作为默认训练协议。
- Phase-1B 应立即扩展到更多模型或 heading/degraded rollout。

### 14.3 后续实验顺序

推荐顺序：

1. 先修复 `ablate_no_lift seed43 clean` 并重建本次报告表。
2. 对 `phnode_full` clean/noisy 的本次结果与既有 canonical catalog 做逐 seed provenance 对照，确认差异来自新重训还是流程差异。
3. 若 `ablate_no_mass_prior` 的 v4-lite 收益仍稳定，可针对它做小规模机制分析，尤其检查 `seed46` 为什么退化。
4. 暂不进入 Phase-1B 全量扩展；除非修复异常后 `v4_lite` 明确改变模型排序或 scientific attribution 仍无法解释。

## 15. 最终判断

本次 `phase1a_oc_v4lite_cleanrun_v1` 完成了 Phase-1A 所要求的大部分 reporting contract：五 seed、三模型、三训练协议、by-seed、by-scenario、by-horizon 与 clean replay cost 都已经导出。

但从科学结论上看，它不是 `v4_lite` 的强阳性结果。最稳妥的阶段结论是：

```text
v4-lite 是有价值的 protocol-sensitivity diagnostic，
但本次 cleanrun v1 没有证明 trajectory-consistent noisy IC
应替代 block-iid noisy IC 成为默认训练设置。
```

在修复 `ablate_no_lift seed43 clean` 并补齐 protocol validation artifact 之前，不建议把这批结果并入 canonical catalog 或作为论文主结论表。
