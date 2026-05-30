# 前代仓库 g3_5_4 — noc/oc 实验完整清单 + 模型命名映射

生成时间：2026-05-30  
对象仓库：`/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_4`（当前 `g3_5_5` 的前代）  
本文件存放位置：`g3_5_5/docs/`（作为当前主线清单 [experiment_full_inventory_zh.md](experiment_full_inventory_zh.md) 的前代补充）。

## 0. 定位

- `g3_5_4` 是 `g3_5_5` 的**前一代仓库**。共 **30 个**训练 run，**全部为单一架构 `ph_se3_full`**（= 新版 `phnode_full`）。
- 这是一条 **大 batch 训练配方扫描（batch / lr / warmup / training-steps）+ noc vs oc 数据对照**线，**不是多模型消融**。架构固定，只扫训练超参与数据是否带海流。
- `g3_5_4` 是独立仓库，不在 `g3_5_5` 的 OC catalog 内。当前论文 §8 **不使用** `g3_5_4` 的任何结果。
- 这条线回答的问题是："大 batch 默认配方该取哪个，noc 与 oc 是否需要不同配方"，属于**训练稳定性 / 精度-效率权衡**研究，早于 `g3_5_5` 的结构化 SE(3) + 海流主线。

---

## 1. 模型命名映射（旧 g3_5_4 → 新 g3_5_5）

旧版用 `model_type` 字符串 + if/elif 分派（`g3_5_4/train_auv_hamnode.py:_build_model`）；新版用声明式注册表 `MODEL_SPECS`（`g3_5_5/auv_model_registry.py`）。类身份经构造参数与 `forward` 逐行比对确认。

| 旧 `model_type` | 旧类 | 新 `model_type` | 新类 | 结构匹配 | 说明 |
|---|---|---|---|---|---|
| `ph_se3_full` | `AUVHamNODE` | `phnode_full` | `AUVHamNODE` | **完全一致** | 同一类、同一结构化 pH 核心（M⁻¹ + 标量势 V(q) + 拆分 D/J/B）。默认 `learn_lift=True, coupled_damping=True, condition_on_velocity=True`。 |
| `ph_se3_nomassinit` | `AUVHamNODE`（`M_init=None`） | `ablate_no_mass_prior` | `AUVHamNODE`（`use_mass_init=False`） | **完全一致** | 仅抑制物理质量先验。`ph_se3_*` → `ablate_*` 改名。 |
| `ph_se3_diagd` | `AUVHamNODE`（`coupled_damping=False`） | `ablate_diag_damping` | `AUVHamNODE`（override `coupled_damping=False`） | **完全一致** | 仅对角阻尼（D_net 6 维 vs 21 维）。 |
| `ph_se3_noj` | `AUVHamNODE`（`learn_lift=False`） | `ablate_no_lift` | `AUVHamNODE`（override `learn_lift=False`） | **完全一致** | 无反对称 lift（J_net=None）。 |
| `ph_se3_buonly` | `AUVHamNODE`（`condition_on_velocity=False`） | `ablate_bu_only` | `AUVHamNODE`（对应 override） | **完全一致** | 执行机构仅以 actuator 状态为条件。 |
| `ph_se3_mergednc` | `AUVHamNODEMergedNC` | `phnode_merged_force` | `PHNodeMergedForce` | **完全一致** | 仅类改名，body 相同（M + V(q) + 合并 F_net）。 |
| `ph_se3_qforce` | `AUVHamNODEQForce` | `phnode_qforce` | `PHNodeQForce` | **完全一致** | 仅改名；结构化 pH，用通用 G_net 广义力替换 −dV/dq，保留拆分 D/J/B。 |
| `mom_se3_unstruct` | `AUVMomentumSE3NODE` | `se3_momentum_blackbox` | `SE3MomentumBlackBox` | **完全一致** | 仅改名；M + 精确 SE(3)，黑箱动量 F_net。 |
| `se3_unstruct` | `AUVUnstructuredSE3NODE` | `se3_accel_blackbox` | `SE3AccelBlackBox` | **完全一致** | 仅改名；hidden_dim_scale=1.88 保留。 |
| `bb_free_unstruct` | `AUVBlackBoxNODE` | `blackbox_fullstate` | `FullStateBlackBox` | **完全一致** | 仅改名；hidden_dim_scale=1.78 保留。 |
| `ham_se3_unstruct` | `AUVUnstructuredHamNODE` | **无对应** | —（未移植） | **已删除** | 单一可学习 H(q,p) 的 SE(3) Hamiltonian 基线，新版**删除**（11 旧 → 10 新）。 |

要点：
- **`ph_se3_full` 与 `phnode_full` 是同一架构**——同一 `AUVHamNODE` 类、同一状态布局、同一海流速度契约（`v_r = v − Rᵀv_c^n`），训练用的结构 flag 完全相同。唯一差异是**装饰性**的：旧版构造函数多一个已弃用的 `condition_dj_on_current` 回退参数（新版移到 trainer 参数解析），对训练出的模型无影响。
- 旧版除 11 个 canonical 名外还接受一批 legacy 别名（`hamnode`/`hamnode_full`→`ph_se3_full`，`merged_nc`→`ph_se3_mergednc`，`blackbox`→`bb_free_unstruct` 等），均映射到 canonical。
- **但 `g3_5_4` 实际只训练了 `ph_se3_full` 一个模型**（10 个 baseline/ablation 仅定义、未训练）。因此前代 checkpoint 命名与新版 `phnode_full` 是 1:1 对应。

---

## 2. 数据集（2 个，schema `auv_dataset_v3`，seed 42）

| 字段 | noc | oc |
|---|---|---|
| 文件 | `auv_noc_traj1000_blk150_s42_32ec4535.pkl` | `auv_oc_traj1000_blk150_s42_89c80d68.pkl` |
| dataset id | `32ec4535` | `89c80d68` |
| ocean_current | **False** | **True** |
| state_dim | **24** `[Δpos3,R9,nu_b6,u_act3,u_cmd3]` | **27**（加 `v_c^n(3)`） |
| 训练/测试轨迹 | 558 / 140 | 559 / 140 |
| blocks/traj | 150（5 pts/block） | 150 |
| current 范围 | [0,0.5]（无流，未用） | [0,0.5]，均值 ‖v_c‖≈0.262 m/s |
| nu_model | `nu_b` | `nu_r = nu_b − [Rᵀv_c^n;0]` |
| .pkl 大小 | 192 MB | 216 MB |

> 注意：`g3_5_4` 的 oc 数据集是 `89c80d68`（**seed 42**），与 `g3_5_5` 主线用的 `d0be9434`（seed 23）**是不同的数据集**。两代 oc 数据不可直接混用。

---

## 3. 全部 30 个 run 明细

`best_loss` = 各 run `training.log` 的 Best test loss。**全部 30 个 run 都含 `block_evaluation.json` + `heldout_evaluation.json` + `rollout_benchmark/`**（无纯训练 run）。rollout 含 `heldout_batch_compare_10_20_30` + `resampled_batch_compare_10_30_60`（场景 PRBS/CHIRP/OU）。

| suite | run | 数据 | seed | batch | lr | warmup | train_steps | best_loss |
|---|---|---|---|---|---|---|---|---|
| run_largebatch_noc | noc_bs2048_seed233 | noc | 233 | 2048 | 5e-3 | 300 | 7000 | 9.99e-05 |
| run_largebatch_noc | noc_bs2048_seed43 | noc | 43 | 2048 | 5e-3 | 300 | 7000 | 1.64e-04 |
| run_largebatch_noc | noc_bs2048_seed44 | noc | 44 | 2048 | 5e-3 | 300 | 7000 | 1.79e-04 |
| run_largebatch_noc | noc_bs4096_seed233 | noc | 233 | 4096 | 6e-3 | 400 | 5000 | 3.87e-04 |
| run_largebatch_noc | noc_bs4096_seed43 | noc | 43 | 4096 | 6e-3 | 400 | 5000 | 2.95e-04 |
| run_largebatch_noc | noc_bs4096_seed44 | noc | 44 | 4096 | 6e-3 | 400 | 5000 | 2.75e-04 |
| run_largebatch_oc | oc_bs2048_seed233 | oc | 233 | 2048 | 5e-3 | 300 | 7000 | **4.57e-01 ⚠ 训练崩溃** |
| run_largebatch_oc | oc_bs2048_seed42 | oc | 42 | 2048 | 5e-3 | 300 | 7000 | 3.68e-03 |
| run_largebatch_oc | oc_bs2048_seed43 | oc | 43 | 2048 | 5e-3 | 300 | 7000 | 3.79e-03 |
| run_largebatch_oc | oc_bs2048_seed44 | oc | 44 | 2048 | 5e-3 | 300 | 7000 | 3.75e-03 |
| run_largebatch_oc | oc_bs4096_seed233 | oc | 233 | 4096 | 6e-3 | 400 | 5000 | 3.93e-03 |
| run_largebatch_oc | oc_bs4096_seed42 | oc | 42 | 4096 | 6e-3 | 400 | 5000 | 3.78e-03 |
| run_largebatch_oc | oc_bs4096_seed43 | oc | 43 | 4096 | 6e-3 | 400 | 5000 | 3.73e-03 |
| run_largebatch_oc | oc_bs4096_seed44 | oc | 44 | 4096 | 6e-3 | 400 | 5000 | 3.91e-03 |
| run_largebatch_oc_aligned | oc_aligned_bs4096_seed233 | oc | 233 | 4096 | 4.5e-3 | 300 | 7000 | 3.71e-03 |
| run_largebatch_oc_aligned | oc_aligned_bs4096_seed42 | oc | 42 | 4096 | 4.5e-3 | 300 | 7000 | 3.82e-03 |
| run_largebatch_oc_aligned | oc_aligned_bs4096_seed43 | oc | 43 | 4096 | 4.5e-3 | 300 | 7000 | 3.68e-03 |
| run_largebatch_oc_aligned | oc_aligned_bs4096_seed44 | oc | 44 | 4096 | 4.5e-3 | 300 | 7000 | 3.69e-03 |
| run_largebatch_confirm/oc_2048_confirm | confirm2048_seed233 | oc | 233 | 2048 | 4.5e-3 | 300 | 7000 | 3.72e-03 |
| run_largebatch_confirm/oc_2048_confirm | confirm2048_seed42 | oc | 42 | 2048 | 4.5e-3 | 300 | 7000 | 3.81e-03 |
| run_largebatch_confirm/oc_2048_confirm | confirm2048_seed43 | oc | 43 | 2048 | 4.5e-3 | 300 | 7000 | 3.77e-03 |
| run_largebatch_confirm/oc_2048_confirm | confirm2048_seed44 | oc | 44 | 2048 | 4.5e-3 | 300 | 7000 | 3.91e-03 |
| run_largebatch_followup/noc_4096_recipe | noc_diag4096_seed233_lr5e-3_wu400_ts5000 | noc | 233 | 4096 | 5e-3 | 400 | 5000 | 4.76e-04 |
| run_largebatch_followup/noc_4096_recipe | noc_diag4096_seed233_lr5e-3_wu400_ts7000 | noc | 233 | 4096 | 5e-3 | 400 | 7000 | 1.42e-04 |
| run_largebatch_followup/noc_4096_recipe | noc_diag4096_seed233_lr6e-3_wu400_ts5000 | noc | 233 | 4096 | 6e-3 | 400 | 5000 | 3.87e-04 |
| run_largebatch_followup/noc_4096_recipe | noc_diag4096_seed233_lr6e-3_wu400_ts7000 | noc | 233 | 4096 | 6e-3 | 400 | 7000 | **1.75e-01 ⚠ ep29 发散** |
| run_largebatch_followup/oc_2048_stability | oc_diag2048_seed233_lr4e-3_wu300_ts7000 | oc | 233 | 2048 | 4e-3 | 300 | 7000 | 3.88e-03 |
| run_largebatch_followup/oc_2048_stability | oc_diag2048_seed233_lr4e-3_wu400_ts7000 | oc | 233 | 2048 | 4e-3 | 400 | 7000 | **1.32e-01 ⚠ ep10 发散** |
| run_largebatch_followup/oc_2048_stability | oc_diag2048_seed233_lr4.5e-3_wu300_ts7000 | oc | 233 | 2048 | 4.5e-3 | 300 | 7000 | 3.72e-03 |
| run_largebatch_followup/oc_2048_stability | oc_diag2048_seed233_lr4.5e-3_wu400_ts7000 | oc | 233 | 2048 | 4.5e-3 | 400 | 7000 | 3.83e-03 |

合计 30（noc 10 + oc 20）。其中 **4 个为记录在案的失败/发散案例**（best_loss ≫ 0.1）：`oc_bs2048_seed233`、`noc_diag4096…lr6e-3_wu400_ts7000`、`oc_diag2048…lr4e-3_wu400_ts7000`——仍保留 rollout 产物作为崩溃/发散样本。

---

## 4. 实验目的与各报告结论（标注来源）

findings 文档在各 suite 下的 `*.md`（8 份）+ `docs/experiment_command_matrix.md`。

**大 batch 配方（`docs/experiment_command_matrix.md` §7）：**
- bs2048 配方：`lr5e-3, min_lr1e-4, warmup300, total_steps7000, epochs200`（仓库声明的默认大 batch 配方）。
- bs4096 配方：`lr6e-3, min_lr1e-4, warmup400, total_steps5000, epochs300`。
- 比较规则：只有当 4096 在不损失 heldout/rollout 指标的前提下改善 wall-clock 才优先。

**noc（`largebatch_noc_report.md`/`_summary.md`）：** "bs2048 是 noc 更好的默认大 batch 配置"。4096 唯一优势是 wall-clock（训练时间 0.784×），但误差系统性变差：best val loss 2.16×，heldout 30s pos median 1.50×（0.1515→0.2265 m），resampled 60s pos median 1.52×（0.3451→0.5260 m）等；求解器失败/无效预测/SO(3) 违例均为 0（非数值不稳定）。"This is not a borderline result."

**oc（`largebatch_oc_report.md`/`_paired_summary.md`）：** oc "不呈现 noc 的模式"。raw all-seed 聚合（让 4096 看似仅 2048 误差的 0.03–0.11×）被**单个 bs2048 seed233 灾难性训练崩溃**主导（epoch 7 Test→inf，best 卡在 0.457，仅 246/7000 步）。在**稳定子集 seed42/43/44**上 2048 vs 4096 接近：best val 1.02×、heldout 30s pos median 1.17×，但 resampled 60s pos median 0.939×（4096 略好）、60s p95 基本持平。结论：要安全默认就用 4096；只比成功 run 则 2048 仍有竞争力，但不足以抵消稳定性风险。

**followup 稳定性扫描（`largebatch_followup.md`）：** noc_4096 最佳稳定配置 `bs4096, lr0.005, wu400, steps7000`（best 1.424e-04）；`lr0.006, wu400, steps7000` 失败（ep29 发散）。oc_2048 最佳 `bs2048, lr0.0045, wu300, steps7000`（best 3.719e-03）；`lr0.004, wu400` 失败（ep10 发散）。结论：**驱动发散的是 lr/warmup 选择，而非单独的 batch size**。

**oc confirm（`oc_confirm_summary.md`/`oc_confirm_vs_baselines.md`）：** 调好的 `bs2048, lr0.0045, wu300, steps7000` 在全 4 seed 上"修复了 seed233 崩溃"（全部达标，零非有限事件，mean best 0.003802）。相对旧 4096：优化质量/局部 vel RMSE/heldout 30s median/completion/发散率更好，但 resampled 60s pos median/p95 更差、慢约 1.22×。报告称这是"配方比较，非纯 batch-size 单变量比较，尚不能宣称 4096 过时"。

**oc aligned（`oc_aligned_vs_confirm.md`）：** 严格对齐 `bs4096, lr4.5e-3, wu300, steps7000` vs confirm2048。aligned4096 四 seed 全稳定，mean best_test 0.98×、resampled 60s pos median 0.923×、rot median 0.879× 更好；confirm2048 在 heldout 30s median/completion/发散率略优、且快约 30%。结论："去掉配方失配后，4096 不再需要旧的稳定性论据；权衡变为 60s 精度 vs 效率/发散裕度"。

（另有 `docs/current_ocean_performance_analysis.md`，是**海流场景物理/架构退化分析**——螺旋桨入流 `‖nu‖`→`nu_r[0]`、B_net 条件化、D/J 海流条件化、actuator loss——不是 batch 配方报告，故不在此归入配方数字。）

---

## 5. 与 g3_5_5 / 论文的关系

- `g3_5_4` 是早期**训练稳定性 / 大 batch 配方 + noc-vs-oc** 研究，只训练 `ph_se3_full` 一个模型；
- `g3_5_5` 在其之后收敛到**结构化 SE(3) port-Hamiltonian + 海流(oc) 主线**，启用 10 模型族、profile 噪声、规范化 catalog，**不再训练 noc**；
- 两代 oc 数据集不同（`89c80d68`/seed42 vs `d0be9434`/seed23），结果不可直接跨代比较；
- 当前论文（`g3_5_5/paper/`）§8 证据来自 g3_5_5 的 Phase-1A（B 区），**不使用 `g3_5_4` 的 noc/oc 结果**。
