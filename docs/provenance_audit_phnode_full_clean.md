# Provenance Audit — phnode_full clean train (catalog vs cleanrun v1)

**日期**：2026-05-12
**调查分支**：`provenance-audit-phnode_full`（派生自 `cx-noise-v3`）
**当前 main HEAD**：`7643dc9`
**完整调查目录**：`analysis/provenance_audit/`

## 0. 调查动机

OC 实验报告 §12 和 `analysis/oc_data_catalog/` 中 `phnode_full clean seed42/46` 60s rollout 给出 **~11 m** 的 5-seed mean，与 `docs/phase1a_oc_v4lite_cleanrun_v1_report.md` 报告的 cleanrun v1 phnode_full clean 的 **0.96 m** 存在 ~11× gap。

由于二者 dataset id 相同（`d0be9434`）、wrapper 调用链相同、显式训练超参相同、noise profile 都是 clean，无法在文档层面直接消解差异 → 启动四阶段 provenance audit。

## 1. 调查路线（四阶段，全部已完成）

| 阶段 | 目的 | 算力 | 状态 |
| --- | --- | --- | --- |
| Phase 1 | 静态 provenance 对齐：catalog A42/A46 vs cleanrun v1 C42–C46 的 dataset / 超参 / wrapper / 训练 log 逐项 diff | 0 | 完成（`phase1_static/`） |
| Phase 2 | 聚合口径对齐：mean vs median、clean vs nominal_eval、4-rollout-per-seed vs 1-rollout-per-seed | 0 | 完成（`phase2_aggregation/`） |
| Phase 3 | 受控复现：在 current main 上重跑 `phnode_full clean seed46` 单 run，判读 fragility 是否仍可复现 | < 5 min Colab L4 | 完成（`phase3_retrain/`） |
| Phase 4 | 归因决策与文档落盘 | 0 | 本文件 |

## 2. 关键发现汇总

### 2.1 Aggregation gap（Phase 2）

报告 §12 表面 11 m vs 0.96 m 的 11× gap **包含三层口径不一致**：

| 维度 | catalog 一侧 | cleanrun v1 一侧 |
| --- | --- | --- |
| eval profile | clean | **nominal_eval** |
| stat | **mean** | median |
| per-seed rollout 数 | 4 | 1 |

**同口径对齐（clean+clean, 60s, 5-seed mean of pos_err_median）**：

| 来源 | seed42 | seed43 | seed44 | seed45 | seed46 | 5-seed mean | 相对 cleanrun v1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| catalog | 4.49 | 0.58 | 0.45 | 0.80 | 46.89 | **10.64 m** | **15.7×** |
| cleanrun v1 | 0.61 | 0.68 | 1.21 | 0.43 | 0.46 | **0.6767 m** | 1× |

**同口径下仍有 15.7× 的真实 gap**，且**完全由 catalog seed42 (7.3×) + seed46 (103×) 训练异常驱动**。seed43/44/45 在 catalog 上甚至比 cleanrun v1 更好或持平。

### 2.2 Catalog A46 训练发散的物理机制（Phase 1）

来自 `checkpoints/sweep_oc_all/sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47/main_phnode_full_seed46/training.log`：

- Epoch 1–23 训练正常收敛，best loss 在 epoch 21 达到 0.27（远高于 cleanrun v1 的 0.004）
- **Epoch 24**：train_loss 突跳到 **2.38e+01**（gradient explosion 前兆）
- **Epoch 25**：train_loss **4.68e+25**, Test inf, SO3 violation 4.55e+11, Fail(train/test) 9/6, **NaN 出现**
- **Epoch 26 起**到 epoch 300 结束：每个 epoch 全部 20 个训练 batch 的 ODE solver 全 fail（solver=0/20, pred=20/20, grad=0/20），整个文件 **275 行 "no successful training batches" warning**
- 模型权重在 epoch 21-25 间被毁坏，optimizer state 被天文数字 grad 污染，无法恢复

### 2.3 当前 main 上 fragility 已不复现（Phase 3 Setup A）

云端 Colab L4 / PyTorch 2.10.0+cu128 / CUDA 12.8 / cuDNN 91002，本地 git commit `7643dc9`，单 run < 5 min：

| 指标 | catalog A46 (2026-04-04) | Phase 3 audit (2026-05-12, current main) |
| --- | --- | --- |
| best_epoch | **21** | **250** |
| best_test_loss | **2.6881e-01** | **4.0471e-03** |
| "no successful training batches" 行数 | **275** | **0** |
| 60s clean rollout pos_err_median | **46.89 m** | **0.4558 m** |
| 60s clean rollout pos_err_mean | **47.69 m** | **0.6141 m** |

**额外强信号**：Phase 3 audit 与 cleanrun v1 C46 在 60s clean rollout 的 mean / median / p90 / p95 / max 全部**浮点 bit-identical**（IEEE 754 比特相同），best_epoch / best_test_loss 也一致 → current main 与 cleanrun v1 时代代码在 clean 训练路径上**数值等价**。

### 2.4 修复来源 = 环境，不是 git commit（Phase 3 Setup B 替代：git log 区间 diff）

在 `[2026-04-04, 2026-04-26]` 区间逐 commit 审阅 + 逐行 diff `_run_epoch / train / se3_trajectory_loss / StateNormalizer / DataLoader 初始化`：

1. **`AUVHamNODE.py` 与 `auv_baselines.py` 在区间内完全无 diff**（479 / 750 行未变）
2. **`train_utils.py` (+1641 行)** 与 **`train_auv_hamnode.py` (+296 行)** 的所有改动**全部围绕 noise v1→v2 重构与 v4_lite 协议**，对 clean 训练路径数学等价 / no-op
3. **`_run_epoch` 训练循环（grad clip / isnan check / skip_invalid_grad）逐行相同**
4. **`se3_trajectory_loss` 在 `frame_weights=None` 时数学等价**于 a2ca101 版本
5. **`StateNormalizer.from_dataset`** 计算逻辑两版本逐行相同

**Smoking-gun**：catalog A46 与 Phase 3 audit 在 **Epoch 1+2 完全 bit-identical**（Train 4.3572e+00 / Test 5.0488e+00 等位等数），证明 deterministic 路径完全等价。Epoch 3 开始数值漂移，来自**非确定性环境层**（cuDNN benchmark 算法选择 / CUDA fp 误差累积）。

更微妙的证据：Phase 3 audit 在 Epoch 13 也曾遭遇 train_loss=31.17（接近 catalog epoch 24 触发爆炸的 23.78），但**侥幸恢复**到 epoch 14 的 0.67。这说明 **grad_clip(max_norm=1.0) + isnan check 在两个版本中是等价但 stochastic 的防护**，catalog A46 进入 4.68e+25 → inf 是「特定 cuDNN 算法选择 + 特定 batch 顺序 + epoch 24 极端 batch」三者耦合的偶然，不是模型/代码缺陷。

**真正的"修复"来自云端环境差异**：catalog 训练用 `auvhamnode/g3_5_5` 云端镜像（PyTorch / CUDA / cuDNN 版本未记录），cleanrun v1 / Phase 3 audit 用 `auvhamnode/g3_5_7` 镜像（PyTorch 2.10 / CUDA 12.8 / cuDNN 91002 / L4 GPU）。两份独立镜像可能使用不同的依赖版本，触发不同的算法选择路径。

## 3. 最终归因（一句话）

> Catalog 时代 `phnode_full clean seed46` 60s rollout 47 m 与 best_loss 0.27 是一次**与云端环境非确定性耦合的灾难性梯度事件**（epoch 24 train_loss 23.78 → epoch 25 4.68e+25 → epoch 26 起 ODE solver 全 fail），不是模型架构缺陷、不是 git 代码 bug、不是 dataset 问题，也不是显式超参问题。当前 main HEAD (`7643dc9`) 在云端 `g3_5_7` 镜像（PyTorch 2.10 / CUDA 12.8 / cuDNN 91002）下重跑该 run 与 cleanrun v1 C46 **浮点 bit-identical** 收敛到 best_loss=4.05e-03 / 60s pos_err_median=0.4558 m，fragility 在该环境下不复现。

## 4. 对原有研究结论的影响

### 4.1 受影响的引用结论（来自 `EXPERIMENT_PROGRESS_TRACKER.md §7`）

| 编号 | 原结论 | 修订后状态 | 说明 |
| --- | --- | --- | --- |
| §7.3 | `main/phnode_full` 有强 stable cluster，但存在真实 bad-outlier seeds：`42` 与 `46` | **stale** | seed42/46 的 "bad outlier" 在 cleanrun v1 / current main 上均已消失。原结论受 catalog 时代未记录的环境因素影响，不能继续作为模型脆弱性论据 |
| §7.4 | noisy training 不是普适增强；它与模型结构强耦合 | **current**（部分需要重新背书） | 「noisy training 对 phnode_full 的主要收益是修复 seed46」依赖 §7.3 — seed46 fragility 已消失后，需以 ablate_no_lift seed44 / ablate_no_mass_prior 退化等独立证据重新背书该结论 |
| §7.5 | noisy training 对 `phnode_full` 的主要收益是修复 `seed46`，不是普遍降低全部 seed 的误差 | **stale** | catalog 时代 seed46 fragility 是环境偶然，不存在「待修复的脆弱性」；该结论的因果链不再成立 |
| §7.6 | `ablate_no_mass_prior` 是当前最稳定受益于 noisy training 的结构模型 | **current** | 不依赖 phnode_full clean fragility，独立证据链有效 |
| §7.2 | 在 PHNODE family 内，clean all-seed 下 `ablate_no_lift` 当前最稳 | **需 recheck** | catalog `ablate_no_lift seed43 clean` 也存在异常（best_epoch=19, best_loss=0.22, 60s ≈ 44 m），与 seed46 fragility 同环境 → 同样可能是 catalog 时代偶然，应在 cleanrun v1 ≡ current main 重训后重新判断 |

### 4.2 受影响的 catalog 数据

- `analysis/oc_data_catalog/canonical_rollout_summary_long.csv` 中 catalog `phnode_full clean seed42/46` 的所有 4 个 rollout 行：5-seed mean 由这些行驱动到 11 m，其中 seed46 (46.89 m median) 与 seed42 (4.49 m median) 都受环境影响
- `analysis/oc_data_catalog/run_inventory.csv` 中对应 run_uid 应增补 `evidence_status = stale_environment_drift` 标记
- 论文/报告中由 §12 表 11 m 引出的所有讨论都应改用同口径锁定后的 cleanrun v1 ≡ current main 基线 0.6767 m

## 5. 防再发生措施

### 5.1 catalog schema 增补字段（强制）

`scripts/build_oc_data_catalog.py` 生成 `run_inventory.csv` 时应额外记录：

| 新字段 | 含义 | 来源 |
| --- | --- | --- |
| `code_revision` | training 时 `git rev-parse HEAD`，云端镜像若不是 git 仓库则人工写入对应本地 commit hash | 训练 wrapper 写入 `<run_dir>/_audit_meta/code_revision.txt`，catalog 构建时读出 |
| `environment_fingerprint` | `python / torch / cuda / cudnn` 版本和 GPU 型号 | 训练 wrapper 写入 `<run_dir>/_audit_meta/environment.txt`，catalog 构建时读出 |
| `evidence_status` | `current` / `stale_environment_drift` / `stale_code_drift` / `superseded_by_<run_uid>` 等 | 人工标记，可选 |

落地工单：写入 `docs/oc_data_catalog_dictionary.md` 的 schema 表，更新 `scripts/build_oc_data_catalog.py` 在下次 catalog 重建时自动注入这些字段。

### 5.2 训练 wrapper 增补

`scripts/train_all_models_noise_profile.sh`（或下游 `train_auv_hamnode.py`）应在写 `config.json` 时同时写：

- `<run_dir>/_audit_meta/code_revision.txt`：`git rev-parse HEAD` 输出 + `git diff HEAD --stat`
- `<run_dir>/_audit_meta/environment.txt`：`python --version / pip show torch / nvidia-smi -L` 输出

Phase 3 audit notebook（`notebook/phase3_provenance_audit_seed46_replay.ipynb`）已示范该模式（Cell 10 + Cell 20），可作为参考。

### 5.3 训练循环 deterministic 模式（可选 ablation）

`train_utils.py` 可增加 `--cudnn_deterministic` 选项设置 `torch.backends.cudnn.deterministic = True` + `torch.backends.cudnn.benchmark = False`，但会牺牲 ~20% 训练速度。建议在 fragility 调查或 paper 重复性实验时启用，日常训练保持默认（非 deterministic）。

## 6. WP-Frag：在新基线下重训的最小工单（可选启动）

若需要把 fragility 完全消除作为论文方法贡献，启动 WP-Frag 子工作包：

- 在 current main ≡ cleanrun v1 的环境下，重跑 catalog §12 的核心矩阵：
  - phnode_full / phnode_qforce / ablate_no_lift / ablate_no_mass_prior
  - × clean_train / nominal_train
  - × seeds 42–46
  - = 4 models × 2 protocols × 5 seeds = 40 runs
- 算力预算：≈ 40 × 5 min = 200 min（一夜 sweep 内完成）
- 输出：新的 `analysis/oc_data_catalog/` + 新的报告章节，把所有 catalog 时代结论刷新为新基线

WP-Frag 是否启动**不由本 audit 决定**，留作用户基于论文需求的决策。

## 7. 调查链 evidence map

- `analysis/provenance_audit/PLAN.md` — 全调查路线 + 各阶段结论摘要
- `analysis/provenance_audit/phase1_static/run_lock.md` — A42/A46 与 C42-C46 的具体 run_uid、磁盘路径、关键 log 数值
- `analysis/provenance_audit/phase1_static/diff_matrix.md` — 7 维静态 diff 表（dataset / 超参 / noise / wrapper / 训练 outcome / git commit）
- `analysis/provenance_audit/phase1_static/cleanrun_train_invocation.txt` — cleanrun v1 invocation 链抓取
- `analysis/provenance_audit/phase1_static/catalog_A42_config.json` / `catalog_A46_config.json` — catalog 训练 config 逐字段证据
- `analysis/provenance_audit/phase1_static/catalog_phnode_full_clean_5seed_60s.csv` — 5-seed × 60s 数值证据
- `analysis/provenance_audit/phase2_aggregation/aggregation_diff.md` — 聚合口径三层 mismatch + 同口径对齐 + per-seed ratio
- `analysis/provenance_audit/phase2_aggregation/same_stat_compare.csv` — per-seed catalog vs cleanrun v1 数值表
- `analysis/provenance_audit/phase3_retrain/findings.md` — Phase 3 audit 决策矩阵 + bit-identical 复现证据
- `analysis/provenance_audit/phase3_retrain/code_fix_search.md` — git log 区间 diff + epoch-by-epoch loss 轨迹对比 + 修复源归因
- `analysis/provenance_audit/phase3_retrain/audit_phase3_seed46_clean_20260512_095957/` — Phase 3 audit run 完整产物（training.log / config.json / best_model.pt / rollout_benchmark / _audit_meta）
- `notebook/phase3_provenance_audit_seed46_replay_completed.ipynb` — Colab 跑 Phase 3 audit 的 notebook，含 cell 输出
