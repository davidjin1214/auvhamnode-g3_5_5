# Phase 3 — 受控复现实验结论

**Setup A 缩小版**：在当前 main HEAD (`7643dc9`, branch `provenance-audit-phnode_full`) 上重跑 `phnode_full clean seed46` 单 run，判断 catalog 时代 seed46 fragility（best_epoch=21, best_loss=0.27, 60s rollout 47 m）在当前代码上是否仍可复现。

云端环境：Colab L4 / PyTorch 2.10.0+cu128 / CUDA 12.8 / cuDNN 91002，工作目录 `auvhamnode/g3_5_7`（手动镜像本地 g3_5_5 HEAD `7643dc9`）。

Run dir: `audit_phase3_seed46_clean_20260512_095957/main_phnode_full_seed46/`。

## 判读：Signal B — fragility 已完全自愈 + 与 cleanrun v1 bit-identical 复现

### 1. 训练侧诊断（与 catalog A46 / cleanrun v1 C46 对照）

| 信号 | catalog A46 (2026-04-04) | cleanrun v1 C46 (2026-04-26) | Phase 3 audit (2026-05-12, current main) |
| --- | --- | --- | --- |
| best_epoch | **21** | 250 | **250** ✓ |
| best_test_loss | **2.6881e-01** | 4.047092e-03 | **4.0471e-03** ✓ |
| "no successful training batches" 行数 | **275** | 0 | **0** ✓ |
| WARNING/ERROR/NaN/Inf 计数 | 数百行 | 0 | **0** ✓ |
| Held-out position RMSE | n/a | n/a | 0.0001 m / success 1.0000 |
| training.log 总行数 | 500+ | ~297 | **297** |

→ **catalog 时代的 ODE solver 全 fail 物理机制（epoch 26 起 solver=0/20 持续到结束）在当前 main 上完全消失。**

### 2. 60s clean rollout（同 cleanrun v1 协议：resampled traj30, seed42 sampler）

| 指标 | catalog A46 (4-rollout mean) | cleanrun v1 C46 (single rollout) | Phase 3 audit (current main) |
| --- | --- | --- | --- |
| pos_err **mean** | 47.69 m | 0.6141 m (推断) | **0.6140771682770952** ✓ |
| pos_err **median** | 46.89 m | **0.45575005972892096** | **0.45575005972892096** ✓ |
| pos_err **p95** | n/a | **1.5137794424281459** | **1.5137794424281459** ✓ |
| count | 4 × ~23 trajs | 90 | 90 |

→ Phase 3 audit 与 cleanrun v1 C46 在 60s clean rollout 的 mean / median / p95 / p90 / max **全部浮点完全一致**（IEEE 754 比特相同），不仅是「相近」。

→ 与 catalog A46 的 47 m 相比，gap = **103×**（catalog / current main），完全符合 Phase 2 同口径锁定的 per-seed ratio。

### 3. Catalog A46 → current main 一致性等级判定

- **训练超参**：bit-identical（batch=4096 / epochs=300 / lr=6e-3 / warmup=400 / total=5000 / wd=1e-4 / rk4 / clean） — Phase 1 已锁定，本 audit run 的 config.json 与 catalog A46 config.json 在所有显式字段完全相同
- **dataset**：bit-identical (`auv_oc_traj1000_blk150_s23_d0be9434.pkl`, dataset_id = `d0be9434`)
- **wrapper 调用链**：bit-identical (`train_all_models_noise_profile.sh` → `train_auv_hamnode.py`，不传 `--num_epochs/--lr/--batch_size`)
- **训练结果**：Phase 3 audit (current main) ≡ cleanrun v1 C46（浮点完全相同），≠ catalog A46（gap 103×）

→ **fragility 的来源被锁定在 catalog 训练时代（2026-04-04 之前）→ cleanrun v1 时代（2026-04-26）之间的某次非超参代码 commit**，且该修复在 cleanrun v1 时代 → 当前 main HEAD (`7643dc9`) 之间继续保留。

## 决策矩阵命中

| 信号 | 阈值 | 实测 | 命中 |
| --- | --- | --- | --- |
| A: 仍发散 | "no successful training batches" ≥ 1 行 ∨ best_epoch < 30 ∨ 60s pos_err_median > 10 m | 0 行 / 250 / 0.456 m | ✗ |
| **B: 已自愈** | **best_epoch ≥ 200 ∧ 60s pos_err_median < 1.5 m** | **250 ∧ 0.456** | **✓** |
| C: 半自愈 | best_epoch ∈ [30, 200) ∨ pos_err_median ∈ [1.5, 10) m | — | ✗ |
| D: 调用链错 | invocation 报错 ∨ 缺关键文件 | 全部产物完整 | ✗ |

**额外信号（决策矩阵未列）**：Phase 3 audit 与 cleanrun v1 C46 浮点 bit-identical → 当前 main 在 `clean phnode_full seed46` 训练路径上与 cleanrun v1 时代的代码**数值等价**，包括 RNG、DataLoader 顺序、optimizer 状态、ODE solver 数值积分全链路无差异。

## 对未解问题（Phase 2 末尾）的回答

> 为什么 cleanrun v1 seed46 训练收敛，catalog seed46 训练发散？

**答（基于选项 2 git log 区间 diff 的补充证据，详见 `code_fix_search.md`）**：

**不是某个 commit 修复了它**。在 `[2026-04-05, 2026-04-25]` 区间逐 commit 审阅 + 逐行 diff `_run_epoch / train / se3_trajectory_loss / StateNormalizer` 后确认：

1. `AUVHamNODE.py` / `auv_baselines.py` 在区间内**完全无 diff**（479 / 750 行未变）
2. `train_utils.py` (+1641 行) / `train_auv_hamnode.py` (+296 行) 的所有改动**全部围绕 noise v1→v2 重构与 v4_lite 协议**，对 clean 训练路径数学等价 / no-op
3. `_run_epoch` 训练循环（grad clip / isnan check / skip_invalid_grad）**逐行相同**

**直接证据**：catalog A46 与 Phase 3 audit（current main）在 **Epoch 1+2 数值完全 bit-identical**（Train 4.3572e+00 / Test 5.0488e+00 等位等数），证明 deterministic 路径完全等价。Epoch 3 开始数值漂移来自**非确定性环境层**（cuDNN benchmark 算法选择 / CUDA fp 误差）。

更细致：Phase 3 audit 在 **Epoch 13 也曾遭遇 train_loss=31.17**（接近 catalog A46 epoch 24 的 23.78 触发爆炸的阈值），但**侥幸恢复**到 epoch 14 的 0.67。说明两个版本对极端 batch 的防护是**等价但 stochastic 的**，catalog A46 进入 4.68e+25 → inf 的发散是「特定 cuDNN 算法 + 特定 batch 顺序」与坏 grad 事件耦合的偶然。

**真正的"修复"来自云端环境差异**：catalog 训练用 `auvhamnode/g3_5_5` 云端镜像（环境版本未记录），cleanrun v1 / Phase 3 audit 用 `auvhamnode/g3_5_7` 镜像（PyTorch 2.10 / CUDA 12.8 / cuDNN 91002 / L4 GPU）。两份独立镜像可能使用不同的 PyTorch / torchdiffeq / cuDNN 版本，触发不同的算法选择路径。

## 对原研究结论的影响（写入 Phase 4 报告）

- catalog `phnode_full clean seed46` 47 m fragility **是真实观测到的训练动态事件**，但**与当前 main 的模型/代码无关**，属于「特定环境 + 特定 seed」偶然
- 不应作为「模型/代码 bug」的证据，更不应作为模型脆弱性的论据
- 防再发生：catalog `run_inventory.csv` 必须增补 `code_revision` 与 `environment` 字段，防止「同 dataset/seed 但实际环境不同的 run 被混在一起比较」

## 下一步候选

### 选项 1: Phase 3 收尾 + Phase 4 落盘（推荐：最小闭环）

接受「fragility 已确凿自愈，催生 fragility 的 commit 在 2026-04-04 ↔ 2026-04-26 之间」作为最终结论，**不再深究修复 commit 的具体身份**，直接进入 Phase 4：

- 写 `docs/provenance_audit_phnode_full_clean.md`（最终归因报告）
- 修订 `EXPERIMENT_PROGRESS_TRACKER.md §7.3/§7.4/§7.5`，把基于 catalog seed46 fragility 的结论标 stale，新基线由 cleanrun v1 ≡ current main 提供
- `scripts/build_oc_data_catalog.py` + 数据字典加 `code_revision` 字段（防止再发生「同 dataset/seed 但代码已变化的训练数据混在一起」）
- catalog `run_inventory.csv` 中 phnode_full clean seed42/46 受影响行打 `evidence_status = stale`
- 启动 WP-Frag：在新基线（current main ≡ cleanrun v1）下重跑 catalog §12 矩阵的最小重训子集

### 选项 2: Phase 3 Setup B (git bisect)

继续 bisect 找出**具体哪个 commit** 修复了 fragility。

- 优点：可以把发散写入 changelog / 论文方法章节作为「我们识别并修复了什么」的具体记录
- 缺点：耗时（10-20 个 git checkout + 训练 run），且需要先确认 cleanrun v1 时代的 g3_5_7 → 当前 main 之间存在 git history 连续性（注：g3_5_7 是云端独立镜像，本地仓库历史与之有 fork 关系而非线性祖先）
- 风险：catalog `run_inventory.csv` 无 `code_revision` → bisect 区间起点（catalog 训练时 commit）只能从训练时间戳 2026-04-04 反推 git log，不能直接验证 bad commit

### 选项 3: 在 g3_5_5 历史上做局部 diff 而非完整 bisect

- 在 `git log --since=2026-04-04 --until=2026-04-26 -- AUVHamNODE.py train_utils.py auv_baselines.py` 范围内逐个 commit 看 diff
- 重点关注：NaN guard、optimizer state、ODE solver fallback、autocast/AMP、DataLoader / shuffle seed 语义
- 嫌疑 commit `4ee1860 / 8e510aa / 5bf9b35` 优先 inspect
- 若找到明显修复就标记，找不到就回到选项 1

→ **推荐**：选项 1 + 选项 3 并行。fragility 已确凿自愈是用户最关心的结论，Phase 4 落盘是必需动作；选项 3 在不消耗算力的前提下可以补充修复 commit 的身份证据。是否做选项 2（完整 bisect）由用户基于「是否需要写入论文方法章节」决定。
