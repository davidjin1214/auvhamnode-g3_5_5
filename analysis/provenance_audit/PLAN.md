# Provenance 对齐与 fragility 复现性调查

调查目标：解释 catalog `phnode_full clean seed42/46` 60s ~11 m 与 cleanrun v1 ~0.96 m 之间的量级 gap，并判定 `phnode_full clean` 是否仍存在 seed42/46 catastrophic failure。

调查分支：`provenance-audit-phnode_full`（派生自 `cx-noise-v3`，`main` 不动）。

## 待确认决定

`.gitignore` 调整方案：已选 **A**（2026-05-12 用户确认）。`.gitignore` 已追加 `!analysis/provenance_audit/`，audit 子树纳入版本控制。

## Phase 1 结果（2026-05-12 已完成）

- A42 / A46 / C42–C46 run_uid + 关键元数据已锁定，见 `phase1_static/run_lock.md`
- cleanrun v1 训练 invocation 链已抓取并核对，见 `phase1_static/cleanrun_train_invocation.txt`
- 静态对齐表已建立，见 `phase1_static/diff_matrix.md`
- 重大发现：
  1. catalog 11 m mean 完全由 seed46 ~48 m 灾难（best_epoch=21, best_loss=0.27）+ seed42 ~5 m（best_loss=0.02）驱动；其它 seed < 1.5 m
  2. catalog A46 训练在 epoch 26 之后完全发散（"no successful training batches"，275 行 warning），best 模型卡在 epoch 21 一个浅薄状态
  3. cleanrun v1 同 seed46 收敛到 epoch 250 / best_loss=0.004，差 67× — 训练稳定性差异是 gap 的核心解释
  4. dataset / 显式超参 / noise / wrapper 全部完全一致 → gap 必由非超参代码变化或环境差异引起
  5. catalog `run_inventory.csv` 仍缺 `code_revision`，无法直接 commit-level diff
- Phase 3 Setup A 可缩小为只重跑 seed46 clean 一个 run（< 1 min）做立即判定

## Phase 1 — 静态 provenance 对齐（不消耗算力）

### 1.1 锁定要比对的具体 run

```bash
grep -E "main_phnode_full_seed4[26]" analysis/oc_data_catalog/run_inventory.csv
grep -E "phnode_full.*clean.*seed4[26]" analysis/oc_data_catalog/canonical_rollout_summary_long.csv | head
```

得到两条 catalog `run_uid` + 磁盘路径（A42 / A46）。

### 1.2 抓 cleanrun v1 训练 cfg（云端 g3_5_7 → 本地最低限度证据）

```bash
jq -r '.cells[] | select(.cell_type=="code") | .source[]' \
  notebook/phase1a_oc_v4lite_formal_workflow_completed.ipynb \
  | grep -E "train_auv_hamnode|--num_epochs|--lr|--noise_profile|phase1a_oc_v4lite_utils" \
  > analysis/provenance_audit/phase1_static/cleanrun_train_invocation.txt

grep -nE "def .*(build|make|construct).*(cfg|config)|num_epochs|learning_rate|warmup|batch_size|precision" \
  scripts/phase1a_oc_v4lite_utils.py \
  > analysis/provenance_audit/phase1_static/cleanrun_default_cfg_refs.txt

cp checkpoints/<A42_path>/config.json analysis/provenance_audit/phase1_static/catalog_A42_config.json
```

### 1.3 输出静态 diff 表 → `analysis/provenance_audit/phase1_static/diff_matrix.md`

字段：dataset filename/hash, num_traj/blocks/ocean_current, num_epochs, LR+scheduler, batch_size, optimizer, ODE solver+rtol/atol, precision, noise_profile_train (两侧应都为 clean), noise_warmup/ramp/mix_ratio, best ckpt 选择口径, git commit at training time (catalog 不存 → 标 unknown)。

## Phase 2 — Aggregation 口径对齐（不消耗算力）

### 2.1 提取两侧聚合公式

```bash
grep -nE "median|mean|aggregate|groupby" scripts/phase1a_oc_v4lite_utils.py
grep -nE "median|mean|aggregate|groupby|final_position_error" scripts/build_oc_data_catalog.py rollout_benchmark_reporting.py
```

### 2.2 用 cleanrun 聚合脚本重算一个 catalog run 的 `Pos Median`

判读：
- ≈ 11 m → 聚合不是 gap 来源，进 Phase 3
- ≈ 0.7 m → 聚合解释了部分 gap，先统一聚合口径

## Phase 3 — 受控复现实验（消耗算力，仅在 Phase 2 未消解时进入）

### Setup A：用 catalog 旧 cfg + 旧 dataset 在当前 main 上重训

```bash
python train_auv_hamnode.py --load-config checkpoints/<A42>/config.json \
  --save_dir checkpoints/audit_setup_A_phnode_full_clean_seed42 --seed 42
```

判读：
- ≈ 11 m → fragility 在 current main + old cfg 下复现 → 进 Setup C（config swap）
- ≈ 0.7 m → fragility 已消失 → 进 Setup B（git bisect）

### Setup B：git bisect

bisect 区间：catalog 训练时 commit ↔ 当前 HEAD。强嫌疑：`4ee1860 / 8e510aa / 5bf9b35`。

用 worktree 隔离：`git worktree add ../audit-bisect cx-noise-v3`。

### Setup C：cfg/dataset 单变量 swap

逐项 swap：epochs/LR → dataset → batch_size/precision → ckpt 选择口径。

### 算力预算

每 run < 1 min（clean 路径，主干已优化）。Setup A 3 runs + bisect ~10 runs + Setup C ~8 runs，总 < 一夜 sweep。

## Phase 4 — 归因决策与文档落盘

无论 Phase 3 走哪一支，产出：

- `docs/provenance_audit_phnode_full_clean.md`（Phase 1 diff + Phase 2 聚合 + Phase 3 结论）
- `EXPERIMENT_PROGRESS_TRACKER.md §7` 增订：§7.3/§7.4/§7.5 逐条标 current/stale/superseded
- `scripts/build_oc_data_catalog.py` + `docs/oc_data_catalog_dictionary.md`：`run_inventory.csv` 加 `code_revision` 字段
- catalog 受影响行打 `evidence_status = stale`
- `docs/phase1_realistic_validation_plan.md §3.3` 末尾追加 "Phase-1A 决策依赖 fragility 复现"

若 fragility 已消失，追加 WP-Frag：在新基线下重跑 phnode_full / phnode_qforce / ablate_no_lift / ablate_no_mass_prior × clean / nominal_train × seeds 42-46 的最小重训。

## 复核要点（来自前次分析二次复核）

- catalog vs cleanrun gap 在报告 §12 直接给的是 11×（eval 协议不一致），匹配条件下接近 16×
- §7.3 / §7.5 直接依赖 fragility；§7.4 部分依赖（结构耦合还有 ablate_no_lift seed44 与 ablate_no_mass_prior 受益两条独立证据）
- 三个 v4-lite 加速 commit（129570a / 41fd824 / e439ed5）在代码层不触及 clean / iid 训练路径 → 代码层不能解释 gap
- `ablate_no_mass_prior seed46` v4-lite 退化为 32%（per-seed），非 27%
- `ablate_no_lift seed43 clean` 异常：best_epoch=19, best_loss=0.2169, 60s rollout ≈ 44 m，必须重跑
- catalog `run_inventory.csv` 字段无 `code_revision` / `git_hash`

## 风险与边界

- catalog A42 原始 config.json 可能不全 → 用 audit 文档显式记录缺口
- cleanrun v1 dataset 是云端 regenerate → 本地用 data_collection.py 同参数重生成
- CUDA 非确定性 → 至少 2 个 replicate 看方差
- 调查途中发现的无关 bug → 不在同一 PR 里修
