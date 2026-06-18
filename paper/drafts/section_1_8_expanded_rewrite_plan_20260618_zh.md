# §1.8「实验结果与结构证据分析」扩展重写——完整实施方案（已与用户定稿）

- 生成时间：2026-06-18
- 分支：`provenance-audit-phnode_full`（非 main，提交用 `docs:` 中文前缀，本地不推送）
- 主稿：`paper/drafts/auvhamnode_thesis_chapter_zh.tex`
- 本次范围：**§1.8（`\label{sec:results-structure-evidence}`）**；§1.9（`\label{sec:discussion}`）/§1.10（`\label{sec:chapter-summary}`）仅因 §1.8 重构需同步交叉引用/机理回指时核对，不主动重写。
- 缘起：用户对上一轮"压缩/去种子腔"版 §1.8（commit `c204b83`）不满——四条硬问题：①展示太少 ②实验层层推进的脉络丢失 ③缺训练曲线等必要插图（全节仅 1 张柱状图）④开头三段防御腔。
- 上游冷启动 prompt：`paper/drafts/section_1_8_results_rewrite_prompt_v2.md`
- 关联记忆：`section1_8_1_9_rewrite_status`、`section8_evidence_merge_rule`、`section8_v4lite_axis`、`feedback_paper_writing_register`、`experiment_inventory`

> **本文件定位**：自包含实施蓝本。所有数值、口径、产物路径、脚本计划、决策均已落档并标注可追溯来源，未来执行/核验无需再反复查找原始数据与文档。

---

## 0. 七项已拍板决策（一览）

| # | 决策点 | 拍板结果 |
|---|---|---|
| ① | 图1 训练曲线叙事 | **诚实三要素**：结构化模型干净收敛 + 黑箱任务损失也收敛但 SO(3) 正交损失全程高(≈0.17，埋下 rollout 发散伏笔) + 无升力 seed43 训练期灾难梯度崩溃。**不画"黑箱训练发散"**（与事实不符） |
| ② | 图6 轨迹可视化 | **做只读式重跑**：轻量给 rollout 增加预测轨迹落盘（不改训练/数值/评估口径），重跑 1–2 个代表 run 出图 |
| ③ | 旧 `fig:s8-two-level-evidence`（唯一柱状图） | **改造为消融阶梯图**（remove-one 边际作用），去掉左侧几何对照（已由图2承载） |
| ④ | 更广 A 区稳定模型处理 + headline | A2 对角阻尼(≈4.2m,≈6×)、A4 执行条件化(崩溃)作为**可信证据并入主消融叙事**（**不单列"受限"、不标数据来源**，见 §0.1）；**headline 细化**：「能量核心最主导」限定在{能量核心/质量先验/升力}惯性-能量子先验组内依次递减，阻尼结构与执行条件化作为**另两条独立结构必要性** |
| ⑤ | 合并广义力 M2 | **补导出并并入主消融叙事**（A 区有 18 run，当前 supplement 未导出；同样不标来源） |
| ⑥ | §1.8 新结构 | 节首(去防御) + §1.8.1 训练收敛 + §1.8.2 几何稳定/结构必要性 + §1.8.3 能量核心与**完整消融阶梯**(并入对角阻尼/合并广义力) + §1.8.4 扰动鲁棒 + §1.8.5 协议等价 + §1.8.6 诊断（**取消独立"受限"小节**） |
| ⑦ | 新增插图 | 5 张新图（训练曲线/时域/扰动梯度/逐轨迹分布/诊断）+ 图6 轨迹；脚本放 `scripts/` + `paper/drafts/figures/<name>/` |

---

## 0.1 A 区/B 区 口径与写作原则（用户 2026-06-18 再次强调，统辖全文）

> 此原则**优先于**本文件其余处仍残留的"受限/镜像脚注"措辞与上游 prompt §B 的"受限证据"框架。凡冲突，以本节为准。

1. **A 区与 B 区无本质区别**：同数据集、模型代码逐行无 diff、同超参/噪声机制/评估口径，仅"不同时期跑、软件版本细微更新"之别。**不是两套实验，是同一套实验的两次运行。**
2. **采信规则**：A 区**总体平稳的结果即可信**，按定量结果采纳。**唯一要弃**的是"A 区极反常且 B 区不复现的单种子"（如完整模型 clean seed42/46 的灾难发散）——当偶发意外丢弃该异常值，**其余 A 区结果照常采信**。
3. **正式论文写法**：**不反复强调数据来源（A 区/B 区）、不铺陈个别种子异常、不加镜像脚注、不单列"受限证据"**。这些属实验报告口径，可留在本文件与 `docs/` 报告，不进正式论文正文。
4. **据此的结构后果**：更广 A 区稳定模型（对角阻尼、执行条件化、合并广义力）**并入主消融叙事**（§1.8.2/§1.8.3），与干净集同列；**取消独立"受限"小节**。退化倍数按单一 phnode_full 基线（0.68）统一计，或定性表述（"约 6 倍"）。
5. **仍守的诚信边界**（与本原则不冲突）：被 provenance 审计推翻的"噪声训练修复种子脆弱性"叙事仍❌不作正面结果；证据空白（P2/P3/Phase-1B/步长敏感性）不得虚构；qforce 的 catalog 0.57（与 B 区 3.76 矛盾、属②的极反常对象）丢弃，主表用 3.76。

---

## 1. 已独立核验的事实台账（每条带产物路径）

| # | 核验结论 | 证据/命令 |
|---|---|---|
| 1.1 | **主表 0.68 与 horizon 表 0.611 不矛盾**：同一产物两列——`posmed_mean_of_seed_medians`=0.6767≈0.68（主表用此）、`posmed_median_of_seed_medians`=0.6110。新图与四表同源 | `analysis/section8_current_evidence/{aggregate.csv,horizon_scenario_aggregate.csv}` 同行两列一致 |
| 1.2 | **horizon 表覆盖极全**：7 模型 × 3 训练协议(clean/iid/v4lite) × 4 评估 profile(clean/nominal_eval/degraded_eval/heading_biased_eval) × 4 场景(PRBS/CHIRP/OU/overall) × 3 时域(10/30/60s)，**含 qforce** | `horizon_scenario_aggregate.csv`（pandas unique 核验） |
| 1.3 | **图1 事实修正**：全状态黑箱**训练期收敛**（train_total≈0.010–0.011、test_total≈0.012–0.014、anyNaN=False、train_success_rate=1、solver_failed=0），"发散"只发生在 rollout。训练期唯一异常征兆=黑箱 `train_so3_orth`≈0.17 vs 结构化≈1e-7 | `training_history.pkl` 抽样 |
| 1.4 | **真正的训练期崩溃只有**无升力消融 seed43（44.4m，nbad=276，多次重训复现、与环境无关） | merge plan A.2 类别(2)；`tab:s8-clean-main` 表注 |
| 1.5 | **图6 预测轨迹未落盘**：rollout 只缓存真值到 `_gt_cache/*.pkl`，预测序列仅在内存用于即时渲染、从不落盘 → 需改 `evaluate_rollout_benchmark.py` 重跑 | run 目录无 `.npz/.npy`；`rollout_benchmark_reporting.py:399` plot_example_rollouts |
| 1.6 | **图4 逐轨迹源可用且同源**：`phase1a_iideval_traj30_*/clean/trajectory_metrics.csv` 含逐轨迹 `final_position_error`（traj30，非受污染的 traj8 探针） | `find ... -name trajectory_metrics.csv` |
| 1.7 | **受限模型逐种子真值**（见 §6）：A2 对角阻尼 6/6 有限一致退化（污染签名相反）、A4 执行条件化 6/6 结构性发散、M2 有 18 A区 run 未导出 | `analysis/section8_current_evidence/catalog_supplement_per_seed.csv` |
| 1.8 | **模型代号映射确认**：A2=ablate_diag_damping、A4=ablate_bu_only、M1=phnode_qforce、M2=phnode_merged_force | `auv_model_registry.py`、`auv_baselines.py` |

---

## 2. 实验推进脉络（多阶段叙事的事实基础）

| 阶段 | 名称（时间/镜像） | 回答的问题 | 结论 | 可用性 |
|---|---|---|---|---|
| **A** | Catalog 主线（04-04~21，`g3_5_5`） | 十模型谁最强、消融价值、noisy 效果 | 跨模型排名，但 2 个种子环境污染 | ⚠️ 受限 |
| **A.1/A.2** | Followup P1-1/P1-2（同期，`g3_5_5`） | noisy 训练是否修复种子脆弱性 | "修复 seed46"叙事 | ❌ 被推翻 |
| **B** | Phase-1A cleanrun v1（04-24~26，`g3_5_7`） | 统一环境下主性能 + 关键消融 + 三协议 | **当前论文主证据** | ✅ 可逐位复现 |
| **C** | Provenance 审计重训（05-12，`g3_5_7`） | seed42/46 异常根因 | = cuDNN 环境 artifact，非模型脆弱 | ✅ 取证完毕 |

来源：`docs/experiment_full_inventory_zh.md`、`docs/experiment_stages_overview.md`（§7 可信度表）、`docs/provenance_audit_phnode_full_clean.md`、`docs/oc_followup_results_p1_p2.md`。

> **B 区两块证据的口径区分（关键，勿混）**：
> - **clean 协议 7 模型**（t2_wpfrag clean 决策）：`tab:s8-clean-main`、`tab:s8-noise` 的来源。物理 checkpoint 在 `checkpoints/sweep_oc_phase1a_decision_clean_t2_wpfrag_<model>/`。
> - **3 模型 × 3 训练协议**（cleanrun v1 决策包，45 run）：`tab:s8-protocol` 的来源（外加 clean 协议的 qforce）。

---

## 3. 证据三级分级清单（逐项给产物路径）

### ✅ 干净主证据（进主排序，B 区 `g3_5_7` 可逐位复现）

| 证据单元 | 覆盖 | 用于 | 产物路径 |
|---|---|---|---|
| 七模型干净主表 | 4 结构化+3 黑箱，clean×clean×5 种子 | 主表/几何层/精度阶梯 | `analysis/section8_current_evidence/aggregate.csv`、`per_seed_long.csv` |
| 误差随时域/场景 | 7 模型×{10,30,60}s×4场景×4profile×3协议 | 图2/图3 | `analysis/section8_current_evidence/horizon_scenario_aggregate.csv`（+`horizon_scenario_per_seed.csv`） |
| 逐轨迹终端误差 | 7 模型×5 种子×30轨迹×3场景，clean 评估 | 图4/图5 | `checkpoints/sweep_oc_phase1a_decision_clean_t2_wpfrag_<model>/<run>_seed*/rollout_benchmark/phase1a_iideval_traj30_*/clean/trajectory_metrics.csv` |
| 训练动力学 | 7 模型×5 种子 clean，250 epoch，train/val 总损失+13 分量 | 图1 | 同上 run 目录 `training_history.pkl` |
| 三协议等价 | 4 结构化×{clean/iid/v4lite}训练×nominal_eval | 协议表 | `aggregate.csv` / horizon 表 train_protocol 列 |
| 内部诊断 | SO(3) 正交性(全模型)、能量跨度(仅结构化族) | 诊断表/图5 | `per_seed_long.csv`（`max_so3_orth_error`、`energy_span_median_60s`） |

### A 区更广模型（catalog `g3_5_5`）：稳定结果可信、并入主排序（见 §0.1）— 详见 §6

| 模型 | A 区 clean 值 | 产物 |
|---|---|---|
| A2 对角阻尼 | 4.17m / 6.83× / 6 种子一致 | `catalog_supplement_per_seed.csv`、`catalog_supplement_aggregate.csv` |
| A4 执行条件化(bu_only) | 6/6 发散 18–54m | 同上 |
| M2 合并广义力 | 18 run 未导出（需补） | `analysis/oc_data_catalog/rollout_run_registry.csv`（`phnode_merged_force`） |
| qforce catalog | 0.57m（与 cleanrun 3.76 矛盾，不可用） | catalog；主表用 B 区 3.76 |

去重口径（**关键，勿踩坑**）：A 区取数须按 `rollout_run_registry.csv` 的 `is_selection_eligible=1` 且优先 `resampled_traj30_*` 对每 (model,train,seed,profile) **唯一去重**，否则混入 traj8 的 `*_iideval_*` 探针 run 导致错误数值。取 `metric_name=final_position_error, stat_name∈{median,p95}, horizon_s=60, scope=overall`。

### ❌ 不可用 / 不得作正面结果

| 项 | 处理 |
|---|---|
| "噪声初值训练修复种子脆弱性"(P1-1/P1-2) | provenance 审计推翻；仅 §1.9 证据治理一句带过 |
| 完整模型 clean seed42/46 | `stale_environment_drift`，排除（`evidence_status_overrides.csv`） |
| smoke/probe/unused/旧噪声、前代仓库 g3_5_4 noc 线 | 不进入本章 |

### 证据空白（不得声称做过）

P2-1（质量先验×升力 2×2，含不存在的"无质量先验且无升力"模型）、P2-2（噪声调度扫描）、P3（remus100_ins / noc）、Phase-1B、步长/求解器阶数敏感性（§1.7 已声明未单独考察）。来源：`docs/experiment_full_inventory_zh.md` §J。

---

## 4. §1.8 新结构提纲（节首去防御 + 7 小节 + 受限单列）

| 小节 | 回答的问题 | 论点 | 表 | 图 |
|---|---|---|---|---|
| 节首（重写） | 本节回答什么 | 四递进问题 + 一句设置回指 §1.7 + 一句证据范围；**无治理/防御开篇** | — | — |
| §1.8.1 训练收敛与块级拟合 | 训练成功？块级拟合够吗？ | 都能稳定训练、拟合块级（必要非充分）；黑箱训练期已露几何漂移征兆；seed43 训练崩溃 | — | 图1 |
| §1.8.2 几何结构与长期稳定性（含结构必要性） | 几何/充分力条件化是否决定长期稳定？ | 黑箱 rollout 发散 vs SE(3) 有限；**执行条件化过窄→崩溃**（第二条结构必要性）；短时接近、长时分化 | `tab:s8-clean-main`(留) | 图2、图6 |
| §1.8.3 能量核心与完整消融阶梯 | 能量核心是否主导精度？各成分边际？ | 完整阶梯（含**对角阻尼≈6×**、能量核心≈5.5×、质量先验≈1.9×、升力≈1.2× + **合并广义力**广义力分解轴）；中位 vs P95 翻转；**headline 细化** | (沿用主表+并入更广模型) | 图4；旧柱状图→阶梯图 |
| §1.8.4 初值扰动鲁棒性与排序收敛 | 优势在哪成立/哪收窄？ | 几何稳定全程稳健、精度领先强扰动下收窄 | `tab:s8-noise`(留) | 图3 |
| §1.8.5 噪声训练协议等价性 | 协议敏感吗？ | iid≈v4lite，排名不变 | `tab:s8-protocol`(留) | — |
| §1.8.6 内部结构诊断 | 为什么？ | 能量跨度、SO(3) 正交性解释机理 | `tab:s8-diag`(留) | 图5 |
| ~~§1.8.7 受限单列~~（**取消**，按 §0.1） | — | 更广 A 区稳定模型已并入 §1.8.2/§1.8.3，不再单列、不标来源 | — | — |
| 小结段 | — | 两级结论收束 + 条件化口径(一次) | — | — |

---

## 5. 插图清单（每图：问题/图型/数据源+筛选/脚本/可用性/实现注意）

| 图 | 回答什么 | 图型 | 数据源 + 筛选 | 脚本（建议名） | 实现注意 |
|---|---|---|---|---|---|
| **图1** 训练/验证收敛 | 训练成功吗？黑箱缺陷何时显现？ | train/val 损失 vs epoch（+SO(3) 正交损失子图） | `training_history.pkl`（clean，代表性种子）；字段 `train_total`/`test_total`/`train_so3_orth` | `scripts/export_section8_training_curves.py` + `figures/section8_training_curves/make_*.py` | 诚实三要素；黑箱靠 `train_so3_orth`≈0.17 露馅；标 seed43 崩溃 |
| **图2** 误差随时域 | 短时接近、长时分化、黑箱发散 | 折线 中位数 vs {10,30,60}s | `horizon_scenario_aggregate.csv`，筛 `train_protocol=clean & eval_profile=clean & scope=overall`；列 `posmed_mean_of_seed_medians`（与主表 0.68 口径一致） | `scripts/export_section8_horizon_curves.py` + `figures/section8_horizon_growth/` | 6 模型有限折线 + 黑箱标"5/5 发散"（七模型同图折线不可行） |
| **图3** 扰动梯度 | 优势在哪收窄？ | 折线 中位数 vs 4 profile | 同表，筛 `train_protocol=clean & horizon_s=60 & scope=overall`，按 `eval_profile` 取列 | 同图2 脚本 | 黑箱四 profile 均发散 |
| **图4** 逐轨迹分布 | 中位 vs P95 为何翻转？ | 箱线/小提琴 终端位置误差 | `trajectory_metrics.csv`（`phase1a_iideval_traj30_*/clean/`），聚合 7 模型×5 种子×30轨迹×3场景；字段 `final_position_error` | `scripts/export_section8_trajectory_distribution.py` + `figures/section8_error_distribution/` | seed43(no_lift,44.4m)单列/标注；黑箱发散轨迹如何呈现需脚本里定 |
| **图5** 内部诊断 | 结构按设计工作吗？ | 跨模型 SO(3) 正交性 + 能量跨度（双子图/分面） | `per_seed_long.csv`；列 `max_so3_orth_error`、`energy_span_median_60s` | `scripts/export_section8_diagnostics.py` + `figures/section8_diagnostics/` | SO(3) 全模型；能量跨度仅结构化族（如实标注） |
| **图6** 轨迹可视化 | 定性 预测 vs 真值 | 3D/分量轨迹叠合 | **需重跑**：rollout 增加预测序列落盘 + gt（`_gt_cache/*.pkl`：`gt_pos(1201,3)`、`gt_rotation`、`gt_nu`、`time`） | 改 `evaluate_rollout_benchmark.py`(只读式增加导出) + 重跑代表 run + `figures/section8_rollout_example/` | 选 完整模型 vs 全状态黑箱，单条 CHIRP 轨迹；不改训练/数值/评估口径 |

### 5.1 现有图生成器风格约定（新脚本必须对齐）

参考 `paper/drafts/figures/make_section8_two_level_evidence.py`、`make_velocity_state_contract.py`：

- **缓存隔离**（脚本开头）：`os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/auvhamnode_mplconfig")`、`XDG_CACHE_HOME=/private/tmp/auvhamnode_xdg_cache`。
- **尺寸**：物理 mm，`MM_TO_IN=1/25.4`，`figsize=(W*MM_TO_IN, H*MM_TO_IN)`（如 150×72 mm）。
- **字体**：`pick_font()` 优先 `Arial→Helvetica→DejaVu Sans→Arial Unicode MS`（+ `PingFang SC/Hiragino Sans GB` 作中文兜底）；**纯英文图**。
- **rcParams**：`font.family=sans-serif`、`svg.fonttype="none"`、`pdf.fonttype=42`、`ps.fonttype=42`、`mathtext.fontset="dejavusans"`、`axes.unicode_minus=False`、`savefig.facecolor="white"`、`font.size≈7`、`axes.linewidth≈0.6`。
- **配色 PALETTE**（语义命名）：`ink #252525`、`muted #62666D`、`rule #D8DDE3`；几何=蓝 `geometry #2E6EBA`、能量=绿 `energy #2F8E86`、风险/黑箱=棕 `risk #9A5B3F`、强调=金 `accent #B8862B`（各带 `_pale` 浅底）；柱状：phnode_full 深蓝、qforce 金、黑箱灰 `#A8AFBA`、消融绿。
- **模型显示名 `DISPLAY`/`AXIS_DISPLAY`**（复用同名）：`phnode_full→"AUVHamNODE"`、`ablate_no_lift→"No Lift"`、`ablate_no_mass_prior→"No Mass Prior"`、`phnode_qforce→"Config Force"`、`se3_momentum_blackbox→"SE(3) mom."`、`se3_accel_blackbox→"SE(3) acc."`、`blackbox_fullstate→"Full-state"`（按现有脚本核对实际值）。
- **输出多格式**：`.svg`(无 dpi)、`.pdf`、`.png`(`dpi=600`)，可选 `.tiff`；`bbox_inches="tight", pad_inches≈0.01–0.02`。
- **路径**：`OUT_DIR=Path(__file__).parent`、`REPO_ROOT=OUT_DIR.parents[2]`、数据从 `REPO_ROOT/"analysis"/"section8_current_evidence"/...` 读。
- **坑**：`diagnostic_plots.csv` 的 `path` 字段是 Colab 绝对路径（`/content/drive/...`），勿照搬；本地用相对路径。

### 5.2 关键产物结构速查

**`training_history.pkl`**（dict，clean 协议 250 epoch）：
- 逐 epoch：`epoch / lr / global_step / epoch_time`
- 训练损失：`train_total` + `train_{position,rotation,actuator,velocity,angular,so3_orth,so3_det,u,v,w,p,q,r}`
- 验证损失：`test_*` 同名全套
- 批次健康：`train_*_batches`、`train_success_rate`
- 黑箱 clean 实测：`train_total`≈0.010–0.011、`test_total`≈0.012–0.014、`train_so3_orth`≈0.17（结构化≈1e-7）

**run 目录命名**（`checkpoints/sweep_oc_phase1a_decision_clean_t2_wpfrag_<model>/`）：
- `main_<model>_seed*`（结构化主模型）、`baseline_<bb>_seed*`（黑箱/qforce）、`ablation_<abl>_seed*`（消融）
- 每 run 含 `config.json / training_history.pkl / best_model.pt / block_evaluation.json / heldout_evaluation.json / rollout_benchmark/ / _audit_meta/`

**`rollout_benchmark/` 子目录**：
- `_gt_cache/*.pkl`（`GroundTruthPayload`：`gt_pos(1201,3)`、`gt_rotation(1201,3,3)`、`gt_nu`、`time(1201,)`）
- `phase1a_iideval_traj30_seed*/{clean,nominal_eval,degraded_eval,heading_biased_eval}/trajectory_metrics.csv`（四 profile 评估，逐轨迹）
- `phase1a_v4eval_traj30_seed*/trajectory_metrics.csv`（v4lite 评估）

**`trajectory_metrics.csv` 列**（共 40 列，关键）：`scenario, trajectory_id, seed, horizon_s, completed_time, failure_reason, final_position_error, final_depth_error, final_rotation_geodesic, final_relative/total_linear/angular_velocity_error, mean/max_position_error, max_so3_det_error, max_so3_orth_error, final_pred_energy, energy_span, max_abs_energy_delta, any_depth_violation_up_to_h`。

---

## 6. 更广 A 区模型：稳定性核验与可信度裁决（论证留档；写作按 §0.1 并入主叙事、不标来源）

> 本节是**实验报告口径的留档**（为何 A 区稳定结果可信、唯一要弃的是哪个单种子异常）。结论：除完整模型 seed42/46 的单种子环境异常外，更广 A 区模型结果平稳、可信、并入主消融。正式论文正文**不复述本节的镜像/种子细节**（§0.1）。

### 6.1 环境差异的精确机制（为何只弃单种子异常）

A 区（catalog，云镜像 `auvhamnode/g3_5_5`）与 B 区（`g3_5_7`，当前 main，可逐位复现）：**同数据集 `d0be9434`、模型代码逐行无 diff（审计 Phase 4）、同超参、同噪声机制、同评估口径**。唯一差异=**未记录、未受控的云端数值栈**（PyTorch/CUDA/cuDNN 版本 + 非确定性算法选择）。

污染的**特定签名**：训练期 cuDNN 耦合梯度爆炸——罕见、单种子、灾难性（完整模型 clean seed46：epoch 24 梯度 4.68e25→inf→275 行"无成功训练批次"；当前环境重训自愈为 0.456m、逐位一致）。在行为正常单元上 A↔B 仅相差约一成、方向不一致（无系统性偏置）。

**R-A 判据（按异常签名而非按镜像）**：污染特征是"多数好+个别坏"的单种子灾难发散。凡呈**相反签名**（多数差+个别好的一致退化）或**全种子一致**（结构性发散/有限稳定）的结果，**不带这种污染**，对数值栈偏移免疫。来源：`docs/section8_evidence_merge_plan.md`、记忆 `section8_evidence_merge_rule`。

### 6.2 逐模型逐种子真值与裁决

数据源：`analysis/section8_current_evidence/catalog_supplement_per_seed.csv`（已按 `is_selection_eligible=1`+`resampled_traj30` 去重）。

| 模型（代号） | A 区逐种子 clean 60s 中位/m | 聚合 | 签名 | 带漂移污染？ | 裁决 |
|---|---|---|---|---|---|
| **对角阻尼 A2** `ablate_diag_damping` | 42=3.93 / 43=4.82 / **44=0.69** / 45=5.60 / 46=5.69 / 47=4.26（6/6 `ok`） | 均值 **4.17**、中位 4.54、P95 均值 **10.29**、6.83× | 多数差+个别好，**一致退化** | **否**（与漂移相反） | **基本可用**；R-A ④ 按定量结果采纳；限制=A 区基线 0.61、未在 B 区逐位重建 |
| **执行条件化 A4** `ablate_bu_only` | 42=23.08 / 43=18.36 / 44=23.85 / 45=53.77 / 46=22.52 / 47=25.20（6/6 `rollout_diverged`） | 全发散，`stability_failure` | **全种子结构性发散** | **否**（偏移免疫） | **可用（定性）**："执行条件化过窄→长期发散"，与黑箱几何结论同级可信 |
| **合并广义力 M2** `phnode_merged_force` | A 区有 18 run、当前 supplement 未导出 | 待补导出 | 待查 | 待查 | **补导出后纳入** |
| **配置广义力 M1** `phnode_qforce` | **B 区 3.76m=主证据（主表 M1）**；A 区另有 0.57m | 0.57 与 3.76 矛盾 | 跨环境单点低值 | 0.57 疑受污 | **不属受限模型**（已在主表）；0.57 仅作"跨环境数值不可直接并列"反例，不单列 |

A 区噪声训练旁证（可选，用于"几何稳定跨噪声线"，来源同 `catalog_supplement`）：
- `blackbox_fullstate` noisy：全发散 ~148m/nan（3 种子）
- `se3_momentum_blackbox` noisy clean：1.79（均值，3 种子）
- `se3_accel_blackbox` noisy clean：5.46（均值）/ 5.70（中位，3 种子）

### 6.3 写作处理（按 §0.1）

A 区稳定结果（对角阻尼、执行条件化、合并广义力）作为可信证据**并入主消融叙事**，正文**不标数据来源、不加镜像脚注、不单列"受限"**。退化倍数按单一 phnode_full 基线（0.68）统一计，或定性表述（"约 6 倍"），避免跨镜像相除的精度纠缠。完整模型 seed42/46 在 B 区 5/5 稳，主表无需提及其异常。

---

## 7. headline 细化与红线一致性

若纳入 A2、A4，合并后阶梯（merge plan 已推演）：

> 执行条件化（A4，**崩溃**）≫ 对角阻尼（A2，**6.8×**）≈ 能量核心（M1，**5.5×**）> 质量先验（1.9×）> 升力耦合（1.2×）

**采纳的细化（merge plan 方案 b，与红线 #5 相容）**：
- 「能量核心是最主导精度先验」**限定在 {能量核心 / 质量先验 / 升力} 惯性-能量子先验组内**依次递减；
- **对角阻尼结构**与**执行条件化**作为**另两条独立的结构必要性**分别陈述。

### 五条设计红线一致性（仅核查、不破坏）

| 红线 | 守法方式 |
|---|---|
| ① 端口哈密顿仅限开放机械核心、τθ≠G(q)u | 结果节不复述理论；机理回指 §1.5/§1.6 引理命题（能量平衡命题、势能功率配对引理、SE(3) 运动学式） |
| ② 海流双口径（契约分段常值 / 数据生成 OU） | 图注不混用，沿用 §1.3 措辞 |
| ③ 普通求解器不严格保 SO(3)/能量 | 图5/诊断节明确"正交性误差小但非零，作诊断量"，不宣称严格保群 |
| ④ 初值扰动=鲁棒性正则、非导航后验 | 图3/§1.8.4 沿用此口径 |
| ⑤ 排名条件化、不外推为普遍必要性 | "当前模型族与数据协议下"全节一次；headline 细化后仍守"依次递减"于子先验组内 |

术语：契约/配置/自由演化/黑箱；M1 正式名「端口哈密顿配置广义力基线」；正文中文描述名、代号仅入表。

---

## 8. 关键数值速查表（来源均已标注，未来直接引用）

> 单位 m。主表口径=`posmed_mean_of_seed_medians`（先轨迹内取中位、再种子间取均值）。

### 8.1 七模型干净主表 `tab:s8-clean-main`（clean 训练/clean 评估，60s）— 来源 `aggregate.csv`/`per_seed_long.csv`（B 区）

| 模型（代号） | N | 中位数 | P95 | 完成率 | 备注 |
|---|---|---|---|---|---|
| 完整模型 M0 `phnode_full` | 5 | 0.68±0.28 | 1.77±0.65 | 0.99 | 最优 |
| 无升力耦合 A3 `ablate_no_lift` | 4 | 0.83±0.35 | 2.15±0.70 | 0.99 | 剔 seed43=44.4m（真实可复现脆弱） |
| 无质量先验 A1 `ablate_no_mass_prior` | 5 | 1.30±0.35 | 4.31±1.26 | 0.99 | |
| SE(3) 动量黑箱 M3 `se3_momentum_blackbox` | 5 | 1.49±0.37 | 3.86±0.87 | 0.99 | |
| SE(3) 加速度黑箱 M4 `se3_accel_blackbox` | 5 | 2.46±1.67 | 9.04±8.47 | 0.98 | 尾部单次主导（重复间中位 1.64） |
| 配置广义力 M1 `phnode_qforce` | 5 | 3.76±0.91 | 11.25±3.19 | 0.99 | |
| 全状态黑箱 M5 `blackbox_fullstate` | 0 | 全部发散 | 80–90m/非有限 | 0–0.87 | 长期稳定性失败 |

phnode_full 时域三点（mean 口径）：10s=0.083、30s=0.272、60s=0.677（median 口径=0.080/0.228/0.611）。**其余模型时域三点脚本导出时按 mean 列复核后写入。**

### 8.2 噪声/profile 表 `tab:s8-noise`（clean 训练，60s 中位）— 来源 `aggregate.csv`（B 区）

| 模型 | clean | nominal | degraded | heading |
|---|---|---|---|---|
| `phnode_full` | 0.68 | 0.96 | 1.76 | 3.13 |
| `ablate_no_lift` | 0.83 | 1.05 | 1.77 | 3.07 |
| `ablate_no_mass_prior` | 1.30 | 1.42 | 2.04 | 3.10 |
| `se3_momentum_blackbox` | 1.49 | 1.62 | 2.13 | 3.00 |
| `se3_accel_blackbox` | 2.46 | 2.62 | 3.20 | 3.83 |
| `blackbox_fullstate` | 全 profile 5/5 发散 | | | |

（已核 `phnode_full` degraded=1.7574、heading=3.1257。）

### 8.3 协议等价表 `tab:s8-protocol`（nominal_eval，60s 中位）— 来源 merge plan T-L3（全 B 区）

| 模型 | clean 训练 | iid 训练 | v4lite 训练 | 两噪声线相对差 |
|---|---|---|---|---|
| `phnode_full` | 0.96 | 1.09 | 1.11 | 1.7% |
| `ablate_no_lift` | 1.05 | 1.00 | 1.27 | 26%（强扰动下收敛<8%） |
| `ablate_no_mass_prior` | 1.42 | 1.46 | 1.40 | 4.1% |
| `phnode_qforce` | 3.74 | 2.63 | 1.59 | 40%（高方差，两协议跨度均≈3m） |

### 8.4 诊断表 `tab:s8-diag`（clean 配置，60s）— 来源 `per_seed_long.csv`（SO(3)=种子间最大值；能量跨度=种子间中位数）

| 模型 | 机械能量跨度中位数 | SO(3) 正交性误差最大值 |
|---|---|---|
| `phnode_full` | ≈17.8 | 1.4e-5 |
| `ablate_no_lift` | ≈18.7（剔异常 seed43 后 N=4） | 1.3e-5 |
| `ablate_no_mass_prior` | ≈1.9 | 1.3e-5 |
| `se3_momentum_blackbox` | 无定义 | 1.2e-5 |
| `se3_accel_blackbox` | 无定义 | 1.4e-5 |
| `phnode_qforce` | 无定义 | 3.4e-4（seed45=3.397e-4，高一量级，真实坐实） |
| `blackbox_fullstate` | 全发散未列 | — |

> ⚠️ 诊断量权威来源是 B 区 `per_seed_long.csv`（列 `max_so3_orth_error`/`energy_span_median_60s`）；A 区 catalog 的 qforce 正交性是不同镜像约 1.5e-5，**勿用**。

### 8.5 更广消融模型（并入 §1.8.3 主阶梯，按 §0.1 不标来源）— 数据 `catalog_supplement_per_seed.csv`

> 倍数列在论文中按**单一 phnode_full 基线 0.68** 统一计（或定性"约 N 倍"）；下表内 A 区基线 0.61 的 6.83× 仅留档参照，二者差异可忽略。

| 模型（代号） | clean 中位/m | P95/m | 倍数(基线0.68/0.61) | N | 判定 |
|---|---|---|---|---|---|
| 对角阻尼 A2 `ablate_diag_damping` | 4.17 | 10.29 | ≈6.1× / 6.83× | 6 | 一致退化（真实效应，可信） |
| 执行条件化 A4 `ablate_bu_only` | 6/6 发散 18–54m | — | — | 6 | 结构性发散（可信） |
| 合并广义力 M2 `phnode_merged_force` | 待补导出 | 待补 | 待补 | 18 run | 补后并入 |

---

## 9. 脚本计划

| 脚本 | 输入 | 输出 | 备注 |
|---|---|---|---|
| `scripts/export_section8_training_curves.py` | 代表 run 的 `training_history.pkl` | 中间 CSV + 给图1 | 选种子/模型在脚本内定 |
| `scripts/export_section8_horizon_curves.py` | `horizon_scenario_aggregate.csv` | 中间 CSV（图2/图3） | 筛选见 §5；统一 mean 口径 |
| `scripts/export_section8_trajectory_distribution.py` | 各 run `phase1a_iideval_traj30_*/clean/trajectory_metrics.csv` | 聚合 CSV（图4） | 7 模型×5 种子聚合；seed43 单列 |
| `scripts/export_section8_diagnostics.py` | `per_seed_long.csv` | 中间 CSV（图5） | SO(3) 全模型/能量跨度仅结构化 |
| 扩展受限导出（在 `export_section8_catalog_supplement.py` 或新脚本） | `canonical_rollout_summary_long.csv` + `rollout_run_registry.csv` | 补 M2 至 `catalog_supplement_*.csv` | 去重口径见 §3 |
| 改 `evaluate_rollout_benchmark.py`（只读式） | — | 预测轨迹序列落盘 | 仅增导出，不改训练/数值/评估口径；重跑代表 run |
| 6 个 `figures/<name>/make_*.py` | 上述中间 CSV | PDF+PNG+SVG | 对齐 §5.1 风格 |

`figures/` 子目录建议：`section8_training_curves/`、`section8_horizon_growth/`、`section8_error_distribution/`、`section8_diagnostics/`、`section8_rollout_example/`、（阶梯图复用/改造 `section8_two_level_evidence/` 或新建 `section8_ablation_ladder/`）。

每脚本保留中间 CSV，**每个数值可由脚本从原始产物复现**。

---

## 10. 执行顺序与交付

1. 写导出/绘图脚本 + 扩展受限导出（补 M2）→ 生成 5 张新图 + 阶梯图中间 CSV。
2. 图6 只读式重跑：给 rollout 增加预测轨迹落盘 → 重跑 1–2 代表 run → 出图6。
3. 改主稿 §1.8（节首 + 7 小节 + §1.8.7 + 6 图嵌入 + headline 细化）+ 同步 §1.9/§1.10 交叉引用。
4. `xelatex` 两遍：**0 Overfull、0 未定义引用**、页数合理。（既存：`.bbl` §1.2 公式段一条 Underfull 属正常，非新引入；新增图表勿引入 Overfull——窄列勿塞长表头、宽图用合适 `width`。）
5. 派一个独立 agent 二次核验：科学内容/数值可追溯到产物、五红线无损、受限证据分级标注得当、交叉引用闭环、术语与 §1.1–§1.7 一致、语域达标（无防御/报告/种子腔）、新增图表数据可复现。
6. 非 main 分支 `docs:` 中文 commit，本地不主动推送，等用户指示。

### 语域硬要求（去防御腔）
- 删"这是证据分层而非结果筛选""区分两类相互独立的排除判据""触发后从定量聚合中移出""需说明该判断建立在……""不构成普遍必要性定理"等防御性元话语的反复出现；**开头不得以证据治理/判定标准开篇**。
- 保留但精炼："在当前模型族与数据协议下"全节一次；普遍必要性保留说明全节一次。
- caveat 下沉：完成率口径、发散/异常判据、被剔除运行、证据治理 → 图表注/脚注/方法回指 §1.7/讨论 §1.9。
- 不反复点名随机种子：减少"五个种子/种子43/种子间标准差/有效种子数 N"；可靠性记账下沉表注，正文用干净陈述（如"该基线在所有评估配置下均长期预测发散"）。
- **不标数据来源（A 区/B 区）、不加镜像脚注、不单列"受限"、不铺陈个别种子异常**（按 §0.1）：A 区稳定结果与干净集同列陈述；无升力那次训练不稳定压成一句或脚注，不再铺成一段。
- 正文不出现源码变量名、函数名、数据集文件名、脚本名、配置键名。

---

## 11. 关键文档/产物索引（未来溯源一站式）

| 类别 | 路径 |
|---|---|
| 主稿 | `paper/drafts/auvhamnode_thesis_chapter_zh.tex` |
| 证据合并方案（表格草值+R-A 口径） | `docs/section8_evidence_merge_plan.md` |
| cleanrun v1 报告 | `docs/phase1a_oc_v4lite_cleanrun_v1_report.md` |
| 实验总账 | `docs/experiment_full_inventory_zh.md`（§J 空白、§K/§L 总账） |
| 阶段总览（§7 可信度表） | `docs/experiment_stages_overview.md` |
| provenance 审计 | `docs/provenance_audit_phnode_full_clean.md`、`docs/oc_followup_results_p1_p2.md` |
| 干净主证据 CSV | `analysis/section8_current_evidence/{aggregate.csv, per_seed_long.csv, horizon_scenario_aggregate.csv, horizon_scenario_per_seed.csv, catalog_supplement_aggregate.csv, catalog_supplement_per_seed.csv, nolift_seed43_disclosure.csv}` |
| catalog（受限/去重） | `analysis/oc_data_catalog/{canonical_rollout_summary_long.csv, rollout_run_registry.csv, evidence_status_overrides.csv}` |
| 物理 checkpoint（B 区 clean 7 模型×5 种子） | `checkpoints/sweep_oc_phase1a_decision_clean_t2_wpfrag_<model>/<run>_seed*/` |
| 选择规则/字典 | `docs/oc_result_selection_policy.md`、`docs/oc_data_catalog_dictionary.md` |
| 现有图生成器（风格参考） | `paper/drafts/figures/make_section8_two_level_evidence.py`、`make_velocity_state_contract.py` |
| 上游 prompt | `paper/drafts/section_1_8_results_rewrite_prompt_v2.md` |

---

## 12. 硬约束（事实诚信/证据闸门）

- §1.8 每个数值、排名、"发散/收敛"判断必须可追溯到上述产物，且尊重 `evidence_status`；`stale_environment_drift` 运行（完整模型 clean seed42/46）不进主排序。
- 不编造/不"优化"数值；需要某个当前证据集没有的数字时**标注让用户确认**，不臆造。
- 不虚构未做过的实验（步长/求解器阶数敏感性、P2/P3、Phase-1B 等空白）。
- 受限证据逐处标 A 区脚注，不与干净集主排序混排。
- 新脚本产出的每个数值，必须能由脚本从原始产物复现（保留脚本与中间 CSV）。

---

*（本方案已与用户定稿；执行中如出现证据集未覆盖的数字需求，标注并请用户确认，不臆造。）*
