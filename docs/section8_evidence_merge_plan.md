# §8 实验章证据合并方案 + 合并后草表

生成时间：2026-05-30  
适用范围：论文 `paper/drafts/auvhamnode_thesis_chapter_zh.tex` 第 8 节重写。  
目标：在 v4-lite 与 iid_noisy_ic **并列为两条噪声训练线** 的前提下，把 **B 区（Phase-1A / g3_5_7）** 与 **A 区（catalog / g3_5_5）** 的可信结果**诚实合并**，补齐当前 §8 缺失的"几何稳定性跨噪声线"和"完整消融阶梯"，并明确每个数字的来源与口径。

> **2026-06-02 更新（R-A 口径）**：本方案 PART B 草表已被 `paper/drafts/section8_rewrite_draft_zh.md`（权威预写稿）取代；本文件保留作方案与推理记录。"双镜像相互印证 / 交叉验证"的旧提法**已废弃**，改为下列 R-A 选择规则。另：PART B 中 `bu_only` 应按发散报（非 45.6×）、`se3_accel` iid 列应为 5.61（非 5.46）、`diag_damping` P95=10.29——详见草案 R1–R3 / D2（下文 PART B 已就地更正）。

前置事实（已一手核实）：
- **A、B 无本质实验差异**：同数据集 `d0be9434`、模型代码逐行无 diff（`docs/provenance_audit_phnode_full_clean.md` §2.4 Phase 4）、同显式超参、同 `iid_noisy_ic` 噪声机制（profile=nominal_train, ref=remus100_dr）、同评估口径（60s / `final_position_error` median / `overall` / PRBS+CHIRP+OU×30 traj / 四 profile）。唯一差异是**未记录、未受控的云端数值栈**（PyTorch/CUDA/cuDNN 版本 + 非确定性算法选择），属混杂因子而非设计因子。
- 因此 A、B **不是相互印证的独立复现**；A↔B 在非异常单元上相差约一成、方向不一致（phnode_full clean A=0.61 vs B=0.68；no_mass_prior A=1.44 vs B=1.30），只说明该未受控环境无系统性偏置。
- **R-A 合并选择规则**：① 两区都有的同一实验 → 取可逐位复现的 **B 区**；② 仅 A 区覆盖的单元 → 取 **A 区**（黑箱噪声训练、`diag_damping`、`bu_only`）；③ 排除仅限"单种子灾难发散 + 已被 B 区重跑推翻"（`phnode_full` clean s42/s46）；④ A 区**多种子一致**结果按定量结果采纳；⑤ 信任按异常签名给（per-anomaly），不按镜像给（per-mirror）。倍数仍各自用同环境基线，逐行标注来源。

> 范围：noc 线属前代仓库 `g3_5_4`，当前论文不使用（见 `docs/g3_5_4_legacy_noc_oc_inventory_zh.md`）。本方案只涉及 OC。

---

# PART A — 合并方案

## A.1 两个证据区

| 区 | 镜像 | 时间 | 角色 | 噪声协议 |
|---|---|---|---|---|
| **B 区** | `g3_5_7` | 04-24~26 | §8 主证据（可逐位复现，带 `_audit_meta/`） | clean / iid_noisy_ic / v4_lite |
| **A 区** | `g3_5_5` | 04-04~24 | 补充已做但 B 区未覆盖的单元（同 iid 协议、同口径） | clean / iid_noisy_ic（=noisy_train） |

## A.2 异常分类法（合并的核心规则）

把"被排除的种子"分成**三类**，处理方式不同，**必须在论文中分别披露**：

| 类别 | 判据 | 实例 | 处理 |
|---|---|---|---|
| **(1) 环境漂移 artifact** | 在历史镜像出现灾难性发散（"no successful training batches" / nbad>0），但**当前环境重训自愈、不可复现**（provenance 审计 Phase 1–4 证实） | A 区 `phnode_full` clean **seed42(4.2m)/seed46(47.9m)** | **排除出当前证据集**；仅在溯源说明中提及。当前环境 B 区该模型 5/5 稳定 |
| **(2) 真实可复现脆弱** | 在**当前环境多次重训均失败**，**与环境无关**（用户已实验确认） | B 区 `ablate_no_lift` clean **seed43(44.4m, nbad=276)** | **诚实记录为该模型的真实失效模式**；中心趋势取稳定簇（N=4），但**单列披露**该种子的值与性质，**不可当作可剔除的 artifact** |
| **(3) 结构性长时发散** | 该模型在某配置下**全部种子**发散（nan / 远超阈值），是结构本身的失败 | `blackbox_fullstate`（clean 与 iid 噪声训练，全 profile） | **不报有限中位数**，报为"长期稳定性失败"，本身即证据 |

> 公平性原则（按用户要求）：对所有模型施加**同一个判据**——"在当前环境是否可复现失败"。phnode_full 的 42/46 因**当前环境不复现**而归类 (1) 排除；no_lift 的 seed43 因**当前环境可复现**而归类 (2) 披露。差别来自证据，而非双重标准。论文应把两者并列说明，让读者看到"全模型在当前环境的真实逐种子行为"。

## A.3 逐表来源映射

| §8 表 | 内容 | 数据来源 | 备注 |
|---|---|---|---|
| **T-L1 几何稳定性** | blackbox vs SE(3) 黑箱 vs 结构化，clean 训练 + iid 噪声训练 | clean 列 = **B 区**；iid-噪声列 = **A 区**（B 区无黑箱噪声训练） | 跨镜像，但结论为定性"发散/有限"，对偏移免疫 |
| **T-L2 能量精度阶梯** | full / no_lift / no_mass_prior / qforce / diag_damping / bu_only，clean | full,no_lift,no_mass_prior,qforce = **B 区**；diag_damping,bu_only = **A 区** | 倍数**同镜像内**计算（见 A.4） |
| **T-rob 跨评估鲁棒性** | 4 结构化 + SE(3) 黑箱，clean 训练，四 profile | **B 区**（结构化、blackbox、se3_*）+ **A 区**（黑箱噪声训练的稳定性旁证可选） | 单镜像主表，最干净 |
| **T-L3 噪声协议敏感性** | clean / iid / v4lite，4 结构化，nominal_eval | **全部 B 区** | 完整模型与无质量先验消融 iid≈v4lite；无升力耦合消融和配置广义力基线是显式例外；分层排序不变 |
| **T-disc 透明度披露** | 所有被排除/失败种子 + 类别 | A 区 + B 区 | 体现 A.2 分类法 |

## A.4 同镜像倍数规则

退化倍数 = 该模型中位数 ÷ **同镜像** `phnode_full` clean 基线，避免跨镜像相除：
- **B 区基线** = phnode_full clean = **0.677 m**（5 种子）。
- **A 区基线** = phnode_full clean 非漂移种子（43,44,45,47）均值 = **0.61 m**（N=4，已剔除环境漂移的 42/46）。
- 两基线相差约 ~10%（0.61 vs 0.68）、方向不一致，确认该未受控环境无系统性偏置；倍数仍各自用本环境基线，并在脚注注明。

## A.5 脚注 / 披露要求（写作时必须落实）

1. 每张合并表对**每一行标注来源镜像**（B 区 g3_5_7 / A 区 g3_5_5）。
2. 倍数表注明"同镜像基线"及其值。
3. A↔B 一致性核验作为"两环境无本质差异"的依据写入脚注或附录（附 phnode_full / no_mass_prior 的 A↔B 逐种子对照）；不表述为"相互印证/独立复现"。
4. A.2 三类异常各自显式披露；**no_lift seed43 必须明确写为"真实可复现脆弱、与环境无关"**，并与 phnode_full 的"环境漂移、当前不复现"对照。
5. A 区黑箱噪声训练只有 N=3（种子 43,44,45），结构化为 N=5（42-46）——表中标注 N，定性结论（发散/稳定）不受 N 影响。

## A.6 工具改造

- 现有 `scripts/export_section8_t2_evidence.py` **只读 B 区 t2_wpfrag**。新增一个并行导出（建议 `scripts/export_section8_catalog_supplement.py`）：
  - 输入：`analysis/oc_data_catalog/{rollout_run_registry.csv, canonical_rollout_summary_long.csv}`。
  - 选择逻辑（**关键**）：每个 (model, train_type, seed, eval_profile) 必须按 `is_selection_eligible=1` 且优先 `resampled_traj30_*` 的 `rollout_run_id` **唯一去重**，否则会混入 traj8 的 `*_iideval_*` 探针 run 导致错误数值（本方案制定时即踩过此坑）。
  - 取数：`metric_name=final_position_error, stat_name∈{median,p95}, horizon_s=60, scope=overall`。
  - 输出：`analysis/section8_current_evidence/catalog_supplement_{aggregate,per_seed}.csv`，列含 `mirror=g3_5_5` 标记，供 §8 表合并。
  - 异常处理：按 A.2 输出 `anomaly_class ∈ {env_drift, genuine_fragility, structural_divergence}` 字段。

---

# PART B — 合并后草表（含数值，paste-ready）

> 指标：60s 自由递推**终端位置误差中位数**（先轨迹内取中位，再种子间取均值），单位 m；P95 同法取种子均值。来源镜像见每行括注。

## 表 1（T-L1）几何运动学结构 → 长期稳定性

clean 列 = clean 训练 / clean 评估；iid-噪声列 = iid_noisy_ic 训练 / nominal_eval(含噪)评估。

| 模型（角色） | clean 训练（中位数/m） | iid 噪声训练（中位数/m, nominal_eval） | 判定 |
|---|---|---|---|
| `blackbox_fullstate`（无几何全黑箱） | **5/5 发散**，85–89m / 非有限（B 区, N=5） | **发散**，~148m，2/3 非有限（A 区, N=3） | 无 SE(3) 几何 → 两种训练线下均长期发散，**噪声训练不能挽救** |
| `se3_accel_blackbox`（保 SE(3)，黑箱加速度） | 2.46（B 区, N=5；尾部由 seed42 主导, P95≈9） | 5.61（A 区, N=3） | 保几何 → 有限但方差/尾部较大 |
| `se3_momentum_blackbox`（保 SE(3)+常质量，黑箱动量） | 1.49（B 区, N=5） | 2.01（A 区, N=3） | 保几何 → 稳定 |
| `phnode_full`（完整结构化, 参照） | 0.68（B 区, N=5） | 1.09（B 区 iid, N=5） | 稳定 |

**结论（可写）**：SE(3) 位姿几何是长时递推稳定的关键结构条件；该结论**在 clean 与噪声-IC 两条训练线下都成立**（旧 §8 仅有 clean 证据，此处由 A 区补齐噪声线）。

## 表 2（T-L2）能量/结构消融阶梯（clean 训练 / clean 评估）

倍数 = 同镜像 phnode_full clean 基线（B 区 0.677 / A 区 0.61）。

| 移除的结构 | 模型 | 中位数/m | P95/m | 退化倍数(同镜像) | 来源 | N |
|---|---|---|---|---|---|---|
| —（完整模型） | `phnode_full` | 0.68 | 1.77 | 1.0× | B 区 | 5 |
| 零功率升力耦合 | `ablate_no_lift` | 0.83 | 2.15 | 1.2× | B 区 | 4 † |
| 质量先验 | `ablate_no_mass_prior` | 1.30 | 4.31 | 1.9× | B 区 | 5 |
| 能量核心（标量势 V，改一般广义力） | `phnode_qforce` | 3.76 | 11.25 | 5.5× | B 区 | 5 |
| 耦合阻尼（D 改对角） | `ablate_diag_damping` | 4.17 | 10.29 | **6.8×** | A 区 | 6 |
| 执行机构条件化（仅 actuator 状态） | `ablate_bu_only` | **6/6 发散（18–54m）** | — | — | A 区 | 6 |

† `no_lift` N=4：seed43 为**真实可复现脆弱**（44.4m, nbad=276，多次重训复现，与环境无关），单列披露，不计入稳定簇中心趋势。见表 5。

**重要含义（需在写作中处理）**：合并后阶梯为
`bu_only(6/6 发散) ≫ diag_damping(6.8×) ≈ qforce 能量核心(5.5×) > no_mass_prior(1.9×) > no_lift(1.2×)`。
即 **耦合阻尼的退化幅度与能量核心相当、甚至略大**（执行机构条件化按 D2 报结构性发散，非有限档）。当前 §8"能量核心是最主导精度先验"的表述需细化为：
- 执行机构条件化是**结构必需**（移除即崩溃）；
- 在保几何前提下，**{能量核心、耦合阻尼}属同一量级的主导精度先验**，质量先验次之，升力耦合边际最小；
- 或：把"能量核心最主导"限定在 **{能量核心 / 质量先验 / 升力}** 这一组惯性-能量子先验内，把阻尼结构与执行条件化作为**另两条独立的结构必要性**分别陈述（更稳妥）。
> A 区 `diag_damping` 6 种子中 5 个聚于 4–6m、仅 seed44=0.69 偏低，呈"多数差+个别好"，与环境漂移"多数好+个别坏"（如 phnode_full s46）相反——这一**多种子一致性**正是按 R-A 将 ~4m 作真实效应、按定量结果采纳的依据，与被排除的单种子漂移性质不同。同环境补跑 `diag_damping` clean×5 可进一步加固，但**非必需**。

## 表 3（T-rob）干净训练下随评估扰动的鲁棒性（60s 中位数/m，B 区）

| 模型 | clean | nominal_eval | degraded_eval | heading_biased_eval |
|---|---|---|---|---|
| `phnode_full` | 0.68 | 0.96 | 1.76 | 3.13 |
| `ablate_no_lift` | 0.83 | 1.05 | 1.77 | 3.07 |
| `ablate_no_mass_prior` | 1.30 | 1.42 | 2.04 | 3.10 |
| `se3_momentum_blackbox` | 1.49 | 1.62 | 2.13 | 3.00 |
| `se3_accel_blackbox` | 2.46 | 2.62 | 3.20 | 3.83 |
| `blackbox_fullstate` | 全部 profile 5/5 发散 | | | |

**结论（可写）**：精度领先在初值扰动增强时收敛（heading 下各结构化模型与 se3_momentum 互相趋近）；但几何稳定性优势（blackbox 全发散 vs 保几何有限）在**所有** profile 下稳健。

## 表 4（T-L3）噪声协议敏感性与条件等价（nominal_eval / iid 评估，60s 中位数/m，B 区）

| 模型 | clean 训练 | **iid 训练** | **v4lite 训练** | iid↔v4lite 差 |
|---|---|---|---|---|
| `phnode_full` | 0.96 | 1.09 | 1.11 | 1.7% |
| `ablate_no_mass_prior` | 1.42 | 1.46 | 1.40 | 4.1% |
| `ablate_no_lift` | 1.05 | 1.00 | 1.27 | 26%（v4lite 单种子尾部，强扰动下收敛 <8%） |
| `phnode_qforce` | 3.74 | 2.63 | 1.59 | 40%（qforce 高方差，两协议种子跨度均 ~3m） |

**结论（可写）**：① 对完整模型和无质量先验消融，**iid≈v4lite**（≤约 5%）；无升力耦合消融与配置广义力基线分别有 26% 与 40% 的显式差异，不能概括为四模型普遍等价；② 模型的**分层排序**不随训练协议改变（full/no_lift 保持头部组但精确次序可互换，no_mass_prior 居中，qforce 居后）；③ 噪声-IC 训练并不系统性优于 clean。→ 因此只在稳定主力模型和分层排序意义下把 v4-lite 与 iid 并列为两条噪声线，不升级为普适协议等价或 v4-lite 更优。

## 表 5（T-disc）逐种子异常透明度披露

| 模型 | 镜像 | 失败/排除种子 | 值/m | 类别（A.2） | 处理 |
|---|---|---|---|---|---|
| `phnode_full` | A 区 g3_5_5 | 42, 46 | 4.2 / 47.9 | (1) 环境漂移，当前环境不复现（审计证实） | 排除出当前证据；溯源说明保留 |
| `phnode_full` | B 区 g3_5_7 | 无 | — | — | 5/5 稳定 |
| `ablate_no_lift` | B 区 g3_5_7 | 43 | 44.4 | (2) **真实可复现脆弱，与环境无关**（多次重训确认） | 诚实记录为 no_lift 失效模式；中心趋势取 N=4 稳定簇 |
| `blackbox_fullstate` | A+B 区 | 全部 | nan / 85–148 | (3) 结构性长时发散 | 报"稳定性失败"，不报有限中位数 |

**可写的对照结论**：在**当前环境**下，完整能量结构模型 `phnode_full` 5 种子全稳；去掉升力耦合的 `ablate_no_lift` 出现**可复现脆弱种子**（seed43）。这是升力耦合结构价值的一个**训练稳定性侧证据**（不止边际精度），但样本为单种子，表述应限定为"出现一个可复现的脆弱种子"，不夸大为普遍稳定性定理。

---

## 附：本方案涉及的取数命令口径（供复核 / 工具实现）

- B 区：`analysis/section8_current_evidence/{aggregate.csv, per_seed_long.csv}`（由 `scripts/export_section8_t2_evidence.py` 生成）。
- A 区：`analysis/oc_data_catalog/canonical_rollout_summary_long.csv` 过滤
  `metric_name=final_position_error & stat_name∈{median,p95} & horizon_s=60.0 & scope=overall`，
  并按 `rollout_run_registry.csv` 的 `is_selection_eligible=1`、优先 `resampled_traj30_*` 对每 (model,train,seed,profile) **唯一去重**。
- A 区 within-mirror 基线：phnode_full clean 剔除 seed42/46 后 N=4 均值 = 0.61 m。
