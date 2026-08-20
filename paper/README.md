# AUVHamNODE 学位论文章节写作入口

> 更新时间：2026-07-26
> 目的：梳理 `paper/` 下 AUVHamNODE 中文学位论文章节相关文档的定位、阅读顺序和后续写作进度。
> 使用原则：本目录中的指南和材料包服务于正式正文写作，但它们本身不是正文。正式正文应保留定义、假设、命题、实验协议、证据边界和讨论，避免保留“建议如何写”“后续再处理”等内部备忘录语气。

---

## 1. 当前文档定位

| 文件 | 定位 | 主要解决的问题 | 何时使用 |
|---|---|---|---|
| [auvhamnode_paper_writing_guide_expert_revised_zh.md](auvhamnode_paper_writing_guide_expert_revised_zh.md) | 完整写作总指南 | 系统给出论文定位、术语、章节结构、理论推导、实验设计、图表、禁用表述和审稿问答 | 需要全面理解 AUVHamNODE 论文应包含什么时 |
| [auvhamnode_formal_writing_companion_zh.md](auvhamnode_formal_writing_companion_zh.md) | 正式写作伴随文件 | 强化“三个账本”、摘要/引言主线、方法章组织、证据阶梯和审稿风险控制 | 准备正式开写、确定叙事顺序和段落逻辑时 |
| [auvhamnode_expert_review_decision_notes_zh.md](auvhamnode_expert_review_decision_notes_zh.md) | 专家复核与落稿决策备忘 | 固化最终论文定位、必须保留的贡献点、理论边界、实验证据红线，以及内部指南到正式正文的转化规则 | 判断“什么能写、什么不能写、如何转化成正文”时 |
| [auvhamnode_thesis_chapter_prewrite_pack_zh.md](auvhamnode_thesis_chapter_prewrite_pack_zh.md) | 学位论文章节开写前材料包 | 集中整理符号表、claim-evidence 表、理论到代码映射、实验矩阵、图表清单、正文种子段落和待补材料 | 真正开始写正文前，作为底稿索引和任务清单使用 |
| [drafts/auvhamnode_thesis_chapter_review_notes_zh.md](drafts/auvhamnode_thesis_chapter_review_notes_zh.md) | 旧稿审查意见与重写约束 | 记录旧中间稿的问题、可保留素材、重写原则和新稿结构 | 继续写正文前，先确认不重复旧稿问题 |
| [drafts/auvhamnode_thesis_chapter_structure_reaudit_20260530.md](drafts/auvhamnode_thesis_chapter_structure_reaudit_20260530.md) | 终稿级结构复审与重构追踪 | 复核第 1.4 节以后章节功能、理论重复、证据链和 `v4_lite` 口径问题，并给出后续修改追踪表 | 进入正文重构前，用作任务清单和进度追踪 |
| [drafts/auvhamnode_section_1_6_energy_review_20260602.md](drafts/auvhamnode_section_1_6_energy_review_20260602.md) | 第 1.6 节专项复核与精修方案 | 记录“结构化模型的能量性质与功率关系”的机械核心边界、命题条件、海流附加功率项和图文一致性精修方案 | 修改第 1.6 节正文或机械核心功率图前使用 |
| [drafts/auvhamnode_thesis_terminology_closure_20260726_zh.md](drafts/auvhamnode_thesis_terminology_closure_20260726_zh.md) | 耗散性质与图内术语闭环记录 | 区分当前实现的严格正定耗散和功率命题所需的半正定条件，并记录图源、导出产物与整章 PDF 的一致性验证 | 核对正定/半正定表述或追溯 2026-07-26 图件修订时 |
| [drafts/figure_review_prompt_20260607.md](drafts/figure_review_prompt_20260607.md) | 新增两张插图的独立审查 prompt | 自包含审查指令：§1.3 `fig:method-architecture-overview` 与 §1.4 `fig:fossen-role-mapping` 两张图及 caption 草稿的审查范围、红线、维度与改进授权（含绕过沙箱的渲染路径与脚本/产物路径） | 在新对话中复审这两张方法图与图注、或据此重画时 |
| [drafts/section_1_7_rewrite_draft_v10_zh.tex](drafts/section_1_7_rewrite_draft_v10_zh.tex)（含 v2–v9 逐版稿与 [v6](drafts/section_1_7_v6_review_report_20260616_zh.md)/[v7](drafts/section_1_7_v7_review_report_20260617_zh.md) 评审报告） | 第 1.7 节扩展重写工作稿 | 记录第 1.7 节『训练目标/实验设置/验证协议』从 v2 起的逐版扩展重写与三方独立评审反馈，作为并入主稿前的迭代底稿 | 修改第 1.7 节正文或核对其评审反馈时 |
| [drafts/auvhamnode_thesis_chapter_zh.tex](drafts/auvhamnode_thesis_chapter_zh.tex) | 当前正式重写主稿 | 当前文件已形成 10 节完整章节初稿：第 1--7 节完成方法、能量性质和验证协议；第 8 节已按 current evidence、B1 anomaly 口径和两级结构证据框架写入结果；第 9--10 节已同步讨论和小结；PDF 已生成 | 后续终稿级复核、局部润色和图表补强的唯一主稿入口 |
| [drafts/deprecated/auvhamnode_thesis_chapter_zh_framework_20260520.tex](drafts/deprecated/auvhamnode_thesis_chapter_zh_framework_20260520.tex) | 已降级旧框架稿 | 保留 2026-05-20 前 10 节框架、旧标题体系和已迁移正文素材 | 仅在追溯旧框架或迁移遗漏素材时查阅 |
| [drafts/deprecated/auvhamnode_thesis_chapter_zh_intermediate_20260519.tex](drafts/deprecated/auvhamnode_thesis_chapter_zh_intermediate_20260519.tex) | 已降级旧中间稿 | 保留旧稿公式、段落和表格素材，但不再作为主稿逐句修改 | 仅在迁移素材时查阅 |

---

> `paper/drafts/` 下有 60 余个文件。哪些是活稿、哪些是历史评审记录，见 [`drafts/INDEX.md`](drafts/INDEX.md)（纯文件清单，不含写作决策）。

## 2. 推荐阅读顺序

### 2.1 首次完整理解

1. 先读 [auvhamnode_paper_writing_guide_expert_revised_zh.md](auvhamnode_paper_writing_guide_expert_revised_zh.md)：建立全局内容边界。
2. 再读 [auvhamnode_formal_writing_companion_zh.md](auvhamnode_formal_writing_companion_zh.md)：理解正式论文叙事主线。
3. 再读 [auvhamnode_expert_review_decision_notes_zh.md](auvhamnode_expert_review_decision_notes_zh.md)：确认最终可写/不可写边界。
4. 最后用 [auvhamnode_thesis_chapter_prewrite_pack_zh.md](auvhamnode_thesis_chapter_prewrite_pack_zh.md)：开始整理正文素材。

### 2.2 准备正式写正文

直接从 [drafts/auvhamnode_thesis_chapter_review_notes_zh.md](drafts/auvhamnode_thesis_chapter_review_notes_zh.md) 和 [drafts/auvhamnode_thesis_chapter_zh.tex](drafts/auvhamnode_thesis_chapter_zh.tex) 开始。写正文时，以重写主稿为唯一主线，必要时回查：

- 概念或推导不清楚：查总指南；
- 叙事顺序不清楚：查正式写作伴随文件；
- 不确定能否这样写：查专家复核与决策备忘。
- 需要迁移符号、claim-evidence、实验矩阵或图表清单：查开写前材料包。

### 2.3 写作中遇到证据风险

优先查 [auvhamnode_expert_review_decision_notes_zh.md](auvhamnode_expert_review_decision_notes_zh.md) 的“实验证据使用红线”和 [auvhamnode_thesis_chapter_prewrite_pack_zh.md](auvhamnode_thesis_chapter_prewrite_pack_zh.md) 的 claim-evidence 表。

---

## 3. 当前统一写作决策

正式学位论文章节应按以下定位展开：

> **AUVHamNODE 的贡献不是替代传统参数化模型，也不是把某一具体平台或工程仿真器逐项改写为严格闭合的哈密顿/端口哈密顿系统，而是提出面向长期状态预测的 AUV 结构化神经动力学建模方法：将位姿几何、能量与耗散约束以及可学习非保守作用组织为可训练、可消融、可长期递推验证的连续时间假设空间。**

标题、摘要和第 1 节必须区分两个层次：**AUV 运动建模**是本章的方法贡献，**长期状态预测**是应用任务和验证场景。若使用“预测”，应明确预测对象是位置、姿态、速度等 AUV 运动状态，而不是笼统写成“AUV 长期预测”。

第 1 节承担引言功能，不应把第 3--6 节的状态约定、海流相对速度、执行器滞后、机械核心和实验协议提前压缩成概念清单。更合适的组织是：AUV 运动建模及长期状态预测需求、传统参数化模型的价值与局限、黑箱学习模型的价值与局限、结构化神经动力学的启发、本文方法的高层定位与章节安排。

2026-05-22 第 1--2 节返工后的补充规则：引言要站在更高层次上建立问题张力、研究缺口和方法切入点，而不是把后文概念提前平铺。增强 AUV 专属性时，应通过“为什么长期递推需要受控动力学结构约束”来推进论证；不要用“六自由度、体坐标速度、相对水速度、耗散、执行器、海流”等概念清单替代引言逻辑。贡献段也不应写成后续章节目录的压缩版，而应概括本章如何把长期状态预测重新表述为受控连续时间动力学学习问题，并说明模型构造和证据检验逻辑。第 2 节已扩写为“preliminaries + 必要研究进展”：它应支撑后续状态约定、模型构造、能量命题和训练验证协议，而不是扩展成完整综述。

2026-05-22 前 3 节复核后的补充规则：第 2 节可以保留较一般的受控连续时间学习语言，但必须通过过渡说明与第 3 节的本文状态表示衔接；第 3 节应明确当前实现中的增强状态定义，即数据保存总体速度、模型内部使用相对水速度，控制命令、海流速度和深度上下文作为单个求解窗口内的分段常值外源变量，实际执行器状态按一阶滞后演化。相关建模基础中已经加入文献角色归纳表，避免把文献综述写成无差别罗列；第 3 节已加入“状态表示与外源变量”假设和仿真基准可观测性边界。后续复核重点仍应转向第 4--5 节，检查模型构造、相对动量动力学和能量命题是否延续这些定义。

2026-05-22 前 3 节正式表达润色后的补充规则：第 1 节关于已有数据驱动方法和文献缺口的表述应保持任务特定和有边界，不把“本文长期状态预测任务仍需要的统一表述”写成“既有研究完全缺失”的绝对断言；第 2--3 节的过渡应写成状态空间和外源变量边界，而不是“后续章节安排”；第 3 节训练监督处应明确速度损失比较的是相对水速度 \(\nu_r\)，避免把相对水速度、总体速度和泛称的水动力速度混用。

2026-05-22 后续章节结构复核后的补充规则：第 1--3 节基本保留，不再继续扩写为新的综述或方法清单；第 4 节以后应从旧的 8 节压缩方法章调整为 10 节扩展结构。新增桥梁节用于承接 Fossen 型能量--功率关系、工程化 AUV 建模相对理论模板的差异、端口哈密顿表述的适用边界和结构保持学习的模型定义阶段约束思想；原“能量平衡与理论边界”应改名为“结构化模型的能量性质与功率关系”，作为主方法的结构性质分析，而不是与主方法并列的新贡献。“机械核心”可在正文中经过定义后作为分析对象或内部简称使用，但不作为正式一级标题中心词；“理论声称边界”不作为正式标题，应转化为“适用条件”“功率关系”“能量性质”等学术表达。后续实验结果应在 current evidence 主表导出后单独成节，不再只附着在“训练目标与验证协议”中。

2026-05-23 第 4 节修订后的补充规则：第 4 节已明确为理论桥梁节，不以 REMUS100 或任何具体工程仿真器作为理论支点；具体平台和仿真器应主要放在实验设置、数据来源或讨论中。第 4 节的功率分析直接采用第 3 节的速度契约：机械存储、阻尼和广义力功率配对使用相对水速度 \(\nu_r\)，位置运动学使用总体速度 \(\nu_b\)。正式中文写作中，autonomous underwater vehicle 中的 vehicle 统一译为“航行器”“水下航行器”或“AUV”。

2026-05-23 第 1--5 节审稿式复核与修订后的补充规则：以高水平审稿和资深中文编辑视角对第 1--5 节做了一次独立复核与修订。主要改动：第 2 节与第 4 节去除 Fossen 功率平衡和端口哈密顿桥梁的重复推导，第 4 节改为回指第 2 节并只保留相对水速度特化；第 2.5 节增补与 SE(3) 哈密顿/端口哈密顿 Neural ODE 及拉格朗日神经网络的对标差异段，并补充李群与结构保持几何积分文献；第 5 节在首次使用处给出与第 6 节正式定义一致的“机械核心”工作定义（含正定耗散项），统一 \(c_\tau\) 记号、执行器时间常数与 coadjoint 约定表述；第 1 节增补四点概念性贡献清单。新增对标文献 `duong2021se3hamnode`（RSS 2021）与 `duong2024liegroupphnode`（IEEE T-RO 2024）的著录细节已于 2026-05-23 核实补全：前者补 `volume = {XVII}` 与 `doi = 10.15607/RSS.2021.XVII.086`；后者确认作者列表（Duong, Altawaitan, Stanley, Atanasov）并补 `volume = {40}`、`pages = {3695--3715}`、`doi = 10.1109/TRO.2024.3428433`，`% NOTE` 已清除，bibtex 编译无 undefined citation。

2026-05-25 第 8 节口径决策与 T2 当前证据重跑（与用户确认，落稿前先定口径=路径 C）：

- 第 8 节六条口径已定：(1) `phnode_full clean` 用 provenance audit 对齐基线 0.6767 m，旧 ~11 m 只入方法论/局限；(2) `ablate_no_lift clean seed43` 异常采用 **B1：按统一 anomaly 判据移出定量聚合（N=4）+ 透明注记**（详见下方第三条，取代原"重跑"口径）；(3) `v4_lite` 见下条；(4) noisy training 写“与结构强耦合、非普适增强”，证据用 `ablate_no_mass_prior`（matched 5/6 seed 受益），**绕开被 seed46 污染的 `phnode_full` clean-vs-noisy 数值**；(5) 主表 `clean+nominal_eval ×{pos_err_median@60s, completion@60s}`，degraded/heading + P95 + 能量/SO(3) 诊断进子表；(6) 真实海试只入第 9 节。
- v4-lite **提高定位**（用户决定）：从“仅协议敏感性脚注”提升为第 8 节一个正式“噪声下结构差异响应诊断”小节，承载四点论断——①完整结构模型在噪声评估/训练下保持稳健；②去掉 lift 在噪声训练下退化 → lift 对噪声鲁棒性有贡献【条件：T2 确认非 seed44 artifact】；③`ablate_no_mass_prior`（仅去 physics-based mass 初始化、子模块仍在）噪声下受益 → 噪声下不宜盲设 mass 初值、宜让模型自学【证据最强】；④clean 与噪声下结构**响应方向相反（delta 符号相反，非排名整体翻转）**。该定位仍**不**声称 v4-lite 训练协议优于主协议，边界 #6 不变。
- **caveat A（必须先过）**：`ablate_no_lift` 噪声退化很可能由单个 seed44 驱动（noisy nominal 3.72 m，clean 同 eval 已偏高 2.03 m；tracker 记为“noisy 下新 seed44 异常”），与 seed46/seed43 同属环境 artifact 风险。论断②在 T2 用 current evidence 确认 seed44 非 artifact 前不得写死。
- **caveat B**：噪声鲁棒性主证据用 matched clean-train vs noisy-train（4 个 eval profile），v4-lite 差异响应作机制佐证；phnode_full 噪声叙事绕开 seed46。
- **T2 已发起**：按模型拆 4 个 notebook（`notebook/t2_wpfrag_{phnode_full,phnode_qforce,ablate_no_lift,ablate_no_mass_prior}.ipynb`，生成器 `notebook/make_t2_notebooks.py`），矩阵=模型×{clean, iid_noisy_ic@nominal_train, v4_lite@nominal_train}×seed42-46，三套 checkpoint 跨 clean/nominal/degraded/heading_biased 评估；补 `_audit_meta` provenance 与 no-successful-batches anomaly 扫描。用户在 Colab 并行执行中。回灌 catalog 并清零 seed44 问题后方可写第 8 节。

2026-05-25 T2 当前证据回灌与分析结果（**本块的 base-rate 补测与"§8 暂缓"前瞻结论已被下方第三条取代，仅留作历史记录**；与用户确认）：

- **数据已回灌并分析**：4 模型 × 3 训练协议（clean/iid_noisy_ic@nominal_train/v4_lite@nominal_train）× seed42-46 全部回灌（`checkpoints/sweep_oc_phase1a_decision_{clean,iid,v4lite}_t2_wpfrag_*`）。因 catalog builder 未识别 `sweep_oc_phase1a_decision_*`/`smoke1_*` 命名（会把单 seed smoke1 误当 primary 污染 canonical），改用**定向导出**（不重建共享 catalog）：脚本 `scripts/export_section8_t2_evidence.py` → `analysis/section8_current_evidence/{per_seed_long,aggregate}.csv`。
- **口径核实**：跨 seed 聚合 = per-seed median → seed 间**取 MEAN**（复现 oc_followup §3.2 的 9.2746 = mean({4.20,0.84,0.73,1.16,47.85,0.87})，对离群 seed 敏感）；旧 noisy training 协议确认 = `nominal_train@iid_noisy_ic`（41 run），故 T2 iid-train 为正确 matched 对照。
- **核心发现：噪声训练"获益"本质是救援环境脆弱的 clean-train seed。** 逐 seed 对比旧 catalog vs T2：旧 phnode_full clean 9.275 全靠 seed46=47.9m+seed42=4.2m **环境伪影**撑高、noisy"巨大获益"=救援 seed46；旧 ablate_no_mass_prior"5/6 受益"主驱动是 noisy 救援偏高的 clean seed45（1.976→1.261）。T2 好环境里这些 clean seed 本就健康 → 噪声训练对结构化模型**无净收益（略损）**，唯一明显获益者是弱基线 `phnode_qforce`（v4lite-tr 把它从 ~3.8m 拉到 ~1.0m）。
- **phnode_full clean = 0.68m 稳健夺冠**（5 seed 无坍缩，与 provenance 审计 0.6767m 吻合）→ 红线 #5 有了全新干净证据，旧 11m 伪影确认不复现。
- **v4-lite 四点论断 ②③ 在 matched 当前证据下不复现（暂挂起）**：② `ablate_no_lift` iid-train 全 seed 健康（seed44=0.83m，旧 3.72m 伪影不复现）→ **caveat A 解决，方向与 ② 相反**：去 lift **不**在噪声训练下退化；③ `ablate_no_mass_prior` 仅 2/5 seed 从 noisy 训练受益、净值略变差 → "噪声下受益"不成立。④"结构响应方向相反"退化为"弱基线获益、结构化模型不获益"。
- **新异常：`ablate_no_lift` clean seed43 在 T2/g3_5_7 可复现坍缩（60s 中位 44.5m，best_loss 0.217，best_epoch 19，确定性逐位复现）**，但旧 catalog 同 seed 健康（0.66m）→ 环境分歧；仅 clean 训练发作（iid/v4lite 同 seed 健康 0.99m）。签名 = epoch~240-300 `no successful training batches (pred=20)`，**与 seed46 同类（pred-divergence）**（早期降到 epoch19/0.217 后发散不恢复，best 取 epoch19）；故 anomaly 扫描应能捕获。seed46 在 g3_5_7 健康、seed43 在 g3_5_7 坍缩 → 同一失败模式在不同 环境×seed 组合上发作，强化"环境敏感 artifact 类"假设，但 seed43 在 g3_5_7 上确定性可复现。
- **用户决策**：seed43 **已确定性复现**（44.5m / loss0.217@epoch19，T2 与补实验首跑逐位一致），故从补实验中**剔除**；补实验改为只测 **base-rate**：clean-only 训 `ablate_no_lift` seeds{47-51}（notebook `notebook/t2supp_nolift_seedscan.ipynb` + 生成器 `notebook/make_t2supp_nolift_notebook.py`，直调 `train_all_models_noise_profile.sh`+`batch_eval_models.sh`，零 driver 改动，suite=`sweep_oc_phase1a_decision_clean_t2supp_nolift`）。§8 → **暂缓**，待 base-rate 定后按当前证据重定四点论断（不得照旧写 ②③）。
- **补实验脚本坑（已修）**：训练须传 `--noise-protocol auto` 而非 `clean`——`clean` 会让训练后多 profile held-out 评估在 `resolve_noise_protocol('clean', profile=nominal_eval)` 处崩溃（train_utils.py:180）并中止整个 sweep（首跑即因此只产出失败的 seed43）。`auto` 对 clean profile=无噪声训练、对 eval profile=iid_noisy_ic，与 phase1a driver 的 `clean auto` 一致。
- **待补实验回答**：去-lift 在 clean 训练下的坍缩 base-rate（seed47-51 有几个 >10m）。≥1 个坍缩 → 真实（罕见）no-lift clean 训练脆弱；全部健康 → seed43 为孤立 环境×seed artifact（seed46 类），§8 按红线 #5 处理、不作模型脆弱性证据。

2026-05-25 第三条（§8 口径定案，base-rate 补实验作废，§8 解除暂缓；与用户确认）：本条取代第二条中"base-rate 待回灌""§8 暂缓"等所有前瞻条目。

- **放弃非对称 base-rate 补测**（用户决定）：原计划只给 `ablate_no_lift` 补 clean seed47-51 测坍缩 base-rate，但只给单一模型加测会与其余 3 模型在不同样本量/不同待遇下比较，破坏 T2「4 模型 × 3 协议 × seed42-46」唯一公平基准（与红线 #5 同属"选择性举证"的反面）。补实验 notebook/suite 作废，`notebook/t2supp_nolift_seedscan.ipynb` 不再执行。
- **seed43 落标准 5 seed 内**：核 4 个 `t2_wpfrag_*_completed.ipynb` 确认 T2 标准 seed = `42 43 44 45 46`，seed43、seed44 均在标准集内（非额外 seed）。
- **§8 口径 = B1（用户拍板）**：seed43 按**统一 anomaly 判据**（`no successful training batches` / nbad>0，与 notebook 对 4 模型一致的扫描）标为 flagged 训练失败，**移出定量聚合**（`no_lift` clean → N=4，over seed{42,44,45,46}），并以透明注记保留——`aggregate.csv` 记 `n_seeds_total`/`n_anomaly_excluded`/`excluded_seeds`/`excluded_seed_posmed`，正文写"5 个 clean seed 中 1 个（seed43）灾难性梯度训练失败、签名同已知数值伪迹类、无 base-rate 故不量化为脆弱性结论"。**不**与 `phnode_full` seed42/46 处置双标（同签名同默认处置）。lift 结构价值改由含噪 rollout 退化承载，不靠 seed43。导出脚本已落实 B1（`scripts/export_section8_t2_evidence.py`，nbad 逐 run 从 `training.log` 计）。
- **全矩阵 anomaly scan 结论（60 run）**：唯一 flagged = `no_lift` clean seed43（nbad=276）。`phnode_full`/`phnode_qforce`/`ablate_no_mass_prior` 全 0/15；`phnode_full` clean seed42/46 在干净镜像恢复正常（0/5）→ 旧 ~11m 确认环境漂移、红线 #5 获独立验证。`no_lift` seed44 在 clean/iid/v4lite 全 nbad=0 → caveat A 训练崩溃面排除（与第二条"②方向相反"一致）。
- **catalog 重建非 §8 路径，且已加护栏**：builder（`derive_experiment_bucket`/`iter_suite_dirs`）不识别 `sweep_oc_phase1a_*` 命名，会把单 seed smoke1 默认当 primary/canonical 污染（实测污染 126 decision + 48 smoke1 行）。已在 `iter_suite_dirs` 加 `sweep_oc_phase1a_*` 跳过护栏并重建，本地 canonical 已洗净（phase1a=0、smoke1=0）且永久防复发。§8 证据仍只读定向导出 `analysis/section8_current_evidence/`，**不读 canonical**。
- **B1 clean/clean 主表头**（pos 60s 中位、seed 间均值）：`phnode_full` 0.677(N=5) ≈ `ablate_no_lift` 0.829(N=4, 剔 seed43@44.38) < `ablate_no_mass_prior` 1.297(N=5) < `phnode_qforce` 3.756(N=5)。§8 可据此写作。

2026-05-25 第四条（§8 主表口径细化 + 黑箱当前证据补齐 Path B；与用户确认）：

- **主表/读法三取舍（用户拍板）**：① co-primary 指标改 {pos_err_median@60s, pos_P95@60s}，completion@60s 降为健全性脚注；② §8 头号证据放**整体结构**而非单个消融；③ seed43 推进 N=1、强 hedge 的"lift 有助训练稳定"读法（不量化）。
- **事实核准（§7 术语对齐）**：`phnode_qforce` = **去标量势能 V(q)/能量核心**基线（`auv_baselines.py:411`，保留 SE(3)+动量+co-adjoint+D/J/B，仅把 −dV/dq 换成通用位形广义力 → 能量跨度 nan 是结构性后果）。§7 已把它归为"结构化 pH 基线"测"能量核心"，与 No Lift/No Mass Prior（消融、查组件边际）分列。故 §8 头号命题精确锚定为**"保守势能/能量核心是长程精度头号承重先验"**；消融阶梯：去能量核心(qforce)×5.5 ≫ 去 mass 先验×1.9 > 去 lift×1.2。
- **旧 catalog 黑箱不可跨表采纳（regime 不可比，非数据问题）**：同数据集 `d0be9434`、同 300ep，但旧 `sweep_oc_core_default_…20260404` suite 里 `phnode_full` clean 3/6 seed 坍缩（s43/44/46）、s42 抬到 4.2，而 T2 干净镜像 0/5 全健康(0.68)；`phnode_qforce` clean 旧=0.57(最好) ↔ T2=3.76(最差)——排名反转、环境漂移（只会变差）无法解释。**qforce builder git 历史已查**：自 `a2ca101`（2026-04-03 21:28 新建 qforce）以来 `auv_baselines.py`/`auv_model_registry.py` 逐字未变，旧 suite（04-04）即用该版；翻转来自 builder 之外（04-04 后 train_utils.py ~10 次提交、噪声/eval 引擎大改）→ 坐实旧 catalog 与 T2 非同一 regime，不能混表。
- **Path B（用户选 B）= 在 T2 regime 廉价重生成黑箱**：把 `blackbox_fullstate`/`se3_momentum_blackbox`/`se3_accel_blackbox` 在 g3_5_7 镜像跑 **clean-only × seed42-46**（每模型 5 run），eval 契约与 T2 clean 套件一致（iid_noisy_ic×4 profile + v4_lite×nominal_eval），得到与 T2 直接可比的黑箱锚点 → §8 可写"结构化 ≫ 黑箱"头号命题（两级故事：结构化≫黑箱；结构化内部 能量核心≫mass≫lift）。
- **driver 增强（向后兼容，已 smoke-test）**：`scripts/run_phase1a_oc_v4lite.sh` 加 `PHASE1A_PROTOCOLS`（默认 `clean iid v4lite`，行为逐字不变；设 `clean` 则只跑 clean）。参数化 train/audit/validate_v4/eval/register 五处；本地桩测确认默认仍三协议、`clean` 时只动 clean 套件且 validate_v4 跳过。**T2 可复现性不受影响**。
- **交付物**：生成器 `notebook/make_t2_blackbox_notebook.py` → 3 个 `notebook/t2_wpfrag_{blackbox_fullstate,se3_momentum_blackbox,se3_accel_blackbox}.ipynb`（22 cell，含 anomaly 扫描，黑箱 collapse-prone 若再坍缩按 B1 同口径处理）。`scripts/export_section8_t2_evidence.py` 的 MODELS 已加 3 黑箱（clean-only，iid/v4lite 发现 glob 自动为空）。
- **待办**：用户在 Colab 跑 3 个黑箱 notebook → 同步 `checkpoints/sweep_oc_phase1a_decision_clean_t2_wpfrag_{黑箱}` → 本地重跑 export，黑箱 clean 行自动汇入 `aggregate.csv`。无需重建共享 catalog。

2026-05-25 第五条（黑箱当前证据回灌 + 两级叙事定案；与用户确认）：Path B 三黑箱 clean×seed42-46 已回灌（各 5 run）、export 已重跑。

- **回灌结果**：3 黑箱**训练全部成功（0/5 nbad）**——与旧 catalog 的 `blackbox_fullstate` nan/86.8 截然不同，说明那是旧 regime 伪迹。但 **`blackbox_fullstate` 长程 rollout 在全部 5 seed 发散**（60s 中位 = 85.5/89.0/nan/83.0/nan，completion = 0.57/0.87/**0**/0.28/**0**）。
- **决策 A（rollout 发散口径）**：这是**有别于 B1 的第二类异常**——训练收敛(nbad=0)但自由递推发散，B1 的 nbad 判据不覆盖。已在 `export_section8_t2_evidence.py` 加 `rollout_diverged` 标记（pos 60s 为 NaN/缺失/>10m），发散 seed 与 train_anomaly 一样**移出定量聚合**并单列（`n_rollout_diverged`/`diverged_seeds`/`diverged_seed_posmed`/`diverged_completion`）；全 seed 发散的模型（`blackbox_fullstate`）**不报裸中位数**，记为"5/5 长程稳定性失败"。其余模型不受影响、无双计。
- **决策 B（两级叙事，取代"结构化≫黑箱"单调命题）**：clean/clean 排名**非单调**，真实故事分两级——
  - **几何层（稳定性命门）**：全黑箱 `blackbox_fullstate` 5/5 发散，而 `se3_momentum_blackbox`（保 SE(3)+常质量、动量黑箱）=1.49 完全稳定 → **SE(3) 流形结构是长程稳定性的必要条件**。
  - **能量层（精度驱动）**：`phnode_full`=0.68 夺冠；去标量势能 V 的 `phnode_qforce`=3.76 是**最差的结构化模型**，甚至不如半结构化黑箱 se3_momentum(1.49)/se3_accel(2.46) → 势能/能量核心是承重件（去 V 比"整段动量换黑箱"更伤，前提保住 SE(3)+质量）。消融阶梯：去能量核心×5.5 ≫ 去 mass×1.9 > 去 lift×1.2。
- **clean/clean 全表**（pos 60s 中位均值, N；P95 见 per_seed）：`phnode_full` 0.68(5) < `ablate_no_lift` 0.83(4,剔s43) < `ablate_no_mass_prior` 1.30(5) < `se3_momentum_blackbox` 1.49(5) < `se3_accel_blackbox` 2.46(5,中位1.64) < `phnode_qforce` 3.76(5,P95~13) < `blackbox_fullstate` **5/5 发散**。
- **qforce hedge 不变**：3.76 与旧 regime 0.57 冲突，但 T2 内部 5 seed 一致[2–4.5]无坍缩，合法 T2 数；"垫底结构化模型"定位带 regime hedge。
- **§8 证据已完整**：当时已确认可按 B1 + 三取舍（co-primary {pos中位,P95}@60s、completion 脚注）+ 两级框架撰写；2026-05-28 同步时，主稿已据此完成 §8 阶段性落稿。

2026-05-28 主稿进度同步：

- **§8 已完成阶段性落稿**：`drafts/auvhamnode_thesis_chapter_zh.tex` 已将 current evidence 写入“实验结果与结构证据分析”，采用 B1 训练异常口径与 rollout 发散口径；主表使用 60 s 位置误差中位数和 P95 并列指标，completion 作为健全性脚注。
- **两级证据框架已落入正文**：几何层写为全状态黑箱 5/5 长期递推发散与 \(\SE(3)\) 动量黑箱稳定之间的对照，支持“\(\SE(3)\) 几何结构是当前模型族内长期稳定递推的关键结构条件”；能量层写为完整模型 0.68 m 取得最优精度、去能量核心的端口哈密顿位形广义力基线 3.76 m 且 P95 较高，支持“机械能量核心是当前模型族中主要精度先验”。
- **§9/§10 与摘要口径已同步**：讨论中保留端口哈密顿适用范围、普通 ODE 数值积分的 SO(3) 边界、非保守力不可唯一辨识、仿真证据与真实海试泛化边界；小结已按两级证据结论收束。
- **README 早期“§8 待写/§9 blocked/§10 pending”类描述已视为过期**：后续应从当前 TeX 主稿出发做终稿级复核，而不是回到“等待撰写 §8”的状态。

2026-05-30 终稿表达同步：

- **R9 正式中文表达优化已完成**：摘要、第 7 节证据口径、第 8 节当前证据来源与异常处理、第 9 节讨论边界和第 10 节小结已做长句压缩、术语中文化和工程化表达降噪。
- **PDF 已重新编译**：`drafts/auvhamnode_thesis_chapter_zh.pdf` 已在 `mytorch1` 环境下通过 `latexmk -xelatex` 重新生成；无 undefined citation/reference、无 overfull hbox、无缺图，仅保留一个参考文献段落 underfull hbox。

2026-05-30 结构复审追加：

- 已新增 `drafts/auvhamnode_thesis_chapter_structure_reaudit_20260530.md`，用于追踪第 1.4 节以后结构重构。复审确认主稿总体未触犯理论红线，但第 1.4--1.6 节存在能量--功率关系重复，第 1.7--1.8 节需要更清楚地区分设计空间和 current evidence，第 1.9--1.10 节需要更充分回应证据边界。
- 2026-05-30 术语复核补充：正式正文中“能量”“功率”“存储函数”“耗散”“端口”“功率平衡”“能量--功率关系”等可作为规范术语使用；“功率配对”应与广义力--速度功率共轭关系相连；“机械核心”必须先定义后使用。避免把“账本”“角色转译”“事前/事后结构分析”等内部写作词带入正文，相关表达改为“能量--功率关系”“结构分解”“结构约束转换”或“由建模后的结构检验转向模型定义阶段的结构约束”。
- 复审确认第 8 节一处优先修正项：当前“轻量协议变体”表值对应 `train_protocol=clean, eval_protocol=v4_lite`，不是 `train_protocol=v4lite` 训练协议结果。后续正文重构时应优先修正该表述，避免把评估协议敏感性误写为训练协议敏感性。

2026-05-31 第 1.4 节理论桥梁全面修订：

- 第 1.4 节已采用“完整桥梁版”而非单纯压缩版：正文保留三个短小节，新增一行非编号结构映射和“Fossen 功率角色到 AUVHamNODE 结构约束”的桥接表，用来连接第 1.2 的能量--功率关系、第 1.3 的速度契约与第 1.5 的模型定义。
- 本轮写法只说明功率角色如何转化为可训练函数类约束，不重新推导 Fossen 能量平衡，不证明第 1.6 的能量命题，不提前定义完整 ODE，也不使用“机械核心”作为第 1.4 的正式分析对象。
- 表述边界已进一步收紧：可学习斜对称分支只继承零功率功能角色，不解释为 Coriolis 项逐项替代；\(\tau_\theta\) 只作为由执行器状态和外源上下文条件化的广义力通道，不写成标准 \(G(q)u\) 端口矩阵；完整航行器--执行器--环境增强系统仍不称为闭合严格端口哈密顿系统。

必须持续遵守的边界：

1. 不写“完整闭合严格端口哈密顿 AUV 系统”。
2. 不写“普通 ODE 数值积分严格保持 SO(3)”。
3. 不把 `B_net` 写成标准端口哈密顿输入矩阵 \(G(q)u\)。
4. 不把海流写成闭合哈密顿环境子系统。
5. 不把旧 `phnode_full clean seed42/46` 异常作为当前模型脆弱性证据。
6. 不把 `v4_lite` 写成已确认优于主协议的最终胜利。
7. 正式正文不保留内部写作指令或章节调度语言。
8. 中文正式正文优先使用“哈密顿”“端口哈密顿”等中文术语，首次出现时可在括号中保留英文原词；避免在中文行文中反复裸用 `Hamiltonian`、`port-Hamiltonian`。

---

## 4. 写作进度看板

状态标签：

- `done`：材料已完成，可直接使用。
- `ready`：材料足够，可以开始写正文。
- `in_progress`：已有正文或骨架，正在扩写。
- `pending`：尚未完成，需要补充。
- `blocked`：需要实验、数据或用户决策后才能推进。
- `paused`：任务本身可继续，但当前应先处理更上游的论证或材料缺口。
- `watch`：可写但必须带边界或证据状态说明。

| 模块 | 状态 | 当前产物 | 下一步 |
|---|---|---|---|
| 总体写作指南 | done | 总指南、伴随文件、决策备忘、材料包 | 后续只在发现具体缺口时补充 |
| 论文主线定位 | done | 专家复核与落稿决策备忘 | 正文中保持“结构化连续时间假设空间”主线 |
| 符号表 | done | 开写前材料包第 2 节 | 写正文时迁移为论文符号表 |
| 数据空间/模型空间定义 | ready | 开写前材料包第 3 节 | 已提前写入正式正文“问题定义” |
| 理论到参数化映射 | done | 开写前材料包第 5 节 | 转化为“可训练参数化”表述 |
| claim-evidence 表 | done | 开写前材料包第 4 节；第 8 节已按 current evidence 和两级结构证据框架落稿 | 终稿前只需复核每个结论是否仍由当前证据支撑 |
| 实验矩阵 | ready | 开写前材料包第 6 节 | 根据实际可用结果确定主表和附表 |
| 图表清单 | done（术语与图文一致性已闭环） | 速度变量关系图、模型定义总览图、机械子系统功率图均已接入；第 8 节已加入训练曲线、时域增长、轨迹示例、消融比较、误差分布、扰动梯度和内部诊断 7 图，另有 OE 投稿变体；2026-07-26 已统一方法图与结果图的英文显示名并重生成受影响产物 | 后续只随全章终稿做版式复核，不再把旧英文标签列为待办 |
| 旧中间稿审查与降级 | done | 审查意见文档 + deprecated 旧稿 | 旧稿只作素材库，不再逐句修改 |
| 旧框架稿降级 | done | `drafts/deprecated/auvhamnode_thesis_chapter_zh_framework_20260520.tex` | 旧 10 节框架只作历史快照和素材备份 |
| 正文草稿 | done（完整章节初稿） | `drafts/auvhamnode_thesis_chapter_zh.tex` 已形成 10 节完整章节：§1--7 完成方法、能量性质和验证协议；§8 已写入 current evidence 主表、鲁棒性表和内部诊断表；§9 讨论、§10 小结和摘要已同步两级证据口径；R9 正式中文表达优化和最新 PDF 编译已完成 | 下一步做终稿级复核：表格数值核对、引用/术语一致性、图表补强和必要时重新编译 |
| 第 1.7–1.10 节终稿级扩展重写 | done（专项修订闭环） | §1.7 协议口径、§1.8 六小节与 7 图、§1.9 可复现性判据及 §1.10 小结均已落稿；2026-07-04 独立审查 23 项已落实或合规处置，2026-07-12 三项图级遗留已完成 | 纳入全章终稿复核与 2024–2026 citation audit，不再单列“等待合并 v7–v10” |
| 第 1--2 节文献支撑返工 | done | 第 1 节已重写为以 AUV 运动建模为中心、长期状态预测为任务场景的 introduction v4；第 2 节已扩写为长论文级“相关建模基础”v1，并补强 Fossen 功率边界、受控时序预测、Neural ODE、科学机器学习、哈密顿/端口哈密顿神经模型和文献角色归纳表；2026-05-23 复核中于第 2.5 节增补 SE(3) (p)H-NODE 与拉格朗日神经网络对标差异段及李群几何积分文献 | 对标文献著录已于 2026-05-23 核实补全（见“对标文献著录核实”项）；终稿前完成 2024--2026 citation audit |
| 对标文献著录核实 | done | `duong2021se3hamnode`（RSS 2021，vol. XVII，DOI 10.15607/RSS.2021.XVII.086）与 `duong2024liegroupphnode`（IEEE T-RO，vol. 40，pp. 3695--3715，2024，DOI 10.1109/TRO.2024.3428433）著录已核实补全；Lagrangian NN/DeLaN/Finzi、李群几何积分等其余条目著录无误 | 注：当前 `plainnat.bst` 不打印 DOI；若终稿需显示 DOI，需换用支持 DOI 的样式或加载相应宏包 |
| 问题定义正文 | done | 当前主稿第 3 节已扩写为完整正文 v1，加入三层速度变量关系图，并补充状态表示与外源变量假设、仿真基准可观测性边界 | 后续仅随模型章符号和外源变量定义调整做一致性修订 |
| 前 3 节正式表达润色 | done | 已收束第 1 节过强文献空缺表述，改写第 2--3 节残留章节调度语气，并校正相对水速度/总体速度表述 | 前 3 节后续只随第 4--5 节符号、边界和引用一致性做局部修订 |
| Fossen-to-structure 桥梁节 | done | 当前主稿第 4 节已全面修订为“从六自由度能量--功率关系到结构保持学习模型”，包含非编号结构映射和桥接表，明确 Fossen 功率角色如何转化为正定质量、势能、耗散、零功率耦合、外部广义力项与 \(\nu_r\) 功率配对约束 | 后续只随第 1.5--1.6 节接口一致性做局部回修 |
| 方法正文 | done | 当前主稿第 5 节“结构化连续时间动力学模型”已完成定义节纯化，并接入模型定义总览图；增强模型态、SE(3) 运动学、相对动量、机械存储、保守/非保守广义力、执行器通道和相对速度动力学已形成统一模型对象 | 第 1.6 节专项复核已完成；后续按专项文档做接口一致性局部精修 |
| 能量性质正文 | done（耗散性质与图文口径已闭环） | 第 6 节已按 4 子节展开；2026-07-26 进一步明确当前参数化给出 \(D_\theta\succ0\)，而能量命题只采用较弱的 \(D_\theta\succeq0\) 条件，并同步更新机械子系统功率图和 Fossen 对应图 | 后续只随全章终稿做交叉引用和版式复核 |
| 评估设置正文 | done | 2026-05-24 将第 7 节补强为 5 子节，把原合并子节拆为“基线模型与结构消融链条”（含模型比较表）与“结构消融设置”（4 项消融逐项映射到第 6 节结构性质）；随后已补齐 current evidence、证据状态、基线文献锚定和 v4_lite 协议敏感性边界 | 后续只随第 8 节表格和最终术语统一做局部一致性修订 |
| T2 当前证据重跑 | done | 4 模型×{clean,iid,v4lite}×seed42-46 已回灌并分析；定向导出 `scripts/export_section8_t2_evidence.py` → `analysis/section8_current_evidence/{per_seed_long,aggregate}.csv`（builder 已加 `sweep_oc_phase1a_*` 护栏、本地 canonical 已洗净；§8 仍只读定向导出） | 见 §3 决策块 2026-05-25 第二/三条；结论触发 §8 叙事重定 |
| 结果主表导出 | done（current evidence, B1） | `per_seed_long.csv` 375 行（含 `train_nbad`/`train_anomaly`）+ `aggregate.csv` 75 行（B1：剔 flagged seed 后保留 `excluded_seeds`/`excluded_seed_posmed`）；clean phnode_full=0.6767m(N=5)、no_lift=0.8288m(N=4)、qforce=3.7564m(N=5) | 已用于 §8 主表；2026-07-25 已重新核对文件行数与 clean headline |
| 结构化 vs 黑箱证据（Path B） | done（已回灌+写入） | 3 黑箱 clean×seed42-46 已回灌、export 已纳入；全部 0/5 nbad（训练成功）。**`blackbox_fullstate` 5/5 rollout 发散**（85-89m/nan），`se3_momentum_blackbox`=1.49、`se3_accel_blackbox`=2.46 稳定。结论：两级叙事（SE(3) 几何→稳定性；能量核心→精度），非单调（qforce 3.76>se3 黑箱） | 已写入 §8；后续只核对表格和措辞边界 |
| rollout 发散口径（决策 A） | done（已写入） | `export_section8_t2_evidence.py` 加 `rollout_diverged`（NaN/缺失/>10m），与 train_anomaly 同样移出聚合并单列；全 seed 发散模型不报裸中位数、记“长期稳定性失败” | §8 已按该口径报告 blackbox_fullstate 5/5 发散失败，不放有限数 |
| driver 协议开关 | done（已 smoke-test） | `run_phase1a_oc_v4lite.sh` 加向后兼容 `PHASE1A_PROTOCOLS`（默认三协议不变，设 `clean` 只跑 clean）；参数化 train/audit/validate/eval/register | 供 Path B clean-only 复用；T2 可复现性不受影响 |
| `phnode_full clean` 结果口径 | done（已写入） | T2 5 seed 全健康，clean 60s=0.68m，与对齐基线 0.6767m 吻合；旧 11m 不复现 | §8 已用 0.68m current evidence；后续只保留 provenance 边界 |
| `ablate_no_lift clean` 结论 | done（B1 已写入） | seed43 落标准 5 seed 内、唯一 flagged（nbad=276）；按统一 anomaly 判据移出聚合 → no_lift clean=0.83m(N=4) ≈ phnode_full，seed43 作透明注记不量化为脆弱性；base-rate 补测已作废（见第三条） | §8 已用 N=4 数 + seed43 注记；后续避免把单次失败量化为模型脆弱性 |
| noisy training 结论 | done（已按当前证据改写） | 旧“ablate_no_mass_prior 5/6 受益”在 matched T2 不复现（仅 2/5、净略损）；当前证据：噪声获益≈救援环境脆弱 clean seed，结构化模型无净收益，仅弱基线 qforce 明显获益 | 正文已避免旧 5/6 叙事；后续只作为协议敏感性和扰动条件讨论 |
| `v4_lite` 结论 | done（协议敏感性） | 论断②（去 lift 噪声退化）与③（去 mass 初值噪声受益）matched 当前证据均不支持；④退化为“弱基线获益” | §8 已将 v4_lite 作为协议敏感性诊断，不写成主协议胜利 |
| caveat A：ablate_no_lift seed44 | resolved（against ②） | T2 中 seed44 iid-train 健康（0.83m），旧 3.72m 噪声退化为环境伪影、不复现 → 论断②失去证据 | 不再以 seed44 噪声退化支持 lift 必要性 |
| 真实海试泛化 | blocked | 当前无真实海试主证据 | 只能写为局限性和未来工作 |

---

## 5. 建议的正式正文目录

2026-05-30 同步后，当前 TeX 主稿已按以下 10 节结构形成完整章节初稿。第 1--7 节完成问题、基础、状态契约、理论桥梁、模型构造、能量性质和验证协议；第 8 节已按 current evidence 写入实验结果与结构证据分析；第 9 节讨论和第 10 节小结已同步结果口径；R9 已完成全文正式表达优化。后续目录不再重开，主要做终稿级一致性复核和图表补强。

1. **研究问题与方法概述**
   - AUV 运动建模任务及其长期状态预测对象：位置、姿态、速度等运动状态
   - 传统参数化物理建模、黑箱学习和结构化神经动力学之间的关系
   - 方法定位、简称和适用范围；避免提前展开后续章节的状态约定和实现细节

2. **相关建模基础**
   - AUV 六自由度坐标、速度与 Fossen 型动力学
   - 水动力参数获取、CFD/EFD/系统辨识和模型校准边界
   - 受控神经时序预测、Neural ODE 与科学机器学习
   - 哈密顿、端口哈密顿与 AUV 功率结构

3. **受控状态表示与海流速度约定**
   - 坐标系、状态与控制
   - 总体速度与相对水速度
   - 数据态、模型态与受控初值问题

4. **从六自由度能量--功率关系到结构保持学习模型**
   - Fossen 型六自由度模型中的能量--功率结构
   - 理论模板与工程化 AUV 建模的差异
   - 端口哈密顿表述在海流、执行器和经验水动力条件下的适用边界
   - 从建模后的结构检验转向模型定义阶段的结构约束

5. **结构化连续时间动力学模型**
   - SE(3) 运动学
   - 相对动量与机械存储函数
   - 非保守广义力与执行器通道
   - 相对动量动力学

6. **结构化模型的能量性质与功率关系**
   - 六自由度机械子系统与存储函数
   - 耗散、零功率耦合与广义力功率
   - 静水条件下的功率平衡命题和证明
   - 海流、执行器与增强状态的适用范围

7. **训练目标、基线体系与验证协议**
   - 结构保持参数化
   - 控制块训练目标
   - 长期递推与证据口径
   - 基线模型和结构消融链条
   - 结构消融设置

8. **实验结果与结构证据分析**
   - 短时控制块预测
   - 长期 rollout、completion rate 和 failure reason
   - 海流速度约定与 OC 场景证据
   - 噪声初值鲁棒性
   - 结构消融、能量诊断和 SO(3) 诊断

9. **讨论**
   - 端口哈密顿适用范围
   - SO(3) 数值漂移和李群积分边界
   - 非保守力可辨识性
   - 仿真证据、噪声训练和真实海试泛化边界

10. **本章小结**

第 6 节的定位需特别明确：它是第 5 节主方法的结构性质分析，不是另一个与主方法并列的新贡献。正式标题不使用“机械核心能量平衡与理论声称边界”；若正文需要“机械核心”这一说法，应先定义为六自由度机械子系统中的位姿、相对动量、存储函数、耗散、零功率耦合和广义力部分。

---

## 6. 近期写作计划

| 阶段 | 任务 | 产物 | 状态 |
|---|---|---|---|
| P0 | 整理指南与材料入口 | 本 README | done |
| P1a | 审查旧中间稿 | `paper/drafts/auvhamnode_thesis_chapter_review_notes_zh.md` | done |
| P1b | 降级旧中间稿 | `paper/drafts/deprecated/auvhamnode_thesis_chapter_zh_intermediate_20260519.tex` | done |
| P1c | 创建正式重写骨架 | `paper/drafts/auvhamnode_thesis_chapter_zh.tex` | done |
| P1d | 降级 10 节旧框架并重开 8 节主稿 | `paper/drafts/deprecated/auvhamnode_thesis_chapter_zh_framework_20260520.tex` + 当前主稿 | done |
| P1e | 重写第 1 节 introduction 论证 | 以 AUV 运动建模为中心、避免后文概念清单化的“研究问题与方法概述”v4 | done |
| P1f | 扩写第 2 节 preliminaries | 长论文级“相关建模基础”v1 + 核心参考文献 | done |
| P2 | 扩写“受控状态表示与海流速度约定” | 正文第 3 节完整正文 v1 + 三层速度变量关系图 | done |
| P2r | 复核并修订前 3 节正式表达 | 第 1 节收束段降清单化；第 2 节补充一般受控流到增强状态外源变量定义的过渡和文献角色表；第 3 节补充状态表示与外源变量假设、仿真基准可观测性边界 | done |
| P2s | 润色前 3 节正式表达 | 收束过强文献空缺/模型局限表述，删除章节安排式元话语，统一相对水速度与总体速度表述 | done |
| P3a | 新增并修订“从六自由度能量--功率关系到结构保持学习模型” | 正文第 4 节完整桥梁版：非编号结构映射 + Fossen 功率角色到 AUVHamNODE 结构约束桥接表 + 最新 PDF 编译 | done（2026-05-31） |
| P3b | 重构“结构化连续时间动力学模型” | 正文第 5 节完整正文 v1 | done |
| P4 | 扩写“结构化模型的能量性质与功率关系” | 正文第 6 节命题、证明和适用范围完整版本（4 子节） | done |
| P5 | 扩写“训练目标、基线体系与验证协议” | 正文第 7 节参数化、损失、rollout、基线和消融完整版本（5 子节） | done |
| P6 | 导出 current evidence 结果表 | 论文结果表底稿 | done（`analysis/section8_current_evidence/`） |
| P7 | 写“实验结果与结构证据分析” | current evidence 主表、扰动表和内部诊断表 | done：§8 已按 B1+rollout_diverged 口径、co-primary 指标和两级框架落稿 |
| P8 | 扩写“讨论”并同步结果口径 | 正文第 9 节完整正文 v1 | done：§9 已同步端口哈密顿边界、SO(3) 数值边界、非保守力可辨识性和真实海试泛化边界 |
| P9 | 写本章小结并回修摘要/第 1 节收束段 | 完整章节初稿 | done：摘要、§1 收束段和 §10 已同步两级证据结论 |
| P10 | 终稿级数值、术语和图表复核 | 第 8 节表格核对、7 张结果图与追踪清单更新 | done：§8 表格与 `analysis/section8_current_evidence/` 对齐；7 图已于 2026-06-18 接入，三项图级遗留于 2026-07-12 完成；2026-07-26 又完成耗散性质区分与图内英文术语闭环 |

### 6.1 当前下一步：完整章节初稿后的终稿级复核

当前主稿已经完成从“证据就绪”到“结果落稿”的转换。第 8 节采用 current evidence 定向导出 `analysis/section8_current_evidence/`，并按 B1 训练异常、rollout 发散、co-primary 指标和两级结构证据框架写入结果；第 9 节和第 10 节已经按这些结果同步讨论与小结。因此后续不应再按“等待写 §8”推进，而应进入终稿级复核。

2026-06-02 已完成第 1.5 节终稿级模型定义复核与模型定义总览图接入。该图采用少公式、强结构语义的英文标签，只作为第 1.5 的定义导航，不替代第 1.6 的机械核心功率图，也不表达完整航行器--执行器--环境系统的闭合储能结构。同日新增 `drafts/auvhamnode_section_1_6_energy_review_20260602.md`，用于追踪第 1.6 节机械核心定义、命题条件、海流附加功率项和图文一致性的局部精修方案。

2026-07-25 状态回查：§1.7–§1.10 专项扩展重写已经闭环。2026-07-04 的 §1.8–§1.9 独立审查共 23 项，已全部落实或按证据说明合规处置；2026-07-12 又完成扰动图轴标签、发散条开放端表达和 5.5×→5.6× 单次舍入口径三项图级遗留。当前下一步是全章终稿一致性与 2024–2026 citation audit，而不是继续合并早期 v7–v10 工作稿。

2026-08-20 全章终稿一致性与 2024–2026 citation audit 已执行，结果如下。**数字口径**：§1.8 四张结果表（主表、扰动表、协议表、诊断表）、消融阶梯倍数与两处 rollout 示例图注数字，已逐格与 `analysis/section8_current_evidence/`（`aggregate.csv`、`figure_data/{ablation_ladder,horizon_growth,diagnostics_summary}.csv`、`figure_data/rollout_example/*.npz`）核对，全部一致，未发现需要修订的数值。**交叉引用**：91 个 label 无重复、71 个 ef 无悬空。**文献库**：45 条全部被正文引用、无未定义引用；2024–2026 的 9 条（后增至 10 条）已按 DOI 与 CrossRef 逐字段核对，仅修正 `moradi2026phnnoutputerror` 作者中间名缩写。**唯一实质缺口**：`jin2025phnodeauv`（JMSE 2025, 13(11):2091）此前未被引用。该文是本章工作的已发表阶段性结果，**不是外部对照工作**，因此不进相关工作对标段：出处声明写在章首贡献段之后（§1，独立一段），说明该文同以 REMUS 100 为平台、评估时域为 10 s、对照对象为时序卷积网络基线（**不是**本章代号 M5 的全状态黑箱；该文亦已做五次随机种子重复，故重复统计不列为本章增量），本章则给出方法定义与能量性质分析、主评估时域取至 60 s、并以结构消融/初值扰动/噪声训练协议组织受控比较。§2.5 与 §1.9 的相关工作段落只保留外部工作，不把本章自己的阶段性成果列为被对标或被补充的对象。PDF 已重新编译（58 页，0 undefined citation、0 overfull box）；改稿经 `zh-academic-writing` 不变量核对：相对改稿前，原有数字、引用键、公式环境无丢失或替换，极性与情态（229 处）、条件限定词（17 处）、引号内命题（5 处）全部一致。

2026-05-30 已完成本轮最小核查：

1. **数值核对**：§8 主表、扰动表和轻量协议表已与 `analysis/section8_current_evidence/aggregate.csv` 对齐；P95、完成率、有效 \(N\)、排除种子和递推发散字段均一致。内部诊断表已按 `per_seed_long.csv` 的干净配置诊断量核对。
2. **口径核对**：正文没有把轻量协议变体写成主协议胜利，没有把无升力耦合随机种子 43 写成定量脆弱性结论，也没有把旧完整模型随机种子 42/46 异常写成当前模型脆弱性。
3. **文字和术语复核**：正文继续采用“当前证据”“长期递推”“完成率”“端口哈密顿”等中文术语；英文主要保留在首次定义、模型别名和结果来源字段中。
4. **图表补强判断（历史记录）**：2026-05-30 当时决定暂不新增结果图；该状态已被 2026-06-18 的 7 图接入和 2026-07-12 的图级修订取代，当前不再把模型结构阶梯图或 rollout 误差增长图列为缺失项。
5. **编译验证**：2026-07-26 已修改 TeX 与图源，并重新生成图件；整章编译和最终 PDF 视觉检查结果见 `drafts/auvhamnode_thesis_terminology_closure_20260726_zh.md`。

---

## 7. 更新规则

后续维护本 README 时，建议只更新三类内容：

1. **文档定位变化**：新增、合并或废弃写作文件时，更新第 1 节。
2. **写作进度变化**：完成草稿、导出结果表、生成图表时，更新第 4 节和第 6 节状态。
3. **证据状态变化**：重跑或修复实验、更新 canonical/evidence status 后，更新第 4 节中相关 `watch` 或 `blocked` 项。

不要把正文内容大段写进 README。README 只负责入口、定位和进度追踪。
