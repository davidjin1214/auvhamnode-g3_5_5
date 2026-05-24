# AUVHamNODE 学位论文章节写作入口

> 更新时间：2026-05-23
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
| [drafts/auvhamnode_thesis_chapter_zh.tex](drafts/auvhamnode_thesis_chapter_zh.tex) | 当前正式重写主稿 | 当前文件中第 1--5 节已按 10 节扩展结构完成阶段性正文；第 6--7 节已切换为新标题并保留待扩写正文；第 8 节结果分析入口已建立 | 后续正文扩写的唯一主稿入口 |
| [drafts/deprecated/auvhamnode_thesis_chapter_zh_framework_20260520.tex](drafts/deprecated/auvhamnode_thesis_chapter_zh_framework_20260520.tex) | 已降级旧框架稿 | 保留 2026-05-20 前 10 节框架、旧标题体系和已迁移正文素材 | 仅在追溯旧框架或迁移遗漏素材时查阅 |
| [drafts/deprecated/auvhamnode_thesis_chapter_zh_intermediate_20260519.tex](drafts/deprecated/auvhamnode_thesis_chapter_zh_intermediate_20260519.tex) | 已降级旧中间稿 | 保留旧稿公式、段落和表格素材，但不再作为主稿逐句修改 | 仅在迁移素材时查阅 |

---

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

2026-05-22 后续章节结构复核后的补充规则：第 1--3 节基本保留，不再继续扩写为新的综述或方法清单；第 4 节以后应从旧的 8 节压缩方法章调整为 10 节扩展结构。新增桥梁节“从 Fossen 能量结构到结构保持学习模型”，用于承接 Fossen 型功率结构、工程化 AUV 建模相对理论模板的差异、端口哈密顿表述的适用边界和结构保持学习的事前约束思想；原“能量平衡与理论边界”应改名为“结构化模型的能量性质与功率关系”，作为主方法的结构性质分析，而不是与主方法并列的新贡献。“机械核心”可在正文中经过定义后作为分析对象或内部简称使用，但不作为正式一级标题中心词；“理论声称边界”不作为正式标题，应转化为“适用条件”“功率关系”“能量性质”等学术表达。后续实验结果应在 current evidence 主表导出后单独成节，不再只附着在“训练目标与验证协议”中。

2026-05-23 第 4 节修订后的补充规则：第 4 节已明确为理论桥梁节，不以 REMUS100 或任何具体工程仿真器作为理论支点；具体平台和仿真器应主要放在实验设置、数据来源或讨论中。第 4 节的功率分析直接采用第 3 节的速度契约：机械存储、阻尼和广义力功率配对使用相对水速度 \(\nu_r\)，位置运动学使用总体速度 \(\nu_b\)。正式中文写作中，autonomous underwater vehicle 中的 vehicle 统一译为“航行器”“水下航行器”或“AUV”。

2026-05-23 第 1--5 节审稿式复核与修订后的补充规则：以高水平审稿和资深中文编辑视角对第 1--5 节做了一次独立复核与修订。主要改动：第 2 节与第 4 节去除 Fossen 功率平衡和端口哈密顿桥梁的重复推导，第 4 节改为回指第 2 节并只保留相对水速度特化；第 2.5 节增补与 SE(3) 哈密顿/端口哈密顿 Neural ODE 及拉格朗日神经网络的对标差异段，并补充李群与结构保持几何积分文献；第 5 节在首次使用处给出与第 6 节正式定义一致的“机械核心”工作定义（含正定耗散项），统一 \(c_\tau\) 记号、执行器时间常数与 coadjoint 约定表述；第 1 节增补四点概念性贡献清单。新增对标文献 `duong2021se3hamnode`（RSS 2021）与 `duong2024liegroupphnode`（IEEE T-RO 2024）的著录细节已于 2026-05-23 核实补全：前者补 `volume = {XVII}` 与 `doi = 10.15607/RSS.2021.XVII.086`；后者确认作者列表（Duong, Altawaitan, Stanley, Atanasov）并补 `volume = {40}`、`pages = {3695--3715}`、`doi = 10.1109/TRO.2024.3428433`，`% NOTE` 已清除，bibtex 编译无 undefined citation。

2026-05-25 第 8 节口径决策与 T2 当前证据重跑（与用户确认，落稿前先定口径=路径 C）：

- 第 8 节六条口径已定：(1) `phnode_full clean` 用 provenance audit 对齐基线 0.6767 m，旧 ~11 m 只入方法论/局限；(2) `ablate_no_lift clean seed43` 异常采用**重跑**（不剔除），由 T2 在 current-main 环境补回；(3) `v4_lite` 见下条；(4) noisy training 写“与结构强耦合、非普适增强”，证据用 `ablate_no_mass_prior`（matched 5/6 seed 受益），**绕开被 seed46 污染的 `phnode_full` clean-vs-noisy 数值**；(5) 主表 `clean+nominal_eval ×{pos_err_median@60s, completion@60s}`，degraded/heading + P95 + 能量/SO(3) 诊断进子表；(6) 真实海试只入第 9 节。
- v4-lite **提高定位**（用户决定）：从“仅协议敏感性脚注”提升为第 8 节一个正式“噪声下结构差异响应诊断”小节，承载四点论断——①完整结构模型在噪声评估/训练下保持稳健；②去掉 lift 在噪声训练下退化 → lift 对噪声鲁棒性有贡献【条件：T2 确认非 seed44 artifact】；③`ablate_no_mass_prior`（仅去 physics-based mass 初始化、子模块仍在）噪声下受益 → 噪声下不宜盲设 mass 初值、宜让模型自学【证据最强】；④clean 与噪声下结构**响应方向相反（delta 符号相反，非排名整体翻转）**。该定位仍**不**声称 v4-lite 训练协议优于主协议，边界 #6 不变。
- **caveat A（必须先过）**：`ablate_no_lift` 噪声退化很可能由单个 seed44 驱动（noisy nominal 3.72 m，clean 同 eval 已偏高 2.03 m；tracker 记为“noisy 下新 seed44 异常”），与 seed46/seed43 同属环境 artifact 风险。论断②在 T2 用 current evidence 确认 seed44 非 artifact 前不得写死。
- **caveat B**：噪声鲁棒性主证据用 matched clean-train vs noisy-train（4 个 eval profile），v4-lite 差异响应作机制佐证；phnode_full 噪声叙事绕开 seed46。
- **T2 已发起**：按模型拆 4 个 notebook（`notebook/t2_wpfrag_{phnode_full,phnode_qforce,ablate_no_lift,ablate_no_mass_prior}.ipynb`，生成器 `notebook/make_t2_notebooks.py`），矩阵=模型×{clean, iid_noisy_ic@nominal_train, v4_lite@nominal_train}×seed42-46，三套 checkpoint 跨 clean/nominal/degraded/heading_biased 评估；补 `_audit_meta` provenance 与 no-successful-batches anomaly 扫描。用户在 Colab 并行执行中。回灌 catalog 并清零 seed44 问题后方可写第 8 节。

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
| claim-evidence 表 | ready | 开写前材料包第 4 节 | 写结果章前按最新 catalog 再校验一次 |
| 实验矩阵 | ready | 开写前材料包第 6 节 | 根据实际可用结果确定主表和附表 |
| 图表清单 | ready | 开写前材料包第 7 节 | 速度变量关系图已完成并接入第 3 节；下一步优先制作“六自由度机械子系统能量/功率结构图” |
| 旧中间稿审查与降级 | done | 审查意见文档 + deprecated 旧稿 | 旧稿只作素材库，不再逐句修改 |
| 旧框架稿降级 | done | `drafts/deprecated/auvhamnode_thesis_chapter_zh_framework_20260520.tex` | 旧 10 节框架只作历史快照和素材备份 |
| 正文草稿 | in_progress | `drafts/auvhamnode_thesis_chapter_zh.tex` 第 1--7 节已完成阶段性正文，10 节结构全部就位；2026-05-23 对 §1--5 做审稿式复核修订，2026-05-24 完成 §6 4 子节重构与 §7 基线/消融拆分，`drafts/auvhamnode_thesis_chapter_zh.pdf` 已按最新 TeX 编译（35 页，无 undefined reference）；§8 实验结果为唯一空节，§9 讨论与 §10 小结正文已在位 | 导出 current evidence 主表后填写第 8 节，并据 §8 结果回修 §9/§10 与摘要口径 |
| 第 1--2 节文献支撑返工 | done | 第 1 节已重写为以 AUV 运动建模为中心、长期状态预测为任务场景的 introduction v4；第 2 节已扩写为长论文级“相关建模基础”v1，并补强 Fossen 功率边界、受控时序预测、Neural ODE、科学机器学习、哈密顿/端口哈密顿神经模型和文献角色归纳表；2026-05-23 复核中于第 2.5 节增补 SE(3) (p)H-NODE 与拉格朗日神经网络对标差异段及李群几何积分文献 | 对标文献著录已于 2026-05-23 核实补全（见“对标文献著录核实”项）；终稿前完成 2024--2026 citation audit |
| 对标文献著录核实 | done | `duong2021se3hamnode`（RSS 2021，vol. XVII，DOI 10.15607/RSS.2021.XVII.086）与 `duong2024liegroupphnode`（IEEE T-RO，vol. 40，pp. 3695--3715，2024，DOI 10.1109/TRO.2024.3428433）著录已核实补全；Lagrangian NN/DeLaN/Finzi、李群几何积分等其余条目著录无误 | 注：当前 `plainnat.bst` 不打印 DOI；若终稿需显示 DOI，需换用支持 DOI 的样式或加载相应宏包 |
| 问题定义正文 | done | 当前主稿第 3 节已扩写为完整正文 v1，加入三层速度变量关系图，并补充状态表示与外源变量假设、仿真基准可观测性边界 | 后续仅随模型章符号和外源变量定义调整做一致性修订 |
| 前 3 节正式表达润色 | done | 已收束第 1 节过强文献空缺表述，改写第 2--3 节残留章节调度语气，并校正相对水速度/总体速度表述 | 前 3 节后续只随第 4--5 节符号、边界和引用一致性做局部修订 |
| Fossen-to-structure 桥梁节 | done | 当前主稿第 4 节“从 Fossen 能量结构到结构保持学习模型”v2，已统一 \(\nu_r\) 功率配对、弱化具体仿真器叙事并补充端口哈密顿适用边界，最新 PDF 已编译 | 后续只随第 5--6 节符号和功率边界做一致性修订 |
| 方法正文 | done | 当前主稿第 5 节“结构化连续时间动力学模型”v1 已完成，补强增强模型态、SE(3) 运动学、相对动量、势能广义力、非保守广义力、执行器通道和相对速度动力学 | 后续只随第 6 节功率边界和符号一致性做局部修订 |
| 能量性质正文 | done | 2026-05-24 将第 6 节按 4 子节展开（六自由度机械子系统与存储函数 / 耗散、零功率耦合与广义力功率 / 静水条件下的功率平衡命题和证明 / 海流、执行器与增强状态的适用范围），保留并复用第 5 节定义与公式、不重复推导，命题加 `\label` 可交叉引用，PDF 已重新编译 | 后续只随第 5/7 节符号与边界一致性做局部修订 |
| 评估设置正文 | done | 2026-05-24 将第 7 节补强为 5 子节，把原合并子节拆为“基线模型与结构消融链条”（含模型比较表）与“结构消融设置”（4 项消融逐项映射到第 6 节结构性质），并明确结果留待第 8 节、不提前写性能结论 | 后续只随第 6 节性质和第 8 节结果口径做一致性修订 |
| T2 当前证据重跑 | in_progress | 4 个按模型 notebook + 生成器（`notebook/t2_wpfrag_*.ipynb`, `make_t2_notebooks.py`），2026-05-25 提交；矩阵=4 模型×{clean, iid_noisy_ic@nominal_train, v4_lite@nominal_train}×seed42-46，跨 4 eval profile，含 `_audit_meta` provenance 与 anomaly 扫描 | 用户 Colab 并行执行中；回灌后本地重建 catalog、导出 §8 主表、复核 seed44 |
| 结果主表导出 | pending（待 T2） | catalog 已有 canonical views（catalog era）；current evidence 待 T2 回灌 | T2 回灌后导出 current evidence 主表，标注 evidence status |
| `phnode_full clean` 结果口径 | decided | 口径=0.6767 m 对齐基线；旧 ~11 m 只入方法论/局限；T2 重跑回灌当前数字 | 写 §8 时用 0.6767 m，并核对校正后家族排名 |
| `ablate_no_lift clean` 结论 | decided（重跑）→ in_progress | 口径=重跑（非剔除）；seed43 由 T2 在 current-main 补回 | T2 回灌后用干净 seed43 重判家族 clean 最稳；并清零 seed44（见 caveat A） |
| noisy training 结论 | decided | 口径=结构强耦合、非普适；证据用 `ablate_no_mass_prior`，绕开 seed46 污染的 phnode_full | 主轴用 matched clean-vs-noisy；写 §8 时按此口径 |
| `v4_lite` 结论 | decided（提高定位） | 升为 §8“噪声下结构差异响应诊断”小节承载四点论断，但仍不声称协议胜利（#6 不变）；论断②条件于 caveat A | T2 确认 seed44 非 artifact 后写入四点论断 |
| caveat A：ablate_no_lift seed44 | open（待 T2） | 噪声退化疑由 seed44 单 seed 驱动（与 seed46/43 同 artifact 风险） | T2 用 current evidence 确认 seed44 非 artifact，否则论断②降级 |
| 真实海试泛化 | blocked | 当前无真实海试主证据 | 只能写为局限性和未来工作 |

---

## 5. 建议的正式正文目录

2026-05-23 复核后，正式正文建议采用以下 10 节结构。当前 TeX 主稿第 1--7 节已按该结构完成阶段性正文（§6/§7 子节于 2026-05-24 补齐）；第 8 节结果分析入口已建立，待 current evidence 主表导出后填写；第 9 讨论、第 10 小结正文已在位。第 1--7 节后续只随符号、引用和后文一致性做局部修订。

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

4. **从 Fossen 能量结构到结构保持学习模型**
   - Fossen 型六自由度模型中的能量--功率结构
   - 理论模板与工程化 AUV 建模的差异
   - 端口哈密顿表述在海流、执行器和经验水动力条件下的适用边界
   - 从事后验证能量结构转向事前约束可学习动力学模型

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
| P3a | 新增并修订“从 Fossen 能量结构到结构保持学习模型” | 正文第 4 节完整正文 v2 + 最新 PDF 编译 | done |
| P3b | 重构“结构化连续时间动力学模型” | 正文第 5 节完整正文 v1 | done |
| P4 | 扩写“结构化模型的能量性质与功率关系” | 正文第 6 节命题、证明和适用范围完整版本（4 子节） | done |
| P5 | 扩写“训练目标、基线体系与验证协议” | 正文第 7 节参数化、损失、rollout、基线和消融完整版本（5 子节） | done |
| P6 | 导出 current evidence 结果表 | 论文结果表底稿 | pending |
| P7 | 写“实验结果与结构证据分析” | current evidence 主表和图 | blocked until P6 |
| P8 | 扩写“讨论”并同步结果口径 | 正文第 9 节完整正文 v1 | blocked until P7 |
| P9 | 写本章小结并回修摘要/第 1 节收束段 | 完整章节初稿 | pending |

### 6.1 当前下一步：P6 → §8

第 1--7 节已形成完整论证链并完成阶段性正文：第 1 节建立 AUV 运动建模和长期状态预测任务，第 2 节给出相关建模基础，第 3 节固定数据态--模型态和海流速度契约，第 4 节完成 Fossen 能量结构到结构保持学习模型的桥梁，第 5 节给出结构化连续时间动力学模型，第 6 节（2026-05-24，P4）按 4 子节给出机械核心定义、功率角色分析、静水功率平衡命题与证明、适用范围，第 7 节（2026-05-24，P5）按 5 子节给出结构保持参数化、控制块训练目标、长期递推与证据口径、基线模型与结构消融链条、结构消融设置。§6 严格定位为 §5 主方法的结构性质分析，引用 §5 公式而非重复推导；§7 各消融逐项映射到 §6 结构性质，并明确性能结论留待 §8。

当前应转入 P6 与 §8。P6 从 catalog 的 `canonical_rollout_*` 视图导出 current evidence 主表（带 evidence status 过滤），然后填写第 8 节“实验结果与结构证据分析”。§8 写作前需先确定证据口径：`phnode_full clean` 用对齐基线、`ablate_no_lift clean seed43` 异常处理、`v4_lite` 定位、noisy training 口径、主表/附表范围、真实海试仅入局限性（见第 4 节看板 `watch`/`blocked` 项）。§8 完成后再据结果回修 §9 讨论、§10 小结与摘要口径。

---

## 7. 更新规则

后续维护本 README 时，建议只更新三类内容：

1. **文档定位变化**：新增、合并或废弃写作文件时，更新第 1 节。
2. **写作进度变化**：完成草稿、导出结果表、生成图表时，更新第 4 节和第 6 节状态。
3. **证据状态变化**：重跑或修复实验、更新 canonical/evidence status 后，更新第 4 节中相关 `watch` 或 `blocked` 项。

不要把正文内容大段写进 README。README 只负责入口、定位和进度追踪。
