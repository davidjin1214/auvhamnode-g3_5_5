# AUVHamNODE 学位论文章节写作入口

> 更新时间：2026-05-22
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
| [drafts/auvhamnode_thesis_chapter_zh.tex](drafts/auvhamnode_thesis_chapter_zh.tex) | 当前正式重写主稿骨架 | 采用“研究问题--相关基础--状态约定--模型构造--能量边界--训练验证--讨论--小结”的 8 节结构，避免把方法代号置于章标题中心 | 后续正文扩写的唯一主稿入口 |
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
| 图表清单 | ready | 开写前材料包第 7 节 | 优先制作速度变量关系图和机械核心图 |
| 旧中间稿审查与降级 | done | 审查意见文档 + deprecated 旧稿 | 旧稿只作素材库，不再逐句修改 |
| 旧框架稿降级 | done | `drafts/deprecated/auvhamnode_thesis_chapter_zh_framework_20260520.tex` | 旧 10 节框架只作历史快照和素材备份 |
| 正文草稿 | in_progress | `drafts/auvhamnode_thesis_chapter_zh.tex` 8 节重写骨架 | 按当前目录继续扩写推导、协议和讨论 |
| 第 1--2 节文献支撑返工 | done | 第 1 节已重写为以 AUV 运动建模为中心、长期状态预测为任务场景的 introduction v4；第 2 节已扩写为长论文级“相关建模基础”v1，并补强 Fossen 功率边界、受控时序预测、Neural ODE、科学机器学习、哈密顿/端口哈密顿神经模型和文献角色归纳 | 后续仅随方法章符号和引用调整做一致性修订 |
| 问题定义正文 | done | 当前主稿第 3 节已扩写为完整正文 v1，并加入速度变量关系图 | 后续仅随模型章符号调整做一致性修订 |
| 方法正文 | in_progress | 当前主稿第 4--5 节已有 SE(3)、能量、非保守力结构和能量命题 | 继续扩写第 4 节模型构造，并检查与第 2 节符号和功率边界一致性 |
| 评估设置正文 | ready | 当前主稿第 6 节已有训练目标、长期递推和结构消融设置 | P4/P5 后按 current evidence 扩写协议和结果章 |
| 结果主表导出 | pending | catalog 已有 canonical views | 导出 current evidence 主表，标注 evidence status |
| `phnode_full clean` 结果口径 | watch | provenance audit 已给 0.6767 m 对齐基线 | 正文避免旧 11 m 脆弱性叙事 |
| `ablate_no_lift clean` 结论 | blocked | seed43 clean 异常需处理 | 用户决定重跑、剔除说明或标注 needs recheck |
| noisy training 结论 | watch | 当前只能写结构相关性 | 避免写成普适增强 |
| `v4_lite` 结论 | watch | 当前是 protocol sensitivity diagnostic | 不并入主胜利，除非后续补证据 |
| 真实海试泛化 | blocked | 当前无真实海试主证据 | 只能写为局限性和未来工作 |

---

## 5. 建议的正式正文目录

当前正式主稿采用以下 8 节结构。若后续篇幅允许，评估和实验结果可单独拆成实验章。

1. **研究问题与方法概述**
   - AUV 运动建模任务及其长期状态预测对象：位置、姿态、速度等运动状态
   - 传统参数化物理建模、黑箱学习和结构化神经动力学之间的关系
   - 方法定位、简称和理论声称边界；避免提前展开后续章节的状态约定和实现细节

2. **相关建模基础**
   - AUV 六自由度坐标、速度与 Fossen 型动力学
   - 水动力参数获取、CFD/EFD/系统辨识和模型校准边界
   - 受控神经时序预测、Neural ODE 与科学机器学习
   - 哈密顿、端口哈密顿与 AUV 功率结构

3. **受控状态表示与海流速度约定**
   - 坐标系、状态与控制
   - 总体速度与相对水速度
   - 数据态、模型态与受控初值问题

4. **结构化连续时间动力学模型**
   - SE(3) 运动学
   - 相对动量与机械存储函数
   - 非保守广义力与执行器通道
   - 相对动量动力学

5. **能量平衡与理论边界**
   - 机械核心定义
   - 静水能量平衡命题和证明
   - 完整增强系统、海流外源和数值积分边界

6. **训练目标与验证协议**
   - 结构保持参数化
   - 控制块训练目标
   - 长期递推与证据口径
   - 结构消融设置

7. **讨论**
   - 端口哈密顿适用范围
   - SO(3) 数值漂移和李群积分边界
   - 非保守力可辨识性
   - 仿真证据、噪声训练和真实海试泛化边界

8. **本章小结**

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
| P2 | 扩写“受控状态表示与海流速度约定” | 正文第 3 节完整正文 v1 + 速度变量关系图 | done |
| P3 | 扩写“结构化连续时间动力学模型” | 正文第 4 节完整正文 v1 | in_progress |
| P4 | 扩写“能量平衡与理论边界” | 正文第 5 节命题、证明和边界完整版本 | ready |
| P5 | 扩写“训练目标与验证协议” | 正文第 6 节参数化、损失、rollout 和消融完整版本 | ready |
| P6 | 扩写“讨论” | 正文第 7 节完整正文 v1 | ready |
| P7 | 导出 current evidence 结果表 | 论文结果表底稿 | pending |
| P8 | 写实验结果章或结果节 | current evidence 主表和图 | blocked until P7 |
| P9 | 根据实验结果修订讨论和证据边界 | 讨论节与结果口径同步版本 | blocked until P7 |
| P10 | 写引言、摘要和章节小结 | 完整章节初稿 | pending |

### 6.1 当前下一步：P3

第 1--2 节返工已完成，可以恢复 P3。P1e 已将第 1 节改写为 introduction v4：主线不再是后续技术点的提前罗列，而是从 AUV 运动建模和长期状态预测任务出发，依次说明传统参数化模型、数据驱动模型和结构化神经动力学的作用与缺口，最后引出本文的 AUVHamNODE。P1f 已将第 2 节扩写为长论文级“相关建模基础”v1：内容覆盖 AUV 六自由度运动与功率结构、水动力参数获取与闭合困难、受控时序预测与 Neural ODE、科学机器学习、哈密顿和端口哈密顿结构的可继承部分，并补充文献角色归纳。

P1e/P1f 已完成以下内容：

1. 明确本章定位：AUV 运动建模是方法贡献，长期状态预测是应用任务和验证场景；预测对象应写为位置、姿态、速度等运动状态。
2. 将“传统方法”改写为更准确的层级：Fossen 型参数化六自由度动力学是建模主线，CFD/EFD/拖曳或 PMM 实验/自由航行试验/系统辨识主要服务于水动力系数获取、模型校准和验证。
3. 将“深度学习”改写为从黑箱受控时序预测到结构化科学机器学习的递进，区分离散状态转移、Neural ODE、PINN/UDE、哈密顿/端口哈密顿结构学习。
4. 将第 1 节重写为引言式论证链，避免把海流相对速度、执行器滞后、机械核心和实验协议等后续内容提前堆砌在引言中。
5. 保持中心定位：本文贡献是结构化连续时间假设空间，不是完整严格端口哈密顿化某一工程平台或仿真器。
6. 将贡献段从后文目录式罗列改为“问题重表述--模型构造--证据检验”的高层收束。
7. 将第 2 节写成“preliminaries + 必要研究进展”，而不是完整综述，也不是短论文式压缩背景。
8. 在第 2 节中明确 Fossen 功率账本的机械核心边界，避免把执行器、海流外源和神经广义力增强系统写成闭合严格端口哈密顿系统。
9. 将端口哈密顿模板中的耗散矩阵记为 \(\mathcal R(z)\)，避免与旋转矩阵 \(R\) 混淆。

P3 的下一步：

1. 扩写第 4 节“结构化连续时间动力学模型”，优先补强相对动量动力学、非保守广义力分支和执行器通道之间的论证衔接。
2. 检查第 4--5 节是否延续第 2 节的功率边界：只把机械核心写成端口哈密顿风格能量平衡，不把完整增强系统写成闭合标准端口哈密顿系统。
3. 确认第 4 节中所有 \(R\)、\(J\)、\(D\)、\(\mathcal R\)、\(J_\theta\)、\(D_\theta\) 的符号角色与第 2 节一致。

---

## 7. 更新规则

后续维护本 README 时，建议只更新三类内容：

1. **文档定位变化**：新增、合并或废弃写作文件时，更新第 1 节。
2. **写作进度变化**：完成草稿、导出结果表、生成图表时，更新第 4 节和第 6 节状态。
3. **证据状态变化**：重跑或修复实验、更新 canonical/evidence status 后，更新第 4 节中相关 `watch` 或 `blocked` 项。

不要把正文内容大段写进 README。README 只负责入口、定位和进度追踪。
