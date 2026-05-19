# AUVHamNODE 学位论文章节写作入口

> 更新时间：2026-05-19  
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
| [drafts/auvhamnode_thesis_chapter_zh.tex](drafts/auvhamnode_thesis_chapter_zh.tex) | 当前正式重写主稿骨架 | 以正式方法章为目标，重建命名、结构、速度契约、机械核心、验证协议和边界 | 后续正文扩写的唯一主稿入口 |
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

> **AUVHamNODE 的贡献不是把 REMUS100 工程模拟器逐项改写成严格端口哈密顿系统，而是把 AUV 六自由度动力学中最关键的几何、能量、耗散、执行器和海流速度约定，转化为可学习、可消融、可长期递推验证的连续时间结构化假设空间。**

必须持续遵守的边界：

1. 不写“完整闭合严格端口哈密顿 AUV 系统”。
2. 不写“普通 ODE 数值积分严格保持 SO(3)”。
3. 不把 `B_net` 写成标准 pH 输入矩阵 \(G(q)u\)。
4. 不把海流写成闭合 Hamiltonian 环境子系统。
5. 不把旧 `phnode_full clean seed42/46` 异常作为当前模型脆弱性证据。
6. 不把 `v4_lite` 写成已确认优于主协议的最终胜利。
7. 正式正文不保留内部写作指令或章节调度语言。

---

## 4. 写作进度看板

状态标签：

- `done`：材料已完成，可直接使用。
- `ready`：材料足够，可以开始写正文。
- `in_progress`：已有正文或骨架，正在扩写。
- `pending`：尚未完成，需要补充。
- `blocked`：需要实验、数据或用户决策后才能推进。
- `watch`：可写但必须带边界或证据状态说明。

| 模块 | 状态 | 当前产物 | 下一步 |
|---|---|---|---|
| 总体写作指南 | done | 总指南、伴随文件、决策备忘、材料包 | 后续只在发现具体缺口时补充 |
| 论文主线定位 | done | 专家复核与落稿决策备忘 | 正文中保持“结构化连续时间假设空间”主线 |
| 符号表 | done | 开写前材料包第 2 节 | 写正文时迁移为论文符号表 |
| 数据空间/模型空间定义 | ready | 开写前材料包第 3 节 | 优先写成正式正文“问题定义” |
| 理论到代码映射 | done | 开写前材料包第 5 节 | 后续可转成“实现细节”表 |
| claim-evidence 表 | ready | 开写前材料包第 4 节 | 写结果章前按最新 catalog 再校验一次 |
| 实验矩阵 | ready | 开写前材料包第 6 节 | 根据实际可用结果确定主表和附表 |
| 图表清单 | ready | 开写前材料包第 7 节 | 优先制作速度契约图和机械核心图 |
| 旧中间稿审查与降级 | done | 审查意见文档 + deprecated 旧稿 | 旧稿只作素材库，不再逐句修改 |
| 正文草稿 | in_progress | `drafts/auvhamnode_thesis_chapter_zh.tex` 重写骨架 | 按正式正文扩写问题定义、方法推导和验证协议 |
| 问题定义正文 | in_progress | 重写骨架已有基本定义 | 扩展状态、控制、坐标系和受控初值问题 |
| 速度契约正文 | in_progress | 重写骨架已有核心公式和命题 | 补充数据空间、模型空间、输出空间的正式论证 |
| 方法正文 | in_progress | 重写骨架已有 SE(3)、能量和非保守力结构 | 补完整能量证明、实现映射和模型对比 |
| 实验协议正文 | in_progress | 重写骨架已有验证协议 | 后续按 current evidence 扩写结果章 |
| 结果主表导出 | pending | catalog 已有 canonical views | 导出 current evidence 主表，标注 evidence status |
| `phnode_full clean` 结果口径 | watch | provenance audit 已给 0.6767 m 对齐基线 | 正文避免旧 11 m 脆弱性叙事 |
| `ablate_no_lift clean` 结论 | blocked | seed43 clean 异常需处理 | 用户决定重跑、剔除说明或标注 needs recheck |
| noisy training 结论 | watch | 当前只能写结构相关性 | 避免写成普适增强 |
| `v4_lite` 结论 | watch | 当前是 protocol sensitivity diagnostic | 不并入主胜利，除非后续补证据 |
| 真实海试泛化 | blocked | 当前无真实海试主证据 | 只能写为局限性和未来工作 |

---

## 5. 建议的正式正文目录

学位论文章节可采用以下结构。若篇幅允许，可以拆成“方法章”和“实验章”两个章节。

1. **研究问题与动机**
   - AUV 六自由度动力学学习挑战
   - 黑箱模型与解析模型的互补缺口
   - 本章目标和贡献

2. **理论基础**
   - Neural ODE
   - Hamiltonian Neural Network
   - Port-Hamiltonian system
   - Fossen 海洋航行器被动性

3. **问题定义与速度契约**
   - 状态、控制和海流变量
   - 数据空间总体速度
   - 模型空间相对水速度
   - 数据态与模型态转换

4. **结构化连续时间模型（AUVHamNODE）**
   - SE(3) 运动学
   - 相对动量与哈密顿量
   - 正定耗散和斜对称零功率耦合
   - 可学习广义力分支
   - 执行器滞后和外源携带通道
   - 机械核心能量平衡命题

5. **神经网络参数化与实现映射**
   - \(M_\theta^{-1}\)、\(V_\theta\)、\(D_\theta\)、\(J_\theta\)、\(\tau_\theta\)
   - `to_ode_state` / `to_data_state`
   - 模型注册、基线和消融

6. **训练与评估协议**
   - REMUS100 风格数据
   - 控制块训练
   - 噪声初值 profile
   - rollout benchmark
   - 指标和 evidence status

7. **实验结果与分析**（需导出 current evidence 后扩写）
   - 短时控制块预测
   - 长期 rollout
   - 海流速度契约验证
   - 噪声初值鲁棒性
   - 结构消融
   - 能量和 SO(3) 诊断

8. **讨论与局限性**
   - 端口哈密顿声称边界
   - 可辨识性边界
   - 数值积分与 SO(3) 漂移
   - 海流与执行器建模边界
   - provenance 和环境漂移
   - 真实数据泛化

---

## 6. 近期写作计划

| 阶段 | 任务 | 产物 | 状态 |
|---|---|---|---|
| P0 | 整理指南与材料入口 | 本 README | done |
| P1a | 审查旧中间稿 | `paper/drafts/auvhamnode_thesis_chapter_review_notes_zh.md` | done |
| P1b | 降级旧中间稿 | `paper/drafts/deprecated/auvhamnode_thesis_chapter_zh_intermediate_20260519.tex` | done |
| P1c | 创建正式重写骨架 | `paper/drafts/auvhamnode_thesis_chapter_zh.tex` | done |
| P2 | 写“问题定义与速度契约” | 正文第 2 节正式扩写 | in_progress |
| P3 | 写“SE(3) 运动学与机械核心” | 正文第 4 节前半 | in_progress |
| P4 | 写“非保守力分解、执行器与海流通道” | 正文第 4 节后半 | in_progress |
| P5 | 写“能量平衡命题与边界” | 命题、证明要点和边界说明 | pending |
| P6 | 写“实现映射与模型对比” | 实现表、模型结构表 | pending |
| P7 | 导出 current evidence 结果表 | 论文结果表底稿 | pending |
| P8 | 写“训练与评估协议” | 正文协议节 | in_progress |
| P9 | 写“实验结果与讨论” | 结果章初稿 | blocked until P7 |
| P10 | 写引言、摘要和章节小结 | 完整章节初稿 | pending |

---

## 7. 更新规则

后续维护本 README 时，建议只更新三类内容：

1. **文档定位变化**：新增、合并或废弃写作文件时，更新第 1 节。
2. **写作进度变化**：完成草稿、导出结果表、生成图表时，更新第 4 节和第 6 节状态。
3. **证据状态变化**：重跑或修复实验、更新 canonical/evidence status 后，更新第 4 节中相关 `watch` 或 `blocked` 项。

不要把正文内容大段写进 README。README 只负责入口、定位和进度追踪。
