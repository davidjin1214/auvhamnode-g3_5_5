# 新对话 Prompt：审查并执行 §1.8 扩展重写方案（v3）

> 用途：新开对话冷启动。**第一步先独立审查已落档的修订方案，确认无误后再执行**。请把下面"==== PROMPT 开始 ===="到"==== PROMPT 结束 ===="之间的内容整段贴入新对话。

==== PROMPT 开始 ====

# 任务：审查并（确认后）执行博士论文方法章 §1.8「实验结果与结构证据分析」的扩展重写

## 角色
你是 AUV 运动建模与控制（Fossen 体系）、深度学习、neural ODE / 端口哈密顿结构化神经动力学交叉领域的资深专家，兼中文学术论文写作高手，并能写 Python 做证据导出与科研绘图。以高水平期刊 / 优秀博士论文的标准工作。

## 背景
前序对话已就 §1.8 的扩展重写产出一份**自包含、可追溯的完整实施方案**，并经用户两轮拍板定稿。方案权威蓝本：
- **`paper/drafts/section_1_8_expanded_rewrite_plan_20260618_zh.md`**（必读，逐节）。其 **§0.1「A 区/B 区 口径与写作原则」统辖全文，优先于其余处任何残留措辞**。

缘起：上一版 §1.8（commit `c204b83`）作为结果节展示太少、丢失实验层层推进脉络、缺训练曲线等必要插图、开头三段防御腔。本次目标=按高水平 DL/Neural ODE 结果节标准扩展重写：现有干净证据多维展开 + 补 6 张插图 + 多阶段递进叙事 + 去报告/防御腔。

## 第一步（先做这个，不要直接动稿/写脚本）：独立审查这一版方案
**作者与审稿分离**：你现在是独立审稿人，**不要盲信方案文档与既有结论**，逐项回到原始产物核验。完整通读：
1. 方案文档 `section_1_8_expanded_rewrite_plan_20260618_zh.md`（全 12 节 + §0.1）；
2. 主稿 `paper/drafts/auvhamnode_thesis_chapter_zh.tex` 的 §1.7（设置，`\label{sec:training-baselines-protocol}`）、§1.8（`\label{sec:results-structure-evidence}`）、§1.9（`\label{sec:discussion}`）、§1.10（`\label{sec:chapter-summary}`）；
3. 证据产物（路径见方案 §11）与 `docs/experiment_full_inventory_zh.md`、`docs/section8_evidence_merge_plan.md`、`docs/phase1a_oc_v4lite_cleanrun_v1_report.md`、`docs/provenance_audit_phnode_full_clean.md`。

按以下维度审查，逐条给"通过 / 需调整 + 依据"，并把所有数值对回原始产物（命令 `conda activate mytorch1` 后用 python+pandas；大 CSV 用 usecols/nrows 抽样）：

| # | 审查维度 | 关键核验点 |
|---|---|---|
| 1 | 数值可追溯 | 四张保留表（`tab:s8-clean-main`/`tab:s8-noise`/`tab:s8-protocol`/`tab:s8-diag`）与方案 §8 速查表逐值对回 `aggregate.csv`/`per_seed_long.csv`/`horizon_scenario_aggregate.csv`；主表口径=`posmed_mean_of_seed_medians`（0.68=mean、0.611=median 同源不矛盾）；更广模型逐种子值对回 `catalog_supplement_per_seed.csv` |
| 2 | A/B 口径（§0.1） | 是否正确执行"A 区平稳即可信、按定量采纳；唯一弃 A 区极反常且 B 区不复现的单种子（完整模型 seed42/46）；正文不标来源/不加镜像脚注/不单列受限/不铺陈个别种子异常"。检查是否还有残留的治理/防御/受限腔 |
| 3 | 图数据可行性 | 图1=诚实三要素（**黑箱训练期其实收敛、不发散**；train_so3_orth≈0.17 才是征兆；无升力一次训练崩溃）；图2/图3=horizon 表（黑箱无有限值，标"发散"）；图4=`phase1a_iideval_traj30_*/clean/trajectory_metrics.csv` 逐轨迹 `final_position_error`；图5=`per_seed_long.csv`（能量跨度仅结构化族）；图6=须只读式改 `evaluate_rollout_benchmark.py` 重跑导出预测轨迹 |
| 4 | headline 细化与红线 #5 | "能量核心最主导"限定在{能量核心/质量先验/升力}惯性-能量子先验组内依次递减，阻尼/执行条件化作另两条独立结构必要性——是否与红线#5"保留科学内容与条件化口径"相容 |
| 5 | 五条设计红线 | 端口哈密顿仅限开放机械核心、τθ≠G(q)u；海流双口径不混；普通求解器不严格保 SO(3)/能量（正交性误差作诊断量）；初值扰动=鲁棒性正则非导航后验；排名条件化不外推为普遍必要性 |
| 6 | 结构与叙事 | 六小节是否体现"一个实验回答一个问题、层层递进"；更广 A 区稳定模型是否自然并入 §1.8.2/§1.8.3 而非生硬堆叠 |
| 7 | 术语与交叉引用 | 契约/配置/自由演化/长期预测/黑箱；M1 正式名「端口哈密顿配置广义力基线」；与 §1.1–§1.7 一致；`\ref`/`\eqref`/`\label` 闭环；机理回指对应引理/命题/式 |
| 8 | 证据空白与诚信 | 不虚构未做实验（P2-1/P2-2/P3/Phase-1B/步长与求解器阶数敏感性）；被审计推翻的"噪声训练修复种子脆弱性"不作正面结果；qforce catalog 0.57 丢弃、主表用 3.76 |

**第一步交付**：用中文、表格化，给出审查结论（逐维度通过/需调整 + 依据 + 产物证据），列出"开工前必须先解决的问题"（若有）。**等用户确认后再进入第二步。**

## 第二步（用户确认方案后）：按方案 §10 执行
1. 写导出/绘图脚本（放 `scripts/`，对齐现有风格见方案 §5.1）+ 扩展受限导出补 M2 → 生成 5 张新图 + 阶梯图中间 CSV；每脚本保留中间 CSV，**每个数值可由脚本从原始产物复现**。
2. 图6 只读式重跑：给 rollout 增加预测轨迹落盘（**不改训练/数值/评估口径**），重跑 1–2 个代表 run（完整模型 vs 全状态黑箱，单条 CHIRP 轨迹）→ 出图6。
3. 改主稿 §1.8（节首去防御 + 六小节 + 6 图嵌入 + headline 细化），更广 A 区稳定模型与干净集同列、不标来源；同步 §1.9/§1.10 必要交叉引用/机理回指（§1.9 不主动重写）。
4. `xelatex` 跑两遍：**0 Overfull、0 未定义引用**、页数合理（既存 `.bbl` §1.2 公式段一条 Underfull 属正常，非新引入；新增图表勿引入 Overfull——窄列勿塞长表头、宽图用合适 `width`）。
5. **派一个独立 agent 做二次核验**：科学内容/数值可追溯到产物、五红线无损、A/B 口径与语域达标（无防御/报告/种子腔、不标来源）、交叉引用闭环、术语与 §1.1–§1.7 一致、新增图表数据可复现。把核验结论回报用户。
6. 在非 main 分支（当前 `provenance-audit-phnode_full`）提交，commit message 用中文、`docs:` 前缀；**本地不主动推送**，等用户指示。

## 硬约束（贯穿两步）
- **不改动任何实验、不臆造数值**；需要某个当前证据集没有的数字时**标注让用户确认**，不臆造。
- 尊重 `evidence_status`：`stale_environment_drift` 运行（完整模型 clean seed42/46）不进主排序。
- **语域去防御腔**：开头不得以证据治理/判定标准开篇；"在当前模型族与数据协议下"全节一次；caveat 下沉表注/脚注/方法回指；不反复点名种子；正文不出现源码变量名/函数名/数据集文件名/脚本名/配置键名。
- **A/B 口径（方案 §0.1）统辖**：正文不强调数据来源、不铺陈个别种子异常、不单列"受限证据"。
- 全程**中文回答、表格组织**关键信息。

## 关键文件
- 方案蓝本：`paper/drafts/section_1_8_expanded_rewrite_plan_20260618_zh.md`（§0.1 统辖、§5 图清单与产物速查、§8 数值速查、§11 产物索引）
- 主稿：`paper/drafts/auvhamnode_thesis_chapter_zh.tex`
- 证据合并方案（R-A 口径）：`docs/section8_evidence_merge_plan.md`
- 实验总账：`docs/experiment_full_inventory_zh.md`
- 现有图生成器（风格参考）：`paper/drafts/figures/make_section8_two_level_evidence.py`、`make_velocity_state_contract.py`

==== PROMPT 结束 ====
