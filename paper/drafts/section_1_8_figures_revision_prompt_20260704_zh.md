# 任务：完成 §1.8 插图的图级修订（承接 2026-07-04 文字修订轮的"可选未做项"）

> 归档说明：本 prompt 供新会话冷启动执行。文字层修订已于 2026-07-04 完成（commit
> 439d109 / 384f5ae / 3228bc1 / 6b1d884，独立核验 PASS with notes），本轮只处理
> 修订说明第四节显式记录的三项图级遗留工单。

## 角色
你是 AUV（水下自主航行器）运动建模与深度学习交叉领域的资深专家、严格的中文博士学位论文审稿人，同时是熟练的 matplotlib 科研绘图工程师。本轮为**图级修订工单**：文字层修订已全部完成并通过独立核验，你只处理遗留的图级事项。

## 环境与路径
- 仓库根：/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/auv_se3node/g3_5_5（分支 provenance-audit-phnode_full）
- 主稿：paper/drafts/auvhamnode_thesis_chapter_zh.tex（行号会漂移，按 \section/\subsection 标题与原文摘录定位）
- 编译：`export PATH="/Library/TeX/texbin:$PATH"` 后
  `latexmk -xelatex -interaction=nonstopmode auvhamnode_thesis_chapter_zh.tex`；
  验收基线：**0 Overfull、0 未定义引用、58 页**（当前 HEAD 即此状态）
- Python：/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin/python
- 目标图：`figures/section8_perturbation_gradient.pdf`（图 1.10，fig:s8-perturbation）与
  `figures/section8_ablation_ladder.pdf`（图 1.8，fig:s8-ablation-ladder）；生成脚本在
  `scripts/` 下（先 `grep -rl "section8_perturbation_gradient\|section8_ablation_ladder" scripts/`
  定位）；数据底座 `analysis/section8_current_evidence/`（figure_data/ 中间 CSV 未入 git
  但在工作区，缺失时可由对应 export 脚本重生成）

## 必读（按顺序）
1. paper/drafts/section_1_8_1_9_revision_notes_20260704_zh.md ——上一轮修订说明，
   **第四节"可选未做项"即本轮工单来源**。
2. paper/drafts/section_1_8_1_9_review_report_20260703_zh.md 中 8-4（图级部分）、
   8-10、8-13 三项原文。
3. 主稿 §1.8.3（阶梯图段与题注）、§1.8.4（扰动段与图 1.10 题注）、§1.9.1
   （"约 5.5 倍"出现处）。
4. 两张目标图的生成脚本全文（改前必须读懂现有取数与格式化逻辑）。

## 工单清单（三项）

### 工单 1：图 1.10 横轴标签（8-4 图级遗留，必做）
现状：横轴标签 "initial-condition perturbation (increasing)" 隐含单轴强度序，与已修订的正文口径冲突（名义/退化为随机强度递增两档，航向偏置为叠加固定偏航偏置的**系统性误差档**，三档按**诱发的终端误差**排列）。改法：去掉 "(increasing)"，改为如 "initial-condition perturbation profile (ordered by induced error)" 或仅 "initial-condition perturbation profile"（按 90 mm 图面空间取舍，刻度标签 clean/nominal/degraded/heading-biased 本身自明）。改后检查图 1.10 题注与新轴标签是否冗余，必要时微调题注（题注中"各档按其诱发的终端误差从小到大排列"的释义应保留，图注自包含优先）。

### 工单 2：图 1.8 发散条画法（8-10，必做）
现状：广义力条件化消融（图内行名 Narrow Actuation）以斜纹条终止于约 8.2 m 处，可能被误读为有限中位数约 8 m（图注虽已声明"不计入有限阶梯"）。改法二选一：条形改为**开放端箭头**延伸出右轴，或条形保留斜纹+轴外注记 "diverged (tens of m)"。图内文字纯英文；与题注"六次重复全部长期发散、终端误差达数十米量级"保持一致。若在图内标注具体量级范围，须先用 pandas 回验 `analysis/section8_current_evidence/catalog_supplement_per_seed.csv`（A4 逐次中位数 18.36–53.77 m）。

### 工单 3：5.5x → 5.6x 口径统一（8-13，随工单 2 顺手做，默认执行）
精确值 5.5511（3.7564/0.6767），一位小数四舍五入为 5.6；图内其余倍数标注（1.2x/1.9x/2.2x/6.2x）均为正确的一位小数四舍五入，唯此处为 5.5x，口径不统一。既然工单 2 必然重生成该图，顺手统一：**先查脚本如何产生 5.5x**（格式化方式或数据列取值），确认后图内改为 5.6x，并同步正文两处"约 5.5 倍"→"约 5.6 倍"（§1.8.3 阶梯段、§1.9.1 能量机理段）。若查明脚本/数据有其他一致性理由支持 5.5，可回退为"三处均维持 5.5 不动"，但须在修订说明中记录理由。

## 绘图工程硬约束
1. **数值零漂移**：本轮只改标签、画法与标注格式，任何条形/曲线/须线的数值一律不得变化。重生成后用改前 PDF 对照（或对照 figure_data CSV）确认。
2. **eval_protocol 过滤坑**（既往实测踩过）：`aggregate.csv`/`horizon_scenario_aggregate.csv` 同一 (model, train, eval_profile) 在 nominal_eval 下有 `iid_noisy_ic` 与 `v4_lite` 两行；取数规则 clean→`eval_protocol=clean`、nominal/degraded/heading→`iid_noisy_ic`。检查目标脚本已有断言；若无，补上。
3. **风格对齐现有图**：mm 尺寸、既有 PALETTE 与字体设置、输出 PDF+PNG+SVG、图内纯英文短名；插图四原则（不强调种子、释义入图注、多子图 a/b 标注、零重叠）。
4. 重生成后将 PNG 用图像查看方式亲自目检两张图（标签完整、无裁切、无重叠、发散条画法不再可误读）。

## 红线与语域（照旧守护）
端口哈密顿语义限开放机械核心；海流双口径不混淆；无"严格保群/保结构积分"声明；初值扰动=鲁棒性侧面；排名结论条件化。图内英文短名属图例惯例；正文与题注不得出现源码标识符、数据集文件名、"A 区/B 区"字样。

## 工作方法
1. 定位并读懂两个绘图脚本 → 逐工单修改 → 重生成（PDF+PNG+SVG）→ PNG 目检。
2. 主稿正文/题注同步改动（工单 1 题注微调、工单 3 正文两处）。
3. latexmk 编译核验（0 Overfull、0 未定义引用、页数 58±浮动排版影响，如页数变化须解释）。
4. commit（中文 paper 前缀，如「paper: 完成 §1.8 插图图级修订（8-4 轴标签/8-10 发散条/8-13 倍数口径）」；图+脚本+tex 同一逻辑单元；**只 commit、不推送**）。
5. 派一个轻量独立核验 agent：三工单落实核验、两图 PNG 目检、数值零漂移抽查（阶梯各条值/扰动曲线端点对 CSV）、图文一致性（题注与图面）、编译状态。
6. 在 paper/drafts/section_1_8_1_9_revision_notes_20260704_zh.md 末尾追加「六、图级工单执行记录（补记）」：逐工单记录改法、脚本与文件、核验结论；第四节相应条目标注"已于图级轮完成"。

## 范围边界
只动：两张目标图及其脚本、工单 3 涉及的正文两处、（如需）两图题注微调、修订说明补记。其余正文、其他图（含图 1.5——无需重生成，8-14 已用文字解决）、analysis/、docs/ 一律不动。
