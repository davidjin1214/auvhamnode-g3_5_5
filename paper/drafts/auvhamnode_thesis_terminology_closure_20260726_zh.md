# AUVHamNODE 耗散性质与图内术语闭环记录

日期：2026-07-26

## 1. 闭环范围

本次处理前两类事项尚未闭环：

1. 正文中的“正定耗散”和“半正定耗散”分别对应什么数学性质，以及当前代码究竟保证哪一种性质；
2. 方法图和结果图仍保留 `mechanical core`、`Force port`、`over-ground velocity`、`Config Force` 等与正式正文不一致或可能误导对象边界的英文标签。

本次只修改模型说明、论文正文、制图脚本、相应导出图件和审计记录，不修改模型计算逻辑、实验协议、训练结果或证据数据。

## 2. 正定与半正定的技术结论

### 2.1 当前实现保证严格正定

`AUVHamNODE.py` 的完整耗散分支构造下三角因子 \(L_D(\xi)\)，并对其对角元施加平滑正值变换和严格正下界。由此 \(L_D\) 非奇异，

\[
D_\theta(\xi)=L_D(\xi)L_D(\xi)^\top\succ0.
\]

对角耗散变体同样对每个对角元施加平滑正值变换和严格正下界，因此也有 \(D_\theta(\xi)\succ0\)。`auv_baselines.py` 中对应结构采用相同性质的参数化。

在 `mytorch1` 环境对两种分支各实例化一次并计算特征值，所得最小特征值分别为：

- 耦合耗散：\(0.4944133\)；
- 对角耗散：\(0.7031460\)。

该数值检查用于验证实现路径，不替代上述由参数化直接得到的解析结论。

### 2.2 功率不等式只需半正定

机械子系统的功率关系含耗散项

\[
-\nu_r^\top D_\theta(\xi)\nu_r.
\]

要推出该项不向机械子系统注入能量，只需

\[
D_\theta(\xi)\succeq0
\quad\Longrightarrow\quad
\nu_r^\top D_\theta(\xi)\nu_r\ge0.
\]

因此，正文统一采用以下分工：

- 描述当前代码和模型函数类时写 \(D_\theta\succ0\) 或“正定耗散参数化”；
- 陈述一般端口哈密顿结构、能量命题的充分条件或功率不增结论时写 \(D_\theta\succeq0\) 或“半正定条件”。

严格正定是当前实现所满足的更强性质，半正定是功率结论所需的较弱条件。二者不再作为同义词混用，也不再保留为待作者二选一的术语问题。

## 3. 图内术语的对象级修订

| 原标签 | 当前标签 | 修订依据 |
|---|---|---|
| `over-ground velocity` | `absolute generalized velocity` | \(\nu_b\) 含线速度与角速度，是六维广义速度，不能缩写成通常只指线速度的 over-ground velocity。 |
| `water-relative velocity` | `water-relative generalized velocity` | 明确 \(\nu_r\) 同样是六维广义速度。 |
| `Open six-DOF mechanical core` | `Open six-DOF mechanical subsystem` | 与正文正式分析对象一致，避免未定义的 `core` 包装语。 |
| `Core state` | `Mechanical state` | 直接说明位形和动量的对象属性。 |
| `Force port`、`external port` | `External generalized force`，并显示 \(\nu_r^\top\tau_\theta\) | \(\tau_\theta\) 单独是广义力项；只有与功率共轭速度共同出现时才构成外部功率配对，且不等同于标准 \(G(q)u\)。 |
| `exogenous open port` | `exogenous variable or state` | 海流、执行器状态和深度上下文在当前图中是外源变量或携带状态，不被包装成闭合端口。 |
| `conditioning input` | `context input` | 表达计算依赖，不引入“条件化接口”这一额外概念。 |
| `power-role transfer` | `mapping by power properties` | 箭头表示功率性质约束的映射，不宣称真实水动力项与神经分支逐项等同。 |
| `Coriolis` | `Coriolis and centripetal` | 与 Fossen 术语和正文“科里奥利--向心项”一致。 |
| `Config Force` | `Configuration-force` | 统一 M1 的英文显示名。 |
| `Narrow Actuation` | `Restricted Force Inputs` | 准确描述 A4 收窄的是广义力分支输入变量范围，而不是执行器本身。 |
| `actuation conditioning` | `force-branch inputs` | 将比较轴落到实际发生变化的模型输入对象。 |

方法图中的耗散标签同步改为 \(D_\theta\succ0\)，与当前参数化一致；正文和证明仍在需要最弱条件的位置保留 \(D_\theta\succeq0\)。

## 4. 修改和再生成范围

修改的主要图源为：

- `figures/make_method_overview_hero.py`；
- `figures/make_mechanical_core_power_structure.py`；
- `figures/make_fossen_role_mapping.py` 及 OE 变体；
- `figures/_section8_style.py`；
- `figures/make_section8_ablation_ladder.py` 及 OE 变体；
- `figures/make_section8_two_level_evidence.py`。

随后在非交互式 Matplotlib `Agg` 后端重新生成 11 组 Python 图件。受共享显示名影响的第 8 节图也一并重新导出，保证 SVG、PDF 和 PNG 三种产物来自同一图源。正式章节 PDF 由更新后的 TeX 和图件重新编译。

## 5. 验证记录

- Python 语法：`AUVHamNODE.py` 和全部受影响制图脚本通过 `python -m py_compile`；
- 实现性质：耦合和对角耗散分支的最小特征值烟雾检查均严格为正；
- 标签残留：对活动 Python 图源和重新生成的 SVG 搜索旧标签，结果为零；
- 独立图件：检查方法图和结果图联系表，并单独放大机械子系统功率图；未见裁切、重叠或不可读标签；
- LaTeX：在 `mytorch1` 环境运行 `latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error auvhamnode_thesis_chapter_zh.tex`，退出状态为 0；
- 最终 PDF：A4、58 页、1,175,022 字节；文本抽取未检出旧英文标签，新标签可在对应页面检出；
- PDF 视觉检查：将全部 58 页渲染为 PNG 检查，并放大核查第 18、21、27、44、45、47、49 页；未发现裁切、重叠、乱码、黑块、图表越界、页码异常或明显分页问题；
- 编译日志：无 undefined reference、undefined citation、Overfull、multiply defined 或致命错误；仅保留参考文献中既有的一处 `Underfull \hbox (badness 1735)`；
- 差异质量：排除 Matplotlib 自动生成 SVG 中的固定行尾空格后，`git diff --check` 通过。未手工编辑生成 SVG。

## 6. 当前结论

这两类事项均已闭环：

1. 耗散术语已有唯一、可复核的使用规则，不再依赖作者凭措辞偏好选择；
2. 图内术语已在生成脚本、独立导出图件、正式 TeX、整章 PDF 和审计记录五个层级保持一致。

本次没有产生新的实验结论，也没有改变任何结果数值。
