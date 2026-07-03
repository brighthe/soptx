# 帧 10：MMC 显式几何高精度离散前向验证数值结果 (T1-V1)

本文档记录了基于 `soptx_heliang` 代码库中 `mmc_cut_mesh_prototype.py` 脚本单次运行（2026-07-02）所产生的基准测试数据与结果图表。这些数据可作为 `dut-postdoc` 仓库下答辩 PPT (如 `template-8min.tex` 的 Frame 10) 的上游事实数据源。

本文的展示框架与答辩口径基于 dut-postdoc 帧级主入口
`research/postdoc-plan/defense-sprint/direction-2-mmc-mmv/frame10_mmc_pipeline_guide.md`。

## 1. 测试用例参数设定
- **背景网格**: $40 \times 20$ 矩形网格，物理域 $[0, 2] \times [0, 1]$，共 800 个单元。
- **组件形态 (MMC Component)**:
  - 形状：倾斜超椭圆 ($m=4$)
  - 中心坐标：$(xc, yc) = (1.0, 0.5)$
  - 几何尺寸：长 $L=1.2$，宽 $W=0.4$
  - 倾角：$\theta = 30^\circ$

## 2. 核心离散分类数据
脚本运用基于 TDF 隐函数的算法，对网格进行了精准分类：
- **Solid (全实体) 单元**: 140 个
- **Void (全空洞) 单元**: 580 个
- **Cut (被切割) 单元**: 80 个

## 3. 高阶积分点数据与计算量压缩
如果采用传统全域 Ersatz 密度法加 $2 \times 2$ 高斯积分，全域需消耗 3200 个积分点。本管线运用精确几何映射后：
- **Solid 侧积分点**: $140 \times 4 = 560$ 个
- **Cut 侧子域映射点**: 468 个 (仅在重构出的实体多边形内三角化布点)
- **有效高阶积分点总计**: **1028** 个

**结论**：在实现完美锐利边界的同时，需要组装计算的积分点数量被大幅压缩。生成的高斯点坐标与权重 `(x, y, weight)` 格式，已具备直连 Krylov / Matrix-Free 算子的能力。

## 4. 产出图表

![MMC 积分点与局部放大约束](mmc_integration_result.png)

- **图表说明**: 上图展示了组件轮廓与网格关系，带有右下角局部放大镜。局部放大区域清晰呈现了**积分点严格截止于实体边界，孔洞侧零渗入**的无伪密度特征。

## 5. 跨库同步材料 (LaTeX 表格)
在 `dut-postdoc` 中撰写 PPT 时，可以直接使用以下精确对齐的表格数据替换模板中的占位符：

```latex
\begin{tabular}{@{}p{0.20\linewidth}p{0.43\linewidth}p{0.34\linewidth}@{}}
    \toprule
    {\color{structure.fg}验证环节} & {\color{structure.fg}展示内容} & {\color{structure.fg}原型跑分结果} \\
    \midrule
    显式几何描述 & MMC/MMV 组件 TDF 与参数化边界 & 中心$(1.0, 0.5), \theta=30^\circ$ \\
    背景网格映射 & 固定网格上的实体/空洞/切割分类 & Solid 140, Void 580, Cut 80 \\
    切割单元处理 & cut cell 边界线段重构与实体侧识别 & 边界精准截断 (见下图) \\
    高阶积分布设 & 实体侧 Gauss 积分点分布 & 有效高斯点压缩至 1028 个 \\
    高阶/混合接口 & 为 Lagrange/胡张混合元提供入口 & 数据流直连无矩阵乘算子 \\
    \bottomrule
\end{tabular}
```

## 6. 上游文档

- 上游进度与决策记录见 `ai/common/progress-frame10_mmc.md`（含复现命令）；帧级主入口见 dut-postdoc `frame10_mmc_pipeline_guide.md`。
