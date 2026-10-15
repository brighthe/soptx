# 问题模型

`soptx.problems` 定义线弹性问题：区域、边界条件与物理载荷，不创建网格，也不持有材料或函数空间对象。公开类一律从 `soptx.problems` 导入；按有无解析解分两族，代码与文档目录同构：

| 族 | 代码 | 文档 |
| --- | --- | --- |
| 工程基准 | `soptx.problems.benchmarks` | [`benchmarks/`](benchmarks/README.md) |
| 制造解 | `soptx.problems.manufactured` | [`manufactured/`](manufactured/README.md) |

## 两族的区别

| | 制造解 | 工程基准 |
|---|---|---|
| 精确解 | 有，暴露 `disp_solution` / `stress_solution` | 无 |
| 体力 | 由精确解反推（$b = -\nabla\cdot\sigma(u_\mathrm{exact})$） | 通常为零（忽略自重） |
| 验证判据 | L2 收敛阶 + 真相对残差 | 真相对残差 + 载荷等效性 |
| 用途 | 验证离散格式的正确性 | 验证载荷路径装配的正确性；演示工程问题的求解流程 |
| `boundary_type` | `dirichlet` 或 `mixed` | `mixed` |

## 模型索引

工程基准按载荷类型分 A 到 D 四组，组别决定哪些分析器能消费，见 [`benchmarks/README.md`](benchmarks/README.md)；类名以载荷类型结尾（`Point`、`Line`、`Traction`、`Patch`）。

| 类 | 组 | 载荷 | Lagrange | Hu--Zhang | 文档 |
| --- | --- | --- | --- | --- | --- |
| `HalfMBBBeamPoint2d` | A | 左上角集中力 | 是 | 否 | [half-mbb-beam-point-2d](benchmarks/half-mbb-beam-point-2d.md) |
| `FullMBBBeamPoint2d` | A | 顶边中点集中力 | 是 | 否 | [full-mbb-beam-point-2d](benchmarks/full-mbb-beam-point-2d.md) |
| `CantileverCornerPoint2d` | A | 右下角集中力 | 是 | 否 | [cantilever-corner-point-2d](benchmarks/cantilever-corner-point-2d.md) |
| `HalfMBBBeamLine3d` | B | 对称面顶棱线载荷 | 是 | 否 | [half-mbb-beam-line-3d](benchmarks/half-mbb-beam-line-3d.md) |
| `FullMBBBeamLine3d` | B | 顶面中线线载荷 | 是 | 否 | [full-mbb-beam-line-3d](benchmarks/full-mbb-beam-line-3d.md) |
| `CantileverBottomEdgeLine3d` | B | 右端底棱线载荷 | 是 | 否 | [cantilever-bottom-edge-line-3d](benchmarks/cantilever-bottom-edge-line-3d.md) |
| `BearingTraction2d` | C | 顶边满布牵引 | 是 | 是 | [bearing-traction-2d](benchmarks/bearing-traction-2d.md) |
| `BridgeTraction2d` | C | 顶边满布牵引 | 是 | 否（支座为两个角点，混合形式无法弱施加点位移） | [bridge-traction-2d](benchmarks/bridge-traction-2d.md) |
| `CantileverMiddlePatch2d` | D | 右端中部区段牵引 | 是 | 是 | [cantilever-middle-patch-2d](benchmarks/cantilever-middle-patch-2d.md) |
| `FixedFixedBeamPatch2d` | D | 底边中部区段牵引 | 是 | 是 | [fixed-fixed-beam-patch-2d](benchmarks/fixed-fixed-beam-patch-2d.md) |
| `HalfFixedFixedBeamPatch2d` | D | 对称面底端区段牵引（左半域） | 是 | 是 | [half-fixed-fixed-beam-patch-2d](benchmarks/half-fixed-fixed-beam-patch-2d.md) |
| `PortalFramePatch2d` | D | 顶边中部区段牵引（七边形域） | 是 | 是 | [portal-frame-patch-2d](benchmarks/portal-frame-patch-2d.md) |

制造解按精确解分文件，2D 与 3D 同解放一处，见 [`manufactured/README.md`](manufactured/README.md)。

| 类 | 维数 | 边界 | 文档 |
| --- | --- | --- | --- |
| `ExponentialSineManufacturedElasticity2D` | 2D | 全 Dirichlet | [exponential-sine-manufactured-elasticity-2d](manufactured/exponential-sine-manufactured-elasticity-2d.md) |
| `MixedBoundaryExponentialSineElasticity2D` | 2D | 混合 | [mixed-boundary-exponential-sine-elasticity-2d](manufactured/mixed-boundary-exponential-sine-elasticity-2d.md) |
| `SinusoidalElasticity2D`（别名 `SinusoidalPlaneStrainElasticity2D`） | 2D | 全 Dirichlet | [sinusoidal-elasticity-2d](manufactured/sinusoidal-elasticity-2d.md) |
| `MixedBoundarySinusoidalElasticity2D` | 2D | 混合 | [mixed-boundary-sinusoidal-elasticity-2d](manufactured/mixed-boundary-sinusoidal-elasticity-2d.md) |
| `HarmonicPoly2D`（别名 `HarmonicPolynomialElasticity2D`） | 2D | 全 Dirichlet，非齐次 | [harmonic-poly-2d](manufactured/harmonic-poly-2d.md) |
| `DivergenceFreePolynomialElasticity3D` | 3D | 全 Dirichlet | [divergence-free-polynomial-elasticity-3d](manufactured/divergence-free-polynomial-elasticity-3d.md) |
| `HarmonicPoly3D`（别名 `HarmonicPolynomialElasticity3D`） | 3D | 全 Dirichlet，非齐次 | [harmonic-poly-3d](manufactured/harmonic-poly-3d.md) |

## 文档约定

每份模型文档只定义模型：一句定义与实现位置、问题描述（示意图、区域与材料、边界与载荷）、使用示例；制造解另给精确解与体力的公式。接口、组规则、离散检查与算例参数分别由类 docstring、本目录的两份 README 与各实验目录承担，不在模型文档中重复。示意图在 [`assets/images/`](assets/images/)，由 [`assets/generate_problem_diagrams.py`](assets/generate_problem_diagrams.py) 从 `assets/tikz/` 的 TikZ 源编译，文件名为类名的 kebab-case。
