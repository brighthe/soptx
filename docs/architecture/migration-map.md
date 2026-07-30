# SOPTX 1.1 迁移表

`1.1.x` 兼容层使用 `DeprecationWarning`；默认不会在普通 Python 运行中显示，可用
`python -Wd` 检查。每次导入一个旧根命名空间只警告一次。有 maintained
替代实现的旧模块仅做对象转发，`2.0.0` 将删除这些旧路径。

| 旧路径 | 新路径 |
| --- | --- |
| `soptx.interpolation.linear_elastic_material.IsotropicLinearElasticMaterial` | `soptx.materials.IsotropicLinearElasticMaterial` |
| `soptx.analysis.integrators.LinearElasticIntegrator` | `soptx.fem.integrators.LinearElasticIntegrator` |
| `soptx.analysis.LagrangeFEMAnalyzer` | `soptx.fem.solvers.LagrangeFEMAnalyzer` |
| `soptx.functionspace.HuZhangFESpace` | `soptx.fem.spaces.HuZhangFESpace` |
| `soptx.interpolation.MaterialInterpolationScheme` | `soptx.topology.interpolation.MaterialInterpolationScheme` |
| `soptx.regularization.*` | `soptx.topology.filters.*` |
| `soptx.optimization.*` | 对应的 `soptx.topology.{objectives,constraints,optimizers,postprocess}` |
| `soptx.utils.{BaseLogged,timer}` | `soptx.core.{BaseLogged,timer}` |
| `TriSolHomoDirHuZhang2d` | `soptx.problems.ExponentialSineManufacturedElasticity2D` |
| `BoxTriLagrange2dData` | `soptx.problems.SinusoidalPlaneStrainElasticity2D` |
| `PolySolPureDirLagrange3d` | `soptx.problems.DivergenceFreePolynomialElasticity3D` |

旧 Problem 类仍保留其网格工厂，作为过渡兼容行为；新语义类没有 `init_mesh`，调用方
必须显式创建并校验网格。

迁移表之外的 `old`、`backup`、内部测试和论文驱动不属于 `1.1.x` wheel
兼容承诺；完整源码保存在 `experiments/legacy` 和 Git 历史中。
