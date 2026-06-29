---
title: "SOPTX PIML 多尺度预测架构理解备忘录"
tags:
  - piml
  - multiscale
  - soptx
  - architecture
  - topology-optimization
status: "draft"
date: 2026-06-29
---

# SOPTX PIML 多尺度预测架构理解备忘录

本文档记录当前对 SOPTX 拓扑优化求解流程、线弹性状态方程路径，以及 **PIML 多尺度预测**
接入方式的共同判断。它的目的不是替代完整设计文档，而是在后续讨论和实现时提供稳定上下文，
避免每次都重新阅读整个代码和计划文件。它与 `docs/matrix_free_architecture_notes.md`
（Matrix-Free 接入备忘录）并列，对应同一拓扑优化框架的另一条分析路径。

本文档中的 PIML 判断大量参考了 `dut-postdoc` 仓库中的相关材料，特别是：

- `C:\workspace\dut-postdoc\research\soptx-piml-multiscale-integration-plan.md`
- `C:\workspace\dut-postdoc\research\piml_multiscale_math_principles.md`
- `C:\workspace\dut-postdoc\research\piml-matrix-free-execution-plan.md`

这些外部材料提供了 PIML 多尺度预测原型的任务背景、数学原则、阶段划分。本文档在此基础上，
结合当前 `soptx_heliang` 仓库的实际代码结构，整理出更贴近 SOPTX 当前实现的架构判断。

> 注意：截至 2026-06-29，soptx 仓库中**尚无 PIML / 多尺度代码**。本文档记录的是
> **接入前的架构判断与默认假设**，实现推进后应同步回填"已完成验证"。

## 1. 当前结论

PIML 多尺度预测应被设计为 **有限元分析器中状态方程求解的一种"分析后端"**，
与 assembled、matrix-free 并列，而**不是**放进优化器，也**不是**让
`LinearElasticIntegrator` 直接理解拓扑优化密度 `rho`。

推荐的职责链路（与 Matrix-Free 备忘录一致，仅多尺度部分不同）：

```text
拓扑优化层
  design variable / rho
      ↓
材料插值层
  rho -> E(rho) -> relative_stiffness / coef
      ↓
有限元分析层
  assembled / matrix-free / piml-multiscale 后端
      ↓
（PIML 多尺度路径）
  粗单元局部密度 -> 多尺度形函数 N̂ -> 等效刚度 K̂^E
      ↓
粗尺度方程
  K^coarse U^coarse = F^coarse
      ↓
细尺度恢复
  u^fine = N̂ u^coarse -> 柔顺度 / 灵敏度
```

核心边界（与 Matrix-Free 完全相同）：

```text
rho                 拓扑优化中的设计/物理密度
E_rho               材料插值得到的绝对杨氏模量
relative_stiffness  E_rho / E0
coef                LinearElasticIntegrator 消费的单元刚度系数
```

`LinearElasticIntegrator` 只消费已处理好的 `coef`，PIML 预测器消费粗单元内的局部密度
（或局部 `coef`），不在内部解释原始 `rho`、SIMP 惩罚、过滤或投影。

## 2. SOPTX 当前拓扑优化流程（复用 Matrix-Free 备忘录的判断）

优化器主循环只稳定调用：

```python
state = analyzer.solve_state(rho_val=rho_phys)
```

优化器不需要知道状态方程是 assembled、matrix-free 还是 piml-multiscale 求解。
因此 PIML 多尺度的切换点应位于 `LagrangeFEMAnalyzer` 内部的求解后端选择，而不是优化器层：

```python
analyzer = LagrangeFEMAnalyzer(..., operator_backend="assembled")
analyzer = LagrangeFEMAnalyzer(..., operator_backend="matrix_free")
analyzer = LagrangeFEMAnalyzer(..., operator_backend="piml_multiscale")
```

## 3. PIML 多尺度状态方程路径

```text
rho_val
  -> interpolation_scheme.interpolate_material(...)
  -> coef（逐细单元相对刚度）
  -> 逐粗单元取局部 coef/密度 rho_local
  -> predictor.predict(rho_local) -> N̂
  -> K̂^E = Σ_f N̂_f^T k^f(coef) N̂_f
  -> Assemble -> K^coarse
  -> apply_bc -> solve K^coarse U^coarse = F^coarse
  -> u^fine = N̂ U^coarse
```

与 `matrix_free_architecture_notes.md` §3 的 assembled 路径对照：PIML 多尺度在
"组装"之前多了"粗/细两级映射 + 形函数预测 + 等效刚度缩聚"三步，但仍落到一个全局
线性系统求解，并保持 `rho -> coef` 语义不变。

## 4. 灵敏度计算路径（原型阶段不替换）

与 Matrix-Free 一致的稳妥路线：

```text
第一阶段：只搭建 PIML 多尺度的"前向"状态分析
保留现有灵敏度局部刚度导数路径
确认前向结果与全尺度/精确缩聚一致后，再考虑多尺度灵敏度（含 ∂N̂/∂ρ）实现
```

> 原型期（答辩前）甚至不要求灵敏度与优化循环，只要求单步前向管道连通（见集成计划 T5/V3）。

## 5. 推荐的 PIML 多尺度接入结构

```text
MultiscaleMeshMapping
  粗/细两级网格、coarse↔fine 自由度映射

MultiscaleShapeFunction
  精确求解局部问题得到 N（基线/真值），并定义预测接口

MultiscalePredictor
  predict(rho_local) -> N̂（或 K̂^E）
    ExactPredictor   精确局部求解（基线/标注）
    MockPredictor    查表 / 解析映射 / 极简 MLP（原型期）
    TrainedPredictor 训练后的结构保持网络（后续）

EquivalentStiffnessAssembler
  K̂^E = Σ N̂_f^T k^f N̂_f，组装 K^coarse

（可选）与 matrix-free 协同
  把 K̂^E 作为局部算子直接喂给 Krylov，避免显式组装 K^coarse
```

预测器接口应与求解后端解耦：分析器只依赖 `MultiscalePredictor.predict`，
不感知其内部是精确求解、查表还是网络推断。

## 6. 阶段划分

### 阶段一：前向管道连通 + 极小预测器（答辩前原型，路线①·子结构缩聚）

构造形式用**子结构静力缩聚（Schur 补）**：N^j = [-(K_ii)^-1 K_ib; I]，K_s^j = (N^j)^T K^j N^j。
数学精确、无边界条件假设，且 K_s^j 正是全局 Matrix-Free 作用消费的逐子结构算子（与 matrix_free 路径咬合）。
Huang 2022 全 EMsFEM 角节点形式保留为长期计划阶段一，作奠基引用。

```text
子结构静力缩聚（精确 K_s^j）+ 接口缩聚组装
mock 预测器与 ExactPredictor 互换（先打通接口）
极小 TrainedPredictor：随机密度离线训练（损失=算子MSE+缩聚刚度MSE），仅取对照论文的预测误差
单步前向：ρ -> predict -> K̂_s^j -> 接口组装（可选求解）
缩聚精确性测试：K_s^j vs 全尺度 Schur 补（机器精度）
预测误差对照团队子结构 PIML 量级（~1e-3）
出图 piml_baseline.pdf + piml_pred_error.pdf
```

### 阶段二：接入状态方程与优化循环

```text
在 LagrangeFEMAnalyzer.solve_state() 内支持 operator_backend="piml_multiscale"
细尺度位移恢复 + 柔顺度
与 assembled 路径结果趋势一致
```

### 阶段三：真实 PIML 网络与高性能协同

```text
用结构保持参数化（对称正定/能量一致）训练 TrainedPredictor 替换 mock
误差传播分析（局部算子误差 -> 位移 -> 灵敏度 -> 拓扑偏差）
把 K̂^E 接入全局 Matrix-Free（见 matrix_free_architecture_notes.md）
GPU/多后端批量推理
```

## 7. 命名建议

```text
rho                 原始/物理密度
rho_local           粗单元内 m 个细单元密度（预测器输入）
coef                积分器刚度缩放系数
N / N_hat           精确 / 预测 多尺度形函数
K_E / K_E_hat       精确 / 预测 粗单元等效刚度
K_coarse            组装后的全局粗刚度
```

如预测器输入已是 `coef`/`relative_stiffness`，字段名不宜继续叫 `rho`，避免误以为
预测器内部负责 SIMP 或材料插值。

## 8. 后续讨论时的默认假设

1. PIML 多尺度接入点在 `LagrangeFEMAnalyzer` 的状态方程求解后端，与 assembled / matrix_free 并列。
2. `LinearElasticIntegrator` 只消费 `coef`，不解释 `rho`；预测器只消费 `rho_local`/局部 `coef`。
3. 第一阶段（B 路线）目标："单步前向管道连通 + 等效刚度对全尺度缩聚一致 + 极小预测器取得对照 Huang 2022 量级（~1e-3）的预测误差"；仍不跑优化循环、不做完整误差传播。
4. 灵敏度计算第一阶段不替换。
5. mock / TrainedPredictor 与 ExactPredictor 接口互换；真值/标注由精确局部求解提供。TrainedPredictor 为极小网络，不追求生产级精度/泛化。
6. 粗尺度方程既可显式组装求解，也可把 K̂^E 接入 matrix-free（与 Matrix-Free 路径协同）。

## 9. 当前重要代码位置

> 待实现，路径为建议；推进后回填实际文件。

```text
C:\workspace\soptx_heliang\soptx\analysis\multiscale\coarse_fine_mesh.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\multiscale_shape.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\equivalent_stiffness.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\piml_predictor.py
C:\workspace\soptx_heliang\soptx\tests\test_equivalent_stiffness_vs_fullscale.py
C:\workspace\soptx_heliang\soptx\examples\piml_baseline_forward.py
```

## 10. 当前验证命令

> 待实现。模块就位后，预期在 `C:\workspace\soptx_heliang` 下运行：

```powershell
.\.venv\Scripts\python.exe -m pytest soptx/tests/test_equivalent_stiffness_vs_fullscale.py soptx/tests/test_shape_function_partition_of_unity.py -q
```
