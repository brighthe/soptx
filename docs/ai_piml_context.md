---
title: "AI PIML 多尺度预测接续上下文"
tags:
  - ai-context
  - piml
  - multiscale
  - soptx
status: "paused"
date: 2026-06-29
---

# AI PIML 多尺度预测接续上下文

这份文件用于后续新开的 AI 窗口快速掌握 SOPTX **PIML 多尺度预测**工作的全局分工。
当前仓库已建立 `ai/common/status.md` 作为 AI 工作线总入口；PIML 工作线暂保留本文作为续接入口，后续可按 Matrix-Free 的方式整理为独立 progress 文档。
PIML 与 Matrix-Free 对应同一拓扑优化框架的两条分析路径（PIML 多尺度预测 vs Matrix-Free 求解）。

## 1. 总体计划

位置：

```text
C:\workspace\dut-postdoc\research\soptx-piml-multiscale-integration-plan.md
```

作用：

```text
说明为什么要把 PIML 多尺度预测接入 SOPTX、总体目标、阶段划分、任务边界和答辩口径。
聚焦"单步前向原型"（宏观密度 -> 预测 -> 等效刚度 -> 组装宏观方程）。
```

它回答的是：

```text
我们要做什么？为什么这样做？原型阶段目标是什么？答辩怎么说才诚实？
```

> 上位长期计划（24 个月）见 `C:\workspace\dut-postdoc\research\piml-matrix-free-execution-plan.md`，
> 本原型对应其 T1.3.1（复现 2022 EMsFEM-PIML）的最小前向核心。

## 2. 数学原则

位置：

```text
C:\workspace\dut-postdoc\research\piml_multiscale_math_principles.md
```

作用：

```text
说明 PIML 多尺度预测的数学核心：

EMsFEM 两级网格、多尺度形函数  u^fine = N u^coarse、
粗尺度等效刚度  K^E = Σ_f N_f^T k^f N_f、
PIML 学习的映射  ρ_local -> N̂，问题无关性来自"形函数是离散 Green 函数"。
```

它回答的是：

```text
PIML 多尺度预测在数学上到底在算什么？
为什么学到的形函数是"问题无关"的？
宏观密度 -> 形函数 -> 等效刚度 这条管道每一步对应什么数学操作？
```

奠基论文笔记：
`C:\workspace\dut-postdoc\literature\topology-opt\Huang2022-problemindependentmachine.md`

## 3. SOPTX 当前实现备忘录

位置：

```text
C:\workspace\soptx_heliang\docs\piml_multiscale_architecture_notes.md
```

作用：

```text
结合当前 soptx_heliang 代码结构，记录 PIML 多尺度预测接入 SOPTX 的架构判断、
职责划分、rho/coef 语义边界、模块结构和阶段划分。
```

它回答的是：

```text
在当前 SOPTX 代码中应该改哪里？
为什么 PIML 多尺度应作为 LagrangeFEMAnalyzer 的一种状态方程求解后端？
为什么预测器只消费 rho_local，不解释原始 rho？
当前实现到了哪一步？
```

## 当前默认判断

后续 AI 窗口应默认采用以下上下文：

1. `dut-postdoc` 保存研究计划和数学原则；`soptx_heliang` 保存与代码强绑定的架构备忘录、接口实现和测试。
2. PIML 多尺度接入点在 `LagrangeFEMAnalyzer` 的状态方程求解后端，与 `assembled` / `matrix_free` 并列（`operator_backend="piml_multiscale"`）。
3. `rho` 属于拓扑优化与材料插值层；`LinearElasticIntegrator` 只消费插值后的 `coef`；PIML 预测器只消费粗单元局部密度 `rho_local`。
4. 第一阶段（答辩前原型，**路线①·子结构静力缩聚**）：构造形式用 Schur 补 N^j=[-(K_ii)^-1 K_ib; I]、K_s^j=(N^j)^T K^j N^j（数学精确、无边界条件假设；正是全局 Matrix-Free 消费的逐子结构算子）。目标：**单步前向管道连通** + **K_s^j 对全尺度 Schur 补机器精度一致** + **极小预测器取得对照团队子结构 PIML 量级（~1e-3）的预测误差**；仍不跑优化循环、不替换灵敏度。Huang 2022 全 EMsFEM 角节点复现保留为长期计划阶段一，作奠基引用。
5. mock（查表/解析）先打通接口；`TrainedPredictor`（极小 MLP，随机密度离线训练，损失=形函数MSE+刚度MSE）取预测误差；`ExactPredictor`（精确局部求解）作真值/标注。三者接口互换；极小网络不追求生产级精度/泛化。
6. 粗尺度方程既可显式组装求解，也可把 K̂^E 接入 Matrix-Free（与 `ai/common/progress-matrix-free.md` 中记录的 Matrix-Free 路径协同）——这是科研计划方向一"主线一 PIML + 主线二 Matrix-Free"的结合点。
7. 截至 2026-06-29，soptx 仓库中尚无 PIML/多尺度代码，本体系为**接入前的计划与架构判断**。

## 当前重要代码位置

> 待实现，路径为建议；推进后回填实际文件（与架构备忘录 §9 保持同步）。

```text
C:\workspace\soptx_heliang\soptx\analysis\multiscale\coarse_fine_mesh.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\multiscale_shape.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\equivalent_stiffness.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\piml_predictor.py
C:\workspace\soptx_heliang\soptx\tests\test_equivalent_stiffness_vs_fullscale.py
C:\workspace\soptx_heliang\soptx\examples\piml_baseline_forward.py
```

## 当前验证命令

> 待实现。模块就位后，预期在 `C:\workspace\soptx_heliang` 下运行：

```powershell
.\.venv\Scripts\python.exe -m pytest soptx/tests/test_equivalent_stiffness_vs_fullscale.py soptx/tests/test_shape_function_partition_of_unity.py -q
```

建议开发分支：

```text
codex/piml-multiscale-prototype
```
