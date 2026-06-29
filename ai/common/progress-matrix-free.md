---
title: "SOPTX Matrix-Free Progress"
tags:
  - ai-context
  - matrix-free
  - soptx
  - progress
status: "active"
date: 2026-06-29
---

# Matrix-Free 结构分析原型 Progress

本文件是 SOPTX Matrix-Free 工作线的续接入口。新窗口续接时，先读 `ai/common/status.md`，再读本文件；需要写代码时，再按本文列出的上下文和代码位置继续阅读。

## 当前结论

截至 2026-06-29，当前结论应表述为：

```text
SOPTX Python/NumPy 原型已完成 Matrix-Free 数学路径、接口闭环和正确性验证。
```

当前不应表述为：

```text
SOPTX 当前 Python 原型已经具备 GPU/MPI 并行性能优势。
```

## 当前进度

1. 阶段一：Matrix-Free MatVec 接口打通完成。
   - `MatrixFreeElasticityOperator.matvec(x)` 与 assembled `K @ x` 已通过一致性验证。
   - 全局算子路径为 `gather -> LinearElasticIntegrator.action -> scatter_add`。

2. 阶段二：状态方程非伴随后端接入完成。
   - `LagrangeFEMAnalyzer.solve_state()` 已支持 `operator_backend="assembled"` 与 `operator_backend="matrix_free"`。
   - matrix-free 后端通过 `MatrixFreeElasticityOperator` 和 `MatrixFreeCGSolver` 求解非伴随状态方程。
   - 上层优化器仍保持 `state = analyzer.solve_state(rho_val=rho_phys)` 的调用形式。
   - 当前不支持 `adjoint=True` 的 matrix-free 路径。

3. 阶段三：2D/3D standard contraction 完成。
   - `LinearElasticIntegrator.action()` 在 2D/3D `standard` 路径使用积分点 contraction。
   - 当前支持 `coef is None`、`coef.shape == (NC,)` 和 `coef.shape == (NC, NQ)`。
   - 测试中禁用 action 装配 fallback 后仍通过，说明当前覆盖路径不再显式形成局部 `Ke`。
   - 多分辨率 contraction 暂不作为当前优先事项。

4. 阶段四：NumPy benchmark 已建立。
   - benchmark 用于验证 Matrix-Free 数学路径、接口闭环、NumPy 后端正确性、基础时间和内存估算。
   - 默认生成 CSV 和格式化 XLSX。
   - 该 benchmark 不是 GPU/MPI 性能 benchmark。

## 已定关键决策

1. Matrix-Free 接入点放在 `LagrangeFEMAnalyzer` 的状态方程求解后端，而不是上层优化器直接管理。
2. `rho` 属于拓扑优化和材料插值层；`LinearElasticIntegrator` 只消费插值后的 `coef` 或 `relative_stiffness`。
3. assembled 与 matrix-free 的一致性测试是当前安全网；修改相关代码后优先运行 `soptx/tests/test_matrix_free_vs_assembled.py`。
4. 当前答辩口径应区分两层：
   - SOPTX 当前原型：证明 Matrix-Free 数学路径、接口闭环和正确性；
   - mfleo/高性能 kernel：作为 MPI/GPU 性能潜力和目标参照。
5. 短期不优先扩展多分辨率 contraction；下一阶段优先转向 GPU/多后端验证与 benchmark。

## 重要代码位置

```text
C:\workspace\soptx_heliang\soptx\analysis\integrators\linear_elastic_integrator.py
C:\workspace\soptx_heliang\soptx\analysis\matrix_free\elasticity_operator.py
C:\workspace\soptx_heliang\soptx\analysis\matrix_free\boundary.py
C:\workspace\soptx_heliang\soptx\analysis\matrix_free\krylov.py
C:\workspace\soptx_heliang\soptx\analysis\lagrange_fem_analyzer.py
C:\workspace\soptx_heliang\soptx\tests\test_matrix_free_vs_assembled.py
C:\workspace\soptx_heliang\soptx\benchmarks\benchmark_matrix_free_elasticity.py
```

## 详细上下文

需要继续理解当前实现说明时，阅读：

```text
C:\workspace\soptx_heliang\docs\matrix_free_architecture_notes.md
```

需要理解研究计划和数学原则时，再阅读：

```text
C:\workspace\dut-postdoc\research\soptx-matrix-free-integration-plan.md
C:\workspace\dut-postdoc\research\matrix_free_math_principles.md
```

注意：当前 Codex 会话可能无法写入 `dut-postdoc` 外部目录；涉及这些文档的更新需确认权限或改为在本仓库生成更新稿。

## 验证命令

在 `C:\workspace\soptx_heliang` 下运行：

```powershell
.\.venv\Scripts\python.exe -m pytest soptx/tests/test_matrix_free_vs_assembled.py -q
```

当前期望结果：

```text
4 passed
```

如遇 pytest cache 权限警告，可使用：

```powershell
.\.venv\Scripts\python.exe -m pytest soptx/tests/test_matrix_free_vs_assembled.py -q -p no:cacheprovider
```

## Benchmark 命令

在 `C:\workspace\soptx_heliang` 下运行：

```powershell
.\.venv\Scripts\python.exe -m soptx.benchmarks.benchmark_matrix_free_elasticity
```

默认输出：

```text
C:\workspace\soptx_heliang\outputs\matrix_free_elasticity_benchmark.csv
C:\workspace\soptx_heliang\outputs\matrix_free_elasticity_benchmark.xlsx
```

如果 CSV 正被 WPS/Excel 占用，可改名输出：

```powershell
.\.venv\Scripts\python.exe -m soptx.benchmarks.benchmark_matrix_free_elasticity --output outputs/matrix_free_elasticity_benchmark_latest.csv --xlsx-output outputs/matrix_free_elasticity_benchmark.xlsx
```

## 下一步计划

建议下一阶段按以下顺序推进：

1. 单机 GPU 后端可行性检查：确认当前 `action/operator/CG` 是否能在 PyTorch 或 JAX 后端运行。
2. GPU/多后端 benchmark：记录 CPU NumPy、PyTorch CPU/GPU 或 JAX CPU/GPU 下的 MatVec 时间、CG 时间、收敛信息和内存估算。
3. 答辩展示数据整理：形成“SOPTX 正确性闭环 + mfleo 高性能趋势参照”的组合表述。
4. 后续再考虑预条件器、`adjoint=True` matrix-free 路径、完整拓扑优化多步闭环和 MPI/多节点并行。

## 新窗口续接提示词

```text
按 C:\workspace\soptx_heliang\ai\common\status.md 续接「Matrix-Free 结构分析原型」。
先复述当前进度、已定关键决策、下一步与实现计划，我确认后再继续。
```
