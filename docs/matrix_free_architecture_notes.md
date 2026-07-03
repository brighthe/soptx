---
title: "SOPTX Matrix-Free 架构理解备忘录"
tags:
  - matrix-free
  - soptx
  - architecture
  - topology-optimization
status: "active"
date: 2026-06-29
---

# SOPTX Matrix-Free 架构理解备忘录

本文档记录当前对 SOPTX 拓扑优化求解流程、线弹性方程装配路径，以及 Matrix-Free 接入方式的共同判断。它的目的不是替代完整设计文档，而是在后续讨论和实现时提供一个稳定的上下文，避免每次都重新阅读整个代码和计划文件。

本文档中的 Matrix-Free 判断大量参考了 `dut-postdoc` 仓库中关于有光环 Matrix-Free 工作的相关内容，特别是：

- `C:\workspace\dut-postdoc\research\postdoc-plan\defense-sprint\direction-1-piml-matrix-free\frame8_matrix_free_pipeline_guide.md`
  （帧 8 单一主入口；原 `soptx-matrix-free-integration-plan.md` 与 `matrix_free_math_principles.md`
  已于 2026-07-02 被其取代删除）

这些外部材料提供了 Matrix-Free 接入 SOPTX 的总体任务背景、数学原则、阶段划分和与 mfleo 的关系。本文档在此基础上，结合当前 `soptx_heliang` 仓库的实际代码结构，整理出更贴近 SOPTX 当前实现的架构判断。

需要注意的是，本文档不是对 `dut-postdoc` 中计划文件的完整复刻，而是面向当前代码实现的工作备忘录。后续如果 `dut-postdoc` 中的 Matrix-Free 方案发生调整，应同步检查本文档中的默认假设是否仍然成立。

如果后续新开 AI 窗口继续本任务，应先阅读 `C:\workspace\soptx_heliang\ai\common\status.md`，在工作线表中定位「Matrix-Free 结构分析原型」，再阅读 `C:\workspace\soptx_heliang\ai\common\progress-frame8_matrix_free.md`。

## 1. 当前结论

Matrix-Free 更适合被设计为 **有限元分析器中的状态方程求解后端**，而不是放进优化器，也不是让 `LinearElasticIntegrator` 直接理解拓扑优化密度 `rho`。

推荐的职责链路是：

```text
拓扑优化层
  design variable / rho
      ↓
材料插值层
  rho -> E(rho) -> relative_stiffness / coef
      ↓
有限元分析层
  assembled backend 或 matrix-free backend
      ↓
线弹性算子
  y = K(rho) x
      ↓
Krylov solver
  solve K u = f
```

核心边界是：

```text
rho                 拓扑优化中的设计/物理密度
E_rho               由材料插值得到的绝对杨氏模量
relative_stiffness  E_rho / E0
coef                LinearElasticIntegrator 消费的单元刚度系数
```

因此，`LinearElasticIntegrator` 应该只消费已经处理好的 `coef`，不应该在内部解释原始 `rho`、SIMP 惩罚、过滤或投影。

## 2. SOPTX 当前拓扑优化流程

典型入口位于优化示例中，例如 `soptx/optimization/test_phd_section3.py`。当前主流程可以概括为：

```text
创建 PDE/model
创建 mesh
创建材料模型 IsotropicLinearElasticMaterial
创建材料插值 MaterialInterpolationScheme
初始化设计变量 d 和密度分布 rho
创建 Filter
创建 LagrangeFEMAnalyzer
创建 ComplianceObjective / VolumeConstraint
创建 OCOptimizer 或 MMAOptimizer
执行 optimizer.optimize(design_variable=d, density_distribution=rho)
```

优化器主循环的核心结构是：

```text
rho_phys = filter.get_initial_density(rho)

for iter:
    state = analyzer.solve_state(rho_val=rho_phys)
    obj_val = objective.fun(density=rho_phys, state=state)
    obj_grad_rho = objective.jac(density=rho_phys, state=state)
    obj_grad_dv = filter.filter_objective_sensitivities(...)
    con_val = constraint.fun(rho_phys)
    con_grad_rho = constraint.jac(rho_phys)
    con_grad_dv = filter.filter_constraint_sensitivities(...)
    dv_new = OC/MMA update(...)
    rho_phys = filter.filter_design_variable(dv_new, rho_phys)
```

这说明优化器并不需要知道状态方程是 assembled 求解还是 matrix-free 求解。它只需要稳定调用：

```python
state = analyzer.solve_state(rho_val=rho_phys)
```

因此 Matrix-Free 的切换点应该位于 `LagrangeFEMAnalyzer` 内部，而不是优化器层。

## 3. 线弹性状态方程路径

当前 `LagrangeFEMAnalyzer.solve_state()` 的基本路径是：

```text
K0 = assemble_stiff_matrix(rho_val)
F0 = assemble_body_force_vector()
K, F = apply_bc(K0, F0)
u = solve K u = F
```

刚度矩阵装配时，密度并不是直接进入 `LinearElasticIntegrator` 被解释，而是先经过材料插值：

```text
rho_val
  -> interpolation_scheme.interpolate_material(...)
  -> E_rho
  -> relative_stiffness = E_rho / E0
  -> integrator.coef = relative_stiffness
  -> BilinearForm(...).assembly(...)
```

这条语义路径非常重要。它意味着 `coef` 已经是线弹性积分器所需的材料缩放系数；积分器不需要知道它来自 SIMP、RAMP、过滤密度还是其他材料插值模型。

## 4. 灵敏度计算路径

当前柔顺度目标通常使用：

```text
c = u^T F
```

灵敏度路径仍然依赖局部刚度导数：

```text
uhe = uh[cell2dof]
diff_KE = analyzer.compute_stiffness_matrix_derivative(rho_val=density)
dc = -uhe^T diff_KE uhe
```

`compute_stiffness_matrix_derivative()` 中的关键语义是：

```text
dE_rho = interpolation_scheme.interpolate_material_derivative(...)
diff_coef_element = dE_rho / E0
ke0 = compute_solid_stiffness_matrix()
diff_ke = diff_coef_element * ke0
```

因此第一阶段 Matrix-Free 不必一次性替换灵敏度计算。更稳妥的路线是：

```text
先替换状态方程求解路径
保留现有灵敏度局部矩阵导数路径
确认优化结果和 assembled 路径一致后，再考虑更深层的 matrix-free 灵敏度实现
```

## 5. 当前框架评价

SOPTX 现有分层总体是合理的：

```text
model / pde          提供几何、边界、载荷和问题定义
interpolation        负责 rho 到材料参数的映射
regularization       负责过滤和投影
analysis             负责有限元空间、刚度、边界和状态方程
optimization         负责 OC/MMA 迭代更新
objective/constraint 负责目标函数和约束函数
```

其中 `LagrangeFEMAnalyzer` 是 Matrix-Free 接入的自然位置，因为它已经负责：

- 有限元空间；
- 刚度装配；
- 载荷装配；
- 边界条件；
- 状态方程求解；
- 刚度导数和应力等分析量。

不过，`LagrangeFEMAnalyzer` 当前职责偏重。Matrix-Free 接入时应尽量避免进一步把它写成一个巨大的过程类。更合适的方向是让它选择和调用后端：

```python
analyzer = LagrangeFEMAnalyzer(..., operator_backend="assembled")
analyzer = LagrangeFEMAnalyzer(..., operator_backend="matrix_free")
```

或者使用等价的 solver/backend 配置对象。

## 6. 推荐的 Matrix-Free 接入结构

推荐把 Matrix-Free 拆成三个层次：

```text
LinearElasticIntegrator.action(...)
  负责局部单元算子作用 xe -> ye

MatrixFreeElasticityOperator
  负责全局 gather -> local action -> scatter_add

MatrixFreeCGSolver / Krylov solver
  负责通过 operator.matvec() 解状态方程
```

其中 `LinearElasticIntegrator.action(...)` 的职责应该是：

```text
输入局部位移 xe
读取或接收 coef
计算局部弹性算子作用 ye
返回 ye
```

第一阶段可以允许 `action()` 内部复用已有局部单元矩阵：

```text
Ke = assembly(space)
ye = Ke @ xe
```

这不是最终的高性能 Matrix-Free kernel，但它可以先稳定接口，并用于验证：

```text
y_ref = K @ x
y_mf = matrix_free_operator.matvec(x)
rel_error = ||y_ref - y_mf|| / ||y_ref||
```

第二阶段再把 `action()` 内部替换成真正的积分点 contraction：

```text
xe
  -> grad u
  -> strain
  -> stress = D : strain
  -> B^T stress
  -> ye
```

## 7. 与 mfleo 的关系

mfleo 的设计思想可以作为参考，但不应机械照搬类名或结构。

mfleo 的核心定位是 Matrix-Free Linear Elasticity Operator 中间件。它不是完整拓扑优化框架，而是把高性能弹性算子接入 MFEM 的 partial assembly 生命周期。

mfleo 的关键模式是：

```text
MFEM ParBilinearForm
  -> MFLEOHexPAIntegrator / MFLEOTetPAIntegrator
  -> AssemblePA() 预计算局部数据
  -> AddMultPA() 执行 y = A x
  -> MFEM Operator::Mult
  -> CG/PCG solve
```

MFEM 负责：

- 全局自由度和 true-dof/local-dof 语义；
- element restriction；
- MPI/网格划分；
- 边界条件；
- 线性系统封装；
- CG/PCG 上层求解。

mfleo kernel 负责：

- 预计算单元、基函数、几何和材料参数数据；
- 在局部执行 `x -> y`；
- 针对 CPU/GPU 做高性能实现。

mfleo 中底层 kernel 消费的是已经准备好的材料参数，例如 `lambda` / `mu` 或 attribute-based material arrays，而不是拓扑优化的原始 `rho`。这与 SOPTX 中 `rho -> E(rho) -> coef` 的职责分离是一致的。

## 8. 对 SOPTX 的具体判断

更现代、也更科学的 SOPTX 接入方式是：

```text
优化器不感知 Matrix-Free
材料插值层仍然负责 rho -> coef
分析器选择 assembled 或 matrix-free 后端
Matrix-Free operator 只实现 y = K x
Krylov solver 通过 matvec 求解状态方程
灵敏度路径第一阶段保持原有实现
```

不推荐的方式包括：

- 在优化器里分支处理 Matrix-Free；
- 让 `LinearElasticIntegrator` 直接处理原始 `rho`；
- 在材料插值层中混入求解器逻辑；
- 第一阶段就同时重写状态求解、灵敏度、过滤和优化器。

## 9. 当前实现阶段的建议

当前阶段建议采用三步走：

### 阶段一：接口打通

目标是验证 assembled operator 与 matrix-free operator 对同一向量的作用一致。

```text
integrator.action() 可以暂时使用 Ke @ xe
MatrixFreeElasticityOperator 实现 gather/action/scatter
加入 y_ref = K @ x 与 y_mf = op.matvec(x) 的一致性测试
```

这一阶段的价值是固定接口、验证自由度顺序、边界处理和 `coef` 语义。

### 阶段二：接入状态方程

目标是在 `LagrangeFEMAnalyzer.solve_state()` 内部支持后端选择：

```text
assembled:
  assemble K
  apply_bc(K, F)
  direct/iterative solve

matrix_free:
  build MatrixFreeElasticityOperator
  apply matrix-free boundary handling
  use CG/PCG solve
```

优化器、目标函数和约束函数的外部调用保持不变。

当前阶段二主体已经完成：`LagrangeFEMAnalyzer` 支持 `operator_backend="assembled"` 和 `operator_backend="matrix_free"`，其中 matrix-free 分支通过 `MatrixFreeElasticityOperator` 和 `MatrixFreeCGSolver` 求解非伴随状态方程。上层仍然调用：

```python
state = analyzer.solve_state(rho_val=rho_phys)
```

当前 matrix-free 分支的关键边界是：

```text
rho_val
  -> interpolation_scheme.interpolate_material(...)
  -> E_rho
  -> relative_stiffness = E_rho / E0
  -> MatrixFreeElasticityOperator(..., rho=relative_stiffness)
  -> MatrixFreeCGSolver.solve(...)
```

也就是说，matrix-free 后端不直接解释原始 `rho`，也不为了获得 `coef` 而调用 `assemble_stiff_matrix()`。

阶段二包含两层验证。第一层是小型手工闭环：

```text
assembled:
  显式组装 K
  手动施加 homogeneous Dirichlet 边界
  用直接法求解 K_bc u = F_bc

matrix-free:
  使用 MatrixFreeElasticityOperator
  使用同一批 fixed dofs 和 RHS
  用 MatrixFreeCGSolver 求解

检查:
  ||u_assembled - u_matrix_free|| / ||u_assembled||
```

这个验证比单纯的 `K @ x` 对比更进一步，因为它覆盖了：

- Matrix-Free operator 在 Krylov 迭代中的使用；
- 固定自由度的输入/输出处理；
- RHS 在固定自由度上的处理；
- 求解得到的位移解是否与 assembled 路径一致。

当前对应测试位于 `soptx/tests/test_matrix_free_vs_assembled.py` 中的 `test_matrix_free_cg_solution_matches_assembled_solution()`。

第二层是正式 analyzer 接入验证：

```text
assembled:
  LagrangeFEMAnalyzer(..., operator_backend="assembled")
  solve_state(rho_val=rho)

matrix-free:
  LagrangeFEMAnalyzer(..., operator_backend="matrix_free")
  solve_state(rho_val=rho)

检查:
  ||u_assembled - u_matrix_free|| / ||u_assembled||
```

当前对应测试位于 `soptx/tests/test_matrix_free_vs_assembled.py` 中的 `test_lagrange_analyzer_matrix_free_state_matches_assembled_state()`。该测试还会替换 matrix-free analyzer 的 `assemble_stiff_matrix()` 为必报错函数，用于确认 matrix-free 状态方程分支没有装配全局刚度矩阵。

阶段二及后续仍有明确未覆盖项：

- `adjoint=True` 路径尚未接入 matrix-free；
- 尚未跑完整拓扑优化多步闭环；
- 尚未加入预条件器；
- `LinearElasticIntegrator.action()` 已在 2D/3D `standard` 路径使用积分点 contraction；多分辨率和更完整的性能优化尚未完成。

### 阶段三：真正 Matrix-Free kernel

目标是替换 `LinearElasticIntegrator.action()` 内部实现：

```text
不再显式形成 Ke
直接在积分点完成 grad/strain/stress/B^T contraction
逐步加入缓存、批量单元、预条件和性能优化
```

这一阶段才是性能收益的主要来源。

阶段三第一步已经完成：当前 `LinearElasticIntegrator.action()` 在 2D、`standard` 路径下通过积分点完成 `strain -> stress -> B^T stress` contraction，并支持 `coef is None`、单元 `coef.shape == (NC,)` 和积分点 `coef.shape == (NC, NQ)`。

阶段三第二步已经完成：同一 contraction 路径已扩展到 3D `standard` 线弹性，并新增 3D 四面体小网格 `K @ x` 与 matrix-free `matvec(x)` 一致性测试。现有 `test_matrix_free_vs_assembled.py` 的 4 个测试已在禁用 action 装配 fallback 的条件下通过。

### 阶段四：最小 Benchmark 与展示数据

当前已新增一个面向答辩/报告第一层结果的 NumPy benchmark：

```text
soptx/benchmarks/benchmark_matrix_free_elasticity.py
```

运行方式：

```powershell
.\.venv\Scripts\python.exe -m soptx.benchmarks.benchmark_matrix_free_elasticity
```

默认输出 CSV，并同时生成同名格式化 XLSX：

```text
outputs/matrix_free_elasticity_benchmark.csv
outputs/matrix_free_elasticity_benchmark.xlsx
```

该 benchmark 的定位是验证和记录：

```text
Matrix-Free 数学路径
接口闭环
NumPy 后端正确性
基础时间与内存估算指标
```

输出字段包括：

```text
case
dim
backend
ncell
ndof
nnz
assembly_time_s
assembled_matvec_time_s
matrix_free_matvec_time_s
rel_matvec_error
cg_converged
cg_iterations
cg_final_residual
matrix_free_solve_time_s
solve_rel_error
assembled_memory_mb
matrix_free_memory_est_mb
```

当前默认 benchmark 覆盖 2D 三角形网格和 3D 四面体网格的小规模案例。它用于支撑“当前 SOPTX Python/NumPy 原型已经完成 Matrix-Free 正确性与接口闭环验证”的结论；它不是 GPU/MPI 高性能 benchmark，不能直接替代 mfleo 的并行/GPU 性能结果。

### 下一阶段方向

当前正确性与接口闭环已经基本完成。若目标是支撑答辩 PPT 中“Matrix-Free 相比传统组装更适合大规模问题”的结果展示，后续优先级应从继续扩大功能覆盖面转向性能层：

```text
优先:
  单机 GPU 后端验证
  GPU/多后端 benchmark
  内存与时间趋势数据整理

暂不优先:
  多分辨率 contraction
  adjoint matrix-free 路径
  MPI/多节点并行
```

原因是当前答辩所需第一层证据已经具备：

```text
K @ x 与 MF(x) 一致
assembled solve 与 matrix-free CG solve 一致
LagrangeFEMAnalyzer 已支持 matrix-free 后端
2D/3D standard action 已经不形成局部 Ke
NumPy benchmark 已能输出 CSV + XLSX
```

但性能层仍需要新的 benchmark 和后端验证支撑，尤其是 GPU/多后端和更大规模内存趋势。当前 SOPTX Python/NumPy benchmark 只能证明正确性和接口闭环，不能直接声明已经达到 mfleo 的 MPI/GPU 性能水平。

## 10. 命名建议

为了避免语义混淆，后续代码中应尽量区分：

```text
rho                 原始或物理密度
coef                积分器刚度缩放系数
relative_stiffness  coef 的具体物理含义之一
material_params     lambda/mu/D 等材料参数
```

如果 Matrix-Free operator 的输入已经是 `relative_stiffness`，则字段名不宜继续叫 `rho`。更清晰的命名是：

```python
coef
relative_stiffness
cell_coef
```

这能避免后续误以为 Matrix-Free operator 内部负责 SIMP 或材料插值。

## 11. 后续讨论时的默认假设

后续除非另有说明，默认采用以下判断：

1. Matrix-Free 接入点在 `LagrangeFEMAnalyzer` 的状态方程求解后端。
2. `LinearElasticIntegrator` 只消费 `coef`，不解释 `rho`。
3. 阶段一接口打通已经完成。
4. 阶段二主体已经完成：`LagrangeFEMAnalyzer.solve_state()` 的非伴随状态方程支持 matrix-free 后端，并通过 analyzer 层一致性测试。
5. 灵敏度计算当前继续沿用现有局部刚度导数路径。
6. 阶段三第一步和第二步已经完成：`LinearElasticIntegrator.action()` 在 2D/3D `standard` 路径使用真正积分点 contraction。
7. 当前已建立 NumPy benchmark，用于正确性、接口闭环、基础时间和内存估算数据输出。
8. 下一阶段优先转向 GPU 后端验证和性能 benchmark；暂不优先扩展多分辨率 contraction。
9. mfleo 的主要借鉴点是职责划分、partial assembly 思想和高性能结果参照，而不是完整迁移 MFEM 结构。
