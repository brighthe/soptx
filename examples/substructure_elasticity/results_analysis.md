# 子结构静力缩聚示例结果分析

本页是 [`verify_full_trace_convergence.py`](verify_full_trace_convergence.py) 与 [`verify_linear_corner_consistency.py`](verify_linear_corner_consistency.py) 的数学—代码映射契约、验收契约与实验证据说明。运行入口和参数说明见 [`README.md`](README.md)；缩聚、接口装配、边界处理与恢复的完整推导见 [`experiments/analysis_capability_substructure/results_analysis.md`](../../experiments/analysis_capability_substructure/results_analysis.md) 第 2—3 节，本页沿用其记号。

## 1. 数学—代码映射契约

### 1.1 共用：局部刚度、精确缩聚与恢复

全部同构子结构共用一个 `SubstructurePrototype`，由 `build_substructures(assembler)` 创建；缩聚沿批量维 $B$（子结构数）一次完成。$i$、$b$ 分别表示子结构内部与接口自由度。

| 数学量 | 代码 | 形状 | 说明 |
|---|---|---|---|
| $i$、$b$ 划分 | `prototype.i_dofs`、`prototype.b_dofs` | `(n_i,)`、`(n_b,)` | 落在子结构外表面上的插值点为接口节点（`_classify_nodes`） |
| $\mathbf K^j$ | `prototype.assemble_local_stiffness_batch(density)` | `(B, n_dof, n_dof)` | 单元系数 `coef = rho ** penal`；`build_substructures` 默认 `penal=3.0`、`rho_min=0.0` |
| $\mathbf N_{\mathrm{int}}^j=-(\mathbf K_{ii}^j)^{-1}\mathbf K_{ib}^j$ | `ExactSchurCondensation(prototype.i_dofs, prototype.b_dofs)` 调用 `condense` 后的 `condensor.N` | `(B, n_i, n_b)` | `bm.linalg.solve(K_ii, K_ib)`，不显式求逆 |
| $\mathbf K_s^j=\mathbf K_{bb}^j-\mathbf K_{bi}^j(\mathbf K_{ii}^j)^{-1}\mathbf K_{ib}^j$ | `condensor.K_s` | `(B, n_b, n_b)` | 以 `bm.matrix_transpose(K_ib)` 代替 $\mathbf K_{bi}^j$，依赖刚度对称 |
| $\mathbf u_i^j=\mathbf N_{\mathrm{int}}^j\mathbf u_b^j$ | `condensor.recover(u_b)` | `(B, n_i)` | `bm.einsum('...ij, ...j -> ...i', ...)` |
| 全场位移 | `assembler.recover_full_displacement(sub_meshes, condensor, system, u_interface)` | `(total_full_dofs,)` | 接口分量按 `system.global_dofs` 写回，内部分量由上式恢复 |

**建模假设：内部自由度不受载。** `ExactSchurCondensation` 只缩聚刚度、不缩聚载荷，恢复式没有 $(\mathbf K_{ii}^j)^{-1}\mathbf f_i^j$ 项，因此只对 $\mathbf f_i^j=\mathbf 0$ 的问题成立。两个脚本在求解前检查该前提：`verify_full_trace_convergence.py` 装配 `pde.loads()` 中的体力并要求全为零，非 `BodyForce` 载荷直接拒绝；`verify_linear_corner_consistency.py` 的 `prepare_consistency_problem` 要求载荷与支承只落在子结构分割面节点上。

### 1.2 `verify_full_trace_convergence.py`

算例为 `HarmonicPoly2D`（$[0,1]^2$，`plane_stress`）与 `HarmonicPoly3D`（$[0,1]^3$，`3D`）：零体力、全外边界给定解析位移。二维本构取原型默认的 `plane_stress`，脚本核对 `prototype.material.hypothesis` 与配置材料一致。加密方式为 `base_sub = 2`，第 `lvl` 层各方向 `n_sub = 2 * 2**lvl`，块内 `n_fine` 固定为 `(2, 2)` / `(2, 2, 2)`；默认层数 `DEFAULT_LEVELS = {2: 4, 3: 3}`。次数只允许 $p\in\{1,2\}$：三次多项式制造解在 $p\ge3$ 时可被精确表示。

| 数学量 | 代码 | 形状 / 说明 |
|---|---|---|
| $\mathbf K_\Gamma=\sum_j\mathbf A_j^{\mathsf T}\mathbf K_s^j\mathbf A_j$ | `assembler.assemble_interface_system(sub_meshes, condensor)` 返回的 `system.stiffness` | `(n_Γ, n_Γ)`，`system.global_dofs` 升序 |
| $D$、$\mathbf u_D$ | `space.boundary_interpolate(gd=pde.dirichlet_bc, ..., method="interp")` 得 `dofs_val`、`fixed_mask`，再经 `assembler.project_global_dofs` / `project_global_vector` 投到接口 | `space = assembler.space_full` |
| $(\mathbf K_\Gamma)_{FF}\mathbf U_F=-(\mathbf K_\Gamma)_{FD}\mathbf U_D$ | `solve_interface_system(system, zero_load, fixed_interface_dofs, prescribed=prescribed_interface, solver=solve_method)` | 右端为零，由给定接口位移驱动；`solve_method` 取 `scipy` 或 `mumps` |
| $\lVert\mathbf u-\mathbf u_h\rVert_0$ | `assembler.full_mesh.error(pde.disp_solution, uh_sub, q=max(4, degree + 3))` | 标量 |
| $\lvert\mathbf u-\mathbf u_h\rvert_1$ | `assembler.full_mesh.error(pde.grad_disp_solution, uh_sub.grad_value, q=...)` | 标量 |
| $h$ | `mesh_size = max(domain_size[d] / total_fine[d])` | 标量 |
| 观测阶 | `math.log(error_ratio) / math.log(mesh_ratio)` | 相邻两层 |

可选的同网格 FA 对照由 `run_convergence_benchmark(..., compare_fa=True)` 开启；命令行不暴露该开关，单独运行脚本时不求解 FA。开启后，FA 刚度取 `LagrangeFEMAnalyzer(operator_level="fa", ...).assemble_stiff_matrix()`，包装为 `InterfaceSystem(full_stiffness, arange(total_full_dofs))`，用同一 `solve_interface_system` 求解。对照量为：

- 应变能：`full_trace` 取 $\frac12\sum_j\mathbf u^{j\mathsf T}\mathbf K^j\mathbf u^j$（`np.einsum("bi,bij,bj->", ...)`），FA 取 $\frac12\mathbf u^{\mathsf T}\mathbf K\mathbf u$；
- 自由自由度残差 `_free_residual`：$\eta=\lVert(\mathbf K\mathbf u)_f\rVert_2/\lVert(\mathbf K\mathbf u_c)_f\rVert_2$，$\mathbf K$ 为全细网格刚度，$\mathbf u_c$ 只在受约束自由度上取给定值。

### 1.3 `verify_linear_corner_consistency.py`

算例为 `HalfMBBBeamRight2d`（定义域 $[0,60]\times[0,20]$，默认 `n_sub=6x2`、`n_fine=5x5`）与 `FullMBBBeam3d`（$[0,6]\times[0,1]\times[0,1]$，默认 `6x2x2`、`4x4x4`），均取 $E=1$、$\nu=0.3$、$P=-1$，固定 Q1。密度 `make_cell_density` 默认 `cell`：在细单元中心取 $0.7+0.3\sin(\pi x)\cos(\pi y)$（三维再乘 $\sin(\pi z)$），坐标按全局细网格归一化；`uniform` 取常数 0.7。载荷与支承由 `make_fa_analyzer(...)` 的 `assemble_external_load()` 与 `tensor_space.boundary_interpolate(...)` 生成。

| 数学量 | 代码 | 形状 / 说明 |
|---|---|---|
| $\mathbf L_j$ | `LinearCornerTraceBasis.from_prototype(prototype)`，矩阵即 `prototype.linear_boundary_matrix` | `(n_b, d·2^d)`，角点（双/三）线性插值 |
| $\mathbf K_c^j=\mathbf L_j^{\mathsf T}\mathbf K_s^j\mathbf L_j$ | `trace_basis.project_stiffness(condensor.K_s)` | `(B, d·2^d, d·2^d)` |
| $\mathbf K_C=\sum_j\mathbf A_{c,j}^{\mathsf T}\mathbf K_c^j\mathbf A_{c,j}$ | `assembler.assemble_macro_system(sub_meshes, ...)` 返回的 `macro.stiffness` | `(total_macro_dofs, total_macro_dofs)` |
| $\mathbf U_\Gamma=\mathbf P\mathbf U_C$ | `assembler.build_linear_corner_projection(sub_meshes, interface_view, trace_basis)`（脚本经 `build_corner_projection` 调用） | SciPy CSR，`(n_Γ, total_macro_dofs)` |
| $\mathbf F_C=\mathbf P^{\mathsf T}\mathbf F_\Gamma$ | `projection.T @ interface_force` | 以原完整载荷的虚功为准，不在角点重新生成物理载荷 |
| $\mathbf C_D\mathbf U_C=\mathbf 0$ | `constraints = projection[fixed_interface]` | $\mathbf P$ 在受约束接口自由度上的行 |
| 约束求解 | `solve_constrained_system(macro, force, constraints)`（缺省 `solver="scipy"`） | 返回 `ConstrainedSolveResult`：`displacement`、`constraint_rank`、`equilibrium_relative_residual`、`constraint_relative_residual`、`mode` |
| 全场位移 | `projection @ q` 后调用 `assembler.recover_full_displacement(...)` | `(total_full_dofs,)` |
| $\mathbf K_\Gamma$（仅诊断） | `assembler.assemble_interface_system(sub_meshes, condensor)` | 只用于 Galerkin 恒等式检查，不计入计时 |

## 2. 验收契约

### 2.1 `verify_full_trace_convergence.py`

| 检查 | 断言 | 常量 / 值 | 失败方式 |
|---|---|---|---|
| 零外载 | 体力装配结果全为零，且无非 `BodyForce` 载荷 | 精确 | `ValueError` |
| 支承落在接口 | 全部 Dirichlet 自由度属于 `system.global_dofs` | 精确 | `ValueError` |
| 材料一致 | `prototype.material.hypothesis == material.hypothesis` | 精确 | `ValueError` |
| 位移有限 | 恢复位移无 NaN / Inf | 精确 | `AssertionError` |
| 误差有效 | 每层 $L^2$、$H^1$ 半范误差为有限正数；相邻层观测阶有限 | 精确 | `AssertionError` |
| 收敛阶 | 最细一对网格：$L^2$ 阶 $\ge p+1-\delta$，$H^1$ 半范阶 $\ge p-\delta$ | `ORDER_MARGIN = 0.20`（$p=1$ 时为 1.8、0.8） | `AssertionError` |
| FA 等价（仅 `compare_fa=True`） | 每层位移相对差、应变能相对差 $\le$ 阈值 | `FA_EQUIVALENCE_TOLERANCE = 1.0e-11` | `AssertionError` |
| FA 残差（仅 `compare_fa=True`） | 每层 `full_trace` 与 FA 的自由残差 $\eta\le$ 阈值 | `FREE_RESIDUAL_TOLERANCE = 1.0e-9` | `AssertionError` |

只有最后一对网格的观测阶参与门禁，中间层的阶数只打印。全部门禁通过后才写出 `full_trace_convergence_{dim}d_{model}_p{degree}_levels{levels}_solver-{solve_method}.json`，`schema_version` 为 `full-trace-convergence-v1`，开启 FA 对照时为 `full-trace-convergence-fa-reference-v2`。

### 2.2 `verify_linear_corner_consistency.py`

求解前的输入检查（`ValueError`）：Dirichlet 给定值必须全为零；载荷必须有限且非零；载荷不得落在非分割面节点上，支承自由度必须全部落在分割面节点上。

`validate_linear_corner_consistency` 要求 `LINEAR_CORNER_CONSISTENCY_KEYS` 中的七项全部有限且不超过 `CONSISTENCY_TOLERANCE = 1.0e-9`，否则 `AssertionError`。记 $\mathbf q$ 为角点解，$\mathbf u$ 为恢复后的全场位移，$\mathbf F$ 为全场载荷，$c=\mathbf F^{\mathsf T}\mathbf u$；各分母另以 `np.finfo(float).tiny` 防零。

| 键 | 定义 |
|---|---|
| `galerkin_relative_error` | $\lVert\mathbf K_C-\mathbf P^{\mathsf T}\mathbf K_\Gamma\mathbf P\rVert_F/\lVert\mathbf P^{\mathsf T}\mathbf K_\Gamma\mathbf P\rVert_F$（`scipy.sparse.linalg.norm`） |
| `equilibrium_relative_residual` | 由 `solve_constrained_system` 给出；乘子路径为 $\lVert\mathbf K_C\mathbf q+\mathbf C^{\mathsf T}\boldsymbol\lambda-\mathbf F_C\rVert/\max(\lVert\mathbf F_C\rVert,\lVert\mathbf K_C\mathbf q\rVert,\lVert\mathbf C^{\mathsf T}\boldsymbol\lambda\rVert)$，坐标消元路径只在自由分量上计算 |
| `constraint_relative_residual` | $\lVert\mathbf C\mathbf q-\mathbf d\rVert/\max(\lVert\mathbf d\rVert,\lVert\mathbf q\rVert)$ |
| `recovered_support_relative_error` | $\lVert\mathbf u_D\rVert/\lVert\mathbf u\rVert$，在全细网格受约束自由度上取值 |
| `internal_relative_residual` | $\lVert(\mathbf K^j\mathbf u^j)_i\rVert/\lVert\mathbf K^j\mathbf u^j\rVert$，对全部子结构合并取范数 |
| `energy_relative_error` | $\lvert\sum_j\mathbf u^{j\mathsf T}\mathbf K^j\mathbf u^j-c\rvert/\lvert c\rvert$，两侧均为两倍应变能 |
| `load_work_relative_error` | $\lvert\mathbf F_C^{\mathsf T}\mathbf q-c\rvert/\lvert c\rvert$ |

本脚本不以相对 FA 的近似误差作为门禁，PASS 只表示角点迹降阶系统代数自洽，不表示其位移或柔顺度与 FA 一致。通过后写出 `linear_corner_consistency_{dim}d_sub-{sub}_fine-{fine}.json`，`schema_version` 为 `linear-corner-consistency-v1`。

## 3. 实验证据

本目录不保存实验证据。正确性与收敛性、密度更新一致性和计算成本的证据统一保存在 [`experiments/analysis_capability_substructure/results_analysis.md`](../../experiments/analysis_capability_substructure/results_analysis.md)，分别见其 [§4 正确性与收敛性](../../experiments/analysis_capability_substructure/results_analysis.md#4-正确性与收敛性)、[§5 密度更新一致性](../../experiments/analysis_capability_substructure/results_analysis.md#5-密度更新一致性) 与 [§6 计算成本](../../experiments/analysis_capability_substructure/results_analysis.md#6-计算成本)。两个入口与该实验的关系如下：

- **`full_trace` 收敛**：实验 task `full_trace_convergence` 直接调用本目录的 `run_convergence_benchmark(..., compare_fa=True)`，使用 MUMPS。该文 §4.4 记录二维、三维最后一对网格的 $L^2$ / $H^1$ 半范阶分别为 2.0006 / 1.0001 与 2.0019 / 1.0003，与同网格 FA 的位移、应变能相对差及自由残差均通过上述阈值（数值转录自该文 §4.4，产物目录时间戳为 `20260914T...`）。
- **`linear_corner` 一致性**：实验 §4 的 `linear_corner_consistency_2d/3d` 由 `_corner_convergence.py` 在 `HarmonicPoly2D/3D` 制造解上执行，**不是**本目录的 MBB 一致性入口，两者同名但对象不同。本脚本只有 `make_fa_analyzer`、`build_corner_projection` 被实验的 `_density_update.py`、`_fa_chunked.py`、`_cost_measurement.py` 复用（服务于 §5、§6），`run_linear_corner_consistency` 本身没有入库证据。

**本目录两个入口在当前代码上均待跑。** 上述 §4.4 证据生成于 2026-09-14，此后 `examples/substructure_elasticity/` 又经提交 `71a03d6`、`5ac85e6`、`b55f265`、`dbdbb33` 修改，不能据此宣称当前入口已通过。

| 入口 | 命令 | 验收条件 | 结果 |
|---|---|---|---|
| `full_trace` 收敛（2D） | `python examples/substructure_elasticity/verify_full_trace_convergence.py --problem HarmonicPoly2D` | §2.1，$p=1$ 时末阶 $\ge$ 1.8 / 0.8 | 待跑 |
| `full_trace` 收敛（3D） | `python examples/substructure_elasticity/verify_full_trace_convergence.py --problem HarmonicPoly3D` | 同上 | 待跑 |
| `linear_corner` 一致性（2D） | `python examples/substructure_elasticity/verify_linear_corner_consistency.py --problem HalfMBBBeamRight2d` | §2.2，七项 $\le 10^{-9}$ | 待跑 |
| `linear_corner` 一致性（3D） | `python examples/substructure_elasticity/verify_linear_corner_consistency.py --problem FullMBBBeam3d` | 同上 | 待跑 |
