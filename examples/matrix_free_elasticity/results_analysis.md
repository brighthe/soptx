# 线弹性 Matrix-Free 2D/3D 实验证据报告

本报告是 `soptx/examples/matrix_free_elasticity` 数值结果的唯一事实源。
运行入口与文件职责见 [`README.md`](README.md)；每道门禁的数学式由知识库维护
（入口见下方第 1 节），阈值本身只在 [`utils/contract.py`](utils/contract.py)
定义一次。

下方带 `<!-- BEGIN/END GENERATED -->` 标记的区块由
`python utils/sync_results.py --dim {2,3}` 从 `outputs/` 下的原始 JSON 生成，
**不要手工编辑**；`--check` 用于只读比对是否漂移。

## 1. 数学—代码映射契约

算子代数、边界条件施加形式与门禁的数学式全部由知识库维护，本仓库不复制推导：

| 数学事实 | 事实源 |
|---|---|
| 五级装配层次、EA/FA 判定口径、跨层级不变量 | `dut-postdoc:concepts/matrix-free/assembly-levels.md#五级分类` |
| Dirichlet 条件在各层级下的施加、并行下 FA 对称消元不成立 | `dut-postdoc:concepts/matrix-free/assembly-levels.md#本质边界条件在各层级下的施加` |
| 五道跨层级门禁的标准数学形式 | `dut-postdoc:concepts/matrix-free/assembly-levels.md#跨层级正确性判据` |
| EA 的算术强度边界及其表述口径 | `dut-postdoc:concepts/matrix-free/assembly-levels.md#算术强度：EA 并没有解决 FA 的瓶颈` |
| MPI 共享自由度、同步归约 $\mathcal S$、幂等投影 $\mathcal C$、加权内积 | `dut-postdoc:concepts/gpu-hpc/distributed-operator-and-shared-dofs.md` |
| 线弹性变分形式与有限元离散 | `dut-postdoc:concepts/linear-elasticity.md#线弹性方程变分形式与有限元离散` |
| 三个制造解的方程、参数与 shape 契约 | [`docs/problems/manufactured-elasticity.md`](../../docs/problems/manufactured-elasticity.md) |

### 1.1 符号—代码映射

设 $\Omega\subset\mathbb R^d$（$d\in\{2,3\}$）的一致剖分为 $\{\Omega_e\}_{e=1}^{N_e}$，
一阶连续 Lagrange 向量元空间的全局自由度数为 $n=\texttt{TGDOF}$，单元自由度数
$m=(d+1)d$。

| 数学符号 | 代码位置 | 代数含义 |
|---|---|---|
| $\mathbf K_e$ | `LinearElasticIntegrator` 计算、`BilinearForm` 在 `'ea'` 下保存的单元刚度张量 | 单元刚度矩阵 |
| $\mathbf R_e$ | `space.cell_to_dof()` | 单元自由度限制算子（gather），转置为 scatter-add |
| $\mathbf K$ | `assemble_stiff_matrix.set('fa')` 返回的 `CSRTensor` | 全局刚度矩阵 |
| $\tilde{\mathbf K}$ | [`DirichletBCOperator`](utils/analyzer.py) 包裹后的算子 | 施加本质边界条件后的算子 |
| $\mathbf E_p$ | [`soptx/fem/distributed.py:_vector_dof_masks`](../../src/soptx/fem/distributed.py) 得到的 DOF 掩码 | rank $p$ 的局部自由度嵌入算子 |
| $r_i$ | `dof_comm.refs(size)` | 自由度 $i$ 的副本数（被多少个 rank 持有） |
| $w_i=1/r_i$ | `dof_comm.dot(size)` / [`soptx/fem/solvers/matrix_free_solver.py`](../../src/soptx/fem/solvers/matrix_free_solver.py) 内部 | 重叠内积权重 |
| $\mathcal S(\cdot)$ | `dof_comm.sync_add(...)` / [`soptx/fem/distributed.py:OverlapOperator`](../../src/soptx/fem/distributed.py) | 跨 rank 共享分量求和 |
| $\boldsymbol\Pi_D,\ \boldsymbol\Pi_I$ | `DirichletBCOperator.is_boundary_dof` 及其补 | Dirichlet／内部自由度上的对角投影 |
| $\bar{\boldsymbol u}$ | `_prescribed_solution` / `ElasticityEAOperator.prescribed_solution` | 边界取给定值、内部取零的基准向量 |

`partition_cells` 产生的单元掩码互不相交且完全覆盖（代码中以 `coverage == 1`
断言），因而知识库中 $\mathbf K=\sum_p\mathbf E_p\mathbf K^{(p)}\mathbf E_p^{\mathsf T}$
的精确分解前提在本实现中成立。

### 1.2 两级算子的保存／省略对象

| 层级 | 保存对象 | 省略对象 | 每次 MatVec |
|---|---|---|---|
| `fa` | 全局 CSR $\mathbf K$（`OPERATOR_STORAGE['fa'] = "global-csr"`） | 无 | 稀疏矩阵乘 |
| `ea` | 单元矩阵集合 $\{\mathbf K_e\}$（`"cached-element-matrices"`） | 全局矩阵 $\mathbf K$ | gather—单元乘—scatter-add |

$\mathbf K_e$ 被完整形成并保存，因此按事实源的判定口径本实现属于 **EA/EbE**，
不是 PA/QA。`contract.OPERATOR_LEVELS` 相应只有 `("ea", "fa")`。

### 1.3 门禁与阈值来源

阈值只在 [`utils/contract.py`](utils/contract.py) 定义一次，本文档不复制数字，
知识库侧也不持有字面量。

| 门禁 | 阶段 | 判据类别 | 阈值常量 |
|---|---|---|---|
| `converged` | 1a | CG 正常退出且无 breakdown | — |
| `true_residual` | 1a | 加权范数下的真残差 | `DEFAULT_RTOL` / `DEFAULT_ATOL` |
| `boundary_dofs` | 1a | Dirichlet 分量与给定值之差 | `BOUNDARY_ABSOLUTE_TOL` |
| `raw_matvec` | 1a | 裸 MatVec 一致 | `MATVEC_RELATIVE_TOL` |
| `dirichlet_matvec` | 1a | 边界后 MatVec 一致 | `MATVEC_RELATIVE_TOL` |
| `operator_symmetry` | 1a | 双线性配对对称且正定 | `SYMMETRY_RELATIVE_TOL` |
| `explicit_solution` | 1a | CG 解与 FA 直解一致 | `EXPLICIT_SOLUTION_RELATIVE_TOL` |
| EA/FA 解一致 | 1a | 解一致 | `EA_FA_SOLUTION_RELATIVE_TOL` |
| 误差单调 | 1a | $E_0>E_1>E_2$ | — |
| 收敛阶 | 1a | 末段观测阶 | `MINIMUM_FINAL_L2_ORDER` |
| 1/2-rank 解一致 | 1b | 解一致 | `PARALLEL_SOLUTION_RELATIVE_TOL` |
| 1/2-rank 误差一致 | 1b | 末档 L2 误差之差 | `PARALLEL_L2_DIFFERENCE_TOL` |

前四道带参照的门禁使用 `REFERENCE_RANDOM_SEED` 生成的固定随机向量，
$\boldsymbol x_{\mathrm{direct}}$ 由 `spsolve` 在 `fa` 系统上独立求出；这些门禁
只在单 rank 非 benchmark 运行下有意义，其余情形记为 `GATE_SKIPPED` 而非通过，
否则 `local_passed` 会在 benchmark 模式下悄悄弱化。加密序列由 `REFINEMENTS`
给出（2D 为 `8,16,32`，3D 为 `4,8,16`），相邻恰为二等分，故观测阶取 $\log_2$。
凡分母出现范数处一律以 `NORM_FLOOR` 兜底。

标注 1b 的两道门禁只在 `--include-parallel` 下参与判定；阶段 1a 的 `comparison`
中不写入对应字段，以免空占位被误读为「已检验」。EA/FA 解一致检验的是对称消元与
算子包裹两种施加方式的等价性，是 1a 的核心代数判据；两道 1b 门禁检验的是
$\mathcal S$ 与 $\mathcal C$ 的实现。

## 2. 当前证据状态

**结论：目前没有可用于验收的正式证据。** 下方两个区块只是 dirty worktree 的开发
证据，且它们是在**阶段 1b 范围**（含 2-rank 算例）下产生的，而当前默认范围已收窄
为**阶段 1a（CPU 串行 EA/FA）**。重放后区块将只含单 rank 结果。

| 项 | 实际值 |
|---|---|
| 区块源 revision | `4cd4e8da17189eb57f9a68cc316bcdf189c084ec` |
| `evidence/*.json` 的 `environment.git_dirty` | **`true`** |
| 距当前 HEAD | `4cd4e8d..HEAD` 共 9 个提交 |
| 1/2-rank 一致性正式证据 | 未固化，`evidence/` 下只有单 rank 产物 |

两点说明：

1. 这两个区块生成时，`utils/sync_results.py` 既没有校验 `git_dirty`，又把
   `git_dirty=false` 当字面量写进正文，因此区块曾错误宣称自己 clean。该缺陷已
   修复：`require_formal_environment` 现在硬性拒绝 `git_dirty != false` 的原始
   JSON，渲染时也改为读取 payload 的真实标志。**在 clean revision 上重放之前，
   `utils/sync_results.py` 会以非零状态退出**，这是预期行为。
2. `4cd4e8d` 之后，`lagrange_fem_analyzer.py`、`linear_elasticity.py` 和
   `manufactured_2d.py` 都在 evidence 依赖路径上发生过改动，所以即便忽略 dirty
   标志，这两个区块也已经不对应当前代码。

因此这两个区块**待在冻结的 clean target revision 上重放**，重放前不得作为验收
结论、对外表达或申报材料的数值来源。

## 3. 历史基线

迁移到 `src/soptx` 与语义 Problem **之前**的三维数值证据保存在
[`evidence/cpu-single-rank-fa-ea-3d-historical.json`](evidence/cpu-single-rank-fa-ea-3d-historical.json)，
只作为原三维实现的历史基线，不作为本次 2D/3D 通用化实现的验收结论。

## 4. 2D CPU 单 rank FA/EA

<!-- BEGIN GENERATED: cpu-single-rank-fa-ea-2d -->

本节由 `utils/sync_results.py --dim 2` 根据 clean-revision 原始 JSON 生成；精简证据见 `evidence/cpu-single-rank-fa-ea-2d.json`。
源 revision：`4cd4e8da17189eb57f9a68cc316bcdf189c084ec`；`git_dirty=true`。

| 网格 | EA-CG 迭代数 | 真相对残差 | 相对 L2 误差 | 边界绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| `8×8` | 38 | `5.13210e-11` | `4.61057e-02` | `0` |
| `16×16` | 89 | `8.96971e-11` | `1.20318e-02` | `0` |
| `32×32` | 188 | `8.95970e-11` | `3.04605e-03` | `0` |

| 网格 | 原始 EA/FA MatVec | Dirichlet EA/FA MatVec | EA-CG/FA 直接解 |
| --- | ---: | ---: | ---: |
| `8×8` | `1.44949e-16` | `1.40930e-16` | `8.67653e-12` |
| `16×16` | `1.62572e-16` | `1.64234e-16` | `6.57436e-12` |
| `32×32` | `1.57536e-16` | `1.56086e-16` | `3.03229e-12` |

相对 L2 误差观测阶为 `1.93809`、`1.98184`。独立 FA 粗网格 `8×8` 在 38 步收敛，真相对残差为 `4.95406e-11`。

<!-- END GENERATED: cpu-single-rank-fa-ea-2d -->

## 5. 3D CPU 单 rank FA/EA

<!-- BEGIN GENERATED: cpu-single-rank-fa-ea-3d -->

本节由 `utils/sync_results.py --dim 3` 根据 clean-revision 原始 JSON 生成；精简证据见 `evidence/cpu-single-rank-fa-ea-3d.json`。
源 revision：`4cd4e8da17189eb57f9a68cc316bcdf189c084ec`；`git_dirty=true`。

| 网格 | EA-CG 迭代数 | 真相对残差 | 相对 L2 误差 | 边界绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| `4×4×4` | 24 | `9.26796e-11` | `6.80637e-01` | `0` |
| `8×8×8` | 64 | `6.69625e-11` | `2.85095e-01` | `0` |
| `16×16×16` | 134 | `8.62632e-11` | `8.91922e-02` | `0` |

| 网格 | 原始 EA/FA MatVec | Dirichlet EA/FA MatVec | EA-CG/FA 直接解 |
| --- | ---: | ---: | ---: |
| `4×4×4` | `2.71490e-16` | `1.98982e-16` | `5.77006e-11` |
| `8×8×8` | `3.17642e-16` | `2.69544e-16` | `1.93814e-11` |
| `16×16×16` | `3.38707e-16` | `2.63232e-16` | `1.98507e-11` |

相对 L2 误差观测阶为 `1.25544`、`1.67645`。独立 FA 粗网格 `4×4×4` 在 24 步收敛，真相对残差为 `9.26796e-11`。

<!-- END GENERATED: cpu-single-rank-fa-ea-3d -->

## 6. 证据边界

本节只说明**这批数字能支持什么结论**。实现层面的能力边界（不实现 PA/QA、无预
条件、EA 的算术强度口径等）由 [`README.md`](README.md) 的「本阶段明确不承诺的
内容」唯一维护，此处不复制。

- **收敛阶未进入渐近区。** 3D 相对 $L^2$ 观测阶为 `1.25544`、`1.67645`，低于 P1
  元的理论二阶。当前三档网格（`4/8/16`）尚未进入渐近区，只能说"误差随加密单调
  下降且末段观测阶过门禁"，不得宣称"收敛阶达到二阶"。2D 的 `1.93809`、`1.98184`
  已接近二阶。
- **一致性结论不等于性能结论。** MatVec 与解的机器精度级一致只支持"EA 与 FA 是
  同一个离散算子"，不支持任何关于完整 solve 时间、迭代数优劣或内存占用的结论——
  本报告目前不含任何计时或峰值内存数据。
- **无跨 rank 结论。** 阶段 1a 只跑单 rank。区块内若出现 2-rank 数字，属于重放前
  的历史遗留，不得引用。
