# 线弹性 Matrix-Free 2D/3D MPI 基线

本目录提供同一套二维/三维线弹性 EA/EbE Matrix-Free 基线：一阶连续
Lagrange 向量元、串行 FA（显式全局 CSR）参考、重叠副本 MPI 和无预条件加权
CG。EA 是默认路径；FA 只支持单 rank，用于同条件数值对照。

支持范围明确限定为二维和三维，不宣称支持任意空间维数。

## 算例配置

维度相关内容集中在 [`cases.py`](cases.py)，求解核心不根据 PDE 类名判断维度。

| 维度 | PDE | 网格 | 位移空间 | 材料假设 |
| --- | --- | --- | --- | --- |
| 2D | `SinusoidalPlaneStrainElasticity2D` | `TriangleMesh` | `shape=(-1, 2)` | `plane_strain`，$E=1,\nu=0.3$ |
| 3D | `DivergenceFreePolynomialElasticity3D` | `TetrahedronMesh` | `shape=(-1, 3)` | `3D`，$\lambda=\mu=1$ |

二维制造解及其体力与 `plane_strain` 本构配套，本阶段不暴露
`plane_stress` 开关。Problem 提供制造解、体力和边界数据；材料参数由独立
`MaterialSpec` 提供，算子构造不再从 PDE 对象隐式读取材料属性。

三个制造解的完整方程、参数与 shape 契约见
[`docs/problems/manufactured-elasticity.md`](../../docs/problems/manufactured-elasticity.md)。
Problem 不创建网格；`ElasticityCase.create_mesh()` 显式组合 Problem、Material
和 Mesh。

EA/FA 两级算子的保存与省略对象、重叠副本的向量表示与算子代数、Dirichlet 施加
方式以及全部数值门禁的数学定义，见 [`math_spec.md`](math_spec.md)。本 README 只
负责使用说明、运行入口和 evidence 状态。

线弹性理论入口为
`dut-postdoc:concepts/linear-elasticity.md#线弹性方程变分形式与有限元离散`；
Matrix-Free 装配层次和 MPI 重叠 DOF 理论入口分别为
`dut-postdoc:concepts/matrix-free/assembly-levels.md#五级分类` 和
`dut-postdoc:concepts/distributed-operator-and-shared-dofs.md`。

## 目录结构与核心模块

- [`operator.py`](operator.py)：核心。2D/3D EA 矩阵无关算子 (`ElasticityEAOperator`) 与 `OverlapOperator`（MPI 共享自由度同步归约）；
- [`solver.py`](solver.py)：核心。分布式加权 Krylov 求解器（`weighted_cg` / `solve_matrix_free_system`）与真残差、边界误差诊断；
- [`cases.py`](cases.py)：2D/3D 物理问题（平面应变与多项式无散场）、材料参数与制造解定义；
- [`minimal_demo.py`](minimal_demo.py)：极简入口。一键运行 2D/3D Matrix-Free CG 求解（0 繁琐 CLI），可单 rank 或 `mpiexec -n 2` 运行；
- [`compare_lagrange.py`](compare_lagrange.py)：交叉比对。与全组装 (FA) CSR 矩阵及 Scipy 直解做 $\sim 10^{-16}$ 机器精度级对照；
- [`math_spec.md`](math_spec.md)：算子代数、重叠副本表示与门禁的 1-to-1 数学定义规范；
- [`utils/`](utils/)：基础设施胶水包，收拢 `contract`、`schema`、`report`、`layout`、`analyzer`、`distributed`、`references`、`postprocess`、`run`、`validate` 和 `sync_results`。

## 环境与运行

极简 Demo（一键运行 Matrix-Free CG 求解）：

```bash
python minimal_demo.py --dim 2
python minimal_demo.py --dim 3
mpiexec -n 2 python minimal_demo.py --dim 2
```

机器精度级交叉比对（校验 Raw MatVec、BC MatVec、算子对称性与 CG/Scipy 直解）：

```bash
python compare_lagrange.py --dim 2
python compare_lagrange.py --dim 3
```

完整自动化验证与证据同步：

```bash
python validate.py --dim all
python sync_results.py --dim all
```

```text
main
  → parse_arguments → create_case(dim) + RunConfig
  → execute
      → build_global_context → partition_cells(split_coordinate=…)
      → distribute → distribute_mesh / distribute_vector_space / serial_references
      → build_distributed_analyzer(operator_level=…)
      → run_solver
          → analyzer.apply_bc(assemble_stiff_matrix(), assemble_body_force_vector())
          → analyzer.solve_system → fealpy cg(dot_product=dof_comm.dot)
          → solver_diagnostics
      → dof_comm.gather_add(local_solution / references)
      → finalize
          → solution_error / write_solution
          → report.local_gates → report.build_payload → report.write_json
```

EA 保存完整单元矩阵集合 $\{\mathbf K_e\}$，每次 MatVec 执行
gather—单元作用—scatter-add；FA 形成并保存全局 CSR。两者对应同一个离散算子，
代数细节见 [`math_spec.md` 第 2–3 节](math_spec.md)。

## 环境与运行

从 SOPTX 仓库根目录执行，使用包含 editable FEALPy、`mpi4py` 和 MPI Runtime
的环境，例如：

```powershell
conda activate ihpcm
python -m pip install -e ".[mpi,test]"
```

二维 EA：

```powershell
mpiexec -n 1 python .\examples\matrix_free_elasticity\run.py `
  --dim 2 --operator-level ea --p 1 --nx 8 --ny 8
```

三维 EA（`--dim` 默认值为 `3`，此处显式写出）：

```powershell
mpiexec -n 1 python .\examples\matrix_free_elasticity\run.py `
  --dim 3 --operator-level ea --p 1 --nx 4 --ny 4 --nz 4
```

串行 FA：

```powershell
mpiexec -n 1 python .\examples\matrix_free_elasticity\run.py `
  --dim 2 --operator-level fa --p 1 --nx 8 --ny 8
```

二维显式传入 `--nz` 会报错。阶段 1 只接受 `p=1` 和 1/2 ranks；EA 支持
1/2 ranks，FA 和 `--benchmark` 只支持单 rank。

## 验证

先跑不需要 MPI 的单元测试（CG 分支、产物命名契约、分区逻辑）：

```powershell
pytest .\examples\matrix_free_elasticity
```

再按维度或同时准备完整验证：

```powershell
python .\examples\matrix_free_elasticity\validate.py --dim 2
python .\examples\matrix_free_elasticity\validate.py --dim 3
python .\examples\matrix_free_elasticity\validate.py --dim all
```

每个维度运行三组单 rank EA 网格、细网格 2-rank EA 和粗网格单 rank FA。
门禁包括：

- CG 收敛、无 breakdown，真实残差满足给定容差；
- Dirichlet DOF 绝对误差不超过 `1e-12`；
- 原始及 Dirichlet 后 EA/FA MatVec 相对误差不超过 `1e-12`；
- EA 算子对称且随机向量能量为正；
- CG 与显式 FA 解相对误差不超过 `1e-8`；
- 单/双 rank 解相对差及粗网格 EA/FA 解相对差不超过 `1e-9`；
- L2 误差严格下降，最后一段观测阶不低于 `1.5`。

每道门禁的精确数学式、所检验的代数等价性及其阈值来源见
[`math_spec.md` 第 5 节](math_spec.md)；阈值本身只在 [`contract.py`](contract.py)
定义一次。

驱动还会确认非法维数、2D 携带 `--nz`、非正网格数、非 `p=1` 以及
FA 多 rank 均以非零状态和对应的明确错误信息退出，且不生成结果产物。

验证完成后同步或只读检查对应维度证据：

```powershell
python .\examples\matrix_free_elasticity\sync_results.py --dim 2
python .\examples\matrix_free_elasticity\sync_results.py --dim 3
python .\examples\matrix_free_elasticity\sync_results.py --dim all --check
```

## 输出协议

单次运行使用 schema version 2，`stage` 为
`soptx/matrix-free-elasticity/stage-1`。`parameters` 明确记录：

- `dimension`、`case`、`domain` 和 `resolution`；
- `material.hypothesis` 及对应材料常数；
- 有限元次数、求解容差、算子层级、存储方式和 MPI 表示。
- `environment` 中的 UTC 时间、运行时版本、Git revision 与 dirty flag；
- 用于 EA/FA 随机向量校验的固定随机种子。

`sync_results.py` 只接受 `git_dirty=false` 的 clean-revision 原始 JSON；
dirty worktree 的验证结果只能作为开发证据。

默认产物写入已忽略的 `outputs/`，文件名包含 `2d` 或 `3d`：

- `*.json`：参数、环境、残差、误差和本地门禁；
- `*.npy`：rank 0 收集的唯一全局位移向量；
- `*.vtu`：三角形或四面体单元重心位移及误差；
- `stage1-validation-{2,3,all}.json`：跨运行门禁。

## 当前证据状态

下方 2D/3D 结果区块由迁移到 `src/soptx` 与语义 Problem **之后**的
`608cedf25038ed690f6db3be5b3f24f92329c5ec` 生成，`git_dirty=false`。其后 HEAD
已经推进，`lagrange_fem_analyzer.py`、`linear_elasticity.py` 和
`manufactured_2d.py` 都在 evidence 依赖路径上发生过改动，因此这两个区块**待在
当前目标 revision 上重放**，在重放完成前只作为迁移期基线。1/2-rank 一致性的正式
证据尚未固化，`evidence/` 下目前只有单 rank 产物。

迁移**之前**的三维数值证据保存在
[`evidence/cpu-single-rank-fa-ea-3d-historical.json`](evidence/cpu-single-rank-fa-ea-3d-historical.json)，
只作为原三维实现的历史基线，不作为本次 2D/3D 通用化实现的验收结论。

### 2D CPU 单 rank FA/EA

<!-- BEGIN GENERATED: cpu-single-rank-fa-ea-2d -->

本节由 `sync_results.py --dim 2` 根据 clean-revision 原始 JSON 生成；精简证据见 `evidence/cpu-single-rank-fa-ea-2d.json`。
源 revision：`4cd4e8da17189eb57f9a68cc316bcdf189c084ec`；`git_dirty=false`。

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

### 3D CPU 单 rank FA/EA

<!-- BEGIN GENERATED: cpu-single-rank-fa-ea-3d -->

本节由 `sync_results.py --dim 3` 根据 clean-revision 原始 JSON 生成；精简证据见 `evidence/cpu-single-rank-fa-ea-3d.json`。
源 revision：`4cd4e8da17189eb57f9a68cc316bcdf189c084ec`；`git_dirty=false`。

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
