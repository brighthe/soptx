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
方式以及全部数值门禁的数学定义，见 [`math_spec.md`](math_spec.md)；实测数值、
证据 provenance 与解释边界见 [`results_analysis.md`](results_analysis.md)。
本 README 只负责使用说明、运行入口和文件职责。

线弹性理论入口为
`dut-postdoc:concepts/linear-elasticity.md#线弹性方程变分形式与有限元离散`；
Matrix-Free 装配层次和 MPI 重叠 DOF 理论入口分别为
`dut-postdoc:concepts/matrix-free/assembly-levels.md#五级分类` 和
`dut-postdoc:concepts/gpu-hpc/distributed-operator-and-shared-dofs.md`。

## 目录结构与核心模块

- [`operator.py`](operator.py)：核心。2D/3D EA 矩阵无关算子 (`ElasticityEAOperator`) 与 `OverlapOperator`（MPI 共享自由度同步归约）；
- [`solver.py`](solver.py)：核心。分布式加权 Krylov 求解器（`weighted_cg` / `solve_matrix_free_system`）与真残差、边界误差诊断；
- [`cases.py`](cases.py)：2D/3D 物理问题（平面应变与多项式无散场）、材料参数与制造解定义；
- [`minimal_demo.py`](minimal_demo.py)：极简入口。一键运行 2D/3D Matrix-Free CG 求解（0 繁琐 CLI），可单 rank 或 `mpiexec -n 2` 运行；
- [`compare_lagrange.py`](compare_lagrange.py)：交叉比对。与全组装 (FA) CSR 矩阵及 Scipy 直解做 $\sim 10^{-16}$ 机器精度级对照；
- [`math_spec.md`](math_spec.md)：算子代数、重叠副本表示与门禁的 1-to-1 数学定义规范；
- [`results_analysis.md`](results_analysis.md)：实验证据报告。2D/3D 实测数值、证据 provenance 与解释边界的唯一事实源，含 `utils/sync_results.py` 生成的结果区块；
- [`utils/`](utils/)：基础设施胶水包，收拢 `contract`、`schema`、`report`、`layout`、`analyzer`、`distributed`、`references`、`postprocess`、`run`、`validate` 和 `sync_results`。

## 环境与运行

使用包含 editable FEALPy、`mpi4py` 和 MPI Runtime 的环境：

```bash
conda activate ihpcm
python -m pip install -e ".[mpi,test]"
```

以下命令均以本示例目录 `examples/matrix_free_elasticity/` 为工作目录。
`utils/` 下的 `run.py`、`validate.py` 和 `sync_results.py` 是可直接执行的驱动
脚本，它们会自行把示例目录加入 `sys.path`，因此按文件路径调用即可，无需
`python -m`。

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

完整 CLI 驱动（单次运行，产物写入 `outputs/`）：

```bash
mpiexec -n 1 python utils/run.py --dim 2 --operator-level ea --p 1 --nx 8 --ny 8
mpiexec -n 1 python utils/run.py --dim 3 --operator-level ea --p 1 --nx 4 --ny 4 --nz 4
mpiexec -n 1 python utils/run.py --dim 2 --operator-level fa --p 1 --nx 8 --ny 8
```

`--dim` 默认值为 `3`，上面三条均显式写出。二维显式传入 `--nz` 会报错。阶段 1
只接受 `p=1` 和 1/2 ranks；EA 支持 1/2 ranks，FA 和 `--benchmark` 只支持单
rank。

`utils/run.py` 的调用链：

```text
main
  → parse_arguments → create_case(dim) + RunConfig
  → execute
      → build_global_context → partition_cells(split_coordinate=…)
      → distribute → distribute_mesh / distribute_vector_space / serial_references
      → build_distributed_analyzer(operator_level=…)
      → run_solver
          → analyzer.apply_bc(assemble_stiff_matrix(), assemble_body_force_vector())
          → analyzer.solve_system → DISTRIBUTED_SOLVERS["cg"]
              → solver.weighted_cg → fealpy cg(dot_product=dof_comm.dot)
          → solver_diagnostics
      → dof_comm.gather_add(local_solution / references)
      → finalize
          → solution_error / write_solution
          → report.local_gates → report.build_payload → report.write_json
```

EA 保存完整单元矩阵集合 $\{\mathbf K_e\}$，每次 MatVec 执行
gather—单元作用—scatter-add；FA 形成并保存全局 CSR。两者对应同一个离散算子，
代数细节见 [`math_spec.md` 第 2–3 节](math_spec.md)。

## 验证

先跑不需要 MPI 的单元测试（CG 分支、产物命名契约、分区逻辑）：

```bash
pytest .
```

再按维度或同时准备完整验证：

```bash
python utils/validate.py --dim 2
python utils/validate.py --dim 3
python utils/validate.py --dim all
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
[`math_spec.md` 第 5 节](math_spec.md)；阈值本身只在
[`utils/contract.py`](utils/contract.py) 定义一次。

驱动还会确认非法维数、2D 携带 `--nz`、非正网格数、非 `p=1` 以及
FA 多 rank 均以非零状态和对应的明确错误信息退出，且不生成结果产物。

验证完成后同步或只读检查对应维度证据：

```bash
python utils/sync_results.py --dim 2
python utils/sync_results.py --dim 3
python utils/sync_results.py --dim all --check
```

## 输出协议

单次运行使用 schema version 3（`utils/schema.py:SCHEMA_VERSION`），`stage` 为
`soptx/matrix-free-elasticity/stage-1`。`parameters` 明确记录：

- `dimension`、`case`、`domain` 和 `resolution`；
- `material.hypothesis` 及对应材料常数；
- 有限元次数、求解容差、算子层级、存储方式和 MPI 表示。
- `environment` 中的 UTC 时间、运行时版本、Git revision 与 dirty flag；
- 用于 EA/FA 随机向量校验的固定随机种子。

`utils/sync_results.py` 只接受 `git_dirty=false` 的 clean-revision 原始 JSON；
dirty worktree 的验证结果只能作为开发证据。

默认产物写入已忽略的 `outputs/`，文件名包含 `2d` 或 `3d`：

- `*.json`：参数、环境、残差、误差和本地门禁；
- `*.npy`：rank 0 收集的唯一全局位移向量；
- `*.vtu`：三角形或四面体单元重心位移及误差；
- `stage1-validation-{2,3,all}.json`：跨运行门禁。

## 数值结果与证据

全部数值结果、证据 provenance 和解释边界由
[`results_analysis.md`](results_analysis.md) 唯一维护，本 README 不复制
数字。其中带 `<!-- BEGIN/END GENERATED -->` 标记的区块由
`utils/sync_results.py` 生成，不要手工编辑。
