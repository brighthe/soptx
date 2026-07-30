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
[`docs/models/manufactured-elasticity.md`](../../docs/models/manufactured-elasticity.md)。
Problem 不创建网格；`ElasticityCase.create_mesh()` 显式组合 Problem、Material
和 Mesh。

> 架构迁移说明：下方已提交 evidence 来自迁移前验证，只作为历史数值基线。切换到
> `src/soptx` 和语义 Problem 后，必须重新执行验证与 evidence 同步，才能声明为
> 本次重构结果。

线弹性理论入口为
`dut-postdoc:concepts/linear-elasticity.md#线弹性方程变分形式与有限元离散`；
Matrix-Free 装配层次和 MPI 重叠 DOF 理论入口分别为
`dut-postdoc:concepts/matrix-free/assembly-levels.md#五级分类` 和
`dut-postdoc:concepts/matrix-free/distributed-operator-and-shared-dofs.md`。

## 模块与调用链

- `run.py`：CLI、case 选择和 MPI 调度；
- `contract.py`：支持范围、默认值、数值门禁和运行配置；
- `layout.py`：验证 case、产物路径、文件名和证据区块标记；
- `cases.py`：PDE、材料、网格、几何维数和输出元数据；
- `operators.py`：材料实例、载荷、EA 单元缓存和串行 FA CSR；
- `solve.py`：EA/FA 共用 Dirichlet 后系统与 CG 调用；
- `distributed.py`、`cg.py`：重叠副本通信和加权 CG；
- `references.py`：EA/FA MatVec、对称性和 `spsolve` 独立参考；
- `postprocess.py`、`report.py`：误差与 VTK 后处理、结果 schema 和 JSON 报告；
- `validate.py`：2D/3D 单 rank、双 rank、FA/EA 与收敛阶门禁；
- `sync_results.py`：按维度同步验证证据和下方结果区块；
- `tests/`：CG、路径契约和 2D/3D 分区逻辑的快速单元测试。

`contract.py` 和 `layout.py` 不依赖 FEALPy、SOPTX 或 mpi4py，因此
`sync_results.py` 在没有 MPI Runtime 的机器上也能做证据检查。每个数值阈值只在
`contract.py` 定义一次，`run.py` 的运行门禁与 `validate.py` 的复核门禁读同一组
常量；每个产物文件名只由 `layout.py` 生成一次，`validate.py` 写出端与
`sync_results.py` 读入端不再各持一份字面量。

```text
main
  → parse_arguments → create_case(dim) + RunConfig
  → execute
      → partition_cells(split_coordinate=case.partition_split_coordinate())
      → distribute_mesh / distribute_vector_space
      → prepare_problem(operator_level=…) → EA 或 FA 装配
      → solve_prepared_problem → solve_cg
      → gather_add
      → solution_error / write_solution
  → report.local_gates → report.build_payload → report.write_json
```

EA 保存完整单元矩阵集合 $\{\mathbf K_e\}$，每次 MatVec 执行
gather—单元作用—scatter-add；FA 形成并保存全局 CSR。两者对应同一个离散算子：

$$
\mathbf K=\sum_e\mathbf R_e^{\mathsf T}\mathbf K_e\mathbf R_e.
$$

## 环境与运行

从 SOPTX 仓库根目录执行，使用包含 editable FEALPy、`mpi4py` 和 MPI Runtime
的环境，例如：

```powershell
conda activate xihe-fealpy
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

重构前的三维数值证据保存在
[`evidence/cpu-single-rank-fa-ea-3d-historical.json`](evidence/cpu-single-rank-fa-ea-3d-historical.json)，
只作为原三维实现的历史基线，不作为本次 2D/3D 通用化实现的验收结论。
以下结果区块必须在新实现完成对应维度验证后再生成。

### 2D CPU 单 rank FA/EA

<!-- BEGIN GENERATED: cpu-single-rank-fa-ea-2d -->

本节由 `sync_results.py --dim 2` 根据 clean-revision 原始 JSON 生成；精简证据见 `evidence/cpu-single-rank-fa-ea-2d.json`。
源 revision：`25226611f174041b85610c652cdd7c4d44e2f9ea`；`git_dirty=false`。

| 网格 | EA-CG 迭代数 | 真相对残差 | 相对 L2 误差 | 边界绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| `8×8` | 38 | `4.95485e-11` | `4.61057e-02` | `0` |
| `16×16` | 89 | `8.99308e-11` | `1.20318e-02` | `0` |
| `32×32` | 188 | `8.95197e-11` | `3.04605e-03` | `0` |

| 网格 | 原始 EA/FA MatVec | Dirichlet EA/FA MatVec | EA-CG/FA 直接解 |
| --- | ---: | ---: | ---: |
| `8×8` | `1.44949e-16` | `1.40930e-16` | `8.88290e-12` |
| `16×16` | `1.62572e-16` | `1.64234e-16` | `6.67864e-12` |
| `32×32` | `1.57536e-16` | `1.56086e-16` | `3.02704e-12` |

相对 L2 误差观测阶为 `1.93809`、`1.98184`。独立 FA 粗网格 `8×8` 在 38 步收敛，真相对残差为 `4.97442e-11`。

<!-- END GENERATED: cpu-single-rank-fa-ea-2d -->

### 3D CPU 单 rank FA/EA

<!-- BEGIN GENERATED: cpu-single-rank-fa-ea-3d -->

尚未生成迁移后 clean-revision 正式证据。

<!-- END GENERATED: cpu-single-rank-fa-ea-3d -->
