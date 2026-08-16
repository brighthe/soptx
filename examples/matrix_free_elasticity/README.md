# 线弹性 Matrix-Free 2D/3D 基线

一阶连续 Lagrange 向量元的 EA/EbE Matrix-Free 线弹性基线，二维、三维共用同一套
算例与门禁。EA 是主路径，FA（显式全局 CSR）作为同离散条件下的黄金参考。

支持范围明确限定为二维和三维，不宣称支持任意空间维数。

## 推进阶段

装配层级固定在 **EA**（FEALPy 当前提供的 matrix-free 就是 EA），沿执行后端推进：

| 阶段 | 范围 | 状态 |
|---|---|---|
| **1a** | CPU 串行 EA/FA | **当前默认范围**，本文档主线 |
| 1b | CPU 并行 EA（MPI 重叠副本） | 代码已就绪，见[下文](#阶段-1b可选的-cpu-并行路径) |
| 1c | 单 GPU EA | 未开始 |

1a 与 1b 共享同一份制造解定义（[`soptx.problems.elasticity`](../../src/soptx/problems/elasticity.py)）和
同一份阈值（[`tools/matrix_free_evidence/contract.py`](../../tools/matrix_free_evidence/contract.py)），因此 1b 的跨 rank 门禁与 1a 的串行
结果始终可比。

## 算例配置

求解核心不根据 PDE 类名判断维度：2D 默认为平面应变正弦制造解与三角形网格，3D 默认为
多项式无散场制造解与四面体网格；EA 正确性验证入口还支持 2D 四边形与 3D 六面体网格。
制造解的完整方程、参数与 shape 契约见
[`docs/problems/manufactured-elasticity.md`](../../docs/problems/manufactured-elasticity.md)。

**凡是能从已有对象问出来的，一律问它，不另存一份。** 区域取
`problem.domain`，弹性常数取 `problem.lam` / `problem.mu`，维数取
`problem.dimension`，网格实体名取 `mesh.Entity("cell").schema.name`，算例名取
`type(problem).__name__`——制造解的解析式与刚度算子必须建立在同一组参数上，各存
一份迟早会漂，而这种漂移 EA/FA 对拍是发现不了的（两边用的是同一个错误常数），
只有 L2 收敛阶门禁才拦得住。

真正由维数决定、又问不出来的是制造解、网格类型和本构假设。
[`verify_ea_correctness.py`](verify_ea_correctness.py) 在模块顶部显式登记全部已验证的
模型—网格组合：

```python
PROBLEM_FACTORIES = {2: {"sinusoidal": ..., "exponential": ...}, 3: {"polynomial": ...}}
MESH_FACTORIES = {2: {"tri": TriangleMesh, "quad": QuadrangleMesh}, 3: {"tet": TetrahedronMesh, "hex": HexahedronMesh}}
MATERIAL_HYPOTHESES = {2: "plane_strain", 3: "3D"}
```

平面降维假设（`plane_strain`/`3D`）属于本构而非制造解，Problem 上没有这个概念，
所以只能定在驱动脚本里。Problem 只提供制造解、体力和边界数据，不创建网格；材料由
驱动脚本显式实例化为 `IsotropicLinearElasticMaterial` 后传入算子，算子构造不从 PDE
对象隐式读取材料属性。

## 文件职责

- [`verify_ea_correctness.py`](verify_ea_correctness.py)：多-rank EA 正确性验证，检查 EA/FA 算子、EA-CG/FA 直接解与并行/串行 EA 解一致性，并可检验单 Rank L2 收敛阶；
- [`benchmark_cpu_ea.py`](benchmark_cpu_ea.py)：单 Rank CPU EA/FA 效率对照，只切换算子层级，记录刚度算子构造、裸 MatVec、CG 和算子长期保存数组字节数；
- [`compare_lagrange.py`](compare_lagrange.py)：交叉比对，与 FA 的 CSR 矩阵及 Scipy 直解做机器精度级对照；
- [`results_analysis.md`](results_analysis.md)：符号—代码映射契约、实测数值、证据 provenance 与证据边界的唯一事实源；

本目录有三个可执行入口与两份文档。制造解已下沉到
[`soptx.problems.elasticity`](../../src/soptx/problems/elasticity.py)（见上节）。证据流水线（`run`、`validate`、`sync_results`
及其 `contract`/`layout`/`schema`/`report`）住在
[`tools/matrix_free_evidence/`](../../tools/matrix_free_evidence/)，因为它同时是
fealpy fork 的 merge 前门禁，不只服务于这一个示例。依赖方向是单向的：那个包不导入
本目录的任何模块，反过来是正确性入口导入它的 `contract`，好让印出的 PASS/FAIL
与正式门禁用同一批阈值。它们的测试在 [`tests/unit/`](../../tests/unit/)，文件名
以 `test_matrix_free_` 开头。

## 实现住在哪里

**Matrix-Free 的实现不在本目录，在 `soptx` 包里**；本目录只是驱动它的 demo 与证据工具：

| 模块 | 内容 |
|---|---|
| [`soptx.problems.elasticity`](../../src/soptx/problems/elasticity.py) | `SinusoidalPlaneStrainElasticity2D` / `DivergenceFreePolynomialElasticity3D`：制造解的单一定义源，自带区域、弹性常数与维数 |
| [`soptx.fem.solvers.matrix_free_solver`](../../src/soptx/fem/solvers/matrix_free_solver.py) | `weighted_cg` / `solve_matrix_free_system` / `PreparedLinearSystem` 与真残差、边界误差诊断。当前无预条件 |
| [`soptx.fem.solvers.matrix_free_analyzer`](../../src/soptx/fem/solvers/matrix_free_analyzer.py) | `DistributedElasticityAnalyzer`（`LagrangeFEMAnalyzer` 的重叠副本子类）与 `DISTRIBUTED_SOLVERS` 登记表 |
| [`soptx.fem.distributed`](../../src/soptx/fem/distributed.py) | `OverlapOperator`（MPI 共享自由度同步归约）、单元分区与向量空间分发 |
| [`soptx.fem.solvers.elasticity_operator`](../../src/soptx/fem/solvers/elasticity_operator.py) | `build_serial_analyzer` / `build_distributed_analyzer` 与 demo 用的 EA 懒装配门面 `ElasticityEAOperator`；只接受 `(space, pde, material)` |
| [`soptx.fem.verification`](../../src/soptx/fem/verification.py) | `serial_references`（FA 黄金参考与 Scipy 直解）、`solution_error`、`relative_difference` |
| [`soptx.numerics`](../../src/soptx/numerics.py) | 求解器默认容差与 `NORM_FLOOR`，由 [`tools/matrix_free_evidence/contract.py`](../../tools/matrix_free_evidence/contract.py) 复出口 |

后两者依赖可选的 `mpi4py`，因此**不**从 `soptx.fem` 的包 `__init__` 导出，需按完整模块路径导入。

**本目录只保留 `README.md` 与 `results_analysis.md` 两份 markdown。** 完整的算子
代数推导、边界条件施加形式与门禁数学式一律由知识库维护，本仓库只保留符号—代码
映射、阈值与实测证据；入口清单见
[`results_analysis.md` 第 1 节](results_analysis.md#1-数学代码映射契约)。

## 环境与运行

本实现与 MPI 厂商实现无关：Open MPI、MPICH 和 Intel MPI 均可使用。关键约束是
运行脚本的 ``python``、其导入的 ``mpi4py`` 与 ``mpiexec`` 必须来自同一个 MPI
运行时；不要把系统启动器与虚拟环境中的 Python 混用。当前项目以 Conda 环境为例：

```bash
conda activate ihpcm
python -m pip install -e ".[mpi,test]"
```

并行前可用下列命令确认 ``mpi4py`` 链接的 MPI 运行时，并始终使用该环境的启动器：

```bash
python -c 'from mpi4py import MPI; print(MPI.Get_library_version())'
$CONDA_PREFIX/bin/mpiexec --version
```

以下命令均以**仓库根目录**为工作目录，按文件路径直接调用即可，无需 `python -m`：
`soptx` 是 editable install，`import soptx` 无需任何 `sys.path` 处理；需要导入
`tools.matrix_free_evidence.contract` 的脚本（`compare_lagrange.py` 与
`tools/` 下的四个脚本）会自行把仓库根加入 `sys.path`。

极简 Demo：

```bash
# 默认二次加密, 验证 n、2n、4n 三档网格的 L2 收敛阶
python examples/matrix_free_elasticity/verify_ea_correctness.py --n 8
python examples/matrix_free_elasticity/verify_ea_correctness.py --model exponential --n 8
python examples/matrix_free_elasticity/verify_ea_correctness.py --model exponential --mesh-type quad --n 8
python examples/matrix_free_elasticity/verify_ea_correctness.py --model polynomial --n 4
python examples/matrix_free_elasticity/verify_ea_correctness.py --model polynomial --mesh-type hex --n 2
# 三次加密, 验证 n、2n、4n、8n 四档网格
python examples/matrix_free_elasticity/verify_ea_correctness.py --model exponential --n 8 --refinements 3
```

运行方式说明：

- `verify_ea_correctness.py` 支持单 Rank 与任意非空条带分区的多 rank MPI EA 正确性验证；每次都检查收敛阶，不承担性能测试或证据产物生成。
- `benchmark_cpu_ea.py` 默认模式为单 Rank CPU EA/FA 对照，固定 NumPy 后端和 CG 参数，逐档记录刚度构造、裸 MatVec、带边界条件的 CG 求解时间及算子长期保存数组字节数；性能测试前必须先通过正确性验证。
- `benchmark_cpu_ea.py --mode mpi-ea-strong` 与 `--mode mpi-ea-weak` 测量 CPU MPI 下 EA 的算子构造、完整系统 MatVec、CG 总时间、每 CG 步时间及其构造加 CG 的流水线时间，均采用所有 rank 中最大 wall time；完整系统 MatVec 还分解 `OverlapOperator` 的输入同步、本地单元核和输出同步。分区器沿 x 方向生成连续条带，支持任意非空正整数 rank 数。
- `--model` 用于选择制造解，并自动确定维度：`sinusoidal`（默认）和 `exponential` 是 2D，`polynomial` 是 3D。只有未指定模型时，才可用 `--dim` 选择该维度的默认模型。
- `--mesh-type` 用于选择网格类型：2D 可选 `tri`（默认，三角形）或 `quad`（四边形），3D 可选 `tet`（默认，四面体）或 `hex`（六面体）。
- 直接运行或传入 `--serial` 时执行单 Rank EA/FA/Scipy 对拍；`--serial` 与 `mpiexec -n 2` 互斥。
- `mpiexec -n 2` 时执行两 rank EA，并在 Root 上检查它与串行 EA 及 FA/Scipy 直接解的一致性。
- `--n` 是最粗网格分辨率，`--refinements` 是连续二倍加密次数，默认值为 `2`，即计算 `n`、`2n`、`4n` 三档网格。该参数至少为 `2`；脚本按 $p=\log_2(E_h/E_{h/2})$ 输出相邻档相对 L2 误差的观测收敛阶。末段阶须不低于与正式门禁共享的 `MINIMUM_FINAL_L2_ORDER`，否则整体正确性结论为 FAIL；正式验收仍应运行 `tools/matrix_free_evidence/validate.py`。

交叉比对，只看两件事——矩阵结果是否一致（裸 MatVec 与施加 Dirichlet 后 MatVec，
应到机器精度）、求解结果是否一致（EA-CG 与 FA Scipy 直解，量级由 CG 停机准则决定）：

```bash
python examples/matrix_free_elasticity/compare_lagrange.py --dim 2
python examples/matrix_free_elasticity/compare_lagrange.py --dim 3
```

CPU EA/FA 效率对照。默认模型为 2D 正弦制造解与三角形网格；时间以 wall time
中位数报告，``FA/EA > 1`` 表示 EA 的相应时间或保存存储更小：

```bash
python examples/matrix_free_elasticity/benchmark_cpu_ea.py --n 16 --levels 3
python examples/matrix_free_elasticity/benchmark_cpu_ea.py \
  --model exponential --mesh-type quad --n 16 --levels 3
python examples/matrix_free_elasticity/benchmark_cpu_ea.py \
  --model polynomial --mesh-type hex --n 2 --levels 3 \
  --output examples/matrix_free_elasticity/outputs/cpu_ea_hex.json
```

该脚本的存储数字仅统计算子为重复 MatVec 长期保存的数组：FA 为 COO 数值与行/列
索引，EA 为单元矩阵与单元到全局 DOF 映射。它**不是**进程峰值内存，不能代替内存
分析工具或用于声称实际峰值内存降低。

CPU MPI EA 初步强/弱扩展。必须以同一个 MPI 环境中的启动器运行；强扩展固定物理
区域与全局网格。弱扩展按 rank 数沿 x 方向同时扩大物理区域和网格数，例如 2D 的
1-rank ``[0, 1] x [0, 1]``、``n x n`` 与 2-rank ``[0, 2] x [0, 1]``、``2n x n``
具有相同单元尺寸和每 rank 单元数。无预条件 CG 的迭代次数仍可能因全局问题变大而
增加，所以端到端求解时间不能单独解释为理想弱扩展效率：

```bash
mpiexec -n 1 python examples/matrix_free_elasticity/benchmark_cpu_ea.py \
  --mode mpi-ea-strong --n 128 --warmup 2 --repeats 5 \
  --output examples/matrix_free_elasticity/outputs/ea_strong_1.json
mpiexec -n 2 python examples/matrix_free_elasticity/benchmark_cpu_ea.py \
  --mode mpi-ea-strong --n 128 --warmup 2 --repeats 5 \
  --output examples/matrix_free_elasticity/outputs/ea_strong_2.json
mpiexec -n 4 python examples/matrix_free_elasticity/benchmark_cpu_ea.py \
  --mode mpi-ea-strong --n 128 --warmup 2 --repeats 5 \
  --output examples/matrix_free_elasticity/outputs/ea_strong_4.json
mpiexec -n 8 python examples/matrix_free_elasticity/benchmark_cpu_ea.py \
  --mode mpi-ea-strong --n 128 --warmup 2 --repeats 5 \
  --output examples/matrix_free_elasticity/outputs/ea_strong_8.json
mpiexec -n 1 python examples/matrix_free_elasticity/benchmark_cpu_ea.py \
  --mode mpi-ea-weak --n 128 --warmup 2 --repeats 5 \
  --output examples/matrix_free_elasticity/outputs/ea_weak_1.json
mpiexec -n 2 python examples/matrix_free_elasticity/benchmark_cpu_ea.py \
  --mode mpi-ea-weak --n 128 --warmup 2 --repeats 5 \
  --output examples/matrix_free_elasticity/outputs/ea_weak_2.json
```

完整 CLI 驱动（单次运行，产物写入本目录的 `outputs/`）：

```bash
python tools/matrix_free_evidence/run.py --dim 2 --operator-level ea --p 1 --nx 8 --ny 8
python tools/matrix_free_evidence/run.py --dim 3 --operator-level ea --p 1 --nx 4 --ny 4 --nz 4
python tools/matrix_free_evidence/run.py --dim 2 --operator-level fa --p 1 --nx 8 --ny 8
```

`--dim` 默认值为 `3`，上面三条均显式写出；二维显式传入 `--nz` 会报错。阶段 1 只
接受 `p=1`。其余参数（`--maxit`、`--rtol`、`--atol`、`--output`、`--summary`）见
`python tools/matrix_free_evidence/run.py --help`。

`run.py` 的调用链：

```text
main
  → parse_arguments → PROBLEM_FACTORIES[dim]() + RunConfig
  → execute
      → build_global_context → partition_cells(split_coordinate=…)
      → distribute → distribute_mesh / distribute_vector_space / serial_references
      → build_distributed_analyzer(operator_level=…)
      → run_solver
          → analyzer.apply_bc(assemble_stiff_matrix(), assemble_body_force_vector())
          → analyzer.solve_system → DISTRIBUTED_SOLVERS["cg"]
              → matrix_free_solver.weighted_cg
                  → fealpy cg(dot_product=dof_comm.dot)
          → solver_diagnostics
      → dof_comm.gather_add(local_solution / references)
      → finalize
          → solution_error / write_solution
          → report.local_gates → report.build_payload → report.write_json
```

单 rank 下所有跨 rank 归约都退化为恒等（$r_i\equiv 1$ 且 $\mathcal S$ 是恒等），
因此串行路径与并行路径走的是同一段代码。

EA 保存完整单元矩阵集合 $\{\mathbf K_e\}$，每次 MatVec 执行
gather—单元作用—scatter-add；FA 形成并保存全局 CSR。两者对应同一个离散算子，
保存／省略对象见 [`results_analysis.md` 第 1.2 节](results_analysis.md)，代数
细节见 `dut-postdoc:concepts/matrix-free/assembly-levels.md#五级分类`。

## 验证（阶段 1a）

先跑单元测试（分区与分布式分析器两个模块需要 `mpi4py`，装不上会自动跳过）：

```bash
python -m pytest tests -q -k matrix_free
```

再按维度或同时运行完整验证：

```bash
python tools/matrix_free_evidence/validate.py --dim 2
python tools/matrix_free_evidence/validate.py --dim 3
python tools/matrix_free_evidence/validate.py --dim all
```

每个维度运行三档单 rank EA 网格加一档单 rank FA 参照。门禁覆盖 CG 收敛与真残差、
Dirichlet 自由度误差、EA/FA 的原始与 Dirichlet 后 MatVec 一致性、算子正定、
CG 与 FA 直解一致、EA/FA 解一致，以及 L2 误差单调下降与最末段观测阶。

**门禁与阈值常量的对照表见 [`results_analysis.md` 第 1.3 节](results_analysis.md)，
精确数学式见 `dut-postdoc:concepts/matrix-free/assembly-levels.md#跨层级正确性判据`；
阈值本身只在 [`tools/matrix_free_evidence/contract.py`](../../tools/matrix_free_evidence/contract.py) 定义一次，本文档不复制数字。**

驱动还会确认非法维数、2D 携带 `--nz`、非正网格数、非 `p=1` 以及 FA 多 rank 均以
非零状态和明确错误信息退出，且不生成结果产物。

验证通过后同步或只读检查证据：

```bash
python tools/matrix_free_evidence/sync_results.py --dim all
python tools/matrix_free_evidence/sync_results.py --dim all --check
```

## 阶段 1b：可选的 CPU 并行路径

MPI 重叠副本实现已经就绪，但**不在 1a 的验证范围内**，需显式打开：

```bash
python tools/matrix_free_evidence/validate.py --dim all --include-parallel
$CONDA_PREFIX/bin/mpiexec -n 2 python examples/matrix_free_elasticity/verify_ea_correctness.py --dim 2
$CONDA_PREFIX/bin/mpiexec -n 4 python examples/matrix_free_elasticity/verify_ea_correctness.py --dim 2
$CONDA_PREFIX/bin/mpiexec -n 8 python examples/matrix_free_elasticity/verify_ea_correctness.py --dim 2
$CONDA_PREFIX/bin/mpiexec -n 2 python tools/matrix_free_evidence/run.py --dim 2 --operator-level ea --p 1 --nx 8 --ny 8
```

必须使用与 `mpi4py` 链接到同一 MPI 运行时的启动器。这里的
`$CONDA_PREFIX/bin/mpiexec` 只是 Conda 环境中的通用写法，不意味着要求 Intel MPI
或 Open MPI。若使用系统 MPI 启动器而其 ABI 与 `mpi4py` 链接的运行时不同，两个
进程会各自退化成单 Rank 串行运行；验证脚本会检测这一情形并以错误退出。

`--include-parallel` 的正式证据流程仍只追加一档 2-rank EA 算例，并启用 1/2-rank
解一致与 L2 误差一致两道跨 rank 门禁。`verify_ea_correctness.py` 与
`benchmark_cpu_ea.py` 的条带分区则支持任意非空正整数 ranks；4/8-rank 的人工复现
结果见 [`results_analysis.md`](results_analysis.md)。FA 和 `tools/matrix_free_evidence/run.py`
中的 FA 路径只支持单 rank——对称消元发生在全局矩阵装配之后，多 rank 下没有插入同步归约的位置（见
`dut-postdoc:concepts/matrix-free/assembly-levels.md#并行下-fa-的对称消元不成立`）。

其代数基础（一致/加和表示、同步归约 $\mathcal S$ 与幂等投影 $\mathcal C$、加权
内积）见 `dut-postdoc:concepts/gpu-hpc/distributed-matrix-free-computing.md#3-共享自由度与重叠副本代数`。
**1a 的证据不包含任何跨 rank 结论**，2 ranks 即便跑通也只验证正确性，不支持任何
扩展性表述。

## 输出与证据

单次运行产物写入已忽略的 `outputs/`，字段与版本的权威定义在
[`tools/matrix_free_evidence/schema.py`](../../tools/matrix_free_evidence/schema.py)，
产物命名契约在
[`tools/matrix_free_evidence/layout.py`](../../tools/matrix_free_evidence/layout.py)。

`sync_results.py` 只接受 `git_dirty=false` 的 clean-revision 原始 JSON；
dirty worktree 的验证结果只能作为开发证据（仓库级政策见
[`docs/validation/evidence-policy.md`](../../docs/validation/evidence-policy.md)）。
它把精简证据写到本目录的 `evidence/`（目录按需创建）。

全部实测数值、证据 provenance 和证据边界由
[`results_analysis.md`](results_analysis.md) 唯一维护，本文档不复制数字。

## 本阶段明确不承诺的内容

本节是**实现层面**的能力边界，是本目录的唯一事实源；这批数字能支持什么结论则由
[`results_analysis.md` 第 3 节](results_analysis.md)维护，二者不互相复制。

- 不实现 PA/QA、UA/NONE，不宣称任何低于 EA 的存储层级；
- 无预条件（`parameters.preconditioner` 恒为 `null`），因此迭代数只反映施加边界
  条件后算子的条件数，不构成任何预条件结论；
- 只支持 $p=1$、$d\in\{2,3\}$ 与沿单一坐标轴的连续条带分区；正式证据流程的跨
  rank 门禁当前仍固定为 2 ranks，4/8-rank 仅有当前工作区的人工复现与性能测量；
- MatVec 一致不替代完整 solve、真残差与解误差；单 kernel 计时不替代端到端时间
  与峰值内存。

**算术强度口径。** EA 的 apply 每单元读取 $m^2$ 个 double 并执行约 $2m^2$ 次
浮点运算，算术强度与 FA 的 SpMV 同量级；EA 改善的是访存的规则性而非总量。因此
1b 的并行加速与 1c 的 GPU 加速，其收益来源都是**并行度与访存规则性**，不得表述
为「Matrix-Free 通过提高算术强度获得加速」——那要到 PA/QA 与 UA/NONE 才成立，
而本阶段明确不实现它们。完整推导与数值见
`dut-postdoc:concepts/matrix-free/assembly-levels.md#算术强度：EA 并没有解决 FA 的瓶颈`。
