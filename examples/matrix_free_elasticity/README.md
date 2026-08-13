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

1a 与 1b 共享同一份算例定义（[`cases.py`](cases.py)）和同一份阈值
（[`utils/contract.py`](utils/contract.py)），因此 1b 的跨 rank 门禁与 1a 的串行
结果始终可比。

## 算例配置

维度相关内容集中在 [`cases.py`](cases.py)，求解核心不根据 PDE 类名判断维度：
2D 为平面应变正弦制造解（`TriangleMesh`），3D 为多项式无散场制造解
（`TetrahedronMesh`）。域、材料常数与网格类型的权威定义在
`cases.py:create_case()`，制造解的完整方程、参数与 shape 契约见
[`docs/problems/manufactured-elasticity.md`](../../docs/problems/manufactured-elasticity.md)。

Problem 只提供制造解、体力和边界数据，不创建网格；材料参数由独立 `MaterialSpec`
提供，算子构造不从 PDE 对象隐式读取材料属性；`ElasticityCase.create_mesh()`
显式组合 Problem、Material 和 Mesh。

## 文件职责

- [`cases.py`](cases.py)：2D/3D 物理问题、材料参数与制造解算例的单一定义源；
- [`minimal_demo.py`](minimal_demo.py)：极简入口，一键运行 2D/3D Matrix-Free CG 求解；
- [`compare_lagrange.py`](compare_lagrange.py)：交叉比对，与 FA 的 CSR 矩阵及 Scipy 直解做机器精度级对照；
- [`results_analysis.md`](results_analysis.md)：符号—代码映射契约、实测数值、证据 provenance 与证据边界的唯一事实源；
- [`utils/`](utils/)：本例专属的胶水与证据工具，收拢 `contract`、`schema`、`report`、`layout`、`analyzer`、`references`、`postprocess`、`run`、`validate` 和 `sync_results`。其中 [`utils/analyzer.py`](utils/analyzer.py) 只负责把 [`cases.py`](cases.py) 的算例拆成分析器构造参数，外加 demo 用的 EA 懒装配门面 `ElasticityEAOperator`。

## 实现住在哪里

**Matrix-Free 的实现不在本目录，在 `soptx` 包里**；本目录只是驱动它的 demo 与证据工具：

| 模块 | 内容 |
|---|---|
| [`soptx.fem.solvers.matrix_free_solver`](../../src/soptx/fem/solvers/matrix_free_solver.py) | `weighted_cg` / `solve_matrix_free_system` / `PreparedLinearSystem` 与真残差、边界误差诊断。当前无预条件 |
| [`soptx.fem.solvers.matrix_free_analyzer`](../../src/soptx/fem/solvers/matrix_free_analyzer.py) | `DistributedElasticityAnalyzer`（`LagrangeFEMAnalyzer` 的重叠副本子类）与 `DISTRIBUTED_SOLVERS` 登记表 |
| [`soptx.fem.distributed`](../../src/soptx/fem/distributed.py) | `OverlapOperator`（MPI 共享自由度同步归约）、单元分区与向量空间分发 |
| [`soptx.numerics`](../../src/soptx/numerics.py) | 求解器默认容差与 `NORM_FLOOR`，由 [`utils/contract.py`](utils/contract.py) 复出口 |

后两者依赖可选的 `mpi4py`，因此**不**从 `soptx.fem` 的包 `__init__` 导出，需按完整模块路径导入。

**本目录只保留 `README.md` 与 `results_analysis.md` 两份 markdown。** 完整的算子
代数推导、边界条件施加形式与门禁数学式一律由知识库维护，本仓库只保留符号—代码
映射、阈值与实测证据；入口清单见
[`results_analysis.md` 第 1 节](results_analysis.md#1-数学代码映射契约)。

## 环境与运行

```bash
conda activate ihpcm
python -m pip install -e ".[mpi,test]"
```

以下命令均以本示例目录 `examples/matrix_free_elasticity/` 为工作目录。`utils/` 下的
`run.py`、`validate.py` 和 `sync_results.py` 会自行把示例目录加入 `sys.path`，按
文件路径直接调用即可，无需 `python -m`。

极简 Demo：

```bash
python minimal_demo.py --dim 2
python minimal_demo.py --dim 3
```

机器精度级交叉比对（Raw MatVec、BC MatVec、算子对称性与 CG/Scipy 直解）：

```bash
python compare_lagrange.py --dim 2
python compare_lagrange.py --dim 3
```

完整 CLI 驱动（单次运行，产物写入 `outputs/`）：

```bash
python utils/run.py --dim 2 --operator-level ea --p 1 --nx 8 --ny 8
python utils/run.py --dim 3 --operator-level ea --p 1 --nx 4 --ny 4 --nz 4
python utils/run.py --dim 2 --operator-level fa --p 1 --nx 8 --ny 8
```

`--dim` 默认值为 `3`，上面三条均显式写出；二维显式传入 `--nz` 会报错。阶段 1 只
接受 `p=1`。其余参数（`--maxit`、`--rtol`、`--atol`、`--output`、`--summary`）见
`python utils/run.py --help`。

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

先跑不需要 MPI 的单元测试：

```bash
pytest .
```

再按维度或同时运行完整验证：

```bash
python utils/validate.py --dim 2
python utils/validate.py --dim 3
python utils/validate.py --dim all
```

每个维度运行三档单 rank EA 网格加一档单 rank FA 参照。门禁覆盖 CG 收敛与真残差、
Dirichlet 自由度误差、EA/FA 的原始与 Dirichlet 后 MatVec 一致性、算子对称正定、
CG 与 FA 直解一致、EA/FA 解一致，以及 L2 误差单调下降与最末段观测阶。

**门禁与阈值常量的对照表见 [`results_analysis.md` 第 1.3 节](results_analysis.md)，
精确数学式见 `dut-postdoc:concepts/matrix-free/assembly-levels.md#跨层级正确性判据`；
阈值本身只在 [`utils/contract.py`](utils/contract.py) 定义一次，本文档不复制数字。**

驱动还会确认非法维数、2D 携带 `--nz`、非正网格数、非 `p=1` 以及 FA 多 rank 均以
非零状态和明确错误信息退出，且不生成结果产物。

验证通过后同步或只读检查证据：

```bash
python utils/sync_results.py --dim all
python utils/sync_results.py --dim all --check
```

## 阶段 1b：可选的 CPU 并行路径

MPI 重叠副本实现已经就绪，但**不在 1a 的验证范围内**，需显式打开：

```bash
python utils/validate.py --dim all --include-parallel
mpiexec -n 2 python minimal_demo.py --dim 2
mpiexec -n 2 python utils/run.py --dim 2 --operator-level ea --p 1 --nx 8 --ny 8
```

`--include-parallel` 追加一档 2-rank EA 算例，并启用 1/2-rank 解一致与 L2 误差
一致两道跨 rank 门禁。EA 支持 1/2 ranks；FA 和 `--benchmark` 只支持单 rank——
对称消元发生在全局矩阵装配之后，多 rank 下没有插入同步归约的位置（见
`dut-postdoc:concepts/matrix-free/assembly-levels.md#并行下-fa-的对称消元不成立`）。

其代数基础（一致/加和表示、同步归约 $\mathcal S$ 与幂等投影 $\mathcal C$、加权
内积）见 `dut-postdoc:concepts/gpu-hpc/distributed-operator-and-shared-dofs.md`。
**1a 的证据不包含任何跨 rank 结论**，2 ranks 即便跑通也只验证正确性，不支持任何
扩展性表述。

## 输出与证据

单次运行产物写入已忽略的 `outputs/`，字段与版本的权威定义在
[`utils/schema.py`](utils/schema.py)，产物命名契约在
[`utils/layout.py`](utils/layout.py)。

`utils/sync_results.py` 只接受 `git_dirty=false` 的 clean-revision 原始 JSON；
dirty worktree 的验证结果只能作为开发证据（仓库级政策见
[`docs/validation/evidence-policy.md`](../../docs/validation/evidence-policy.md)）。

全部实测数值、证据 provenance 和证据边界由
[`results_analysis.md`](results_analysis.md) 唯一维护，本文档不复制数字。其中带
`<!-- BEGIN/END GENERATED -->` 标记的区块由 `utils/sync_results.py` 生成，不要
手工编辑。

## 本阶段明确不承诺的内容

本节是**实现层面**的能力边界，是本目录的唯一事实源；这批数字能支持什么结论则由
[`results_analysis.md` 第 6 节](results_analysis.md)维护，二者不互相复制。

- 不实现 PA/QA、UA/NONE，不宣称任何低于 EA 的存储层级；
- 无预条件（`parameters.preconditioner` 恒为 `null`），因此迭代数只反映施加边界
  条件后算子的条件数，不构成任何预条件结论；
- 只支持 $p=1$、$d\in\{2,3\}$、1/2 ranks；阶段 1a 只跑单 rank，2 ranks 属阶段
  1b 且只验证正确性，不支持任何扩展性结论；
- MatVec 一致不替代完整 solve、真残差与解误差；单 kernel 计时不替代端到端时间
  与峰值内存。

**算术强度口径。** EA 的 apply 每单元读取 $m^2$ 个 double 并执行约 $2m^2$ 次
浮点运算，算术强度与 FA 的 SpMV 同量级；EA 改善的是访存的规则性而非总量。因此
1b 的并行加速与 1c 的 GPU 加速，其收益来源都是**并行度与访存规则性**，不得表述
为「Matrix-Free 通过提高算术强度获得加速」——那要到 PA/QA 与 UA/NONE 才成立，
而本阶段明确不实现它们。完整推导与数值见
`dut-postdoc:concepts/matrix-free/assembly-levels.md#算术强度：EA 并没有解决 FA 的瓶颈`。
