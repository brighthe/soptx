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

- [`solver.py`](solver.py)：核心。`weighted_cg` / `solve_matrix_free_system` 与真残差、边界误差诊断。当前无预条件；
- [`cases.py`](cases.py)：2D/3D 物理问题、材料参数与制造解算例的单一定义源；
- [`minimal_demo.py`](minimal_demo.py)：极简入口，一键运行 2D/3D Matrix-Free CG 求解；
- [`compare_lagrange.py`](compare_lagrange.py)：交叉比对，与 FA 的 CSR 矩阵及 Scipy 直解做机器精度级对照；
- [`math_spec.md`](math_spec.md)：符号—代码映射、算子代数、门禁数学式与能力边界；
- [`results_analysis.md`](results_analysis.md)：实测数值、证据 provenance 与证据边界的唯一事实源；
- [`utils/`](utils/)：基础设施胶水包，收拢 `contract`、`schema`、`report`、`layout`、`analyzer`、`distributed`、`references`、`postprocess`、`run`、`validate` 和 `sync_results`。其中 [`utils/analyzer.py`](utils/analyzer.py) 持有分析器构造与 EA 缓存门面 `ElasticityEAOperator`，[`utils/distributed.py`](utils/distributed.py) 持有 1b 用的 `OverlapOperator`（MPI 共享自由度同步归约）。

理论入口在知识库侧：线弹性变分形式与离散见
`dut-postdoc:concepts/linear-elasticity.md#线弹性方程变分形式与有限元离散`，
装配层次见 `dut-postdoc:concepts/matrix-free/assembly-levels.md#五级分类`，
MPI 重叠 DOF 见 `dut-postdoc:concepts/gpu-hpc/distributed-operator-and-shared-dofs.md`。

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
              → solver.weighted_cg → fealpy cg(dot_product=dof_comm.dot)
          → solver_diagnostics
      → dof_comm.gather_add(local_solution / references)
      → finalize
          → solution_error / write_solution
          → report.local_gates → report.build_payload → report.write_json
```

单 rank 下所有跨 rank 归约都退化为恒等（见 [`math_spec.md`](math_spec.md) 第 3.3
节），因此串行路径与并行路径走的是同一段代码。

EA 保存完整单元矩阵集合 $\{\mathbf K_e\}$，每次 MatVec 执行
gather—单元作用—scatter-add；FA 形成并保存全局 CSR。两者对应同一个离散算子，
代数细节见 [`math_spec.md` 第 2–3 节](math_spec.md)。

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

**每道门禁的精确数学式与阈值见 [`math_spec.md` 第 5 节](math_spec.md)；阈值本身
只在 [`utils/contract.py`](utils/contract.py) 定义一次，本文档不复制数字。**

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
[`math_spec.md`](math_spec.md) 第 4 节）。

其代数基础（一致/加和表示、同步归约 $\mathcal S$ 与幂等投影 $\mathcal C$、加权
内积）见 [`math_spec.md` 第 3 节](math_spec.md)。**1a 的证据不包含任何跨 rank
结论**，2 ranks 即便跑通也只验证正确性，不支持任何扩展性表述。

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
