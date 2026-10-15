# 归档

本页登记从主线移出、只保留在 git 标签上的代码。恢复时以标签为准, 不在主线保留副本.

## `archive/distributed-mpi`

| 项 | 内容 |
|---|---|
| 标签 | `archive/distributed-mpi` (带注释标签, 指向 `a5f8a9b`) |
| 归档日期 | 2026-10-10 |
| 原因 | 分布式/MPI 架构 (重叠自由度布局、`OverlapOperator` 与 `ConstrainedOperator` 的包装次序、分析器钩子划分) 是否合理尚未评审, 不让未定的设计约束主线架构; 以后重新设计时以标签上的代码为参考, 不直接搬回 |

移出的内容:

| 类别 | 路径或接口 |
|---|---|
| 分布式实现 | `src/soptx/fem/distributed/` (`EntityMPI`, `OverlapOperator`, 分区, 分布式网格与空间), `src/soptx/fem/analyzers/distributed_analyzer.py`, `build_distributed_analyzer` |
| 求解层 | `src/soptx/solvers/overlap.py` (`weighted_cg`, `weighted_norm`), `cg` / `CGSolver` 的 `dot_product` 参数 |
| 分析器钩子 | `LagrangeFEMAnalyzer` 的 `dof_comm` 参数与属性、多 rank 守卫, `reduce_load` / `wrap_operator` 扩展点 |
| Matrix-Free 一步求解 | `src/soptx/fem/matrix_free/solve.py` (`solve_ea_system`, `solve_matrix_free_system`); `ElasticityEAOperator` 保留, 只去掉 `dof_comm` 分支 |
| 数值常数 | `soptx.core.numerics` 中只服务于上述路径的 `DEFAULT_MAX_ITERATIONS`, `DEFAULT_RTOL`, `DEFAULT_ATOL`, `RESIDUAL_REFRESH`; `NORM_FLOOR` 保留 |
| 证据工具链 | `tools/matrix_free_evidence/` |
| 实验与示例 | `experiments/matrix_free_capability/` (图 2 成图数据), `examples/matrix_free_elasticity/` |
| 测试 | `test_distributed_analyzer.py`, `test_matrix_free_partition.py`, `test_matrix_free_cg.py`, `test_matrix_free_evidence_{layout,schema}.py`, operator_level 集成测试中的分布式钩子用例 |
| 打包与 CI | `pyproject.toml` 的 `mpi` extra, CI 的 `matrix-free-fast` job |

`soptx.core.mpi_runtime` 不在归档之列: MUMPS 直接法依赖它激活 MPI 上下文, `mpi4py` 改由
`mumps` extra 提供.

查看或恢复:

```bash
git show archive/distributed-mpi --stat
git checkout archive/distributed-mpi -- src/soptx/fem/distributed
```
