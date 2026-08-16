# 经典子结构有限元与静力缩聚范例 (Substructure FE Elasticity Demo)

本目录是 2D/3D 线弹性问题的**精确子结构静力缩聚基线**。它只包含经典有限元、SIMP 局部刚度和 Schur 补代数，**不含** PIML 预测器、Matrix-Free、Krylov/GPU 或拓扑优化迭代闭环。

## 文件职责

| 文件 | 职责 |
|---|---|
| `compare_lagrange.py` | 端到端缩聚接口系统与 Lagrange FEM 的 2D/3D MBB 交叉验证；通过时写出 JSON 证据。 |
| `results_analysis.md` | 数学—代码映射契约、验证边界、验收阈值与证据解释。 |

> 核心模块 `SubstructurePrototype`、`SubstructureMesh`、`FEAStaticCondensation`、
> `GlobalAssembler`、`solve_interface_system` 位于
> [`src/soptx/fem/substructure/`](../../src/soptx/fem/substructure/)，通过
> `from soptx.fem.substructure import ...` 导入。

## 调用范式

张量化调用链如下，物理模型、外载与约束全部取自
[`soptx.problems.elasticity`](../../src/soptx/problems/elasticity/)，脚本不硬编码几何、材料或边界条件：

1. **共享参考子结构。** 全部子结构同构，因此构造一个 `SubstructurePrototype`，
   由所有 `SubstructureMesh` 通过 `prototype=proto` 共享；离散结构、自由度划分与
   单位密度单元刚度只构造一次。
2. **批量缩聚。** `prototype.assemble_local_stiffness_batch(density)` 一次得到
   `(B, n_dof, n_dof)` 的局部刚度，再交给**单个** `FEAStaticCondensation` 沿前导维
   一次算完全部 Schur 补，取代逐子结构的 Python 循环。
3. **外载与约束来自 PDE。** 外载取 `analyzer.force_vector`（施加 Dirichlet 条件
   *之前*的外载向量），固定自由度取
   `tensor_space.boundary_interpolate(gd=pde.dirichlet_bc, ...)` 的掩码，
   使两条路径的载荷与约束在结构上同源，而非靠脚本两处手写对齐。
4. **接口求解。** `solve_interface_system(system, load, fixed_dofs)` 承担
   「施加位移约束 + 稀疏直接求解」，装配器本身不携带载荷与求解策略。

## 运行与验收

全局接口缩聚与 Lagrange 全装配的交叉验证（MBB 梁）：

```bash
python examples/substructure_elasticity/compare_lagrange.py --dim 2
```

```bash
python examples/substructure_elasticity/compare_lagrange.py --dim 3
```

`compare_lagrange.py` 要求柔度和全节点位移的相对误差均不超过 `1e-12`；不满足时以异常失败，且不写任何文件。通过时分别生成：

- `outputs/lagrange_comparison_2d.json`
- `outputs/lagrange_comparison_3d.json`

`--output-dir` 用于改写这两份 JSON 证据的落盘目录，缺省即上面的 `outputs/`（按脚本所在位置解析，与从哪个目录发起命令无关）。传相对路径会按**当前工作目录**解析，可能落到 `examples/*/outputs/` 之外而不被 `.gitignore` 忽略，一般无需指定。

这些 JSON 是当前可复核的数值证据；计时仅在相同硬件、依赖版本和计时边界下可比较，不能单独用于性能结论。
