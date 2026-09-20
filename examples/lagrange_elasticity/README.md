# 拉格朗日位移有限元范例 (Lagrange Displacement FEM Demo)

本目录是线弹性问题的 **CPU 串行全装配（FA）拉格朗日位移元基线**。它只包含单机串行的全局刚度装配、边界条件对称消元与直接/迭代求解，**不含** GPU、Matrix-Free/MPI 并行、子结构静力缩聚、PINN/PIML 代理或拓扑优化迭代闭环。

其他技术栈的入口见 [`examples/index.md`](../index.md)。本目录产出的正确性证据被 [`experiments/elasticity_paradigm_comparison/`](../../experiments/elasticity_paradigm_comparison/) 直接取用为参考路径的证据来源，不在该实验中重复验证。

## 能力边界

- **有限元次数**：支持任意正整数多项式次数 $p \ge 1$，已实测 $p=1$ 与 $p=2$。
- **网格类型**：2D 支持 `TriangleMesh` / `QuadrangleMesh`，3D 支持 `TetrahedronMesh` / `HexahedronMesh`。张量积网格的已知问题见 [`docs/known-issues/fealpy-patches.md`](../../docs/known-issues/fealpy-patches.md) 第一节。
- **求解器**：`scipy`（缺省，直接法）、`mumps`（直接法，需额外安装 PyMUMPS）和 `cg`（迭代法）。

详细验证范围与证据边界见 [`results_analysis.md`](results_analysis.md)。

## 文件职责

| 文件 | 职责 |
|---|---|
| [`manufactured_convergence_demo.py`](manufactured_convergence_demo.py) | 制造解基准：多层嵌套加密下的 $L_2$ 观测收敛阶与真相对残差，验证离散正确性。 |
| [`concentrated_load_demo.py`](concentrated_load_demo.py) | 集中力工程基准（MBB 梁）：真相对残差与载荷等效性验证；可选导出最密层位移场为 VTU。 |
| [`results_analysis.md`](results_analysis.md) | 数学—代码映射、验收契约、实测证据、结果解释与已知缺口。 |

物理模型统一来自 [`soptx.problems.elasticity`](../../src/soptx/problems/elasticity/)；制造解和工程基准分别见 [`docs/problems/manufactured-elasticity.md`](../../docs/problems/manufactured-elasticity.md) 与 [`docs/problems/engineering-benchmarks.md`](../../docs/problems/engineering-benchmarks.md)。

## 快速运行

前置：SOPTX 需以 editable 方式安装（`pip install -e .`）。

### 制造解 $L_2$ 收敛阶验证

```bash
python examples/lagrange_elasticity/manufactured_convergence_demo.py \
    --dim 2 --mesh-type tri --model sinusoidal
```

```bash
python examples/lagrange_elasticity/manufactured_convergence_demo.py \
    --dim 2 --mesh-type quad --model mixed-sinusoidal
```

```bash
python examples/lagrange_elasticity/manufactured_convergence_demo.py \
    --dim 3 --mesh-type tet --model divfree-poly --base 4 --levels 5 \
    --solver mumps --assembly-method fast --mumps-sym 1
```

### 集中力载荷路径验证

```bash
python examples/lagrange_elasticity/concentrated_load_demo.py
```

```bash
python examples/lagrange_elasticity/concentrated_load_demo.py \
    --dim 3 --problem mbb-half-3d --nx 30 --ny 10 --nz 10 --levels 1
```

## 关键参数

- 两个脚本共同支持 `--degree`、`--levels`、`--solver` 和 `--output-dir`；`cg` 的容差与迭代上限通过 `--rtol`、`--atol` 和 `--maxiter` 设置。
- 制造解脚本通过 `--dim`、`--mesh-type`、`--model` 和 `--base` 选择验证路径；`--assembly-method` 与 `--mumps-sym` 用于比较装配和 MUMPS 配置。
- 制造解脚本的 `--operator-level` 取 `fa`（缺省）、`ea` 或 `pa`，在同一条误差链上换算子层级；`ea` 与 `pa` 不持有显式矩阵，只能配 `--solver cg`，其他组合在入口即被拒。缺省档 `fa` 不进产物文件名，`ea`/`pa` 各占一个文件。
- 集中力脚本通过 `--problem`、`--dim`、`--mesh-type` 和 `--nx/--ny/--nz` 选择算例与网格；`--save-vtu` 可导出最密层位移场。
- 完整参数和缺省值以脚本的 `--help` 为准。

验收阈值、数学—代码映射、证据产物、实测结果、内存口径和已知缺口统一见 [`results_analysis.md`](results_analysis.md)。
