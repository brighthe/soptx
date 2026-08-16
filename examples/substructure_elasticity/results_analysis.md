# 子结构静力缩聚验证、契约与证据报告

本报告承载本算例的数学—代码映射契约、验证边界、验收阈值与证据产物；数值结论以同一次运行生成的 JSON 为准，避免把不同网格或边界条件下的历史数值混为一组证据。

## 1. 数学—代码映射契约

全部同构子结构共用一个 `SubstructurePrototype`，缩聚沿批量维 $B$ 一次完成，下表中的
`...` 表示可变前导维。

| 数学对象 | 当前代码 | 形状 | 含义 |
|---|---|---|---|
| $K^j$ | `prototype.assemble_local_stiffness_batch(density)` | `(B, n_dof, n_dof)` | SIMP 局部刚度 $K^j=(\rho^j)^p K_{\mathrm{unit}}$，一次批量装配。 |
| $K_{ii}^j$ | `K_local[..., i_dofs[:, None], i_dofs]` | `(..., n_i, n_i)` | 子结构内部自由度刚度。 |
| $K_{ib}^j$ | `K_local[..., i_dofs[:, None], b_dofs]` | `(..., n_i, n_b)` | 内部—接口耦合刚度。 |
| $N^j$ | `-bm.linalg.solve(K_ii, K_ib)` | `(..., n_i, n_b)` | 内部位移恢复映射：$u_i^j=N^j u_b^j$。 |
| $K_s^j$ | `K_bb - K_bi @ bm.linalg.solve(K_ii, K_ib)` | `(..., n_b, n_b)` | Schur 补缩聚刚度。 |
| $K_\mathcal{B}$ | `InterfaceSystem.stiffness` | `(n_I, n_I)` | 全局接口系统的 Scatter-Add 装配结果。 |
| $u_\mathcal{B}$ | `solve_interface_system(system, load, fixed_dofs)` | `(n_I,)` | 施加位移约束后的接口位移。 |

`bm.linalg.solve(K_ii, K_ib)` 与逐列施加单位接口位移、求解局部 Dirichlet 问题在代数上等价；实现不显式求逆。

### 建模假设：内部自由度不受载

当前 `FEAStaticCondensation` 只缩聚刚度，不缩聚载荷，恢复关系固定为 $u_i^j = N^j u_b^j$。

这**不是**相对 Huang 2023 的实现缺口，而是与论文一致的建模假设。论文式 (6) 把子结构平衡方程的右端项直接写成 $(f_{jb}^h,\ \mathbf 0)^{\mathsf T}$，并明言"不失一般性地假设与 $u_{ji}^h$ 相关的外部载荷为零"；由此论文式 (7) 中的"缩聚载荷" $\tilde f_{jb}^h = f_{jb}^h - K_{jbi}^h (K_{jii}^h)^{-1} f_{ji}^h$ 在 $f_{ji}^h = \mathbf 0$ 下退化为 $f_{jb}^h$ 本身，其恢复式同样没有 $+(K_{jii}^j)^{-1} f_i^j$ 项。

**因此本模块与论文同样只对内部自由度不受载的问题成立**：集中载荷、面载荷必须作用在接口自由度上。体力（自重、热载）作用在内部自由度上，$f_i \ne \mathbf 0$，此时必须真正实现
$f_s^j = f_b^j - K_{bi}^j (K_{ii}^j)^{-1} f_i^j$ 并在恢复式中补上 $(K_{ii}^j)^{-1} f_i^j$——论文未覆盖该类问题，本实现亦未覆盖。

`compare_lagrange.py` 使用的 MBB 梁集中载荷落在接口自由度上，故不受此限制。

## 2. 验证对象与职责

| 脚本 | 物理模型 | 验证内容 | 不验证的内容 |
|---|---|---|---|
| `compare_lagrange.py --dim 2` | `HalfMBBBeamRight2d`，$[0,60]×[0,20]$；`6×2` 子结构、每块 `5×5` Q4 单元 | 全局接口缩聚解与 Lagrange 全装配解的一致性 | Matrix-Free、Krylov/GPU 和端到端加速。 |
| `compare_lagrange.py --dim 3` | `FullMBBBeam3d`；`6×2×2` 子结构、每块 `4×4×4` 六面体单元 | 同上；使用缩小的 $[0,6]×[0,1]×[0,1]$ 计算域 | 对 Huang 2023 全尺寸问题的性能复现。 |

两条路径必须使用相同的密度场、荷载和 Dirichlet 固定 DOF 集合。这一点由构造保证而非人工对齐：密度场经 `assembler.reconstruct_global_field()` 展开给全尺度路径，外载取 `analyzer.force_vector`（施加 Dirichlet 条件*之前*的外载向量），固定 DOF 取 `tensor_space.boundary_interpolate(gd=pde.dirichlet_bc, threshold=pde.is_dirichlet_boundary())` 的掩码。固定 DOF 的物理语义由 PDE 类定义；缩聚路径只将该全局集合经 `project_global_dofs()` 投影到接口系统。

`compare_lagrange.py` 先以精确 $K_s$ 组装并求解接口系统，再恢复全场（含由 $u_i=N u_b$ 回代的内部自由度），并与 `LagrangeFEMAnalyzer` 的全装配解比较。局部回代关系是这条链路的一环，因此被端到端的机器精度一致性传递地覆盖，本目录不再单设隔离该步骤的脚本。

## 3. 验收契约与证据产物

`compare_lagrange.py` 把「柔度相对误差和全节点位移相对误差均 $\le$ `1e-12`」实现为运行时断言：超出即以异常失败且不写任何文件，通过则落盘 JSON 证据。

```bash
python examples/substructure_elasticity/compare_lagrange.py --dim 2
```

```bash
python examples/substructure_elasticity/compare_lagrange.py --dim 3
```

证据缺省写入脚本同级的 `outputs/lagrange_comparison_{2d,3d}.json`（按脚本所在位置解析，与从哪个目录发起命令无关）；`--output-dir` 可改写该目录，传相对路径时按当前工作目录解析，一般无需指定。

每份 JSON 记录问题名称、子结构划分 `n_sub`、子结构细网格 `n_fine`、全尺度自由度、Lagrange 自由度、缩聚接口自由度、两条路径的柔度、位移/柔度相对误差、计时与验收阈值。只有这些字段来自同一次运行时，才可写入研究报告或基金材料。

## 4. 实测证据

下表逐字转录自一次运行生成的 `outputs/lagrange_comparison_{2d,3d}.json`，不做任何加工。

**运行环境**

| 项 | 值 |
|---|---|
| soptx commit | `2b079f3`（工作区含未提交改动，非干净修订） |
| fealpy commit | `824dc4f39`（`~/workspace/fealpy`，editable install） |
| Python / 平台 | 3.12.13 / Linux 6.18.33.2 WSL2 x86-64, glibc 2.39 |
| NumPy / SciPy | 2.5.1 / 1.18.0 |
| `bm` 后端 | `numpy` |
| CPU | Intel Core i9-14900KF（32 逻辑核，单核串行执行） |

**验收指标**

| 指标 | 2D (`HalfMBBBeamRight2d`) | 3D (`FullMBBBeam3d`) |
|---|---|---|
| 子结构划分 `n_sub` | `6 × 2` | `6 × 2 × 2` |
| 子结构细网格 `n_fine` | `5 × 5` | `4 × 4 × 4` |
| 全尺度总自由度 | 682 | 6075 |
| Lagrange 求解自由度 (free) | 670 | 6023 |
| 接口自由度 | 298 | 4131 |
| 缩聚接口求解自由度 (free) | 286 | 4079 |
| Lagrange 柔度 | 406.9258523926437 | 228.78611503496938 |
| 缩聚柔度 | 406.9258523927062 | 228.7861150348482 |
| **柔度相对误差** | **1.5352e-13** | **5.2971e-13** |
| **位移相对误差** | **1.7228e-13** | **6.6851e-13** |
| 验收阈值 | `1e-12` | `1e-12` |
| 结论 | 通过 | 通过 |

两个维度的相对误差都比阈值低约 3 个数量级，落在双精度累积舍入的量级上，支持「两条路径在同一离散问题上精确代数等价」这一结论。

**计时（仅供同环境下的相对参考）**

| 路径 | 2D | 3D |
|---|---|---|
| Lagrange 全装配求解 | 0.0125 s | 0.6199 s |
| 缩聚接口求解 | 0.0019 s | 0.1993 s |

计时未做预热与多次取样，且缩聚路径的计时边界不含局部批量缩聚的离线代价；受硬件、依赖版本与计时边界影响，**不得**单独用于算法加速比归因。

> 上述数值由 Claude 在重构验证过程中运行产生，非用户本人执行。复核请重跑 §3 的两条命令并以新生成的 JSON 为准；若数值与本表不一致，以 JSON 为准并更新本节。本文档不保存脱离脚本、commit、运行环境和 JSON 证据的「当前实测值」。
