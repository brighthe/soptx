# 子结构静力缩聚验证、契约与证据报告

本报告承载本算例的数学—代码映射契约、两类验证边界、验收阈值与证据产物；数值结论以同一次运行生成的 JSON 为准，避免把不同脚本、网格或边界条件下的历史数值混为一组证据。

## 1. 数学—代码映射契约

| 数学对象 | 当前代码 | 含义 |
|---|---|---|
| $K_{ii}^j$ | `K_local[i_dofs[:, None], i_dofs]` | 子结构内部自由度刚度。 |
| $K_{ib}^j$ | `K_local[i_dofs[:, None], b_dofs]` | 内部—接口耦合刚度。 |
| $N^j$ | `-bm.linalg.solve(K_ii, K_ib)` | 内部位移恢复映射：$u_i^j=N^j u_b^j$。 |
| $K_s^j$ | `K_bb - K_bi @ bm.linalg.solve(K_ii, K_ib)` | Schur 补缩聚刚度。 |
| $K_\mathcal{B}$ | `K_global` in `solve_condensed_fea` | 全局接口系统的 Scatter-Add 装配结果。 |

`bm.linalg.solve(K_ii, K_ib)` 与逐列施加单位接口位移、求解局部 Dirichlet 问题在代数上等价；实现不显式求逆。

## 2. 验证对象与职责

| 脚本 | 物理模型 | 验证内容 | 不验证的内容 |
|---|---|---|---|
| `minimal_demo.py` | 2D/3D 悬臂梁；分别为 `4×2`、`4×2×2` 子结构 | 已知接口位移下的局部 Schur 补形函数恢复 $u_i=N u_b$ | 全局接口系统求解、接口自由度降阶、Lagrange 交叉验证。 |
| `compare_lagrange.py --dim 2` | `HalfMBBBeamRight2d`；`6×2` 子结构、每块 `5×5` Q4 单元 | 全局接口缩聚解与 Lagrange 全装配解的一致性 | Matrix-Free、Krylov/GPU 和端到端加速。 |
| `compare_lagrange.py --dim 3` | `FullMBBBeam3d`；`6×2×2` 子结构、每块 `4×4×4` 六面体单元 | 同上；使用缩小的 $[0,6]×[0,1]×[0,1]$ 计算域 | 对 Huang 2023 全尺寸问题的性能复现。 |

两条路径必须使用相同的密度场、荷载和 Dirichlet 固定 DOF 集合。3D MBB 的固定 DOF 由 `GlobalAssembler._compute_fixed_dofs("mbb")` 与 `FullMBBBeam3d` 的语义对齐。

- `minimal_demo.py` 先得到全尺度参考解，再用其接口位移通过 $u_i=N u_b$ 回代内部自由度：仅验证局部缩聚与恢复关系，**不能**报告接口系统降阶或端到端缩聚求解性能。
- `compare_lagrange.py` 先以精确 $K_s$ 组装并求解接口系统，再恢复全场，并与 `LagrangeFEMAnalyzer` 的全装配解比较：这是全局缩聚正确性的验收脚本。

## 3. 验收契约与证据产物

- 局部恢复：相对 $L_2$ 误差应小于 `1e-12`。
- 全局交叉验证：柔度相对误差和全节点位移相对误差均应小于等于 `1e-12`。
- `compare_lagrange.py` 将上述阈值实现为运行时断言；通过后写入 `outputs/lagrange_comparison_{2d,3d}.json`。
- 文档不保存脱离脚本、commit、运行环境和 JSON 证据的「当前实测值」。

运行下列命令：

```bash
python compare_lagrange.py --dim 2 --output-dir outputs
python compare_lagrange.py --dim 3 --output-dir outputs
```

脚本在误差超过 `1e-12` 时以异常失败；通过时生成 `outputs/lagrange_comparison_2d.json` 与 `outputs/lagrange_comparison_3d.json`。每份 JSON 记录问题名称、全尺度自由度、Lagrange 自由度、缩聚接口自由度、两条路径的柔度、位移/柔度相对误差、计时与验收阈值。只有这些字段来自同一次运行时，才可写入研究报告或基金材料。

## 4. 解释边界

- 缩聚接口自由度必须来自 `solve_condensed_fea` 返回的 `interface_free`；全尺度参考解的 `free_dofs` 不得标作接口自由度。
- 机器精度级一致性支持「同一离散问题上的精确代数等价」，不直接支持性能加速、PIML 可靠性或大规模可扩展性结论。
- 计时受硬件、依赖版本、预热与计时边界影响；除非同一运行记录完整环境，否则不得用于算法速度归因。

## 5. 后续工作

当前模块是精确有限元基线；PIML 数据集、预测器、结构检查/OOD/精确回退，以及 Matrix-Free/Krylov/GPU 接口属于后续工作。
