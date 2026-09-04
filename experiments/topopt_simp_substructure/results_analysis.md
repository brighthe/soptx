# 结果与验收结论

> 状态：**尚未运行**。本目录刚建立，`outputs/` 为空，下表在首次运行后回填。
> 未运行前不得在任何地方引用本目录的数值。

## 1 验收口径

| 层级 | 判据 | 阈值 | 依据 |
| --- | --- | --- | --- |
| L0 结构性 | `full_trace` 迹基为单位阵，`n_trace_dofs == n_boundary_dofs` | 精确相等 | `FullTraceBasis` 定义 |
| L1 单步一致性 | 首步柔度与 FA 同工况之差 | 相对差 ≤ 1e-10 | 代数等价，只剩线性解法器与舍入 |
| L2 轨迹一致性 | 逐迭代柔度最大相对差、体积分数最大绝对差 | ≤ 1e-8 / ≤ 1e-10 | 同一离散系统 + 同一 OC 更新 |
| L3 拓扑一致性 | 最终密度场相对 L2 / L∞ | ≤ 1e-8 | 同上 |
| L4 迭代数 | 收敛步数 | 完全相同 | 收敛判据只依赖柔度序列 |

L1–L4 由 `compare.py` 的三阶段输出直接给出；阈值取「舍入量级」而非工程容差，
因为 `full_trace` 与 FA 之间**不存在方法误差**，任何可见偏差都应当追到实现。

`linear_corner` 工况不适用上表：它相对 FA 有式 (16) 的迹降阶误差，验收口径是
与 `full_trace` 对照给出该项误差的量级，而不是要求一致。

## 2 结果

| 工况 | trace | 单元数 | 子结构数 | 迭代步 | 最终柔度 | 体积分数 | 相对 FA 柔度差 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mbb_2d_full_trace` | `full_trace` | 待运行 | 待运行 | — | — | — | — | 未运行 |
| `mbb_3d_full_trace` | `full_trace` | 待运行 | 待运行 | — | — | — | — | 未运行 |
| `mbb_2d_linear_corner` | `linear_corner` | 待运行 | 待运行 | — | — | — | — | 未运行 |
| `mbb_3d_linear_corner` | `linear_corner` | 待运行 | 待运行 | — | — | — | — | 未运行 |

## 3 当前边界与未决项

1. **FA 侧参照缺位**：`experiments/topopt_simp_fa/cases.toml` 当前 4 个工况全部是
   悬臂梁与轴承装置，没有 MBB 梁。因此四条工况的 `fa_reference` 为空，
   L1–L4 全部无法执行。补齐路径有两条：在 FA 侧注册同参数 MBB 工况（推荐，
   保持 FA 作为唯一参照侧），或在本目录内加一条完整组装参照（会与 FA 目录职责重叠）。
2. **单元序一致性**：FA 侧密度向量按结构化网格单元序排列，本目录的密度向量经
   `split_global_cell_field` / `merge_substructure_cell_field` 往返。二者在
   `n_sub × n_fine` 结构化划分下应当同序，但首次对照必须人工核对一次，
   `compare.py` 的阶段 3 已就此显式标注。
3. **同质子结构假设**：`assemble_interface_system` 走的 `interface_indices` 要求各
   子结构边界自由度数相同，当前四条工况（均匀 `n_fine`）满足；非均匀划分需另行处理。
4. **`linear_corner` 工况已注册、尚未运行**：必须先通过 `full_trace` 门禁，
   再运行并解释 `linear_corner` 结果，以便把迹降阶误差与实现误差彻底分开。
