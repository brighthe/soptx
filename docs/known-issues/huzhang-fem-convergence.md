# 胡张元混合有限元求解链不收敛修复记录

> 提交：`fa73d4d`（主修复）、`c4a2d37`（div_basis 简化）
> 日期：2026-08-05
> 结论：根因是 FEALPy 4.0.0（fealpy_stable）`grad_shape_function` 默认返回
> 参考坐标导数而非物理梯度，导致 2D `div_basis` 散度错 2 倍；修复后
> degree 2/3/4 全部恢复收敛，pytest 81 项通过。

## 问题现象

`soptx` 从 `fealpy_heliang`（3.4.0）迁移到 `fealpy_stable`（4.0.0）后，
`examples/huzhang_elasticity/minimal_demo.py` 出现：

- σ L2 误差停滞在 O(1)（~8-10），不随网格加密收敛；u 同样停滞（~1-2）
- div 误差 3 阶收敛（散度部分正确）
- 残差 1e-16、对称性 1e-19 均通过——**系统自洽但物理错误**，判据无法暴露
- degree 2（jump-penalty 路径）甚至秩亏/求解崩溃（SuperLU DGEMV 非法）

## 定位过程（关键实验）

1. **旧仓库对照**：`soptx_heliang` + `fealpy_heliang`（3.4.0）同一流程
   完美收敛（σ 4 阶）→ 锁定差异在 FEALPy API 层，而非数学问题。
2. **逐项数值验证**（新旧环境同网格对比）：A 矩阵（含无散度方向）、
   B 矩阵元素、F_body、插值链（P2 恒等 1e-15）、强加链、solve_state 流程
   全部逐位正确 → 缩小到 `div_basis`。
3. **有限差分验证**：`div_basis` 输出 vs 解析散度，在旧网格（中心扇型单元）
   上失败（符号/分量错）；3D 空间通过——差异在调用参数 `variables='x'`。
4. **验证教训**：FD 验证必须在**多种网格（含非对称单元）**上做——对称单元
   上参考导数与物理导数巧合一致，单点验证会误判。

## 根因与修复清单

| # | 问题 | 根因 | 修复 | 提交 |
| --- | --- | --- | --- | --- |
| 1 | σ/u 不收敛（总根因） | 2D `div_basis` 用 `grad_shape_function` 默认返回**参考坐标导数**（= 物理梯度的 1/2）；3D 一直带 `variables='x'` 故正确 | `div_basis` 改调 `grad_shape_function(bc, p, variables='x')` 取笛卡尔物理梯度 | `fa73d4d`、`c4a2d37` |
| 2 | degree 2 div 发散 | jump-penalty 缩放 `E/L0²·hF` 量级过大，惩罚块远超柔度块，压坏 div 约束 | 改为 `0.01·模量/hF`（γ/hF 标准 DG 型，γ 取小比例系数） | `fa73d4d` |
| 3 | degree 2 秩亏/求解崩溃 | fealpy `bmat` 在 blocks 全非 None 时走 hstack/vstack 分支，`B.T` 与 `-J` 组合丢失 `-J` 块（K 秩 96/135） | 改 `scipy.sparse.bmat`（显式零块）+ `CSRTensor.from_scipy` | `fa73d4d` |
| 4 | 全 Dirichlet（u_D≠0）问题失效 | `assemble_displacement_bc_vector` 恒返回零（只支持齐次） | 实现 `∫_Γ_D (τ·n)·u_D` 自然边界向量（`face_to_cell`/`face_unit_normal`/`basis(index=...)` 适配） | `fa73d4d` |
| 5 | 求解后缓存的 K 被破坏 | fealpy `spsolve` 经 `to_scipy()`（共享内存视图）传给 scipy SuperLU，**原地修改矩阵**（列置换/缩放） | 缓存用 `K.copy()`（`CSRTensor.copy()`） | `fa73d4d` |
| 6 | degree 2 组装 API 报错 | 2D 下 `cell_to_face_sign` 改名 `cell_to_edge_sign` | 按 `mesh.top_dimension()` 分派 | `fa73d4d` |

另：`examples/huzhang_elasticity/minimal_demo.py` 的 `mesh.error` 改位置参数
（FEALPy 4.0 API）。

## 验证结果

| 验证项 | 结果 |
| --- | --- |
| demo degree 3（松弛开/关） | u 3 阶、**σ 4 阶**（1.2e-6）、div 3 阶；残差/对称性通过 |
| degree 2（jump-penalty） | u 2 阶、σ 2.4 阶、div 1.6 阶 |
| degree 4 | u 4 阶、**σ 5 阶**（1.3e-8） |
| from_box 无松弛对照 | σ 4 阶（1.1e-5），与 checkerboard 一致 |
| 3D 空间 | `div_basis` 无同类问题（`variables='x'` 已正确，FD 3.4e-10） |
| pytest | 81 passed + 17 subtests |

## 遗留待办

- jump-penalty 的 `0.01·模量/hF` 系数为数值标定，需理论核查（degree 2
  收敛阶 σ 2.4 vs 理论 3 阶）
- 3D 无松弛求解链整体可用性尚未端到端验证（`div_basis` 已确认正确）
