# 经典子结构有限元算例代码规范 (Math Spec)

> **理论事实源**：完整物理原理、变分推导与 Schur 补代数性质见知识库概念页：`C:\workspace\dut-postdoc\concepts\substructural-condensation.md`。

---

## 1. 代码变量与数学符号映射表 (Symbol-to-Code Mapping)

| 数学符号 | `substructure.py` / `assembler.py` 代码变量 | 代数含义与算式 |
|---|---|---|
| $\mathbf{K}_{ii}^j$ | `K_ii = K_local[np.ix_(i_dofs, i_dofs)]` | 局部内部自由度刚度矩阵 |
| $\mathbf{K}_{ib}^j$ | `K_ib = K_local[np.ix_(i_dofs, b_dofs)]` | 局部内部与接口耦合刚度矩阵 |
| $\mathbf{N}^j$ | `N = -np.linalg.solve(K_ii, K_ib)` | 子结构多尺度形函数矩阵 ($\boldsymbol{u}_i^j = \mathbf{N}^j \boldsymbol{u}_b^j$) |
| $\mathbf{K}_s^j$ | `K_s = K_bb - K_bi @ np.linalg.solve(K_ii, K_ib)` | Schur 补缩聚刚度矩阵 ($\mathbf{K}_s^j \boldsymbol{u}_b^j = \mathbf{f}_b^j$) |
| $\mathbf{K}_{\text{global}}$ | `K_global` (in `solve_condensed_fea`) | 全局接口 Scatter-Add 粗系统刚度矩阵 |
| $\boldsymbol{u}_i^j$ | `u_sub_i_recovered = condensor.recover(u_sub_b)` | 由接口位移回代恢复出的子结构内部细观位移 |

> **实现等价性补充**：
> 代码中使用的矩阵消元 `np.linalg.solve(K_ii, K_ib)` 在数学结果上与 **Huang 2023 论文中“依次施加单位接口位移基 $\boldsymbol{u}_{b,k} = \mathbf{e}_k$ 求解局部 Dirichlet 问题”** 100% 严格等价。

---

## 2. 接口系统 Scatter-Add 装配 (Interface System Assembly)

`solve_condensed_fea` 中的粗尺度接口系统按以下步骤装配：

1. **接口 DOF 收集** (`build_interface_dofs`)：遍历所有子结构，取各子结构边界节点 DOF 的**全局编号**，去重后构成接口 DOF 集合
   $$
   \mathcal{B} = \bigcup_j \{\text{global-dof}( \text{boundary node of } \Omega^j )\},
   \qquad n_{\mathcal{B}} = |\mathcal{B}|.
   $$
2. **Scatter-Add 装配**：对每个子结构 $j$，将其缩聚刚度 $\mathbf{K}_s^j$（大小 $n_b^j \times n_b^j$）通过
   局部边界 DOF → 全局接口 DOF 的映射散加进 $n_{\mathcal{B}} \times n_{\mathcal{B}}$ 的接口刚度
   $$
   \mathbf{K}_{\mathcal{B}} = \sum_j \mathbf{R}^{j,\mathsf T} \mathbf{K}_s^j \mathbf{R}^j,
   $$
   其中 $\mathbf{R}^j$ 为装配算子（`get_substructure_global_dofs` 提供局部→全局 DOF 编号，
   `global_to_interface` 提供全局→接口序号）。代码中 Python 双层循环 `K_global[i_glob, j_glob] += K_s[i_local, j_local]`
   即逐元素等价于上式矩阵散加。
3. **右端项与 BC 施加**：接口右端项 $\boldsymbol{F}_{\mathcal{B}}$ 只在加载 DOF 处非零；Dirichlet 固定 DOF
   从接口系统中消去（`interface_free = setdiff1d(arange(n_interface), interface_fixed)`）。
4. **求解与恢复**：求解接口系统 $\mathbf{K}_{\mathcal{B}} \boldsymbol{u}_{\mathcal{B}} = \boldsymbol{F}_{\mathcal{B}}$ 得接口位移，
   再对每个子结构用 $\boldsymbol{u}_i^j = \mathbf{N}^j \boldsymbol{u}_b^j$ 恢复内部位移。

> **边界条件一致性**：路径 A（LagrangeFEMAnalyzer）的 Dirichlet 边界取自 `FullMBBBeam3d` / `HalfMBBBeamRight2d`，
> 路径 B（`solve_condensed_fea` 的 `bc_type="mbb"`）必须使用**完全相同**的固定 DOF 集合，否则两路径约束不同、
> 柔度会出现系统性偏差（3D 下曾因路径 B 误用"整面 ux=0"而差约 6 倍，已对齐修复）。

---

## 3. 算例数值验收标准 (Acceptance Criteria)

1. **纯粹性**：100% 纯经典有限元代数求解，0 神经网络/PyTorch 依赖；
2. **缩聚与位移恢复精度**（`minimal_demo.py`）：
   恢复出的细观位移 $\boldsymbol{U}_{\text{recovered}}$ 与全尺度有限元直解 $\boldsymbol{U}_{\text{ref}}$ 的相对 $L_2$ 误差必须满足：
   $$
   \frac{\|\boldsymbol{U}_{\text{recovered}} - \boldsymbol{U}_{\text{ref}}\|_2}{\|\boldsymbol{U}_{\text{ref}}\|_2} < 10^{-12} \quad (\text{实测 } 2.32 \times 10^{-16} \text{ 机器精度})
   $$
3. **两路径交叉对比精度**（`compare_lagrange.py`）：子结构缩聚解与 LagrangeFEMAnalyzer 全装配解的
   结构柔度相对误差与位移场全节点相对误差均应 $\lesssim 10^{-12}$：
   - 2D：柔度误差 $4.88 \times 10^{-13}$，位移误差 $5.17 \times 10^{-13}$；
   - 3D：柔度误差 $1.83 \times 10^{-13}$，位移误差 $2.24 \times 10^{-13}$。
