# 子结构 full_trace 与 linear_corner 接口的拓扑优化对照

硬件：Intel Core i9-14900KF（24 核 32 线程，8P+16E），运行内存 45 GiB；NVIDIA GeForce RTX 5080，显存 16 GB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（内置 scipy-openblas 0.3.34），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.13.0（CUDA 13.0）；PyMUMPS 0.4.0，链接系统 MUMPS 5.6.2（OpenBLAS 0.3.26、Open MPI 4.1.6、ScaLAPACK 2.2.1，单进程运行）；pyevtk 1.7.0。

## 1. 研究对象

对照相同细网格、相同子结构划分下，子结构边界位移的两种表示 `full_trace` 与 `linear_corner` 对 Huang2023 三维 MBB 梁上的完整优化结果。

两侧的子结构位移都用基函数插值表示：$u^j=N^jq^j$，$q^j$ 为待求系数，$N^j$ 为子结构的多尺度基函数，每一列是某个系数取单位值、其余为零时子结构的静力位移，其边界节点行给出边界位移，内部节点行给出内部位移。两侧的内部位移都由边界位移精确回代得到，差别只在系数 $q^j$ 的选取：`full_trace` 取全部边界节点位移，`linear_corner` 取 8 个角点位移，基函数随之确定。

| 项目 | full_trace | linear_corner |
|---|---|---|
| 子结构边界位移 | 全部边界节点位移：$u_b^j=q^j$ |  8 个角点位移 $q^j$ 三线性插值：$u_b^j=Lq^j$ |
| 刚度算子 | 各子结构 Schur 补 $K_s^j$ 组装为全局边界矩阵 $K_\Gamma$ | 各子结构投影到角点的 Schur 补 $L^{\mathsf T}K_s^jL$ 组装为全局角点矩阵 $K_C=P^{\mathsf T}K_\Gamma P$ |
| 未知量 | 全局边界位移 $U_\Gamma$ | 全局角点位移 $U_C$ 与插值得到的边界位移 $U_\Gamma=PU_C$ |

### 1.1 矩阵组装、边界参数化、局部缩聚与全局组装

**矩阵组装**。首先将细网格单元刚度 $E(\rho_e)K^0$ 组装为子结构总刚 $K^j$，$K^0$ 为单位模量的单元刚度，并按边界 $b$、内部 $i$ 分块：

$$
K^j=\sum_{e\in\Omega_j}(G_e^j)^{\mathsf T}E(\rho_e)K^0\,G_e^j
=\begin{bmatrix}K_{bb}^j&K_{bi}^j\\K_{ib}^j&K_{ii}^j\end{bmatrix}\in\mathbb R^{n_d\times n_d}
$$

**边界位移的参数化**。子结构平衡方程按同一分块写为

$$
\begin{bmatrix}K_{bb}^j&K_{bi}^j\\K_{ib}^j&K_{ii}^j\end{bmatrix}
\begin{bmatrix}u_b^j\\u_i^j\end{bmatrix}=
\begin{bmatrix}f_b^j\\0\end{bmatrix},
$$

其中 $f_b^j$ 为子结构边界节点上的节点力，组装后边界右端项只剩外载荷 $F_\Gamma$；内部载荷为零。边界自由度在子结构中占多数，两侧用不同的参数 $q^j$ 表示边界位移，$u_b^j=\Psi q^j$：

- `full_trace` 取 $\Psi=I$，$q^j=u_b^j$ 就是全部边界节点位移，不做降维。
- `linear_corner` 取 $\Psi=L$，$q^j$ 为 8 个角点的 24 个位移，$L\in\mathbb R^{n_b\times24}$ 为角点位移到全部边界节点位移的三线性插值矩阵。

**局部缩聚**。第二行 $K_{ib}^ju_b^j+K_{ii}^ju_i^j=0$ 即内部平衡方程。代入 $u_b^j=\Psi q^j$，解出内部位移对参数的响应：

$$
u_i^j=N_i^jq^j,\qquad N_i^j=-(K_{ii}^j)^{-1}K_{ib}^j\Psi ,
$$

$N_i^j$ 的每一列是某个参数取单位值、其余为零时内部节点的静力位移，两侧都由内部平衡方程精确解出。再把 $u_b^j=\Psi q^j$、$u_i^j=N_i^jq^j$ 代入第一行，消去内部自由度，并左乘 $\Psi^{\mathsf T}$ 把 $n_b$ 个方程投影到 $q^j$ 的维数，得到只含 $q^j$ 的缩聚平衡方程：

$$
\Psi^{\mathsf T}\bigl(K_{bb}^j-K_{bi}^j(K_{ii}^j)^{-1}K_{ib}^j\bigr)\Psi\,q^j=\Psi^{\mathsf T}f_b^j .
$$

括号内即 Schur 补 $K_s^j=K_{bb}^j-K_{bi}^j(K_{ii}^j)^{-1}K_{ib}^j\in\mathbb R^{n_b\times n_b}$，缩聚刚度为

$$
\Psi^{\mathsf T}K_s^j\Psi=
\begin{cases}
K_s^j\in\mathbb R^{n_b\times n_b}, & \texttt{full\_trace}\ (\Psi=I),\\[4pt]
K_c^j=L^{\mathsf T}K_s^jL\in\mathbb R^{24\times24}, & \texttt{linear\_corner}\ (\Psi=L).
\end{cases}
$$

`full_trace` 不施加任何约束，缩聚刚度就是 Schur 补；`linear_corner` 的 $K_c^j$ 是 $K_s^j$ 在角点插值空间上的 Galerkin 投影，引入了边界位移约束，是 Ritz 近似。

**全局未知量与组装**。相邻子结构共享边界，各子结构的 $q^j$ 不独立，而是同一个全局未知量 $Q$ 的分量，$q^j=A^jQ$，$A^j$ 为限制矩阵（布尔定位矩阵，即以子结构为单元的 cell2dof）。三层量的对应如下：

| | 边界位移 $u_b^j$ | 子结构参数 $q^j$ | 全局未知量 $Q$ |
|---|---|---|---|
| `full_trace` | $u_b^j=q^j$ | 全部边界节点位移，$n_b$ 个 | 全局边界位移 $U_\Gamma$，$q^j=A_b^jU_\Gamma$ |
| `linear_corner` | $u_b^j=Lq^j$ | 8 个角点的 24 个位移 | 全局角点位移 $U_C$，$q^j=A_c^jU_C$ |

把 $q^j=A^jQ$ 代入各子结构的缩聚方程并求和，得到全局系统刚度：

$$
K_\Gamma=\sum_j(A_b^j)^{\mathsf T}K_s^jA_b^j\in\mathbb R^{N_\Gamma\times N_\Gamma},\qquad
K_C=\sum_j(A_c^j)^{\mathsf T}K_c^jA_c^j\in\mathbb R^{N_C\times N_C}.
$$

两个全局未知量由全局插值 $P$ 联系，$U_\Gamma=PU_C$，且 $A_b^jP=LA_c^j$。`full_trace` 下单个 $K_s^j$ 只是局部对象，组装后的 $K_\Gamma$ 是细网格总刚对接口自由度的 Schur 补，$K_\Gamma U_\Gamma=F_\Gamma$ 与细网格 FEM 代数等价。

### 1.2 求解与位移恢复

`full_trace` 首先得到全局边界位移：

$$
K_\Gamma U_\Gamma=F_\Gamma .
$$

`linear_corner` 首先得到全局角点位移：

$$
K_CU_C=F_C ,
$$

其中 $F_C=P^{\mathsf T}F_\Gamma=\sum_j(A_c^j)^{\mathsf T}L^{\mathsf T}f_b^j$ 为投影到角点的载荷。

提取各子结构的边界位移参数 $q^j$，由 $u_b^j=\Psi q^j$ 得到子结构边界位移，回代内部位移：

$$
q^j=
\begin{cases}
A_b^jU_\Gamma, & \texttt{full\_trace},\\[4pt]
A_c^jU_C, & \texttt{linear\_corner},
\end{cases}\qquad
u_b^j=\Psi q^j,\qquad
u_i^j=N_i^jq^j.
$$

将边界与内部位移按全局编号恢复为全场位移 $U^\Gamma$、$U^C$：

$$
U^\Gamma=R_\Gamma^{\mathsf T}U_\Gamma+\sum_j(R_i^j)^{\mathsf T}u_i^j,\qquad
U^C=R_\Gamma^{\mathsf T}PU_C+\sum_j(R_i^j)^{\mathsf T}u_i^j .
$$

两侧内部位移都由精确解出的 $N_i^j$ 回代得到，精确满足细网格内部平衡，差别只在边界位移 $u_b^j$。

### 1.3 Ritz 近似

对于同一物理密度场，在相同材料、载荷与支承下，`linear_corner` 的角点系统是 `full_trace` 边界系统在角点插值子空间 $U_\Gamma=PU_C$ 上的 Galerkin 投影：

$$
K_C=\sum_j\bigl(LA_c^{j}\bigr)^{\mathsf T}K_s^{j}\bigl(LA_c^{j}\bigr)=\sum_j\bigl(A_b^{j}P\bigr)^{\mathsf T}K_s^{j}\bigl(A_b^{j}P\bigr)=P^{\mathsf T}K_\Gamma P,\qquad F_C=P^{\mathsf T}F_\Gamma
$$

本算例内部载荷为零，$F_\Gamma$ 同时表示原载荷的边界分量与缩聚后的载荷。结合内部位移回代，可写为：

$$
\text{full\_trace:}\quad
\begin{cases}
K_\Gamma U_\Gamma=F_\Gamma,\\
U_I^\Gamma=-K_{II}^{-1}K_{I\Gamma}U_\Gamma.
\end{cases}
\qquad \text{linear\_corner:}\quad
\begin{cases}
P^{\mathsf T}K_\Gamma P\,U_C=P^{\mathsf T}F_\Gamma,\\
U_I^C=-K_{II}^{-1}K_{I\Gamma}P\,U_C.
\end{cases}
$$

两侧内部位移均精确回代，但角点系统仅满足投影后的边界平衡。因此两侧一般不代数等价，角点插值解是完整边界解的 Ritz 近似。

以下误差关系针对同一物理密度场、相同材料、载荷与齐次支承，并假设两侧线性系统精确求解。记恢复后的全场位移为 $U^\Gamma$、$U^C$，柔顺度分别为 $C_\Gamma=F_\Gamma^{\mathsf T}U_\Gamma$、$C_C=F_C^{\mathsf T}U_C$，则：

$$
C_\Gamma-C_C
=\|U^\Gamma-U^C\|_K^2\ge0,\qquad
\|v\|_K^2=v^{\mathsf T}Kv.
$$

能量范数取在满足齐次支承的自由位移空间上。对于 $C_\Gamma>0$，相对柔顺度误差与相对能量误差为：

$$
\varepsilon_C=\frac{C_\Gamma-C_C}{C_\Gamma},\qquad
\varepsilon_E=\frac{\|U^\Gamma-U^C\|_K}{\|U^\Gamma\|_K}
=\sqrt{\varepsilon_C}.
$$


## 2. 实验设置

实验脚本为 [`run_substructure_full_trace_vs_linear_corner.py`](run_substructure_full_trace_vs_linear_corner.py)。

本实验均采用 NumPy / CPU：`BACKEND = 'numpy'`，`DEVICE = 'cpu'`；单线程：线程数由 `--threads` 指定，默认 `NUM_THREADS = 1`，脚本在导入 numpy/scipy 之前把全部 BLAS / OpenMP 线程变量设为该值，取值记入 `config.json` 的 `threads`，输出目录名带后缀 `_t1`。

### 2.1 模型参数

![MBB 梁的几何、载荷与支承](../topopt_piml_substructure/assets/Huang2023_Fig5.png)

采用整体三维 MBB 梁模型，长、高、宽分别记为 $L_x,L_y,L_z$，比例为 $6:1:1$，以细单元边长为长度单位（$h=1$），计算域 $\Omega=[0,L_x]\times[0,L_y]\times[0,L_z]$ 的尺寸由网格剖分数确定。顶面中心施加竖直向下的单位集中力 $F=1$。支承取在两端底边的四个角点：左端两角点铰支（$u_x=u_y=u_z=0$），右端两角点滚支（$u_y=0$）。

| 参数 | 数值或设置 |
|---|---|
| 几何与计算域 | 整体三维 MBB 梁，$L_x:L_y:L_z=6:1:1$，尺寸由网格确定：`domain = (0, nx*h, 0, ny*h, 0, nz*h)`，`FullMBBBeamPoint3d(domain=domain, ...)` |
| 长度约定 | 以细单元边长为长度单位，$h=1$：`ELEMENT_SIZE = 1.0`；网格须满足 $n_x:n_y:n_z=6:1:1$ |
| 载荷 | 顶面中心集中力，沿 $y$ 方向向下，合力为 $-1$：`LOAD = -1.0` |
| 支承 | 齐次 Dirichlet 边界条件：左端两底角点 $u_x=u_y=u_z=0$，右端两底角点 $u_y=0$：`FullMBBBeamPoint3d` |
| 实体材料杨氏模量 $E_0$ | $1$：`E0 = 1.0` |
| 弱材料杨氏模量 $E_{\min}$ | $10^{-7}$：`EMIN = 1.0e-7` |
| 泊松比 $\nu$ | $0.3$：`NU = 0.3` |

### 2.2 分析参数

**有限元离散**

| 参数 | 数值或设置 |
|---|---|
| 有限元 | 线性元：`SPACE_DEGREE = 1` |
| 数值积分 | 二阶高斯积分：`INTEGRATION_ORDER = 2`，传入 `integration_order=INTEGRATION_ORDER` |
| 单元刚度积分 | `fast` 积分实现：`LinearElasticIntegrator(..., method='fast')` |
| 分析网格 | 结构化六面体网格，各方向单元数由 `--grid` 设定，须满足 $6:1:1$；也可由 `--n-sub` 与 `--n-fine` 推出：`mesh = HexahedronMesh.from_box(box=domain, nx=nx, ny=ny, nz=nz, ...)` |

**子结构缩聚**

| 参数 | 数值或设置 |
|---|---|
| 子结构划分 | 子结构数由 `--n-sub` 设定，每个子结构含 $m\times m\times m$ 个细单元，$m$ 由 `--n-fine` 设定，须满足 $\text{grid}=n_{\text{sub}}\times m$，缺省项由其余两项推出：`StructuredSubstructureLayout(..., n_sub=n_sub, n_fine=(m, m, m), full_mesh=mesh)` |
| 边界位移表示 | full_trace 保留全部边界自由度，$\Psi=I$：`SubstructureAnalyzer(..., trace='full_trace')`；linear_corner 取角点三线性插值，$\Psi=L$：`SubstructureAnalyzer(..., trace='linear_corner')` |
| 分批大小 | 每批处理 256 个子结构：`chunk_size=256` |

**线性求解参数**

| 参数 | 数值或设置 |
|---|---|
| 求解器 | CG：`SOLVER = 'cg'`；预条件子由 `--preconditioner` 指定，可选 `none`、`jacobi`、`gmg`（默认 `jacobi`，`gmg` 仅支持 full_trace）；实际取值记入 `config.json` 与结果表 |
| 停机准则 | 相对右端项的容差 $10^{-6}$：`CG_TOLERANCE = 1.0e-6`，`rtol=CG_TOLERANCE`，`atol=0`，`rtol_reference='rhs'` |
| 最大 CG 步数 | `CG_MAXITER = 20000` |
| 迭代初值 | 首轮为零向量；第 2 轮起取上一轮的解：`x0=previous_interface`，`full_trace` 为全局边界位移，`linear_corner` 为全局角点位移 |

### 2.3 优化参数

| 参数 | 数值或设置 |
|---|---|
| 优化目标 | 最小化柔顺度：`ComplianceObjective(..., state_variable='u', diff_mode='manual')` |
| 体积约束 | 物理体积分数 $0.12$：`VOLFRAC = 0.12` |
| 初始密度 | 均匀取 $0.12$：`rho = bm.full((n_cells, ), VOLFRAC, ...)` |
| 材料插值 | 修正 SIMP：`interpolation_method='msimp'` |
| SIMP 惩罚指数 $q$ | $3$：`PENALTY = 3.0` |
| 密度过滤 | `filter_type='density'` |
| 过滤半径 | $3h=3$：`FILTER_RADIUS_CELLS = 3.0` |
| 灵敏度过滤 | `filter_objective_sensitivities(...)`，`filter_constraint_sensitivities(...)` |
| 优化器 | OC：`OCOptimizer.update_design_variable(..., **OC_OPTIONS)` |
| 移动限 | $0.2$：`move_limit=0.2` |
| 阻尼指数 | $0.5$：`damping_coef=0.5` |
| 设计变量下界 | $0$：`design_variable_min=0.0` |
| Lagrange 乘子 | 二分法：`initial_lambda=1.0e9`，`bisection_tol=1.0e-3` |
| 体积可行性容差 | $10^{-6}$：`VOLUME_TOLERANCE = 1.0e-6` |
| 收敛判据 | 柔顺度相对变化连续 5 轮小于 $2\times10^{-4}$：`TOLERANCE = 2.0e-4`，`CONVERGENCE_WINDOW = 5` |
| 最大迭代数 | 由 `--max-iter` 指定，实际取值记入 `config.json` |

## 3. 结果对照

```bash
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace full_trace --threads 1
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace linear_corner --threads 1
```

### 3.1 linear_corner 近似的正确性

在同一物理密度场上分别用 full_trace 与 linear_corner 各做一次分析（固定密度，不过滤、不做 OC 更新，求解配置同优化运行），核对第 1.3 节的 Ritz 关系：$C_C\le C_\Gamma$；按能量范数直接算的 $\varepsilon_E=\|U^\Gamma-U^C\|_K/\|U^\Gamma\|_K$ 与 $\sqrt{\varepsilon_C}$ 吻合到 CG 容差；恢复后的 $U^C$ 代回细网格方程，残差 $r=KU^C-F$ 的内部分量 $r_I$ 在 CG 容差量级，接口分量 $r_\Gamma$ 非零但与角点子空间正交，$P^{\mathsf T}r_\Gamma$ 在自由角点分量上为 CG 容差量级。密度场取初始均匀密度（两侧首轮共用）。

| 物理密度场 | $C_\Gamma$ | $C_C$ | $\varepsilon_C$ | $\varepsilon_E$ | $\|r_I\|_2/\|F\|_2$ | $\|P^{\mathsf T}r_\Gamma\|_2/\|P^{\mathsf T}F_\Gamma\|_2$ |
|---|---|---|---|---|---|---|
| 初始均匀密度，$m=5$（$42\times7\times7$） | $2562.4025$ | $1238.0029$ | $0.517$ | $0.719$ | $3.6\times10^{-13}$ | $7.7\times10^{-7}$ |
| 初始均匀密度，$m=7$（$30\times5\times5$） | $2562.4025$ | $1132.5640$ | $0.558$ | $0.747$ | $3.7\times10^{-13}$ | $7.7\times10^{-7}$ |

残差两列为 linear_corner 的值，投影只计自由角点分量；未投影的接口残差 $\|r_\Gamma\|_2/\|F\|_2$ 为 $1.08$、$1.06$，投影后与角点系统 CG 的相对残差一致（相对差小于 $10^{-8}$），即 $K_C=P^{\mathsf T}K_\Gamma P$、$F_C=P^{\mathsf T}F_\Gamma$ 在细网格方程上成立。full_trace 的内部残差为 $2.2\times10^{-12}$，接口残差 $9.3\times10^{-7}$ 即 CG 的相对残差。均匀密度 $m=5$ 行与两次 3 轮探针首轮的柔顺度一致（相对差小于 $10^{-10}$）；$\varepsilon_E$ 与 $\sqrt{\varepsilon_C}$ 的相对差为 $1.9\times10^{-8}$。$m=7$ 的 $C_\Gamma$ 与 $m=5$ 相对差 $1.7\times10^{-11}$，full_trace 不依赖子结构划分；子结构变大后 $\varepsilon_C$ 由 $0.517$ 增至 $0.558$，两种划分不嵌套，此为观察而非理论保证。重分析由 [`reanalyze_full_trace_vs_linear_corner.py`](reanalyze_full_trace_vs_linear_corner.py) 完成。

均匀密度上 linear_corner 把柔顺度低估一半。本节只核对近似的性质是否与理论一致，不以误差大小作判据；近似对设计的影响见 3.2，成本见第 4 节。

### 3.2 优化结果

| 指标 | full_trace | linear_corner |
|---|---|---|
| 接口自由度（约束前） | $425\,232$ | $8\,256$ |
| 终止 / 轮数 | 待填 | 待填 |
| 最终 $C_{\mathrm{opt}}$ | 待填 | 待填 |
| 最终体积分数 | 待填 | 待填 |
| 每轮 CG 步数（首轮 / 第 2 轮起均值 / 最大） | 待填 | 待填 |
| 末轮 CG 相对残差 | 待填 | 待填 |
| 末轮 z 向物理密度对称误差（诊断） | 待填 | 待填 |

两份设计的质量以同一精确模型下的真实柔顺度 $C_\Gamma(\rho_\Gamma)$、$C_\Gamma(\rho_C)$ 比较，并同时看体积分数：前者即 full_trace 运行的最终 $C_{\mathrm{opt}}$（停止时不再做 OC 更新，最终密度即末轮分析所用密度），后者对 $\rho_C$ 用 full_trace 重分析一次得到；同一次重分析给出的 $C_C(\rho_C)$ 应与 linear_corner 自报的 $C_{\mathrm{opt}}$ 一致。linear_corner 自己报告的 $C_{\mathrm{opt}}=C_C(\rho_C)$ 低估真实柔顺度，不用于设计比较。

| 设计差异 | linear_corner 相对 full_trace |
|---|---|
| 真实柔顺度相对变化 $\bigl(C_\Gamma(\rho_C)-C_\Gamma(\rho_\Gamma)\bigr)/C_\Gamma(\rho_\Gamma)$ | 待重分析 |
| 物理密度相对 $L^2$ 差 $\|\rho_C-\rho_\Gamma\|_2/\|\rho_\Gamma\|_2$ | 待填 |
| 阈值 $0.5$ 的二值拓扑不一致单元比例 | 待填 |

图：两侧柔顺度随轮数的收敛曲线（同图，纵轴对数）；两侧最终拓扑（同视角、同色标）。文件名待定。

两侧轨迹从首轮起就分开（探针第 2、3 轮柔顺度 full_trace 为 $655.05$、$358.87$，linear_corner 为 $571.70$、$315.75$），各轮密度不再相同，逐轮柔顺度之差不是 Ritz 误差，只反映轨迹差异。

数据来源：`results/full_trace_hex_210x35x35_m5_cg_jacobi_t1/`、`results/linear_corner_hex_210x35x35_m5_cg_jacobi_t1/`，对 $\rho_C$ 的重分析结果 `results/reanalysis_*/`（目录名待定）。

## 4. 计算成本

主表均对照 CG + Jacobi. 即使阶段功能相同, 也分别记录两侧实测值. 不同优化轮数会影响总成本, 应与每轮成本共同解读.

### 4.1 耗时 (s)

准备阶段取 `summary.setup_stages` 中的 `seconds`. 迭代阶段取第 2 轮起的均值, 首轮另列; 仅一轮时稳态指标为“不适用”.

准备阶段:

| 阶段 | full_trace | linear_corner | 说明 |
|---|---|---|---|
| 网格 | 待填 | 待填 | 细网格与子结构布局 |
| 分析器 | 待填 | 待填 | 材料、参考子结构与接口空间 |
| 过滤矩阵 | 待填 | 待填 | 密度过滤矩阵 |
| 目标与约束 | 待填 | 待填 | 目标、体积约束与梯度准备 |

迭代阶段:

| 阶段 | full_trace | linear_corner | 说明 |
|---|---|---|---|
| 过滤 | 待填 | 待填 | 设计密度过滤为物理密度 |
| 组装 | 待填 | 待填 | 局部组装、内部消元与接口组装 |
| 边界条件 | 待填 | 待填 | 载荷与边界条件处理 |
| 求解 | 待填 | 待填 | 接口求解及全场位移恢复 |
| 灵敏度 | 待填 | 待填 | 柔顺度与灵敏度计算 |
| 输出 | 待填 | 待填 | VTU/PVD 输出 |
| OC | 待填 | 待填 | 仅统计实际执行 OC 的轮次 |
| 首轮组装 | 待填 | 待填 | 单列首次缓存建立成本 |
| 首轮求解及恢复 | 待填 | 待填 | 与稳态均值区分 |

整个流程:

| 项目 | full_trace | linear_corner | 说明 |
|---|---|---|---|
| 准备合计 / s | 待填 | 待填 | 准备阶段耗时之和 |
| 每轮合计 / s | 待填 | 待填 | 逐轮实际阶段耗时求和, 再对第 2 轮起取均值 |
| 总耗时 / min | 待填 | 待填 | 准备合计加 `summary.total_seconds`, 换算为分钟 |
| 最终设计 FEM 重分析 / s | 待重分析 | 待重分析 | 独立记录, 不混入优化耗时 |

末轮未执行 OC 时不补算 OC. OC 阶段均值仅含实际执行的轮次, 因此每轮合计不能直接取表中各阶段均值之和. 阶段和与包含阶段外开销的 `iteration_seconds` 区分记录.

数据来源: 两侧实际输出目录的 `summary.json` 与 `history.json`, 路径待填.

### 4.2 内存 (GiB)

统计进程 RSS. 下表保留阶段峰值的双列记录, 不在缺少数据时推断常驻与临时内存组成. 阶段峰值为 RSS 绝对值, 不可相加; 后续若补充分项解释, 须注明测量口径与数据依据.

准备阶段:

| 阶段 | 峰值 full_trace | 峰值 linear_corner | 说明 |
|---|---|---|---|
| 网格 | 待填 | 待填 | 网格与布局构造期间 |
| 分析器 | 待填 | 待填 | 分析器与接口空间构造期间 |
| 过滤矩阵 | 待填 | 待填 | 过滤矩阵构造期间 |
| 目标与约束 | 待填 | 待填 | 目标与约束准备期间 |

迭代阶段 (分别取各阶段跨全部优化轮次的最大峰值):

| 阶段 | 峰值 full_trace | 峰值 linear_corner | 说明 |
|---|---|---|---|
| 过滤 | 待填 | 待填 | 物理密度计算 |
| 组装 | 待填 | 待填 | 局部缩聚、接口 CSR 及构造中间量 |
| 边界条件 | 待填 | 待填 | 载荷与边界条件处理 |
| 求解 | 待填 | 待填 | 接口求解及全场位移恢复 |
| 灵敏度 | 待填 | 待填 | 单元级能量与梯度计算 |
| 输出 | 待填 | 待填 | 场文件写出 |
| OC | 待填 | 待填 | 仅包含实际执行的轮次 |

整个流程:

| 项目 | full_trace | linear_corner | 说明 |
|---|---|---|---|
| 峰值: 准备阶段 | 待填 | 待填 | 准备各阶段 `peak_gib` 的最大值 |
| 峰值: 迭代阶段 | 待填 | 待填 | 实际迭代阶段 `peak_gib` 的最大值 |
| 峰值: 全程 (入口记录) | 待填 | 待填 | `summary.process_peak_rss_gib`, 沿用入口记录口径 |

数据来源: 两侧 `summary.setup_stages`、`history[].stages` 与 `summary.process_peak_rss_gib`. 阶段明细与汇总按各自字段记录, 有差异时先核对采样与汇总时点.

可选的 full_trace + GMG 结果另行补充, 沿用以上统计口径, 与主对照分开记录. 其与 linear_corner + Jacobi 的差异包含接口和求解配置两方面影响.

## 参考文献

[1] Huang2023, 第 4 节公共设置与第 4.1 节 MBB 梁算例, 图 5–7. 沿用现有实验文档的文献标识.
