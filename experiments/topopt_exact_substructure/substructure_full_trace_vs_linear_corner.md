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
u_i^j=N_i^jq^j,
$$

其中回代矩阵随 $\Psi$ 不同而不同：

$$
N_i^j=-(K_{ii}^j)^{-1}K_{ib}^j\Psi=
\begin{cases}
-(K_{ii}^j)^{-1}K_{ib}^j\in\mathbb R^{n_i\times n_b}, & \texttt{full\_trace}\ (\Psi=I),\\[4pt]
-(K_{ii}^j)^{-1}K_{ib}^jL\in\mathbb R^{n_i\times24}, & \texttt{linear\_corner}\ (\Psi=L).
\end{cases}
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

误差的阶。记精确解为 $u$，细网格空间为 $V_h$，角点插值子空间为 $V_H=\{PU_C\}\subset V_h$，空间嵌套给出 $\|u-U^C\|_K^2=\|u-U^\Gamma\|_K^2+\|U^\Gamma-U^C\|_K^2$。第一项是细网格 FEM 的误差，解光滑时对 $h$ 一阶。第二项：$V_H$ 在子结构面上的迹空间与尺寸为 $H$ 的 $Q_1$ 粗网格元相同，内部是同一迹的能量极小延拓，内部载荷为零时 $U^\Gamma$ 在子结构内离散调和，故 $\|U^\Gamma-U^C\|_K\le\inf_{V_H\in Q_1(H)}\|U^\Gamma-V_H\|_K\le C\,H$，对 $H$ 一阶；$m=1$ 时两侧恒等。$m$ 固定、$h\to0$ 时两种方法对精确解都是能量范数一阶，linear_corner 的常数约为 $m$ 倍。


## 2. 实验设置

实验脚本为 [`run_substructure_full_trace_vs_linear_corner.py`](run_substructure_full_trace_vs_linear_corner.py)。

计算后端与设备由 `--backend`、`--device` 指定，默认 `numpy`、`cpu`，`cuda` 须配 `pytorch`；3.1、3.2 节的 240、390 档采用 PyTorch，分别在 GPU 与 CPU 上运行；单线程：线程数由 `--threads` 指定，默认 `NUM_THREADS = 1`，脚本在导入 numpy/scipy 之前把全部 BLAS / OpenMP 线程变量设为该值，取值记入 `config.json` 的 `threads`，输出目录名带后缀 `_t1`。

### 2.1 模型参数

模型为 [`FullMBBBeamPoint3d`](../../docs/problems/benchmarks/full-mbb-beam-point-3d.md)：整体三维 MBB 梁，长、高、宽分别记为 $L_x,L_y,L_z$，比例为 $6:1:1$，以细单元边长为长度单位（$h=1$），计算域 $\Omega=[0,L_x]\times[0,L_y]\times[0,L_z]$ 的尺寸由网格剖分数确定。顶面中心施加竖直向下的单位集中力 $F=1$。支承取在两端底边的四个角点：左端两角点铰支（$u_x=u_y=u_z=0$），右端两角点滚支（沿 $x$ 可滚动，$u_y=u_z=0$）。

| 参数 | 数值或设置 |
|---|---|
| 几何与计算域 | 整体三维 MBB 梁，$L_x:L_y:L_z=6:1:1$，尺寸由网格确定：`domain = (0, nx*h, 0, ny*h, 0, nz*h)`，`FullMBBBeamPoint3d(domain=domain, ...)` |
| 长度约定 | 以细单元边长为长度单位，$h=1$：`ELEMENT_SIZE = 1.0`；网格须满足 $n_x:n_y:n_z=6:1:1$ |
| 载荷 | 顶面中心集中力，沿 $y$ 方向向下，合力为 $-1$：`LOAD = -1.0` |
| 支承 | 齐次 Dirichlet 边界条件：左端两底角点 $u_x=u_y=u_z=0$，右端两底角点 $u_y=u_z=0$：`FullMBBBeamPoint3d` |
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

### 3.1 内存占用

**非零元个数**

| 网格 | $K_\Gamma$ 维数 | $K_\Gamma$ 非零元 | $K_C$ 维数 | $K_C$ 非零元 | 维数比 | 非零元比 |
|---|---|---|---|---|---|---|
| $210\times35\times35$ | $425\,232$ | $3.65\times10^8$ | $8\,256$ | $5.53\times10^5$ | 51.5 | 660 |
| $240\times40\times40$ | $625\,539$ | $5.43\times10^8$ | $11\,907$ | $8.16\times10^5$ | 52.5 | 666 |
| $390\times65\times65$ | $2\,578\,644$ | $2.32\times10^9$ | $46\,452$ | $3.38\times10^6$ | 55.5 | 685 |

PyTorch 后端：`--backend pytorch`；单线程：`--threads 1`。

```bash
# GPU
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace full_trace --grid 240 40 40 --max-iter 3 --backend pytorch --device cuda --gpu-memory-limit 14 --threads 1
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace linear_corner --grid 240 40 40 --max-iter 3 --backend pytorch --device cuda --gpu-memory-limit 14 --threads 1
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace full_trace --grid 390 65 65 --max-iter 3 --backend pytorch --device cuda --gpu-memory-limit 14 --threads 1
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace linear_corner --grid 390 65 65 --max-iter 3 --backend pytorch --device cuda --gpu-memory-limit 14 --threads 1
```

单位 GiB，取第 1 轮各阶段 torch.cuda.max_memory_allocated 的记录。

| 部分 | full_trace 240 | linear_corner 240 | full_trace 390 | linear_corner 390 |
|---|---|---|---|---|
| 共有部分（常驻） | 0.71 | 0.69 | 3.05 | 2.94 |
| 刚度算子（常驻） | 6.62 | 0.02 | 未测 | 0.10 |
| 恢复矩阵（常驻） | 0 | 0.11 | 0 | 0.45 |
| 边界条件（常驻） | 0.08 | 0.05 | 未执行 | 0.20 |
| 装配阶段峰值 | 10.46 | 2.77 | > 14 | 5.44 |
| 求解阶段峰值 | 7.47 | 0.87 | 未执行 | 3.70 |
| 恢复阶段峰值 | 10.56 | 0.88 | 未执行 | 3.74 |

表中取第 1 轮。full_trace 的恢复阶段峰值高于装配阶段峰值，是逐批重算局部组装与缩聚的瞬时量叠在 7.41 GiB 常驻之上所致；linear_corner 的装配阶段峰值 2.77 远高于其常驻 0.13，为分批局部缩聚的瞬时量，与网格规模无关。第 2 轮起阶段峰值略高，第 3 轮全程峰值 full_trace 240 为 10.58、linear_corner 240 为 2.82、linear_corner 390 为 5.68，比表中高 0.02、0.05、0.24。full_trace 390 在第 1 轮装配阶段触及 14 GiB 上限，失败点记在 `failure.json`，共有部分取其过滤阶段结束时的占用。

两侧回代矩阵 $N_i^j$（1.2 节）的规模相差 $n_b/24$ 倍（$m=5$ 时 $n_i=192$，$n_b=456$，相差 19 倍），恢复矩阵一行由此而来：`linear_corner` 的 $N_i^j$ 合计 $n_{\text{sub}}\times n_i\times24$ 个数（210、240、390 档为 0.071、0.105、0.453 GiB），装配时缓存，恢复只做回代，表中按此解析值填写；`full_trace` 若缓存需 2.0 GiB（240 档）、8.6 GiB（390 档），故不缓存，恢复时逐批重算 $(K_{ii}^j)^{-1}K_{ib}^j$，为瞬时开销，可由 `chunk_size` 调节。

```bash
# CPU
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace full_trace --grid 240 40 40 --max-iter 3 --backend pytorch --device cpu --cpu-memory-limit 40 --threads 1
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace linear_corner --grid 240 40 40 --max-iter 3 --backend pytorch --device cpu --cpu-memory-limit 40 --threads 1
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace full_trace --grid 390 65 65 --max-iter 3 --backend pytorch --device cpu --cpu-memory-limit 40 --threads 1
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace linear_corner --grid 390 65 65 --max-iter 3 --backend pytorch --device cpu --cpu-memory-limit 40 --threads 1
```

单位 GiB，取第 1 轮进程 RSS 的记录。

| 部分 | full_trace 240 | linear_corner 240 | full_trace 390 | linear_corner 390 |
|---|---|---|---|---|
| 共有部分（常驻） | 2.03 | 2.02 | 6.38 | 6.32 |
| 刚度算子（常驻） | 6.58 | 0.08 | 未测 | 0.04 |
| 恢复矩阵（常驻） | 0 | 0.11 | 0 | 0.45 |
| 边界条件（常驻） | 0.12 | 0.10 | 未执行 | 0.19 |
| 装配阶段峰值 | 11.74 | 4.15 | > 40 | 8.75 |
| 求解阶段峰值 | 8.78 | 2.31 | 未执行 | 7.01 |
| 恢复阶段峰值 | 11.90 | 2.31 | 未执行 | 7.04 |

表中取第 1 轮。CPU 上的共有部分比 GPU 高，为进程本身与已导入的库（含 torch）的驻留内存及主机侧网格、过滤矩阵的副本；其余各行与 GPU 表的规律相同。全程峰值 full_trace 240 为 11.90（第 1 轮恢复阶段）、linear_corner 240 为 4.26（第 2 轮）、linear_corner 390 为 9.82（准备阶段构造过滤矩阵时）。full_trace 390 在第 1 轮装配阶段触及 40 GiB 上限（阶段内峰值 39.45），失败点记在 `failure.json`，共有部分取其过滤阶段结束时的占用。

数据来源：`outputs/pytorch_{cuda,cpu}_{full_trace,linear_corner}_hex_{240x40x40,390x65x65}_m5_cg_jacobi_t1/`（待固化到 `results/`）。

### 3.2 linear_corner 下的完整优化结果

```bash
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace linear_corner --grid 390 65 65 --backend pytorch --device cuda --gpu-memory-limit 14 --threads 1
```

| 指标 | linear_corner 390 |
|---|---|
| 终止状态 / 轮数 | 收敛 / 68 |
| 最终 $C$ | $6.974166$ |
| 最终物理体积分数 | $0.1199990$ |
| CG 步数，首轮 / 均值 / 最大 / 总计 | 471 / 1050 / 1633 / $71\,421$ |
| 真残差，末轮 / 最大 | $9.9\times10^{-7}$ / $1.0\times10^{-6}$ |
| 物理密度不对称量 $x$ / $z$ | $3.1\times10^{-4}$ / $2.9\times10^{-7}$ |
| 每轮耗时，装配 / 求解 / 恢复 / 整轮（均值） | 1.15 / 1.02 / 0.03 / 4.03 |
| 全程耗时，准备 / 优化循环 | 28 / 275 |
| 全程峰值显存 | 5.68 |
| 最终拓扑 | 下图，阈值 0.5 |

![linear_corner 390 最终拓扑](assets/linear_corner_hex_390x65x65_final_topology.png)

数据来源：`results/pytorch_cuda_linear_corner_hex_390x65x65_m5_cg_jacobi_t1/`（由 `outputs/` 同名目录固化，不含 VTU 帧）；拓扑图由 `tools/render_density_topology.py` 从其中的 `density_final.npy` 以默认视角渲染。
