# 子结构与传统有限元方法的结果对照

硬件：Intel Core i9-14900KF（24 核 32 线程，8P+16E），运行内存 45 GiB；NVIDIA GeForce RTX 5080，显存 16 GB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（内置 scipy-openblas 0.3.34），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.13.0（CUDA 13.0）；PyMUMPS 0.4.0，链接系统 MUMPS 5.6.2（OpenBLAS 0.3.26、Open MPI 4.1.6、ScaLAPACK 2.2.1，单进程运行）；pyevtk 1.7.0。

## 1. 研究对象

对照相同细网格下，传统有限元 `fem` 与精确子结构 `substructure`（`full_trace`）在 Huang2023 三维 MBB 梁上的拓扑优化结果。

两侧采用相同的细网格、单元与材料插值，过滤、灵敏度与 OC 更新也完全相同，差别只在线性系统的求解路径。`fem` 直接在全场上求解；`substructure` 先在每个子结构内精确消去内部自由度，在全局边界上求解，再回代内部位移。

| 项目 | fem | substructure |
|---|---|---|
| 刚度算子 | 细网格单元刚度直接组装为全局矩阵 $K$ | 各子结构 Schur 补 $K_s^j$ 组装为全局边界矩阵 $K_\Gamma$ |
| 未知量 | 全场位移 $U$ | 全局边界位移 $U_\Gamma$ |
| 全场位移 | 求解直接得到 | 由 $U_\Gamma$ 逐子结构回代内部位移 $u_i^j=-(K_{ii}^j)^{-1}K_{ib}^ju_b^j$ |

### 1.1 矩阵组装与局部缩聚

`fem`：将细网格单元刚度 $K_e(\rho_e)$ 直接组装为细网格总刚：

$$
K=\sum_e A_e^{\mathsf T}K_e(\rho_e)A_e\in\mathbb R^{N_h\times N_h}.
$$

`substructure`：首先将细网格单元刚度 $K_e(\rho_e)$ 组装为子结构总刚 $K^j$，并按边界 $b$、内部 $i$ 分块：

$$
K^j=\sum_{e\in\Omega_j}(G_e^j)^{\mathsf T}K_e(\rho_e)\,G_e^j
=\begin{bmatrix}K_{bb}^j&K_{bi}^j\\K_{ib}^j&K_{ii}^j\end{bmatrix}\in\mathbb R^{n_d\times n_d}
$$

再消去内部自由度 $i$，得到子结构的 Schur 补 $K_s^j$，即边界自由度 $b$ 上的缩聚刚度矩阵：

$$
K_s^j=K_{bb}^j-K_{bi}^j(K_{ii}^j)^{-1}K_{ib}^j.
$$

将各子结构的缩聚刚度矩阵组装为全局边界总刚：

$$
K_\Gamma=\sum_j(A_b^j)^{\mathsf T} K_s^j A_b^j\in\mathbb R^{N_\Gamma\times N_\Gamma}.
$$

**$K_\Gamma$ 的稀疏结构**。设共 $M$ 个子结构，每个含 $m\times m\times m$ 个三线性六面体细单元。$K$ 中每个节点只与同属一个单元的 $3\times3\times3$ 个节点耦合。$K_s^j$ 则是满阵：子结构内部节点连通，且每个边界节点都与内部节点相邻，消去内部自由度后，任意两个边界节点都经内部耦合。因此 $K_\Gamma$ 在子结构内稠密、在子结构间稀疏，与 $K$ 的对比如下：

| | $K$ | $K_\Gamma$ |
|---|---|---|
| 局部矩阵 | $K_e\in\mathbb R^{24\times24}$ | $K_s^j\in\mathbb R^{n_b\times n_b}$，$n_b=3(6m^2+2)$ |
| 维数 | $N_h$ | $N_\Gamma=N_h-3M(m-1)^3\approx\dfrac3m N_h$ |
| 每行非零元 | $81$（内部节点） | $3(11m^2-2m+3)$（公共面内节点） |
| 非零元总数 | $\approx81N_h$ | $\approx\dfrac3m N_h\cdot33m^2\approx1.2m\cdot81N_h$ |

上述非零元增长对应显式存储 Schur 补的路径。也可保留全部边界自由度，通过内部线性系统求解隐式计算 $K_\Gamma v$，不形成局部 Schur 补或全局接口 CSR；这不引入 `linear_corner` 的边界插值降阶，但需承担每次算子作用中的内部求解及相关缓存成本。方法差异与实测显存结果见 [显式与局部刚度隐式 Schur 补的显存对比](substructure_schur_explicit_vs_implicit.md)。下文实测结果仍属于显式路径，不代表隐式方法的性能。

### 1.2 求解与位移恢复

`fem` 直接求解得到全场位移：

$$
KU=F.
$$

`substructure` 首先得到全局边界位移：

$$
K_\Gamma U_\Gamma=F_\Gamma.
$$

提取各子结构的边界位移，回代内部位移：

$$
u_b^j=A_b^jU_\Gamma,\qquad
u_i^j=-(K_{ii}^j)^{-1}K_{ib}^ju_b^j.
$$

将边界与内部位移按全局编号恢复为全场位移 $U$：

$$
U=R_\Gamma^{\mathsf T} U_\Gamma+\sum_j(R_i^j)^{\mathsf T} u_i^j.
$$

### 1.3 代数等价

本算例内部载荷为零，原载荷的边界分量与缩聚后的载荷相同，统一记为 $F_\Gamma$。线性方程组的分块表示为：

$$
U=\begin{bmatrix}U_\Gamma\\U_I\end{bmatrix},\qquad
K=\begin{bmatrix}
K_{\Gamma\Gamma}&K_{\Gamma I}\\
K_{I\Gamma}&K_{II}
\end{bmatrix},\qquad
F=\begin{bmatrix}
F_\Gamma\\
0
\end{bmatrix}.
$$

消去内部自由度后，得到全局边界刚度矩阵：

$$
K_\Gamma=K_{\Gamma\Gamma}-K_{\Gamma I}K_{II}^{-1}K_{I\Gamma}.
$$

结合内部位移回代：

$$
KU=F
\quad\Longleftrightarrow\quad
\begin{cases}
K_\Gamma U_\Gamma=F_\Gamma,\\
U_I=-K_{II}^{-1}K_{I\Gamma}U_\Gamma.
\end{cases}
$$

在相同离散、材料、载荷与边界条件下，`full_trace` 保留全部边界自由度，仅精确消去内部自由度。因此，精确缩聚与回代得到的全场解与传统有限元解代数等价；实际数值差异来自浮点运算与线性求解误差。


## 2. 实验设置

实验脚本为 [`run_substructure_vs_fem.py`](run_substructure_vs_fem.py)。

计算后端与设备由 `--backend`、`--device` 指定，默认 `numpy`、`cpu`，`cuda` 须配 `pytorch`；3.1 节采用 NumPy / CPU，3.2 节的 300、390 档采用 PyTorch，分别在 GPU 与 CPU 上运行。单线程：`NUM_THREADS = 1`，脚本在导入 numpy/scipy 之前把全部 BLAS / OpenMP 线程变量设为 1，取值记入 `config.json` 的 `threads`。

### 2.1 模型参数

![MBB 梁的几何、载荷与支承](../topopt_piml_substructure/assets/Huang2023_Fig5.png)

采用整体三维 MBB 梁模型，长、高、宽分别记为 $L_x,L_y,L_z$，比例为 $6:1:1$，以细单元边长为长度单位（$h=1$），计算域 $\Omega=[0,L_x]\times[0,L_y]\times[0,L_z]$ 的尺寸由网格剖分数确定。顶面中心施加竖直向下的单位集中力 $F=1$。支承取在两端底边的四个角点：左端两角点铰支（$u_x=u_y=u_z=0$），右端两角点滚支（沿 $x$ 可滚动，$u_y=u_z=0$）。

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
| 分析网格 | 结构化六面体网格，各方向单元数由 `--grid` 设定，须满足 $6:1:1$；子结构运行中也可由 `--n-sub` 与 `--n-fine` 推出：`mesh = HexahedronMesh.from_box(box=domain, nx=nx, ny=ny, nz=nz, ...)` |

**子结构缩聚**

| 参数 | 数值或设置 |
|---|---|
| 子结构划分 | 子结构数由 `--n-sub` 设定，每个子结构含 $m\times m\times m$ 个细单元，$m$ 由 `--n-fine` 设定，须满足 $\text{grid}=n_{\text{sub}}\times m$，缺省项由其余两项推出：`StructuredSubstructureLayout(..., n_sub=n_sub, n_fine=(m, m, m), full_mesh=mesh)` |
| 边界位移表示 | 保留全部边界自由度，$\Psi=I$：`SubstructureAnalyzer(..., trace='full_trace')` |
| 分批大小 | 每批处理 256 个子结构：`chunk_size=256` |

**线性求解参数**

| 参数 | 数值或设置 |
|---|---|
| 求解器 | CG：`SOLVER = 'cg'`；预条件子由 `--preconditioner` 指定，可选 `none`、`jacobi`、`gmg`（默认 `gmg`），`fem` 另可选 `scipy`、`mumps` 直接法；实际取值记入 `config.json` 与结果表 |
| 停机准则 | 相对右端项的容差 $10^{-6}$：`CG_TOLERANCE = 1.0e-6`，`rtol=CG_TOLERANCE`，`atol=0`，`rtol_reference='rhs'` |
| 最大 CG 步数 | `CG_MAXITER = 20000` |
| 迭代初值 | 首轮为零向量；第 2 轮起取上一轮的解：`x0=previous_displacement`，`fem` 为全场位移，`full_trace` 为全局边界位移 |

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

### 3.1 结果一致性

```bash
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method fem --preconditioner jacobi --grid 210 35 35 --max-iter 3
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method substructure --preconditioner jacobi --grid 210 35 35 --max-iter 3
```

| 指标 | fem | substructure |
|---|---|---|
| 终止 | 达到迭代上限，3 轮 | 达到迭代上限，3 轮 |
| 逐轮柔顺度 $C$ | $2562.4025186829$，$655.0507016027$，$358.8702793925$ | $2562.4025185972$，$655.0505983310$，$358.8704086800$ |
| 第 3 轮体积分数 | $0.119954$ | $0.119954$ |
| 每轮 CG 步数 | 1295，2252，2299 | 787，1376，1417 |
| 末轮 CG 相对残差 | $9.86\times10^{-7}$ | $9.26\times10^{-7}$ |

两侧逐轮柔顺度相对差不超过 $3.6\times10^{-7}$，第 3 轮全场位移相对 $L^2$ 差为 $4.5\times10^{-7}$，均与 CG 容差 $10^{-6}$ 相符。

### 3.2 内存占用

**非零元个数**

| 网格 | $K$ 维数 | $K$ 非零元 | $K_\Gamma$ 维数 | $K_\Gamma$ 非零元 | 非零元比 |
|---|---|---|---|---|---|
| $210\times35\times35$ | $820\,368$ | $6.38\times10^7$ | $425\,232$ | $3.65\times10^8$ | 5.72 |
| $240\times40\times40$ | $1\,215\,363$ | $9.50\times10^7$ | $625\,539$ | $5.43\times10^8$ | 5.72 |
| $390\times65\times65$ | $5\,109\,588$ | $4.05\times10^8$ | $2\,578\,644$ | $2.32\times10^9$ | 5.72 |

PyTorch 后端：`--backend pytorch`；单线程：`--threads 1`。

```bash
# GPU
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method fem --preconditioner jacobi --grid 240 40 40 --max-iter 3 --backend pytorch --device cuda --gpu-memory-limit 14
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method substructure --preconditioner jacobi --grid 240 40 40 --max-iter 3 --backend pytorch --device cuda --gpu-memory-limit 14
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method fem --preconditioner jacobi --grid 390 65 65 --max-iter 3 --backend pytorch --device cuda --gpu-memory-limit 14
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method substructure --preconditioner jacobi --grid 390 65 65 --max-iter 3 --backend pytorch --device cuda --gpu-memory-limit 14
```

单位 GiB，取第 1 轮各阶段 torch.cuda.max_memory_allocated 的记录。

| 部分 | fem 240 | substructure 240 | fem 390 | substructure 390 |
|---|---|---|---|---|
| 共有部分（常驻） | 0.67 | 0.71 | 2.86 | 3.05 |
| 刚度算子（常驻） | 1.64 | 6.62 | 未测 | 未测 |
| 边界条件（常驻） | 0.05 | 0.08 | 未执行 | 未执行 |
| 装配阶段峰值 | 3.78 | 10.46 | > 14 | > 14 |
| 求解阶段峰值 | 3.79 | 10.56 | 未执行 | 未执行 |

表中取第 1 轮。第 2 轮起 CSR 骨架已缓存，装配只做数值阶段，但各阶段开始时显存中仍保留上一轮的载荷、Dirichlet 掩码与热启动初值等跨轮数据，阶段峰值略高于第 1 轮，第 3 轮起基本不再增长：fem 240 全程峰值 3.85（第 3 轮装配），比表中高 0.06；substructure 240 全程峰值 10.59（第 3 轮求解），比表中高 0.03。

```bash
# CPU
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method fem --preconditioner jacobi --grid 240 40 40 --max-iter 3 --backend pytorch --device cpu --cpu-memory-limit 40
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method substructure --preconditioner jacobi --grid 240 40 40 --max-iter 3 --backend pytorch --device cpu --cpu-memory-limit 40
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method fem --preconditioner jacobi --grid 390 65 65 --max-iter 3 --backend pytorch --device cpu --cpu-memory-limit 40
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method substructure --preconditioner jacobi --grid 390 65 65 --max-iter 3 --backend pytorch --device cpu --cpu-memory-limit 40
```

单位 GiB，取第 1 轮进程 RSS 的记录。

| 部分 | fem 240 | substructure 240 | fem 390 | substructure 390 |
|---|---|---|---|---|
| 共有部分（常驻） | 待测 | 待测 | 待测 | 待测 |
| 刚度算子（常驻） | 待测 | 待测 | 待测 | 待测 |
| 边界条件（常驻） | 待测 | 待测 | 待测 | 待测 |
| 装配阶段峰值 | 待测 | 待测 | 待测 | 待测 |
| 求解阶段峰值 | 待测 | 待测 | 待测 | 待测 |

数据来源：`outputs/pytorch_{cuda,cpu}_fa_hex_{240x40x40,390x65x65}_cg_jacobi/`、`outputs/pytorch_{cuda,cpu}_full_trace_hex_{240x40x40,390x65x65}_m5_cg_jacobi/`（待固化到 `results/`）。
