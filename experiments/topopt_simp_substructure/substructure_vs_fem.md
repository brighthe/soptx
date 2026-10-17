# 子结构法与传统有限元法的拓扑优化结果对照

硬件：Intel Core i9-14900KF（24 核 32 线程，8P+16E），运行内存 45 GiB；NVIDIA GeForce RTX 5080，显存 16 GB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（内置 scipy-openblas 0.3.34），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.13.0（CUDA 13.0）；PyMUMPS 0.4.0，链接系统 MUMPS 5.6.2（OpenBLAS 0.3.26、Open MPI 4.1.6、ScaLAPACK 2.2.1，单进程运行）；pyevtk 1.7.0。

## 1. 研究对象

对照相同细网格下，传统有限元 `fem` 与精确子结构 `substructure`（`full_trace`）在 Huang2023 三维 MBB 梁上的拓扑优化结果，比较结果一致性、计算耗时与内存。

| 项目 | fem | substructure |
|---|---|---|
| 刚度算子 | 细网格全局 CSR 矩阵 $K$ | 各子结构局部 Schur 补装配为全局边界 CSR 矩阵 $K_\Gamma$ |
| 未知量 | 全场位移 $U$ | 全局边界位移 $U_\Gamma$ |

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

### 1.2 求解与内部位移回代

`fem` 直接求解得到全场位移：

$$
KU=F
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

本实验均采用 NumPy / CPU：`BACKEND = 'numpy'`，`DEVICE = 'cpu'`；单线程：`NUM_THREADS = 1`，脚本在导入 numpy/scipy 之前把全部 BLAS / OpenMP 线程变量设为 1，取值记入 `config.json` 的 `threads`。

### 2.1 模型参数

![MBB 梁的几何、载荷与支承](../topopt_piml_substructure/assets/Huang2023_Fig5.png)

采用整体三维 MBB 梁模型，长、高、宽分别记为 $L_x,L_y,L_z$，比例为 $6:1:1$，计算域为 $\Omega=[0,L_x]\times[0,L_y]\times[0,L_z]$。顶面中心施加竖直向下的单位集中力 $F=1$。支承取在两端底边的四个角点：左端两角点铰支（$u_x=u_y=u_z=0$），右端两角点滚支（$u_y=0$）。

| 参数 | 数值或设置 |
|---|---|
| 几何与计算域 | 整体三维 MBB 梁，长、高、宽为 $210,35,35$：`domain = (0, 210, 0, 35, 0, 35)`，`FullMBBBeamPoint3d(domain=domain, ...)` |
| 长度约定 | 由计算域与网格推出细单元边长 $h=210/210=1$：`spacing = domain_size / grid` |
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
| 分析网格 | 结构化六面体网格 $210\times35\times35$：`mesh = HexahedronMesh.from_box(box=domain, nx=210, ny=35, nz=35, ...)` |

**子结构缩聚**

| 参数 | 数值或设置 |
|---|---|
| 子结构划分 | $42\times7\times7$ 个子结构，每个含 $5\times5\times5$ 个细单元：`StructuredSubstructureLayout(..., n_sub=(42, 7, 7), n_fine=(5, 5, 5), full_mesh=mesh)` |
| 边界迹空间 | 保留全部边界自由度，$\boldsymbol\Psi=\boldsymbol I$：`SubstructureAnalyzer(..., trace='full_trace')` |
| 分批大小 | 每批处理 256 个子结构：`chunk_size=256` |

**线性求解参数**

| 参数 | 数值或设置 |
|---|---|
| 求解器 | CG + GMG（MGCG）：`SOLVER = 'cg'`，`PRECONDITIONER = 'gmg'` |
| 停机准则 | 绝对容差为 $10^{-6}\lVert F\rVert$：`CG_TOLERANCE = 1.0e-6`，`rtol=0` |
| 最大 CG 步数 | `CG_MAXITER = 20000` |
| 迭代初值 | 首轮均为零向量；第 2 轮起取上一轮拓扑优化的位移解：`x0=previous_displacement`，`fem` 使用全场位移，`full_trace` 使用全局边界位移 |

预条件子：FA 用细网格几何多重网格 $B\approx\tilde K^{-1}$（V 循环，每层 1 次光滑，粗层不超过 20 000 自由度时直接求解）。full_trace 不为 $\tilde K_\Gamma$ 另建层次，而是把同一个 $B$ 限制到接口：

$$
B_\Gamma=R\,B\,R^{T},\qquad R\in\{0,1\}^{N_\Gamma\times N_h}\ \text{为接口提取矩阵}
$$

依据是分块矩阵的恒等式 $R\tilde K^{-1}R^{T}=\tilde K_\Gamma^{-1}$（Schur 补的逆等于逆矩阵的对应块），因此 $B\approx\tilde K^{-1}$ 蕴含 $B_\Gamma\approx\tilde K_\Gamma^{-1}$。作用一次 $B_\Gamma$ 的主要代价仍是一次全场 V 循环：接口残差零延拓到全场，做一次全场 V 循环，取回接口分量。这解释了两侧 CG 步数相近。

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
| 最大迭代数 | $300$：`--max-iter` 默认值 `300` |

## 3. 结果对照

### 3.1 结果一致性

两侧优化轨迹逐轮一致，终止于同一轮，最终设计相同。

| 指标 | FA | full_trace |
|---|---|---|
| 终止 | 收敛，68 轮 | 收敛，68 轮 |
| 最终柔顺度 $C$ | $18.9095002249$ | $18.9095000516$ |
| 最终体积分数 | $0.119999$ | $0.119999$ |
| 每轮 CG 步数 | 首轮 13，最大 18，均值 13.7，总计 930 | 首轮 12，最大 16，均值 12.7，总计 862 |
| 末轮 CG 相对残差 | $5.73\times10^{-7}$ | $6.03\times10^{-7}$ |
| 物理密度 $z$ 向不对称量 | $1.8\times10^{-15}$ | $1.3\times10^{-15}$ |
| 设计密度 $z$ 向不对称量 | $0$ | $0$ |

数据来源：`results/fa_hex_210x35x35_cg/`、`results/full_trace_hex_210x35x35_m5_cg/` 的 `summary.json` 与 `history.json`。

两侧差异：

| 量 | 最大差 |
|---|---|
| 逐轮柔顺度相对差 | $1.3\times10^{-7}$ |
| 最终柔顺度相对差 | $9.2\times10^{-9}$ |
| 最终物理密度 | $5.0\times10^{-6}$ |
| 最终设计密度 | $6.0\times10^{-5}$ |
| 最终位移相对 $L^2$ 差 | $1.0\times10^{-7}$ |
| 阈值 $0.5$ 二值化拓扑 | 逐单元相同，实体单元 $29\,403$ 个 |


数据来源：柔顺度相对差来自两侧 `history.json`，其余来自 `results/final_fields.json` 。

### 3.2 计算成本

以下保留历史运行的原始数据与阶段说明。历史子结构约束处理计入求解阶段；当前 CG、直接法及一般线性约束的系统准备均已移至 `apply_bc`，新运行将其计入边界条件阶段。此处未重跑或重新分摊历史耗时与内存数据。

装配复杂度：三维 Q1 单元下，子结构规模与边长 $m$ 的关系为

$$
n_i=3(m-1)^3,\qquad n_b=3\bigl[(m+1)^3-(m-1)^3\bigr]=18m^2+6,\qquad N_s=\lvert\mathcal T_h\rvert/m^3
$$

| 步骤 | 运算量（散加个数 / flop） | 对 $\lvert\mathcal T_h\rvert$、$m$ 的阶 |
|---|---|---|
| FA，组装全局刚度 $K$ | $\lvert\mathcal T_h\rvert n_e^2$ | $\Theta(\lvert\mathcal T_h\rvert)$ |
| full_trace，稠密子结构局部组装 $K^j$ | $N_s(n_i+n_b)^2$ | $\Theta(\lvert\mathcal T_h\rvert m^3)$ |
| full_trace，子结构缩聚单刚 $K_s^j$ | $N_s\bigl(\tfrac23n_i^3+2n_i^2n_b+2n_in_b^2\bigr)$ | $\Theta(\lvert\mathcal T_h\rvert m^6)$ |
| full_trace，组装全局边界总刚 $K_\Gamma$ | $N_s n_b^2$ | $\Theta(\lvert\mathcal T_h\rvert m)$ |

求解复杂度：

| 项目 | FA | full_trace |
|---|---|---|
| 未知量 | $N_h=820\,368$ | $N_\Gamma=425\,232$ |
| 矩阵每行非零元 | 内部节点 27 个邻点，$\le 81$ | 面内节点与两个子结构的全部边界节点耦合，$3(2\times152-36)=804$；棱与角点节点更多，平均约 850 |
| 非零元总数 | $\approx 81N_h=6.6\times10^{7}$ | $\approx 850N_\Gamma=3.6\times10^{8}$，上界 $N_sn_b^2=4.3\times10^{8}$ |
| 每步 CG：矩阵向量积 | 1 次 $\tilde K$ | 1 次 $\tilde K_\Gamma$，约 5 倍 |
| 每步 CG：预条件 | 1 次全场 V 循环 | 1 次全场 V 循环，相同 |
| 热启动初值 | 上一轮全场位移 | 上一轮接口位移 |
| 求解后 | 无 | 恢复：按批重做局部装配与 $K_{ii}^j$ 分解回代 |
| 实测每轮求解（s，第 2 轮起均值） | 2.85 | 28.31 |

回代按批重新做局部装配与 $K_{ii}^j$ 的分解，不保存 $(K_{ii}^j)^{-1}K_{ib}^j$（每个子结构 $n_i\times n_b$，全部约 1.3 GiB），以重算换内存。

全局未知量减少不代表计算成本必然降低：局部缩聚使边界矩阵更稠密，还增加了内部位移恢复的工作。以下保留原有耗时与内存统计口径。

#### 3.2.1 耗时（s）

两侧相同的阶段：

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 准备：网格 | 1.16 | 生成节点与单元，eager 构造面、边及其关联数组 |
| 准备：过滤矩阵 | 1.26 | 构造过滤矩阵 $H$ |
| 准备：目标与约束 | 0.14 | 体积约束梯度及其过滤 |
| 迭代：过滤 | 0.03 | 设计密度过滤为物理密度 |
| 迭代：灵敏度 | 0.16 | 柔顺度 $c$、逐单元灵敏度 $\partial c/\partial\rho_e$ 及其过滤 |
| 迭代：输出 | 0.22 | 写出本轮 VTU 并更新 PVD |
| 迭代：OC | 0.05 | 更新设计变量 |

两侧不同的阶段：

| 阶段 | FA | full_trace | 说明 |
|---|---|---|---|
| 准备：分析器 | 0.03 | 0.30 | **FA**：构造材料、插值方案、函数空间与积分子<br>**full_trace**：在此之上另建各子结构网格、参考子结构与边界迹空间，并把载荷与支承投影到全局边界未知量 |
| 迭代：装配 | 0.59 | 29.19 | **FA**：把缓存的 $K^0$ 按单元系数缩放后散加进 CSR 值数组，见上文 $K$ 的组装式；首轮另建 CSR 骨架，2.8 s<br>**full_trace**：对 2 058 个子结构按批（256 个一批）做局部装配与精确 Schur 缩聚，再把每个 $456^2$ 的局部 Schur 补散加进全局边界 CSR，见上文 $K^j$、$K_s^j$、$K_\Gamma$ 三步；首轮另建全局边界 CSR 符号模式，40.1 s |
| 迭代：边界条件 | 0.06 | 0.01 | 两侧每轮重建全场右端项<br>**FA**：另施加约束，复制 CSR 值数组并对称消元；首轮 0.9 s<br>**full_trace**：不改写全局边界矩阵，返回构造时算好的全局边界载荷，约束在求解阶段施加；首轮 0.6 s |
| 迭代：求解 | 2.85 | 28.31 | 两侧均为 MGCG，以上一轮位移热启动，CG 步数相近<br>**FA**：在全场 $K$ 上迭代<br>**full_trace**：全局边界 CSR 每行约 850 个非零元，是 $K$ 的 10 倍，矩阵向量积相应变贵；另含全场位移恢复，要再做一次局部装配与内部回代 |

整个流程：

| 项目 | FA | full_trace | 说明 |
|---|---|---|---|
| 每轮合计 / s | 4.0 | 58.0 | 迭代各阶段之和，第 2 轮起均值 |
| 总耗时 / min | 4.6 | 65.9 | 准备加 68 轮迭代 |

数据来源：`results/fa_hex_210x35x35_cg/`、`results/full_trace_hex_210x35x35_m5_cg/` 的 `summary.json` 与 `history.json`。

#### 3.2.2 内存

单位 GiB，统计量为进程 RSS。共同阶段按内存用途归组，不代表两侧实测值完全相同；所有数值保留原统计口径，不取两侧均值。显示为 0.00 仅表示按当前精度舍入后的结果。

**表 A　常驻内存的建立**

阶段行记录 RSS 净变化，可因内存释放而为负；起点与累计状态记录 RSS 本身。

两侧共同的阶段（统一说明内存用途，分别保留实测值）：

| 阶段 | 内存 / GiB | 说明 |
|---|---|---|
| 起点 | 0.07 | 解释器与导入 |
| 准备：网格 | 0.41 | 节点坐标、单元以及eager 构造的面、边与其关联数组 |
| 准备：过滤矩阵 | FA 0.30；full_trace 0.33 | 过滤矩阵 $H$，COO 存储 |
| 准备：目标与约束 | 0.00 | 体积约束梯度 |
| 首轮：过滤 | 0.00 | 物理密度 |
| 第 2 轮起 | 0.00 | 热启动保留的位移已计入首轮求解 |

两侧存储内容或处理方式不同的阶段：

| 阶段 | FA | full_trace | 说明 |
|---|---|---|---|
| 准备：分析器 | 0.00 | 0.00 | 函数空间与积分子等对象；full_trace 的子结构布局与边界迹空间在本档也不足 0.01 |
| 首轮：装配 | 1.44 | 5.81 | FA 为 CSR 刚度与装配缓存；full_trace 为全局边界 CSR 的符号模式（行指针与列号）、子结构编号与边界自由度映射，以及持有到求解结束的全局边界 CSR 值数组（约 2.7，见表 B） |
| 首轮：边界条件 | 0.48 | −0.04 | FA 复制一份值数组作为系统矩阵；full_trace 不改写全局边界矩阵，无副本 |
| 首轮：求解 | −0.21 | −2.35 | 两侧建立多重网格层次并保留上一轮位移；FA 释放装配缓存与值数组副本，full_trace 释放全局边界 CSR 值数组与恢复临时量 |

累计状态（按执行顺序列出）：

| 阶段 | FA | full_trace | 说明 |
|---|---|---|---|
| 准备结束 | 0.78 | 0.81 | 累计 |
| 首轮结束 | 2.52 | 4.27 | 累计 |
| 稳态每轮起点 | 2.53 | 4.29 | 累计 |

数据来源：各行为相邻阶段结束时 RSS 之差，准备各行取 `summary.json` 的 `setup_stages.<阶段>.after_gib`，首轮各行取 `history[0]` 的 `stages.<阶段>.after_gib`；起点取 `setup_stages.网格` 的 `peak_gib − net_gib`；首轮结束取 `history[0]` 的 `stages.OC.after_gib`，已含首轮灵敏度、输出与 OC 的小幅变化（FA +0.04，full_trace +0.03）；稳态每轮起点取第 2 轮起各轮 `stages.过滤` 的 `peak_gib − net_gib` 的中位数。

**表 B　各阶段相对起点的峰值增量**

各行记录阶段峰值 RSS 减去阶段起点 RSS，可能包含新增常驻量及跨阶段持有量，不能全部解释为阶段结束即释放的临时量，也不能相加得到全程峰值。

两侧共同的阶段（统一说明内存用途，分别保留实测值）：

| 阶段 | 内存 / GiB | 说明 |
|---|---|---|
| 准备：网格 | 0.68 | 构造面、边时的全体候选及其排序去重的中间量 |
| 准备：过滤矩阵 | FA 1.03；full_trace 1.05 | 分块生成的行号、列号与权重列表及其拼接 |
| 准备：目标与约束 | FA 0.14；full_trace 0.18 | 梯度过滤的中间量 |
| 迭代：过滤 | 0.00 | 无 |
| 迭代：灵敏度 | FA 0.24；full_trace 0.23 | 按单元取出的位移 $u_e$ 与逐单元灵敏度 |
| 迭代：输出 | 0.06 | VTU 写出缓冲 |
| 迭代：OC | 0.00 | 二分法中的候选密度 |

两侧存储内容或处理方式不同的阶段：

| 阶段 | FA | full_trace | 说明 |
|---|---|---|---|
| 准备：分析器 | 0.00 | 0.00 | 函数空间构造的中间量 |
| 迭代：装配 | 0.24 | 5.84 | FA 为逐分量块的缩放权重与散加中间量；full_trace 含两部分，全局边界 CSR 值数组约 2.7 持有到求解结束（装配结束时 RSS 为 6.96，比起点高 2.67），其余约 3.2 为缩聚批次的局部刚度、局部 Schur 补与散加索引等中间量，装配结束即释放 |
| 迭代：边界条件 | 0.48 | 0.00 | FA 复制并改写值数组；full_trace 无副本 |
| 迭代：求解 | 0.18 | 3.34 | 两侧都含粗层算子重建与 CG 向量；full_trace 另含全局边界 MG 的限制与延拓、全场位移恢复时逐批重做局部装配与内部回代的中间量，起点已含全局边界 CSR 值数组 |

数据来源：准备各行取 `summary.json` 的 `setup_stages.<阶段>.net_gib`；迭代各行取 `history.json` 第 2 轮起各轮 `stages.<阶段>.net_gib` 的中位数。

首轮的阶段峰值另含建立常驻的过程量：full_trace 首轮装配 8.93，FA 首轮装配 1.74，所在时刻的 RSS 分别为 9.73 与 2.52，均未超过全程峰值（取 `history[0]` 对应阶段的 `net_gib` 与 `peak_gib`）。

**表 C　整个流程**

| 项目 | FA | full_trace | 说明 |
|---|---|---|---|
| 准备结束 | 0.78 | 0.81 | 表 A 累计 |
| 稳态每轮起点 | 2.53 | 4.29 | 表 A 累计 |
| 迭代阶段峰值 | 3.20 | 10.32 | FA 在第 25 轮求解，2.53 + 0.48 + 0.18，即稳态起点加值数组副本加求解临时量；full_trace 在第 27 轮求解，4.29 + 2.67 + 3.34，即稳态起点加全局边界 CSR 值数组加求解临时量 |
| 全程峰值 | 3.20 | 10.32 | 两侧均即迭代阶段峰值；准备阶段峰值在过滤矩阵，1.51 与 1.53 |

数据来源：全程峰值取 `summary.json` 的 `process_peak_rss_gib`；迭代阶段峰值取 `history.json` 全部轮次各阶段 `peak_gib` 的最大值。

首档 $390\times65\times65$ 上 full_trace 每轮约 300 s、峰值 44 GB，FA 为 33 s、19 GB，前 12 轮 CG 步数逐轮相同，因代价过高未跑完（中止运行，未入库）。

## 参考文献

[1] Huang2023，第 4 节公共设置与第 4.1 节 MBB 梁算例，图 5–7。
