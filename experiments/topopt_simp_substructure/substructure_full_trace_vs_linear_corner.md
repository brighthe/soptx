# 子结构 full_trace 与 linear_corner 接口的拓扑优化对照

硬件：Intel Core i9-14900KF（24 核 32 线程，8P+16E），运行内存 45 GiB；NVIDIA GeForce RTX 5080，显存 16 GB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（内置 scipy-openblas 0.3.34），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.13.0（CUDA 13.0）；PyMUMPS 0.4.0，链接系统 MUMPS 5.6.2（OpenBLAS 0.3.26、Open MPI 4.1.6、ScaLAPACK 2.2.1，单进程运行）；pyevtk 1.7.0。

## 1. 研究对象

对照相同细网格、相同子结构划分下，`full_trace` 与 `linear_corner` 接口迹空间对 Huang2023 三维 MBB 梁 拓扑优化设计质量和计算成本的影响。两者均采用局部精确内部消元，最终设计质量通过统一细网格 FEM 重分析评价。

| 项目 | full_trace | linear_corner |
|---|---|---|
| 刚度算子 | 各子结构局部 Schur 补装配为完整接口 CSR 矩阵 $K_\Gamma$ | 各子结构投影后的 Schur 补装配为角点接口 CSR 矩阵 $K_C=T^{\mathsf T}K_\Gamma T$ |
| 未知量 | 保留全部接口位移未知量 $U_\Gamma$ | 保留角点位移未知量 $U_C$，接口位移由 $U_\Gamma=T U_C$ 插值得到 |

## 2. 实验设置

实验脚本为 [`run_substructure_full_trace_vs_linear_corner.py`](run_substructure_full_trace_vs_linear_corner.py)。

本实验均采用 NumPy / CPU：`BACKEND = 'numpy'`，`DEVICE = 'cpu'`；单线程：`NUM_THREADS = 1`，脚本在导入 numpy/scipy 之前把全部 BLAS / OpenMP 线程变量设为 1，取值记入 `config.json` 的 `threads`。

### 2.1 模型参数

![MBB 梁的几何、载荷与支承](../topopt_piml_substructure/assets/Huang2023_Fig5.png)

采用整体三维 MBB 梁模型，长、高、宽分别记为 $L_x,L_y,L_z$，比例为 $6:1:1$，计算域为 $\Omega=[0,L_x]\times[0,L_y]\times[0,L_z]$。顶面中心施加竖直向下的单位集中力 $F=1$。支承取在两端底边的四个角点：左端两角点铰支（$u_x=u_y=u_z=0$），右端两角点滚支（$u_y=0$）。

| 参数 | 数值或设置 |
|---|---|
| 几何与计算域 | 整体三维 MBB 梁，长、高、宽为 $210,35,35$：`domain = (0, 210, 0, 35, 0, 35)`，`FullMBBBeamPoint3d(domain=domain, ...)` |
| 长度约定 | 默认剖分下 $h=1$；计算域固定，实际边长按 $h_i=L_i/n_i$ 计算并记录到 `spacing` |
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
| 分析网格 | 结构化六面体网格：`mesh = HexahedronMesh.from_box(box=domain, nx=210, ny=35, nz=35, ...)` |

**子结构缩聚**

| 参数 | 数值或设置 |
|---|---|
| 子结构划分 | $42\times7\times7$ 个子结构，每个含 $5\times5\times5$ 个细单元：`StructuredSubstructureLayout(..., n_sub=(42, 7, 7), n_fine=(5, 5, 5), full_mesh=mesh)` |
| 接口空间 | full_trace 保留全部接口自由度，$\boldsymbol\Psi=\boldsymbol I$：`SubstructureAnalyzer(..., trace='full_trace')`；linear_corner 采用角点线性插值空间：`SubstructureAnalyzer(..., trace='linear_corner')` |
| 分批大小 | 每批处理 256 个子结构：`chunk_size=256` |

**线性求解参数**

| 参数 | 数值或设置 |
|---|---|
| 求解器 | CG + GMG（MGCG）：`SOLVER = 'cg'`，`PRECONDITIONER = 'gmg'` |
| 停机准则 | 绝对容差为 $10^{-6}\lVert F\rVert$：`CG_TOLERANCE = 1.0e-6`，`rtol=0` |
| 最大 CG 步数 | `--cg-maxiter 20000`，对应 `maxiter=args.cg_maxiter` |
| 迭代初值 | 首轮均为零向量；第 2 轮起取上一轮拓扑优化的接口位移解：`x0=previous_displacement`，`full_trace` 为完整接口位移，`linear_corner` 为角点位移 |

### 2.3 优化参数

| 参数 | 数值或设置 |
|---|---|
| 优化目标 | 最小化柔顺度 |
| 体积约束 | 物理体积分数 $0.12$：`VOLFRAC = 0.12` |
| 材料插值 | 修正 SIMP：`interpolation_method='msimp'` |
| SIMP 惩罚指数 $q$ | $3$：`PENALTY = 3.0` |
| 密度过滤 | `filter_type='density'` |
| 过滤半径 | $3h$：`FILTER_RADIUS_CELLS = 3.0` |
| 优化器 | OC |
| 移动限 | $0.2$：`move_limit=0.2` |
| 阻尼指数 | $0.5$：`damping_coef=0.5` |
| 收敛判据 | 柔顺度相对变化连续 5 轮小于 $2\times10^{-4}$：`TOLERANCE = 2.0e-4`，`CONVERGENCE_WINDOW = 5` |
| 体积可行性容差 | $10^{-6}$：`VOLUME_TOLERANCE = 1.0e-6` |
| 最大迭代数 | 正式优化 `--max-iter 300`；小网格验收 `--max-iter 2` |

###
## 3. 结果对照

同一细网格 $210\times35\times35$ 划分为 $42\times7\times7=2\,058$ 个子结构，每个含 $5^3=125$ 个细单元、192 个内部自由度和 456 个边界自由度。

子结构有限元 `full_trace`：全局接口共 $141\,744$ 个节点、$425\,232$ 个自由度。

子结构有限元 `linear_corner`：全局角点接口共 $2\,752$ 个节点、$8\,256$ 个自由度。

记号如下，矩阵一列给出形状：

| 符号 | 含义 | 取值（$m=5$） |
|---|---|---|
| $\mathcal T_h$，$\lvert\mathcal T_h\rvert$ | 细网格及其单元数 | 257 250 |
| $N_{h}$ | 细网格自由度数 | 820 368 |
| $n_{e}$ | 单元自由度数 | 24 |
| $m$ | 子结构每边的细单元数 | 5 |
| $N_{s}$ | 子结构数 | $42\times7\times7=2\,058$ |
| $n_{i},\ n_{b}$ | 子结构内部、边界自由度数 | 192，456 |
| $n_{c}$ | 子结构角点自由度数，$3\times2^3$ | 24 |
| $N_{\Gamma}$ | 全局边界自由度数 | 425 232 |
| $N_{C}$ | 全局角点自由度数，$3(n_x+1)(n_y+1)(n_z+1)$，$(n_x,n_y,n_z)=(42,7,7)$ | $3\times43\times8\times8=8\,256$ |
| $\rho_e$，$\tilde\rho_e$ | 单元 $e$ 的设计密度、过滤后的物理密度 | |
| $c(\tilde\rho_e)=E(\tilde\rho_e)/E_0$ | 相对刚度系数 | |
| $K^{0}$ | 缓存的参考单元刚度 | $n_e\times n_e$ |
| $A_{e}$ | 单元到细网格的提取矩阵，元素取 0 或 1 | $n_e\times N_h$ |
| $K$，$F$ | 细网格总刚与载荷 | $N_h\times N_h$，$N_h$ |
| $K^{j}$ | 子结构 $\Omega_j$ 的总刚，按边界 $b$、内部 $i$ 分块 | $(n_b+n_i)\times(n_b+n_i)$ |
| $K_{s}^{j}=K_{bb}^{j}-K_{bi}^{j}(K_{ii}^{j})^{-1}K_{ib}^{j}$ | 子结构缩聚刚度（Schur 补） | $n_b\times n_b$ |
| $A_{b}^{j}$ | 子结构边界到全局边界的提取矩阵，元素取 0 或 1 | $n_b\times N_\Gamma$ |
| $K_{\Gamma}$，$F_{\Gamma}$ | 全局边界总刚与载荷 | $N_\Gamma\times N_\Gamma$，$N_\Gamma$ |
| $U_{\Gamma}$ | 全局边界位移 | $N_\Gamma$ |
| $L$ | 局部角点插值矩阵（Huang2023 式 (16)） | $n_b\times n_c=456\times24$ |
| $A_{c}^{j}$ | 子结构角点到全局角点的提取矩阵，元素取 0 或 1 | $n_c\times N_C$ |
| $P$ | 全局迹映射，$U_\Gamma=PU_C$ | $N_\Gamma\times N_C=425\,232\times8\,256$ |
| $K_{C}$，$F_{C}$，$U_{C}$ | 全局角点总刚、载荷与位移 | $N_C\times N_C$，$N_C$，$N_C$ |

两侧共用的局部组装与精确 Schur 缩聚见 substructure_vs_fem.md 第 3 节的第 1、2 步，得到每个子结构的缩聚刚度 $K_s^j$。分叉从 $K_s^j$ 之后开始。

**full_trace** 把 $K_s^j$ 当作“单刚”，组装成全局边界总刚：

$$
K_\Gamma=\sum_{j=1}^{N_s}\bigl(A_b^{j}\bigr)^{T}K_s^{j}A_b^{j}\in\mathbb R^{N_\Gamma\times N_\Gamma}
$$

**linear_corner** 先把 $K_s^j$ 投影到角点自由度，再组装成全局角点总刚：

1. 用角点位移插值接口位移。记子结构 8 个角点的位移为 $q^{j}=A_c^{j}U_C\in\mathbb R^{n_c}$，边界节点 $x_b$ 处的位移取角点的三线性插值：

$$
u_b^{j}=L\,q^{j},\qquad L=\bigl[w_c(x_b)\,\delta_{kl}\bigr]\in\mathbb R^{n_b\times n_c},\qquad
w_c(x)=\prod_{d=1}^{3}(1-\xi_d)^{1-s_{c,d}}\,\xi_d^{\,s_{c,d}},\quad \xi_d=\frac{x_d-x_{0,d}}{mh}
$$

$s_{c,d}\in\{0,1\}$ 为角点 $c$ 在 $d$ 方向的位置；$\sum_c w_c\equiv1$，$L$ 精确再现任意仿射位移场；$L$ 与 $\tilde\rho$ 无关，对所有子结构相同。相邻子结构在公共面上共享 4 个角点，两侧插值在该面上一致，因此各子结构的 $L$ 可拼成全局迹映射 $P\in\mathbb R^{N_\Gamma\times N_C}$，满足 $A_b^{j}P=LA_c^{j}$，即 $U_\Gamma=PU_C$。

2. 把缩聚刚度投影到角点自由度：

$$
K_C^{j}=L^{T}K_s^{j}L\in\mathbb R^{n_c\times n_c}
$$

3. 把 $K_C^j$ 当作“单刚”，组装成全局角点总刚：

$$
K_C=\sum_{j=1}^{N_s}\bigl(A_c^{j}\bigr)^{T}K_C^{j}A_c^{j}\in\mathbb R^{N_C\times N_C}
$$

$K_C$ 的组装式与 $K_\Gamma$ 的组装式同构，只是把 $n_b\times n_b$ 的缩聚单刚 $K_s^j$ 换成 $n_c\times n_c$ 的投影单刚 $L^TK_s^jL$，把边界提取矩阵 $A_b^j$ 换成角点提取矩阵 $A_c^j$。

求解与恢复。记 $F_\Gamma$ 为细网格载荷在接口自由度上的分量，$F_C=P^{T}F_\Gamma$ 为其在角点上的等效载荷（对任意 $U_C$ 满足 $F_C^{T}U_C=F_\Gamma^{T}PU_C$）；支承点都在角点上，两侧都按固定自由度消元。

$$
\text{full\_trace:}\quad K_\Gamma U_\Gamma=F_\Gamma,\qquad u_b^{j}=A_b^{j}U_\Gamma,\qquad u_i^{j}=-\bigl(K_{ii}^{j}\bigr)^{-1}K_{ib}^{j}u_b^{j}
$$

$$
\text{linear\_corner:}\quad K_CU_C=F_C,\qquad u_b^{j}=L\,A_c^{j}U_C,\qquad u_i^{j}=-\bigl(K_{ii}^{j}\bigr)^{-1}K_{ib}^{j}u_b^{j}
$$

两侧恢复内部位移的公式相同，只是 $u_b^j$ 的来源不同。报告的柔顺度分别为 $C_\Gamma=F_\Gamma^TU_\Gamma$ 与 $C_C=F_C^TU_C$，都等于各自恢复后全场位移的 $F^TU$。

两者的关系。把 $A_b^{j}P=LA_c^{j}$ 代入 $K_C$ 的组装式：

$$
K_C=\sum_j\bigl(LA_c^{j}\bigr)^{T}K_s^{j}\bigl(LA_c^{j}\bigr)=\sum_j\bigl(A_b^{j}P\bigr)^{T}K_s^{j}\bigl(A_b^{j}P\bigr)=P^{T}K_\Gamma P,\qquad F_C=P^{T}F_\Gamma
$$

即 linear_corner 的角点系统就是 full_trace 的接口系统限制到子空间 $U_\Gamma=PU_C$ 上的 Galerkin 投影。substructure_vs_fem.md 中 $K_\Gamma$ 是 $K$ 的 Schur 补，消去的是内部自由度，解不变，所以 FA 与 full_trace 代数等价；这里 $P$ 是 $N_\Gamma\times N_C$ 的瘦矩阵，不可逆，消去的是接口上非角点的位移自由度，解会改变。因此两侧不等价，linear_corner 是 full_trace 的 Ritz 近似，差别可以写成：

1. 柔顺度单调偏低，偏低量等于位移误差的能量范数平方：

$$
C_C\le C_\Gamma,\qquad C_\Gamma-C_C=\bigl\|U^{\Gamma}-U^{C}\bigr\|_K^{2}=\bigl\|U_\Gamma-PU_C\bigr\|_{K_\Gamma}^{2},\qquad \varepsilon_\Gamma=\frac{C_\Gamma-C_C}{C_\Gamma}
$$

2. 最佳逼近：$\|U^{\Gamma}-U^{C}\|_K=\min\{\|U^{\Gamma}-V\|_K:\ V_\Gamma\in\operatorname{range}P\}$。接口迹为仿射场时误差为零；$m=1$ 时 $n_b=n_c$、$L=I$，两侧相同。

3. 灵敏度公式同形：子空间不随 $\tilde\rho$ 变化，$\partial C_C/\partial\tilde\rho_e=-c'(\tilde\rho_e)(u_e^{C})^{T}K^0u_e^{C}$，与 full_trace 只差代入的恢复位移。此为推导结论，代码中 `ComplianceObjective.jac` 对两种接口走同一路径，尚未用有限差分做数值核对。

装配复杂度：三维 Q1 单元下 $n_i=3(m-1)^3$，$n_b=18m^2+6$，$n_c=24$，$N_s=\lvert\mathcal T_h\rvert/m^3$。

| 步骤 | 运算量（散加个数 / flop） | 对 $\lvert\mathcal T_h\rvert$、$m$ 的阶 | $m=5$ |
|---|---|---|---|
| 共用：局部组装 $K^j$ | $N_s(n_i+n_b)^2$ | $\Theta(\lvert\mathcal T_h\rvert m^3)$ | $8.6\times10^{8}$ |
| 共用：缩聚 $K_s^j$ | $N_s\bigl(\tfrac23n_i^3+2n_i^2n_b+2n_in_b^2\bigr)$ | $\Theta(\lvert\mathcal T_h\rvert m^6)$ | $2.4\times10^{11}$ |
| full_trace：组装 $K_\Gamma$ | $N_sn_b^2$ | $\Theta(\lvert\mathcal T_h\rvert m)$ | $4.3\times10^{8}$ |
| linear_corner：投影 $L^TK_s^jL$ | $N_sn_bn_c(n_b+n_c)$ | $\Theta(\lvert\mathcal T_h\rvert m)$ | $1.1\times10^{10}$ |
| linear_corner：组装 $K_C$ | $N_sn_c^2$ | $\Theta(\lvert\mathcal T_h\rvert/m^3)$ | $1.2\times10^{6}$ |
| 接口未知量 | $N_\Gamma$ / $N_C$ | $\Theta(\lvert\mathcal T_h\rvert/m)$ / $\Theta(\lvert\mathcal T_h\rvert/m^3)$ | $425\,232$ / $8\,256$ |
| CG 每步 matvec 非零元上界 | $N_sn_b^2$ / $N_sn_c^2$ | $\Theta(\lvert\mathcal T_h\rvert m)$ / $\Theta(\lvert\mathcal T_h\rvert/m^3)$ | $4.3\times10^{8}$ / $1.2\times10^{6}$ |

$m=5$ 一列按公式计算，不是实测；非零元一行是各子结构稠密块之和的上界，共享面会使实际值略小。两侧的主要成本都在共用的缩聚一步；linear_corner 多出的投影与缩聚相比可忽略，省下的是全局矩阵规模与接口未知量数，代价是上面的模型误差。


### 3.1 优化结果

| 项目 | full_trace + Jacobi | linear_corner + Jacobi |
|---|---|---|
| 日期 / 硬件 / 线程配置 | 待填 | 待填 |
| Git revision / 工作区状态 | 待填 | 待填 |
| 软件环境 / BLAS | 待填 | 待填 |
| 原始结果目录 | 待填 | 待填 |
| 小网格流程验收 | 待执行 | 待执行 |
| 正式运行状态 | 待执行 | 待执行 |
| 线性求解失败或其他异常 | 待填 | 待填 |

两侧优化结果:

| 指标 | full_trace + Jacobi | linear_corner + Jacobi |
|---|---|---|
| 接口自由度 (约束前, 核对值) | 待填 | 待填 |
| 优化终止状态 / 轮数 | 待填 | 待填 |
| 最终 C_opt | 待填 | 待填 |
| 最终体积分数 | 待填 | 待填 |
| 最终 C_Fine (FEM 重分析) | 待重分析 | 待重分析 |
| 柔顺度相对偏差 ε_C | 待重分析 | 待重分析 |
| CG 步数 (首轮 / 均值 / 最大 / 总计) | 待填 | 待填 |
| CG 相对残差 (末轮 / 最大) | 待填 | 待填 |
| z 向物理密度对称误差 | 待填 | 待填 |
| 最终拓扑图 / 场文件链接 | 待填 | 待填 |

| 两份主对照设计的差异指标 | linear_corner 相对 full_trace |
|---|---|
| C_Fine 相对变化 (带符号, 以 full_trace 设计为分母) | 待重分析 |
| 最终物理密度相对 L2 差 (以 full_trace 密度范数为分母) | 待填 |
| 阈值 0.5 的二值拓扑不一致单元数 / 总单元数 | 待填 |



### 3.2 最终设计的统一重分析 (待准备)

分别使用两侧 `density_final.npy` 中的物理密度, 在相同细网格 FEM、材料插值、载荷与支承下仅求解一次状态方程. 不重新过滤物理密度, 不执行 OC 更新. 两侧重分析采用相同求解配置与验收容差.

记各自优化报告值为 $C_{\mathrm{opt}}$, 细网格重分析值为 $C_{\mathrm{Fine}}$. 记录接口模型在自身最终设计上的偏差:

\[
\varepsilon_C=\frac{|C_{\mathrm{opt}}-C_{\mathrm{Fine}}|}{|C_{\mathrm{Fine}}|}.
\]

设计质量比较采用两份设计的 $C_{\mathrm{Fine}}$, 并同时报告体积分数. 拓扑比较采用相同视角与色标展示物理密度, 辅以密度相对 L2 差和阈值 0.5 的二值拓扑不一致比例; 这些是设计差异指标, 不是 linear_corner 的求解正确性判据.

当前两个优化入口未提供固定最终密度的 FEM 重分析命令. 此步骤须后续检查并复用仓库能力, 明确入口后再执行; 本文不虚构命令, 不将优化柔顺度填入 C_Fine 栏. 重分析成本独立记录, 不混入优化耗时.

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
| 装配 | 待填 | 待填 | 局部装配、内部消元与接口装配 |
| 边界条件 | 待填 | 待填 | 载荷与边界条件处理 |
| 求解 | 待填 | 待填 | 接口求解及全场位移恢复 |
| 灵敏度 | 待填 | 待填 | 柔顺度与灵敏度计算 |
| 输出 | 待填 | 待填 | VTU/PVD 输出 |
| OC | 待填 | 待填 | 仅统计实际执行 OC 的轮次 |
| 首轮装配 | 待填 | 待填 | 单列首次缓存建立成本 |
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
| 装配 | 待填 | 待填 | 局部缩聚、接口 CSR 及构造中间量 |
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
