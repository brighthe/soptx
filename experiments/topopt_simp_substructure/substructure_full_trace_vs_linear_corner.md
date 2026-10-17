# 子结构 full_trace 与 linear_corner 接口的拓扑优化对照

硬件：Intel Core i9-14900KF（24 核 32 线程，8P+16E），运行内存 45 GiB；NVIDIA GeForce RTX 5080，显存 16 GB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（内置 scipy-openblas 0.3.34），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.13.0（CUDA 13.0）；PyMUMPS 0.4.0，链接系统 MUMPS 5.6.2（OpenBLAS 0.3.26、Open MPI 4.1.6、ScaLAPACK 2.2.1，单进程运行）；pyevtk 1.7.0。

## 1. 研究对象

对照相同细网格、相同子结构划分下，子结构边界位移的两种表示 `full_trace` 与 `linear_corner` 对 Huang2023 三维 MBB 梁拓扑优化设计质量和计算成本的影响。

| 项目 | full_trace | linear_corner |
|---|---|---|
| 子结构边界位移 | 全部边界节点位移为未知量，$\Psi=I$ | 由 8 个角点位移三线性插值，$\Psi=L$ |
| 刚度算子 | 各子结构 Schur 补装配为全局边界 CSR 矩阵 $K_\Gamma$ | 各子结构投影到角点的 Schur 补装配为全局角点 CSR 矩阵 $K_C=P^{\mathsf T}K_\Gamma P$ |
| 未知量 | 全局边界位移 $U_\Gamma$ | 全局角点位移 $U_C$，边界位移由 $U_\Gamma=PU_C$ 插值得到 |

### 1.1 矩阵组装、局部缩聚与角点投影

首先将细网格单元刚度 $K_e(\rho_e)$ 组装为子结构总刚 $K^j$，并按边界 $b$、内部 $i$ 分块：

$$
K^j=\sum_{e\in\Omega_j}(G_e^j)^{\mathsf T}K_e(\rho_e)\,G_e^j
=\begin{bmatrix}K_{bb}^j&K_{bi}^j\\K_{ib}^j&K_{ii}^j\end{bmatrix}\in\mathbb R^{n_d\times n_d}
$$


子结构平衡方程按同一分块写为

$$
\begin{bmatrix}K_{bb}^j&K_{bi}^j\\K_{ib}^j&K_{ii}^j\end{bmatrix}
\begin{bmatrix}u_b^j\\u_i^j\end{bmatrix}=
\begin{bmatrix}f_b^j\\0\end{bmatrix},
$$

其中 $f_b^j$ 为作用在子结构边界节点上的力；内部载荷为零。由第二行解出内部位移

$$
u_i^j=-(K_{ii}^j)^{-1}K_{ib}^ju_b^j ,
$$

代入第一行，消去内部自由度：

$$
\bigl(K_{bb}^j-K_{bi}^j(K_{ii}^j)^{-1}K_{ib}^j\bigr)u_b^j=f_b^j .
$$

这是只含边界位移的缩聚平衡方程 $K_s^ju_b^j=f_b^j$，其系数矩阵即缩聚刚度，也就是 Schur 补：

$$
K_s^j=K_{bb}^j-K_{bi}^j(K_{ii}^j)^{-1}K_{ib}^j\in\mathbb R^{n_b\times n_b}.
$$

这一步是局部精确消元，两侧相同。

消元后每个子结构只剩边界位移 $u_b^j$ 作为未知量，但边界自由度在子结构中占多数。记子结构边界位移的表示为 $u_b^j=\Psi q^j$，以 $q^j$ 作为子结构的未知量：`full_trace` 保留全部边界自由度，`linear_corner` 用少量参数表示边界位移，以缩小全局系统。两侧的差别就在 $\Psi$ 的取法：

- `full_trace` 取 $\Psi=I$，$q^j=u_b^j$ 为全部边界自由度，不做降维。消元后不再施加任何约束，缩聚刚度就是 Schur 补 $K_s^j$。
- `linear_corner` 取 $\Psi=L$，$q^j$ 为 8 个角点的 24 个位移，$L\in\mathbb R^{n_b\times24}$ 为角点位移到全部边界节点位移的三线性插值矩阵。将 $u_b^j=Lq^j$ 代入 $K_s^ju_b^j=f_b^j$ 并左乘 $L^{\mathsf T}$，得 $L^{\mathsf T}K_s^jL\,q^j=L^{\mathsf T}f_b^j$，缩聚刚度为 $K_c^j=L^{\mathsf T}K_s^jL$。

两种缩聚刚度统一写为 $\Psi^{\mathsf T}K_s^j\Psi$：

$$
\Psi^{\mathsf T}K_s^j\Psi=
\begin{cases}
K_{s}^j\in\mathbb R^{n_b\times n_b}, & \texttt{full\_trace}\ (\Psi=I),\\[4pt]
K_c^j=L^{\mathsf T}K_s^jL\in\mathbb R^{24\times24}, & \texttt{linear\_corner}\ (\Psi=L).
\end{cases}
$$

`full_trace` 的 $K_s^j$ 与细网格 FEM 代数等价；`linear_corner` 的 $K_c^j$ 是 $K_s^j$ 在角点插值空间上的 Galerkin 投影，引入了边界位移约束，是 Ritz 近似。

再将各自的子结构缩聚刚度矩阵组装为全局系统刚度：

$$
K_\Gamma=\sum_j(A_b^j)^{\mathsf T}K_s^jA_b^j\in\mathbb R^{N_\Gamma\times N_\Gamma},\qquad
K_C=\sum_j(A_c^j)^{\mathsf T}K_c^jA_c^j\in\mathbb R^{N_C\times N_C},
$$

其中 $A_b^j,A_c^j$ 分别提取子结构的边界位移与角点位移，$P$ 为全局的角点到边界插值，满足 $A_b^jP=LA_c^j$。

### 1.2 求解与位移恢复

`full_trace`：$K_\Gamma U_\Gamma=F_\Gamma$，$q^j=A_b^jU_\Gamma$；`linear_corner`：$K_CU_C=F_C$，$q^j=A_c^jU_C$。两侧都用形函数矩阵恢复子结构位移，再按全局编号拼成全场位移：

$$
u^j=N^jq^j,\qquad
U^\Gamma=R_\Gamma^{\mathsf T}U_\Gamma+\sum_j(R_i^j)^{\mathsf T}u_i^j,\qquad
U^C=R_\Gamma^{\mathsf T}PU_C+\sum_j(R_i^j)^{\mathsf T}u_i^j .
$$

展开即回代 $u_i^j=-(K_{ii}^j)^{-1}K_{ib}^ju_b^j$，其中 $u_b^j=A_b^jU_\Gamma$ 或 $u_b^j=LA_c^jU_C$；两侧内部位移都精确满足细网格内部平衡。

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

## 3. 结果对照

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
