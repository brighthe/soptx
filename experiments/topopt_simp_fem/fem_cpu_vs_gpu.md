# 传统有限元 CPU 与 GPU 拓扑优化结果对照

硬件：AMD Ryzen 9 9950X（16 核 32 线程），运行内存 45 GiB（WSL2 上限 48 GB）；NVIDIA GeForce RTX 5070 Ti（70 SM，8960 CUDA 核心），显存 16 GiB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（conda-forge，链接 BLIS 2.0 与 netlib LAPACK），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.11.0（CUDA 12.8）；PyMUMPS 0.4.0，链接 MUMPS 5.8.2（BLIS 2.0、netlib LAPACK、Open MPI 5.0.11、ScaLAPACK 2.2.0）；pyevtk 1.7.0。

## 1. 研究对象

对照传统有限元不同的设备 `CPU` 与 `GPU` 在 Huang2023 三维 MBB 梁上的完整优化结果。

|      | CPU | GPU |
| ---- | --- | --- |
| 设备 | AMD Ryzen 9 9950X，16 线程 | NVIDIA GeForce RTX 5070 Ti，70 SM |
| 内存 | 主机内存 45 GiB | 显存 16 GiB |

## 2. 实验设置

实验脚本为 [`run_fem_cpu_vs_gpu.py`](run_fem_cpu_vs_gpu.py)，两侧以 `--device cpu`、`--device cuda` 各运行一次，结果经 [`sync_results.py`](sync_results.py) 固化到 `results/`。

本实验均采用 PyTorch 后端：`BACKEND = 'pytorch'`。

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

#### 有限元离散

| 参数 | 数值或设置 |
|---|---|
| 有限元 | 八节点六面体单元（Q1）：`HexahedronMesh.from_box(...)` |
| 位移插值次数 $p$ | 每个参考坐标方向一次，$p=1$：`SPACE_DEGREE = 1`，传入 `space_degree=SPACE_DEGREE` |
| 数值积分 | 二阶高斯积分：`INTEGRATION_ORDER = 2`，传入 `integration_order=INTEGRATION_ORDER` |
| 单元刚度积分 | `fast` 积分实现，只存一份实体参考单元刚度 $K^0$：`--assembly-method fast`；分析器逐单元检查几何，判定 `from_box` 网格为单一平移类 |
| 分析网格 | 各方向单元数为 $390\times65\times65$：`--grid 390 65 65` |

#### 线性系统求解

| 参数 | 数值或设置 |
|---|---|
| 线性求解器 | 几何多重网格预条件共轭梯度法（MGCG）：`--solver cg --precond gmg` |
| 多重网格 | V 循环，三线性延拓，粗网格各向单元数取 $\lceil n/2\rceil$，Galerkin 粗层算子（第 2 层按单元组合 $\sum_c s_cP_c^{\mathsf T}K^0P_c$）；加权 Jacobi 前后各 1 步，$\omega=4/(3\cdot1.1\,\hat\lambda)$；最粗层不超过 $20000$ 个自由度，直接分解：`--precond gmg`，沿用默认 `gmg_sweeps=1`、`gmg_omega=None`（自动估计）、`gmg_coarse_max_dofs=20000`、`gmg_coarse_solver='scipy'` |
| 停机准则（PCG） | $\lVert b-Au\rVert_2\le10^{-6}\lVert b\rVert_2$，$b$ 为施加边界条件后的右端项（集中力不落在节点上时按节点分摊，$\lVert b\rVert_2$ 小于力的模长）：`CG_TOLERANCE = 1.0e-6`，`rtol=CG_TOLERANCE`，`atol=0.0`，`rtol_reference='rhs'`，`norm_type='unpreconditioned'`，`verify_residual=True` |
| 初值（PCG） | 首轮取 Dirichlet 基准向量（本例为零位移），之后以上一轮位移热启动：`x0 = analyzer.prescribed_solution if previous_displacement is None else previous_displacement` |
| 最大迭代数（PCG） | 最多 $20000$ 步，逐轮记录实际步数：`--cg-maxiter 20000`，传入 `maxiter=args.cg_maxiter` |

### 2.3 优化参数

| 参数 | 数值或设置 |
|---|---|
| 优化目标 | 最小化柔顺度：`ComplianceObjective(..., state_variable='u', diff_mode='manual')` |
| 体积约束 | 物理体积分数 $\frac{1}{N_F}\sum_e\tilde\rho_e=0.12$，作用于物理密度 $\tilde\rho$：`VOLFRAC = 0.12`，`VolumeConstraint(..., volume_fraction=VOLFRAC, ...)` |
| 初始密度 | 均匀取 $0.12$：`rho = bm.full((n_cells, ), VOLFRAC, ...)` |
| 材料插值 | 修正 SIMP：$E(\rho)=E_{\min}+\rho^{q}(E_0-E_{\min})$，$\rho\in[0,1]$：`interpolation_method='msimp'` |
| SIMP 惩罚指数 $q$ | $3$（区别于位移插值次数 $p$）：`PENALTY = 3.0` |
| 密度过滤 | $\tilde\rho_e=\sum_i H_{ei}\rho_i/\sum_i H_{ei}$，$H_{ei}=\max(0,\ r_{\min}-d_{ei})$，$d_{ei}$ 为单元形心距离：`Filter(..., filter_type='density', rmin=radius, ...)` |
| 过滤半径 | $r_{\min}=3h=3$：`FILTER_RADIUS_CELLS = 3.0`，`radius = FILTER_RADIUS_CELLS * h` |
| 灵敏度过滤 | 按链式法则过滤：`filter_objective_sensitivities(...)`，`filter_constraint_sensitivities(...)` |
| 优化器 | OC：`OCOptimizer.update_design_variable(..., **OC_OPTIONS)` |
| 移动限 | $0.2$：`move_limit=0.2` |
| 阻尼指数 | $0.5$：`damping_coef=0.5` |
| 设计变量下界 | $0$：`design_variable_min=0.0` |
| Lagrange 乘子 | 二分法，初始区间 $[0,10^9]$，相对精度 $10^{-3}$：`initial_lambda=1.0e9`，`bisection_tol=1.0e-3` |
| 体积可行性容差 | $10^{-6}$，物理体积分数超过目标值的允许量：`VOLUME_TOLERANCE = 1.0e-6` |
| 收敛判据 | $r_k=\lvert c_k-c_{k-1}\rvert/c_k$，$r_{k-4},\dots,r_k$ 均小于 $2\times10^{-4}$ 时停止：`TOLERANCE = 2.0e-4`，`CONVERGENCE_WINDOW = 5` |
| 最大迭代数 | $300$：`MAX_ITER = 300` |

## 3. 结果一致性

| 指标 | CPU | GPU |
| --- | --- | --- |
| 终止 | 待填 | 待填 |
| 最终柔顺度 $C$ | 待填 | 待填 |
| 最终体积分数 | 待填 | 待填 |
| 每轮 CG 步数 | 待填 | 待填 |
| 末轮 CG 相对残差 | 待填 | 待填 |
| 物理密度 $z$ 向不对称量 | 待填 | 待填 |
| 设计密度 $z$ 向不对称量 | 待填 | 待填 |

数据来源：待填。

两侧差异：

| 量 | 最大差 |
| --- | --- |
| 逐轮柔顺度相对差 | 待填 |
| 最终物理密度 | 待填 |
| 最终设计密度 | 待填 |
| 最终位移相对 $L^2$ 差 | 待填 |
| 阈值 $0.5$ 二值化拓扑 | 待填 |

数据来源：待填。

## 4. 计算成本

### 4.1 耗时（s）

| 阶段 | CPU | GPU | 说明 |
|---|---|---|---|
| 准备：网格 | 待填 | 待填 | |
| 准备：分析器 | 待填 | 待填 | |
| 准备：过滤矩阵 | 待填 | 待填 | |
| 准备：目标与约束 | 待填 | 待填 | |
| 迭代：装配 | 待填 | 待填 | |
| 迭代：边界条件 | 待填 | 待填 | |
| 迭代：求解 | 待填 | 待填 | |
| 迭代：过滤 | 待填 | 待填 | |
| 迭代：灵敏度 | 待填 | 待填 | |
| 迭代：输出 | 待填 | 待填 | |
| 迭代：OC | 待填 | 待填 | |

整个流程：

| 项目 | CPU | GPU | 说明 |
|---|---|---|---|
| 每轮合计 / s | 待填 | 待填 | 迭代各阶段之和，第 2 轮起均值 |
| 总耗时 / min | 待填 | 待填 | 准备加全部迭代 |

数据来源：待填。

### 4.2 内存（GiB）

| 项目 | CPU | GPU | 说明 |
|---|---|---|---|
| 主机内存峰值 | 待填 | 待填 | 进程 RSS |
| 显存峰值 | – | 待填 | |

数据来源：待填。

## 参考文献

[1] Huang2023，第 4 节公共设置与第 4.1 节 MBB 梁算例，图 5–7。
