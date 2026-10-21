# 传统有限元 EA 与 FA 拓扑优化结果对照

硬件：AMD Ryzen 9 9950X（16 核 32 线程），运行内存 45 GiB（WSL2 上限 48 GB）；NVIDIA GeForce RTX 5070 Ti，显存 16 GB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（conda-forge，链接 BLIS 2.0 与 netlib LAPACK），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.11.0（CUDA 12.8）；PyMUMPS 0.4.0，链接 MUMPS 5.8.2（BLIS 2.0、netlib LAPACK、Open MPI 5.0.11、ScaLAPACK 2.2.0）；pyevtk 1.7.0。

## 1. 研究对象

对照传统有限元刚度算子的两种层级 `EA` 与 `FA` 在 Huang2023 三维 MBB 梁上的完整拓扑优化结果。这里的 `EA` 是共享参考单元的 EA：标准 EA 为每个单元常驻一份单元刚度 $K_e$，而本算例的结构化网格上各单元只差平移，单元刚度可写为 $K_e=E(\rho_e)K^0$，$K^0$ 为单位模量的单元刚度，于是全体单元共用一份 $K^0$，只需常驻 $K^0$ 与逐单元模量 $E(\rho_e)$。

两侧的网格、单元、材料插值、边界条件、过滤、灵敏度、OC 更新与 CG 求解完全相同，差别只在刚度算子的存储与作用方式：`FA` 每轮装配全局矩阵 $K$，`EA` 不形成 $K$，矩阵向量积逐单元完成。

| 项目 | EA（共享参考单元） | FA |
|---|---|---|
| 常驻的刚度数据 | 一份参考单元刚度 $K^0$ 与单元模量 $E(\rho_e)$ | 全局 CSR 矩阵 $K$ |
| 每轮更新 | 只更新 $E(\rho_e)$ | 按 $E(\rho_e)$ 缩放 $K^0$，重新装配 $K$ 的值数组 |
| 矩阵向量积 | 逐单元以 $K^0$ 作用，按 $E(\rho_e)$ 缩放后散加 | CSR 稀疏矩阵向量积 $Kx$ |

`FA`：每轮将单元刚度装配为全局矩阵，以 CSR 格式存储，矩阵向量积直接在 $K$ 上计算：

$$
K=\sum_e A_e^{\mathsf T}\bigl(E(\rho_e)K^0\bigr)A_e\in\mathbb R^{N_h\times N_h},\qquad y=Kx.
$$

`EA`：不形成 $K$，只常驻 $K^0$ 与 $E(\rho_e)$。矩阵向量积先提取单元向量 $A_ex$，以共享的 $K^0$ 作用并乘 $E(\rho_e)$，再散加回全局向量：

$$
y=\sum_e A_e^{\mathsf T}\bigl(E(\rho_e)K^0A_ex\bigr).
$$

两式是同一个线性算子，只是求和顺序不同：`FA` 先把单元贡献合并成矩阵元再与 $x$ 相乘，`EA` 先逐单元相乘再散加。因此两侧结果只差浮点舍入。代价也落在不同地方：`FA` 要常驻全局矩阵并每轮装配，`EA` 省去这两项，但每次矩阵向量积都要逐单元重算。

## 2. 实验设置

实验脚本为 [`run_fem_ea_vs_fa.py`](run_fem_ea_vs_fa.py)，两侧以 `--level ea`、`--level fa` 各运行一次，结果经 [`sync_results.py`](sync_results.py) 固化到 `results/`。

本实验均采用 NumPy / CPU：`BACKEND = 'numpy'`，`DEVICE = 'cpu'`；单线程：`NUM_THREADS = 1`。

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
| 刚度算子层级 | 对照变量，EA 不形成全局矩阵，FA 装配全局 CSR 矩阵：命令行 `--level ea` 或 `--level fa`，传入 `operator_level=args.level` |
| 分析网格 | 结构化六面体网格，各方向单元数由 `--grid` 设定，须满足 $6:1:1$：`mesh = HexahedronMesh.from_box(box=domain, nx=nx, ny=ny, nz=nz, ...)` |

**线性求解参数**

| 参数 | 数值或设置 |
|---|---|
| 求解器 | CG：`SOLVER = 'cg'`；预条件子由 `--preconditioner` 指定，可选 `none`、`jacobi`、`gmg`，本实验取默认的 `jacobi`：`PRECONDITIONER = 'jacobi'` |
| 停机准则 | 相对右端项的容差 $10^{-6}$：`CG_TOLERANCE = 1.0e-6`，`rtol=CG_TOLERANCE`，`atol=0`，`rtol_reference='rhs'` |
| 最大 CG 步数 | `CG_MAXITER = 20000` |
| 迭代初值 | 首轮为零向量；第 2 轮起取上一轮的解：`x0=previous_displacement`，两侧均为全场位移 |

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
python experiments/topopt_simp_fem/run_fem_ea_vs_fa.py --level ea --max-iter 3
python experiments/topopt_simp_fem/run_fem_ea_vs_fa.py --level fa --max-iter 3
```

### 3.1 结果一致性

| 指标 | EA | FA |
|---|---|---|
| 终止 | 达到迭代上限，3 轮 | 达到迭代上限，3 轮 |
| 逐轮柔顺度 $C$ | $2562.4025186343$，$655.0507013970$，$358.8702794654$ | $2562.4025186829$，$655.0507016027$，$358.8702793925$ |
| 第 3 轮体积分数 | $0.119954$ | $0.119954$ |
| 每轮 CG 步数 | 1295，2253，2299 | 1295，2252，2299 |
| 末轮 CG 相对残差 | $9.96\times10^{-7}$ | $9.86\times10^{-7}$ |

两侧逐轮柔顺度相对差不超过 $3.1\times10^{-10}$，第 3 轮全场位移相对 $L^2$ 差为 $3.4\times10^{-8}$，阈值 $0.5$ 下的二值化拓扑逐单元相同，均在 CG 容差 $10^{-6}$ 以内。

### 3.2 计算成本

| 阶段 | 方法 | 对应步骤 | 耗时 / s | 常驻 / GiB | 临时 / GiB |
|---|---|---|---|---|---|
| 组装 | EA | 只更新单元模量 $E(\rho_e)$ | 0.00 | 0.07 | 0.00 |
| | FA | 按 $E(\rho_e)$ 缩放 $K^0$，散加为 $K$ | 0.54 | 1.41 | 0.25 |
| 边界条件 | EA | 包装为 `ConstrainedOperator` | 0.01 | 0.00 | 0.00 |
| | FA | 复制值数组并对称消元 | 0.05 | 0.00 | 0.48 |
| 求解 | EA | CG，逐单元矩阵向量积 | 122.32 | 0.00 | 0.14 |
| | FA | CG，CSR 矩阵向量积 | 54.96 | 0.00 | 0.02 |

耗时与临时内存取第 2、3 轮的均值，常驻内存为首轮该阶段新增并保留到轮末的量。

耗时：EA 的组装只更新单元模量，几乎不耗时；FA 每轮组装 0.54 s，首轮另建 CSR 骨架与装配缓存，为 2.73 s。差异主要在求解：两侧 CG 步数几乎相同，按步均摊 EA 约 53.7 ms/步，FA 约 24.2 ms/步，EA 每步约慢 2.2 倍。每轮合计（另含过滤、灵敏度、输出、OC）EA 为 122.7 s，FA 为 55.9 s，约慢 2.2 倍。

内存：FA 组装的常驻量是全局 CSR 矩阵 $K$ 及其装配缓存；施加边界条件时另复制一份值数组，持有到求解结束才释放，故计入临时。EA 只常驻 `cell2dof` 与单元模量，边界条件不复制数据，求解的临时内存为逐单元矩阵向量积的中间数组。整个运行的峰值：EA 为 1.50（出现在准备阶段构造过滤矩阵），FA 为 2.98（出现在求解阶段，含边界条件复制的值数组）。其余内存（网格、过滤矩阵等）两侧相同，约 1.03。

数据来源：`results/ea_hex_210x35x35_cg_jacobi/`、`results/fa_hex_210x35x35_cg_jacobi/`、`results/final_fields.json`。

## 4. 实验结论

1. **结果一致**：EA 与 FA 前 3 轮的逐轮柔顺度相对差不超过 $3.1\times10^{-10}$，每轮 CG 步数至多相差 1 步，阈值 $0.5$ 下的二值化拓扑逐单元相同。两种刚度算子层级对优化结果等价。
2. **耗时**：本文设置（NumPy / CPU，单线程，Jacobi 预条件 CG）下 FA 更快，每轮 55.9 s 对 122.7 s。差异几乎全在求解：EA 不形成全局矩阵，每次矩阵向量积都要逐单元作用 $K^0$，每步 53.7 ms 对 24.2 ms。FA 每轮多出的装配 0.54 s（首轮 2.73 s）相对求解可以忽略。EA 的单元作用经由 BLAS，FA 的稀疏矩阵向量积不经由 BLAS，多线程下两者的快慢可能改变，尚待验证。
3. **内存**：EA 更省，全程峰值 1.50 对 2.98 GiB。FA 多出的部分包括常驻的 CSR 矩阵与装配缓存（1.41 GiB），以及施加边界条件时复制的值数组（0.48 GiB）。前者是全局矩阵形式固有的开销；后者属于当前实现，改为原地消元即可消除。
