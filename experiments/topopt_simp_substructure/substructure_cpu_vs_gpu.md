# 子结构有限元 CPU 与 GPU 拓扑优化结果对照

硬件：AMD Ryzen 9 9950X（16 核 32 线程），运行内存 45 GiB（WSL2 上限 48 GB）；NVIDIA GeForce RTX 5070 Ti（70 SM，8960 CUDA 核心），显存 16 GiB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（conda-forge，链接 BLIS 2.0 与 netlib LAPACK），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.11.0（CUDA 12.8）；PyMUMPS 0.4.0，链接 MUMPS 5.8.2（BLIS 2.0、netlib LAPACK、Open MPI 5.0.11、ScaLAPACK 2.2.0）；pyevtk 1.7.0。

## 1. 研究对象

对照子结构边界位移的两种表示 `full_trace` 与 `linear_corner` 各自在不同设备 `CPU` 与 `GPU` 上对 Huang2023 三维 MBB 梁的拓扑优化，比较单轮迭代的计算代价与两侧结果的一致性。

|      | CPU | GPU |
| ---- | --- | --- |
| 设备 | AMD Ryzen 9 9950X，16 线程 | NVIDIA GeForce RTX 5070 Ti，70 SM |
| 内存 | 主机内存 45 GiB | 显存 16 GiB |

### 1.1 full_trace

各步骤都在 `--device` 所选的设备上执行，首轮的一次性准备在主机上完成：

| 步骤 | 说明 |
|---|---|
| 子结构局部组装 $K^j$ 与缩聚 $K_s^j$ | 按批的稠密运算，每批 256 个子结构 |
| 组装全局边界总刚 $K_\Gamma$ | 符号模式只在首轮于主机上构建一次，之后移到所选设备，每轮只重填数值 |
| 求解 $K_\Gamma U_\Gamma=F_\Gamma$ | 约束只在首轮于主机上解析一次，CG 迭代与真实残差验收都在所选设备上 |
| 内部位移回代 $u_i^j$ | 逐批重做局部组装与缩聚后回代；子结构全局编号在主机上逐个生成后拷入 |

### 1.2 linear_corner

各步骤都在 `--device` 所选的设备上执行，首轮的一次性准备在主机上完成：

| 步骤 | 说明 |
|---|---|
| 子结构局部组装 $K^j$ 与缩聚 $K_s^j$ | 按批的稠密运算，每批 256 个子结构；不显式形成 $K_s^j$，先把边界块投影到角点（$K_{ib}^jL$、$L^{\mathsf T}K_{bb}^jL$），内部求解只对 24 个右端项进行 |
| 角点投影 $K_c^j=L^{\mathsf T}K_s^jL$ | 与缩聚在同一批内完成：$K_c^j=L^{\mathsf T}K_{bb}^jL-(K_{ib}^jL)^{\mathsf T}(K_{ii}^j)^{-1}K_{ib}^jL$，代数上等于 $L^{\mathsf T}K_s^jL$；同时缓存角点恢复矩阵 $-(K_{ii}^j)^{-1}K_{ib}^jL$ |
| 组装全局角点总刚 $K_C$ | 符号模式只在首轮于主机上构建一次，之后移到所选设备，每轮只重填数值 |
| 求解 $K_CU_C=F_C$ | 约束只在首轮于主机上解析一次，CG 迭代与真实残差验收都在所选设备上 |
| 内部位移回代 $u_i^j$ | 直接用组装时缓存的角点恢复矩阵回代，不重做组装与缩聚；子结构全局编号在主机上逐个生成后拷入 |

## 2. 实验设置

实验脚本为 [`run_substructure_cpu_vs_gpu.py`](run_substructure_cpu_vs_gpu.py)，在仓库根目录依次运行，运行命令见各研究对象一节；除 `--trace`、`--device` 与 `--max-iter` 外，其余参数取默认值：`--grid 210 35 35`、`--operator-level fa`、`--preconditioner jacobi`、`--threads 16`。

计算后端：两侧均为 PyTorch：`BACKEND = 'pytorch'`。线程数：两侧均为 16，即 9950X 的物理核数：`--threads 16`（默认）。GPU 侧的线程数只作用于主机端运算。

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
| 分析网格 | 结构化六面体网格，各方向单元数由 `--grid` 设定，须满足 $6:1:1$；子结构运行中也可由 `--n-sub` 与 `--n-fine` 推出：`mesh = HexahedronMesh.from_box(box=domain, nx=nx, ny=ny, nz=nz, ...)` |

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
| 迭代初值 | 首轮为零向量；第 2 轮起取上一轮的解：`x0=previous_displacement`，`full_trace` 为全局边界位移，`linear_corner` 为全局角点位移 |

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

## 3. full_trace

```bash
# full_trace
python experiments/topopt_simp_substructure/run_substructure_cpu_vs_gpu.py --trace full_trace --device cpu --max-iter 3
python experiments/topopt_simp_substructure/run_substructure_cpu_vs_gpu.py --trace full_trace --device cuda --max-iter 3
```

### 3.1 结果一致性

| 指标 | CPU | GPU |
| --- | --- | --- |
| 终止 | 达到迭代上限，3 轮 | 达到迭代上限，3 轮 |
| 逐轮柔顺度 $C$ | $2562.4025185843$，$655.0505960037$，$358.8704188436$ | $2562.4025188758$，$655.0506050008$，$358.8704009722$ |
| 第 3 轮体积分数 | $0.119954$ | $0.119954$ |
| 每轮 CG 步数 | 787，1377，1413 | 787，1375，1417 |
| 末轮 CG 相对残差 | $9.61\times10^{-7}$ | $9.84\times10^{-7}$ |

### 3.2 计算成本

| 阶段 | 对应步骤 | 设备 | 耗时 / s | 常驻 / GiB | 临时 / GiB |
|---|---|---|---|---|---|
| 组装 | 局部组装、缩聚与 $K_\Gamma$ 组装 | CPU | 6.76 | 4.46 | 3.12 |
| | | GPU | 1.13 | 4.44 | 3.12 |
| 接口求解 | 求解 $K_\Gamma U_\Gamma=F_\Gamma$ | CPU | 99.15 | 0.01 | 0.03 |
| | | GPU | 8.87 | 0.00 | 0.04 |
| 回代 | 内部位移回代 | CPU | 3.86 | 0.00 | 3.12 |
| | | GPU | 0.66 | 0.00 | 3.14 |

耗时：GPU 的组装约快 6.0 倍，接口求解约快 11.2 倍，回代约快 5.8 倍；接口求解按 CG 步数均摊，CPU 约 71 ms/步，GPU 约 6.4 ms/步。每轮合计（另含边界条件与过滤、灵敏度、输出、OC）CPU 为 110.1 s，GPU 为 10.9 s，约快 10.1 倍。

内存：组装的常驻量就是全局边界总刚 $K_\Gamma$ 的 CSR 矩阵及其散加槽位表。组装的临时内存为逐批缩聚的中间量，回代的临时内存为逐批重做局部组装与缩聚的中间量；接口求解只用到 CG 的几个工作向量，几乎不占内存。整个运行的峰值：CPU 运行的主机内存为 9.30；GPU 运行的显存为 8.13，主机内存为 7.60（出现在首轮组装，$K_\Gamma$ 的符号模式在主机上构建）。

数据来源：`results/cpu_fa_full_trace_hex_210x35x35_m5_cg_jacobi_t16/`、`results/cuda_fa_full_trace_hex_210x35x35_m5_cg_jacobi_t16/`。

## 4. linear_corner

```bash
# linear_corner
python experiments/topopt_simp_substructure/run_substructure_cpu_vs_gpu.py --trace linear_corner --device cpu --max-iter 3
python experiments/topopt_simp_substructure/run_substructure_cpu_vs_gpu.py --trace linear_corner --device cuda --max-iter 3
```

### 4.1 结果一致性

| 指标 | CPU | GPU |
| --- | --- | --- |
| 终止 | 达到迭代上限，3 轮 | 达到迭代上限，3 轮 |
| 逐轮柔顺度 $C$ | $1238.0028583982$，$571.6995019388$，$315.7506794519$ | $1238.0028584446$，$571.6995017126$，$315.7506760357$ |
| 第 3 轮体积分数 | $0.119974$ | $0.119974$ |
| 每轮 CG 步数 | 252，420，441 | 252，418，442 |
| 末轮 CG 相对残差 | $9.72\times10^{-7}$ | $8.57\times10^{-7}$ |

### 4.2 计算成本

| 阶段 | 对应步骤 | 设备 | 耗时 / s | 常驻 / GiB | 临时 / GiB |
|---|---|---|---|---|---|
| 组装 | 局部组装、缩聚、角点投影与 $K_C$ 组装 | CPU | 2.30 | 0.15 | 1.91 |
| | | GPU | 0.194 | 0.09 | 1.95 |
| 接口求解 | 求解 $K_CU_C=F_C$ | CPU | 0.095 | 0.00 | 0.00 |
| | | GPU | 0.487 | 0.00 | 0.00 |
| 回代 | 内部位移回代 | CPU | 0.009 | 0.00 | 0.00 |
| | | GPU | 0.005 | 0.00 | 0.01 |

耗时：GPU 的组装约快 11.9 倍，GPU 侧组装中批量求解 $K_{ii}^j$ 约占六成；接口求解反而慢约 5 倍：$K_C$ 只有 $8256$ 个未知量，每步 CG 的计算量只需微秒量级，GPU 每步约 1.1 ms，基本是与问题规模无关的固定开销，且逐轮波动明显（第 1–3 轮依次约 1.1、0.9、1.3 ms/步），CPU 约 0.2 ms/步；回代直接用缓存的恢复矩阵，两侧都只要几毫秒。每轮合计（另含边界条件与过滤、灵敏度、输出、OC）CPU 为 2.68 s，GPU 为 0.92 s，约快 2.9 倍；GPU 每轮中在主机上写 VTU 的输出约 0.22 s，约占四分之一，限制了加速比。

内存：组装的常驻量主要是缓存的角点恢复矩阵 $-(K_{ii}^j)^{-1}K_{ib}^jL$（每个子结构 $192\times24$，共约 0.071），另有 $K_C$ 的 CSR 矩阵约 0.006，与 GPU 侧实测一致；CPU 侧偏高来自 RSS 的统计口径。组装的临时内存为逐批缩聚的中间量；接口求解与回代几乎不占内存，回代直接用缓存的恢复矩阵，不重做缩聚。整个运行的峰值：CPU 运行的主机内存为 3.75（出现在组装阶段）；GPU 运行的显存为 2.54，主机内存为 2.84（出现在准备阶段建过滤矩阵）。

数据来源：`results/cpu_fa_linear_corner_hex_210x35x35_m5_cg_jacobi_t16/`、`results/cuda_fa_linear_corner_hex_210x35x35_m5_cg_jacobi_t16/`。