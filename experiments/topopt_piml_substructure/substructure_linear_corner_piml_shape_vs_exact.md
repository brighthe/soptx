# linear_corner 下形函数预测 PIML 与精确子结构的拓扑优化结果对照

硬件：Intel Core i9-14900KF（24 核 32 线程，8P+16E），运行内存 45 GiB；NVIDIA GeForce RTX 5080，显存 16 GB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（内置 scipy-openblas 0.3.34），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.13.0（CUDA 13.0）；PyMUMPS 0.4.0，链接系统 MUMPS 5.6.2（OpenBLAS 0.3.26、Open MPI 4.1.6、ScaLAPACK 2.2.1，单进程运行）；pyevtk 1.7.0。


## 1. 研究对象

对照相同细网格、相同子结构划分及 `linear_corner` 下，精确子结构（Exact）与形函数预测 PIML（shape 路线）在 Huang2023 三维 MBB 梁上的完整优化结果。

两侧的子结构位移都用基函数插值表示：$u^j=N^jq^j$，$q^j$ 为 8 个角点的 24 个位移，$N^j$ 为子结构的多尺度基函数。其边界节点行 $L$ 由角点位移三线性插值给出，两侧相同；内部节点行给出内部位移，`Exact` 取某个角点位移为单位值、其余为零时子结构的静力位移，即由边界位移精确回代，`PIML_shape` 由网络预测。两侧共用全局角点未知量 $U_C$ 与载荷 $F_C$，差别只在基函数内部部分 $N_i^j$ 的来源。

| 项目 | Exact | PIML shape |
|---|---|---|
| 子结构内部位移 | 边界位移精确回代：$u_i^j=-(K_{ii}^j)^{-1}K_{ib}^ju_b^j$ | 网络预测延拓：$\widehat u_i^j=\widehat N_i^jq^j$，$\widehat N_i^j$ 由网络按子结构内各单元的模量预测，一般不满足内部平衡 |
| 刚度算子 | 各子结构 Schur 补投影到角点 $L^{\mathsf T}K_s^jL$，组装为全局角点矩阵 $K_C$ | 各子结构用预测的基函数按变分式 $(\widehat N^j)^{\mathsf T}K^j\widehat N^j$ 构造，组装为 $\widehat K_C$，内部未消元，比 $L^{\mathsf T}K_s^jL$ 偏硬 |

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

其中 $f_b^j$ 为子结构边界节点上的节点力，组装后边界右端项只剩外载荷 $F_\Gamma$；内部载荷为零。两侧都取 `linear_corner` 参数化：边界位移为 8 个角点位移的三线性插值，$u_b^j=Lq^j$，$q^j$ 为子结构 8 个角点的 24 个位移，$L\in\mathbb R^{n_b\times24}$ 与材料无关，所有子结构相同。

**局部缩聚**。第二行 $K_{ib}^ju_b^j+K_{ii}^ju_i^j=0$ 即内部平衡方程。代入 $u_b^j=Lq^j$，内部位移对角点位移的响应记为 $N_i^j\in\mathbb R^{n_i\times24}$，$n_i$ 为子结构内部自由度数，$u_i^j=N_i^jq^j$。两侧的差别只在 $N_i^j$ 怎么来：

- `Exact` 解内部平衡方程：$N_i^j=-(K_{ii}^j)^{-1}K_{ib}^jL$。
- `PIML_shape` 不解方程，由网络以子结构内各细单元的模量 $E(\rho_e)$ 为输入直接预测 $\widehat N_i^j$（网络输出经刚体约束补全为 $n_i\times24$）。$\widehat N_i^j$ 一般不满足内部平衡 $K_{ii}^j\widehat N_i^j=-K_{ib}^jL$。

子结构位移 $u^j=N^jq^j$，$N^j=[L;N_i^j]$ 为多尺度基函数，缩聚刚度按变分式 $K_c^j=(N^j)^{\mathsf T}K^jN^j$ 构造，展开为四项：

$$
K_c^j=L^{\mathsf T}K_{bb}^jL+L^{\mathsf T}K_{bi}^jN_i^j
+(N_i^j)^{\mathsf T}\bigl(K_{ib}^jL+K_{ii}^jN_i^j\bigr).
$$

将两侧的 $N_i^j$ 代入：

- `Exact`：$N_i^j$ 满足内部平衡，括号项为零，$K_c^j=L^{\mathsf T}K_{bb}^jL+L^{\mathsf T}K_{bi}^jN_i^j=L^{\mathsf T}K_s^jL$，其中 $K_s^j=K_{bb}^j-K_{bi}^j(K_{ii}^j)^{-1}K_{ib}^j$ 为 Schur 补。
- `PIML_shape`：$\widehat N_i^j$ 不满足内部平衡，括号项不为零，$\widehat K_c^j$ 须保留四项；若只取前两项，刚度不对称且无正定性保证。对预测的 $\widehat K_c^j$ 逐子结构记录三项诊断量：刚体模态下的内力（$\widehat K_c^jZ_c$，$Z_c$ 为 24 个角点自由度上的 6 个刚体模态）、相对内部固支刚度 $L^{\mathsf T}K_{bb}^jL$ 的刚化幅度、变形子空间上的条件数。它们只用于衡量预测质量，不触发替换，见 §2.4。

**全局组装**。各子结构的 $q^j$ 是全局角点位移 $U_C$ 的分量，$q^j=A_c^jU_C$，$A_c^j$ 为限制矩阵。两侧以相同方式组装全局角点刚度：

$$
K_C=\sum_j(A_c^j)^{\mathsf T}K_c^jA_c^j,\qquad
\widehat K_C=\sum_j(A_c^j)^{\mathsf T}\widehat K_c^jA_c^j\in\mathbb R^{N_C\times N_C},
$$

$N_C$ 为全局角点自由度数，两侧同阶。

### 1.2 求解与位移恢复

两侧先解全局角点系统得到角点位移，提取各子结构的参数 $q^j$，由 $u_b^j=Lq^j$ 得到边界位移，再由各自的 $N_i^j$ 得到内部位移。`Exact` 的内部位移是精确回代，`PIML_shape` 的内部位移是预测延拓：

$$
\text{Exact:}\quad
\begin{cases}
K_CU_C=F_C,\\
q^j=A_c^jU_C,\\
u_b^j=Lq^j,\quad u_i^j=N_i^jq^j,
\end{cases}
\qquad
\text{PIML\_shape:}\quad
\begin{cases}
\widehat K_C\widehat U_C=F_C,\\
\widehat q^j=A_c^j\widehat U_C,\\
\widehat u_b^j=L\widehat q^j,\quad \widehat u_i^j=\widehat N_i^j\widehat q^j .
\end{cases}
$$

其中 $F_C=P^{\mathsf T}F_\Gamma=\sum_j(A_c^j)^{\mathsf T}L^{\mathsf T}f_b^j$ 为投影到角点的载荷。

两侧的子结构位移都按 1.1 写为 $u^j=N^jq^j$，`Exact` 取 $N^j=[L;N_i^j]$，`PIML_shape` 取 $\widehat N^j=[L;\widehat N_i^j]$。两侧按全局编号拼成全场位移的方式相同：

$$
U=R_\Gamma^{\mathsf T}PU_C+\sum_j(R_i^j)^{\mathsf T}u_i^j,\qquad
\widehat U=R_\Gamma^{\mathsf T}P\widehat U_C+\sum_j(R_i^j)^{\mathsf T}\widehat u_i^j .
$$

`Exact` 的内部位移由内部平衡方程解出，恒满足 $K_{ib}^ju_b^j+K_{ii}^ju_i^j=0$。`PIML_shape` 的内部位移由网络预测，不是该方程的解；记预测误差 $D^j=\widehat N_i^j-N_i^j$，代入得

$$
K_{ib}^j\widehat u_b^j+K_{ii}^j\widehat u_i^j=K_{ii}^jD^j\widehat q^j ,
$$

只要 $D^j\ne0$ 就不为零。这部分内力在内部节点上没有外力与之平衡，$\widehat u^j$ 因而不是平衡状态。

### 1.3 代理误差关系

以下关系针对同一物理密度场、相同材料、载荷与齐次支承，并假设两侧系统正定且精确求解。

**局部刚度**。由 1.2 的预测误差 $D^j$ 与 Exact 的内部平衡方程，四项式相减得

$$
\widehat K_c^j-K_c^j=(D^j)^{\mathsf T}K_{ii}^jD^j\succeq0 ,
$$

预测只会使角点刚度变硬，组装后 $\widehat K_C\succeq K_C$。

**柔顺度**。两侧恢复的全场位移共用边界部分 $L$，在子结构边界上连续，是细网格空间的协调子空间元素；载荷只落在边界节点上，$F^{\mathsf T}U=F_C^{\mathsf T}U_C$。于是

$$
\widehat C\le C_{\mathrm{Exact}}\le C_{\mathrm{FEM}},\qquad
C_{\mathrm{FEM}}-\widehat C=\|U_{\mathrm{FEM}}-\widehat U\|_K^2 .
$$

两个不等号来源不同：$C_{\mathrm{Exact}}\le C_{\mathrm{FEM}}$ 来自边界位移的角点插值，是 linear_corner 自身的 Ritz 误差，两侧共有；$\widehat C\le C_{\mathrm{Exact}}$ 来自刚度不等式，是 PIML 额外引入的。能量恒等式是协调 Ritz 近似的通用结论，对两侧都成立。

**评价准则**。PIML 报告的柔顺度偏低不表示设计更好。评价以统一细网格重分析 $C_{\mathrm{Fine}}$ 为准，对 PIML 恒有 $C_{\mathrm{Fine}}\ge C_{\mathrm{opt}}$；§3.1 同时报告 $C_{\mathrm{FEM}}$、$C_{\mathrm{Exact}}$、$\widehat C$，以分离两层误差。

## 2. 实验设置

实验脚本为 [`run_substructure_exact_vs_piml_shape.py`](run_substructure_exact_vs_piml_shape.py)。

本实验均采用 NumPy / CPU：`BACKEND = 'numpy'`，`DEVICE = 'cpu'`；单线程：`NUM_THREADS = 1`（`--threads` 的默认值），脚本在导入 numpy/scipy/torch 之前把全部 BLAS / OpenMP 线程变量设为该值，torch 推理同用，取值记入 `config.json` 的 `threads`，输出目录名带 `_t1`。

### 2.1 模型参数

![MBB 梁的几何、载荷与支承](assets/Huang2023_Fig5.png)

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

### 2.4 PIML 形函数预测

本节为两侧的全部差别。离线两步决定权重，在线脚本不执行。

**离线阶段：样本生成**。

```bash
python experiments/topopt_piml_substructure/generate_samples.py \
  --n-sub 78 13 13 --n-fine 5 --fine-cell-size 1.0 --nu 0.3 --trace-kind linear_corner \
  --sampling independent_uniform --n-train 400000 --n-validation 40000 \
  --min-modulus 1e-7 --sample-seed 2026 --targets shape \
  --backend numpy --device cpu --generation-batch-size 32
```

产物写入脚本旁的 `outputs/independent_15_layer/samples/`，目录名由参数自动拼出：`mbb_linear_corner_m5_h1_sub78x13x13_nu0p3_emin1e-7_shape_train400000_val40000_seed2026`。

| 参数 | 数值或设置 |
|---|---|
| 子结构 | $m=5$，$h=1$，$\nu=0.3$，线性元，linear_corner 接口：`--n-fine 5`，`--fine-cell-size 1.0`，`--nu 0.3`，`--trace-kind linear_corner`；`--n-sub 78 13 13` 只决定记入 `generation_config.json` 的整体域尺寸，不影响局部样本 |
| 采样量 | 各细单元的归一化杨氏模量 $\widehat E_e$，逐单元独立均匀采样：`--sampling independent_uniform` |
| 采样范围 | $\widehat E_e\in[10^{-7},1]$，上界固定为 $1$：`--min-modulus 1e-7` |
| 样本数 | 训练 $400\,000$，验证 $40\,000$：`--n-train 400000`，`--n-validation 40000` |
| 采样种子 | $2026$，驱动材料采样的随机流：`--sample-seed 2026` |
| 标签 | 精确内部形函数 $N_i^j=-(K_{ii}^j)^{-1}K_{ib}^jL$，只写形函数标签：`--targets shape` |
| 生成后端 | NumPy / CPU，float64：`--backend numpy`，`--device cpu`，`--generation-batch-size 32` |

**离线阶段：网络训练**。

```bash
python experiments/topopt_piml_substructure/train_network.py \
  --samples-dir experiments/topopt_piml_substructure/outputs/independent_15_layer/samples/mbb_linear_corner_m5_h1_sub78x13x13_nu0p3_emin1e-7_shape_train400000_val40000_seed2026 \
  --route shape --num-networks 4 \
  --hidden-dims 60 80 100 120 140 160 180 200 180 160 140 120 100 80 60 \
  --activations tanh elu tanh elu tanh elu tanh elu elu tanh elu tanh elu tanh elu \
  --input-normalization none --loss decoded_full_matrix_mse --stiffness-loss-weight 0 \
  --epochs 500 --batch-size 256 --optimizer adam --lr 1e-3 --weight-decay 0 --patience 40 \
  --train-seed 2026 --backend pytorch --device cpu
```

权重写入 `outputs/independent_15_layer/training/shape/<数据集名>_trainseed2026/shape_best.pt`，即在线脚本 `SHAPE_WEIGHTS` 的默认值。

| 参数 | 数值或设置 |
|---|---|
| 网络数 | $4$ 个网络共用同一输入，形函数的独立条目按下标连续均分给各网络，输出拼接后按刚体约束补全：`--num-networks 4` |
| 隐藏层宽度 | $[60,80,100,120,140,160,180,200,180,160,140,120,100,80,60]$：`--hidden-dims 60 80 ... 60` |
| 隐藏层激活 | $[\tanh,\mathrm{elu},\tanh,\mathrm{elu},\tanh,\mathrm{elu},\tanh,\mathrm{elu},\mathrm{elu},\tanh,\mathrm{elu},\tanh,\mathrm{elu},\tanh,\mathrm{elu}]$：`--activations tanh elu ... elu` |
| 输入归一化 | 不做：`--input-normalization none` |
| 损失 | 网络输出经刚体约束补全为 $\widehat N_i^j$ 后与精确 $N_i^j$ 的逐元素 MSE：`--loss decoded_full_matrix_mse` |
| 刚度一致性损失 | 叠加于基本损失之上的加权项：由 $\widehat N_i^j$ 按变分式构造的角点刚度与精确刚度标签的相对误差；权重 $0$ 即不启用：`--stiffness-loss-weight 0` |
| 优化器 | Adam：`--optimizer adam` |
| 初始学习率 | $10^{-3}$：`--lr 1e-3` |
| 权重衰减 | $0$：`--weight-decay 0` |
| batch size | $256$：`--batch-size 256` |
| 提前停止 | 验证损失 $40$ 轮未降：`--patience 40` |
| 最大轮数 | $500$：`--epochs 500` |
| 训练种子 | $2026$，驱动网络初始化与每轮样本打乱：`--train-seed 2026` |
| 训练后端 | PyTorch / CPU，float64：`--backend pytorch`，`--device cpu` |

**在线阶段：预测、缩聚与恢复**。仅 `--side piml_shape`：每轮装配时以各子结构的 $E(\rho_e)/E_0$ 为输入，网络预测 $\widehat N_i^j$，缩聚按 §1.1，求得角点位移后内部位移同样用 $\widehat N_i^j$ 回代；启动时核对权重的局部配置与本实验一致。

| 参数 | 数值或设置 |
|---|---|
| 权重文件 | 上述训练得到的 `shape_best.pt`：`--shape-weights`，默认 `SHAPE_WEIGHTS` |

每轮另按子结构记录三项预测质量比值的分布到 `history.json` 的 `condensation_report`，只记录不替换：刚体零空间残量 $\|\widehat K_c^jZ_c\|_F/\|\widehat K_c^j\|_F$、相对 $L^{\mathsf T}K_{bb}^jL$ 的刚化比 $\lambda_{\max}(\widehat K_c^j-L^{\mathsf T}K_{bb}^jL)/\lambda_{\max}(L^{\mathsf T}K_{bb}^jL)$、变形子空间条件数倒数 $\lambda_{\min}/\lambda_{\max}$。

## 3. 结果对照

```bash
python experiments/topopt_piml_substructure/run_substructure_exact_vs_piml_shape.py --side exact
python experiments/topopt_piml_substructure/run_substructure_exact_vs_piml_shape.py --side piml_shape
```

### 3.1 固定同一密度场的分析对照

先固定物理密度进行双侧分析，不执行 OC 更新。样本包括均匀场、非对称非均匀场及优化中间场；对优化场记录其来源，避免仅在单侧最终设计上评价。使用同一输入密度比较，才能分离代理误差与优化轨迹变化。

| 指标 | Exact | PIML shape | 对照指标 |
|---|---|---|---|
| 密度场来源及文件标识 | 待填 | 同一文件，待填 | 输入一致性待核对 |
| 基函数内部部分 $N_i^j$ | 待填 | 待填 | 相对 Frobenius 误差，待填 |
| 局部角点刚度 | 待填 | 待填 | 相对 Frobenius 误差，待填 |
| 全局角点位移 | 待填 | 待填 | 相对 L2 差，待填 |
| 恢复后细网格位移 | 待填 | 待填 | 相对 L2 差，待填 |
| 柔顺度 | 待填 | 待填 | 带符号相对差，待填 |
| 物理密度 / 设计密度梯度 | 待填 | 待填 | 分别报告相对差与方向差，待填 |
| 梯度方向有限差分检查 | 待填 | 待填 | 步长及误差，待填 |
| 实际系统真残差 | 待填 | 待填 | 容差待确定 |
| 预测、缓存及回退数量 | 不适用或按实际策略记录 | 待填 | 覆盖比例待填 |

矩阵与位移相对差以 Exact 范数为分母，零范数时另报绝对差；柔顺度带符号差为（C_PIML-C_Exact）/|C_Exact|。局部指标同时记录分布与最大值，不仅报告平均值。

PIML 的 N 通常随材料变化。对固定载荷的角点系统，柔顺度导数应依据实际模型：

$$
\frac{\mathrm d C}{\mathrm d\rho_e}
=-U_C^T\frac{\mathrm d\widehat K_C}{\mathrm d\rho_e}U_C.
$$

其中预测基函数的导数一般不能自动忽略；若载荷映射也随材料变化，还需包含载荷导数项。Exact 内部平衡导致的消项不能直接移植到近似形函数。应记录使用完整梯度还是近似梯度，并用有限差分核对，否则完整优化差异不能仅归因于状态求解误差。

### 3.2 完整拓扑优化结果

| 运行信息 | Exact linear_corner | PIML shape linear_corner |
|---|---|---|
| 日期 / 硬件 / 软件 / BLAS / 线程 | 待填 | 待填 |
| Git revision / 工作区状态 | 待填 | 待填 |
| 原始输出目录 / 配置记录 | 待填 | 待填 |
| 求解器与预条件器 | CG + Jacobi，同 §2 | CG + Jacobi，同 §2 |
| 小规模验收 / 正式运行状态 | 待填 | 待填 |
| 求解失败、回退或其他异常 | 待填 | 待填 |
| 权重路径、sha256、训练代码版本 | 不适用 | 待填，取 `config.json` 的 `checkpoint` |
| 实际网络结构、激活、输出分组及 dtype | 不适用 | 待填 |
| 训练 / 验证 / 独立测试数据及分布 | 不适用 | 待填，取训练目录的 `run_config.json` 与样本 `manifest.json` |
| 输入排序、预处理和额外缩放 | 不适用 | 待填，取 `checkpoint.input_normalization` |
| 权重训练结果与验收状态 | 不适用 | 跑满 500 轮，最优轮 491，验证损失 $6.39\times10^{-6}$；无独立测试集，验证损失为唯一泛化指标；README 记录默认权重尚未通过正式复现前置验收 |

| 指标 | Exact linear_corner | PIML shape linear_corner |
|---|---|---|
| 全局角点自由度，约束前 | 待填 | 待填 |
| 优化终止状态 / 轮数 | 待填 | 待填 |
| 最终 C_opt | 待填 | 待填 |
| 最终物理体积分数 | 待填 | 待填 |
| 最终 C_Fine，统一 FEM 重分析 | 待填 | 待填 |
| 自身设计上的模型偏差 | 待填 | 待填 |
| 线性迭代数，首轮 / 均值 / 最大 / 总计 | 待填 | 待填 |
| 真残差，末轮 / 最大 | 待填 | 待填 |
| 物理密度对称误差 | 待填 | 待填 |
| 预测 / 缓存 / 回退累计数量 | 待填 | 待填 |
| 最终拓扑图 / 场文件 | 待填 | 待填 |

| 两份最终设计的差异 | PIML 相对 Exact |
|---|---|
| C_Fine 带符号相对变化 | 待填 |
| 物理密度相对 L2 差 | 待填 |
| 阈值 0.5 的二值拓扑不一致比例 | 待填 |

优化收敛曲线应同时展示柔顺度与体积分数，拓扑图采用相同视角、阈值和色标。两侧可能有不同优化轮数，不能用总耗时单独推断单次分析加速比。

### 3.3 最终设计的统一 FEM 重分析

对两侧最终物理密度，使用同一细网格 FEM、材料插值、载荷、支承及求解设置各分析一次。不再次过滤，不执行 OC 更新。记录：

$$
\varepsilon_{\mathrm{model}}=
\frac{|C_{\mathrm{opt}}-C_{\mathrm{Fine}}|}{|C_{\mathrm{Fine}}|},\qquad
\Delta_{\mathrm{design}}=
\frac{C_{\mathrm{Fine}}(\rho_{\mathrm{PIML}})-C_{\mathrm{Fine}}(\rho_{\mathrm{Exact}})}
{|C_{\mathrm{Fine}}(\rho_{\mathrm{Exact}})|}.
$$

前者评价各自模型在自身设计上的偏差，后者比较最终设计质量，必须结合体积分数解读。对 PIML 最终设计还应补做 Exact linear_corner 重分析，以分离同一设计上的代理偏差与角点插值偏差。重分析入口和求解配置待确定，实测数据待填。

## 4. 计算成本

成本分为离线训练、在线分析与完整优化。同一硬件和线程条件下对照，推理设备或精度不同须明示。不预设 PIML 一定更快：网络推理之外，细尺度局部刚度构造、N 转置 K N、数据转换及恢复仍可能占主要成本。

### 4.1 在线耗时（s）

| 准备阶段 | Exact | PIML shape |
|---|---|---|
| 网格与子结构布局 | 待填 | 待填 |
| 材料、参考刚度、边界位移表示与分析器 | 待填 | 待填 |
| 权重加载、契约检查与推理初始化 | 不适用 | 待填 |
| 过滤矩阵、目标与约束 | 待填 | 待填 |
| 准备合计 | 待填 | 待填 |

| 每轮阶段 | Exact | PIML shape |
|---|---|---|
| 密度过滤及材料插值 | 待填 | 待填 |
| 局部细尺度刚度构造 | 待填 | 待填 |
| 精确内部消元及角点投影 | 待填 | 回退部分待填，无回退则不适用 |
| 输入转换、网络推理及基函数补全 | 不适用 | 待填 |
| 预测基函数构造角点刚度 | 不适用 | 待填 |
| 全局角点矩阵组装 | 待填 | 待填 |
| 载荷与边界条件 | 待填 | 待填 |
| 线性求解 | 待填 | 待填 |
| 全场位移恢复 | 待填 | 待填 |
| 柔顺度与灵敏度 | 待填 | 待填 |
| 输出 | 待填 | 待填 |
| OC 更新 | 待填 | 待填 |
| 首轮完整耗时 | 待填 | 待填 |
| 第 2 轮起每轮完整耗时均值 | 待填 | 待填 |

上述是目标计时分项，现有入口是否提供相应字段尚待核实。如果实际实现融合了若干步骤，报告可测的合并阶段，不凭空拆分耗时。缓存判别与回退耗时应归入相应阶段并另记次数，避免重复计时。GPU 推理若启用，需明确同步与主机设备传输计时边界。

| 汇总 | Exact | PIML shape |
|---|---|---|
| 固定同一密度场的单次分析耗时 | 待填 | 待填 |
| 准备合计 / s | 待填 | 待填 |
| 全部优化迭代耗时 / s | 待填 | 待填 |
| 准备与优化总耗时 / min | 待填 | 待填 |
| 最终设计 FEM 重分析 / s，单列 | 待填 | 待填 |
| 单次分析 / 完整优化加速比 | 基准 | 待填 |

加速比定义为 Exact 耗时除以 PIML 耗时，两者必须使用相同计时范围。每轮总耗时先对实际阶段求和再取均值，与包含阶段外开销的墙钟时间区分。OC 仅统计实际执行轮次；只有一轮时稳态均值不适用。首轮、模型预热及缓存建立单列，不隐去其成本。

### 4.2 内存（GiB）

| 阶段峰值或汇总 | Exact RSS | PIML RSS |
|---|---|---|
| 网格与布局准备 | 待填 | 待填 |
| 分析器、参考数据与模型加载 | 待填 | 待填 |
| 过滤矩阵及优化准备 | 待填 | 待填 |
| 局部计算与整体组装 | 待填 | 待填 |
| 载荷与边界条件 | 待填 | 待填 |
| 求解与位移恢复 | 待填 | 待填 |
| 灵敏度计算 | 待填 | 待填 |
| 场输出与 OC | 待填 | 待填 |
| 准备阶段峰值 | 待填 | 待填 |
| 优化阶段峰值 | 待填 | 待填 |
| 全进程峰值 | 待填 | 待填 |

RSS 为进程常驻内存绝对值，阶段峰值不可相加。记录测量工具、采样方式及是否包含子进程。若启用 GPU，显存峰值单列表记录，不与 RSS 混合；框架 allocated、reserved 与设备进程用量采用何种口径需说明。不在缺少证据时推断临时内存组成。

### 4.3 离线成本与摊销

| PIML 离线项目 | 耗时 / 资源记录 |
|---|---|
| 样本生成与精确标签 | 待填 |
| 网络训练与选模 | 待填 |
| 独立测试与模型验收 | 待填 |
| 数据集 / 权重存储量 | 待填 |
| 可复用的几何、材料与边界表示范围 | 待填 |

离线成本不混入单轮在线成本，但需与在线收益并列报告。只有在每次任务在线成本确实降低且模型可复用时，才讨论摊销次数；不以未测量的网络推理耗时宣称整体加速。

## 5. 验证状态与后续步骤

本轮仅重写文档，未运行测试、训练、benchmark 或优化。后续按以下顺序推进，运行前另行明确命令、验收阈值并取得授权：

1. 检查并复用现有分析和优化入口，确认两侧相同的 L、编号、材料映射、载荷及支承；确定求解器配置。
2. 选择与局部尺寸相容的权重，固定模型与输入契约；明确缓存、均匀近似、质量诊断量和梯度实现。
3. 在小规模固定密度上核对局部刚度、载荷、位移、柔顺度、梯度及真残差，完成计时与内存字段检查。
4. 完成短程优化流程验收后再执行正式对照，记录原始配置、逐轮数据和失败情况。
5. 对最终设计进行统一 FEM 重分析，填写结果与成本表，再形成精度与性能结论。

参考文档中的实测结果或历史软件版本不自动适用于本文。本目录 README 说明尚无完整 PIML 拓扑优化执行入口，因此这里的实验设计不代表该入口已经接通；在检查现有能力前不提供虚构运行命令。

## 6. 未来工作

以下是本实验不做、但方法上值得做的事，与本实验的完成步骤（§5）分开。

1. **预测质量检查与改用 Exact**。soptx 已实现三项检查及未通过者改用 Exact 缩聚（§2.4），Huang2023 无此步骤，本实验只记录三项诊断量不替换。若 §3.1 的诊断显示某类密度场的预测质量明显偏差，可启用替换；届时结果须标明为预测与精确缩聚的混合流程，并记录替换比例，1.3 的不等式在被替换的子结构上以 $D^j=0$ 平凡成立。
2. **直接预测刚度路线**。Huang2023 的路线一直接由网络预测 $\widehat K_c^j$，本文不比较。论文自述该路线在柔性机构算例上结果更差，且预测的刚度与预测的 $N^j$ 一般不满足变分式，1.3 的关系对它不成立。
3. **自洽损失**。Huang2023 在训练末期加入第二类损失，即由预测 $N^j$ 经变分式算出的刚度与直接预测刚度之差。本文网络是否使用记入 §2.4，可作为改进训练的选项。
4. **均匀替代的变分化**。现行均匀子结构替代用缩放的实体刚度，不是变分构造，阈值量级内可能破坏 1.3 的不等式。可改为用实体基函数经变分式 $(N^j)^{\mathsf T}K^jN^j$ 构造，代价是每个均匀子结构多一次矩阵乘。
