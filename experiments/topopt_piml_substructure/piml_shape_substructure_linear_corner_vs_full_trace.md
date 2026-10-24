# PIML shape 子结构 linear_corner 与 full_trace 的对照分析

硬件：Intel Core i9-14900KF（24 核 32 线程，8P+16E），运行内存 45 GiB；NVIDIA GeForce RTX 5080，显存 16 GB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（内置 scipy-openblas 0.3.34），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.13.0（CUDA 13.0）；PyMUMPS 0.4.0，链接系统 MUMPS 5.6.2（OpenBLAS 0.3.26、Open MPI 4.1.6、ScaLAPACK 2.2.1，单进程运行）；pyevtk 1.7.0。

## 1. 研究对象

形函数预测 `PIML_shape` 下，子结构边界位移的两种表示 `linear_corner` 与 `full_trace` 对 Huang2023 三维 MBB 梁上的完整优化结果。

两侧的子结构位移都写为 $u^j=N^jq^j$，$N^j=[\Psi;N_i^j]$，边界位移 $u_b^j=\Psi q^j$。PIML 路线不解内部平衡方程，由网络按子结构内各细单元的模量输出内部部分 $\widehat N_i^j\in\mathbb R^{n_i\times n_q}$ 的 free 列 $\widehat N_{i,F}^j\in\mathbb R^{n_i\times(n_q-6)}$，pivot 列由刚体约束补全，再按变分式 $\widehat K_c^j=(\widehat N^j)^{\mathsf T}K^j\widehat N^j$ 构造缩聚刚度。两种表示的差别只在 $\Psi$，由此决定 $n_q$、网络输出维数与每一步的运算量。

三维问题中记 $n_i$、$n_b$ 为子结构内部与边界自由度数，刚体模态 6 个，$N_C$、$N_\Gamma$ 为全局角点与全局边界自由度数。对 $m\times m\times m$ 个 Q1 细单元的子结构，内部节点 $(m-1)^3$ 个，边界节点 $(m+1)^3-(m-1)^3=6m^2+2$ 个，每节点 3 个自由度：

$$
n_i=3(m-1)^3,\qquad n_b=3(6m^2+2)=18m^2+6 .
$$

| 项目 | PIML linear_corner | PIML full_trace |
|---|---|---|
| 边界位移参数化 | $\Psi=L\in\mathbb R^{n_b\times n_q}$，角点位移的多重线性插值 | $\Psi=I\in\mathbb R^{n_b\times n_b}$，全部边界节点位移 |
| 子结构参数个数 $n_q$ | $24$ | $n_b$ |
| 网络输出 $\widehat N_{i,F}^j$ | $n_i\times18$ | $n_i\times(n_b-6)$ |
| 刚体约束补全后的 $\widehat N_i^j$ | $n_i\times24$ | $n_i\times n_b$ |
| 全局未知量 | 角点位移 $\widehat U_C$，$N_C$ 个 | 边界位移 $\widehat U_\Gamma$，$N_\Gamma$ 个 |

## 2. 实验设置

本文为分析文档，不设专门入口。PIML linear_corner 的在线运行用 [`run_piml_substructure.py`](run_piml_substructure.py)；Exact 两种表示的运行记录取自 [`run_substructure_full_trace_vs_linear_corner.py`](../topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py)；局部运算量的计数见 §3.2 的数据来源。

### 2.1 模型参数

![MBB 梁的几何、载荷与支承](assets/Huang2023_Fig5.png)

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

### 2.4 PIML 形函数预测

**离线阶段：样本生成**。

| 参数 | 数值或设置 |
|---|---|
| 子结构 | $m=5$，$h=1$，$\nu=0.3$，线性元：`--n-fine 5`，`--fine-cell-size 1.0`，`--nu 0.3`；`--n-sub 78 13 13` 只决定记入 `generation_config.json` 的整体域尺寸，不影响局部样本 |
| 边界位移表示 | linear_corner 取角点三线性插值，$\Psi=L$：`--trace-kind linear_corner`；full_trace 保留全部边界自由度，$\Psi=I$：`--trace-kind full_trace` |
| 采样量 | 各细单元的归一化杨氏模量 $\widehat E_e$，逐单元独立均匀采样：`--sampling independent_uniform` |
| 采样范围 | $\widehat E_e\in[10^{-7},1]$，上界固定为 $1$：`--min-modulus 1e-7` |
| 样本数 | 训练 $400\,000$，验证 $40\,000$：`--n-train 400000`，`--n-validation 40000` |
| 采样种子 | 驱动材料采样：`--sample-seed 2026` |
| 标签 | 形函数标签：`--targets shape` |
| 生成后端 | NumPy / CPU，float64：`--backend numpy`，`--device cpu`，`--generation-batch-size 32` |

**离线阶段：网络训练**。

| 参数 | 数值或设置 |
|---|---|
| 网络数 | $4$ 个网络共用同一输入，形函数的独立条目按下标连续均分给各网络，输出拼接后按刚体约束补全：`--num-networks 4` |
| 隐藏层宽度 | $[60,80,100,120,140,160,180,200,180,160,140,120,100,80,60]$：`--hidden-dims 60 80 ... 60` |
| 隐藏层激活 | $[\tanh,\mathrm{elu},\tanh,\mathrm{elu},\tanh,\mathrm{elu},\tanh,\mathrm{elu},\mathrm{elu},\tanh,\mathrm{elu},\tanh,\mathrm{elu},\tanh,\mathrm{elu}]$：`--activations tanh elu ... elu` |
| 损失 | 网络输出经刚体约束补全为 $\widehat N_i^j$ 后与精确 $N_i^j$ 的逐元素 MSE：`--loss decoded_full_matrix_mse` |
| 优化器 | Adam：`--optimizer adam` |
| 初始学习率 | $10^{-3}$：`--lr 1e-3` |
| batch size | $256$：`--batch-size 256` |
| 提前停止 | 验证损失 $40$ 轮未降：`--patience 40` |
| 最大轮数 | $500$：`--epochs 500` |
| 训练种子 | 驱动网络初始化：`--train-seed 2026` |
| 训练后端 | PyTorch / CPU，float64：`--backend pytorch`，`--device cpu` |

## 3. 结果对照

### 3.1 离线成本

```bash
# 样本生成
python experiments/topopt_piml_substructure/generate_samples.py \
  --n-sub 78 13 13 --n-fine 5 --fine-cell-size 1.0 --nu 0.3 --trace-kind linear_corner(full_trace) \
  --sampling independent_uniform --n-train 400000 --n-validation 40000 \
  --min-modulus 1e-7 --sample-seed 2026 --targets shape \
  --backend numpy --device cpu --generation-batch-size 32
```

每个样本装配同一个 $648\times648$ 的局部刚度，取 $K_{ii}^j$（$192\times192$）与 $K_{ib}^j$（$192\times456$），解 $N_i^j=-(K_{ii}^j)^{-1}K_{ib}^j\Psi$，只写 free 列 $N_{i,F}^j$。差别全部来自 $\Psi$：

| 项目 | linear_corner | full_trace |
|---|---|---|
| $\Psi$ 与求解右端数 | $L$，24 | $I$，456 |
| 每样本标签 $N_{i,F}^j$ | $192\times18$，展平 $3\,456$ 维 | $192\times450$，展平 $86\,400$ 维 |
| 40 万训练样本标签（float64） | 10.3 GiB（实测） | 257.5 GiB（解析） |
| 数据集目录 | `mbb_linear_corner_m5_h1_sub78x13x13_nu0p3_emin1e-7_shape_train400000_val40000_seed2026` | `mbb_full_trace_m5_h1_sub78x13x13_nu0p3_emin1e-7_shape_train400000_val40000_seed2026`(未生成) |

**网络训练**。

```bash
# 网络训练（linear_corner 侧）
python experiments/topopt_piml_substructure/train_network.py \
  --samples-dir experiments/topopt_piml_substructure/outputs/independent_15_layer/samples/mbb_linear_corner_m5_h1_sub78x13x13_nu0p3_emin1e-7_shape_train400000_val40000_seed2026 \
  --route shape --num-networks 4 \
  --hidden-dims 60 80 100 120 140 160 180 200 180 160 140 120 100 80 60 \
  --activations tanh elu tanh elu tanh elu tanh elu elu tanh elu tanh elu tanh elu \
  --epochs 500 --batch-size 256 --optimizer adam --lr 1e-3 --patience 40 \
  --train-seed 2026 --backend pytorch --device cpu
```

训练脚本不带迹参数，接口类型由样本目录的 `manifest.json` 决定。`full_trace` 侧只需把 `--samples-dir` 指向 full_trace 数据集，但 `--num-networks` 与 `--hidden-dims` 没有现成取法：4 网与论文隐藏结构是对每网 864 维输出给定的，21 600 维输出下须显式指定。差别全部来自输出维数：

| 项目 | linear_corner | full_trace |
|---|---|---|
| 每网输出维数（4 网均分） | $864$ | $21\,600$ |
| 网络结构 | 论文给定，2.4 节 | 待定，README 要求显式指定 |
| 训练结果 | 跑满 500 轮，最优轮 491，验证 MSE $6.39\times10^{-6}$ | 无 |
| 权重 | `outputs/independent_15_layer/training/shape/<数据集名>_trainseed2026/shape_best.pt` | 无 |

相同隐藏结构下，末层 $60\to21\,600$ 的权重与偏置就有 $1.32\times10^6$ 个，占每网参数量的 83%，一个 60 维的末隐藏层要线性生成 21 600 个输出，没有理由期望达到 `linear_corner` 侧的验证损失，而要加宽网络就进一步推高 3.2 节的推理成本。训练以 `mmap_mode='r'` 逐批读盘，每轮要扫一遍 257.5 GiB 的标签，500 轮的 I/O 本身已不可行。同 trace 的 stiffness 路线独立输出 $101\,475$ 个，情况相同。

数据来源：训练结果见 PIML 对照文档 §3.1。

### 3.2 局部运算量

按实际实现逐子结构计数：Exact 为 `ExactSchurCondensation.condense`（`full_trace`，一次批量 `solve` 得 $K_{ii}^{-1}K_{ib}$，再 $K_{bb}-K_{bi}K_{ii}^{-1}K_{ib}$）与 `streaming.py` 的角点路径（`linear_corner`，先投影 $K_{ib}L$、$L^{\mathsf T}K_{bb}L$，再对 24 个右端 `solve`）；PIML 为 `ShapeFunctionCondensation._variational_stiffness` 的四项式 $\Psi^{\mathsf T}K_{bb}\Psi+A+A^{\mathsf T}+\widehat N_i^{\mathsf T}K_{ii}\widehat N_i$，$A=(K_{ib}\Psi)^{\mathsf T}\widehat N_i$。计数规则：$(m\times k)(k\times n)$ 矩阵乘计 $2mkn$，$n$ 阶 $r$ 个右端的 solve 计 $\tfrac23n^3+2n^2r$，全连接层 $a\to b$ 计 $2ab$；网络取 2.4 节的 4 个 15 层网络（隐藏层宽 $60\sim200$，输入 125），`full_trace` 侧假定沿用同一隐藏层结构、只把输出均分给 4 个网络，是其推理成本的下界。子结构取 $m=5$，单位 MFlop。

| 步骤 | Exact linear_corner | PIML linear_corner | Exact full_trace | PIML full_trace |
|---|---|---|---|---|
| 边界块投影 $K_{ib}\Psi$、$\Psi^{\mathsf T}K_{bb}\Psi$ | 14.7 | 14.7 | 0（$\Psi=I$） | 0 |
| 内部求解 $K_{ii}^{-1}(\cdot)$ | 6.5 | 不做 | 38.3 | 不做 |
| 刚度合成 | 0.2 | 2.2 | 79.8 | 193.3 |
| 网络推理 | 不做 | 2.5 | 不做 | 12.5 |
| **合计** | **21.4** | **19.5** | **118.2** | **205.8** |
| PIML / Exact | | 0.91 | | 1.74 |

PIML 省掉的只有 solve 一项，代价是四项式中多出 $\widehat N_i^{\mathsf T}K_{ii}\widehat N_i$，其运算量 $2n_i^2n_q+2n_in_q^2$ 随 $n_q$ 增长。`linear_corner` 下 $n_q=24$，该项可忽略，局部运算量比 Exact 少约 9%；`full_trace` 下 $n_q=n_b=456$，该项达 113 MFlop，局部运算量反而是 Exact 的 1.74 倍，网络推理还未计入末层之外可能需要的更宽结构。由第 1 节的 $n_i=3(m-1)^3$、$n_b=18m^2+6$，Exact 的 LU 项 $\tfrac23n_i^3\sim m^9$ 在 $m$ 很大时才会超过 PIML 的 $2n_i^2n_b\sim m^8$，但那时网络输出 $n_in_b\sim m^5$ 已不可训练（$m=7$ 时 $n_i=648$，$n_b=888$，输出约 $5.8\times10^5$），见 3.1 节。

`linear_corner` 下 9% 的局部节省也说明，PIML 在该表示下的收益并不来自浮点运算量，而至多来自把批量 LU 换成 GPU 更友好的矩阵乘；这一点属于 PIML 对照文档 §4 的实测范围，本文不展开。

数据来源：维数 $n_i$、$n_b$ 与独立输出数由 `SubstructurePrototype` 与 `IndependentPredictionDecoder` 读出；计数脚本在 scratchpad 中临时运行，未入库。

### 3.3 全局成本与加速比上界

PIML 只改变局部缩聚，全局系统的规模、非零元、求解与恢复与 Exact 同一表示完全相同。Exact 对照文档 §3.2 给出两种表示的全局矩阵规模：240 档 $K_\Gamma$ 维数 $625\,539$、非零元 $5.43\times10^8$，$K_C$ 维数 $11\,907$、非零元 $8.16\times10^5$，相差 52 倍与 666 倍；GPU 上 `full_trace` 的刚度算子常驻 6.62 GiB，390 档装配阶段超过 14 GiB 显存上限失败，而 `linear_corner` 的 390 档全程峰值 5.68 GiB。这些都与网络无关。

**单轮耗时与 Amdahl 上界**。取 Exact 对照文档记录的 240 档运行（单线程），`full_trace` 的「装配」阶段未再细分，含局部组装、局部缩聚与全局 CSR 组装。把整个装配阶段都算作 PIML 可以省掉的部分，得到局部缩聚成本归零时单轮加速比的上界：

| Exact full_trace，240 档 | 装配 / s | 求解 / s（CG 步数） | 恢复 / s | 整轮 / s | 加速比上界 |
|---|---|---|---|---|---|
| GPU，第 1 轮 | 14.1 | 20.9（898） | 3.2 | 38.7 | $\le1.57$ |
| GPU，第 2 轮 | 4.1 | 34.5（1554） | 2.2 | 41.3 | $\le1.11$ |
| CPU，第 1 轮 | 43.0 | 259.5（898） | 27.2 | 336.7 | $\le1.15$ |

第 1 轮 GPU 装配含一次性初始化，第 2 轮起为稳态。同网格 Exact linear_corner 的第 2 轮整轮耗时 GPU 1.1 s、CPU 13.8 s，其中 CPU 的装配 12.9 s 由局部刚度生成 6.4 s 与边界块投影 5.5 s 构成，24 个右端的 solve 只占 0.9 s。即无论哪种表示，内部求解都不是主项；`full_trace` 的主项是对 $K_\Gamma$ 的 CG 求解（第 2 轮起占整轮 84%），PIML 不触及。

**恢复矩阵**。`linear_corner` 下 $N_i^j$ 每子结构 $192\times24$（0.035 MiB），装配时缓存，恢复只做回代；`full_trace` 下 $192\times456$（0.668 MiB），390 档 13 182 个子结构合计 8.60 GiB，Exact 对照文档因此不缓存而逐批重算。PIML full_trace 面临同样的选择：缓存 $\widehat N_i^j$ 需 8.60 GiB，或每轮恢复时重新推理一遍。

数据来源：全局矩阵规模与显存见 Exact 对照文档 §3.2；单轮各阶段耗时取 `outputs/pytorch_{cuda,cpu}_full_trace_hex_240x40x40_m5_cg_jacobi_t1/history.json` 与同网格 `linear_corner` 目录的记录。

## 4. 结论与适用边界

PIML shape 路线只在 `linear_corner` 下有意义：$n_q=24$ 使预测对象小、四项式便宜、全局系统也小，网络替换的局部缩聚在整轮成本中占可观比例，且 Ritz 误差是两侧共有的。改到 `full_trace` 后：

1. 精度只能变差，把与 FEM 等价的 Exact full_trace 换成一个只会偏硬的近似，误差全部由代理引入（1.1 节）；
2. 离线需要 25 倍的标签，且无文献结构可循（3.1 节）；
3. 局部运算量不省反增，为 Exact 的 1.74 倍（3.2 节）；
4. 整轮成本的主项是 $K_\Gamma$ 的求解与常驻内存，与网络无关，局部缩聚归零也只能把单轮提速至多 1.11 倍（3.3 节）。

因此本目录不实现、不训练、不运行 PIML full_trace；在线代码 `piml_corner.py` 只接受 `linear_corner` 保持不变，`generate_samples.py` 与独立条目编解码器对 `full_trace` 的支持仅作为标签工具保留。若要在 `linear_corner` 之外寻找 PIML 的位置，方向是介于两者之间的低维边界参数化（如角点加棱中点、边界低阶模态），使 $n_q$ 仍远小于 $n_b$，本文不讨论。
