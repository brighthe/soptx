# 子结构 linear_corner FA 与 EA 的拓扑优化对照

硬件：Intel Core i9-14900KF（24 核 32 线程，8P+16E），运行内存 45 GiB；NVIDIA GeForce RTX 5080，显存 16 GB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（内置 scipy-openblas 0.3.34），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.13.0（CUDA 13.0）；PyMUMPS 0.4.0，链接系统 MUMPS 5.6.2（OpenBLAS 0.3.26、Open MPI 4.1.6、ScaLAPACK 2.2.1，单进程运行）；pyevtk 1.7.0。

## 1. 研究对象

对照子结构 `linear_corner` 中子结构缩聚的两种处理方式 FA 与 EA 在 Huang2023 三维 MBB 梁完整拓扑优化中的计算成本，并验证各配置结果一致；另考察位移恢复中形函数内部块的缓存与重算。

| 变量 | 显式做法 | matrix-free 做法 |
|---|---|---|
| 子结构缩聚 | FA：组装完整的子结构刚度 $K^j$，抽取各块后与稠密 $L$ 相乘 | EA：不形成 $K^j$；$K_{ib}^jL$、$K_{bb}^jL$ 由单元刚度逐单元作用在 $L$ 的 24 列上得到，只组装 $K_{ii}^j$ |
| 位移恢复 | 缓存形函数内部块 $N_i^j$，回代时直接相乘 | 重算：不保存 $N_i^j$，回代时重新组装并分解 $K_{ii}^j$，每个子结构只求解 1 个右端 |

组合成以下配置：

| 配置 | 缩聚 | 位移恢复 | 用途 |
|---|---|---|---|
| FA-缓存 | FA | 缓存 | 基线，即现有实现（提交 `8469602`） |
| EA-缓存 | EA | 缓存 | 单独考察 EA 缩聚 |
| EA-重算 | EA | 重算 | 两处都用 matrix-free，内存最省 |
| FA-重算 | FA | 重算 | 只做小算例验收，单独考察不改缩聚时重算的代价 |

记号沿用 [`substructure_full_trace_vs_linear_corner.md`](substructure_full_trace_vs_linear_corner.md) §1。本算例 $m=5$、Q1 元：每个子结构含 125 个细单元，$n_d=648$、$n_b=456$、$n_i=192$，角点自由度 $n_q=24$；$L\in\mathbb R^{456\times24}$ 为局部角点到边界节点的三线性插值矩阵。

### 1.1 角点缩聚、位移恢复与求解

各配置都采用直接角点缩聚：

$$
B^j=K_{ib}^jL,\qquad K_{ii}^jX^j=B^j,\qquad
K_C^j=L^{\mathsf T}K_{bb}^jL-(B^j)^{\mathsf T}X^j .
$$

$K_{ii}^j$ 需要做 Cholesky 分解，因此各配置都显式组装 $K_{ii}^j$。全局装配（$K_C$ 均散加为 CSR）、约束处理、CG + Jacobi 求解与灵敏度计算在各配置中完全一致。

### 1.2 子结构缩聚：FA

子结构刚度由单元刚度散加得到：

$$
K^j=\sum_{e\in\Omega_j}(G_e^j)^{\mathsf T}\bigl[c(\tilde\rho_e)K^0\bigr]G_e^j .
$$

实现上先用 `einsum` 得到 $(b,125,24,24)$ 的单元刚度，再用 `bincount` 散加成 $(b,648,648)$，然后按 $i$、$b$ 抽取各块，最后与稠密 $L$ 相乘（`SubstructurePrototype.iter_local_stiffness_batches` 与 `assemble_corner_system_with_recovery`）。

### 1.3 子结构缩聚：EA

结构化布局下各子结构共用局部编号，$G_e^j$ 与 $j$ 无关，下文记为 $G_e$；按边界列与内部列拆成 $G_{e,b}\in\mathbb R^{24\times456}$、$G_{e,i}\in\mathbb R^{24\times192}$，并记单元系数 $c_e^j=c(\tilde\rho_e)$。

$K_{ib}^jL$ 与 $K_{bb}^jL$ 是子结构刚度对 $L$ 的 24 列的作用，按 EA 逐单元完成：gather、单元刚度相乘、scatter-add。

$$
L_e=G_{e,b}L\in\mathbb R^{24\times24},\qquad
Y_e^j=c_e^jK^0L_e,
$$

$$
B^j=\sum_e G_{e,i}^{\mathsf T}Y_e^j,\qquad
L^{\mathsf T}K_{bb}^jL=\sum_e L_e^{\mathsf T}Y_e^j .
$$

$L_e$ 是 $L$ 中对应单元 $e$ 边界自由度的行（单元 $e$ 的内部自由度对应零行），与密度无关，构造时取出一次即可。$K_{ii}^j$ 只对单元刚度的内部–内部分量散加：

$$
K_{ii}^j=\sum_e G_{e,i}^{\mathsf T}\bigl[c_e^jK^0\bigr]G_{e,i}.
$$

与 FA 相比，EA 不形成 $648\times648$ 的 $K^j$，也不再抽取各块、不再与 $456\times24$ 的 $L$ 做稠密矩阵乘。

### 1.4 位移恢复：缓存与重算

求得角点位移后，第 $j$ 个子结构的角点未知量为 $q^j$，内部位移为

$$
u_i^j=N_i^jq^j=-(K_{ii}^j)^{-1}K_{ib}^jLq^j .
$$

**缓存**：装配时保存 $N_i^j=-X^j\in\mathbb R^{192\times24}$，回代只需一次 $192\times24$ 的矩阵乘向量。390 档缓存约 0.453 GiB，210 档约 0.071 GiB。

**重算**：装配时用完 $X^j$ 即丢弃。回代时逐批重新组装并分解 $K_{ii}^j$，再求解

$$
K_{ii}^ju_i^j=-r^j,\qquad
r^j=K_{ib}^j(Lq^j)=\sum_e G_{e,i}^{\mathsf T}\bigl[c_e^jK^0\bigl(L_eq^j\bigr)\bigr].
$$

$r^j$ 由 EA 对单个向量作用得到；FA-重算则先组装 $K^j$，再抽取 $K_{ib}^j$ 计算。每个子结构只求解 1 个右端，而装配阶段要求解 24 个，因此重算的主要代价是 $K_{ii}^j$ 的组装与分解。

重算即 Ma2026 的做法：其并行实现不保存多尺度形函数，回代时由 PIML 模型重新预测，并称之为 matrix-free 实现。两者的区别在于，Ma2026 重新得到整个形函数；而精确缩聚下回代只需要 $N_i^jq^j$，不必重算整个 $N_i^j$。

### 1.5 等价性

精确算术下，四种配置得到的 $K_C$、全场位移、柔顺度、灵敏度与优化轨迹都相同，只有运算顺序不同带来的舍入差异。本文**不涉及**角点插值相对于 full_trace 的近似误差，不做 Ritz 分析，也不做 $C_{\mathrm{Fine}}$ 重分析。

### 1.6 预期收益（估算，未实测）

| 项目 | FA | EA |
|---|---|---|
| 每批（`chunk_size=256`）临时数组 | $K^j$ 约 0.80 GiB，加上单元刚度 $(b,125,24,24)$ 约 0.14 GiB | $Y_e^j$ 约 0.14 GiB，$K_{ii}^j$ 约 0.07 GiB，$B^j$ 约 0.01 GiB |
| 每个子结构的主要运算 | 散加 $648^2$ 个元素，抽取后做 $192\times456\times24$ 与 $24\times456\times456$ 的矩阵乘 | 125 次 $24\times24\times24$ 的单元矩阵乘，加上 $K_{ii}^j$ 的散加 |

| 项目 | 缓存 | 重算 |
|---|---|---|
| 常驻内存 | 390 档 $N_i^j$ 约 0.453 GiB | 无 |
| 每轮额外耗时 | 无 | 重新组装并分解 $K_{ii}^j$，求解 1 个右端 |

依据：提交 `2226a7b` 在 $210\times35\times35$ 档的分项计时中，装配时间约 20–22% 用于子结构刚度生成，53–55% 用于块抽取与投影，约 25% 用于局部求解（含 $K_{ii}^j$ 分解与 24 个右端），CSR 散加约 0.1%。EA 针对前两项；重算的代价与局部求解同量级。

## 2. 实验设置

模型、材料、离散、线性求解与优化参数同 [`substructure_full_trace_vs_linear_corner.md`](substructure_full_trace_vs_linear_corner.md) §2，以入口脚本顶部配置为准。正式对照用 $390\times65\times65$（子结构划分 $78\times13\times13$，$m=5$），CG + Jacobi，运行 FA-缓存、EA-缓存、EA-重算三种配置。下表只列本文特有的设置。

| 参数 | 数值或设置 |
|---|---|
| 子结构边界位移 | `trace='linear_corner'`，各配置相同 |
| 缩聚方式 | FA：现有 `assemble_corner_system_with_recovery`；EA：待实现，分析器选项名待定 |
| 位移恢复方式 | 缓存：现有实现；重算：待实现，分析器选项名待定 |
| 求解阶段计时 | 待实现：分别记录 CG 与位移恢复，重算配置另记 $K_{ii}^j$ 的组装与分解 |
| 线程 | 不强制线程数，外部环境变量原样记录到 `config.json` |
| 实验入口 | 待定：扩展 `run_substructure_full_trace_vs_linear_corner.py`，或另建入口 |

## 3. 等价性验收

参照 [`verify_corner_direct.py`](verify_corner_direct.py)：对同一物理密度分别用各配置装配并求解，以 FA-缓存为基准，比较以下四项的相对误差，要求均小于 $10^{-8}$；各配置的求解残差也均小于 $10^{-8}$。

| 密度 | 配置 | $K_C$ | 全场位移 | 柔顺度 | 灵敏度 |
|---|---|---|---|---|---|
| 均匀密度，$60\times10\times10$ | EA-缓存 / EA-重算 / FA-重算 | 待填 | 待填 | 待填 | 待填 |
| 随机密度，$60\times10\times10$ | EA-缓存 / EA-重算 / FA-重算 | 待填 | 待填 | 待填 | 待填 |
| 第 94 轮真实密度，$210\times35\times35$ | EA-缓存 / EA-重算 / FA-重算 | 待填 | 待填 | 待填 | 待填 |

## 4. 计算成本

正式数据取自 390 档完整优化。FA-重算只在 210 档第 94 轮密度上做单次计时，不跑完整优化。

### 4.1 装配分项耗时（s，第 2 轮起取均值）

分项沿用 `assembly_breakdown` 的口径。EA 中 `local_stiffness` 指 $K_{ii}^j$ 的形成，`block_projection` 指以 EA 作用形成 $B^j$ 与 $L^{\mathsf T}K_{bb}^jL$。

| 分项 | FA-缓存 | EA-缓存 | EA-重算 | 说明 |
|---|---|---|---|---|
| local_stiffness | 待填 | 待填 | 待填 | 子结构刚度或 $K_{ii}^j$ 的形成 |
| block_projection | 待填 | 待填 | 待填 | 块抽取与投影，或 EA 作用 |
| local_solve | 待填 | 待填 | 待填 | 求解 $K_{ii}^jX^j=B^j$，各配置应相近 |
| global_csr | 待填 | 待填 | 待填 | 各配置应相近 |
| 装配合计 | 待填 | 待填 | 待填 | |

### 4.2 求解阶段分项耗时（s，第 2 轮起取均值）

| 分项 | FA-缓存 | EA-缓存 | EA-重算 | 说明 |
|---|---|---|---|---|
| CG | 待填 | 待填 | 待填 | 各配置应相近 |
| 位移恢复 | 待填 | 待填 | 待填 | 重算含 $K_{ii}^j$ 的组装、分解与单右端求解 |
| 求解合计 | 待填 | 待填 | 待填 | |

### 4.3 每轮与全程

| 项目 | FA-缓存 | EA-缓存 | EA-重算 | 说明 |
|---|---|---|---|---|
| 每轮合计 / s | 待填 | 待填 | 待填 | |
| 优化轮数 | 待填 | 待填 | 待填 | 应相同，或仅因舍入相差少量轮次 |
| 总耗时 / min | 待填 | 待填 | 待填 | |

### 4.4 内存（GiB）

| 项目 | FA-缓存 | EA-缓存 | EA-重算 | 说明 |
|---|---|---|---|---|
| 装配阶段峰值 RSS | 待填 | 待填 | 待填 | EA 降低每批临时数组 |
| 求解阶段峰值 RSS | 待填 | 待填 | 待填 | 重算不常驻 $N_i^j$ |
| 全程峰值 RSS | 待填 | 待填 | 待填 | `summary.process_peak_rss_gib` |
| $N_i^j$ 缓存 | 待填 | 待填 | 不适用 | 390 档约 0.453 GiB |

## 5. 不在本文范围内

- 全局 $K_C$ 的 EA：$210\times35\times35$ 档前两轮实测中，$K_C$ 的 CSR 散加与 CG 全部迭代合计约占每轮 1%，且不组装 $K_C$ 会失去直接法与代数预条件。
- 以参考张量形成 $B^j$ 与 $K_{cc}^j$：把 $L$ 预先乘进 $(G_e\hat N)^{\mathsf T}K^0(G_e\hat N)$ 可能更快，但属于降维空间装配，不是 EA，留作后续优化。
- 内部方程改用迭代法：$m=5$ 时 $K_{ii}^j$ 只有 $192\times192$，Cholesky 更快；$m=10$ 时再考虑。
- full_trace 的 matrix-free Schur 算子。
