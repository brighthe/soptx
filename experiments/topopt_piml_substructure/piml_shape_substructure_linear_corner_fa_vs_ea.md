# PIML shape 子结构 linear_corner FA 与 EA 的拓扑优化结果对照

硬件：Intel Core i9-14900KF（24 核 32 线程，8P+16E），运行内存 45 GiB；NVIDIA GeForce RTX 5080，显存 16 GB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（内置 scipy-openblas 0.3.34），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.13.0（CUDA 13.0）；PyMUMPS 0.4.0，链接系统 MUMPS 5.6.2（OpenBLAS 0.3.26、Open MPI 4.1.6、ScaLAPACK 2.2.1，单进程运行）；pyevtk 1.7.0。

## 1. 研究对象

对照形函数预测 PIML（shape 路线）在 `linear_corner` 下采用不同局部矩阵组装方式 `FA` 与 `EA` 时，在 Huang2023 三维 MBB 梁上的完整优化结果与计算成本。精确子结构侧的同类对照见 [`substructure_linear_corner_fa_vs_ea.md`](../topopt_simp_substructure/substructure_linear_corner_fa_vs_ea.md)（下称“Exact FA/EA 文档”）。

| 项目 | FA（现有实现） | EA（待实现） |
|---|---|---|
| 局部刚度 | 先组装完整的 $K^j\in\mathbb R^{648\times648}$，再按自由度切片 | 不形成 $K^j$ |
| $K_{cc}^j=L^{\mathsf T}K_{bb}^jL$、$B^j=K_{ib}^jL$ | 切片后与稠密 $L$ 相乘 | $S\,\mathcal R^{C}$、$S\,\mathcal R^{B}$，参考张量与 Exact EA 共用 |
| 第四项 $(\widehat N_i^j)^{\mathsf T}K_{ii}^j\widehat N_i^j$ | 由切片得到的 $K_{ii}^j$ 计算 | 待定，见 §1.3 |
| 精确回退 | 完整边界 Schur 补后投影到 $L$ | 直接角点缩聚，复用 Exact EA 的三个块 |
| 全局 $\widehat K_C$ | CSR | CSR |

记号沿用 Exact FA/EA 文档 §1 与 [`piml_shape_substructure_linear_corner.md`](piml_shape_substructure_linear_corner.md)（下称“PIML 对照文档”）§1.1。本算例 $m=5$、Q1 元：每个子结构含 125 个细单元，$n_d=648$、$n_b=456$、$n_i=192$，角点自由度 $n_q=24$；$L\in\mathbb R^{456\times24}$ 为局部角点到边界节点的三线性插值矩阵，$\widehat N^j=[L;\widehat N_i^j]$。

### 1.1 共同部分

以下环节与组装方式无关，两侧完全一致：

- **网络推理**：输入为各细单元的单元系数 $c_e^j=c(\tilde\rho_e)$，输出 3456 个独立条目，经刚体约束补全为 $\widehat N_i^j\in\mathbb R^{192\times24}$。网络输入不依赖局部刚度，因此给定相同的密度、权重、推理设备与批量划分，两侧的 $\widehat N_i^j$ 逐位相同。
- **门禁**：刚体零空间残量（作用于 $\widehat K_C^j$）、刚化上界（作用于 $K_{cc}^j$ 与 $\widehat K_C^j-K_{cc}^j$ 的特征值）、变形子空间条件数（作用于 $\widehat K_C^j$ 在变形子空间上的投影）；阈值取 `ShapeFunctionCondensation` 的默认值，两侧相同。
- **均匀子结构**：满足本目录 README 的判据时取 $\widehat K_C^j=\frac{E(\rho_m^j)}{E_0}K_C^{\mathrm{solid}}$，不需要局部刚度。现有 `PIMLShapeReduction.reduce_many` 中未见该判别，实现状态待核实。
- **全局与后处理**：全局角点 CSR 装配、载荷投影、约束处理、线性求解、位移恢复 $\widehat u_i^j=\widehat N_i^jq^j$，以及冻结形函数的单元能量灵敏度。

### 1.2 FA：组装局部刚度后切片

局部刚度由单元刚度散加得到：

$$
K^j=\sum_{e\in\Omega_j}(G_e^j)^{\mathsf T}\bigl[c_e^jK^0\bigr]G_e^j .
$$

`reduce_many` 接收 $(b,648,648)$ 的 $K^j$：先由 `_trace_blocks` 切出 $K_{ii}^j$，并计算 $K_{ib}^jL$ 与 $L^{\mathsf T}K_{bb}^jL$；再由 `_variational_stiffness` 展开四项式

$$
\widehat K_C^j=K_{cc}^j+(B^j)^{\mathsf T}\widehat N_i^j+(\widehat N_i^j)^{\mathsf T}B^j+(\widehat N_i^j)^{\mathsf T}K_{ii}^j\widehat N_i^j .
$$

未通过门禁或预测非有限的子结构，由 `fallback_solver.condense` 对其 $K^j$ 做完整边界 Schur 补（$456$ 个右端），再经 `_reduce_exact` 投影为 $L^{\mathsf T}K_s^jL$ 与 $N_i^jL$。

### 1.3 EA：由参考张量形成四项式所需的块

前三项所需的 $K_{cc}^j$ 与 $B^j$ 恰为 Exact EA 的两个块：

$$
B=S\,\mathcal R^{B},\qquad K_{cc}=S\,\mathcal R^{C},
$$

其中 $S\in\mathbb R^{b\times125}$ 为一批子结构的单元系数，$\mathcal R^{B}\in\mathbb R^{125\times4608}$、$\mathcal R^{C}\in\mathbb R^{125\times576}$ 的定义见 Exact FA/EA 文档 §1.3，直接复用，不另行实现。

第四项依赖逐子结构预测的 $\widehat N_i^j$，不能预计算，候选形成方式有三种：

- **方案 a**：只对单元刚度的内部–内部分量做 `bincount` 散加，得到 $(b,192,192)$ 的 $K_{ii}^j$，再计算 $(\widehat N_i^j)^{\mathsf T}K_{ii}^j\widehat N_i^j$；与 Exact EA 的方案 a 共用 $K_{ii}^j$ 的形成代码。
- **方案 b**：$K_{ii}=S\,\mathcal A$，$\mathcal A\in\mathbb R^{125\times36864}$（约 37 MB），再同方案 a 计算第四项。
- **方案 c**：逐单元计算。取 $Z_e^j=G_{e,i}\widehat N_i^j\in\mathbb R^{24\times24}$（按局部编号取 $\widehat N_i^j$ 的行，边界自由度对应行置零），则

$$
(\widehat N_i^j)^{\mathsf T}K_{ii}^j\widehat N_i^j=\sum_e c_e^j\,(Z_e^j)^{\mathsf T}K^0Z_e^j ,
$$

  完全不形成 $K_{ii}^j$。若取 $W_e^j=G_e\widehat N^j$ 代替 $Z_e^j$，同一求和直接给出完整的 $\widehat K_C^j$，但刚化上界门禁仍需 $K_{cc}^j$，故 $S\,\mathcal R^{C}$ 不能省。

三种方案先在小算例上比较耗时与内存，选定后再进入正式对照。

精确回退在 EA 下改用直接角点缩聚：对回退子集形成 $K_{ii}^j$（方案 a 或 b），求解 $K_{ii}^jX^j=B^j$，取 $K_C^j=K_{cc}^j-(B^j)^{\mathsf T}X^j$、$N_i^j=-X^j$，与 Exact EA 完全相同。它与 FA 的完整边界 Schur 补在精确算术下等价，但右端数由 456 降为 24。

### 1.4 等价性

给定相同的 $\widehat N_i^j$，精确算术下 FA 与 EA 得到的 $K_{cc}^j$、$B^j$、第四项及回退结果都相同，因此 $\widehat K_C$、全场位移、柔顺度、灵敏度与优化轨迹都相同，只有运算顺序不同带来的舍入差异。与 Exact FA/EA 相比，有三处可能把舍入差异放大为离散差异，须在验收中单独检查：

1. **门禁判定翻转**：比值落在阈值附近时，舍入差异可能改变通过与回退的子结构集合，进而改变 $\widehat K_C$ 的来源。
2. **回退路径不同**：FA 用完整边界 Schur 补，EA 用直接角点缩聚，舍入差异大于其余环节。
3. **推理批量**：GEMM 内核可能随批量大小变化，两侧须保持相同的 `chunk_size` 与推理设备，才能保证 $\widehat N_i^j$ 逐位相同。

本文**不涉及** PIML 相对 Exact 的代理误差（见 PIML 对照文档 §1.3 与 §3），也不做 $C_{\mathrm{Fine}}$ 重分析。

### 1.5 预期收益（估算，未实测）

每个子结构的主要乘加次数（MAC）与每批（`chunk_size=256`）的局部临时数组：

| 项目 | FA | EA 方案 a | EA 方案 c |
|---|---|---|---|
| 局部刚度临时数组 | $K^j$：约 0.80 GiB | $K_{ii}^j$：约 0.07 GiB | $Z_e^j$：约 0.14 GiB，可再分块 |
| 形成 $K_{cc}^j$、$B^j$ | 约 $5.3\times10^6$ | 约 $0.65\times10^6$ | 同方案 a |
| 第四项及交叉项 | 约 $1.1\times10^6$ | 约 $1.1\times10^6$ | 约 $3.5\times10^6$ |
| 散加 | $125\times576$ 项散入 $648^2$ | $125\times144$ 项散入 $192^2$ | 无 |

估算口径：FA 的 $K_{ib}^jL$ 约 $192\times456\times24$，$L^{\mathsf T}K_{bb}^jL$ 约 $24\times456\times456+24\times24\times456$；EA 的 GEMM 为 $125\times4608+125\times576$；第四项 $192^2\times24+192\times24^2$；方案 c 为 $125\times2\times24^3$。

与 Exact 不同，PIML 没有局部求解，这部分时间换成了网络推理与门禁（含三次 $24\times24$ 量级的特征值分解）。Exact FA/EA 文档 §1.5 引用的分项占比（提交 `2226a7b`）不适用于本文；EA 的实际收益取决于推理与门禁在装配中的占比，须先完成 FA 的分项计时（§4.1）再下结论。

## 2. 实验设置

模型、材料、离散、线性求解、优化参数及 PIML 模型同 PIML 对照文档 §2，以入口脚本顶部配置为准。正式对照沿用该文档的 $210\times35\times35$（子结构划分 $42\times7\times7$，$m=5$）；$390\times65\times65$ 档待 $210$ 档完成后再定。下表只列本文特有的设置。

| 参数 | 数值或设置 |
|---|---|
| 子结构边界位移 | `trace='linear_corner'`，两侧相同 |
| 网络权重 | 两侧同一权重文件，待填 |
| 推理设备与批量 | 两侧相同，待填 |
| 门禁阈值 | `rigid_tol=1e-10`、`excess_rtol=2e-2`、`rcond_min=1e-8`（现有默认值），两侧相同 |
| 均匀子结构判别 | 两侧相同，待实现状态核实后填写 |
| 局部刚度层级 | FA：现有 `PIMLShapeReduction.reduce_many`；EA：待实现，分析器选项名待定 |
| 第四项形成方式 | 待 §1.3 小算例比较后确定 |
| 精确回退 | FA：完整边界 Schur 补；EA：直接角点缩聚 |
| 线程 | 不强制线程数，外部环境变量原样记录到 `config.json` |
| 实验入口 | 待定；目前尚无 PIML 完整拓扑优化入口 |

## 3. 等价性验收

参照 [`verify_corner_direct.py`](../topopt_simp_substructure/verify_corner_direct.py)：对同一物理密度分别用 FA 和 EA 装配并求解，要求 $\widehat N_i^j$ 逐位相同、门禁判定集合相同，其余四项的相对误差均小于 $10^{-8}$；两侧求解残差也均小于 $10^{-8}$。为分别检验预测分支与回退分支，另加“禁用回退”与“强制全部回退”两组。

| 密度 | 回退设置 | $\widehat N_i^j$ | 门禁集合 | $\widehat K_C$ | 全场位移 | 柔顺度 | 灵敏度 |
|---|---|---|---|---|---|---|---|
| 均匀密度，$60\times10\times10$ | 正常 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 随机密度，$60\times10\times10$ | 正常 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 随机密度，$60\times10\times10$ | 禁用回退 | 待填 | 不适用 | 待填 | 待填 | 待填 | 待填 |
| 随机密度，$60\times10\times10$ | 强制全部回退 | 不适用 | 不适用 | 待填 | 待填 | 待填 | 待填 |
| 优化中间密度，$210\times35\times35$，来源待定 | 正常 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

门禁集合不同时，记录翻转的子结构数及其门禁比值与阈值的距离，不计入相对误差。

## 4. 计算成本

### 4.1 装配分项耗时（s，第 2 轮起取均值）

现有 PIML 路径是否提供 `assembly_breakdown` 的相应字段尚待核实；若实现融合了若干步骤，报告可测的合并阶段，不凭空拆分。

| 分项 | FA | EA | 说明 |
|---|---|---|---|
| local_stiffness | 待填 | 待填 | FA 为 $K^j$；EA 为 $K_{ii}^j$（方案 a/b），方案 c 为空 |
| block_projection | 待填 | 待填 | FA 为切片与 $L$ 投影；EA 为参考张量 GEMM |
| inference | 待填 | 待填 | 输入转换、推理与刚体补全，两侧应相近 |
| variational | 待填 | 待填 | 四项式；方案 c 含逐单元第四项 |
| gates | 待填 | 待填 | 两侧应相近 |
| fallback | 待填 | 待填 | 另记回退子结构数 |
| global_csr | 待填 | 待填 | 两侧应相近 |
| 装配合计 | 待填 | 待填 | |
| 首轮装配 | 待填 | 待填 | EA 的首轮含参考张量预计算 |

### 4.2 每轮与全程

| 项目 | FA | EA | 说明 |
|---|---|---|---|
| 每轮合计 / s | 待填 | 待填 | |
| 优化轮数 | 待填 | 待填 | 两侧应相同，或仅因舍入相差少量轮次 |
| 累计回退子结构数 | 待填 | 待填 | 两侧应相同或相近 |
| 总耗时 / min | 待填 | 待填 | |

### 4.3 内存（GiB）

| 项目 | FA | EA | 说明 |
|---|---|---|---|
| 装配阶段峰值 RSS | 待填 | 待填 | 预期差别主要在这里 |
| 全程峰值 RSS | 待填 | 待填 | `summary.process_peak_rss_gib` |
| 参考张量 | 不适用 | 待填 | $\mathcal R^{B}$、$\mathcal R^{C}$，以及方案 b 的 $\mathcal A$ |
| 网络权重与推理缓冲 | 待填 | 待填 | 两侧相同；GPU 推理时显存单列 |
| 形函数内部块缓存 | 待填 | 待填 | 两侧相同 |

## 5. 验证状态与后续步骤

本轮仅新建文档，未实现、未运行。后续按以下顺序推进，运行前另行明确命令、验收阈值并取得授权：

1. 完成 Exact EA 的实现与等价性验收（Exact FA/EA 文档 §3），得到可复用的 $\mathcal R^{B}$、$\mathcal R^{C}$ 与 $K_{ii}^j$ 形成代码。
2. 接通 PIML 完整拓扑优化入口（PIML 对照文档 §5），并核实均匀子结构判别的实现状态。
3. 在 $210\times35\times35$ 上做一次 PIML FA 的装配分项计时；若局部刚度与块投影占比不显著，在此终止并在本文记录结论。
4. 在小算例上比较 §1.3 的三种第四项方案，选定后实现 EA。
5. 完成 §3 的等价性验收后再执行正式对照，填写 §4。

## 6. 不在本文范围内

- PIML 相对 Exact 的代理误差与 $C_{\mathrm{Fine}}$ 重分析：见 PIML 对照文档。
- `stiffness` 路线：直接预测 $\widehat K_C^j$，不形成四项式；现有 `ReducedStiffnessCondensation` 中局部刚度用于精确导出恢复矩阵与回退，不进入 $\widehat K_C^j$。
- 全局 $\widehat K_C$ 的 FA 与 EA：理由同 Exact FA/EA 文档 §5。
- 形函数内部块保存与重算：留待 GPU 实验。

## 参考资料

- [Exact FA/EA 文档](../topopt_simp_substructure/substructure_linear_corner_fa_vs_ea.md)：参考张量 $\mathcal R^{B}$、$\mathcal R^{C}$ 与 $K_{ii}^j$ 的方案 a/b。
- [PIML 对照文档](piml_shape_substructure_linear_corner.md)：四项式、代理误差关系与公共实验设置。
- [本目录 README](README.md)：网络输入、均匀子结构判别与训练配置。
