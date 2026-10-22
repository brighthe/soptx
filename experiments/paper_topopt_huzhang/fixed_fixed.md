# 两端固支梁柔顺度最小化（论文 5.2.1 节）

## 1. 研究对象

在体积约束下最小化柔顺度，考察 Lagrange 位移元（LFEM）$p = 2, 3, 4$ 与 Hu--Zhang 混合元（HZMFEM）$k = 2, 3, 4$ 共六组离散下的优化过程与最终构型，并把六个冻结设计在六种离散下交叉重分析（论文表 5.3），区分"设计不同"与"离散与泛函不同"两种差异来源。六组共用同一设计域、材料插值与优化参数，只有分析链与阶次不同。

| | LFEM | HZMFEM |
| --- | --- | --- |
| 阶次 $k$ 的含义 | 位移插值阶 $p = k$ | 应力插值阶 $k$，位移阶 $k - 1$ |
| 柔顺度泛函 | 外载功 $f^\mathsf{T} u$：`ComplianceObjective(state_variable="u")` | 互补能 $\sigma^\mathsf{T} A(\rho)\, \sigma$：`ComplianceObjective(state_variable="sigma")` |
| $k = 2$ | 原生格式 | 跳量稳定化格式（$k \le d$），`stabilization_coefficient = "fixed"` |
| $k = 3, 4$ | 原生格式 | 原生格式，无需稳定化 |

两类泛函在离散层面不相等（$k \le 2$ 的跳量稳定化格式中，互补能与牵引对偶功一般也不相等），故六个优化所得柔顺度不能直接横比，交叉重分析在同一列内用同一泛函比较不同设计。

模型为 [`HalfFixedFixedBeamPatch2d`](../../docs/problems/benchmarks/half-fixed-fixed-beam-patch-2d.md)：完整梁 $160\ \mathrm{mm} \times 20\ \mathrm{mm}$ 两端固支、底边中点受竖直局部牵引，几何与载荷关于竖直中线对称，只算左半域 $80\ \mathrm{mm} \times 20\ \mathrm{mm}$；左端 $x = 0$ 完全固支，对称面 $x = 80$ 上法向位移 $u_x = 0$、切向牵引 $\sigma_{xy} = 0$，顶边与底边其余部分为零牵引。柔顺度按完整结构（半域值 $\times 2$）报告。

## 2. 实验设置

实验脚本为 [`run_fixed_fixed.py`](run_fixed_fixed.py)，缺省一次跑完六组，也可只跑其中几组：

```bash
python experiments/paper_topopt_huzhang/run_fixed_fixed.py                      # 六组
python experiments/paper_topopt_huzhang/run_fixed_fixed.py --analyzer huzhang   # 只跑 HZMFEM 三组
```

开跑前统一检查全部运行目录，任一已存在即拒绝覆盖；每组开跑前写 `config.json` 并打印设置摘要。

### 2.1 模型参数

| 参数 | 数值或设置 |
|---|---|
| 设计域 | 左半域 $80\ \mathrm{mm} \times 20\ \mathrm{mm}$（完整梁 $160 \times 20$），平面应力：模型类默认 `domain=(0, 80, 0, 20)`，`PLANE_TYPE = "plane_stress"`，`FULL_STRUCTURE_FACTOR = 2.0` |
| 载荷 | 完整梁底边中点竖直合力 $P = -3$ N，名义分布于宽 $l = 1$ mm 的区间 $[79.5, 80.5]$，强度 $P/l$；左半域只保留 $[79.5, 80]$，合力 $P/2$ 由模型类自动给出：`LOAD = -3.0`，`LOAD_WIDTH = 1.0` |
| 实体材料 | $E_0 = 30$ MPa，$\nu = 0.4$（可压缩）：`E0 = 30.0`，`NU = 0.4` |

### 2.2 分析参数

| 参数 | 数值或设置 |
|---|---|
| 网格 | $80 \times 20$ 个矩形按棋盘式交替对角线剖成三角形，共 $3200$ 个单元，$h = 1$ mm：`TriangleMesh.from_checkerboard_box(box=problem.domain, nx=80, ny=20)` |
| 阶次 | $k \in \{2, 3, 4\}$，传入 `space_degree=order`；命令行 `--analyzer`、`--order` 可选子集 |
| 数值积分 | $2k + 2$ 阶：`integration_order=2 * order + 2` |
| 载荷离散 | 载荷区宽 $0.5$ mm，不与 $h = 1$ mm 的网格节点对齐；先把区间常值牵引精确 $L^2$ 投影到底边的连续 P1 迹空间（合力精确保持），再作为同一个连续牵引交给两条分析链：LFEM 经 Neumann 边界弱施加，HZMFEM 经应力自由度强施加，两条路径看到同一个载荷泛函：`project_patch_traction_to_p1_trace(...)`，`load_discretization = "p1_trace_l2_projection"` |
| 对称面 | LFEM：$u_x = 0$ 作 Dirichlet 条件；HZMFEM：切向牵引 $\sigma_{nt} = 0$ 强施加、法向牵引自由，$u_n = 0$ 自然满足：`is_symmetry_boundary` |
| LFEM 装配 | 标准逐单元装配：`ASSEMBLY_METHOD = "standard"` |
| HZMFEM 稳定化 | $k = 2$ 用跳量稳定化，系数取固定基材值（默认 `stabilization_coefficient = "fixed"`）；$k = 3, 4$ 原生格式。$E_0 = 30$ 未归一化，跳量积分器会打印"未进行归一化"的提示：鞍点系统条件数约 $O(E^2) \approx 10^3$，远在 MUMPS 的承受范围内，可忽略 |
| HZMFEM 角点松弛 | 关闭：`USE_RELAXATION = False`。Hu 与 Ma（2021）4.1 节的角点松弛只针对两条牵引边在角点处牵引数据不一致的情形；本模型对称面上的两个角点 $(80, 0)$、$(80, 20)$ 中，对称面只强施加切向牵引，与相邻底边（载荷区）或顶边（零牵引）对共享分量 $\sigma_{xy}$ 的要求都是 $0$，数据一致；两个固支角点 $(0, 0)$、$(0, 20)$ 是位移–牵引角点，不出现约束冲突 |
| 线性求解器 | MUMPS 直接法：`SOLVER = "mumps"` |
| 终态验收 | 优化结束后在最终物理密度上重解一次，柔顺度与相对平衡残差 `relative_equilibrium_residual`（LFEM 只计非 Dirichlet 自由度）都取自这次求解；`summary.json` 的 `converged` 为末次设计变化 $\le 10^{-2}$ |

### 2.3 优化参数

| 参数 | 数值或设置 |
|---|---|
| 优化目标 | 最小化柔顺度（泛函见第 1 节）：`ComplianceObjective(diff_mode="manual")` |
| 约束 | 体积分数 $\le 0.4$：`VolumeConstraint`，`VOLFRAC = 0.4` |
| 材料插值 | 修正 SIMP，只插值 Young 模量：$E(\rho) = E_{\min} + \rho^{q}(E_0 - E_{\min})$，$q = 3$，$E_{\min} = 10^{-9}$：`interpolation_method="msimp"`，`PENALTY = 3.0`，`EMIN = 1.0e-9`，`target_variables=["E"]` |
| 初始密度 | 均匀 $0.4$，即体积分数：`relative_density=VOLFRAC` |
| 密度过滤 | 线性密度过滤，$r_{\min} = 2.4$ mm：`Filter(filter_type="density", rmin=FILTER_RADIUS, filter_q=3)`，`FILTER_RADIUS = 2.4` |
| 优化器 | MMA，移动限 $0.2$，渐近线初值 $0.5$：`MMAOptimizer`，`MOVE_LIMIT = 0.2`，`ASYMP_INIT = 0.5` |
| 停止准则 | 相邻两步设计变量变化的无穷范数 $\le 10^{-2}$，或达到 $500$ 步：`CHANGE_TOLERANCE = 1.0e-2`，`MAX_ITERATIONS = 500` |

## 3. 结果

六组均按停止准则收敛（末次设计变化 $\le 10^{-2}$，`converged: true`）。下表取自各运行目录的 `summary.json`：迭代步数为 `optimization_iterations`，柔顺度为 `compliance`（半域）乘 `full_structure_factor = 2` 的完整结构值，平衡残差为 `relative_equilibrium_residual`。LFEM 与 HZMFEM 的柔顺度出自不同泛函（第 1 节），列内只宜同链比较；同一泛函下的横比见表 5.3 的交叉重分析。

| 分析链 | $k$ | 迭代步数 | 柔顺度（完整结构，N·mm） | 体积分数 | 平衡残差 |
|---|---|---|---|---|---|
| LFEM | 2 | 230 | 31.7309 | 0.4000 | $5.2 \times 10^{-12}$ |
| LFEM | 3 | 230 | 31.8252 | 0.4000 | $1.3 \times 10^{-11}$ |
| LFEM | 4 | 254 | 31.8491 | 0.4000 | $2.8 \times 10^{-11}$ |
| HZMFEM | 2 | 284 | 32.7774 | 0.4000 | $6.3 \times 10^{-12}$ |
| HZMFEM | 3 | 277 | 32.1166 | 0.4000 | $1.9 \times 10^{-7}$ |
| HZMFEM | 4 | 319 | 31.9139 | 0.4000 | $1.8 \times 10^{-10}$ |

体积约束在六组中都处于激活状态（体积分数 $0.4000$）。HZMFEM 的柔顺度随阶次单调下降（$32.78 \to 32.12 \to 31.91$），LFEM 三组在 $31.73 \sim 31.85$ 之间。HZMFEM $k = 3$ 的平衡残差为 $1.9 \times 10^{-7}$，其余各组不超过 $2 \times 10^{-10}$。

### 3.1 论文插图与表格

| 图表 | 产物 case | 内容 | 数据来源 |
|---|---|---|---|
| 图 5.2 | `compliance-topology` | 六组最终构型，左半域密度按对称面镜像为全域 | 六份 `density_final.vtu` |
| 图 5.3 | `compliance-convergence` | 两条分析链各三阶的柔顺度收敛历史与体积分数 | 六份 `history.json` |
| 表 5.3 | `table.py compliance-reanalysis` | 六个冻结设计在六种离散下的柔顺度交叉重分析 | `postprocess/frozen_reanalysis.json`，表格写 `results/tables/table5_3.md` |

成图命令 `python experiments/paper_topopt_huzhang/plot.py <case>`；表 5.3 由 `python experiments/paper_topopt_huzhang/table.py compliance-reanalysis` 重分析并生成。

### 3.2 产物文件

每组运行写 `outputs/compliance-fixed-fixed-half/<run>/`，确认后经 `sync_results.py` 固化到 `results/compliance-fixed-fixed-half/<run>/`，运行名 `analyzer-{lfem,huzhang}__order-{2,3,4}`。

| 文件 | 内容 |
|---|---|
| `config.json` | 开跑前记录的全部设置：模型、离散（含角点松弛开关）、优化参数与代码版本戳记 |
| `summary.json` | 终态：柔顺度（半域）与完整结构系数、体积分数、迭代步数、是否收敛、平衡残差、稳定化系数、代码版本戳记 |
| `history.json` | 逐步记录：步号、设计变化量、耗时与标量历史（柔顺度、体积分数） |
| `density_initial.vtu` | 初始设计经过滤后的物理密度，只含 `density` |
| `density_final.vtu` | 最终物理密度 `density`，插图与交叉重分析的输入 |
| `displacement_final.npy`、`stress_final.npy` | 终态求解的位移自由度向量；`stress_final.npy` 只有 Hu--Zhang 写出，是应力自由度向量，编号随 `use_relaxation` 而定（本算例不松弛，即原始 Hu--Zhang 空间的编号） |

逐步帧 `iterations/iter_NNNN.vtu` 与 `evolution.pvd` 在优化结束后一次写入运行目录并留在原处，需要在 Windows ParaView 中查看时手动拷到查看目录；`sync_results.py` 不拷帧。

入库的 LFEM 三组是 2026-09-10 的运行（`ad95594`），只有 `summary.json`、`history.json` 与 `density_final.vtu`。HZMFEM 三组是 2026-10-15 用当前代码（`c0b5a69`）关闭角点松弛重跑的，产物齐全；此前开松弛的三组（`a08562b`、`ad95594`）已被替换，可从 git 历史取回。

### 3.3 角点松弛对照

HZMFEM 三组此前曾以开松弛跑过（四个几何角点全部松弛），2026-10-15 关闭松弛用当前代码重跑。两组的代码版本不同：关松弛为 `c0b5a69`，开松弛为 `a08562b`（$k = 2$）与 `ad95594`（$k = 3, 4$，FEALPy 移植前），故下表只说明关闭松弛后结论不变，差异不能全部归于松弛开关。开松弛数据取自 `git c0b5a69` 版本的 `results/compliance-fixed-fixed-half/analyzer-huzhang__order-{2,3,4}/`（`summary.json`、`density_final.vtu`），可用 `git show` 取回。

| $k$ | 松弛 | 代码 | 迭代步数 | 柔顺度（完整结构，N·mm） | 体积分数 | 平衡残差 |
|---|---|---|---|---|---|---|
| 2 | 关 | `c0b5a69` | 284 | 32.7774 | 0.4000 | $6.3 \times 10^{-12}$ |
|  | 开 | `a08562b` | 311 | 32.7619 | 0.4000 | $4.3 \times 10^{-12}$ |
| 3 | 关 | `c0b5a69` | 277 | 32.1166 | 0.4000 | $1.9 \times 10^{-7}$ |
|  | 开 | `ad95594` | 301 | 32.1091 | 0.4000 | $3.0 \times 10^{-10}$ |
| 4 | 关 | `c0b5a69` | 319 | 31.9139 | 0.4000 | $1.8 \times 10^{-10}$ |
|  | 开 | `ad95594` | 342 | 31.9098 | 0.4000 | $1.8 \times 10^{-11}$ |

终态物理密度的差异：

| $k$ | 柔顺度相对差（关 / 开 − 1） | $\tilde\rho = 0.5$ 两侧翻转的单元 | 最大 $\lvert\Delta\tilde\rho\rvert$ | 固支角点 $5$ mm 内（翻转 / 平均 $\lvert\Delta\tilde\rho\rvert$） | 对称面角点 $5$ mm 内（翻转 / 平均 $\lvert\Delta\tilde\rho\rvert$） | 其余平均 $\lvert\Delta\tilde\rho\rvert$ |
|---|---|---|---|---|---|---|
| 2 | +4.7e-04 | 0（0.00%） | 0.012 | 0 / 0.0000 | 0 / 0.0000 | 0.0001 |
| 3 | +2.3e-04 | 0（0.00%） | 0.014 | 0 / 0.0000 | 0 / 0.0000 | 0.0001 |
| 4 | +1.3e-04 | 2（0.06%） | 0.016 | 0 / 0.0000 | 0 / 0.0000 | 0.0004 |

三组的最终设计几乎相同：单元密度的最大差不超过 $0.016$，至多 $2$ 个单元在 $\tilde\rho = 0.5$ 两侧翻转，固支角点与对称面角点 $5$ mm 内无差异；柔顺度相对差在 $5 \times 10^{-4}$ 以内且随阶次减小。关松弛的迭代步数比开松弛少二十余步，只是 MMA 在停止阈值附近触发时机不同。

## 4. 实验结论

- 六组离散均在 $230 \sim 319$ 步内满足同一停止准则，最终体积分数接近上限 $0.4$。采用 LFEM $p=3,4$ 与 HZMFEM $k=3,4$ 四种离散分别进行统一再分析时，六组最终设计的柔顺度最大相对差约为 $0.86\%$，小于各组优化所得柔顺度之间的 $3.30\%$；相对差按最大值与最小值之差除以最小值计算。这表明六组设计在统一再分析下的性能较为接近，不能直接根据各自优化时报告的柔顺度判断设计优劣。低阶稳定化与高阶非稳定化 Hu--Zhang 格式均获得了与位移法相近的主要承载构型及再分析柔顺度，支持所提框架在不同阶次下用于柔顺度最小化问题。
- HZMFEM 不需要角点松弛（2.2 节）：按 Hu 与 Ma 的判据本模型没有牵引数据不一致的角点；关闭松弛后 $k = 2, 3, 4$ 的最终设计与开松弛几乎相同，柔顺度相对差在 $5 \times 10^{-4}$ 以内（3.3 节）。两次运行的代码版本也不同，差异不能全部归于松弛开关。
