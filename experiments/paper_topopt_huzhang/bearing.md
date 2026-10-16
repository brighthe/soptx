# 二维轴承装置近不可压缩柔顺度最小化（论文 5.2.2 节）

## 1. 研究对象

在体积约束下最小化柔顺度，用可压缩与近不可压缩两组材料考察线性位移元的体积闭锁及其对最终构型的影响：每组材料以 Lagrange 位移元（LFEM）$p = 1, 2$ 与 Hu--Zhang 混合元（HZMFEM）$k = 2$ 各做一次优化，共六组；再把六个冻结设计在四种离散下重分析，以 HZMFEM $k = 4$ 为参考（论文表 5.4），并在全实体域上做 $h$ 收敛考察（论文图 5.5），区分"$p = 2$ 不闭锁"与"$p = 2$ 的闭锁被细网格掩盖"。

| | LFEM | HZMFEM |
| --- | --- | --- |
| 阶次 | 位移插值阶 $p = 1, 2$；$p = 1$ 用作线性位移元体积闭锁的对照 | 应力插值阶 $k = 2$，位移阶 $1$；重分析另用 $k = 4$ 作参考 |
| 柔顺度泛函 | 外载功 $f^\mathsf{T} u$：`ComplianceObjective(state_variable="u")` | 互补能 $\sigma^\mathsf{T} A(\rho)\, \sigma$：`ComplianceObjective(state_variable="sigma")` |
| $\nu \to 1/2$ | 低阶元在本网格上体积闭锁 | 应力–位移混合格式，不闭锁；$k = 2$ 为跳量稳定化格式（$k \le d$），`stabilization_coefficient = "fixed"` |

两组材料：

| 组名（运行目录） | 实体 $\nu_0$ | 材料插值 |
|---|---|---|
| `nu-0.3` | $0.3$ | 只插值 Young 模量 |
| `nu-0.4999` | $0.4999$ | 同时插值 Young 模量与 Poisson 比，空材料退化为可压缩弱材料，避免空单元的体积闭锁污染实体区 |

模型为 [`BearingTraction2d`](../../src/soptx/problems/benchmarks/bearing.py)：设计域 $120\ \mathrm{mm} \times 40\ \mathrm{mm}$，底边完全固支，左右侧边自由（零牵引），顶边竖直向下均布牵引。

## 2. 实验设置

实验脚本为 [`run_bearing.py`](run_bearing.py)，缺省一次跑完两组材料 $\times$ 三种离散，也可只跑其中几组：

```bash
python experiments/paper_topopt_huzhang/run_bearing.py                                          # 六组
python experiments/paper_topopt_huzhang/run_bearing.py --analyzer huzhang                       # 只跑 HZMFEM 两组
python experiments/paper_topopt_huzhang/run_bearing.py --group nu-0.4999 --analyzer lfem --order 1
```

运行目录为 `outputs/bearing/<组名>/<运行名>/`，同步时算例名取 `bearing/<组名>`。开跑前统一检查全部运行目录，任一已存在即拒绝覆盖；每组开跑前写 `config.json` 并打印设置摘要。

### 2.1 模型参数

| 参数 | 数值或设置 |
|---|---|
| 设计域 | $120\ \mathrm{mm} \times 40\ \mathrm{mm}$，平面应变：`DOMAIN = (0.0, 120.0, 0.0, 40.0)`，`PLANE_TYPE = "plane_strain"` |
| 载荷 | 顶边全长竖直均布牵引 $t = -0.08$ N/mm，合力 $P = t \times 120 = -9.6$ N：`TRACTION = -0.08`。载荷标记只取顶边（`marker=self._on_top_boundary`），见 2.2 节"角点松弛"一行 |
| 实体材料 | $E_0 = 1$ MPa；$\nu_0$ 按材料组取 $0.3$ 或 $0.4999$：`E0 = 1.0`，`GROUPS = {"nu-0.3": (0.3, "E"), "nu-0.4999": (0.4999, "E+nu")}` |

### 2.2 分析参数

| 参数 | 数值或设置 |
|---|---|
| 网格 | $120 \times 40$ 个矩形剖成三角形，共 $9600$ 个单元，$h = 1$ mm；左半取 `/`、右半取 `\`，关于中线 $x = 60$ 镜像对称；除两个顶角格子互为镜像地翻转外，内部顶点周围都是 $6$ 个三角形，即低阶位移元体积闭锁的经典构型：`TriangleMesh.from_mirror_box(box=problem.domain, nx=120, ny=40)`，`MESH_TYPE = "triangle-single-diagonal-symmetric"` |
| 离散 | 每组材料三种：`RUNS = (("lfem", 1), ("lfem", 2), ("huzhang", 2))`；命令行 `--group`、`--analyzer`、`--order` 可选子集 |
| 数值积分 | $2k + 2$ 阶：`integration_order=2 * order + 2` |
| LFEM 装配 | 标准逐单元装配：`ASSEMBLY_METHOD = "standard"` |
| HZMFEM 稳定化 | $k = 2$ 用跳量稳定化，系数取固定基材值（默认 `stabilization_coefficient = "fixed"`） |
| HZMFEM 角点松弛 | 关闭：`USE_RELAXATION = False`。Hu 与 Ma（2021）4.1 节的角点松弛只针对两条牵引边在角点处牵引数据不一致的情形；本模型顶边两个角点 $(0, 40)$、$(120, 40)$ 是牵引–牵引角点，顶边受法向压载 $(0, t)$、侧边为零牵引，两侧对 $\sigma_{xy}$ 的要求都是 $0$，数据一致（$\sigma = \mathrm{diag}(0, t)$ 同时满足两侧）；底边两个角点是位移–牵引角点，不出现约束冲突。分析器在边端点上逐点取牵引值，载荷标记若取 `is_traction_boundary`（顶边与侧边），侧边顶端端点落在 $y = 40$ 上会取到顶边载荷，角点两侧数据被误判为不相容；2026-10-16 起载荷标记只取顶边，与 `BridgeTraction2d` 一致，LFEM 载荷向量不变（逐位相同） |
| 线性求解器 | MUMPS 直接法：`SOLVER = "mumps"` |
| 终态验收 | 优化结束后在最终物理密度上重解一次，柔顺度与相对平衡残差 `relative_equilibrium_residual`（LFEM 只计非 Dirichlet 自由度）都取自这次求解；`summary.json` 的 `converged` 为末次设计变化 $\le 10^{-2}$ |

### 2.3 优化参数

| 参数 | 数值或设置 |
|---|---|
| 优化目标 | 最小化柔顺度（泛函见第 1 节）：`ComplianceObjective(diff_mode="manual")` |
| 约束 | 体积分数 $\le 0.35$：`VolumeConstraint`，`VOLFRAC = 0.35` |
| 材料插值 | 修正 SIMP：$E(\rho) = E_{\min} + \rho^{q}(E_0 - E_{\min})$，$q = 3$，$E_{\min} = 10^{-9}$；`nu-0.4999` 组另插值 Poisson 比 $\nu(\rho) = \nu_{\mathrm{void}} + \rho^{q_\nu}(\nu_0 - \nu_{\mathrm{void}})$，$\nu_{\mathrm{void}} = 0.3$，$q_\nu = 1$（只在 $\nu_0 \ge 0.49$ 时生效）：`interpolation_method="msimp"`，`PENALTY = 3.0`，`EMIN = 1.0e-9`，`NU_VOID = 0.3`，`NU_PENALTY = 1.0` |
| 初始密度 | 均匀 $0.35$，即体积分数：`relative_density=VOLFRAC` |
| 密度过滤 | 线性密度过滤，$r_{\min} = 2$ mm：`Filter(filter_type="density", rmin=FILTER_RADIUS, filter_q=3)`，`FILTER_RADIUS = 2.0` |
| 优化器 | OC，移动限 $0.2$，阻尼指数 $0.5$；体积约束乘子在 $[0, 10^9]$ 上二分，相对收敛容差 $(l_2 - l_1)/(l_2 + l_1) \le 10^{-3}$。MMA 及其渐近线扫描不能复现三拱构型，故用 OC：`OCOptimizer`，`MOVE_LIMIT = 0.2`，`DAMPING = 0.5`，`INITIAL_LAMBDA = 1.0e9`，`BISECTION_TOL = 1.0e-3` |
| 停止准则 | 相邻两步设计变量变化的无穷范数 $\le 10^{-2}$，或达到 $1000$ 步：`CHANGE_TOLERANCE = 1.0e-2`，`MAX_ITERATIONS = 1000` |

## 3. 结果

六组均按停止准则收敛（末次设计变化 $\le 10^{-2}$，`converged: true`）。下表取自各运行目录的 `summary.json`：迭代步数为 `optimization_iterations`，柔顺度为全域 `compliance`，平衡残差为 `relative_equilibrium_residual`。LFEM 与 HZMFEM 的柔顺度出自不同泛函（第 1 节），且各自含本离散的闭锁或离散误差，同一参考下的横比见表 5.4 的交叉重分析。

| 材料组 | 分析链 | 迭代步数 | 柔顺度（N·mm） | 体积分数 | 平衡残差 |
|---|---|---|---|---|---|
| `nu-0.3` | LFEM $p = 1$ | 281 | 119.329 | 0.3500 | $2.2 \times 10^{-13}$ |
| `nu-0.3` | LFEM $p = 2$ | 351 | 122.733 | 0.3500 | $8.9 \times 10^{-13}$ |
| `nu-0.3` | HZMFEM $k = 2$ | 255 | 127.379 | 0.3500 | $2.0 \times 10^{-10}$ |
| `nu-0.4999` | LFEM $p = 1$ | 372 | 76.435 | 0.3500 | $1.2 \times 10^{-10}$ |
| `nu-0.4999` | LFEM $p = 2$ | 245 | 98.061 | 0.3500 | $4.1 \times 10^{-10}$ |
| `nu-0.4999` | HZMFEM $k = 2$ | 382 | 101.358 | 0.3500 | $1.5 \times 10^{-10}$ |

体积约束在六组中都处于激活状态（体积分数 $0.3500$）。同一离散下，近不可压组的柔顺度都低于可压缩组；其中 LFEM $p = 1$ 降得最多（$119.33 \to 76.44$，$-36\%$），LFEM $p = 2$ 与 HZMFEM $k = 2$ 都降约 $20\%$。Poisson 比增大本身就使平面应变下的结构变刚，$p = 1$ 多出的降幅与线性位移元在 $\nu_0 \to 1/2$ 时体积闭锁、离散过刚的预期一致，定量结论以表 5.4 与图 5.5 为准。

### 3.1 论文插图与表格

| 图表 | 产物 case | 内容 | 数据来源 |
|---|---|---|---|
| 图 5.5 | `bearing-solid-h-convergence` | 全实体域（$\rho \equiv 1$）上各离散柔顺度相对误差随 $h$ 的收敛，两档 $\nu_0$，网格 $30 \times 10$ 至 $240 \times 80$ 四级，参考为 HZMFEM $k = 4$ 前三级的 Aitken 外推 | `plot.py bearing-h-locking` 导出的 `postprocess/solid_h_sweep.json` |
| 图 5.6 | `bearing-topologies` | 两组材料 $\times$ 三种离散的最终构型 | 六份 `density_final.vtu` |
| 表 5.4 | `table.py bearing-reanalysis` | 六个冻结设计的优化所得柔顺度、HZMFEM $k = 4$ 再分析柔顺度及各离散相对后者的偏差 | `postprocess/frozen_reanalysis.json`，表格写 `results/tables/table5_4.md` |

成图命令 `python experiments/paper_topopt_huzhang/plot.py <case>`，缺数据时入口会提示先运行对应的重分析动词；表 5.4 由 `python experiments/paper_topopt_huzhang/table.py bearing-reanalysis` 重分析并生成。

### 3.2 产物文件

每组运行写 `outputs/bearing/<组名>/<run>/`，确认后经 `sync_results.py`（算例名取 `bearing/<组名>`）固化到 `results/bearing/<组名>/<run>/`，运行名 `analyzer-lfem__order-{1,2}` 与 `analyzer-huzhang__order-2`。

| 文件 | 内容 |
|---|---|
| `config.json` | 开跑前记录的全部设置：材料组、模型、离散（含角点松弛开关）、优化参数（含材料插值与 OC 参数）与代码版本戳记 |
| `summary.json` | 终态：柔顺度（全域）、体积分数、迭代步数、是否收敛、平衡残差、稳定化系数、代码版本戳记 |
| `history.json` | 逐步记录：步号、设计变化量、耗时与标量历史（柔顺度、体积分数） |
| `density_initial.vtu` | 初始设计经过滤后的物理密度，只含 `density` |
| `density_final.vtu` | 最终物理密度 `density`，插图与交叉重分析的输入 |
| `displacement_final.npy`、`stress_final.npy` | 终态求解的位移自由度向量；`stress_final.npy` 只有 Hu--Zhang 写出，是应力自由度向量，编号随 `use_relaxation` 而定（本算例不松弛，即原始 Hu--Zhang 空间的编号） |

逐步帧 `iterations/iter_NNNN.vtu` 与 `evolution.pvd` 在优化结束后一次写入运行目录并留在原处，需要在 Windows ParaView 中查看时手动拷到查看目录；`sync_results.py` 不拷帧。

入库的 LFEM 四组是 2026-09-14 的运行（`ad95594`），只有 `summary.json`、`history.json` 与 `density_final.vtu`。HZMFEM 两组是 2026-10-15 用当前代码（`c0b5a69`）关闭角点松弛、并修正载荷标记后重跑的，产物齐全；此前开松弛的两组（`f70fe7f`）已被替换，可从 git 历史取回。

### 3.3 角点松弛对照

HZMFEM 两组此前曾以开松弛跑过（四个几何角点全部松弛），2026-10-15 关闭松弛用当前代码重跑。两组之间有三处不同：角点松弛开关；代码版本（关松弛为 `c0b5a69`，开松弛为 `f70fe7f`）；载荷标记（2.2 节）。开松弛时各边按自身端点写入分裂后的角点自由度，侧边顶端端点取到顶边载荷，等于在侧边顶端单元内多施加一段从 $t$ 线性衰减到 $0$ 的剪切牵引；这一点由代码结构推断，未作数值量化。故下表只说明关闭松弛后结论不变，差异不能归于松弛开关。开松弛数据取自 `git c0b5a69` 版本的 `results/bearing/<组名>/analyzer-huzhang__order-2/`（`summary.json`、`density_final.vtu`），可用 `git show` 取回。

| 材料组 | 松弛 | 代码 | 迭代步数 | 柔顺度（N·mm） | 体积分数 | 平衡残差 |
|---|---|---|---|---|---|---|
| `nu-0.3` | 关 | `c0b5a69` | 255 | 127.379 | 0.3500 | $2.0 \times 10^{-10}$ |
|  | 开 | `f70fe7f` | 464 | 127.678 | 0.3500 | $1.0 \times 10^{-10}$ |
| `nu-0.4999` | 关 | `c0b5a69` | 382 | 101.358 | 0.3500 | $1.5 \times 10^{-10}$ |
|  | 开 | `f70fe7f` | 343 | 102.374 | 0.3500 | $4.4 \times 10^{-10}$ |

终态物理密度的差异：

| 材料组 | 柔顺度相对差（关 / 开 − 1） | $\tilde\rho = 0.5$ 两侧翻转的单元 | 最大 $\lvert\Delta\tilde\rho\rvert$ | 底边固支角点 $5$ mm 内（翻转 / 平均 $\lvert\Delta\tilde\rho\rvert$） | 顶边载荷角点 $5$ mm 内（翻转 / 平均 $\lvert\Delta\tilde\rho\rvert$） | 其余平均 $\lvert\Delta\tilde\rho\rvert$ |
|---|---|---|---|---|---|---|
| `nu-0.3` | -2.3e-03 | 88（0.92%） | 0.497 | 0 / 0.0025 | 8 / 0.0284 | 0.0099 |
| `nu-0.4999` | -9.9e-03 | 116（1.21%） | 0.658 | 2 / 0.0384 | 6 / 0.0273 | 0.0113 |

两组都收敛到相同的三拱拓扑，差异是杆件边界的小幅平移（翻转单元约 $1\%$）。顶边载荷角点附近的平均密度差为其余区域的 $2.4 \sim 2.9$ 倍，与该处载荷标记的修正一致；近不可压组的柔顺度差约 $1\%$，大于可压缩组的 $0.2\%$。迭代步数一组减少、一组增加，没有一致的方向。

## 4. 实验结论

- 六组离散都在 $245 \sim 382$ 步内按同一准则收敛，体积约束激活于 $0.35$。近不可压组中 LFEM $p = 1$ 的柔顺度显著低于其余两种离散，与线性位移元的体积闭锁一致；$p = 2$ 是否闭锁、闭锁是否被细网格掩盖，以表 5.4 与图 5.5 为准。
- HZMFEM 不需要角点松弛（2.2 节）：按 Hu 与 Ma 的判据本模型没有牵引数据不一致的角点；载荷标记修正后，关闭松弛的 $k = 2$ 两组收敛到与开松弛相同的拓扑，柔顺度相差 $0.2\%$ 与 $1.0\%$（3.3 节）。两次运行之间还有代码版本与载荷标记的差别，差异不能归于松弛开关。
