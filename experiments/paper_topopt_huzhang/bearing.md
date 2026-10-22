# 二维轴承装置近不可压缩柔顺度最小化（论文 5.2.2 节）

## 1. 研究对象

在体积约束下最小化柔顺度，用可压缩与近不可压缩两组材料考察线性位移元的体积闭锁及其对最终构型的影响：每组材料以 Lagrange 位移元（LFEM）$p = 1, 2$ 与 Hu--Zhang 混合元（HZMFEM）$k = 2,3$ 各做一次优化，共八组；再把八个冻结设计在五种离散下重分析，以 HZMFEM $k = 4$ 为参考（论文表 5.4），并在全实体域上做 $h$ 收敛考察（论文图 5.5），按相同位移逼近次数配对比较 LFEM $p=1$ 与 HZMFEM $k=2$、LFEM $p=2$ 与 HZMFEM $k=3$，分别考察状态分析精度与最终设计性能，并将抗自锁结论限定在本算例范围内。

| | LFEM | HZMFEM |
| --- | --- | --- |
| 阶次 | 位移插值阶 $p = 1, 2$；$p = 1$ 用作线性位移元体积闭锁的对照 | 应力插值阶 $k = 2,3$，位移阶分别为 $1,2$；重分析另用 $k = 4$ 作参考 |
| 柔顺度泛函 | 外载功 $f^\mathsf{T} u$：`ComplianceObjective(state_variable="u")` | 互补能 $\sigma^\mathsf{T} A(\rho)\, \sigma$：`ComplianceObjective(state_variable="sigma")` |
| $\nu \to 1/2$ | 本算例全实体测试中，线性位移元 $p=1$ 表现出明显体积自锁，$p=2$ 的柔顺度偏差较小且随加密减小 | 应力–位移混合格式，本算例全实体测试中未见明显自锁；$k = 3$ 无需稳定化，$k = 2$ 为跳量稳定化格式（$k \le d$），`stabilization_coefficient = "fixed"` |

两组材料：

| 组名（运行目录） | 实体 $\nu_0$ | 材料插值 |
|---|---|---|
| `nu-0.3` | $0.3$ | 只插值 Young 模量 |
| `nu-0.4999` | $0.4999$ | 同时插值 Young 模量与 Poisson 比，空材料退化为可压缩弱材料，避免空单元的体积闭锁污染实体区 |

模型为 [`BearingTraction2d`](../../src/soptx/problems/benchmarks/bearing.py)：设计域 $120\ \mathrm{mm} \times 40\ \mathrm{mm}$，底边完全固支，左右侧边自由（零牵引），顶边竖直向下均布牵引。

## 2. 实验设置

实验脚本为 [`run_bearing.py`](run_bearing.py)，缺省一次跑完两组材料 $\times$ 四种离散，也可只跑其中几组：

```bash
python experiments/paper_topopt_huzhang/run_bearing.py                                          # 八组
python experiments/paper_topopt_huzhang/run_bearing.py --analyzer huzhang                       # 只跑 HZMFEM 四组
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
| 离散 | 每组材料四种：`RUNS = (("lfem", 1), ("huzhang", 2), ("lfem", 2), ("huzhang", 3))`；命令行 `--group`、`--analyzer`、`--order` 可选子集 |
| 数值积分 | $2k + 2$ 阶：`integration_order=2 * order + 2` |
| LFEM 装配 | 标准逐单元装配：`ASSEMBLY_METHOD = "standard"` |
| HZMFEM 稳定化 | $k = 2$ 用跳量稳定化，系数取固定基材值（默认 `stabilization_coefficient = "fixed"`）；$k = 3$ 不启用跳量稳定化 |
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

八组均按停止准则收敛（末次设计变化 $\le 10^{-2}$，`converged: true`）。下表取自各运行目录的 `summary.json`：迭代步数为 `optimization_iterations`，柔顺度为全域 `compliance`，平衡残差为 `relative_equilibrium_residual`。LFEM 与 HZMFEM 的柔顺度出自不同泛函（第 1 节），且各自含本离散的闭锁或离散误差，同一参考下的横比见表 5.4 的交叉重分析。

| 材料组 | 分析链 | 迭代步数 | 柔顺度（N·mm） | 体积分数 | 平衡残差 |
|---|---|---|---|---|---|
| `nu-0.3` | LFEM $p=1$ | 281 | 119.329 | 0.34999973 | 2.25e-13 |
| `nu-0.3` | HZMFEM $k=2$ | 255 | 127.379 | 0.35001646 | 1.98e-10 |
| `nu-0.3` | LFEM $p=2$ | 351 | 122.733 | 0.35000409 | 8.87e-13 |
| `nu-0.3` | HZMFEM $k=3$ | 302 | 124.831 | 0.35002277 | 1.93e-09 |
| `nu-0.4999` | LFEM $p=1$ | 372 | 76.435 | 0.34999923 | 1.16e-10 |
| `nu-0.4999` | HZMFEM $k=2$ | 382 | 101.358 | 0.34998744 | 1.53e-10 |
| `nu-0.4999` | LFEM $p=2$ | 245 | 98.061 | 0.35000351 | 4.09e-10 |
| `nu-0.4999` | HZMFEM $k=3$ | 351 | 99.531 | 0.34999865 | 3.55e-09 |

体积约束在八组中都处于激活状态（体积分数 $0.3500$）。同一离散下，近不可压组的柔顺度都低于可压缩组；其中 LFEM $p = 1$ 降得最多（$119.33 \to 76.44$，$-36\%$），LFEM $p = 2$ 与 HZMFEM $k = 2,3$ 都降约 $20\%$。Poisson 比增大本身就使平面应变下的结构变刚，$p = 1$ 多出的降幅与线性位移元在 $\nu_0 \to 1/2$ 时体积闭锁、离散过刚的预期一致，定量结论以表 5.4 与图 5.5 为准。

### 3.1 论文插图与表格

| 图表 | 产物 case | 内容 | 数据来源 |
|---|---|---|---|
| 图 5.5 | `bearing-solid-h-convergence` | 全实体域（$\rho \equiv 1$）上各离散柔顺度相对误差随 $h$ 的收敛，两档 $\nu_0$，网格 $30 \times 10$ 至 $240 \times 80$ 四级，展示 LFEM $p=1,2$ 与 HZMFEM $k=2,3$，参考为 HZMFEM $k=4$ 前三级按观测阶的 Richardson 外推（等比误差模型下采用 Aitken delta-squared 形式） | `plot.py bearing-h-locking` 导出的 `postprocess/solid_h_sweep.json` |
| 图 5.6 | `bearing-topologies` | 两组材料 $\times$ 四种离散的最终构型 | 八份 `density_final.vtu` |
| 表 5.4 | `table.py bearing-reanalysis` | 八个冻结设计的优化所得柔顺度、HZMFEM $k = 4$ 再分析柔顺度及各离散相对后者的偏差 | `postprocess/frozen_reanalysis.json`，表格写 `results/tables/table5_4.md` |

成图命令 `python experiments/paper_topopt_huzhang/plot.py --case <case>`，缺数据时入口会提示先运行对应的重分析动词；表 5.4 由 `python experiments/paper_topopt_huzhang/table.py bearing-reanalysis` 重分析并生成。

### 3.2 产物文件

每组运行写 `outputs/bearing/<组名>/<run>/`，确认后经 `sync_results.py`（算例名取 `bearing/<组名>`）固化到 `results/bearing/<组名>/<run>/`，运行名 `analyzer-lfem__order-{1,2}` 与 `analyzer-huzhang__order-{2,3}`。

| 文件 | 内容 |
|---|---|
| `config.json` | 开跑前记录的全部设置：材料组、模型、离散（含角点松弛开关）、优化参数（含材料插值与 OC 参数）与代码版本戳记 |
| `summary.json` | 终态：柔顺度（全域）、体积分数、迭代步数、是否收敛、平衡残差、稳定化系数、代码版本戳记 |
| `history.json` | 逐步记录：步号、设计变化量、耗时与标量历史（柔顺度、体积分数） |
| `density_initial.vtu` | 初始设计经过滤后的物理密度，只含 `density` |
| `density_final.vtu` | 最终物理密度 `density`，插图与交叉重分析的输入 |
| `displacement_final.npy`、`stress_final.npy` | 终态求解的位移自由度向量；`stress_final.npy` 只有 Hu--Zhang 写出，是应力自由度向量，编号随 `use_relaxation` 而定（本算例不松弛，即原始 Hu--Zhang 空间的编号） |

逐步帧 `iterations/iter_NNNN.vtu` 与 `evolution.pvd` 在优化结束后一次写入运行目录并留在原处，需要在 Windows ParaView 中查看时手动拷到查看目录；`sync_results.py` 不拷帧。

入库的 LFEM 四组是 2026-09-14 的运行（`ad95594`），仅保存 `summary.json`、`history.json` 与 `density_final.vtu`，未保存独立配置、初始密度及终态位移向量。HZMFEM $k=2$ 两组是 2026-10-15 关闭角点松弛、修正载荷标记后重跑的结果（`c0b5a69`）；$k=3$ 两组于 2026-10-17 补充完成（`7cafae9`，工作区含未提交修改），分别经过 302 与 351 步收敛。四组 HZMFEM 均保存上述七类文件。此前开启角点松弛的 $k=2$ 两组（`f70fe7f`）已被替换，可从 git 历史取回，历史对照见第 3.3 节。

支撑论文图表的数据已齐全：`postprocess/frozen_reanalysis.json` 保存八个设计在五种离散下的 40 项再分析结果；`postprocess/solid_h_sweep.json` 保存两组材料下 LFEM $p=1,2$ 与 HZMFEM $k=2,3$ 的四级网格结果，以及 HZMFEM $k=4$ 的前三级外推参考数据。表 5.4 保存于 `results/tables/table5_4.md`，图 5.5、5.6 保存于 `results/figures/`。这不表示历史 LFEM 运行的全部原始文件均已归档；逐步密度帧也不在入库范围内。

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

## 3.4 图 5.5 补算: HZMFEM k=3 (2026-10-17 完成)

为按相同位移逼近次数比较 LFEM p=1,2 与 HZMFEM k=2,3, 补算全实体域的 HZMFEM k=3. 本计划不包含拓扑优化, 不新增 k=1. 固定 rho=1, 泊松比取 0.3 与 0.4999, 网格为 30x10、60x20、120x40、240x80, 共 8 次状态求解. 几何、载荷、网格生成与角点不松弛设置沿用 `run_bearing.build`; k=3 无需跳量稳定化.

在 WSL 的仓库根目录、已配置的 Python 环境执行:

```bash
python experiments/paper_topopt_huzhang/plot.py bearing-h-locking --analyzer huzhang --order 3
```

脚本默认读取已有 `results/bearing/postprocess/solid_h_sweep.json`, 只补缺失结果, 保留已有 LFEM、HZMFEM k=2、k=4 的数值及外推参考值, 无需 `--resume` 参数. 已有文件必须包含两组材料四级网格和 k=4 前三级参考数据. 新增计算单独记录溯源信息, 不把旧数据归于本次代码版本. 重复执行会跳过已保存的 k=3 结果; 本次任务全部完成后才原子替换结果文件.

已用 WSL `ihpcm` 环境完成 8 次状态求解, 两组材料均有四个有限且为正的 k=3 柔顺度结果. 与运行前备份逐项比较, 原有柔顺度、耗时、`reference_values` 及原始溯源字段不变. 新结果的溯源单独写入 `incremental_runs`; 本次代码工作区含未提交修改, `reproducible=false`, 未将其标为干净提交下的复现实验.

| 网格 | nu=0.3 柔顺度 (N·mm) | 偏差绝对值 (%) | nu=0.4999 柔顺度 (N·mm) | 偏差绝对值 (%) |
|---|---:|---:|---:|---:|
| 30x10 | 26.0197705026 | 0.03009581 | 15.5544043123 | 0.38981719 |
| 60x20 | 26.0148455234 | 0.01116228 | 15.5204037552 | 0.17037390 |
| 120x40 | 26.0130207007 | 0.00414695 | 15.5055661975 | 0.07461068 |
| 240x80 | 26.0123422284 | 0.00153864 | 15.4990635320 | 0.03264177 |

参考柔顺度分别为 26.011941998007167 与 15.494006013556433 N·mm, 仍来自 k=4 前三级外推. k=3 在两组材料下的偏差均随加密单调减小; 可压缩组最细网格偏差低于既定参考值比较尺度 0.003%, 不作精细定量解释. 图 5.5 已改为展示 p=1,2 与 k=2,3, 并生成 PNG/PDF 同步至论文目录; k=4 数据保留作参考. 未新增 k=3 拓扑优化或改动表 5.4.

成图命令:

```bash
python experiments/paper_topopt_huzhang/plot.py --case bearing-solid-h-convergence
```

## 3.5 图 5.6 与表 5.4 扩充: HZMFEM k=3 (2026-10-17 完成入库与统一再分析)

按用户确认方案, 在原六组优化之外新增两组 HZMFEM k=3, 保持网格、材料插值、过滤和 OC 参数不变. k=3 无需跳量稳定化, 角点松弛关闭. 两组结果已完成并经 sync_results.py 入库, 与原六组一起进行统一再分析.

此前代跑在第 17 步按用户要求停止; 用户随后重跑并完成两组优化. 两组分别经 302 与 351 步收敛, 末次设计变化为 0.00991281 与 0.00942352. 终态有限值及密度范围检查通过. 八组最大体积分数超出量为 2.2773e-5. 图 5.6 已生成两列四行并同步论文目录.

WSL 仓库根目录运行命令:

```bash
python experiments/paper_topopt_huzhang/run_bearing.py --analyzer huzhang --order 3
```

已检查 `summary.json` 的 `converged`、体积分数与平衡残差, 并执行以下入库、成图和再分析命令:

```bash
python experiments/paper_topopt_huzhang/sync_results.py bearing/nu-0.3 analyzer-huzhang__order-3
python experiments/paper_topopt_huzhang/sync_results.py bearing/nu-0.4999 analyzer-huzhang__order-3
python experiments/paper_topopt_huzhang/plot.py --case bearing-topologies
python experiments/paper_topopt_huzhang/table.py bearing-reanalysis
```

图 5.6 使用两列四行, 行序为 LFEM p=1 / HZMFEM k=2 / LFEM p=2 / HZMFEM k=3, 左右分别为 nu=0.3 / 0.4999. 表 5.4 已纳入八个最终设计, 各用 LFEM p=1,2 与 HZMFEM k=2,3,4 再分析; 保留 k=4 共同评价列, 其余列报告相对偏差. 40 个再分析值均为有限正值, 八组自身离散一致性检查全部通过, 最大相对差 3.82e-11. 新图表已同步草稿, 未重跑优化. 两组入库文件各 7 个, 内容哈希与输出目录逐一一致. 本次优化与再分析均记录 dirty 工作区溯源, 未标为干净提交下的可复现实验.

统一采用 k=4 评价时, 可压缩组四设计最大相对差为 1.0346%; 近不可压缩组 LFEM p=1 设计比 HZMFEM k=2、k=3 设计分别高 24.2879% 与 24.2693%, 而 p=2、k=2、k=3 三设计最大相对差仅 0.7948%. 两种 Hu--Zhang 设计彼此相差约 0.015%, 不据此宣称高阶优化设计更优. 相对差统一按最大最小值之差除以最小值计算.

## 4. 实验结论

- 八组优化均在 $245 \sim 382$ 步内达到设计变化容差，最终体积分数接近 $0.35$，最大超出量约为 $2.28\times10^{-5}$。
- 全实体测试中，近不可压缩组 LFEM $p=1$ 在 $120\times40$ 网格上的柔顺度偏差约为 $16.81\%$，加密至 $240\times80$ 后仍约为 $7.13\%$，表现出明显体积自锁。LFEM $p=2$ 与 HZMFEM $k=2,3$ 在 $120\times40$ 网格上的偏差分别约为 $0.11\%$、$0.15\%$ 和 $0.075\%$，并随加密减小。抗自锁结论限定于本算例；参考值比较尺度以下（$0.003\%$）的偏差不作精细定量解释。
- 采用 HZMFEM $k=4$ 统一再分析时，可压缩组四个设计的柔顺度最大相对差约为 $1.03\%$；近不可压缩组 LFEM $p=1$ 所得设计的柔顺度比 HZMFEM $k=2,3$ 所得设计均高约 $24.3\%$，而 LFEM $p=2$ 与 HZMFEM $k=2,3$ 三个设计的最大相对差仅为 $0.79\%$。最大相对差按最大值与最小值之差除以最小值计算，不能依据各组自身离散下较低的优化所得柔顺度判断设计更优。
- 综合全实体测试与统一再分析，本算例中 HZMFEM $k=2,3$ 均减轻了线性位移元自锁对优化结果的影响，所得设计与 LFEM $p=2$ 性能接近。提高 Hu--Zhang 阶次降低了全实体分析的柔顺度偏差，稳定化低阶格式与无需稳定化的高阶格式均获得了性能相近的优化设计。
- HZMFEM 不需要角点松弛（2.2 节）：按 Hu 与 Ma 的判据本模型没有牵引数据不一致的角点；载荷标记修正后，关闭松弛的 $k = 2$ 两组收敛到与开松弛相同的拓扑，柔顺度相差 $0.2\%$ 与 $1.0\%$（3.3 节）。两次运行之间还有代码版本与载荷标记的差别，差异不能归于松弛开关。
