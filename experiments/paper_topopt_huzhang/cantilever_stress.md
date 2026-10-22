# 二维悬臂梁局部应力约束体积最小化（论文 5.2.3 节）

## 1. 研究对象

论文当前展示 LFEM $p=3$ 与 HZMFEM $k=2,3$ 三组离散。LFEM $p=3$ 与 HZMFEM $k=2$ 的单元内应力多项式次数均为二次，不代表实际应力精度相同；$k=3$ 用于展示三次应力原生混合格式。模型、载荷、过滤与验收口径一致；论文统一采用乘子上限 $\lambda_{\max}=10^4$。实际运行中 LFEM 的上限为 10000，HZMFEM 为 3000，三组全程及最终更新后的最大乘子均小于 3000，上界截断均未激活；实际配置与上限等价性说明见第 3 节。

| | LFEM | HZMFEM |
| --- | --- | --- |
| 阶次 $k$ 的含义 | 位移插值阶 $p = k$ | 应力插值阶 $k$，位移阶 $k - 1$ |
| 应力读数 | 由位移梯度经本构得到，单元间不连续 | 应力自由度直接给出，$H(\mathrm{div})$ 协调 |
| $k = 2$ | 原生格式 | 跳量稳定化格式（$k \le d$），`stabilization_coefficient = "fixed"` |
| $k = 3, 4$ | 原生格式 | 原生格式，无需稳定化 |

模型为 [`CantileverMiddlePatch2d`](../../docs/problems/benchmarks/cantilever-middle-patch-2d.md)：左端固支，右边界中点处的竖直合力 $P$ 分布于宽 $l$ 的区段成为均布牵引。

## 2. 实验设置

实验脚本为 [`run_cantilever_stress.py`](run_cantilever_stress.py)，每组一次运行：

```bash
python experiments/paper_topopt_huzhang/run_cantilever_stress.py --analyzer lfem --order 3 --lambda-max 10000
python experiments/paper_topopt_huzhang/run_cantilever_stress.py --analyzer huzhang --order 2
python experiments/paper_topopt_huzhang/run_cantilever_stress.py --analyzer huzhang --order 3
```

### 2.1 模型参数

模型定义见模型文档，此处只列算例层取值。

| 参数 | 数值或设置 |
|---|---|
| 设计域 | $80\ \mathrm{mm} \times 40\ \mathrm{mm}$，平面应力：`L, H = 80.0, 40.0`，`PLANE_TYPE = "plane_stress"` |
| 载荷 | 右边界中点竖直合力 $P = -400$ N，分布于宽 $l = 6$ mm 的区段，强度 $\bar t = P/l$：`LOAD = -400.0`，`LOAD_WIDTH = 6.0`；离散方式见 2.2 节"载荷离散" |
| 实体材料 | $\nu = 0.25$；$E_0$ 物理参考值 $70000$ MPa，计算取 $1$ MPa：密度、Poisson 比与载荷固定且位移边界齐次时，弹性模量整体同比缩放不改变应力响应，$E_{\min}$ 与 $k = 2$ 格式的稳定化系数随之同比缩放，载荷与许用应力保持物理值：`E0 = 1.0`，`NU = 0.25` |
| 许用应力 | $\bar\sigma = 180$ MPa，保持物理值：`STRESS_LIMIT = 180.0` |

### 2.2 分析参数

| 参数 | 数值或设置 |
|---|---|
| 网格 | $80 \times 40$ 个矩形按棋盘式交替对角线剖成三角形，共 $6400$ 个单元，$h = 1$ mm：`TriangleMesh.from_checkerboard_box(box=problem.domain, nx=80, ny=40)` |
| 阶次 | 命令行 `--order k`，$k \in \{2, 3, 4\}$，传入 `space_degree=order` |
| 数值积分 | $2k + 2$ 阶：`integration_order=2 * order + 2` |
| LFEM 装配 | 标准逐单元装配：`ASSEMBLY_METHOD = "standard"` |
| HZMFEM 角点松弛 | 关闭：`USE_RELAXATION = False`。Hu 与 Ma（2021）4.1 节的角点松弛只针对两条牵引边在角点处牵引数据不一致的情形；本模型两个自由角点 $(80, 0)$、$(80, 40)$ 是牵引–牵引角点，相邻两边都是零牵引，满足一致性条件，常规节点插值即可精确施加；两个固支角点 $(0, 0)$、$(0, 40)$ 是位移–牵引角点，只由顶/底边写一次牵引，不出现约束冲突；载荷区端点在直边上，不是几何角点。与此前开松弛结果的对照见 3.3 节 |
| 线性求解器 | MUMPS 直接法：`SOLVER = "mumps"` |
| 载荷离散 | 区段常值牵引节点插值到右边界（沿 $y$ 均匀 $40$ 段）的连续 $\mathbb P_1$ 迹空间：载荷区内部节点取 $\bar t$，端点取左右极限的平均 $\bar t/2$，其余节点取 $0$；经 `traction=` 注入问题，LFEM 与 HZMFEM 看到同一个离散载荷：`build_problem()` 调用 `interpolate_patch_traction_to_p1_trace`，`summary.json` 记为 `load_discretization: \"p1_trace_interpolation\"`。端点为节点时合力与一阶矩精确，非零牵引只延伸到端点外一条边（$[16, 17]$、$[23, 24]$），该边所在单元在被动实体区内，再往外严格为零、不变号。不用 5.2.1 节的 $L^2$ 投影：投影在载荷区外带逐单元衰减约 $0.27$ 倍的振荡尾巴（$y = 16, 24$ 处约 $0.13\,\lvert\bar t\rvert$ 的反向牵引），落在被动实体区外的低密度边界单元上；HZMFEM 把这些单元的应力钉在牵引数据上，松弛约束在低密度处允许的应力又很小，优化器只能补材料或卡住。实测（2026-10-16，`deb2eb9`，不松弛、全受约束单元验收）：$k = 3$ 收敛但用了 $422$ 次 MMA 更新（论文侧 blis1 阶跃载荷、开松弛为 $315$），第 $271 \sim 419$ 步 $g$ 停在 $7.26 \times 10^{-3}$，卡住的是 $(79.67, 15.33)$、$(79.67, 24.67)$ 两个边界单元，终态被补到 $\rho = 0.61 / 0.38$；$k = 2$ 跑满 $1000$ 步不收敛，同两个单元 $\rho \approx 0.07$、$g \approx 9.2 \times 10^{-3}$，其余单元均在容差内（同一构型换阶跃载荷重算 $\max g = 1.6 \times 10^{-3}$）。两组运行目录保留在 `outputs/`，不入库 |
| 载荷区端点 | $(80, 17)$、$(80, 23)$ 必须落在网格节点上，节点插值才保持合力；由 `interpolate_patch_traction_to_p1_trace` 检查，不是节点即报错。被动实体区以这两点为圆心。注入 P1 迹载荷后 `uses_step_traction` 为 False，分析器构造时的 `check_traction_patch_alignment` 自动跳过 |
| 终态验收 | 优化结束后在最终物理密度上重解一次：记录相对平衡残差 `relative_equilibrium_residual`（LFEM 只计非 Dirichlet 自由度），并在判据集合（2.4 节）上复算相对超限量，记为 `max_relative_violation_accepted`，集合单元数记为 `accepted_cells`；`summary.json` 的 `converged` 是优化器停止准则与这次复验 $\le 5 \times 10^{-3}$ 的合取 |

### 2.3 应力约束与被动实体区

| 参数 | 数值或设置 |
|---|---|
| 约束形式 | 逐单元表观 von Mises 应力比的 $\varepsilon$ 松弛约束，$\varepsilon = 10^{-3}$：`EpsilonRelaxedStressFormulation(epsilon=EPSILON)`，`LagrangeStressConstraint` / `HuZhangStressConstraint` |
| 载荷端点被动实体区 | 以 `problem.traction_patch_endpoints` 两点为中心、半径 $r_{\mathrm{pad}} = 1.5$ mm $= l/4$，按单元重心判定，共 $16$ 个单元（设计域面积的 $0.25\%$）：`LOAD_PAD_RADIUS = 1.5`，`build_exemption_mask(...)` |
| 固支角点被动实体区 | 半径 $0$，即不处置：`SUPPORT_PAD_RADIUS = 0.0` |
| 掩码的两个用途 | 应力豁免：区内不施加局部应力约束，`exemption_mask=pad_mask`；实体保留：区内物理密度钉为 $1$、密度灵敏度置零，`Filter(..., passive_mask=pad_mask)` |

右端牵引在区段端点处跳变 $P/l$，这是载荷模型自带的混合边界间断点：载荷分布化只消除了集中力的 $r^{-1}$ 奇异，消不掉端点奇异，密度设计也消不掉。本算例的载荷竖直作用在竖直边上，是**切向**区段牵引；半平面受切向均布牵引时，沿边界的正应力在端点处对数奇异，$\sigma_{tt} \sim \frac{2\bar t}{\pi}\ln\frac{r_1}{r_2}$（Johnson, *Contact Mechanics*, 第 2 章）。门式框架（[`portal_frame.md`](portal_frame.md)）的载荷是法向区段牵引，端点应力有界，故那里不设被动实体区。

全实体（$\rho \equiv 1$）网格加密核实了这一点。下表为实体应力比在固定物理半径内的最大值：端点区每加密一档增加约 $0.14$，即 $\ln(1/h)$ 增长（理论参照 $(2\bar t/\pi)\ln 2/\bar\sigma = 0.163$，实测取 von Mises 与单元评价点最大值，故略低）；P1 $L^2$ 投影只把起点降低 $0.002 \sim 0.010$，不改变增长；远场按 $O(h)$ 收敛。

| 区域 | 离散 / 载荷 | $h = 1$ | $h = 0.5$ | $h = 0.25$ | 逐级增量 |
|---|---|---|---|---|---|
| 载荷端点 $r \le 1.5$ | LFEM $p = 2$ / 阶跃 | 0.6131 | 0.7595 | 0.9023 | $+0.146,\ +0.143$ |
| | LFEM $p = 2$ / P1 $L^2$ 投影 | 0.6054 | 0.7544 | 0.8981 | $+0.149,\ +0.144$ |
| | HZMFEM $k = 2$ / 阶跃 | 0.6142 | 0.7477 | 0.8879 | $+0.134,\ +0.140$ |
| | HZMFEM $k = 2$ / P1 $L^2$ 投影 | 0.6124 | 0.7380 | 0.8799 | $+0.126,\ +0.142$ |
| 远场 $20 \le x \le 60$ | 四组相同 | 0.48436 | 0.49104 | 0.49440 | $+0.0067,\ +0.0034$ |
| 固支角点 $r \le 1.5$ | LFEM $p = 2$ | 0.8652 | 0.9915 | 1.1424 | $+0.126,\ +0.151$ |
| | HZMFEM $k = 2$ | 0.8313 | 0.9584 | 1.1065 | $+0.127,\ +0.148$ |

$h = 1$ 时端点区的全实体读数只有 $0.61$，单看论文网格并不危险；设被动实体区的依据是上面这一条与网格无关的论证：载荷总要通过一块实体传入结构，该处应力随 $h \to 0$ 无界，约束不可能对一切网格满足；它是边界条件的性质而非设计的缺陷，既约束不了，也不该让优化器去动它，故划为被动实体区（论文记作 $\Omega_{\mathrm{pad}}$）。固支角点同样奇异（增量逐档变大，比对数更强），但优化器可以挖除该处材料绕开它，不必处置。两条分析链施加同一份掩码，保证对照在同一验收区域上进行。

豁免与实体保留**必须成对施加**。2026-09-16 的对照（Hu--Zhang $k = 2$）：只豁免不保留时从 $1000$ 步不收敛变为 $536$ 步收敛，但优化器发现该邻域不再受约束，把原本被应力约束逼着保持 $\rho \approx 0.98$ 的单元减到 $0.67$ 以换体积，实体应力比从 $0.68$ 涨到 $1.45$，`max_solid_stress_ratio_solid_region` 由稳定的 $1.01 \sim 1.03$ 跳到 $1.452$，豁免掩盖了真实过应力。

实现上的四条边界：

- 实体保留施加在**过滤与投影之后**（`Filter._enforce_passive_solid`）。只钉设计变量不够：$r_{\min} = 6 \gg h = 1$，宽过滤下保留单元的物理密度仍由邻域决定，达不到满密度。
- $\tilde\rho$ 在被动实体区内与设计变量无关，故 $\partial\tilde\rho / \partial z = 0$，过滤器在委托给策略之前先把这些行的密度灵敏度清零。
- 豁免只改约束集合的成员，不改 AL 的归一化基数：`fun` 返回张量的形状不变，带与不带被动实体区的两次运行罚项量级严格可比。豁免单元的约束值取严格可行的常值 $-1$，对目标与灵敏度的贡献恒为 $0$；真实值仍可由 `compute_unexempted_constraint` 取回，`summary.json` 单列 `max_constraint_pad` 与 `max_solid_stress_ratio_pad`，受约束区域单列为 `max_solid_stress_ratio_constrained`，被动实体区掩盖了多大的应力可直接读出。
- 半径记在各运行目录的 `config.json` 与 `summary.json` 中；当前三组的确切目录见第 3 节，其中 LFEM p=3 目录含乘子上限后缀。

核对走 `plot.py discretization-probe`：构型冻结后在自身离散下重解，不施加豁免，被动实体区的真实读数一并取回。

### 2.4 优化参数

| 参数 | 数值或设置 |
|---|---|
| 优化目标 | 最小化体积分数：`VolumeObjective` |
| 约束 | 2.3 节的局部应力约束，经增广 Lagrange 函数并入目标：`AugmentedLagrangianObjective` |
| 材料插值 | 修正 SIMP，只插值 Young 模量：$E(\rho) = E_{\min} + \rho^{q}(E_0 - E_{\min})$，$q = 3.5$，$E_{\min} = 10^{-9}$（随 $E_0$ 同比缩放）：`interpolation_method="msimp"`，`PENALTY = 3.5`，`EMIN = 1.0e-9`，`target_variables=["E"]` |
| 初始密度 | 均匀 $0.5$，满密度启动会导致 ALM 发散：`INITIAL_DENSITY = 0.5` |
| 密度过滤 | 线性密度过滤，$r_{\min} = 6$ mm：`Filter(filter_type="projection", rmin=FILTER_RADIUS, filter_q=3, ...)` |
| 投影 | tanh 投影，$\eta = 0.5$，$\beta$ 从 $1$ 起每 $5$ 个外层步加 $1$，直至 $10$：`PROJECTION = {"projection_type": "tanh", "beta": 1.0, "beta_max": 10.0, "continuation_iter": 5, "beta_increment": 1.0, "continuation_strategy": "additive"}` |
| 优化器 | AL-MMA，外层更新乘子与罚参数，内层做 MMA：`ALMMMAOptimizer` |
| 罚参数 | $\mu_0 = 50$，每个外层步无条件更新 $\mu \leftarrow \min(1.1\,\mu,\ 10^4)$：`MU_0 = 50.0`，`ALPHA = 1.1`，`MU_MAX = 1.0e4`，`mu_update_rule = "unconditional"`（默认） |
| 乘子 | 带安全阈的投影更新 $\lambda \leftarrow \min(\lambda + \mu h,\ \lambda_{\max})$，$h = \max(g, -\lambda/\mu)$，初值 $0$，$\lambda_{\max} = 10^4$ 为论文统一设置；实际 LFEM p=3 采用 `--lambda-max 10000`，HZMFEM k=2,3 采用原默认上限 `LAMBDA_MAX = 3000.0`，未激活上限的等价性见第 3 节。初值 `LAMBDA_0 = 0.0`，更新接口 `AugmentedLagrangianObjective.update_multipliers` |
| MMA | 移动限 $0.15$，渐近线最小间距 $10^{-4}$：`MOVE_LIMIT = 0.15`，`ASYMPTOTE_MIN_DISTANCE = 1.0e-4` |
| 迭代上限 | 外层最多 $200$ 步，每外层步最多 $5$ 次内层 MMA 更新；达到上限未满足 C3 时以 `max-al-iterations-reached` 终止：`MAX_AL_ITERATIONS = 200`，`MMA_ITERS_PER_AL = 5` |
| 内层停止 | `legacy` 规则：每次 MMA 更新并重解后，若相邻两次更新间设计变量的平均绝对变化 $< 2 \times 10^{-3}$ 且判据集合上 $\delta_g \le 5 \times 10^{-3}$，提前进入外层更新，否则做满 $5$ 次：`INNER_STOP_RULE = "legacy"`。`INNER_RELATIVE_TOLERANCE`、`INNER_ABSOLUTE_TOLERANCE` 只对 `projected_gradient` 规则生效，本算例不用 |
| 停止准则 | 每个外层步结束时判定：C0 连续化终止，$\beta$ 已达 $\beta_{\max}$ 且本步未发生 $\beta$ 更新；C1 设计稳定，外层步首末设计变量的平均绝对变化 $\delta_\rho < 2 \times 10^{-3}$（发生 $\beta$ 更新的步强制不满足）；C2 松弛可行，判据集合上相对超限量 $\delta_g \le 5 \times 10^{-3}$；C3 持续，C0 至 C2 连续 $3$ 个外层步同时满足，任一步不满足计数清零。KKT 验收（C4）关闭：`CHANGE_TOLERANCE = 2.0e-3`，`CHANGE_MEASURE = "mean"`，`STRESS_TOLERANCE = 5.0e-3`，`HOLD_STEPS = 3`，`kkt_acceptance_enabled = False`（默认） |
| 判据集合 | 被动实体区外的全部受约束单元，共 $6384$ 个，即论文 4.6.2 节的 $\mathcal E_{\mathrm{acc}}$；内层提前退出、外层 C2 与终态验收共用：`ACCEPTANCE_SOLID_THRESHOLD = None`。取 $(0, 1)$ 内的值时只验收 $\tilde\rho \ge$ 该值的单元，此前的运行用过 $0.5$，超容差的单元全在 $0.1 \le \tilde\rho < 0.5$ 的灰度区。"实体区最大应力比"只作诊断，仍按 $\tilde\rho \ge 0.5$ 统计：`SOLID_REPORT_THRESHOLD = 0.5` |

## 3. 当前论文结果（2026-10-17 更新）

最新三组均采用 P1 节点插值牵引、全部 6384 个非豁免单元验收，HZMFEM 不松弛角点。
以下运行目录在 `results/cantilever-middle-2d-stress/` 下：

- LFEM p=3：`analyzer-lfem__order-3__lambda-max-10000.0`。
- HZMFEM k=2：`analyzer-huzhang__order-2`。
- HZMFEM k=3：`analyzer-huzhang__order-3`。

旧 LFEM p=3 无后缀目录不作为当前图表来源；替换前的 HZMFEM 入库目录已备份到
`outputs/archive-results-20261017/cantilever-middle-2d-stress/`。原 outputs 运行及旧论文图件保留。
三组运行均含 dirty 工作区溯源，原始 config 和 summary 如实保存，不重写乘子上限或版本戳记。

表 5.5 对应的优化终态及牵引连续性统计。下表额外保留实体带单元数供实验追溯，论文表 5.5 不展示该列。

| 离散 | MMA 更新次数 | 体积分数 (%) | 最大约束值 | 实体带单元数 | 最大归一化牵引跳量 |
|---|---:|---:|---:|---:|---:|
| LFEM $p=3$ | 249 | 34.4807 | 4.711311e-03 | 1780 | 1.180699e-01 |
| HZMFEM $k=2$ | 314 | 34.8770 | 4.903359e-03 | 1765 | 1.572811e-16 |
| HZMFEM $k=3$ | 381 | 34.9713 | 4.850877e-03 | 1775 | 1.805537e-16 |

全程及最后外层更新后的最大乘子分别为 408.60、1129.64、2253.97，均未触及 3000。

论文正文统一采用 $\lambda_{\max}=10^4$。实际运行中，LFEM p=3 的上限为 10000，HZMFEM k=2,3 的上限为 3000；原始配置与运行记录保留实际值。由于两组 HZMFEM 全程及最终乘子更新均未触发 3000 的上界截断，在其余设置及初始状态相同的条件下，将上限放宽至 10000 不改变乘子更新及后续迭代，因此已有结果适用于正文的统一上限设置。此处为未激活上限的等价性说明，未重新运行优化。
三组终态冻结再分析均复现 summary 的最大约束值；HZMFEM 相对牵引跳量验收通过。
数值及输入密度摘要见 `postprocess/comparison_manifest.json`；牵引跳量统计限于密度大于 0.9 且位于被动区外的单元，相邻单元不作密度筛选。

### 3.1 论文插图与表格

当前论文图表对应关系如下，图件位于 `results/figures/`，表格位于 `results/tables/`。

| 编号 | 文件 | 内容 |
|---|---|---|
| 图 5.8 | `stress_cantilever_comparison_topologies.png` | 三行两列的物理密度与表观应力比 |
| 图 5.9 | `stress_cantilever_history_principal.png/pdf` | 上排为体积分数与最大约束值历史，下排为三组终态主应力散点 |
| 表 5.5 | `table5_5.md` | MMA 更新次数、体积分数、最大约束值及最大归一化牵引跳量 |

图 5.8 与表 5.5 的生成入口为：

```bash
python experiments/paper_topopt_huzhang/plot.py --case stress-cantilever-comparison
```

该入口对三组已入库终态做冻结再分析，不启动优化；同时生成的独立收敛历史图与牵引跳量图作为辅助产物保留，不再对应正文图号。图 5.9 的生成入口为：

```bash
python experiments/paper_topopt_huzhang/plot.py --case stress-cantilever-history-principal
```

该入口读取已有 `history.json` 与 `postprocess/principal_stress_comparison/` 中经来源校验的散点数据，仅重新排版，不执行状态求解或优化。散点数据的冻结导出入口为 `plot.py --case stress-cantilever-principal`。

牵引跳量在正文仅保留定义、统计范围与表 5.5，不再单列插图。旧的 `stress-cubic-*`、`stress-hz-orders-topologies`、`stress-traction-jump` 入口，以及独立历史图、散点图和牵引跳量图保留供历史追溯；当前图号以上表为准。

### 3.2 产物文件

每组运行写 `outputs/cantilever-middle-2d-stress/<run>/`，确认后经 `sync_results.py` 固化到 `results/cantilever-middle-2d-stress/<run>/`，当前论文使用的运行名如下，旧批次无后缀的 LFEM p=3 及其他阶次目录不作为当前图表来源：

```text
analyzer-lfem__order-3__lambda-max-10000.0/
analyzer-huzhang__order-2/
analyzer-huzhang__order-3/
```

| 文件 | 内容 |
|---|---|
| `config.json` | 开跑前记录的全部设置：模型、应力约束、离散（含角点松弛开关）、优化参数与代码版本戳记 |
| `history.json` | 逐 MMA 更新的记录：步号、设计变化量、耗时与标量历史（目标、体积分数、最大约束等） |
| `outer_history.json` | 逐外层 AL 步的内层诊断（`ALMMMAOptimizer.outer_history`：$\beta$、$\mu$、内层残差与目标、内层步数、是否接受、外层首末设计变化、最大约束与超限量），只在 `inner_stop_rule = "projected_gradient"` 时记录；本算例用 `legacy`，三组均为空列表 `[]`，`summary.json` 的 `last_inner_diagnostics` 同理为 `null` |
| `summary.json` | 终态验收：体积分数、迭代数、终止原因、各口径的最大应力比与超限量、被动实体区诊断、平衡残差 |
| `final_optimizer_state.npz` | AL 终态：设计变量 `design`、物理密度 `density`、乘子 `lamb`、罚参数 `mu`，供冻结复算与内层诊断，不支持精确断点续算 |
| `displacement_final.npy`、`stress_final.npy` | 终态求解的位移自由度向量；`stress_final.npy` 只有 Hu--Zhang 写出，是应力自由度向量，编号随 `use_relaxation` 而定（本算例不松弛，即原始 Hu--Zhang 空间的编号） |
| `density_initial.vtu` | 第 0 帧的副本：初始设计经过滤与实体保留处理后的物理密度，只含 `density`，用来查看被动实体区 |
| `density_final.vtu` | 最终构型：`density` 与单元最大表观应力比 `von_mises_normalized`，冻结重分析与插图的输入 |

逐步帧 `iterations/iter_NNNN.vtu` 与 `evolution.pvd` 在优化结束后由优化器记录的历史一次写入运行目录并留在原处：第 0 帧是初始构型（只含 `density`），第 1 帧到末帧每次 MMA 更新一帧，字段由 `VTU_FIELDS` 决定。脚本不迁移帧，需要在 Windows ParaView 中查看时手动拷到 `C:\workspace\soptx-results\paper_topopt_huzhang\cantilever-middle-2d-stress\<run>\`；`sync_results.py` 不拷帧。

当前三组的文件清单和目录以上述第 3 节为准。冻结再分析的数值、输入密度摘要与乘子统计见 `postprocess/comparison_manifest.json`；逐单元约束、应力比与牵引跳量见 `postprocess/discretization_probe/`；主应力散点数组与来源清单见 `postprocess/principal_stress_comparison/`。这些数据支撑当前图 5.8、图 5.9 与表 5.5，旧批次数据只用于历史追溯。

### 3.3 历史角点松弛对照（旧载荷、旧验收口径）

以下为历史记录，不代表当前 results 中的三组来源；其中 2026-10-15 的关闭松弛结果位于旧 outputs 目录。HZMFEM 此前曾以开松弛跑过（四个几何角点全部松弛），随后关闭松弛重跑。两组的代码版本不同：关松弛为 `0218413`，开松弛为 `5a8ab5a`（$k = 2$）与 `ad95594`（$k = 3, 4$，FEALPy 移植前），故下表只说明关闭松弛后结论不变，差异不能全部归于松弛开关。开松弛数据取自 `git 0218413` 版本的 `results/cantilever-middle-2d-stress/analyzer-huzhang__order-{2,3,4}/`（`summary.json`、`final_optimizer_state.npz`），可用 `git show` 取回。

| $k$ | 松弛 | 代码 | MMA 更新 | 体积分数 | $g_{\max}$（判据集合） | $g_{\max}$（全域） | 实体区最大应力比 | 被动区最大应力比 | 平衡残差 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 关 | `0218413` | 257 | 0.3504 | $3.17 \times 10^{-3}$ | $9.56 \times 10^{-3}$ | 1.041 | 0.627 | $3.34 \times 10^{-11}$ |
|  | 开 | `5a8ab5a` | 253 | 0.3486 | $2.79 \times 10^{-3}$ | $7.28 \times 10^{-3}$ | 1.027 | 0.638 | $2.78 \times 10^{-11}$ |
| 3 | 关 | `0218413` | 256 | 0.3513 | $3.30 \times 10^{-3}$ | $6.94 \times 10^{-3}$ | 1.031 | 0.646 | $1.45 \times 10^{-8}$ |
|  | 开 | `ad95594` | 256 | 0.3501 | $2.82 \times 10^{-3}$ | $4.80 \times 10^{-3}$ | 1.022 | 0.648 | $1.62 \times 10^{-8}$ |
| 4 | 关 | `0218413` | 252 | 0.3518 | $2.80 \times 10^{-3}$ | $1.46 \times 10^{-2}$ | 1.039 | 0.648 | $4.66 \times 10^{-7}$ |
|  | 开 | `ad95594` | 258 | 0.3530 | $4.92 \times 10^{-3}$ | $1.25 \times 10^{-2}$ | 1.061 | 0.643 | $4.22 \times 10^{-8}$ |

终态物理密度的差异：

| $k$ | 体积分数差（关 − 开） | $\tilde\rho = 0.5$ 两侧翻转的单元 | 其中固支角点 $5$ mm 内 | 平均 $\lvert\Delta\tilde\rho\rvert$：固支角点 $5$ mm 内 / 其余 |
|---|---|---|---|---|
| 2 | +0.0017 | 99（1.55%） | 0 | 0.0009 / 0.0158 |
| 3 | +0.0012 | 73（1.14%） | 0 | 0.0015 / 0.0125 |
| 4 | -0.0011 | 59（0.92%） | 0 | 0.0006 / 0.0109 |

三组都收敛到相同拓扑，差异是杆件边界的小幅平移，体积分数差有正有负；固支角点是位移–牵引角点，也是应力奇点，若松弛在此起作用，差异应集中于该处，实测该处无翻转单元、密度差比其余区域小一个量级。

## 4. 当前实验结论

- LFEM p=3 与 HZMFEM k=2,3 均通过全部非豁免单元验收，主要承载布局相近。
- 三组更新次数为 249、314、381；本组结果不支持 HZMFEM 优化收敛更快的结论。
- 体积分数分别为 34.4807%、34.8770%、34.9713%；各自在自身离散下验收，不能直接据此排序设计性能。
- LFEM p=3 的实体带最大归一化牵引跳量约为 0.1181，HZMFEM k=2,3 处于舍入误差量级。这验证混合应力空间的牵引连续性，不等价于应力误差估计，也不单独证明高阶精度优势。

## LFEM p=2 乘子上限单因素试算（2026-10-17，已完成，非当前正文组合）

基准运行 `outputs/cantilever-middle-2d-stress/analyzer-lfem__order-2/` 保留不变。
该运行在 1000 次 MMA 更新后未通过全非豁免单元验收，最大约束违反量为
`0.007947683853317433`；两个持续超限单元的乘子达到上限 `3000`。

本次只将乘子上限提高至 `10000`，其余优化与离散参数保持不变，从相同初始设计开始，
不续接基准终态。运行命令：

```bash
python experiments/paper_topopt_huzhang/run_cantilever_stress.py --analyzer lfem --order 2 --lambda-max 10000
```

试算写入独立目录
`outputs/cantilever-middle-2d-stress/analyzer-lfem__order-2__lambda-max-10000.0/`，
目录已存在时仍拒绝覆盖。实际乘子上限同步记录到启动摘要、`config.json` 和 `summary.json`。
验收仍要求全部 6384 个非豁免单元的最大约束违反量不超过 `0.005`，并满足完整收敛条件。
关注单元 1776、3363 的违反量与乘子变化，同时检查是否有其他单元超限。
此项为参数诊断试算，不与基准结果混称为相同参数下的比较。上限 10000 的运行已在 940 次 MMA 更新后收敛，体积分数 35.3360%，最大约束值 0.00498743，最大乘子 9815.31；该组不进入当前正文三组比较。

## 历史图表调整记录

以下保留各次调整时的图号与产物说明，不代表当前图表目录；当前悬臂梁图表以第 3.1 节为准。

### 2026-10-17: 恢复三组主应力散点图

- 新入口 `python experiments/paper_topopt_huzhang/plot.py --case stress-cantilever-principal`，使用当前 LFEM p=3 (上限 10000) 与 HZMFEM k=2,3 入库终态，冻结再分析而不重跑优化。
- 图 5.9 使用全部非豁免单元的形心表观主应力，一行三列、统一坐标及色标。红色椭圆为表观 von Mises 许用应力参考边界，不等同于 epsilon 松弛约束的验收边界。
- 数据与源密度摘要保存在 `postprocess/principal_stress_comparison/`；图件为 `stress_cantilever_comparison_principal.png/pdf`。原图 5.9--5.13 顺延至 5.10--5.14，门式框架散点说明引用新图 5.9。此前记录中的图号保留历史含义。


### 2026-10-17: 合并优化历史与主应力图

当前图 5.9 由 `plot.py --case stress-cantilever-history-principal` 生成，上排为体积和约束历史，下排为 LFEM p=3 与 HZMFEM k=2,3 的终态主应力散点。复用已有历史及已核验的散点数据，不重新求解；输出 `stress_cantilever_history_principal.png/pdf`。原独立图保留为历史产物，正文不再引用；牵引跳量现为图 5.10，门式框架为图 5.11--5.13。此前记录中的编号按历史版本理解。


### 2026-10-17: 牵引跳量仅保留表格

正文删除原图 5.10 的牵引跳量空间分布，保留定义与统计范围，以表 5.5 展示量化证据；表格删除实体带单元数列，底层统计及原图件仍保留。门式框架图号调整为 5.10--5.12。生成表格的代码同步更新，未重新计算。
