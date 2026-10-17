# 二维悬臂梁局部应力约束体积最小化（论文 5.2.3 节）

## 1. 研究对象

在局部应力约束下最小化体积，考察 Lagrange 位移元（LFEM）$p = 2, 3, 4$ 与 Hu--Zhang 混合元（HZMFEM）$k = 2, 3, 4$ 共六组离散下的优化过程、最终构型与约束满足情况。六组共用同一设计域、材料插值、应力约束模型与优化参数，只有分析链与阶次不同。

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
for m in lfem huzhang; do for k in 2 3 4; do
    python experiments/paper_topopt_huzhang/run_cantilever_stress.py --analyzer $m --order $k; done; done
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
- 半径记在各运行目录的 `summary.json` 中，目录名只含分析链与阶次。

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
| 乘子 | 带安全阈的投影更新 $\lambda \leftarrow \min(\lambda + \mu h,\ \lambda_{\max})$，$h = \max(g, -\lambda/\mu)$，初值 $0$，$\lambda_{\max} = 3000$：`LAMBDA_0 = 0.0`，`LAMBDA_MAX = 3000.0`，`AugmentedLagrangianObjective.update_multipliers` |
| MMA | 移动限 $0.15$，渐近线最小间距 $10^{-4}$：`MOVE_LIMIT = 0.15`，`ASYMPTOTE_MIN_DISTANCE = 1.0e-4` |
| 迭代上限 | 外层最多 $200$ 步，每外层步最多 $5$ 次内层 MMA 更新；达到上限未满足 C3 时以 `max-al-iterations-reached` 终止：`MAX_AL_ITERATIONS = 200`，`MMA_ITERS_PER_AL = 5` |
| 内层停止 | `legacy` 规则：每次 MMA 更新并重解后，若相邻两次更新间设计变量的平均绝对变化 $< 2 \times 10^{-3}$ 且判据集合上 $\delta_g \le 5 \times 10^{-3}$，提前进入外层更新，否则做满 $5$ 次：`INNER_STOP_RULE = "legacy"`。`INNER_RELATIVE_TOLERANCE`、`INNER_ABSOLUTE_TOLERANCE` 只对 `projected_gradient` 规则生效，本算例不用 |
| 停止准则 | 每个外层步结束时判定：C0 连续化终止，$\beta$ 已达 $\beta_{\max}$ 且本步未发生 $\beta$ 更新；C1 设计稳定，外层步首末设计变量的平均绝对变化 $\delta_\rho < 2 \times 10^{-3}$（发生 $\beta$ 更新的步强制不满足）；C2 松弛可行，判据集合上相对超限量 $\delta_g \le 5 \times 10^{-3}$；C3 持续，C0 至 C2 连续 $3$ 个外层步同时满足，任一步不满足计数清零。KKT 验收（C4）关闭：`CHANGE_TOLERANCE = 2.0e-3`，`CHANGE_MEASURE = "mean"`，`STRESS_TOLERANCE = 5.0e-3`，`HOLD_STEPS = 3`，`kkt_acceptance_enabled = False`（默认） |
| 判据集合 | 被动实体区外的全部受约束单元，共 $6384$ 个，即论文 4.6.2 节的 $\mathcal E_{\mathrm{acc}}$；内层提前退出、外层 C2 与终态验收共用：`ACCEPTANCE_SOLID_THRESHOLD = None`。取 $(0, 1)$ 内的值时只验收 $\tilde\rho \ge$ 该值的单元，此前的运行用过 $0.5$，超容差的单元全在 $0.1 \le \tilde\rho < 0.5$ 的灰度区。"实体区最大应力比"只作诊断，仍按 $\tilde\rho \ge 0.5$ 统计：`SOLID_REPORT_THRESHOLD = 0.5` |

## 3. 结果

六组均按停止准则收敛（`termination_reason: criterion-met`）。下表取自各运行目录的 `summary.json`；MMA 更新次数为 `optimization_iterations`，$g_{\max}$ 为相对超限量，应力比为未加权的实体材料 von Mises 应力与 $\bar\sigma$ 之比。

| 分析链 | $k$ | MMA 更新 | 体积分数 | $g_{\max}$（判据集合） | $g_{\max}$（全域） | 实体区最大应力比 | 被动区最大应力比 | 平衡残差 |
|---|---|---|---|---|---|---|---|---|
| LFEM | 2 | 245 | 0.3528 | $3.75 \times 10^{-3}$ | $6.77 \times 10^{-3}$ | 1.026 | 0.648 | $7.1 \times 10^{-12}$ |
| LFEM | 3 | 248 | 0.3439 | $3.92 \times 10^{-3}$ | $6.36 \times 10^{-3}$ | 1.020 | 0.633 | $2.0 \times 10^{-11}$ |
| LFEM | 4 | 256 | 0.3495 | $4.70 \times 10^{-3}$ | $1.10 \times 10^{-2}$ | 1.028 | 0.642 | $4.2 \times 10^{-11}$ |
| HZMFEM | 2 | 257 | 0.3504 | $3.17 \times 10^{-3}$ | $9.56 \times 10^{-3}$ | 1.041 | 0.627 | $3.3 \times 10^{-11}$ |
| HZMFEM | 3 | 256 | 0.3513 | $3.30 \times 10^{-3}$ | $6.94 \times 10^{-3}$ | 1.031 | 0.646 | $1.4 \times 10^{-8}$ |
| HZMFEM | 4 | 252 | 0.3518 | $2.80 \times 10^{-3}$ | $1.46 \times 10^{-2}$ | 1.039 | 0.648 | $4.7 \times 10^{-7}$ |

列的来源：体积分数 `volume_fraction`（含被动实体区）；判据集合 `max_relative_violation_solid_region`，全域 `max_relative_violation`（含 $\tilde\rho < 0.5$ 的单元，不作验收）；实体区最大应力比 `max_solid_stress_ratio_solid_region`；被动区最大应力比 `max_solid_stress_ratio_pad`（冻结重分析取回的真实读数，均低于 $1$，被动实体区没有掩盖过应力）；平衡残差 `relative_equilibrium_residual`。

### 3.1 论文插图

| 图 | 产物 case | 内容 | 数据来源 |
|---|---|---|---|
| 5.8 | `stress-cubic-topologies` | $k = 3$ 两种离散的最终构型与表观应力比 | `plot.py discretization-probe` 导出的 `postprocess/discretization_probe/*fields.npz` |
| 5.9 | `stress-cubic-convergence` | $k = 3$ 的收敛历史与主应力空间内的单元应力分布 | `history.json`、`summary.json`；`plot.py export --run lfem-k3 --run huzhang-k3` |
| 5.10 | `stress-hz-orders-topologies` | Hu--Zhang $k = 2$（稳定化）与 $k = 4$（原生）的构型与应力比 | 同图 5.8 |
| 5.11 | `stress-traction-jump` | 冻结构型实体带上逐单元牵引跳量的空间分布与累积分布 | 六份 `fields.npz` |

成图命令 `python experiments/paper_topopt_huzhang/plot.py <case>`，缺数据时入口会提示先运行对应的重分析动词。

### 3.2 产物文件

每组运行写 `outputs/cantilever-middle-2d-stress/<run>/`，确认后经 `sync_results.py` 固化到 `results/cantilever-middle-2d-stress/<run>/`，运行名 `analyzer-{lfem,huzhang}__order-{2,3,4}`。

| 文件 | 内容 |
|---|---|
| `config.json` | 开跑前记录的全部设置：模型、应力约束、离散（含角点松弛开关）、优化参数与代码版本戳记 |
| `history.json` | 逐 MMA 更新的记录：步号、设计变化量、耗时与标量历史（目标、体积分数、最大约束等） |
| `outer_history.json` | 逐外层 AL 步的内层诊断（`ALMMMAOptimizer.outer_history`：$\beta$、$\mu$、内层残差与目标、内层步数、是否接受、外层首末设计变化、最大约束与超限量），只在 `inner_stop_rule = "projected_gradient"` 时记录；本算例用 `legacy`，六组均为空列表 `[]`，`summary.json` 的 `last_inner_diagnostics` 同理为 `null` |
| `summary.json` | 终态验收：体积分数、迭代数、终止原因、各口径的最大应力比与超限量、被动实体区诊断、平衡残差 |
| `final_optimizer_state.npz` | AL 终态：设计变量 `design`、物理密度 `density`、乘子 `lamb`、罚参数 `mu`，供冻结复算与内层诊断，不支持精确断点续算 |
| `displacement_final.npy`、`stress_final.npy` | 终态求解的位移自由度向量；`stress_final.npy` 只有 Hu--Zhang 写出，是应力自由度向量，编号随 `use_relaxation` 而定（本算例不松弛，即原始 Hu--Zhang 空间的编号） |
| `density_initial.vtu` | 第 0 帧的副本：初始设计经过滤与实体保留处理后的物理密度，只含 `density`，用来查看被动实体区 |
| `density_final.vtu` | 最终构型：`density` 与单元最大表观应力比 `von_mises_normalized`，冻结重分析与插图的输入 |

逐步帧 `iterations/iter_NNNN.vtu` 与 `evolution.pvd` 在优化结束后由优化器记录的历史一次写入运行目录并留在原处：第 0 帧是初始构型（只含 `density`），第 1 帧到末帧每次 MMA 更新一帧，字段由 `VTU_FIELDS` 决定。脚本不迁移帧，需要在 Windows ParaView 中查看时手动拷到 `C:\workspace\soptx-results\paper_topopt_huzhang\cantilever-middle-2d-stress\<run>\`；`sync_results.py` 不拷帧。

入库的 LFEM 三组是 2026-09-18 与 2026-10-07 的运行：`density_initial.vtu` 由当时查看目录的第 0 帧回填（与常量重建逐位相同），没有 `config.json`、`displacement_final.npy`。HZMFEM 三组是 2026-10-15 用 `git 0218413` 关闭角点松弛重跑的，产物齐全；此前开松弛的三组（`5a8ab5a`、`ad95594`）已被替换，可从 git 历史取回。

### 3.3 角点松弛对照

HZMFEM 三组此前曾以开松弛跑过（四个几何角点全部松弛），2026-10-15 关闭松弛用当前代码重跑并替换入库。两组的代码版本不同：关松弛为 `0218413`，开松弛为 `5a8ab5a`（$k = 2$）与 `ad95594`（$k = 3, 4$，FEALPy 移植前），故下表只说明关闭松弛后结论不变，差异不能全部归于松弛开关。开松弛数据取自 `git 0218413` 版本的 `results/cantilever-middle-2d-stress/analyzer-huzhang__order-{2,3,4}/`（`summary.json`、`final_optimizer_state.npz`），可用 `git show` 取回。

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

## 4. 实验结论

- 六组离散都在 $245 \sim 257$ 次 MMA 更新内按同一准则收敛，最终体积分数在 $0.344 \sim 0.352$ 之间，判据集合上的相对超限量均在 $5 \times 10^{-3}$ 容差内。
- 实体区最大应力比在 $1.02 \sim 1.04$，与 $\varepsilon$ 松弛和 $\delta_g$ 容差的量级一致；被动实体区内冻结重分析得到的真实应力比为 $0.63 \sim 0.65$，豁免没有掩盖过应力。
- Hu--Zhang 的应力自由度 $H(\mathrm{div})$ 协调，冻结构型实体带上的单元间牵引跳量整幅为零，LFEM 则非零（图 5.11）；两类离散的构型与约束满足情况在同一验收口径下可比。
- HZMFEM 不需要角点松弛（2.2 节）：按 Hu 与 Ma 的判据本模型没有牵引数据不一致的角点；关闭松弛后 $k = 2, 3, 4$ 收敛到与开松弛相同的拓扑，体积分数相差不超过 $0.002$，固支角点附近无差异（3.3 节）。两次运行的代码版本也不同，差异不能全部归于松弛开关。
