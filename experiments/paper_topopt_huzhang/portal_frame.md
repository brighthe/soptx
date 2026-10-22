# 二维门式框架局部应力约束体积最小化（论文 5.2.4 节）

## 1. 研究对象

在非矩形设计域与 Gmsh 非结构三角形网格上做局部应力约束下的体积最小化，考察所提框架对含斜边、凹角设计域的推广。论文只报 Hu--Zhang 混合元（HZMFEM）$k = 3$（二维中无需稳定化的最低阶，$k \ge d + 1$）；阶次对比已由 5.2.3 节给出，本节不重复。LFEM 运行只作诊断，见第 4 节。

模型为 [`PortalFramePatch2d`](../../docs/problems/benchmarks/portal-frame-patch-2d.md)：$L \times H$ 矩形去掉底边中央三角形缺口，左支座段固定铰、右支座段滚动铰，顶边中点处的竖直合力 $P$ 分布于宽 $l$ 的区段成为均布牵引。几何与载荷参数取自 Giraldo-Londoño 与 Paulino（2021，PolyStress）图 7 与其 `PortalDomain.m`；载荷与材料按自洽单位制取值，不声称复现其数值。

## 2. 实验设置

实验脚本为 [`run_portal_frame.py`](run_portal_frame.py)，论文组的命令：

```bash
python experiments/paper_topopt_huzhang/run_portal_frame.py --analyzer huzhang --order 3 \
  --load-discretization p1-interpolation --load-pad-radius 0 --lambda-max 10000
```

### 2.1 模型参数

模型定义见模型文档，此处只列算例层取值。

| 参数 | 数值或设置 |
|---|---|
| 设计域 | $12\ \mathrm{mm} \times 6\ \mathrm{mm}$ 矩形去掉底边中央三角形缺口，缺口顶点 $(b, 0)$、$(L/2, h_c)$、$(L - b, 0)$，$b = 0.55$ mm，$h_c = 3.5$ mm，平面应力：`L, H = 12.0, 6.0`，`SUPPORT_WIDTH = 0.55`，`CORNER_HEIGHT = 3.5`，`PLANE_TYPE = "plane_stress"` |
| 支座 | 左支座段 $x \in [0, b]$ 固定铰 $u_x = u_y = 0$；右支座段 $x \in [L - b, L]$ 滚动铰 $u_y = 0$。支座不对称，状态场不严格对称，必须求全域 |
| 载荷 | 顶边中点竖直合力 $P = -300$ N，分布于以 $x = L/2$ 为中心、宽 $l = 1$ mm 的区段，强度 $\bar t = P/l$；通过节点插值映射至顶边连续 $\mathbb P_1$ 迹空间（载荷区端点取半值）后施加：`LOAD = -300.0`，`LOAD_WIDTH = 1.0`，`load_discretization="p1_trace_interpolation"` |
| 实体材料 | $\nu = 0.25$；$E_0$ 物理参考值 $100000$ MPa，计算取 $1$ MPa（同比缩放理由同 5.2.3 节）：`E0 = 1.0`，`NU = 0.25` |
| 许用应力 | $\bar\sigma = 1000$ MPa：`STRESS_LIMIT = 1000.0` |

### 2.2 分析参数

| 参数 | 数值或设置 |
|---|---|
| 网格 | Gmsh 在左半域五边形上生成非结构三角形网格（Frontal-Delaunay，目标尺寸 $h = 0.0625$ mm），再关于 $x = L/2$ 镜像成全域，网格严格对称；顶边载荷区与其余段用 transfinite 均匀剖分，使载荷区端点落在节点上；共 $31\,554$ 个单元、$16\,083$ 个节点：`MESH_SIZE = 0.0625`，`MESH_ALGORITHM = 6`。$h$ 须为 $0.5/k$ 形式以整除顶边分段 |
| 阶次 | $k = 3$：应力 $\mathbb P_3$、位移不连续 $\mathbb P_2$，`--order 3` 传入 `space_degree=order` |
| 数值积分 | $2k + 2 = 8$ 阶：`integration_order=2 * order + 2` |
| HZMFEM 角点松弛 | 关闭：`USE_RELAXATION = False`。各几何角点处相邻两边的牵引均为零，数据相容；未松弛的角点由分析器按 Hu 与 Ma（2021）式 (4.1) 联立两侧强约束施加。凹角顶点与支座段内端通常不满足两单元松弛拓扑，无法统一松弛 |
| 设计对称化 | 支座不对称而几何与载荷对称，由过滤器对 $x = L/2$ 做坐标折叠（PolyFilter 的 `'Y'` 选项）：`Filter(..., symmetry_planes={0: problem.symmetry_axis})` |
| 线性求解器 | MUMPS 直接法：`SOLVER = "mumps"` |
| 终态验收 | 同 5.2.3 节：优化结束后在最终物理密度上重解一次，记录相对平衡残差并在判据集合上复算 $\delta_g$ |

规模：HZMFEM $k = 3$ 应力与位移合计 $901\,427$ 个自由度。

### 2.3 应力约束：不设被动实体区

| 参数 | 数值或设置 |
|---|---|
| 约束形式 | 同 5.2.3 节：逐单元形心表观 von Mises 应力比的 $\varepsilon$ 松弛约束，$\varepsilon = 10^{-3}$ |
| 载荷端点被动实体区 | 不设，全部单元受应力约束：`LOAD_PAD_RADIUS = 0.0` |
| 支座端点被动实体区 | 不设：`SUPPORT_PAD_RADIUS = 0.0` |
| 凹角顶点 | 不设：优化从凹角应力集中区去料是本算例要观察的现象 |

本算例采用法向区间牵引。连续 P1 节点插值下的全实体网格加密检查未显示载荷端点附近的应力发散，因此不设置载荷端点被动实体区；凹角与支座附近同样不设被动区。全部单元均参与局部应力约束、停止判据中的应力检查与终态验收。

P1 节点插值端点诊断已原样归档到 `results/portal-frame-2d-stress/postprocess/endpoint_p1_interpolation_20261018/endpoint_h_sweep.json`。在 $h=0.125,0.0625,0.03125$ mm 下，LFEM $p=2$ 的端点邻域最大形心实体应力比分别为 0.6486、0.6840、0.6996；HZMFEM $k=3$ 前两级为 0.6527、0.6872，最细一级未计算。该检查支持当前不设端点被动区的选择，不替代优化终态的全域验收。旧载荷诊断与被动区对照保留在第 4 节。

### 2.4 优化参数

除下表所列各项外，材料插值、初始密度、投影与 $\beta$ 连续化、AL 罚参数与乘子安全阈、MMA 参数与内层停止规则均同 5.2.3 节 2.4 节（$\mu_0 = 50$，$\mu_{\max} = 10^4$，$\lambda_{\max} = 10^4$，移动限 $0.15$，$\beta$ 从 $1$ 起每 $5$ 个外层步加 $1$ 至 $10$）。

| 参数 | 数值或设置 |
|---|---|
| 密度过滤 | $r_{\min} = 0.25$ mm，与 PolyStress 一致（$r_{\min}/h = 4$）：`FILTER_RADIUS = 0.25` |
| 停止准则 C3 | C0 至 C2 连续 $10$ 个外层步同时满足（5.2.3 节为 $3$）：`HOLD_STEPS = 10` |
| 罚参数初值 | $\mu_0 = 50$，与 5.2.3 节相同，不取 PolyStress 表 3 的 $10$（见第 4 节）：`MU_0 = 50.0` |
| 判据集合 | 全部受约束单元（本算例无被动实体区，即全部单元），与 5.2.3 节同口径：`ACCEPTANCE_SOLID_THRESHOLD = None`。当前运行从优化开始即采用全部单元验收；旧运行的密度筛选及等价性核查保留在第 4 节 |

当前运行实际采用 `hold_steps=10`，与 5.2.3 节的 3 步分别如实记录。第 52--61 个外层步连续满足 C0--C2，采用第 61 步结束时的最终设计，不将其解释为 `hold_steps=3` 的首次停止结果。历史 hold 对照见第 4 节。

## 3. 当前论文结果（2026-10-18 更新）

当前来源为 `results/portal-frame-2d-stress/analyzer-huzhang__order-3__p1-interpolation__all-cells__lambda-max-10000.0/`。HZMFEM $k=3$ 采用连续 P1 节点插值牵引、无被动区、全部 31554 个单元验收、乘子上限 10000 与 `hold_steps=10`，已按停止准则收敛。

| MMA 更新次数 | 外层步数 | 体积分数 | 全域最大约束值 | 最大表观应力比 | 实体区最大应力比 | 相对平衡残差 | 乘子截断数 |
|---|---|---|---|---|---|---|---|
| 278 | 61 | 0.2356840147 | 0.0042426110 | 0.9997972890 | 1.0223761383 | $5.3865\times10^{-9}$ | 0 |

数值取自当前运行的 `summary.json`；实体区最大应力比仅作诊断，按物理密度不小于 0.5 统计，不是停止判据的验收范围。论文采用体积分数 $23.57\%$、最大局部约束值 $4.243\times10^{-3}$，低于容差 $0.005$。

构型由上部弦杆、斜杆及凹角上方的连接杆件组成，形成从顶边载荷区通向两侧支座的主要传力路径。高应力区域主要沿承载杆件分布，部分杆件的表观应力接近许用水平，交汇处保留少量中间密度区域。应力与主应力散点均采用最终设计在自身离散下的单元形心值。

冻结再分析在全部单元上复现最大约束值 $0.0042426107161$，与运行摘要一致并通过验收。在 $\rho_{\mathrm{phys}}>0.9$ 的实体区域内，最大归一化牵引跳量为 $2.6871749455\times10^{-16}$，处于浮点舍入误差量级；相对跳量检查值为 $5.8146\times10^{-12}<10^{-10}$。这些结果体现了高阶 Hu--Zhang 元在所检查非结构化网格上的几何适应能力与跨单元法向牵引连续性，支持其用于含斜边与凹角设计域的局部应力约束优化，不据此宣称设计性能优于 LFEM。

### 3.1 论文插图与数据来源

| 图 | 文件 / 入口 | 内容与来源 |
|---|---|---|
| 5.10 | 论文侧 `fig5_11_portal_schematic.png` | 几何尺寸、载荷与边界条件示意图，文件名保留历史编号 |
| 5.11 | `portal_mesh.png/pdf`；`portal-mesh` | 非结构化全域网格与凹角、载荷区间局部放大 |
| 5.12 | `stress_portal_hz_k3_topology.png`；`stress-portal-topology` | 最终物理密度与表观应力比，使用当前终态及冻结场数据 |
| 5.13 | `stress_portal_hz_k3_convergence.png/pdf`；`stress-portal-convergence` | 体积分数、全域最大约束值历史与全部单元的归一化表观主应力散点 |

图 5.11--5.13 的图件位于 `results/figures/`。牵引跳量仅在正文报告统计结果，不单列插图。当前后处理数据位于 `results/portal-frame-2d-stress/postprocess/`：

- `discretization_probe/huzhang-3__probe.json`：来源目录、密度 SHA-256、冻结验收与牵引跳量统计；同名前缀 `__fields.npz` 保存逐单元数据。实体带统计单元数 6112 保留在记录中，正文不列出。
- `stress_fields/fig_data_portal-huzhang-k3.npz` 及配套 JSON：网格、密度、表观主应力与 von Mises 应力比等成图数据。
- `endpoint_p1_interpolation_20261018/endpoint_h_sweep.json`：当前 P1 节点插值的全实体端点诊断；旧 `endpoint_h_sweep.json` 保留为历史载荷诊断，不覆盖。

### 3.2 产物与溯源

当前运行已从 `outputs/portal-frame-2d-stress/analyzer-huzhang__order-3__p1-interpolation__all-cells__lambda-max-10000.0/` 经 `sync_results.py` 固化至同名 `results/` 目录。原始配置和摘要记录 `git_revision=ee06c0594f024396c9cedab5f7b0ae102907d55e`、`git_dirty=false`，生成时间为 `2026-10-18T03:00:12+00:00`；后处理另有各自的版本记录，不覆盖运行溯源。

已入库的核心文件包括 `config.json`、`summary.json`、`history.json`、`outer_history.json`、`final_optimizer_state.npz`、`displacement_final.npy`、`stress_final.npy`、`density_initial.vtu` 与 `density_final.vtu`。历史分别包含 278 次 MMA 更新与 61 个外层步，终态保存全部 31554 个单元的设计变量、物理密度、乘子和罚参数，供冻结分析，不支持精确断点续算。

逐步 `iterations/` 帧与 `evolution.pvd` 不属于核心入库文件。旧的无后缀 `results/portal-frame-2d-stress/analyzer-huzhang__order-3/` 保留旧载荷运行，不作为当前论文结果来源。P1 端点诊断由原输出文件逐字节复制到第 3.1 节所列独立目录，并附 `archive-manifest.json` 记录源路径、大小与 SHA-256；此次归档未重新计算。

## 4. 历史结果与诊断记录（不作为当前论文结果）

以下保留旧载荷、旧验收口径及 LFEM 对照，不用当前默认配置解释旧运行。

### 4.1 旧载荷端点与被动区诊断

与 5.2.3 节不同，本算例不设载荷端点被动实体区，与 PolyStress 一致（其 `PortalDomain.m` 以等分节点力加载，`Passive = []`，全部单元受约束）。依据是两个算例载荷方向不同：悬臂梁的竖向载荷作用在竖直边上，是**切向**区段牵引，区段端点处 $\sigma_{xx}$ 有对数奇异；门式框架的竖向载荷作用在水平顶边上，是**法向**区段牵引，端点处应力只间断、不奇异，且投影到 $\mathbb P_1$ 迹空间后间断也被抹平。

全实体（$\rho = 1$）加密验证（`plot.py portal-endpoint-check`，`postprocess/endpoint_h_sweep.json`）：网格 $h = 0.125 / 0.0625 / 0.03125$，单元形心实体 von Mises 应力比的最大值如下。

| 离散 | 载荷端点 $r \le 0.25$ 内 | 凹角顶点 $r \le 0.25$ 内（奇异对照） |
|---|---|---|
| LFEM $p = 1$ | $0.683 / 0.706 / 0.711$ | $1.42 / 1.89 / 2.50$ |
| LFEM $p = 2$ | $0.655 / 0.685 / 0.700$ | $1.22 / 1.62 / 2.15$ |
| LFEM $p = 3$ | $0.653 / 0.685 / 0.700$ | $1.05 / 1.41 / 1.86$ |
| HZMFEM $k = 2$ | $0.665 / 0.690 / 0.702$ | $1.58 / 2.09 / 2.77$ |
| HZMFEM $k = 3$ | $0.659 / 0.688$ / — | $1.31 / 1.76$ / — |

载荷端点邻域各离散收敛到约 $0.70$，最后一级加密只变约 $2\%$，端点正下方单元（$0.55 \sim 0.64$）还低于顶边跨中（约 $0.71$），最大值来自上弦弯曲而非端点；凹角顶点每次 $h$ 减半增约 $1.3$ 倍，是奇异的。支座段内端 $(b, 0)$、$(L - b, 0)$ 同样随加密增长（LFEM $p = 2$：$1.66 \to 2.46$），亦为奇异点，未设被动区，由优化器去料处理。

曾以 $r_{\mathrm{pad}} = 0.25$（$118$ 个单元）跑过 HZMFEM $k = 3$（第 4 节）：被动区内最大实体应力比 $1.168$，体积 $0.2401$。端点既无奇异，这一超限是豁免被优化器利用（被动区不受约束，载荷下方不必加材料），与 5.2.3 节 2.3 节记录的"只豁免不保留"属同一类现象；去掉被动区后体积反降为 $0.2362$，全域最大表观应力比为 $1.0004$。

### 4.2 旧 L2 投影运行结果

HZMFEM $k = 3$ 按停止准则收敛（`termination_reason: criterion-met`）。下表取自运行目录的 `summary.json`，列含义同 5.2.3 节。

| 分析链 | $k$ | MMA 更新 | 体积分数 | $g_{\max}$（$\tilde\rho \ge 0.5$） | $g_{\max}$（全部受约束单元） | 最大表观应力比 | 实体区最大应力比 | 平衡残差 | 乘子截断 |
|---|---|---|---|---|---|---|---|---|---|
| HZMFEM | 3 | 315 | 0.2362 | $1.26 \times 10^{-3}$ | $1.91 \times 10^{-3}$ | 1.0004 | 1.015 | $2.7 \times 10^{-8}$ | 0 |

列的来源：$g_{\max}$ 分别为 `max_relative_violation_solid_region` 与 `max_relative_violation`；最大表观应力比 `max_apparent_stress_ratio`（全域，即应力图色标上限）；实体区最大应力比 `max_solid_stress_ratio_solid_region`；平衡残差 `relative_equilibrium_residual`。耗时 $2.71$ h。

验收口径的等价性：本次运行的 C2 与内层停止判据用的是 $\tilde\rho \ge 0.5$；逐次核对 `history.json` 的 $315$ 次 MMA 更新，按全部受约束单元判定时 $\delta_g \le 5 \times 10^{-3}$ 的结论**每一次都相同**（两口径的 $g_{\max}$ 只在 $25$ 次更新上不同，且都同在容差一侧），故本次运行的轨迹与终止步与改用全部受约束单元验收（5.2.3 节现行口径）完全一致，终态全部受约束单元上 $g_{\max} = 1.91 \times 10^{-3}$。

构型特征：顶部上弦、斜杆、中央菱形与贴着凹角顶点通过的横杆，与 PolyStress 图 8 的定性特征一致；横杆压在凹角顶点上（图 8 亦然），沿斜边的材料只延伸约 $0.1$ mm，小于 $r_{\min}$；跨中上弦实体厚约 $0.47$ mm。主要杆件基本满应力，全域最大表观应力比 $1.0004$，应力图色标可直接取 $[0, 1]$。中部交汇处只剩小块淡灰（中部灰度面积 $0.97$，灰度单元实体应力比中位 $0.69$）。整体布局（斜肩短柱）与图 8 的直柱贯通横梁不同，第 4 节的网格加密对照显示该差异随网格分辨率改变。

#### 历史图表对应关系（旧编号）

以下图号、数值与路径描述旧运行；当前同名后处理文件已更新为 P1 节点插值结果，不能用于复核这些旧统计。当前图表以第 3.1 节为准。

| 图 | 产物 case | 内容 | 数据来源 |
|---|---|---|---|
| 5.11(b) | `portal-mesh` | 全域网格与凹角、载荷区间两处局部放大 | 运行目录的 `density_final.vtu` |
| 5.12 | `stress-portal-topology` | 最终构型与表观应力比（含设计域轮廓） | `plot.py discretization-probe --case portal-frame-2d-stress` 导出的 `postprocess/discretization_probe/huzhang-3__fields.npz` |
| 5.13 | `stress-portal-convergence` | 迭代历史（全部受约束单元上的 $g_{\max}$）与全部受约束单元的表观主应力散点 | `history.json`、`summary.json`；`plot.py export --run portal-huzhang-k3` |

图 5.11(a) 示意图是论文侧 TikZ。牵引跳量统计取自 `huzhang-3__probe.json`：$\overline\rho_e > 0.9$ 的单元 $6090$ 个，$A_e$ 中位 $9.4 \times 10^{-17}$、最大 $2.634 \times 10^{-16}$；探针两道验收门（Hu--Zhang 相对跳量 $4.4 \times 10^{-12} < 10^{-10}$，判据集合 $g_{\max}$ 复现 $1.263 \times 10^{-3}$）均通过。2.3 节的全实体端点验证由 `plot.py portal-endpoint-check` 产出 `postprocess/endpoint_h_sweep.json`。

#### 旧运行产物与溯源

运行写 `outputs/portal-frame-2d-stress/analyzer-huzhang__order-3/`，产物种类与 5.2.3 节 3.2 节相同（含 `stress_final.npy`、`iterations/` 帧 $316$ 帧）。该旧运行现已入库至同名 `results/` 目录；以下保留当时的入库命令供追溯：

```bash
python experiments/paper_topopt_huzhang/sync_results.py portal-frame-2d-stress analyzer-huzhang__order-3
```

溯源：运行于 2026-10-16（`generated_at_utc` 07:38:45），`config.json` 记 `git c0b5a69`、`git_dirty = true`。数值上与提交 `7a58f4b` 一致：`c0b5a69..HEAD` 间 Hu--Zhang 数值路径（分析器、函数空间、约束、过滤、优化器、问题类）只有 `mma.py` / `oc.py` 增记初始帧两行；`run_portal_frame.py` 在此区间只经 `7a58f4b` 一次提交，其常量与本次 `config.json` 逐项相同。其余运行时未提交的改动（网格拓扑的 `device=` 参数、`vtk_export.py` 输出转换）不影响 numpy 后端上的数值。

### 4.3 LFEM 与参数诊断

`HOLD_STEPS = 10` 的来历：原以为 $3$ 步时平均变化判据在中部灰度区尚未定型时就已满足；LFEM $p = 3$ 的两次对照（第 4 节）显示 $3$ 步与 $10$ 步的终态几乎相同，灰度区在停机前已稳定，这一理由不成立。保留 $10$ 只多耗约 $18$ 次 MMA 更新，无害。



以下运行均在 `outputs/portal-frame-2d-stress/` 下，不入库。除第一行外均设载荷端点被动实体区 $r_{\mathrm{pad}} = 0.25$（$118$ 个单元）。

| 运行目录 | 离散 | $\mu_0$ / hold | MMA 更新 | 体积分数 | $g_{\max}$（$\tilde\rho \ge 0.5$ / 全域） | 收敛 | 说明 |
|---|---|---|---|---|---|---|---|
| `analyzer-huzhang__order-3__pad0.25` | HZMFEM $k = 3$，$h = 0.0625$ | 50 / 10 | 279 | 0.2401 | $3.21 \times 10^{-3}$ / $4.54 \times 10^{-3}$ | 是 | 被动区内最大实体应力比 $1.168$（受约束区域 $1.011$）；去掉被动区对应旧 L2 投影无被动区运行（第 4.2 节） |
| `analyzer-lfem__order-3__old` | LFEM $p = 3$，$h = 0.0625$ | 10 / 10 | 808 | 0.3446 | $4.84 \times 10^{-3}$ / $1.78 \times 10^{-2}$ | 是 | $42$ 个乘子触 $\lambda_{\max}$；连续化前期严重超限，$\beta$ 到顶时拓扑被锁成多细杆网格 |
| `analyzer-lfem__order-3__mu50-hold3` | LFEM $p = 3$，$h = 0.0625$ | 50 / 3 | 265 | 0.2271 | $3.98 \times 10^{-3}$ / $4.04 \times 10^{-3}$ | 是 | 中部有三角灰度填角；逐步帧未保留 |
| `analyzer-lfem__order-3` | LFEM $p = 3$，$h = 0.0625$ | 50 / 10 | 283 | 0.2258 | $2.90 \times 10^{-3}$ / $3.15 \times 10^{-3}$ | 是 | 与 hold 3 几乎相同，灰度填角约第 235 步起稳定不变 |
| `analyzer-lfem__order-1` | LFEM $p = 1$，$h = 0.0625$ | 50 / 10 | 277 | 0.2147 | $2.43 \times 10^{-3}$ / $5.77 \times 10^{-3}$ | 是 | 同网格换阶次：三角灰度填角消失，整体布局同 $p = 3$ |
| `analyzer-lfem__order-1__h0.025` | LFEM $p = 1$，$h = 0.025$ | 50 / 10 | 1000 | 0.2122 | $2.21 \times 10^{-2}$ / $2.40 \times 10^{-2}$ | 否 | 布局接近 PolyStress 图 8(a)（直柱 + $y \approx 3.3$ 横梁）；$196\,708$ 个单元下 $\mu$、$\lambda$ 上限相对不足（AL 按约束数 $N$ 归一化，压同一峰值所需 $\lambda$ 约随 $N$ 增长），违反位置来回跳 |

结论：

- **$\mu_0$**：PolyStress 的 $\mu_0 = 10$ 在本算例的 LFEM $p = 3$ 上失败，$\mu_0 = 50$（同 5.2.3 节）给出体积 $0.226$ 的干净拓扑。
- **灰度填角**：LFEM $p = 3$ 中部的三角灰度团单元实体应力比中位 $0.91$、$18\%$ 乘子激活（实体单元 $1.7\%$），是被应力约束锁住的交汇凹角填角，不是低应力区。同网格的 $p = 1$ 没有它，HZMFEM $k = 3$ 基本没有它（中部灰度面积 $1.29 \to 1.00$，应力比中位 $0.76$，乘子激活 $13.5\%$；去掉被动区后为 $0.97$、$0.69$、$12.2\%$）。
- **布局**：随网格分辨率变化（$p = 1$ 的 $h = 0.0625 \to 0.025$），与阶次无关。

冻结设计交叉重分析（2026-10-15 一次性脚本，未入库；对角项逐位复现 `summary.json`；两设计均带 $r_{\mathrm{pad}} = 0.25$ 被动区），$\tilde\rho \ge 0.5$ 且不在被动区的单元上的 $g_{\max}$：

| 设计 \ 重分析离散 | LFEM $p = 1$ | $p = 2$ | $p = 3$ | $p = 4$ | HZMFEM $k = 3$ |
|---|---|---|---|---|---|
| LFEM $p = 1$ 设计 | $2.4 \times 10^{-3}$ | $9.4 \times 10^{-2}$ | $9.1 \times 10^{-2}$ | $1.3 \times 10^{-1}$ | $2.6 \times 10^{-1}$ |
| LFEM $p = 3$ 设计 | $2.7 \times 10^{-1}$ | $7.3 \times 10^{-2}$ | $2.9 \times 10^{-3}$ | $5.9 \times 10^{-2}$ | $3.6 \times 10^{-1}$ |

- $p = 1$ 设计在任何高阶重分析下超限 $9\% \sim 26\%$：线性元形心应力抹平交汇凹角的峰值，其黑白构型与较低体积不能作为设计更优的依据。
- $p = 3$ 设计中部灰度团置 $0$（尖角）后 $g_{\max}$ 为 $3.4 / 3.9 / 5.1$（$p = 3 / 4$ / HZ $k = 3$），随离散加细增大；置 $1$（实体倒圆）后体积 $+4.5\%$ 仍超限 $0.17 \sim 0.59$。冻结替换下灰度既不能删也不能直接实体化。
- $p = 3$ 设计在 HZMFEM $k = 3$ 下的最大超限位于凹角顶点 $(6, 3.5)$ 附近（材料内角约 $245°$，奇异）：各设计只在自身离散下可行。


## 历史实施记录

以下按实施顺序保留记录，其中“未启动”“未入库”描述当时状态；当前完成状态与来源以第 3 节为准。

### 2026-10-18: P1 节点插值与全域验收的新运行设置

新运行采用 `p1_trace_interpolation`, 端点取半值; 不设置载荷或支座被动区.
`ACCEPTANCE_SOLID_THRESHOLD=None`, 停止判据与终态验收覆盖全部受约束单元;
`LAMBDA_MAX=10000`, `HOLD_STEPS=10`, 其他物理与优化参数保持不变.
运行目录增加 `__p1-interpolation__all-cells__lambda-max-10000` 后缀, 不覆盖原结果.
本次只修改运行设置, 未启动优化; 前述已入库数值仍属于原 L2 投影运行.
原运行的密度验收阈值为 0.5, 乘子上限为 3000, 不得用新默认配置解释旧记录.
新的冻结分析默认跟随节点插值设置; 旧结果的再分析必须恢复其原载荷配置, 不可混用.

运行命令 (WSL, soptx 根目录):

```bash
/home/brighthe/miniconda3/envs/ihpcm/bin/python experiments/paper_topopt_huzhang/run_portal_frame.py --analyzer huzhang --order 3
```

2026-10-18 全实体端点诊断见 `outputs/portal-endpoint-p1-interpolation-20261018/endpoint_h_sweep.json`.
LFEM p=2 三级端点邻域最大形心应力比为 0.6486/0.6840/0.6996,
HZMFEM k=3 前两级为 0.6527/0.6872; 未发现必须增加端点被动区的证据.
此检查不代替新优化终态的全域约束验收.


### 2026-10-18: 命令行参数化与冻结复算

`run_portal_frame.py` 新增 `--load-discretization` (`p1-interpolation` /
`p1-l2-projection`), `--load-pad-radius` (有限非负数, mm) 和 `--lambda-max`
(有限正数). 默认分别为节点插值、0 和 10000. 正半径同时用于应力豁免与过滤后的实体固定.
实际取值写入 config/summary, 目录名编码载荷方式、正的被动区半径及乘子上限;
新的浮点上限后缀为 `10000.0`, 原有目录不移动、不覆盖.
冻结探针及主应力场导出读取运行目录保存的载荷方式与网格尺寸, 旧配置缺少载荷字段时
读取旧 summary 中的 `p1_trace_l2_projection`; 缺失或冲突记录报错. 被动区按原运行摘要
恢复, 冻结密度不重新过滤. 验收阈值仍采用原运行摘要, 不把旧阈值 0.5 改写为全域.

```bash
/home/brighthe/miniconda3/envs/ihpcm/bin/python experiments/paper_topopt_huzhang/run_portal_frame.py \
  --analyzer huzhang --order 3 \
  --load-discretization p1-interpolation --load-pad-radius 0 --lambda-max 10000
```

本轮未执行优化或静力求解.


### 2026-10-18: 入口与共用构造模块拆分

`run_portal_frame.py` 仅定义 `parse_args` 与 `main`. 默认参数、`run_label`,
`build_problem`, `build_mesh`, `format_run_header` 移至同目录的
`portal_frame_common.py`. 端点诊断、冻结探针与主应力场导出直接引用共用模块,
不再导入门式框架运行入口. 所有函数体保持不变, 运行命令、CLI 默认值、目录名
和配置记录格式保持不变. 本轮不生成网格、不执行状态求解或优化.


### 2026-10-18: 与悬臂梁入口统一组织

本次调整替代上一次共用模块拆分: 默认参数、`parse_args`、`build_problem`、
`format_run_header`、`main` 集中在 `run_portal_frame.py`, 与悬臂梁脚本一致.
删除 `run_label`, 目录名直接在 `parse_args` 中拼接; `RUN_LABEL` 保留默认目录模板.
仅 Gmsh 镜像网格函数移至 `portal_frame_mesh.py`, 显式接收网格尺寸与算法参数.
移除此前的 `portal_frame_common.py`, 分析模块同步引用运行脚本配置与独立网格函数.
命令、默认参数、目录名称、原运行配置读取机制不变; 本轮不启动计算.


### 2026-10-18: 逐轮保存运行结果

`run_portal_frame.py` 通过 ALM-MMA 的可选输出回调复用 `IterationFrameWriter`.
过滤后的初始物理密度立即写为第 0 帧和 `density_initial.vtu`; 每次 MMA 更新并完成新设计
状态分析后写出对应的密度、表观应力比帧, 更新 `evolution.pvd` 与 `history.json`.
每个 ALM 外层完成后更新 `outer_history.json`, 记录更新后的乘子最大值、罚参数和停止计数.
历史 JSON 通过同目录临时文件替换; 初始帧不计入 MMA 更新次数.
终态冻结再分析、最终场、`final_optimizer_state.npz` 与 `summary.json` 仍在正常结束后生成.
优化参数、状态求解与停止判据保持不变, 新增磁盘 I/O 会增加运行耗时.
逐轮文件只用于查看与核查, 不支持精确断点续算. 已启动的旧进程不会自动采用新输出逻辑.

## 2026-10-18: P1 节点插值新结果入库与论文更新

- 来源: `outputs/portal-frame-2d-stress/analyzer-huzhang__order-3__p1-interpolation__all-cells__lambda-max-10000.0`, 经 `sync_results.py` 固化到同名 `results/` 目录; 旧运行保留.
- 连续 P1 节点插值牵引, 无被动区, 全部 31554 个单元验收, 乘子上限 10000.
- **实际运行 hold_steps=10, 未修改原始配置与终止记录.** 第 52--61 个外层步连续满足 C0--C2, 第 61 步结束, 累计 278 次 MMA 更新. 第 54 步 (263 次 MMA 更新) 的 hold_count 为 3; 论文采用第 61 步的最终设计, 不将其冒记为 hold_steps=3 的首次停止结果.
- 论文按实际运行设置表述: 5.2.3 采用 N_hold=3, 5.2.4 采用 N_hold=10. 不再使用统一为“连续至少 3 步”的表述.
- 最终体积分数 0.23568401473285952, 最大局部约束值 0.00424261101844331, 最大表观 von Mises 应力比 0.9997972889696697, 外层平均设计变化量 0.00025886574958888835.
- 最新冻结再分析复现 summary 的最大约束值, 全部单元验收通过; rho_phys>0.9 实体带最大归一化牵引跳量约 2.687e-16, 相对跳量门限检查通过. 精确结果见 `postprocess/discretization_probe/huzhang-3__probe.json`.
- 图 5.12/5.13 改用本次运行的冻结应力与优化历史. 正文体积分数更新为 23.57%, 最大约束值为 4.243e-3, 跳量为 2.687e-16. 本次仅后处理, 未重跑优化.