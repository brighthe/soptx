# Hu--Zhang 拓扑优化投稿论文复现实验

## 目录结构

```text
experiments/paper_topopt_huzhang/
|-- README.md                   # 本文件: 目录结构、参数注册约定、调用方式
|-- results_analysis.md         # 实测数据、机理分析与全部复现命令 (端到端流水线见其 §7)
|-- run.py                      # 执行入口: 按 --case 跑算例, 写运行产物
|-- compare.py                  # 后处理入口: 插图 / 论文表 / 冻结指标
|-- cases.toml                  # [[cases]] 算例参数注册表
|-- config.py                   # 路径常量、TOML 加载校验与参数拍平
|-- pipeline.py                 # 组装层: 共享原语 + 三族算例装配器 + 模型名注册表
|-- driver.py                   # 优化/状态对比驱动 (--case 的实现) + 能量恒等式诊断
|-- convergence.py              # 制造解收敛阶验证 (--case 的另一条驱动)
|-- provenance.py               # Git revision、环境与产物摘要
|-- report.py                   # 产出: 论文表 5.1 / 5.2 (由 compare.py table 调用)
|-- metrics.py                  # 产出: 梯度校验 / 冻结设计指标 / 插图 npz 导出
|-- bearing_reanalysis.py       # 产出: 轴承冻结设计交叉再分析 + nu 扫描 (论文表 5.3 / 5.4)
|-- stress_cross_evaluation.py  # 产出: 应力算例一份构型 x 七条离散的应力比与可行性余量交叉表
|-- discretization_probe.py     # 产出: 应力算例冻结构型的离散敏感性探针 (实验 A)
|-- edge_jump.py                # 内边法向牵引跳量 [[sigma.n]]; 仅被 discretization_probe 调用
|-- plots/                      # 唯一子目录: 每张图一个模块, 文件名为 <算例族>_<产物>
|                               # (成图落在 outputs/figures/, 故不与之同名)
|   |-- _base.py                # 十个成图模块的共用底座: 字体/vtu/产物定位/落盘
|   |-- manufactured_mesh.py    # 唯一不读运行产物的一张: 制造解算例的棋盘格剖分示意
|   |-- compliance_topology.py compliance_convergence.py compliance_k1_comparison.py
|   |-- bearing_topologies.py bearing_highorder_topologies.py
|   `-- stress_topologies.py stress_convergence.py stress_max_ratio_history.py
|                               # 论文图号只在各模块 docstring 首行的括注里
`-- outputs/                    # 运行产物, 不提交
```

## 已注册算例

| case id | `role` | 模型 | 组装入口 | 论文位置 |
| --- | --- | --- | --- | --- |
| `manufactured-native` | `convergence-verification` | `MixedBoundarySinusoidalElasticity2D` | `convergence.py: run_convergence_suite` | 表 5.1 ($k=3,4$ 原生格式) |
| `manufactured-stabilized` | `convergence-verification` | `MixedBoundarySinusoidalElasticity2D` | `convergence.py: run_convergence_suite` | 表 5.2 ($k=1,2$ 矩阵跳量稳定化) |
| `compliance-fixed-fixed-half` | `optimization-baseline` | `FixedFixedBeamHalfDomain2d` | `pipeline.py: build_fixed_fixed_*` | 5.2.1 节 / 图 5.2~5.3 |
| `bearing-compressible` | `incompressible-baseline` | `BearingDevice2d` | `pipeline.py: build_bearing_*` | 5.2.2 节 / 图 5.5、表 5.3~5.4 |
| `bearing-incompressible` | `incompressible-study` | `BearingDevice2d` | `pipeline.py: build_bearing_*` | 5.2.2 节 / 图 5.5、表 5.3~5.4 |
| `cantilever-middle-2d-stress` | `stress-constrained` | `CantileverMiddle2d` | `pipeline.py: build_stress_*` | 5.2.3 节 / 图 5.7~5.9、表 5.5 |

## 参数注册约定

每个 `[[cases]]` 必须包含三类独立参数:

- `[cases.model]`: `name` 和 `parameters`, 选择 `soptx.problems` 中的物理模型并给出载荷、材料、平面假设等参数。
- `[cases.discretization]`: 三角剖分方式 `mesh_type` (取值见 `pipeline.MESH_TYPES`, 由 `pipeline.create_mesh` 分派)、网格剖分、受控比较阶次 `comparison_orders`、可选的 `supplementary_orders` (只放宽 `--order` 白名单, 不进缺省与 `--full`)、角点松弛和线性求解器。
- `[cases.optimization]`: 体积分数、过滤器、材料插值、优化器 (`oc` / `mma` / `al_mma`) 及其迭代参数。

`mesh_type` 的两种取值:

| 取值 | 对角线规则 | 用途 |
|---|---|---|
| `triangle-checkerboard` | `(i+j)` 偶数 `/`, 奇数 `\` | 固支梁、悬臂梁与制造解算例的注册值; 轴承算例的棋盘格对照 |
| `triangle-single-diagonal-symmetric` | 左半 `/`, 右半 `\`, 两个顶角落翻转 | 两条轴承 case 的注册值; 低阶位移元体积闭锁对照 (与左右对称问题同对称性) |

生成器见 `soptx.mesh.create_huzhang_checkerboard_mesh` / `create_huzhang_symmetric_single_diagonal_mesh`。

材料插值对象 `interpolation` 取三值: `E` 只插值 Young 模量 (Poisson 比固定为实体值), `E+nu` 同时按论文式 (4.3) 插值 Poisson 比 (只允许近不可压缩材料 `nu >= 0.49`, 可压缩材料上直接报错), `auto` 按材料自动决定。两条轴承 case 已显式登记 (`bearing-compressible` 为 `E`, `bearing-incompressible` 为 `E+nu`), 参数 `nu_penalty_factor` / `void_poisson_ratio` 走 `--override`。

两条 `manufactured-*` 的 `stabilization` 字段直接驱动 `HuZhangMFEMAnalyzer` 的同名入参, 取值 `none` / `matrix_jump` / `vector_jump`, 只在 $p \le GD$ 时生效; `manufactured-native` 只能声明 `none`, 写别的会被 `convergence.py` 拒绝。

## 调用方式

`run.py` 解题并落盘, `compare.py` 在其产物之上做二次数值试验与成图: 八个动词里只有 `table` 是纯
格式化 (由 `summary.json` 重算论文表 5.1 / 5.2), 其余七个都会按冻结设计重新组装并求解 (代码中
的 `REANALYSIS` 集合), 给出 `run.py` 不产的数据 —— 表 5.3 / 5.4 的轴承交叉再分析与 $\nu$ 扫描、
应力算例的七条离散交叉表与离散敏感性探针、伴随灵敏度的有限差分校验, 都只能由这里产生。两者都以 `--case` 驱动,
但 case 是两套命名:

| | `run.py` | `compare.py` |
|---|---|---|
| 一条 case | 一道要解的题 | 一件要产出的成果 (多数需重新求解) |
| id 来源 | `cases.toml` 的 `[[cases]]` | `plots/` 下声明了 `SOURCE_CASE` / `REQUIRED_RUNS` 的模块, id 取模块文件名 |
| 派发 | 由 `role` 决定 (`convergence-verification` 归 `convergence.py`, 其余归 `driver.py`) | 由模块自描述决定, 绘图数据缺失或过期时自行触发冻结重分析 |
| 列出 | `run.py --list` | `compare.py --list` |

调用方拿到 id 即可运行, 不必先知道用哪个动词; 不支持按文件路径直接执行 `driver.py` / `convergence.py`。

省略 `--analyzer` / `--order` / `--degree` 时只展开一个组合: 方法取 `huzhang` (该 case 未注册时退回 `methods` 首项), 阶次取 `comparison_orders` 的最小值。论文那套方法/阶次对比是显式动作, 用 `--full` 展开成 `methods x comparison_orders` 全集, 也可以用 `--analyzer all` / `--order 2 3` 精确指定; 注册表里的 `methods` / `comparison_orders` 同时是白名单, 越界直接报错。`summary.json` 按方法与阶次为键增量合并, 因此分多次单跑与一次 `--full` 得到的论文表一致。

覆盖参数有两个通道:

| 通道 | 覆盖的字段 | 直接报错的情形 |
|---|---|---|
| 具名开关 `--mesh-type` / `--interpolation` / `--analyzer` / `--order` / `--nx` / `--ny` / `--optimizer` / `--stabilization` / `--load-discretization` | 最常用的字段 | 与 `--override` 重复指定同一字段 |
| `--override KEY=VALUE` (可重复) | `[cases.discretization]` / `[cases.optimization]` 的键, 另加运行组合维度 `analyzer` / `order` (多值用逗号分隔) | 字段名写错、类型转换失败 |

取值按配置对象现有取值的类型转换, 不静默取一边。两个驱动认的参数并不重叠 (`--stabilization` / `--degree` / `--levels` 只属于收敛验证, `--analyzer` / `--order` / `--nx` / `--optimizer` 只属于优化), 因此覆盖参数只允许配合单个 `--case`。`--load-discretization` 在固支梁上默认 `p1_trace_l2_projection`; `point_force` 用节点集中力替换分布牵引, 仅允许 `--analyzer lfem`, 不支持 `--mode state-compare`, 此时 `load_width` 不参与载荷计算, 产物标签含 `load_discretization-point_force`。

## 载荷引入垫片 (应力算例)

右端牵引在贴片端点 `(80, 17)` 与 `(80, 23)` 处跳变 `P/l`, 这是载荷模型自带的混合边界
间断点: 载荷分布化只消除点载荷的 `r^-1` 奇异, 消不掉端点奇异, 密度设计也消不掉。实测
该处实体应力比随阶次单调上升 (LFEM `k=1..4` 为 2.77 / 3.94 / 4.19 / 4.50, Hu--Zhang
`k=2` 为 4.31), 无收敛迹象。

`cases.toml` 的 `load_pad_radius` (默认 `1.5 = l/4`) 把这两个端点的固定物理半径邻域
划为"载荷引入垫片": 载荷总要通过一块实体传入结构, 该处的奇异性是边界条件的性质而
非设计的缺陷, 既约束不了, 也不该让优化器去动它。掩码按单元重心判定
(`soptx.topology.constraints.exemption`), 两条分析链施加同一份掩码, 保证对照在同一
验收区域上进行。掩码有两个用途, **必须成对施加**:

1. **应力豁免** —— 垫片内不施加局部应力约束 (`apply_exemption`);
2. **实体保留 (passive solid)** —— 垫片内物理密度钉为 1, 密度灵敏度置零
   (`apply_passive_solid`)。

只做第 1 条会被优化器利用。2026-09-16 的对照 (Hu--Zhang `k=2`): 只豁免不保留时确实从
1000 步不收敛变为 536 步收敛, 但优化器发现该邻域不再受约束, 把原本被应力约束逼着保持
`rho ~ 0.98` 的单元减到 0.67 以换体积, 实体应力比从 0.68 涨到 1.45,
`max_solid_stress_ratio_solid_region` 由历年稳定的 1.01-1.03 跳到 1.452 —— 豁免掩盖了
真实过应力。故实体保留不是可选项。

实现上的四条边界:

- 实体保留施加在**过滤/投影之后** (`Filter._enforce_passive_solid`)。只钉设计变量是
  不够的: `rmin = 6 >> h = 1`, 宽过滤下保留单元的物理密度仍由邻域决定, 达不到满密度;
- `rho_phys` 在垫片内与设计变量无关, 故链式法则中 `d rho_phys / d z = 0`, 过滤器在
  委托给策略之前先把这些行的密度灵敏度清零;
- 豁免只改约束集合的成员, 不改 AL 的归一化基数 —— `fun` 返回张量的形状不变, 带垫片
  与不带垫片的两次运行罚项量级严格可比。豁免点的约束值取严格可行的常值 (`-1.0`),
  对目标与灵敏度的贡献恒为 0; 真实值仍可由 `compute_unexempted_constraint` 取回,
  `summary.json` 单列 `max_constraint_pad` 与 `max_solid_stress_ratio_pad`, 并把受约束
  区域单列为 `max_solid_stress_ratio_constrained`, 垫片掩盖了多大的应力可直接读出;
- 非零半径恒进产物目录名 (`__load_pad_radius-1.5`), 2026-09-16 之前的产物不会被覆盖;
  `--override load_pad_radius=0` 复原旧行为与旧目录名。

核对用 `compare.py stress-cross-eval`: 把一份构型冻结, 用 LFEM `k=1..4` 与 Hu--Zhang
`k=2..4` 各前向求解一次, 在同一密度场上读同一批单元, 把"离散"与"构型"两个变量分开,
并按 `eta = rho / rho_crit - 1` 报可行性余量。

## 离散敏感性探针 (实验 A)

`stress-cross-eval` 回答"同一构型换一条离散, 读出的应力比差多少"; `discretization-probe`
在同一批前向解上继续追问这个差**从哪来、是否大到让约束判定失效**。设计冻结, 只让离散变,
四件事各自独立成表:

1. **离散间散布, 按密度带分解** —— $\rho$ 分 10 档, 逐档报各离散的 $\max g$ 与散布
   $\Delta = \max - \min$, 同时报 $\Delta / \delta_g$ ($\delta_g = 5\times 10^{-3}$,
   即停止容差)。散布按**逐单元**取差再取最大, 比直接差两条离散的全局最大值严格: 后者
   的两个最大值可能落在不同单元上。另出 argmax 单元身份表与两两交集大小 —— 若各离散把
   $g_\max$ 放在不同单元上, 横比本身就无意义。全部 7 条与"可信子集"
   {`lfem-3`, `lfem-4`, `huzhang-3`, `huzhang-4`} 各算一遍; `huzhang-2` 如实报出但不进
   可信子集 (见 `docs/known-issues/`: 跳量稳定化的惩罚系数不随密度插值);
2. **胞内采样敏感性** —— 两族 analyzer 的 `compute_stress_state` 默认
   `integration_order=1`, 三角形上即形心一个点, 于是**约束只读一个形心样本, 与 $k$ 无关**。
   探针用同一 `state` 重采样 $q \in \{1, 3, 4\}$ (不重解), 报
   $\max_q g_e - g_e(\text{形心})$ 随 $k$ 的走势, 即高阶单元的胞内起伏被形心采样丢了多少;
3. **内边法向牵引跳量** ($\texttt{edge\_jump.py}$) —— 逐单元应力约束要有意义, 前提是
   "该单元的应力"良定义。Hu--Zhang 的应力是 $H(\mathrm{div}, S)$ 协调的原始变量,
   $[[\sigma \cdot n]] \equiv 0$; LFEM 的应力由位移求导, 跨边有跳。报相对跳量的
   中位数 / P95 / 最大值并按密度带分解, 重点与实体带的可行余量 (1%--4%) 对照。
   跳量一律测**表观应力**: 连续介质中真实牵引跨面连续, 而实体应力
   $\sigma^{\mathrm{sol}} = \sigma^{\mathrm{app}} / m_E$ 因 $m_E$ 逐单元常值必然跳,
   测它没有意义;
4. **细网格双参考** (`--reference`, 默认关) —— 在 160x80 上建 `huzhang-4` 与 `lfem-4`
   两条参考, 密度用 `singularity_h_probe.map_density_to_mesh` 作面积保持的分片常数延拓,
   细单元 $g$ 按父单元取 max 归约。除各粗离散相对参考的 $\Delta$ 外, 一并报**两条参考彼此
   的分歧**: 若它已超 $\delta_g$, 参考解自身不可信, 该块结论作废。

两条硬门, 不过则进程返回码为 1:

- Hu--Zhang 的相对跳量必须 $< 10^{-10}$ (协调性的直接后果)。不满足即判定 `edge_jump.py`
  有错, 第 3 项无效;
- $q = 1$ 重采样出的 $g_e$ 必须与 `constraint.fun` 的结果逐单元相等 —— 证明重采样路径与
  生产路径同口径, 第 2 项的差值才归因于采样而非归因于旁路。

```bash
# 默认对两份 k=3 无 pad 构型 (huzhang / lfem 各一) 交叉跑, 避免交叉表那种来源偏置
~/miniconda3/envs/ihpcm/bin/python compare.py discretization-probe
# 加细网格双参考; 分钟级
~/miniconda3/envs/ihpcm/bin/python compare.py discretization-probe --reference
# 指定构型 (可重复), 跳过牵引跳量
~/miniconda3/envs/ihpcm/bin/python compare.py discretization-probe --design <运行目录名> --no-jump
```

产物落在 `outputs/cantilever-middle-2d-stress/postprocess/discretization_probe/`:
`<design>__probe.json` (全部聚合表 + provenance + 构型 sha256)、`<design>__fields.npz`
(逐单元原始场, 所有离散 x 所有采样阶次, 使求解成为一次性成本)、
`<design>__<disc>.vtu` ($g$ / 各阶采样的 $g$ / 相对跳量, 供出图)。
