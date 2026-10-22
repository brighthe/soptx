# Hu--Zhang 拓扑优化投稿论文复现实验

本目录产出论文第 5 章的全部数值结果 (表 5.1~5.4、图 5.2~5.3 / 5.5~5.6 / 5.8~5.11), 结果入库于 `results/`. 图 5.1 / 5.4 / 5.7 是论文侧 TikZ 示意图, 不在本目录.

## 目录结构

```text
experiments/paper_topopt_huzhang/
|-- README.md
|-- run_manufactured.py         # 5.1 节 / 表 5.1~5.2: MixedBoundarySinusoidalElasticity2D 前向求解 -> outputs/manufactured-convergence/<run>/
|-- manufactured.md             # 5.1 节实验记录: 设置、结果指针与角点松弛开/关对照
|-- run_fixed_fixed.py          # 5.2.1 节 / 图 5.2~5.3、表 5.3: HalfFixedFixedBeamPatch2d, MMA -> outputs/compliance-fixed-fixed-half/<run>/
|-- fixed_fixed.md              # 5.2.1 节实验记录: 设置、结果与角点松弛开/关对照
|-- run_bearing.py              # 5.2.2 节 / 图 5.5~5.6、表 5.4: BearingTraction2d, OC, 两组材料 -> outputs/bearing/{nu-0.3,nu-0.4999}/<run>/
|-- bearing.md                  # 5.2.2 节实验记录: 设置、结果与角点松弛开/关对照
|-- run_cantilever_stress.py    # 5.2.3 节 / 图 5.8~5.11: CantileverMiddlePatch2d, AL-MMA -> outputs/cantilever-middle-2d-stress/<run>/
|-- cantilever_stress.md        # 5.2.3 节实验记录: 设置、被动实体区、结果与角点松弛开/关对照
|-- run_portal_frame.py         # 5.2.4 节 / 图 5.11~5.13: PortalFramePatch2d (全域, 左铰右滚), gmsh 半域网格镜像成全域, 过滤器对 x = L/2 设计对称化, AL-MMA -> outputs/portal-frame-2d-stress/<run>/
|-- portal_frame_mesh.py        # 门式框架 Gmsh 半域剖分与镜像网格构造, 参数由运行脚本传入
|-- portal_frame.md             # 5.2.4 节实验记录: 设置、HZ k=3 结果与 LFEM 诊断对照 (灰度填角、交叉重分析)
|-- sync_results.py             # 把 outputs/<case>/<run> 固化到 results/ (确认后入库; 已有产物须 --force)
|-- table.py                    # 表格入口: 冻结设计交叉再分析 + 论文表格 (表 5.3 / 5.4)
|-- plot.py                     # 绘图入口: 插图数据 (重分析) + 成图
|-- config.py                   # 共用路径常量 (table.py / plot.py、analysis/ 与 plots/ 使用; run 脚本各自定义)
|-- analysis/                   # table.py / plot.py 背后的数值计算, 由入口派发, 不直接运行
|   |-- compliance_reanalysis.py    # 表 5.3: 固支梁六个冻结设计 x 六种离散的柔顺度交叉再分析
|   |-- bearing_reanalysis.py       # 表 5.4: 轴承冻结设计 x 五种离散的柔顺度交叉再分析
|   |-- bearing_h_locking.py        # 图 5.5 的数据: 轴承全实体域 h 收敛闭锁考察
|   |-- discretization_probe.py     # 图 5.8 / 5.10 / 5.12 的数据: 应力算例 (--case 悬臂梁 / 门式框架) 冻结构型重分析, 应力比与牵引跳量
|   |-- stress_metrics.py           # 图 5.9 / 5.13 的数据: 应力算例插图 npz 导出 (主应力面板)
|   |-- portal_endpoint_check.py    # 5.2.4 节: 门式框架全实体域载荷端点应力有界性验证 (不设被动区的依据)
|   `-- provenance.py               # 上述再分析的溯源戳记
|-- plots/                      # 每张图一个模块, 文件名即 plot.py --case 的 id; 共用底座 _base.py
|-- outputs/                    # run 脚本的运行产物 (不入库, .gitignore 整体忽略): <case>/<run>/, 已有目录拒绝覆盖
|                               #   优化类 run 脚本的逐步帧 iterations/iter_NNNN.vtu + evolution.pvd 留在运行目录,
|                               #   需要时手动迁移到 Windows 查看目录 config.VIEW_DIR/<case>/<run>/
`-- results/                    # 论文证据, 入库 (config.OUTPUT_DIR), 只经 sync_results.py 写入
    |-- manufactured-convergence/{relaxation-off,relaxation-on}/
    |                                  # 5.1 节: 不松弛 (论文口径) 与松弛对照, 各含数据 json 与 table5_1 / table5_2
    |-- <case>/<run>/                  # 优化运行: config / summary / history / outer_history 四份 json, final_optimizer_state.npz,
    |                                  #   displacement_final.npy (Hu--Zhang 另有 stress_final.npy), density_initial.vtu / density_final.vtu
    |-- <case>/postprocess/            # 冻结设计再分析与插图数据
    |-- tables/                        # 表 5.1~5.4 (Markdown, 可直接贴入草稿); 表 5.1 / 5.2 经 sync_results.py --paper-tables 拷入
    `-- figures/                       # 成图
```

## 载荷端点被动实体区 (应力算例)

右端牵引在贴片端点 `(80, 17)` 与 `(80, 23)` 处跳变 `P/l`, 这是载荷模型自带的混合边界
间断点: 载荷分布化只消除点载荷的 `r^-1` 奇异, 消不掉端点奇异, 密度设计也消不掉。实测
该处实体应力比随阶次单调上升 (LFEM `k=1..4` 为 2.77 / 3.94 / 4.19 / 4.50, Hu--Zhang
`k=2` 为 4.31), 无收敛迹象。

`run_cantilever_stress.py` 的 `LOAD_PAD_RADIUS` (`1.5 = l/4`) 把这两个端点的固定物理半径邻域
划为"被动实体区" (论文记作 Omega_pad): 载荷总要通过一块实体传入结构, 该处的奇异性是边界条件的性质而
非设计的缺陷, 既约束不了, 也不该让优化器去动它。掩码按单元重心判定
(`soptx.topology.constraints.exemption`), 两条分析链施加同一份掩码, 保证对照在同一
验收区域上进行。掩码有两个用途, **必须成对施加**:

1. **应力豁免** —— 被动实体区内不施加局部应力约束 (`apply_exemption`);
2. **实体保留 (passive solid)** —— 被动实体区内物理密度钉为 1, 密度灵敏度置零
   (`apply_passive_solid`)。

只做第 1 条会被优化器利用。2026-09-16 的对照 (Hu--Zhang `k=2`): 只豁免不保留时确实从
1000 步不收敛变为 536 步收敛, 但优化器发现该邻域不再受约束, 把原本被应力约束逼着保持
`rho ~ 0.98` 的单元减到 0.67 以换体积, 实体应力比从 0.68 涨到 1.45,
`max_solid_stress_ratio_solid_region` 由历年稳定的 1.01-1.03 跳到 1.452 —— 豁免掩盖了
真实过应力。故实体保留不是可选项。

实现上的四条边界:

- 实体保留施加在**过滤/投影之后** (`Filter._enforce_passive_solid`)。只钉设计变量是
  不够的: `rmin = 6 >> h = 1`, 宽过滤下保留单元的物理密度仍由邻域决定, 达不到满密度;
- `rho_phys` 在被动实体区内与设计变量无关, 故链式法则中 `d rho_phys / d z = 0`, 过滤器在
  委托给策略之前先把这些行的密度灵敏度清零;
- 豁免只改约束集合的成员, 不改 AL 的归一化基数 —— `fun` 返回张量的形状不变, 带被动实体区
  与不带被动实体区的两次运行罚项量级严格可比。豁免点的约束值取严格可行的常值 (`-1.0`),
  对目标与灵敏度的贡献恒为 0; 真实值仍可由 `compute_unexempted_constraint` 取回,
  `summary.json` 单列 `max_constraint_pad` 与 `max_solid_stress_ratio_pad`, 并把受约束
  区域单列为 `max_solid_stress_ratio_constrained`, 被动实体区掩盖了多大的应力可直接读出;
- 半径记在各运行目录的 `summary.json` 中 (目录名只含分析链与阶次, 见 `run_cantilever_stress.RUN_LABEL`)。

核对走 `plot.py discretization-probe`: 构型冻结后在自身离散下重解, 不施加豁免,
被动实体区的真实读数一并取回。

## 当前悬臂梁图表入口（2026-10-17）

论文 5.2.3 采用 LFEM p=3 与 HZMFEM k=2,3；通过 `plot.py --case stress-cantilever-comparison` 生成图 5.8--5.10 和表 5.5。运行目录、参数差异及历史结果边界见 `cantilever_stress.md` 第 3 节。入口复用冻结再分析与牵引跳量验收，不运行优化。

- 当前悬臂梁主应力散点图（图 5.9）：`plot.py --case stress-cantilever-principal`，从当前 LFEM p=3 与 HZMFEM k=2,3 终态冻结导出；后续收敛历史、牵引跳量为图 5.10、5.11，门式框架图号顺延为 5.12--5.14。


### 2026-10-17: 合并优化历史与主应力图

当前图 5.9 由 `plot.py --case stress-cantilever-history-principal` 生成，上排为体积和约束历史，下排为 LFEM p=3 与 HZMFEM k=2,3 的终态主应力散点。复用已有历史及已核验的散点数据，不重新求解；输出 `stress_cantilever_history_principal.png/pdf`。原独立图保留为历史产物，正文不再引用；牵引跳量现为图 5.10，门式框架为图 5.11--5.13。此前记录中的编号按历史版本理解。


### 2026-10-17: 牵引跳量仅保留表格

正文删除原图 5.10 的牵引跳量空间分布，保留定义与统计范围，以表 5.5 展示量化证据；表格删除实体带单元数列，底层统计及原图件仍保留。门式框架图号调整为 5.10--5.12。生成表格的代码同步更新，未重新计算。


## 门式框架运行选项 (2026-10-18)

`run_portal_frame.py` 支持 `--load-discretization p1-interpolation|p1-l2-projection`,
`--load-pad-radius` (mm, 默认 0) 和 `--lambda-max` (默认 10000).
新运行按参数生成独立目录, 不覆盖旧 L2 投影结果. 冻结再分析读取指定运行的载荷配置,
不使用当前默认值替代旧配置; 具体命令与历史差异见 `portal_frame.md`.
