# Hu--Zhang 拓扑优化投稿论文复现实验

本目录产出论文第 5 章的全部数值结果 (表 5.1~5.4、图 5.2~5.3 / 5.5~5.6 / 5.8~5.11), 结果入库于 `results/`. 图 5.1 / 5.4 / 5.7 是论文侧 TikZ 示意图, 不在本目录.

## 目录结构

```text
experiments/paper_topopt_huzhang/
|-- README.md
|-- run_manufactured.py         # 5.1 节 / 表 5.1~5.2: MixedBoundarySinusoidalElasticity2D 前向求解 -> results/manufactured-convergence/
|-- run_fixed_fixed.py          # 5.2.1 节 / 图 5.2~5.3、表 5.3: FixedFixedBeamHalfDomain2d, MMA -> results/compliance-fixed-fixed-half/
|-- run_bearing.py              # 5.2.2 节 / 图 5.5~5.6、表 5.4: BearingDevice2d, OC, 两组材料 -> results/bearing/{nu-0.3,nu-0.4999}/
|-- run_cantilever_stress.py    # 5.2.3 节 / 图 5.8~5.11: CantileverMiddle2d, AL-MMA -> results/cantilever-middle-2d-stress/
|-- table.py                    # 表格入口: 冻结设计交叉再分析 + 论文表格 (表 5.3 / 5.4)
|-- plot.py                     # 绘图入口: 插图数据 (重分析) + 成图
|-- config.py                   # 共用路径常量 (table.py / plot.py、analysis/ 与 plots/ 使用; run 脚本各自定义)
|-- analysis/                   # table.py / plot.py 背后的数值计算, 由入口派发, 不直接运行
|   |-- compliance_reanalysis.py    # 表 5.3: 固支梁六个冻结设计 x 六种离散的柔顺度交叉再分析
|   |-- bearing_reanalysis.py       # 表 5.4: 轴承冻结设计 x 四种离散的柔顺度交叉再分析
|   |-- bearing_h_locking.py        # 图 5.5 的数据: 轴承全实体域 h 收敛闭锁考察
|   |-- discretization_probe.py     # 图 5.8 / 5.10 / 5.11 的数据: 应力算例冻结构型重分析, 应力比与牵引跳量
|   |-- stress_metrics.py           # 图 5.9 的数据: 应力算例插图 npz 导出 (主应力面板)
|   `-- provenance.py               # 上述再分析的溯源戳记
|-- plots/                      # 每张图一个模块, 文件名即 plot.py --case 的 id; 共用底座 _base.py
`-- results/                    # 论文证据, 入库 (config.OUTPUT_DIR); 改 run 脚本常量后重跑会覆盖同名运行目录
    |-- <case>/<run>/                  # 优化运行: summary.json / history.json / density_final.vtu
    |-- <case>/postprocess/            # 冻结设计再分析与插图数据
    |-- tables/                        # 表 5.1~5.4 (Markdown, 可直接贴入草稿)
    `-- figures/                       # 成图
```

## 载荷引入垫片 (应力算例)

右端牵引在贴片端点 `(80, 17)` 与 `(80, 23)` 处跳变 `P/l`, 这是载荷模型自带的混合边界
间断点: 载荷分布化只消除点载荷的 `r^-1` 奇异, 消不掉端点奇异, 密度设计也消不掉。实测
该处实体应力比随阶次单调上升 (LFEM `k=1..4` 为 2.77 / 3.94 / 4.19 / 4.50, Hu--Zhang
`k=2` 为 4.31), 无收敛迹象。

`run_cantilever_stress.py` 的 `LOAD_PAD_RADIUS` (`1.5 = l/4`) 把这两个端点的固定物理半径邻域
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
- 半径恒进产物目录名 (`__load_pad_radius-1.5`, 见 `run_cantilever_stress.run_label`)。

核对走 `plot.py discretization-probe`: 构型冻结后在自身离散下重解, 不施加豁免,
被动实体区的真实读数一并取回。
