# Hu--Zhang 拓扑优化投稿论文复现实验

本目录产出论文第 5 章的全部数值结果 (表 5.1~5.4、图 5.2~5.3 / 5.5~5.6 / 5.8~5.11), 结果入库于 `results/`. 图 5.1 / 5.4 / 5.7 是论文侧 TikZ 示意图, 不在本目录.

论文第 5.2 节的 HZMFEM `k=2` 优化、冻结设计再分析和应力后处理统一使用固定稳定化系数: `HuZhangMFEMAnalyzer` 默认 `stabilization_coefficient="fixed"`, 各 run 脚本不另行指定即为固定系数. 结果一致性检查不等于灵敏度公式验证.

## 复现命令

在实验目录下运行 (解释器为 `~/miniconda3/envs/ihpcm/bin/python`, 下文简写为 `python`). 优化运行耗时较长; 再分析与成图只读 `results/` 中已有的运行.

```bash
# 5.1 节: 制造解收敛阶 (表 5.1 / 5.2)
python run_manufactured.py

# 5.2.1 节: 两端固支梁 (表 5.3, 图 5.2 / 5.3)
python run_fixed_fixed.py                                   # 六组优化
python table.py compliance-reanalysis                       # 表 5.3
python plot.py --case compliance-topology                   # 图 5.2
python plot.py --case compliance-convergence                # 图 5.3

# 5.2.2 节: 二维轴承 (表 5.4, 图 5.5 / 5.6)
python run_bearing.py                                       # 两组材料 x 三种离散
python table.py bearing-reanalysis                          # 表 5.4
python plot.py bearing-h-locking                            # 图 5.5 的数据 (全实体 h 收敛)
python plot.py --case bearing-solid-h-convergence           # 图 5.5
python plot.py --case bearing-topologies                    # 图 5.6

# 5.2.3 节: 悬臂梁局部应力约束 (图 5.8~5.11)
python run_cantilever_stress.py                             # 六组优化
python plot.py discretization-probe                         # 图 5.8 / 5.10 / 5.11 的数据
python plot.py --case stress-cubic-topologies               # 图 5.8
python plot.py --case stress-cubic-convergence              # 图 5.9 (插图数据自动导出)
python plot.py --case stress-hz-orders-topologies           # 图 5.10
python plot.py --case stress-traction-jump                  # 图 5.11
```

成图模块默认把图同步到论文仓库 `papers/huzhang-topopt/figures/` (`config.PAPER_FIGURE_DIR`). 核对数据前不想覆盖论文中的图时, 在命令前加 `HUZHANG_PAPER_FIGDIR=/nonexistent`: 同步目录不存在即跳过, 图只写到 `results/figures/`.

## 目录结构

```text
experiments/paper_topopt_huzhang/
|-- README.md                   # 本文件: 目录结构、算例与入口、调用方式
|-- run_manufactured.py         # 5.1 节: 制造解收敛阶验证 + 表 5.1 / 5.2 (自包含, 参数为文件顶部常量)
|-- run_fixed_fixed.py          # 5.2.1 节: 两端固支梁六组 MMA 优化 (自包含, 参数为文件顶部常量)
|-- run_bearing.py              # 5.2.2 节: 轴承两组材料 x 三种离散 OC 优化 (同上)
|-- run_cantilever_stress.py    # 5.2.3 节: 悬臂梁六组 AL-MMA 应力约束优化 (同上)
|-- table.py                    # 表格入口: 冻结设计交叉再分析 + 论文表格 (表 5.3 / 5.4)
|-- plot.py                     # 绘图入口: 插图数据 (重分析) + 成图
|-- config.py                   # 路径常量与源码路径注入 (各入口、analysis/ 与 plots/ 共用)
|-- analysis/                   # table.py / plot.py 背后的数值计算, 由入口派发, 不直接运行
|   |-- compliance_reanalysis.py    # 表 5.3: 固支梁六个冻结设计 x 六种离散的柔顺度交叉再分析
|   |-- bearing_reanalysis.py       # 表 5.4: 轴承冻结设计 x 四种离散的柔顺度交叉再分析
|   |-- bearing_h_locking.py        # 图 5.5 的数据: 轴承全实体域 h 收敛闭锁考察, 两档 nu x 四级网格 x 四种离散
|   |-- discretization_probe.py     # 图 5.8 / 5.10 / 5.11 的数据: 应力算例冻结构型重分析, 应力比与牵引跳量
|   |-- stress_metrics.py           # 图 5.9 的数据: 应力算例插图 npz 导出 (主应力面板)
|   `-- provenance.py               # 上述再分析的溯源戳记: Git revision、环境与产物摘要
|-- plots/                      # 每张图一个模块, 文件名为 <算例族>_<产物> (成图落在 results/figures/)
|   |-- _base.py                # 成图模块的共用底座: 字体/vtu/产物定位/落盘 (论文口径: Palatino, 版心 5.9 in; 分片常值云图只出 PNG 600 dpi, 线图出 PDF+PNG)
|   |-- compliance_topology.py compliance_convergence.py         # 图 5.2 / 5.3
|   |-- bearing_solid_h_convergence.py bearing_topologies.py      # 图 5.5 / 5.6
|   |-- stress_cubic_topologies.py stress_cubic_convergence.py   # 图 5.8 / 5.9: LFEM p=3 与 HZMFEM k=3 主对比
|   `-- stress_hz_orders_topologies.py stress_traction_jump.py   # 图 5.10 / 5.11: HZMFEM k=2,4 构型; 六个构型实体带的牵引跳量
|                               # 图号以各模块 docstring 首行括注为准
`-- results/                    # 论文证据, 入库: 第 5 章全部数值结果 (config.OUTPUT_DIR)
    |-- manufactured-convergence/      # 5.1 节: manufactured_convergence.json (全部数据 + 参数 + 溯源戳记)
    |-- <case>/<run>/                  # 5.2 节优化运行: summary.json / history.json / density_final.vtu
    |                                  # (轴承两组材料同属一个算例: bearing/nu-0.3/<run>/ 与 bearing/nu-0.4999/<run>/;
    |                                  #  应力算例另有 final_optimizer_state.npz / outer_history.json)
    |-- <case>/postprocess/            # 5.2 节冻结设计再分析与插图数据
    |-- tables/                        # 全部论文表格: table5_1.md / table5_2.md (run_manufactured.py),
    |                                  # table5_3.md / table5_4.md (table.py); 可直接贴入草稿
    `-- figures/                       # 成图 (plot.py)
```

论文引用的数字只以 `results/` 为准, 是否可复现以其中 JSON 的 `provenance.reproducible` 为准。`results/` 的内容与 git 中一致, 不存逐步帧。

各运行的逐步帧只供 ParaView 查看, 由 run 脚本直接写到 Windows 本地盘的查看目录 `C:\workspace\soptx-results\paper_topopt_huzhang\` (脚本常量 `VIEW_ROOT`; Windows 下的 ParaView 经 `\\wsl.localhost` 读大批帧很慢), 目录结构与 `results/` 一一对应, 每个运行只含 `vtu/` 帧和与之同级的时间序列集合文件 `evolution.pvd` (在 ParaView 中打开后者即可按迭代步播放, 末帧即最终构型); `density_final.vtu` 只在 `results/`, 每个文件只存一处。帧在优化结束后从内存一次写出, 写 Windows 盘不拖慢迭代。该盘不可用时脚本退回写 `results/`, 此时 `vtu/` 与 `evolution.pvd` 由 `.gitignore` 排除, 不会入库。

`plot.py discretization-probe` (图 5.8 / 5.10 / 5.11 的数据) 同理: `__probe.json` 入库; 逐单元场 `__fields.npz` 每构型约 12 MB, 留在 `results/` 但不入库, 克隆后需重跑该探针再成图; 查看用的 `.vtu` 写到查看目录 (`config.VIEW_DIR`) 下的同名路径。

## 算例与入口

| 算例目录 (`results/` 下) | run 脚本 | 模型 | 优化器 | 论文位置 |
| --- | --- | --- | --- | --- |
| `manufactured-convergence` | `run_manufactured.py` | `MixedBoundarySinusoidalElasticity2D` | (前向求解) | 5.1 节 / 表 5.1~5.2 |
| `compliance-fixed-fixed-half` | `run_fixed_fixed.py` | `FixedFixedBeamHalfDomain2d` | MMA | 5.2.1 节 / 图 5.2~5.3、表 5.3 |
| `bearing` (`nu-0.3` / `nu-0.4999`) | `run_bearing.py` | `BearingDevice2d` | OC | 5.2.2 节 / 图 5.5~5.6、表 5.4 |
| `cantilever-middle-2d-stress` | `run_cantilever_stress.py` | `CantileverMiddle2d` | AL-MMA | 5.2.3 节 / 图 5.8~5.11 |

### 参数

每个 run 脚本自包含: 算例参数是文件顶部的常量 (模型、离散、优化三组), 直接调用 soptx 的 API, 结构只有 `parse_args` / `build` / `main` 三个函数。`build(method, order)` 组装一条分析链 (问题、网格、分析器、初始密度、目标与约束), 优化运行与冻结设计再分析 (表 5.3 / 5.4、图 5.5、图 5.8~5.11 的数据) 共用它, 保证再分析与优化同一份组装代码。溯源戳记在运行开始时盖, 记录本进程实际加载的代码版本。

受控比较协议: 给定阶次 `k`, LFEM 取位移阶 `p=k`, HZMFEM 取应力阶 `k` (位移阶 `k-1`), 统一积分阶 `q=2k+2`。

三角剖分方式:

| 剖分 | 对角线规则 | 用于 |
|---|---|---|
| 棋盘格 (`create_huzhang_checkerboard_mesh`) | `(i+j)` 偶数 `/`, 奇数 `\` | 制造解、固支梁、悬臂梁 |
| 对称单向对角 (`create_huzhang_symmetric_single_diagonal_mesh`) | 左半 `/`, 右半 `\`, 两个顶角落翻转 | 轴承: 低阶位移元在其上出现经典体积自锁, 且与左右对称问题同对称性 |

材料插值对象: 固支梁与悬臂梁只插值 Young 模量; 轴承的可压缩组 (`nu = 0.3`) 只插值 Young 模量, 近不可压缩组 (`nu = 0.4999`) 按论文式 (4.4) 同时插值 Young 模量与 Poisson 比 (`run_bearing.GROUPS`, 参数 `NU_VOID` / `NU_PENALTY`)。

## 调用方式

各 run 脚本缺省跑论文该算例的全部组合; 调试单组时用 `--analyzer` / `--order` (轴承另有 `--group`) 只跑其中几组, 例如:

```bash
python run_fixed_fixed.py --analyzer huzhang --order 2
python run_bearing.py --group nu-0.4999 --analyzer lfem --order 1
```

`table.py` 与 `plot.py` 在 run 脚本的产物之上做二次数值试验, 分别产出表格与插图。`table.py` 的两个子命令做表 5.3 / 5.4 的冻结设计交叉再分析并写出论文表格; `plot.py` 的三个数据动词 (`export` / `bearing-h-locking` / `discretization-probe`, 代码中的 `REANALYSIS` 集合) 为图 5.5、5.8~5.11 按冻结设计重新求解, `--case` 成图。`plot.py --case` 的 id 是一件产物 (一张图), 取 `plots/` 下声明了 `SOURCE_CASE` / `REQUIRED_RUNS` 的模块文件名, 与算例目录是两套命名; 缺数据时会提示补跑哪个 run 脚本或数据动词。`table.py --list` 与 `plot.py --list` 列出全部子命令与产物。

参数只在 run 脚本的常量里, 命令行没有覆盖通道。对照实验 (换优化器、网格、插值对象等) 需改常量; 运行目录名只由分析链、阶次与应力算例的协议标签决定, 改常量后重跑会写入同名目录、覆盖论文结果, 因此对照前先提交 `results/`, 用完以 git 恢复。

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

核对走 `plot.py discretization-probe` (下节): 构型冻结后在自身离散下重解, 不施加豁免,
被动实体区的真实读数一并取回。

## 冻结构型重分析与牵引跳量 (图 5.8 / 5.10 / 5.11)

`discretization-probe` 把论文 5.2.3 节的六份最终构型 (LFEM `p=2..4`, Hu--Zhang `k=2..4`,
pad 1.5 mm, 判据集合 $\rho \ge 0.5$) 冻结, 各在优化所用的离散下重解一次状态方程, 导出
逐单元的表观应力比、约束值 $g$ 与内边法向牵引绝对跳量 $A_e$ (求值见 `analysis/discretization_probe.py` 第二节)。
分析链由 `run_cantilever_stress.build(load_pad_radius=0)` 组装。
逐单元应力约束要有意义, 前提是"该单元的应力"良定义: Hu--Zhang 的应力是
$H(\mathrm{div}, S)$ 协调的原始变量, $[[\sigma \cdot n]] \equiv 0$; LFEM 的应力由位移
求导, 跨边有跳。$A_e$ 取单元各内边归一化跳量 RMS 的最大值, 统计限于实体带
($\rho > 0.9$ 且不在被动实体区)。跳量一律测**表观应力**: 连续介质中真实牵引跨面连续,
而实体应力 $\sigma^{\mathrm{sol}} = \sigma^{\mathrm{app}} / m_E$ 因 $m_E$ 逐单元常值必然跳,
测它没有意义。Hu--Zhang 的 $A_e$ 处于舍入量级, 其具体数值随求解器线程调度在
$10^{-16}$ 量级上波动, 只宜按量级引用。

两条硬门, 不过则进程返回码为 1:

- Hu--Zhang 的相对跳量必须 $< 10^{-10}$ (协调性的直接后果)。不满足即判定跳量求值有错;
- 判据集合上的最大 $g$ 必须复现 `summary.json` 的 `max_relative_violation_solid_region`,
  否则掩码或阈值口径与优化器不同。

```bash
# 默认重分析六份构型
python plot.py discretization-probe
# 指定构型 (可重复; 须是 run_cantilever_stress.py 写出的运行目录)
python plot.py discretization-probe --design <运行目录名>
```

产物落在 `results/cantilever-middle-2d-stress/postprocess/discretization_probe/`:
`<design>__probe.json` (跳量统计 + 验收门 + provenance + 构型 sha256, 入库)、`<design>__fields.npz`
(逐单元场, 供图 5.8 / 5.10 / 5.11 读取, 不入库); 查看用的 `<design>__<disc>.vtu` 写到查看目录下的同名路径。
成图所需的网格取自被冻结运行的 `density_final.vtu`, 不依赖查看目录。
