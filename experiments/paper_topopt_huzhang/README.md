# Hu--Zhang 拓扑优化投稿论文复现实验

## 论文结果集 (固定稳定化系数)

论文第 5.2 节的 HZMFEM `k=2` 优化、冻结设计再分析和应力后处理统一使用固定系数. `HuZhangMFEMAnalyzer` 默认采用 `stabilization_coefficient="fixed"`, 普通 `run.py`/`compare.py` 即为固定系数. 对照实验可在构造分析器时显式指定 `stabilization_coefficient="density_dependent"`; 该选项独立于跳量形式 `stabilization` 和网格缩放律 `stabilization_scaling`.

当前论文结果集为 `outputs/fixed_coefficient_optimization/20260923T013529873586Z/` (下记 `R`): 四个 `k=2` 优化结果实际保存在该目录, 未受影响的 LFEM 和高阶结果以符号链接复用 `outputs/<case>/` 的运行. `outputs/<case>/` 下同名的 `k=2` 目录是密度相关系数的旧结果, 只作历史对比. 系数与源码摘要见 `R/manifest.json` 与 `R/paper_refresh_manifest.json`.

优化走 `run.py --output`, 后处理走 `compare.py --output-root` (须写在最前面):

```bash
# 在实验目录运行; 新优化应给一个新的独立结果集, 保留既有结果.
~/miniconda3/envs/ihpcm/bin/python run.py --case compliance-fixed-fixed-half --analyzer huzhang --order 2 --output outputs/fixed_coefficient_optimization/<new-run>
# 对结果集 R 重算再分析 / 重绘
~/miniconda3/envs/ihpcm/bin/python compare.py --output-root outputs/fixed_coefficient_optimization/20260923T013529873586Z compliance-reanalysis
~/miniconda3/envs/ihpcm/bin/python compare.py --output-root outputs/fixed_coefficient_optimization/20260923T013529873586Z --case bearing-topologies
```

新结果集需自行把未受影响的运行链接进来 (R 中的链接清单可作模板). 结果一致性检查不等于灵敏度公式验证.

## 目录结构

```text
experiments/paper_topopt_huzhang/
|-- README.md                   # 本文件: 目录结构、参数注册约定、调用方式
|-- results_analysis.md         # 实测数据、机理分析与全部复现命令 (端到端流水线见其 §6)
|-- run.py                      # 执行入口: 按 --case 跑优化算例, 写运行产物
|-- manufactured_convergence.py # 自包含: 制造解收敛阶验证 + 论文表 5.1 / 5.2 (论文 5.1 节, 不经 run.py)
|-- compare.py                  # 后处理入口: 插图 / 冻结设计再分析
|-- cases.toml                  # [[cases]] 算例参数注册表
|-- config.py                   # 路径常量、TOML 加载校验与参数拍平
|-- pipeline.py                 # 组装层: 共享原语 + 三族算例装配器 + 模型名注册表
|-- driver.py                   # 优化驱动 (--case 的实现) + 能量恒等式诊断
|-- provenance.py               # Git revision、环境与产物摘要
|-- metrics.py                  # 产出: 应力算例插图 npz 导出 (图 5.9 主应力面板)
|-- compliance_reanalysis.py    # 产出: 固支梁六个冻结设计 x 六种离散的柔顺度交叉再分析 (论文 5.2.1 节)
|-- bearing_reanalysis.py       # 产出: 轴承冻结设计 x 四种离散的柔顺度交叉再分析 (论文表 5.4)
|-- bearing_h_locking_probe.py  # 产出: 轴承全实体域 h 收敛闭锁考察, 两档 nu x 四级网格 x 四种离散 (论文图 5.5)
|-- discretization_probe.py     # 产出: 应力算例冻结构型重分析, 应力比与牵引跳量 (图 5.8 / 5.10 / 5.11)
|-- plots/                      # 唯一子目录: 每张图一个模块, 文件名为 <算例族>_<产物>
|                               # (成图落在 outputs/figures/, 故不与之同名)
|   |-- _base.py                # 八个成图模块的共用底座: 字体/vtu/产物定位/落盘 (论文口径: Palatino, 版心 5.9 in; 分片常值云图只出 PNG 600 dpi, 线图出 PDF+PNG)
|   |-- compliance_topology.py compliance_convergence.py         # 图 5.2 / 5.3
|   |-- bearing_solid_h_convergence.py bearing_topologies.py      # 图 5.5 / 5.6
|   |-- stress_cubic_topologies.py stress_cubic_convergence.py   # 图 5.8 / 5.9: LFEM p=3 与 HZMFEM k=3 主对比
|   `-- stress_hz_orders_topologies.py stress_traction_jump.py   # 图 5.10 / 5.11: HZMFEM k=2,4 构型; 六个构型实体带的牵引跳量
|                               # 图 5.1 / 5.4 / 5.7 是论文侧 TikZ 示意图, 不在本目录; 图号以各模块 docstring 首行括注为准
`-- outputs/                    # 运行产物, 不提交
```

## 已注册算例

| case id | `role` | 模型 | 组装入口 | 论文位置 |
| --- | --- | --- | --- | --- |
| `compliance-fixed-fixed-half` | `optimization-baseline` | `FixedFixedBeamHalfDomain2d` | `pipeline.py: build_fixed_fixed_*` | 5.2.1 节 / 图 5.2~5.3 |
| `bearing-compressible` | `incompressible-baseline` | `BearingDevice2d` | `pipeline.py: build_bearing_*` | 5.2.2 节 / 图 5.5~5.6、表 5.4 |
| `bearing-incompressible` | `incompressible-study` | `BearingDevice2d` | `pipeline.py: build_bearing_*` | 5.2.2 节 / 图 5.5~5.6、表 5.4 |
| `cantilever-middle-2d-stress` | `stress-constrained` | `CantileverMiddle2d` | `pipeline.py: build_stress_*` | 5.2.3 节 / 图 5.8~5.11 |

## 参数注册约定

每个 `[[cases]]` 必须包含三类独立参数:

- `[cases.model]`: `name` 和 `parameters`, 选择 `soptx.problems` 中的物理模型并给出载荷、材料、平面假设等参数。
- `[cases.discretization]`: 三角剖分方式 `mesh_type` (取值见 `pipeline.MESH_TYPES`, 由 `pipeline.create_mesh` 分派)、网格剖分、受控比较阶次 `comparison_orders`、可选的 `supplementary_orders` (只放宽 `--order` 白名单, 不进缺省与 `--full`)、角点松弛和线性求解器。
- `[cases.optimization]`: 体积分数、过滤器、材料插值、优化器 (`oc` / `mma` / `al_mma`) 及其迭代参数。

`mesh_type` 的两种取值:

| 取值 | 对角线规则 | 用途 |
|---|---|---|
| `triangle-checkerboard` | `(i+j)` 偶数 `/`, 奇数 `\` | 固支梁、悬臂梁的注册值 (制造解脚本同用此剖分); 轴承算例的棋盘格对照 |
| `triangle-single-diagonal-symmetric` | 左半 `/`, 右半 `\`, 两个顶角落翻转 | 两条轴承 case 的注册值; 低阶位移元体积闭锁对照 (与左右对称问题同对称性) |

生成器见 `soptx.mesh.create_huzhang_checkerboard_mesh` / `create_huzhang_symmetric_single_diagonal_mesh`。

材料插值对象 `interpolation` 取三值: `E` 只插值 Young 模量 (Poisson 比固定为实体值), `E+nu` 同时按论文式 (4.3) 插值 Poisson 比 (只允许近不可压缩材料 `nu >= 0.49`, 可压缩材料上直接报错), `auto` 按材料自动决定。两条轴承 case 已显式登记 (`bearing-compressible` 为 `E`, `bearing-incompressible` 为 `E+nu`), 参数 `nu_penalty_factor` / `void_poisson_ratio` 走 `--override`。

论文 5.1 节的制造解收敛阶 (表 5.1 / 5.2) 不是优化算例, 不进注册表: `manufactured_convergence.py` 把参数写成文件顶部常量, 一次跑完 $k=1,\dots,4$, 写出 `outputs/manufactured_convergence/summary.json` (顶层一个溯源戳记) 与 `table5_1.md` / `table5_2.md`。`--degree` 只回显, 不覆盖 `summary.json`。

```bash
python experiments/paper_topopt_huzhang/manufactured_convergence.py
```

## 调用方式

`run.py` 解题并落盘, `compare.py` 在其产物之上做二次数值试验与成图: 五个动词都会按冻结设计重新
组装并求解 (代码中的 `REANALYSIS` 集合), 给出 `run.py` 不产的数据 —— 表 5.3 / 5.4 的冻结设计交叉再分析、图 5.5 的
全实体域 h 收敛、应力算例的插图场导出与冻结构型重分析, 都只能由这里产生。两者都以 `--case` 驱动,
但 case 是两套命名:

| | `run.py` | `compare.py` |
|---|---|---|
| 一条 case | 一道要解的题 | 一件要产出的成果 (多数需重新求解) |
| id 来源 | `cases.toml` 的 `[[cases]]` | `plots/` 下声明了 `SOURCE_CASE` / `REQUIRED_RUNS` 的模块, id 取模块文件名 |
| 派发 | 一律归 `driver.py` | 由模块自描述决定, 绘图数据缺失或过期时自行触发冻结重分析 |
| 列出 | `run.py --list` | `compare.py --list` |

调用方拿到 id 即可运行; 不支持按文件路径直接执行 `driver.py`。

省略 `--analyzer` / `--order` 时只展开一个组合: 方法取 `huzhang` (该 case 未注册时退回 `methods` 首项), 阶次取 `comparison_orders` 的最小值。论文那套方法/阶次对比是显式动作, 用 `--full` 展开成 `methods x comparison_orders` 全集, 也可以用 `--analyzer all` / `--order 2 3` 精确指定; 注册表里的 `methods` / `comparison_orders` 同时是白名单, 越界直接报错。

覆盖参数有两个通道:

| 通道 | 覆盖的字段 | 直接报错的情形 |
|---|---|---|
| 具名开关 `--mesh-type` / `--interpolation` / `--analyzer` / `--order` / `--nx` / `--ny` / `--optimizer` / `--filter-type` | 最常用的字段 | 与 `--override` 重复指定同一字段 |
| `--override KEY=VALUE` (可重复) | `[cases.discretization]` / `[cases.optimization]` 的键, 另加运行组合维度 `analyzer` / `order` (多值用逗号分隔) | 字段名写错、类型转换失败 |

取值按配置对象现有取值的类型转换, 不静默取一边。覆盖参数针对单条 case 的注册值, 因此只允许配合单个 `--case`。

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

核对走 `compare.py discretization-probe` (下节): 构型冻结后在自身离散下重解, 不施加豁免,
被动实体区的真实读数一并取回。

## 冻结构型重分析与牵引跳量 (图 5.8 / 5.10 / 5.11)

`discretization-probe` 把论文 5.2.3 节的六份最终构型 (LFEM `p=2..4`, Hu--Zhang `k=2..4`,
pad 1.5 mm, 判据集合 $\rho \ge 0.5$) 冻结, 各在优化所用的离散下重解一次状态方程, 导出
逐单元的表观应力比、约束值 $g$ 与内边法向牵引绝对跳量 $A_e$ (求值见 `discretization_probe.py` 第二节)。
逐单元应力约束要有意义, 前提是"该单元的应力"良定义: Hu--Zhang 的应力是
$H(\mathrm{div}, S)$ 协调的原始变量, $[[\sigma \cdot n]] \equiv 0$; LFEM 的应力由位移
求导, 跨边有跳。$A_e$ 取单元各内边归一化跳量 RMS 的最大值, 统计限于实体带
($\rho > 0.9$ 且不在被动实体区)。跳量一律测**表观应力**: 连续介质中真实牵引跨面连续,
而实体应力 $\sigma^{\mathrm{sol}} = \sigma^{\mathrm{app}} / m_E$ 因 $m_E$ 逐单元常值必然跳,
测它没有意义。

两条硬门, 不过则进程返回码为 1:

- Hu--Zhang 的相对跳量必须 $< 10^{-10}$ (协调性的直接后果)。不满足即判定跳量求值有错;
- 判据集合上的最大 $g$ 必须复现 `summary.json` 的 `max_relative_violation_solid_region`,
  否则掩码或阈值口径与优化器不同。

```bash
# 默认重分析六份构型
~/miniconda3/envs/ihpcm/bin/python compare.py discretization-probe
# 指定构型 (可重复)
~/miniconda3/envs/ihpcm/bin/python compare.py discretization-probe --design <运行目录名>
```

产物落在 `outputs/cantilever-middle-2d-stress/postprocess/discretization_probe/`:
`<design>__probe.json` (跳量统计 + 验收门 + provenance + 构型 sha256)、`<design>__fields.npz`
(逐单元场, 供图 5.8 / 5.10 / 5.11 读取)、`<design>__<disc>.vtu` (同一批场, 供 ParaView 查看)。
