# Hu--Zhang 拓扑优化投稿论文复现实验

本目录保存投稿论文的可执行拓扑优化算例。物理模型位于 `src/soptx/problems/elasticity/`, 本目录只通过 `cases.toml` 选择模型、配置离散与优化参数, 并调用当前 `soptx.topology` 的公共接口。

## 多算例结构

```text
experiments/huzhang_topopt_paper/
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
|-- plots/                      # 唯一子目录: 每张图一个模块, 文件名为 <算例族>_<产物>
|                               # (成图落在 outputs/figures/, 故不与之同名)
|   |-- _base.py                # 八个成图模块的共用底座: 字体/vtu/产物定位/落盘
|   |-- compliance_topology.py compliance_convergence.py compliance_k1_comparison.py
|   |-- bearing_topologies.py bearing_highorder_topologies.py
|   `-- stress_topologies.py stress_convergence.py stress_highorder_topologies.py
|                               # 论文图号只在各模块 docstring 首行的括注里
`-- outputs/                    # 运行产物, 不提交
```

模块平铺在实验根目录, 与 `experiments/` 下其余八个实验同构 (`run.py` / `config.py` /
`provenance.py` / `pipeline.py` / `report.py` / `metrics.py` 等名字全部沿用
`experiments/` 下其它实验已有的词汇, 不新造模块名); 只有 `plots/` 单独成目录, 因为它是八个
同构模块的集合, 且成图落在 `outputs/figures/`, 与之同名会混淆。

三族物理算例的构造合在 `pipeline.py` 一个文件里, 与 `topopt_simp_ea/pipeline.py` 同构:
`cases.toml` 只存数值, 「数值 -> soptx 对象」的构造 (各物理类构造签名不同、跨字段校验、
局部牵引的 P1 迹投影、AL-MMA 选项) 是代码, TOML 表达不了。两族柔顺度算例共有的 17 个字段
由 `compliance_config_fields` / `validate_compliance_config` 统一解析校验, 各族只写自己的
载荷字段。

产出层按「作用在已有产物上」聚合, 而不是一个动词一个文件: `report.py` 出论文表,
`plots/_base.py` 收拢八个成图模块的共用成图口径,
`metrics.py` 的三个入口 (`run_gradient_check` / `run_frozen_metrics` / `run_export`) 共用同一条
`pipeline` 悬臂梁装配器给出的分析链。`compare.py` 的 `COMMAND_MODULES` 用
`"模块:入口函数"` 记录派发, 故子命令与模块不必一一对应。

`sys.path` 由两个入口通过 `config.bootstrap_source_path()` 统一注入, 顶层各模块一律用
`from config import ...` 这类绝对导入, `plots/` 内部改用 `from ._base import ...`
的包内相对导入, **不支持按文件路径直接执行**（如
`python plots/compliance_topology.py`）, 一切经 `run.py` 或 `compare.py`.

算例一律由 `--case` 驱动。case 归属哪个驱动由其 `role` 决定
(`convergence-verification` -> `convergence.py`, 其余 -> `driver.py`),
调用方拿到 id 即可运行, 不必先知道用哪个动词:

```text
run.py --list                    # 列出全部算例: id / mesh / analyzer / order / optimizer / role
run.py --case <id>               # 单跑一个 case, 可重复指定
run.py --case <id> <id> ...      # 一次给出多个 case id (也可重复 --case)
run.py --all                     # 跑全部 ready 算例, 按 cases.toml 顺序
run.py --all --dry-run           # 只打印派发计划, 不执行
run.py --case <id> --full        # 展开成注册表声明的完整对比组
```

不带其它参数时参数取自 `cases.toml`, 且**只跑一个组合**。`--case` 之后可直接追加该驱动
认识的覆盖参数, `run.py` 原样转交, 由驱动自身的 argparse 校验:

```text
run.py --case manufactured-stabilized --stabilization none
run.py --case compliance-fixed-fixed-half --analyzer all --order 2
```

具名开关只覆盖最常用的那几个字段; 其余字段走通用通道 `--override KEY=VALUE`, 可重复
给出, 字段名就是 `cases.toml` 里 `[cases.discretization]` / `[cases.optimization]` 的键,
另加运行组合维度 `analyzer` / `order` (多值用逗号分隔), 与 `topopt_simp_fa` /
`topopt_simp_ea` 的 `--override` 同一口径:

```text
run.py --case compliance-fixed-fixed-half --override optimizer=mma
run.py --case compliance-fixed-fixed-half --override penalty_factor=4.0 filter_radius=3.0
run.py --case compliance-fixed-fixed-half --override analyzer=huzhang order=2
run.py --case compliance-fixed-fixed-half --override analyzer=all order=2,3
```

取值按配置对象里现有取值的类型转换; 字段名写错、类型不对, 或与具名开关重复指定同一字段,
都直接报错而不是静默取一边。产物目录第二层是参数标签: `analyzer` 与 `order` 恒进,
其余字段只在覆盖了注册值时按字段名追加 `__<字段>-<取值>`, 探索性运行不会盖掉注册运行的
产物; 实际生效的覆盖同时记进 `summary.json` 的 `overrides` 字段。

**缺省是单值, 不是全集。** 优化算例的 `--analyzer` / `--order` 与收敛算例的 `--degree`
省略时只展开一个组合: 方法取 `huzhang` (该 case 未注册时退回 `methods` 首项), 阶次取
`comparison_orders` 的最小值 —— 裸跑一条 case 就是一次运行, 便于探索与调试。论文那套
方法/阶次对比是显式动作, 用 `--full` 展开成 `methods x comparison_orders` 全集 (如
`compliance-fixed-fixed-half` 的 `2 x [2,3,4] = 6` 组), 也可以用 `--analyzer all` /
`--order 2 3` 精确指定。注册表里的 `methods` / `comparison_orders` 同时是这两个选项的
白名单, 越界直接报错。`summary.json` 按方法与阶次为键增量合并, 因此分多次单跑与一次
`--full` 得到的论文表一致。

两个驱动认的参数并不重叠 (`--stabilization` / `--degree` / `--levels` 只属于收敛验证,
`--analyzer` / `--order` / `--nx` / `--optimizer` 只属于优化), 因此覆盖参数只允许配合
**单个** `--case`; 选中多条时无法确定按哪个驱动解释, 直接报错而不是猜。

**case id 只标研究对象, 不标参数取值。** 纯离散层的扫描 (网格加密、阶次) 不另立 case,
一律用覆盖参数跑; 只有换模型类 (如固支梁的半域对称版) 或换连续问题 (如泊松比从可压缩推到
近不可压) 才另立一条 case。论文里悬臂梁应力算例的细网格结果就是这样产生的:

```text
run.py --case cantilever-middle-2d-stress                              # 注册默认网格 80x40
run.py --case cantilever-middle-2d-stress --nx 160 --ny 80 --order 2 3  # 细网格 160x80
```

网格偏离注册值时进产物目录标签 (`analyzer-<链>__nx-<值>__ny-<值>__order-<k>`),
两套网格的产物并存互不覆盖;
但 `run.py --all` 只按各 case 的注册默认网格跑缺省组合, 上面第二条命令必须单独执行。

作用在已有产物上的后处理归另一个入口 `compare.py`, 与算例选择无关:

```text
compare.py --list             # 列出产物 case: case-id / source-case / 说明
compare.py --case <case-id>   # 整理出一件产物, 如 compliance-topology
compare.py figure <图号>      # 尚未迁成产物 case 的图: 5.5 / 5.7 / 5.8 / 5.9 / supp-*, 或 --all
compare.py table              # 由 summary.json 重算论文表 5.1 / 5.2
compare.py export [--check]   # 冻结重分析导出插图场数据
compare.py gradients          # 伴随灵敏度的有限差分校验
compare.py metrics            # 冻结设计的论文口径指标复算
```

其中 `export` / `gradients` / `metrics` 会按冻结设计重新组装并求解, 不是纯读产物, 但它们
产出的是论文校验数字而非运行产物 (不写 `outputs/<case>/<run>/`), 故归在 `compare.py`;
`--help` 用 `[重分析]` 标出这三个。老写法 `run.py figure ...` 会被 `run.py`
接住并提示改用 `compare.py`。

一件产物 = 一条 `--case`, 与 `run.py --case` 同一个词: 那边一条 case 是一道要解的题,
这边一条 case 是一件要整理出来的产物。产物 case 不另立注册表, 由 `plots/` 下声明了
`SOURCE_CASE` / `REQUIRED_RUNS` 的模块自描述, case id 取模块文件名, 论文图号只留在各
模块 docstring 首行的括注里, 排版改号不波及命令行。未声明的图仍走 `figure <图号>`,
迁一个删一条, 迁完 `figure` 子命令一并去掉。

配置层的回归测试在 `tests/experiments/test_huzhang_paper_config.py`, 只覆盖不启动求解的纯配置逻辑。

每个 `[[cases]]` 必须包含三类独立参数:

- `[cases.model]`: `name` 和 `parameters`, 选择 `soptx.problems` 中的物理模型并给出载荷、材料、平面假设等参数。
- `[cases.discretization]`: 网格类型 `mesh_type`（台账字段, 供 `--list` 显示；实际网格由 `pipeline.create_huzhang_checkerboard_mesh` 构造）、网格剖分、受控比较阶次 `comparison_orders`、可选的 `supplementary_orders`（只放宽 `--order` 白名单, 不进缺省与 `--full`）、角点松弛和线性求解器。
- `[cases.optimization]`: 体积分数、过滤器、材料插值和 OC 迭代参数。

`pipeline.py` 的 `ASSEMBLERS` 把模型名映射到装配器, 当前注册四个模型:

| 模型 | 装配器 | case id |
| --- | --- | --- |
| `FixedFixedBeamCenterLoad2d` | `build_fixed_fixed_*` | —（不再注册, 见「全域固定梁模型」） |
| `FixedFixedBeamHalfDomain2d` | `build_fixed_fixed_*` | `compliance-fixed-fixed-half` |
| `BearingDevice2d` | `build_bearing_*` | `bearing-compressible` / `bearing-incompressible` |
| `CantileverMiddle2d` | `build_stress_*` | `cantilever-middle-2d-stress` |

每个装配器是一组 `build_config` / `build_analysis_pipeline` / `build_pipeline`, 由 `CaseAssembler`
打包; `driver.py` 只调 `pipeline.assembler_for(model_name)`, 不关心内部布局。新增论文算例时,
先在核心库实现模型, 再增加 `[[cases]]`，最后在 `ASSEMBLERS` 注册装配器。未注册模型会明确报错,
不会隐式复用固定梁参数。

例外: `manufactured-native` (表 5.1, k=3/4 原生格式) 与 `manufactured-stabilized` (表 5.2, k=1/2 矩阵跳量稳定化) 描述的是前向制造解验证参数, 由 `convergence.py` 读取并驱动, 不走 `driver.py` 的算例组装器链路; `run.py` 按 `role` 把它们派给 `convergence.py`, 因此 `--case` / `--all` 都能正常涵盖; 只有直接以模块方式调用 `driver.py` 时才需要注意绕开这两条。

两条 case 的 `stabilization` 字段**直接驱动** `HuZhangMFEMAnalyzer` 的同名入参, 取值 `none` / `matrix_jump` / `vector_jump`。该参数只在 `p <= GD` 时生效; `p >= GD+1` 的原生格式本身稳定, 分析器忽略它, 因此 `manufactured-native` 只能声明 `none`, 写别的会被 `convergence.py` 直接拒绝, 免得跑出一组名义上加了稳定化、实际没加的数。

低阶失稳消融:

```bash
# k=1,2 不加稳定化项, 用于验证论文中「低阶必须稳定化」的论断
python run.py --case manufactured-stabilized --stabilization none

# 换用向量跳量形式
python run.py --case manufactured-stabilized --stabilization vector_jump
```

消融产物写入 `outputs/manufactured_convergence/ablation_<方法>.json`, **不进 `summary.json`**: 后者是 `compare.py table` 生成论文表 5.1 / 5.2 的唯一数据源, 按阶次为键增量合并, 消融结果落进去会静默顶掉同阶次的论文数值。`--stabilization` 只允许配合单个 `--case`。

## 命令

```bash
# 列出全部算例及其缺省组合
python experiments/huzhang_topopt_paper/run.py --list

# 只校验指定 case 的配置和组合, 不运行优化 (--full 才校验完整对比组)
python experiments/huzhang_topopt_paper/run.py \
  --case compliance-fixed-fixed-half --full --check-only

# 固定初始密度下的 LFEM/Hu--Zhang 单次状态对比, 不更新设计变量
python experiments/huzhang_topopt_paper/run.py \
  --case compliance-fixed-fixed-half --analyzer all \
  --mode state-compare --solver scipy

# 单条 Hu--Zhang 计算链
python experiments/huzhang_topopt_paper/run.py \
  --case compliance-fixed-fixed-half --analyzer huzhang --order 3

# 全部论文离散组合 (methods x comparison_orders)
python experiments/huzhang_topopt_paper/run.py --case compliance-fixed-fixed-half --full
```

临时调试可覆盖网格、迭代次数和求解器, 覆盖结果不能作为默认投稿证据:

```bash
python experiments/huzhang_topopt_paper/run.py \
  --case compliance-fixed-fixed-half --analyzer huzhang --order 2 \
  --nx 8 --ny 2 --max-iterations 1 --solver scipy
```

优化模式的每个组合写入 `outputs/<case-id>/analyzer-<链>__order-<k>[__<字段>-<取值>...]/`，包含 `density_final.vtu`、`history.json` 和 `summary.json`。`state-compare` 写入 `outputs/<case-id>/state-comparison-<nx>x<ny>/state_comparison.json`，只包含一次状态分析比较数据。结果一律以各组合目录下的 `summary.json` 为准。

## 证据与成图

运行产物 `outputs/` 整体不入版本控制, 论文数字的溯源依据是产物自带的戳记: 每次落盘时
`provenance.run_stamp()` 把本仓库与 fealpy 副本的 revision/dirty 写进该次运行的
`summary.json` (收敛验证写进 `provenance_by_degree`), 因此过期目录会自报版本, 不会被
后来的汇总统一改标成最新版本。

```bash
# 制造解收敛阶: 按阶次增量写入, 单独重算某个 k 不会覆盖其余阶次
python experiments/huzhang_topopt_paper/run.py --case manufactured-native --degree 3

# 应力算例插图数据: 对每次运行的 density_final.vtu 做冻结重分析并落盘 npz
python experiments/huzhang_topopt_paper/compare.py export
python experiments/huzhang_topopt_paper/compare.py export --check   # 只校验, 不覆盖
```

`provenance.py` 记录 Git revision、工作区是否 dirty、环境与产物 sha256; 快照的 `reproducible` 字段为 `false` 时（工作区不干净或取不到 revision）该批数字不能直接作为定稿证据。插图统一经 `report.save_figure` 落盘到 `outputs/figures/`, 论文图件目录存在时同步一份, 可用环境变量 `HUZHANG_PAPER_FIGDIR` 覆盖。

## 全域固定梁模型（不再注册, 仅作对称降维参照）

论文 5.2.1 节在左半计算域上求解, 故全域 case 已从注册表移除; `FixedFixedBeamCenterLoad2d` 仍留在 `pipeline.ASSEMBLERS` 中, 作对称降维一致性核对的参照, 需要重跑时按下述参数临时注册一条 case 即可。它使用全域 `160 mm x 20 mm`、两端固定、底边中点载荷 `P=-3 N`、`E=30 MPa`、`nu=0.4`、平面应力、`Vbar=0.4`、`rmin=2.4 mm`。每个受控比较阶次取 `comparison_orders = [2, 3, 4]`（$k=1$ 因位移空间为 $P_0$ 缺失刚体旋转模态不适用于拓扑演化，详见 [`docs/fem/huzhang-mixed-fem-implementation.md`](../../docs/fem/huzhang-mixed-fem-implementation.md)），均成对运行 `LFEM p=k` 与 `Hu--Zhang k=k`，并统一使用积分阶 `q=2k+2`。

默认 `hx=1 mm`, 而 `load_width=1 mm` 的连续载荷以中点为中心，会切过相邻两条底边的半边。Hu--Zhang 的牵引强施加不能精确表达边内跳变，网格对齐也救不了（跳变点上的顶点自由度是单值的）。因此两条分析链使用的不是原始阶跃牵引，而是它在底边连续 P1 迹空间上的 L2 投影：`build_problem(parameters, n_cells=nx)` 调用 `soptx.fem.project_patch_traction_to_p1_trace` 得到该投影，再注入 `FixedFixedBeamCenterLoad2d(traction=...)`。投影精确保持合力 `P`，且能被 LFEM 的边界积分与 Hu--Zhang 的迹插值同时精确重现，所以两种方法的差异可以归因于离散格式本身。

结构合力核查（从解出的场反算真正传进结构的力：Hu--Zhang 取 $\int_{\Gamma_N}\sigma_h\cdot n$，LFEM 取支座反力）与密度无关，因此不在本目录重复：它由 [`examples/huzhang_elasticity/concentrated_load_demo.py`](../../examples/huzhang_elasticity/concentrated_load_demo.py) 在实体材料（`rho=1`、无材料插值）下承担。本目录的 `--mode state-compare` 只保留与密度相关的部分：`rho=0.4` + msimp 插值下的柔顺度对比、体积分数、真相对残差与能量恒等式诊断。

## 左半域对称降维算例

`compliance-fixed-fixed-half` 显式选择 `FixedFixedBeamHalfDomain2d`，采用左半设计域对称降维设置。它把完整域关于竖直中线 `x=80 mm` 对称降维为左半域 `80 mm x 20 mm`：左端 `x=0` 完全固支，对称面 `x=80` 施加对称约束（法向位移 `u_x=0` 与切向牵引 `sigma_xy=0`），底部对称面底端施加局部牵引。

对称面的离散处理：

- LFEM 走分量级 Dirichlet：`is_dirichlet_boundary_dof_x` 同时标记左端与对称面（`u_x=0`），`is_dirichlet_boundary_dof_y` 只标记左端（`u_y=0`），对称面切向位移自由。
- Hu--Zhang 走分量级本质边界：`is_symmetry_boundary` 标记对称面，`HuZhangMFEMAnalyzer` 仅强加对称面切向牵引分量 `sigma_nt=0`，法向牵引 `sigma_nn` 自由；法向位移 `u_n=0` 在切向牵引固定后由变分自然满足，位移边界项贡献为零。

载荷名义区间 `load_width=1 mm` 关于对称面对称，左半域只保留其左半边，因此合力自动为完整域的一半 `P/2=1.5 N`；左半域柔顺度为完整域的一半。`run.py` 的终端输出、`history.json`、`summary.json` 与 `state-compare` 均保存计算域（半域）柔顺度，与半域 `density_final.vtu` 对应；摘要以 `compliance_domain = "half"`、`full_structure_factor = 2.0` 标明口径。仅在 `compliance-convergence` 和 `compliance-topology` 中将柔顺度显式乘以 2 展示完整结构，后者同时镜像补全拓扑，密度值与体积分数不变，且不回写运行产物。旧摘要若缺少口径字段，需先核查迁移，不能直接当成半域值再次乘 2；已启动或暂停的旧进程不会自动加载此协议。网格剖分 `80 x 20`（对应完整域 `160 x 20` 的一半），`filter_radius=2.4 mm` 保持物理长度不变。受控比较阶次、积分阶、材料与优化参数与全域参照完全一致，两组结果应给出相同的完整结构柔顺度（LFEM `C≈31.94–32.07`、Hu--Zhang `C≈32.32–33.17`，差异仅来自对称边界处理的离散误差；实测收敛值见 `results_analysis.md`）。

### 对称降维的一致性验证

对称面稳定化在均匀密度与收敛设计下表现不同，分两层验证：

**单次状态分析（`--mode state-compare`，`rho=0.4` 均匀密度）**：

- `k=3, 4`（高阶原生格式，无跳量稳定化）与 LFEM 全阶：左半域结果与完整域**逐位一致**（`k=3` 下 LFEM `184.017`、Hu--Zhang `184.080` 均与完整域相同）。
- `k=2`（低阶跳量稳定化）：Hu--Zhang 左半域与完整域存在约 `3%` 的偏差。根源是稳定化格式下的对称降维不等价：完整域把对称面 `x=80` 当作内部面，跳量惩罚施加的是两侧对称梯度之差（对称时仅法向位移 `u_x` 有跳量，切向为零）；左半域把对称面当作边界，仅强加切向牵引 `sigma_nt=0`，未施加等价的分量级跳量惩罚（只惩罚法向位移跳量 `u_x`），因此均匀密度诊断下缺少对称面稳定化。曾尝试补全对称面分量级跳量惩罚（仅法向位移）与法向位移 `u_x=0` 的强加，两者均无法消除该偏差；且 `u_x` 强加会破坏 `k=3,4` 左半域与完整域的逐位一致性，已回退。诊断确认完整域解本身完全对称、对称面内部面稳定化对柔顺度无贡献，故该偏差是低阶稳定化格式下对称降维的固有数值差异，不是可单点修复的缺陷。

**完整 OC 优化（收敛设计）**：`k=2` Hu--Zhang 左半域与完整域各跑一条优化链（`--max-iterations 200 --solver scipy`），收敛柔顺度分别为 `33.071` 与 `33.171`（相对差异 `0.30%`），体积分数均为 `0.400`；以 `rho>0.5` 二值化的最终拓扑**3200/3200 单元完全一致**（最大密度差 `0.019`，平均差 `1.6e-4`）。优化收敛到 0/1 密度后，对称面附近由实体/空腔结构自身承载法向位移约束，均匀密度下的稳定化偏差不再放大，因此 `k=2` 的收敛设计与完整域一致。

结论：左半域 case 的 `k=2,3,4` 收敛设计均可作为投稿证据；均匀密度诊断下 `k=2` 的约 `3%` 偏差仅存在于优化前的固定密度分析，不影响收敛设计。