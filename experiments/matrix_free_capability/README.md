# Matrix-Free 能力证据的成图数据

本目录把图 2「Matrix-Free 算子 / Krylov 求解」的四格数据, 从散落的手工运行收成一次可调度、可校验、可入库的采集。四格各有注册算例。

本目录不实现任何物理或数值算法: 它按 [`cases.toml`](cases.toml) 以子进程调用 `examples/` 下已有的脚本, 再把产物收成带溯源的快照。物理模型在 `src/soptx/problems/`, 算法在 `src/soptx/fem/`, 阈值契约在 `tools/matrix_free_evidence/contract.py`。

## 数据组键名与图面位置错开

`cases.toml` / `config.py` / 快照 `panels.*` 里的 `panel` 是**数据组**标识, 不是图面位置。自 `2026-08-24` 版式改造起两套编号错开:

| 数据组 | 内容 | 图面位置 | 命令 | 本目录表名 | 切图文件 |
|---|---|---|---|---|---|
| `c` | 设备对照(CPU / 单卡 GPU) | **(d)** | `run.py --panel c` | `表 c-*` | `..._panel_d.{png,svg}` |
| `d` | 进程级 MPI 强扩展 | **(c)** | `run.py --panel d` | `表 d-*` | `..._panel_c.{png,svg}` |

键名按采集顺序固定(改名会作废全部历史快照), 图面位置按阅读顺序排, 两者没有理由必须一致。引用时说清是哪一套: 说「图 2(d)」指图面, 说「`panels.d`」指快照字段。

有效访存带宽不是第五组数据, 它由数据组 `c` 的耗时现场换算, 只进 (d) 的结论框与 [`results_analysis.md`](results_analysis.md) §5.4 的表 c-2。

## 文件职责

```text
experiments/matrix_free_capability/
|-- cases.toml         # 数据点注册表: 跑哪条命令、产物落在哪
|-- config.py          # cases.toml 的加载、校验与 argv 构造
|-- provenance.py      # git revision / dirty / 时间戳 / 环境 / 源文件 sha256
|-- collect.py         # 产物 -> 快照; 门禁判定与跨源一致性核对
|-- report.py          # 快照 -> results_analysis.md 里的 <!-- BEGIN generated --> 区间
|-- run.py             # CLI 调度入口
|-- diag_body_force.py # 一次性诊断: 体力向量组装的分步峰值与分块原型 (不进 cases.toml)
|-- results_analysis.md # 按图面四格组织的证据报告: 当前结论、读法与未闭环待办
|-- figure_data/
|   |-- fig2_data.json                          # 入库快照 (由 .gitignore 的否定规则放行)
|   |-- fig2_matrix_free.{png,svg}              # 合并图 (2x2): 进申请书的那一张
|   |-- fig2_matrix_free_panel_{a,b,c,d}.{png,svg}  # 单格图: 供本目录文档按格引用
|   `-- fig2_matrix_free_panel_c_internal.{png,svg}  # (c) 内部拆解图: 不进申请书
`-- outputs/           # 原始运行产物, 忽略
```

`figure_data/` 同时放快照和定稿成图。成图不是本目录画的 —— 绘图代码在申请书仓库的 `assets/make_figs.py`, 与图 1~4 共用配色和坐标轴样式层, 搬过来只会让四张图各画各的; 这里放的是副本, 让 [`results_analysis.md`](results_analysis.md) 能对着图讲。单格图与合并图共用同一批 `draw_panel_*` 函数、只换画布(单格 4.6 in 宽, 合并图 2x2 里每格约 2.9 in), 因此不会互相漂移。每种图存 PNG 与 SVG 两份: Markdown 引 SVG, DOCX 只能插 PNG(`python-docx` 的 `add_picture` 不认 SVG)。`outputs/figures/` 下每次重绘都变的中间产物仍整体忽略。

[`diag_body_force.py`](diag_body_force.py) 不属于调度管线, 不在 `cases.toml` 里, 也不写快照。它服务于 [`results_analysis.md`](results_analysis.md) §3.3 的一条归因结论: 表 b-2 的 `load` 阶段增量是累积高水位之差, 会被 `operator` 更高的水位遮盖, 因此单靠主 benchmark 量不出体力组装的真实开销。该脚本刻意不装配刚度算子, 让后续每一步的增量都可见, 并附一个按单元分块的原型用于量化「能省多少」。用法见其模块 docstring。

## 三处设计取舍

- **为什么不放在 `examples/` 下。** 采集要横跨 `lagrange_elasticity`(收敛阶)与 `matrix_free_elasticity`(EA/FA 对照)两个目录取数, 放进任一个都违反「example 目录不能交叉污染」。体裁上它与 `experiments/huzhang_topopt_paper/` 的 `make_fig4_*.py` 相同, 都是面向外部交付物的成图 campaign。
- **为什么每个数据点独占一个进程。** (b) 的峰值内存取 `resource.getrusage(RUSAGE_SELF).ru_maxrss`, 是进程级高水位、无法按对象归因; 同一进程里先建 FA 再建 EA, 测出的 EA 峰值是被 FA 抬高过的。故 (b) 的十个数据点(FA/EA 各五档)一个都不能合并。
- **TOML 描述运行, Python 描述读取。** `cases.toml` 只说「跑什么」; 取哪些字段、门禁怎么判写在 `collect.py` 里, 因为那里带判断逻辑, 写不进 TOML。

## 命令

```bash
# 列出全部数据点及产物状态
python experiments/matrix_free_capability/run.py --list

# 只打印将要执行的命令, 不运行 (建议先看一遍)
python experiments/matrix_free_capability/run.py --all --check-only

# 跑某一格 / 某个数据点 / 全部; 跑完自动 collect
python experiments/matrix_free_capability/run.py --panel a
python experiments/matrix_free_capability/run.py --case b-fa-n64
python experiments/matrix_free_capability/run.py --all

# 从已有产物收快照, 不重跑
python experiments/matrix_free_capability/run.py --collect
```

⚠️ `--all` 里最重的是 `b-fa-n80`(1,594,323 自由度, 峰值 `33.0 GiB`, 数分钟), 次重的是 `b-fa-n64`(823,875 自由度, `17.0 GiB`)。本机可用内存 47 GiB, 单进程跑得下, 但 `b-fa-n80` 已占去七成, 绝不要与其他大内存进程并行。

⚠️ `c-dev-*` 四档跑在单块 RTX 5080(16 GiB)上, 最细的 `c-dev-n64` 峰值分配 12.8 GiB, 占整卡 80%。不要自行往 `cases.toml` 加更细的档 —— `n=80` 试过一次, WDDM 驱动不报 OOM 而是换页到主机内存, 表现为利用率 100% 但速度掉两个量级。

## 逐档数值去哪看

本目录只管把数字取对、记住它出自哪个 revision。逐档实测值、口径声明与结论边界在两个 example 目录里:

- 收敛阶与六种装配/求解器配置对照: [`examples/lagrange_elasticity/results_analysis.md`](../../examples/lagrange_elasticity/results_analysis.md)
- EA/FA 功能验证、效率对照与进程峰值内存: [`examples/matrix_free_elasticity/results_analysis.md`](../../examples/matrix_free_elasticity/results_analysis.md)

图面各格的实测数据、口径与读法边界见 [`results_analysis.md`](results_analysis.md)。

## 采集契约与快照结构

[`figure_data/fig2_data.json`](figure_data/) 约 28 KB, 入库(`.gitignore` 第 27 行的否定规则放行, 是仓库里第一条 `!` 规则)。

| 顶层字段 | 内容 |
|---|---|
| `schema_version` | 结构版本, 消费方据此判断兼容性 |
| `figure` | 图号、标题、四问、消费方路径 |
| `provenance` | revision / branch / dirty / 采集时间 / 主机 / 平台 / Python / NumPy / FEALPy / 物理内存总量 |
| `reproducible` | 有 revision 且 `dirty = false` 时为 `true` |
| `panels.a` / `.b` / `.c` / `.d` | 四组数据, 均为 `status = "measured"`; 某组没有注册 case 时退回 `status = "placeholder"` 与原因 |
| `sources` | 22 条: case id、role、仓库根相对路径、`sha256`、字节数 |
| `gate_failures` | 门禁失败项; 空列表即全过 |
| `notes` | 非致命提示 |

溯源由采集侧补, 因为产出侧没有: `stage1_evidence.py` 与 `benchmark_cpu_ea.py` 不写 `environment` 块, 原始产物无法自证出自哪个 revision。本次快照的实际溯源渲染在 [`results_analysis.md`](results_analysis.md) §6.4。

## 门禁清单

由 [`collect.py`](collect.py) 在采集时判定, 失败项进 `gate_failures` 并使 `run.py --collect` 返回非零。

| 对象 | 判据 | 挡住什么 |
|---|---|---|
| 阈值本身 | `MINIMUM_FINAL_L2_ORDER`、`EA_FA_SOLUTION_RELATIVE_TOL` 与 `contract.py` 一致 | 本目录复述的阈值与契约漂移 |
| FA 误差链 | `passed = true`; 档数 `= 5`; `base_subdivisions = 4`; 末段阶 `>= 1.5` | 忘传 `--base 4` 得到的另一条链; 档数不足看不出趋势 |
| EA 环 | 各维 `passed = true`; EA/FA 解相对差**逐档** `< 1e-9` | matrix-free 与参考路径已不同解 |
| EA 环显式参考 | 单 rank 算例 CG/spsolve 直接解相对差最大值 `< 1e-8`(`EXPLICIT_SOLUTION_RELATIVE_TOL`) | stage-1 回归或产物被替换时, 表 a-2「CG 解已站在直接解上」失去支撑而不自知 |
| 跨链核对 | FA 与 EA 在 `n = 8/16/32` 的相对 L2 误差值逐档比对, 最差档相对差 `> 1e-8`(`EA_FA_ERROR_CHAIN_RELATIVE_TOL`)则记 note | 两条链已不是同一个离散 |
| 峰值内存 | 逐项核对 `operator_level`/`resolution`/`assembly_method`/`mesh_type`/`dimension`/`degree`/`mode`; `solved` 与 `cg_converged` 为真 | 旧产物被当成新的读进来; 漏传 `--assembly-method fast` |
| 峰值内存配对 | FA 与 EA 档位集合一致; 两者 CG 迭代数逐档相同 | 半套数据成图; 两个层级实为不同算子 |
| 峰值内存分阶段 | 每个产物都含 `STAGE_ORDER` 的全部阶段(`baseline`/`mesh`/`operator`/`load`/`bc`/`solve`), 缺则直接抛 `CollectError` | 旧插桩产物(`load` 与 `bc` 合并)被静默当成新口径读进表 b-2 与表 b-3 |
| 设备对照身份 | 逐项核对 `mode`/`resolution`/`operator_level`/`assembly_method`/`mesh_type`/`dimension`/`degree`/`backend`; 两侧 `cg_converged` 为真; 产物自判 `gate_passed` 为真 | 旧产物或改过参数的产物被当成本 case 的; CPU 侧偷偷换成 `numpy` 后端却仍当「设备对照」读 |
| 设备对照等价性 | CPU 与 GPU 解相对差 `<= 1e-9`(`DEVICE_SOLUTION_RELATIVE_TOL`); 两侧 CG 迭代数逐档相同 | 两边算的不是同一个问题 —— 那样加速比没有意义 |
| 设备对照同轴 | 设备对照的自由度序列是 (b) 的**前缀**(同起点、同档位, 允许早一步停); 各档 warmup/repeats 一致、同一块 GPU | 图面 (b)(d) 横轴错位或交错却上下对读。⚠️ 判据是前缀而非子集 —— 设备对照因单卡显存止于 `n=64`, 但跳档或换起点仍必须报错 |
| MPI 强扩展身份 | 逐项核对 `ranks`/`mode`/`model`/`mesh_type`/`assembly_method`; 各档 `converged` 为真 | 旧产物混入; ⚠️ 特别是 `--assembly-method` 曾在 MPI 分支上静默失效, 核对产物字段是唯一能挡住它的地方 |
| MPI 强扩展代数不变性 | 各档自由度必须相同(否则不是强扩展); CG 迭代数逐档相同; 真实相对残差跨档相对离散 `<= 1e-3`(`MPI_RESIDUAL_RELATIVE_TOL`) | 分区改变了代数结果却仍被当成加速。残差判据故意放松到 `1e-3` —— 归约次序随分区变, 末位浮点差异是正常的, 硬判等值会误报 |
| MPI 强扩展口径 | 必须有 1 进程档(强扩展的分母); 各档 warmup/repeats 一致 | 缺分母时加速比无定义; 不同计时口径的秒数被放进同一条曲线 |

`EA_FA_ERROR_CHAIN_RELATIVE_TOL` 不进第一行的契约核对: 它比对的是 `examples/lagrange_elasticity` 的 FA 链与本管线 EA 三档两个不同来源, 只在采集时发生, 而 `contract.py` 覆盖的是 stage-1 管线自身的门禁。该阈值与旧判据「最少吻合 8 位有效数字」数学等价。

⚠️ 门禁不校验「这次是不是刚跑的」。`sources[].sha256` 提供内容指纹, 但判断产物新旧仍需人 —— 彻底的解法是让上游脚本自己写 `environment` 块(见「未闭环事项」第 2 条)。

## 不属于本图的旧产物

`examples/matrix_free_elasticity/outputs/` 下的 `ea_strong_profiled_{1,2,4,8}.json` 与 `ea_weak_{1,2}.json`(`2026-08-18`)是 `mpi-ea-strong` / `mpi-ea-weak` 的真实产物, 四档 CG 迭代数恒为 `760`、相对残差 `9.6e-11` —— 分区没有改变代数, 正确性是干净的。但它们不能进本图, 三条理由:

| 理由 | 事实 |
|---|---|
| 问题不同 | 2D tri `128x128`、`33,282` 自由度; 本图是 3D tet, 最大 `823,875` 自由度 |
| 规模过小导致曲线拐头 | 每 rank 降到约 `4` 千自由度后局部核不再下降, P=8 反比 P=4 慢(`0.896 s` vs `0.619 s`) —— 问题过小的伪影, 不是方法性质 |
| 不在采集契约内 | 产物落在 `examples/.../outputs/`, 未过 `collect.py` 门禁 |

引用这批 2D 结论必须回到该目录核对, 也不得把它们与图面 (c) 的 3D 五档写进同一句话。同理, `examples/gpu_elasticity/` 那批旧 GPU 结果走 2D 三角形 + 全组装路径, 与图面 (d) 不是同一个算子, 见 [`results_analysis.md`](results_analysis.md) §5.5。

## 未闭环事项

本节只列未闭环的; 已闭环的口径与边界写在 [`results_analysis.md`](results_analysis.md) 各格的「读法与边界」里。

1. **当前快照不可复现。** 工作区 dirty, `reproducible = false`。正式投递前须在 clean revision 上重跑 `run.py --all`。
2. **上游脚本缺 `environment` 块。** `stage1_evidence.py` 与 `benchmark_cpu_ea.py` 都不写该块, 采集侧只能记「采集时」的 revision, 不是「运行时」的。
3. **`--mumps-sym` 未纳入门禁校验。** 该字段已随 `_collect_fa_chain` 进快照并渲染进表 a-1 的配置行, 但 `collect.py` 只如实转录、不校验它与 [`cases.toml`](cases.toml) 是否一致。要闭环得把 case 的 `args` 与产物的 `solver_options` 对一遍。
4. **2D 从 SciPy 改判 MUMPS 的对照只做过一次, 未纳入门禁。** 两次结果的逐档绝对 `L2` 误差相对差为 `1.2e-15` ~ `2.3e-11`(由粗到细), 末段观测阶 `1.9951778248379` -> `1.9951778248708` —— 求解器不改变离散在本链上是实测的。但旧产物已不再生成, 要复核得手动重跑 `--solver scipy` 再逐档比; 做成常驻证据应在 `cases.toml` 加一个只跑不入图的对照 case。
5. **成图靠人工同步。** `figure_data/` 下的图是从申请书仓库 `assets/` 拷来的副本。各图之间不会漂移(同一次渲染切出), 但它们与本目录的表之间会 —— `make_figs.py` 重绘后必须重新拷贝, 否则图落后于表。
6. **三处并行配置未落盘。** 均属记录缺口而非结论缺陷:

   | 缺口 | 现状 | 影响 |
   |---|---|---|
   | 设备对照的 `torch_threads` | 脚本已写该字段(`benchmark_device_ea.py:161`), 但当前快照 `panels.c.cpu.torch_threads` 四档全为 `null` —— 快照早于该行。重跑 `c-dev-*` 即可补上 | 事后无从从产物本身判定分母是单核还是满核 |
   | (a) EA 环的 `backend` | `stage1_evidence.py` 不调 `set_backend`, 走 fealpy 硬编码默认 `numpy`, 但产物不记该字段 | 当前可由代码反推, 一旦上游改默认值, 历史产物就无法区分 |
   | (a) FA 链的 BLAS 线程数 | 运行时既不设也不记, 而实测该段默认会拉起最多 32 条 OpenBLAS 线程 | 同一条链在不同机器/不同 `OMP_NUM_THREADS` 下耗时可差 `1.2` 倍以上, 跨机引用会静默失真 |

7. **跨格对照表由人工维护。** [`results_analysis.md`](results_analysis.md) §4.4 那四行取自两个不同的 role(`mpi-strong` 与 `device-speedup`), `collect.py` 没有跨格提取器, 是全库唯一一处手抄数字的表。要闭环应加一个跨格核对函数, 按「同 `n`、同装配、同迭代数」把两个 role 的记录对起来并断言残差同量级。
8. **MPI 五档是单次采样。** 实跑用 `--warmup 0 --repeats 1`(脚本默认 `1` / `3`), 秒数不带误差棒。曲线形状与正确性结论(迭代数逐档恒为 `493`)不受影响, 但这批数自 `2026-08-24` 起已经画在图面 (c) 上, 正式投递前须按默认口径重跑 `run.py --panel d` 并重绘图 2。
9. **`dot_fn` 的强制同步未量化。** `matrix_free/krylov.py:146-147` 的 `float(bm.sum(x * y))` 使 GPU 侧每次求解被强制同步约 `986` 次(`n=64`)。这是一处不需自定义 kernel 就能改的点, 但修掉它能带来多少加速尚无实测 —— 需要 Nsight 时间线才能定量, 在此之前不得写成收益。
