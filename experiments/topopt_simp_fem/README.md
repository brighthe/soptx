# 传统有限元拓扑优化: EA 与 FA 对照

本目录验证传统有限元 (Q1 六面体, Lagrange 位移元) 下 EA 与 FA 两种刚度算子层级的拓扑优化结果一致性与计算代价。两条流程除算子层级外完全相同, 由同一脚本以 `--level` 切换, 算例为 Huang2023 的三维 MBB 梁首档。

| 文件 | 内容 |
|---|---|
| [`run_fem_ea_vs_fa.py`](run_fem_ea_vs_fa.py) | 运行入口, `--level ea` 为一份参考单元刚度乘逐单元系数、不形成全局矩阵, `--level fa` 为全局 CSR 稀疏矩阵; `--grid`、`--preconditioner` 选网格与 CG 预条件子 |
| [`fem_ea_vs_fa.md`](fem_ea_vs_fa.md) | EA 与 FA 公共实验参数、结果一致性与耗时、内存对照 |
| [`run_fem_preconditioners.py`](run_fem_preconditioners.py) | CG 预条件子有效性对照: 固定密度场下逐档加密网格, 以 `--preconditioner` 选择 none / jacobi / gmg, 比较 CG 步数随单元尺寸的增长, 见 [预条件子对照](#预条件子对照) |
| [`fem_preconditioners.md`](fem_preconditioners.md) | 预条件子对照的实验设置、各密度下的步数与耗时、结论 |
| [`fem_preconditioners/`](fem_preconditioners/) | 入库的预条件子筛选输入密度 (48x8x8): `make_densities.py` 合成的均匀 `uniform.npy`、0/1 桁架 `binary.npy` 与过滤桁架 `gray.npy`, 生成参数见 `sources.json` |
| [`sync_results.py`](sync_results.py) | 把选定的两侧运行从 `outputs/` 固化到 `results/` |
| [`results/`](results/) | 入库的结果, `fem_ea_vs_fa.md` 的数据来源: 两侧 `config.json`, `summary.json`, `history.json` 与末轮场比对 `final_fields.json` |

公共参数统一见 [实验设置](fem_ea_vs_fa.md#2-实验设置)。

当前运行入口统一采用 `rtol=CG_TOLERANCE`、`atol=0`、`rtol_reference='rhs'`、
`norm_type='unpreconditioned'`、`verify_residual=True`, 由 CG 根据施加边界条件后的实际
右端项计算停机阈值, 不再手算 `load_norm`. EA 与 FA 共用这套设置.
`results/` 中 210x35x35 Jacobi 的 EA 与 FA 结果按这套设置生成; 390x65x65 GMG 的旧结果 (`*_hex_390x65x65_cg*`, `final_fields_390x65x65_gmg.json`) 保留生成时的配置, `fem_ea_vs_fa.md` 不再引用.

## 运行

在仓库根目录、已安装 SOPTX 的 Python 环境中执行, 两侧各为一次独立运行:

```bash
python experiments/topopt_simp_fem/run_fem_ea_vs_fa.py --level ea --max-iter 3
python experiments/topopt_simp_fem/run_fem_ea_vs_fa.py --level fa --max-iter 3
python experiments/topopt_simp_fem/run_fem_ea_vs_fa.py --level fa --grid 48 8 8 --preconditioner gmg
```

线程固定为单线程: 脚本顶部的 `NUM_THREADS = 1` 在导入 numpy/scipy 之前写入全部 BLAS / OpenMP 线程变量, 覆盖 shell 中的已有设置, 实际取值与已加载的 BLAS 库记入 `config.json` 的 `threads`。EA 的单元作用经由 BLAS, FA 的稀疏矩阵向量积与 `bincount` 不经由 BLAS, 多线程会让对照混入库的线程支持差异; 且不设置时 NumPy 链接的 BLIS 默认单线程, SciPy 内置的 OpenBLAS 默认用满全部核, 口径不一。

每次运行只跑一侧: 内存按进程 RSS 记录, 两侧同进程连跑会互相污染。命令行有必填的 `--level` 与可选的 `--grid NX NY NZ` (默认常量 `GRID = (210, 35, 35)`)、`--preconditioner {none,jacobi,gmg}` (默认常量 `PRECONDITIONER = 'jacobi'`)、`--max-iter N` (默认常量 `MAX_ITER = 300`), 其余设置 (后端与设备、MGCG 选项与 VTU 场) 均为脚本顶部常量, 改设置须改常量。

结果写入本目录的 `outputs/{level}_hex_{NX}x{NY}x{NZ}_cg_{preconditioner}/`, 如默认配置为 `outputs/{ea,fa}_hex_210x35x35_cg_jacobi/`; 目录名含层级、网格与预条件子, 不同配置互不覆盖。2026-10-15 之前的运行目录名不含预条件子 (如 `ea_hex_390x65x65_cg`), 保持原名。`outputs/` 是不入库的临时目录, 输出目录已存在时脚本直接清空再重写; 需要保留的结果先用 `sync_results.py` 固化到 `results/` 或自行备份。

## 预条件子对照

`run_fem_preconditioners.py` 不执行优化, 在固定物理密度场上从基准网格逐档加密 (各方向单元数翻倍, 物理域固定, 第 k 档单元尺寸为基准的 $2^{-k}$), 每档以所选预条件子求解同一个线性系统。$K$ 的条件数随加密必然按 $O(h^{-2})$ 增长, 与预条件子无关; 判断预条件子是否有效看的是 $\kappa(M^{-1}K)$, 即 CG 步数随 $h$ 的增长。停机判据与 `run_fem_ea_vs_fa.py` 相同。每次运行为一种预条件子 × 一种密度, 筛选对 jacobi 与 gmg 各跑三种密度 (none 在高反差下不可行, 可选作参照), 例如 gmg:

```bash
python experiments/topopt_simp_fem/run_fem_preconditioners.py --preconditioner gmg --density uniform
python experiments/topopt_simp_fem/run_fem_preconditioners.py --preconditioner gmg --density gray
python experiments/topopt_simp_fem/run_fem_preconditioners.py --preconditioner gmg --density binary
```

- 命令行有必填的 `--preconditioner {none,jacobi,gmg}`、`--density {uniform,gray,binary}` 与可选的 `--levels K`, 其余设置 (`OPERATOR_LEVELS` 及求解参数) 均为脚本顶部常量, 改设置须改常量。
- `--density` 按名字读 `fem_preconditioners/{name}.npy`, 存的是单元物理密度 (读入后直接计算刚度, 不再过滤; 不是设计变量), 其形状即基准网格, 名字作为输出目录的密度标签; 文件缺失时提示先运行 `make_densities.py`。筛选用的三组密度由 [`fem_preconditioners/make_densities.py`](fem_preconditioners/make_densities.py) 合成并入库, 生成参数见 `sources.json`, 基准网格均为 48x8x8:
  - `uniform.npy`: 处处取 `VOLFRAC = 0.12`, 对应优化第 1 轮的常系数问题, 作基线。
  - `binary.npy`: 合成的 0/1 桁架 (上弦贯穿全宽承接加载点, 下弦、拱腿、竖杆与斜杆只在 z 两侧面各一层, 杆宽 1.5 个基准单元, 体积分数 0.176, 面连通), 代表成形后的高反差细杆件 ($E_0/E_{\min} = 10^7$), 是 GMG 的已知弱点。
  - `gray.npy`: 以优化所用的同一密度过滤器 (半径 3 个单元) 过滤 `binary.npy`, 最大密度 0.73、中间密度占 47%, 代表结构已成雏形但密度仍灰的中期; 无预条件 CG 的步数峰值正出现在中期 (48x8x8 优化第 15 轮前后)。
- 三个文件由 `make_densities.py` 一次生成并整份重写 `sources.json`, 不执行优化, 任意基准网格瞬间完成; 换基准网格会覆盖这三个文件:

```bash
python experiments/topopt_simp_fem/fem_preconditioners/make_densities.py --grid 48 8 8
```

- 粗网格密度以 $2^k \times 2^k \times 2^k$ 分片常数注入各档, 各档几何与刚度反差相同。注入保留的是粗网格的杆件, 比真实细网格优化的细杆件温和, 结论偏乐观, 通过筛选的预条件子仍需一次完整优化确认。
- `--levels K` 为网格加密次数, 共 K + 1 档 (默认取常量 `LEVELS = 2`, 即 48x8x8 至 192x32x32; `--levels 3` 再加 384x64x64, 约 485 万自由度, none 与 jacobi 在单线程下很慢); `OPERATOR_LEVELS` 默认只含 `ea`, `fa` 为同一算子, 加入时同档两层级的解逐分量比较, 仅作交叉核对。
- 各档在同一进程内依次执行, 单档出错只记录不中断; 只比较步数与耗时, 不记录内存。耗时分解: `solve_seconds` 为实测求解耗时 (含 `solve_system` 内部的预条件子构造); 求解后另构造一次预条件子单独计时为 `preconditioner_setup_seconds` (GMG 层次结构已缓存, 即优化中每轮重建的成本), 再以 CG 内部同样的 `A @ p`、`M @ r` 对随机向量重复作用取平均得 `matvec_seconds`、`apply_seconds`; `other_seconds` = 实测 − (构造 + 步数 × (matvec + 作用)), 含向量运算、GMG 首次建立层次结构与调用开销。
- 结果写入 `outputs/preconditioner_{none,jacobi,gmg}_{密度}_{NX}x{NY}x{NZ}_L{K}/` (已存在即清空重写): `config.json`、逐档的 `results.json`、含斜率的 `summary.json` 与逐档位移 `displacement_{层级}_{NX}x{NY}x{NZ}.npy`, 运行结束打印步数表。

判据: 以收敛档的 $\log(\text{niter})$ 对 $\log h$ 做最小二乘拟合, 斜率接近 0 为与网格无关的最优预条件 (期望 gmg); 接近 $-1$ 为步数 $O(h^{-1})$ 的未改善行为 (期望 none 与 jacobi, jacobi 的常数更小)。同一密度下不同预条件子运行之间, 同一档的位移 NPY 逐分量比较, 最大相对差应在 `CG_TOLERANCE` 量级; 脚本本身不做跨运行比较。

## 输出与结果固化

运行脚本把结果写入 `outputs/{ea,fa}_hex_{NX}x{NY}x{NZ}_cg_{preconditioner}/` (不入库), 再由 `sync_results.py` 固化到 `results/`。

在仓库根目录执行 `python experiments/topopt_simp_fem/sync_results.py` 固化默认运行 (`outputs/{ea,fa}_hex_210x35x35_cg_jacobi/`)；旧命名的运行目录须以 `--ea`、`--fa` 显式指定；可用 `--ea`、`--fa` 与 `--results-dir` 指定来源与目标。固化两侧 JSON 和末轮场比对统计，NPY 本身不入库。

## ParaView 帧

每轮帧 `iterations/iter_XXXX.vtu` 与串成 ParaView 时间序列的 `evolution.pvd` 写在结果目录 (`outputs/<运行名>/`)，运行结束后仍留在原处，不自动迁移；第 0 帧 `iter_0000.vtu` 是初始设计经过滤与实体保留单元处理后的物理密度 (只含 `density`)，在计时之前写出，便于查看固定单元；由 `soptx.postprocess.iteration_frames.IterationFrameWriter` 原子写出。结果目录顶层另存末帧、初始帧的副本 `result_final.vtu`、`result_initial.vtu`，末帧与最终 NPY 对应同一次分析。

帧体积大 ($190\times32\times32$ 每帧约 14 MB，$390\times65\times65$ 每帧约 120 MB)，长跑前留意 WSL 盘空间。需要在 Windows ParaView 中查看时，在本实验目录下手动迁移，例如：

```bash
python -m soptx.postprocess.sync_visualization --source-dir outputs/fa_hex_210x35x35_cg_jacobi --destination-dir /mnt/c/workspace/soptx-results/topopt_simp_fem/fa_hex_210x35x35_cg_jacobi --move
```

`--move` 迁移后删除 WSL 中的源帧，位置记在结果目录的 `visualization_location.json`；它要求结果目录已有 `summary.json`，中断的运行没有该文件，此时去掉 `--move` 只复制，确认后自行删除 WSL 中的帧。
