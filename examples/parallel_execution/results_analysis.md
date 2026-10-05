# 线程级并行扩展性结果分析

本页是 [`benchmark_thread_scaling.py`](benchmark_thread_scaling.py) 的测量量—代码映射契约、验收契约与实验证据报告。运行入口、参数说明与测量条件见 [`README.md`](README.md)。本目录只提供工具，不存放结论证据：一次要被引用的扫描须冻结到 `experiments/`，在那里写它自己的 `results_analysis.md`。

## 1. 测量量—代码映射契约

### 1.1 进程结构与线程钉死

| 环节 | 代码 | 口径 |
|---|---|---|
| 派发 | `main` → `run_child` | 每个线程档 $p$ 起一个新子进程（`--worker p`），父进程不 import 重型库 |
| 钉死 native 池 | `THREAD_ENV_VARS` | 子进程 env 中 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、`NUMEXPR_NUM_THREADS` 同时置为 $p$ |
| 钉死 ATen 池 | `run_worker` | 能 import `torch` 时 `torch.set_num_threads(p)`，与 `--backend` 无关；interop 池只记录不设置 |
| 档位过滤 | `main` | 只保留 $1 \le p \le$ `os.cpu_count()` 的档，升序去重；被丢弃的档打印 `[note]` |
| 结果回传 | `JSON_BEGIN` / `JSON_END` | 子进程 stdout 中两标记之间为一档的 JSON；缺标记即 `RuntimeError`，整次扫描中止 |

不做核绑定：脚本不设置 affinity，也不设置 `OMP_PROC_BIND` 一类变量。

### 1.2 工况（不计时）

`build_workload` 一次建好、各段共享，构造开销不计入任何段：

| 项 | 取值 |
|---|---|
| 问题 | `SinusoidalPlaneStrainElasticity2D()`，单位正方形，全 Dirichlet 制造解 |
| 材料 | `IsotropicLinearElasticMaterial`，$E = 1$，$\nu = 0.3$，`plane_strain` |
| 网格 | `QuadrangleMesh` / `TriangleMesh`（`--mesh-type`）的 `from_box`，$n \times n$ |
| 分析器 | `LagrangeFEMAnalyzer`，`space_degree = order`，`integration_order = order + 3`，`solve_method = "cg"`，`topopt_algorithm = None` |
| 滤波器 | `Filter`，`filter_type = "density"`，`density_location = "element"`，`rmin = --rmin-cells` $\times h$，$h$ 为单元尺寸 |

`--dim` 只接受 2，3D 在 `build_workload` 中抛 `NotImplementedError`。

### 1.3 段注册表

`_register` 返回的 `SegmentSpec`；`kind` 是判读曲线的先验，不是测量结果。计时一律为 `time.perf_counter()` 包住下表“计时范围”一列。

| 段 | `kind` | 计时范围 | 不计时的准备 | 指纹 `signature` |
|---|---|---|---|---|
| `assemble` | `compute` | `analyzer.assemble_stiff_matrix()` | — | `K_norm`：`'fa'` 取稀疏矩阵非零值 `K.values` 的 2-范数，`'ea'` 取常驻单元矩阵 `K.element_matrices` 的范数 |
| `cg_solve` | `compute` | `analyzer.solve_system(K, F, uh, rtol=1e-12, atol=1e-12, maxiter=5000)` | 装配、体力、`apply_bc`，首次调用后缓存于 `ctx.cache["system"]` | `u_norm` $= \lVert u_{h} \rVert_{2}$，`niter` |
| `filter_spmv` | `bandwidth` | `filter_obj.filter_objective_sensitivities(rho, grad)` | 输入 $\rho \equiv 0.5$、`grad = linspace(0, 1, NC)`，缓存于 `ctx.cache["filter_input"]` | `out_norm` $= \lVert \mathrm{out} \rVert_{2}$ |
| `simp_update` | `bandwidth` | 未实现（`fn=None`） | — | — |
| `direct_factor` | `compute` | 未实现（`fn=None`） | — | — |

选中未实现或未知段时 `_resolve_segments` 以 `SystemExit` 退出，不静默跳过。

`filter_spmv` 在 `density` 策略下实际执行 `apply_exemption`、一次 `H.matmul` 与若干次逐元素运算（加权源项、`maximum` 稳定化、除以 `Hs`），比段描述里的“spmv + 两次 elementwise”多几步逐元素运算，但都是带宽受限类操作。

### 1.4 统计量

每段先跑 `--warmup` 次（不计），再跑 `--repeat` 次，样本升序排列：

| 量 | 代码 | 定义 |
|---|---|---|
| $T_{p}$ | `median` | `samples[len(samples) // 2]`，偶数次取上中位数 |
| $T_{p}^{\min}$ | `min` | `samples[0]`，只入 JSON，不入终端表 |
| 指纹 | `signature` | 取最后一次计时重复的返回值 |
| 加速比 | `report` | $S_{p} = T_{p_{0}} / \max(T_{p}, 10^{-12})$，$p_{0}$ 为档位中的最小线程数 |
| 效率 | `report` | $E_{p} = S_{p} / p$ |

只有 $p_{0} = 1$ 时 $S_{p}$、$E_{p}$ 才是通常意义的加速比与并行效率；`--threads` 不含 1 时，$E_{p}$ 的分母仍是绝对线程数 $p$。

### 1.5 环境可信度

`probe_environment` 每档记录 env、`os.cpu_count()`、`sched_getaffinity` 核数、torch 线程数与 `threadpoolctl` 的 `pools`。`_assess_trust` 命中以下任一条即 `trustworthy: false`，并把理由写入 `caveats`：

| 条件 | 判据 |
|---|---|
| WSL2 | `/proc/version` 含 `microsoft` |
| 线程数未核实 | 未安装 `threadpoolctl`（`verified: false`） |
| 核集合未知 | 平台无 `os.sched_getaffinity` |
| 大小核混合 | `_detect_hybrid_topology`：`/proc/cpuinfo` 的 model name 含 `i5/i7/i9-12/13/14` 或 `Ultra`，启发式 |

终端警告横幅只依据第一档（最小 $p$）的 `environment`；JSON 中每档各自保留一份。

## 2. 验收契约

`check_invariants` 以第一档（最小 $p$）为参照，逐档、逐段、逐指纹键比较；参照中缺失的键跳过。

| 指纹键 | 断言 | 阈值 |
|---|---|---|
| `K_norm`、`u_norm`、`out_norm` | $\lvert v_{p} - v_{p_{0}} \rvert / \max(\lvert v_{p_{0}} \rvert, 10^{-30}) \le$ `--tol` | 默认 $10^{-10}$ |
| `niter` | $\lvert n_{p} - n_{p_{0}} \rvert \le 1$ | 硬编码 1，不受 `--tol` 控制 |

- 任一项失败：终端打印 `[invariant] 解随线程数漂移  [FAIL]` 与逐条差异，进程返回码为 1；全部通过打印 `[OK]`，返回码 0。
- `--json` 给定时无论成败都落盘，顶层含 `invariant_passed`、`invariant_failures`、`runs`。
- 只有一个有效档位时该检查空过。
- 不检查的项：CG 是否收敛（`info["converged"]` 不进指纹）、解是否逼近制造解、计时本身的离散度；`trustworthy` 只是标注，不影响返回码。

## 3. 实验证据

待跑。尚无入库的扫描结果。

开发机运行在 WSL2 下且为大小核混合 CPU，`_assess_trust` 必然给出 `trustworthy: false`，在其上扫描出的加速比只能用于自查，不得写入本页、证据页、申请书或论文。可引用的扫描须满足：

1. 在同构核、可绑核的节点上运行，并安装 `threadpoolctl`，使 `trustworthy: true`、`verified: true`；
2. `invariant_passed: true`；
3. 冻结到 `experiments/` 下的对应主题目录，附 git revision、JSON 原件与命令行，按该目录规则写 `results_analysis.md`。

本页在冻结证据出现后只登记其路径，不复制数字。

## 4. 已知口径偏差

以下一点是当前代码的行为，判读结果前须知晓：

| 项 | 现象 | 影响 |
|---|---|---|
| `'ea'` 下的 `cg_solve` 初值 | `_seg_cg_solve` 不传 `x0`，而 `solve_system` 要求 `'ea'` 的初值由调用方给出满足 Dirichlet 值的 prescribed solution | 跨线程指纹仍可比，但 `'ea'` 的 `niter` 与解不与 `'fa'` 同口径 |
