# 并行执行性能测量 (Parallel Execution)

本目录按**并行层级**组织性能测量脚本：线程级（节点内共享内存）、进程级（MPI）、
以及两者与设备级的组合。设备级（CPU vs GPU）的正确性与性能对比在
`../gpu_elasticity/`，本目录不重复。

对应的研究口径与状态账在 `dut-postdoc` 的
`research/piml-matrix-free-gpu/project-plan.md` §三「并行与高性能执行」；
本目录只提供**工具**，不存放结论证据。一次具体扫描如果要被论文或申请书引用，
冻结副本放 `experiments/`，按该目录规则写 `results_analysis.md`。

## 能力边界

- **能做**：在固定工况下，测同一段代码随线程数的 wall time 变化，并核验解不
  随线程数漂移。
- **不能做**：给出本机可信的扩展性曲线。见下面「测量条件」。
- **当前范围**：2D 线弹性制造解工况，`numpy` / `pytorch` 两个 CPU 后端。3D、
  MPI 布局扫描、直接法分解段尚未接入。

## 文件职责

| 文件 | 职责 |
|---|---|
| `benchmark_thread_scaling.py` | 线程级扩展性 benchmark：父进程派发，每个线程档一个干净子进程 |

## 为什么必须起子进程

OpenBLAS 与 MKL 在 `import numpy` / `import torch` 的那一刻就建好线程池。进程
内再改 `os.environ` 对它们无效；`torch.set_num_threads` 只管 ATen 自己那一个
池，管不到 scipy 背后的 OpenBLAS。所以**一个线程档位 = 一个新进程**，父进程只
负责拼 env、派发、收 JSON。

三个变量必须同时钉死（`OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
`MKL_NUM_THREADS`，另加 `NUMEXPR_NUM_THREADS`），只设其中一个会测到混合状态。

## 为什么必须有带宽受限对照段

只测计算受限段时，曲线压不上去无法区分「代码没并行」和「撞内存墙」，结论不可
证伪。所以段注册表里 `kind` 分两类，默认段集合两类都选：

| 段 | 类别 | 预期 |
|---|---|---|
| `assemble` | 计算受限 | 随线程数上涨 |
| `cg_solve` | 计算受限 | 随线程数上涨，是算子作用乘迭代数的总账 |
| `filter_spmv` | **带宽受限** | 少数线程即饱和 |
| `simp_update` | 带宽受限 | 尚未实现 |
| `direct_factor` | 计算受限 | 尚未实现 |

未实现的段选中时直接报错，不静默跳过。

## 正确性优先于时间

线程数只改归约顺序，不改数学解。每一档都与单线程档比指纹（矩阵范数、解范数、
CG 迭代数），相对差超过 `--tol`（默认 `1e-10`）即判失败，进程返回码非零。这一
项挂了说明并行路径上有 race 或非确定性归约超出预期，比性能数字难看严重得多。

## 测量条件（重要）

脚本自己探测运行环境，命中以下任何一条就在 JSON 里写 `trustworthy: false`，并
在终端输出顶部打警告横幅：

- 运行在 WSL2 下——暴露的是虚拟化后的同构拓扑，核绑定无效，vCPU 到物理核的映射
  由 Windows 动态决定；
- CPU 为 P-core/E-core 混合架构——不绑核时线程落在性能不同的核上，加速比没有单
  一分母；
- 未安装 `threadpoolctl`——各库线程数是声明值而非核实值。

**当前开发机同时命中前两条**，因此本脚本在这台机器上只能用于自查（比如确认某段
到底吃不吃线程），扫描出的加速比不得写进证据页、申请书或论文。可信的曲线需要在
同构核、可绑核的节点上重跑。

## 可选依赖

```bash
pip install threadpoolctl
```

装了才能列出每个已加载 native 库的真实 `num_threads`。不装脚本照跑，但环境记录
里标 `verified: false`——scipy 自带的 OpenBLAS wheel 符号被改名，
`openblas_get_num_threads` 取不到，没有别的办法核实。

## 快速运行

列出可选段：

```bash
python examples/parallel_execution/benchmark_thread_scaling.py --list
```

默认扫描（`numpy` 后端，128×128 四边形网格，三个已实现段）：

```bash
python examples/parallel_execution/benchmark_thread_scaling.py --threads 1,2,4,8,16
```

只看带宽受限对照，并落盘 JSON：

```bash
python examples/parallel_execution/benchmark_thread_scaling.py --threads 1,4,16 --segments filter_spmv --json outputs/thread_scaling_filter.json
```

`pytorch` 后端（走 ATen 线程池而非 OpenBLAS）：

```bash
python examples/parallel_execution/benchmark_thread_scaling.py --backend pytorch --threads 1,4,16
```

## 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--threads` | `1,2,4,8,16` | 线程档位；超过 `os.cpu_count()` 的档会被丢弃 |
| `--segments` | `assemble,cg_solve,filter_spmv` | 被测段 |
| `--backend` | `numpy` | `numpy` 走 OpenBLAS，`pytorch` 走 ATen |
| `--n` | `128` | 每方向网格数 |
| `--order` | `1` | 位移空间次数 |
| `--operator-level` | `fa` | `fa` 全装配，`ea` 单元级 matrix-free |
| `--repeat` / `--warmup` | `3` / `1` | 计时次数取中位数 / 预热次数 |
| `--tol` | `1e-10` | 跨线程档指纹的相对差上限 |
| `--json` | 无 | 结果落盘路径 |

## 待补

- `simp_update`、`direct_factor` 两段只注册了位，实现待接；
- 3D 工况；
- `benchmark_mpi_layout.py`：进程级布局扫描（`1×N` / `N×1` / `N×M`），依赖
  `src/soptx/solvers/direct.py` 先从 `set_centralized_sparse` + `job=6` 改造为
  可分布输入、可复用符号分解的接口。
