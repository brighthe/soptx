# 帧 8：Matrix-Free 无矩阵高性能求解原型验证数值结果

本文档记录了基于 `soptx_heliang` 代码库中 `soptx/benchmarks/benchmark_matrix_free_elasticity.py` 多后端运行（2026-06-30）所产生的基准测试数据。这些数据可作为 `dut-postdoc` 仓库下答辩 PPT（`template-8min.tex` 的 Frame 8「方向一 · Matrix-Free 无矩阵高性能求解原型」）的上游事实数据源；帧 8 右栏四条证据链的数值均出自本文（证据 ④ 为外部锚点，见 §4）。

本文的展示框架与答辩口径基于 dut-postdoc 帧级主入口
`research/postdoc-plan/defense-sprint/direction-1-piml-matrix-free/frame8_matrix_free_pipeline_guide.md`
（原 `soptx-matrix-free-integration-plan.md` / `matrix_free_math_principles.md` 已被其取代删除）。

> **本文为帧 8 的数值单一事实源（SSOT）**：§2/§3 的全部数值表、复现命令、deck LaTeX 只在此维护；
> `ai/common/progress-frame8_matrix_free.md` 只留 headline 并指回本文，dut-postdoc guide / deck
> 的数字均从本文派生（各带出处行）。重跑 benchmark 后**只改本文一处**。§4 为外部锚点（mfleo，
> 非本仓库实测），出处见 §4。
> **数值源**：`soptx/benchmarks/benchmark_matrix_free_elasticity.py`（§2 正确性 / §3 GPU 趋势 @ 2026-06-30）；
> CSV `outputs/matrix_free_elasticity_benchmark_latest.csv`（§2）与 `..._large.csv`（§3）（outputs/ 默认 gitignore，重跑即得）。

## 1. 测试用例参数设定

- **离散**: 2D 三角网 / 3D 四面体（`from_box`），P1 位移元，$E=1$，$\nu=0.3$（2D 平面应力），非均匀单元系数 `coef = linspace(0.4, 1.0, NC)`，`q=4` standard contraction（测试中禁用装配 fallback，不形成局部 $K_e$）。
- **三档配置**: `numpy-cpu` / `pytorch-cpu` / `pytorch-cuda`；driver 为每个配置 spawn 独立子进程（隔离全局后端态），GPU 计时含 `torch.cuda.synchronize` 与 warm-up。
- **环境**: RTX 5070 Ti（16 GB，sm_120），`torch 2.11.0+cu128`。
- **数据文件**: `outputs/matrix_free_elasticity_benchmark_latest.csv`（小规模正确性档）、`outputs/matrix_free_elasticity_benchmark_large.csv`（大规模 GPU 趋势档）+ 对应 XLSX。

## 2. 正确性档（小规模·CG 收敛，帧 8 证据 ①②）

算例 2D $n=4$–$32$、3D $n=2$–$8$（ndof 50–2,187），三档配置全部：

| 指标 | 实测范围 |
|---|---|
| MatVec 等价性 ‖MF(x)−Kx‖/‖Kx‖ | **7.9e-16 – 3.5e-14** |
| CG 收敛 | 全部收敛（tol 1e-10） |
| CG 相对残差 | 2.7e-12 – 1.1e-11 |
| ‖u‖ 三档一致性 | 逐位一致（如 2d_n32 三档均 84399.2558819149，≥13 位有效数字） |

- 帧 8 上 MatVec 写作 $10^{-15}$–$10^{-13}$、残差写作 $10^{-11}$–$10^{-10}$，均为保守口径（实测更好）。
- 完整 `solve_state(operator_backend="matrix_free")` 链路（RHS 装配 + Dirichlet + matvec + CG）另有 pytorch CPU 与 numpy 的一致性实证（HalfMBBBeamRight2d，rel_err 7.9e-15），已固化为仓库测试。

## 3. 单卡 GPU MatVec 加速趋势（大规模 2D，帧 8 证据 ③）

| 算例 | ndof | PyTorch CPU | PyTorch CUDA | 装配内存 (MB) | Matrix-Free 内存估计 (MB) |
|---|---|---|---|---|---|
| 2d_n64 | 8,450 | 1.46x | 2.16x | 2.66 | 0.25 |
| 2d_n128 | 33,282 | 2.35x | 7.83x | 10.57 | 1.0 |
| 2d_n256 | 132,098 | 2.31x | **11.91x** | **42.14** | **4.0** |

- 加速比为 matrix-free **MatVec 算子**计时 vs NumPy CPU；帧 8 上写作"13.2 万 DOF：11.9×；内存估计 42.1 → 4.0 MB"。
- 大规模档 rel_matvec ~1e-13（算子正确）；但无预条件 CG 在 maxiter=200 内**未收敛**，`‖u‖` 为截断迭代值（三档间 ~1e-4 浮点次序差异，非正确性问题）——这正是"预条件器是下一步刚需"的依据。
- 小规模下 GPU 反而慢于 NumPy（kernel 启动 + Python CG 循环开销主导），交叉点约 ndof~1e4。

## 4. GPU/MPI 并行基础（帧 8 证据 ④，外部锚点·非本仓库实测）

来源为 **mfleo 仓库**（MFEM/C++，PA / Matrix-Free 算子）：

- 650 万 DOF hex，GPU + OpenMPI，`mpirun = 1–32`：相对基线总时间约 **3.72×–12.74×**；
- P2 tet（Jacobi / Chebyshev 预条件）：64 核 CPU 约 1.18×–1.21×，GPU 多数配置 4×+。

**引用纪律（必须区分实测与外推）**：MPI+GPU 架构跑通 + 机器级零误差一致性校验（`rel_error=0.0`，`proxy_report.json`）是**实测**；千万级 DoF 的耗时/内存表是**基于真实记录的外推投影**，不可当实测报。

## 5. 复现命令

在 `C:\workspace\soptx_heliang` 下运行（outputs/ 默认 gitignore，重跑即得）：

```powershell
# 小规模三档正确性档（§2）
.\.venv\Scripts\python.exe -m soptx.benchmarks.benchmark_matrix_free_elasticity `
  --output outputs/matrix_free_elasticity_benchmark_latest.csv `
  --xlsx-output outputs/matrix_free_elasticity_benchmark.xlsx

# 大规模 2D GPU MatVec 趋势档（§3）
.\.venv\Scripts\python.exe -m soptx.benchmarks.benchmark_matrix_free_elasticity `
  --cases-2d 64,128,256 --cases-3d "" --repeat 5 --cg-maxiter 200 `
  --output outputs/matrix_free_elasticity_benchmark_large.csv `
  --xlsx-output outputs/matrix_free_elasticity_benchmark_large.xlsx

# 正确性安全网（期望 6 passed：4 numpy + 2 pytorch）
.\.venv\Scripts\python.exe -m pytest soptx/tests/test_matrix_free_vs_assembled.py -q -p no:cacheprovider
```

## 6. 跨库同步材料 (LaTeX)

帧 8 右栏（`template-8min.tex`）当前采用的四条证据链即本文数据的展示化：

```latex
\mfbadge{1}~\textbf{MatVec 等价}（2D/3D，不形成 $K_e$）
  $\|\mathrm{MF}(x)-Kx\|/\|Kx\|$：$10^{-15}$--$10^{-13}$
\mfbadge{2}~\textbf{状态方程一致}（NumPy / PyTorch / CUDA）
  小规模 CG 解 $\equiv$ 组装直解，残差 $10^{-11}$--$10^{-10}$
\mfbadge{3}~\textbf{GPU MatVec 加速}（单卡趋势）
  13.2 万 DOF：$\mathbf{11.9\times}$；内存估计 $42.1\to4.0$ MB
\mfbadge{4}~\textbf{GPU/MPI 并行基础}（PA / Matrix-Free 算子）
  650 万 DOF：mpirun 1--32，$\mathbf{3.72\times}$--$\mathbf{12.74\times}$
```

## 7. 诚实边界（答辩口径提醒）

- GPU 加速结论**仅限 matrix-free MatVec 算子**的单卡计时趋势，不是端到端求解性能；大规模 CG 无预条件未收敛，预条件器为下一步刚需。
- 证据 ④ 证明的是 GPU+MPI 并行**算子工程基础**（mfleo 承接），不等同于 SOPTX 已具备端到端 GPU/MPI 高性能求解能力，也不等同于博士后方向一的 PIML × Matrix-Free 一体化系统已完成。

## 8. 上游文档

- 上游进度与决策记录见 `ai/common/progress-frame8_matrix_free.md`（权威实时态，含逐步实证记录）；实现备忘录见 `docs/matrix_free_architecture_notes.md`；帧级主入口见 dut-postdoc `frame8_matrix_free_pipeline_guide.md`。
