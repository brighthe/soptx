---
title: "SOPTX Matrix-Free Progress"
tags:
  - ai-context
  - matrix-free
  - soptx
  - progress
status: "active"
date: 2026-06-29
---

# Matrix-Free 结构分析原型 Progress

本文件是 SOPTX Matrix-Free 工作线的续接入口。新窗口续接时，先读 `ai/common/status.md`，再读本文件；需要写代码时，再按本文列出的上下文和代码位置继续阅读。

## 当前结论

截至 2026-06-30，当前结论应表述为：

```text
SOPTX Matrix-Free 求解链路已在 NumPy CPU / PyTorch CPU / PyTorch CUDA 三档跑通,
三档结果一致, 正确性由仓库测试 (含 pytorch 后端用例) 固化;
GPU 上 matrix-free MatVec 已观测到随规模上升的加速趋势 (ndof≈1.3e5 时约 12x vs NumPy)。
```

当前不应表述为：

```text
SOPTX 已具备端到端 GPU/MPI 高性能求解能力。
```

（GPU 加速结论目前限于 matrix-free **MatVec 算子**的单卡计时趋势; CG 求解尚无预条件器,
大规模未收敛, 端到端求解性能与 MPI/多节点仍属后续工作。）

## 当前进度

1. 阶段一：Matrix-Free MatVec 接口打通完成。
   - `MatrixFreeElasticityOperator.matvec(x)` 与 assembled `K @ x` 已通过一致性验证。
   - 全局算子路径为 `gather -> LinearElasticIntegrator.action -> scatter_add`。

2. 阶段二：状态方程非伴随后端接入完成。
   - `LagrangeFEMAnalyzer.solve_state()` 已支持 `operator_backend="assembled"` 与 `operator_backend="matrix_free"`。
   - matrix-free 后端通过 `MatrixFreeElasticityOperator` 和 `MatrixFreeCGSolver` 求解非伴随状态方程。
   - 上层优化器仍保持 `state = analyzer.solve_state(rho_val=rho_phys)` 的调用形式。
   - 当前不支持 `adjoint=True` 的 matrix-free 路径。

3. 阶段三：2D/3D standard contraction 完成。
   - `LinearElasticIntegrator.action()` 在 2D/3D `standard` 路径使用积分点 contraction。
   - 当前支持 `coef is None`、`coef.shape == (NC,)` 和 `coef.shape == (NC, NQ)`。
   - 测试中禁用 action 装配 fallback 后仍通过，说明当前覆盖路径不再显式形成局部 `Ke`。
   - 多分辨率 contraction 暂不作为当前优先事项。

4. 阶段四：NumPy benchmark 已建立。
   - benchmark 用于验证 Matrix-Free 数学路径、接口闭环、NumPy 后端正确性、基础时间和内存估算。
   - 默认生成 CSV 和格式化 XLSX。
   - 该 benchmark 不是 GPU/MPI 性能 benchmark。

## 第一步可行性检查结果（2026-06-30）

下一阶段「单机 GPU/多后端」的第一步「单机 GPU 后端可行性检查」已完成代码勘察 + CPU 实证。

### 结论

1. matrix-free 链路（`elasticity_operator` / `krylov` / `boundary` / `integrator.action`）全部经
   `fealpy.backend.backend_manager`（`bm`）访问后端，无硬编码 numpy，结构上可切换后端。
2. **阻断点（已实证）**：`elasticity_operator.scatter_add` 使用 `bm.add_at`。fealpy 三后端语义不一致：
   - numpy（`np.add.at`）、jax（`.at[].add`）对重复索引**累加**，正确；
   - pytorch（`a[idx] += src`）对重复索引**不累加、非确定性**，错误。
   - FEM scatter 的全局 DOF 必然重复，故当前源码切 pytorch 后端 matvec 直接算错。
3. **次要移植性问题（已实证）**：pytorch 后端默认 `float32`，需显式 `float64` 才能与装配出的
   float64 刚度矩阵对齐（实测 `K.matmul(x)` 报 `expected Double but found Float`）。
4. **环境**：本机有 NVIDIA RTX 5070 Ti（16 GB，Blackwell/sm_120，驱动 610.62）。venv 当前已装
   `torch 2.12.1+cpu`；尚未装 GPU 版 torch；jax 未装（且 jax GPU 在原生 Windows 不支持，需 WSL2）。
   原生 Windows GPU 现实路径是 PyTorch + cu128。

### CPU 实证数据（2D 三角网 n=8，ndof=162，768 散射条目→162 唯一 DOF）

| 后端 | (A) 现状 `add_at` rel_err | (B) `index_add` rel_err |
|---|---|---|
| numpy | 5.0e-15 ✅ | 5.0e-15 ✅ |
| pytorch | 8.87e-1 ❌ | 4.4e-15 ✅ |

实证脚本（scratchpad，未纳入仓库）：`mf_backend_probe.py`。

### 修改进展（均已应用，2026-06-30）

1. `elasticity_operator.scatter_add`：`bm.add_at` → `bm.index_add`。
   - numpy 安全网 `test_matrix_free_vs_assembled.py` 仍 `4 passed`，无回归；
   - pytorch 后端真实 `op.matvec` 路径从 rel_err 8.87e-1 → 4.4e-15，已正确。
2. `lagrange_fem_analyzer._solve_state_matrix_free`：传给 CG 求解器的 `F` 改为裸张量 `F[:]`。
   - 根因：`F` 是 fealpy `Function` 包装对象，pytorch 后端下 `torch.zeros_like(Function)` 失败；
     `F[:]` 在 numpy/pytorch 下均返回底层数组（与 assembled cg 路径既有惯例一致）。

### 仍待处理（非阻断）

- pytorch 后端默认 `float32`，与 float64 刚度矩阵不匹配（`expected Double but found Float`）。
  实证中以 `torch.set_default_dtype(torch.float64)` 规避；后续多后端入口需固化统一 dtype 策略
  （显式构造 float64，或入口处统一设默认 dtype）。

## 第二步结果：pytorch CPU 全链路实证（2026-06-30）

完整 `solve_state(operator_backend="matrix_free")` 求解链（RHS 装配 + Dirichlet + matvec + CG）已在
pytorch CPU 后端跑通并与 numpy 一致。算例：HalfMBBBeamRight2d，nx=6/ny=2，ndof=42。

| 后端 | CG 收敛/步数 | matrix_free vs assembled rel_err | ‖u_mf‖₂ |
|---|---|---|---|
| numpy | True / 44 | 7.08e-14 | 5.43612416e+02 |
| pytorch | True / 44 | 7.90e-15 | 5.43612416e+02 |

两后端 ‖u‖ 一致到 8 位有效数字，跨后端正确性确认。实证脚本：`mf_solve_probe.py`（scratchpad，未纳入仓库）。

**已固化进仓库测试（2026-06-30）**：`test_matrix_free_vs_assembled.py` 新增两个 pytorch 后端用例
（`..._matvec_..._pytorch` 与 `..._state_..._pytorch`），用 `pytest.importorskip("torch")`，torch 缺失时
自动 skip，`finally` 中恢复后端与默认 dtype。已反向验证测试有效性：退回 `add_at` 时
pytorch matvec 用例 FAIL（rel_err 0.76）而 numpy 用例仍 PASS，证明该用例精确捕获 numpy 覆盖不到的
pytorch 散射 bug。

## 第三步结果：多后端 benchmark 与 GPU 趋势（2026-06-30）

环境：RTX 5070 Ti（16 GB，sm_120）；venv 已装 `torch 2.11.0+cu128`，CUDA 可用。

`benchmark_matrix_free_elasticity.py` 已重写为多后端：driver 为每个 `(backend, device)` 配置 spawn
独立子进程（隔离全局 backend 态与 CUDA 默认 device），GPU 计时含 `torch.cuda.synchronize` 与 warm-up，
汇总为带 `backend/device` 列与 `*_speedup_vs_numpy` 列的 CSV/XLSX。配置：`numpy-cpu / pytorch-cpu /
pytorch-cuda`，CUDA 不可用时自动 SKIP。

**正确性（小规模, CG 收敛档）**：三档 `|u|` 对每个算例逐位一致（如 2d_n32 三档均 8.439926e+04），
rel_matvec ~1e-15，CG 均收敛。输出：`outputs/matrix_free_elasticity_benchmark_latest.csv`。

**GPU MatVec 加速趋势（大规模 2D, matvec 计时）**：随规模单调上升，到 ndof≈1.3e5 时 CUDA 约 12x。
输出：`outputs/matrix_free_elasticity_benchmark_large.csv`。

> **数值 SSOT = `docs/frame8_matrix_free_pipeline_results.md` §3**：完整三档加速表在此维护，
> 重跑 benchmark 只改那里。headline：ndof≈1.3e5（2d_n256）CUDA **≈11.9x** vs NumPy CPU
> （交叉点约 ndof~1e4，小规模 GPU 反而慢）。

口径提醒：

- GPU 加速结论**仅限 matrix-free MatVec 算子**的单卡计时趋势, 不是端到端求解性能。
- 大规模 CG（无预条件）在 maxiter 内未收敛, 大规模行的 `|u|` 是截断迭代值, 三档间有 ~1e-4 浮点
  次序差异, 非正确性问题（rel_matvec ~1e-13 已证算子正确）。这恰好凸显「预条件器」是后续刚需。
- 小规模下 GPU 反而慢于 NumPy（kernel 启动 + Python CG 循环开销主导）, 交叉点约在 ndof~1e4 起。

## 已定关键决策

1. Matrix-Free 接入点放在 `LagrangeFEMAnalyzer` 的状态方程求解后端，而不是上层优化器直接管理。
2. `rho` 属于拓扑优化和材料插值层；`LinearElasticIntegrator` 只消费插值后的 `coef` 或 `relative_stiffness`。
3. assembled 与 matrix-free 的一致性测试是当前安全网；修改相关代码后优先运行 `soptx/tests/test_matrix_free_vs_assembled.py`。
4. 当前答辩口径应区分两层：
   - SOPTX 当前原型：证明 Matrix-Free 数学路径、接口闭环和正确性；
   - mfleo/高性能 kernel：作为 MPI/GPU 性能潜力和目标参照。
   - **并行层次对照（mfleo vs SOPTX 当前）**：mfleo 是 MFEM/C++ 之上的两层并行——
     **MPI 多节点分布式（域分解，跨节点）+ 单元/高斯点级 GPU SIMT（节点内）**；SOPTX 当前是
     Python 原生张量化（NumPy/PyTorch/JAX），已覆盖**节点内 GPU SIMT 那一层**（本轮单卡
     PyTorch CUDA MatVec 加速即对应此层），**尚缺 MPI 跨节点分布式那一层**。
   - **引用 mfleo 数据时必须分清实测与外推**：MPI+GPU 架构跑通 + 机器级零误差一致性校验
     （`rel_error=0.0`，`proxy_report.json`）是**实测**；千万级 DoF 的耗时/内存表是**基于真实
     记录的外推投影**，不可当实测报。
5. 短期不优先扩展多分辨率 contraction；下一阶段优先转向 GPU/多后端验证与 benchmark。

## 重要代码位置

```text
C:\workspace\soptx_heliang\soptx\analysis\integrators\linear_elastic_integrator.py
C:\workspace\soptx_heliang\soptx\analysis\matrix_free\elasticity_operator.py
C:\workspace\soptx_heliang\soptx\analysis\matrix_free\boundary.py
C:\workspace\soptx_heliang\soptx\analysis\matrix_free\krylov.py
C:\workspace\soptx_heliang\soptx\analysis\lagrange_fem_analyzer.py
C:\workspace\soptx_heliang\soptx\tests\test_matrix_free_vs_assembled.py
C:\workspace\soptx_heliang\soptx\benchmarks\benchmark_matrix_free_elasticity.py
```

## 详细上下文

需要继续理解当前实现说明时，阅读：

```text
C:\workspace\soptx_heliang\docs\matrix_free_architecture_notes.md
```

需要理解研究计划和数学原则时，再阅读：

```text
C:\workspace\dut-postdoc\research\postdoc-plan\defense-sprint\direction-1-piml-matrix-free\frame8_matrix_free_pipeline_guide.md
（帧 8 单一主入口；原 soptx-matrix-free-integration-plan.md 与 matrix_free_math_principles.md
已于 2026-07-02 被其取代删除。预条件子数学原则 + Jacobi/Chebyshev 选型依据以 guide 为准，
详细推导引用 mfleo preconditioner_v0.1.md。）
```

帧 8 数值快照（deck 上游事实源）见 `docs/frame8_matrix_free_pipeline_results.md`。

注意：当前 Codex 会话可能无法写入 `dut-postdoc` 外部目录；涉及这些文档的更新需确认权限或改为在本仓库生成更新稿。

## 验证命令

在 `C:\workspace\soptx_heliang` 下运行：

```powershell
.\.venv\Scripts\python.exe -m pytest soptx/tests/test_matrix_free_vs_assembled.py -q
```

当前期望结果：

```text
6 passed
```

其中 4 个为 numpy 用例、2 个为 pytorch 后端用例。pytorch 用例依赖 venv 中已装 `torch`（当前为
`torch 2.11.0+cu128`，GPU 版）；若环境未装 torch, 这 2 个用例会 skip（显示 `4 passed, 2 skipped`），属预期行为。

如遇 pytest cache 权限警告，可使用：

```powershell
.\.venv\Scripts\python.exe -m pytest soptx/tests/test_matrix_free_vs_assembled.py -q -p no:cacheprovider
```

## Benchmark 命令

完整复现命令（小规模正确性档 + 大规模 GPU 趋势档）统一维护于数值 SSOT
`docs/frame8_matrix_free_pipeline_results.md` §5，此处不复写，避免命令漂移。

progress 专属备忘：driver 为每个配置 spawn 子进程（CUDA 不可用自动 SKIP）；可选
`--configs numpy-cpu,pytorch-cpu,pytorch-cuda`（子集）、`--worker`（内部单配置模式，
driver 自动调用，一般不手动用）。输出文件：

```text
outputs/matrix_free_elasticity_benchmark_latest.csv  (+ benchmark.xlsx) —— 小规模三档正确性
outputs/matrix_free_elasticity_benchmark_large.csv   (+ _large.xlsx)    —— 大规模 GPU MatVec 趋势
```

## 下一步计划

建议下一阶段按以下顺序推进：

1. ~~单机 GPU 后端可行性检查：确认当前 `action/operator/CG` 是否能在 PyTorch 或 JAX 后端运行。~~
   **已完成（2026-06-30，见上「第一步可行性检查结果」）**：结构可移植；阻断点为 `scatter_add` 的
   `add_at`→`index_add`，外加 pytorch 默认 float32 的 dtype 处理。
2. ~~应用修改并在 pytorch CPU 后端跑通完整状态方程。~~ **已完成（见上「第二步结果」）**：
   `scatter_add` 与 `F[:]` 两处修改已应用，numpy 安全网无回归，pytorch CPU 全链路与 numpy 一致。
3. ~~GPU/多后端 benchmark：记录三档 MatVec/CG 时间与收敛信息。~~ **已完成（见上「第三步结果」）**：
   benchmark 重写为多后端子进程结构，三档跑通，GPU MatVec 到 ndof≈1.3e5 约 12x vs NumPy。
### 方向定位（2026-06-30 确定）

**采用方向 A**：SOPTX 专注「正确性闭环 + 单卡 GPU + PIML 张量同图集成」这条差异化主线；
**MPI 跨节点分布式定位为远期、并由 mfleo（MFEM/C++）承接作为性能参照**，SOPTX 短期内不自研原生 MPI。

依据：
- mfleo 的 MPI 域分解是成熟工程（MFEM+HYPRE），自研新意低、答辩易被追问「为何不用 MFEM」；
- SOPTX 的原创点是 PIML 等效刚度场与 matrix-free 算子**同图、数据驻留 GPU、天然可微**，这是
  mfleo C++ 栈做不到的，也正是原 `matrix_free_math_principles.md` 第 7 节「博士后计划核心」所写
  （该文档现已并入 `frame8_matrix_free_pipeline_guide.md`）；
- 入站考核答辩需「窄而深、论断都有实测背书」，A 风险小、与计划书契合；规模故事用 mfleo 作参照补足。
- 战略选型最终仍需与郭旭团队确认；但现有计划文档证据支持 A。

### 下一步（按优先级）

1. **预条件器（下一刚需，立即动手）**：当前无预条件 CG 在大规模未收敛, 是端到端 GPU 求解性能的
   主要瓶颈。优先做 Jacobi/对角预条件等 matrix-free 友好的预条件器, 让大规模 CG 收敛,
   再补**端到端单卡 GPU 求解计时**（把「MatVec 12x」升级为「整个 solve 几倍」）。
2. **PIML 等效刚度场接入**：让 AI 预测的 `coef`/相对刚度直接喂入 matrix-free 算子, 验证张量同图、
   显存驻留、可微路径（方向 A 的核心卖点）。
3. **答辩数据整理**：形成「SOPTX 正确性闭环 + 单卡 GPU 求解加速 + PIML 集成 + mfleo（MPI+GPU）作为
   规模参照」的组合表述；引用 mfleo 时区分实测（架构 + `rel_error=0.0`）与外推（千万 DoF 耗时/内存）。
4. **远期（不在答辩冲刺范围）**：`adjoint=True` matrix-free 路径、完整拓扑优化多步闭环；
   MPI/多节点分布式定位为远期且优先由 mfleo 承接（见上「方向定位」）。

## 新窗口续接提示词

```text
按 C:\workspace\soptx_heliang\ai\common\status.md 续接「Matrix-Free 结构分析原型」。
先复述当前进度、已定关键决策、下一步与实现计划，我确认后再继续。
```
