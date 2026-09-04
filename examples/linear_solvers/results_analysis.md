# 线性求解器层结果分析

本页是 [`verify_direct_solvers.py`](verify_direct_solvers.py) 与 [`verify_cg_solver.py`](verify_cg_solver.py) 的数学—代码映射契约、验收契约与实验证据报告。运行入口和参数说明见 [`README.md`](README.md)。

## 1. 数学—代码映射契约

### 1.1 直接法

`DirectSolver.setup` 分解、`solve` 回代，两个后端的映射如下。

| 数学量 | scipy | MUMPS |
|---|---|---|
| 分解 $A = LU$ 或 $LDL^{\mathsf T}$ | `splu(A.tocsc(copy=True))` | `job=4`，`sym!=0` 时只传 `tril(A)` |
| 回代 $x = U^{-1} L^{-1} b$ | `lu.solve(b)` | `job=3`，只支持一维右端项 |
| 报告量 $\rho(x) = \lVert b - Ax\rVert / \lVert b\rVert$ | 对完整 $A$ 算 | 对完整 $A$ 算 |
| `converged` | $\rho(x) \le$ `residual_tol` | 同左 |

`sym` 是声明不是检测。声明与矩阵不符时后端解的是另一个方程组，只有对完整 $A$ 算残差才暴露得出来。

### 1.2 PCG 递推

求解 $Ax=b$，$A$ 对称正定，预条件子 $M^{-1}$ 对称正定。`cg.py` 里 `_cg_impl` 的变量与数学量一一对应：

| 数学量 | 代码变量 | 更新式 |
|---|---|---|
| $r_{k} = b - A x_{k}$ | `r` | $r_{k+1} = r_{k} - \alpha_{k} A p_{k}$ |
| $z_{k} = M^{-1} r_{k}$ | `z` | `M @ r`，`M is None` 时 `z = r` |
| $\rho_{k} = (r_{k}, z_{k})$ | `rhssq` | `_dot(r, z)` |
| $\gamma_{k} = (p_{k}, A p_{k})$ | `curvature` | `_dot(p, Ap)` |
| $\alpha_{k} = \rho_{k} / \gamma_{k}$ | `alpha` | 冻结列置 0 |
| $\beta_{k} = \rho_{k+1} / \rho_{k}$ | `beta` | — |
| $p_{k+1} = z_{k+1} + \beta_{k} p_{k}$ | `p` | $p_{0} = z_{0}$ |

`_dot` 是可插拔内积：默认逐列点积，分布式时由 `overlap.weighted_cg` 注入重叠加权内积。

### 1.3 停机判据

判据对齐 MFEM `CGSolver`，逐列独立：

$$
q_{k} \le \tau^{2}, \qquad \tau^{2} = \max\bigl(\mathrm{rtol}^{2}\, q_{0},\ \mathrm{atol}^{2}\bigr)
$$

| `norm_type` | 判据量 $q_{k}$ | 代码 |
|---|---|---|
| `'natural'` | $(r_{k}, M^{-1} r_{k})$ | `_column_squares(_dot(r, z))`，负值截为 0 |
| `'unpreconditioned'` | $\lVert r_{k}\rVert_{2}^{2}$ | `_squares(r)` |
| `'preconditioned'` | $\lVert M^{-1} r_{k}\rVert_{2}^{2}$ | `_squares(z)` |

`tol_sq` 即 $\tau^{2}$，`reference_sq` 即 $q_{0}$，`recursive_sq` 即递推得到的 $q_{k}$。无预条件时三档判据量相同，因此迭代数必须相同。

进入循环前先做一次判定，$q_{0} \le \tau^{2}$ 时 `niter = 0`；$\lVert b\rVert < 10^{-15}$ 时直接返回零解。

### 1.4 真残差刷新

`residual_refresh = m > 0` 时每 $m$ 步重算 $r_{k} = b - A x_{k}$ 并得 `true_sq`。判定为

$$
\text{reached} = (\texttt{recursive\_sq} \le \tau^{2}) \ \lor\ (\texttt{true\_sq} \le \tau^{2})
$$

真残差只能额外促成收敛，不能否决递推残差已达标的列。`CGSolver` 的 `info['relres']` 与刷新无关，一律在返回前按 $\lVert b - Ax\rVert / \lVert b\rVert$ 现算。

### 1.5 退出原因

| `reason` | 触发条件 | 代码位置 |
|---|---|---|
| `CONVERGED_RTOL` (2) | 达标且 $\mathrm{rtol}^{2} q_{0} \ge \mathrm{atol}^{2}$ | `converged_codes` |
| `CONVERGED_ATOL` (3) | 达标且 $\mathrm{rtol}^{2} q_{0} < \mathrm{atol}^{2}$ | `converged_codes` |
| `DIVERGED_ITS` (-3) | 循环结束仍活跃 | `_finalize` |
| `DIVERGED_DTOL` (-4) | $q_{k} > \mathrm{divtol}^{2}\, q_{0}$ | `div_sq` |
| `DIVERGED_BREAKDOWN` (-5) | 上一步 $\rho_{k} \le 0$ | `_freeze(rhssq <= 0)` |
| `DIVERGED_INDEFINITE_MAT` (-8) | $\gamma_{k} \le 0$ | `negative` |
| `DIVERGED_NANORINF` (-9) | $q_{k}$ 或 $\gamma_{k}$ 非有限 | `_freeze(~isfinite)` |

多列时 `info['reason']` 取各列码的最小值，`info['column_reasons']` 保留逐列码；`converged` 为全部列 `reason > 0`。

### 1.6 `'ea'` 算子与初值

`'ea'` 的 `apply_bc` 不改写矩阵，返回

$$
A = \Pi_{I} K \Pi_{I} + \Pi_{D}
$$

$\Pi_{I}$、$\Pi_{D}$ 分别是内部与 Dirichlet 自由度的投影。右端项在 Dirichlet 自由度上取边界插值 $u_{D}$，初值 $x_{0}$ 必须在这些分量上等于 $u_{D}$，否则 CG 在 Dirichlet 分量上多花一步且解在内部与 `'fa'` 不一致。脚本取 `analyzer._prescribed_solution` 为 $x_{0}$，与 `solve_state` 一致。

`assemble_operator_diagonal` 在 `'ea'` 下给出 $\operatorname{diag}(A)$：内部分量为 $\operatorname{diag}(K)$ 的逐单元散加，Dirichlet 分量为 1；`'fa'` 下对称消元后的矩阵对角与之相同。

## 2. 验证对象与职责

| 对象 | 被验证 | 由谁负责 |
|---|---|---|
| `spsolve` / `DirectSolver` 残差、矩阵所有权、参数守卫 | 是 | `verify_direct_solvers.py` |
| 三后端耗时、对称性收益、因子复用 | 否 | 无脚本，需要选型依据时另起 |
| `CGSolver` 递推、判据、`info` | 是 | `verify_cg_solver.py` |
| `DiagonalPreconditioner` 接入 `M=` 位 | 是 | `verify_cg_solver.py` |
| `assemble_operator_diagonal` 两层级一致 | 是 | `verify_cg_solver.py` |
| `'fa'` 与 `'ea'` 算子作用一致 | 否 | `matrix_free_elasticity/`、`soptx.fem.verification` |
| 离散格式收敛阶 | 否，`solution` 一项只借用其判定 | `lagrange_elasticity/` |
| 多 rank `weighted_cg` | 否 | `matrix_free_elasticity/` |

## 3. 验收契约

记直接法解为 $x_{\mathrm{d}}$，任一 CG 解为 $x$，相对差 $\delta(x, y) = \lVert x - y\rVert_{2} / \lVert y\rVert_{2}$，真相对残差 $\rho(x) = \lVert b - Ax\rVert_{2} / \lVert b\rVert_{2}$。阈值定义在各脚本的 `TOLERANCES`，下表为当前值或初值，首轮运行后按 §4 证据校准并同步。

### 3.1 直接法

| 检查 | 断言 | 阈值键 | 值 |
|---|---|---|---|
| `residual` | 每档对称性声明下 $\rho(x) \le$ 阈值 | `residual_relative` | $10^{-10}$ |
| `solution` | 逐层 $\lVert u - u_{h}\rVert_{0}$ 与观测阶 $\log_{2}(e_{2h}/e_{h})$，末档观测阶不低于 `manufactured_convergence_demo.py` 的门禁 | 借用 | 借用 |
| `ownership` | 求解前后 `data` 与 `indices` 的相对扰动 $\le$ 阈值 | `matrix_perturbation_relative` | 0 |
| `guard` | 非法 solver 名、非法 `sym` 均抛 `ValueError` | — | 精确 |

### 3.2 迭代法

case 轴是预条件子（`none`、`jacobi`），下表每一行在每个 case 下各判一次；记该 case
的 CG 解为 $x_{M}$，`precond` 一行只对 $M$ 非空的 case 生效。

| 检查 | 断言 | 阈值键 | 初值 |
|---|---|---|---|
| `consistency` | $\delta(x_{M}, x_{\mathrm{d}}) \le$ 阈值 | `solution_rel_diff` | $10^{-6}$ |
| `consistency` | $\rho(x_{M}) \le$ 阈值 | `true_relres` | $10^{-10}$ |
| `level` | $\delta(x_{\mathrm{ea}}, x_{\mathrm{fa}}) \le$ 阈值 | `solution_rel_diff` | $10^{-6}$ |
| `level` | $\lvert n_{\mathrm{ea}} - n_{\mathrm{fa}}\rvert \le$ 阈值 | `niter_gap` | 1 |
| `norm` | 三档 `norm_type` 各自 $\delta(x, x_{\mathrm{d}}) \le$ 阈值，且 `converged` | `solution_rel_diff` | $10^{-6}$ |
| `norm` | 无预条件时三档 `niter` 相等 | — | 精确 |
| `refresh` | $\delta(x_{m=20}, x_{m=0}) \le$ 阈值 | `solution_rel_diff` | $10^{-6}$ |
| `refresh` | $\lvert \texttt{info['relres']} - \rho(x)\rvert / \rho(x) \le$ 阈值 | `relres_rel_diff` | $10^{-6}$ |
| `batch` | 两列各自 $\delta(x_{:,j}, x^{\mathrm{single}}_{j}) \le$ 阈值 | `solution_rel_diff` | $10^{-6}$ |
| `batch` | `batch_first` 两种布局 $\delta \le$ 阈值 | `solution_rel_diff` | $10^{-6}$ |
| `contract` | `converged == (reason > 0)`，本 case 全部求解 | — | 精确 |
| `contract` | $x_{0} = x_{\mathrm{d}}$、`atol=1e-8` 时 `niter == 0` 且 `reason == CONVERGED_ATOL` | — | 精确 |
| `contract` | `maxit=3` 时 `converged is False` 且 `reason == DIVERGED_ITS` | — | 精确 |
| `contract` | `DiagonalPreconditioner(diag)` 对含 0、负元素或非一维的 `diag` 抛 `ValueError` | — | 精确 |
| `contract` | `CGSolver` 对非法 `norm_type`、`residual_refresh < 0`、`divtol <= 0` 抛 `ValueError` | — | 精确 |
| `precond` | $\delta(d_{\mathrm{ea}}, d_{\mathrm{fa}}) \le$ 阈值 | `diag_rel_diff` | $10^{-12}$ |
| `precond` | 两层级各自 $\delta(x_{M}, x_{\mathrm{d}}) \le$ 阈值 | `solution_rel_diff` | $10^{-6}$ |
| `precond` | 两层级各自 $n_{M} < n_{\mathrm{none}}$ | — | 严格 |
| `solution` | 每层两层级 `converged` | — | 精确 |
| `solution` | 每层 $\rho(x_{\mathrm{fa}}) \le$ 阈值 | `true_relres` | $10^{-10}$ |
| `solution` | 末档 $L_{2}$ 观测阶 $\log_{2}(e_{2h}/e_{h})$ 不低于 `manufactured_convergence_demo.py` 的门禁 | 借用 | 借用 |

`solution` 以外各项的求解参数固定为 `rtol=1e-12`、`atol=1e-12`、`maxit=10000`，与 `LagrangeFEMAnalyzer` 的 `'cg'` 默认值同口径。`solution_rel_diff` 初值 $10^{-6}$ 留出了 $\kappa(A)\cdot\mathrm{rtol}$ 的放大余量，$n=40$、$p=1$ 时 $\kappa(A)$ 约 $10^{4}$。

## 4. 实测证据

尚无入库的实测证据。各脚本首轮运行后按以下表格填写，并把 `outputs/` 下 JSON 的 git revision 记入表头。

### 4.1 直接法：残差与耗时

| 维数与网格 | $p$ | $n$ | 自由度 | scipy $\rho$ / s | mumps $\rho$ / s | pardiso $\rho$ / s |
|---|---:|---:|---:|---:|---:|---:|
| 2D `tri` | 1 | 40 | 待填 | 待填 | 待填 | 待填 |
| 3D `tet` | 1 | 12 | 待填 | 待填 | 待填 | 待填 |

### 4.2 直接法：制造解误差与观测收敛阶

每个后端一张表，脚本缺省 `--levels 5`、`--base` 取收敛脚本的 `BASE_SUBDIVISIONS[2] = 8`（2D `tri`，$p=1$，模型 `sinusoidal`）。$h = 1/n$，order 为相邻两层 $L_{2}$ 误差比值的以 2 为底对数，首层无上一层可比。`residual` 是该层对完整矩阵算的相对残差。

scipy：

| $n$ | gdof | $h$ | $\lVert u - u_{h}\rVert_{0}$ | order | residual |
|---:|---:|---:|---:|---:|---:|
| 8 | 待填 | 0.125 | 待填 | — | 待填 |
| 16 | 待填 | 0.0625 | 待填 | 待填 | 待填 |
| 32 | 待填 | 0.03125 | 待填 | 待填 | 待填 |
| 64 | 待填 | 0.015625 | 待填 | 待填 | 待填 |
| 128 | 待填 | 0.0078125 | 待填 | 待填 | 待填 |

mumps：

| $n$ | gdof | $h$ | $\lVert u - u_{h}\rVert_{0}$ | order | residual |
|---:|---:|---:|---:|---:|---:|
| 8 | 待填 | 0.125 | 待填 | — | 待填 |
| 16 | 待填 | 0.0625 | 待填 | 待填 | 待填 |
| 32 | 待填 | 0.03125 | 待填 | 待填 | 待填 |
| 64 | 待填 | 0.015625 | 待填 | 待填 | 待填 |
| 128 | 待填 | 0.0078125 | 待填 | 待填 | 待填 |

### 4.3 直接法：逐项判定

| 检查 | scipy | mumps | pardiso |
|---|---|---|---|
| `residual` | 待填 | 待填 | 待填 |
| `solution` | 待填 | 待填 | 不适用 |
| `ownership` | 待填 | 待填 | 待填 |
| `guard` | 待填 | 待填 | 不适用 |

### 4.4 迭代法：逐层迭代数、残差与制造解误差

列同 `solution` 的 stdout 表，一个 case 一张；2D `tri` $p=1$，$n = 8 \ldots 128$。
两张表的 $\lVert u-u_{h}\rVert_{0}$ 与观测阶两列应相同，`jacobi` 的 `niter` 应逐层
低于 `none`。

case `none`：

| $n$ | 自由度 | `niter(fa)` | `niter(ea)` | `relres` | $\rho(x_{\mathrm{fa}})$ | $\lVert u-u_{h}\rVert_{0}$ | 观测阶 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | — |
| 16 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 32 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 128 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

case `jacobi`：

| $n$ | 自由度 | `niter(fa)` | `niter(ea)` | `relres` | $\rho(x_{\mathrm{fa}})$ | $\lVert u-u_{h}\rVert_{0}$ | 观测阶 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | — |
| 16 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 32 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 128 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

### 4.5 迭代法：逐项判定

2D `tri` $p=1$，$n=40$（`solution` 除外）：

| 检查 | `none` | `jacobi` |
|---|---|---|
| `consistency` | 待填 | 待填 |
| `level` | 待填 | 待填 |
| `norm` | 待填 | 待填 |
| `refresh` | 待填 | 待填 |
| `batch` | 待填 | 待填 |
| `contract` | 待填 | 待填 |
| `precond` | 不适用 | 待填 |
| `solution` | 待填 | 待填 |

## 5. 证据边界与复现

### 5.1 适用范围

| 维度 | 覆盖 |
|---|---|
| 执行 | 串行 CPU，`float64`，FEALPy vendor fork |
| 算子 | `'fa'` CSR 稀疏矩阵、`'ea'` `DirichletBCOperator` |
| 系统 | SPD 弹性刚度系统，Dirichlet 对称消元 |
| 直接法后端 | SuperLU、MUMPS、MKL PARDISO（后者未接入 `soptx.solvers`） |
| 预条件 | 无预条件、Jacobi |
| 内积 | 默认逐列点积 |

### 5.2 尚未覆盖

| 项 | 由谁覆盖或何时覆盖 |
|---|---|
| 鞍点系统上的直接法（`--system saddle`） | `HuZhangMFEMAnalyzer` 接线后 |
| 三后端耗时、多右端项与因子复用的横向对比 | 需要选型依据时另起脚本 |
| MUMPS 分阶段 `job=1/2/3` 的因子复用 | `soptx.solvers` 新增分阶段接口后 |
| 多 rank `weighted_cg` | `matrix_free_elasticity/` |
| GPU 后端 | `gpu_elasticity/` |
| 对称不定系统的迭代法 | `MINRESSolver` 落地后 |
| 多重网格 / AMG 预条件 | `Multigrid` / `AMGSolver` 落地后 |
| `'natural'` 范数下 $(r, M^{-1} r) < 0$ 的截断语义 | 非 SPD 预条件子出现后 |

### 5.3 复现命令

```bash
python examples/linear_solvers/verify_direct_solvers.py --case scipy mumps pardiso --json
```

```bash
python examples/linear_solvers/verify_cg_solver.py --case none jacobi --json
```

```bash
python examples/linear_solvers/verify_cg_solver.py --case none jacobi --order 2 --json
```

```bash
python examples/linear_solvers/verify_cg_solver.py --case none jacobi --dim 3 --n 16 --levels 3 --json
```
