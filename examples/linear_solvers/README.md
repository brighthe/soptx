# 线性求解器层 (Linear Solvers)

本目录验证 `soptx.solvers`：直接法一侧检查稀疏 LU/LDLᵀ 分解后端是否解对、求解是否
破坏调用方持有的矩阵、对称性标志与非法参数是否被守住；迭代法一侧按预条件子分
case（无预条件、Jacobi），每个 case 下检查 `CGSolver` 在同一离散算子的 `'fa'` 稀疏
矩阵与 `'ea'` matrix-free 两种形态上是否同解、是否与直接法一致、三种判据范数是否只
改变迭代数而不改变解、真残差刷新与批量右端项的判定是否守规矩、`info` 契约是否自洽，
并在加密序列上逐层记录迭代数、真残差与制造解误差。

有限元装配在这里只是造出一个真实的 SPD 弹性刚度系统的手段，不是被验证的对象；
离散格式本身的正确性由 `lagrange_elasticity/` 与 `huzhang_elasticity/` 负责，
`'fa'` 与 `'ea'` 算子作用结果的一致性由 `matrix_free_elasticity/` 与
`soptx.fem.verification` 负责。

## 为什么单独立目录

求解器是横跨格式与算子层级的**求解器层**关注点：同一套直接法后端要同时服务位移
元的 SPD 系统与胡张混合元的对称不定鞍点系统，同一个 `CGSolver` 要同时服务 `'fa'`
的稀疏矩阵、`'ea'` 的 `DirichletBCOperator` 与分布式的重叠加权内积。它不属于任何
一个物理问题目录，因此按「新技术栈 → 新目录」的约定单独成目录；直接法与迭代法
是同一技术栈的两个文件，不再分目录。

## 求解器在 SOPTX 中的角色

| 角色 | 谁在用 | 要求 |
|---|---|---|
| 正确性基准 | `soptx.fem.verification` 用直接法校验 EA matrix-free 算子 | **可信**：矩阵不被改写，语义无歧义 |
| 鞍点系统的唯一解法 | `HuZhangMFEMAnalyzer`（CG 对对称不定系统无效） | **能力**，且不成为瓶颈 |
| 中小规模 FA 的默认求解器 | `LagrangeFEMAnalyzer`（`solve_method` 默认 `mumps`） | 每次 OC 迭代**成本可预测、无浪费** |
| `'ea'` 的唯一解法 | `LagrangeFEMAnalyzer(operator_level='ea')`，`solve_method` 只能是 `'cg'` | **可信**：与 `'fa'` 直接法同解 |
| 分布式求解内核 | `DistributedAnalyzer` 经 `overlap.weighted_cg` 注入 `dot_product` | 内积可插拔，判据不依赖全局矩阵 |
| 大规模路线的骨架 | EA + CG + 多重网格（`Multigrid` / `AMGSolver` 尚为占位） | 预条件子经 `M=` 位接入，接口稳定 |

直接法的目标**不是**支撑大规模问题：2D 下分解浮点量约 $O(n^{1.5})$、3D 下约
$O(n^{2})$，内存更差。大规模路线是 EA + CG + 多重网格，本目录只在串行、CPU、
`float64` 下验证 CG 的算法正确性；多 rank 的 `weighted_cg` 由
`matrix_free_elasticity/` 覆盖，多重网格落地前没有第六个角色的证据。

## 直接法后端

| `backend` | 底层 | 并行 | 对称性 | 调用路径 |
|---|---|---|---|---|
| `scipy` | SuperLU（scipy 内置） | 串行 | 不利用 | `soptx.solvers.spsolve` |
| `mumps` | MUMPS（多波前） | MPI 分布式 | `sym=0/1/2` | `soptx.solvers.spsolve` |
| `pypardiso` | MKL PARDISO | OpenMP 共享内存 | `mtype=11/2/-2` | **脚本内本地分支**（尚未接入 `soptx.solvers`） |

`scipy` 零额外依赖，保证「正确性基准」这个角色在任何机器上都能跑；`mumps` 与
PARDISO 提供对称性利用与分阶段能力。pypardiso 接入 `soptx.solvers` 之前先在这里量
清楚它的矩阵所有权语义，接入后 `_pardiso_solve` 应当删除、改走同一个 `spsolve` 入口。

两个必须绕开的陷阱，脚本里已处理：

- **对称性标志的三角约定相反**：MUMPS `sym=1/2` 只吃**下**三角，MKL PARDISO
  `mtype=2/-2` 只吃**上**三角。这正是后端中立的 `Symmetry` 枚举要盖住的差异。
- **PARDISO 的因子缓存会伪造计时**：`PyPardisoSolver.solve()` 会把因子存进
  `factorized_A` 并在同一矩阵上静默复用；模块级 `pypardiso.spsolve` 更是共用一个
  单例。脚本每次计时都新建求解器实例，测的才是完整的「分析 + 分解 + 回代」。

## 迭代法被验证的对象

| 对象 | 位置 | 检查什么 |
|---|---|---|
| `CGSolver` / 函数式 `cg` | `src/soptx/solvers/cg.py` | PCG 递推、停机判据、`norm_type`、`residual_refresh`、批量右端项、`info` |
| `DiagonalPreconditioner` | `src/soptx/solvers/preconditioners.py` | 作为 `M=` 接入后解不变、两层级迭代数都严格下降 |
| `assemble_operator_diagonal` | `src/soptx/fem/analyzers/lagrange_fem_analyzer.py` | `'fa'` 与 `'ea'` 取出的对角一致（Jacobi 的输入） |
| `DirichletBCOperator` 作为 CG 的算子 | `fealpy.fem.dirichlet_bc_operator`（vendor fork） | 只用 `@` 就能被 `CGSolver.setup` 接受并解对 |

`DirectSolver` 在迭代法检查里只充当参考解。

## 文件职责

| 文件 | 职责 |
|---|---|
| `verify_direct_solvers.py` | 直接法四项检查的可执行入口，一个 case 一个后端；同时持有两个脚本共用的模型、网格、材料与分析器构造（`build_analyzer` / `build_system`） |
| `verify_cg_solver.py` | CG 八项检查的可执行入口，一个 case 一个预条件子；按路径加载 `verify_direct_solvers.py` 借用问题构造与排版工具 |
| `results_analysis.md` | 数学—代码映射契约 + 验收契约 + 实验证据报告 |
| `outputs/` | 运行产物（JSON），已由 `.gitignore` 忽略 |

判定阈值定义在各脚本的 `TOLERANCES`，属于本目录，不属于 `soptx`。

## 直接法：四项检查（`verify_direct_solvers.py`）

| 检查 | 内容 | scipy | mumps | pardiso | 服务于 |
|---|---|---|---|---|---|
| `residual` | 对**完整矩阵**算相对残差 $\lVert b - Ax\rVert/\lVert b\rVert$ 落在阈值内 | ✓ | ✓ | ✓ | 正确性基准 |
| `solution` | 制造解的 $L_{2}$ 观测收敛阶不低于门禁，复用 `lagrange_elasticity/manufactured_convergence_demo.py` 的加密序列与判定 | ✓ | ✓ | — | 正确性基准 |
| `ownership` | 求解后调用方持有的矩阵数值与索引均未被改写，分 FEALPy 稀疏算子与原生 scipy CSR 两档输入 | ✓ | ✓ | ✓ | 正确性基准 |
| `guard` | 非法 solver 名与非法 `sym` 值均抛 `ValueError` | ✓ | ✓ | — | 正确性基准 |

`residual` 必须对完整矩阵算：`sym=1/2` 时后端只读一侧三角，把非对称矩阵按对称
声明进去会静默解另一个方程组，后端自报的内部残差仍然很小。通过时脚本只打印横幅
与 `solution` 的逐层误差表，`residual`、`ownership`、`guard` 都只在失败时打出整组
结果；它们照样参与判定，残差数字、自由度与算子导出格式进 JSON。`ownership` 的证据
强弱取决于算子的实际导出格式：若导出为 COO，`tocsr()` 本身即复制，后端碰不到调用方
内存，JSON 里会标注。

pardiso 只有 `residual` 与 `ownership`：它尚未接入 `soptx.solvers.spsolve`，凡是要走
那个入口才成立的检查对它都不存在，接入后应当补齐。

对称性一轴（`--symmetry`）是**声明**而不是检测：general / spd / indefinite 分别对应
MUMPS 的 `sym=0/1/2` 与 PARDISO 的 `mtype=11/2/-2`。脚本装配的是 SPD 系统，三档
声明都合法，因此三档都应当给出同一个正确解。

跨后端的耗时、多右端项与因子复用对比不在本目录：它们是选型依据而非门禁，需要时
另起脚本。

## 迭代法：case 与检查（`verify_cg_solver.py`）

case 轴是预条件子，与直接法的「一个 case 一个后端」同构；每个 case 在 `'fa'` 与
`'ea'` 两个层级上都跑。

| case | 预条件子 | 层级 |
|---|---|---|
| `none` | 无预条件，`M=None` | `fa ea` |
| `jacobi` | `DiagonalPreconditioner(diag)`，`diag` 由 `assemble_operator_diagonal` 取出 | `fa ea` |

`ChebyshevSmoother`、`AMGSolver`、`Multigrid` 尚为 `NotImplementedError` 占位，不在
注册表里；落地后加一条注册项即可，可用性由脚本探测，显式点名的 case 不可用即判失败。

每个 case 下跑八项检查：

| 检查 | 内容 | 类型 | 固定参数 | 服务于 |
|---|---|---|---|---|
| `consistency` | **与直接法一致**：`CGSolver` 解与 `DirectSolver('scipy')` 解的相对差、对完整算子算的真残差均落在阈值内 | 正确性 | `'fa'` | `'ea'` 解法 |
| `level` | **算子层级无关**：同一参数、同一初值下 `'fa'` 与 `'ea'` 的解相对差落在阈值内，迭代数相差不超过 1；两层级的 Dirichlet 基准向量一致 | 正确性 | — | `'ea'` 解法 |
| `norm` | **判据范数**：`'natural'` / `'unpreconditioned'` / `'preconditioned'` 三档都收敛且解相对差落在阈值内；无预条件时三档迭代数相同 | 正确性 | `'fa'` | `'ea'` 解法 |
| `refresh` | **真残差口径**：`residual_refresh=0` 与 `=20` 解相对差落在阈值内；`info['relres']` 与脚本独立算的 $\lVert b-Ax\rVert/\lVert b\rVert$ 相对差落在阈值内 | 正确性 | `'fa'` 与 `'ea'` 各跑一遍 | `'ea'` 解法、分布式内核 |
| `batch` | **批量右端项逐列判定**：两列尺度相差 $10^{3}$ 的右端项同时求解，每列与单独求解的解相对差落在阈值内；`batch_first` 两种布局同解 | 正确性 | `'fa'`，`x0=0` | `'ea'` 解法 |
| `contract` | **`info` 契约**：本 case 全部求解 `converged` 与 `reason > 0` 一致；`x0` 取直接法解且 `atol=1e-8` 时 `niter == 0`、`reason=CONVERGED_ATOL`；`maxit=3` 时 `converged=False`、`reason=DIVERGED_ITS`；`DiagonalPreconditioner` 收到含非正元素或非一维的 `diag`、`CGSolver` 收到非法 `norm_type` / `residual_refresh` / `divtol` 时均抛 `ValueError` | 守卫 | — | `'ea'` 解法、分布式内核 |
| `precond` | **预条件子收益**（仅 `M` 非空的 case）：`'fa'` 与 `'ea'` 取出的 `diag` 相对差落在阈值内；两层级各自接入 `M` 后解相对差落在阈值内，迭代数相对无预条件严格下降 | 正确性 + 效率 | `norm_type='unpreconditioned'` | `'ea'` 解法、大规模骨架 |
| `solution` | **制造解逐层收敛**：加密序列上逐层求解，两层级都收敛、`'fa'` 真残差落在阈值内、$L_{2}$ 观测阶末档不低于 `manufactured_convergence_demo.py` 的门禁；逐层表是通过时的唯一输出 | 正确性 | `--levels` / `--base` | `'ea'` 解法 |

`solution` 以外的检查里两个层级的 `x0` 都取 `'ea'` 的 Dirichlet 基准向量：`'ea'` 的
`apply_bc` 不改写右端项上的 Dirichlet 分量，而是把算子包成
$A = \Pi_I K \Pi_I + \Pi_D$，Dirichlet 值靠初值携带，与
`LagrangeFEMAnalyzer.solve_state` 的做法相同；`'fa'` 用同一初值后两个系统在内部自由度
上完全相同，迭代数才可比。

`batch` 的两列尺度刻意拉开，用来抓「按整体范数判定」这类错误：若判据不是逐列独立，
小尺度那一列会在未收敛时被大尺度列的收敛带过。

通过时脚本只打印横幅、`cases:` 行与每个 case 的逐层表（`n`、`gdof`、两层级 `niter`、
`relres`、真残差、$\lVert u-u_{h}\rVert_{0}$、观测阶），其余检查只在失败时打出整组
结果；它们照样参与判定，各检查的数字进 JSON。两张表对照即预条件子的收益：`jacobi`
的 `niter` 逐层低于 `none`，而误差与观测阶两列相同。`precond` 对 `none` 记 `skipped`，
不算失败。

`'natural'` 范数下 $(r, M^{-1} r)$ 为负会被截为 0 并按收敛处理而非 breakdown，
这是 `cg.py` 当前的语义选择；`DiagonalPreconditioner` 校验 `diag > 0`，SPD 系统上
触发不到，本目录不为它设检查。

## 运行

```bash
python examples/linear_solvers/verify_direct_solvers.py --list
```

```bash
python examples/linear_solvers/verify_direct_solvers.py --case scipy mumps --json
```

```bash
python examples/linear_solvers/verify_cg_solver.py --list
```

```bash
python examples/linear_solvers/verify_cg_solver.py --case none jacobi --json
```

```bash
python examples/linear_solvers/verify_cg_solver.py --dim 3 --n 16 --levels 3 --json
```

公共参数：`--case` 要跑的 case、`--n` 每方向单元数（默认 40）、`--order` 有限元
阶数（默认 1）、`--dim` 2 或 3（默认 2）、`--model`（默认 2D `sinusoidal`、3D
`polynomial`）、`--mesh`（默认 2D `tri`、3D `tet`）、`--levels` / `--base` 控制
`solution` 的加密序列（默认 5 层，最粗一层取 `manufactured_convergence_demo.py` 的
`BASE_SUBDIVISIONS`）、`--json` 把结果写入 `outputs/`。`verify_direct_solvers.py`
的 `--case` 必选，另有 `--symmetry`；`verify_cg_solver.py` 的 `--case` 默认全部，另有
`--checks`。3D 默认加密到 $n=64$，无预条件 CG 到 $10^{-12}$ 很重，用 `--levels 3`。

两个脚本里显式点名的 case 不可用即判失败；`verify_cg_solver.py` 现有两个 case 都只
依赖 SciPy。退出码 0 表示全部通过，1 表示有检查未过。

## 相关

- 求解层的统一接口、注册表与 `info` 契约：`docs/solvers.md`
- EA matrix-free 的黄金参考路径：`src/soptx/fem/verification.py`
  （该文件刻意直接调用 scipy，不走 `soptx.solvers`，理由见其行内注释）
- `'fa'` / `'ea'` 算子作用一致性与多 rank `weighted_cg`：`matrix_free_elasticity/`
- 断言型单元测试：`tests/unit/test_solvers_direct*.py`、`tests/unit/test_solvers_cg*.py`
