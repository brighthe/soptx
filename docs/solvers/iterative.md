# 迭代法

本文说明 `soptx.solvers` 的迭代求解。参数与默认值见源码 docstring; 求解层的整体结构见
[`architecture.md`](architecture.md), 预条件子见 [`preconditioners.md`](preconditioners.md)。
文中 SPD 指对称正定 (symmetric positive definite)。

迭代法的原理、收敛理论与各方法的取舍见 `dut-postdoc:concepts/linear-solvers/iterative-methods.md`,
本文的分类沿用该页, 只记录 SOPTX 的实现事实。

## 迭代法分类与实现状态

迭代法每步用一个易于计算的近似逆 $B \approx A^{-1}$ (预条件子) 作用于残差, 再按某种规则
构造更新。按两条轴分类:

- **轴一: 外层迭代格式**, 决定怎样使用 $B$, 即怎样选方向与步长;
- **轴二: 预条件子的构造**, 决定 $B$ 是什么。

具体求解器是两条轴的组合。不预条件时 $B = I$, 只剩轴一。

本文给出轴一与两轴的组合, 轴二见 [`preconditioners.md`](preconditioners.md)。实现状态的标注:

- **已实现**: 可直接使用;
- **仅作组件**: 已实现, 但只在多重网格内部使用, 不单独使用;
- **占位**: 类已声明, 构造即抛 `NotImplementedError`;
- **未实现**: 无。

### 轴一: 外层迭代格式

| 外层格式 | 收敛由什么决定 | SOPTX |
|---|---|---|
| 定常迭代 (Richardson, 矩阵分裂) | 谱半径 $\rho(I - BA)$ | **仅作组件**: `JacobiSmoother` 固定次数扫描, 作多重网格光滑子; 多重网格也只作 CG 的预条件子, 不作定常迭代求解器 |
| Chebyshev 迭代 | 谱区间估计 | 未实现, 计划作多重网格光滑子 |
| Krylov 子空间方法 | $BA$ 的整体谱分布 | 见下表 |

Krylov 子空间方法:

| 方法 | 适用系统 | 固定预条件子 | 变预条件子 | SOPTX |
|---|---|---|---|---|
| CG | SPD | PCG | FCG | **已实现**: `CGSolver` 与函数式 `cg`, 注册键 `cg`; `M=` 位即 PCG。FCG 未实现 |
| MINRES, SYMMLQ | 对称不定 | 预条件 MINRES (要求 SPD 预条件子) | — | 未实现 |
| FOM, GMRES | 一般 | 左 / 右预条件 GMRES | FGMRES | 未实现 |
| BiCGSTAB, TFQMR | 非对称 | 预条件 BiCGSTAB | — | 未实现 |

### 轴二: 预条件子的构造

松弛类、不完全分解、多项式、多重网格与区域分解五类构造及 SOPTX 的实现状态见
[`preconditioners.md`](preconditioners.md)「构造类别与实现状态」。

### 组合

按"预条件子-外层方法"命名, 加粗为 SOPTX 已实现的组合:

| 预条件子 \ 外层格式 | 定常迭代 | Chebyshev 迭代 | Krylov 子空间方法 |
|---|---|---|---|
| 无预条件 | Richardson | Chebyshev 迭代 | **CG** |
| 松弛类 | Jacobi, Gauss–Seidel, SOR | Chebyshev-Jacobi | **Jacobi-PCG**, SSOR-PCG |
| 不完全分解 | 很少单独使用 | 很少使用 | IC-PCG, ILU-GMRES |
| 多重网格 | 多重网格求解器 | 很少使用 | **MGCG** (GMG), AMG-PCG |
| 区域分解 | Schwarz | 很少使用 | Schwarz-PCG |

SOPTX 实现的都在 Krylov 一列, 且外层只有 CG。这是有意的取舍:

- 主线是 SPD 的线弹性与拓扑优化系统, MGCG 已经覆盖;
- 多重网格作独立求解器时受最难消去的误差分量拖累, SIMP 的系数高对比度下尤其明显,
  外层套 CG 可吸收少数离群特征值, 更稳健;
- Gauss–Seidel, SOR, SSOR 与不完全分解需要显式矩阵且有串行依赖, 与 matrix-free 和 GPU
  主线冲突;
- 非对称 Krylov 方法暂无需求, SOPTX 目前没有非对称系统;
- 对称不定的 Hu–Zhang 鞍点系统是唯一明确的缺口, 需要 MINRES 与鞍点块预条件子。

预条件子须是固定、对称、正定的线性算子; 嵌套 Krylov 或按容差停止的内层迭代需要 FCG
一类的变预条件方法, 不能直接填进 CG 的 `M=` 位, 见 [`architecture.md`](architecture.md)
「组合约束」。

子结构分析器在接口 Schur 补系统上用 CG 求解, 属于非重叠子结构在界面上做 Krylov 迭代的
路线; 它不是通用的区域分解框架, 不计入上表。

## CG

要求算子 SPD。`requires` 为空, 只需算子支持 `@`, 因此 'fa' 的稀疏矩阵与 'ea' 的
matrix-free 算子都能用。

### 停机判据

判据对齐 MFEM `CGSolver::Mult`, 逐列成立:

$$
\|r_n\|^2 \le \max\left(\mathrm{rtol}^2 \|r_0\|^2,\ \mathrm{atol}^2\right)
$$

- 参照量取初始残差 $\|r_0\|$, 不是右端项 $\|b\|$, 二者只在 $x_0 = 0$ 时相等。此时 `rtol`
  表示残差的下降倍数; 热启动时 $\|r_0\|$ 已经很小, 同一个 `rtol` 会比以 $\|b\|$ 为参照
  更严。需要与初值无关的精度目标时, 传 `rtol=0` 与 `atol=tol·‖b‖`; 拓扑优化的实验脚本
  (`run_fem_ea_vs_fa.py`, 子结构路径) 即如此。
- 判据两边在同一范数下度量, 范数由 `norm_type` 选择 (命名对齐 PETSc `KSPNormType`):


| `norm_type`          | 范数                                           | 额外开销    |
| -------------------- | -------------------------------------------- | ------- |
| `'natural'` (默认)     | $\sqrt{r^{\mathsf T} M^{-1} r}$, 即 CG 递推自带的量 | 无       |
| `'unpreconditioned'` | $\|r\|_2$                                    | 每步多一次归约 |
| `'preconditioned'`   | $\|M^{-1} r\|_2$                             | 每步多一次归约 |


不带预条件子时三者数值相同。

### 退出原因

`info['reason']` 给出 `ConvergedReason`, 取值与整数编码沿用 PETSc `KSPConvergedReason`,
约定 `reason > 0` 等价于收敛:


| 原因                                  | 触发条件                                                                              |
| ----------------------------------- | --------------------------------------------------------------------------------- |
| `CONVERGED_RTOL` / `CONVERGED_ATOL` | 判据成立, 按 $\max$ 中哪一项生效区分                                                           |
| `DIVERGED_ITS`                      | `maxit` 耗尽                                                                        |
| `DIVERGED_DTOL`                     | 判据量相对初值放大超过 `divtol` 倍                                                            |
| `DIVERGED_INDEFINITE_MAT`           | 非正曲率 $p^{\mathsf T} A p \le 0$; 一律停机, 跟随 PETSc 而非 MFEM (MFEM 只告警), 以暴露误用在鞍点系统上的情形 |
| `DIVERGED_INDEFINITE_PC`            | $r^{\mathsf T} M^{-1} r < 0$, 预条件子不正定; 在收敛判定之前拦下, 免得 natural 范数下被截断为零而误判收敛 |
| `DIVERGED_BREAKDOWN`                | $r^{\mathsf T} M^{-1} r = 0$ 而残差尚未达标                                              |
| `DIVERGED_NANORINF`                 | 判据量出现 NaN 或 Inf                                                                   |


右端项为零时不迭代, 直接返回零解并记为收敛。

### 批量右端项

二维右端项按 N 次独立求解处理: 每列有自己的阈值、退出原因与冻结时刻。达标或失效的列
把步长置零, 不再参与曲率与发散判定, 否则已收敛列下溢的曲率会把整批误判成 breakdown。
逐列原因在 `info['column_reasons']`, `info['reason']` 取其中最小值, 优先暴露失败。

### 收敛时的真残差验证 `verify_residual`

有限精度下 CG 的递推残差会与真残差 $b - Ax$ 漂开, 可能在真残差尚未达标时就宣布收敛。
`verify_residual=True` 时, 递推残差宣布达标 (或到达最后一次迭代) 的那一步重算真残差,
把活跃列的递推残差替换为它, 并重算预条件残差与 $(r, z)$, 再以替换后的量判定。虚假达标
因此被否决, 未达标的列以替换后的残差继续迭代; 搜索方向不动, 冻结列保持原值。每次验证
多一次 matvec 与一次预条件子作用, 结果写进 `info['true_residual']`。

只在达标时替换, 不做周期替换: 周期性地用真残差替换递推残差会破坏 CG 递推的局部正交性,
病态问题上迭代数可增加 15%–30%; 只在达标时替换, 算子精确时迭代轨迹与标准 CG 完全一致。
代价是不提高可达精度, 只保证报告收敛时真残差确实达标; SOPTX 的容差在双精度下远高于
可达精度, 这已足够。

### 报告量 `relres`

`CGSolver` 求解后按 $\|b - Ax\| / \|b\|$ 现算一次真残差填进 `info['relres']`, 二维右端
按 Frobenius 范数, 以便与 `DirectSolver` 的 `relres` 直接比较。它是**报告量**, 与停机判据
不同口径: 判据逐列、参照 $\|r_0\|$, `relres` 聚合、参照 $\|b\|$。参照量 $\|r_0\|$ 另由
`info['reference_norm']` 给出。

### 监视与日志

`monitor(iteration, norm, residual, final)` 的签名照 MFEM `IterativeSolverMonitor`,
从第 0 步 (初始残差) 开始回调, 停机后再回调一次; `print_level` 对应 MFEM 的
`PrintLevel`: 0 静默, 1 停机摘要, 2 每步残差。

### 默认值

默认值分两层, 有意为之:

| 层 | 定义处 | 面向 | 策略 |
|---|---|---|---|
| 求解层 | `CGSolver` / `cg` 的签名 | 通用代数调用 | 判据范数 `'natural'`, 不验证真残差 |
| 有限元层 | `fem/linear_solve.py` 的 `DEFAULT_CG_*` | `LagrangeFEMAnalyzer` 未给容差的调用 | 容差比求解层更严, 面向有限元正确性验证; 带预条件子时判据范数改为 `'unpreconditioned'` (与 2-范数的 `relres` 同口径), 并默认开启 `verify_residual` |

具体数值以定义处为准。许多测试与示例依赖有限元层不给容差时的严格默认值, 两层合并会改变
它们的数值行为, 因此不合并。子结构路径在调用处显式给出全部参数, 不受两层默认值影响。
命名上求解层用 `maxit`, 分析器选项用 `maxiter`, 暂不统一。

### 与 FEALPy 的差异

`cg.py` 以 FEALPy `fealpy/solver/cg.py` 为参考重新设计。上游实现的不同之处:


| 项        | FEALPy 上游                                           | SOPTX                       |
| -------- | --------------------------------------------------- | --------------------------- |
| 相对判据的参照量 | $\mathrm{rtol} \cdot |b|_2$                         | $\mathrm{rtol} \cdot |r_0|$ |
| 判据两边的范数  | 左边 $\sqrt{r^{\mathsf T} M^{-1} r}$, 右边 $|b|_2$, 不一致 | 一致, 可选                      |
| 批量右端项    | 聚合为 Frobenius 范数判一次                                 | 逐列判定、逐列冻结                   |
| 非正曲率     | 不检查                                                 | 停机                          |
| 退出信息     | `residual`, `niter`                                 | `ConvergedReason` 与完整诊断     |




## MINRES (未实现)

Hu–Zhang 混合元的鞍点系统对称但不定, CG 的收敛理论在其上不成立, 目前只能用直接法。
将来的迭代路径是预条件 MINRES, 要求预条件子自身正定。FEALPy 的 `fealpy/solver/minres.py`
可作算法参考; 鞍点系统的难点在块预条件子, 不在 MINRES 本身。原有的 `MINRESSolver` 占位类
已删除, 实现时再按 `LinearSolver` 契约新建。

## 测试与示例


| 路径                                            | 覆盖                                                                         |
| --------------------------------------------- | -------------------------------------------------------------------------- |
| `tests/unit/test_solvers_cg.py`               | 函数式 `cg`: 正确性, 批量右端项, 判据范数, 退出原因, 逐列冻结, `divtol`, `monitor`, `print_level` |
| `tests/unit/test_solvers_cg_solver.py`        | `CGSolver`: 能力协商, 预条件子级联, `relres`, 入口类型归一化                                |
| `examples/linear_solvers/verify_cg_solver.py` | 与直接法一致, 'fa' / 'ea' 同解, 判据范数, 真残差, 批量右端项, `info` 契约, 逐层迭代数                 |


