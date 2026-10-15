# 直接法

本文说明 `soptx.solvers.direct` 的直接法: 有状态的 `DirectSolver` 与无状态的函数式
`spsolve`。参数与默认值见源码 docstring; 求解层的整体结构见 [`architecture.md`](architecture.md)。

## 后端与用途

| 注册键 | 后端 | 说明 |
|---|---|---|
| `scipy` | SuperLU (`scipy.sparse.linalg.splu`) | 零额外依赖, 支持二维的多右端项 |
| `mumps` | MUMPS (PyMUMPS) | 只支持一维右端项, 多右端须逐列调用 `solve`; `sym` 取 1 或 2 时只传下三角 |

两个后端都只在 CPU 上运行, 求解层没有 GPU 直接法。

直接法在 SOPTX 中的用途:

- `'fa'` 层级的中小规模算例;
- Hu–Zhang 鞍点系统 (目前唯一可用的解法);
- 子结构接口系统;
- 几何多重网格的最粗层;
- 作为精确逆预条件子, 验证两个层级是同一个离散算子。

3D 嵌套剖分的因子存储是 $\mathcal{O}(n^{4/3})$, 大规模细网格不宜直接分解, 走 MGCG。

## `DirectSolver`

相对 `spsolve` 的实质区别是**分解被缓存**: `setup` 做分解, `solve` 只做回代。同一矩阵要
反复回代时这是必需的: 作多重网格最粗层时每个 V 循环回代一次, 作 CG 的预条件子时每步
回代一次。矩阵数值改变 (如密度更新) 后须重新 `setup`。

- 需要显式矩阵 (`requires = {CAP_MATRIX}`), 'ea' 层级的 matrix-free 算子在 `setup` 处被拒。
- scipy 路径先复制矩阵再分解, SuperLU 的原地改写不会破坏调用方持有的矩阵。
- 每次 `solve` 后按 $\|b - Ax\| / \|b\|$ 现算真残差填进 `relres`, 并以 `residual_tol`
  判定 `converged`: 接近奇异的系统下两个后端都可能不报错地给出错误解。
- 持有 MUMPS 上下文这一非 Python 资源; 重复 `setup`, `close()`, `with` 语句与析构都会释放它。
- MUMPS 分解报工作空间不足 (`INFOG(1) = -8/-9`) 时, 按 `ICNTL(14) = 100, 400, 1600`
  依次放大重试, 用尽仍失败才抛出。
- MUMPS 路径自行调用 `soptx.core.mpi_runtime.ensure_mpi_initialized`, 调用方不必准备 MPI 上下文。

## 函数式 `spsolve`

每次调用都重新分解, 适合一次性求解; 同一矩阵多次求解应改用 `DirectSolver`。`src/` 中
已无调用方, 只在测试、示例与实验中使用。

## MUMPS 的 `job` 划分

两条 MUMPS 路径的 `job` 划分不同:

| 路径 | `job` | 说明 |
|---|---|---|
| `DirectSolver.setup` | 4 | 符号分析 + 数值分解, 因子在实例内缓存 |
| `DirectSolver._solve` | 3 | 只回代 |
| 函数式 `spsolve` | 6 | 三阶段一次做完 |

`job=4` 仍把符号分析与数值分解绑在一起。拓扑优化每步在同一网格上重装配, 稀疏结构不变,
因此每次 `setup` 都在重做符号分析; 拆成全程一次 `job=1` 加每步 `job=2` 是尚未做的改造。
它与"分解不跨调用复用" (分析器求解后即 `close()`) 是两层不同的浪费。

两条路径都用 `set_centralized_sparse` 传矩阵、在 host 上取回解, 只能单进程运行;
改成分布式输入需设 `ICNTL(18)=3`。

## 依赖

`mumps` 后端需要 PyMUMPS 与 MUMPS 库, 两种装法:

- 推荐 `conda install -c conda-forge pymumps`, MUMPS 库随之装好;
- `pip install -e ".[mumps]"`, 从源码编译 PyMUMPS, 须预装系统 MUMPS 库与头文件
  (Ubuntu 24.04: `libmumps-dev`)。

未安装时选用 `mumps` 后端会抛 `ImportError` 并给出上述提示。`scipy` 随 SciPy 自带。

## 与 FEALPy 的关系

`direct.py` 以 FEALPy `fealpy/solver/direct.py` 为参考重新设计。函数式路径与来源一致,
差异见各函数 docstring; `DirectSolver` 为 SOPTX 自有, FEALPy 无对应类。FEALPy 的
`DirectSolverManager` 另支持 Pardiso, CHOLMOD 与 cupy, SOPTX 未引入。

直接法的并行模型、三阶段接口语义与各实现的 GPU 支持对照见
`dut-postdoc:concepts/linear-solvers/direct-methods.md`, 本文只记录本仓库的实现事实。

## 测试与示例

| 路径 | 覆盖 |
|---|---|
| `tests/unit/test_solvers_direct.py` | `spsolve` |
| `tests/unit/test_solvers_direct_solver.py` | `DirectSolver`: 分解缓存, 能力协商, `ICNTL(14)` 重试, 资源释放 |
| `examples/linear_solvers/verify_direct_solvers.py` | 完整矩阵残差, 收敛阶, 矩阵是否被改写, `sym` 守卫 |
