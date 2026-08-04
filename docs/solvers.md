# 求解器

本文说明 SOPTX 求解状态方程 `K u = f` 时的求解方式、各分析器上的可用组合,以及
`mumps` 后端的依赖。默认值与运行行为以源码为事实源,这里的描述与分析器 docstring
保持一致。

## 求解方式

| 名称 | 类别 | 分派入口 | 额外依赖 | 说明 |
|------|------|----------|----------|------|
| `scipy` | 直接法 | `fealpy.solver.spsolve(solver='scipy')` | 无 | 纯 SciPy 生态, 未装 MUMPS 时的回退选择 |
| `mumps` | 直接法 | `fealpy.solver.spsolve(solver='mumps')` | PyMUMPS 包 + 系统 MUMPS 库 | 大规模稀疏直接求解 |
| `cg` | 迭代法 | `fealpy.solver.cg` | 无 | 只要求 matvec, 不装配全局矩阵 |

`scipy` 与 `mumps` 都是直接法, 差别在数值后端; 分析器对它们走同一条
`fealpy.solver.spsolve(K, F, solver=...)` 分派路径, 切换求解器只改一个参数,
不影响装配与边界处理。

## 各分析器的可用组合

| 分析器 | 状态方程结构 | 可用 `solve_method` |
|--------|--------------|---------------------|
| `LagrangeFEMAnalyzer` | 对称正定弹性刚度阵 | `'fa'` 层级: `'scipy'`/`'mumps'`; `'ea'` 层级: `'cg'` |
| `HuZhangMFEMAnalyzer` | 对称不定鞍点系统 | `'scipy'`/`'mumps'` |

两个分析器的默认 `solve_method` 都是 `'mumps'`。

- `LagrangeFEMAnalyzer` 的 `operator_level='ea'` 只保留单元矩阵, 没有可分解的
  全局矩阵, 直接法会被 `solve_system` 拒绝, 应改用 `'cg'`。
- `HuZhangMFEMAnalyzer` 的状态方程是鞍点系统, 只能用直接法, 构造期即拒绝
  迭代解法。

## 依赖

`mumps` 后端需要:

- PyMUMPS 包: `pip install pymumps`;
- 系统 MUMPS 库, 如 Debian/Ubuntu 的 `libdmumps-5-dev` 及配套依赖。

`scipy` 与 `cg` 后端随 SciPy 自带, 无额外依赖。

## 其它调用点

旧拓扑优化目标 `src/soptx/topology/objectives/compliance.py` 直接调用
`fealpy.solver.spsolve(..., solver='mumps')`, 使用相同的 PyMUMPS 后端。

> 机器相关的运行环境配置(如 MPI ABI)不在本仓库文档范围, 见各机器的环境说明。
