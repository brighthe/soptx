# 预条件子

本文说明 CG 可用的预条件子: 实现位置、使用条件与构造方式。参数与默认值见源码 docstring;
预条件子须满足的数学条件 (线性、对称、正定) 见 [`architecture.md`](architecture.md)
「组合约束」。

本文对应迭代法分类的轴二 (预条件子的构造); 轴一 (外层迭代格式) 与两轴的组合见
[`iterative.md`](iterative.md)「迭代法分类与实现状态」, 状态标注的含义也见该处。

## 构造类别与实现状态

| 构造类别 | 代表 | 能否配 PCG | SOPTX |
|---|---|---|---|
| 松弛类 | Jacobi, Gauss–Seidel, SOR, SSOR, 块 Jacobi | Jacobi, SSOR, 块 Jacobi 可以; 单向 Gauss–Seidel 不对称, 不可以 | Jacobi **已实现**: `DiagonalPreconditioner`, 键 `jacobi`; 阻尼 Jacobi **仅作组件** (多重网格光滑子); 其余未实现 |
| 不完全分解 | ILU, IC, ILUT | IC 可以; ILU 不对称, 不可以 | 未实现 |
| 多项式 | Chebyshev 多项式 | $q$ 在谱上取正值时可以 | 未实现, 只计划作光滑子 |
| 多重网格 | GMG, p-多重网格, AMG | 满足固定、对称、正定条件时可以 | GMG **已实现**: `Multigrid` + `StructuredHexHierarchy`, V 循环, 键 `gmg`; AMG 与 p-多重网格未实现 |
| 区域分解 | 加性 / 乘性 Schwarz | 加性 Schwarz 可以 | 未实现 |

此外 SOPTX 允许把直接法 (`DirectSolver`, 键 `scipy` / `mumps`) 作为精确逆填进 `M=` 位,
它不属于上述构造类别, 只用于验证两个装配层级是同一个离散算子。

未实现类别的取舍:

- **Chebyshev**: 只用 matvec 与谱区间估计, 适合 matrix-free 与 GPU, 计划作为 GMG 的光滑子;
- **AMG**: 需要显式矩阵, 矩阵变化后通常须重建, 定位为 'fa' 旁路与 GMG 的对照基线; 计划改为
  代数层次构造器, 复用 `Multigrid`, 不另起一套循环;
- **Gauss–Seidel, SOR, SSOR, 不完全分解**: 需要显式矩阵, 前代回代有串行依赖, 与 matrix-free
  和 GPU 主线冲突, 不在计划内;
- **区域分解**: 原分布式线已归档 (见 [`docs/archive.md`](../archive.md)), 暂无需求。

FEALPy 另有加性 Schwarz、AMS (H(curl)) 与 GAMG (AMG 已实现, 支持 V/W/F 循环, Gauss–Seidel
光滑), 均需要显式矩阵; 补 AMG 时可参考其 `amg_core`。

## 已实现的预条件子

`LagrangeFEMAnalyzer` 按 `solver_options['precond']` 查 `soptx.fem.preconditioners`
注册表; 不给 `precond` 即不带预条件子。

| 键 | 类别 | 核心实现 | 使用条件 | 用途 |
|---|---|---|---|---|
| `jacobi` | 对角 | `solvers/preconditioners.py` 的 `DiagonalPreconditioner` | 算子能给出对角; 'fa' 与 'ea' 都可用 | 最便宜, 迭代数随网格加密明显增长 |
| `gmg` | 几何多重网格 | `solvers/multigrid.py` 的 `Multigrid` + `fem/multigrid/structured_hex.py` 的 `StructuredHexHierarchy` | 结构化六面体网格, Q1 元, numpy 后端, 单元密度的拓扑优化 | 3D 主线, 迭代数基本不随网格加密增长 |
| `scipy` / `mumps` | 直接法精确逆 | `solvers/direct.py` 的 `DirectSolver` | 需要显式矩阵: 'fa' 直接可用, 'ea' 须设 `preconditioner_level='fa'`; 只在 CPU 上运行 | 仅用于验证: CG 应一步收敛 |

注意 `solvers/preconditioners.py` 只实现了对角预条件子与多重网格的光滑子, GMG 与直接法
分别在各自模块里; 统一按键分派的是 fem 侧的注册表。

公共入口是 `LagrangeFEMAnalyzer._make_preconditioner`: 查表构造、确定预条件算子、`setup`,
并在能力协商失败时补上下文报错。预条件子先在 fem 侧的算子上 `setup` 再交给 CG, CG 只对
未 `setup` 的预条件子做级联。

## `preconditioner_level`

预条件子默认绑定主算子本身。给出 `preconditioner_level` 时, 分析器另建一个该层级的算子
只供预条件子使用 (`_preconditioner_operator`), 例如主算子走 'ea' 省内存, 直接法预条件子
仍落在 'fa' 的显式矩阵上。两者都已施加边界条件, 对预条件子而言等价。

## 对角 (`jacobi`)

$M^{-1} r = r / \mathrm{diag}(A)$。对角在 `setup` 时经 `solvers.base.operator_diagonal`
向算子要, 也可在构造时显式给出:

- 'fa': 扫对称消元后稀疏矩阵的 COO 取主对角;
- 'ea' 等层级: 边界条件算子 `ConstrainedOperator.diagonal()` 在 Dirichlet 自由度上取 1,
  其余转发到装配层级 (`fem/levels/`) 各自的 `diagonal()`;
- 子结构路径: 自行算好接口矩阵的对角, 构造时显式传入。

对角含非正元素时报错, 不能用于 SPD 系统。

## 几何多重网格 (`gmg`)

由三部分组成:

| 部分 | 位置 | 内容 |
|---|---|---|
| V 循环 | `solvers/multigrid.py` 的 `Multigrid` | 通用循环引擎, 层次由外部传入; 前后光滑次数须相等, 使 V 循环对称; 作预条件子时零初值跑一次循环 |
| 层次 | `fem/multigrid/structured_hex.py` 的 `StructuredHexHierarchy` | 网格粗化、三线性延拓与 Galerkin 粗层 |
| 光滑子与最粗层 | `solvers/preconditioners.py` 的 `JacobiSmoother`; `solvers/direct.py` 的 `DirectSolver` | 每层加权 Jacobi; 最粗层直接法精确求解 |

### 层次的构造

- 粗网格各方向单元数取 $\lceil n/2 \rceil$, 奇数边上伸出计算域的子单元系数取零。
- 延拓 $P$ 为三线性插值; $P_0$ 把受约束细自由度所在的行置零, 粗层只作用在自由自由度上,
  支承不必落在粗节点上。
- 粗层遵循 Galerkin 原则 $A_{l+1} = P_0^{\mathsf T} A_l P_0$。第 1 层到第 2 层按单元组合
  $K_E = \sum_{c=1}^{8} s_{e(c)} P_c^{\mathsf T} K^0 P_c$, 不需要细层全局矩阵, 因此 'ea' 也能用;
  更粗层直接做稀疏 Galerkin 投影。
- 粗化持续到最粗层自由度不超过上限为止, 至少粗化一层。
- 层次的拓扑部分首次构造后缓存, 换空间时作废; 每次求解按当前单元系数更新粗层。

粗层系数取自装配层级的单元系数, 只有 'fa' 与 'ea' 下单元密度的拓扑优化带这组系数,
其余场景报错。层次内部用 `scipy.sparse` 构造, 目前只支持 numpy 后端。

### 光滑子

加权 Jacobi $x \leftarrow x + \omega D^{-1}(b - Ax)$。$\omega$ 缺省时取
$4 / (3 \cdot 1.1\, \hat\lambda)$, $\hat\lambda$ 为 `estimate_lambda_max` 对 $D^{-1}A$ 的
幂迭代估计 (Rayleigh 商, 不超过真值, 1.1 抵消低估)。固定的 $\omega = 0.6$ 不稳健: 单元
系数剧烈跳变时 $\lambda_{\max}(D^{-1}A)$ 可达 4.6, 光滑发散, 作 PCG 预条件子会失去正定性。

### 选项

分析器读取 `gmg_omega`, `gmg_sweeps`, `gmg_coarse_solver` 与 `gmg_coarse_max_dofs`,
含义与默认值见 `LagrangeFEMAnalyzer._gmg_preconditioner` 的 docstring。

### 子结构变体

`SubstructureAnalyzer` 在 `trace='full_trace'` 下用 `RestrictedPreconditioner`
(`fem/substructure/implicit.py`) 把细网格 GMG 限制到接口自由度: $R B R^{\mathsf T}$, $B$ 为
细网格上的一次 V 循环。`linear_corner` 下接口未知量不是细网格自由度, 不支持。这条路径
不经过注册表。

## 直接法精确逆 (`scipy` / `mumps`)

`LinearSolver.__matmul__` 本就是"零初值解一次"的预条件子模式, 直接法填进 `M=` 位即为精确
逆, CG 应一步收敛。主要用途是验证两个装配层级确实是同一个离散算子。分解求解后随
`run_solver` 一并释放。

## 测试

| 路径 | 覆盖 |
|---|---|
| `tests/unit/test_fem_preconditioners.py` | 注册表: 已注册键, `gmg` 委托分析器, 未知键报错 |
| `tests/unit/test_multigrid_solver.py` | `JacobiSmoother`, `estimate_lambda_max`, `Multigrid` (一维 Poisson) |
| `tests/unit/test_multigrid_structured_hex.py` | 粗层与代数 Galerkin 逐元一致, MGCG 与直接法同解, 层次跨密度复用, 'fa' / 'ea' 迭代数一致 |
| `tests/integration/test_lagrange_fem_analyzer_preconditioner_level.py` | `preconditioner_level`, Jacobi 与直接法预条件 |
| `tests/unit/test_substructure_analyzer.py`, `tests/unit/test_implicit_schur.py` | 子结构路径的 GMG 与 Jacobi |
