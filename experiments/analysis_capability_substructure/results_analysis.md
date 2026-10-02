# 精确子结构分析

## 1. 研究范围与验证对象

本文研究给定密度场下的精确子结构静力分析，覆盖 `full_trace + 精确` 和 `linear_corner + 精确` 两种组合，考察分析正确性、密度切换后的计算一致性及计算成本。验证范围包括局部刚度装配、局部静力缩聚、接口装配求解和内部位移恢复，不包含优化器密度更新。

| 精确分析组合 | 精度参考基准 | 主要评价内容 |
|---|---|---|
| `full_trace + 精确` | 同网格 FA；制造解算例另以解析解评价收敛阶 | 与 FA 的等价性及收敛阶，密度切换后的计算一致性，相对 FA 的计算成本 |
| `linear_corner + 精确` | 同迹空间精确参照（由 FA 刚度与全场延拓构造）；同网格 FA 用于评价降阶误差 | 同迹空间内的计算正确性及相对 FA 的降阶误差，密度切换后的计算一致性，相对 FA 的计算成本 |

与 [PIML 子结构分析](../analysis_capability_piml_substructure/results_analysis.md) 的章节对应如下：本文第 2 节对应 PIML 文档 §2.1 的子结构配置与其 §6.1 第 2 步的整体结构创建；本文第 3 节对应 PIML 文档 §6.1 第 3—7 步，其中 PIML 以网络预测代替本文 3.2 节的精确缩聚，其余步骤相同。PIML 文档中的同接口空间精确参考即本文第 2—3 节的流程。

## 2. 子结构与接口空间构建

子结构与接口空间的构建是分析之前的第一步：先确定整体布局与共享的参考子结构，再创建精确缩聚器与接口空间。缩聚器只记录内部与接口自由度的划分，实际缩聚在第 3.2 节随密度执行。这些对象只依赖网格与接口空间，与密度无关，密度切换时可以复用。

### 2.1 子结构配置与接口空间

下面集中列出构建时需要设置的参数，再创建子结构及接口空间。子结构参数与 [PIML 文档 §2.1](../analysis_capability_piml_substructure/results_analysis.md) 一致，整体排列取其 §6.1 的 `n_sub = (78, 13, 13)`；因此本段即 PIML 同接口空间精确参考的构建过程。本段为示意配置，未登记进 [cases.toml](cases.toml)；第 4—6 节各算例的实际配置见对应小节。第 2—3 节的代码由 [walkthrough.py](walkthrough.py) 按同一顺序、同一变量名串成可运行脚本，默认取小规模排列 `12x2x2`，`--n-sub 78 13 13` 即本段排列。

相关源码：[assembler.py](../../src/soptx/fem/substructure/assembler.py) 的 `GlobalAssembler`、[mesh.py](../../src/soptx/fem/substructure/mesh.py) 的 `build_substructures` 与 `SubstructurePrototype`、[condensation.py](../../src/soptx/fem/substructure/condensation.py) 的 `ExactSchurCondensation`、[traces/](../../src/soptx/fem/substructure/traces/) 的 `FullTraceBasis` 与 `LinearCornerTraceBasis`

```python
from scipy.sparse import csr_matrix, identity
from soptx.fem.substructure import (
    ExactSchurCondensation, FullTraceBasis, GlobalAssembler, InterfaceDofsView,
    LinearCornerTraceBasis, build_substructures,
)

# 子结构配置.
cell_size = (1.0, 1.0, 1.0)    # 三维单位立方体.
n_fine = (5, 5, 5)             # 各方向细单元数, 共 125 个细单元.
nu = 0.3                       # 泊松比.
trace_kind = "linear_corner"   # 接口空间: linear_corner 或 full_trace.
hypothesis = None              # 三维各向同性线弹性; 二维可选 plane_stress 或 plane_strain.
E_base = 1.0                   # 实体材料杨氏模量.

# 整体排列, 13,182 个子结构, 1,647,750 个细单元.
n_sub = (78, 13, 13)
domain = tuple(c * n for c, n in zip(cell_size, n_sub))

# 创建全局装配器: 记录整体布局与材料参数, 负责后续接口系统装配与自由度映射.
assembler = GlobalAssembler(
    domain, n_sub, n_fine, E_base=E_base, nu=nu, hypothesis=hypothesis,
)

# 创建子结构: build_substructures 内部先创建一次参考子结构 prototype,
# 再铺开全部子结构 sub_meshes, 它们共用这同一个 prototype.
# positions 为各子结构在子结构网格中的整数位置.
prototype, sub_meshes, positions = build_substructures(assembler)
# prototype 默认 SIMP 惩罚指数 penal=3.0, rho_min=0.0; 输入为密度.

# 精确缩聚器: 按内部与边界自由度划分.
condensor = ExactSchurCondensation(prototype.i_dofs, prototype.b_dofs)

# 创建接口空间: 全局接口自由度, 局部迹基 T_j 与全局迹延拓 P (U_Gamma = P U_C).
interface_dofs = assembler.build_interface_dofs(sub_meshes)
interface_view = InterfaceDofsView(global_dofs=interface_dofs)
if trace_kind == "full_trace":
    trace = FullTraceBasis.from_prototype(prototype)
    projection = csr_matrix(identity(len(interface_dofs)))  # T_j = I, 故 P = I.
else:
    trace = LinearCornerTraceBasis.from_prototype(prototype)
    projection = assembler.build_linear_corner_projection(
        sub_meshes, interface_view, trace,
    )
```

接口空间分两层：局部迹基 `trace` 给出单个子结构的 $\mathbf T_j$，由参考子结构构造一次、全部子结构共用；全局层面，`interface_dofs` 为全部子结构接口自由度升序去重后的全局编号，迹延拓 $\mathbf P$ 把接口系统的未知量 $\mathbf U_C$ 映射为完整接口位移 $\mathbf U_\Gamma=\mathbf P\mathbf U_C$，供边界处理（§3.4）与位移恢复（§3.5）使用。`full_trace` 的接口系统即以 `interface_dofs` 为自由度（`assemble_interface_system` 内部调用同一个 `build_interface_dofs`），因此 $\mathbf U_C=\mathbf U_\Gamma$，$\mathbf P=\mathbf I$；`linear_corner` 的 $\mathbf U_C$ 为宏观角点位移，$\mathbf P$ 由各子结构的 $\mathbf L_j$ 按全局编号拼成。两种接口空间此后走同一条代码路径，分支只出现在这里。`interface_view` 只携带 `global_dofs`，为载荷与支承投影及位移恢复提供完整接口编号。

`build_substructures()` 按装配器布局铺开全部子结构。全部子结构同构，离散结构、自由度划分与单位密度单元刚度 `KE_unit` 只在 `prototype` 中构造一次；`positions` 记录各子结构在子结构网格中的整数位置，用于局部到全局的自由度映射。

### 2.2 接口迹空间

两种接口的局部 Schur 缩聚都是精确计算，内部延拓算子 $\mathbf N_{\mathrm{int}}^j=-(\mathbf K_{ii}^j)^{-1}\mathbf K_{ib}^j$ 也相同，差别只在接口位移被限制在哪个空间内。

| 迹空间 | 迹基 $\mathbf T_j$ | 接口坐标 $\mathbf q_j$ | 内部位移恢复 |
|---|---|---|---|
| `full_trace` | $\mathbf I$，保留完整接口 | $\mathbf u_b^j$，完整接口位移 | 接口系统直接给出 $\mathbf u_b^j$，按 $\mathbf u_i^j=\mathbf N_{\mathrm{int}}^j\mathbf u_b^j$ 恢复 |
| `linear_corner` | $\mathbf L_j$，二维沿边线性插值、三维沿面双线性插值 | $\mathbf u_c^j$，宏观角点位移 | 先由 $\mathbf U_\Gamma=\mathbf P\mathbf U_C$ 迹延拓得到 $\mathbf u_b^j$，再用同一恢复式 |

设空间维数为 $d$，`n_fine` 各方向为 $m_1,\dots,m_d$（Q1 单元），则单个子结构的内部自由度数 $n_i$、完整接口自由度数 $n_b$ 与角点接口自由度数 $n_c$ 为

$$
n_i=d\prod_{k=1}^{d}(m_k-1),\qquad
n_b=d\left[\prod_{k=1}^{d}(m_k+1)-\prod_{k=1}^{d}(m_k-1)\right],\qquad
n_c=d\,2^d.
$$

本文各算例的子结构规模如下。

| 算例 | 维数 | `n_fine` | $n_i$ | $n_b$（`full_trace`） | $n_c$（`linear_corner`） |
|---|---|---|---:|---:|---:|
| 正确性与收敛性 | 2 | `2x2` | 2 | 16 | 8 |
| 正确性与收敛性 | 3 | `2x2x2` | 3 | 78 | 24 |
| 密度更新一致性、二维计算成本 | 2 | `5x5` | 32 | 40 | 8 |
| 密度更新一致性 | 3 | `4x4x4` | 81 | 294 | 24 |
| §2.1 示例（与 PIML 文档一致） | 3 | `5x5x5` | 192 | 456 | 24 |

## 3. 精确分析主线

沿用第 2 节的 `assembler`、`prototype`、`sub_meshes`、`condensor`、`trace`、`interface_view` 和 `projection`，给定全局单元密度 `density` 与弹性问题 `pde`（提供载荷与支承，如 `FullMBBBeam3d`）后，一次分析依次执行局部刚度装配、静力缩聚、接口装配、边界处理与求解、完整位移恢复。

数学符号沿用[子结构有限元与静力缩聚](C:/workspace/dut-postdoc/concepts/substructural-condensation.md)。$i$、$b$ 分别表示子结构内部与接口自由度。当前实现采用内部载荷 $\mathbf f_i^j=\mathbf0$ 的模型；`full_trace` 保留完整接口，`linear_corner` 在精确缩聚后引入接口迹降阶。以下代码摘录核心语句，省略参数检查、计时与日志。

| 步骤 | 输入 | 输出 | 接口 |
|---|---|---|---|
| 3.1 局部刚度装配 | 全局单元密度 | 各子结构局部刚度 $\mathbf K^j$ | `split_global_cell_field`、`assemble_local_stiffness_batch` |
| 3.2 静力缩聚 | $\mathbf K^j$ | $\mathbf K_s^j$、$\mathbf N_{\mathrm{int}}^j$ | `ExactSchurCondensation.condense` |
| 3.3 接口装配 | $\mathbf K_s^j$、迹基 | $\mathbf K_\Gamma$ 或 $\mathbf K_C$ | `assemble_trace_system` |
| 3.4 边界处理与求解 | 接口系统、载荷、支承 | $\mathbf U_\Gamma$ 或 $\mathbf U_C$ | `solve_interface_system`、`solve_constrained_system` |
| 3.5 完整位移恢复 | 接口位移、$\mathbf N_{\mathrm{int}}^j$ | 全局位移 | `recover_full_displacement` |

### 3.1 局部刚度装配

设 $\mathbf A_e$ 为子结构内的单元自由度提取矩阵，$\mathbf K_e^0$ 为实体材料的单元刚度，则

$$
\mathbf K^j=\sum_{e\in j}s_e\,\mathbf A_e^{\mathsf T}\mathbf K_e^0\mathbf A_e,
\qquad
s_e=\frac{E(\rho_e)}{E_0}.
$$

```python
local_density = assembler.split_global_cell_field(density)
local_stiffness = prototype.assemble_local_stiffness_batch(local_density)
```

原型复用实体单元刚度 `KE_unit`（即 $\mathbf K_e^0$），先按上式计算 `coef`（即 $s_e$），再由 `bm.einsum('be, eij -> beij', coef, self.KE_unit)` 得到各单元刚度，最后用 `bm.bincount` 散加为批量局部矩阵。

### 3.2 静力缩聚

相关源码：[condensation.py](../../src/soptx/fem/substructure/condensation.py) 的 `ExactSchurCondensation.condense`

$$
\mathbf N_{\mathrm{int}}^j=-(\mathbf K_{ii}^j)^{-1}\mathbf K_{ib}^j,\qquad
\mathbf K_s^j=\mathbf K_{bb}^j-\mathbf K_{bi}^j(\mathbf K_{ii}^j)^{-1}\mathbf K_{ib}^j.
$$

```python
condensed, recovery = condensor.condense(local_stiffness)

# condense() 内部核心语句.
K_ii = K_local[..., self.i_dofs[:, None], self.i_dofs]
K_ib = K_local[..., self.i_dofs[:, None], self.b_dofs]
K_bb = K_local[..., self.b_dofs[:, None], self.b_dofs]
invK_ii_K_ib = bm.linalg.solve(K_ii, K_ib)
self.N = -invK_ii_K_ib
self.K_s = K_bb - bm.matrix_transpose(K_ib) @ invK_ii_K_ib
```

`bm.linalg.solve` 沿批量维求解，不显式构造逆矩阵；实现利用弹性刚度的对称性，以 $(\mathbf K_{ib}^j)^{\mathsf T}$ 表示 $\mathbf K_{bi}^j$。代码中的 `self.K_s` 对应 $\mathbf K_s^j$，`self.N` 对应 $\mathbf N_{\mathrm{int}}^j$，分别用于接口装配与内部位移恢复。

### 3.3 接口装配

相关源码：[assembler.py](../../src/soptx/fem/substructure/assembler.py) 的 `GlobalAssembler.assemble_trace_system`、`assemble_interface_system` 与 `assemble_macro_system`

迹基 $\mathbf T_j$ 与接口坐标 $\mathbf q_j$ 的定义见 2.2 节，$\mathbf u_b^j=\mathbf T_j\mathbf q_j$，迹空间刚度为

$$
\mathbf K_r^j=\mathbf T_j^{\mathsf T}\mathbf K_s^j\mathbf T_j.
$$

`full_trace` 下 $\mathbf K_r^j=\mathbf K_s^j$；`linear_corner` 下记 $\mathbf K_c^j=\mathbf L_j^{\mathsf T}\mathbf K_s^j\mathbf L_j$。分别以 $\mathbf A_j$、$\mathbf A_{c,j}$ 提取局部完整接口与角点自由度，全局装配为

$$
\mathbf K_\Gamma=\sum_j\mathbf A_j^{\mathsf T}\mathbf K_s^j\mathbf A_j,\qquad
\mathbf K_C=\sum_j\mathbf A_{c,j}^{\mathsf T}\mathbf K_c^j\mathbf A_{c,j}.
$$

两条路径统一调用

```python
system = assembler.assemble_trace_system(
    sub_meshes, condensor, trace_basis=trace,
)
```

`assemble_trace_system` 根据迹基类型选择已有底层装配：`FullTraceBasis` 直接进入完整接口 CSR pattern 装配，不计算 $\mathbf I^{\mathsf T}\mathbf K_s^j\mathbf I$；`LinearCornerTraceBasis` 先计算 $\mathbf L_j^{\mathsf T}\mathbf K_s^j\mathbf L_j$，再装配宏观角点系统。`FullTraceBasis` 当前仍以 `bm.eye` 保存恒等迹矩阵，但统一入口不会用该矩阵执行刚度投影。未知迹基因没有明确的全局自由度映射而直接报错。

接口拓扑不变时，`full_trace` 可复用 CSR pattern；`linear_corner` 可按批投影并装配。两者的局部 Schur 缩聚均为精确计算，`linear_corner` 的近似来自接口位移被限制在角点迹空间内。

### 3.4 边界处理与求解

相关源码：[problem_adapter.py](../../src/soptx/fem/substructure/problem_adapter.py) 的 `project_problem_conditions_to_interface_system`、[solve.py](../../src/soptx/fem/substructure/solve.py) 的 `solve_interface_system` 与 `solve_constrained_system`

载荷与支承先映射到完整接口，再经迹延拓 $\mathbf P$（§2.1）变换到接口系统的未知量 $\mathbf U_C$ 上：载荷为 $\mathbf F_C=\mathbf P^{\mathsf T}\mathbf F_\Gamma$；完整接口上的支承变为线性约束 $\mathbf C_D\mathbf U_C=\mathbf d_D$，其中 $\mathbf C_D$ 取 $\mathbf P$ 在受约束接口自由度上的行。约化为独立约束后求解

$$
\begin{pmatrix}\mathbf K_C&\mathbf C_D^{\mathsf T}\\\mathbf C_D&\mathbf0\end{pmatrix}
\begin{pmatrix}\mathbf U_C\\\boldsymbol\lambda_D\end{pmatrix}
=\begin{pmatrix}\mathbf F_C\\\mathbf d_D\end{pmatrix},
$$

$\boldsymbol\lambda_D$ 为支承约束反力的乘子。`full_trace` 下 $\mathbf K_C=\mathbf K_\Gamma$，$\mathbf C_D$ 为单位阵的若干行（坐标选择约束），上式退化为直接消元：记 $F$、$D$ 为接口的自由与给定位移自由度集合，求解 $(\mathbf K_\Gamma)_{FF}(\mathbf U_\Gamma)_F=(\mathbf F_\Gamma)_F-(\mathbf K_\Gamma)_{FD}(\mathbf U_\Gamma)_D$，不引入乘子。

```python
conditions = project_problem_conditions_to_interface_system(
    pde, assembler, interface_view,
)
macro_force = projection.T @ conditions.interface_force
constraints = projection[conditions.interface_fixed_dofs]
solved = solve_constrained_system(
    system, macro_force, constraints, solver="mumps",
)
macro_u = solved.displacement
```

`project_problem_conditions_to_interface_system` 按 `pde.loads()` 与 `pde.is_dirichlet_boundary()` 在细网格节点上生成载荷与支承（`full_force`、`full_fixed_dofs`），再取出其中的完整接口分量（`interface_force`、`interface_fixed_dofs`），即 $\mathbf F_\Gamma$ 与受约束接口自由度；[analyzers/substructure.py](../../src/soptx/fem/analyzers/substructure.py) 的 `FullInterfaceSubstructureAnalyzer` 也由它准备载荷与支承。静力缩聚不缩聚载荷，载荷或支承落在子结构内部自由度上时直接报错。该函数只读取支承位置、按零位移处理，不检查 `pde.dirichlet_bc` 是否为零；点力作用点不在节点上时取最近的边界节点。

`solve_constrained_system` 忽略零行且右端为零的约束，将重复或线性相关的行约化为独立约束；每行只有一个非零元的坐标选择约束复用 `solve_interface_system` 消元（`full_trace` 总是如此，返回的 `mode` 标签为“固定角点消元”），混合约束通过稀疏 Lagrange 乘子系统求解。返回值同时给出约束秩、平衡相对残差与约束相对残差。非零给定位移通过 `prescribed` 参数传入；本文的密度更新与成本工况均采用零位移支承。

第 4—6 节的实测入口在载荷与求解两处与上面的写法不同。载荷与支承由 `_density_update._conditions` 经同网格 FA 分析器的 `assemble_external_load` 与 `boundary_interpolate` 生成，点力要求精确落在节点上，并另行检查非齐次约束；作用点落在节点上时两者应给出相同的载荷与支承（经 `_conditions` 的工况均满足这一条件，否则 FA 路径直接报错），但尚无单元测试逐项比对；[walkthrough.py](walkthrough.py) 的同网格 FA 自检可间接检查。求解按接口空间分别调用：`full_trace` 直接以 `system` 投影载荷与支承并调用 `solve_interface_system`，与上面的统一写法求解同一消元方程，只是省去 $\mathbf P=\mathbf I$ 的构造与约束秩检查。`solve_constrained_system` 用于第 4、5 节的 `linear_corner` 工况。第 6 节成本工况的 `_cost_measurement.py` 未调用该函数，而是内联实现同一鞍点系统：对 `projection[full_fixed]`（受约束接口自由度对应的 $\mathbf P$ 行）的活跃列做列主元 QR 提取独立约束行，用 `scipy.sparse.bmat` 组装鞍点矩阵后以 MUMPS 求解。两种实现尚未做逐项对照。

### 3.5 完整位移恢复

相关源码：[condensation.py](../../src/soptx/fem/substructure/condensation.py) 的 `StaticCondensationBase.recover`、[assembler.py](../../src/soptx/fem/substructure/assembler.py) 的 `GlobalAssembler.recover_full_displacement`

先得到完整接口位移，再按 $\mathbf u_i^j=\mathbf N_{\mathrm{int}}^j\mathbf u_b^j$ 恢复各子结构内部位移：

```python
interface_u = bm.asarray(
    projection @ bm.to_numpy(macro_u), dtype=bm.float64,
)
displacement = assembler.recover_full_displacement(
    sub_meshes, condensor, interface_view, interface_u,
)
```

恢复器用 `bm.einsum('...ij, ...j -> ...i', self.N, u_b_bm)` 计算内部位移，再按全局编号写回。若内部载荷非零，还需载荷缩聚与 $\mathbf w_i^j=(\mathbf K_{ii}^j)^{-1}\mathbf f_i^j$ 恢复项；当前流程未包含这两项。

## 4. 正确性与收敛性

第 4—6 节的验证实验均调用第 3 节的流程，共用同一入口。全部工况登记在 [cases.toml](cases.toml)，通过 [run.py](run.py) 执行；以下命令在本实验目录下运行：

```bash
# 列出全部已注册工况.
python run.py --list

# 运行单个工况; 密度更新与成本工况可加 --monitor 显示 Worker 的 CPU 与内存.
python run.py --case <case-id>
```

每次运行在 `outputs/<case-id>/<UTC timestamp>/` 下新建目录，先写入 `run_config.json`，不覆盖已有输出。各工况的具体命令与保存文件见 4.3、5.3 与 6.1 节。

### 4.1 算例设置

取 $E=4/3$、$\nu=1/3$、$\rho=1$，体力为零。

| 算例 | 定义域 | 本构 |
|---|---|---|
| `HarmonicPoly2D` | $[0,1]^2$ | 平面应力 |
| `HarmonicPoly3D` | $[0,1]^3$ | 三维弹性 |

`full_trace` 在细网格外边界施加解析位移；`linear_corner` 在宏观边界角点赋值，其余边界位移由迹插值确定。

二维与三维解析位移分别为

$$
\mathbf u_{2D}(x,y)=\bigl(-12x^2y+4y^3,\,-4x^3+12xy^2\bigr)^{\mathsf T},
$$

$$
\mathbf u_{3D}(x,y,z)=\bigl(-12x^2y+4y^3,\,4z^3-12x^2z-4x^3+12xy^2,\,0\bigr)^{\mathsf T}.
$$

### 4.2 网格与子结构划分

二维采用规则四边形网格，三维采用规则六面体网格。两种接口均固定每个子结构的单元数，逐层增加子结构数量，使子结构尺寸和细网格尺寸同时减半。

| 维度（两种接口共用） | 子结构划分 `n_sub` | 块内单元数 `n_fine` | 全局细网格 |
|---|---|---|---|
| 二维 | `2x2 -> 4x4 -> 8x8 -> 16x16` | `2x2` | `4x4 -> 8x8 -> 16x16 -> 32x32` |
| 三维 | `2x2x2 -> 4x4x4 -> 8x8x8` | `2x2x2` | `4x4x4 -> 8x8x8 -> 16x16x16` |

### 4.3 运行命令与保存文件

`full_trace_convergence` 复用 `examples/substructure_elasticity/verify_full_trace_convergence.py`，并由统一入口开启同网格 FA 对照；`linear_corner_convergence` 由 [_corner_convergence.py](_corner_convergence.py) 执行，同迹空间参照由全细网格 FA 刚度与全场延拓构造，并另解一次细网格 FA 作为精度参照。

```bash
python run.py --case full_trace_convergence_2d
python run.py --case full_trace_convergence_3d
python run.py --case linear_corner_consistency_2d
python run.py --case linear_corner_consistency_3d
```

| 保存文件 | 内容 |
|---|---|
| `run_config.json` | 工况登记参数与运行选项 |
| `full_trace_convergence_<d>d_harmonic-poly_p1_levels<k>_solver-mumps.json` | `full_trace` 各加密层误差、观测阶、FA 对照及判定 |
| `linear_corner_convergence_<d>d_harmonic-poly_p1_levels<k>_solver-mumps.json` | `linear_corner` 各加密层一致性指标、解析解误差、观测阶及相对 FA 的误差 |

### 4.4 验证结果

`full_trace` 的 $L^2$、$H^1$ 半范收敛阶下限为 1.8、0.8。`linear_corner` 以同迹空间一致性验收，误差与阶数只报告；参照由 FA 刚度与全场延拓构造，内部延拓与缩聚路径共用。

四个 case 均通过验收，观测到 $L^2$ 约二阶、$H^1$ 半范约一阶收敛。本节结果均使用 MUMPS。

| case-id / 参照 | 最细网格 $L^2$ 误差 | 最细网格 $H^1$ 半范误差 | 最后 $L^2$ 阶 | 最后 $H^1$ 半范阶 | 判定 |
|---|---|---|---|---|---|
| `full_trace_convergence_2d` | $1.0087\times10^{-3}$ | $2.5001\times10^{-1}$ | 2.0006 | 1.0001 | PASS |
| `full_trace_convergence_3d` | $5.5267\times10^{-3}$ | $6.8471\times10^{-1}$ | 2.0019 | 1.0003 | PASS |
| `linear_corner_consistency_2d` | $4.6916\times10^{-3}$ | $5.3401\times10^{-1}$ | 2.0012 | 1.0001 | 一致性 PASS；阶数只报告 |
| `linear_corner_consistency_3d` | $2.3631\times10^{-2}$ | $1.4261$ | 2.0057 | 1.0006 | 一致性 PASS；阶数只报告 |
| FA（二维，`32x32`） | $1.0087\times10^{-3}$ | $2.5001\times10^{-1}$ | 2.0006 | 1.0001 | 阶数只报告 |
| FA（三维，`16x16x16`） | $5.5267\times10^{-3}$ | $6.8471\times10^{-1}$ | 2.0019 | 1.0003 | 阶数只报告 |

`linear_corner` 全部加密层、全部一致性指标的最大值：二维为 $2.08\times10^{-15}$，三维为 $4.61\times10^{-15}$，均低于 $10^{-9}$。

`full_trace` 与同网格 FA 的对照如下，各项取全部加密层的最大值。位移与应变能相对差均低于 $10^{-11}$，自由自由度相对残差均低于 $10^{-9}$。

| 维度 | 位移相对差 | 应变能相对差 | `full_trace` 相对残差 | FA 相对残差 | 判定 |
|---|---|---|---|---|---|
| 二维 | $1.41\times10^{-15}$ | $3.49\times10^{-15}$ | $8.73\times10^{-16}$ | $7.68\times10^{-16}$ | PASS |
| 三维 | $1.04\times10^{-15}$ | $3.44\times10^{-15}$ | $7.06\times10^{-16}$ | $5.75\times10^{-16}$ | PASS |

应变能取 $\frac12\mathbf u^{\mathsf T}K\mathbf u$；位移、应变能相对差分别以 FA 位移向量的范数、FA 应变能归一化。

全细网格相对平衡残差为

$$
\eta_{\mathrm{full}}
=\frac{\|K_{ff}u_f+K_{fc}u_c\|_2}{\|K_{fc}u_c\|_2},
$$

其中 $K$ 为全细网格刚度，$f$、$c$ 表示自由和受约束自由度；外载为零，由 $u_c$ 驱动。`full_trace` 代入恢复后的完整位移，FA 代入直接求解的位移。

`linear_corner` 相对 FA 的误差随加密减小，最细网格结果如下；误差包含边界迹插值和接口降阶的影响。

| 维度 | 最细网格 | 位移相对误差 | 应变能相对误差 |
|---|---|---|---|
| 二维 | `32x32` | 0.2233% | 0.1242% |
| 三维 | `16x16x16` | 0.9744% | 0.4797% |

结果来源：

- [二维 full_trace 收敛验证](outputs/full_trace_convergence_2d/20260914T012353860755Z/full_trace_convergence_2d_harmonic-poly_p1_levels4_solver-mumps.json)
- [三维 full_trace 收敛验证](outputs/full_trace_convergence_3d/20260914T012356132147Z/full_trace_convergence_3d_harmonic-poly_p1_levels3_solver-mumps.json)
- [二维 linear_corner 一致性与收敛验证](outputs/linear_corner_consistency_2d/20260914T012402105441Z/linear_corner_convergence_2d_harmonic-poly_p1_levels4_solver-mumps.json)
- [三维 linear_corner 一致性与收敛验证](outputs/linear_corner_consistency_3d/20260914T012404571145Z/linear_corner_convergence_3d_harmonic-poly_p1_levels3_solver-mumps.json)

## 5. 密度更新一致性

### 5.1 算例设置

二维采用 `CantileverCorner2d`（定义域 $[0,2]\times[0,1]$），三维采用 `FullMBBBeam3d`（定义域 $[0,6]\times[0,1]\times[0,1]$），均取 $E=1$、$\nu=0.3$、$P=-1$，两个算例均验证 `full_trace` 与 `linear_corner`。固定网格、子结构划分、材料参数、载荷与支承，仅更新单元密度。

密度依次切换为均匀场、非均匀场 A、非均匀场 B、初始均匀场。每步比较两条路径的结果：更新路径复用网格、子结构编号、迹映射及同一个 `ExactSchurCondensation` 实例，按第 3 节流程重新计算；重建路径按当前密度独立重新构造上述对象。最后检查是否恢复初始状态。两种接口各自对照，不要求 `linear_corner` 与 FA 等价。

### 5.2 网格与子结构划分

二维采用规则四边形网格，三维采用规则六面体网格。两种接口使用相同划分，各次密度切换保持网格不变。

| 算例 | 子结构划分 `n_sub` | 块内单元数 `n_fine` | 全局细网格 | 全细网格位移自由度 | `full_trace` 接口自由度 | `linear_corner` 接口自由度 |
|---|---|---|---|---:|---:|---:|
| `CantileverCorner2d` | `8x4` | `5x5` | `40x20` | 1,722 | 698 | 90 |
| `FullMBBBeam3d` | `6x2x2` | `4x4x4` | `24x8x8` | 6,075 | 4,131 | 189 |

自由度数均为施加支承约束前的数量，`linear_corner` 按宏观角点的位移分量计数。

### 5.3 运行命令与保存文件

实现见 [_density_update.py](_density_update.py)。验证在独立 Worker 进程中执行；更新路径每步分析结束后清空 `K_s` 和 `N`，比较结果先写入临时矩阵文件，释放后再构建重建路径，避免两套全量状态同时驻留内存。

```bash
python run.py --case full_trace_density_update_2d --monitor
python run.py --case full_trace_density_update_3d --monitor
python run.py --case linear_corner_density_update_2d --monitor
python run.py --case linear_corner_density_update_3d --monitor
```

| 保存文件 | 内容 |
|---|---|
| `run_config.json` | 工况登记参数、运行选项及密度更新协议 |
| `density_uniform.npy`、`density_pattern_a.npy`、`density_pattern_b.npy` | 三份实际使用的密度场 |
| `density_update_request.json`、`density_update_worker_result.json` | Worker 请求与返回记录 |
| `<case-id>_sub-<n_sub>_fine-<n_fine>.json` | 各密度状态的对照指标、残差、峰值内存及判定 |
| `worker_stdout.log`、`worker_stderr.log` | Worker 输出日志 |

比较用的临时矩阵文件在退出前清理。

### 5.4 验证结果

四个 case 均通过验证。相对差取全部密度状态、全部对照指标的最大值；残差另取更新与重建两条路径的最大值。

| case-id | 更新与重建的最大相对差 | 最大接口相对平衡残差 | 结果 |
|---|---:|---:|---|
| `full_trace_density_update_2d` | 0 | $1.32\times10^{-13}$ | PASS |
| `full_trace_density_update_3d` | 0 | $1.21\times10^{-13}$ | PASS |
| `linear_corner_density_update_2d` | 0 | $1.29\times10^{-14}$ | PASS |
| `linear_corner_density_update_3d` | 0 | $4.07\times10^{-14}$ | PASS |

相对差覆盖局部刚度、缩聚刚度、恢复矩阵、接口刚度、完整位移和应变能，接口稀疏结构也完全一致。恢复均匀场后，各 case 均通过与初始状态的对照。

接口相对平衡残差为

$$
\eta_{\mathrm{eq}}=\frac{\|(S u_\Gamma-f_\Gamma)_F\|_2}{\|(f_\Gamma)_F\|_2},
$$

其中 $S$、$u_\Gamma$、$f_\Gamma$ 分别为所选迹空间下的接口刚度、接口位移和接口载荷，$F$ 为未施加位移约束的接口自由度集合。`linear_corner` 检查角点降阶系统的平衡。

相对差与残差分别低于 $10^{-11}$、$10^{-9}$。由于每步主动清空数值状态，本验证覆盖重新计算与重建的一致性，不覆盖数值缓存的自动失效。

数据来源：

- [full_trace_density_update_2d](outputs/full_trace_density_update_2d/20260911T132310577837Z/full_trace_density_update_2d_sub-8x4_fine-5x5.json)
- [full_trace_density_update_3d](outputs/full_trace_density_update_3d/20260911T132316914143Z/full_trace_density_update_3d_sub-6x2x2_fine-4x4x4.json)
- [linear_corner_density_update_2d](outputs/linear_corner_density_update_2d/20260911T132326662756Z/linear_corner_density_update_2d_sub-8x4_fine-5x5.json)
- [linear_corner_density_update_3d](outputs/linear_corner_density_update_3d/20260911T132331743498Z/linear_corner_density_update_3d_sub-6x2x2_fine-4x4x4.json)

## 6. 计算成本

### 6.1 二维计算成本

#### 算例设置

采用 `CantileverCorner2d`，在相同网格、非均匀密度场 A、材料参数、载荷与支承下比较 FA、`full_trace` 和 `linear_corner`。各路径每次在独立进程中完整构建并分析，试运行 1 次、正式测量 3 次，报告正式测量的中位数。

#### 网格与子结构划分

采用规则四边形网格，共 512 万个单元。

| 子结构划分 `n_sub` | 块内单元数 `n_fine` | 全局细网格 | 全细网格位移自由度 | 接口自由度 | 接口自由求解自由度 |
|---|---|---|---:|---:|---:|
| `640x320` | `5x5` | `3200x1600` | 10,249,602 | 3,696,002 | 3,692,800 |

#### 运行命令与保存文件

实现见 [_cost_measurement.py](_cost_measurement.py)；FA 路径使用 [_fa_chunked.py](_fa_chunked.py) 的分块 pattern 装配。三条路径的精度对照在计时之外由 [collect_cost_comparison.py](collect_cost_comparison.py) 完成，输入为各路径首个正式样本的 `record.json`。

```bash
# 1. 三条路径的独立进程成本测量.
python run.py --case fa_cost_2d --monitor
python run.py --case full_trace_cost_2d --monitor
python run.py --case linear_corner_cost_2d --monitor

# 2. full_trace 首个正式样本未保存全场位移时, 补存一次参考位移.
python run.py --case full_trace_reference_2d --monitor

# 3. 三路径精度对照; 将 <时间戳> 替换为实际输出目录.
python collect_cost_comparison.py \
  --fa outputs/fa_cost_2d/<时间戳>/measurement_01/record.json \
  --full-trace outputs/full_trace_reference_2d/<时间戳>/measurement_01/record.json \
  --linear-corner outputs/linear_corner_cost_2d/<时间戳>/measurement_01/record.json \
  --output outputs/fa_cost_2d/<时间戳>/three_route_comparison.json
```

| 保存文件 | 内容 |
|---|---|
| `run_config.json` | 工况登记参数、计时与内存口径 |
| `warmup_01/`、`measurement_0k/` | 每次独立进程的 `request.json`、`record.json`（分阶段耗时、峰值 RSS、残差）及日志；首个正式样本另存 `displacement.npy` |
| `<case-id>_sub-640x320_fine-5x5.json` | 物理设置与输入指纹、测量环境、各样本及统计量、验收结果 |
| `three_route_comparison.json` | `full_trace`、`linear_corner` 相对 FA 的位移与应变能误差 |

#### 验证结果

三条路径各完成 1 次试运行和 3 次独立进程正式测量，均通过自身平衡与支承检查。同一路径三次密度与位移指纹一致，应变能相对差为 0。

FA 检查全细网格自由自由度平衡，`full_trace` 检查自由接口平衡，`linear_corner` 检查含约束反力的宏观平衡及线性约束。精度检查在性能计时之外进行。

| 路径 | 最大相对平衡残差 | 相对 FA 位移误差 | 相对 FA 应变能误差 |
|---|---:|---:|---:|
| FA | $2.03\times10^{-11}$ | — | — |
| `full_trace` | $1.08\times10^{-11}$ | $4.11\times10^{-9}$ | $2.70\times10^{-9}$ |
| `linear_corner` | $3.28\times10^{-12}$ | 3.12% | 11.06% |

各路径平衡残差均低于 $10^{-9}$。`full_trace` 相对 FA 的位移和应变能误差分别为 $4.11\times10^{-9}$ 和 $2.70\times10^{-9}$，数值结果高度一致。`linear_corner` 的两项误差分别为 3.12% 和 11.06%，反映角点迹降阶带来的精度损失。

成本对照取三次正式测量的中位数，时间单位为 s；试运行不参与统计，"—"表示无此阶段。各阶段与第 3 节流程的对应关系：局部刚度装配对应 3.1，局部缩聚对应 3.2，接口 pattern 与数值装配、迹投影与宏观装配对应 3.3，边界处理与求解对应 3.4，位移恢复对应 3.5。

| 阶段或指标 | FA | `full_trace` | `linear_corner` |
|---|---:|---:|---:|
| 全局／局部刚度装配 | 15.58 | 4.17 | 4.20 |
| 局部缩聚 | — | 25.09 | 22.95 |
| 接口 pattern 首次准备与构建 | — | 5.27 | — |
| 接口映射与数值装配 | — | 6.01 | — |
| 迹投影与宏观装配 | — | — | 1.65 |
| 边界处理与求解 | 124.01 | 98.59 | 7.67 |
| 位移恢复 | — | 1.54 | 1.60 |
| **分析阶段合计** | **139.22** | **140.57** | **38.63** |
| 问题准备（单列） | 150.77 | 156.13 | 164.78 |
| 峰值 RSS（GiB） | 34.22 | 39.52 | 25.14 |

三次分析阶段合计的范围分别为：FA 115.76–146.68 s、`full_trace` 136.99–143.61 s、`linear_corner` 37.22–39.58 s。每次合计取各分析阶段时间之和，再取三次中位数；各分项中位数相加不必等于合计中位数。残差检查、证据读写及阶段间日志不计入分析时间。

每个进程均重新构建装配结构。FA 使用每批 65536 个单元的分块 pattern 装配，pattern 构建与单位单元刚度准备计入全局装配；`full_trace` 的 pattern 成本单列。峰值 RSS 在完整位移得到后、正确性检查前读取，包含依赖加载、问题准备及结构分析。

本档 `full_trace` 与 FA 的分析耗时接近，峰值内存更高；`linear_corner` 分析耗时约为 FA 的 27.7%，但伴随上述降阶误差。FA 三次耗时存在明显波动，本组结果不支持两条精确路径间的小幅速度差异结论，也不代表规模增长趋势。

数据来源：[FA 成本](outputs/fa_cost_2d/20260914T022228026256Z/fa_cost_2d_sub-640x320_fine-5x5.json)、[full_trace 成本](outputs/full_trace_cost_2d/20260914T005217358092Z/full_trace_cost_2d_sub-640x320_fine-5x5.json)、[linear_corner 成本](outputs/linear_corner_cost_2d/20260914T020225864268Z/linear_corner_cost_2d_sub-640x320_fine-5x5.json)、[三路径精度对照](outputs/fa_cost_2d/20260914T022228026256Z/three_route_comparison.json)。

`full_trace` 另运行一次以保存全场位移，其位移指纹与应变能和原成本样本一致，仅用于精度对照，不并入计时统计：[参照结果](outputs/full_trace_reference_2d/20260914T021646825235Z/full_trace_cost_2d_sub-640x320_fine-5x5.json)。FA 原全批装配在内存接近上限时以 -9 退出；分块装配通过小规模矩阵、位移和能量对照后用于本次正式测量：[分块装配验证](outputs/fa_chunked_verification_2d/20260914T020500000000Z/fa_chunked_verify_2d_sub-8x4_fine-5x5.json)。

原连续密度切换记录保留在 [历史 JSON](outputs/full_trace_density_update_2d/20260911T092934958152Z/full_trace_density_update_2d_sub-640x320_fine-5x5.json)，不再作为本节正式性能表。

### 6.2 三维计算成本（待接入）

三维计算成本拟采用 Huang2023 的 `FullMBBBeam3d`，按论文规模及规模序列比较 FA、`full_trace` 与 `linear_corner`。当前成本工况的 Worker 仅支持二维（`_cost_measurement.py` 对 `dim != 2` 直接报错），`cases.toml` 中尚无三维成本工况。

| 路径 | 子结构划分 `n_sub` | 块内单元数 `n_fine` | 分析阶段合计 | 峰值 RSS | 相对 FA 位移误差 | 结果目录 / 汇总 |
|---|---|---|---:|---:|---:|---|
| FA | — | — | — | — | — | 尚未接入 |
| `full_trace` | — | — | — | — | — | 尚未接入 |
| `linear_corner` | — | — | — | — | — | 尚未接入 |

## 7. 产物来源与测试环境

新产物按 `outputs/<case-id>/<UTC timestamp>/` 隔离，运行配置由 `run_config.json` 记录；`outputs/` 不纳入版本控制，本文链接的产物位于主工作区。

以下测试环境取自二维成本工况结果 JSON 的 `environment` 字段：

* **操作系统**：Linux 6.18.33.2-microsoft-standard-WSL2 x86_64，glibc 2.39
* **计算软件栈**：Python 3.12.13，NumPy 2.5.1，SciPy 1.18.0，FEALPy 4.0.0，SOPTX 1.1.0.dev0
* **硬件设备**：32 个逻辑 CPU；未设置 `OMP_NUM_THREADS` 等线程数环境变量
