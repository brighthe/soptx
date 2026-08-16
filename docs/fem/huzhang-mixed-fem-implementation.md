# 胡张混合有限元实现

> 对应博士论文第五章。SOPTX 的胡张元实现覆盖 2D/3D 单纯形网格、任意次
> Bubble 丰富应力空间、角点松弛、低阶跳量稳定化，以及混合边界条件。

## 数学形式

胡张元是一种基于 $H(\mathrm{div})$ 顺应应力空间的混合有限元方法。直接在
应力-位移混合形式上离散：

$$
\begin{aligned}
&\text{求 } (\sigma_h, u_h) \in \Sigma_h \times V_h \text{ 满足} \\
&\quad \int_{\Omega} C^{-1} \sigma_h : \tau_h \; dx
      + \int_{\Omega} \mathrm{div}\,\tau_h \cdot u_h \; dx
      = \int_{\Gamma_D} (\tau_h \cdot n) \cdot u_D \; ds
      \quad \forall \tau_h \in \Sigma_h \\
&\quad \int_{\Omega} \mathrm{div}\,\sigma_h \cdot v_h \; dx
      = \int_{\Omega} f \cdot v_h \; dx
      \quad \forall v_h \in V_h
\end{aligned}
$$

- **应力空间 $\Sigma_h$**：$H(\mathrm{div})$ 顺应、对称张量值。在单纯形的
  subsimplex（顶点/边/单元面/单元体）上通过多指标构造 Bubble 基函数，
  保证法向迹连续。对称性通过 `symmetry_span_array` 将对称指标展开为
  独立分量。
- **位移空间 $V_h$**：分片不连续 Lagrange $(P_{k-1})$，张量值
  （维度 = GD）。
- **鞍点结构**：$k \ge \mathrm{GD}+1$ 时满足 inf-sup 条件，刚度矩阵为
  $\begin{bmatrix} A & B \\ B^T & 0 \end{bmatrix}$；$k \le \mathrm{GD}$ 时
  需跳量稳定化 $\begin{bmatrix} A & B \\ B^T & -J \end{bmatrix}$。

SOPTX 的实现直接操作应力空间基底，不为位移元引入额外的 Bubble 或
缩聚——这在拓扑优化中保持了应力场的物理可解释性。

## 程序架构

### 文件布局

```
src/soptx/fem/
├── spaces/
│   ├── huzhang_fe_space.py          ← 工厂，按 mesh.top_dimension() 分派
│   ├── huzhang_fe_space_2d.py       ← 2D 应力空间（DOF 枚举、basis、div_basis、
│   │                                  角点松弛 TM、create_huzhang_checkerboard_mesh）
│   └── huzhang_fe_space_3d.py       ← 3D 应力空间（无松弛）
├── integrators/
│   ├── huzhang_stress_integrator.py ← A 块：∫ C⁻¹ σ : τ（柔度双线性型）
│   ├── huzhang_mix_integrator.py    ← B 块：∫ div σ · u（应力-位移耦合）
│   └── jump_penalty_integrator.py   ← J 块：低阶跳量稳定化
└── solvers/
    └── huzhang_mfem_analyzer.py     ← 求解器：装配、边界条件、求解、后处理
```

### 核心类关系

```
HuZhangFESpace.__new__  ──按 TD 分派──▶  HuZhangFESpace2d  (应力空间 Σ_h)
                                     ▶  HuZhangFESpace3d

HuZhangMFEMAnalyzer
  ├── 持有 HuZhangFESpace            (应力空间，degree = p)
  ├── 持有 TensorFunctionSpace       (位移空间 V_h，Lagrange P_{p-1}，DG)
  ├── 持有 HuZhangStressIntegrator   (A 块装配器)
  ├── 持有 HuZhangMixIntegrator      (B 块装配器)
  └── 按需创建 JumpPenaltyIntegrator  (J 块装配器，p ≤ GD 时)
```

应力空间输出对称张量基底（2D: `[xx, xy, yy]`，3D: `[xx, xy, xz, yy, yz, zz]`），
通过 `symmetry_span_array` 编码对称指标。位移空间是标准的
`TensorFunctionSpace(LagrangeFESpace(p-1, DG), shape=(-1, GD))`。

### 求解流程

`solve_state()` 按以下顺序执行：

```
1. 标记边界
   bc → is_traction_boundary     → self._essential_bc  (牵引 Γ_N，强施加)
   bc → is_displacement_boundary → self._natural_bc    (位移 Γ_D，弱施加)

2. 组装刚度矩阵  K = [[A,  B   ],
                      [B^T, 0]]  或  [[A,  B   ],
                                       [B^T, -J]]   (p ≤ GD 时)

3. 组装右端项    F = [F_natural, -F_body]
                  F_natural: ∫_{Γ_D} (τ·n)·u_D   (位移边界，弱形式)
                  F_body:    ∫ f · v              (体力)

4. 施加本质边界  牵引 Γ_N 上 σ·n = g_N → 置 1 置 0 法修改 K, F

5. 求解          直接法 (MUMPS / scipy SuperLU)

6. 解向量拆分    X[:gdof_sigma] = σ_h,  X[gdof_sigma:] = u_h
```

步骤 2–3 中，若启用角点松弛（`TM` 变换矩阵），A 和 B 分别在装配后做
`TM.T @ A @ TM` 和 `TM.T @ B` 变换。右端项 `F_natural` 在累加到全局
向量后同样做 `TM.T @ F_vec`。

步骤 5 求解后，刚度矩阵 `K` 以 `K.copy()` 缓存——SuperLU 会原地修改
`to_scipy()` 的共享内存视图，不复制会导致后续伴随求解复用错误的矩阵。

### 矩阵装配细节

**A 块**（HuZhangStressIntegrator）：单元循环，消费柔度张量的 lambda0、lambda1 两个不变量。支持密度插值时更新系数后重新装配。装配结果缓存于 _cached_stress_matrix。

二维必须与 IsotropicLinearElasticMaterial.hypothesis 保持一致。对于 plane_stress，lambda0 = (1 + nu) / E、lambda1 = nu / E；对于 plane_strain，lambda1 = nu * (1 + nu) / E。不得把平面应力算例沿用平面应变柔度，否则 Hu--Zhang 与位移法会产生系统性的柔顺度偏差。

**B 块**（`HuZhangMixIntegrator`）：位移空间与应力空间的混合双线性型，
积分 `div σ_h · v_h`。在构造器中一次性装配，缓存于 `_cached_mix_matrix`。

**J 块**（`JumpPenaltyIntegrator`）：面循环（内部面 + Dirichlet 边界面），
矩阵跳量（`method='matrix_jump'`），系数论文式物理量纲缩放
`α·h_F`（`α = μ/L₀²`，`penalty_scaling='physical_h'` 默认；旧
`γ/h_F` 型保留作回归对比）。仅在 `p ≤ GD` 时参与 K 装配。

## 实现特性

### 网格支持

**仅支持单纯形网格**：2D 三角形、3D 四面体。不支持四边形和六面体。
这是当前软件实现的支持范围，不是胡张元理论的永久限制。

### 次数与稳定化分支

| 条件 | 刚度矩阵 | 说明 |
|---|---|---|
| `p >= GD + 1` | `[[A, B], [B^T, 0]]` | inf-sup 稳定，不需惩罚 |
| `p <= GD` | `[[A, B], [B^T, -J]]` | 低阶跳量稳定化 |

跳量稳定化施加在内部面和 Dirichlet 边界面上，使用矩阵跳量
（`method='matrix_jump'`）。penalty 系数默认论文式物理量纲缩放
`α·h_F`（`α = μ/L₀²`，$L_0$ 为计算域特征尺度；`penalty_scaling='physical_h'`），
整体随 $h_F^2\to0$ 弱一致衰减，细层收敛恢复。早期 `γ/h_F` 型缩放
（γ 取弹性模量的 1%）乘面测度后净效果 O(γ) 常数、细层阶塌陷，已从默认路径
弃用，`penalty_scaling='gamma_hinv'` 保留作回归对比。完整数学过程与缩放律论证
见知识库概念页 `C:\workspace\dut-postdoc\concepts\huzhang\huzhang-mixed-fem.md`。

已验证收敛的 degree：2（跳量稳定化，σ 2 阶、$H(\mathrm{div})$ 1 阶降阶）、
3（无惩罚，σ 4 阶）、4（无惩罚，σ 5 阶）。

**k = 1 的适用边界**：k = 1（$P_1$ 应力 / $P_0$ 位移）加跳量稳定化后静力可收敛，
因此静力算例开放该阶次（`examples/huzhang_elasticity/concentrated_load_demo.py`
的 `SUPPORTED_ORDERS = 1, 2, 3, 4`）。但 $P_0$ 位移不完备包含刚体位移空间（RM），
在变密度演化中会使低密度区应变能评估失真、诱发非物理拓扑（博士论文 §5.6.2），
故拓扑优化侧 `experiments/huzhang_topopt_paper/cases.toml` 的 `comparison_orders`
下限取 2。完整论证（稳定性 / 静力收敛 / 拓扑可用三层的区分）见知识库概念页
`C:\workspace\dut-postdoc\concepts\huzhang\huzhang-mixed-fem.md` §3.2 与 §5 末。

### 角点松弛

胡张应力空间的对称张量约束在角点附近会过度限制应力场，导致收敛阶
退化。角点松弛通过在每个几何角点的两个 incident 三角形上引入额外的
DOF 变换来解除这一约束。

**拓扑要求（当前实现）**：每个几何角点必须恰好连接 2 个三角形，且两者
共享一条从角点出发的内部边。这就要求网格从规则四边形按**棋盘格交替
对角线**剖分得到：

![checkerboard 网格剖分](checkerboard-mesh.png)

左图为 `QuadrangleMesh.from_box` 的 2×2 规则四边形网格；右图为
`create_huzhang_checkerboard_mesh` 剖分结果，四边形的对角线方向按
`(i+j) % 2` 交替（蓝色 = 左对角线 `\`，红色 = 右对角线 `/`）。
四个几何角点（红点）各连接 2 个三角形且共享一条内部边，满足角点
松弛的拓扑条件。`nx` 和 `ny` 必须为正偶数。

松弛通过 DOF 变换矩阵 `TM` 实现：构造时计算变换矩阵，装配时施加到
基函数和载荷向量上。**3D 不支持角点松弛**——`HuZhangFESpace3d` 忽略
`use_relaxation` 参数，无松弛的 3D 求解链尚未端到端验证。

### 2D vs 3D

| 特性 | 2D | 3D |
|---|---|---|
| 空间类 | `HuZhangFESpace2d` | `HuZhangFESpace3d` |
| 角点松弛 | 支持 | **不支持** |
| `div_basis` | `variables='x'` 修复后正确 | 原生正确（一直带 `variables='x'`） |
| 端到端验证 | 完成（degree 2/3/4） | **未完成** |

工厂类 `HuZhangFESpace.__new__` 按 `mesh.top_dimension()` 自动分派。

### 边界条件

混合边界条件通过 `MixedBoundaryElasticityProblem` 协议定义：

- **位移边界 $\Gamma_D$**：弱施加（自然边界），通过
  $\int_{\Gamma_D} (\tau \cdot n) \cdot u_D \; ds$ 进入右端项。支持非齐次位移。
- **牵引边界 $\Gamma_N$**：强施加（本质边界），直接修改刚度矩阵对应行。
  使用 `face_to_cell` / `face_unit_normal` 适配 FEALPy 4.0 API。

纯 Dirichlet 问题是混合边界的退化形式（AllDisplacementBoundaryMixin）。

### 工程集中载荷的程序架构

博士论文第五章将理想集中力转换为局部边界上的等效均布牵引 t_bar = P / l。在 SOPTX 中，连续物理定义与有限元离散定义必须分层，不能将含边内跳跃的连续牵引函数直接同时交给两种方法。

    src/soptx/problems/elasticity/fixed_fixed.py
      FixedFixedBeamCenterLoad2d
      ├── 连续物理模型：domain, P, load_width=l, E, nu, plane_type
      ├── 原始局部均布牵引 t_bar=P/l（默认 traction_bc）
      ├── traction_patch / traction_level / traction_intensity：贴片几何的唯一出处
      └── traction=...：可选注入，用等价的连续牵引替换 traction_bc/neumann_bc

    src/soptx/fem/boundary_loads.py
      project_patch_traction_to_p1_trace(line, n_cells, level, patch, intensity)
      ├── 将 t_bar L2 投影至该边的连续 P1 迹空间，得 t_h（右端项按解析重叠积分）
      └── 返回 P1TraceLoad：可直接当作牵引函数调用，resultant() 用于核查合力

    experiments/huzhang_topopt_paper/cases.toml
      └── load_discretization = p1_trace_l2_projection

    experiments/huzhang_topopt_paper/fixed_fixed_beam.py
      └── build_problem(parameters, n_cells=nx)：投影 + 注入，得到唯一的分析问题

    examples/huzhang_elasticity/concentrated_load_demo.py
      └── build_problem(nx)：同一投影 + 注入，实体材料下核查载荷等效性

    LagrangeFEMAnalyzer
      └── Neumann 边界弱积分 t_h

    HuZhangMFEMAnalyzer
      └── 应力法向迹本质边界强施加 sigma_h*n=t_h

FixedFixedBeamCenterLoad2d 是核心物理模型，不保存网格、边界自由度或投影系数：它只
把贴片几何以三个属性的形式暴露出来，并接受一个替换牵引。投影本身是与具体物理问题
无关的通用能力，因此落在 `soptx.fem` 而不是实验目录——它只需要计算域、边界剖分数
和贴片区间，不需要网格对象。实验层剩下的只是「用哪个 n_cells 去投影」这一个决定。

选择连续 P1 迹空间是有意的：Hu--Zhang 空间在每条边上的法向迹是跨边连续的 k 次多项式，连续 P1 迹空间对任意 k >= 1 都是它的子空间。因此 Hu--Zhang 的边界自由度插值可精确表示 t_h，LFEM 也能对同一 t_h 做 Neumann 积分。直接对原始阶跃牵引做 Hu--Zhang 节点插值会在共享端点扩散载荷，使离散合力偏离 P，不能作为方法对照的载荷。

共同载荷必须满足以下验收条件：

- integral_GammaN t_h ds = P（`tests/unit/test_p1_trace_load_projection.py`）。
- integral_GammaN x t_h ds = x_mid P，即保持中点载荷的一阶矩（同上）。
- 结构合力守恒：Hu--Zhang 取 integral_GammaN sigma_h*n ds，LFEM 取支座反力 sum (K u)|_GammaD，两者均等于 P（`examples/huzhang_elasticity/concentrated_load_demo.py`，实体材料 rho=1）。这一条抓的是「载荷落在被强加自由度上被静默吞掉」——此时残差依然为 0，只有它能报警；它同时端到端覆盖了「LFEM 的边界积分与 Hu--Zhang 的迹插值拿到同一个载荷泛函」，因此不再单列。
- 对固定密度状态，LFEM 验证 fTu = uKu；Hu--Zhang 报告 sigmaAsigma、sigmaBu 和牵引对偶功 sigmaAsigma + sigmaBu（`experiments/huzhang_topopt_paper/run.py --mode state-compare`，rho=0.4）。

P1 投影在贴片外带有几何衰减的振荡尾（P1 质量矩阵 Green 函数，每单元约 0.27）。网格过粗时该尾部触及固支端，那一部分载荷会被 Dirichlet 自由度真实吞掉，量级约 P * 0.27^(nx/2)；demo 将其单列为 `吞掉` 一列，n_x >= 40 时已远在 1e-6 相对容差之下。

Hu--Zhang 的位移变量属于分片不连续 L2 空间，不能直接将其解释为单值边界位移并套用位移法的边界外力功。对于 k >= 3，无跳量稳定化时 sigmaBu 应接近零。对于 k=2，sigmaBu 可反映跳量稳定化对离散能量的贡献，不应与高阶情形混作严格等价基准。

### 求解器

状态方程是鞍点系统（对称不定）。理论上 MINRES 等迭代法可解对称不定
系统，但鞍点矩阵条件数随网格加密而增长，未经预条件处理的迭代法收敛
极慢或发散。有效的块预条件子（基于 Schur 补近似）实现复杂，当前
SOPTX 未提供。因此，当前实现仅提供直接法，构造期即拒绝
`solve_method` 为迭代法的配置：

| 选项 | 底层 | 要求 |
|---|---|---|
| `solve_method='mumps'` | PyMUMPS + 系统 MUMPS 库 | `pip install pymumps` |
| `solve_method='scipy'` | `scipy.sparse.linalg.spsolve` (SuperLU) | 无额外依赖 |

注意 SuperLU 会原地修改矩阵（列置换/缩放），缓存刚度矩阵时需用
`K.copy()`（`CSRTensor.copy()`）而非 `to_scipy()` 的共享内存视图。

## FEALPy 3.4 → 4.0 迁移要点

以下 6 项是 SOPTX 从 FEALPy 3.4.0（`fealpy_heliang`，已退役的另一条 fork 线）
迁移到 4.0.0 时适配的 API 差异。这些是 4.0.0 自身的 API 变化或行为差异，上游
`suanhai/develop` 和本地 fork 中都一样，修复均落在 SOPTX 侧。本地 fork 独有的
改进（张量积网格 5 缺陷修复）见
[`../known-issues/fealpy-tensor-product-mesh.md`](../known-issues/fealpy-tensor-product-mesh.md)。

| # | 要点 | 修复 | 影响范围 |
|---|---|---|---|
| 1 | `grad_shape_function` 默认返回参考坐标导数 | 2D/3D 统一调 `grad_shape_function(bc, p, variables='x')` | `div_basis` |
| 2 | `bmat` 在 blocks 全非 None 时走 hstack/vstack 丢块 | 改用 `scipy.sparse.bmat` → `CSRTensor.from_scipy` | 刚度矩阵装配 |
| 3 | `spsolve` 经 `to_scipy()` 共享内存，SuperLU 原地修改 | 缓存用 `K.copy()` | 伴随求解复用 |
| 4 | `cell_to_face_sign` → 2D 改名 `cell_to_edge_sign` | 按 `mesh.top_dimension()` 分派 | jump-penalty |
| 5 | `mesh.edgedata` 用户数据字典已移除 | 边界标记改用分析器 `_essential_bc/_natural_bc` 持有 | BC 装配 |
| 6 | `bc_to_point` 返回 `(NC, NQ, GD)` 带单元维 | 直接使用返回值，不再 `[0]` 取首个单元 | 诊断/后处理 |

API 差异的完整清单见 dut-postdoc 的 `fealpy4-api-notes.md`。

## 示例与测试

**可运行示例**：
* 制造解收敛验证：[`examples/huzhang_elasticity/manufactured_convergence_demo.py`](../../examples/huzhang_elasticity/manufactured_convergence_demo.py)
* 集中力工程基准：[`examples/huzhang_elasticity/concentrated_load_demo.py`](../../examples/huzhang_elasticity/concentrated_load_demo.py)

```bash
# 制造解收敛阶与残差验证
python examples/huzhang_elasticity/manufactured_convergence_demo.py                # degree=3, 带松弛
python examples/huzhang_elasticity/manufactured_convergence_demo.py --degree 2      # 低阶跳量稳定化
python examples/huzhang_elasticity/manufactured_convergence_demo.py --no-relaxation # 关闭松弛

# 集中力工程基准载荷路径与合力守恒验证
python examples/huzhang_elasticity/concentrated_load_demo.py --model fixed-fixed --nx 160 --ny 20 --degrees 2 3
```

**pytest**（6 个测试文件）：

| 测试 | 范围 |
|---|---|
| `tests/integration/test_huzhang_accepts_maintained_problems.py` | 端到端：分析器驱动维护中的 Problem，松弛开/关 |
| `tests/unit/test_huzhang_corner_relaxation.py` | 角点松弛的两单元拓扑、变换矩阵行为 |
| `tests/unit/test_huzhang_compliance_coefficients.py` | 平面应力/平面应变柔度系数分支 |
| `tests/unit/test_problem_protocol_conformance.py` | `MixedBoundaryElasticityProblem` 协议符合性 |
| `tests/unit/test_compatibility_api.py` | 旧 `soptx.functionspace` 与新 `soptx.fem.spaces` 兼容 |
| `tests/experiments/test_huzhang_paper_runner.py` | `run.py` 配置覆盖论文 case matrix |

**论文 evidence**：[`experiments/huzhang_topopt_paper/`](../../experiments/huzhang_topopt_paper/)
包含 7 个 case 的 evidence matrix（前向制造解、边界 ablation、灵敏度、
柔度固定、近不可压缩、悬臂梁应力约束、冻结设计），degree 1–4，统一
三角形网格。

## 已知限制与开放问题

1. **全 Dirichlet 制造解路径**：`sinusoidal` / `exp-sine` 全位移边界问题理论上可走 `AllDisplacementBoundaryMixin` 的混合形式边界接口，但该路径未经充分独立测试，当前算例聚焦于混合边界条件。
2. **低阶跳量稳定化在混合边界下的行为**：$k\le 2$ 时跳量惩罚项 $c(\boldsymbol{u}_h,\boldsymbol{v}_h)$ 加在内部面与位移边界（$\Gamma_D$）上，不施加于 $\Gamma_N$。这使得 $k=2$ 在混合边界下的 $H(\mathrm{div})$ 误差出现向 1 阶的降阶，属于物理与离散截断的预期现象；惩罚系数采用物理量纲缩放 $\alpha=\mu/L_0^2\cdot h_F$（$\alpha=\mu$ 于单位域）。
3. **网格拓扑约束**：角点松弛要求每个几何角点恰好连接两个三角形且共享内部边，因此当前实现固定使用 `triangle-checkerboard` 剖分；扩展到任意三角形网格需要重新推导 `_get_corner_data` 的 incident 单元判定逻辑。
4. **3D 扩展**：3D 无松弛求解链 `div_basis` 已确认正确（有限差分 $3.4\times 10^{-10}$），但 3D 混合边界制造解与端到端松弛集成仍留作后续扩展。
5. **空间阶次与拓扑优化适用性**：$k=1$ 时的位移空间为 $P_0$（分片常数），无法表达二维刚体旋转模态（线性场 $\boldsymbol{u}=[-\omega y, \omega x]^{\mathsf T}$）。在实体材料单次线弹性求解下由跳量惩罚约束尚能收敛，但在变密度拓扑优化迭代中会导致悬臂细杆等结构出现虚假刚度硬化；因此拓扑优化论文实验矩阵将阶次下限定为 $k\ge 2$（位移空间 $P_1$，具备刚体位移与旋转完备性）。
