# 胡张混合有限元实现

> SOPTX 的胡张元实现覆盖 2D/3D 单纯形网格、任意次 Bubble 丰富应力空间、角点松弛、低阶跳量稳定化，以及混合边界条件。

## 程序架构

SOPTX 的实现直接操作应力空间基底，不为位移元引入额外的 Bubble 或缩聚——这在拓扑优化中保持了应力场的物理可解释性。

### 文件布局

```
src/soptx/fem/
├── spaces/
│   ├── huzhang_fe_space.py          ← 工厂，按 mesh.top_dimension() 分派
│   ├── huzhang_fe_space_2d.py       ← 2D 应力空间（含角点松弛）
│   └── huzhang_fe_space_3d.py       ← 3D 应力空间（无松弛）
├── integrators/
│   ├── huzhang_stress_integrator.py ← A 块：∫ C⁻¹ σ : τ（柔度双线性型）
│   ├── huzhang_mix_integrator.py    ← B 块：∫ div σ · u（应力-位移耦合）
│   └── jump_penalty_integrator.py   ← J 块：低阶跳量稳定化
└── analyzers/
    └── huzhang_mfem_analyzer.py     ← 分析器：装配、边界条件、求解、后处理

src/soptx/mesh/
└── structured_triangle.py           ← 与角点松弛兼容的结构网格生成器
```

两个空间类都实现 DOF 枚举、`basis` 与 `div_basis`；`huzhang_fe_space_2d.py` 另外提供角点松弛变换矩阵 `TM` 的构造。结构网格生成器只构造 `TriangleMesh`、不依赖任何应力空间对象，因此独立放在 `soptx.mesh` 下；`soptx.fem` 与 `soptx.fem.spaces` 保留了向后兼容的导入别名。

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

应力空间输出对称张量基底（2D: `[xx, xy, yy]`，3D: `[xx, xy, xz, yy, yz, zz]`），通过 `symmetry_span_array` 编码对称指标。位移空间是标准的 `TensorFunctionSpace(LagrangeFESpace(p-1, DG), shape=(-1, GD))`。

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
                  F_body:    ∫ f · v              (体力，写入时取负号)

4. 施加本质边界  牵引 Γ_N 上 σ·n = g_N → 置 1 置 0 法修改 K, F

5. 求解          直接法 (MUMPS / scipy SuperLU)

6. 解向量拆分    X[:gdof_sigma] = σ_h,  X[gdof_sigma:] = u_h
```

步骤 2–3 中，若启用角点松弛（`TM` 变换矩阵），A 和 B 分别在装配后做 `TM.T @ A @ TM` 和 `TM.T @ B` 变换。右端项 `F_natural` 在累加到全局向量后同样做 `TM.T @ F_vec`。

### 矩阵装配细节

**A 块**（`HuZhangStressIntegrator`）：单元循环，消费柔度张量的 `lambda0`、`lambda1` 两个不变量。密度插值更新系数后需要重新装配，是拓扑优化迭代中唯一随设计变量变化的块。

> 二维柔度系数必须与 `IsotropicLinearElasticMaterial.hypothesis` 保持一致：`plane_stress` 取 `lambda0 = (1 + nu) / E`、`lambda1 = nu / E`；`plane_strain` 取 `lambda1 = nu * (1 + nu) / E`。不得把平面应力算例沿用平面应变柔度，否则胡张元与位移法会产生系统性的柔顺度偏差（回归测试 `tests/unit/test_huzhang_compliance_coefficients.py`）。

**B 块**（`HuZhangMixIntegrator`）：位移空间与应力空间的混合双线性型，积分 `div σ_h · v_h`。只依赖网格与空间，在构造器中一次性装配后不再重算。

**J 块**（`JumpPenaltyIntegrator`）：面循环（内部面 + Dirichlet 边界面），矩阵跳量（`method='matrix_jump'`），系数论文式物理量纲缩放 `α·h_F`（`α = μ/L₀²`，`penalty_scaling='physical_h'`）。仅在 `p ≤ GD` 时参与 K 装配。

### 边界条件与外载荷

线弹性问题的边界条件与外载荷在位移元与混合元中呈现严格的变分对偶性：位移法强施加位移、弱施加载荷，混合法强施加载荷、弱施加位移。混合边界条件通过 `MixedBoundaryElasticityProblem` 协议定义：

- **位移边界 $\Gamma_D$**：弱施加（自然边界），通过 $\int_{\Gamma_D} (\tau \cdot n) \cdot u_D \; ds$ 进入右端项。支持非齐次位移。
- **牵引边界 $\Gamma_N$**：强施加（本质边界），直接修改刚度矩阵对应行。使用 `face_to_cell` / `face_unit_normal` 适配 FEALPy 4.0 API。

纯 Dirichlet 问题是混合边界的退化形式（`AllDisplacementBoundaryMixin`）。

通用的载荷协议分层、PDE 接口契约、连续分布力与集中力（`boundary_loads.py` 解析重叠 $L^2$ 投影）的程序数据流详见载荷处理架构文档：

> **详细规范**：参见通用有限元文档 [`load-handling-implementation.md`](load-handling-implementation.md)。

胡张元特有的实现要点只有以下六条：

1. **角点松弛变换对边界项的作用**：若启用角点松弛（`space_sigma.use_relaxation == True`），弱施加的位移边界右端项 `F_natural`（$\int_{\Gamma_D} (\boldsymbol{\tau}\cdot\boldsymbol{n})\cdot\boldsymbol{u}_D$）在累加到全局向量后，必须经转置变换矩阵投影：`F_natural = TM.T @ F_natural`。
2. **拓扑优化中的非齐次牵引提升（Lifting）**：在前向求解与拓扑优化灵敏度分析中，混合法将总应力分解为齐次部分与设计无关的非齐次提升场：$\boldsymbol{\sigma} = \boldsymbol{\sigma}_0 + \boldsymbol{\sigma}_g$。目标函数与灵敏度求导始终基于总应力 $\boldsymbol{s} = \boldsymbol{s}_0 + \boldsymbol{s}_g$ 进行，防止代数消元法遗漏与材料密度相关的提升交叉项。
3. **牵引边界的强施加只接受连续 $P_1$ 迹载荷**：牵引数据必须先经 `project_patch_traction_to_p1_trace` 投影到连续分片一次迹空间，得到 $t_h$ 后再写入边界自由度；不得把含边内跳跃的原始阶跃牵引直接做节点插值。判据是迹空间的包含关系——胡张空间的法向迹跨边连续且逐边为 $k$ 次多项式，连续 $P_1$ 对任意 $k\ge1$ 都是它的子空间，故对 $t_h$ 的强插值是恒等映射，与位移法对同一 $t_h$ 的 Neumann 弱积分构成同一个载荷泛函；阶跃牵引不在迹空间内，插值会改写跨越载荷区端点那条边上的数据，使离散合力偏离 $P$，不能作为方法对照的载荷。完整论证见 `dut-postdoc:concepts/external-loads.md` §3.4–§3.5，在胡张形式下的判据陈述见 `dut-postdoc:concepts/huzhang/huzhang-mixed-fem.md` §2.5.2，本条只记程序契约。
4. **位移变量不能用于边界外力功核算**：胡张的位移变量属于分片不连续 $L^2$ 空间，不能解释为单值边界位移去套位移法的 $f^\mathsf{T} u = u^\mathsf{T} K u$；能量核对改看 $\sigma^\mathsf{T} A \sigma$ 与 $\sigma^\mathsf{T} B u$，其中 $k \ge 3$ 无跳量稳定化时 $\sigma^\mathsf{T} B u$ 应接近零，$k = 2$ 时该量含跳量稳定化对离散能量的贡献，不构成与位移法的严格等价基准。
5. **牵引强施加用的必须是外法向**：边界应力自由度定义在边标架 $(\boldsymbol{n}_f, \boldsymbol{t}_f)$ 上，而 `mesh.face_unit_normal()` 的朝向由全局边定向决定，边界上通常有相当一部分指向单元内部（`triangle-single-diagonal-symmetric` 的 $120\times 40$ 网格上，顶边 120 条全部朝内，全体边界边 160/320 朝内）。当牵引以二分量向量 $\boldsymbol{t}=\boldsymbol{\sigma}\cdot\boldsymbol{n}_{\mathrm{out}}$ 给出时（`gd_vals.shape[-1] == 2` 分支），$\boldsymbol{n}_f$ 在 $\boldsymbol{\sigma}:\boldsymbol{n}_f\otimes\boldsymbol{n}_f$ 与 $\boldsymbol{\sigma}:\boldsymbol{n}_f\otimes\boldsymbol{t}_f$ 中各只出现一次，两个分量都必须乘上 `boundary_outward_sign(mesh, index)` 给出的朝外符号 $s=\operatorname{sign}\big((\boldsymbol{c}_F-\boldsymbol{c}_K)\cdot\boldsymbol{n}_f\big)$。漏乘是 $O(1)$ 的数据错误，不随网格加密或阶次提高消失。若牵引以 Voigt 应力张量三分量给出（`shape[-1] == 3` 分支），$\boldsymbol{n}_f$ 出现两次、翻转自洽，不需要该符号。同一符号也用于弱施加位移边界的 $\int_{\Gamma_D}(\boldsymbol{\tau}\cdot\boldsymbol{n})\cdot\boldsymbol{u}_D$ 项，$\boldsymbol{u}_D=\boldsymbol{0}$ 时看不出差别，非齐次位移边界会整体反号。

   后果与载荷的空间分布有关，须逐算例判读，不能一概认为结果作废：若某条牵引边界上的 $s$ 取值一致，效果是该边载荷整体反号，柔顺度 $C=\boldsymbol{f}^{\mathsf T}\boldsymbol{K}^{-1}\boldsymbol{f}$ 与灵敏度 $\partial C/\partial\rho_e=-\boldsymbol{u}_e^{\mathsf T}(\partial\boldsymbol{K}_e/\partial\rho_e)\boldsymbol{u}_e$ 都是 $\boldsymbol{u}$ 的二次型，取值不变，优化轨迹与最终构型也不变，只有 $\boldsymbol{\sigma}$、$\boldsymbol{u}$ 整体差一个负号；若 $s$ 在同一条牵引边界上参差，则是真实的载荷分布畸变，结果不可用。定向判据的理论陈述见 `dut-postdoc:concepts/huzhang/huzhang-mixed-fem.md` §2.5.1。

6. **顶点自由度被多条边界边重复写入时取平均**：牵引数据按边成批写入 `uh[e2d]`，几何顶点同时属于相邻两条边界边，因而被写两次。直接赋值是「后写者胜」，胜者由全局边编号决定、与几何无关；牵引数据在该顶点两侧不连续时（贴片载荷的两个端点即是），端点值被整个判给其中一侧，离散载荷失去镜像对称并产生净力矩，误差是 $O(1)$ 的一个节点值，不随 $k$ 收敛。`_average_boundary_writes` 改为对同一自由度的多次写入取平均：数据连续处各次写入本就相同，平均是恒等操作；数据间断处给出唯一的、与边编号无关的对称值，且贴片两端各取半值恰好保住载荷合力。

   角点是例外：几何角点两侧的边标架不同，各次写入表达在不同标架里，取平均与后写者胜同样没有良定义的极限。该情形的正确处理是 `use_relaxation=True` 的角点自由度分裂，分裂后每条边各有独立自由度、写入次数为 1，本机制自动退化为直接赋值。共享顶点三种情形的良定义性讨论见 `dut-postdoc:concepts/huzhang/huzhang-mixed-fem.md` §2.5.3。

## 实现特性

### 网格支持

**仅支持单纯形网格**：2D 三角形、3D 四面体。不支持四边形和六面体。

这不是实现上的取舍，而是胡张元构造本身的前提。应力空间的每个基函数都写成"标量 Lagrange 基 × 对称张量标架"的乘积（`basis` 中的 `scalar_part * tensor_part`），两个因子各自绑死在单纯形上：

- **标量因子按重心多重指标枚举**：`bm.multi_index_matrix(p, TD)` 给出 $TD+1$ 个分量的多重指标 $\alpha$，自由度归属直接由"$\alpha$ 的非零分量集合 = 该自由度所属的子单纯形"判定（`dof_classfication` 中的 `flag0` / `flag1`）。子实体与重心指标子集的一一对应是单纯形独有的结构，四边形上没有重心坐标，这套枚举无从建立；`number_of_multiindex` 也只实现了 $p+1$、$(p+1)(p+2)/2$、$(p+1)(p+2)(p+3)/6$ 三档，即 1/2/3 维单纯形上 $P_p$ 的维数。
- **张量标架按子单纯形维数分级**：$i$ 维子单纯形上，$TD(TD+1)/2$ 个对称张量标架分量中恰有 $i(i+1)/2$ 个跨单元断开、其余保持连续（`dof_classfication` 中的 `N_c = NS - i*(i+1)//2`）。以 2D 为例：顶点上 3 个分量全连续，边上 2 连续 1 断开，单元内部 3 个全断开。正是这个分级同时给出 $H(\mathrm{div})$ 协调性与逐点对称性，而它的计数只在单纯形的面格结构上成立。

标架本身由 `dof_frame` 给出：边上取该边的单位法向与单位切向，顶点与单元内部取笛卡尔基，再经 `symmetry_span_array` 张成对称张量。

矩形与长方体网格上的对称张量 $H(\mathrm{div})$ 协调元是另一族构造，不是把胡张元的基底搬到四边形上就能得到，需要另写空间类，不在当前范围内。

### 自由度分层与装配映射

胡张应力空间的自由度按几何实体（顶点、边、单元）分层编号，`HuZhangFEDof2d.number_of_global_dofs` 与 `cell_to_dof` 按此装配：

- **顶点自由度**：每个几何顶点 $NS = d(d+1)/2$ 个点值自由度（2D 即 $\sigma_{xx},\sigma_{xy},\sigma_{yy}$），非角点顶点全局共享，松弛角点扩展为 4 个；
- **边内部自由度**：每条边 $2(k-1)$ 个，对应局部标架 $(\boldsymbol n_f,\boldsymbol t_f)$ 下的法向迹分量 $\sigma_{nn},\sigma_{nt}$，相邻单元共享（这正是 $H(\mathrm{div})$ 协调性的载体）；
- **单元内部自由度**：`number_of_internal_local_dofs('cell')` 返回 `ldof - NS*3 - 2*(p-1)*3`，2D 化简为 $3k(k-1)/2$，它同时容纳每边 $k-1$ 个**单元私有的纯切向分量** $\sigma_{tt}$（合计 $3(k-1)$）与 $3(k-1)(k-2)/2$ 个张量泡函数。纯切向分量不参与跨边连续，故代码把它记在 cell 账上而非 edge 账上。

2D 三角形网格上的逐阶计数（$NS=3$，单元局部总数 $\mathrm{ldof}_\sigma = 3(k+1)(k+2)/2$）：

| 阶次 $k$ | 每顶点 | 每边内部 $2(k-1)$ | 每单元内部 $3k(k-1)/2$ | 单元局部总 $\mathrm{ldof}_\sigma$ | 松弛角点附加 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 3 | 0 | 0 | **9** | $+1$ |
| **2** | 3 | 2 | 3 | **18** | $+1$ |
| **3** | 3 | 4 | 9 | **30** | $+1$ |
| **4** | 3 | 6 | 18 | **45** | $+1$ |

单元局部总数按 $3\times$ 顶点 $+\,3\times$ 边内部 $+$ 单元内部 核算，例如 $k=3$：$9+12+9=30$。全局规模（$N_v$ 顶点、$N_e$ 边、$N_c$ 单元、$N_{\mathrm{corner}}$ 松弛角点）：

$$
\mathrm{gdof}_\sigma = 3 N_v + 2(k-1) N_e + N_c\,\frac{3k(k-1)}{2} + N_{\mathrm{corner}},
$$

与 `number_of_global_dofs` 的 `NC*cldof + NE*eldof + NN*nldof + self.NCP` 逐项对应。

**装配流程**：先按上述三层顺序分配全局编号（顶点段 → 边段 → 单元段），再对每个合格角点追加 1 个自由度。`node_to_internal_dof` 在 `use_relaxation` 为真时把追加编号拼进 `corner2dof`，使该角点持有 $(d_0,d_1,d_2,d_3)$ 四个索引；`cell_to_dof` 按角点两条边界边的归属，把前两个（法向型，两单元共享）与后两个（切向型，分侧私有）分别写入 $K^+$、$K^-$ 的局部映射。基底层面的解耦由 `_transform_matrix` 生成的 $4\times4$ 块变换 `TM` 完成——它把这 4 个原始自由度旋到两侧边标架对齐的方向上，装配时施加到基函数与载荷向量。

角点松弛的网格拓扑前提与两种可用生成器见下文「角点松弛」；理论依据见 `dut-postdoc:concepts/huzhang/huzhang-mixed-fem.md` §3.4 与 §2.5.3。

### 次数与稳定化分支
分析器参数 `stabilization_coefficient` 默认取 `"fixed"`, 使用基材剪切模量 $\mu_0$ 定标, 不随密度更新. 若需复现密度相关对照, 在构造 `HuZhangMFEMAnalyzer` 时显式传入 `stabilization_coefficient="density_dependent"`, 惩罚系数再乘面两侧相对剪切模量 $\mu(\rho)/\mu_0$ 的调和平均. 该参数与跳量形式 `stabilization`、网格缩放律 `stabilization_scaling` 独立; 原生高阶分支不受影响.

| 条件 | 刚度矩阵 | 说明 |
|---|---|---|
| `p >= GD + 1` | `[[A, B], [B^T, 0]]` | inf-sup 稳定，不需惩罚 |
| `p <= GD` | `[[A, B], [B^T, -J]]` | 低阶跳量稳定化 |

跳量稳定化施加在内部面和 Dirichlet 边界面上，使用矩阵跳量（`method='matrix_jump'`）。penalty 系数取论文式物理量纲缩放 `α·h_F`（`α = μ/L₀²`，$L_0$ 为计算域特征尺度；`penalty_scaling='physical_h'`），整体随 $h_F^2\to0$ 弱一致衰减，细层收敛恢复。

已验证收敛的 degree：2（跳量稳定化，σ 2 阶、$H(\mathrm{div})$ 1 阶降阶）、3（无惩罚，σ 4 阶）、4（无惩罚，σ 5 阶）。制造解收敛阶由两个 case 覆盖：`manufactured-native`（`comparison_orders = [3, 4]`，原生高阶格式）与 `manufactured-stabilized`（`comparison_orders = [1, 2]`，低阶跳量稳定化格式）。

**k = 1 的适用边界**（本文档关于 $k=1$ 的唯一详述处，已知限制 5 只作交叉引用）：$k = 1$（$P_1$ 应力 / $P_0$ 位移）加跳量稳定化后静力可收敛，因此静力算例开放该阶次——[`examples/huzhang_elasticity/concentrated_load_demo.py`](../../examples/huzhang_elasticity/concentrated_load_demo.py) 的 `SUPPORTED_DEGREES = (1, 2, 3, 4)` 是 `--degrees` 的取值白名单，制造解 case `manufactured-stabilized` 也把 $k = 1$ 列入 `comparison_orders`。但 $P_0$ 位移不完备包含刚体位移空间（RM），在变密度演化中会使低密度区应变能评估失真、诱发非物理拓扑（博士论文 §5.6.2），故**拓扑优化族 case** 的 `comparison_orders` 下限取 2；$k = 1$ 只以 `supplementary_orders = [1]` 单列，供补充失效专题 `supp-k1` 取数，既不进缺省也不进 `--full`。

### 角点松弛

胡张应力空间的对称张量约束在角点附近会过度限制应力场，导致收敛阶退化。角点松弛通过在每个几何角点的两个 incident 三角形上引入额外的 DOF 变换来解除这一约束。

**拓扑要求（当前实现）**：每个几何角点必须恰好连接 2 个三角形，且两者共享一条从角点出发的内部边。矩形域上的规则四边形网格按对角线剖分即可满足该条件，`soptx.mesh` 提供两个这样的生成器，棋盘格只是其中一种剖分方式：

| 生成器 | 对角线规则 | 尺寸约束 | 内部结点扇 | 用途 |
|---|---|---|---|---|
| `create_huzhang_checkerboard_mesh` | `(i+j)` 偶数取 `/`，奇数取 `\` | `nx`、`ny` 为正偶数 | 4 / 8 三角形交替 | 固支梁、悬臂梁与制造解算例的注册值 |
| `create_huzhang_symmetric_single_diagonal_mesh` | 左半 `i < nx/2` 取 `/`，右半取 `\`，顶角落 `(0,ny-1)`、`(nx-1,ny-1)` 分别翻转为 `\`、`/` | `nx` 为偶数 ≥ 2，`ny` ≥ 2 | 一律 6 三角形（中缝结点左 3 右 3） | 两条轴承 case 的注册值；低阶位移元体积闭锁对照（与左右对称问题同对称性） |

![两种与角点松弛兼容的结构网格](assets/corner-relaxation-meshes.png)

左为 `create_huzhang_checkerboard_mesh`、右为 `create_huzhang_symmetric_single_diagonal_mesh` 的剖分结果，均取 `nx = 6`、`ny = 4`。两者的四个几何角点（绿点）都各连接 2 个三角形并共享一条内部边，满足上述拓扑条件；右图的红色虚线为镜像中缝，两个浅蓝单元是为满足角点条件而相对于单向对角规则翻转的四边形。插图由 `tools/plot_huzhang_meshes.py` 调用上述生成器绘制。

松弛通过 DOF 变换矩阵 `TM` 实现：构造时计算变换矩阵，装配时施加到基函数和载荷向量上。**3D 不支持角点松弛**——`HuZhangFESpace3d` 忽略 `use_relaxation` 参数，无松弛的 3D 求解链尚未端到端验证。

### 求解器

状态方程是鞍点系统（对称不定）。理论上 MINRES 等迭代法可解对称不定系统，但鞍点矩阵条件数随网格加密而增长，未经预条件处理的迭代法收敛极慢或发散。有效的块预条件子（基于 Schur 补近似）实现复杂，当前 SOPTX 未提供。因此，当前实现仅提供直接法，构造期即拒绝 `solve_method` 为迭代法的配置：

| 选项 | 底层 | 要求 |
|---|---|---|
| `solve_method='mumps'` | PyMUMPS + 系统 MUMPS 库 | `pip install pymumps` |
| `solve_method='scipy'` | `scipy.sparse.linalg.spsolve` (SuperLU) | 无额外依赖 |

## FEALPy 3.4 → 4.0 迁移要点

以下 6 项是 SOPTX 从 FEALPy 3.4.0（`fealpy_heliang`，已退役的另一条 fork 线）迁移到 4.0.0 时适配的 API 差异。这些是 4.0.0 自身的 API 变化或行为差异，上游 `suanhai/develop` 和本地 fork 中都一样，修复均落在 SOPTX 侧。本地 fork 独有的改进（张量积网格 5 缺陷修复）见 [`../known-issues/fealpy-patches.md`](../known-issues/fealpy-patches.md) 第一节。

| # | 要点 | 修复 | 影响范围 |
|---|---|---|---|
| 1 | `grad_shape_function` 默认返回参考坐标导数 | 2D/3D 统一调 `grad_shape_function(bc, p, variables='x')` | `div_basis` |
| 2 | `bmat` 在 blocks 全非 None 时走 hstack/vstack 丢块 | 改用 `scipy.sparse.bmat` → `CSRTensor.from_scipy` | 刚度矩阵装配 |
| 3 | `spsolve` 经 `to_scipy()` 共享内存，SuperLU 原地修改 | 当时在调用方缓存 `K.copy()`；现已下沉为 `DirectSolver` 内部的保护性复制 | 伴随求解复用 |
| 4 | `cell_to_face_sign` → 2D 改名 `cell_to_edge_sign` | 按 `mesh.top_dimension()` 分派 | jump-penalty |
| 5 | `mesh.edgedata` 用户数据字典已移除 | 边界标记改用分析器 `_essential_bc/_natural_bc` 持有 | BC 装配 |
| 6 | `bc_to_point` 返回 `(NC, NQ, GD)` 带单元维 | 直接使用返回值，不再 `[0]` 取首个单元 | 诊断/后处理 |

## 已知限制与开放问题

1. **全 Dirichlet 制造解路径**：`sinusoidal` / `exp-sine` 全位移边界问题理论上可走 `AllDisplacementBoundaryMixin` 的混合形式边界接口，但该路径未经充分独立测试，当前算例聚焦于混合边界条件。
2. **低阶跳量稳定化在混合边界下的行为**：$k\le 2$ 时跳量惩罚项 $c(\boldsymbol{u}_h,\boldsymbol{v}_h)$ 加在内部面与位移边界（$\Gamma_D$）上，不施加于 $\Gamma_N$。这使得 $k=2$ 在混合边界下的 $H(\mathrm{div})$ 误差出现向 1 阶的降阶，属于物理与离散截断的预期现象；惩罚系数采用物理量纲缩放 $\alpha=\mu/L_0^2\cdot h_F$（$\alpha=\mu$ 于单位域）。
3. **网格拓扑约束**：角点松弛要求每个几何角点恰好连接两个三角形且共享内部边（`_get_corner_data` 对不满足的角点直接报错）。当前 `triangle-checkerboard`、`triangle-single-diagonal-symmetric` 两种结构剖分满足该要求；`TriangleMesh.from_box` 的纯单向对角剖分在右下、左上角点只有 1 个三角形，不能直接使用。推广到一般顶点扇需要把 `TM` 改为每角点 $3m\times(m+2)$ 的长方块并拆分组装/求解两套自由度计数，$m=1$ 角点还需单独处理两条牵引边写入同一节点自由度的冲突（$m\ge 2$ 的常规角点由上文要点 6 的角点分裂覆盖，$m=1$ 无从分裂，该冲突仍开放）。
4. **3D 扩展**：3D 无松弛求解链 `div_basis` 已确认正确（有限差分 $3.4\times 10^{-10}$），但 3D 混合边界制造解与端到端松弛集成仍留作后续扩展。
5. **空间阶次与拓扑优化适用性**：$k=1$ 的位移空间为 $P_0$（分片常数），无法表达二维刚体旋转模态 $\boldsymbol{u}=[-\omega y, \omega x]^{\mathsf T}$，因而静力可用但拓扑优化不可用。详见[次数与稳定化分支](#次数与稳定化分支)一节的「k = 1 的适用边界」。
