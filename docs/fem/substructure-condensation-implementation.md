# 子结构静力缩聚实现

> 对应 Huang 2023 PIML 增强子结构方法的精确基线。SOPTX 的子结构缩聚覆盖
> 2D/3D 规则矩形/六面体子域、SIMP 材料插值、Schur 补消元、全局接口系统
> Scatter-Add 装配、全场位移恢复，以及与 PIML 预测器的统一接口。

## 数学形式

### 子结构分块与 Schur 补消元

对第 $j$ 个子结构 $\Omega^j$，在无内部载荷（$\mathbf{f}_i^j = \mathbf{0}$）假设下，
局部有限元方程写为 $2 \times 2$ 分块形式：

$$
\begin{bmatrix}
\mathbf{K}_{ii}^j & \mathbf{K}_{ib}^j \\
\mathbf{K}_{bi}^j & \mathbf{K}_{bb}^j
\end{bmatrix}
\begin{bmatrix}
\mathbf{u}_i^j \\ \mathbf{u}_b^j
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{0} \\ \mathbf{f}_b^j
\end{bmatrix}
$$

其中下标 $i$ 为内部自由度，$b$ 为接口（边界）自由度。由第一行消去 $\mathbf{u}_i^j$：

$$
\mathbf{u}_i^j = - (\mathbf{K}_{ii}^j)^{-1} \mathbf{K}_{ib}^j \mathbf{u}_b^j
= \mathbf{N}^j \mathbf{u}_b^j
$$

代入第二行得 Schur 补缩聚刚度矩阵：

$$
\mathbf{K}_s^j = \mathbf{K}_{bb}^j - \mathbf{K}_{bi}^j (\mathbf{K}_{ii}^j)^{-1} \mathbf{K}_{ib}^j
= (\mathbf{N}^j)^{\mathsf{T}} \mathbf{K}^j \mathbf{N}^j
$$

**多尺度形函数矩阵** $\mathbf{N}^j \in \mathbb{R}^{n_i \times n_b}$ 和
**缩聚刚度矩阵** $\mathbf{K}_s^j \in \mathbb{R}^{n_b \times n_b}$ 是
缩聚的两组核心代数产物，也是 PIML 的学习目标。

### 全局接口系统

各子结构的 $\mathbf{K}_s^j$ 经布尔映射矩阵 $\mathbf{L}_j$ 做 Scatter-Add 装配：

$$
\mathbf{K}_{\text{global}} = \sum_{j=1}^{M} \mathbf{L}_j^{\mathsf{T}} \mathbf{K}_s^j \mathbf{L}_j,
\qquad
\mathbf{K}_{\text{global}} \mathbf{U}_b = \mathbf{F}_b
$$

宏观边界条件和外载荷只在全局接口系统阶段进入。求解 $\mathbf{U}_b$ 后，
各子结构内部位移通过 $\mathbf{u}_i^j = \mathbf{N}^j \mathbf{u}_b^j$ 恢复。

### 路线 A 与路线 B

Huang 2023 §3.4 定义了两条 PIML 预测路线，对应两种学习对象：

- **路线 A：预测形函数。** $\boldsymbol{\rho}^j \mapsto \widehat{\mathbf{N}}^j$，
  再构造 $\widehat{\mathbf{K}}_s^j = (\widehat{\mathbf{N}}^j)^{\mathsf{T}} \mathbf{K}^j \widehat{\mathbf{N}}^j$。
  保持形函数、内部位移恢复与缩聚刚度之间的显式构造关系。
两条路线共享相同的局部输入 $\boldsymbol{\rho}^j$（逐单元 SIMP 密度）、
精确标签 $(\mathbf{N}_{\mathrm{exact}}^j, \mathbf{K}_{s,\mathrm{exact}}^j)$、
全局接入方式和下游评价指标。SOPTX 同时支持两条路线的真值计算与 PIML 代理评测。

精确缩聚的完整推导、刚体模态、能量一致性和细尺度恢复见
`dut-postdoc:concepts/substructural-condensation.md`。
PIML 局部—全局契约见
`dut-postdoc:concepts/piml/piml-substructural.md`。

## 程序架构

### 文件布局

```
src/soptx/fem/substructure/              ← 核心库 (成熟)
├── __init__.py                           ← 导出 SubstructureMesh, FEAStaticCondensation, GlobalAssembler
├── mesh.py                               ← SubstructureMesh: 2D/3D 子结构网格管理
├── condensation.py                       ← StaticCondensationBase, FEAStaticCondensation
├── piml_surrogate.py                     ← 路线 A: ShapeFunctionSurrogateNet, ShapeFunctionCondensation
│                                            路线 B: PIMLSurrogateNet, PIMLStaticCondensation
│                                            共用: SurrogateContractError
└── assembler.py                          ← GlobalAssembler, InterfaceSystem

examples/
├── substructure_elasticity/              ← 精确缩聚基线 (成熟)
│   ├── convergence_rate.py              ← 连续制造解 2D/3D 多层网格 L2 收敛阶验证 (理论 2.00 阶)
│   ├── compare_lagrange.py              ← 端到端缩聚 vs Lagrange 全装配交叉验证 (1e-12 双精度等价)
│   ├── results_analysis.md              ← 数学—代码映射契约与验收阈值
│   └── README.md
│
└── piml_substructure_elasticity/         ← PIML 代理体系 (路线 A + 路线 B)
    ├── convergence_rate.py              ← 制造解 2D/3D 渐近一致性收敛阶 (PIML 保持 2.00 阶)
    ├── verify_stiffness_route.py                  ← 宏观整梁代理 vs 精确缩聚，两层误差 + 误差归因诊断
    ├── verify_shape_function_route.py    ← 路线 A 消融: 闭式恒等式, 二阶扫描, 解层对比, 门禁标定
    ├── plot_local_recovery.py           ← 云图版式原型; ⚠️ (c)(d) 为合成数据, 不可引用
    ├── compare_piml_pinn.py              ← PIML 与 PINN 的同问题比较
    ├── results_analysis.md                ← PIML 契约、已知问题与后续工作
    └── README.md
```

> 核心模块已迁入 `src/soptx/fem/substructure/`，所有 example 通过
> `from soptx.fem.substructure import ...` 导入，不再需要 `sys.path` 操作。

### 核心类关系

```
SubstructureMesh                          (soptx.fem.substructure.mesh)
  ├── 持有 QuadrangleMesh / HexahedronMesh   (2D/3D 自动)
  ├── 持有 TensorFunctionSpace + LinearElasticIntegrator
  ├── 节点分类: internal_nodes / boundary_nodes → i_dofs / b_dofs
  ├── rigid_basis / deformation_basis        (接口自由度上的刚体模态基与其正交补)
  └── assemble_local_stiffness(density_field) → K_local

FEAStaticCondensation                    (soptx.fem.substructure.condensation, bm 后端)
  ├── condense(K_local) → (K_s, N)         (bm.linalg.solve, 不显式求逆)
  └── recover(u_b) → u_i                   (u_i = N @ u_b)

GlobalAssembler                           (soptx.fem.substructure.assembler)
  ├── build_interface_dofs(...)            (全局接口 DOF 集合)
  ├── assemble_interface_system(...)       (K_s Scatter-Add → InterfaceSystem)
  ├── project_global_vector/dofs(...)      (全局载荷/约束投影到接口)
  └── recover_full_displacement(...)       (接口位移 → 全场位移)

StaticCondensationBase                   (soptx.fem.substructure.condensation, 抽象基类)
  └── condense(K_local, rho_local=None) → (K_s, N)   (统一接口)
      ├── FEAStaticCondensation             → 精确 Schur 补消元
      └── PIMLStaticCondensation           → 网络推理 + 结构检查 + 失败回退
```

### 求解流程（8 个完整步骤与代码映射）

> **理论事实源**：数学推导、刚体模态解析证明与多尺度形函数理论详见 `dut-postdoc:concepts/substructural-condensation.md`。本节定义工程落地中的 8 步代数流转与张量形状契约。

子结构有限元静力缩聚的全流程由以下 8 个步骤构成：

```
                       子结构静力缩聚全流程 (8 个步骤)
  
 [阶段一: 局部刚度提取]  步骤 1: 缓存单位刚度模板 KE_unit (4, 8, 8)
                             │
                             ▼
                         步骤 2: SIMP 批量缩放单元刚度 KE (4, 4, 8, 8)
                             │
                             ▼
                         步骤 3: 向量化散加成子结构刚度 K_local (4, 18, 18)
                             │
                             ▼
                         步骤 4: 内部/接口分块切片 K_ii (4, 2, 2), K_ib, K_bb
                             │
                             ▼
 [阶段二: 局部静力缩聚]  步骤 5: Schur 补消元，产出 K_s (4, 16, 16) 与恢复矩阵 N (4, 2, 16)
                             │
                             ▼
 [阶段三: 宏观接口求解]  步骤 6: FEALPy COOTensor 散加装配为全局接口 CSRTensor K_B
                             │
                             ▼
                         步骤 7: 接口边界投影 + fealpy.solver.spsolve 求解接口位移 u_B
                             │
                             ▼
 [阶段四: 细尺度位移恢复]步骤 8: 矩阵乘法回代 u_i = N @ u_b，拼合输出全场位移 U
```

#### 步骤 1：构造参考原型并缓存单位单元刚度模板 $K_E^0$
* **数学与物理**：同构子结构共享相同的网格与尺寸，每个细单元在单位密度（$\rho=1$）下的基准刚度 $K_E^0$ 是一致的，高斯数值积分**只需计算一次**。
* **对应代码**：[`SubstructurePrototype.__init__()`](../../src/soptx/fem/substructure/mesh.py)
* **张量形状**：`KE_unit` $\to$ `(NC, n_edof, n_edof)`（以 $2\times 2$ 细网格为例为 `(4, 8, 8)`）。

#### 步骤 2：SIMP 材料插值与批量单元刚度缩放
* **数学与物理**：根据 SIMP 幂律公式 $K_E(\rho_{b,e}) = \rho_{b,e}^p K_E^0$，通过张量乘法一次性算出全部子结构的所有细单元实际刚度。
* **对应代码**：[`SubstructurePrototype._assemble_chunk()`](../../src/soptx/fem/substructure/mesh.py)
* **张量形状**：`KE` $\to$ `(b, NC, n_edof, n_edof)`（以 4 子结构为例为 `(4, 4, 8, 8)`）。

#### 步骤 3：向量化散加（Scatter-Add）生成子结构总刚度矩阵 $K_{\text{local}}$
* **数学与物理**：利用局部单元到自由度映射 `cell2dof`，把各细单元刚度累加到子结构自身的全部自由度上：$K_{\text{local}}^j = \sum_e L_e^{\mathsf{T}} K_E^{j,e} L_e$。
* **对应代码**：[`SubstructurePrototype._assemble_chunk()`](../../src/soptx/fem/substructure/mesh.py) 中的 `bm.bincount` 散加
* **张量形状**：`K_local` $\to$ `(b, n_dof, n_dof)`（以 4 子结构为例为 `(4, 18, 18)`）。

#### 步骤 4：内部与接口自由度分块切片
* **数学与物理**：根据坐标边界判别生成的 `i_dofs`（内部）和 `b_dofs`（接口）索引，将局部刚度矩阵切片为三个子块：
  $$\mathbf{K}_{\text{local}} = \begin{bmatrix} \mathbf{K}_{ii} & \mathbf{K}_{ib} \\ \mathbf{K}_{bi} & \mathbf{K}_{bb} \end{bmatrix}$$
* **对应代码**：[`FEAStaticCondensation.condense()`](../../src/soptx/fem/substructure/condensation.py)
* **张量形状**：$K_{ii}$ `(b, n_i, n_i)` $\to$ `(4, 2, 2)`；$K_{ib}$ `(b, n_i, n_b)` $\to$ `(4, 2, 16)`；$K_{bb}$ `(b, n_b, n_b)` $\to$ `(4, 16, 16)`。

#### 步骤 5：批量求解 Schur 补缩聚刚度 $K_s$ 与位移恢复矩阵 $N$
* **数学与物理**：利用内部平衡方程消去内部自由度：
  $$N = - K_{ii}^{-1} K_{ib}, \quad K_s = K_{bb} - K_{ib}^{\mathsf{T}} K_{ii}^{-1} K_{ib} = K_{bb} + K_{ib}^{\mathsf{T}} N$$
* **对应代码**：[`FEAStaticCondensation.condense()`](../../src/soptx/fem/substructure/condensation.py) 中的 `bm.linalg.solve`
* **张量形状**：$K_s$ `(b, n_b, n_b)` $\to$ `(4, 16, 16)`；$N$ `(b, n_i, n_b)` $\to$ `(4, 2, 16)`。

#### 步骤 6：全局接口系统装配（生成 FEALPy 原生 `CSRTensor`）
* **数学与物理**：将各子结构的缩聚刚度 $K_s^j$ 按接口拓扑映射矩阵 $L_j$ 散加为整个大结构的全局接口刚度矩阵：
  $$K_{\mathcal{B}} = \sum_{j=1}^M L_j^{\mathsf{T}} K_s^j L_j$$
* **对应代码**：[`GlobalAssembler.assemble_interface_system()`](../../src/soptx/fem/substructure/assembler.py)
* **张量形状**：`system.stiffness` $\to$ `(n_interface, n_interface)`（FEALPy `CSRTensor`）。

#### 步骤 7：施加宏观边界条件并用 `fealpy.solver.spsolve` 求解接口位移
* **数学与物理**：外载和 Dirichlet 约束投影到接口，对自由自由度子系统进行稀疏求解：
  $$K_{\text{free}} u_{\text{free}} = F_{\text{free}} - K_{\text{fixed}} u_{\text{fixed}}$$
* **对应代码**：[`solve_interface_system()`](../../src/soptx/fem/substructure/solve.py)
* **张量形状**：`u` $\to$ `(n_interface,)`。

#### 步骤 8：内部细尺度位移回代与全场位移恢复
* **数学与物理**：根据子结构接口位移 $u_b^j$，通过矩阵乘法回代恢复所有子结构内部细节点位移 $u_i^j = N^j u_b^j$，并拼合得到全场位移场 $U$。
* **对应代码**：[`GlobalAssembler.recover_full_displacement()`](../../src/soptx/fem/substructure/assembler.py)
* **张量形状**：`U_full` $\to$ `(total_full_dofs,)`。

---

### 全流程张量形状演变表

| 步骤 | 变量名称 | 张量形状 (以 4 子结构, $2\times 2$ 细网格为例) | 物理对象与含义 |
|---|---|---|---|
| **步骤 1** | `KE_unit` | `(4, 8, 8)` | 单子结构内部 4 个细单元的单位密度基准刚度矩阵 |
| **步骤 2** | `KE` | `(4, 4, 8, 8)` | 4 个子结构 $\times$ 4 个细单元的实际单元刚度矩阵 |
| **步骤 3** | `K_local` | `(4, 18, 18)` | 4 个子结构的局部全自由度刚度矩阵 |
| **步骤 4** | `K_ii` / `K_ib` / `K_bb` | `(4, 2, 2)` / `(4, 2, 16)` / `(4, 16, 16)` | 内部刚度块 / 耦合刚度块 / 接口刚度块 |
| **步骤 5** | `K_s` / `N` | `(4, 16, 16)` / `(4, 2, 16)` | 子结构 Schur 补缩聚刚度矩阵 / 内部位移恢复矩阵 |
| **步骤 6** | `system.stiffness` | `(n_interface, n_interface)` | 宏观全局接口刚度矩阵 (FEALPy `CSRTensor`) |
| **步骤 7** | `u` ($u_{\mathcal{B}}$) | `(n_interface,)` | 宏观全局接口位移向量 |
| **步骤 8** | `U_full` | `(total_full_dofs,)` | 包含所有细节点自由度的全场位移向量 |

---

### PIML 代理模型与 8 步流程的结合机制

子结构静力缩聚在 SOPTX 中作为 **PIML（物理增强机器学习 / Huang 2023）代理模型的精确基线与即插即用接口**：

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        子结构 8 步全流程与 PIML 接入点                  │
│                                                                        │
│  [步骤 1～4]  局部网格构造、SIMP 刚度提取与分块 (K_ii, K_ib, K_bb)     │
│                     │                                                  │
│                     ▼                                                  │
│  [步骤 5: 核心分支点 ── 可即插即用替换]                                 │
│    ├── 经典有限元路径 (FEAStaticCondensation):                         │
│    │     数值消元求逆: invK_ii_K_ib = solve(K_ii, K_ib)                │
│    │     产出精确真值: (K_s,exact, N_exact)  ──> 作为 PIML 训练标签    │
│    │                                                                   │
│    ├── PIML 路线 A (预测形函数):                                       │
│    │     代理预测形函数: ρ ──> 神经网络 ──> N_pred                      │
│    │     能量一致构造刚度: K_s = K_bb + K_bi @ N_pred                  │
│    │     极速细尺度回代: u_i = N_pred @ u_b                            │
│    │                                                                   │
│    └── PIML 路线 B (直接预测缩聚刚度):                                 │
│          网络极速推理: ρ ──> 神经网络 ──> 变形空间 Cholesky 条目 L     │
│          物理结构保障: K_s = R_perp @ (L @ L^T) @ R_perp^T (保零空间)   │
│          安全门禁机制: 特征值退化时自动回退到 FEAStaticCondensation    │
│                     │                                                  │
│                     ▼                                                  │
│  [步骤 6～8]  全局接口装配 (K_B) ──> 接口求解 (u_B) ──> 细尺度位移恢复  │
└────────────────────────────────────────────────────────────────────────┘
```

1. **唯一替换点：步骤 5**  
   PIML 代理模型（无论是预测形函数的路线 A，还是直接预测缩聚刚度的路线 B）**只替换步骤 5 中的数值消元**。步骤 1～4 的网格构建与局部刚度提取、步骤 6～8 的全局接口装配、稀疏求解与细尺度位移恢复 **100% 完全复用**。
2. **多态基类统一契约**  
   抽象基类 [`StaticCondensationBase`](../../src/soptx/fem/substructure/condensation.py) 统一了 `condense(K_local, rho_local)` 与 `recover(u_b)` 接口。`FEAStaticCondensation` 负责精确求解并产出训练标签，`PIMLStaticCondensation` 负责极速推理并在异常时无缝回退精确基线。

---

## 实现特性

### 2D/3D 统一

`SubstructureMesh` 和 `GlobalAssembler` 的维度通过 `len(box_span)` 或
`len(domain_size)` 自动推断。关键维度的分支点：

| 位置 | 2D 行为 | 3D 行为 |
|---|---|---|
| 网格 | `QuadrangleMesh.from_box` | `HexahedronMesh.from_box` |
| 材料 | `hypothesis='plane_stress'` | 默认（3D 本构） |
| DOF 编号 | `2 * gnode + k` (k=0,1) | `3 * gnode + k` (k=0,1,2) |
| 节点分类 | 检查 x, y 两面距 | 检查 x, y, z 三对面距 |
| 全局节点索引 | `gx * n_full_nodes_y + gy` | `(gx * n_full_nodes_y + gy) * n_full_nodes_z + gz` |

节点分类使用向量化 `bm.abs` + `bm.nonzero`，避免 Python 逐节点循环。

### 节点分类约定

对规则矩形/六面体子结构与一阶四边形/六面体细网格，采用节点级分类：

- 节点坐标到子结构任一边界面距离 < $\varepsilon = 10^{-7}$ → **接口（边界）节点**
- 其余 → **内部节点**

节点 $n$ 的第 $k$ 个位移分量自由度为 $d \cdot n + k$（$d$ 为空间维度，
$k = 0, \dots, d-1$）。$n_i$、$n_b$ 由该约定唯一确定，$\mathbf{N}^j \in \mathbb{R}^{n_i \times n_b}$、
$\mathbf{K}_s^j \in \mathbb{R}^{n_b \times n_b}$ 的维度与此一致。

### PIML 路线 A 实现机制（多尺度形函数预测）

路线 A 以多尺度形函数 $\mathbf{N} \in \mathbb{R}^{n_i \times n_b}$ 为学习目标，由
`ShapeFunctionCondensation` 实现，与路线 B 的 `PIMLStaticCondensation` 并列。

**参数化。** 子结构做刚体运动时内部位移完全由接口位移决定且与密度无关，即
$\mathbf{N}\mathbf{R}_{\text{rigid}} = \boldsymbol{\Phi}_i$ 对一切密度成立。因此取

$$\widehat{\mathbf{N}} = \boldsymbol{\Phi}_i \mathbf{R}_{\text{rigid}}^{\mathsf T} + \mathbf{M}\mathbf{R}_\perp^{\mathsf T},$$

网络只输出变形子空间上的 $\mathbf{M} \in \mathbb{R}^{n_i \times m}$。这与路线 B 的
Cholesky-on-$\mathbf{R}_\perp$ 参数化同构，也是 Huang 2023 式 (13)(14) 几何约束的等价
实现：那里用求和约束逐条消元，这里用刚体模态正交补一次性消掉。

$\boldsymbol{\Phi}_i$ 与 $\mathbf{R}_{\text{rigid}}$ 同源，由
`SubstructurePrototype.rigid_interior_modes` **解析**给出（同一转动中心的刚体位移场在
内部自由度上的取值，经 QR 的上三角因子换基），不含任何有限元装配；实测它与有限元
缩聚导出的 $\mathbf{N}\mathbf{R}_{\text{rigid}}$ 相差 $4.0\times10^{-16}$。

**二阶效应。** 记 $\mathbf{N} = \mathbf{N}^* + \mathbf{E}$，式 (17) 的误差有精确闭式

$$\widehat{\mathbf{N}}_{\text{full}}^{\mathsf T}\mathbf{K}\widehat{\mathbf{N}}_{\text{full}} - \mathbf{K}_s = \mathbf{E}^{\mathsf T}\mathbf{K}_{ii}\mathbf{E},$$

一阶项被 $\mathbf{K}_{ii}\mathbf{N}^* = -\mathbf{K}_{ib}$ 精确抵消。由此得到三条**构造
性质**，对应的检查因而无需进入门禁：

1. $\mathbf{K}_{ii}$ 正定 $\Rightarrow$ 误差半正定 $\Rightarrow$ 式 (17) 只会**高估**
   刚度，结构柔度只会被低估，与变分原理一致；正定性无需判定。
2. $\widehat{\mathbf{K}}_s$ 限制到变形子空间上对**任意** $\mathbf{N}$ 都严格正定：若
   $\boldsymbol{v}$ 属该子空间且 $\boldsymbol{v}^{\mathsf T}\widehat{\mathbf{K}}_s\boldsymbol{v}=0$，
   则 $\mathbf{N}_{\text{full}}\boldsymbol{v}$ 落在 $\mathbf{K}$ 的零空间即刚体模态中，
   其接口分量 $\boldsymbol{v}$ 因而属刚体子空间，与假设矛盾。秩亏是构造性质。
3. $\widehat{\mathbf{K}}_s\mathbf{R}_{\text{rigid}} = \mathbf{0}$ 精确成立：参数化保证
   $\widehat{\mathbf{N}}\mathbf{R}_{\text{rigid}} = \boldsymbol{\Phi}_i$，于是
   $\mathbf{N}_{\text{full}}\mathbf{R}_{\text{rigid}}$ 恰是刚体位移场，被 $\mathbf{K}$ 零化。

实测：受控扰动扫描的 log-log 斜率 $2.0030$（理论 2）跨 7 个数量级，放大系数
$C=0.8445$；训练网络的形函数误差 $\varepsilon_N = 8.97\%$ 经式 (17) 压到
$\varepsilon_{K17} = 0.44\%$。

**门禁机制与精确回退。** 没有保证的是**高估的幅度**——$\mathbf{E}$ 任意大时
$\mathbf{E}^{\mathsf T}\mathbf{K}_{ii}\mathbf{E}$ 也任意大。门禁按代价递增分三道：

| 门禁 | 判据 | 缺省阈值 | 实测读数 | 性质 |
|---|---|---|---|---|
| 刚体零空间残量 | $\lVert\widehat{\mathbf{K}}_s\mathbf{R}_{\text{rigid}}\rVert/\lVert\widehat{\mathbf{K}}_s\rVert$ | $10^{-10}$ | $1.2\times10^{-16}$ | 复核性质 3，拦配置错误 |
| 刚化上界 | $\lambda_{\max}(\widehat{\mathbf{K}}_s-\mathbf{K}_{bb})/\lambda_{\max}(\mathbf{K}_{bb})$ | $2\times10^{-2}$ | $8.1\times10^{-3}$ | **唯一实质门禁** |
| 变形子空间条件数 | $\lambda_{\min}/\lambda_{\max}$ 于 $\mathbf{R}_\perp$ 上 | $10^{-8}$ | $6.1\times10^{-3}$ | 性质 2 只保证定性非零 |

刚化上界取 $\mathbf{K}_{bb}$ 为参照量有两重理由：它是式 (17) 取 $\mathbf{N}=0$（内部
完全固支）的结果，可由 $\mathbf{K}_{\text{local}}$ 直接切片得到而**无需任何分解**；且
精确形函数下 $\mathbf{K}_s - \mathbf{K}_{bb} = -\mathbf{K}_{bi}\mathbf{K}_{ii}^{-1}\mathbf{K}_{ib}$
半负定，判据严格满足。注意 $\mathbf{K}_{bb}-\mathbf{K}_s$ 秩亏（实测 $\operatorname{rank}=24 < n_b=40$），
故 $\lambda_{\max}$ 在精确解处恰为 $0$——严格不等式判据会落在边界上，必须用相对容差。

阈值 $2\times10^{-2}$ 对应形函数误差约 $22\%$，正是本算例中路线 A 与路线 B 精度相当的
交叉点：越过它路线 A 不再有优势，回退精确缩聚反而更合算。留出集上最大读数
$8.1\times10^{-3}$，裕度 $2.5$ 倍。故障注入确认门禁切在声明位置：输出放大 $1.40$ 倍
通过（$1.68\times10^{-2}$），$1.45$ 倍回退（$2.15\times10^{-2}$）。

门禁代价为 $O(n_b^3)$（两次接口尺度的特征值分解），精确缩聚为 $O(n_i^3)$。内部自由度
按体积增长而接口自由度按表面增长，故子结构越大门禁越便宜：实测 5×5 时门禁是一次精确
缩聚的 $5.6$ 倍，12×12 降到 $0.55$ 倍，20×20 降到 $0.34$ 倍。**因此小子结构上整套 PIML
代理（含门禁）比直接精确缩聚更慢**，示例目录的 5×5 配置是精度验证载体而非加速演示。

**细尺度位移恢复。** $\mathbf{u}_i = \widehat{\mathbf{N}}\mathbf{u}_b$ 只需单次矩阵-向量
乘法。注意路线 A 的 $\mathbf{N}$ 是预测量而路线 B 的 $\mathbf{N}$ 取精确值，因此路线 A 的
解层误差同时包含形函数误差对恢复的直接影响，**评判标准严于路线 B**；即便如此，完整 MBB
梁上路线 A 的全场位移误差 $0.153\%$ 仍优于路线 B。

### PIML 路线 B 实现机制（变形子空间 Cholesky 参数化）

自由漂浮子结构的 $\mathbf{K}_s$ 以刚体模态为**精确**零空间：若 $(\boldsymbol{u}_b,\boldsymbol{u}_i)$ 是刚体运动，则 $\mathbf{K}[\boldsymbol{u}_b;\boldsymbol{u}_i]=\mathbf{0}$，于是 $\boldsymbol{u}_i=\mathbf{N}\boldsymbol{u}_b$ 且 $\mathbf{K}_s\boldsymbol{u}_b=\mathbf{0}$（对 P1 拉格朗日单元精确成立，实测 $\lVert\mathbf{K}_s\mathbf{R}_{\text{rigid}}\rVert/\lVert\mathbf{K}_s\rVert\sim10^{-17}$）。

`SubstructurePrototype` 据此提供三个与密度无关、全部同构子结构共享的惰性缓存属性：`n_rigid`（二维 3，三维 6）、`rigid_basis` $(n_b, n_{\text{rigid}})$ 与 `deformation_basis` $(n_b, n_b-n_{\text{rigid}})$。基由平动与转动模态在接口自由度上的取值解析给出，再经完整 QR 正交化；`SubstructureMesh` 转发这三个属性。

代理按 $\widehat{\mathbf{K}}_s = \mathbf{R}_\perp\mathbf{L}\mathbf{L}^{\mathsf T}\mathbf{R}_\perp^{\mathsf T}$ 重构，$\mathbf{R}_\perp$ 即 `deformation_basis`。由此秩亏成为构造性质：$\widehat{\mathbf{K}}_s$ 在刚体子空间上恒为零，在变形子空间上正定；训练目标 $\operatorname{cholesky}(\mathbf{R}_\perp^{\mathsf T}\mathbf{K}_s\mathbf{R}_\perp)$ 无需正则。

**门禁机制与精确回退**：
`PIMLStaticCondensation` 判定 $\mathbf{L}$ 的对角线相对尺度（$\min\lvert\operatorname{diag}\rvert / \max\lvert\operatorname{diag}\rvert > 10^{-8}$）。预测异常或退化时自动回退到 `FEAStaticCondensation`，保证求解闭环绝对安全。

---

### 与 MFEM StaticCondensation 的关系

名字 "StaticCondensation" 和 Schur 补代数公式可追溯到包括 MFEM 在内的有限元传统。
但 SOPTX 的架构设计与 MFEM 的 `mfem::StaticCondensation` 有本质差异：

| 设计维度 | MFEM | SOPTX |
|---|---|---|
| 缩聚粒度 | 逐单元 | 逐子结构（一组单元的整块缩聚） |
| DOF 分类 | FE 空间自动识别（`GetNumElementInteriorDofs`） | 对子结构包围盒手工坐标容差判断 |
| 全局装配 | `S` 矩阵自动装配（含 conforming 约束） | 手工构建 interface DOF map + Scatter-Add |
| 缩聚数学 | 逐单元 LU 分解 | 整块 `linalg.solve(K_ii, K_ib)` |
| 右端项 | `ReduceRHS` 处理非零内部载荷 | 假设 $\mathbf{f}_i = \mathbf{0}$ |
| 并行 | 完整 MPI 支持 | 无（并行归 `matrix_free_elasticity/`） |
| Trace 空间 | 自动从 FEColl 构造 | 无 trace 空间概念 |

SOPTX 的架构直接对应 Huang 2023 §2.2 的子结构缩聚框架——规则的 $m \times m$（或
$m \times m \times m$）子域、节点位置分类、逐子结构整块缩聚、Scatter-Add 全局装配——
而非 MFEM 的单元级 hybridized 消元路线。

---

## 开放问题与后续工作

1. **路线 A 的端到端大规模训练泛化**：路线 A 已由 `ShapeFunctionCondensation` 落库并配三道门禁，在 5×5 子结构、完整 MBB 梁上完成与路线 B 的同预算消融（`verify_shape_function_route.py`）。尚未做的是：换子结构尺寸与算例后重新标定 `excess_rtol`（当前值锚定在本算例的路线交叉点上，可移植性未验证），以及在复杂拓扑优化迭代分布下的 OOD 泛化评估——留出集密度为独立均匀采样，与优化过程中出现的密度场分布不同。
2. **全局 Matrix-Free 融合**：当前 `GlobalAssembler` 使用 FEALPy 原生 `CSRTensor` 组装稀疏矩阵；向无矩阵算子作用接口（Matrix-Free Operator-Action）的进一步加速融合属于后续工作。
