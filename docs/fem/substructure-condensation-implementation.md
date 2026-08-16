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
- **路线 B：直接预测缩聚刚度。** $\boldsymbol{\rho}^j \mapsto \widehat{\mathbf{K}}_s^j$。
  推理更直接，但不天然保持与 $\mathbf{N}^j$ 的能量构造关系。

两条路线共享相同的局部输入 $\boldsymbol{\rho}^j$（逐单元 SIMP 密度）、
精确标签 $(\mathbf{N}_{\mathrm{exact}}^j, \mathbf{K}_{s,\mathrm{exact}}^j)$、
全局接入方式和下游评价指标。SOPTX 当前精确基线同时支持两条路线的真值计算；
PIML 代理原型仅实现了路线 B 雏形。

精确缩聚的完整推导、刚体模态、能量一致性和细尺度恢复见
`dut-postdoc:concepts/substructural-condensation.md`。
PIML 局部—全局契约见
`dut-postdoc:concepts/piml/piml-substructural.md`。

## 程序架构

### 文件布局

```
src/soptx/fem/substructure/              ← 核心库（成熟）
├── __init__.py                           ← 导出 SubstructureMesh, FEAStaticCondensation, GlobalAssembler
├── mesh.py                               ← SubstructureMesh: 2D/3D 子结构网格管理
├── condensation.py                       ← StaticCondensationBase, FEAStaticCondensation
├── piml_surrogate.py                     ← PIMLSurrogateNet, PIMLStaticCondensation, SurrogateContractError
└── assembler.py                          ← GlobalAssembler, InterfaceSystem

examples/
├── substructure_elasticity/              ← 精确缩聚基线（成熟）
│   ├── compare_lagrange.py              ← 端到端缩聚 vs Lagrange 全装配交叉验证
│   ├── results_analysis.md              ← 数学—代码映射契约与验收阈值
│   └── README.md
│
└── piml_substructure_elasticity/         ← PIML 代理原型（早期）
    ├── compare_exact.py                  ← 代理 vs 精确缩聚，两层误差 + 误差归因诊断
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

### 求解流程

缩聚全局流程由调用方编排，装配器和通用线性求解器按以下顺序协作：

```
1. 构建接口 DOF 集合
   遍历所有子结构的 b_dofs → 去重排序 → 全局→接口映射字典

2. K_s Scatter-Add 装配
   for each (sub_mesh, condensor):
       获取子结构边界 DOF 的全局编号
       映射到接口 DOF 索引
       K_global[b_interface[i], b_interface[j]] += K_s[i, j]

3. 由 Problem 构建全局载荷与 Dirichlet 固定 DOF
   → project_global_vector/dofs(...) 投影到接口系统

4. 施加强制边界条件并调用通用稀疏求解器
   interface_free = setdiff(所有接口DOF, interface_fixed)
   K_free @ u_b_free = F_free  →  scipy spsolve

5. 全场位移恢复
   接口位移写入 U_full
   for each condensor: u_sub_i = recover(u_sub_b)
   内部位移写入 U_full
```

步骤 3–6 对精确缩聚和 PIML 预测使用相同的代码路径——唯一的变量是
`condensor.K_s` 和 `condensor.N` 的来源（精确计算或网络预测）。

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

对规则矩形/六面体子结构与 Q4/六面体细网格，采用节点级分类：

- 节点坐标到子结构任一边界面距离 < $\varepsilon = 10^{-7}$ → **接口（边界）节点**
- 其余 → **内部节点**

节点 $n$ 的第 $k$ 个位移分量自由度为 $d \cdot n + k$（$d$ 为空间维度，
$k = 0, \dots, d-1$）。$n_i$、$n_b$ 由该约定唯一确定，$\mathbf{N}^j \in \mathbb{R}^{n_i \times n_b}$、
$\mathbf{K}_s^j \in \mathbb{R}^{n_b \times n_b}$ 的维度与此一致。

> 换用非规则子结构、非节点自由度或非规则编号时，必须重新显式定义划分与编号，
> 否则局部标签、PIML 预测与全局映射三者无法保持同一契约。

### 精确缩聚的实现选择

`bm.linalg.solve(K_ii, K_ib)` 与逐列施加单位接口位移、求解局部 Dirichlet 问题
在代数上等价；实现不显式求逆。这与 Huang 2023 式 (6)–(7) 的数学过程一致。

### PIML 统一接口与精确回退

`StaticCondensationBase` 定义了 `condense(K_local, rho_local=None) → (K_s, N)` 和
`recover(u_b) → u_i` 的统一接口。`PIMLStaticCondensation` 的 `rho_local` 参数用于
将子结构单元密度传入神经网络；精确缩聚中该参数被忽略。

`PIMLStaticCondensation` 在以下条件自动触发精确回退：

- 模型未加载（`model is None`）
- 密度输入缺失（`rho_local is None`）
- 网络推理异常
- 网络输出含 `NaN` 或 `Inf`
- 预测的 $\mathbf{L}$ 对角线数值退化（$\min\lvert\operatorname{diag}\rvert / \max\lvert\operatorname{diag}\rvert \le$ `rcond_min`，缺省 $10^{-8}$）

回退时调用 `FEAStaticCondensation.condense()`，与精确基线使用完全相同的计算路径。
网络输出维与参数化所需的独立条目数不符属**配置错误**而非数值退化，抛
`SurrogateContractError` 而不回退——静默回退只会把它掩盖成一次精度下降。

### 变形子空间上的 Cholesky 参数化

自由漂浮子结构的 $\mathbf{K}_s$ 以刚体模态为**精确**零空间：若 $(\boldsymbol{u}_b,\boldsymbol{u}_i)$ 是刚体运动，则 $\mathbf{K}[\boldsymbol{u}_b;\boldsymbol{u}_i]=\mathbf{0}$，于是 $\boldsymbol{u}_i=\mathbf{N}\boldsymbol{u}_b$ 且 $\mathbf{K}_s\boldsymbol{u}_b=\mathbf{0}$（对 P1 拉格朗日单元精确成立，实测 $\lVert\mathbf{K}_s\mathbf{R}_{\text{rigid}}\rVert/\lVert\mathbf{K}_s\rVert\sim10^{-17}$）。

`SubstructurePrototype` 据此提供三个与密度无关、全部同构子结构共享的惰性缓存属性：`n_rigid`（二维 3，三维 6）、`rigid_basis` $(n_b, n_{\text{rigid}})$ 与 `deformation_basis` $(n_b, n_b-n_{\text{rigid}})$。基由平动与转动模态在接口自由度上的取值解析给出，再经完整 QR 正交化；`SubstructureMesh` 转发这三个属性。

代理按 $\widehat{\mathbf{K}}_s = \mathbf{R}_\perp\mathbf{L}\mathbf{L}^{\mathsf T}\mathbf{R}_\perp^{\mathsf T}$ 重构，$\mathbf{R}_\perp$ 即 `deformation_basis`。由此秩亏成为构造性质：$\widehat{\mathbf{K}}_s$ 在刚体子空间上恒为零，在变形子空间上正定；训练目标 $\operatorname{cholesky}(\mathbf{R}_\perp^{\mathsf T}\mathbf{K}_s\mathbf{R}_\perp)$ 无需正则（限制后实测最小特征值 $1.0\times10^{-2}$、条件数 $\approx94$，三维为 $1.3\times10^{-3}$、$\approx270$）。

**门禁必须作用在 $\mathbf{L}$ 上**：$\widehat{\mathbf{K}}_s$ 按构造 $\lambda_{\min}\equiv0$，沿用「要求 $\lambda_{\min}>10^{-8}$」会恒定触发回退。这与历史上「重构后扣除 $10^{-6}\mathbf{I}$ 再要求 $\lambda_{\min}>10^{-8}$」导致 24/24 静默回退是同一个失效模式。改判 $\mathbf{L}$ 的对角线相对尺度后，推理路径也不再需要每次求一次 `eigvalsh`。

`compare_exact.py` 每次运行都做两项自检：校验解析刚体基确为 $\mathbf{K}_s$ 的零空间；以一组**精确**的 Cholesky 条目走一遍推理路径，检验训练目标的条目排布与推理期的重构是同一个参数化，并把由此测得的相对误差作为本次运行的误差下界报出（实测约 $3.8\times10^{-8}$，即 `float32` 精度；旧的全空间参数化为 $2.6\times10^{-4}$）。

改造前的实测证据：$\mathbf{L}\mathbf{L}^{\mathsf T}$ 在刚体子空间注入的伪刚度仅为最小非零特征值的 $1.6\%$，却因装配后接口位移 $99.98\%$ 是刚体运动而被平方量级放大，贡献了观测刚化的 $96.8\%$。该效应由 `compare_exact.py` 的零空间污染与能量归因指标量化，改造后转为对上述构造的运行期校验。

### PIML 代理网络（当前路线 B 雏形）

`PIMLSurrogateNet` 复用 `soptx.ml.MLP`，采用两层宽度为 128 的 SiLU 隐藏层。输入为展平后的子结构单元密度 $\boldsymbol{\rho}^j$，输出为变形子空间上 Cholesky 因子的独立条目（$5\times5$ 二维子结构为 $37\cdot38/2=703$ 个）。`PIMLStaticCondensation` 将其重构为 $\mathbf{L}$，投影回全部接口自由度并执行门禁；预测异常或未通过门禁时回退精确缩聚。

网络构造、张量 shape 与层序列见 [`docs/ml/mlp.md`](../ml/mlp.md)。本节只维护该网络在路线 B 缩聚链中的输入、输出和结构保持职责。

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

### 验收契约

| 验证类型 | 阈值 | 脚本 |
|---|---|---|
| 全局缩聚 vs Lagrange 交叉验证（柔度） | < $10^{-12}$ | `substructure_elasticity/compare_lagrange.py` |
| 全局缩聚 vs Lagrange 交叉验证（位移） | < $10^{-12}$ | `substructure_elasticity/compare_lagrange.py` |

机器精度级一致性支持"同一离散问题上的精确代数等价"，不直接支持性能加速、
PIML 可靠性或大规模可扩展性结论。

## 示例与测试

### 精确基线

**端到端交叉验证**：[`examples/substructure_elasticity/compare_lagrange.py`](../../examples/substructure_elasticity/compare_lagrange.py)

```bash
python examples/substructure_elasticity/compare_lagrange.py --dim 2   # 2D MBB 梁
python examples/substructure_elasticity/compare_lagrange.py --dim 3   # 3D MBB 梁
```

精确 $K_s$ 组装并求解全局接口系统，恢复全场（含由 $u_i = N u_b$ 回代的内部自由度），
与 `LagrangeFEMAnalyzer` 全装配解比较。通过时生成
`outputs/lagrange_comparison_{2d,3d}.json` 作为可复核数值证据。

局部回代关系 $u_i = N u_b$ 是这条链路的一环，被端到端的机器精度一致性传递地覆盖，
因此不再单设隔离该步骤的脚本。

### PIML 代理原型

[`examples/piml_substructure_elasticity/compare_exact.py`](../../examples/piml_substructure_elasticity/compare_exact.py)

```bash
python examples/piml_substructure_elasticity/compare_exact.py
```

在 12×2 完整 MBB 梁（`FullMBBBeam2d`，共 24 个子结构）上，以精确批量缩聚为基线对比 PIML 代理缩聚。用 300 个随机密度样本（一次批量装配 + 一次批量缩聚生成）训练 MLP，预测的 $\widehat{\mathbf{K}}_s$ 进入全局接口系统。误差分两层报告：算子层为 $\mathbf{K}_s$ 的相对 Frobenius 误差（代理本征精度，与外载无关），解层为接口位移、全场位移与柔度的相对误差（算子误差经接口系统放大后的后果）。

> 精确缩聚在此仅作基线，不再重复验证其与 Lagrange 全装配的等价性——该结论由
> `examples/substructure_elasticity/compare_lagrange.py` 建立，因此本脚本不构造
> 全尺度参考解。代理精度不设阈值断言；`--strict` 下若存在回退到精确缩聚的子结构
> 则以异常失败，使"代理是否真正生效"成为可回归状态。

## 开放问题

1. **路线 A 尚未实现。** 当前 PIML 代理仅预测 $\mathbf{K}_s$（路线 B 雏形）。
   预测 $\widehat{\mathbf{N}}^j$ 再由 $\widehat{\mathbf{K}}_s^j = (\widehat{\mathbf{N}}^j)^{\mathsf{T}} \mathbf{K}^j \widehat{\mathbf{N}}^j$
   构造的路线 A 尚未实现。路线 A 保持形函数—刚度—能量构造关系，Huang 2023
   指出这在实际中更鲁棒（即使 $\mathbf{N}$ 误差较大，由式 (17) 构造的
   $\mathbf{K}_s$ 仍接近精确解）。

2. **预测 $\mathbf{K}_s$ 的泛化尚未验证。** 当前 PIML 示例已将预测算子装配到
   全局接口系统并计算接口位移、柔顺度和细尺度恢复，但训练集、网格和载荷均很小，
   尚不能外推为泛化或加速结论。

3. **能量一致性断裂。** 当前路线 B 实现在预测成功时取精确 $\mathbf{N}$ 但替换
   预测 $\mathbf{K}_s$，$\mathbf{K}_s = \mathbf{N}^{\mathsf{T}} \mathbf{K} \mathbf{N}$
   的构造关系不成立。Huang 2023 §4.2 也坦承"预测的形函数和预测的缩聚刚度矩阵
   通常无法满足式 (17) 所描述的关系"。

4. **结构性质未完整硬保持。** 当前 Cholesky 参数化已保证缩聚刚度的对称正定构造，且保留特征值门禁与精确回退；刚体运动约束（Huang 2023 式 13–15）、分片统一性／线性完备性等结构条件尚未建立。

5. **训练/测试分布不一致。** 当前训练样本来自 $\text{Uniform}(0.3, 1.0)$，
   测试使用 sin·cos 密度场，报告的 `rel_err_Ks` 不是有定义的测试指标。
   缺少统一的 train/val/test 划分与 OOD 评估。

6. **全局 Matrix-Free 融合未开始。** 当前 `GlobalAssembler` 显式组装
   `K_global` 稀疏矩阵；向 Matrix-Free 算子作用接口的转换属于后续工作。
