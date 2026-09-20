# 有限元载荷处理与装配实现

> 本文档说明 `soptx/fem` 如何承接物理载荷对象，并将其离散装配为代数方程右端项与边界条件；
> 物理载荷协议与算例生产仅作输入规格简述。连续适定性与变分依据见 `dut-postdoc:concepts/external-loads.md`。

## 1. 架构总览与数据流

载荷对象纯粹描述物理语义与几何作用实体，不绑定具体的网格、有限元空间或装配方式。同一个物理载荷对象由消费方（位移法分析器、胡张混合法分析器或子结构适配器）按各自的变分原理进行解释与离散。

### 1.1 模块职责划分

```
protocols/loads.py (定义 Load 协议与几何实体维度)
        │
        ▼
problems/loads.py (提供 4 类 frozen dataclass 实现)
        │
        ▼
┌──────────────────────────────────────────────┐
│ soptx/fem 离散与装配核心                      │
│ ├─ load_projection.py (点力 / 线载荷节点投影)  │
│ ├─ boundary_loads.py  (连续 P1 迹 L2 投影与核验)│
│ ├─ integrators/       (边界与体积积分器)       │
│ └─ analyzers/         (位移元装配 / 混合元强插值)│
└──────────────────────────────────────────────┘
```

| 载荷类型 | 协议与 Dataclass | 作用实体 | 余维数 | 主要离散手段 |
|---|---|---|:---:|---|
| **体力** | `BodyForce` / `BodyForceLoad` | 区域 $\Omega$ | 0 | 单元体积数值积分（`SourceIntegrator`） |
| **面牵引** | `BoundaryTraction` / `BoundaryTractionLoad` | 边界 $\Gamma_N$ | 1 | 边界数值面积分 或 应力法向迹插值/提升 |
| **线载荷** | `LineTraction` / `LineTractionLoad` | 曲线 / 棱线 | 3D: 2, 2D: 1 | 高阶线单元 Gauss–Legendre 节点投影 |
| **集中力** | `PointForce` / `PointForceLoad` | 作用点 $\boldsymbol{x}_0$ | $d$ | 节点插值匹配投影（仅位移法，`mode="exact"`） |

消费方一律基于协议类型 `isinstance` 分派，不依赖私有字段。

### 1.2 两种基本离散手段

| 离散手段 | 适用载荷 | 核心入口 | 输出排布与行为 |
|---|---|---|---|
| **节点投影** | `PointForce`、`LineTraction` | `load_projection.project_nodal_loads` | 节点优先排布（`dim * node + component`）。若传入体力或面牵引则显式报错。 |
| **变分积分** | `BoundaryTraction`、`BodyForce` | `LinearForm` + `SourceIntegrator` / `LagrangeBoundarySourceIntegrator` | 排布由 `TensorFunctionSpace.dof_priority` 决定，直接装配进全局载荷向量。 |

---

## 2. 两类有限元的载荷装配机制

### 2.1 位移型有限元（`LagrangeFEMAnalyzer`）

位移法将外载荷作为自然边界条件或体力泛函弱进入平衡方程右端项。

#### 边界类型门禁
装配前由 `_non_body_loads_by_boundary_type` 统一做边界语义拦截：
- `'mixed'`：正常装配体力、边界牵引、线载荷与点力（以及伴随载荷双列）；
- `'dirichlet'`：纯位移边界问题（如制造解算例）。自然边界项不应进入右端，若 `pde.loads()` 仍声明了非体力载荷，显式抛出错误，严禁静默丢弃；
- 其他未定义边界类型：报错拒绝。

#### 分布载荷数值积分
- **体力**：通过 `SourceIntegrator(source=load.body_force, q=self._integration_order)` 组装；
- **边界牵引**：通过 `LagrangeBoundarySourceIntegrator` 逐面做数值高斯积分。判定函数 `threshold` 在**面重心**上调用一次，面重心命中则整个面计入积分。

#### 节点力投影与高阶线载荷
对于集中力和线载荷，通过 `load_projection.py` 计算一致节点力并叠加：
- **点力装配模式**：
  - `mode="exact"`（全尺度位移元默认）：作用点在 $\ell_\infty$ 范数下与唯一网格插值节点的距离 $\leqslant 10^{-10}$；合力精确填入该节点自由度，命中数不为 1 则报错；
  - `mode="nearest_boundary"`（子结构 `problem_adapter`）：按物理边界坐标面筛选候选节点后就近均分，防止边界载荷被吸附至子结构内部节点。
- **高阶线载荷一致节点力**：
  - `project_line_traction` 将命中节点按线走向排序，每 `degree` 个区间组合为一个高阶一维 Lagrange 单元，单元内采用 `degree + 2` 点 Gauss–Legendre 求积；
  - 约束：`degree` 必须等于位移空间阶次，节点在几何上必须严格共线等距（相对判据 $10^{-9} \times$ 单元长度）；
  - 常值线载荷 $q$、单元长 $h$ 下的一致节点力系数理论值：
    - $p=1$：$[qh/2,\ qh/2]$；
    - $p=2$：$[qh/6,\ 4qh/6,\ qh/6]$；
    - $p=3$：$[qh/8,\ 3qh/8,\ 3qh/8,\ qh/8]$。
    若将高阶节点串误当作 $P_1$ 串处理，合力虽守恒但分布完全失真，必须用一致节点力装配。

### 2.2 应力—位移混合有限元（`HuZhangMFEMAnalyzer`）

在 Hellinger–Reissner 变分原理下，未知量包含独立对称应力 $\boldsymbol{\sigma}_h$ 与不连续位移 $\boldsymbol{u}_h$。

#### 变分拒绝门禁
在 `_validated_loads` 中，若发现 `PointForce` 或 `LineTraction`，**立即抛出 `TypeError`**：
```python
raise TypeError(
    f"Hu--Zhang 不能直接离散 {type(load).__name__}: "
    "混合位移空间 [L2]^d 无迹, 余维不小于 1 的配对都不存在; "
    "请先按物理特征尺度正则化, 或投影为 BoundaryTraction."
)
```
- 理论原因：位移检验空间 $\boldsymbol{V} = [L^2(\Omega)]^d$ 无逐点值泛函；应力法向迹空间 $H^{-1/2}(\partial\Omega)$ 无法容纳点测度（Sobolev 负指标临界失败，见理论页附录 A.2）。

#### 体力装配
体力进入分块鞍点方程第二行（位移平衡方程右端项），采用 `SourceIntegrator` 积分后填入时按方程符号约定取负。

#### 边界牵引强施加与边重心判定
边界牵引在混合法中属于**本质边界条件**，须在应力法向迹自由度上强插值或 Lifting（拓扑优化中统一采用 Lifting 分解 $\boldsymbol{\sigma} = \boldsymbol{\sigma}_0 + \boldsymbol{\sigma}_g$ 以确保对密度的伴随求导一致性）。

- **关键防御机制（边重心判定）**：
  在调用 `set_dirichlet_bc` 时，必须使用 `_prescribed_traction_on_edges`（按边重心 `bm.mean(points, axis=-2)` 判定整边是否受载），而非逐点判定：
  - *致命陷阱*：若在边上的迹插值点 `(NEb, p+1, GD)` 逐点判定，当载荷区与挖除区（如对称边界）相邻时，处于分界点处的插值点牵引被判为 0，而 `boundary_interpolate` 仍将该点打上 `isDDof` 标记，导致在交界点强加 $\sigma_{nn} = 0$；
  - *后果*：导致离散合力凭空亏损，亏损量等于端点迹系数乘该阶 Newton–Cotes 端点权（$p=2,3,4$ 时分别亏损 $1/6$、$1/8$、$7/90$）；改用边重心整边全有或全无后彻底消除该隐患。

---

## 3. 局部与集中载荷的正则化、迹投影与核验

### 3.1 集中载荷的特征尺度正则化

由于混合法严格拒绝点力，在进行位移法与混合法的受控对比时，物理上的集中力必须在 Problem 层按物理接触特征尺度（如压头宽度、轴承接触面宽度）改写为局部均布牵引（如 `FixedFixed` 基准题中设置 $w = 1\,\mathrm{mm}$，牵引强度 $t = P/w$）：
- **物理量而非数值参数**：$w$ 是固定物理建模量，网格加密时解收敛到该物理问题的连续解；
- **拒绝分析器自动转换**：$w \to 0$ 对应发散的不适定点力问题，分析器若私自代选 $w$ 属于隐式篡改物理题意。

### 3.2 连续 $P_1$ 迹空间 $L^2$ 投影

当分布牵引的载荷区间与有限元边界网格单元不对齐时，直接强施加或数值求积均存在误差。`fem/boundary_loads.py` 提供了 `project_patch_traction_to_p1_trace`：

求解连续分片一次迹函数 $\boldsymbol{t}_h \in [W_h^1(\Gamma_N)]^d$，使得：
$$
\int_{\Gamma_N} \boldsymbol{w}_h \cdot \boldsymbol{t}_h \,\mathrm{d}s
= \int_{\Gamma_N} \boldsymbol{w}_h \cdot \bar{\boldsymbol{t}}_l \,\mathrm{d}s,
\qquad \forall\, \boldsymbol{w}_h \in [W_h^1(\Gamma_N)]^d
$$

| 实现核心 | 机理与数学保证 |
|---|---|
| **解析重叠积分** | 单元积分区间解析截断为 $[a, b] = [\max(x_0, \text{left}), \min(x_1, \text{right})]$。**严禁使用高斯数值求积**，解析计算基函数反导数以消除不连续跳变带来的求积误差。 |
| **严格守恒律** | 常数与坐标线性函数天然属于检验空间，数学上严格保证**合力守恒**与**一阶力矩守恒**。 |
| **质量矩阵求解** | 组装 $(n_{\text{cells}} + 1)$ 阶三对角质量矩阵，直接解线性系统得到迹系数。 |
| **两法无偏施加** | 输出标准的 `P1TraceLoad` 对象：位移法通过其 `__call__` 进行弱面积分，混合法通过其迹点值进行强插值，使两法看到完全相同的离散载荷泛函。 |

投影结果在载荷区外带有几何衰减的振荡尾（三对角质量矩阵的 Green 函数，每单元约 0.27）。网格过粗时尾部触及 Dirichlet 边界，那部分载荷会被强加自由度真实吞掉，量级约 $P \cdot 0.27^{n_x/2}$——此时残差依然为 0，只有 §3.3 的合力核验能报警；$n_x \ge 40$ 起已在 $10^{-6}$ 相对容差之下。

### 3.3 离散合力与核验门禁

常规有限元程序以“面/边重心落在载荷区内”进行整面选取。当载荷端点落在单元内部时，会整面多算或少算，产生相对误差可达 $O(h_F / w)$ 的静默偏差（计算照常收敛，但收敛到错误的合力）。

`boundary_loads.py` 提供了自动核验工具：
- `boundary_load_resultant(force, dimension)`：将全局已装配载荷向量按自由度求和还原离散合力；
- `check_boundary_load_resultant(force, dimension, expected)`：输出 `LoadResultantReport`，比对绝对误差、相对误差并在超出容差（默认 $10^{-10}$）时触发门禁。

---

## 4. 已知限制

| 模块 / 限制项 | 具体约束与现状 |
|---|---|
| **算子层级 `'fa'`** | 对称消元没有重叠归约插入点，目前仅支持单 rank（`lagrange_fem_analyzer.apply_bc('fa')`）。 |
| **算子层级 `'ea'`** | 暂不支持伴随双列右端项。 |
| **边界类型定义** | `boundary_type` 当前仅覆盖 `'mixed'` 与 `'dirichlet'`。 |
| **子结构分布载荷** | `problem_adapter._assemble_integrated_load` 当前仅验证了 $p=1$ 全尺度 Lagrange 空间。 |
| **宏观角点系统** | 仅支持 `PointForce` 与 `LineTraction`，且固定按 $degree=1$ 投影。 |
| **内部自由度载荷** | 完整接口缩聚拒绝直接作用在子结构内部自由度上的载荷与约束。 |
| **面求值形状** | `CantileverMiddle2d._step_traction` 仅接受按面成批的求值点 `(..., NP, GD)`。 |
| **转置分支测试** | `_assemble_non_body_loads` 中 `dof_priority=True` 时的转置分支暂无测试覆盖。 |

---

## 相关文档

- `dut-postdoc:concepts/external-loads.md` — 载荷正则性、集中力适定性与 $P_1$ 迹投影的连续层数学理论（特别是 §3.2 与 附录 A.2）
- `dut-postdoc:concepts/huzhang/huzhang-mixed-fem.md` — 胡张混合有限元变分原理与对偶边界条件
- [`huzhang-mixed-fem-implementation.md`](huzhang-mixed-fem-implementation.md) — 胡张元特有的边界与载荷实现要点（角点松弛等）
- [`substructure-condensation-implementation.md`](substructure-condensation-implementation.md) — 静力缩聚求解流程与载荷投影上下文
- [`../problems/engineering-benchmarks.md`](../problems/engineering-benchmarks.md) — 各工程基准算例的载荷与边界设置
