# PA 部分装配内存机制与容量能力

## 实验设置与测量口径

### 运行环境


| 项目     | 配置                                                           |
| ------ | ------------------------------------------------------------ |
| 宿主系统   | Windows 11 Pro，物理内存 64 GB                                    |
| 实验系统   | WSL2 Ubuntu 24.04 LTS，内核 `6.18.33.2-microsoft-standard-WSL2` |
| CPU    | Intel Core i9-14900KF，WSL 可见 32 个逻辑核                         |
| WSL 内存 | 配置上限 `memory=48GB`，系统报告总内存约 47 GiB（`MemTotal`），swap 12 GiB   |
| 软件版本   | Python 3.12.13，numpy 2.5.1，scipy 1.18.0，torch 2.13.0+cu130   |


### 问题与执行方式

采用三维线弹性制造解问题（`DivergenceFreePolynomialElasticity3D`），使用 FEALPy 的 `TetrahedronMesh.from_box` 构建四节点四面体网格，采用 $p=1$ 的 Lagrange 向量有限元，积分参数取默认值 $q=p+3=4$。三个坐标方向均划分为 $n$ 份，单元数为 $N_C = 6n^3$，位移自由度数为 $N_{dof} = 3(n+1)^3$。实验采用单进程 CPU 执行方式，每个数据点在独立进程中测量。

### 测量口径

- **内存统计**：以 `VmRSS` 记录常驻内存，以逐阶段重置的 `VmHWM` 记录阶段峰值；阶段净增为阶段峰值与阶段起始 `VmRSS` 之差，全程绝对峰值取各阶段峰值的最大值。
- **容量预算**：以进程绝对峰值 RSS 不超过 45 GiB 为容量评估条件。
- **数值精度**：内存数值统一按「不小于 1 GiB 者保留一位小数、小于 1 GiB 者取整数 MiB」给出，耗时统一保留两位有效数字；同一表中各列为独立读数，舍入后按「起点 + 净增 = 峰值」相加可有 0.1 GiB 的偏差。

---



## 一、装配前：import 底座与网格构建

本节测量 PA 算子装配前的内存开销，即固定网格下只需支付一次、后续各轮计算可直接复用的部分，与每轮重复支付的算子构建与计算相区分。PA 与 FA 的前置成本存在明确分流：

- **公共基础**：模块导入，以及网格、有限元空间和材料对象的构建（第 1、2 小节）。
- **CSR 符号结构免除**：FA 的 `pattern` 路线必须通过 `build_csr_pattern` 预先建立 CSR 骨架与槽位映射；PA 算子在求积点级进行局部张量收缩，从根本上省去了这一阶段（第 3 小节）。



### 1. import 底座（不随 n）

`run.py` 显式加载 FEALPy、SOPTX 及底层依赖栈（NumPy、SciPy、PyTorch、SymPy），静态导入底座产生的常驻内存为 **607 MiB**。三档工况进入 `mesh` 阶段前的实测读数分别为 607.7、605.4、607.8 MiB，不随网格规模 $n$ 变化，与 FA 实测底座（606 MiB）及 EA 实测底座（607 MiB）一致。

数据来源：`outputs/cache_fast_n{32,48,64}.json` 的 `mesh_before_MiB`。

### 2. 网格与空间构建（随 n）

PA 的 `mesh` 阶段把网格、有限元空间、材料与算子门面的构建合并为一段测量，不含任何装配；阶段结束后执行 `gc.collect()` 与 `malloc_trim()`，再读取构建后 RSS。峰值 RSS 与构建后 RSS 均为进程绝对量，构建后 RSS 净增以各次测量的 import 底座为基准。


| n   | $N_{dof}$ | 构建期峰值 RSS（GiB） | 构建后 RSS（MiB） | 构建后 RSS 净增（MiB） | 构建耗时（s） |
| --- | --------- | -------------- | ------------ | --------------- | ------- |
| 32  | 107,811   | 1.3            | 705          | 98              | 12      |
| 48  | 352,947   | 3.0            | 914          | 309             | 36      |
| 64  | 823,875   | 6.2            | 1329         | 721             | 86      |


三档的构建期峰值均远高于构建后常驻量（$n=64$ 为 6.2 GiB 对 1.3 GiB），瞬时量在阶段结束时绝大部分归还，容量评估需计入这一峰值。

三点口径说明：

- 本表的构建后 RSS 是 `malloc_trim` **之后**的读数，故低于 FA 与 EA 同规模的对应列（未 trim，$n=32$ 分别为 782、779 MiB）；三档净增 98、309、721 MiB 与 FA 报告中网格构建的 trim 后残留（96、308、719 MiB）一致，两者并不矛盾，跨报告横向比较时须按同一口径取值。
- 该读数即第二章阶段 1 的起点 RSS（0.7、0.9、1.3 GiB），故本章到下一章的水位链条连续。
- PA 的 `mesh` 为合并阶段，无法像 EA 的分段测量那样单独给出 `space` 子阶段的开销；本目录也未测 $n=96$、$n=124$ 两档，相应数据点缺失。

数据来源：`outputs/cache_fast_n{32,48,64}.json` 的 `mesh_*` 字段。

### 3. 与 FA 路线对比（免 CSR 符号结构）

FA 的模式先行装配路线（`fast + pattern`）在进入单刚计算前，必须针对有限元空间调用 `build_csr_pattern`，构建张量级 CSR 骨架及分量块槽位映射，在 $n=32, 48, 64$ 时分别引入 47、215、519 MiB 的常驻开销（FA 报告第一章第 3 小节）。

PA 算子属于无矩阵范式（Matrix-Free），在整个生命周期内不显式组装全局 CSR 矩阵，因而**完全免除 CSR 符号结构的构建与常驻开销**。网格与空间构建完成后，即可直接进入积分点数据缓存与算子计算。

---



## 二、阶段 1：积分点数据计算与缓存（随 n）



### 1. 算子构建与积分点缓存机制

PA 算子通过分析器调用 `assemble_stiff_matrix()` 完成构建，并在内部构建常驻算子实例；与 EA 缓存单刚张量不同，PA 彻底舍弃单元刚度矩阵 $K_e$ 的计算与存储，改为缓存各单元求积点上的几何与材料数据以及自由度映射 `cell2dof`。

PA/QA 层级：

$$
y = \sum_e G_e^{\mathsf T} B_e^{\mathsf T} \left[ D_e \left( B_e \left( G_e x \right) \right) \right]
$$

PA 不形成 $B_e$，而是把它拆成三个因子：

$$
B_e = S\,\Gamma(J_e)\,\hat B
$$

$D_e$ 逐积分点是一个标量乘一个常量矩阵：

$$
D_{e,q} = w_q\,\lvert J_{e,q}\rvert\,\rho_e\,D
$$

相关源码实现：

- 分析器工厂与装配入口：
  - `[src/soptx/fem/analyzers/builders.py](../../src/soptx/fem/analyzers/builders.py)` 中的 `build_serial_analyzer`
  - `[src/soptx/fem/analyzers/lagrange_fem_analyzer.py](../../src/soptx/fem/analyzers/lagrange_fem_analyzer.py)` 中的 `assemble_stiff_matrix`
- PA 算子构建与积分点缓存逻辑：
  - `[src/soptx/fem/levels/partial.py](../../src/soptx/fem/levels/partial.py)` 中的 `PartialAssembly.build`
- 算子内核类：
  - `[src/soptx/fem/kernels/reference_basis.py](../../src/soptx/fem/kernels/reference_basis.py)` 中的 `ReferenceBasis`
  - `[src/soptx/fem/kernels/geometric_factors.py](../../src/soptx/fem/kernels/geometric_factors.py)` 中的 `GeometricFactors`
  - `[src/soptx/fem/kernels/qfunction.py](../../src/soptx/fem/kernels/qfunction.py)` 中的 `LinearElasticQFunction`
  - `[src/soptx/fem/kernels/restriction.py](../../src/soptx/fem/kernels/restriction.py)` 中的 `ElementRestriction`
- 算子计算核（只吃数组）：
  - `[src/soptx/fem/kernels/gradients.py](../../src/soptx/fem/kernels/gradients.py)` 中的 `physical_gradient` 与 `physical_gradient_transpose`
  - `[src/soptx/fem/kernels/qfunction.py](../../src/soptx/fem/kernels/qfunction.py)` 中的 `weighted_stress` 与 `weighted_stress_diagonal`

入口 `analyzer.assemble_stiff_matrix()` 经 `registry.create_level('pa', ...)` 分派至 `PartialAssembly.build`。该方法分 build 与 setup 两段：build 段取参考单元上的基函数 `ReferenceBasis`（按单元形状、$p$、$q$ 进程级缓存）；setup 段由模块级函数 `quadrature_geometry` 从同一份 Jacobi 矩阵算出 $J^{-1}$ 与逐点权重，分别装进 `GeometricFactors` 与 `LinearElasticQFunction`，再构造分量布局的单元限制算子 `ElementRestriction`。走查脚本 `walkthrough.py` 手工拼装 PA 时调的是同一个函数：

```python
# 节选自 src/soptx/fem/levels/partial.py
@classmethod
def build(cls, space, integrator, pattern=None, *,
          reference_basis=None, **kwargs) -> "PartialAssembly":
    if not isinstance(integrator, LinearElasticIntegrator):
        raise TypeError(
            "PA / UA 层级目前只支持 LinearElasticIntegrator, 得到 "
            f"{type(integrator).__name__}"
        )

    ctx = integrator.fetch_context(space)
    scalar_space, mesh = ctx.scalar_space, ctx.mesh

    # (嵌入流形网格的 NotImplementedError 检查从略)

    # ---- build: 只依赖单元形状, p 与 q, 与网格几何无关 ----
    if reference_basis is None:
        reference_basis = ReferenceBasis.build(scalar_space=scalar_space, q=ctx.q)

    # ---- setup: 依赖网格几何与当前设计变量, 每张网格算一次 ----
    jacobi_inverse, weighted_measure = quadrature_geometry(ctx)

    geometric_factors = GeometricFactors(jacobi_inverse=jacobi_inverse)

    qfunction = LinearElasticQFunction(
                        elastic_matrix=integrator.material.elastic_matrix()[0, 0],
                        weighted_measure=weighted_measure,
                        coef=integrator.coef)

    # 分量布局的 G: 张量空间的自由度排序在这里一次换好, B 只认 (NC, ldof, GD)
    restriction = ElementRestriction.from_integrator(integrator, space)

    return cls(space=space,
            restriction=restriction,
            reference_basis=reference_basis,
            geometric_factors=geometric_factors,
            qfunction=qfunction)
```

```python
# 节选自 src/soptx/fem/levels/partial.py
def quadrature_geometry(ctx) -> Tuple[TensorLike, TensorLike]:
    mesh, bcs, ws = ctx.mesh, ctx.bcs, ctx.ws

    # entity_view 的 index 用 None 表示全体
    geo_index = None if ctx.index is _S else ctx.index
    jacobi = mesh.entity_view('cell').jacobi_matrix(bcs, index=geo_index)

    # inv 后下标变成 (NC, NQ, TD, GD), 即 d xi_r / d x_b
    jacobi_inverse = bm.linalg.inv(jacobi)

    if isinstance(mesh, SimplexMesh):
        weighted_measure = ws[None, :] * ctx.cell_measure[:, None]
    else:
        weighted_measure = ws[None, :] * bm.abs(bm.linalg.det(jacobi))

    return jacobi_inverse, weighted_measure
```

块内各张量与数学对象的对应如下：


| 数学对象                      | 来源                                                             | 落入的内核类                   |
| ------------------------- | -------------------------------------------------------------- | ------------------------ |
| $\hat B$                  | `reference_basis.grad`                                         | `ReferenceBasis`         |
| $\Gamma(J_e)$             | `jacobi_inverse`                                               | `GeometricFactors`       |
| $w_q\lvert J_{e,q}\rvert$ | `weighted_measure`                                             | `LinearElasticQFunction` |
| $\rho_e$                  | `integrator.coef`                                              | `LinearElasticQFunction` |
| $D$                       | `integrator.material.elastic_matrix()`                         | `LinearElasticQFunction` |
| $S$                       | `qfunction.strain_map`，由 `LinearElasticQFunction.__init__` 现生成 | `LinearElasticQFunction` |
| $G$                       | `ElementRestriction.from_integrator(integrator, space)` | `ElementRestriction`     |


两点补充。其一，张量空间的自由度排序 `dof_priority` 不对应任何数学对象，只是 L 向量的布局：`ElementRestriction.from_integrator` 在构造时用 FEALPy 的 `flatten_indices` 把 `cell2dof` 一次重排成 $(N_C, \text{ldof}, GD)$，此后 `gather` 直接交出规范布局，$B$ 不再重排，与 MFEM 的 E 向量约定一致。其二，`weighted_measure` 按网格类型分两支——非单纯形取 $w_q\lvert J_{e,q}\rvert$，单纯形取 $w_q\lvert T_e\rvert$，因为 FEALPy 的重心坐标求积权重之和为 1 而非参考单元测度；若在该支误用 $\lvert J\rvert$，三角形差 2 倍、四面体差 6 倍。本实验是四面体网格，走的是后一支。

$G$ 与另外三个内核同出 `build`，不再有 "内核构造" 与 "算子构造" 的切分。早先把三个浮点内核拆进一个 `build_kernels`、把 $G$ 留在 `build`，理由是 UA 层级借用同一个 `build_kernels`；这条理由不成立：$G$ 只由整数索引 `cell2dof` 定义，不参与任何浮点运算，"UA 与 PA 逐位相同" 的论证管不到它。切分反而让 `ElementRestriction(...)` 的构造在 `partial.py` 与 `unassembled.py` 里字面重复了一遍。现在两个层级都走 `ElementRestriction.from_integrator`，取法只有一处。

四个内核一并存入算子实例，后续 `operator @ x` 只读它们，不再触碰网格与空间对象。

### 2. 峰值 RSS 与常驻 RSS 实测对比（随 n）

下表统计 `assemble_stiff_matrix('pa')` 在各规模下的实测数据：


| n   | $N_C$     | $N_{dof}$ | 起点 RSS（GiB） | 峰值净增（GiB） | 阶段峰值 RSS（GiB） | 缓存后常驻（GiB） | 构建耗时（s） |
| --- | --------- | --------- | ----------- | --------- | ------------- | ---------- | ------- |
| 32  | 196,608   | 107,811   | 0.7         | 0.7       | 1.4           | 1.0        | 1.35    |
| 48  | 663,552   | 352,947   | 0.9         | 2.6       | 3.5           | 2.0        | 3.38    |
| 64  | 1,572,864 | 823,875   | 1.3         | 5.9       | 7.2           | 3.8        | 8.16    |


相比于 FA 和 EA 在阶段 1 需要耗时 4.9 s 生成并缩并 $K_e$，PA 在 $n=64$ 下仅需计算几何逆雅可比与求积权重，耗时为 8.16 s，且常驻增量中没有任何单元刚度阵。

数据来源：`outputs/cache_fast_n{32,48,64}.json`。

---



## 三、阶段 2：算子乘的内存机制与执行效率（随 n）



### 1. 算子乘机制与理论工作区构成

PA 的无矩阵算子乘通过链式张量收缩实现：

$$
y = K x = G^T B^T D B G x
$$

其中 $G$ 为限制算子，$B$ 为梯化求值算子，$D$ 为逐点本构算子。算子类 `PartialAssembly`（`[src/soptx/fem/levels/partial.py](../../src/soptx/fem/levels/partial.py)`）实现 `operator @ x` 的核心代码如下：

```python
class PartialAssembly:
    def __matmul__(self, x: TensorLike) -> TensorLike:
        # 基函数算子 B 的两份数据：参考基函数梯度与逐单元 J^{-1}，供下面两步共用
        reference_grad = self._reference_basis.grad
        jacobi_inverse = self._geometric_factors.jacobi_inverse
        qfunction = self._qfunction

        # 1. Gather (G): 根据 cell2dof 提取局部自由度向量 x_E
        x_E = self._restriction.gather(x)

        # 2. 梯化求值 (B): 将单元位移转换为求积点位移梯度张量 grad_u
        grad_u = physical_gradient(x_E, reference_grad=reference_grad,
                                jacobi_inverse=jacobi_inverse)

        # 3. 逐点本构 (D): 计算求积点上乘了积分权重的应力 s_Q
        s_Q = weighted_stress(grad_u, weighted_coef=qfunction.weighted_coef,
                            elastic_matrix=qfunction.elastic_matrix,
                            strain_map=qfunction.strain_map)

        # 4. 梯化转置 (B^T): 将 s_Q 投影回单元局部内力向量 y_E
        y_E = physical_gradient_transpose(s_Q, reference_grad=reference_grad,
                                        jacobi_inverse=jacobi_inverse)

        # 5. Scatter-add (G^T): 将单元贡献累加回全局向量 y
        return self._restriction.scatter_add(y_E)
```

执行步骤与张量形态如下：

1. **Gather（$G$）**：根据 `cell2dof` 提取局部自由度向量 $x_E$（形状 $(N_C, 4, 3)$，即 ldof $\times$ GD 的分量布局）；
2. **梯化求值（$B$）**：利用参考梯度与 $J^{-1}_q$ 将位移计算为各求积点上的位移梯度张量 $\nabla u$（形状 $(N_C, 20, 3, 3)$）；
3. **逐点本构（$D$）**：由各求积点应变计算柯西应力张量 $\sigma$（形状 $(N_C, 20, 6)$）；
4. **梯化转置（$B^T$）**：将求积点应力投影回单元局部节点内力向量 $y_E$（形状 $(N_C, 4, 3)$）；
5. **Scatter-Add（$G^T$）**：将单元力累加回全局向量 $y$。

逐步对应到 §2.1 的数学对象，实际执行的张量收缩如下：


| 步骤  | 数学                              | einsum 下标                                |
| --- | ------------------------------- | ---------------------------------------- |
| 1   | $x_E = G_e x$                   | `gather`，按 `cell2dof` 取值                 |
| 2a  | $\partial u_d / \partial \xi_r$ | `'qir, cid -> cqrd'`（$\hat B$）           |
| 2b  | $\partial u_d / \partial x_b$   | `'cqrb, cqrd -> cqdb'`（$\Gamma$）         |
| 3a  | $\varepsilon_s$                 | `'sdb, cqdb -> cqs'`（$S$，至此凑齐 $B_e x_e$） |
| 3b  | $\sigma_s$                      | `'st, cqt -> cqs'` 再乘 `weighted_coef`（$D_{e,q}$） |
| 4a  | $S^{\mathsf T}\sigma$           | `'sdb, cqs -> cqdb'`                     |
| 4b  | $\Gamma^{\mathsf T}$            | `'cqrb, cqdb -> cqrd'`                   |
| 4c  | $\hat B^{\mathsf T}$            | `'qir, cqrd -> cid'`                     |
| 5   | $\sum_e G_e^{\mathsf T}$        | `scatter_add`                            |


$6\times12$ 的 $B_e$ 在任何时刻都不出现，中间量最大只到 $(N_C, 20, 3, 3)$ 的梯度张量——这正是 PA 的常驻量不带 ldof 维的原因。

在步骤 2 与步骤 3 期间，由于当前单核实现基于密集张量 einsum，求积点位移梯度与应力张量需显式物化。在最大规模 $n=64$ 下，$\nabla u$ 理论占用 2.11 GiB，$\sigma$ 理论占用 1.41 GiB，加上 einsum 中间收缩张量，计算期存活的局部张量理论容量合计达 **8~9 GiB**。

### 2. 算子乘多档实测对比（随 n）

三档规模下的实测数据如下：


| n   | $N_C$     | $N_{dof}$ | 工作区峰值净增（GiB） | 稳态常驻净增（MiB） | 首次耗时（s） | 稳态耗时（s） |
| --- | --------- | --------- | ------------ | ----------- | ------- | ------- |
| 32  | 196,608   | 107,811   | 1.3          | **0**       | 1.17    | 1.23    |
| 48  | 663,552   | 352,947   | 3.9          | **0**       | 3.62    | 3.33    |
| 64  | 1,572,864 | 823,875   | 9.0          | **0**       | 8.52    | 8.04    |


在连续调用下，首次调用物化工作区后，后续重复调用满足**稳态零内存增长**。单次算子乘耗时随单元数呈线性增长。

数据来源：`outputs/cache_matvec_continuous_fast_n{32,48,64}.json`。

---



## 四、求解前内存汇总

PA 路线在进入 Krylov 迭代求解器之前的全流程包含网格与空间构建、积分点几何量缓存以及算子乘应用。下面按 `[run.py](run.py)` 的实际调用顺序摊平，剥去 `StageMeter` 计量与 `ElasticityPAOperator` 门面的包装后，链路即为：

```python
# mesh 阶段 (run.py:167)：构建问题、网格、有限元空间与材料
problem, mesh, vs, material = build_problem_space(n)

# mesh 阶段 (run.py:103)：构造分析器，此时不触发任何装配
#   积分阶 q = degree + 3 = 4，在 builders._analyzer_arguments 中定死，不经 cases.toml
analyzer = build_serial_analyzer(vs, problem, material,
                                 degree=1, operator_level="pa", assembly_method="fast")

# cache 阶段 (run.py:119)：构造 PA 算子实例，缓存积分点几何量与 cell2dof
#   返回的是 PartialAssembly 算子而非矩阵，内部机制见第二章
pa_op = analyzer.assemble_stiff_matrix()

# assemble 阶段 (run.py:120-121)：体力右端项与 Dirichlet 边界条件
#   apply_bc 不改写任何数据，只把 pa_op 包进 ConstrainedOperator
load_vector = analyzer.assemble_body_force_vector()
operator, load = analyzer.apply_bc(pa_op, load_vector)

# matvec 阶段 (run.py:319)：无矩阵算子乘应用，本章峰值即出现在这一步
y = pa_op @ x

# 以下两步属 solve 面板，不计入本章「求解前」统计 (run.py:426-434)
#   diag 由 PartialAssembly.diagonal() 闭式给出，cg 每迭代一步调用一次 operator @ x
# diag = analyzer.assemble_operator_diagonal(operator)
# x, info = cg(operator, load, analyzer.prescribed_solution,
#              DiagonalPreconditioner(diag), atol=0.0, rtol=tol, maxit=maxiter)
```

`run.py` 中这条链被拆在 `_build_facade`、`ElasticityPAOperator.assemble` 与各 `measure_*` 三处：拆分是为了让阶段边界与 `VmHWM` 重置对齐，与算子本身的语义无关。

### 1. 内存的累积方式（以 n=64 为例）

PA 的全流程同样严格遵循两条规则：**常驻内存累积，阶段峰值不累积**。

下表以最大规模 $n=64$（157.3 万单元、82.4 万自由度）连续实测记录走完全程。每行满足「起点常驻 + 自身净增 = 阶段峰值」，每行的结束常驻即下一行的起点常驻：


| 阶段      | 起点常驻（GiB） | 自身净增（GiB） | 阶段峰值（GiB） | 结束常驻（GiB） |
| ------- | --------- | --------- | --------- | --------- |
| 网格构建    | 0.6       | 5.6       | 6.2       | 1.3       |
| 积分点几何缓存 | 1.3       | 5.9       | 7.2       | 3.8       |
| 算子乘应用   | 3.8       | 9.0       | **12.8**  | 4.1       |


与 FA 和 EA 全程峰值出现在阶段 1（单刚计算）不同，**PA 的全流程绝对峰值出现在阶段 2（算子乘应用）**。这是因为低阶四面体在 $q=4$ 下包含 20 个求积点，单核 Python 张量收缩同时物化了全网格求积点的梯度张量（净增 9.0 GiB），导致进程瞬时冲高至 **12.8 GiB**，计算完毕后临时张量全部释放，常驻平稳回落至 4.1 GiB。

### 2. 多档汇总

下表汇总三档规模下容量评估直接用到的核心量：


| n   | $N_{dof}$ | 全过程峰值 RSS（GiB） | 准备后常驻 RSS（GiB） | 峰值所在阶段     |
| --- | --------- | -------------- | -------------- | ---------- |
| 32  | 107,811   | 2.4            | 1.3            | 阶段 2：算子乘应用 |
| 48  | 352,947   | 5.9            | 2.3            | 阶段 2：算子乘应用 |
| 64  | 823,875   | 12.8           | 4.1            | 阶段 2：算子乘应用 |


数据来源：`outputs/cache_matvec_continuous_fast_n{32,48,64}.json` 与 `outputs/cache_fast_n{32,48,64}.json`。