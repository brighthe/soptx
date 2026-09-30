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

### 算子结构与阶段划分

PA/QA 层级：

$$
y = \sum_e G_e^{\mathsf T} B_e^{\mathsf T} \left[ D_e \left( B_e \left( G_e x \right) \right) \right]
$$

PA 不形成 $B_e$，而是把它逐积分点拆成三个因子：

$$
B_{e,q} = S\,\Gamma(J_{e,q})\,\hat B_q
$$

$D_e$ 逐积分点是一个标量乘一个常量矩阵：

$$
D_{e,q} = w_q\,\lvert T_e\rvert\,\rho_e\,D
$$

PA 各阶段的产物如下：


| 阶段     | PA 中的对应                                                                                                                                  | 常驻产物                                                                 |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| build  | `ReferenceBasis.build` 在参考单元上求对**参考坐标**的基函数梯度 $\hat B$，只依赖单元形状、$p$、$q$，按进程级缓存                                                            | $\hat B$：$(20, 4, 3)$，约 1.9 KiB，不随 $N_C$ 变化                          |
| setup  | `quadrature_geometry` 算出 $J^{-1}$ 与 $w_q\lvert T_e\rvert$，分别装入 `GeometricFactors` 与 `LinearElasticQFunction`；`ElementRestriction` 给出 $G$ | $J^{-1}$：$(N_C, 20, 3, 3)$；`weighted_measure`：$(N_C, 20)$；`cell2dof` |
| update | `PartialAssembly.update` 只重算 $(N_C, N_Q)$ 个逐点标量，不碰几何因子，也不重新积分                                                                            | 新的 `weighted_coef`：$(N_C, 20)$                                       |
| apply  | gather → $B$ → $\mathcal D$ → $B^{\mathsf T}$ → scatter-add                                                                              | 无                                                                    |


优化中各阶段按 setup →（update → apply × m）× k 循环：每轮先按新密度 update，再在 Krylov 求解中调用 m 次 apply；第一轮的密度在 setup 时随 `coef` 一并传入，不另调 update。

---



## 一、装配前：import 底座与网格构建

本节测量 PA 算子装配前的内存开销，即固定网格下只需支付一次、后续各轮计算可直接复用的部分，与每轮重复支付的算子构建与计算相区分。PA 与 FA 的前置成本存在明确分流：

- **公共基础**：模块导入，以及网格、有限元空间和材料对象的构建（第 1、2 小节）。
- **CSR 符号结构免除**：FA 的 `pattern` 路线必须通过 `build_csr_pattern` 预先建立 CSR 骨架与槽位映射；PA 算子在求积点级进行局部张量收缩，从根本上省去了这一阶段（第 3 小节）。



### 1. import 底座（不随 n）

测量开始前一次导入以下模块，NumPy、SciPy、PyTorch、SymPy 由它们连带加载：

```python
from fealpy.backend import backend_manager as bm
bm.set_backend("numpy")

import fealpy.functionspace
import fealpy.mesh
import fealpy.sparse
import soptx.fem.integrators
import soptx.fem.matrix.csr_pattern
import soptx.materials
import soptx.problems.elasticity
```

静态导入底座产生的常驻内存为 **607 MiB**。该底座由共享依赖共同构成，不随网格规模 $n$ 变化，与 FA 实测底座（606 MiB）及 EA 实测底座（607 MiB）一致。

数据来源：`outputs/cache_fast_n{32,48,64}.json`。

### 2. 网格与空间构建（随 n）

`mesh` 阶段合并测量网格、有限元空间、材料与算子门面的构建，不含任何装配；阶段结束后执行 `gc.collect()` 与 `malloc_trim()`，再读取构建后 RSS。峰值与构建后 RSS 均为进程绝对量，净增以同一进程的 import 底座为基准。


| n   | $N_{dof}$ | 构建期峰值 RSS（GiB） | 构建后 RSS（MiB） | 构建后 RSS 净增（MiB） | 构建耗时（s） |
| --- | --------- | -------------- | ------------ | --------------- | ------- |
| 32  | 107,811   | 1.3            | 705          | 98              | 12      |
| 48  | 352,947   | 3.0            | 914          | 309             | 36      |
| 64  | 823,875   | 6.2            | 1329         | 721             | 86      |


数据来源：主仓库 soptx 的 `experiments/pa_assembly_capability/outputs/cache_fast_n{32,48,64}.json`。

### 3. 与 FA 路线对比（免 CSR 符号结构）

FA 的模式先行装配路线（`fast + pattern`）在进入单刚计算前，必须针对有限元空间调用 `build_csr_pattern`，构建张量级 CSR 骨架及分量块槽位映射，在 $n=32, 48, 64$ 时分别引入 47、215、519 MiB 的常驻开销（FA 报告第二章第 2 小节）。

PA 算子属于无矩阵范式（Matrix-Free），在整个生命周期内不显式组装全局 CSR 矩阵，因而**完全免除 CSR 符号结构的构建与常驻开销**。网格与空间构建完成后，即可直接进入积分点数据缓存与算子计算。

---



## 二、setup：积分点数据计算与缓存（随 n）



### 1. 算子构建与积分点缓存机制

PA 在 setup 阶段不形成 $K_e$，也不做任何缩并，只算出并缓存逐积分点的两类数据：

$$
J_{e,q}^{-1}, \qquad w_q\,\lvert T_e\rvert
$$

- **$J_{e,q}^{-1}$ 是几何因子。** 应变需要基函数对物理坐标的梯度，而 build 阶段的 $\hat B$ 只有对参考坐标的梯度，且全网格共用一份。二者由链式法则联系：

$$
\nabla_x \varphi_i(x_{e,q}) = J_{e,q}^{-\mathsf T}\, \nabla_\xi \hat\varphi_i(\xi_q)
$$

- **$w_q\lvert T_e\rvert$ 是加权测度。** 单纯形上重心坐标求积权重之和为 1，所以乘的是单元体积 $\lvert T_e\rvert$，而不是 $\lvert J_{e,q}\rvert$（四面体上 $\lvert J\rvert = 6\lvert T_e\rvert$）。

setup 同时由 `ElementRestriction` 从 `cell2dof` 构造单元限制算子 $G_e$，与上述数据共同定义全局刚度算子：

$$
K x = \sum_e G_e^{\mathsf T} \sum_q B_{e,q}^{\mathsf T} \left[ \left( w_q\,\lvert T_e\rvert\,\rho_e\,D \right) \left( B_{e,q} \left( G_e x \right) \right) \right]
$$

与 EA 对照：EA 在 setup 把 $\sum_q$ 和 $B$、$D$ 全部缩并成 $K_e$；PA 把这些因子留到 apply，按结合律从右往左逐步作用于向量，不形成单元矩阵。

相关源码：

- 分析器工厂与装配入口：
  - `[src/soptx/fem/analyzers/builders.py](../../src/soptx/fem/analyzers/builders.py)` 中的 `build_serial_analyzer`
  - `[src/soptx/fem/analyzers/lagrange_fem_analyzer.py](../../src/soptx/fem/analyzers/lagrange_fem_analyzer.py)` 中的 `assemble_stiff_matrix`
- PA 算子构建与积分点缓存：
  - `[src/soptx/fem/levels/partial.py](../../src/soptx/fem/levels/partial.py)` 中的 `PartialAssembly.build` 与 `quadrature_geometry`
- 算子内核类：
  - `[src/soptx/fem/kernels/reference_basis.py](../../src/soptx/fem/kernels/reference_basis.py)` 中的 `ReferenceBasis`
  - `[src/soptx/fem/kernels/geometric_factors.py](../../src/soptx/fem/kernels/geometric_factors.py)` 中的 `GeometricFactors`
  - `[src/soptx/fem/kernels/qfunction.py](../../src/soptx/fem/kernels/qfunction.py)` 中的 `LinearElasticQFunction`
  - `[src/soptx/fem/kernels/restriction.py](../../src/soptx/fem/kernels/restriction.py)` 中的 `ElementRestriction`

```python
# 分析器入口: create_level('pa', ...) 分派至 PartialAssembly.build(space, integrator)
analyzer = build_serial_analyzer(vs, problem, material, degree=1, operator_level="pa", assembly_method="fast")
pa_op = analyzer.assemble_stiff_matrix()

# ---- PartialAssembly.build 内部 ----
# build 阶段: 只依赖单元形状、p 与 q, 按进程级缓存, 换网格不重算
reference_basis = ReferenceBasis.build(scalar_space=scalar_space, q=ctx.q)   # hat B: (NQ, ldof, TD)

# setup 阶段: 依赖网格几何, 每张网格算一次
jacobi_inverse, weighted_measure = quadrature_geometry(ctx)   # J^{-1}: (NC, NQ, TD, GD); w_q |T_e|: (NC, NQ)
geometric_factors = GeometricFactors(jacobi_inverse=jacobi_inverse)
qfunction = LinearElasticQFunction(elastic_matrix=D, weighted_measure=weighted_measure,
                                   coef=integrator.coef)   # D 与 rho_e; S (strain_map) 在构造时现生成

# 单元限制 G: 分量布局 (NC, ldof, GD), dof_priority 在这里一次换好, B 不再重排
restriction = ElementRestriction.from_integrator(integrator, space)

# 四个内核拼成 PA 算子, 即 pa_op; 此后 operator @ x 只读它们, 不再触碰网格与空间对象
pa = PartialAssembly(space=space, restriction=restriction, reference_basis=reference_basis,
                     geometric_factors=geometric_factors, qfunction=qfunction)
```



### 2. 峰值 RSS 与常驻 RSS 实测对比（随 n）

下表统计三档规模下 setup 阶段的内存与耗时。各内存列统一换算为 GiB，保留一位小数：


| n   | $N_C$     | $N_{dof}$ | 起点 RSS（GiB） | 峰值净增（GiB） | 阶段峰值 RSS（GiB） | 缓存后常驻 RSS（GiB） | 缓存耗时（s） |
| --- | --------- | --------- | ----------- | --------- | ------------- | -------------- | ------- |
| 32  | 196,608   | 107,811   | 0.7         | 0.7       | 1.4           | 1.0            | 1.35    |
| 48  | 663,552   | 352,947   | 0.9         | 2.6       | 3.5           | 2.0            | 3.38    |
| 64  | 1,572,864 | 823,875   | 1.3         | 5.9       | 7.2           | 3.8            | 8.16    |


表中呈现了两个核心特征：

1. **峰值 RSS 与常驻 RSS 的落差**：计算期间 Jacobi 矩阵、其逆与中间临时量同时存活，形成高于常驻量的阶段瞬时峰值；计算完成后临时量释放，常驻 RSS 回落为后续算子的常驻基线。以 $n=64$ 为例，阶段峰值为 7.2 GiB，缓存后常驻为 3.8 GiB。
2. **随网格规模的线性缩放**：从 $n=32$ 增至 $n=64$，单元数增至 8 倍，峰值净增约为 8.0 倍，耗时约为 6.1 倍。与 EA 相比，PA 虽不形成 $K_e$，三项都更高（$n=64$）：
  - 峰值净增 5.9 对 5.3 GiB；
  - 计算后多出的常驻量 2.5 对 2.1 GiB。PA 的这部分可逐项对上理论值：$J^{-1}$ 2160 MiB、`weighted_measure` 240 MiB、`cell2dof` 144 MiB，合计 2544 MiB，实测 2610 MiB，其余约 66 MiB 为分配器未归还的残留。其中 $J^{-1}$ 按积分点存储，而 P1 四面体上 20 点取值相同，逐单元存储只需约 108 MiB；
  - 耗时 8.16 对 5.0 s。耗时未按子步骤拆分计时，差距来自哪一步本实验不下结论。

数据来源：`outputs/cache_fast_n{32,48,64}.json`。

---



## 三、update：密度更新的内存代价（随 n）



### 1. 密度更新机制

拓扑优化每轮迭代都要按新的密度更新算子。PA 的 update 只改积分点系数 $D_{e,q}$ 中的 $\rho_e$，$G$、$\hat B$、$J^{-1}$ 都不变：

$$
D_{e,q}(\rho) = w_q\,\lvert T_e\rvert\,\rho_e\,D
$$

其中 $w_q\lvert T_e\rvert$ 即 setup 缓存的 `weighted_measure`，$D$ 为常量矩阵，update 只需重算乘积 $w_q\lvert T_e\rvert\rho_e$，即 `weighted_coef`，形状 $(N_C, 20)$。

与 EA 对照：EA 的 update 对象是整块 $K_e$（每单元 144 个数），单元密度下还要一份 $K_e^0$ 作缩放基准；PA 的 update 对象每单元只有 20 个数，缩放基准 `weighted_measure` 本身就是 setup 的常驻量，不需另存备份。逐点密度 $(N_C, N_Q)$ 下 PA 同样只重算这一数组，不像 EA 要退回重新积分。

相关源码：

- 层级复用：
  - `[src/soptx/fem/analyzers/lagrange_fem_analyzer.py](../../src/soptx/fem/analyzers/lagrange_fem_analyzer.py)` 中的 `LagrangeFEMAnalyzer.assemble_stiff_matrix`
- 逐点系数更新：
  - `[src/soptx/fem/levels/partial.py](../../src/soptx/fem/levels/partial.py)` 中的 `PartialAssembly.update`
  - `[src/soptx/fem/kernels/qfunction.py](../../src/soptx/fem/kernels/qfunction.py)` 中的 `LinearElasticQFunction.update`

```python
# 更新入口: 优化主流程每轮调 analyzer.assemble_stiff_matrix(rho), 第二轮起复用已有的 PA 层级
# 几何因子与参考梯度都不随密度变, 只重算 (NC, NQ) 个逐点标量
pa_op.update(new_coef)

# ---- LinearElasticQFunction.update 内部 ----
# setup 时 coef 为 None, weighted_coef 只是 weighted_measure 的别名
if coef is None:
    weighted_coef = weighted_measure
elif coef.shape == (NC, ):                             # 单元密度
    weighted_coef = weighted_measure * coef[:, None]   # 新分配 (NC, NQ)
elif coef.shape == (NC, NQ):                           # 逐点密度
    weighted_coef = weighted_measure * coef
self._coef = coef                                      # 持有本轮密度
self._weighted_coef = weighted_coef
```

按此机制，单元密度下 update 的内存行为可预期为：

- 首轮：断开与 `weighted_measure` 的别名，新分配一份 `weighted_coef`，常驻 +1 份；另加 qfunction 持有的本轮 $\rho$（$N_C$ 个数，$n=64$ 为 12 MiB）；
- 后续轮：新的 `weighted_coef` 先分配、赋值后才释放上一轮的，瞬时峰值 +1 份，常驻不变。

$n=64$ 时一份 `weighted_coef` 为 240 MiB，是 EA 一份 $K_e$（1728 MiB）的 20/144。PA 只有这一种写法，不需像 EA 那样在 scale / inplace / reassemble 之间取舍。

### 2. 密度更新多档实测对比（随 n）

update 面板在同一进程内依次测 setup → update_first → update_rest（共 5 轮），每轮直接调用 `pa_op.update(rho)`，与分析器复用层级时的路径相同。$\rho_e$ 取 $[10^{-3}, 1]$ 上的均匀随机数，在计时之外生成；每轮用前 1000 个单元核对 `weighted_coef` 与 `weighted_measure * rho[:, None]`，相对误差上限 $10^{-12}$。各内存列单位为 MiB：


| n   | $N_C$     | `weighted_coef` 理论（MiB） | 首轮 update 常驻净增 | 后续轮峰值净增 | 首轮耗时（ms） | 稳态耗时（ms） |
| --- | --------- | ----------------------- | -------------- | ------- | -------- | -------- |
| 32  | 196,608   | 30                      | 待测             | 待测      | 待测       | 待测       |
| 48  | 663,552   | 101                     | 待测             | 待测      | 待测       | 待测       |
| 64  | 1,572,864 | 240                     | 待测             | 待测      | 待测       | 待测       |


列的口径（对应 JSON 中 `stages` 的字段）：

- 首轮 update 常驻净增：`update_first` 的 `after_kib - before_kib`，即第一次 update 后常驻的变化；
- 后续轮峰值净增：`update_rest` 的 `net_kib`（峰值减阶段起点），即稳态下每轮的瞬时工作区；
- 首轮 / 稳态耗时：`update_seconds_first` 与 `update_seconds_rest_median`。

数据来源：`outputs/update_fast_n{32,48,64}.json`（待测）。

---



## 四、apply：算子乘的内存机制与执行效率（随 n）

本章测量不经 update，即 $\rho_e = 1$，算子直接作用于 setup 的产物。update 前后参与 apply 的数组形状不变，apply 的工作区与耗时不随 $\rho_e$ 的取值变化，本章结论同样适用于优化迭代中 update 之后的 apply。



### 1. 算子乘机制与理论工作区构成

EA 的 apply 是逐单元稠密小矩阵乘向量。PA 不常驻 $K_e$，apply 按结合律把第二章的各因子从右往左逐步作用于向量：

$$
y = K x = G^{\mathsf T} B^{\mathsf T} \mathcal{D} B G x
$$

其中 $G$ 为单元限制算子，$B$ 由各积分点上的 $B_{e,q} = S\Gamma(J_{e,q})\hat B_q$ 组成，$\mathcal D$ 为块对角算子，每个积分点上的块为 $D_{e,q} = w_q\lvert T_e\rvert\rho_eD$。

相关源码：

- 算子作用入口：
  - `[src/soptx/fem/levels/partial.py](../../src/soptx/fem/levels/partial.py)` 中的 `PartialAssembly.__matmul__`
- 单元限制算子的 gather 与 scatter-add：
  - `[src/soptx/fem/kernels/restriction.py](../../src/soptx/fem/kernels/restriction.py)` 中的 `ElementRestriction.gather` 与 `ElementRestriction.scatter_add`
- $B$ 与 $B^{\mathsf T}$：
  - `[src/soptx/fem/kernels/gradients.py](../../src/soptx/fem/kernels/gradients.py)` 中的 `physical_gradient` 与 `physical_gradient_transpose`
- 逐点本构 $\mathcal D$：
  - `[src/soptx/fem/kernels/qfunction.py](../../src/soptx/fem/kernels/qfunction.py)` 中的 `weighted_stress`

```python
# 算子作用入口: pa_op 为第二章 assemble_stiff_matrix() 返回的 PA 算子
y = pa_op @ x

# ---- PartialAssembly.__matmul__ 内部 ----
# apply 阶段: 每次 MatVec; 形状按 NQ = 20, ldof = 4, GD = TD = 3

# G: 全局 -> 单元
x_E = restriction.gather(x)  # (NC, ldof, GD)

# B 的 hat B 与 Gamma(J): 参考梯度, 再乘 J^{-1} 得物理梯度
grad_u = physical_gradient(x_E, reference_grad=reference_grad,
                           jacobi_inverse=jacobi_inverse)  # (NC, NQ, GD, GD)

# B 的 S, 乘 D_{e,q}, 再乘 S^T: 应变 -> 乘了 w_q |T_e| rho_e 的应力 -> 与 grad u 共轭的量
s_Q = weighted_stress(grad_u, weighted_coef=qfunction.weighted_coef,
                      elastic_matrix=qfunction.elastic_matrix,
                      strain_map=qfunction.strain_map)  # (NC, NQ, GD, GD)

# B^T 的剩余两步: 乘 J^{-T}, 再对 q 求和送回单元自由度
y_E = physical_gradient_transpose(s_Q, reference_grad=reference_grad,
                                  jacobi_inverse=jacobi_inverse)  # (NC, ldof, GD)

# G^T: 单元 -> 全局
y = restriction.scatter_add(y_E)
```



### 2. 算子乘多档实测对比（随 n）

三档规模下的实测数据如下：


| n   | $N_C$     | $N_{dof}$ | 工作区峰值净增（GiB） | 稳态常驻净增（MiB） | 首次耗时（s） | 稳态耗时（s） |
| --- | --------- | --------- | ------------ | ----------- | ------- | ------- |
| 32  | 196,608   | 107,811   | 1.3          | **0**       | 1.17    | 1.23    |
| 48  | 663,552   | 352,947   | 3.9          | **0**       | 3.62    | 3.33    |
| 64  | 1,572,864 | 823,875   | 9.0          | **0**       | 8.52    | 8.04    |


数据来源：`outputs/cache_matvec_continuous_fast_n{32,48,64}.json`。

---



## 五、求解前内存汇总

PA 路线在进入 Krylov 迭代求解器之前的全流程包含网格与空间构建、积分点几何量缓存以及算子乘应用，不含 update。下面按 `[run.py](run.py)` 的实际调用顺序摊平，剥去 `StageMeter` 计量与 `ElasticityPAOperator` 门面的包装后，链路即为：

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


与 FA 和 EA 全程峰值出现在 setup（单刚计算）不同，**PA 的全流程绝对峰值出现在 apply（算子乘）**。这是因为低阶四面体在 $q=4$ 下包含 20 个求积点，单核 Python 张量收缩同时物化了全网格求积点的梯度张量（净增 9.0 GiB），导致进程瞬时冲高至 **12.8 GiB**，计算完毕后临时张量全部释放，常驻平稳回落至 4.1 GiB。

### 2. 多档汇总

下表汇总三档规模下容量评估直接用到的核心量：


| n   | $N_{dof}$ | 全过程峰值 RSS（GiB） | 准备后常驻 RSS（GiB） | 峰值所在阶段    |
| --- | --------- | -------------- | -------------- | --------- |
| 32  | 107,811   | 2.4            | 1.3            | apply：算子乘 |
| 48  | 352,947   | 5.9            | 2.3            | apply：算子乘 |
| 64  | 823,875   | 12.8           | 4.1            | apply：算子乘 |


数据来源：`outputs/cache_matvec_continuous_fast_n{32,48,64}.json` 与 `outputs/cache_fast_n{32,48,64}.json`。