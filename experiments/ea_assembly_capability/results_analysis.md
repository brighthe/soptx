# EA 单元装配内存机制与容量能力

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


| 网格  | 模型                                        | 有限元次数 $p$ | 积分参数 $q$ |
| --- | ----------------------------------------- | --------- | -------- |
| 三角形 | `ExponentialSineManufacturedElasticity2D` | 可选，默认 1   | $p+3$    |
| 四边形 | `ExponentialSineManufacturedElasticity2D` | 可选，默认 1   | $p+1$    |
| 四面体 | `DivergenceFreePolynomialElasticity3D`    | 可选，默认 1   | $p+3$    |
| 六面体（默认） | `DivergenceFreePolynomialElasticity3D`    | 可选，默认 1   | $p+1$    |


两种 EA：

- **标准 EA**：逐单元保存 $K_e$，形状 $(N_C, \mathrm{LDOF}, \mathrm{LDOF})$。
- **共享参考 EA**：每个平移类保存一份参考单元矩阵 $K_k^0$，形状 $(N_k, \mathrm{LDOF}, \mathrm{LDOF})$，另逐单元保存 $s_e = E(\rho_e)/E_0$，apply 时按 $K_e = s_e K^0_{k(e)}$ 缩放。

### 算子结构与阶段划分

两种 EA 算子：

$$
\text{标准 EA：}\ y = \sum_e G_e^{\mathsf T} K_e\, G_e x,
\qquad
\text{共享参考 EA：}\ y = \sum_{k=1}^{N_k} \sum_{e \in \mathcal C_k} s_e\, G_e^{\mathsf T} K_k^0\, G_e x
$$

其中 $N_C$ 为单元数，$\mathrm{LDOF}$ 为单元自由度数（$p=1$ 时三角形 6、四边形 8、四面体 12、六面体 24），$G_e$ 为由 `cell2dof` 第 $e$ 行确定的单元限制算子（全局 → 单元 $e$），$K_e$ 为单元 $e$ 的 $\mathrm{LDOF} \times \mathrm{LDOF}$ 稠密刚度矩阵。只差平移的单元归为一个平移类，$N_k$ 为平移类数，$\mathcal C_k$ 为第 $k$ 类的单元集合，$K_k^0$ 为该类的参考单元矩阵（$s_e = 1$ 时的单元矩阵）。`from_box` 网格的 $N_k$ 由剖分方式决定，与 $n$、$p$ 无关：四边形、六面体为 1，三角形为 2（每个正方形剖成两个朝向相反的三角形），四面体为 6（每个立方体剖成 6 个四面体）。

两种 EA 各阶段的产物如下：


| 阶段     | 标准 EA                                                                                                                                                                                                            | 共享参考 EA                                                                                                                  |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| build  | 两者相同：与网格规模无关的张量 $\Lambda$（代码中为 `S`）由 `fetch_fast_assembly` 将基函数导数与求积权重缩并得到，与 setup 的几何量在同一次调用中算出；常驻 $\Lambda$，KiB 量级                                                                                             | 同左                                                                                                                       |
| setup  | 逐单元算出几何量（单纯形为 `grad_lambda` 与 `cm`，四边形、六面体为 `invJT` 与 `cm`），与 $\Lambda$、材料、密度一起积出全部 $K_e$；`ElementRestriction` 给出 $G_e$。常驻 $K_e$ $(N_C, \mathrm{LDOF}, \mathrm{LDOF})$、`cell2dof`；几何量在 `keep_data` 下也常驻，供 `reassemble` 路线重新积分，非 apply 所需 | 按 `from_box` 的单元编号约定取平移类，$k(e) = e \bmod N_k$，不比对几何；每类只对一个代表单元积出 $K_k^0$；`ElementRestriction` 给出 $G_e$。常驻 $K_k^0$ $(N_k, \mathrm{LDOF}, \mathrm{LDOF})$、`cell2dof` 与 $s_e$，$k(e)$ 由编号现算、不常驻 |
| update | 由 $K_e^0$ 逐单元缩放后写回 $K_e$，每轮写 $N_C\, \mathrm{LDOF}^2$ 个数                                                                                                                                                                        | 只写回 $s_e$，每轮写 $N_C$ 个数                                                                                                   |
| apply  | gather → 逐单元 $K_e x_e$ → scatter-add                                                                                                                                                                             | gather → 逐类 $K_k^0 x_e$（一次矩阵乘）→ 乘 $s_e$ → scatter-add                                                                    |


优化中各阶段按 setup →（update → apply × $N_{it}$）× $N_{opt}$ 循环：每轮先按新密度 update，再在 Krylov 求解中调用 $N_{it}$ 次 apply；第一轮的密度在 setup 时随 `coef` 一并传入，不另调 update。

---

## 一、装配前：import 底座与网格构建

本节测量 EA 算子装配前的内存开销，即模块导入与网格、空间、材料对象的构建。它们先于算子的四个阶段，固定网格下只付一次，与 setup 阶段的单刚计算相区分。EA 与 FA 的前置成本存在明确分流：

- **公共基础**：模块导入，以及网格、有限元空间和材料对象的构建。
- **CSR 符号结构免除**：FA 的 `pattern` 路线必须通过 `build_csr_pattern` 预先建立 CSR 骨架与槽位映射；EA 算子直接在局部单元级进行矩阵向量乘并累加，从根本上省去了这一阶段。

### 1. import 底座（不随 n）

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

静态导入底座产生的常驻内存为 **607 MiB**。该底座由共享依赖共同构成，不随网格规模 $n$ 变化，与 FA 实测底座（606 MiB）完全一致。

数据来源：`outputs/mesh_build_staged_n32.json`。

### 2. 网格与空间构建（随 n）

`mesh` 阶段合并测量网格、有限元空间、材料与算子门面的构建，不含任何装配；阶段结束后执行 `gc.collect()` 与 `malloc_trim()`，再读取构建后 RSS。峰值与构建后 RSS 均为进程绝对量，净增以同一进程的 import 底座为基准。


| n   | $N_{dof}$ | 构建期峰值 RSS（GiB） | 构建后 RSS（MiB） | 构建后 RSS 净增（MiB） | 构建耗时（s） |
| --- | --------- | -------------- | ------------ | --------------- | ------- |
| 32  | 107,811   | 1.3            | 705          | 97              | 11      |
| 48  | 352,947   | 3.0            | 916          | 308             | 36      |
| 64  | 823,875   | 6.2            | 1329         | 721             | 86      |


数据来源：主仓库 soptx 的 `experiments/ea_assembly_capability/outputs/cache_fast_n{32,48,64}.json`。

### 3. 与 FA 路线对比（免 CSR 符号结构）

FA 的模式先行装配路线（`fast + pattern`）在进入单刚计算前，必须针对有限元空间调用 `build_csr_pattern`，构建张量级 CSR 骨架及分量块槽位映射，在 $n=32, 48, 64$ 时分别引入 47、215、519 MiB 的常驻开销（FA 报告第二章第 2 小节）。

EA 算子属于广义 Matrix-Free：不形成全局稀疏矩阵，但保存稠密单元矩阵。它在整个生命周期内不显式组装全局 CSR 矩阵，因而**完全免除 CSR 符号结构的构建与常驻开销**。网格与空间构建完成后，即可直接进入 setup 阶段的单刚计算与缓存。

---

## 二、setup：单刚计算与缓存（随 n）

### 1. 算子构建与单刚缓存机制

两种 EA 的单元矩阵积分式相同，区别在积分对象与密度的去向：

$$
\begin{aligned}
\text{标准 EA：}\quad & K_e = s_e \int_{T_e} B_e^{\mathsf T} D^0 B_e \,\mathrm{d}x, && e = 1, \dots, N_C \\
\text{共享参考 EA：}\quad & K_k^0 = \int_{T_{e_k}} B_{e_k}^{\mathsf T} D^0 B_{e_k} \,\mathrm{d}x, && k = 1, \dots, N_k
\end{aligned}
$$

其中 $T_e$ 为单元 $e$ 所占区域，$B_e$ 为其应变–位移矩阵，$D^0$ 为实体材料的本构矩阵，$e_k$ 为第 $k$ 个平移类的代表单元；积分按"问题与执行方式"表中的积分参数 $q$ 取求积公式。同类单元只差平移，$B_e$ 平移后相同，故每类积一次即可。

|  | 标准 EA | 共享参考 EA |
| --- | --- | --- |
| 积分对象 | 全部 $N_C$ 个单元 | $N_k$ 个代表单元 |
| 密度 $s_e$ | 积进 $K_e$ | 单独存储，不参与积分 |
| 单元映射数据（`grad_lambda`/`invJT`、`cm`） | 全部单元，`keep_data` 下常驻 | 仅代表单元 |
| 常驻产物 | $K_e$ $(N_C, \mathrm{LDOF}, \mathrm{LDOF})$ | $K_k^0$ $(N_k, \mathrm{LDOF}, \mathrm{LDOF})$、$s_e$ |

平移类按 `from_box` 的单元编号约定取：每个格子（正方形或立方体）按固定模板剖出 $N_k$ 个单元并连续编号，同一模板的单元只差平移、局部顶点顺序也相同，故 $k(e) = e \bmod N_k$，前 $N_k$ 个单元即各类代表。该约定只对 `from_box` 生成、未经重编号的网格成立，由 [`tests/unit/test_structured_box.py`](../../tests/unit/test_structured_box.py) 守住。两种 EA 都由 `ElementRestriction` 造出 $G_e$ 并常驻 `cell2dof`。

相关源码（分析器入口只接标准 EA；共享参考 EA 不进层级注册表，由调用方显式构造，复用其中的积分子与 `ElementRestriction`）：

- 分析器工厂与装配入口：
  - [`src/soptx/fem/analyzers/builders.py`](../../src/soptx/fem/analyzers/builders.py) 中的 `build_serial_analyzer`
  - [`src/soptx/fem/analyzers/lagrange_fem_analyzer.py`](../../src/soptx/fem/analyzers/lagrange_fem_analyzer.py) 中的 `assemble_stiff_matrix`
- EA 算子构建与单刚缓存：
  - [`src/soptx/fem/levels/element.py`](../../src/soptx/fem/levels/element.py) 中的 `ElementAssembly.build`
- 共享参考 EA 算子：
  - [`src/soptx/fem/levels/shared_reference.py`](../../src/soptx/fem/levels/shared_reference.py) 中的 `SharedReferenceElementAssembly`
  - 代表单元积分：[`src/soptx/fem/integrators/linear_elastic_integrator.py`](../../src/soptx/fem/integrators/linear_elastic_integrator.py) 中的 `LinearElasticIntegrator.assembly`（`standard` / `voigt` / `fast` 均按 `index` 截取）
- 单元限制算子：
  - [`src/soptx/fem/kernels/restriction.py`](../../src/soptx/fem/kernels/restriction.py) 中的 `ElementRestriction`

**标准 EA**（适用网格：任意网格，包括一般非结构化网格；逐单元积分，不要求单元之间有平移关系）：

```python
# 网格: 一般非结构化网格, node、cell 可来自任意网格生成器; 默认六面体, 其余三种换成 TriangleMesh、QuadrangleMesh、TetrahedronMesh
mesh = HexahedronMesh(node, cell)
vs = TensorFunctionSpace(LagrangeFESpace(mesh, p=p, ctype='C'), shape=(-1, GD))

# 分析器入口: create_level('ea', ...) 分派至 ElementAssembly.build(space, integrator)
analyzer = build_serial_analyzer(vs, problem, material, degree=p, operator_level="ea", assembly_method="fast")
ea_op = analyzer.assemble_stiff_matrix()

# ---- ElementAssembly.build 内部 ----
# setup 阶段: 依赖网格几何与拓扑, 每张网格算一次
# 单元矩阵 K_e: build 与 setup 在 fetch_fast_assembly 的同一次调用中完成, 缩并张量 S 属于 build,
# invJT、cm 属于 setup (单纯形为 grad_lambda、cm; keep_data 下常驻在 integrator 上), 再与材料、密度一起积进 K_e;
# K_e 的初值含 s_e (即 coef), 按阶段属于第一次 update
K_e = integrator.assembly(space)  # (NC, ldof * GD, ldof * GD) = (NC, LDOF, LDOF)

# 单元限制 G: 扁平布局 (NC, ldof * GD), 只依赖网格拓扑 (cell2dof), 单元内自由度顺序与 K_e 的行列一致
g = ElementRestriction.from_integrator(integrator, space, layout='flat')

# G 与 K_e 拼成 EA 算子, 即 ea_op
ea = ElementAssembly(space, restriction=g, element_matrices=K_e)
```

**共享参考 EA**（适用网格：等距结构化网格；只对代表单元积分，要求同类单元之间只差平移）：

```python
# 网格: from_box 结构化网格; 默认六面体, 其余三种换成 TriangleMesh、QuadrangleMesh、TetrahedronMesh.from_box
mesh = HexahedronMesh.from_box(box, nx=n, ny=n, nz=n)
space = TensorFunctionSpace(LagrangeFESpace(mesh, p=p, ctype='C'), shape=(-1, GD))

# 本实验绕开分析器直接构造, 不生成 _cached_ke0
# 平移类: from_box 把每个格子剖出的 N_k 个单元连续编号 (四面体 6, 三角形 2, 四边形、六面体 1),
# 故 k(e) = e % N_k, 前 N_k 个单元即各类代表; 不比对几何, k(e) 用时由编号现算, 不常驻
reps = bm.arange(N_k)  # (N_k, ), 六面体 N_k = 1, 即只取单元 0

# 参考单元矩阵 K_k^0: 只对 N_k 个代表单元积分, coef=None 即 s_e = 1
# 积分子的 standard / voigt / fast 均按 index 截取; 这里与标准 EA 同取 fast, 换另两种只差舍入
K0 = LinearElasticIntegrator(material, coef=None, q=q, index=reps, method='fast').assembly(space)  # (N_k, LDOF, LDOF)

# 单元限制 G: 与标准 EA 相同, 取全体单元 (积分子不带 index, 这里只用到其 cell2dof)
g = ElementRestriction.from_integrator(LinearElasticIntegrator(material, q=q), space, layout='flat')

# G、K_k^0 与 s_e 拼成共享参考 EA 算子; N_k 即 K0 的第一维, k(e) 由它按编号算出
ea = SharedReferenceElementAssembly(space, restriction=g, reference_matrices=K0, scale=s)
```

### 2. 峰值 RSS 与常驻 RSS 实测对比（随 n）

下表统计三档规模下 setup 阶段的内存与耗时。各内存列统一换算为 GiB，保留一位小数：


| n   | $N_C$     | $N_{dof}$ | `cell2dof`（MiB） | 起点 RSS（GiB） | 峰值净增（GiB） | 阶段峰值 RSS（GiB） | 缓存后常驻 RSS（GiB） | 缓存耗时（s） |
| --- | --------- | --------- | --------------- | ----------- | --------- | ------------- | -------------- | ------- |
| 32  | 196,608   | 107,811   | 18              | 0.7         | 0.7       | 1.4           | 1.0            | 0.69    |
| 48  | 663,552   | 352,947   | 61              | 0.9         | 2.3       | 3.2           | 1.9            | 2.1     |
| 64  | 1,572,864 | 823,875   | 144             | 1.3         | 5.3       | 6.6           | 3.4            | 5.0     |


表中呈现了两个核心特征：

1. **峰值 RSS 与常驻 RSS 的落差**：计算期间由于中间梯度缩并分块与刚度分块同时存活，形成了高于常驻量的阶段瞬时峰值（机制与 FA 完全同源，详见 FA 报告第二章第 3 小节）；计算完成后临时分块释放，常驻 RSS 回落并沉淀为后续算子的常驻基线。以 $n=64$ 为例，阶段峰值为 6.6 GiB，缓存后常驻为 3.4 GiB。
2. **随网格规模的线性缩放**：从 $n=32$ 增至 $n=64$，单元数增至 8 倍，峰值净增约为 7.4 倍，耗时约为 7.3 倍，峰值净增近似随单元数 $N_C$ 线性增长，增幅略低于单元数本身。与 FA 相比，两者的峰值净增几乎相同（$n=64$ 均为 5.3 GiB），但计算后多出的常驻量 EA 更大：EA 为 2.1 GiB，FA 为 1.75 GiB，差额约 376 MiB。其中 `cell2dof` 映射占 144 MiB；`fetch_fast_assembly` 的几何缓存 `grad_lambda` 与 `cm` 占约 156 MiB（EA 经分析器开启 `keep_data`，FA 阶段 1 用裸 integrator，不保留）；其余约 76 MiB 为分配器未归还的残留。

数据来源：`outputs/cache_fast_n{32,48,64}.json`。

---

## 三、update：密度更新的内存代价（随 n）

### 1. 密度更新机制

拓扑优化每轮迭代都要按新的密度更新算子，两种 EA 的 update 都不改 $G_e$。前提是本构只差一个逐单元标量，即 $D_e = s_e D^0$：密度为单元常数、只插值杨氏模量、泊松比固定。此时单元矩阵对 $s_e$ 线性，两种 EA 的 update 分别为：

$$
\begin{aligned}
\text{标准 EA：}\quad & K_e \leftarrow s_e\, K_e^0, && e = 1, \dots, N_C \\
\text{共享参考 EA：}\quad & s_e \leftarrow E(\rho_e)/E_0, && e = 1, \dots, N_C
\end{aligned}
$$

其中 $K_e^0$ 为单元 $e$ 的实体单元矩阵（$s_e = 1$ 时的 $K_e$），同类单元的 $K_e^0$ 均等于 $K_{k(e)}^0$。逐点密度、多分辨率与泊松比插值不满足该前提：标准 EA 只能按新系数重新积分，共享参考 EA 不适用。

|  | 标准 EA | 共享参考 EA |
| --- | --- | --- |
| 缩放基准 | 逐单元 $K_e^0$ $(N_C, \mathrm{LDOF}, \mathrm{LDOF})$，须在 $K_e$ 之外另存 | $K_k^0$ $(N_k, \mathrm{LDOF}, \mathrm{LDOF})$，setup 已常驻 |
| 每轮写入 | $N_C\, \mathrm{LDOF}^2$ 个数 | $N_C$ 个数 |
| 前提不满足时 | 重新积分，峰值与 setup 相当 | 不适用 |

标准 EA 的 $K_e^0$ 在分析器中即实体单元矩阵缓存 `_cached_ke0`，灵敏度也用这一份，EA 只持有引用。共享参考 EA 要在优化流程中保住内存上的节省，灵敏度也须改用 $K_{k(e)}^0$，不再生成 `_cached_ke0`。

相关源码（共享参考 EA 尚未接入分析器与灵敏度，除其 `update` 外下列均为标准 EA 的路径）：

- 单元矩阵更新：
  - [`src/soptx/fem/levels/element.py`](../../src/soptx/fem/levels/element.py) 中的 `ElementAssembly.update` 与 `ElementAssembly.set_element_matrices`
  - [`src/soptx/fem/levels/shared_reference.py`](../../src/soptx/fem/levels/shared_reference.py) 中的 `SharedReferenceElementAssembly.update`（只换 $s_e$，$K_k^0$ 不动）
- 层级复用与 $K_e^0$ 共享：
  - [`src/soptx/fem/analyzers/lagrange_fem_analyzer.py`](../../src/soptx/fem/analyzers/lagrange_fem_analyzer.py) 中的 `LagrangeFEMAnalyzer.assemble_stiff_matrix`
- 按新密度重新积分：
  - [`src/soptx/fem/integrators/linear_elastic_integrator.py`](../../src/soptx/fem/integrators/linear_elastic_integrator.py) 中的 `LinearElasticIntegrator.coef`与 `LinearElasticIntegrator.assembly`

```python
# 更新入口: 优化主流程每轮调 analyzer.assemble_stiff_matrix(rho), 第二轮起复用已有的 EA 层级
# 单元密度 (NC, ) 下 K_e 对 coef = s_e 线性, EA 由 K_e^0 逐单元缩放即可, 不必重新积分
ea_op.update(new_coef)  # K_e <- new_coef_e * K_e^0, 原地写回, 不分配新的 K_e

# ---- ElementAssembly.update 内部 ----
# K_e^0 为分析器的实体单元矩阵缓存 _cached_ke0 (敏度也用这一份), EA 只持有引用
if reference is not None and coef.shape == (NC, ):
    bm.multiply(reference, coef[:, None, None], out=self._element_matrices)
else:
    # 逐点密度、多分辨率、泊松比插值等其余形状: 按新系数重新积分
    self._integrator.coef = coef
    self._element_matrices = self._integrator.assembly(space)  # 每轮重付 setup 的积分峰值
```

### 2. 密度更新多档实测对比（随 n）

update 面板在同一进程内依次测 setup → keep_K0 → update_first → update_rest（共 5 轮），三种写法各起一个子进程，互不干扰峰值：

- `scale`：`K_e0` 取 setup 所得 $K_e$ 的别名，每轮 `ea_op.set_element_matrices(rho[:, None, None] * K_e0)`；
- `inplace`：`K_e0` 为独立副本，每轮 `np.multiply(K_e0, rho[:, None, None], out=ea_op.element_matrices)` 原地写回；
- `reassemble`：不存 $K_e^0$，每轮 `integrator.coef = rho` 后 `ea_op.set_element_matrices(integrator.assembly(space))`。

本实验绕过材料插值，把 $[10^{-3}, 1]$ 上的均匀随机数（上述代码中的 `rho`）直接作为 $E(\rho_e)/E_0$ 传入，在计时之外生成；每轮用前 1000 个单元核对 $\frac{E(\rho_e)}{E_0} K_e^0$，相对误差上限 $10^{-12}$。各内存列单位为 MiB：


| n   | $N_C$     | $K_e$ 理论（MiB） | mode         | 备份 $K_e^0$ 常驻净增 | 首轮 update 常驻净增 | 后续轮峰值净增 | 首轮耗时（ms） | 稳态耗时（ms） |
| --- | --------- | ------------- | ------------ | --------------- | -------------- | ------- | -------- | -------- |
| 32  | 196,608   | 216           | `scale`      | 0               | 216            | 215     | 29.5     | 30.8     |
| 32  | 196,608   | 216           | `inplace`    | 216             | 0              | 0       | 21.2     | 20.9     |
| 32  | 196,608   | 216           | `reassemble` | —               | 0              | 701     | 403      | 359      |
| 48  | 663,552   | 729           | `scale`      | 0               | 729            | 728     | 298      | 102      |
| 48  | 663,552   | 729           | `inplace`    | 729             | 0              | 0       | 70.5     | 70.1     |
| 48  | 663,552   | 729           | `reassemble` | —               | 0              | 2,372   | 1,482    | 1,390    |
| 64  | 1,572,864 | 1,728         | `scale`      | 0               | 1,728          | 1,727   | 248      | 250      |
| 64  | 1,572,864 | 1,728         | `inplace`    | 1,728           | 0              | 0       | 172      | 171      |
| 64  | 1,572,864 | 1,728         | `reassemble` | —               | 0              | 5,614   | 3,597    | 3,326    |


列的口径（对应 JSON 中 `stages` 的字段）：

- 备份 $K_e^0$ 常驻净增：`keep_K0` 的 `after_kib - before_kib`；`reassemble` 无此阶段；
- 首轮 update 常驻净增：`update_first` 的 `after_kib - before_kib`，即第一次 update 后常驻的变化；
- 后续轮峰值净增：`update_rest` 的 `net_kib`（峰值减阶段起点），即稳态下每轮的瞬时工作区；
- 首轮 / 稳态耗时：`update_seconds_first` 与 `update_seconds_rest_median`。

三种写法的实测与机制预期逐项吻合：`scale` 与 `inplace` 的内存各项均为一份 $K_e$ 的整数倍（误差 < 1 MiB），`reassemble` 的峰值与 setup 相当，均随 $N_C$ 线性增长；三档规模下核对的相对误差均不超过 $3 \times 10^{-16}$：

- `scale`：`K_e0 = K_e` 只是别名，备份本身不占内存（0）；首轮 `rho * K_e0` 新分配一份 $K_e$，此后 $K_e^0$ 与 $K_e$ 分成两块，常驻 +1 份 $K_e$；稳态每轮乘法先分配新 $K_e$、赋值后才释放上一轮的，瞬时峰值再 +1 份 $K_e$。$n=64$ 时进程峰值 6956 MiB，已超过 setup 峰值 6748 MiB。
- `inplace`：备份即复制一份 $K_e$（+1），此后原地写回，常驻与瞬时均为 0，进程峰值保持在 setup 峰值。耗时也是三者中最低，稳态耗时与 $K_e$ 大小成正比（约 0.1 ms/MiB，即读 $K_e^0$、写 $K_e$ 共约 20 GB/s）。`scale` 稳态慢约 45%，多出的是每轮新分配内存的缺页代价；$n=48$ 时其前两轮约 300 ms，第三轮起降到 102 ms，为分配器复用已释放内存之后的稳态。
- `reassemble`：不存 $K_e^0$，常驻不变（0）；但每轮峰值净增与第二章 setup 相当（$n=32, 48, 64$ 下 setup 为 733、2372、5418 MiB），且叠在已常驻的旧 $K_e$ 之上，$n=64$ 时进程峰值达 9114 MiB（8.9 GiB），比 setup 峰值高 2.3 GiB。耗时为 `inplace` 的约 17–20 倍。

结论：单元密度下 EA 的 update 应采用 `inplace` 写法，代价是常驻多一份 $K_e^0$（$n=64$ 为 1.7 GiB），换来每轮零额外内存与最低耗时；`scale` 的写法最直观，但常驻与 `inplace` 相同、每轮还多一份 $K_e$ 的瞬时峰值，无任何优势；`reassemble` 省下 $K_e^0$ 的常驻，却让每轮峰值高于 setup，只在逐点密度下不可避免。`ElementAssembly.update` 已按此实现：单元密度下走 `inplace`，其余形状走 `reassemble`；且 $K_e^0$ 与柔顺度、应力敏度共用分析器的 `_cached_ke0`，优化流程中本就常驻，`inplace` 的备份代价在这里不再另计。

数据来源：`outputs/update_{scale,inplace,reassemble}_fast_n{32,48,64}.json`。

---

## 四、apply：算子乘的内存机制与执行效率（随 n）

本章测量不经 update，即 $E(\rho_e)/E_0 = 1$，算子直接作用于 setup 的产物。update 前后参与 apply 的数组形状不变，apply 的工作区与耗时不随 $\rho_e$ 的取值变化，本章结论同样适用于优化迭代中 update 之后的 apply。

### 1. 算子乘机制与理论工作区构成

FA 在单刚之后还要把 $K_e$ 求和成全局稀疏矩阵，apply 是稀疏矩阵乘向量。EA 舍弃了全局总刚的组装与存储，apply 直接逐单元作用：

$$
y = K x = \sum_e G_e^{\mathsf T} K_e\, G_e x
$$

其中 $G_e$ 为由 `cell2dof` 确定的单元限制算子（$x_e = G_e x$），$\sum_e G_e^{\mathsf T}$ 为对应的 scatter-add。

相关源码：

- 算子作用入口：
  - [`src/soptx/fem/levels/element.py`](../../src/soptx/fem/levels/element.py) 中的 `ElementAssembly.__matmul__`
- 单元限制算子的 gather 与 scatter-add：
  - [`src/soptx/fem/kernels/restriction.py`](../../src/soptx/fem/kernels/restriction.py) 中的 `ElementRestriction.gather` 与 `ElementRestriction.scatter_add`

```python
# 算子作用入口: ea_op 为第二章 assemble_stiff_matrix() 返回的 EA 算子
y = ea_op @ x

# ---- ElementAssembly.__matmul__ 内部 ----
# apply 阶段: 每次 MatVec

# G: 全局 -> 单元
x_E = g.gather(x)  # (NC, ldof * GD[, NB])

# K_e: 逐单元小矩阵乘向量
y_E = bm.einsum('cij, cj... -> ci...', K_e, x_E)  # (NC, ldof * GD[, NB])

# G^T: 单元 -> 全局
y = g.scatter_add(y_E)
```

EA 保留了单元内的预计算，却放弃了单元间的合并：FA 的 scatter-add 把共享节点上的多份贡献合成一个非零元，EA 则逐单元各存一份。参与 apply 的常驻算子数据因此高于 FA，$n=32, 48, 64$ 时 EA 的 $K_e$ 为 216、729、1728 MiB，FA 组装后的 CSR（`values` 与 `col`）为 71、237、558 MiB，约为其 3 倍（FA 报告第四章）。

### 2. 算子乘多档实测对比（随 n）

三档规模下的实测数据如下：


| n   | $N_C$     | $N_{dof}$ | 工作区峰值净增（MiB） | 稳态常驻净增（MiB） | 首次耗时（ms） | 稳态耗时（ms） |
| --- | --------- | --------- | ------------ | ----------- | -------- | -------- |
| 32  | 196,608   | 107,811   | 35.9         | **0**       | 22       | 20.3     |
| 48  | 663,552   | 352,947   | 181.5        | **0**       | 62       | 63.7     |
| 64  | 1,572,864 | 823,875   | 430.8        | **0**       | 153      | 154.3    |


数据来源：`outputs/cache_matvec_continuous_fast_n{32,48,64}.json`。

---

## 五、求解前内存汇总

EA 路线在进入 Krylov 迭代求解器（如 Jacobi-PCG）之前，全流程包含网格与空间构建、setup（单刚与自由度映射缓存）以及 apply（算子乘），不含 update，执行顺序如下：

```python
# 1. 构建问题、网格、有限元空间与材料
problem, mesh, vs, material = build_problem_space(n)

# 2. setup: 构造 EA 算子实例并缓存单刚 K_e 与自由度映射 cell2dof
analyzer = build_serial_analyzer(vs, problem, material, degree=1, operator_level="ea", assembly_method="fast")
ea_op = analyzer.assemble_stiff_matrix()

# 3. apply: 无矩阵算子乘
y = ea_op @ x
```

### 1. 内存的累积方式（以 n=64 为例）

与 FA 一致，EA 的全流程同样严格遵循两条规则：**常驻内存累积，阶段峰值不累积**。

下表以最大规模 $n=64$（157.3 万单元、82.4 万自由度）连续实测记录走完全程，三个阶段按执行顺序排列。每行满足「起点常驻 + 自身净增 = 阶段峰值」，每行的结束常驻即下一行的起点常驻。


| 阶段    | 起点常驻（GiB） | 自身净增（GiB） | 阶段峰值（GiB） | 结束常驻（GiB） |
| ----- | --------- | --------- | --------- | --------- |
| 网格构建  | 0.6       | 5.6       | 6.2       | 1.3       |
| setup | 1.3       | 5.3       | **6.6**   | 3.4       |
| apply | 3.4       | 0.4       | 3.8       | 3.4       |


### 2. 多档汇总

下表汇总三档规模下容量评估直接用到的两个核心量：


| n   | $N_{dof}$ | 全过程峰值 RSS（GiB） | 准备后常驻 RSS（GiB） | 峰值所在阶段     |
| --- | --------- | -------------- | -------------- | ---------- |
| 32  | 107,811   | 1.4            | 1.0            | setup：单刚缓存 |
| 48  | 352,947   | 3.2            | 1.9            | setup：单刚缓存 |
| 64  | 823,875   | 6.6            | 3.4            | setup：单刚缓存 |


数据来源： `outputs/mesh_build_staged_n{32,48,64}.json`、`outputs/cache_fast_n{32,48,64}.json` 与 `outputs/cache_matvec_continuous_fast_n{32,48,64}.json`。