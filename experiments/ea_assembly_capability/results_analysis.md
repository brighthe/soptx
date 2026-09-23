# EA 单元装配无矩阵算子的内存机制与容量能力

## 实验设置与测量口径

### 运行环境


| 项目     | 配置                                                           |
| ------ | ------------------------------------------------------------ |
| 宿主系统   | Windows 11 Pro，物理内存 64 GB                                    |
| 实验系统   | WSL2 Ubuntu 24.04 LTS，内核 `6.18.33.2-microsoft-standard-WSL2` |
| CPU    | Intel Core i9-14900KF，WSL 可见 32 个逻辑核                         |
| WSL 内存 | 配置上限 `memory=48GB`，系统报告总内存约 47 GiB（`MemTotal`），swap 12 GiB   |
| 软件版本   | Python 3.12.13，numpy 2.5.1，scipy 1.18.0，torch 2.13.0+cu130   |


GPU 为 NVIDIA GeForce RTX 5080，显存 16 GB；相关结果以 `_cuda` 标记，不纳入本文 CPU 容量分析。

### 问题与执行方式

采用三维线弹性制造解问题，使用 FEALPy 的 `TetrahedronMesh.from_box` 构建四节点四面体网格，采用 $p=1$ 的 Lagrange 有限元，积分参数取默认值 $q=p+3=4$。三个坐标方向均划分为 $n$ 份，单元数为 $N_C = 6n^3$，位移自由度数为 $N_{dof} = 3(n+1)^3$。实验采用单进程 CPU 执行方式，每个数据点在独立进程中测量。

### 测量口径

- **内存统计**：以 `VmRSS` 记录常驻内存，以逐阶段重置的 `VmHWM` 记录阶段峰值；阶段净增为阶段峰值与阶段起始 `VmRSS` 之差，全程绝对峰值取各阶段峰值的最大值。
- **表头约定**：凡标「峰值」的列取 `VmHWM`，其余 RSS 列一律为 `VmRSS` 常驻读数，列名中的「构建后」「计算后」「缓存后」「结束」指该读数所处的时刻；常驻读数均在对象未释放、未调用垃圾回收或内存归还的状态下读取。
- **容量预算**：以进程绝对峰值 RSS 不超过 45 GiB 为容量评估条件。
- **数值精度**：本文关注内存量级与占比，不追求逐字节精确。内存数值统一按「不小于 1 GiB 者保留一位小数、小于 1 GiB 者取整数 MiB」给出，耗时统一保留两位有效数字；同一表中各列为独立读数，舍入后按「起点 + 净增 = 峰值」相加可有 0.1 GiB 的偏差。

---



## 一、装配前：import 底座与网格构建

本节测量 EA 算子装配前的内存开销，即固定网格下只需支付一次、后续各轮计算可直接复用的部分，与每轮重复支付的单刚计算相区分。EA 与 FA 的前置成本存在明确分流：

- **公共基础**：模块导入，以及网格、有限元空间和材料对象的构建（第 1、2 小节）。
- **CSR 符号结构免除**：FA 的 `pattern` 路线必须通过 `build_csr_pattern` 预先建立 CSR 骨架与槽位映射；EA 算子直接在局部单元级进行矩阵向量乘并累加，从根本上省去了这一阶段（第 3 小节）。



### 1. import 底座（不随 n）

`run.py` 显式加载 FEALPy、SOPTX 及底层依赖栈（NumPy、SciPy、PyTorch、SymPy），静态导入底座产生的常驻内存为 **607 MiB**。该底座由共享依赖共同构成，不随网格规模 $n$ 变化，与 FA 实测底座（606 MiB）完全一致。

数据来源：`outputs/mesh_build_staged_n32.json`。

### 2. 网格与空间构建（随 n）

下表统计网格、空间与材料构建过程的内存与耗时，不包含单刚计算。峰值 RSS 与构建后 RSS 均为进程绝对量；构建后 RSS 净增以各次测量的 import 底座为基准。


| n   | $N_{dof}$ | 构建期峰值 RSS（GiB） | 构建后 RSS（MiB） | 构建后 RSS 净增（MiB） | 构建耗时（s） |
| --- | --------- | -------------- | ------------ | --------------- | ------- |
| 32  | 107,811   | 1.3            | 779          | 172             | 15      |
| 48  | 352,947   | 3.0            | 1059         | 452             | 53      |
| 64  | 823,875   | 6.2            | 1391         | 784             | 120     |
| 96  | 2,738,019 | 19.5           | 3059         | 2452            | 420     |
| 124 | 5,859,375 | **41.0**       | 5880         | 5273            | 880     |


五档测量的峰值均出现在网格构建阶段，`space` 阶段的内存净增与耗时在当前记录精度下均为零。

数据来源：`outputs/mesh_build_staged_n{32,48,64,96,124}.json`。

### 3. 与 FA 路线对比（免 CSR 符号结构）

FA 的模式先行装配路线（`fast + pattern`）在进入单刚计算前，必须针对有限元空间调用 `build_csr_pattern`，构建张量级 CSR 骨架及分量块槽位映射，在 $n=32, 48, 64$ 时分别引入 47、215、519 MiB 的常驻开销（FA 报告第一章第 3 小节）。

EA 算子属于无矩阵范式（Matrix-Free），在整个生命周期内不显式组装全局 CSR 矩阵，因而**完全免除 CSR 符号结构的构建与常驻开销**。网格与空间构建完成后，即可直接进入单刚计算与算子缓存。

---



## 二、阶段 1：单刚计算与缓存（随 n）



### 1. 算子构建与单刚缓存机制

EA 算子通过分析器调用 `assemble_stiff_matrix()` 完成单刚计算，并在内部构建常驻算子实例，缓存单刚张量 $K_e$ 与自由度映射 `cell2dof`。

相关源码实现：

- 分析器工厂与装配入口：
  - [`src/soptx/fem/analyzers/builders.py`](../../src/soptx/fem/analyzers/builders.py) 中的 `build_serial_analyzer` 
  - [`src/soptx/fem/analyzers/lagrange_fem_analyzer.py`](../../src/soptx/fem/analyzers/lagrange_fem_analyzer.py) 中的 `assemble_stiff_matrix`
- EA 算子构建与单刚缓存逻辑：
  - [`src/soptx/fem/levels/element.py`](../../src/soptx/fem/levels/element.py) 中的 `ElementAssembly.build`
- 算子限制类：
  - [`src/soptx/fem/kernels/restriction.py`](../../src/soptx/fem/kernels/restriction.py) 中的 `ElementRestriction`

核心代码如下：

```python
# 1. 创建 EA 分析器（内部根据 material, degree 等自动实例化 self._integrator = LinearElasticIntegrator(...)）
analyzer = build_serial_analyzer(vs, problem, material, degree=1, operator_level="ea", assembly_method="fast")

# 2. 单刚计算与缓存入口
ea_op = analyzer.assemble_stiff_matrix()

# 3. 分析器内部的分派逻辑：
#    level = create_level('ea', space=self._tensor_space, integrator=self._integrator)
#    分派至 ElementAssembly.build(space, integrator) 内部执行：
integrator = analyzer._integrator
const_integrator = integrator.const(space)  # 计算并缓存单刚张量 K_e, 形状 (NC, 12, 12)

# 构造限制算子 (G 算子，缓存全局自由度映射 cell2dof)
restriction = ElementRestriction(
    cell2dof=const_integrator.to_global_dof(space),
    global_dofs=space.number_of_global_dofs(),
)
```



### 2. 峰值 RSS 与常驻 RSS 实测对比（随 n）

单刚计算采用 `fast` 算法。与 FA 阶段 1 仅保留 $K_e$ 不同，EA 算子在计算完成后，不仅保留单刚张量 $K_e$（float64，$(N_C, 12, 12)$），还长期常驻保留全局自由度映射 `cell2dof`（int64，$(N_C, 12)$），供后续算子乘重复调用。三档规模下 `cell2dof` 的理论存储量分别为 18、61、144 MiB；与 $K_e$（216、729、1728 MiB）相加，两者理论存储量合计分别为 234、790、1872 MiB。

下表统计三档规模下单刚阶段的内存与耗时。各内存列统一换算为 GiB，保留一位小数：


| n   | $N_C$     | $N_{dof}$ | `cell2dof`（MiB） | 起点 RSS（GiB） | 峰值净增（GiB） | 阶段峰值 RSS（GiB） | 缓存后常驻 RSS（GiB） | 缓存耗时（s） |
| --- | --------- | --------- | --------------- | ----------- | --------- | ------------- | -------------- | ------- |
| 32  | 196,608   | 107,811   | 18              | 0.7         | 0.7       | 1.4           | 1.0            | 0.69    |
| 48  | 663,552   | 352,947   | 61              | 0.9         | 2.3       | 3.2           | 1.9            | 2.1     |
| 64  | 1,572,864 | 823,875   | 144             | 1.3         | 5.3       | 6.6           | 3.4            | 5.0     |


表中呈现了两个核心特征：

1. **峰值 RSS 与常驻 RSS 的落差**：计算期间由于中间梯度缩并分块与刚度分块同时存活，形成了高于常驻量的阶段瞬时峰值（机制与 FA 完全同源，详见 FA 报告第二章第 2 小节）；计算完成后临时分块释放，常驻 RSS 回落并沉淀为后续算子的常驻基线。以 $n=64$ 为例，阶段峰值为 6.6 GiB，缓存后常驻为 3.4 GiB。
2. **随网格规模的线性缩放**：从 $n=32$ 增至 $n=64$，单元数增至 8 倍，峰值净增约为 7.4 倍，耗时约为 7.3 倍，峰值净增近似随单元数 $N_C$ 线性增长，增幅略低于单元数本身。与 FA 相比，两者的峰值净增几乎相同（$n=64$ 均为 5.3 GiB），而缓存后常驻 RSS 略高于 FA（3.4 对 3.1 GiB），差额约 0.3 GiB，其中常驻保留的 `cell2dof` 映射占 144 MiB，其余为分配器未归还的残留。

数据来源：`outputs/cache_fast_n{32,48,64}.json`。

---



## 三、阶段 2：算子乘的内存机制与执行效率（随 n）



### 1. 算子乘机制与理论工作区构成

FA 的阶段 2 是装配全局稀疏矩阵。EA 彻底舍弃了全局总刚的组装与存储，在求解过程中直接以无矩阵算子乘替代：

$$
y = K x = \sum_{e=1}^{N_C} P_e^T K_e P_e x
$$

其中 $P_e$ 为由 `cell2dof` 确定的限制算子（$x_e = P_e x$）。算子类 `ElementAssembly`（[`src/soptx/fem/levels/element.py`](../../src/soptx/fem/levels/element.py)）实现 `operator @ x` 的核心代码如下：

```python
class ElementAssembly:
    def __matmul__(self, x: TensorLike) -> TensorLike:
        # 1. Gather: 根据 cell2dof 提取每个单元的局部自由度向量 x_e
        x_E = self._restriction.gather(x)

        # 2. Einsum: 并行计算单元级矩阵向量积 y_e = K_e @ x_e
        y_E = bm.einsum('cij, cj... -> ci...', self._K_e, x_E)

        # 3. Scatter-add: 将单元贡献累加回全局向量 y
        return self._restriction.scatter_add(y_E)
```

在执行过程中，步骤 1 提取的单元位移张量 $x_E$ 与步骤 2 计算的单元力张量 $y_E$ 形状均为 $(N_C, 12)$，类型为 float64，理论大小各为 $96 N_C\text{ 字节}$（单块大小在三档规模下分别为 18、61、144 MiB）。在 einsum 缩并与 scatter-add 累加期间，底层还存在同等规模的中间临时张量，使得计算期存活的局部张量理论容量合计约为 **36~54、122~182、288~432 MiB**。

### 2. 算子乘多档实测对比（随 n）

三档规模下的实测数据如下：


| n   | $N_C$     | $N_{dof}$ | 工作区峰值净增（MiB） | 稳态常驻净增（MiB） | 首次耗时（ms） | 稳态耗时（ms） |
| --- | --------- | --------- | ------------ | ----------- | -------- | -------- |
| 32  | 196,608   | 107,811   | 35.9         | **0**       | 22       | 20.3     |
| 48  | 663,552   | 352,947   | 181.5        | **0**       | 62       | 63.7     |
| 64  | 1,572,864 | 823,875   | 430.8        | **0**       | 153      | 154.3    |


数据来源：`outputs/cache_matvec_continuous_fast_n{32,48,64}.json`。

---



## 四、求解前内存汇总

EA 路线在进入 Krylov 迭代求解器（如 Jacobi-PCG）之前，全流程包含网格与空间构建、单刚与自由度映射缓存以及算子乘应用，执行顺序如下：

```python
# 1. 构建问题、网格、有限元空间与材料
problem, mesh, vs, material = build_problem_space(n)

# 2. 构造 EA 算子实例并缓存单刚 K_e 与自由度映射 cell2dof (阶段 1)
analyzer = build_serial_analyzer(vs, problem, material, degree=1, operator_level="ea", assembly_method="fast")
ea_op = analyzer.assemble_stiff_matrix()

# 3. 无矩阵算子乘应用 (阶段 2)
y = ea_op @ x
```



### 1. 内存的累积方式（以 n=64 为例）

与 FA 一致，EA 的全流程同样严格遵循两条规则：**常驻内存累积，阶段峰值不累积**。

下表以最大规模 $n=64$（157.3 万单元、82.4 万自由度）连续实测记录走完全程，三个阶段按执行顺序排列。每行满足「起点常驻 + 自身净增 = 阶段峰值」，每行的结束常驻即下一行的起点常驻。


| 阶段      | 起点常驻（GiB） | 自身净增（GiB） | 阶段峰值（GiB） | 结束常驻（GiB） |
| ------- | --------- | --------- | --------- | --------- |
| 网格构建    | 0.6       | 5.6       | 6.2       | 1.3       |
| 单刚与映射缓存 | 1.3       | 5.3       | **6.6**   | 3.4       |
| 算子乘应用   | 3.4       | 0.4       | 3.8       | 3.4       |




### 2. 多档汇总

下表汇总三档规模下容量评估直接用到的两个核心量：


| n   | $N_{dof}$ | 全过程峰值 RSS（GiB） | 准备后常驻 RSS（GiB） | 峰值所在阶段    |
| --- | --------- | -------------- | -------------- | --------- |
| 32  | 107,811   | 1.4            | 1.0            | 阶段 1：单刚缓存 |
| 48  | 352,947   | 3.2            | 1.9            | 阶段 1：单刚缓存 |
| 64  | 823,875   | 6.6            | 3.4            | 阶段 1：单刚缓存 |


数据来源： `outputs/mesh_build_staged_n{32,48,64}.json`、`outputs/cache_fast_n{32,48,64}.json` 与 `outputs/cache_matvec_continuous_fast_n{32,48,64}.json`。