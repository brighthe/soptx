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
- **数据版本**：第二、三章与第五章的 setup、apply 数据测于 2026-09-09 至 09-18，早于 `kernels/levels` 分层重构（34c4451、96c59a0）。重构后常驻项仍为 $K_e$ 与 `cell2dof`，机制等价，但数值待按当前代码重测后替换；网格构建路径未受重构影响，第一章数据保留。
- **数值精度**：本文关注内存量级与占比，不追求逐字节精确。内存数值统一按「不小于 1 GiB 者保留一位小数、小于 1 GiB 者取整数 MiB」给出，耗时统一保留两位有效数字；同一表中各列为独立读数，舍入后按「起点 + 净增 = 峰值」相加可有 0.1 GiB 的偏差。

---

## 一、装配前：import 底座与网格构建

本节测量 EA 算子装配前的内存开销，即模块导入与网格、空间、材料对象的构建。它们先于算子的四个阶段，固定网格下只付一次，与 setup 阶段的单刚计算相区分。EA 与 FA 的前置成本存在明确分流：

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

EA 算子属于无矩阵范式（Matrix-Free），在整个生命周期内不显式组装全局 CSR 矩阵，因而**完全免除 CSR 符号结构的构建与常驻开销**。网格与空间构建完成后，即可直接进入 setup 阶段的单刚计算与缓存。

---

## 二、setup：单刚计算与缓存（随 n）

### 1. 算子构建与单刚缓存机制

EA 算子：

$$
y = G^T K_e G x
$$

其中 $G$ 为由 `cell2dof` 确定的单元限制算子（全局 → 单元），$K_e$ 为逐单元的稠密刚度矩阵。

EA 各阶段的产物如下：


| 阶段     | EA 中的对应                                                                              | 常驻产物             |
| ------ | ------------------------------------------------------------------------------------ | ---------------- |
| build  | 无独立阶段：参考基函数梯度在 `integrator.assembly` 内部现算，积分完成即丢弃                                    | 无                |
| setup  | `integrator.assembly(space)` 一次积出 $K_e$，参考基、几何、材料与密度全部乘入；`ElementRestriction` 给出 $G$ | $K_e$、`cell2dof` |
| update | 单元密度下由 $K_e^0$ 逐单元缩放后调用 `update` 替换 $K_e$                                            | 新的 $K_e$         |
| apply  | gather → 逐单元小矩阵乘向量 → scatter-add                                                     | 无                |


因此 EA 的常驻项只有两项，且都逐单元存储、随 $N_C$ 线性增长：$K_e$（float64，$(N_C, 12, 12)$）与 `cell2dof`（int64，$(N_C, 12)$）；**全网格共享项为空**，参考基与本构矩阵都已积进 $K_e$，不单独保留。测量时 `assemble_stiff_matrix()` 不传 `rho_val`，取基准实体刚度，因此这里测到的 $K_e$ 即 $K_e^0$，也就是第四章中单元密度更新需另存的那一份。

相关源码：

- 分析器工厂与装配入口：
  - `[src/soptx/fem/analyzers/builders.py](../../src/soptx/fem/analyzers/builders.py)` 中的 `build_serial_analyzer`
  - `[src/soptx/fem/analyzers/lagrange_fem_analyzer.py](../../src/soptx/fem/analyzers/lagrange_fem_analyzer.py)` 中的 `assemble_stiff_matrix`
- EA 算子构建与单刚缓存：
  - `[src/soptx/fem/levels/element.py](../../src/soptx/fem/levels/element.py)` 中的 `ElementAssembly.build`
- 单元限制算子：
  - `[src/soptx/fem/kernels/restriction.py](../../src/soptx/fem/kernels/restriction.py)` 中的 `ElementRestriction`

```python
# 分析器入口: create_level('ea', ...) 分派至 ElementAssembly.build(space, integrator)
analyzer = build_serial_analyzer(vs, problem, material, degree=1, operator_level="ea", assembly_method="fast")
ea_op = analyzer.assemble_stiff_matrix()

# ---- ElementAssembly.build 内部 ----
# setup 阶段: 依赖网格几何与拓扑, 每张网格算一次
# 单元矩阵 K_e: EA 没有独立的 build 段, 参考基在 assembly 内部现算, 与几何、材料、密度一起积进 K_e;
# K_e 的初值含 coef, 按阶段属于第一次 update
K_e = integrator.assembly(space)  # (NC, ldof * GD, ldof * GD)

# 单元限制 G: 扁平布局 (NC, ldof * GD), 只依赖网格拓扑 (cell2dof), 单元内自由度顺序与 K_e 的行列一致
g = ElementRestriction.from_integrator(integrator, space, layout='flat')

# G 与 K_e 拼成 EA 算子, 即 ea_op
ea = ElementAssembly(space, restriction=g, element_matrices=K_e)
```

### 2. 峰值 RSS 与常驻 RSS 实测对比（随 n）

单刚计算采用 `fast` 算法。与 FA 的单刚阶段仅保留 $K_e$ 不同，EA 在 setup 后还常驻 `cell2dof`（`build` 内各处引用的是同一个 `cell2dof` 数组，只计一次）。三档规模下 `cell2dof` 的理论存储量分别为 18、61、144 MiB；与 $K_e$（216、729、1728 MiB）相加，两者理论存储量合计分别为 234、790、1872 MiB。

下表统计三档规模下 setup 阶段的内存与耗时。各内存列统一换算为 GiB，保留一位小数：


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

## 三、apply：算子乘的内存机制与执行效率（随 n）

### 1. 算子乘机制与理论工作区构成

FA 在单刚之后还要把 $K_e$ 求和成全局稀疏矩阵，apply 是稀疏矩阵乘向量。EA 舍弃了全局总刚的组装与存储，apply 直接逐单元作用：

$$
y = K x = G^T K_e G x
$$

其中 $G$ 为由 `cell2dof` 确定的单元限制算子（$x_E = G x$），$G^T$ 为对应的 scatter-add。

相关源码：

- 算子作用入口：
  - `[src/soptx/fem/levels/element.py](../../src/soptx/fem/levels/element.py)` 中的 `ElementAssembly.__matmul__`
- 单元限制算子的 gather 与 scatter-add：
  - `[src/soptx/fem/kernels/restriction.py](../../src/soptx/fem/kernels/restriction.py)` 中的 `ElementRestriction.gather` 与 `ElementRestriction.scatter_add`

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

### 2. 算子乘多档实测对比（随 n）

三档规模下的实测数据如下：


| n   | $N_C$     | $N_{dof}$ | 工作区峰值净增（MiB） | 稳态常驻净增（MiB） | 首次耗时（ms） | 稳态耗时（ms） |
| --- | --------- | --------- | ------------ | ----------- | -------- | -------- |
| 32  | 196,608   | 107,811   | 35.9         | **0**       | 22       | 20.3     |
| 48  | 663,552   | 352,947   | 181.5        | **0**       | 62       | 63.7     |
| 64  | 1,572,864 | 823,875   | 430.8        | **0**       | 153      | 154.3    |


数据来源：`outputs/cache_matvec_continuous_fast_n{32,48,64}.json`。

---

## 四、update：密度更新的内存代价（随 n）

### 1. 密度更新机制

拓扑优化每轮迭代都要按新的密度更新算子。EA 的 update 只改 $K_e$，$G$ 不变：

$$
y = G^T K_e(\rho) G x, \qquad K_e(\rho) = \rho_e K_e^0
$$

其中 $K_e^0$ 为基准单刚（`coef` 全为 1 时的单元矩阵），第二个等式只对单元密度成立。

相关源码：

- 单元矩阵更新：
  - `[src/soptx/fem/levels/element.py](../../src/soptx/fem/levels/element.py)` 中的 `ElementAssembly.update` 与 `ElementAssembly.set_element_matrices`
- 层级复用与 $K_e^0$ 共享：
  - `[src/soptx/fem/analyzers/lagrange_fem_analyzer.py](../../src/soptx/fem/analyzers/lagrange_fem_analyzer.py)` 中的 `LagrangeFEMAnalyzer.assemble_stiff_matrix`
- 按新密度重新积分：
  - `[src/soptx/fem/integrators/linear_elastic_integrator.py](../../src/soptx/fem/integrators/linear_elastic_integrator.py)` 中的 `LinearElasticIntegrator.coef`与 `LinearElasticIntegrator.assembly`

```python
# 更新入口: 优化主流程每轮调 analyzer.assemble_stiff_matrix(rho), 第二轮起复用已有的 EA 层级
# 单元密度 (NC, ) 下 K_e 对 rho_e 线性, EA 由 K_e^0 逐单元缩放即可, 不必重新积分
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

第 2 节的三种写法用 `ElementAssembly.set_element_matrices` 直接替换 $K_e$，绕开上面的分派，以便分别测量。



### 2. 密度更新多档实测对比（随 n）

update 面板在同一进程内依次测 setup → keep_K0 → update_first → update_rest（共 5 轮），三种写法各起一个子进程，互不干扰峰值：

- `scale`：`K_e0` 取 setup 所得 $K_e$ 的别名，每轮 `ea_op.set_element_matrices(rho[:, None, None] * K_e0)`；
- `inplace`：`K_e0` 为独立副本，每轮 `np.multiply(K_e0, rho[:, None, None], out=ea_op.element_matrices)` 原地写回；
- `reassemble`：不存 $K_e^0$，每轮 `integrator.coef = rho` 后 `ea_op.set_element_matrices(integrator.assembly(space))`。

$\rho_e$ 取 $[10^{-3}, 1]$ 上的均匀随机数，在计时之外生成；每轮用前 1000 个单元核对 $\rho_e K_e^0$，相对误差上限 $10^{-12}$。各内存列单位为 MiB：


| n   | $N_C$     | $K_e$ 理论（MiB） | mode         | 备份 $K_e^0$ 常驻净增 | 首轮 update 常驻净增 | 后续轮峰值净增 | 首轮耗时（ms） | 稳态耗时（ms） |
| --- | --------- | ------------ | ------------ | --------------- | -------------- | ------- | -------- | -------- |
| 32  | 196,608 | 216 | `scale` | 0 | 216 | 215 | 29.5 | 30.8 |
| 32  | 196,608 | 216 | `inplace` | 216 | 0 | 0 | 21.2 | 20.9 |
| 32  | 196,608 | 216 | `reassemble` | — | 0 | 701 | 403 | 359 |
| 48  | 663,552 | 729 | `scale` | 0 | 729 | 728 | 298 | 102 |
| 48  | 663,552 | 729 | `inplace` | 729 | 0 | 0 | 70.5 | 70.1 |
| 48  | 663,552 | 729 | `reassemble` | — | 0 | 2,372 | 1,482 | 1,390 |
| 64  | 1,572,864 | 1,728 | `scale` | 0 | 1,728 | 1,727 | 248 | 250 |
| 64  | 1,572,864 | 1,728 | `inplace` | 1,728 | 0 | 0 | 172 | 171 |
| 64  | 1,572,864 | 1,728 | `reassemble` | — | 0 | 5,614 | 3,597 | 3,326 |


列的口径（对应 JSON 中 `stages` 的字段）：

- 备份 $K_e^0$ 常驻净增：`keep_K0` 的 `after_kib - before_kib`；`reassemble` 无此阶段；
- 首轮 update 常驻净增：`update_first` 的 `after_kib - before_kib`，即第一次 update 后常驻的变化；
- 后续轮峰值净增：`update_rest` 的 `net_kib`（峰值减阶段起点），即稳态下每轮的瞬时工作区；
- 首轮 / 稳态耗时：`update_seconds_first` 与 `update_seconds_rest_median`。

三种写法的实测与机制预期逐项吻合：`scale` 与 `inplace` 的内存各项均为一份 $K_e$ 的整数倍（误差 < 1 MiB），`reassemble` 的峰值与 setup 相当，均随 $N_C$ 线性增长；三档规模下核对的相对误差均不超过 $3 \times 10^{-16}$：

- **`scale`**：`K_e0 = K_e` 只是别名，备份本身不占内存（0）；首轮 `rho * K_e0` 新分配一份 $K_e$，此后 $K_e^0$ 与 $K_e$ 分成两块，常驻 +1 份 $K_e$；稳态每轮乘法先分配新 $K_e$、赋值后才释放上一轮的，瞬时峰值再 +1 份 $K_e$。$n=64$ 时进程峰值 6956 MiB，已超过 setup 峰值 6748 MiB。
- **`inplace`**：备份即复制一份 $K_e$（+1），此后原地写回，常驻与瞬时均为 0，进程峰值保持在 setup 峰值。耗时也是三者中最低，稳态耗时与 $K_e$ 大小成正比（约 0.1 ms/MiB，即读 $K_e^0$、写 $K_e$ 共约 20 GB/s）。`scale` 稳态慢约 45%，多出的是每轮新分配内存的缺页代价；$n=48$ 时其前两轮约 300 ms，第三轮起降到 102 ms，为分配器复用已释放内存之后的稳态。
- **`reassemble`**：不存 $K_e^0$，常驻不变（0）；但每轮峰值净增与第二章 setup 相当（$n=32, 48, 64$ 下 setup 为 733、2372、5418 MiB），且叠在已常驻的旧 $K_e$ 之上，$n=64$ 时进程峰值达 9114 MiB（8.9 GiB），比 setup 峰值高 2.3 GiB。耗时为 `inplace` 的约 17–20 倍。

结论：单元密度下 EA 的 update 应采用 `inplace` 写法，代价是常驻多一份 $K_e^0$（$n=64$ 为 1.7 GiB），换来每轮零额外内存与最低耗时；`scale` 的写法最直观，但常驻与 `inplace` 相同、每轮还多一份 $K_e$ 的瞬时峰值，无任何优势；`reassemble` 省下 $K_e^0$ 的常驻，却让每轮峰值高于 setup，只在逐点密度下不可避免。`ElementAssembly.update` 已按此实现：单元密度下走 `inplace`，其余形状走 `reassemble`；且 $K_e^0$ 与柔顺度、应力敏度共用分析器的 `_cached_ke0`，优化流程中本就常驻，`inplace` 的备份代价在这里不再另计。

数据来源：`outputs/update_{scale,inplace,reassemble}_fast_n{32,48,64}.json`。



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