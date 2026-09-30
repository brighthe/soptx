# FA 显式装配内存机制与容量能力

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

FA 算子：

$$
y = K x, \qquad K = \sum_e G_e^{\mathsf T} K_e\, G_e
$$

其中 $G_e$ 为由 `cell2dof` 确定的单元限制算子，$K_e$ 为逐单元的稠密刚度矩阵，以 CSR 格式常驻全局矩阵 $K$。

FA 各阶段的产物如下：


| 阶段     | FA 中的对应                                                                                                                                  | 常驻产物                                             |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| build  | 无独立阶段：fast 路线的张量 $\Lambda$ 由 `fetch_fast_assembly` 与 setup 的几何量在同一次调用中算出                                                                 | $\Lambda$，约 2 KiB                                |
| setup  | 符号阶段 `build_csr_pattern` 建立 CSR 骨架与槽位映射；`integrator.assembly(space)` 积出 $K_e$；数值阶段 `assemble_csr` 按槽位把 $K_e$ 累加进 `pattern.buffer`，得到 $K$ | $K$（`crow`、`col`、`values`）、槽位映射；$K_e$ 为局部量，装配后释放 |
| update | 层级不可复用：按新 `coef` 重新积分 $K_e$，复用骨架，数值阶段原地覆盖 `pattern.buffer`                                                                               | 新的 $K$（`values` 原地覆盖）                            |
| apply  | CSR 稀疏矩阵乘向量                                                                                                                              | 无                                                |


优化中各阶段按 setup →（update → apply × m）× k 循环：每轮先按新密度 update，再在 Krylov 求解中调用 m 次 apply；第一轮的密度在 setup 时随 `coef` 一并传入，不另调 update。

---



## 一、装配前：import 底座与网格构建

本节测量 FA 算子装配前的内存开销，即模块导入与网格、空间、材料对象的构建。它们先于算子的四个阶段，固定网格下只付一次。CSR 符号结构只依赖网格拓扑，同样只建一次，但属于算子自身的 setup，放在第二章。

### 1. import 底座（不随 n）

分步导入依赖栈，各步常驻 RSS 增量如下（首项含进程初始内存）：


| 导入阶段                             | RSS 增量（MiB） |
| -------------------------------- | ----------- |
| Python、探针 + numpy + scipy.sparse | 48          |
| `torch`                          | 458         |
| `sympy`                          | 30          |
| FEALPy + SOPTX 后续导入              | 69          |
| **合计 / 最终 RSS**                  | **606**     |


底座不随 $n$ 变化，与 EA 实测（607 MiB）一致。

数据来源：`outputs/import_baseline_20260909T021748Z.json`。

### 2. 网格与空间构建（随 n）

`mesh` 阶段合并测量网格、有限元空间与材料的构建，不含任何装配。峰值与构建后 RSS 均为进程绝对量，净增以同一进程的 import 底座为基准。


| n   | $N_{dof}$ | 构建期峰值 RSS（GiB） | 构建后 RSS（MiB） | 构建后 RSS 净增（MiB） | 构建耗时（s） |
| --- | --------- | -------------- | ------------ | --------------- | ------- |
| 32  | 107,811   | 1.3            | 782          | 172             | 11      |
| 48  | 352,947   | 3.0            | 1067         | 458             | 37      |
| 64  | 823,875   | 6.2            | 1393         | 783             | 86      |
| 96  | 2,738,019 | 19.5           | 3064         | 2454            | 300     |
| 124 | 5,859,375 | **41.1**       | 5883         | 5273            | 660     |


五档的峰值均出现在网格构建，`space` 阶段的净增与耗时在记录精度下为零。构建期峰值远高于构建后常驻（$n=124$ 为 41.1 对 5.7 GiB），$n=124$ 已接近 45 GiB 预算，容量评估须计入这一峰值。

数据来源：`outputs/mesh_build_n{32,48,64,96,124}.json`。

---



## 二、setup：符号结构、单刚计算与总刚生成（随 n）



### 1. 装配机制

FA 的 setup 分三步。单刚与 EA 相同，按积分点求和：

$$
K_e = \sum_q B_{e,q}^{\mathsf T} \left( w_q\,\lvert T_e\rvert\,\rho_e\,D \right) B_{e,q}
$$

结果形状为 $(N_C, 12, 12)$，本章测量不传密度，即 $\rho_e = 1$。随后把 $K = \sum_e G_e^{\mathsf T} K_e G_e$ 显式装配为 CSR 矩阵，拆成两段：

- **符号阶段**：`build_csr_pattern` 在标量空间上建立 CSR 骨架（`crow`、`col`）与槽位映射（`slot_base`、`row_deg`），只依赖网格拓扑，首次装配时建好并交回分析器，后续各轮复用；
- **数值阶段**：`assemble_csr` 按 $3\times3$ 分量块，把 $K_e$ 的元素原子累加到骨架对应的槽位 `pattern.buffer` 上。

相关源码：

- 分析器工厂与装配入口：
  - `[src/soptx/fem/analyzers/builders.py](../../src/soptx/fem/analyzers/builders.py)` 中的 `build_serial_analyzer`
  - `[src/soptx/fem/analyzers/lagrange_fem_analyzer.py](../../src/soptx/fem/analyzers/lagrange_fem_analyzer.py)` 中的 `assemble_stiff_matrix`
- FA 层级与模式先行装配：
  - `[src/soptx/fem/levels/full.py](../../src/soptx/fem/levels/full.py)` 中的 `FullAssembly.build`
  - `[src/soptx/fem/bilinear_form.py](../../src/soptx/fem/bilinear_form.py)` 中的 `BilinearForm.assembly`
- 符号阶段与数值阶段：
  - `[src/soptx/fem/matrix/csr_pattern.py](../../src/soptx/fem/matrix/csr_pattern.py)` 中的 `build_csr_pattern` 与 `assemble_csr`

```python
# 分析器入口: create_level('fa', ..., pattern=self._csr_pattern) 分派至 FullAssembly.build
analyzer = build_serial_analyzer(vs, problem, material, degree=1, operator_level="fa", assembly_method="fast")
K = analyzer.assemble_stiff_matrix()

# ---- FullAssembly.build -> BilinearForm.assembly(method="pattern") 内部 ----
# 符号阶段: 首次装配时惰性建立, 交回分析器缓存, 固定网格下只执行一次
if self._pattern is None:
    self._pattern = build_csr_pattern(space)

# 单刚: 与 EA 相同的 fast 路线, K_e 为局部量, 装配结束即释放
K_e = integrator.assembly(space)  # (NC, ldof * GD, ldof * GD)

# 数值阶段: 按分量块原子累加进 pattern.buffer
K = assemble_csr(K_e, self._pattern)  # CSRTensor
```

本实验不经分析器，按上述顺序直接调用三步，与生产路径等价。第 2 小节给出三步在同一进程内的多档实测，第 3 小节给出 $n=32$ 下单刚方法与总刚路线的选择依据。

### 2. 峰值 RSS 与常驻 RSS 实测对比（随 n）

三步在同一进程内连续测量，每步重置 `VmHWM`。起点为网格与空间已建成时的常驻 RSS；setup 峰值取三步阶段峰值的最大值，setup 后常驻在保留 $K_e$、符号结构与 $K$ 的状态下读取：


| n   | $N_C$     | 起点 RSS（GiB） | 符号 峰值净增 / 常驻增量（MiB） | 单刚 峰值净增（GiB） | 数值 峰值净增（MiB） | setup 峰值 RSS（GiB） | setup 后常驻 RSS（GiB） | 耗时 符号 / 单刚 / 数值（s） |
| --- | --------- | ----------- | ------------------- | ------------ | ------------ | --------------- | ---------------- | ------------------- |
| 32  | 196,608   | 0.8         | 47 / 47             | 0.7          | 24           | 1.5             | 1.1              | 0.17 / 0.64 / 0.28  |
| 48  | 663,552   | 1.0         | 215 / 215           | 2.2          | 117          | 3.5             | 2.2              | 0.52 / 2.0 / 0.96   |
| 64  | 1,572,864 | 1.4         | 735 / 519           | 5.3          | 514          | **7.1**         | 3.9              | 1.5 / 4.6 / 2.4     |


表中呈现了两个核心特征：

1. **峰值由单刚决定**：三档 setup 峰值均落在单刚一步，原因是 `A_ab`、`KK_ab` 与 $K_e$ 同时存活（第 3 小节）。从 $n=32$ 增至 $n=64$，单元数增至 8 倍，单刚峰值净增约 7.5 倍、耗时约 7.3 倍，近似随 $N_C$ 线性增长。单刚净增与 EA 的 setup 相同（$n=64$ 均为 5.3 GiB），FA 的 setup 峰值高出 0.5 GiB（7.1 对 6.6 GiB），差额即单刚之前已常驻的符号结构。
2. **符号与数值两步的常驻与名义值吻合**：$n=64$ 符号结构常驻 519 MiB，名义值为 `col` 276 + `slot_base` 192 + `row_deg` 48 + `crow` 6 = 522 MiB；数值阶段常驻增量 276 MiB，恰为 `buffer` 的名义值——它由 `np.zeros(nnz)` 申请，calloc 零页在首次写入前不计入 RSS，故落在数值阶段而非符号阶段。符号阶段峰值高出常驻约 200 MiB，是构建期瞬时量。小规模读数受分配器复用空闲页影响（如 $n=32$ 数值阶段净增 24 MiB，低于 `buffer` 名义 35 MiB），宜按量级理解。

数据来源：`outputs/full_fast_pattern_n{32,48,64}_presolver_rss.json`。

### 3. 单刚方法与总刚路线的选择依据（n=32）

**单刚方法**：三种方法生成的 $K_e$ 相同（理论 216 MiB），`fast` 的峰值与耗时均最低：


| method     | 阶段峰值 RSS（GiB） | 峰值净增（GiB） | 单刚耗时（s） |
| ---------- | ------------- | --------- | ------- |
| `fast`     | 1.4           | 0.6       | 0.65    |
| `standard` | 5.8           | 5.1       | 4.1     |
| `voigt`    | 11.9          | 11.1      | 6.2     |


数据来源：`outputs/stage1_{fast,standard,voigt}_n32.json`。

`fast` 由张量 $\Lambda$（代码中为 `S`）与重心坐标梯度缩并出 `A_ab`，组合成刚度分块 `KK_ab`，再写入 $K_e$；峰值由下列数组同时存活形成，两组分块共 432 MiB，为 $K_e$ 的两倍（数组存储量，与 RSS 净增口径不同）：

```python
cm, glambda_x, S = self.fetch_fast_assembly(space)
A_xx = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 0], glambda_x[..., 0], cm)
KK_11 = D00 * A_xx + D55 * (A_yy + A_zz)
KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_11)
# 其余 A_ab / KK_ab 同理
```


| 对象                      | 形状 / 数量          | 存储量（MiB） |
| ----------------------- | ---------------- | -------- |
| 梯度缩并分块 `A_ab`           | 9 个 `(NC, 4, 4)` | 216      |
| 刚度分块 `KK_ab`            | 9 个 `(NC, 4, 4)` | 216      |
| 单元刚度矩阵 $K_e$（代码中为 `KK`） | `(NC, 12, 12)`   | 216      |
| 重心坐标梯度 `glambda_x`      | `(NC, 4, 3)`     | 18       |
| **主要数组合计**              |                  | **666**  |


数据来源：`outputs/stage1_probe_fast_n32.json`。

**总刚路线**：`measure_stage2()` 以与网格拓扑一致的合成输入单独测量总刚生成，输入准备不计入窗口；三元组 28,311,552 个，最终非零元 4,619,817 个。

```python
# coalesce
K = COOTensor(np.stack([I, J]), V, spshape=(Ndof, Ndof)).coalesce().tocsr()
# scipy
K = sp.coo_matrix((V, (I, J)), shape=(Ndof, Ndof)).tocsr()
# pattern: 符号阶段 + 数值阶段
K = assemble_csr(K_e, build_csr_pattern(space))
```


| route      | 生成机制              | 峰值 RSS（GiB） | 首轮净增（MiB） | 骨架复用后净增（MiB） | 首轮耗时（s） | 骨架复用后耗时（s） |
| ---------- | ----------------- | ----------- | --------- | ------------ | ------- | ---------- |
| `coalesce` | 全长三元组排序、去重与累加     | 2.1         | 1435      | 1435         | 1.8     | 1.8        |
| `scipy`    | 按行分桶后合并重复项        | 1.3         | 593       | 593          | 0.30    | 0.30       |
| `pattern`  | 标量 CSR 骨架与分量块槽位累加 | 0.9         | 140       | **35**       | 0.43    | **0.29**   |


「首轮」含符号阶段，「骨架复用后」只含数值阶段；`coalesce` 与 `scipy` 没有可前置的部分，两列相同。`pattern` 的净增在两种口径下均最低，骨架复用后为 `scipy` 的约 1/17、`coalesce` 的约 1/41；耗时与 `scipy` 持平，比 `coalesce` 快约 6 倍。三条路线输入载体不同，绝对峰值不宜直接比较。

数据来源：`outputs/stage2_{coalesce,scipy,pattern}_n32.json`。

`pattern` 的骨架建在标量空间上（$snnz=513313$），按 $3\times3$ 分量块展开为张量级（$\text{nnz}=9snnz$），槽位映射只保留标量级的 `slot_base` 与 `row_deg`。各对象的名义存储量如下：


| 对象               | 类型与长度             | 存储量（MiB） | 生命周期               |
| ---------------- | ----------------- | -------- | ------------------ |
| CSR 行指针 `crow`   | int64，$N_{dof}+1$ | 1        | 符号阶段建成后常驻          |
| CSR 列索引 `col`    | int64，nnz         | 35       | 符号阶段建成后常驻          |
| 槽位基址 `slot_base` | int64，$16N_C$     | 24       | 符号阶段建成后常驻          |
| 行度数 `row_deg`    | int64，$4N_C$      | 6        | 符号阶段建成后常驻          |
| 数值缓冲区 `buffer`   | float64，nnz       | 35       | 符号阶段申请，数值阶段首次写入后常驻 |
| 块内槽位 `slot`      | int64，$16N_C$     | 24       | 数值阶段瞬时，每个分量块一份     |
| `_slot_of` 内部瞬时量 | —                 | 6        | 数值阶段瞬时             |
| **主要对象合计**       |                   | **131**  | 其中常驻 101、瞬时 30     |


名义值与 RSS 读数不宜逐项对账：同一符号阶段在两次独立测量中记到 47 与 104 MiB，跨在名义常驻 66 MiB（不含 `buffer`）两侧，差异来自分配器能否复用前序阶段释放的空闲页。槽位映射可直接核对：`resident_map_MiB` 记为 30 MiB，与名义值一致。

数据来源：`outputs/stage2_pattern_n32.json`。

---



## 三、update：每轮重新积分与数值装配（随 n）



### 1. 密度更新机制

FA 的 update 不能像 EA 那样只替换 $K_e$：密度变化后 $K$ 的每个非零元都要重新求和。`FullAssembly.update` 接受的是装配好的全局矩阵而非系数，分析器的 `_level_reusable` 对 `'fa'` 返回 False，每轮重建层级，只复用 CSR 骨架：

$$
K(\rho) = \sum_e G_e^{\mathsf T} K_e(\rho)\, G_e
$$

相关源码：

- 层级重建与骨架复用：
  - `[src/soptx/fem/analyzers/lagrange_fem_analyzer.py](../../src/soptx/fem/analyzers/lagrange_fem_analyzer.py)` 中的 `LagrangeFEMAnalyzer.assemble_stiff_matrix` 与 `LagrangeFEMAnalyzer._level_reusable`
- 数值阶段复用缓冲区：
  - `[src/soptx/fem/matrix/csr_pattern.py](../../src/soptx/fem/matrix/csr_pattern.py)` 中的 `assemble_csr`

```python
# 更新入口: 优化主流程每轮调 analyzer.assemble_stiff_matrix(rho)
# 'fa' 不可复用, 每轮重建 FullAssembly, 但传入上一轮交回的骨架
level = create_level('fa', space=space, integrator=integrator, pattern=self._csr_pattern)

# ---- BilinearForm.assembly 内部 ----
# 符号阶段: 骨架已存在, 跳过
# 单刚: 按新 coef 重新积分, 每轮重付第二章第 2 小节的积分峰值
K_e = integrator.assembly(space)
# 数值阶段: buffer 取 pattern.buffer, 原地置零后累加, 新旧 K 共用同一份 values
K = assemble_csr(K_e, pattern)
```

由此预期每轮 update 由两部分组成：重新积分的峰值净增与第二章第 2 小节的单刚相同（$n=64$ 约 5.3 GiB）；数值阶段复用 `pattern.buffer`，不新增常驻，只有槽位瞬时量。

### 2. 数值阶段多档实测（随 n）

本实验未单独设 update 面板。下表取第二章第 2 小节同一次运行中的数值阶段，即骨架建成、$K_e$ 已算出后的第一次数值装配：


| n   | $N_{dof}$ | 起点 RSS（GiB） | 峰值净增（MiB） | 阶段峰值 RSS（GiB） | 常驻增量（MiB） | 耗时（s） |
| --- | --------- | ----------- | --------- | ------------- | --------- | ----- |
| 32  | 107,811   | 1.0         | 24        | 1.1           | 24        | 0.28  |
| 48  | 352,947   | 2.1         | 117       | 2.2           | 117       | 0.96  |
| 64  | 823,875   | 3.6         | 514       | 4.1           | 276       | 2.4   |


常驻增量是 `buffer` 零页的首次写入：$n=48$、$64$ 分别为 117、276 MiB，与名义值（117、276 MiB）一致；$n=32$ 为 24 MiB，低于名义 35 MiB，部分落在已释放的空闲页中。$n=64$ 峰值净增再多出 238 MiB，对应块内槽位 192 + `_slot_of` 瞬时量 48 = 240 MiB 的名义值；$n=32$、$48$ 的槽位瞬时量落在单刚一步释放的空闲页中，未显现在 RSS 上。第二轮起 `buffer` 已常驻，每轮只剩槽位瞬时量（$n=64$ 名义约 240 MiB），该口径未实测。

结论：FA 每轮 update 的峰值由重新积分决定，与 setup 的单刚峰值相当，量级与 EA 的 `reassemble` 写法相同（$n=64$ 实测 5614 MiB，见 EA 报告第三章）；数值阶段的代价（$n=64$ 为 2.4 s、约 0.24 GiB 瞬时量）次之。与 EA 的 `inplace` 同理，单元密度下 $K_e(\rho) = \rho_e K_e^0$，FA 也可由敏度计算已缓存的 $K_e^0$ 逐单元缩放后直接进入数值阶段，省去每轮的积分峰值，当前实现未采用。

数据来源：`outputs/full_fast_pattern_n{32,48,64}_presolver_rss.json`。

---



## 四、apply：稀疏矩阵乘向量

本实验未测 FA 的 apply。机制上 `FullAssembly.__matmul__` 即一次 CSR SpMV：

$$
y = K x
$$

每次读 `values` 与 `col`（每个非零元 16 B）、`crow` 与 $x$，写 $y$，工作区只有输出向量（$n=64$ 为 6.3 MiB）。update 只覆盖 `values`，形状不变，apply 的代价不随 $\rho_e$ 变化。参与 apply 的常驻矩阵理论量如下（nnz 按 `from_box` 网格的拓扑精确计数，$n=32$ 与第二章第 3 小节一致）：


| n   | nnz        | $K$ 理论（MiB） | EA 的 $K_e$ 理论（MiB） |
| --- | ---------- | ----------- | ------------------ |
| 32  | 4,619,817  | 71          | 216                |
| 48  | 15,369,273 | 237         | 729                |
| 64  | 36,168,777 | 558         | 1,728              |


FA 的算子数据约为 EA 的 1/3，apply 的访存量相应更少。

---



## 五、求解前内存汇总

采用 `fast + pattern` 在真实网格上完成 setup，执行顺序复刻 `BilinearForm.assembly(method="pattern")`：

```python
# 1. 构建问题、网格、有限元空间与材料
problem, mesh, vs, material = build_problem_space(n)

# 2. setup: 符号阶段 -> 单刚 -> 数值阶段
pattern = build_csr_pattern(vs)
K_e = integrator.assembly(vs)
K = assemble_csr(K_e, pattern)
```

符号结构在 update 各轮中始终常驻，单刚计算的起点必然包含它，故本章把 `build_csr_pattern` 置于单刚之前，容量预算不会低估。

### 1. 内存的累积方式（以 n=64 为例）

**常驻内存累积，阶段峰值不累积**。每行满足「起点常驻 + 自身净增 = 阶段峰值」，结束常驻即下一行的起点常驻。


| 阶段         | 起点常驻（GiB） | 自身净增（GiB） | 阶段峰值（GiB） | 结束常驻（GiB） |
| ---------- | --------- | --------- | --------- | --------- |
| 网格构建       | 0.6       | 5.6       | 6.2       | 1.4       |
| setup：符号阶段 | 1.4       | 0.7       | 2.1       | 1.9       |
| setup：单刚计算 | 1.9       | 5.3       | **7.1**   | 3.6       |
| setup：数值阶段 | 3.6       | 0.5       | 4.1       | 3.9       |


起点常驻单调上升，阶段峰值上下起伏，最大值 7.1 GiB 即全过程峰值。网格构建涨了 5.6 GiB，结束时只留下 0.8 GiB，下一阶段从 1.4 GiB 而非 6.2 GiB 起步。

### 2. 多档汇总


| n   | $N_{dof}$ | 全过程峰值 RSS（GiB） | 生成后常驻 RSS（GiB） | 峰值所在阶段     |
| --- | --------- | -------------- | -------------- | ---------- |
| 32  | 107,811   | 1.5            | 1.1            | setup：单刚计算 |
| 48  | 352,947   | 3.5            | 2.2            | setup：单刚计算 |
| 64  | 823,875   | 7.1            | 3.9            | setup：单刚计算 |


三档峰值均落在单刚计算，瓶颈是 `A_ab`、`KK_ab` 与 $K_e$ 同时存活（第二章第 3 小节）。生成后常驻只有峰值的一半到七成，按常驻估算 $n=64$ 会低估 3.2 GiB。

本表的生成后常驻保留了 $K_e$（$n=64$ 约 1.7 GiB）；生产路径中 $K_e$ 是 `BilinearForm.assembly` 的局部量，装配后即释放，实际归还量取决于分配器行为。

数据来源：`outputs/full_fast_pattern_n{32,48,64}_presolver_rss.json`。