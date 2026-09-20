   # 精确子结构分析

## 1. 研究范围与算例设计

本实验验证 `full_trace` 与 `linear_corner` 的分析正确性、密度更新一致性及计算成本。各算例统一采用 Q1、CPU 显式组装与 MUMPS 求解，覆盖局部缩聚、接口求解和内部位移恢复，不包含优化器密度更新。

| 验证内容 | 算例 | 主要指标 |
|---|---|---|
| 正确性与收敛性 | `HarmonicPoly2D`、`HarmonicPoly3D` | 位移误差、收敛阶、与 FA 的差异 |
| 密度更新一致性 | Huang2022 二维 `CantileverCorner2d`、Huang2023 三维 `FullMBBBeam3d`，采用较小规模 | 密度切换后，更新与重建的结果差异 |
| 二维计算成本 | Huang2022 二维 `CantileverCorner2d`，采用论文规模及规模序列 | 分阶段耗时、总耗时、峰值内存及降阶误差 |
| 三维计算成本 | Huang2023 三维 `FullMBBBeam3d`，采用论文规模及规模序列 | 分阶段耗时、总耗时、峰值内存及降阶误差 |

`full_trace` 验证与同网格 FA 的等价性；`linear_corner` 验证同迹空间内的计算正确性，并单独评价相对 FA 的降阶误差。密度更新实验固定较小规模网格，验证连续切换后的计算一致性；性能实验采用固定密度，分别增加子结构数量和单个子结构网格规模，比较单次分析的耗时、内存和降阶误差。

### 1.1 接口迹空间

两种接口的局部 Schur 缩聚都是精确计算，内部延拓算子 $\mathbf N_{\mathrm{int}}^j=-(\mathbf K_{ii}^j)^{-1}\mathbf K_{ib}^j$ 也相同，差别只在接口位移被限制在哪个空间内。

| 迹空间 | 迹基 $\mathbf T_j$ | 接口坐标 $\mathbf q_j$ | 内部位移恢复 |
|---|---|---|---|
| `full_trace` | $\mathbf I$，保留完整接口 | $\mathbf u_b^j$，完整接口位移 | 接口系统直接给出 $\mathbf u_b^j$，按 $\mathbf u_i^j=\mathbf N_{\mathrm{int}}^j\mathbf u_b^j$ 恢复 |
| `linear_corner` | $\mathbf L_j$，二维沿边线性插值、三维沿面双线性插值 | $\mathbf u_c^j$，宏观角点位移 | 先由 $\mathbf U_\Gamma=\mathbf P\mathbf U_C$ 迹延拓得到 $\mathbf u_b^j$，再用同一恢复式 |

## 2. 正确性与收敛性

### 2.1 算例设置

取 $E=4/3$、$\nu=1/3$、$\rho=1$，体力为零。

| 算例 | 定义域 | 本构 |
|---|---|---|
| `HarmonicPoly2D` | $[0,1]^2$ | 平面应力 |
| `HarmonicPoly3D` | $[0,1]^3$ | 三维弹性 |

`full_trace` 在细网格外边界施加解析位移；`linear_corner` 在宏观边界角点赋值，其余边界位移由迹插值确定。

二维与三维解析位移分别为

$$
\mathbf u_{2D}(x,y)=\bigl(-12x^2y+4y^3,\,-4x^3+12xy^2\bigr)^{\mathsf T},
$$

$$
\mathbf u_{3D}(x,y,z)=\bigl(-12x^2y+4y^3,\,4z^3-12x^2z-4x^3+12xy^2,\,0\bigr)^{\mathsf T}.
$$

### 2.2 网格与子结构划分

二维采用规则四边形网格，三维采用规则六面体网格。两种接口均固定每个子结构的单元数，逐层增加子结构数量，使子结构尺寸和细网格尺寸同时减半。

| 维度（两种接口共用） | 子结构划分 `n_sub` | 块内单元数 `n_fine` | 全局细网格 |
|---|---|---|---|
| 二维 | `2x2 -> 4x4 -> 8x8 -> 16x16` | `2x2` | `4x4 -> 8x8 -> 16x16 -> 32x32` |
| 三维 | `2x2x2 -> 4x4x4 -> 8x8x8` | `2x2x2` | `4x4x4 -> 8x8x8 -> 16x16x16` |

### 2.3 验证结果

`full_trace` 的 $L^2$、$H^1$ 半范收敛阶下限为 1.8、0.8。`linear_corner` 以同迹空间一致性验收，误差与阶数只报告；参照由 FA 刚度与全场延拓构造，内部延拓与缩聚路径共用。

四个 case 均通过验收，观测到 $L^2$ 约二阶、$H^1$ 半范约一阶收敛。本节结果均使用 MUMPS。

| case-id / 参照 | 最细网格 $L^2$ 误差 | 最细网格 $H^1$ 半范误差 | 最后 $L^2$ 阶 | 最后 $H^1$ 半范阶 | 判定 |
|---|---|---|---|---|---|
| `full_trace_convergence_2d` | $1.0087\times10^{-3}$ | $2.5001\times10^{-1}$ | 2.0006 | 1.0001 | PASS |
| `full_trace_convergence_3d` | $5.5267\times10^{-3}$ | $6.8471\times10^{-1}$ | 2.0019 | 1.0003 | PASS |
| `linear_corner_consistency_2d` | $4.6916\times10^{-3}$ | $5.3401\times10^{-1}$ | 2.0012 | 1.0001 | 一致性 PASS；阶数只报告 |
| `linear_corner_consistency_3d` | $2.3631\times10^{-2}$ | $1.4261$ | 2.0057 | 1.0006 | 一致性 PASS；阶数只报告 |
| FA（二维，`32x32`） | $1.0087\times10^{-3}$ | $2.5001\times10^{-1}$ | 2.0006 | 1.0001 | 阶数只报告 |
| FA（三维，`16x16x16`） | $5.5267\times10^{-3}$ | $6.8471\times10^{-1}$ | 2.0019 | 1.0003 | 阶数只报告 |

`linear_corner` 全部加密层、全部一致性指标的最大值：二维为 $2.08\times10^{-15}$，三维为 $4.61\times10^{-15}$，均低于 $10^{-9}$。

`full_trace` 与同网格 FA 的对照如下，各项取全部加密层的最大值。位移与应变能相对差均低于 $10^{-11}$，自由自由度相对残差均低于 $10^{-9}$。

| 维度 | 位移相对差 | 应变能相对差 | `full_trace` 相对残差 | FA 相对残差 | 判定 |
|---|---|---|---|---|---|
| 二维 | $1.41\times10^{-15}$ | $3.49\times10^{-15}$ | $8.73\times10^{-16}$ | $7.68\times10^{-16}$ | PASS |
| 三维 | $1.04\times10^{-15}$ | $3.44\times10^{-15}$ | $7.06\times10^{-16}$ | $5.75\times10^{-16}$ | PASS |

应变能取 $\frac12\mathbf u^{\mathsf T}K\mathbf u$；位移、应变能相对差分别以 FA 位移向量的范数、FA 应变能归一化。

全细网格相对平衡残差为

$$
\eta_{\mathrm{full}}
=\frac{\|K_{ff}u_f+K_{fc}u_c\|_2}{\|K_{fc}u_c\|_2},
$$

其中 $K$ 为全细网格刚度，$f$、$c$ 表示自由和受约束自由度；外载为零，由 $u_c$ 驱动。`full_trace` 代入恢复后的完整位移，FA 代入直接求解的位移。

`linear_corner` 相对 FA 的误差随加密减小，最细网格结果如下；误差包含边界迹插值和接口降阶的影响。

| 维度 | 最细网格 | 位移相对误差 | 应变能相对误差 |
|---|---|---|---|
| 二维 | `32x32` | 0.2233% | 0.1242% |
| 三维 | `16x16x16` | 0.9744% | 0.4797% |

结果来源：

- [二维 full_trace 收敛验证](outputs/full_trace_convergence_2d/20260914T012353860755Z/full_trace_convergence_2d_harmonic-poly_p1_levels4_solver-mumps.json)
- [三维 full_trace 收敛验证](outputs/full_trace_convergence_3d/20260914T012356132147Z/full_trace_convergence_3d_harmonic-poly_p1_levels3_solver-mumps.json)
- [二维 linear_corner 一致性与收敛验证](outputs/linear_corner_consistency_2d/20260914T012402105441Z/linear_corner_convergence_2d_harmonic-poly_p1_levels4_solver-mumps.json)
- [三维 linear_corner 一致性与收敛验证](outputs/linear_corner_consistency_3d/20260914T012404571145Z/linear_corner_convergence_3d_harmonic-poly_p1_levels3_solver-mumps.json)

## 3. 密度更新一致性

### 3.1 算例设置

二维采用 `CantileverCorner2d`，三维采用 `FullMBBBeam3d`，两个算例均验证 `full_trace` 与 `linear_corner`。固定网格、子结构划分、材料参数、载荷与支承，仅更新单元密度。

密度依次切换为均匀场、非均匀场 A、非均匀场 B、初始均匀场。每步比较复用网格、映射与缩聚对象后重新计算的结果，与按当前密度独立重建的结果；最后检查是否恢复初始状态。两种接口各自对照，不要求 `linear_corner` 与 FA 等价。

### 3.2 网格与子结构划分

二维采用规则四边形网格，三维采用规则六面体网格。两种接口使用相同划分，各次密度切换保持网格不变。

| 算例 | 子结构划分 `n_sub` | 块内单元数 `n_fine` | 全局细网格 | 全细网格位移自由度 | `full_trace` 接口自由度 | `linear_corner` 接口自由度 |
|---|---|---|---|---:|---:|---:|
| `CantileverCorner2d` | `8x4` | `5x5` | `40x20` | 1,722 | 698 | 90 |
| `FullMBBBeam3d` | `6x2x2` | `4x4x4` | `24x8x8` | 6,075 | 4,131 | 189 |

自由度数均为施加支承约束前的数量，`linear_corner` 按宏观角点的位移分量计数。

### 3.3 验证结果

四个 case 均通过验证。相对差取全部密度状态、全部对照指标的最大值；残差另取更新与重建两条路径的最大值。

| case-id | 更新与重建的最大相对差 | 最大接口相对平衡残差 | 结果 |
|---|---:|---:|---|
| `full_trace_density_update_2d` | 0 | $1.32\times10^{-13}$ | PASS |
| `full_trace_density_update_3d` | 0 | $1.21\times10^{-13}$ | PASS |
| `linear_corner_density_update_2d` | 0 | $1.29\times10^{-14}$ | PASS |
| `linear_corner_density_update_3d` | 0 | $4.07\times10^{-14}$ | PASS |

相对差覆盖局部刚度、缩聚刚度、恢复矩阵、接口刚度、完整位移和应变能，接口稀疏结构也完全一致。恢复均匀场后，各 case 均通过与初始状态的对照。

接口相对平衡残差为

$$
\eta_{\mathrm{eq}}=\frac{\|(S u_\Gamma-f_\Gamma)_F\|_2}{\|(f_\Gamma)_F\|_2},
$$

其中 $S$、$u_\Gamma$、$f_\Gamma$ 分别为所选迹空间下的接口刚度、接口位移和接口载荷，$F$ 为未施加位移约束的接口自由度集合。`linear_corner` 检查角点降阶系统的平衡。

相对差与残差分别低于 $10^{-11}$、$10^{-9}$。本验证覆盖重新计算与重建的一致性，不覆盖数值缓存的自动失效。

数据来源：

- [full_trace_density_update_2d](outputs/full_trace_density_update_2d/20260911T132310577837Z/full_trace_density_update_2d_sub-8x4_fine-5x5.json)
- [full_trace_density_update_3d](outputs/full_trace_density_update_3d/20260911T132316914143Z/full_trace_density_update_3d_sub-6x2x2_fine-4x4x4.json)
- [linear_corner_density_update_2d](outputs/linear_corner_density_update_2d/20260911T132326662756Z/linear_corner_density_update_2d_sub-8x4_fine-5x5.json)
- [linear_corner_density_update_3d](outputs/linear_corner_density_update_3d/20260911T132331743498Z/linear_corner_density_update_3d_sub-6x2x2_fine-4x4x4.json)

## 4. 二维计算成本

### 4.1 算例设置

采用 `CantileverCorner2d`，在相同网格、非均匀密度场 A、材料参数、载荷与支承下比较 FA、`full_trace` 和 `linear_corner`。各路径每次在独立进程中完整构建并分析，试运行 1 次、正式测量 3 次，报告正式测量的中位数。

### 4.2 网格与子结构划分

采用规则四边形网格，共 512 万个单元。

| 子结构划分 `n_sub` | 块内单元数 `n_fine` | 全局细网格 | 全细网格位移自由度 | 接口自由度 | 接口自由求解自由度 |
|---|---|---|---:|---:|---:|
| `640x320` | `5x5` | `3200x1600` | 10,249,602 | 3,696,002 | 3,692,800 |

### 4.3 验证结果

三条路径各完成 1 次试运行和 3 次独立进程正式测量，均通过自身平衡与支承检查。同一路径三次密度与位移指纹一致，应变能相对差为 0。

FA 检查全细网格自由自由度平衡，`full_trace` 检查自由接口平衡，`linear_corner` 检查含约束反力的宏观平衡及线性约束。精度检查在性能计时之外进行。

| 路径 | 最大相对平衡残差 | 相对 FA 位移误差 | 相对 FA 应变能误差 |
|---|---:|---:|---:|
| FA | $2.03\times10^{-11}$ | — | — |
| `full_trace` | $1.08\times10^{-11}$ | $4.11\times10^{-9}$ | $2.70\times10^{-9}$ |
| `linear_corner` | $3.28\times10^{-12}$ | 3.12% | 11.06% |

各路径平衡残差均低于 $10^{-9}$。`full_trace` 相对 FA 的位移和应变能误差分别为 $4.11\times10^{-9}$ 和 $2.70\times10^{-9}$，数值结果高度一致。`linear_corner` 的两项误差分别为 3.12% 和 11.06%，反映角点迹降阶带来的精度损失。

成本对照取三次正式测量的中位数，时间单位为 s；试运行不参与统计，“—”表示无此阶段。

| 阶段或指标 | FA | `full_trace` | `linear_corner` |
|---|---:|---:|---:|
| 全局／局部刚度装配 | 15.58 | 4.17 | 4.20 |
| 局部缩聚 | — | 25.09 | 22.95 |
| 接口 pattern 首次准备与构建 | — | 5.27 | — |
| 接口映射与数值装配 | — | 6.01 | — |
| 迹投影与宏观装配 | — | — | 1.65 |
| 边界处理与求解 | 124.01 | 98.59 | 7.67 |
| 位移恢复 | — | 1.54 | 1.60 |
| **分析阶段合计** | **139.22** | **140.57** | **38.63** |
| 问题准备（单列） | 150.77 | 156.13 | 164.78 |
| 峰值 RSS（GiB） | 34.22 | 39.52 | 25.14 |

三次分析阶段合计的范围分别为：FA 115.76–146.68 s、`full_trace` 136.99–143.61 s、`linear_corner` 37.22–39.58 s。每次合计取各分析阶段时间之和，再取三次中位数；各分项中位数相加不必等于合计中位数。残差检查、证据读写及阶段间日志不计入分析时间。

每个进程均重新构建装配结构。FA 使用每批 65536 个单元的分块 pattern 装配，pattern 构建与单位单元刚度准备计入全局装配；`full_trace` 的 pattern 成本单列。峰值 RSS 在完整位移得到后、正确性检查前读取，包含依赖加载、问题准备及结构分析。

本档 `full_trace` 与 FA 的分析耗时接近，峰值内存更高；`linear_corner` 分析耗时约为 FA 的 27.7%，但伴随上述降阶误差。FA 三次耗时存在明显波动，本组结果不支持两条精确路径间的小幅速度差异结论，也不代表规模增长趋势。

数据来源：[FA 成本](outputs/fa_cost_2d/20260914T022228026256Z/fa_cost_2d_sub-640x320_fine-5x5.json)、[full_trace 成本](outputs/full_trace_cost_2d/20260914T005217358092Z/full_trace_cost_2d_sub-640x320_fine-5x5.json)、[linear_corner 成本](outputs/linear_corner_cost_2d/20260914T020225864268Z/linear_corner_cost_2d_sub-640x320_fine-5x5.json)、[三路径精度对照](outputs/fa_cost_2d/20260914T022228026256Z/three_route_comparison.json)。

`full_trace` 另运行一次以保存全场位移，其位移指纹与应变能和原成本样本一致，仅用于精度对照，不并入计时统计：[参照结果](outputs/full_trace_reference_2d/20260914T021646825235Z/full_trace_cost_2d_sub-640x320_fine-5x5.json)。FA 原全批装配在内存接近上限时以 -9 退出；分块装配通过小规模矩阵、位移和能量对照后用于本次正式测量：[分块装配验证](outputs/fa_chunked_verification_2d/20260914T020500000000Z/fa_chunked_verify_2d_sub-8x4_fine-5x5.json)。

原连续密度切换记录保留在 [历史 JSON](outputs/full_trace_density_update_2d/20260911T092934958152Z/full_trace_density_update_2d_sub-640x320_fine-5x5.json)，不再作为本节正式性能表。

## 附录 A. 精确子结构分析的关键实现

数学符号沿用[子结构有限元与静力缩聚](C:/workspace/dut-postdoc/concepts/substructural-condensation.md)。以下摘录当前实现的核心语句，省略参数检查、计时与日志，不作为独立脚本。$i$、$b$ 分别表示子结构内部与接口自由度。当前实现采用内部载荷 $\mathbf f_i^j=\mathbf0$ 的模型；`full_trace` 保留完整接口，`linear_corner` 在精确缩聚后引入接口迹降阶。

### A.1 局部刚度装配

本实验采用 SIMP 系数 $\rho_e^p$，其中 $p=3$ 为惩罚指数。设 $\mathbf A_e$ 为子结构内的单元自由度提取矩阵，则

$$
\mathbf K^j=\sum_{e\in j}\mathbf A_e^{\mathsf T}\rho_e^p\mathbf K_e^0\mathbf A_e.
$$

```python
local_density = assembler.split_global_cell_field(density)
local_stiffness = prototype.assemble_local_stiffness_batch(local_density)
```

原型复用单位密度单元刚度 `KE_unit`，通过 `coef = rho_chunk ** self.penal` 和 `bm.einsum('be, eij -> beij', coef, self.KE_unit)` 得到各单元刚度，再用 `bm.bincount` 散加为批量局部矩阵。

源码：[mesh.py：SubstructurePrototype.assemble_local_stiffness_batch](//wsl.localhost/Ubuntu-24.04/home/brighthe/workspace/soptx/src/soptx/fem/substructure/mesh.py:585)、[mesh.py：SubstructurePrototype._assemble_chunk](//wsl.localhost/Ubuntu-24.04/home/brighthe/workspace/soptx/src/soptx/fem/substructure/mesh.py:673)、[_cost_measurement.py：_worker](//wsl.localhost/Ubuntu-24.04/home/brighthe/workspace/soptx/experiments/analysis_capability_substructure/_cost_measurement.py:142)。

### A.2 静力缩聚

$$
\mathbf N_{\mathrm{int}}^j=-(\mathbf K_{ii}^j)^{-1}\mathbf K_{ib}^j,\qquad
\mathbf K_s^j=\mathbf K_{bb}^j-\mathbf K_{bi}^j(\mathbf K_{ii}^j)^{-1}\mathbf K_{ib}^j.
$$

```python
K_ii = K_local[..., self.i_dofs[:, None], self.i_dofs]
K_ib = K_local[..., self.i_dofs[:, None], self.b_dofs]
K_bb = K_local[..., self.b_dofs[:, None], self.b_dofs]
invK_ii_K_ib = bm.linalg.solve(K_ii, K_ib)
self.N = -invK_ii_K_ib
self.K_s = K_bb - bm.matrix_transpose(K_ib) @ invK_ii_K_ib
```

`bm.linalg.solve` 沿批量维求解，不显式构造逆矩阵；实现利用弹性刚度的对称性，以 $(\mathbf K_{ib}^j)^{\mathsf T}$ 表示 $\mathbf K_{bi}^j$。代码中的 `self.K_s` 对应 $\mathbf K_s^j$，`self.N` 对应 $\mathbf N_{\mathrm{int}}^j$，分别用于接口装配与内部位移恢复。

源码：[condensation.py：FEAStaticCondensation.condense](//wsl.localhost/Ubuntu-24.04/home/brighthe/workspace/soptx/src/soptx/fem/substructure/condensation.py:157)。

### A.3 接口空间与全局装配

设局部迹矩阵为 $\mathbf T_j$，降阶接口坐标为 $\mathbf q_j$，则 $\mathbf u_b^j=\mathbf T_j\mathbf q_j$，迹空间刚度为

$$
\mathbf K_r^j=\mathbf T_j^{\mathsf T}\mathbf K_s^j\mathbf T_j.
$$

`full_trace` 取 $\mathbf T_j=\mathbf I$；`linear_corner` 取 $\mathbf T_j=\mathbf L_j$、$\mathbf q_j=\mathbf u_c^j$，其局部刚度为 $\mathbf K_c^j=\mathbf L_j^{\mathsf T}\mathbf K_s^j\mathbf L_j$。二维沿边线性插值，三维沿面双线性插值。分别以 $\mathbf A_j$、$\mathbf A_{c,j}$ 提取局部完整接口与角点自由度，全局装配为

$$
\mathbf K_\Gamma=\sum_j\mathbf A_j^{\mathsf T}\mathbf K_s^j\mathbf A_j,\qquad
\mathbf K_C=\sum_j\mathbf A_{c,j}^{\mathsf T}\mathbf K_c^j\mathbf A_{c,j}.
$$

两条路径统一调用

```python
system = assembler.assemble_trace_system(
    sub_meshes,
    condensor,
    trace_basis=trace_basis,
)
```

`assemble_trace_system` 根据迹基类型选择已有底层装配：`FullTraceBasis` 直接进入完整接口 CSR pattern 装配，不计算 $\mathbf I^{\mathsf T}\mathbf K_s^j\mathbf I$；`LinearCornerTraceBasis` 先计算 $\mathbf L_j^{\mathsf T}\mathbf K_s^j\mathbf L_j$，再装配宏观角点系统。`FullTraceBasis` 当前仍以 `bm.eye` 保存恒等迹矩阵，但统一入口不会用该矩阵执行刚度投影。未知迹基因没有明确的全局自由度映射而直接报错。

接口拓扑不变时，`full_trace` 可复用 CSR pattern；`linear_corner` 可按批投影并装配。两者的局部 Schur 缩聚均为精确计算，`linear_corner` 的近似来自接口位移被限制在角点迹空间内。

源码：[assembler.py：GlobalAssembler.assemble_trace_system](//wsl.localhost/Ubuntu-24.04/home/brighthe/workspace/soptx/src/soptx/fem/substructure/assembler.py:761)、[assembler.py：GlobalAssembler.assemble_interface_system](//wsl.localhost/Ubuntu-24.04/home/brighthe/workspace/soptx/src/soptx/fem/substructure/assembler.py:1012)、[assembler.py：GlobalAssembler.assemble_macro_system](//wsl.localhost/Ubuntu-24.04/home/brighthe/workspace/soptx/src/soptx/fem/substructure/assembler.py:584)、[full.py：FullTraceBasis.from_prototype](//wsl.localhost/Ubuntu-24.04/home/brighthe/workspace/soptx/src/soptx/fem/substructure/traces/full.py:27)、[linear_corner.py：LinearCornerTraceBasis.from_prototype](//wsl.localhost/Ubuntu-24.04/home/brighthe/workspace/soptx/src/soptx/fem/substructure/traces/linear_corner.py:21)。

### A.4 边界条件与求解

`full_trace` 将载荷和支承映射到接口系统，记 $F$、$D$ 为接口的自由与给定位移自由度集合，按 $(\mathbf K_\Gamma)_{FF}(\mathbf U_\Gamma)_F=(\mathbf F_\Gamma)_F-(\mathbf K_\Gamma)_{FD}(\mathbf U_\Gamma)_D$ 消去给定位移自由度，再用 MUMPS 求解：

```python
interface_force = assembler.project_global_vector(system, force)
interface_fixed = assembler.project_global_dofs(system, fixed)
interface_u = solve_interface_system(
    system, interface_force, interface_fixed, solver="mumps"
)
```

非零给定位移通过 `prescribed` 参数传入。上述成本工况采用零位移支承。

`linear_corner` 的全局迹延拓矩阵为 $\mathbf P$，满足 $\mathbf U_\Gamma=\mathbf P\mathbf U_C$；宏观载荷为 $\mathbf F_C=\mathbf P^{\mathsf T}\mathbf F_\Gamma$。另记支承约束矩阵为 $\mathbf C_D$、给定位移向量为 $\mathbf d_D$，则 $\mathbf C_D\mathbf U_C=\mathbf d_D$。二维成本工况从 `projection[full_fixed]` 中提取独立约束，求解

$$
\begin{pmatrix}\mathbf K_C&\mathbf C_D^{\mathsf T}\\\mathbf C_D&\mathbf0\end{pmatrix}
\begin{pmatrix}\mathbf U_C\\\boldsymbol\lambda_D\end{pmatrix}
=\begin{pmatrix}\mathbf F_C\\\mathbf d_D\end{pmatrix}.
$$

```python
saddle = bmat(
    [[stiffness, independent.T], [independent, None]], format="csc"
)
linear_solver = create(request["solve_method"])
try:
    solved_all, _ = linear_solver.setup(saddle).solve(rhs)
finally:
    linear_solver.close()
```

本实验的 `solve_method` 为 `mumps`，$\boldsymbol\lambda_D$ 为支承约束反力的乘子；其平衡检查包含该反力。

源码：[solve.py：solve_interface_system](//wsl.localhost/Ubuntu-24.04/home/brighthe/workspace/soptx/src/soptx/fem/substructure/solve.py:36)、[_cost_measurement.py：_worker](//wsl.localhost/Ubuntu-24.04/home/brighthe/workspace/soptx/experiments/analysis_capability_substructure/_cost_measurement.py:142)。

### A.5 完整位移恢复

先得到完整接口位移，再按 $\mathbf u_i^j=\mathbf N_{\mathrm{int}}^j\mathbf u_b^j$ 恢复各子结构内部位移：

```python
if route == "linear_corner":
    interface_u = bm.asarray(
        projection @ bm.to_numpy(macro_u), dtype=bm.float64
    )
displacement = assembler.recover_full_displacement(
    sub_meshes, condensor, full_view, interface_u
)
```

恢复器用 `bm.einsum('...ij, ...j -> ...i', self.N, u_b_bm)` 计算内部位移，再按全局编号写回。若内部载荷非零，还需载荷缩聚与 $\mathbf w_i^j=(\mathbf K_{ii}^j)^{-1}\mathbf f_i^j$ 恢复项；当前上述流程未包含这两项。

源码：[condensation.py：StaticCondensationBase.recover](//wsl.localhost/Ubuntu-24.04/home/brighthe/workspace/soptx/src/soptx/fem/substructure/condensation.py:123)、[assembler.py：GlobalAssembler.recover_full_displacement](//wsl.localhost/Ubuntu-24.04/home/brighthe/workspace/soptx/src/soptx/fem/substructure/assembler.py:1100)。
