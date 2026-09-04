# FEALPy fork 补丁详述

本页是 vendor fork（`~/workspace/fealpy`，分支 `main`）相对上游
`suanhaitech/fealpy:develop` 全部**存活补丁**的技术正文，按 fealpy 子系统分四节。

> **可变状态不在本页。** 分叉量、`main`/`origin` 的当前 SHA、丢弃判据的核对现状统一由
> [README](README.md) 的补丁总表维护；本页只写不变量——缺陷机理、修改点、回归验证、
> 丢弃判据本身，以及补丁所在的提交 SHA（SHA 不随分支前进而过期）。
>
> **效果已丢弃的补丁不在本页**，只在 README 总表留一行。它们的代码对比针对的是上游
> v0.4 之前的旧架构，移植时不可照搬；确需查看时 `git -C ~/workspace/fealpy show <SHA>`
> 直接看原始 diff。本页保留原有的「缺陷 N」编号不重排，编号不连续是正常的。

| 节 | 子系统 | 补丁 |
|---|---|---|
| [一](#一mesh--functionspace-层) | `fealpy/mesh/`、`fealpy/functionspace/` | `0758339`、`fbfe39e`、`ce8aa8ae9`、`13e8ebb75`、`bc2ea8ecb` |
| [二](#二backend--sparse-层) | `fealpy/backend/`、`fealpy/sparse/` | `09783f643`、`ceb0c61`、`88cf4fa`、`c33db98ae` |
| [三](#三solver--distributed-层) | `fealpy/solver/`、`fealpy/distributed/` | `875496d`、`40016dc56`、`4c5887d`、`30ca15599` |
| [四](#四functional-层) | `fealpy/functional.py` | `a1ec1086d` |

---

## 一、mesh / functionspace 层

四条存活缺陷。**缺陷 8 与 9 是全部补丁里风险最高的两条**：都是静默错误——装配出来的
刚度矩阵照样对称正定，CG 照样收敛，残差照样下降，只有观测收敛阶会塌。上游 742 个
测试在这两个缺陷存在时全部通过，**上游测试不构成保护**。

| 缺陷 | 补丁所在提交 | 触及文件 |
|---|---|---|
| 1 | `0758339` | `functionspace/tensor_space.py`（`lagrange_fe_space.py` 侧已取上游） |
| 5 | `fbfe39e` | `functionspace/lagrange_fe_space.py` |
| 8 | `ce8aa8ae9` | `mesh/view/entity_view.py` |
| 9 | `13e8ebb75`（修复）+ `bc2ea8ecb`（测试） | `mesh/schema/entity_schema.py`、`mesh/schema/classic/base.py`、`mesh/ipoints.py`、`tests/mesh/unit/test_interpolation_delta.py`（新增） |

### 1.1 缺陷 1：重心坐标 tuple 的 `TD` 计算错误

- **现象**：三角形/四面体在 `mesh.error(exact, uh)` 处抛
  `ValueError: Size of label 'l' for operand 1 (6) does not match previous terms (4)`。
- **机理**：上游在 `value()` / `grad_value()` 中判断 `if isinstance(bc, tuple): TD = len(bc)`。
  单纯形网格返回的是单元素元组 `(array(NQ, 3),)`，`len(bc)` 为 1，2D/3D 问题被当成 1D 处理。
- **修改点**：`TD = len(bc)` → `TD = sum(item.shape[-1] - 1 for item in bc)`。
- **当前只剩半条**：上游 `1fede55` 已在 `lagrange_fe_space.py` 侧加了等价的
  `len(bc) == 1` 分支，该侧补丁已丢弃取上游；`tensor_space.py` 侧上游仍是
  `TD = len(bc)`，补丁保留。
- **丢弃判据**：下式返回非空即可丢弃。

  ```bash
  git -C ~/workspace/fealpy show suanhai/develop:fealpy/functionspace/tensor_space.py | grep -n "len(bc) == 1"
  ```

- **回归测试**：`tests/functionspace/unit/test_barycentric_tuple_dimension.py`。

### 1.2 缺陷 5：`face_basis` / `edge_basis` 指向单元基函数

- **现象**：装配 traction 面力积分时传入面重心坐标（长度 2），单元基函数校验失败
  （期望长度 3）。
- **机理**：上游硬编码别名 `face_basis = basis`。
- **修改点**（`fealpy/functionspace/lagrange_fe_space.py`）：`face_basis` 改调
  `mesh.face_shape_function`、`edge_basis` 改调 `mesh.edge_shape_function`，
  均补 `[None, ...]` 单元轴。
- **丢弃判据**：下式返回空即可丢弃。上游至今仍是别名，且该文件已有 5 个上游新提交，
  移植时需重新落位。

  ```bash
  git -C ~/workspace/fealpy show suanhai/develop:fealpy/functionspace/lagrange_fe_space.py | grep -n "face_basis = basis"
  ```

- **回归测试**：`tests/functionspace/unit/test_face_basis_entity_dimension.py`。

### 1.3 缺陷 8：v0.4 经典 `shape_function` 路径反置换张量积基函数列序

合并上游 v0.4（`382869aec`）后立刻实测发现的**新缺陷**，是旧缺陷 2/3 的物理症状
（收敛阶塌陷）在新架构中的重现，但落点完全不同。

- **现象**：2D 四边形 $p=1$ 的 $L_2$ 误差不随网格加密下降，32×32 观测阶 **0.1113**
  （门禁 1.80）。残差正常收敛，求解器不报错——典型的静默错位。
- **根因**：`fealpy/mesh/view/entity_view.py` 的 `_legacy_shape_function` 先正确调用
  `schema.lagrange_basis_function(bcs, p)`（输出已排成 `_node_keys()` 序），随后
  `_restore_legacy_basis_order` 经 `_legacy_basis_indices` 取
  `_lagrange_basis_permutation(p)` 的**逆置换**，把列序打回张量积枚举序：

  ```python
  if schema.type_id in {"lagrange_prism", "lagrange_hexahedron"}:
      orders = (order,) if type(order) is int else order
      if all(value == 1 for value in orders):
          return None          # 只有 prism / hex 且各向阶次全为 1 才豁免
  ```

  而 `cell_to_ipoint` 给出的局部自由度序正是 `_node_keys()` 序，两者错位。豁免名单里
  **没有 `lagrange_quadrilateral`**，且多了 `all(value == 1)` 的阶次条件——六面体
  $p=1$ 因此侥幸正确，四边形所有阶次全错。`LagrangeFESpace.basis` 走的恰好是
  `mesh.shape_function` → `_legacy_shape_function` 这条路，错位直接作用在单元刚度装配上。
- **修复**（`ce8aa8ae9`）：`_legacy_basis_indices` 的豁免判据改为按**是否张量积 schema**
  整体豁免，不再看阶次。

  ```python
  if isinstance(schema, _TensorProductOrderSchema):
      return None
  ```

  单纯形的 `_lagrange_basis_permutation` 是恒等置换，反置换对其无害，无需区别对待。
- **实测**：delta 判据 $\phi_i(\mathbf{x}_j) = \delta_{ij}$ 下，Quadrangle Q1 错位由
  2/4 归零；Hex1（豁免分支侥幸正确）、三角形/四面体 $p=1..3$（置换为恒等）修复前后均为 0。
  收敛阶复测 2D Q1 = 2.0006、3D Hex1 = 2.0019。修复后上游 742 个测试全部通过——
  **上游自己的 `test_lagrange_fe_space_p1_p2.py` 并未覆盖基函数列序与 `cell_to_ipoint`
  的一致性**，这是它能合入 `develop` 的原因。
- **丢弃判据**：上游 `_legacy_basis_indices` 改为按张量积与否整体豁免（或等价地，经典
  路径不再做这次反置换），且四边形 $p=1$ 的观测收敛阶实测达 2.0。

### 1.4 缺陷 9：张量积 `multi_index` 的列序不是顶点序

缺陷 8 的补丁只修好了 $p=1$。$p \ge 2$ 的张量积单元仍然错位，根因与缺陷 8 无关，
在 `fealpy/mesh/ipoints.py` 一侧。

- **现象**（修复缺陷 8 后重测 delta 判据）：Quad Q2 错位 7/9、Hex2 错位 26/27、
  Hex3 错位 63/64；Quad Q3 更严重——插值点 $(4/9, 1/3)$ **根本不在网格节点上**。
- **根因**：`EntitySchema.multi_index` 的 docstring 写的是「with one column per vertex」，
  但张量积 schema 的实现不满足该契约：`multi_index_tensorprod` 按**因子嵌套顺序**给列
  编号（四边形 $s = i_\eta \cdot 2 + i_\xi$），而 `local_vertices()`、`_node_keys()`、
  `_local_face_groups()` 一律用逆时针顶点序。两者相差一个固定置换，四边形是
  $(0, 1, 3, 2)$、六面体是 $(0, 1, 3, 2, 4, 5, 7, 6)$——正是 `quadrilateral.py` /
  `hexahedron.py` 里早就存在的 `_tp_to_contract`，`bc_to_point` 收缩顶点坐标用的就是它。

  这一个根因分出三条后果：

  | | 后果 | 从哪一阶暴露 |
  |---|---|---|
  | **A** | **插值点坐标算错**。`ipoints()` 把 `mi / sum(mi)` 当作对 `local_vertices()` 的重心权重去插顶点坐标。四边形 $p=3$ 内部点 `mi = (4,2,2,1)` 按逆时针顶点算得 $(4/9, 1/3)$，正确键应为 $(4/9, 2/9, 1/9, 2/9)$、坐标 $(1/3, 1/3)$ | $p \ge 3$。$p=1$ 顶点走 `top_dimension() == 0` 另一分支；$p=2$ 内部点权重 $(1/4,1/4,1/4,1/4)$ 在 $2 \leftrightarrow 3$ 对换下不变而侥幸正确；边内部点来自 1D 子实体（2 槽位无置换可言） |
  | **B** | **自由度序算错**。`to_ipoint_permutation` 用 `_local_face_groups(schema)`（顶点序）去给 `mi`（槽位序）的行分类，分类本身就是错的。实测 `to_ipoint_permutation(quad, (2,2)) = (0,6,1,2,8,3,4,7,5)` | $p \ge 2$ |
  | **C** | **面内部自由度的定向没被重排**。`_vertex_orientation_to_ipoint_permutation` 拿 `schema.orientation` 当字典键，而查表用的 `global_permutations` 给出的是**顶点置换**。四边形 `orientation` 是重心分量**列**的置换，8 个元素里有 4 个不是顶点对称，与 `vertex_permutations()` 只交出 4 个，六面体 6 个面里有一半的内部自由度根本没被重排 | $p \ge 3$。$p \le 2$ 每面只有 1 个内部点、置换必为恒等 |

  三棱柱是恒等置换（底面三角形 3 槽位 × 高度 2 槽位，恰好与顶点序一致），所以这个映射
  **不能**从 `_lagrange_kernel_node_keys()` 推导：三棱柱的 kernel 因子序与 `multi_index`
  的因子序不同，推出来是 $(0, 3, 1, 4, 2, 5)$，实测应为恒等。必须由 schema 显式给出。
- **修复**（`13e8ebb75`）：

  | 文件 | 改动 |
  |---|---|
  | `mesh/schema/entity_schema.py` | 新增 `multi_index_vertex_columns()` classmethod 钩子，默认返回 `None` 表示两套约定一致；把 `multi_index` 的列语义写成显式契约 |
  | `mesh/schema/classic/base.py` | `_TensorProductOrderSchema` 上实现该钩子，统一返回具体 Schema 已有的 `_tp_to_contract`；三棱柱不定义该常量，与全部单纯形一样落到 `None` |
  | `mesh/ipoints.py` | 新增 `_vertex_column_permutation()`；`ipoints()` 求重心权重前先对齐列（后果 A）；`to_ipoint_permutation()` 对 `_TensorProductOrderSchema` 直接返回 `None`（后果 B）；`_vertex_orientation_to_ipoint_permutation()` 改用 `vertex_permutations()` 作键、用重心键直接查表取代 `multi_index_sort` 的排序启发式（后果 C） |

  后果 B 之所以是「返回 `None`」而不是「把列序修对」：`to_ipoint` 的拼装顺序
  （顶点 → 各维子实体内部点 → 单元内部点）本来就是 `_node_keys()` 序，而张量积单元的
  `shape_function` 经缺陷 8 的补丁后也已是 `_node_keys()` 序，两侧本就对齐，再排一次
  只会错位。单纯形不走这条豁免：它们的 `shape_function` 经 `_legacy_basis_indices`
  反置换后是 kernel 序，仍需原来的置换，因此单纯形一侧一行未动。
- **实测**：delta 判据逐单元检查，网格取多单元（quad $3\times3$、hex $2\times2\times2$、
  tri $3\times3$、tet $2\times2\times2$）使共享边/面的相对定向真正被触发：

  | 阶段 | quad $p=1/2/3/4$ | hex $p=1/2/3/4$ | tri / tet $p=1..4$ |
  |---|---|---|---|
  | 修复前 | ✓ / ✗ 7/9 / 点不在网格上 / — | ✓ / ✗ 26/27 / ✗ 63/64 / — | 全 ✓ |
  | 仅修后果 A | ✓ / ✓ / ✓ / ✓ | ✓ / ✓ / ✗ 10/64 / — | 全 ✓ |
  | 修复后（A+B+C） | 全 ✓ | 全 ✓ | 全 ✓ |

  「仅修后果 A」剩下的 10 个错位全部落在面内部块，且是面内 4 个点之间的置换（面 1/2/5
  一次对换，面 3 一个 4-循环），面 0/4 恰好命中那 4 个碰巧对得上的键——正是后果 C 的
  指纹。单纯形一侧 $p=1..4$ 的 `max|M - I|` 始终在 $3 \times 10^{-15}$ 量级，修复前后
  逐位不变。
- **回归测试**：`tests/mesh/unit/test_interpolation_delta.py`（fork 内），22 个用例，
  `numpy` 后端。

  | 测试 | 覆盖 |
  |---|---|
  | `test_basis_is_delta_at_interpolation_points[quad/hex/tri/tet × p=1..4]` | 16 用例，delta 判据本身；同时断言 `interpolation_points` 点数与 `number_of_global_ipoints` 一致、每个插值点确实落在参考网格节点上（后果 A 会在这里先炸） |
  | `test_multi_index_columns_are_barycentric_in_vertices[edge/tri/tet/quad/hex/prism]` | 6 用例，根因本身：`multi_index(..., internal=True)` 按 `multi_index_vertex_columns()` 对齐后，归一化结果必须逐项等于 `_node_keys()` 的内部点尾段。它不依赖网格装配，只比对 schema 自身的两套编号，三棱柱在其中作为「钩子必须返回 `None`」的反例存在 |

  ```bash
  cd ~/workspace/fealpy && ~/miniconda3/envs/ihpcm/bin/python -m pytest tests/mesh/unit/test_interpolation_delta.py -q
  ```

  **该测试同时兼覆盖缺陷 8 以及已丢弃的缺陷 2、3、6 的症状域，是唯一能抓住这类静默
  错位的自动化保护，下一轮移植上游后必须先跑它。**
- **丢弃判据**：上游 `EntitySchema.multi_index` 的列确实「one column per vertex」，
  或 `ipoints` / `to_ipoint_permutation` 不再假定列序即顶点序。判据直接跑上面这条
  pytest，22 个用例全绿才算成立。
- **为什么以前没被发现**：delta 判据是唯一能抓住这类错误的诊断，而 $p \ge 2$ 的张量积
  元在本仓库和上游都没有收敛阶门禁，SOPTX 又全程走 $p=1$。

### 1.5 收敛阶验收基准

全 Dirichlet 线弹性制造解（Harmonic Polynomial）下的实测观测阶，是 mesh 层补丁的
端到端验收判据。$p=3$ 档在缺陷 9 修复前不可达（内部插值点被算到网格外）。

| 单元 | $p=1$（理论 2.0） | $p=2$（理论 3.0） | $p=3$（理论 4.0） |
|---|---|---|---|
| Triangle | 1.982 | 3.0000 | — |
| Quadrangle | 1.998 | 3.002 | 3.998 / 4.001 |
| Hexahedron | 1.955 | 2.997 | — |

### 1.6 对 SOPTX 的影响与调用契约

[`LagrangeFEMAnalyzer`](../../src/soptx/fem/analyzers/lagrange_fem_analyzer.py)、
[`SubstructurePrototype`](../../src/soptx/fem/substructure/mesh.py) 与
[`GlobalAssembler`](../../src/soptx/fem/substructure/assembler.py) 依赖上述修复，才能在
2D/3D 张量积网格下把全装配与缩聚装配对齐到物理自由度；
[`verify_full_trace_convergence.py`](../../examples/substructure_elasticity/verify_full_trace_convergence.py) 可用
`--degree 1` 或 `--degree 2` 验证对应收敛阶。

SOPTX 的 `--degree` 默认为 1，子结构模块本身也基于 $p=1$：缺陷 9 修复前所有门禁与算例
都走 $p=1$ 路径、不受影响；修复后 $p \ge 2$ 的四边形/六面体元才可用。

---

## 二、backend / sparse 层

| 组 | 性质 | 补丁 |
|---|---|---|
| [2.1](#21-pytorch-后端-numpy-语义对齐上游缺陷) PyTorch 后端 numpy 语义对齐（4 条） | 上游缺陷修复 | `09783f643` |
| [2.2](#22-pytorch-算子补齐与稀疏容器-device能力增强) PyTorch 算子补齐与稀疏容器 `.device` | 能力增强 | `ceb0c61`、`88cf4fa` |
| [2.3](#23-backend-层静态类型标注补齐能力增强纯静态) backend 层静态类型标注补齐（6 个缺口） | 能力增强（纯静态） | `c33db98ae` |

### 2.1 PyTorch 后端 numpy 语义对齐（上游缺陷）

**契约依据**：`fealpy/backend/base.py` 把 `split`、`cumsum`、`cumprod`、`concat` 等函数
列入 `ATTRIBUTE_MAPPING`，numpy 后端不加包装直接委托 `np.*`，因此这些函数的跨后端契约
就是 numpy 语义，PyTorch 后端偏离即为缺陷。`add_at` 不在 `ATTRIBUTE_MAPPING` 内（各后端
各自实现），但 numpy 实现为 `np.add.at`、jax 实现为 `.at[].add()`，契约同样是 numpy 的。

**触发场景**：SOPTX 子结构 Matrix-Free 算子验证
（`examples/piml_substructure_elasticity/verify_matrix_free_ea_operator.py --backend pytorch`）。
同一脚本 numpy 后端全部通过，pytorch 后端在三个位置依次崩溃（缺陷 1–3）；修复后两个后端
7/7 判据一致通过。缺陷 4 由该脚本运行时刷出的 `logger.warning` 暴露。

| # | 缺陷 | 触及文件 | 上游行为 | 修复 |
|---|---|---|---|---|
| **1** | `bm.split` 拒收 1-D 序列分割点，且 int 不整除时静默截断 | `backend/pytorch_backend.py` | 只接受 int 与 `Tensor`，传 list/tuple/ndarray 抛 `ValueError`（崩在 `multi_index_tensorprod` 的 `bm.split(..., (2,), axis=-1)`）；int 不整除时静默 `//` 产生错误分段 | 非 int 输入统一经 `torch.as_tensor` 规范化为 1-D 张量后走原有「分割点 → 段长」换算；int 分支先校验整除性，不整除按 numpy 报错。全库检索无调用点走 int 分支，行为变化风险为零 |
| **2** | `bm.cumsum` / `bm.cumprod` 不带 `axis` 即崩溃 | `backend/pytorch_backend.py` | `_dim_to_axis` 的「`axis=None` 就不传 `dim`」策略只对 `dim` 有默认值的 torch 函数成立，这两个的 `dim` 必填 → `TypeError`（崩在 `CSRTensor.__getitem__` 的 `bm.cumsum(bm.bincount(...))`） | 改专用包装，`axis=None` 时展平后按 `dim=0` 计算；签名覆盖库内全部调用形态。`cumulative_sum`（Array API 名）保持原映射不动 |
| **3** | `CSRTensor.__getitem__` 把 Python list 塞进 `bm.concat` | `sparse/csr_tensor.py`（3 处） | `np.concatenate` 宽容接受 list，`torch.cat` 拒收 → `stiffness[free, free]` 抛 `TypeError` | `[0]` / `[nrow-1]` 改为 `bm.zeros((1,), **kwargs)` / `bm.full((1,), ..., **kwargs)`，`kwargs` 复用该分支既有的 `bm.context(...)`，dtype/device 与被拼接的索引张量一致；算法不变 |
| **4** | `bm.add_at` 不累加重复索引 | `backend/pytorch_backend.py` | 实现为 `a[indices] += src`。torch 的 advanced-indexing 复合赋值先 gather 再 scatter，重复索引下只写回任意一项、静默丢弃其余；包装器每次调用**无条件**打印 `logger.warning` 自认该行为 | 改用 `index_put_(..., accumulate=True)`（torch 侧对应 `np.add.at` 的原语），`slice` 分量显式抛 `NotImplementedError` 并指向 `index_add`；语义对齐后删除该警告。7 形态探针对照 `np.add.at`：新实现 7/7 一致，旧实现 **6/7 错误**（唯一正确的是 bool 掩码，掩码天然无法重复） |

四条是一类问题：后端通用代码或包装器偏离 numpy 语义，numpy 后端下不可见，换 PyTorch
后端才暴露。缺陷 1 的 int 截断与**缺陷 4 是静默错误**，缺陷 2、3 是显式崩溃。

- **丢弃判据**：四条语义全部对齐，且纯上游检出上
  `verify_matrix_free_ea_operator.py --backend pytorch` 判据全通过。

#### 缺陷 4 在 SOPTX 主线上的危害面

以下调用点索引均为 `cell2node.reshape(-1)`，二维四边形网格每个内部节点被 4 个单元重复
引用，**旧实现下每个节点只收到一个单元的贡献**：

| 文件 | 位置 | 受影响量 |
|---|---|---|
| `src/soptx/topology/objectives/compliance.py` | `:171`、`:183` | 节点密度柔顺度灵敏度 |
| `src/soptx/topology/objectives/mechanism.py` | `:112` | 机构位移目标灵敏度 |
| `src/soptx/topology/constraints/volume.py` | `:116` | 体积约束梯度 |
| `src/soptx/topology/constraints/apparent_stress.py` | `:131` | 表观应力约束梯度 |
| `src/soptx/topology/constraints/vanishing_stress.py` | `:200` | 消失应力约束梯度 |
| `src/soptx/filters/strategies.py` | `:153`、`:283` | 密度过滤的节点侧散射累加 |

即 `density_location` 取 `node` / `node_multiresolution` 时，pytorch 后端下这些梯度是
**静默错误**的（numpy 后端不受影响）。修复前的 pytorch 后端拓扑优化结果应视为不可信。

#### 回归测试

| 测试 | 位置 | 覆盖 | 结果 |
|---|---|---|---|
| `test_backends.py::test_split` | fork `test/backend/`（实现上游占位 stub） | 缺陷 1：int 等分、tuple 分割点、list 不等长分割、后端张量分割点，4 组 × numpy/pytorch | 8/8 |
| `test_backends.py::test_add_at` | fork `test/backend/`（新增） | 缺陷 4：1-D 重复索引 + 数组 src、标量 src、2-D 索引张量、行索引入 2-D、tuple `(row, col)` 索引、bool 掩码，6 组 × numpy/pytorch；**每组都含重复索引**，期望值按 `np.add.at` 手算 | 12/12 |
| `verify_matrix_free_ea_operator.py --backend pytorch` | SOPTX `examples/piml_substructure_elasticity/` | 缺陷 1–3 端到端 | 7/7 |
| 同上 `--backend numpy` | 同上 | 缺陷 3 的修改在 numpy 后端无回归 | 7/7，数值与修改前逐位一致 |

- 两个测试均落在上游自带的 `test/backend/` 而非 fork 自建的 `tests/`，是对总账「补丁
  pytest 放 fork `tests/`」约定的**例外**：`test_split` 上游已留占位 stub，`test_add_at`
  与之同类同文件，就地实现使上游日后自行补测试时能直接产生冲突提示。
- `test_add_at` 对输入取 `.copy()` 后再喂 `bm.from_numpy`：`add_at` 原地修改，而
  `NumPyBackend.from_numpy` 返回数组本身，不复制会让 parametrize 共享的数据被前一个
  后端污染。
- 缺陷 2、3 尚无 fealpy 内的独立 pytest，目前仅由 SOPTX 侧验证脚本端到端覆盖。
- 顺带删除了 `test_backends.py` 首行遗留的 `import ipdb`（环境未装、文件内无使用），
  它阻断整个测试文件的 pytest 收集。

```bash
cd ~/workspace/fealpy/test/backend && ~/miniconda3/envs/ihpcm/bin/python -m pytest test_backends.py -k "(test_split or test_add_at) and (numpy or pytorch)" -v
```

#### 附：PyTorch 后端运行期警告治理

`verify_matrix_free_ea_operator.py --backend pytorch` 原有 4 段运行期警告，现已全部消除：
`add_at` 那两条随缺陷 4 修复删除；`check_invariants` 那条在 `pytorch_backend.py` 新增类
属性 `_SPARSE_KWARGS = {'check_invariants': False}` 并应用到全部 5 个稀疏构造点（布局
不变量由 fealpy 自己的 `COOTensor`/`CSRTensor` 维护，且 spmm 在 Krylov 迭代中每步都跑，
逐次校验是纯开销）；`Sparse CSR tensor support is in beta state` 来自 PyTorch C++ 侧的
`TORCH_WARN_ONCE`，无 Python API 可关，**fealpy 侧不动**，改在 SOPTX 的验证脚本里按消息
原文 + `UserWarning` 精确过滤——库不该替调用方决定警告策略。

### 2.2 PyTorch 算子补齐与稀疏容器 `.device`（能力增强）

**目的**：打通 `gpu_elasticity` 中的 GPU 端有限元装配与稀疏求解。上游 `PyTorchBackend`
缺这两个算子，`COOTensor`/`CSRTensor` 也只有 `device_put` 方法而没有 `device` 属性，
与 `Mesh.device` 不统一。

| 补丁 | 触及文件 | 修改点 |
|---|---|---|
| `ceb0c61` | `backend/pytorch_backend.py` | 补 `take_along_axis`（封装 `torch.take_along_dim`，自动把 indices 转 `Long`）与 `unique_counts`（封装 `torch.unique(return_counts=True)`） |
| `88cf4fa` | `sparse/{coo,csr,sparse}_tensor.py` | `COOTensor`/`CSRTensor` 暴露 `.device` property（返回底层索引张量的 device），基类 `SparseTensor` 声明为 `NotImplementedError` |

- **丢弃判据**：上游 `PyTorchBackend` 自带这两个算子；上游三个 sparse 文件自带 `device`
  **property**（而非仅 `device_put` 方法）。
- **无 pytest 覆盖**，仅由 `gpu_elasticity` 下的算例端到端使用。

### 2.3 backend 层静态类型标注补齐（能力增强，纯静态）

`bm` 门面的静态签名由手写的 `backend/manager.pyi` 提供，`TensorLike` 则是一个只用于
`register()` 的 ABC（`backend/base.py`）。两者都比运行期真实能力窄：写对的调用被 Pyright
判为错误。这不影响运行，但把真错误淹没在假阳性里，等于没有静态检查。SOPTX 全量扫描
（`pyright src/soptx`）中这类假阳性有 **681 条**，占总诊断数 21%（3200 → 2519）。

| # | 缺口 | 位置 | 修复 | 消除假阳性 |
|---|---|---|---|---|
| 1 | `set_at`/`add_at` 的 `indices` 只接受张量 | `manager.pyi` | 新增索引别名 `_IT` | 335 |
| 2 | `TensorLike` 缺位运算 dunder（`~`、`&`、`\|`、`^`） | `base.py` | 补 7 个 dunder 声明 | 247 |
| 3 | 逐元素二元运算不接受标量操作数 | `manager.pyi` | 新增操作数别名 `_OT`，改 28 个二元函数与 `clip` | 46 |
| 4 | `TensorLike` 缺 `__float__`/`__int__`/`__index__` | `base.py` | 补 3 个 dunder 声明 | 34 |
| 5 | `BackendManager` 未声明 `random`/`linalg` 子模块，`LinalgModule` 缺 `norm` | `manager.pyi` | 声明两个子模块属性并补 `norm` | 30 |
| 6 | `from_dlpack`/`matrix_transpose`/`cumulative_sum` 三个声明漏写 `self` | `manager.pyi` | 三处各补 `self` | 未计入 |

缺口 6 是后续在 `src/soptx/fem/substructure/condensation.py` 用 `bm.matrix_transpose` 时
才触发的，681 条的统计早于该补丁；它性质也不同（声明笔误而非标注偏窄）。余下约 18 条是
连带消除的（表达式类型从 `Unknown` 变确定后，元组解包与 `einsum` 的报错一并消失）。

全部改动只影响类型检查：`.pyi` 不参与运行，`base.py` 新增的方法体是 `...`，而
`TensorLike` 没有任何子类（`grep -rn "class .*(TensorLike"` 为空），后端类型是通过
`TensorLike.register()` 挂上去的、不进 MRO，因此这些声明在运行期不可达。

**无 pytest 覆盖，也不该有。** 类型标注不产生运行期行为，能检查它的是类型检查器本身。
判据是诊断数对照（3200 → 2519，新增诊断位置为 0），或直接看本仓库三个入口文件（期望
0 errors；补丁丢失后会重新报错，主要落在缺口 1、3、5）：

```bash
cd ~/workspace/soptx && pyright src/soptx/fem/boundary_loads.py experiments/huzhang_topopt_paper/pipeline.py experiments/huzhang_topopt_paper/run.py
```

**逐缺口丢弃判据**：下面三条 grep 都返回非零时本补丁可整体丢弃。

```bash
F=~/workspace/fealpy; git -C $F show suanhai/develop:fealpy/backend/base.py | grep -c "__invert__\|__float__"; git -C $F show suanhai/develop:fealpy/backend/manager.pyi | grep -c "linalg: LinalgModule"; git -C $F show suanhai/develop:fealpy/backend/manager.pyi | grep -c "def matrix_transpose(self"
```

---

## 三、solver / distributed 层

| 组 | 性质 | 补丁 |
|---|---|---|
| [3.1](#31-cg-可插拔内积补丁栈) CG 可插拔内积与真残差刷新（含 fork 自造的 5 条回归） | 能力增强 + fork 内部回归修复 | `875496d` + `40016dc56`、`4c5887d` |
| [3.2](#32-mumps-直接解法暴露对称性标志-sym) MUMPS 直接解法暴露对称性标志 `sym` | 能力增强 | `30ca15599` |

### 3.1 CG 可插拔内积补丁栈

**能力那一半**（`875496d`、`4c5887d`）：上游 `cg()` 只支持标准内积，无法注入 MPI 重叠
修正内积或预条件内积，也没有真残差重算。SOPTX 的 `matrix_free_elasticity` 需要分布式
并行求解与通信掩盖，因此 `fealpy.solver.cg` 签名增加可选参数 `dot_product=None`（提供时
替换默认向量点积）与 `residual_refresh`（定期用 $\|Ax - b\|$ 刷新递推残差）；
`fealpy.distributed.entity_mpi.EntityMPI` 新增 `dot()` 工厂方法（`4c5887d`），构造重叠
分界面的加权一致内积。

**回归那一半**（`40016dc56`）：`875496d` 在加这两个入口的同时，把上游原本正确的批量右端项、
预条件残差内积与 `maxit=None` 三条路径改坏了，`40016dc56` 逐条恢复。

因此本补丁栈的**丢弃判据与其余补丁相反**：它不会因为上游进步而失效，也不能单独回退——
`875496d` 与 `40016dc56` 是一个整体。

#### 五条回归

| # | 缺陷 | 上游 `suanhai/develop` 行为 | `875496d` 引入的回归 | `40016dc56` 修复 |
|---|---|---|---|---|
| **1** | 预条件残差内积退化为 `r · r` | `rTr = sum(r * z, axis=0)`，全程 $M$-内积 | 初值写成 `_dot(r, r)`，循环内仍是 `_dot(r_new, z_new)`，两者量纲不一致 | `rhssq = _dot(r, z)`；初始残差检查同时从 2-范数改成 `max(_collapse(rhssq), 0.0) ** 0.5`，使入口检查与循环内判据同源 |
| **2** | 批量右端项（`b.ndim == 2`）崩溃 | `alpha`/`beta` 逐列为 `(batch,)`，收敛判据取 `sqrt(sum(rTr))` | `_norm` 里的 `float()` 与 `curvature <= 0.0` 作用在 `(batch,)` 上，`batch >= 2` 即抛 `TypeError` / `ValueError` | `_collapse`（求和后开方，判收敛）与 `_worst`（取最小，判 breakdown）两个归约助手 |
| **3** | `maxit=None` 崩溃 | `while True` + `maxit is not None`，签名承诺的无界迭代可用 | 改成 `range(1, maxit + 1)` → `None + 1` 抛 `TypeError` | `iterations = count(1) if maxit is None else range(1, maxit + 1)` |
| **4** | `dot_product` 与批量右端项静默不相容 | 无此参数 | 自定义内积返回单个标量、无法驱动逐列步长，但未做任何校验，会用同一步长推进所有列 | `b.ndim == 2` 且传了 `dot_product` 时在入口抛 `NotImplementedError` |
| **5** | 零右端项判定绕过自定义内积 | `bm.linalg.norm(b)`（串行下无碍） | 沿用 `bm.linalg.norm(b)`，量的是**本地**切片 | `rhs_norm = _norm(b)` 走 `_dot`，是一次全局归约 |

缺陷 1 与 5 是**静默错误**，缺陷 2、3 是显式崩溃，缺陷 4 是缺失的入参校验。串行、无预
条件、`maxit` 给定整数的调用路径——也就是 `875496d` 当时唯一验证过的那条——五条全都不可见。

三点值得单独说明：

- **缺陷 1 只有第一步错**：正确值应为 $\alpha_0 = (r_0^{\mathsf T} z_0)/(p_0^{\mathsf T} A p_0)$、
  $\beta_0 = (r_1^{\mathsf T} z_1)/(r_0^{\mathsf T} z_0)$，回归版本的分母写成
  $r_0^{\mathsf T} r_0$。之后递推自洽，所以表现为收敛变慢而非发散，在谱较平的测试矩阵上
  完全看不出来；不带预条件时 `z = r`、两者恒等，这是它长期未被发现的原因。
- **`_collapse` 与 `_worst` 语义刻意不同**：收敛判据把整批当作一个整体（与上游
  `sqrt(sum(rTr))` 一致），而非正曲率是单列的性质，一列坏掉不应被其余列的正贡献求和掩盖。
  步长本身仍是逐列的：`alpha = rhssq / curvature` 为 `(batch,)`，`alpha * p` 依广播作用到
  `(dof, batch)`，与上游的 `alpha[None, ...]` 等价。
- **缺陷 5 在 MPI 下是死锁**：若某 rank 分到的 `b` 恰好全零（子域内无载荷），它会当场
  return 而其余 rank 继续迭代，后者卡死在下一次集合通信上。

#### 回归测试

`test/solver/test_cg.py`（fork 内，`40016dc56` 新增），11 个测试函数、13 个用例，`numpy`
后端，不依赖网格与装配，失败即指向 `fealpy/solver/cg.py` 本身：

| 测试 | 覆盖 |
|---|---|
| `test_single_vector` | 基线：串行 1-D 右端项对 `np.linalg.solve` |
| `test_batched_right_hand_side[1/3]` | 缺陷 2：`batch = 1` 与 `batch = 3` 都须解出每一列 |
| `test_batch_first_matches_batch_last` | `batch_first=True` 的转置语义 |
| `test_columns_are_solved_independently` | 缺陷 2：两列量级相差 $10^6$，防止整批塌缩成一个步长 |
| `test_maxit_none_runs_until_converged` | 缺陷 3 |
| `test_maxit_exhausted_reports_failure` | `maxit` 耗尽时 `converged=False`、`niter` 正确 |
| `test_jacobi_preconditioner[(60,)/(60,2)]` | 缺陷 1：矩阵按 `logspace(0, 3, n)` 行列缩放——**对角平坦时 Jacobi 预条件是单位阵的倍数，`r·r` 与 `r·z` 无从区分**，必须拉开谱 |
| `test_custom_dot_product_reproduces_default` | 串行恒等式：普通点积作为 `dot_product` 传入，迭代步数与解都不得改变 |
| `test_custom_dot_product_rejects_batches` | 缺陷 4 |
| `test_residual_refresh_reports_true_residual` | `residual_refresh` 报告的 `true_residual` 与 $\|Ax - b\|$ 相符 |
| `test_zero_right_hand_side[(20,)/(20,3)]` | 缺陷 5 的串行侧 |

该文件落在上游自带的 `test/solver/` 而非 fork 自建的 `tests/`，是对总账约定的**例外**：
上游 `test/solver/` 已是既有目录，其余求解器的测试都在那里（11 个文件，其中 7 个是 Krylov
方法），但没有 `test_cg.py`；就地新建以便上游日后补测试时能产生冲突提示。

未覆盖：缺陷 5 的多 rank 行为（需 MPI 环境），以及 `dot_product` 接分布式内积的实际
正确性——后者只由 SOPTX 侧 `src/soptx/fem/matrix_free/krylov.py` 的 matrix-free 门禁
端到端覆盖。

```bash
cd ~/workspace/fealpy && ~/miniconda3/envs/ihpcm/bin/python -m pytest test/solver/test_cg.py -v
```

#### 对 SOPTX 的影响与丢弃判据

`src/soptx/fem/matrix_free/krylov.py:157` 是 `dot_product` / `residual_refresh` 的唯一
调用点，传入 `EntityMPI.dot()` 构造的重叠修正内积。它走 1-D 右端项路径，因此缺陷 2、3
影响不到它；缺陷 1 影响到它当且仅当该路径启用预条件，缺陷 5 则在任何存在空载荷子域的
分区上都可能触发。

本补丁栈修的是 fork 自己的回归，不存在「上游修好了就删」的情形。唯一的删除条件是
**整栈还原**——即 `krylov.py` 不再调用这两个参数，此时应把 `fealpy/solver/cg.py` 直接
checkout 回上游版本，连同 `875496d` 一起丢掉，而**不是**保留 `875496d` 再单独回退
`40016dc56`（后者会让上述五条回归重新生效）。`4c5887d`（`EntityMPI.dot()`）的判据是
常规的：上游 `EntityMPI` 自带 `dot()` 即删。

### 3.2 MUMPS 直接解法暴露对称性标志 `sym`

- **目的**：位移有限元的刚度阵经对称消元后仍是对称正定，按一般非对称矩阵分解会白白付出
  约一倍的因子存储与运算量。`soptx` 的三维制造解基准在 `n=64`（823,875 自由度）上因此
  多占十余 GiB。
- **上游现状**：`_mumps_solve` 内部写死 `ctx = DMumpsContext()`，实际总是 `SYM=0`，且没有
  任何形参可以改。
- **修改点**（`30ca15599`，`fealpy/solver/direct.py`）：`_mumps_solve(A, b, sym=0)` 校验
  `sym ∈ {0,1,2}`，非零时用 `scipy.sparse.tril` 只取下三角后再交给 MUMPS（`SYM=1/2` 下
  若把上三角一并传入，对称位置会被重复计入）；`spsolve` 透传该形参，其余后端忽略；默认值
  保持 `0`，既有调用方行为不变。
- **风险**：`sym=1` 声明矩阵对称正定，MUMPS 据此跳过数值主元选取；`sym=1`/`2` 都会直接
  丢弃严格上三角部分且**不做对称性校验**。对非对称矩阵传非零 `sym` 会静默求解另一个方程组。
- **实测收益**（`soptx` 三维四面体 P1 制造解，`n=64`，823,875 自由度，`fast` 装配，进程
  峰值 RSS；三者解一致，相对 $L^2$ 误差逐位相同）：

  | `sym` | 峰值 RSS | 墙钟 | 最细档单档耗时 |
  |---|---|---|---|
  | `0`（原行为） | 28.74 GiB | 3:08 | 154.9 s |
  | `1`（对称正定） | 17.38 GiB | 3:08 | 150.3 s |
  | `2`（一般对称） | 17.32 GiB | 4:07 | 205.8 s |

  `sym=1` 在同等墙钟下省 11.4 GiB，是位移元的推荐取值；`sym=2` 因保留数值主元选取而慢
  约三分之一。
- **调用侧**：`LagrangeFEMAnalyzer.solve_system` 从 `kwargs` 读 `sym` 并透传，缺省仍为 `0`；
  `examples/lagrange_elasticity/manufactured_convergence_demo.py` 以 `--mumps-sym` 暴露。
- **丢弃判据**：上游 `_mumps_solve` / `spsolve` 自带对称性入口，且 `sym` 非零时只传下三角。
- **无 pytest 覆盖**，上述收益由一次性的峰值 RSS 测量核对。

> ⚠️ **移植后有静默失效风险。** 上游 `1fede55` 把 **`spsolve` 的默认 solver 从 `"mumps"`
> 改成了 `"scipy"`**，改的正是本补丁修改的那一行签名（合并 `382869aec` 时该文件冲突，
> 手工解为保 fork 的 `sym` + 收上游的新默认值）。若调用侧不显式传 `solver="mumps"` 就会走
> scipy 分支，而 scipy 分支忽略 `sym`——透传的 `sym` 会**静默失效**：结果依然正确，只是峰值
> RSS 悄悄从 17.4 GiB 涨回 28.7 GiB。已核对 SOPTX 全部调用点都显式传了 `solver=`，当前
> 不受影响；下一轮移植后需重新确认这一点。

---

## 四、functional 层

### 4.1 `linear_integral` 把与单元无关的基函数沿单元轴广播物化

补丁 `a1ec1086d`，是 `fealpy/functional.py` 的第一个也是唯一一个 fork 补丁。

| 项 | 内容 |
|---|---|
| 触及文件 | `fealpy/functional.py`（`linear_integral`，两处收缩分支） |
| 症状 | 载荷向量组装的瞬态内存随单元数线性增长，`n=64` 的三维四面体算例上单此一项约 `4.4 GiB` |
| 根因 | `bm.einsum` 恒带 `optimize=True`，收缩路径按 FLOP 数选、对内存无感，把长度为 `1` 的单元轴广播成逐单元中间量 |
| 影响面 | 基函数与单元无关的调用方，即 Lagrange 单纯形元的全部 `SourceIntegrator` 路径 |
| 修法 | 先把求积权重折进与单元无关的小基函数，剩下一个两操作数收缩，路径不再有自由度可选 |
| 数值影响 | 无。仅改变求和次序，48 组参照比对中 30 组逐位相同，最差相对偏差 `3.522e-16` |

#### 触发条件与根因

`linear_integral(basis, weights, measure, source)` 中 `basis` 的形状是 `(C, Q, I, ...)`。
对**单纯形上的 Lagrange 基**，基函数在参考单元上定义、与具体单元无关，
`LagrangeFESpace.basis()` 因此返回 `C = 1` 的数组并依赖广播——三维四面体 P1、`q=4` 求积下
它的形状是 `(1, 20, 12, 3)`，**仅 5,760 字节**。而 `measure` 与 `source` 是逐单元的。

上游把四个操作数一次交给 `einsum`（`'c, q, cqid, ...cq -> ...cid'` 与
`'c, q, cqid, ...cqd -> ...ci'`），而 fork 的 NumPy 后端对 `einsum` 恒定注入
`optimize=True`（`fealpy/backend/numpy_backend.py:94`，NumPy 自身缺省是 `False`）。
`optimize=True` 把四操作数收缩拆成一串两两收缩，**拆法只按 FLOP 数打分、不计中间量的
字节数**；它选中的路径先把 `measure`（或 `weights`）与 `basis` 结合，于是那个长度为 `1` 的
单元轴被广播到 `C`——5,760 字节的数组被物化成与网格同量级的中间量。

`np.einsum_path` 在生产尺寸（`NC = 1,572,864`、`NQ = 20`、`I = 12`、`D = 3`）上给出：

| 分支 | 改前最大中间量 | 改后最大中间量 | 输出本身 | 改前 FLOP | 改后 FLOP |
|---|---:|---:|---:|---:|---:|
| 一 `...cq -> ...cid` | `1.132e9` 元素（`8.434 GiB`） | `5.662e7`（`0.422 GiB`） | `5.662e7` | `3.454e9` | `2.265e9` |
| 二 `...cqd -> ...ci` | `3.775e8` 元素（`2.813 GiB`） | `1.887e7`（`0.141 GiB`） | `1.887e7` | `3.039e9` | `2.265e9` |

改后两条分支的最大中间量都**等于输出本身**，即除结果外不再有额外的逐单元中间量。
FLOP 同时下降，因为原路径把权重和面积重复乘进了被广播的大数组。

#### 修改点

`fealpy/functional.py::linear_integral` 的 `is_tensor(source)` 分支：

```python
dof_shape = basis.shape[3:]
basis = basis.reshape(*basis.shape[:3], -1) # (C, Q, I, dof_numel)
# A basis that does not depend on the cell (Lagrange bases on simplices)
# has a cell axis of length 1. einsum chooses its path by FLOP count
# alone and broadcasts that axis to C, materialising an intermediate as
# large as the mesh. Folding the weights into the basis first leaves a
# two-operand contraction, whose largest intermediate is the result.
cellwise_basis = basis.shape[0] != 1

if source.ndim <= 2 + int(batched):
    source = fill_axis(source, 3 if batched else 2)
    if cellwise_basis:
        r = bm.einsum(f'c, q, cqid, ...cq -> ...cid', measure, weights, basis, source)
    else:
        kernel = basis[0] * weights[:, None, None]          # (Q, I, dof_numel)
        r = bm.einsum('...cq, qid -> ...cid', source, kernel) * measure[:, None, None]
    return bm.reshape(r, r.shape[:-1] + dof_shape)
else:
    source = fill_axis(source, 4 if batched else 3)
    if cellwise_basis:
        return bm.einsum(f'c, q, cqid, ...cqd -> ...ci', measure, weights, basis, source)
    kernel = bm.swapaxes(basis[0], -1, -2) * weights[:, None, None]  # (Q, dof_numel, I)
    return bm.einsum('...cqd, qdi -> ...ci', source, kernel) * measure[:, None]
```

三点设计取舍：

1. **用 `basis.shape[0] != 1` 做门，不改原路径。** 基函数逐单元的调用方（`gphi` 系列：
   `diffusion_integrator`、`scalar_mass_integrator`、`nonlinear_*`）`shape[0] == NC`，走的
   仍是上游那两行，行为逐字节不变。新路径只对 `C == 1` 生效，那正是缺陷发生的条件。
2. **保留两行原 `einsum` 的 `f''` 前缀。** 这两个字符串里没有插值，前缀是上游的冗余写法；
   保持逐字节相同能让将来与上游做三方合并时这两行不产生差异。
3. **`measure` 在收缩之后再乘。** 它是 `(C,)`，放最后是一次逐单元缩放，不引入新的中间量；
   若像上游那样先并入收缩，就又给了路径优化器广播的机会。

`bm.swapaxes` 与 `bm.einsum` 在 NumPy 与 PyTorch 后端上均已存在；其余后端未做数值验证。

#### 回归验证

**数值等价**：改动前的实现在 48 组形状组合上的输出已先行落盘为参照（`dof_shape ∈
{(), (3,), (2,3)}` × 单元轴 ∈ `{1, NC}` × 8 种 `source` 形态，含 `batched` 与 `NC = 1`
退化档），改动后逐组比对：30 组逐位相同，最差相对偏差 `3.522e-16`，全部通过。未逐位
相同的 18 组都走新路径，偏差量级即浮点求和次序变化的固有量级。

**端到端**：SOPTX 三维制造解基准
（[`experiments/matrix_free_capability/`](../../experiments/matrix_free_capability/results_analysis.md)，
`DivergenceFreePolynomialElasticity3D`、`tet` P1、`n=64`、823,875 自由度、EA 算子层级）
修复前后的进程峰值 RSS：`load` 阶段增量 `+3.248 GiB` → `+0.000 GiB`，进程峰值
`8.986 GiB` → `5.749 GiB`，相对 FA 的扣基线内存比 `1.91` → `3.02`。`load` 增量归零表示
载荷组装的峰值已完全落在刚度算子的高水位之下；CG 迭代数与真残差不变（`493` 次、`1.05e-10`）。

微基准（`NQ=20`、`I=12`、`D=3`、单元无关基、`NC=1,572,864`）上分支二耗时
`1722.6 ms` → `48.0 ms`、分支一 `7237.3 ms` → `117.2 ms`。加速比（35.9 / 61.8 倍）远高于
FLOP 比（1.34 / 1.52 倍），说明原路径的时间主要花在分配与写出那块大中间量上，而非算术。

⚠️ **尚无 pytest 覆盖。** 上述比对由一次性脚本完成，未沉淀成 fork 内的回归测试。

#### 丢弃判据

上游满足以下任一条件时，本补丁应在移植时直接删除，不做三方合并：

- 上游 `linear_integral` 自身对 `basis.shape[0] == 1` 做了分支处理；或
- 上游 `NumPyBackend.einsum` 不再无条件注入 `optimize=True`；或
- 上游 `LagrangeFESpace.basis()` 不再返回 `C = 1` 的数组。

判据脚本：在上游检出上以 `NC = 1572864` 跑上面那组 `np.einsum_path`，若两条分支的最大
中间量均等于输出本身，则缺陷已在上游消失。

---

## 五、验证环境

| 项 | 值 |
|---|---|
| 主机 | WSL2 Ubuntu 24.04，47 GiB 可用内存 |
| Python / 环境 | conda env `ihpcm`，Python 3.12.13 |
| NumPy | 2.5.1 |
| PyTorch | 2.13.0+cu130 |
| mesh / functionspace 算例 | 全 Dirichlet 线弹性制造解（Harmonic Polynomial），quad / hex / tri / tet，$p = 1 \ldots 4$ |
| functional 算例 | 3D `tet` P1，`DivergenceFreePolynomialElasticity3D`，`integration_order = 4`（`NQ = 20`） |

各补丁与上游的碰撞情况（合并 `382869aec` 时）：

| 子系统 | 碰撞 |
|---|---|
| `backend/`、`sparse/` | 无：相关文件自分叉点起上游 0 个提交，全部自动合并 |
| `solver/cg.py`、`distributed/entity_mpi.py` | 无：零冲突自动合并（上游 `cg()` 至今无 `dot_product`） |
| `solver/direct.py` | **有冲突**，手工解为保 fork 的 `sym` + 收上游的新默认值，见 [3.2](#32-mumps-直接解法暴露对称性标志-sym) 的 ⚠️ |
| `functional.py` | 无：上游 0 个未合并提交，自动合并 |
| `mesh/`、`functionspace/` | 上游 v0.4 整层重写，四条补丁的落点消失、其中两条的症状换落点重现（缺陷 8、9） |
