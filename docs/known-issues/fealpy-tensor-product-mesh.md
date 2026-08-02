# 上游缺陷：FEALPy 4.0.0 mesh 重构回归

FEALPy `4.0.0-alpha` 的 mesh 重构引入四处回归。同一批判据在早期版本
（`fealpy_heliang`）上全部通过，因此这不是设计取舍。

缺陷 1 影响**所有网格类型**，是复现其余三处的前置条件；缺陷 2–4 只影响张量积
网格（四边形、六面体）。

| # | 缺陷 | 位置 | 修复 |
| --- | --- | --- | --- |
| 1 | 重心坐标 tuple 的 `TD` 计算错误 | `functionspace/lagrange_fe_space.py:218,229`<br>`functionspace/tensor_space.py:282,293` | `TD = sum(item.shape[-1] - 1 for item in bc)` |
| 2 | 四边形 `shape_function` 输出顺序与单元节点顺序错位 | `mesh/schema/classic/quadrilateral.py:67-82` | 输出重排 `[0, 1, 3, 2]` |
| 3 | 四边形 `grad_shape_function_reference` 输出顺序与单元节点顺序错位 | `mesh/schema/classic/quadrilateral.py:106-129` | 输出重排 `[0, 2, 3, 1]` |
| 4 | 六面体 `bc_to_point` 未展平积分点维 | `mesh/schema/classic/hexahedron.py:53-69` | 返回前展平方向维 |

缺陷 2 的四边形一路**不抛异常**：残差合格、刚度矩阵对称、刚体位移零空间正确，
只有观测收敛阶和插值一致性能发现。任何在此版本上使用四边形网格的计算都会得到
错误结果而不自知。

四处修复均已作为源码改动落在 `C:\workspace\fealpy_stable` 的 `stable` 分支
（`0758339`），全部判据通过，尚未向上游提交。`jacobi_matrix` **不需要**单独
修改：它消费 `grad_shape_function_reference`，后者对齐后自动正确。

## 环境与版本对照

| 检出 | revision | 日期 | 结果 |
| --- | --- | --- | --- |
| `fealpy` | `2f17532`（`v4.0.0-alpha-15`） | 2026-07-26 | 5 项判据 **FAIL** |
| `fealpy_stable` | `0758339`（基于 `2f17532`） | 2026-08-02 | **ALL PASSED** |
| `fealpy_heliang` | `18c9afe` | 2026-06-25 | **ALL PASSED** |

Python 3.12.13，NumPy 2.5.1，backend `numpy`。

`fealpy_stable` 是本地维护的可用版本：以 `2f17532` 为基线、只加这四处缺陷修复，
`origin` 指向个人 fork、`upstream` 指向算海仓库。SOPTX 的 conda 环境
（`xihe-fealpy`）已 editable 安装该检出。

**在纯净的 `2f17532` 上复现时，会先撞上缺陷 1。**`mesh.error` 对**三角形**就会
抛出下述异常，根本走不到张量积的判据。因此上游复现时需要先应用缺陷 1 的修复，
才能观察到缺陷 2–4 的现象。

两版的网格目录结构不同：`4.0.0-alpha` 把全部网格收敛为单个 `Mesh` 类，配
`factory.py` 工厂与重写 `__instancecheck__` 的 metaclass；早期版本是
`SimplexMesh(HomogeneousMesh)` 这样的常规类继承，并保留 `quadrangle_mesh.py`、
`hexahedron_mesh.py` 等独立文件。

## 复现与判据

[`reproduce_tensor_product_issue.py`](reproduce_tensor_product_issue.py) 只依赖
FEALPy 自身接口，判据取自有限元的基础性质，与具体 PDE 无关：

- **A** 线性函数的节点插值必须被 `basis` 精确重现
- **B** 线性函数的梯度必须被 `grad_basis` 精确重现
- **C** `Σ_q ws·detJ` 必须等于 `entity_measure`
- **D** 扭曲（非矩形）单元上 `jacobi_matrix` 必须与 `bc_to_point` 的数值微分一致
- **E** `bc_to_point` 返回 `(NC, NQ, GD)`，积分点维与 `ws` 一致

当前版本输出：

```
quadrangle
  [PASS] quadrangle E bc_to_point 形状: (4, 16, 2), 应为 (4, 16, 2)
  [FAIL] quadrangle A 插值一致性: max err = 8.013e-01
  [PASS] quadrangle B 梯度一致性: max err = 2.665e-15
  [FAIL] quadrangle C 面积一致性: sum ws*detJ = 0.1303168572, entity_measure = 0.2500000000

hexahedron
  [FAIL] hexahedron E bc_to_point 形状: (8, 4, 4, 4, 3), 应为 (8, 64, 3)
  [PASS] hexahedron A 插值一致性: max err = 2.665e-15
  [PASS] hexahedron B 梯度一致性: max err = 3.553e-15
  [PASS] hexahedron C 面积一致性: sum ws*detJ = 0.1250000000, entity_measure = 0.1250000000

distorted quadrangle
  [FAIL] D jacobi_matrix @ (0.25, 0.4): |det| = 0.3810000000, 真值 = 0.9419999999, max|J - J_num| = 8.00e-01
  [FAIL] D jacobi_matrix @ (0.7, 0.6): |det| = 0.4660000000, 真值 = 0.9600000001, max|J - J_num| = 1.14e+00

FAILED (5)
```

三角形与四面体全部通过，两个版本一致。

两条诊断经验，都是踩过坑得来的：

**判据 D 不可省略。**均匀矩形网格会掩盖节点错位——错误的重排 `[0, 1, 3, 2]` 在
矩形单元上恰好给出正确的 `detJ = 0.0625`，只有扭曲单元能把它区分开。诊断过程中
曾据此得出错误结论。

**不要用 `J` 是否为对角阵作判据。**参考坐标与物理坐标的轴对应关系可以是交换的，
修复后 `J` 为反对角矩阵，`|det J|` 才是有意义的量。

## 缺陷 1：重心坐标 tuple 的 `TD` 计算错误

**影响所有网格类型，包括单纯形。**这是复现缺陷 2–4 的前置条件。

**症状**：对三角形网格调用 `mesh.error(exact, uh, q=...)` 直接抛异常。

```
mesh/view/entity_view.py:172  integrand → v2 = f2(bcs)
  → functionspace/function.py:19   space.value(self.array, bcs, index=index)
  → functionspace/tensor_space.py:287
        val = bm.einsum('cql..., cl... -> cq...', phi, uh[e2dof, ...])
ValueError: Size of label 'l' for operand 1 (6) does not match previous terms (4)
```

三角形 P1 张量空间的 `ldof = 3 节点 × 2 分量 = 6`，而 `phi` 的对应维是 4。

**根因**：`value` 与 `grad_value` 在 `bc` 为 tuple 时用元素个数当拓扑维数：

```python
if isinstance(bc, tuple):
    TD = len(bc)              # 错
else:
    TD = bc.shape[-1] - 1
```

`mesh.error` 会把**单纯形**的 `bcs` 也包装成长度为 1 的 tuple 再传入，此时
`len(bc) = 1`，而正确的拓扑维数应为 `2`。

| 网格 | `bc` 形态 | `len(bc)` | `sum(item.shape[-1]-1)` | 正确 `TD` |
| --- | --- | --- | --- | --- |
| triangle | `(array(NQ, 3),)` | `1` ✗ | `2` ✓ | `2` |
| tetrahedron | `(array(NQ, 4),)` | `1` ✗ | `3` ✓ | `3` |
| quadrangle | `(array(NQ,2), array(NQ,2))` | `2` ✓ | `2` ✓ | `2` |
| hexahedron | 三个 `(NQ,2)` | `3` ✓ | `3` ✓ | `3` |

`len(bc)` 只在张量积网格上碰巧正确——每个方向贡献一维。单纯形被包成单元素
tuple 后就错了。

### 修复

`lagrange_fe_space.py:218,229` 与 `tensor_space.py:282,293`，四处相同：

```python
if isinstance(bc, tuple):
    TD = sum(item.shape[-1] - 1 for item in bc)
```

对两类网格都成立：重心坐标每段的最后一维减一即为该段贡献的拓扑维数。

## 缺陷 2：四边形 `shape_function` 与单元节点顺序错位

**症状**：四边形连线性函数的节点插值都无法重现，误差 `8.0e-01`。

`quadrilateral.py:67-82`：

```python
return bm.tensorprod(
    bm.simplex_shape_function(bcs[1], p[1], mi1),   # η 在外
    bm.simplex_shape_function(bcs[0], p[0], mi0),   # ξ 在内
)
```

展平顺序为 `(0,0),(1,0),(0,1),(1,1)`，而单元节点按逆时针存储
`(0,0),(1,0),(1,1),(0,1)`，第 3、4 位错开。

调用链（`p=1`）：

```
LagrangeFESpace.basis            functionspace/lagrange_fe_space.py:146
  → mesh.shape_function(bc, p)   mesh/view/fealpy_api.py:153
  → Entities(-1)[0].shape_function(...)   mesh/view/entity_view.py:412
  → self.schema.shape_function(bcs, p)    mesh/view/entity_view.py:446
```

后果是载荷向量在节点间错配。注意**总载荷不变**，所以 `sum(F)` 一类的检验无法
发现它——诊断早期「四边形 `sum(F)` 与三角形一致」曾被误当作载荷正确的证据。

## 缺陷 3：四边形 `grad_shape_function_reference` 与单元节点顺序错位

**症状**：四边形 L2 误差随网格加密反而增大，观测收敛阶为负，提高积分阶无效。

```
                8×8         16×16       32×32        观测阶
triangle     3.2602e-02  8.5078e-03  2.1539e-03    1.938  1.982
quadrangle   2.3950e-01  3.0546e-01  3.2480e-01   -0.351 -0.089
```

`quadrilateral.py:106-129` 用 `einsum('im,jn->ijmn', ξ, η)`，展平顺序为
`(0,0),(0,1),(1,0),(1,1)`——与同一文件 `shape_function` 的顺序**相反**。

`jacobi_matrix`（`quadrilateral.py:199-212`）直接把它与未重排的
`positions[cell]` 相乘：

```python
gphi = cls.grad_shape_function_reference(bcs, p=(1, 1))
J = bm.einsum('cim, qin -> cqmn', node[cell], gphi)
```

在 `[0, 0.25]²` 的单元上 `J` 应为 `diag(0.25, 0.25)`，实得
`[[0.21528408, 0.21528408], [0.25, 0]]`，`detJ = 0.05382102`（应为 `0.0625`），
`Σ_q ws·detJ = 0.0325792143`（应为 `0.0625`）。

`jacobi_matrix` 还被 `mesh/schema/entity_schema.py:146` 的
`grad_shape_function_cartesian` 用来把参考梯度转成物理梯度，因此
`grad_basis(variable='x')` 同样受影响。

**为什么表现为不收敛**：装配用 `detJ` 作积分权重，`detJ` 偏小使刚度矩阵整体
缩放一个常数因子 `c`，解变为 `u_h ≈ u/c`。误差 `‖u_h − u‖ ≈ ‖u‖·|1/c − 1|`
与网格无关，观测阶因而趋近 0。实测误差 `2.40e-01 → 3.05e-01 → 3.25e-01` 正在
收敛到一个常数，与该分析一致。

### 重排的确定方法

在**扭曲四边形**上以 `bc_to_point` 的数值微分为真值，穷举全部 24 种置换，只有
一种在两个参考点上同时命中：

| 参考点 | 真值 `\|det\|` | 未重排 | `[0,1,3,2]` | `[0,2,3,1]` |
| --- | --- | --- | --- | --- |
| `(0.25, 0.4)` | `0.9420` | `0.3810` | `0.9090` | `0.9420` ✓ |
| `(0.7, 0.6)` | `0.9600` | `0.4660` | `0.9820` | `0.9600` ✓ |

### 同一文件内已有正确处理

`bc_to_point`（`quadrilateral.py:58-65`）对同一件事**做了**重排：

```python
points = ctx.block.positions[quad[:, [0, 1, 3, 2]]]
```

六面体的 `bc_to_point`（`hexahedron.py:65-66`）还留了注释：

```python
# Contract order is bottom/top cyclic; tensor-product order is different.
points = ctx.block.positions[cell[:, [0, 1, 3, 2, 4, 5, 7, 6]]]
```

可见两套顺序的差异是已知的，只是 `shape_function` 与
`grad_shape_function_reference` 未做对应处理。

### 六面体的 `jacobi_matrix` 没有问题

六面体 `jacobi_matrix`（`hexahedron.py:149-161`）虽然同样直接使用
`positions[cell]` 而未重排，但其参考基函数编号恰好与单元节点顺序一致，用角点
求值法推导出的重排是恒等置换，实测 `detJ = 0.125` 与 `entity_measure` 相符。
此处**不需要**修改。

## 缺陷 4：六面体 `bc_to_point` 未展平积分点

**症状**：六面体装配体力项直接抛异常。

```
fealpy/functional.py:61  linear_integral
  → fealpy/utils/utils.py:104  fill_axis
RuntimeError: The dimension of the input should be smaller than 3,
              but got shape (8, 4, 4, 4, 3)
```

`hexahedron.py:53-69` 末行的 einsum 输出保留了三个方向的积分点维：

```python
u, v, w = bcs
return bm.einsum("ia,jb,kc,ncbae->nkjie", u, v, w, points)   # (NC, k, j, i, GD)
```

而 `ws` 是 `(64,)`、`phi` 是 `(1, 64, …)`，均已展平。四边形的实现
（`quadrilateral.py:58-65`）在最后一步就合成了单一积分点维。

| 网格 | `ws` | `bc_to_point`（4.0.0-alpha） | `bc_to_point`（早期版本） |
| --- | --- | --- | --- |
| triangle | `(10,)` | `(8, 10, 2)` | `(8, 10, 2)` |
| quadrangle | `(16,)` | `(4, 16, 2)` | `(4, 16, 2)` |
| hexahedron | `(64,)` | `(8, 4, 4, 4, 3)` ✗ | `(8, 64, 3)` ✓ |

## 统一修复方案

缺陷 1 独立修复（见上）。缺陷 2–4 的思路是把 schema 的输出**统一对齐到单元节点
的逆时针序**，而不是在各消费者处分别重排：

```python
# lagrange_fe_space.py / tensor_space.py  value 与 grad_value        缺陷 1
TD = sum(item.shape[-1] - 1 for item in bc)

# quadrilateral.py  shape_function 返回前                            缺陷 2
phi = phi[..., [0, 1, 3, 2]]

# quadrilateral.py  grad_shape_function_reference 返回前             缺陷 3
gphi = gphi[..., [0, 2, 3, 1], :]

# hexahedron.py  bc_to_point 返回前                                  缺陷 4
result = bm.reshape(result, (result.shape[0], -1, result.shape[-1]))
```

三点说明：

1. **`jacobi_matrix` 无需改动**。它消费 `grad_shape_function_reference`，后者
   对齐后 `einsum('cim,qin->cqmn', positions[cell], gphi)` 两侧顺序自然一致。
   反过来若只在 `jacobi_matrix` 内重排 `node`，则只修了一个消费者，
   `grad_shape_function_cartesian` 等仍是错的。
2. 展平顺序 `(k, j, i)` 与 `TensorProductQuadrature` 的 `ws` 顺序一致，无需转置。
3. 两处重排**只对 `p=1` 验证过**，高阶元需按同一约定重新推导。判断是否为一阶元
   请用形状（`phi.shape[-1] == 4`）而非 `p == (1, 1)`：`entity_view.py:444` 会把
   标量 `p=1` 包成 `(1,)`，用后者写的补丁会静默失效。

### 必须整套施加

只修其中一处会让**互相抵消的错误被拆开**。仅修 `jacobi_matrix` 时，四边形的
梯度一致性反而从 `2.7e-15` 恶化到 `4.7e+00`，端到端观测阶从 `-0.351, -0.089`
变为 `0.089, 0.011`——误差下降但依然不收敛。

### 验证结果

三处补丁一起施加后全部判据通过：

| 判据 | triangle | quadrangle | hexahedron |
| --- | --- | --- | --- |
| D 扭曲单元 `J` vs 数值微分 | — | `5.7e-11` ✓ | — |
| C `Σ_q ws·detJ` vs `entity_measure` | 不适用¹ | `0.25` = `0.25` ✓ | `0.125` = `0.125` ✓ |
| A 插值一致性 | `8.9e-16` ✓ | `8.9e-16` ✓ | `2.7e-15` ✓ |
| B 梯度一致性 | `0.0` ✓ | `1.8e-15` ✓ | `3.6e-15` ✓ |

¹ 单纯形的 `detJ` 等于 `d!×测度`，且装配走 `entity_measure` 分支不使用 `detJ`，
该判据对单纯形不适用。

线弹性制造解（P1/Q1，全 Dirichlet）的 L2 收敛阶：

```
              8×8/2³      16×16/4³    32×32/8³      观测阶
triangle     3.2602e-02  8.5078e-03  2.1539e-03   1.938  1.982
quadrangle   7.8988e-03  1.9839e-03  4.9659e-04   1.993  1.998
hexahedron   4.6470e-02  1.6810e-02  4.3349e-03   1.467  1.955
```

四边形从 `-0.351, -0.089` 恢复到 `1.993, 1.998`，六面体从崩溃恢复到 `1.955`，
均趋向理论阶 2。四边形误差小于同规模三角形，与 Q1 优于 P1 的预期一致。

## 建议纳入回归测试

判据 A–E 与具体 PDE 无关，成本很低，可直接作为网格层的回归测试。其中 D 需要一个
扭曲单元，建议固定使用：

```python
node = [[0.0, 0.0], [1.0, 0.2], [1.3, 1.1], [0.1, 0.9]]
```

---

以上为可直接提交上游的技术事实。以下为 SOPTX 本地专有内容。

## 对 SOPTX 的影响范围

- **缺陷 1 影响单纯形网格**，任何经 `mesh.error` 计算误差的代码都会撞上，
  包括 SOPTX 的全部二维、三维算例。**在未修复的 FEALPy 上，SOPTX 的单纯形算例
  同样跑不通**，不只是四边形、六面体。
- 缺陷 2–4 只影响四边形与六面体。SOPTX 的装配对单纯形走 `entity_measure`
  分支，不调用 `jacobi_matrix`，故现有 evidence 的**数值结论**不受其影响。
- 四边形一路是静默的错误结果，比六面体的直接崩溃更危险。
- **evidence 的环境记录需要注意**：`report.py` 记录的 `fealpy` 版本号是
  `4.0.0`，官方检出与 `fealpy_stable` 无法据此区分，而两者的单纯形结果一个跑不
  通、一个正常。重放前需确认环境实际指向哪个检出；`fealpy_stable` 的
  revision 可作为 provenance 的补充记录。

## SOPTX 侧的处置

- `examples/lagrange_elasticity/minimal_demo.py` 与
  `examples/matrix_free_elasticity` 目前只使用单纯形网格。
- **放开四边形、六面体入口的判据**：`reproduce_tensor_product_issue.py` 在目标
  FEALPy 检出上全部 PASS。`fealpy_stable` 已满足该判据，官方 `fealpy` 未满足。
  若为 SOPTX 增加张量积网格入口，需同时明确它依赖 `fealpy_stable`——在官方
  检出上这些入口会静默给出错误结果，而 SOPTX 无法在自己这一层判断 `detJ`
  是否可信。
- 顺带记录一处 SOPTX 自身的隐患（当前未致错，但口径应统一）：刚度装配对非单纯
  形用 `detJ`（`src/soptx/fem/integrators/linear_elastic_integrator.py:77`），
  体力装配始终用 `cm`（`src/soptx/fem/integrators/source_integrator.py:50`）。
  即便上游修复，同一个积分由两条不同口径加权仍应收敛到一处。

## 维护 fealpy_stable

`fealpy_stable` 是**过渡性**的：这些缺陷会逐条上报给 FEALPy 维护者，上游修复后
对应的本地改动即可撤除，最终两个检出归于一致，`fealpy_stable` 退役。上游修复
周期较长，故先以本地分支支撑研究工作。

三个检出的分工：

| 路径 | 分支 | 定位 |
| --- | --- | --- |
| `C:\workspace\fealpy` | `develop` | 官方状态，保持纯净，不做本地修改 |
| `C:\workspace\fealpy_stable` | `stable` | 官方基线 + 四处缺陷修复，SOPTX 的 conda 环境指向它 |

`fealpy_stable` 配了三个 remote：`origin` → 个人 fork，`upstream` → 算海仓库，
`local` → `C:\workspace\fealpy`（用于与本地官方检出对比）。

### 查看两者差异

```bash
git -C C:\workspace\fealpy_stable fetch local
git -C C:\workspace\fealpy_stable diff --stat local/develop stable
git -C C:\workspace\fealpy_stable log --oneline local/develop..stable
```

`stable` 始终保持「官方基线 + 若干修复提交」的形态，因此 `log` 的输出就是尚未
被上游接受的全部改动。

### 跟进上游更新

```bash
git -C C:\workspace\fealpy pull
git -C C:\workspace\fealpy_stable fetch local
git -C C:\workspace\fealpy_stable rebase local/develop stable
```

`git rebase <新基线> <要搬的分支>` 的双参数形式：把 `stable` 上不属于
`local/develop` 的提交摘下来，重新贴到前进后的 `local/develop` 之上（执行时会
自动切到 `stable`）。这样 `stable` 始终是「最新官方 + 本地修复」，而不会停留在
旧基线上。

维护者修好某一处后，本地对应改动在 rebase 时会变成空提交，git 提示
`The previous cherry-pick is now empty`，`git rebase --skip` 跳过即可，差异自动
缩小。若上游采用了不同的修法则会冲突，此时采纳上游版本、删除本地改动。

两点后果：

- **rebase 会改写提交 hash**。本文档记录的 `stable` revision（`0758339`）在第一次
  rebase 后即失效，需同步更新「环境与版本对照」表。
- 若 `stable` 已 push 到 `origin`，rebase 后需要
  `git push --force-with-lease`，因为历史被改写。

当 `log local/develop..stable` 为空时，两者已一致，`fealpy_stable` 可以退役：
把 conda 环境改回 `pip install -e C:\workspace\fealpy` 即可。

### 判断上游是否已修复

不必读代码，直接用判据脚本测官方检出：

```bash
python -c "import sys, runpy; sys.path.insert(0, r'C:\workspace\fealpy'); runpy.run_path(r'C:\workspace\soptx\docs\known-issues\reproduce_tensor_product_issue.py', run_name='__main__')"
```

输出 `ALL PASSED` 即表示官方版本已不再需要 `fealpy_stable`。同一条命令把路径换成
`C:\workspace\fealpy_stable` 可以验证本地修复仍然有效。
