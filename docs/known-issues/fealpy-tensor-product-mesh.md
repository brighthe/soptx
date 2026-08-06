# 上游缺陷：FEALPy 4.0.0 mesh 重构回归

FEALPy `4.0.0-alpha` 的 mesh 重构引入五处回归，同一批判据在早期版本
（`fealpy_heliang`）上全部通过。

| # | 缺陷 | 位置 | 修复 |
| --- | --- | --- | --- |
| 1 | 重心坐标 tuple 的 `TD` 计算错误 | `functionspace/lagrange_fe_space.py:218,229`<br>`functionspace/tensor_space.py:282,293` | `TD = sum(item.shape[-1] - 1 for item in bc)` |
| 2 | 四边形 `shape_function` 输出顺序与节点错位 | `mesh/schema/classic/quadrilateral.py:67-82` | 返回前重排 `[0, 1, 3, 2]` |
| 3 | 四边形 `grad_shape_function_reference` 输出顺序与节点错位 | `mesh/schema/classic/quadrilateral.py:106-129` | 返回前重排 `[0, 2, 3, 1]` |
| 4 | 六面体 `bc_to_point` 未展平积分点维 | `mesh/schema/classic/hexahedron.py:53-69` | 返回前 `reshape` 展平方向维 |
| 5 | `face_basis`/`edge_basis` 指向单元 `shape_function` | `functionspace/lagrange_fe_space.py:150-151` | 改调 `mesh.face_shape_function` / `mesh.edge_shape_function` |

缺陷 1 影响**所有网格类型**，是复现缺陷 2–4 的前置条件。缺陷 2–4 只影响张量积
网格。缺陷 5 影响所有网格类型，但只在施加 Neumann/traction 边界时走到。

**缺陷 2 不抛异常**：残差合格、刚体位移零空间正确，只有观测收敛阶和插值一致性
能发现。在此版本上使用四边形网格会静默得到错误结果。

---

## 缺陷 1：重心坐标 tuple 的 `TD` 计算错误

**症状**：三角形 `mesh.error(exact, uh)` 抛异常：
```
ValueError: Size of label 'l' for operand 1 (6) does not match previous terms (4)
```

**根因**：`value`/`grad_value` 在 `bc` 为 tuple 时用 `len(bc)` 当拓扑维数。
单纯形被包成单元素 tuple → `len(bc) = 1`，正确值应为 `2`。张量积网格碰巧正确。

| 网格 | `bc` 形态 | `len(bc)` | 正确 `TD` |
| --- | --- | --- | --- |
| triangle | `(array(NQ, 3),)` | 1 ✗ | 2 |
| tetrahedron | `(array(NQ, 4),)` | 1 ✗ | 3 |
| quadrangle | `(array(NQ,2), array(NQ,2))` | 2 ✓ | 2 |
| hexahedron | 三个 `(NQ,2)` | 3 ✓ | 3 |

**修复**（四处相同）：
```python
if isinstance(bc, tuple):
    TD = sum(item.shape[-1] - 1 for item in bc)
```

## 缺陷 2：四边形 `shape_function` 与节点顺序错位

**症状**：四边形连线性函数插值都无法重现，误差 `8.0e-01`。

**根因**：`tensorprod(η, ξ)` 展平为 `(0,0),(1,0),(0,1),(1,1)`，而单元节点
逆时针序为 `(0,0),(1,0),(1,1),(0,1)`，第 3、4 位错开。

**修复**：返回前重排 `phi = phi[..., [0, 1, 3, 2]]`。

## 缺陷 3：四边形 `grad_shape_function_reference` 与节点顺序错位

**症状**：四边形 L2 观测阶为负（`-0.351, -0.089`），误差不降反升。

**根因**：`einsum('im,jn->ijmn', ξ, η)` 展平为 `(0,0),(0,1),(1,0),(1,1)`，
与同一文件 `shape_function` 的 `(0,0),(1,0),(0,1),(1,1)` **相反**。
`jacobi_matrix` 把未重排的 `gphi` 直接与 `positions[cell]` 相乘，`detJ` 偏小，
刚度矩阵整体缩放一个常因子 `c`，解变为 `u_h ≈ u/c`，误差与网格无关 → 观测阶趋近 0。

**修复**：返回前重排 `gphi = gphi[..., [0, 2, 3, 1], :]`。

> `jacobi_matrix` **无需改动**：它消费 `grad_shape_function_reference`，后者对齐后
> 自然正确。`bc_to_point` 在同一文件内已做了 `[0, 1, 3, 2]` 重排——两种顺序的差异
> 是已知的，只是 `shape_function` 和 `grad_shape_function_reference` 未做对应处理。

## 缺陷 4：六面体 `bc_to_point` 未展平积分点维

**症状**：六面体装配体力项抛异常：
```
RuntimeError: The dimension of the input should be smaller than 3,
              but got shape (8, 4, 4, 4, 3)
```

**根因**：`einsum("ia,jb,kc,ncbae->nkjie", ...)` 输出 `(NC, k, j, i, GD)`，
保留了三个方向的积分点维，而 `ws` 是 `(64,)`、`phi` 是 `(1, 64, …)`，均已展平。

**修复**：返回前 `reshape(result, (result.shape[0], -1, result.shape[-1]))`。

## 缺陷 5：`face_basis` / `edge_basis` 指向单元 `shape_function`

**症状**：单纯形装配 traction 边界积分抛异常：
```
ValueError: triangle shape_function expects last dimension 3, got 2
```

**根因**：`lagrange_fe_space.py:150-151` 把面/边基函数绑定为单元基函数的别名
（`face_basis = basis`）。面上的积分点是面实体的重心坐标（三角形边 2 分量，
四面体面 3 分量），而 `mesh.shape_function` 按单元维度校验，不匹配。

| 网格 | face 重心坐标 | `face_basis` 结果 |
| --- | --- | --- |
| triangle | `(3, 2)` | 抛异常 |
| tetrahedron | `(6, 3)` | 抛异常 |
| quadrangle | `(3, 2)` | 返回 `(1, 9, 4)` ✗（应为边上 2 个基） |

四边形不抛异常但返回错误结果。`mesh.face_shape_function` / `mesh.edge_shape_function`
已提供正确入口，只是未被使用。

**修复**：
```python
def face_basis(self, bc, index=_S):
    return self.mesh.face_shape_function(bc, self.p, index=index)[None, ...]

def edge_basis(self, bc, index=_S):
    return self.mesh.edge_shape_function(bc, self.p, index=index)[None, ...]
```

---

## 对 SOPTX 的影响

- 缺陷 1：影响所有网格，SOPTX 单纯形算例同样跑不通。
- 缺陷 2–4：只影响张量积网格。单纯形装配走 `entity_measure` 分支不调 `jacobi_matrix`，
  已有 evidence 不受影响。但四边形一路**静默错误**，比六面体直接崩溃更危险。
- 缺陷 5：只影响 mixed 边界模型。纯 Dirichlet 算例碰不到。受影响的是
  `src/soptx/model/` 下 `boundary_type='mixed'` 的模型和 `MixedBoundary*` 制造解。
  Hu–Zhang 混合元路径不经过 `LagrangeFESpace.face_basis`，不受影响。
- `examples/lagrange_elasticity/minimal_demo.py` 开放了张量积网格和混合边界入口，
  因此**依赖 `fealpy_stable`**：官方检出上四边形静默错误，混合边界抛异常。

---

## 环境与版本对照

| 检出 | revision | 结果 |
| --- | --- | --- |
| `fealpy` | `2f17532`（`v4.0.0-alpha-15`） | 5 项 FAIL |
| `fealpy_stable` | `fbfe39e`（缺陷 1–4 在 `0758339`） | ALL PASSED |
| `fealpy_heliang` | `18c9afe` | ALL PASSED |

Python 3.12.13，NumPy 2.5.1，backend `numpy`。

修复后的验证结果（P1/Q1，全 Dirichlet 线弹性制造解）：

```
              8×8        16×16       32×32       观测阶
triangle     3.2602e-02  8.5078e-03  2.1539e-03  1.938  1.982
quadrangle   7.8988e-03  1.9839e-03  4.9659e-04  1.993  1.998
hexahedron   4.6470e-02  1.6810e-02  4.3349e-03  1.467  1.955
```

## 验证脚本

[`reproduce_tensor_product_issue.py`](reproduce_tensor_product_issue.py) 只依赖
FEALPy 自身接口，包含五条判据（A–E），可用于判断上游是否已修复：

```bash
python -c "import sys, runpy; sys.path.insert(0, r'...\fealpy'); \
  runpy.run_path(r'...\soptx\docs\known-issues\reproduce_tensor_product_issue.py')"
```

输出 `ALL PASSED` 即表示官方版本不再需要 `fealpy_stable`。
