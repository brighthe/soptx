# 上游缺陷：mesh view 重命名 `etype` 后未同步内部调用方

FEALPy 4.0.0 的 mesh 重构把 `Mesh.quadrature_formula` 的实体选择参数从 `etype`
改名为 `name_or_topdim`，但仓库内仍有 21 处调用方按关键字传 `etype=`，这些路径
一旦被触发就抛 `TypeError`。

| # | 缺陷 | 位置 | 修复 |
| --- | --- | --- | --- |
| 1 | `quadrature_formula` 参数改名未同步内部调用方 | `mesh/view/fealpy_api.py` | 新增可选 `etype` 作为 `name_or_topdim` 的别名，非 `None` 时优先 |

---

## 缺陷 1：`etype` 关键字调用抛 `TypeError`

**症状**：

```
TypeError: Mesh.quadrature_formula() got an unexpected keyword argument 'etype'
```

**根因**：新的 mesh view 签名为

```python
def quadrature_formula(self, q, name_or_topdim="cell", qtype="legendre"):
```

而下列模块仍按旧名传参（`mesh_old/` 下 14 处使用的是自带 `etype` 形参的旧类，
不受影响）：

| 模块 | 调用点数 |
| --- | --- |
| `functionspace/scaled_monomial_space_2d.py` | 10 |
| `functionspace/scaled_monomial_space_3d.py` | 4 |
| `functionspace/parametric_lagrange_fe_space.py` | 2 |
| `fsi/coupling_interface.py` | 2 |
| `meshopt/shapeopt/{adjoint_solver,objective}.py` | 2 |
| `ml/modules/module.py` | 1 |

**修复**：在 mesh view 上接受旧参数名，而不是逐处改调用方——21 处分布在四个互不
相关的子系统里，逐处修改会与上游产生大面积冲突，而 shim 只占一行。

```python
# mesh/view/fealpy_api.py, Mesh.quadrature_formula
def quadrature_formula(self, q: int, name_or_topdim: str | int = "cell",
                       qtype: str = "legendre",
                       etype: str | int | None = None):
    target = name_or_topdim if etype is None else etype
    return self.Entity(target).quadrature_formula(q, qtype)
```

纯向后兼容，默认行为不变。

---

## SOPTX 影响范围

SOPTX 自身**没有**任何一处用 `etype=` 关键字调用 `quadrature_formula`（全部为
位置参数或 `q=`），因此不直接触发。受影响的是 SOPTX 间接经过的 fealpy 内部路径，
主要是 `ScaledMonomialSpace` 与 `ParametricLagrangeFESpace`。

---

## 回归测试

**无 pytest 覆盖。** 这是纯向后兼容 shim，丢失后受影响的是 fealpy 内部 21 处调用
方，SOPTX 自身不用 `etype=` 关键字，因此 SOPTX 测试集捕捉不到。判据见下节 grep。

## 环境与版本对照

| 检出 | 状态 |
| --- | --- |
| 上游 `suanhai/develop` (`2f17532`, `v4.0.0-alpha-15`) | 缺陷 1 存在 |
| 本地 fork | 已修复：`824dc4f` |

## 后续

这是 shim 而非根治。上游把 21 处调用方统一改名后，本补丁应在 merge 时丢弃；
判据是上游 `grep -rn "quadrature_formula(.*etype=" fealpy | grep -v /old/ | grep -v mesh_old`
返回空。
