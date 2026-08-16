# 上游缺陷：backend 层类型标注窄于运行期真实能力

`bm` 门面的静态签名由手写的 `backend/manager.pyi` 提供，`TensorLike` 则是一个只
用于 `register()` 的 ABC（`backend/base.py`）。两者都比运行期真实能力窄：写对的
调用被 Pyright 判为错误。这不影响运行，但把 IDE 的红波浪线变成噪声——真错误淹没
在假阳性里，等于没有静态检查。

在 SOPTX 全量扫描（`pyright src/soptx`）中，这类假阳性有 **681 条**，占总诊断数的
21%（3200 → 2519）。

| # | 缺口 | 位置 | 修复 | 消除的假阳性 |
| --- | --- | --- | --- | --- |
| 1 | `set_at`/`add_at` 的 `indices` 只接受张量 | `manager.pyi` | 新增索引别名 `_IT` | 335 |
| 2 | `TensorLike` 缺位运算 dunder（`~`、`&`、`\|`、`^`） | `base.py` | 补 7 个 dunder 声明 | 247 |
| 3 | 逐元素二元运算不接受标量操作数 | `manager.pyi` | 新增操作数别名 `_OT`，改 28 个二元函数与 `clip` | 46 |
| 4 | `TensorLike` 缺 `__float__`/`__int__`/`__index__` | `base.py` | 补 3 个 dunder 声明 | 34 |
| 5 | `BackendManager` 未声明 `random`/`linalg` 子模块，`LinalgModule` 缺 `norm` | `manager.pyi` | 声明两个子模块属性并补 `norm` | 30 |
| 6 | `from_dlpack`/`matrix_transpose`/`cumulative_sum` 三个声明漏写 `self` | `manager.pyi` | 三处各补 `self` | 未计入下述统计 |

缺口 6 是后续在 `src/soptx/fem/substructure/condensation.py` 用到 `bm.matrix_transpose`
时才触发的，上表 681 条的统计早于该补丁，不含它；它与缺口 1–5 性质也不同（是声明
笔误而非标注偏窄），详见下文。

余下约 18 条是连带消除的（表达式类型从 `Unknown` 变确定后，元组解包与 `einsum`
的报错一并消失）。全部改动只影响类型检查：`.pyi` 不参与运行，`base.py` 里新增的
方法体是 `...`，而 `TensorLike` 没有任何子类（`grep -rn "class .*(TensorLike"`
为空），后端类型是通过 `TensorLike.register()` 挂上去的，不进 MRO，因此这些声明
在运行期不可达。

---

## 缺口 1：`set_at`/`add_at` 的索引只接受张量

**症状**：

```
无法将“tuple[EllipsisType, Literal[1]]”类型的参数分配给函数“set_at”中类型为
“TensorLike”的参数“indices” (reportArgumentType)
```

**根因**：上游签名把索引位声明成了张量本身。

```python
def set_at(self, x: _DT, indices: _DT, src: Union[_DT, Number, bool], /) -> _DT: ...
```

但各后端的实现走的是 NumPy 风格花式索引，整数、切片、`Ellipsis`、`None` 以及它们
组成的元组全都合法。SOPTX 里 `bm.set_at(value, (..., 1), traction_y)`、
`bm.set_at(rhs, cell, ...)`、`bm.set_at(uh_bd, ~isBdDof, 0.0)` 都属于此类。

**修复**：新增索引别名，只改这两个函数的 `indices` 位，`src` 与返回值不动。

```python
# manager.pyi
_IT = Union[_DT, int, slice, EllipsisType, None, Tuple[Any, ...], List[Any]]

def set_at(self, x: _DT, indices: _IT, src: Union[_DT, Number, bool], /) -> _DT: ...
def add_at(self, x: _DT, indices: _IT, src: Union[_DT, Number], /) -> _DT: ...
```

`index_add`/`scatter`/`scatter_add` 的 `index` 是真正的索引张量，不在此列，保持原样。

---

## 缺口 2：`TensorLike` 缺位运算 dunder

**症状**：

```
类型“TensorLike”和“TensorLike”不支持运算符“&” (reportOperatorIssue)
预期类型为 "TensorLike"时，类型“TensorLike”不支持运算符“~"
```

**根因**：`TensorLike` 声明了比较、算术、矩阵乘的 dunder，唯独没有位运算。而布尔
掩码的合并与取反（`~isBdDof`、`is_left & is_bottom`）是有限元代码里最常见的写法。

**修复**：补齐 7 个声明，右操作数按各后端实际支持的形式放宽到 `bool | int | 自身`。

```python
# base.py, class TensorLike
def __invert__(self: _Self) -> _Self: ...
def __and__(self: _Self, other: Union[bool, int, _Self]) -> _Self: ...
def __rand__(self: _Self, other: Union[bool, int, _Self]) -> _Self: ...
def __or__(self: _Self, other: Union[bool, int, _Self]) -> _Self: ...
def __ror__(self: _Self, other: Union[bool, int, _Self]) -> _Self: ...
def __xor__(self: _Self, other: Union[bool, int, _Self]) -> _Self: ...
def __rxor__(self: _Self, other: Union[bool, int, _Self]) -> _Self: ...
```

---

## 缺口 3：逐元素二元运算不接受标量操作数

**症状**：

```
无法将“float”类型的参数分配给函数“divide”中类型为“TensorLike”的参数“x2”
无法将“Literal[0]”类型的参数分配给函数“clip”中类型为“TensorLike”的参数“x_min”
```

**根因**：Element-wise 段的二元函数一律写成 `(self, x1: _DT, x2: _DT, /)`。Python
Array API 标准确实规定这些函数的实参是数组，但 fealpy 绑定的每一个后端都对 Python
标量做广播，`bm.divide(x, hx)`、`bm.clip(idx, 0, nx - 1)` 是全仓库通行的写法。签名
按标准写、实现按后端跑，中间这道缝就落在调用方头上。

**修复**：新增操作数别名，替换 Element-wise 段的 28 个二元函数（`add`、`subtract`、
`multiply`、`divide`、`pow`、`maximum`、`minimum`、比较与逻辑运算等）以及 `clip`
的上下界。`linalg` 子模块内同形状的签名（`matmul`、`solve`、`vecdot` 等）**不改**，
那里的操作数确实必须是张量。

```python
# manager.pyi
_OT = Union[_DT, Number, bool]

def divide(self, x1: _OT, x2: _OT, /) -> _DT: ...
def clip(self, x: _DT, x_min: Optional[_OT], x_max: Optional[_OT], /) -> _DT: ...
```

返回类型保持 `_DT` 不变。

---

## 缺口 4：`TensorLike` 缺标量转换 dunder

**症状**：

```
“TensorLike”与协议“SupportsFloat”不兼容 —— “__float__”不存在
“TensorLike”与协议“SupportsIndex”不兼容 —— “__index__”不存在
```

**根因**：`float(bm.linalg.norm(r))`、`int(bm.sum(mask))` 这类零维张量取标量的写法，
在 `TensorLike` 上找不到对应声明。

**修复**：

```python
# base.py, class TensorLike
def __float__(self) -> float: ...
def __int__(self) -> int: ...
def __index__(self) -> int: ...
```

---

## 缺口 5：`random`/`linalg` 子模块与 `linalg.norm` 未声明

**症状**：

```
无法访问类“BackendManager”的属性“linalg” —— 属性“linalg”未知
无法访问类“LinalgModule”的属性“norm”
```

**根因**：`bm.linalg` 与 `bm.random` 由 `BackendManager.__getattr__` 转发到当前
后端（`manager.py:63`），静态分析器不解析 `__getattr__`，而 `.pyi` 里虽然定义了
`RandomModule`/`LinalgModule` 两个类，却没有把它们挂到 `BackendManager` 上。
`LinalgModule` 另外按 Array API 只声明了 `vector_norm`/`matrix_norm`，而 numpy、
pytorch、jax 三个后端都保留着 `norm`，SOPTX 用的正是 `norm`。

**修复**：

```python
# manager.pyi, class BackendManager
### submodules ###
random: RandomModule
linalg: LinalgModule

# manager.pyi, class LinalgModule
def norm(self, x: _DT, ord: Union[int, float, str, None] = None,
         axis: Axes = None, keepdims: bool = False) -> _DT: ...
```

`norm` 的 `ord` 在各后端都是位置或关键字皆可，因此不加 `/`——SOPTX 里存在
`bm.linalg.norm(d, bm.inf)` 这种位置传法。

`SparseModule` 类同样存在，但运行期 `hasattr(bm, 'sparse')` 为 `False`，**不声明**：
声明一个运行期不存在的属性只会把假阳性换成假阴性。

---

## 缺口 6：三个方法声明漏写 `self`，被 Pyright 当成绑定错误

**症状**：

```
无法访问 "BackendManager" 类的 "matrix_transpose" 属性
  无法绑定 "matrix_transpose" 方法, 因为 "BackendManager" 不能赋值给 "x" 参数
    "BackendManager" 与 "TensorLike" 不兼容
```

与缺口 1–5 不同，这一条不是「标注偏窄」，而是**声明写错**：属性本身存在，但第一个
参数位没有 `self`，Pyright 于是把 `bm` 本体绑进 `x`，报的是参数类型不兼容。因此它
无法用「放宽类型」消除，只能补 `self`。

**根因**：`BackendManager` 里有三个方法声明漏写 `self`，与相邻声明的写法不一致——
例如同一段里的 `matmul`、`tensordot`、`vecdot` 都带 `self`：

```python
# manager.pyi, class BackendManager —— 修复前
def from_dlpack(x: Any, /) -> _DT: ...                                   # 153
def matrix_transpose(x: _DT, /) -> _DT: ...                              # 261
def cumulative_sum(x: _DT, /, *, axis: Optional[int] = None, ...) -> _DT: ...  # 321
```

`matrix_transpose` 在同一个 `.pyi` 的 `BackendProxy`（第 57 行）里写的是
`def matrix_transpose(self, x: _DT, /)`，可见 `BackendManager` 侧是笔误。

**修复**：三处各补一个 `self`。

```python
# manager.pyi, class BackendManager
def from_dlpack(self, x: Any, /) -> _DT: ...
def matrix_transpose(self, x: _DT, /) -> _DT: ...
def cumulative_sum(self, x: _DT, /, *, axis: Optional[int] = None,
                   dtype=None, include_initial: bool = False) -> _DT: ...
```

三者在运行期都可用，走的是与 `matmul` 完全相同的 `__getattr__` 转发路径，补 `self`
与运行期语义一致：

- `matrix_transpose`：numpy 后端取 `np.matrix_transpose`（NumPy ≥ 2.0；本机 2.5.1），
  pytorch 后端在 `pytorch_backend.py:161` 显式实现为 `x.transpose(-1, -2)`；
- 三者均登记在 `backend/base.py` 的标准函数名单中。

一键复查是否还有同类漏写：

```bash
awk '/^class /{cls=$0} /^    def /{ if ($0 !~ /\((self|cls)[,)]/) print NR": "cls" | "$0 }' \
    ~/workspace/fealpy/fealpy/backend/manager.pyi
```

修复后应无输出。

---

## SOPTX 影响范围

纯静态。运行期行为逐条比对无变化，冒烟检查：

```bash
python -c "
from fealpy.backend import backend_manager as bm
from fealpy.backend.base import TensorLike
x = bm.zeros((3,), dtype=bm.float64)
print(float(bm.sum(x)), isinstance(x, TensorLike))
"
```

诊断数对照（`pyright src/soptx`，同一检出、同一 `pyrightconfig.json`）：

| 检出 | 诊断总数 |
| --- | --- |
| 打补丁前 | 3200 |
| 打补丁后 | 2519 |

差值 681 条全部是假阳性；补丁没有在任何新位置产生诊断（按 `(文件, 行号)` 比对，
新增位置为 0），29 条措辞变化的诊断都落在补丁前就已经报错的行上，那些是 SOPTX
自身的 `Optional` 未收窄等真问题。

---

## 回归测试

**无 pytest 覆盖，也不该有。** 类型标注不产生运行期行为，pytest 断言不到；能检查
它的是类型检查器本身。判据用上面的诊断数对照，或直接看本仓库两个入口文件：

```bash
cd ~/workspace/soptx
pyright src/soptx/fem/boundary_loads.py \
        experiments/huzhang_topopt_paper/fixed_fixed_beam.py \
        experiments/huzhang_topopt_paper/run.py
# 期望：0 errors
```

补丁丢失后这三个文件（SOPTX 侧代码不变）会重新报错，主要落在 `bm.set_at` 的元组
索引、`bm.clip` 的标量上下界和 `bm.linalg` 上——即缺口 1、3、5。

---

## 环境与版本对照

| 检出 | 状态 |
| --- | --- |
| 上游 `suanhai/develop` (`7b76f0d`, `v4.0.1-1`) | 缺口 1–6 全部存在 |
| fork `main` (`824dc4f`) | 与上游逐字节相同——本补丁是这两个文件的首次改动 |
| 本地 fork 工作区 | **已修复，尚未提交**：`fealpy/backend/base.py`、`fealpy/backend/manager.pyi` |

提交前本页的补丁 SHA 一栏为空，`README.md` 的补丁清单同样标记为「工作区」。提交
后需要回填 SHA。

## 丢弃判据

六条都是**补齐**而非行为修改，上游不会天然产生冲突（除非上游自己重写这两个文件）。
逐条判据：

| 缺口 | 丢弃条件 |
| --- | --- |
| 1、3、5 | 上游 `manager.pyi` 自带 `_IT`/`_OT` 等价放宽，且 `BackendManager` 已声明 `random`/`linalg` |
| 2、4 | 上游 `base.py` 的 `TensorLike` 自带位运算与标量转换 dunder |
| 6 | 上游 `manager.pyi` 的三个声明已带 `self` |

一键检查：

```bash
git -C ~/workspace/fealpy show suanhai/develop:fealpy/backend/base.py | grep -c "__invert__\|__float__"
git -C ~/workspace/fealpy show suanhai/develop:fealpy/backend/manager.pyi | grep -c "linalg: LinalgModule"
git -C ~/workspace/fealpy show suanhai/develop:fealpy/backend/manager.pyi | grep -c "def matrix_transpose(self"
```

三条都返回非零时，本补丁可整体丢弃。
