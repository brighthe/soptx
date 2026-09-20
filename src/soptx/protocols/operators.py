"""线性算子的公共结构化契约.

本模块只描述"能被线性求解层作用"这一最小语义, 不规定算子内部是显式稀疏
矩阵还是 matrix-free 的单元级实现. 契约不持有 Mesh、FunctionSpace 或
Material.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fealpy.typing import TensorLike


@runtime_checkable
class SupportsMatmul(Protocol):
    """能被迭代解法作用的算子: 稀疏矩阵, matrix-free 算子或预条件子.

    'fa' 层级下是 CSRTensor/COOTensor, 'ea' 层级下是层层包装的
    ConstrainedOperator. 两者唯一的共性就是支持 ``@``, 所以按该协议
    标注, 而不是枚举具体类型 -- 枚举会随实现漂移.
    """

    def __matmul__(self, other: TensorLike) -> TensorLike: ...
