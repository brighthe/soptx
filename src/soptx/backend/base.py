# 移植自 brighthe/fealpy ``fealpy/backend/base.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""后端代理基类与张量类型协议.

``TensorLike`` 是各后端张量类型的统一抽象 (numpy 的 ``ndarray``、pytorch 的
``Tensor`` 在注册后端时登记为其虚拟子类); ``BackendProxy`` 是各后端代理类的基类,
按本模块的名称映射从后端库复制函数与属性.
"""

from __future__ import annotations

__all__ = ["dtype", "device", "Number", "Size", "Index", "TensorLike"]

from abc import ABCMeta
from typing import(
    Union, Optional, Dict, Tuple, Any, Type, NewType, TypeVar, overload
)

import logging

logger = logging.getLogger(__name__)

_Self = TypeVar("_Self")
_DT = TypeVar("_DT")
dtype = NewType("dtype", object)
device = NewType("device", object)
Number = Union[int, float, complex]
Size = Tuple[int, ...]
Index = Union[int, slice, "TensorLike"]


class TensorLike(metaclass=ABCMeta):
    """各后端张量类型的统一抽象, 只用于 ``isinstance`` 判断与类型标注.

    后端注册时把其张量类 (``DATA_CLASS``) 登记为本类的虚拟子类; 下列方法只声明
    接口, 不提供实现.
    """
    @property
    def dtype(self) -> dtype:
        """元素的数据类型."""
        ...
    @property
    def device(self) -> device:
        """所在设备."""
        ...
    @property
    def mT(self: _Self) -> _Self:
        """最后两轴转置后的张量."""
        ...
    @property
    def ndim(self) -> int:
        """维数."""
        ...
    @property
    def shape(self) -> Tuple[int, ...]:
        """形状."""
        ...
    @property
    def size(self) -> int:
        """元素个数."""
        ...
    @property
    def T(self: _Self) -> _Self:
        """转置后的张量."""
        ...

    def __len__(self) -> int: ...
    # 零维张量的标量转换. 已注册的后端都实现了这些方法, 因此 `float(x)`、`int(x)`
    # 以及把零维张量用作下标都是合法的.
    def __float__(self) -> float: ...
    def __int__(self) -> int: ...
    def __index__(self) -> int: ...
    def __getitem__(self: _Self, index: Union[int, _Self, slice, tuple]) -> _Self: ...
    def __setitem__(self: _Self, index: Union[int, _Self, slice, tuple], value: Any) -> None: ...
    def __eq__(self: _Self, other: Union[int, _Self]) -> _Self: ...
    def __ne__(self: _Self, other: Union[int, _Self]) -> _Self: ...
    def __lt__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __gt__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __le__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __ge__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __add__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __radd__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __iadd__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __sub__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __rsub__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __isub__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __mul__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __rmul__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __imul__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __truediv__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __rtruediv__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __itruediv__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __matmul__(self: _Self, other: _Self) -> _Self: ...
    def __pow__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    def __rpow__(self: _Self, other: Union[Number, _Self]) -> _Self: ...
    # 按位运算, 主要用于组合布尔掩码.
    def __invert__(self: _Self) -> _Self: ...
    def __and__(self: _Self, other: Union[bool, int, _Self]) -> _Self: ...
    def __rand__(self: _Self, other: Union[bool, int, _Self]) -> _Self: ...
    def __or__(self: _Self, other: Union[bool, int, _Self]) -> _Self: ...
    def __ror__(self: _Self, other: Union[bool, int, _Self]) -> _Self: ...
    def __xor__(self: _Self, other: Union[bool, int, _Self]) -> _Self: ...
    def __rxor__(self: _Self, other: Union[bool, int, _Self]) -> _Self: ...
    def __neg__(self: _Self) -> _Self: ...
    def __pos__(self: _Self) -> _Self: ...
    def __abs__(self: _Self) -> _Self: ...
    @overload
    def reshape(self: _Self, newshape: Size, /) -> _Self: ...
    @overload
    def reshape(self: _Self, *newshape: int) -> _Self: ...
    def reshape(self: _Self, *newshape) -> _Self:
        """改变形状."""
        ...


# NOTE: 下面列出的名字是什么?
#
# 这些是映射表: 键为 SOPTX 后端中属性与函数的名字, 值为后端库 (如 numpy) 中对应的名字.
#
#   - 某个函数若没有在后端子类中手动实现, 就按映射表从后端库复制过来.
#
#   - 每个映射的格式为 {target_name: source_name}: `target_name` 是在 SOPTX 中调用该
#     函数或属性所用的名字, `source_name` 是它在后端库中的原名.
#
#   - 例如映射 {'transpose': 'permute'} 表示 SOPTX 的 `transpose` 复制自后端库的
#     `permute`, 于是可以用 `backend_manager.transpose(x)` 调用 `permute`.
#
#   - 下面写出的其实是目标名. `_make_default_mapping` 以相同的源名构造默认映射;
#     各后端子类导入这些默认映射, 再按后端的实际情况修改.


def _make_default_mapping(*names: str):
    return {k: k for k in names}


# NOTE: 新增属性时只需在这里加上目标名, 再逐个后端确认是否支持;
# 必要时在后端文件中修改源名.
#
ATTRIBUTE_MAPPING = _make_default_mapping(
    # 常量
    'pi', 'e', 'nan', 'inf', 'newaxis',
    'dtype', 'device',
    # 数据类型
    'bool',
    'uint8', 'uint16', 'uint32', 'uint64',
    'int8', 'int16', 'int32', 'int64',
    'float16', 'float32', 'float64',
    'complex64', 'complex128',
)

# NOTE: 新增函数的步骤:
#
# 1. 把目标函数名加到对应类别下.
#
# 2. 在类型存根文件 (manager.pyi) 中补上该函数的类型标注 (参数与返回值).
#
# 3. 逐个后端确认是否支持, 分以下几种情况:
#
#    - 不支持: 手动实现.
#    - 支持但名字不同: 在后端文件中修改源名.
#    - 支持但参数或返回值的格式不同: 在后端子类中写包装函数.
#    - 完全相同: 无需处理.
#
FUNCTION_MAPPING = _make_default_mapping(

    ### 创建函数 ###
    # Python array API 标准 v2023.12
    'array',
    'asarray',
    'arange', 'linspace',
    'empty', 'zeros', 'ones', 'full',
    'empty_like', 'zeros_like', 'ones_like', 'full_like',
    'eye', 'meshgrid',
    'tril', 'triu',

    # 非标准
    'tensor',

    ### 数据类型函数 ###
    # Python array API 标准 v2023.12
    'astype', 'can_cast',
    'finfo', 'iinfo',
    'isdtype', 'result_type',

    ### 逐元素函数 ###
    # Python array API 标准 v2023.12
    'abs', 'acos', 'acosh', 'add', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
    'bitwise_and', 'bitwise_left_shift', 'bitwise_invert', 'bitwise_or',
    'bitwise_right_shift', 'bitwise_xor',
    'ceil', 'clip', 'conj', 'copysign', 'cos', 'cosh',
    'divide',
    'equal', 'exp', 'expm1',
    'floor', 'floor_divide',
    'greater', 'greater_equal',
    'hypot',
    'imag', 'isfinite', 'isinf', 'isnan',
    'less', 'less_equal', 'log', 'log1p', 'log2', 'log10', 'logaddexp', 'logical_and',
    'logical_not', 'logical_or', 'logical_xor',
    'maximum', 'minimum', 'multiply',
    'negative', 'not_equal',
    'positive', 'pow',
    'real', 'remainder', 'round',
    'sign', 'signbit', 'sin', 'sinh', 'square', 'sqrt', 'subtract',
    'tan', 'tanh', 'trunc',

    # 非标准
    'arcsin', 'arccos', 'arctan', 'arctan2', 'arcsinh', 'arccosh', 'arctanh',
    'power',

    ### 索引函数 ###
    # Python array API 标准 v2023.12
    'take', 'take_along_axis',

    ### 检查 ###
    # Python array API 标准 v2023.12
    # 非标准

    ### 线性代数函数 ###
    # Python array API 标准 v2023.12
    'matmul', 'matrix_transpose',
    'tensordot',
    'vecdot',
    # 非标准
    'cross',
    'dot',
    'einsum',
    'trace',

    ### 变形函数 ###
    # Python array API 标准 v2023.12
    'broadcast_arrays', 'broadcast_to',
    'concat',
    'expand_dims',
    'flip',
    'moveaxis',
    'permute_dims',
    'repeat', 'reshape', 'roll',
    'squeeze', 'stack',
    'tile',
    'unstack',
    # 非标准
    'concatenate', 'insert',
    'swapaxes', 'split', 'transpose',

    ### 查找函数 ###
    # Python array API 标准 v2023.12
    'argmax', 'argmin', 'nonzero', 'searchsorted', 'where',

    # 非标准
    'bincount', 'isin',

    ### 集合函数 ###
    # Python array API 标准 v2023.12
    'unique_all', 'unique_counts', 'unique_inverse', 'unique_values',

    # 非标准
    'setdiff1d',
    'unique',

    ### 排序函数 ###
    # Python array API 标准 v2023.12
    'argsort', 'sort',
    # 非标准
    'lexsort',

    ### 统计函数 ###
    # Python array API 标准 v2023.12
    'cumulative_sum',
    'max', 'mean', 'min',
    'prod',
    'std', 'sum',
    'var',
    # 非标准
    'cumsum', 'cumprod',

    ### 工具函数 ###
    # Python array API 标准 v2023.12
    'all', 'any',
    # 非标准
    'allclose',
    'copy',
    'size',

    ### 函数式编程 ###
    'apply_along_axis',
)

TRANSFORMS_MAPPING = _make_default_mapping(
    'grad', 'hessian', 'jvp', 'vjp', 'jacfwd', 'jacrev', 'vmap'
)


class ModuleProxy():
    """把后端库的属性与函数挂到代理类上的工具基类."""
    @classmethod
    def attach_attributes(cls, mapping: Dict[str, str], source: Any, /):
        """按 ``{目标名: 源名}`` 映射把 ``source`` 的属性复制到类上; 源名为空或不存在时跳过."""
        for target_key, source_key in mapping.items():
            if (source_key is None) or (source_key == ''):
                continue
            if hasattr(source, source_key):
                setattr(cls, target_key, getattr(source, source_key))

    @classmethod
    def attach_methods(cls, mapping: Dict[str, str], source: Any, /):
        """按 ``{目标名: 源名}`` 映射把 ``source`` 的函数作为静态方法复制到类上.

        类中已手动实现的同名方法不被覆盖; 源中不存在的函数记一条 info 日志后跳过.
        """
        for target_key, source_key in mapping.items():
            if (source_key is None) or (source_key == ''):
                continue
            if hasattr(cls, target_key):
                # 已手动实现的方法不从源复制.
                logger.debug(f"`{target_key}` already defined. "
                             f"Skip the copy from {source.__name__}.")
                continue
            if hasattr(source, source_key):
                setattr(cls, target_key, staticmethod(getattr(source, source_key)))
            else:
                logger.info(f"`{source_key}` not found in {source.__name__}. "
                            f"Method `{target_key}` remains unimplemented.")

    @classmethod
    def show_unsupported(cls, signal: bool, function_name: str, arg_name: str) -> None:
        """``signal`` 为真时警告: 本后端不支持函数 ``function_name`` 的参数 ``arg_name``, 该参数被忽略."""
        if signal:
            logger.warning(f"{cls.__name__} does not support the "
                           f"'{arg_name}' argument in the function {function_name}. "
                           f"The argument will be ignored.")


class BackendProxy(ModuleProxy):
    """所有后端代理类的基类.

    子类以 ``class XxxBackend(BackendProxy, backend_name='xxx')`` 定义, 定义时自动
    登记到可用后端表, 并把 ``DATA_CLASS`` 登记为 ``TensorLike`` 的虚拟子类.

    Raises
    ------
    ValueError
        ``backend_name`` 为空.
    """
    DATA_CLASS: Optional[Type] = None
    _available_backends: Dict[str, Type["BackendProxy"]] = {}

    def __init_subclass__(cls, backend_name: str, **kwargs):
        super().__init_subclass__(**kwargs)

        if backend_name != "":
            cls._available_backends[backend_name.lower()] = cls
            cls.backend_name = backend_name
            TensorLike.register(cls.DATA_CLASS)
        else:
            raise ValueError("Backend name cannot be empty.")

    @classmethod
    def is_tensor(cls, obj: Any, /) -> bool:
        """判断对象是否为本后端的张量类型."""
        return isinstance(obj, cls.DATA_CLASS)

    # NOTE: 本类是后端系统的基类, 不要在这里实现任何工具函数.
