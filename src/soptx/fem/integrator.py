# 移植自 brighthe/fealpy ``fealpy/fem/integrator.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""积分子基类、类型标记类与组合工具."""

from typing import (
    Union, Optional, Any, TypeVar, Tuple, List, Dict, Callable,
    Generic, Protocol, overload
)

from ..typing import Index
from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..decorator.variantmethod import VariantMeta

import logging

logger = logging.getLogger(__name__)


__all__ = [
    'Integrator',
    'NonlinearInt',
    'LinearInt',
    'OpInt',
    'SrcInt',
    'CellInt',
    'FaceInt',
    'ConstIntegrator',
    'GroupIntegrator'
]

class Mesh(Protocol):
    """积分子对网格的最小接口要求, 仅用于类型标注."""
    def entity(self, etype: Union[int, str]) -> TensorLike:
        """返回某类实体的顶点编号数组, 第 0 轴为实体; ``Integrator.size`` 取其长度."""
        ...

Self = TypeVar('Self')
Space = TypeVar("Space")
INdex = Union[int, slice, Tuple[int, ...], TensorLike]
_SpaceGroup = Union[Space, Tuple[Space, ...]]
_OpIndex = Optional[Index]
_Region = Union[Callable[[Mesh], TensorLike], TensorLike, None]


def enable_cache(func: Self) -> Self:
    """按 ``space`` 参数缓存方法结果的装饰器.

    适用于取空间基函数、实体测度等积分素材的方法: 系数或源项改变后重新装配时,
    这些素材无需重算. 缓存键为 ``(方法名, id(space))``, 只在 ``indices`` 为 None
    且积分子开启了 ``keep_data`` 时生效.

    用 ``Integrator.keep_data(True)`` 开启缓存.
    """
    def wrapper(integrator_obj, space, /, indices=None) -> TensorLike:
        """有缓存时直接返回, 否则计算; 给出 ``indices`` 时不使用缓存."""
        if (indices is None) and (integrator_obj._keep_data):
            assert hasattr(integrator_obj, '_cache')
            _cache = integrator_obj._cache
            key = (func.__name__, id(space))

            if key in _cache:
                return _cache[key]

            data = func(integrator_obj, space)
            _cache[key] = data

            return data
        else:
            if indices is None:
                return func(integrator_obj, space)
            return func(integrator_obj, space, indices)

    return wrapper


class Integrator(metaclass=VariantMeta):
    """积分子基类.

    积分子在给定空间上对被积函数做积分, 输出实体上的张量: 第 0 轴为实体, 第 1 轴
    为各局部自由度, 其后可以有附加维.

    积分子有 "区域" (``region``) 的概念, 以网格实体编号指定积分范围, 输出的第 0 轴
    长度即区域内的实体数.

    子类须实现两个方法::

        def to_global_dof(space, /, indices=None): ...
        def assembly(space, /, indices=None): ...

    前者给出局部自由度到全局自由度的映射, 后者计算积分. ``indices`` 用于在积分区域
    内再选一个子集, 最终参与积分的实体编号在方法内用
    ``index = self.entity_selection(indices)`` 取得. 注意 SOPTX 的具体积分子均未
    实现 ``indices`` 参数.

    Parameters
    ----------
    keep_data : bool, optional
        是否开启 ``enable_cache`` 缓存. 默认 False.

    Notes
    -----
    ``enable_cache`` 装饰器可缓存部分积分素材, 适合在迭代算法中反复装配而空间
    不变的积分子, 见 ``integrator.enable_cache``.
    """
    _region: _Region = None
    etype: str

    def __init__(self, keep_data=False, *args, **kwds) -> None:
        self._cache: Dict[Tuple[str, int], Any] = {}
        self.keep_data(keep_data)

    ### START: 缓存 ###
    def keep_data(self, status_on=True, /):
        """设置是否保留 ``@enable_cache`` 修饰的方法所缓存的积分素材; 关闭时清空缓存."""
        self._keep_data = status_on
        if not status_on:
            self._cache.clear()
        return self

    def clear(self) -> None:
        """清空积分子的缓存."""
        self._cache.clear()
    ### END: 缓存 ###

    ### START: 积分区域 ###
    def set_region(self, region: _Region, /):
        """设置积分区域: 网格实体编号, 或接收网格并返回编号的函数; 同时清空缓存."""
        self._region = region
        self.clear()
        return self

    def get_region(self):
        """返回积分区域: 网格实体编号, 或接收网格并返回编号的函数."""
        return self._region

    def entity_selection(self, indices: _OpIndex = None, *, mesh: Optional[Mesh] = None) -> Index:
        """确定参与积分的实体.

        Parameters
        ----------
        indices : Index, optional
            在积分区域内再选的子集; 区域为布尔掩码时按其真值位置的序号选取.
        mesh : Mesh, optional
            网格; 积分区域为函数时必须给出.

        Returns
        -------
        Index
            参与积分的实体编号, 区域与 ``indices`` 都未给出时为全切片.

        Raises
        ------
        RuntimeError
            积分区域为函数而未给出网格.
        TypeError
            积分区域不是张量而又给出了 ``indices``.
        """
        if self._region is None:
            if indices is None:
                return slice(None, None, None)
            else:
                return indices
        else:
            if callable(self._region):
                if mesh is None:
                    raise RuntimeError("Mesh must be provided in entity_selection "
                    "when region is given as a callable.")
                full_region = self._region(mesh)
            else:
                full_region = self._region
            if indices is None:
                return full_region
            else:
                if bm.is_tensor(full_region):
                    if full_region.dtype == bm.bool:
                        return bm.nonzero(full_region)[0][indices]
                    return full_region[indices]
                else:
                    raise TypeError(f"region of type '{full_region.__class__.__name__}' "
                                    "is not supported when indices is given.")

    def size(self, mesh: Mesh, /) -> int:
        """积分区域内的实体数; 未设区域时为网格上 ``etype`` 类实体的总数.

        Raises
        ------
        RuntimeError
            未设区域且积分子没有 ``etype``.
        TypeError
            积分区域不是张量.
        """
        if self._region is None:
            if not hasattr(self, 'etype'):
                raise RuntimeError("etype of Integrator should be specified to detect "
                "the number of entities when region is `None`.")
            else:
                return mesh.entity(self.etype).shape[0]
        else:
            if callable(self._region):
                full_region = self._region(mesh)
            else:
                full_region = self._region
            if bm.is_tensor(full_region):
                if full_region.dtype == bm.bool:
                    return bm.sum(full_region, dtype=bm.int64)
                else:
                    return full_region.shape[0]
            else:
                raise TypeError(f"region of type '{full_region.__class__.__name__}' "
                                "is not supported when indices is given.")
    ### END: 积分区域 ###

    def const(self, space: _SpaceGroup, /):
        """在 ``space`` 上算出积分与自由度映射, 冻结为 ``ConstIntegrator``."""
        value = self.assembly(space)
        to_gdof = self.to_global_dof(space)
        return ConstIntegrator(value, to_gdof)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def __call__(self, *args, **kwargs):
        return self.assembly(*args, **kwargs)

    def to_global_dof(self, space: _SpaceGroup, /, indices: _OpIndex = None) -> Union[TensorLike, Tuple[TensorLike, ...]]:
        """返回积分实体的局部自由度到全局自由度的映射, 由子类实现."""
        raise NotImplementedError

    def assembly(self, space: _SpaceGroup, /, indices: _OpIndex = None) -> TensorLike:
        """在实体上计算积分的默认方法, 由子类实现."""
        raise NotImplementedError

    ### 运算

    def __add__(self, other: 'Integrator'):
        if isinstance(other, Integrator):
            return GroupIntegrator(self, other)
        else:
            return NotImplemented

    __iadd__ = __add__


# 以下积分子类只作类型标记

class NonlinearInt(Integrator):
    """非线性积分子: 不要求线性的积分子基类."""
    pass

class LinearInt(Integrator):
    """线性积分子: 积分关于 ``u`` 与 ``v`` 都是线性的积分子基类."""
    pass

class OpInt(Integrator):
    """算子积分子: 同时涉及试探函数 ``u`` 与检验函数 ``v`` 的积分子基类."""
    pass

class SrcInt(Integrator):
    """源项积分子: 只涉及检验函数 ``v`` 的积分子基类."""
    pass

class CellInt(Integrator):
    """单元积分子: 在网格单元上积分的积分子基类."""
    etype = 'cell'

class FaceInt(Integrator):
    """面积分子: 在网格面上积分的积分子基类."""
    etype = 'face'

class EdgeInt(Integrator):
    """边积分子: 在网格边上积分的积分子基类."""
    etype = 'edge'


##################################################
### 积分工具
##################################################

_GT = TypeVar('_GT')

class ConstIntegrator(Integrator, Generic[_GT]):
    """取给定值的积分项: 把一个张量包装成积分子.

    Parameters
    ----------
    value : TensorLike
        积分值, 第 0 轴为实体.
    to_gdof : TensorLike or tuple of TensorLike, optional
        自由度映射; 需要调用 ``to_global_dof`` 时必须给出.
    """
    def __init__(self, value: TensorLike, to_gdof: Optional[_GT] = None):
        super().__init__()
        self.value = value
        self.to_gdof = to_gdof
        self._region = slice(None)

    def set_region(self, region, /):
        """记录区域但不起作用 (会给出警告), 积分值已固定."""
        logger.warning("`set_region` has no effect for ConstIntegrator.")
        return super().set_region(region)

    def to_global_dof(self, space, /, indices: _OpIndex = None) -> _GT:
        """返回给定的自由度映射, 给出 ``indices`` 时取其子集.

        Raises
        ------
        RuntimeError
            构造时未给出 ``to_gdof``.
        """
        if self.to_gdof is None:
            raise RuntimeError("to_gdof not defined for ConstIntegrator.")
        if indices is None:
            return self.to_gdof
        if isinstance(self.to_gdof, (tuple, list)):
            return self.to_gdof.__class__(tg[indices] for tg in self.to_gdof)
        return self.to_gdof[indices]

    def assembly(self, space, /, indices: _OpIndex = None):
        """返回给定的积分值, 给出 ``indices`` 时取其子集."""
        if indices is None:
            return self.value
        return self.value[indices]


class GroupIntegrator(Integrator):
    """把多个积分项合为一个.

    要求所有子积分子的 ``to_global_dof`` 输出相同, 即积分区域与局部-全局自由度
    关系一致. 组合后子积分子只提供积分值, 自由度映射取第一个子积分子的
    ``to_global_dof``.

    Parameters
    ----------
    *ints : Integrator
        子积分子; 其中的 ``GroupIntegrator`` 会被展开.
    region : TensorLike, optional
        给出时替换所有子积分子的积分区域; 为 None 时不替换.

    Raises
    ------
    ValueError
        没有给出子积分子.
    TypeError
        子积分子不是 ``Integrator``.
    """
    def __init__(self, *ints: Integrator, region: Optional[TensorLike] = None):
        super().__init__('assembly')
        self.ints: List[Integrator] = [] # 不含 GroupIntegrator 的子积分子.
        if len(ints) == 0:
            raise ValueError("No integrators provided.")
        for integrator in ints:
            if isinstance(integrator, GroupIntegrator):
                self.ints.extend(integrator)
            elif isinstance(integrator, Integrator):
                self.ints.append(integrator)
            else:
                raise TypeError(f"Unsupported type {integrator.__class__.__name__} "
                                "found in the inputs.")
        if region is not None:
            self.set_region(region)

    def __repr__(self):
        return "GroupIntegrator[" + ", ".join([repr(i) for i in self.ints]) + "]"

    def __iter__(self):
        yield from self.ints

    def __len__(self):
        return len(self.ints)

    def __getitem__(self, index: int):
        return self.ints[index]

    def __iadd__(self, other: Integrator): # 原地并入, 不新建组
        if isinstance(other, GroupIntegrator):
            self.ints.extend(other)
        elif isinstance(other, Integrator):
            self.ints.append(other)
        else:
            return NotImplemented
        return self

    @property
    def etype(self) -> str:
        """第一个子积分子的实体类型."""
        return self.ints[0].etype

    def set_region(self, region: TensorLike, /) -> None:
        """把积分区域同时设到所有子积分子与组本身."""
        for integrator in self.ints:
            integrator.set_region(region)
        return super().set_region(region)

    def to_global_dof(self, space: _SpaceGroup, /, indices: _OpIndex = None):
        """取第一个子积分子的自由度映射."""
        if indices is None:
            return self.ints[0].to_global_dof(space)
        return self.ints[0].to_global_dof(space, indices=indices)

    @overload
    def assembly(self, space: _SpaceGroup, /, indices: _OpIndex = None) -> TensorLike: ...
    def assembly(self, space: _SpaceGroup, /, *args, **kwargs):
        """各子积分子的积分值相加; 维数不同时在较少维者前补一个批量轴后相加.

        Raises
        ------
        RuntimeError
            子积分子输出的共同前导维形状不一致.
        """
        ct = self.ints[0].assembly(space, *args, **kwargs)

        for int_ in self.ints[1:]:
            new_ct = int_.assembly(space, *args, **kwargs)
            fdim = min(ct.ndim, new_ct.ndim)
            if ct.shape[:fdim] != new_ct.shape[:fdim]:
                raise RuntimeError(f"The output of the integrator {int_.__class__.__name__} "
                                   f"has an incompatible shape {tuple(new_ct.shape)} "
                                   f"with the previous {tuple(ct.shape)}.")
            if new_ct.ndim > ct.ndim:
                ct = new_ct + ct[None, ...]
            elif new_ct.ndim < ct.ndim:
                ct = ct + new_ct[None, ...]
            else:
                ct = ct + new_ct

        return ct
