# -*- coding: utf-8 -*-
"""装配层级注册表: 字符串键 -> :class:`AssemblyLevelExtension` 子类.

analyzer 侧只认字符串 (配置里的 ``operator_level``), 不认类. 注册表把这层映射收在
一处, 取代 ``assemble_stiff_matrix`` 上按层级分派的 ``variantmethod`` 变体 -- 加一
个层级只需在它自己的模块里 ``@register_level("名字")`` 并实现 ``build``, 不必回头
改 analyzer.

结构与 :mod:`soptx.solvers.registry` 一致: 同一个模式用在两根不同的轴上, 一个分派
求解器后端, 一个分派算子的常驻形式.

注册表只负责造出层级实例, 不判断该层级配不配合某个求解器 -- 那件事在
:meth:`LinearSolver.setup` 的算子能力协商里做, 因此把只能吃显式矩阵的后端与只有
matvec 的层级同时登记进来是安全的.
"""

from typing import Any, Callable, Dict, Tuple, Type, TypeVar

from .base import AssemblyLevelExtension

# 键 -> 层级类. 由各层级模块导入时通过 register_level 填充.
_REGISTRY: Dict[str, Type[AssemblyLevelExtension]] = {}

# 装饰器要原样透传被装饰的类型: 写成 Type[AssemblyLevelExtension] 会把子类擦成基类,
# 调用方拿到的 PartialAssembly.build 就只剩基类签名, 层级特有的属性全部报未知.
_LevelType = TypeVar("_LevelType", bound=AssemblyLevelExtension)


def register_level(
    name: str,
) -> Callable[[Type[_LevelType]], Type[_LevelType]]:
    """把层级类注册到 ``name`` 键下的类装饰器.

    Parameters
    ----------
    name : 注册键, 与配置里的 ``operator_level`` 取值一致.

    Returns
    -------
    接收类并原样返回的装饰器, 返回类型与传入类型相同.

    Raises
    ------
    KeyError
        该键已被占用 (重复注册通常意味着模块被重复导入或键名撞车).
    """
    def _decorate(cls: Type[_LevelType]) -> Type[_LevelType]:
        if name in _REGISTRY:
            raise KeyError(
                f"装配层级键 {name!r} 已被 {_REGISTRY[name].__name__} 占用, "
                f"不能再注册给 {cls.__name__}"
            )
        _REGISTRY[name] = cls
        return cls

    return _decorate


def create_level(name: str, **kwargs: Any) -> AssemblyLevelExtension:
    """按键构造层级实例.

    构造走层级类的 ``build`` 而不是 ``__init__``: ``__init__`` 收的是已经算好的
    常驻数据 (矩阵 / 单元矩阵), ``build`` 收的是函数空间与积分子, 后者才是各层级
    统一的入口签名.

    Parameters
    ----------
    name : 注册键.
    **kwargs : 转发给 ``build`` 的参数, 各层级共同接受 ``space``, ``integrator``
        与 ``pattern``, 用不上的层级原样忽略.

    Returns
    -------
    构造完成的层级实例.

    Raises
    ------
    KeyError
        键未注册.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"未注册的装配层级键: {name!r}; 可用: {', '.join(available_levels())}"
        )

    return cls.build(**kwargs)


def available_levels() -> Tuple[str, ...]:
    """已注册的全部键, 按字典序返回."""
    return tuple(sorted(_REGISTRY))
