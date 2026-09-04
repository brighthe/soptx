"""求解器注册表: 字符串键 -> :class:`~soptx.solvers.base.LinearSolver` 子类.

analyzer 侧只认字符串 (配置里的 ``solve_method``), 不认类. 注册表把这层映射
收在一处, 取代 ``lagrange_fem_analyzer.solve_system`` 与
``huzhang_mfem_analyzer`` 里两份重复的 if-elif 分派 -- 加一种后端只需在该后端
自己的模块里 ``@register("名字")``, 不必回头改 analyzer.

能不能用不由注册表判断: 注册表只负责造出实例, 算子能力协商在
:meth:`LinearSolver.setup` 里做. 因此注册 AMG 这类只能吃显式矩阵的后端是安全
的 -- 它在 'ea' 层级下会在 setup 处抛 :class:`OperatorCapabilityError`, 而不是
在查表处被静默排除.

一个类可以占多个键: :func:`register` 接受预置构造参数, 于是
``DirectSolver`` 用 ``backend`` 的两个取值注册成 ``"scipy"`` 与 ``"mumps"``
两个键, 不必为一个字符串多造一个类.

尚未实现的后端 (AMG, MINRES, 多重网格) 暂不注册: 它们的 ``__init__`` 直接抛
:class:`NotImplementedError`, 登记进来只会让 :func:`available` 谎报可用键.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Type

from .base import LinearSolver

# 键 -> (求解器类, 预置构造参数). 由各后端模块导入时通过 register 填充.
_REGISTRY: Dict[str, Tuple[Type[LinearSolver], Dict[str, Any]]] = {}


def register(
    name: str, **preset: Any
) -> Callable[[Type[LinearSolver]], Type[LinearSolver]]:
    """把求解器类注册到 ``name`` 键下的类装饰器.

    Parameters
    ----------
    name : str
        注册键, 与配置里的 ``solve_method`` 取值一致.
    **preset
        该键固有的构造参数, 用于把同一个类分成多个键 (如 ``backend``).
        :func:`create` 的同名参数覆盖它.

    Returns
    -------
    callable
        接收类并原样返回的装饰器.

    Raises
    ------
    KeyError
        该键已被占用 (重复注册通常意味着模块被重复导入或键名撞车).
    """
    def _decorate(cls: Type[LinearSolver]) -> Type[LinearSolver]:
        if name in _REGISTRY:
            raise KeyError(
                f"求解器键 {name!r} 已被 {_REGISTRY[name][0].__name__} 占用, "
                f"不能再注册给 {cls.__name__}"
            )
        _REGISTRY[name] = (cls, dict(preset))
        return cls

    return _decorate


def create(name: str, **kwargs: Any) -> LinearSolver:
    """按键造出求解器实例, 不做 setup.

    Parameters
    ----------
    name : str
        注册键.
    **kwargs
        转发给求解器构造函数的参数 (容差, 最大迭代数等), 覆盖注册时的预置参数.

    Returns
    -------
    LinearSolver
        未绑定算子的求解器实例, 调用方自行 ``setup(op)``.

    Raises
    ------
    KeyError
        键未注册.
    """
    entry = _REGISTRY.get(name)
    if entry is None:
        raise KeyError(
            f"未注册的求解器键: {name!r}; 可用: {', '.join(available())}"
        )
    cls, preset = entry
    return cls(**{**preset, **kwargs})


def available() -> Tuple[str, ...]:
    """已注册的全部键, 按字典序返回."""
    return tuple(sorted(_REGISTRY))
