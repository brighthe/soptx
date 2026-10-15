"""Lagrange 位移有限元分析器的预条件子注册表: 字符串键 -> 预条件子工厂.

与 :mod:`soptx.solvers.registry` 同形, 但落在 fem 侧: 几何多重网格 ``'gmg'`` 的层次要用
分析器状态 (单元系数, Dirichlet 掩码, 层次缓存), 求解层没有网格, 造不出它. 因此工厂
的入参是分析器本身, 而不只是构造参数.

工厂只构造、不 setup: 预条件子绑哪个算子 (主算子或 ``preconditioner_level`` 另建的
算子) 以及能力协商失败时的报错由分析器的 ``_make_preconditioner`` 负责. 加一种预条件子
只需在本模块 ``@register_preconditioner("名字")``, 不必回头改分析器.

子结构分析器的接口系统另有求解路径, 暂不经过本注册表.
"""

from typing import Any, Callable, Dict, Tuple

from soptx.solvers import DiagonalPreconditioner, LinearSolver
from soptx.solvers import create as create_solver

# 工厂签名: (analyzer, options) -> 尚未 setup 的预条件子; options 为求解调用的 kwargs
PreconditionerFactory = Callable[[Any, Dict[str, Any]], LinearSolver]

# 键 -> 工厂
_REGISTRY: Dict[str, PreconditionerFactory] = {}


def register_preconditioner(name: str) -> Callable[[PreconditionerFactory], PreconditionerFactory]:
    """把预条件子工厂注册到 ``name`` 下的装饰器.

    Parameters
    ----------
    name : 注册键, 与 ``solver_options['precond']`` 的取值一致. 每种预条件子只有一个键,
        不设别名.

    Returns
    -------
    decorator : 接收工厂并原样返回的装饰器.

    Raises
    ------
    KeyError
        该键已被占用.
    """
    def _decorate(factory: PreconditionerFactory) -> PreconditionerFactory:
        if name in _REGISTRY:
            raise KeyError(f"预条件子键 {name!r} 已被占用")
        _REGISTRY[name] = factory
        return factory

    return _decorate


def create_preconditioner(name: str, analyzer: Any, options: Dict[str, Any]) -> LinearSolver:
    """按键造出尚未 setup 的预条件子.

    Parameters
    ----------
    name : 注册键.
    analyzer : 调用方分析器, 供需要分析器状态的工厂 (如 ``'gmg'``) 使用.
    options : 求解调用的选项, 原样交给工厂.

    Returns
    -------
    M : 尚未 setup 的预条件子.

    Raises
    ------
    KeyError
        键未注册.
    """
    factory = _REGISTRY.get(name)
    if factory is None:
        raise KeyError(f"未知的预条件子类型: {name!r}; 可用: {', '.join(available_preconditioners())}")

    return factory(analyzer, options)


def available_preconditioners() -> Tuple[str, ...]:
    """已注册的全部键, 按字典序返回."""
    return tuple(sorted(_REGISTRY))


@register_preconditioner('jacobi')
def _jacobi(analyzer: Any, options: Dict[str, Any]) -> LinearSolver:
    """对角预条件子; 对角在 setup 时向预条件算子要."""
    return DiagonalPreconditioner()


@register_preconditioner('gmg')
def _gmg(analyzer: Any, options: Dict[str, Any]) -> LinearSolver:
    """几何多重网格 V 循环, 层次由分析器构造, 见 ``LagrangeFEMAnalyzer._gmg_preconditioner``."""
    return analyzer._gmg_preconditioner(**options)


@register_preconditioner('scipy')
def _scipy(analyzer: Any, options: Dict[str, Any]) -> LinearSolver:
    """SuperLU 直接法作预条件子; 精确逆, 主要用于验证两个层级是同一个离散算子."""
    return create_solver('scipy')


@register_preconditioner('mumps')
def _mumps(analyzer: Any, options: Dict[str, Any]) -> LinearSolver:
    """MUMPS 直接法作预条件子; 用途同 ``'scipy'``."""
    return create_solver('mumps')
