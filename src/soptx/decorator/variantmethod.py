# 移植自 brighthe/fealpy ``fealpy/decorator/variantmethod.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""变体方法: 一个方法名下注册多种实现, 按实例选择当前使用哪一种.

典型用法::

    class Scheme:
        @variantmethod('simp')
        def interpolate(self, rho): ...

        @interpolate.register('ramp')
        def interpolate(self, rho): ...

    s = Scheme()
    s.interpolate.set('ramp')   # 为该实例选定变体
    s.interpolate(rho)          # 调用选定的变体
    s.interpolate['simp'](rho)  # 绕过选定, 直接调用指定变体

Notes
-----
选定的变体记在描述符的 ``key_table`` 中, 以实例为键, 未选定时用默认键 (即
``variantmethod(key)`` 的 ``key``). 只有在以 ``VariantMeta`` 为元类的类中, 子类才
能继承并扩充父类注册的变体.
"""

from typing import (
    Tuple, Dict,
    Any, Type, Callable,
    TypeVar, overload, ParamSpec, Generic, Concatenate,
    Optional,
)
from functools import partial

__all__ = "variantmethod", "VariantMeta"

_T = TypeVar("_T")
_P = ParamSpec("_P")
_R_co = TypeVar("_R_co", covariant=True)


class Variantmethod(Generic[_T, _P, _R_co]):
    """持有多个变体实现的方法描述符.

    Parameters
    ----------
    func : callable, optional
        默认变体的实现; 为 None 时创建空表, 供 ``VariantMeta`` 合并使用.
    key : Any, optional
        默认变体的键.

    Attributes
    ----------
    virtual_table : dict
        变体键到实现函数的映射.
    key_table : dict
        实例到其选定变体键的映射.
    default_key : Any
        实例未选定时使用的变体键.
    """
    __slots__ = ('virtual_table', 'key_table', 'default_key')
    virtual_table : Dict[Any, Callable]
    key_table : Dict[_T, Any]

    def __init__(self, func: Optional[Callable] = None, key: Any = None):
        if func is None:
            self.virtual_table = {}
        else:
            self.virtual_table = {key: func}
        self.key_table = {}
        self.default_key = key

    @property
    def __func__(self) -> Callable[Concatenate[_T, _P], _R_co]:
        """默认变体的实现函数."""
        assert len(self.virtual_table) >= 1, "variants can not be empty"
        return self.virtual_table[self.default_key]

    @property
    def __name__(self) -> str:
        """默认变体实现的函数名, 也是该方法在类中的名字."""
        return self.__func__.__name__

    # def __get__(self, obj: _T, objtype: Type[_T]) -> Callable[_P, _R_co]:
    @overload
    def __get__(self, obj: None, objtype: Type[_T]) -> "Variantmethod[_T, _P, _R_co]": ...
    @overload
    def __get__(self, obj: _T, objtype: Type[_T]) -> "VariantHandler[_T, _P, _R_co]": ...
    def __get__(self, obj, objtype):
        if obj is None:
            return self

        return VariantHandler(self, obj, objtype)

    def __set__(self, obj: _T, val: Any):
        raise RuntimeError("Variantmethod has no setter.")

    def __len__(self) -> int:
        return len(self.virtual_table)

    def __getitem__(self, key: Any) -> Callable[Concatenate[_T, _P], _R_co]:
        if key in self.virtual_table:
            return self.virtual_table[key]
        else:
            return self.__func__

    def __contains__(self, item: Any) -> bool:
        return item in self.virtual_table

    def register(self, key: Any, /):
        """返回把函数注册为键 ``key`` 的变体的装饰器.

        装饰器返回描述符本身, 因此被修饰的函数可以与默认变体同名.

        Parameters
        ----------
        key : Any
            变体键.

        Returns
        -------
        callable
            装饰器, 注册后返回本描述符.
        """
        def decorator(func: Callable) -> Variantmethod[_T, _P, _R_co]:
            """把 ``func`` 登记为变体 ``key``, 返回描述符本身."""
            self.virtual_table[key] = func
            return self
        return decorator

    def get_key(self, obj: _T):
        """返回实例选定的变体键, 未选定时返回默认键."""
        if obj in self.key_table:
            return self.key_table[obj]
        else:
            return self.default_key

    def set_key(self, obj:_T, val: Any):
        """为实例选定变体键."""
        self.key_table[obj] = val

    def update(self, other: "Variantmethod", /) -> None:
        """并入另一个描述符的变体与实例选定; 本描述符为空时同时继承其默认键."""
        if len(self.virtual_table) == 0:
            self.default_key = other.default_key

        self.virtual_table.update(other.virtual_table)
        self.key_table.update(other.key_table)


class VariantHandler(Generic[_T, _P, _R_co]):
    """变体方法绑定到实例后的调用入口, 由 ``Variantmethod.__get__`` 创建.

    Parameters
    ----------
    vm : Variantmethod
        所属的描述符.
    obj : object
        绑定的实例.
    objtype : type
        实例所属的类.
    """
    def __init__(self, vm: Variantmethod[_T, _P, _R_co], obj: _T, objtype: Type[_T]):
        self.vm = vm
        self.instance = obj
        self.owner = objtype

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs):
        key = self.vm.get_key(self.instance)
        func = self.vm[key]
        return func.__get__(self.instance, self.owner)(*args, **kwargs)

    def __getitem__(self, val: Any) -> Callable[_P, _R_co]:
        func = self.vm[val]
        return func.__get__(self.instance, self.owner)

    def __contains__(self, item: Any) -> bool:
        return item in self.vm.virtual_table

    def set(self, val: Any):
        """为当前实例选定变体键 ``val``."""
        self.vm.set_key(self.instance, val)


@overload
def variantmethod(func: Callable[Concatenate[_T, _P], _R_co]) -> Variantmethod[_T, _P, _R_co]: ...
@overload
def variantmethod(key: Any) -> Callable[[Callable[Concatenate[_T, _P], _R_co]], Variantmethod[_T, _P, _R_co]]: ...
def variantmethod(arg):
    """把方法声明为变体方法.

    既可直接修饰 (``@variantmethod``, 默认键为 None), 也可带键修饰
    (``@variantmethod('simp')``, 默认键为该键).

    Parameters
    ----------
    arg : callable or Any
        被修饰的函数, 或默认变体的键.

    Returns
    -------
    Variantmethod or callable
        直接修饰时返回描述符; 带键修饰时返回以该键构造描述符的装饰器.
    """
    if isinstance(arg, Callable):
        return Variantmethod(arg)
    else:
        return partial(Variantmethod, key=arg)


def update_dispatch(dispatch_map: Dict[str, Variantmethod], dispobj: Variantmethod):
    """把描述符按方法名并入分派表, 同名的变体合并到一处."""
    name = dispobj.__name__

    if name not in dispatch_map:
        dispatch_map[name] = Variantmethod()

    dispatch_map[name].update(dispobj)


class VariantMeta(type):
    """让子类继承并扩充父类变体方法的元类.

    创建类时, 为每个变体方法名新建一个描述符, 先并入各父类的同名描述符, 再并入
    本类定义的, 最后挂到类上. 子类于是同时拥有父类与自身注册的变体.
    """
    def __init__(self, name: str, bases: Tuple[type, ...], dict: Dict[str, Any], /, **kwds: Any):
        dispatch_map: Dict[str, Variantmethod] = {}

        for base in bases:
            for attr in dir(base):
                val = getattr(base, attr)
                if isinstance(val, Variantmethod):
                    update_dispatch(dispatch_map, val)

        for val in dict.values():
            if isinstance(val, Variantmethod):
                update_dispatch(dispatch_map, val)

        for attr, disp in dispatch_map.items():
            setattr(self, attr, disp)

        return type.__init__(self, name, bases, dict, **kwds)
