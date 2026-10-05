# 移植自 brighthe/fealpy ``fealpy/decorator/input_produce.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""批量输入装饰器."""

from functools import wraps
from ..backend.base import TensorLike

def multi_input(func):
    """让函数既接受单组参数, 也接受一组参数序列并逐组调用.

    唯一的位置参数是序列 (列表、元组或张量) 且其每个元素也是序列时, 把每个元素
    解包为一组位置参数分别调用 ``func``; 此时每个关键字参数也须是序列, 第 ``i``
    次调用取其第 ``i`` 个元素. 张量会先转成 Python 列表. 其余情况直接调用 ``func``.

    Parameters
    ----------
    func : callable
        被修饰的函数.

    Returns
    -------
    callable
        包装后的函数; 批量调用时返回各次结果组成的列表.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """按输入形式决定逐组调用还是直接调用."""
        # 判断只有一个参数传入, 并且它是一个列表, 且其中的每个元素也是列表或元组
        if len(args) == 1 and isinstance(args[0], (list, tuple, TensorLike)) and all(isinstance(item, (list, tuple, TensorLike)) for item in args[0]):
            results = []
            param_sets = args[0]
            if isinstance(args[0], TensorLike):
                param_sets = param_sets.tolist()
            for idx, param_set in enumerate(param_sets):
                # param_set 应该是一个列表或元组, 使用 *param_set 来解包调用原函数
                kwarg = {}
                for k, v in kwargs.items():
                    val = v[idx]
                    if isinstance(val, TensorLike):
                        val = val.tolist()
                    kwarg[k] = val
                results.append(func(*param_set, **kwarg))

            return results
        # 如果都不是序列, 直接调用
        return func(*args, **kwargs)
    return wrapper