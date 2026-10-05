# 移植自 brighthe/fealpy ``fealpy/decorator/coordinates.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""坐标类型标记装饰器.

Notes
-----
每个装饰器给被修饰函数加一个 ``coordtype`` 属性, 取值即装饰器名. 调用方据此决定
传入的点用哪种坐标表示, 例如 ``process_coef_func`` 对 ``'barycentric'`` 直接传
重心坐标, 对其余取值先映射为直角坐标.
"""
from functools import wraps

def cartesian(func):
    """把函数标记为接受直角坐标, 即设置 ``coordtype = 'cartesian'``."""
    @wraps(func)
    def add_attribute(*args, **kwargs):
        """原样转发调用, 只用于携带 ``coordtype`` 属性."""
        return func(*args, **kwargs)
    add_attribute.__dict__['coordtype'] = 'cartesian' 
    return add_attribute 

def barycentric(func):
    """把函数标记为接受重心坐标, 即设置 ``coordtype = 'barycentric'``."""
    @wraps(func)
    def add_attribute(*args, **kwargs):
        """原样转发调用, 只用于携带 ``coordtype`` 属性."""
        return func(*args, **kwargs)
    add_attribute.__dict__['coordtype'] = 'barycentric' 
    return add_attribute 

def polar(func):
    """把函数标记为接受极坐标, 即设置 ``coordtype = 'polar'``."""
    @wraps(func)
    def add_attribute(*args, **kwargs):
        """原样转发调用, 只用于携带 ``coordtype`` 属性."""
        return func(*args, **kwargs)
    add_attribute.__dict__['coordtype'] = 'polar' 
    return add_attribute 

def spherical(func):
    """把函数标记为接受球坐标, 即设置 ``coordtype = 'spherical'``."""
    @wraps(func)
    def add_attribute(*args, **kwargs):
        """原样转发调用, 只用于携带 ``coordtype`` 属性."""
        return func(*args, **kwargs)
    add_attribute.__dict__['coordtype'] = 'spherical' 
    return add_attribute 

def cylindrical(func):
    """把函数标记为接受柱坐标, 即设置 ``coordtype = 'cylindrical'``."""
    @wraps(func)
    def add_attribute(*args, **kwargs):
        """原样转发调用, 只用于携带 ``coordtype`` 属性."""
        return func(*args, **kwargs)
    add_attribute.__dict__['coordtype'] = 'cylindrical' 
    return add_attribute 
