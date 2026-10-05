# 移植自 brighthe/fealpy ``fealpy/mesh/uniform_mesh/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""均匀结构网格的占位类.

这些类没有任何实现, 仅供 ``isinstance`` 判断; 由于不会被实例化, 相应判断恒为假.
"""

__all__ = [
    'UniformMesh',
    'UniformMesh1d',
    'UniformMesh2d',
    'UniformMesh3d'
]

# NOTE: 以下均为占位类

class UniformMesh:
    """均匀结构网格占位类."""
    pass


class UniformMesh1d:
    """一维均匀结构网格占位类."""
    pass


class UniformMesh2d:
    """二维均匀结构网格占位类."""
    pass


class UniformMesh3d:
    """三维均匀结构网格占位类."""
    pass
