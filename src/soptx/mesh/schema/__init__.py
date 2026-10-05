# 移植自 brighthe/fealpy ``fealpy/mesh/schema/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""公开的不可变实体 Schema, 以及描述符与解析器接口.

由 :mod:`soptx.mesh.schema.classic` 中的类名构造具体的 Schema 值, 按值比较, 并经
``SchemaDescriptor`` 序列化其身份. 几何插值由 Schema 值决定; 独立的有限元参考基函数
通过其 ``lagrange_basis_function`` 接口取得, 不改变几何身份.
"""

from .classic import *
from .descriptor import *
from .entity_schema import *
from .local_entity import *
from .polygon import *
from .registry import *
