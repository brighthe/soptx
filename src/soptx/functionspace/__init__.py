# 移植自 brighthe/fealpy ``fealpy/functionspace/__init__.py`` @ f474a5775, 仅保留 Lagrange 与张量空间.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

from .space import FunctionSpace
from .function import Function
from .dofs import LinearMeshCFEDof
from .lagrange_fe_space import LagrangeFESpace
from .tensor_space import TensorFunctionSpace

__all__ = [
    'FunctionSpace',
    'Function',
    'LinearMeshCFEDof',
    'LagrangeFESpace',
    'TensorFunctionSpace',
]
