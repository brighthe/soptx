# 移植自 brighthe/fealpy ``fealpy/mesh/topology/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""网格拓扑: 由根单元派生子实体分区、建立实体间关系并判定边界."""

from .boundary import *
from .builder import *
from .local_entity import *
