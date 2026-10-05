# 移植自 brighthe/fealpy ``fealpy/mesh/generation/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""通用网格生成算法."""

from .box import Box1d, Box2d, Box3d

__all__ = ["Box1d", "Box2d", "Box3d"]
