# -*- coding: utf-8 -*-
"""在装配层级之上再包一层的算子.

装配层级 (``soptx.fem.levels``) 只负责同一个离散算子以什么形式常驻, 不知道边界
条件. 边界条件是包在层级外面的一层: ``ConstrainedOperator(level)``.
"""

from .constrained import ConstrainedOperator

__all__ = [
    "ConstrainedOperator",
]
