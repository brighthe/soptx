# -*- coding: utf-8 -*-
"""在装配层级之上再包一层的算子.

装配层级 (``soptx.fem.levels``) 只负责同一个离散算子以什么形式常驻, 不知道边界
条件, 也不知道并行布局. 这两件事各是一层包装, 包装次序固定为

    ConstrainedOperator(OverlapOperator(level))

即先并行后边界: 跨 rank 归约要先把各 rank 的局部作用拼成完整的算子作用, 再谈哪些
自由度被约束. ``OverlapOperator`` 在 ``soptx.fem.distributed`` 里, 因为它依赖可选
的 mpi4py.
"""

from .constrained import ConstrainedOperator

__all__ = [
    "ConstrainedOperator",
]
