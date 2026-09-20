# -*- coding: utf-8 -*-
"""矩阵自由算子分解中与物理无关的计算核.

    A = P^T G^T B^T D B G P

G 是 ``ElementRestriction`` (L 向量与 E 向量之间的聚集散加), B 是 ``DofToQuad``
(单元自由度与积分点之间的插值求导), D 是 ``QFunction`` (积分点上的逐点作用). 三者
都不认识具体的装配层级, 由 ``soptx.fem.levels`` 里的层级类组合使用.

严格说 D 是唯一带物理的一环: ``QFunction`` 是抽象基类, 具体方程各写一个实现, 本子
包目前只有线弹性的 ``LinearElasticQFunction``.
"""

from .dof_to_quad import DofToQuad
from .qfunction import LinearElasticQFunction, QFunction, strain_map
from .restriction import ElementRestriction

__all__ = [
    "DofToQuad",
    "ElementRestriction",
    "LinearElasticQFunction",
    "QFunction",
    "strain_map",
]
