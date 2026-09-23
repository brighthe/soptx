# -*- coding: utf-8 -*-
"""矩阵自由算子分解中与物理无关的计算核.

    A = P^T G^T B^T D B G P

G 是 ``ElementRestriction`` (L 向量与 E 向量之间的聚集散加), B 是 ``ReferenceBasis``
加 ``GeometricFactors`` (单元自由度与积分点之间的插值求导, 前者持参考单元上的基函数,
后者持逐单元的 J^{-1}, 两者都是纯数据; ``gradients`` 模块里的收缩只吃这两个数组), D 是
``weighted_stress`` (积分点上的逐点作用, 所需数组由 ``LinearElasticQFunction`` 持有).
三者都不认识具体的装配层级, 由 ``soptx.fem.levels`` 里的层级类组合使用.

严格说 D 是唯一带物理的一环, 目前只有线弹性一种实现, 不设抽象基类.

记号约定
--------
- B 沿用 libCEED / MFEM 的记号, 指基函数算子, 交出的是完整的 grad u; 它不是有限元
  教材里的应变位移矩阵. 后者等于 B 再接一个取对称并按 Voigt 排列的
  ``strain_map``, 那一项归在 D 一侧.
- 形状里的 NB 是批量维, 即同时作用的向量个数 (多列右端项), 一律放在最后, 单列时省
  略; 它与算子 B 无关.
- E 向量在 PA / UA 里一律是 (NC, ldof, GD) 的分量布局, 由 ``ElementRestriction`` 交
  出, B 不做自由度重排.
"""

from .geometric_factors import GeometricFactors
from .gradients import (
    physical_basis_gradients,
    physical_gradient,
    physical_gradient_transpose,
)
from .qfunction import (
    LinearElasticQFunction,
    strain_map,
    weighted_stress,
    weighted_stress_diagonal,
)
from .reference_basis import ReferenceBasis, build_cache_keys, clear_build_cache
from .restriction import ElementRestriction

__all__ = [
    "build_cache_keys",
    "clear_build_cache",
    "ElementRestriction",
    "GeometricFactors",
    "physical_basis_gradients",
    "physical_gradient",
    "physical_gradient_transpose",
    "LinearElasticQFunction",
    "ReferenceBasis",
    "strain_map",
    "weighted_stress",
    "weighted_stress_diagonal",
]
