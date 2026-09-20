"""局部应力约束的豁免区域: 掩码构造与掩码施加.

载荷贴片端部、混合边界角点等位置的应力奇异性不能由密度设计消除, 网格加密时
该处应力比随阶次与分辨率单调增长, 不收敛到有限值. 若把这些点留在约束集合中,
ALM 会用密度去追一个发散量, 表现为个别单元长期卡住、外层永不满足 C2.

标准处置是"局部应力豁免": 把奇异点的固定物理尺寸邻域从约束集合中剔除, 只在
其余区域施加局部应力约束. 豁免区必须按固定物理尺寸定义, 不随网格加密缩小,
否则豁免的是"越来越小的一块", 等于没有豁免.

只做豁免是不够的, 必须同时把该邻域设为实体保留 (passive solid). 2026-09-16 的
对照给出了理由: 悬臂梁应力算例只豁免不保留时, Hu--Zhang k=2 确实从 1000 步不
收敛变为 536 步收敛, 但优化器发现该邻域不再受约束, 于是把原本被应力约束逼着
保持 rho~0.98 的单元减到 0.67 以换取体积 —— 减料是白赚的, 代价是实体应力比从
0.68 涨到 1.45. 该次运行的 ``max_solid_stress_ratio_solid_region`` 为 1.452,
而此前全部运行都稳定在 1.01-1.03 (恰是 epsilon 松弛允许的余量). 也就是说, 豁免
一旦留下设计自由度, 就会被优化器利用, 掩盖真实的过应力.

因此本模块提供的掩码有两个用途, 必须成对施加:

- 应力约束侧: ``apply_exemption`` 把该邻域的约束值与灵敏度换成严格可行常值;
- 密度设计侧: ``apply_passive_solid`` 把该邻域的物理密度钉在 1, 并把对应的
  密度灵敏度置零, 使这些单元既不参与设计更新, 也不产生减料动机.

物理上这就是"载荷引入垫片": 载荷总要通过一块实体传入结构, 该处的应力奇异性
是边界条件的性质而非设计的缺陷, 既约束不了, 也不该让优化器去动它.

本模块只负责几何判定与张量施加, 不决定邻域中心在哪里: 中心由物理问题给出
(如 ``CantileverMiddle2d.traction_patch_endpoints``), 半径由算例配置给出.

References
----------
.. [1] 博士论文第 2.3.3 节: 局部应力豁免与实体保留.
.. [2] dut-postdoc/concepts/density-topopt/stress-constrained-topopt.md 第 5.2 节.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

__all__ = [
    "EXEMPT_CONSTRAINT_VALUE",
    "PASSIVE_SOLID_DENSITY",
    "apply_exemption",
    "apply_passive_solid",
    "build_exemption_mask",
    "validate_exemption_mask",
]

# 豁免点在约束集合中的替代值. 取严格可行的有限值, 使 AL 的
# h = max(g, -lambda/mu) 恒为负、乘子被拉回 0, 同时不引入 inf/nan
# (compute_relative_violation 对非有限值直接报错).
EXEMPT_CONSTRAINT_VALUE = -1.0

# 实体保留单元的物理密度. 取 1.0 (满密度), 即载荷引入垫片按实体材料计入体积,
# 其密度不再是设计变量的函数, 故对应的密度灵敏度恒为 0.
PASSIVE_SOLID_DENSITY = 1.0


def build_exemption_mask(
    mesh,
    centers: Sequence[Sequence[float]],
    radius: float,
) -> TensorLike:
    """构造应力约束豁免单元的布尔掩码.

    单元重心落入任一豁免中心的 ``radius`` 邻域即视为豁免单元, 按几何判定,
    不依赖单元编号顺序, 结构化与非结构化网格通用.

    Parameters
    ----------
    mesh : HomogeneousMesh
        应力评价所在的网格, 需支持 ``entity_barycenter('cell')``.
    centers : Sequence[Sequence[float]]
        豁免中心坐标序列, 每项长度须与网格几何维一致.
    radius : float
        豁免半径, 有限非负数. 取 0 时返回全 False 掩码.

    Returns
    -------
    TensorLike
        形状 ``(NC,)`` 的布尔张量, True 表示该单元的应力评价点被剔除.
    """
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius 必须为有限非负数")

    barycenter = mesh.entity_barycenter("cell")
    n_cells = barycenter.shape[0]
    mask: TensorLike = bm.zeros((n_cells,), dtype=bm.bool)
    if radius == 0.0 or len(centers) == 0:
        return mask

    geometry_dimension = barycenter.shape[-1]
    radius_squared = radius * radius
    for center in centers:
        if len(center) != geometry_dimension:
            raise ValueError(
                f"豁免中心维度 {len(center)} 与网格几何维 {geometry_dimension} 不一致"
            )
        distance_squared = bm.zeros((n_cells,), dtype=barycenter.dtype)
        for axis, coordinate in enumerate(center):
            offset = barycenter[..., axis] - float(coordinate)
            distance_squared = distance_squared + offset * offset
        mask = bm.logical_or(mask, distance_squared <= radius_squared)

    return mask


def validate_exemption_mask(
    mask: Optional[TensorLike],
    n_cells: int,
) -> Optional[TensorLike]:
    """校验豁免掩码的形状与类型, 全 False 掩码归一为 None.

    Parameters
    ----------
    mask : TensorLike, optional
        形状 ``(NC,)`` 的布尔张量, 或 None 表示不豁免.
    n_cells : int
        网格单元数, 用于形状校验.

    Returns
    -------
    TensorLike or None
        校验通过的掩码; 未豁免任何单元时返回 None, 使下游走无掩码快路径.
    """
    if mask is None:
        return None
    if mask.shape != (n_cells,):
        raise ValueError(
            f"exemption_mask 形状必须为 ({n_cells},), 实际为 {tuple(mask.shape)}"
        )
    if not bool(bm.any(mask)):
        return None
    return mask


def apply_exemption(
    values: TensorLike,
    mask: Optional[TensorLike],
    fill: float,
) -> TensorLike:
    """把豁免单元上的约束量或灵敏度置为指定常值.

    掩码定义在单元上, 对单元内的全部评价点 (多分辨率子单元、积分点) 一并生效,
    因此沿 ``values`` 的尾部各轴广播.

    Parameters
    ----------
    values : TensorLike
        形状 ``(NC, ...)`` 的约束值或其对中间量的偏导数.
    mask : TensorLike, optional
        形状 ``(NC,)`` 的布尔掩码; None 时原样返回.
    fill : float
        豁免单元上的替代值: 约束量取 ``EXEMPT_CONSTRAINT_VALUE``, 灵敏度取 0.

    Returns
    -------
    TensorLike
        与 ``values`` 同形状的张量.
    """
    if mask is None:
        return values
    if values.shape[0] != mask.shape[0]:
        raise ValueError(
            f"约束量首轴长度 {values.shape[0]} 与豁免掩码长度 {mask.shape[0]} 不一致"
        )
    broadcast_shape = (mask.shape[0],) + (1,) * (values.ndim - 1)
    broadcast_mask = bm.reshape(mask, broadcast_shape)
    return bm.where(broadcast_mask, bm.full_like(values, fill), values)


def apply_passive_solid(
    density: TensorLike,
    mask: Optional[TensorLike],
) -> TensorLike:
    """把实体保留单元的物理密度钉为满密度.

    施加点在过滤/投影之后: 设计变量经 ``rho_phys = P(H z)`` 得到物理密度后,
    再把保留单元覆写为 1. 这样无论过滤半径多大、邻域怎么变化, 该处都是实体;
    只钉设计变量是不够的 —— 宽过滤下保留单元的物理密度仍由邻域决定.

    覆写使 ``rho_phys`` 在这些单元上与设计变量无关, 因此链式法则中对应的
    ``d rho_phys / d z`` 为 0, 调用方须同步把这些行的密度灵敏度置零
    (``apply_exemption(grad, mask, 0.0)``), 否则梯度与被优化的量不一致.

    Parameters
    ----------
    density : TensorLike
        形状 ``(NC, ...)`` 的物理密度场.
    mask : TensorLike, optional
        形状 ``(NC,)`` 的布尔掩码; None 时原样返回.

    Returns
    -------
    TensorLike
        与 ``density`` 同形状, 保留单元取 ``PASSIVE_SOLID_DENSITY``.
    """
    return apply_exemption(density, mask, PASSIVE_SOLID_DENSITY)
