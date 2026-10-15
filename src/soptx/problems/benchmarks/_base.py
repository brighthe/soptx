"""工程基准问题的公共基类."""

from __future__ import annotations

from typing import Sequence

from soptx.backend import backend_manager as bm
from soptx.core.numerics import GEOMETRIC_TOL
from soptx.typing import TensorLike


class ElasticityBenchmark:
    """工程基准问题共用的几何、主载荷幅值与参考材料.

    子类以类属性给出 ``dimension``; 三维子类另以类属性 ``plane_type = "3D"`` 覆盖本类的
    同名 property, 构造函数不再收该参数. 边界、载荷与各能力层成员由子类按
    ``soptx.protocols`` 的接口分层实现, 构造参数顺序统一为
    ``domain, P, <类特有参数>, E, nu, plane_type[, traction]``.

    Parameters
    ----------
    domain : Sequence[float]
        轴对齐包围盒 ``(x0, x1, y0, y1[, z0, z1])``; 非盒形问题另以 ``vertices`` 给出几何.
    P : float
        主载荷的合力, 负值向下.
    E : float
        实体材料的 Young 模量.
    nu : float
        实体材料的 Poisson 比.
    plane_type : str
        本构假设, ``'plane_stress'``, ``'plane_strain'`` 或 ``'3D'``.
    """

    dimension: int
    boundary_type = "mixed"

    def __init__(
        self,
        domain: Sequence[float],
        *,
        P: float,
        E: float,
        nu: float,
        plane_type: str,
    ) -> None:
        self._domain = tuple(float(value) for value in domain)
        self._eps = GEOMETRIC_TOL
        self._P = float(P)
        self._E = float(E)
        self._nu = float(nu)
        self._plane_type = str(plane_type)

    @property
    def domain(self) -> tuple[float, ...]:
        """轴对齐包围盒 ``(x0, x1, y0, y1[, z0, z1])``."""
        return self._domain

    @property
    def P(self) -> float:
        """主载荷的合力, 负值向下."""
        return self._P

    @property
    def E(self) -> float:
        """实体材料的 Young 模量."""
        return self._E

    @property
    def nu(self) -> float:
        """实体材料的 Poisson 比."""
        return self._nu

    @property
    def plane_type(self) -> str:
        """本构假设."""
        return self._plane_type

    def _on_x(self, points: TensorLike, value: float) -> TensorLike:
        """标记横坐标等于 ``value`` 的点."""
        return bm.abs(points[..., 0] - value) < self._eps

    def _on_y(self, points: TensorLike, value: float) -> TensorLike:
        """标记纵坐标等于 ``value`` 的点."""
        return bm.abs(points[..., 1] - value) < self._eps

    def _on_z(self, points: TensorLike, value: float) -> TensorLike:
        """标记第三坐标等于 ``value`` 的点."""
        return bm.abs(points[..., 2] - value) < self._eps
