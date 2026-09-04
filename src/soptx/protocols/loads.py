"""物理载荷的公共结构化契约.

本模块只描述载荷的物理语义及作用的几何实体, 不规定有限元装配方式. 同一个载荷对象
可以由 LFEM、Hu--Zhang、子结构或其他离散方法分别解释. 契约不持有 Mesh、
FunctionSpace 或 Material.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from fealpy.typing import TensorLike


LoadKind = Literal[
    "point_force",
    "line_traction",
    "boundary_traction",
    "body_force",
]


@runtime_checkable
class Load(Protocol):
    """所有物理载荷共享的最小契约.

    Notes
    -----
    ``dimension`` 是物理空间维数, ``support_dimension`` 是载荷作用的几何实体的维数.
    例如三维点力的两者分别为 3 和 0, 三维线载荷分别为 3 和 1. 这两个量描述
    物理对象, 不表示载荷最终通过节点写入还是数值积分装配.
    """

    @property
    def kind(self) -> LoadKind:
        """返回载荷的稳定物理类型标识."""
        ...

    @property
    def dimension(self) -> int:
        """返回载荷所在物理空间的维数."""
        ...

    @property
    def support_dimension(self) -> int:
        """返回载荷作用的几何实体的维数."""
        ...


@runtime_checkable
class PointForce(Load, Protocol):
    """作用在任意物理点上的集中力契约."""

    @property
    def kind(self) -> Literal["point_force"]:
        """返回 ``"point_force"``."""
        ...

    @property
    def point(self) -> Sequence[float]:
        """返回载荷作用点的物理坐标, 长度必须等于 ``dimension``."""
        ...

    def force(self, *, time: float | None = None) -> TensorLike:
        """返回集中力向量, 形状为 ``(dimension,)``.

        Parameters
        ----------
        time : float, optional
            载荷求值时刻. 静力载荷可以忽略该参数.
        """
        ...


@runtime_checkable
class LineTraction(Load, Protocol):
    """沿一维曲线分布的牵引载荷契约.

    三维实体边载荷属于本契约. 二维区域的边界线载荷优先使用
    :class:`BoundaryTraction`, 以便分析器按边界实体统一积分.
    """

    @property
    def kind(self) -> Literal["line_traction"]:
        """返回 ``"line_traction"``."""
        ...

    def is_load_line(self, points: TensorLike) -> TensorLike:
        """返回与 ``points`` 前导形状一致的布尔载荷区域标记."""
        ...

    def traction(
        self,
        points: TensorLike,
        *,
        tangents: TensorLike | None = None,
        time: float | None = None,
    ) -> TensorLike:
        """返回单位长度上的力, 形状为 ``points.shape``.

        Parameters
        ----------
        points : TensorLike
            线载荷求值点, 最后一维长度为 ``dimension``.
        tangents : TensorLike, optional
            对应点处的单位切向量. 与方向无关的给定牵引可以忽略.
        time : float, optional
            载荷求值时刻.
        """
        ...


@runtime_checkable
class BoundaryTraction(Load, Protocol):
    """作用在计算域边界上的分布牵引载荷契约.

    对二维问题, 本契约表示边界线载荷; 对三维问题, 本契约表示表面载荷.
    标量压力可在后续作为本契约的专门化对象, 利用 ``normals`` 生成向量牵引.
    """

    @property
    def kind(self) -> Literal["boundary_traction"]:
        """返回 ``"boundary_traction"``."""
        ...

    def is_load_boundary(self, points: TensorLike) -> TensorLike:
        """返回与 ``points`` 前导形状一致的布尔载荷边界标记."""
        ...

    def traction(
        self,
        points: TensorLike,
        *,
        normals: TensorLike | None = None,
        time: float | None = None,
    ) -> TensorLike:
        """返回单位边界测度上的牵引向量, 形状为 ``points.shape``.

        Parameters
        ----------
        points : TensorLike
            边界求值点, 最后一维长度为 ``dimension``.
        normals : TensorLike, optional
            对应点处的单位外法向量. 与法向无关的给定牵引可以忽略.
        time : float, optional
            载荷求值时刻.
        """
        ...


@runtime_checkable
class BodyForce(Load, Protocol):
    """作用在区域内部、以单位体积计量的体力契约."""

    @property
    def kind(self) -> Literal["body_force"]:
        """返回 ``"body_force"``."""
        ...

    def body_force(
        self,
        points: TensorLike,
        *,
        time: float | None = None,
    ) -> TensorLike:
        """返回单位体积上的力, 形状为 ``points.shape``.

        Parameters
        ----------
        points : TensorLike
            区域内部求值点, 最后一维长度为 ``dimension``.
        time : float, optional
            载荷求值时刻.
        """
        ...


@runtime_checkable
class LoadProvider(Protocol):
    """直接提供物理载荷对象的 Problem 契约."""

    def loads(self) -> Sequence[Load]:
        """按稳定顺序返回该 Problem 的全部非零物理载荷."""
        ...


__all__ = [
    "BodyForce",
    "BoundaryTraction",
    "LineTraction",
    "Load",
    "LoadKind",
    "LoadProvider",
    "PointForce",
]
