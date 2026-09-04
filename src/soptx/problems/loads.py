"""物理载荷契约的通用可调用实现."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from fealpy.decorator import cartesian


@dataclass(frozen=True)
class BodyForceLoad:
    """由可调用体力场构造的 ``BodyForce`` 对象."""

    dimension: int
    value: Callable[..., Any] = field(repr=False)
    kind: Literal["body_force"] = field(init=False, default="body_force")

    @property
    def support_dimension(self) -> int:
        """返回体力作用实体（单元）的维数."""
        return self.dimension

    @cartesian
    def body_force(self, points: Any, *, time: float | None = None) -> Any:
        """在物理坐标处求值体力场."""
        del time
        return self.value(points)


@dataclass(frozen=True)
class PointForceLoad:
    """由作用点和合力向量构造的 ``PointForce`` 对象."""

    point: Sequence[float]
    vector: Sequence[float]
    kind: Literal["point_force"] = field(init=False, default="point_force")

    def __post_init__(self) -> None:
        """验证作用点与力向量的维数一致."""
        point = tuple(float(value) for value in self.point)
        vector = tuple(float(value) for value in self.vector)
        if not point or len(point) != len(vector):
            raise ValueError("point 与 vector 必须具有相同的正维数.")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "vector", vector)

    @property
    def dimension(self) -> int:
        """返回集中力所在物理空间的维数."""
        return len(self.point)

    @property
    def support_dimension(self) -> int:
        """返回集中力作用实体（点）的维数零."""
        return 0

    def force(self, *, time: float | None = None) -> tuple[float, ...]:
        """返回集中力合力向量."""
        del time
        return tuple(self.vector)


@dataclass(frozen=True)
class LineTractionLoad:
    """由区域标记和单位长度牵引构造的 ``LineTraction`` 对象."""

    dimension: int
    marker: Callable[..., Any] = field(repr=False)
    value: Callable[..., Any] = field(repr=False)
    kind: Literal["line_traction"] = field(init=False, default="line_traction")

    @property
    def support_dimension(self) -> int:
        """返回线载荷作用实体（线）的维数一."""
        return 1

    @cartesian
    def is_load_line(self, points: Any) -> Any:
        """返回载荷作用线的布尔标记."""
        return self.marker(points)

    @cartesian
    def traction(
        self,
        points: Any,
        *,
        tangents: Any | None = None,
        time: float | None = None,
    ) -> Any:
        """返回单位长度上的牵引向量."""
        del tangents, time
        return self.value(points)


@dataclass(frozen=True)
class BoundaryTractionLoad:
    """由边界标记和牵引函数构造的 ``BoundaryTraction`` 对象."""

    dimension: int
    marker: Callable[..., Any] = field(repr=False)
    value: Callable[..., Any] = field(repr=False)
    kind: Literal["boundary_traction"] = field(
        init=False,
        default="boundary_traction",
    )

    @property
    def support_dimension(self) -> int:
        """返回计算域边界的维数."""
        return self.dimension - 1

    @cartesian
    def is_load_boundary(self, points: Any) -> Any:
        """返回载荷作用边界的布尔标记."""
        return self.marker(points)

    @cartesian
    def traction(
        self,
        points: Any,
        *,
        normals: Any | None = None,
        time: float | None = None,
    ) -> Any:
        """返回单位边界测度上的牵引向量."""
        del normals, time
        return self.value(points)


__all__ = [
    "BodyForceLoad",
    "BoundaryTractionLoad",
    "LineTractionLoad",
    "PointForceLoad",
]
