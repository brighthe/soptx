"""边界牵引的离散表示.

强施加 (本质边界) 与弱施加 (自然边界) 对同一个连续牵引给出的离散载荷并不相同:
弱施加走边界数值积分, 强施加走迹空间插值. 只要牵引在单元内部不连续, 两者都不精确,
且不精确的方式不同 —— 强施加尤其没有出路, 因为跨边连续的迹空间在跳变点处只有一个
单值自由度, 装不下跳跃.

本模块把这类载荷先投影到边界的连续 P1 迹空间上: 常数落在该空间内, 所以 L2 投影
精确保持合力; 而连续分片线性函数又能被足够高阶的迹空间精确插值、被足够高阶的
数值积分精确积分. 于是两条离散路径看到的是同一个离散载荷泛函.
"""

from __future__ import annotations

from dataclasses import dataclass

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike


@dataclass(frozen=True)
class P1TraceLoad:
    """轴对齐边界直线上的连续分片线性牵引.

    Attributes
    ----------
    - start : 载荷所在直线沿 ``axis`` 方向的起点坐标
    - h : 该直线上 P1 网格的单元尺寸
    - coefficients : ``(n_cells + 1,)`` 的 P1 节点系数
    - level : 该直线在垂直方向上的坐标, 用于把非本条边界上的点置零
    - axis : 直线所沿的坐标轴 (二维中 0 表示水平边, 1 表示竖直边)
    - component : 牵引作用的分量下标
    - tolerance : 判定"点落在该直线上"的几何容差
    """

    start: float
    h: float
    coefficients: TensorLike
    level: float
    axis: int = 0
    component: int = 1
    tolerance: float = 1.0e-12

    @cartesian
    def __call__(self, points: TensorLike) -> TensorLike:
        """在物理坐标 ``points`` 上求值, 非本条边界上的点返回零向量."""
        along = points[..., self.axis]
        across = points[..., 1 - self.axis]

        n_cells = self.coefficients.shape[0] - 1
        coordinate = bm.divide(bm.subtract(along, bm.full_like(along, self.start)), self.h)
        cell = bm.clip(bm.astype(bm.floor(coordinate), bm.int64), 0, n_cells - 1)
        xi = bm.subtract(coordinate, bm.astype(cell, coordinate.dtype))
        value_along = bm.add(
            bm.multiply(bm.subtract(1.0, xi), self.coefficients[cell]),
            bm.multiply(xi, self.coefficients[cell + 1]),
        )

        on_line = bm.less_equal(
            bm.abs(bm.subtract(across, bm.full_like(across, self.level))),
            bm.full_like(across, self.tolerance),
        )
        traction = bm.zeros(points.shape, **bm.context(points))
        traction = bm.set_at(traction, (..., self.component), value_along)

        return bm.where(bm.expand_dims(on_line, axis=-1), traction, bm.zeros_like(traction))

    # 兼容按名字调用的场景 (fealpy 的部分接口按 ``obj.traction(points)`` 取值)
    @cartesian
    def traction(self, points: TensorLike) -> TensorLike:
        """``__call__`` 的具名别名."""
        return self(points)

    def resultant(self) -> float:
        """返回该 P1 载荷沿直线的合力分量, 用于核查投影是否守恒."""
        weights = bm.full_like(self.coefficients, self.h)
        weights = bm.set_at(weights, 0, self.h / 2.0)
        weights = bm.set_at(weights, -1, self.h / 2.0)
        # 内点权重为 h, 端点为 h/2, 即梯形公式; 对 P1 函数精确
        return float(bm.sum(bm.multiply(weights, self.coefficients)))


def project_patch_traction_to_p1_trace(
    *,
    line: tuple[float, float],
    n_cells: int,
    level: float,
    patch: tuple[float, float],
    intensity: float,
    axis: int = 0,
    component: int = 1,
) -> P1TraceLoad:
    """把区间常值牵引精确 L2 投影到边界的连续 P1 迹空间.

    投影目标是使 ``\\int (\\Pi g - g) v = 0`` 对该空间中所有 ``v`` 成立. 常数函数
    在空间内, 因此 ``\\int \\Pi g = \\int g``, 合力被精确保持.

    右端项按贴片与单元的**解析重叠**积分, 不使用数值积分: 被积函数在单元内部
    不连续时, 数值积分本身就是误差来源, 而这正是本函数要消除的东西.

    Parameters
    ----------
    - line : 载荷所在直线沿 ``axis`` 方向的起止坐标 ``(start, end)``
    - n_cells : 该直线上的均匀单元数
    - level : 该直线在垂直方向上的坐标
    - patch : 常值牵引所占的区间 ``(left, right)``, 允许与单元边界不对齐
    - intensity : 贴片区间内的常值牵引强度 (即合力除以贴片长度)
    - axis : 直线所沿的坐标轴
    - component : 牵引作用的分量下标

    Notes
    -----
    质量矩阵按稠密矩阵组装并求解. 该矩阵是 ``(n_cells + 1)`` 阶三对角阵, 边界
    分辨率在 1e3 量级以内时代价可忽略; 若将来需要更细的边界剖分, 应换成三对角
    求解而不是继续放大稠密矩阵.
    """
    start, end = float(line[0]), float(line[1])
    left, right = float(patch[0]), float(patch[1])
    if n_cells <= 0:
        raise ValueError("n_cells 必须为正整数.")
    if end <= start:
        raise ValueError("line 的终点必须大于起点.")
    if right <= left:
        raise ValueError("patch 的右端必须大于左端.")
    if left < start or right > end:
        raise ValueError("patch 必须落在 line 之内.")
    if axis not in (0, 1):
        raise ValueError("axis 仅支持 0 或 1.")

    h = (end - start) / n_cells
    rhs = bm.zeros((n_cells + 1,), dtype=bm.float64)
    mass = bm.zeros((n_cells + 1, n_cells + 1), dtype=bm.float64)
    for cell in range(n_cells):
        x0, x1 = start + cell * h, start + (cell + 1) * h

        # 贴片与本单元的重叠区间, 空重叠时右端项无贡献
        a, b = max(x0, left), min(x1, right)
        if b > a:
            length = b - a
            difference = b * b - a * a
            rhs = bm.set_at(rhs, cell, rhs[cell] + intensity * (x1 * length - 0.5 * difference) / h)
            rhs = bm.set_at(rhs, cell + 1, rhs[cell + 1] + intensity * (0.5 * difference - x0 * length) / h)

        # P1 单元质量矩阵 h/6 * [[2, 1], [1, 2]]
        mass = bm.set_at(mass, (cell, cell), mass[cell, cell] + h / 3.0)
        mass = bm.set_at(mass, (cell + 1, cell + 1), mass[cell + 1, cell + 1] + h / 3.0)
        mass = bm.set_at(mass, (cell, cell + 1), mass[cell, cell + 1] + h / 6.0)
        mass = bm.set_at(mass, (cell + 1, cell), mass[cell + 1, cell] + h / 6.0)

    return P1TraceLoad(
        start=start,
        h=h,
        coefficients=bm.linalg.solve(mass, rhs),
        level=float(level),
        axis=axis,
        component=component,
    )
