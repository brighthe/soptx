"""网格划分器 (Mesh Partitioner) 模块.

负责将全局网格单元切分为满足互斥性 (Disjoint) 与完备性 (Exhaustive) 的各 rank 单元布尔掩码.
"""

from __future__ import annotations

__all__ = [
    "AXIS_NAMES",
    "SUPPORTED_RANKS",
    "partition_cells",
    "partition_strategy_label",
]

from fealpy.backend import backend_manager as bm
from fealpy.mesh import Mesh
from fealpy.typing import TensorLike

AXIS_NAMES = ("x", "y", "z")

#: 条带分区器接受任意正整数 rank 数; 每个分区仍必须至少拥有一个单元.
SUPPORTED_RANKS: tuple[int, ...] | None = None


def partition_cells(
    mesh: Mesh,
    mpi_size: int,
    *,
    split_coordinate: float,
    axis: int = 0,
) -> list[TensorLike]:
    """将网格单元划分为互斥且完备的连续条带分区.

    参数:
        mesh: 待切分的全局网格对象.
        mpi_size: MPI 进程总数, 必须为正整数.
        split_coordinate: 两 rank 时沿指定轴的几何二分中点坐标. 三个及以上
            rank 时保留该参数以兼容旧调用, 实际边界由单元重心坐标均匀分成
            ``mpi_size`` 个连续条带.
        axis: 几何切分轴 (0 为 x 轴, 1 为 y 轴, 2 为 z 轴).

    返回:
        各 rank 所分配单元的一维布尔掩码列表 list[TensorLike].
    """
    if mpi_size <= 0:
        raise ValueError(f"MPI rank 数必须为正整数, 当前收到 {mpi_size}")

    number_of_cells = mesh.number_of_cells()
    if mpi_size > number_of_cells:
        raise ValueError(
            f"MPI rank 数 {mpi_size} 超过单元数 {number_of_cells}, 必然出现空分区"
        )
    if mpi_size == 1:
        masks: list[TensorLike] = [bm.ones((number_of_cells,), dtype=bm.bool)]
    else:
        coordinates = mesh.Entity("cell").barycenter()[:, axis]
        if mpi_size == 2:
            masks = [
                coordinates < split_coordinate,
                coordinates >= split_coordinate,
            ]
        else:
            lower = float(bm.min(coordinates))
            upper = float(bm.max(coordinates))
            if not upper > lower:
                raise ValueError(
                    f"切分轴 {AXIS_NAMES[axis]!r} 上的单元重心坐标退化, 无法生成条带分区"
                )
            width = (upper - lower) / mpi_size
            masks = []
            for rank in range(mpi_size):
                left = lower + rank * width
                right = lower + (rank + 1) * width
                if rank == mpi_size - 1:
                    masks.append((coordinates >= left) & (coordinates <= upper))
                else:
                    masks.append((coordinates >= left) & (coordinates < right))

    coverage = bm.zeros((number_of_cells,), dtype=bm.int8)
    for rank, mask in enumerate(masks):
        mask = bm.asarray(mask, dtype=bm.bool)
        if mask.shape != (number_of_cells,):
            raise ValueError(f"分区 {rank} 的掩码形状异常: {mask.shape}")
        if not bm.any(mask):
            raise ValueError(f"分区 {rank} 的掩码为空 (empty partition), 未分配到任何单元")
        coverage += bm.astype(mask, bm.int8)

    if not bm.all(coverage == 1):
        raise ValueError("单元分区掩码必须严格满足互斥 (Disjoint) 与完备 (Exhaustive) 覆盖契约")

    return masks


def partition_strategy_label(
    mpi_size: int,
    split_coordinate: float,
    axis: int = 0,
) -> str:
    """生成记录在运行摘要中的可读网格分区策略标签描述."""
    if mpi_size == 1:
        return "all-cells"
    if mpi_size > 2:
        return (
            "non-overlapping-cells-striped-along-"
            f"{AXIS_NAMES[axis]}-into-{mpi_size}-parts"
        )
    return (
        "non-overlapping-cells-split-at-"
        f"{AXIS_NAMES[axis]}={split_coordinate:g}"
    )
