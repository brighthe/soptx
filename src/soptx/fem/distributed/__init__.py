"""SOPTX 分布式有限元与并行计算模块 (soptx.fem.distributed).

包含实体级 MPI 通信抽象 (EntityMPI)、网格划分与分发 (partition_cells, distribute_mesh)、
标量/向量有限元空间分布式限制 (distribute_space, distribute_vector_space)
以及重叠副本全局算子代数 (OverlapOperator).
"""

from __future__ import annotations

__all__ = [
    # 实体级 MPI 通信
    "EntityMPI",
    "SharingPair",
    "SparseData1D",
    "dist_from_masks",
    "mapped_masks",
    # 网格划分器
    "AXIS_NAMES",
    "SUPPORTED_RANKS",
    "partition_cells",
    "partition_strategy_label",
    # 分布式网格分发
    "DistMeshResult",
    "MeshComm",
    "distribute_mesh",
    # 分布式有限元空间分发
    "DistSpaceResult",
    "DistributedVectorSpace",
    "distribute_space",
    "distribute_vector_space",
    # 重叠算子代数
    "OverlapOperator",
]

from .entity_mpi import (
    EntityMPI,
    SharingPair,
    SparseData1D,
    dist_from_masks,
    mapped_masks,
)
from .mesh import (
    DistMeshResult,
    MeshComm,
    distribute_mesh,
)
from .operator import OverlapOperator
from .partitioner import (
    AXIS_NAMES,
    SUPPORTED_RANKS,
    partition_cells,
    partition_strategy_label,
)
from .space import (
    DistSpaceResult,
    DistributedVectorSpace,
    distribute_space,
    distribute_vector_space,
)
