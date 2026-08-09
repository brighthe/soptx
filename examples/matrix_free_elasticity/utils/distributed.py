from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mpi4py.MPI import Comm
import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.distributed import (
    DistMeshResult,
    EntityMPI,
    dist_from_masks,
    distribute_space,
)
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import Mesh
from fealpy.typing import TensorLike

from utils import contract

AXIS_NAMES = ("x", "y", "z")


class OverlapOperator:
    """把局部算子提升为重叠副本布局下的全局算子

    ``__matmul__`` 的三步对应 math_spec.md 第 3.3 节的
    :math:`\\mathcal S \\circ K_{\\mathrm{loc}} \\circ \\mathcal C`: 先把加和表示
    的输入投影成一致表示, 作用局部算子, 再把结果同步归约回加和表示。单 rank 下
    ``refs`` 恒为 1 且 ``sync_add`` 是恒等, 因此串行与并行走同一段代码。

    ``__getattr__`` 转发未定义属性, 使被包装的 ``BilinearForm`` 在调用方看来
    与未包装时一致。
    """

    def __init__(self, local_operator: Any, dof_comm: EntityMPI) -> None:
        self.local_operator = local_operator
        self.dof_comm = dof_comm

    def __matmul__(self, vector: TensorLike) -> TensorLike:
        references = self.dof_comm.refs(vector.shape[-1])
        consistent_vector = self.dof_comm.sync_add(vector) / references
        local_result = self.local_operator @ consistent_vector
        return self.dof_comm.sync_add(local_result)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.local_operator, name)


@dataclass(frozen=True)
class DistributedVectorSpace:
    """A local vector space together with its overlap communicator."""

    space: TensorFunctionSpace
    dof_comm: EntityMPI


def partition_cells(
    mesh: Mesh,
    mpi_size: int,
    *,
    split_coordinate: float,
    axis: int = 0,
) -> list[np.ndarray]:
    """Split cells into disjoint, exhaustive per-rank masks."""

    number_of_cells = mesh.number_of_cells()
    if mpi_size == 1:
        masks = [np.ones(number_of_cells, dtype=bool)]
    elif mpi_size == 2:
        coordinates = np.asarray(
            mesh.Entity("cell").barycenter()
        )[:, axis]
        masks = [
            coordinates < split_coordinate,
            coordinates >= split_coordinate,
        ]
    else:
        raise ValueError(
            f"stage 1 supports {contract.SUPPORTED_RANKS} ranks, "
            f"received {mpi_size}"
        )

    coverage = np.zeros(number_of_cells, dtype=np.int8)
    for rank, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (number_of_cells,):
            raise ValueError(f"partition {rank} has shape {mask.shape}")
        if not np.any(mask):
            raise ValueError(f"partition {rank} is empty")
        coverage += mask.astype(np.int8)
    if not np.all(coverage == 1):
        raise ValueError("cell partitions must be disjoint and exhaustive")
    return masks


def partition_strategy_label(
    mpi_size: int,
    split_coordinate: float,
    axis: int = 0,
) -> str:
    """Human-readable partition description recorded in the run summary."""

    if mpi_size == 1:
        return "all-cells"
    return (
        "non-overlapping-cells-split-at-"
        f"{AXIS_NAMES[axis]}={split_coordinate:g}"
    )


def _vector_dof_masks(
    space: TensorFunctionSpace,
    cell_masks: list[np.ndarray],
) -> list[np.ndarray]:
    cell_to_dof = np.asarray(space.cell_to_dof())
    number_of_dofs = space.number_of_global_dofs()
    result: list[np.ndarray] = []

    for cell_mask in cell_masks:
        selected = cell_to_dof[np.asarray(cell_mask, dtype=bool)]
        dof_mask = np.zeros(number_of_dofs, dtype=bool)
        dof_mask[selected.reshape(-1)] = True
        result.append(dof_mask)
    return result


def distribute_vector_space(
    scalar_space: LagrangeFESpace | None,
    vector_space: TensorFunctionSpace | None,
    distributed_mesh: DistMeshResult,
    cell_masks: list[np.ndarray] | None,
    *,
    components: int,
    root: int,
    comm: Comm,
) -> DistributedVectorSpace:
    """Restrict an interleaved vector Lagrange space to each rank."""

    rank = comm.Get_rank()
    if components not in (2, 3):
        raise ValueError(
            f"vector space must have 2 or 3 components, got {components}"
        )
    if rank == root:
        if scalar_space is None or vector_space is None or cell_masks is None:
            raise ValueError("root rank must provide spaces and cell masks")
        if vector_space.dof_priority:
            raise ValueError(
                "the stage-1 layout must use interleaved shape=(-1, GD)"
            )

        masks = _vector_dof_masks(vector_space, cell_masks)
        global_cell_to_dof = np.asarray(vector_space.cell_to_dof())
        cell_dofs_by_rank = [
            np.asarray(global_cell_to_dof[mask], dtype=np.int64)
            for mask in cell_masks
        ]
    else:
        masks = None
        cell_dofs_by_rank = None

    masks = comm.bcast(masks, root=root)
    expected_global_cell_dofs = comm.scatter(cell_dofs_by_rank, root=root)

    if not hasattr(Mesh, "edge_to_ipoint") and hasattr(Mesh, "face_to_ipoint"):
        setattr(Mesh, "edge_to_ipoint", getattr(Mesh, "face_to_ipoint"))

    if comm.Get_size() == 1:
        local_scalar = scalar_space
    else:
        local_scalar = distribute_space(
            scalar_space,
            distributed_mesh,
            root=root,
            comm=comm,
        ).space
    local_vector = TensorFunctionSpace(
        local_scalar,
        shape=(-1, components),
    )

    local_mask = np.asarray(masks[rank], dtype=bool)
    global_indices = np.flatnonzero(local_mask)
    global_to_local = np.full(local_mask.size, -1, dtype=np.int64)
    global_to_local[global_indices] = np.arange(global_indices.size)
    expected_local_cell_dofs = global_to_local[expected_global_cell_dofs]

    if np.any(expected_local_cell_dofs < 0):
        raise ValueError("a cell references a vector DOF outside its partition")
    if not np.array_equal(
        np.asarray(local_vector.cell_to_dof()),
        expected_local_cell_dofs,
    ):
        raise ValueError("local and restricted-global cell DOF maps differ")
    if local_vector.number_of_global_dofs() != global_indices.size:
        raise ValueError("local vector-space size does not match its DOF mask")

    dof_comm = dist_from_masks(
        [bm.asarray(mask, dtype=bm.bool) for mask in masks],
        comm=comm,
    )
    return DistributedVectorSpace(local_vector, dof_comm)
