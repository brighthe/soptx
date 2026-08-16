"""分布式有限元空间分发 (Distributed Space) 模块.

负责将全局标量 Lagrange 空间与多维交错向量空间 (TensorFunctionSpace) 分发并限制到各个 MPI 进程,
并构建对应的自由度跨进程通信子 dof_comm.
"""

from __future__ import annotations

__all__ = [
    "DistSpaceResult",
    "DistributedVectorSpace",
    "distribute_space",
    "distribute_vector_space",
]

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, TypeVar

from mpi4py.MPI import Comm, COMM_WORLD
import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import FunctionSpace, LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import Mesh
from fealpy.typing import TensorLike

from . import entity_mpi as _de
from . import mesh as _dm

_ST_co = TypeVar("_ST_co", bound="FunctionSpace")
_S = slice(None)


class DistSpaceResult(NamedTuple, Generic[_ST_co]):
    """标量分布式有限元空间分发结果包装."""

    space: _ST_co
    dofcomm: _de.EntityMPI


@dataclass(frozen=True)
class DistributedVectorSpace:
    """局部向量有限元空间及其重叠通信子."""

    space: TensorFunctionSpace
    dof_comm: _de.EntityMPI


def _mk_distributed_space_type(kind: type[_ST_co]) -> type[_ST_co]:
    """动态重构带有局部 DOF 映射重载的分布式空间类型."""

    def __repr__(self: _ST_co) -> str:
        return f"<Distributed{kind.__name__} object at {hex(id(self))}>"

    def cell_to_dof(self: Any, index: Any = _S) -> Any:
        return self.cell2dof[index]

    def face_to_dof(self: Any, index: Any = _S) -> Any:
        return self.face2dof[index]

    def edge_to_dof(self: Any, index: Any = _S) -> Any:
        return self.edge2dof[index]

    class_namespace = {
        "__repr__": __repr__,
        "cell_to_dof": cell_to_dof,
        "face_to_dof": face_to_dof,
        "edge_to_dof": edge_to_dof,
    }
    return type(f"Dist{kind.__name__}", (kind,), class_namespace)  # type: ignore


def _face_entity_name(mesh: Mesh, entity_comms: dict[str, _de.EntityMPI]) -> str:
    """返回网格最高余维一实体的通信器名称.

    二维网格的 face 为 ``segment``; 三维单纯形网格的 face 为 ``tri``,
    三维六面体网格的 face 为 ``quad``. 该选择必须由拓扑维数决定, 不能仅依据
    ``quad`` 是否存在, 因为二维四边形网格的 ``quad`` 是根单元而不是 face.
    """
    if mesh.top_dimension() == 2:
        return "segment"
    for name in ("tri", "quad"):
        if name in entity_comms:
            return name
    raise ValueError("三维网格缺少 tri 或 quad 面实体通信器.")


def _edge_to_dof(space: Any) -> TensorLike:
    """返回全局边与标量自由度的映射.

    对 ``p=1`` 连续 Lagrange 空间, 边自由度就是端点节点自由度. 部分 FEALPy
    结构网格尚未实现 ``Mesh.edge_to_ipoint``; 不能把三维面的 ``face_to_ipoint``
    伪装成边映射, 否则六面体网格会以面索引访问边 DOF 并越界.
    """
    if hasattr(space.mesh, "edge_to_ipoint"):
        return space.edge_to_dof()
    if getattr(space, "p", None) != 1:
        raise NotImplementedError("缺少 edge_to_ipoint 时仅支持 p=1 边 DOF 映射.")
    return bm.asarray(space.mesh.Entity("segment").indices, copy=True)


def distribute_space(
    space: _ST_co | None,
    distributed_mesh: _dm.DistMeshResult,
    *,
    root: int = 0,
    comm: Comm | None = None,
) -> DistSpaceResult[_ST_co]:
    """将全局标量有限元空间限制并分发至各 MPI 进程.

    参数:
        space: 待分发的全局标量有限元空间 (仅在 root 进程必须提供).
        distributed_mesh: 各进程持有的分布式网格分发结果.
        root: Root 进程编号.
        comm: 使用的 MPI 通信子.

    返回:
        DistSpaceResult: 当前进程的局部标量空间与自由度通信子 dofcomm.
    """
    if comm is None:
        comm = COMM_WORLD

    pmesh, mcomm = distributed_mesh
    root_entity = mcomm.entities[mcomm.root_entity_name]
    face_entity_name = _face_entity_name(pmesh, mcomm.entities)
    face_entity = mcomm.entities[face_entity_name]
    edge_entity = mcomm.entities["segment"]
    all_cell_global_indices = comm.gather(root_entity._global_indices, root=root)
    all_face_global_indices = comm.gather(face_entity._global_indices, root=root)
    all_edge_global_indices = comm.gather(edge_entity._global_indices, root=root)

    lcell2dof = None
    lface2dof = None
    ledge2dof = None
    gdata: dict[str, Any] = {}

    if comm.Get_rank() == root:
        assert space is not None, "root: 在 Root 进程必须提供全局标量空间 space."
        assert all_cell_global_indices is not None
        assert all_face_global_indices is not None
        assert all_edge_global_indices is not None

        # ``FunctionSpace`` 基类未声明网格拓扑接口, 但本分发路径只接受 Lagrange
        # 空间及其运行时子类, 因此在此处通过 ``Any`` 表达该受限运行时契约.
        lagrange_space: Any = space
        is3D = lagrange_space.mesh.top_dimension() == 3
        cell2dof = lagrange_space.cell_to_dof()
        face2dof = lagrange_space.face_to_dof()
        edge2dof = _edge_to_dof(lagrange_space)
        lcell2dof = [bm.asarray(cell2dof[m], copy=True) for m in all_cell_global_indices]
        lface2dof = [bm.asarray(face2dof[m], copy=True) for m in all_face_global_indices]
        ledge2dof = [bm.asarray(edge2dof[m], copy=True) for m in all_edge_global_indices]

        gdata = {
            "space_type": type(lagrange_space),
            "p": getattr(lagrange_space, "p", 1),
            "is3D": is3D,
            "NDOF": lagrange_space.number_of_global_dofs(),
        }

    gdata = comm.bcast(gdata, root=root)
    lcell2dof = comm.scatter(lcell2dof, root)
    lface2dof = comm.scatter(lface2dof, root)
    ledge2dof = comm.scatter(ledge2dof, root)

    dof_mask = bm.zeros((gdata["NDOF"],), dtype=bm.bool)
    dof_mask[lcell2dof] = True
    dof_masks = comm.alltoall([dof_mask] * comm.Get_size())
    dofcomm = _de.dist_from_masks(dof_masks, comm=comm)

    space_type = _mk_distributed_space_type(gdata["space_type"])
    pspace = space_type(pmesh.fealpy_api(), gdata["p"])

    local_index = _dm._make_local_index(gdata["NDOF"], dof_mask)
    pspace.cell2dof = bm.asarray(local_index[lcell2dof], copy=True)
    pspace.face2dof = bm.asarray(local_index[lface2dof], copy=True)
    pspace.edge2dof = bm.asarray(local_index[ledge2dof], copy=True)

    return DistSpaceResult(pspace, dofcomm)


def _vector_dof_masks(
    space: TensorFunctionSpace,
    cell_masks: Sequence[TensorLike],
) -> list[np.ndarray]:
    """推导各 rank 单元对应的向量自由度布尔掩码列表."""
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
    distributed_mesh: _dm.DistMeshResult,
    cell_masks: Sequence[TensorLike] | None,
    *,
    components: int,
    root: int = 0,
    comm: Comm | None = None,
) -> DistributedVectorSpace:
    """将交错布局的多维向量拉格朗日有限元空间分发并限制到各个 rank.

    参数:
        scalar_space: 全局标量拉格朗日空间 (仅在 root 进程必须提供).
        vector_space: 全局交错向量空间 (仅在 root 进程必须提供).
        distributed_mesh: 分布式网格分发结果.
        cell_masks: 各 rank 单元掩码序列 (仅在 root 进程必须提供).
        components: 向量场分量维数 (2 或 3).
        root: Root 进程编号 (默认 0).
        comm: 使用的 MPI 通信子.

    返回:
        DistributedVectorSpace: 局部向量有限元空间与自由度通信子 dof_comm.
    """
    if comm is None:
        comm = COMM_WORLD

    rank = comm.Get_rank()
    if components not in (2, 3):
        raise ValueError(f"向量有限元空间必须具有 2 或 3 个分量, 收到 {components}")

    if rank == root:
        if scalar_space is None or vector_space is None or cell_masks is None:
            raise ValueError("Root 进程必须提供标量空间、向量空间与单元掩码列表.")
        if vector_space.dof_priority:
            raise ValueError("阶段 1 向量空间布局必须使用交错排列 shape=(-1, GD).")

        masks = _vector_dof_masks(vector_space, cell_masks)
        global_cell_to_dof = np.asarray(vector_space.cell_to_dof())
        cell_dofs_by_rank = [
            np.asarray(global_cell_to_dof[np.asarray(mask, dtype=bool)], dtype=np.int64)
            for mask in cell_masks
        ]
    else:
        masks = None
        cell_dofs_by_rank = None

    masks = comm.bcast(masks, root=root)
    expected_global_cell_dofs = comm.scatter(cell_dofs_by_rank, root=root)

    if comm.Get_size() == 1:
        local_scalar = scalar_space
    else:
        local_scalar = distribute_space(
            scalar_space,
            distributed_mesh,
            root=root,
            comm=comm,
        ).space

    assert local_scalar is not None, "局部标量空间构造失败 (为 None)."
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
        raise ValueError("单元引用了其所属分区之外的未知向量自由度.")
    if not np.array_equal(
        np.asarray(local_vector.cell_to_dof()),
        expected_local_cell_dofs,
    ):
        raise ValueError("局部空间与全局限制导出的 cell_to_dof 映射不一致.")
    if local_vector.number_of_global_dofs() != global_indices.size:
        raise ValueError("局部向量空间自由度总数与 DOF 掩码数量不匹配.")

    dof_comm = _de.dist_from_masks(
        [bm.asarray(mask, dtype=bm.bool) for mask in masks],
        comm=comm,
    )
    return DistributedVectorSpace(local_vector, dof_comm)
