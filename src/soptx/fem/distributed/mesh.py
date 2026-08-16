"""分布式网格分发 (Distributed Mesh) 模块.

负责将全局网格及其几何实体 (Cell/Face/Edge/Node) 分割并分发给各 MPI 进程,
并自动构建各维度实体的跨进程通信拓扑集合 DistMeshResult.
"""

from __future__ import annotations

__all__ = [
    "DistMeshResult",
    "MeshComm",
    "distribute_mesh",
]

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from mpi4py.MPI import Comm, COMM_WORLD

from fealpy.backend import backend_manager as bm
from fealpy.mesh import EntitySector, Mesh, MeshBlock, Relation
from fealpy.typing import TensorLike

from . import entity_mpi as _de


class MeshComm(NamedTuple):
    """当前网格各几何实体的跨进程通信拓扑集合."""

    entities: dict[str, _de.EntityMPI]
    root_entity_name: str


class DistMeshResult(NamedTuple):
    """分布式网格分发结果包装 (包含当前 rank 的局部网格与通信拓扑)."""

    mesh: Mesh
    comm: MeshComm


def _make_local_index(num: int, mask: TensorLike, nonlocal_value: int = 0) -> TensorLike:
    """构建全局索引到局部连续紧凑索引的重编号映射表."""
    local_index = bm.full((num,), nonlocal_value, dtype=bm.int32)
    lnum = int(bm.sum(mask))
    local_index[mask] = bm.arange(lnum, dtype=bm.int32)
    return local_index


def _build_masks_by_sector(
    storage: MeshBlock,
    root_name: str,
    root_masks: Sequence[TensorLike],
) -> dict[str, list[TensorLike]]:
    """从最高维根实体 (如单元 cell) 的切分掩码级联推导所有低维实体的所属掩码."""
    num_parts = len(root_masks)
    block_dims = {
        name: storage.get_sector(name).schema.top_dim
        for name in storage.sectors
    }
    root_dim = block_dims[root_name]

    masks_by_sector: dict[str, list[TensorLike]] = {
        name: [
            bm.full(
                (storage.get_sector(name).indices.shape[0],),
                False,
                dtype=bm.bool,
            )
            for _ in range(num_parts)
        ]
        for name in storage.sectors
    }
    masks_by_sector[root_name] = [bm.asarray(mask, dtype=bm.bool) for mask in root_masks]

    outgoing_relations: dict[str, list[tuple[str, Relation]]] = defaultdict(list)
    for (src_name, tgt_name), relation in storage.relations.items():
        if relation.src_indices is not None:
            continue
        if block_dims[src_name] <= block_dims[tgt_name]:
            continue
        outgoing_relations[src_name].append((tgt_name, relation))

    dims_desc = sorted(
        {dim for dim in block_dims.values() if dim <= root_dim},
        reverse=True,
    )
    for pid in range(num_parts):
        for dim in dims_desc:
            for src_name, src_dim in block_dims.items():
                if src_dim != dim:
                    continue

                src_mask = masks_by_sector[src_name][pid]
                if not bool(bm.any(src_mask)):
                    continue

                for tgt_name, relation in outgoing_relations.get(src_name, []):
                    selected_tgt = bm.reshape(bm.asarray(relation.tgt_indices[src_mask]), (-1,))
                    if int(selected_tgt.shape[0]) == 0:
                        continue
                    masks_by_sector[tgt_name][pid][selected_tgt] = True

    return masks_by_sector


def _build_local_storage(
    storage: MeshBlock,
    masks_by_sector: Mapping[str, Sequence[TensorLike]],
    part_id: int,
) -> MeshBlock:
    """为指定进程截取并构建局部网格存储块 MeshBlock (自动重编局部紧凑索引)."""
    root_names = list(storage.root_entity_names)
    if len(root_names) != 1:
        raise ValueError("构建局部存储时仅支持单一根实体.")

    block_dims = {name: storage.get_sector(name).schema.top_dim for name in storage.sectors}
    block_maps: dict[str, TensorLike] = {}
    selected_indices_by_block: dict[str, TensorLike] = {}

    for name, block in storage.sectors.items():
        block_mask = bm.asarray(masks_by_sector[name][part_id])
        block_map = _make_local_index(block.indices.shape[0], block_mask, nonlocal_value=-1)
        block_maps[name] = block_map
        selected_indices_by_block[name] = bm.asarray(block.indices[block_mask])

    local_relations: dict[tuple[str, str], Relation] = {}
    for key, relation in storage.relations.items():
        src_name, tgt_name = key
        if relation.src_indices is not None:
            continue
        if block_dims[src_name] <= block_dims[tgt_name]:
            continue

        src_mask = bm.asarray(masks_by_sector[src_name][part_id])
        local_tgt_indices = bm.asarray(block_maps[tgt_name][relation.tgt_indices[src_mask]])
        if bool(bm.any(local_tgt_indices < 0)):
            raise ValueError(
                f"分区 {part_id}: 关系 {key!r} 中存在未包含的非局部目标实体."
            )

        local_relations[key] = Relation(
            src_name=src_name,
            tgt_name=tgt_name,
            tgt_indices=local_tgt_indices,
            src_indices=None,
        )

    position_mask = bm.full((len(storage.positions),), False, dtype=bm.bool)
    for indices in selected_indices_by_block.values():
        flat = bm.reshape(bm.asarray(indices), (-1,))
        if int(flat.shape[0]) == 0:
            continue
        position_mask[flat] = True

    position_map = _make_local_index(len(storage.positions), position_mask, nonlocal_value=-1)
    lpos = bm.asarray(storage.positions[position_mask])
    local_storage = MeshBlock(positions=lpos, root_entity_names=root_names)

    for name, block in storage.sectors.items():
        local_indices = bm.asarray(position_map[selected_indices_by_block[name]])
        if bool(bm.any(local_indices < 0)):
            raise ValueError(
                f"分区 {part_id}: 块 {name!r} 中包含未包含的非局部位置节点."
            )

        local_storage.add_sector(
            EntitySector(
                schema_name=name,
                indices=local_indices,
                attributes=dict(block.attributes),
            ),
            root=(name in root_names),
        )

    local_storage.relations.update(local_relations)
    return local_storage


def distribute_mesh(
    mesh: Mesh | None,
    cell_masks: Sequence[TensorLike] | None,
    *,
    root: int = 0,
    comm: Comm | None = None,
) -> DistMeshResult:
    """基于 MPI 将全局网格切分并分发至各个进程, 自动构建跨进程实体通信拓扑.

    参数:
        mesh (Mesh | None): 待切分的全局网格对象, 仅在 root 进程必须提供, 其余 rank 传入 None.
        cell_masks (Sequence[TensorLike] | None): 各进程分配的单元布尔掩码序列,
            仅在 root 进程必须提供, 掩码总数必须等于 MPI 进程数, 其余 rank 传入 None.
        root (int, 可选): 负责切分与分发的 Root 进程号. 默认值为 0.
        comm (Comm, 可选): 使用的 MPI 通信子. 若为 None 则使用 MPI.COMM_WORLD.

    返回:
        DistMeshResult (NamedTuple):
        - mesh (Mesh): 当前进程所分配到的局部子网格对象 (Local Submesh).
        - comm (MeshComm): 当前网格各几何实体的跨进程通信拓扑集合:
            - entities: 包含各实体扇区 (cell, face, edge, node) 对应 EntityMPI 通信器的字典.
            - root_entity_name: 最高维根实体的名称 (通常为 "cell").
    """
    if comm is None:
        comm = COMM_WORLD

    local_storage_list: list[MeshBlock] | None = None
    gdata: dict[str, Any] = {}

    # 1. 仅在 Root 进程执行参数校验、低维实体掩码级联推导与局部存储块切分
    if comm.Get_rank() == root:
        assert mesh is not None, "root: 在 Root 进程必须提供待分发的全局网格 mesh."
        assert cell_masks is not None, "root: 在 Root 进程必须提供单元划分掩码列表 cell_masks."

        if len(cell_masks) != comm.Get_size():
            raise ValueError("root: cell_masks 的数量必须精确等于 MPI 进程总数.")

        root_names = list(mesh.block.root_entity_names)
        if len(root_names) != 1:
            raise ValueError("root: 网格必须包含且仅包含一个根实体 (通常为 cell).")

        root_name = root_names[0]
        expected_num_cells = mesh.block.get_sector(root_name).indices.shape[0]

        for cell_mask in cell_masks:
            if not bm.any(cell_mask):
                raise ValueError("root: 所有进程的单元切分掩码必须非空.")
            if int(cell_mask.shape[0]) != int(expected_num_cells):
                raise ValueError("root: 单元掩码的长度与网格根实体 (cell) 总数不一致.")

        # 级联推导所有低维几何实体 (node/edge/face) 在各 rank 上的重叠掩码
        masks_by_sector = _build_masks_by_sector(mesh.block, root_name, cell_masks)
        # 为每个进程截取并构建局部网格存储块 (自动重编局部紧凑索引)
        local_storage_list = [
            _build_local_storage(mesh.block, masks_by_sector, pid)
            for pid in range(comm.Get_size())
        ]

        gdata = {
            "root_name": root_name,
            "masks_by_sector": masks_by_sector,
        }

    # 2. MPI 集合通信: 广播实体掩码元数据, 散射分发各自的局部网格存储
    gdata = comm.bcast(gdata, root)
    lstorage = comm.scatter(local_storage_list, root)
    pmesh = Mesh(lstorage)

    # 3. 基于广播的实体掩码, 各 rank 本地构建各几何实体的跨进程通信拓扑 EntityMPI
    entities_comm: dict[str, _de.EntityMPI] = {}
    for name, masks in gdata["masks_by_sector"].items():
        entities_comm[name] = _de.dist_from_masks(masks, comm=comm)

    return DistMeshResult(
        pmesh.fealpy_api(),
        MeshComm(entities_comm, gdata["root_name"]),
    )
