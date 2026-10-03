# 移植自 brighthe/fealpy ``fealpy/mesh/ipoints.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

from collections.abc import Iterable
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

from ..backend import bm
from ..backend import Tensor, dtype

if TYPE_CHECKING:
    from .schema import EntitySchema
    from .view.entity_view import EntityView
    from .view.mesh_view import MeshView


__all__ = [
    "MultiIndex",
    "multi_index_sort",
    "multi_index_tensorprod",
    "to_ipoint",
    "to_ipoint_permutation",
    "ipoints",
]


class MultiIndex:
    @classmethod
    def multi_index_matrix(cls, p: int, n: int, *, dtype: dtype | None = None) -> Tensor:
        """Generate the multi-index matrix for interpolation points of
        degree p with n vertices. The multi-index matrix is of shape
        (C(p+n-1, n-1), n) and each row corresponds to the multi-index of
        an interpolation point.

        Parameters:
            p (int): Degree of interpolation.
            n (int): Number of vertices in the Simplex.
            dtype (dtype, optional): Data type of the output tensor. If None, it will
                default to int32.

        Returns:
            Tensor: A tensor of shape (C(p+n-1, n-1), n) containing the multi-indices.
        """
        if dtype is None:
            dtype = bm.int32

        sep = bm.flip(bm.asarray(
            tuple(combinations_with_replacement(range(p+1), n-1)),
            dtype=dtype
        ), axis=0)
        raw = bm.zeros((sep.shape[0], n+1), dtype=dtype)
        raw[:, -1] = p
        raw[:, 1:-1] = sep
        return (raw[:, 1:] - raw[:, :-1])

    @classmethod
    def multi_index_inner(cls, p: int, n: int, *, dtype: dtype | None = None) -> Tensor:
        """Generate the multi-index corresponding to the inner interpolation
        points of degree p with n vertices.

        See also: `multi_index_matrix`."""
        if p < n:
            if dtype is None:
                dtype = bm.int32
            return bm.zeros((0, n), dtype=dtype)
        return cls.multi_index_matrix(p - n, n, dtype=dtype) + 1


def multi_index_sort(multi_index: Tensor, /) -> Tensor:
    """Return the indices to sort multi-indices according to the predefined orientation."""
    NV = multi_index.shape[-1]
    count = bm.sum(multi_index != 0, axis=1)
    nonzero_row, nonzero_col = bm.nonzero(multi_index)
    rank = bm.zeros_like(count, dtype=bm.int64)
    rank = bm.index_add(rank, nonzero_row, NV**nonzero_col) # type: ignore[call-overload]
    arg = bm.lexsort(tuple(multi_index.T) + (rank, count)) # type: ignore[call-overload]
    return arg


def _local_face_groups(
    schema: "EntitySchema",
) -> list[tuple[int, list[tuple[int, ...]]]]:
    """Return ``(top_dim, vertex rows)`` for every local subentity group."""
    groups: list[tuple[int, list[tuple[int, ...]]]] = []
    for top_dim in range(1, schema.top_dim):
        for group in schema.local_entity_groups(top_dim):
            child_vertices = group.schema.local_vertices()
            rows = [
                tuple(row[vertex] for vertex in child_vertices)
                for row in group.local_node_indices
            ]
            groups.append((group.schema.top_dim, rows))
    return groups


def _multi_index_sort_by_local_groups(
    multi_index: Tensor,
    local_groups: list[tuple[int, list[tuple[int, ...]]]],
) -> Tensor:
    num_multi_index = multi_index.shape[0]
    NV = multi_index.shape[-1]
    TOTAL = bm.sum(multi_index, axis=-1)
    topdim = bm.zeros((num_multi_index,), dtype=bm.int8)
    face_type_index = bm.zeros((num_multi_index,), dtype=bm.int8)
    face_instance_index = bm.zeros((num_multi_index,), dtype=bm.int8)
    weights = bm.zeros_like(multi_index, dtype=bm.int64)

    for fti, (TD, local_faces) in enumerate(local_groups):
        for fii, local_face in enumerate(local_faces):
            mask = bm.logical_and(
                bm.sum(multi_index[:, local_face], axis=-1) == TOTAL,
                bm.all(multi_index[:, local_face] != 0, axis=-1)
            )
            mask_idx = bm.nonzero(mask)[0]
            topdim[mask_idx] = TD
            face_type_index[mask_idx] = fti
            face_instance_index[mask_idx] = fii
            new_weights = bm.zeros((mask_idx.shape[0], NV), dtype=bm.int64)
            new_weights[:, local_face] = NV**bm.arange(len(local_face))[None, :]
            weights[mask_idx] = new_weights

    mask = bm.all(multi_index != 0, axis=-1)
    topdim[mask] = 127

    rank = bm.sum(multi_index * weights, axis=-1)
    arg = bm.lexsort((rank, face_instance_index, face_type_index, topdim))
    return arg


def multi_index_tensorprod(
    broadcast_multi_index: Tensor,
    split_indices: tuple[int, ...] | None = None
) -> Tensor:
    """Compute the tensor product between split multi-indices.

    Do nothing if split_indices is None, as no other operand is provided to
    perform the tensor product with."""
    from functools import reduce

    if split_indices is not None:
        mi_tuple = bm.split(broadcast_multi_index, split_indices, axis=-1)
    else:
        return broadcast_multi_index

    def kron_last_dim(a: Tensor, b: Tensor) -> Tensor:
        if bm.size(a) == 0:
            return a
        return (a[..., :, None] * b[..., None, :]).reshape(*a.shape[:-1], -1) # type: ignore[return-value]

    return reduce(kron_last_dim, reversed(mi_tuple))


def _vertex_column_permutation(schema: "EntitySchema") -> list[int] | None:
    """Column indices reordering ``multi_index`` into local vertex numbering.

    Return ``None`` when ``multi_index`` already numbers its columns by local
    vertex, which is the case for every simplex-based Schema.
    """
    columns = schema.multi_index_vertex_columns()
    if columns is None:
        return None
    inverse = [0] * len(columns)
    for column, vertex in enumerate(columns):
        inverse[vertex] = column
    return inverse


def _vertex_orientation_to_ipoint_permutation(
    schema: "EntitySchema",
    order: tuple[int, ...],
) -> dict[tuple[int, ...], Tensor]:
    """Build vertex-orientation -> local ipoint permutation.

    Keys are the vertex automorphisms reported by ``vertex_permutations()``,
    matching what ``EntityView.global_permutations`` produces.  Values are the
    column permutation carrying the sub-entity's own internal-point order into
    the order the parent expects under that relative orientation.
    """
    # 用重心键直接查表, 不走 multi_index_sort.
    #
    # 原实现取 tensorprod=False 的多重指标, 并以 schema.orientation 为键. 这两
    # 处对四边形都不成立: orientation 是重心分量列的置换 (含 (2, 0, 3, 1) 这类
    # 非顶点对称), 而 global_permutations 给出的是顶点置换, 8 个键里只有 4 个
    # 碰巧对得上, 其余定向的面内部点根本没被重排. 后果同样是静默的: 六面体
    # p >= 3 时部分面的内部自由度与基函数错位.
    mi = schema.multi_index(order, internal=True, tensorprod=True)
    columns = _vertex_column_permutation(schema)
    if columns is not None:
        mi = mi[:, columns]
    keys = [tuple(int(weight) for weight in row) for row in bm.to_numpy(mi)]
    lookup = {key: index for index, key in enumerate(keys)}

    result: dict[tuple[int, ...], Tensor] = {}
    for vertex_order in schema.vertex_permutations():
        permuted = [tuple(key[vertex] for vertex in vertex_order) for key in keys]
        result[tuple(vertex_order)] = bm.asarray(
            [lookup[key] for key in permuted],
            dtype=bm.int64,
        )
    return result


def to_ipoint(
    mesh: "MeshView",
    entity: "EntityView | str",
    order: int,
) -> Tensor:
    """Get the interpolation point indices for the given entity and order,
    in unstructured meshes.
    The interpolation point indices are ordered from lower-dimensional
    sub-entities to higher-dimensional entities, and the interpolation points
    of each sub-entity are ordered according to the vertex orientation.

    Parameters:
        mesh (Mesh): The mesh object.
        entity (EntityView | str): The target entity view or an unambiguous
            sector id / role accepted by ``mesh.entity_view``.
        order (int): The degree of interpolation.

    Returns:
        Tensor: A tensor of shape (num_entities, num_ip) containing the
            interpolation point indices.
    """
    from .view.entity_view import EntityView

    if isinstance(entity, EntityView):
        tgt_entity = entity
    else:
        tgt_entity = mesh.entity_view(entity)

    td = tgt_entity.top_dimension()
    root_id = tgt_entity.sector.source_cell_sector_id or tgt_entity.sector_id

    subentities_by_dim: dict[int, list[EntityView]] = {dim: [] for dim in range(td + 1)}
    for sector in mesh.block.sectors.values():
        if sector.id == "node":
            continue
        if sector.source_cell_sector_id == root_id and sector.schema.top_dim < td:
            subentities_by_dim[sector.schema.top_dim].append(
                EntityView(mesh.block, sector.id)
            )
    subentities_by_dim[td] = [tgt_entity]

    collected: list[Tensor] = []

    local_vertices = tgt_entity.schema.local_vertices()
    vertex_node_ids = bm.reshape(
        tgt_entity.indices[:, local_vertices],
        (-1,),
    )
    _, vertex_inverse = bm.unique(vertex_node_ids, return_inverse=True)
    vertex_map = bm.reshape(
        vertex_inverse,
        (tgt_entity.size(), len(local_vertices)),
    )
    collected.append(vertex_map)
    ip_cursor = 0 if vertex_map.size == 0 else int(bm.max(vertex_map)) + 1

    for dim in range(1, td + 1):
        for subentity in subentities_by_dim[dim]:
            num_sub_entity = subentity.size()
            num_internal_ip = subentity.num_multi_index(order, internal=True)
            if num_internal_ip == 0:
                continue

            sub_map = bm.arange(
                ip_cursor,
                ip_cursor + num_sub_entity * num_internal_ip,
                dtype=bm.int64,
            )
            sub_map = bm.reshape(sub_map, (num_sub_entity, num_internal_ip))

            if dim == td:
                full_map = bm.reshape(sub_map, (tgt_entity.size(), -1))
                collected.append(full_map)
                ip_cursor += num_sub_entity * num_internal_ip
                continue

            tgt_to_sub = tgt_entity.to(subentity).tgt_indices
            full_map = bm.reshape(sub_map[tgt_to_sub], (-1, num_internal_ip))
            global_vo = tgt_entity.global_permutations(subentity)
            global_vo = bm.reshape(global_vo, (-1, global_vo.shape[-1]))

            for vo, do in _vertex_orientation_to_ipoint_permutation(
                subentity.schema,
                (order,),
            ).items():
                vo = bm.asarray(vo, dtype=bm.uint8, device=full_map.device)
                vo_mask = bm.all(global_vo == vo[None, :], axis=-1)
                full_map = bm.where(vo_mask[:, None], full_map[:, do], full_map)

            full_map = bm.reshape(full_map, (tgt_entity.size(), -1))
            collected.append(full_map)
            ip_cursor += num_sub_entity * num_internal_ip

    if not collected:
        return bm.zeros(
            (tgt_entity.size(), 0),
            dtype=bm.int64,
            device=mesh.block.positions.device,
        )

    result = bm.concat(collected, axis=1)
    permutation = to_ipoint_permutation(tgt_entity.schema, (order,))

    if permutation is not None:
        permutation = bm.device_put(permutation, result.device)
        result = result[:, permutation]
    return result


def to_ipoint_permutation(schema: type["EntitySchema"], order: tuple[int, ...]) -> Tensor | None:
    """Column permutation from topological ipoint order to basis order.

    ``to_ipoint`` builds interpolation-point indices in topological order
    (lower-dimensional sub-entities first). Schemas whose basis functions
    use a different local ordering may override this hook to return the
    column permutation that aligns the mapping with ``multi_index`` and
    shape-function order.
    """
    from .schema.classic.base import _TensorProductOrderSchema

    # 张量积单元 (四边形/六面体/三棱柱) 不做这次重排.
    #
    # to_ipoint 的拼装顺序 (顶点 -> 各维子实体内部点 -> 单元内部点) 恰好就是
    # _node_keys() 序, 而张量积单元的 shape_function 也已经排成 _node_keys()
    # 序 (见 view/entity_view.py 的 _legacy_basis_indices), 两侧本就对齐.
    # 再排一次反而错位; 且这里的排序依据 _local_face_groups 用的是顶点序,
    # 与 multi_index 的列序不一致, 排出来的结果本身也是错的.
    if isinstance(schema, _TensorProductOrderSchema):
        return None

    mi = schema.multi_index(order, tensorprod=True)
    natural_to_topological = _multi_index_sort_by_local_groups(
        mi,
        _local_face_groups(schema),
    )
    return bm.argsort(natural_to_topological)


def ipoints(mesh: "MeshView", order: int | tuple[int, ...], names: Iterable[str]) -> Tensor:
    """Get the interpolation points for the given entity and order.

    Parameters:
        mesh (MeshView): The mesh object.
        order (int | tuple[int, ...]): The degree of interpolation.
        names (Iterable[str]): The names of the entities for which to compute
            interpolation points. For example, ["tet", "hex"].

    Returns:
        Tensor: A tensor of shape (num_ip, GD) containing the interpolation points.
    """
    if isinstance(order, int):
        order = (order,)
    if not order or any(p <= 0 for p in order):
        raise ValueError(f"order must be positive, got {order!r}")

    device = mesh.block.positions.device

    collected = []
    for subentity in [mesh.entity_view(entity) for entity in names]:
        if subentity.top_dimension() == 0:
            cell_view = mesh.cell_view
            local_vertices = cell_view.schema.local_vertices()
            vertex_ids = bm.reshape(
                cell_view.indices[:, local_vertices],
                (-1,),
            )
            unique_vertex_ids = bm.unique(vertex_ids)
            points = mesh.block.positions[unique_vertex_ids]
            collected.append(bm.reshape(points, (-1, mesh.geo_dimension())))
            continue

        mi = subentity.schema.multi_index(order, internal=True)
        mi = bm.device_put(mi, device)
        if mi.shape[0] == 0:
            continue

        vertices = subentity.indices[:, subentity.schema.local_vertices()]
        points = mesh.block.positions[vertices]
        # points 的第 2 个轴按局部顶点序号排列, 而张量积单元的 multi_index 按
        # 因子嵌套序编号列, 两者相差一个固定置换 (四边形为 (0, 1, 3, 2)).
        # 不对齐就会把权重扣到错误的顶点上: 四边形 p >= 3 的单元内部点会落到
        # 网格上不存在的位置, 而 p <= 2 因为内部点权重恰好对称而侥幸正确.
        columns = _vertex_column_permutation(subentity.schema)
        if columns is not None:
            mi = mi[:, columns]
        weights = mi / bm.sum(mi, axis=-1, keepdims=True)
        points = bm.einsum("qv, evd -> eqd", weights, points)
        collected.append(bm.reshape(points, (-1, mesh.geo_dimension())))

    if not collected:
        return bm.zeros((0, mesh.geo_dimension()), dtype=mesh.block.positions.dtype, device=device)

    return bm.concat(collected, axis=0)
