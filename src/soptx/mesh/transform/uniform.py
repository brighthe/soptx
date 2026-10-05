# 移植自 brighthe/fealpy ``fealpy/mesh/transform/uniform.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""各类网格 Schema 的一致加密算法."""

from __future__ import annotations

from collections.abc import Callable

from ...backend import bm
from ...backend import Tensor
from ...sparse import csr_matrix
from ..storage import MeshBlock
from ..topology.builder import TopologyBuilder

__all__ = [
    "uniform_refine",
    "uniform_refine_edge",
    "uniform_refine_triangle",
    "uniform_refine_quadrilateral",
    "uniform_refine_tetrahedron",
    "uniform_refine_prism",
    "uniform_refine_hexahedron",
]


RefineFunc = Callable[..., list | None]


def _arange_like(start: int, stop: int, ref: Tensor) -> Tensor:
    return bm.arange(start, stop, dtype=ref.dtype, device=bm.get_device(ref))


def _empty_like(shape: tuple[int, int], ref: Tensor) -> Tensor:
    return bm.zeros(shape, dtype=ref.dtype, device=bm.get_device(ref))


def _barycenter(block: MeshBlock, indices: Tensor) -> Tensor:
    return bm.mean(block.positions[indices], axis=1)


def _root_names(block: MeshBlock) -> set[str]:
    return set(block.root_cell_sector_ids)


def _child_sector_id(block: MeshBlock, root_id: str, child_schema_name: str) -> str:
    """由根分区与子实体的 Schema 名找出派生分区的 id."""
    matches: list[str] = []
    for sector in block.sectors.values():
        if (
            sector.source_cell_sector_id == root_id
            and sector.schema.name == child_schema_name
        ):
            matches.append(sector.id)
    if not matches:
        raise KeyError(
            f"no derived {child_schema_name!r} sector found for root "
            f"{root_id!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous derived {child_schema_name!r} sectors for root "
            f"{root_id!r}: {matches}"
        )
    return matches[0]


def _ensure_child_sector(
    block: MeshBlock,
    root_id: str,
    child_schema_name: str,
) -> str:
    """返回派生子分区的 id, 不存在时先重建拓扑."""
    try:
        child_id = _child_sector_id(block, root_id, child_schema_name)
    except KeyError:
        _rebuild_topology(block)
        child_id = _child_sector_id(block, root_id, child_schema_name)

    if (root_id, child_id) not in block.relations:
        _rebuild_topology(block)
        child_id = _child_sector_id(block, root_id, child_schema_name)

    return child_id


def _rebuild_topology(block: MeshBlock) -> None:
    roots = _root_names(block)
    block.sectors = {
        name: sector
        for name, sector in block.sectors.items()
        if name in roots or name == "node"
    }
    node = block.sectors["node"]
    node.indices = bm.arange(
        len(block.positions),
        dtype=bm.int64,
        device=bm.get_device(block.positions),
    ).reshape((-1, 1))
    block.relations.clear()
    block._cache_boundary_info = None
    TopologyBuilder.construct(block)


def _replace_topology(
    block: MeshBlock,
    root_name: str,
    root_indices: Tensor,
    positions: Tensor,
) -> None:
    """以受控的原子替换提交一次根分区加密."""
    block.replace_root_topology(positions, root_name, root_indices)


def _sector(block: MeshBlock, name: str) -> Tensor:
    return block.get_sector(name).indices


def _prolongation_from_edges(
    old_positions: Tensor,
    edge: Tensor,
    edge2new_node: Tensor,
) -> csr_matrix:
    nn = len(old_positions)
    ne = len(edge)
    shape = (nn + ne, nn)
    values = bm.ones(nn + 2 * ne, **bm.context(old_positions))
    values = bm.set_at(values, bm.arange(nn, nn + 2 * ne), 0.5)

    i0 = bm.arange(nn, dtype=edge.dtype, device=bm.get_device(edge))
    i = bm.concat((i0, edge2new_node, edge2new_node), axis=0)
    j = bm.concat((i0, edge[:, 0], edge[:, 1]), axis=0)
    return csr_matrix((values, (i, j)), shape)


def _refine_edge_once(block: MeshBlock, root_id: str) -> csr_matrix | None:
    edge = _sector(block, root_id)
    node = block.positions
    nn = len(node)
    nc = len(edge)

    new_node = (node[edge[:, 0]] + node[edge[:, 1]]) / 2
    new_index = _arange_like(nn, nn + nc, edge)

    left = bm.stack((edge[:, 0], new_index), axis=1)
    right = bm.stack((new_index, edge[:, 1]), axis=1)
    new_edge = bm.concat((left, right), axis=0)
    new_positions = bm.concat((node, new_node), axis=0)
    _replace_topology(block, root_id, new_edge, new_positions)


def uniform_refine_edge(
    block: MeshBlock,
    n: int = 1,
    returnim: bool = False,
    *,
    root_id: str = "edge",
    **kwargs,
) -> list | None:
    """一致加密区间网格, 原地修改网格块.

    每次加密把每个区间分成 2 个子区间 (取中点).

    Parameters
    ----------
    block : MeshBlock
        网格块.
    n : int, optional
        加密次数, 默认 1.
    returnim : bool, optional
        是否返回各次加密的延拓矩阵. 默认 False.
    root_id : str, optional
        被加密的根分区 id, 默认 ``edge``.

    Returns
    -------
    list of scipy.sparse.csr_matrix or None
        ``returnim`` 为 True 时返回延拓矩阵列表 (从最细一层到最粗一层),
        否则返回 None.
    """
    im = [] if returnim else None
    for _ in range(n):
        if returnim:
            edge = _sector(block, root_id)
            node = block.positions
            nc = len(edge)
            nn = len(node)
            new_index = _arange_like(nn, nn + nc, edge)
            im.append(_prolongation_from_edges(node, edge, new_index))
        _refine_edge_once(block, root_id)
    if returnim:
        im.reverse()
        return im
    return None


def _refine_triangle_once(block: MeshBlock, root_id: str) -> csr_matrix | None:
    edge_id = _ensure_child_sector(block, root_id, "edge")
    tri = _sector(block, root_id)
    edge = _sector(block, edge_id)
    cell2edge = block.relations[(root_id, edge_id)].tgt_indices
    node = block.positions
    nn = len(node)
    ne = len(edge)

    edge2new = _arange_like(nn, nn + ne, tri)
    new_node = (node[edge[:, 0]] + node[edge[:, 1]]) / 2

    p = bm.concat((tri, edge2new[cell2edge]), axis=1)
    new_tri = bm.concat(
        (p[:, [0, 5, 4]], p[:, [5, 1, 3]], p[:, [4, 3, 2]], p[:, [3, 4, 5]]),
        axis=0,
    )
    new_positions = bm.concat((node, new_node), axis=0)
    _replace_topology(block, root_id, new_tri, new_positions)


def uniform_refine_triangle(
    block: MeshBlock,
    n: int = 1,
    surface=None,
    interface=None,
    returnim: bool = False,
    *,
    root_id: str = "tri",
    **kwargs,
) -> list | None:
    """一致加密三角形网格, 原地修改网格块.

    每次加密把每个三角形分成 4 个子三角形 (连接各边中点).

    Parameters
    ----------
    block : MeshBlock
        网格块.
    n : int, optional
        加密次数, 默认 1.
    surface, interface : optional
        未使用.
    returnim : bool, optional
        是否返回各次加密的延拓矩阵. 默认 False.
    root_id : str, optional
        被加密的根分区 id, 默认 ``tri``.

    Returns
    -------
    list of scipy.sparse.csr_matrix or None
        ``returnim`` 为 True 时返回延拓矩阵列表 (从最细一层到最粗一层),
        否则返回 None.
    """
    im = [] if returnim else None
    for _ in range(n):
        edge_id = _ensure_child_sector(block, root_id, "edge")
        if returnim:
            edge = _sector(block, edge_id)
            node = block.positions
            edge2new = _arange_like(len(node), len(node) + len(edge), _sector(block, root_id))
            im.append(_prolongation_from_edges(node, edge, edge2new))
        _refine_triangle_once(block, root_id)
    if returnim:
        im.reverse()
        return im
    return None


def _refine_quadrilateral_once(block: MeshBlock, root_id: str) -> None:
    edge_id = _ensure_child_sector(block, root_id, "edge")
    quad = _sector(block, root_id)
    edge = _sector(block, edge_id)
    cell2edge = block.relations[(root_id, edge_id)].tgt_indices
    node = block.positions
    nn = len(node)
    ne = len(edge)
    nc = len(quad)

    edge_center = _barycenter(block, edge)
    cell_center = _barycenter(block, quad)
    edge2center = _arange_like(nn, nn + ne, quad)
    cell_center_idx = _arange_like(nn + ne, nn + ne + nc, quad)[:, None]

    cp = [quad[:, i:i + 1] for i in range(4)]
    ep = [edge2center[cell2edge[:, i]][:, None] for i in range(4)]
    children = (
        bm.concat([cp[0], ep[0], cell_center_idx, ep[3]], axis=1),
        bm.concat([ep[0], cp[1], ep[1], cell_center_idx], axis=1),
        bm.concat([cell_center_idx, ep[1], cp[2], ep[2]], axis=1),
        bm.concat([ep[3], cell_center_idx, ep[2], cp[3]], axis=1),
    )
    new_quad = bm.reshape(bm.stack(children, axis=1), (-1, 4))

    new_positions = bm.concat([node, edge_center, cell_center], axis=0)
    _replace_topology(block, root_id, new_quad, new_positions)


def uniform_refine_quadrilateral(
    block: MeshBlock,
    n: int = 1,
    surface=None,
    interface=None,
    returnim: bool = False,
    *,
    root_id: str = "quad",
    **kwargs,
) -> list | None:
    """一致加密四边形网格, 原地修改网格块.

    每次加密把每个四边形分成 4 个子四边形 (取各边中点与单元中心).

    Parameters
    ----------
    block : MeshBlock
        网格块.
    n : int, optional
        加密次数, 默认 1.
    surface, interface : optional
        未使用.
    returnim : bool, optional
        是否返回各次加密的延拓矩阵. 默认 False.
    root_id : str, optional
        被加密的根分区 id, 默认 ``quad``.

    Returns
    -------
    list of scipy.sparse.csr_matrix or None
        ``returnim`` 为 True 时返回延拓矩阵列表 (从最细一层到最粗一层),
        否则返回 None.
    """
    im = [] if returnim else None
    for _ in range(n):
        edge_id = _ensure_child_sector(block, root_id, "edge")
        if returnim:
            quad = _sector(block, root_id)
            edge = _sector(block, edge_id)
            node = block.positions
            nn = len(node)
            ne = len(edge)
            nc = len(quad)
            shape = (nn + ne + nc, nn)
            values = bm.ones(nn + 2 * ne + 4 * nc, **bm.context(node))
            values = bm.set_at(values, bm.arange(nn, nn + 2 * ne), 0.5)
            values = bm.set_at(values, bm.arange(nn + 2 * ne, nn + 2 * ne + 4 * nc), 0.25)
            i0 = bm.arange(nn, dtype=quad.dtype, device=bm.get_device(quad))
            i1 = _arange_like(nn, nn + ne, quad)
            i2 = _arange_like(nn + ne, nn + ne + nc, quad)
            i = bm.concat((i0, i1, i1, i2, i2, i2, i2), axis=0)
            j = bm.concat((i0, edge[:, 0], edge[:, 1], quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]), axis=0)
            im.append(csr_matrix((values, (i, j)), shape))
        _refine_quadrilateral_once(block, root_id)
    if returnim:
        im.reverse()
        return im
    return None


def _refine_tetrahedron_once(block: MeshBlock, root_id: str) -> None:
    edge_id = _ensure_child_sector(block, root_id, "edge")
    tet = _sector(block, root_id)
    edge = _sector(block, edge_id)
    cell2edge = block.relations[(root_id, edge_id)].tgt_indices
    node = block.positions
    nn = len(node)
    ne = len(edge)
    nc = len(tet)

    edge2new = _arange_like(nn, nn + ne, tet)
    new_node = (node[edge[:, 0]] + node[edge[:, 1]]) / 2
    new_positions = bm.concat((node, new_node), axis=0)

    p = edge2new[cell2edge]
    new_tet = _empty_like((8 * nc, 4), tet)
    new_tet = bm.set_at(new_tet, (slice(4 * nc), 3), tet.T.flatten())
    new_tet = bm.set_at(new_tet, (slice(nc), slice(3)), p[:, [0, 2, 1]])
    new_tet = bm.set_at(new_tet, (slice(nc, 2 * nc), slice(3)), p[:, [0, 3, 4]])
    new_tet = bm.set_at(new_tet, (slice(2 * nc, 3 * nc), slice(3)), p[:, [1, 5, 3]])
    new_tet = bm.set_at(new_tet, (slice(3 * nc, 4 * nc), slice(3)), p[:, [2, 4, 5]])

    l = bm.zeros((nc, 3), dtype=block.positions.dtype, device=bm.get_device(block.positions))
    node = new_positions
    l = bm.set_at(l, (slice(None), 0), bm.sum((node[p[:, 0]] - node[p[:, 5]]) ** 2, axis=1))
    l = bm.set_at(l, (slice(None), 1), bm.sum((node[p[:, 1]] - node[p[:, 4]]) ** 2, axis=1))
    l = bm.set_at(l, (slice(None), 2), bm.sum((node[p[:, 2]] - node[p[:, 3]]) ** 2, axis=1))

    idx = bm.argmin(l, axis=1)
    table = bm.array(
        [(1, 3, 4, 2, 5, 0), (0, 2, 5, 3, 4, 1), (0, 4, 5, 1, 3, 2)],
        dtype=tet.dtype,
        device=bm.get_device(tet),
    )
    t = table[idx]
    rows = bm.arange(nc, dtype=tet.dtype, device=bm.get_device(tet))
    new_tet = bm.set_at(new_tet, (slice(4 * nc, 5 * nc), 0), p[rows, t[:, 0]])
    new_tet = bm.set_at(new_tet, (slice(4 * nc, 5 * nc), 1), p[rows, t[:, 1]])
    new_tet = bm.set_at(new_tet, (slice(4 * nc, 5 * nc), 2), p[rows, t[:, 4]])
    new_tet = bm.set_at(new_tet, (slice(4 * nc, 5 * nc), 3), p[rows, t[:, 5]])
    new_tet = bm.set_at(new_tet, (slice(5 * nc, 6 * nc), 0), p[rows, t[:, 1]])
    new_tet = bm.set_at(new_tet, (slice(5 * nc, 6 * nc), 1), p[rows, t[:, 2]])
    new_tet = bm.set_at(new_tet, (slice(5 * nc, 6 * nc), 2), p[rows, t[:, 4]])
    new_tet = bm.set_at(new_tet, (slice(5 * nc, 6 * nc), 3), p[rows, t[:, 5]])
    new_tet = bm.set_at(new_tet, (slice(6 * nc, 7 * nc), 0), p[rows, t[:, 2]])
    new_tet = bm.set_at(new_tet, (slice(6 * nc, 7 * nc), 1), p[rows, t[:, 3]])
    new_tet = bm.set_at(new_tet, (slice(6 * nc, 7 * nc), 2), p[rows, t[:, 4]])
    new_tet = bm.set_at(new_tet, (slice(6 * nc, 7 * nc), 3), p[rows, t[:, 5]])
    new_tet = bm.set_at(new_tet, (slice(7 * nc, 8 * nc), 0), p[rows, t[:, 3]])
    new_tet = bm.set_at(new_tet, (slice(7 * nc, 8 * nc), 1), p[rows, t[:, 0]])
    new_tet = bm.set_at(new_tet, (slice(7 * nc, 8 * nc), 2), p[rows, t[:, 4]])
    new_tet = bm.set_at(new_tet, (slice(7 * nc, 8 * nc), 3), p[rows, t[:, 5]])

    _replace_topology(block, root_id, new_tet, new_positions)


def uniform_refine_tetrahedron(
    block: MeshBlock,
    n: int = 1,
    returnim: bool = False,
    *,
    root_id: str = "tet",
    **kwargs,
) -> list | None:
    """一致加密四面体网格, 原地修改网格块.

    每次加密把每个四面体分成 8 个子四面体 (取各边中点).

    Parameters
    ----------
    block : MeshBlock
        网格块.
    n : int, optional
        加密次数, 默认 1.
    returnim : bool, optional
        是否返回各次加密的延拓矩阵. 默认 False.
    root_id : str, optional
        被加密的根分区 id, 默认 ``tet``.

    Returns
    -------
    list of scipy.sparse.csr_matrix or None
        ``returnim`` 为 True 时返回延拓矩阵列表 (从最细一层到最粗一层),
        否则返回 None.
    """
    im = [] if returnim else None
    for _ in range(n):
        edge_id = _ensure_child_sector(block, root_id, "edge")
        if returnim:
            edge = _sector(block, edge_id)
            node = block.positions
            edge2new = _arange_like(len(node), len(node) + len(edge), _sector(block, root_id))
            im.append(_prolongation_from_edges(node, edge, edge2new))
        _refine_tetrahedron_once(block, root_id)
    if returnim:
        im.reverse()
        return im
    return None


def _refine_prism_once(block: MeshBlock, root_id: str) -> None:
    edge_id = _ensure_child_sector(block, root_id, "edge")
    quad_id = _ensure_child_sector(block, root_id, "quad")
    prism = _sector(block, root_id)
    edge = _sector(block, edge_id)
    quad = _sector(block, quad_id)
    c2e = block.relations[(root_id, edge_id)].tgt_indices
    c2q = block.relations[(root_id, quad_id)].tgt_indices
    node = block.positions
    nn = len(node)
    ne = len(edge)
    nq = len(quad)
    nc = len(prism)

    edge_center = _barycenter(block, edge)
    quad_center = _barycenter(block, quad)
    e = c2e + nn
    q = c2q + nn + ne

    new_prism = _empty_like((8 * nc, 6), prism)
    new_prism = bm.set_at(new_prism, (slice(0, None, 8), slice(None)), bm.stack([prism[:, 0], e[:, 0], e[:, 2], e[:, 3], q[:, 0], q[:, 2]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(1, None, 8), slice(None)), bm.stack([e[:, 0], e[:, 1], e[:, 2], q[:, 0], q[:, 1], q[:, 2]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(2, None, 8), slice(None)), bm.stack([prism[:, 1], e[:, 1], e[:, 0], e[:, 4], q[:, 1], q[:, 0]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(3, None, 8), slice(None)), bm.stack([prism[:, 2], e[:, 2], e[:, 1], e[:, 5], q[:, 2], q[:, 1]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(4, None, 8), slice(None)), bm.stack([e[:, 3], q[:, 0], q[:, 2], prism[:, 3], e[:, 6], e[:, 8]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(5, None, 8), slice(None)), bm.stack([q[:, 0], q[:, 1], q[:, 2], e[:, 6], e[:, 7], e[:, 8]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(6, None, 8), slice(None)), bm.stack([e[:, 4], q[:, 1], q[:, 0], prism[:, 4], e[:, 7], e[:, 6]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(7, None, 8), slice(None)), bm.stack([e[:, 5], q[:, 2], q[:, 1], prism[:, 5], e[:, 8], e[:, 7]], axis=1))

    new_positions = bm.concat([node, edge_center, quad_center], axis=0)
    _replace_topology(block, root_id, new_prism, new_positions)


def uniform_refine_prism(
    block: MeshBlock,
    n: int = 1,
    returnim: bool = False,
    *,
    root_id: str = "prism",
    **kwargs,
) -> list | None:
    """一致加密三棱柱网格, 原地修改网格块.

    每次加密把每个三棱柱分成 8 个子三棱柱 (取各边中点与四边形面中心).

    Parameters
    ----------
    block : MeshBlock
        网格块.
    n : int, optional
        加密次数, 默认 1.
    returnim : bool, optional
        是否返回各次加密的延拓矩阵. 默认 False.
    root_id : str, optional
        被加密的根分区 id, 默认 ``prism``.

    Returns
    -------
    list of scipy.sparse.csr_matrix or None
        ``returnim`` 为 True 时返回延拓矩阵列表 (从最细一层到最粗一层),
        否则返回 None.

    Notes
    -----
    三棱柱尚未生成延拓矩阵, ``returnim`` 为 True 时返回空列表.
    """
    im = [] if returnim else None
    for _ in range(n):
        _refine_prism_once(block, root_id)
    if returnim:
        return im
    return None


def _refine_hexahedron_once(block: MeshBlock, root_id: str) -> None:
    edge_id = _ensure_child_sector(block, root_id, "edge")
    quad_id = _ensure_child_sector(block, root_id, "quad")
    hex_ = _sector(block, root_id)
    edge = _sector(block, edge_id)
    quad = _sector(block, quad_id)
    c2e = block.relations[(root_id, edge_id)].tgt_indices
    c2f = block.relations[(root_id, quad_id)].tgt_indices
    node = block.positions
    nn = len(node)
    ne = len(edge)
    nf = len(quad)
    nc = len(hex_)

    edge_center = _barycenter(block, edge)
    face_center = _barycenter(block, quad)
    cell_center = _barycenter(block, hex_)
    c2n = hex_
    c2e = c2e + nn
    c2f = c2f + nn + ne
    c2c = _arange_like(nn + ne + nf, nn + ne + nf + nc, hex_)

    cell = _empty_like((8 * nc, 8), hex_)
    assignments = [
        (0, [c2n[:, 0], c2e[:, 0], c2e[:, 3], c2f[:, 0], c2e[:, 4], c2f[:, 4], c2f[:, 2], c2c]),
        (1, [c2n[:, 1], c2e[:, 1], c2e[:, 0], c2f[:, 0], c2e[:, 5], c2f[:, 3], c2f[:, 4], c2c]),
        (2, [c2n[:, 3], c2e[:, 2], c2e[:, 1], c2f[:, 0], c2e[:, 7], c2f[:, 5], c2f[:, 3], c2c]),
        (3, [c2n[:, 2], c2e[:, 3], c2e[:, 2], c2f[:, 0], c2e[:, 6], c2f[:, 2], c2f[:, 5], c2c]),
        (4, [c2n[:, 4], c2e[:, 11], c2e[:, 8], c2f[:, 1], c2e[:, 4], c2f[:, 2], c2f[:, 4], c2c]),
        (5, [c2n[:, 5], c2e[:, 8], c2e[:, 9], c2f[:, 1], c2e[:, 5], c2f[:, 4], c2f[:, 3], c2c]),
        (6, [c2n[:, 7], c2e[:, 9], c2e[:, 10], c2f[:, 1], c2e[:, 7], c2f[:, 3], c2f[:, 5], c2c]),
        (7, [c2n[:, 6], c2e[:, 10], c2e[:, 11], c2f[:, 1], c2e[:, 6], c2f[:, 5], c2f[:, 2], c2c]),
    ]
    for offset, cols in assignments:
        cell = bm.set_at(cell, (slice(offset, None, 8), slice(None)), bm.stack(cols, axis=1))

    new_positions = bm.concat(
        [node, edge_center, face_center, cell_center],
        axis=0,
    )
    _replace_topology(block, root_id, cell, new_positions)


def uniform_refine_hexahedron(
    block: MeshBlock,
    n: int = 1,
    surface=None,
    interface=None,
    returnim: bool = False,
    *,
    root_id: str = "hex",
    **kwargs,
) -> list | None:
    """一致加密六面体网格, 原地修改网格块.

    每次加密把每个六面体分成 8 个子六面体 (取各边中点、各面中心与单元中心).

    Parameters
    ----------
    block : MeshBlock
        网格块.
    n : int, optional
        加密次数, 默认 1.
    surface, interface : optional
        未使用.
    returnim : bool, optional
        是否返回各次加密的延拓矩阵. 默认 False.
    root_id : str, optional
        被加密的根分区 id, 默认 ``hex``.

    Returns
    -------
    list of scipy.sparse.csr_matrix or None
        ``returnim`` 为 True 时返回延拓矩阵列表 (从最细一层到最粗一层),
        否则返回 None.
    """
    im = [] if returnim else None
    for _ in range(n):
        edge_id = _ensure_child_sector(block, root_id, "edge")
        quad_id = _ensure_child_sector(block, root_id, "quad")
        if returnim:
            hex_ = _sector(block, root_id)
            edge = _sector(block, edge_id)
            quad = _sector(block, quad_id)
            node = block.positions
            nn = len(node)
            ne = len(edge)
            nf = len(quad)
            nc = len(hex_)
            shape = (nn + ne + nf + nc, nn)
            values = bm.ones(nn + 2 * ne + 4 * nf + 8 * nc, **bm.context(node))
            values = bm.set_at(values, bm.arange(nn, nn + 2 * ne), 0.5)
            values = bm.set_at(values, bm.arange(nn + 2 * ne, nn + 2 * ne + 4 * nf), 0.25)
            values = bm.set_at(values, bm.arange(nn + 2 * ne + 4 * nf, nn + 2 * ne + 4 * nf + 8 * nc), 0.125)
            i0 = bm.arange(nn, dtype=hex_.dtype, device=bm.get_device(hex_))
            i1 = _arange_like(nn, nn + ne, hex_)
            i2 = _arange_like(nn + ne, nn + ne + nf, hex_)
            i3 = _arange_like(nn + ne + nf, nn + ne + nf + nc, hex_)
            i = bm.concat((i0, i1, i1, i2, i2, i2, i2, i3, i3, i3, i3, i3, i3, i3, i3), axis=0)
            j = bm.concat((i0, edge[:, 0], edge[:, 1], quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3], hex_[:, 0], hex_[:, 1], hex_[:, 2], hex_[:, 3], hex_[:, 4], hex_[:, 5], hex_[:, 6], hex_[:, 7]), axis=0)
            im.append(csr_matrix((values, (i, j)), shape))
        _refine_hexahedron_once(block, root_id)
    if returnim:
        im.reverse()
        return im
    return None


_DISPATCH: dict[str, RefineFunc] = {
    "edge": uniform_refine_edge,
    "tri": uniform_refine_triangle,
    "quad": uniform_refine_quadrilateral,
    "tet": uniform_refine_tetrahedron,
    "prism": uniform_refine_prism,
    "hex": uniform_refine_hexahedron,
}


def uniform_refine(
    block: MeshBlock,
    n: int = 1,
    schema_name: str | None = None,
    *,
    root_id: str | None = None,
    **kwargs,
) -> list | None:
    """一致加密网格块, 原地修改; 按根分区的具体 Schema 分派到对应的加密函数.

    Parameters
    ----------
    block : MeshBlock
        网格块.
    n : int, optional
        加密次数, 默认 1.
    schema_name : str, optional
        ``root_id`` 的过渡别名, 为经典单根调用方保留; 新代码应使用 ``root_id``.
    root_id : str, optional
        被加密的根分区 id, 默认取第一个根单元分区.
    **kwargs
        传给具体的加密函数, 如 ``returnim``.

    Returns
    -------
    list or None
        具体加密函数的返回值.

    Raises
    ------
    ValueError
        网格块没有根实体, 或 ``root_id`` 不是根单元分区.
    NotImplementedError
        该 Schema 没有一致加密实现.
    """
    if root_id is None:
        root_id = schema_name

    if root_id is None:
        if not block.root_cell_sector_ids:
            raise ValueError("cannot refine a MeshBlock without root entities")
        root_id = block.root_cell_sector_ids[0]

    if root_id not in block.root_cell_sector_ids:
        raise ValueError(
            f"root sector {root_id!r} is not a root cell sector"
        )

    schema_name = block.get_sector(root_id).schema.name
    try:
        refine = _DISPATCH[schema_name]
    except KeyError as exc:
        raise NotImplementedError(f"uniform refinement is not implemented for {schema_name!r}") from exc
    return refine(block, n=n, root_id=root_id, **kwargs)
