# 移植自 brighthe/fealpy ``fealpy/mesh/generation/box.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""一、二、三维长方体区域的结构网格生成器.

各生成器先生成节点与按张量积顺序编号的内部单元 (``initialize``), 再按所需单元类型
重排或细分, 构造只含一个根分区的网格块.
"""

from dataclasses import dataclass, field
from typing import Any, NamedTuple

from ...backend import Tensor, bm
from ..schema import (
    HexahedronSchema,
    PrismSchema,
    PyramidSchema,
    QuadrilateralSchema,
    EdgeSchema,
    TetrahedronSchema,
    TriangleSchema,
)
from ..storage import EntitySector, MeshBlock
from ..view.mesh_view import MeshView


class BoxCache(NamedTuple):
    """生成器缓存的节点与内部单元."""
    node: Tensor
    cell: Tensor


@dataclass(slots=True)
class Box1d:
    """区间上的结构网格生成器.

    Parameters
    ----------
    box : list of float, optional
        区间 ``[x0, x1]``, 默认 ``[0, 1]``.
    nx : int, optional
        剖分数, 默认 10.
    device : optional
        设备.
    """
    box: list[float] = field(default_factory=list)
    nx: int = 10
    device: Any | None = None
    _cache: BoxCache | None = field(default=None, init=False)

    def __post_init__(self):
        if not self.box:
            self.box = [0, 1]

    def initialize(self):
        """返回节点与区间单元, 结果会缓存."""
        if self._cache is not None:
            return self._cache.node, self._cache.cell

        box = self.box
        nx = self.nx

        x = bm.linspace(box[0], box[1], nx + 1, dtype=bm.float64, device=self.device)
        node = bm.reshape(x, (-1, 1))
        idx = bm.arange(nx + 1, dtype=bm.int32, device=self.device)
        cell = bm.stack((idx[:-1], idx[1:]), axis=1)

        self._cache = BoxCache(node=node, cell=cell)
        return node, cell

    def clear(self) -> None:
        """清除缓存."""
        self._cache = None

    def nodalize(self) -> MeshView:
        """生成只含节点坐标、没有单元的网格."""
        node, _ = self.initialize()

        block = MeshBlock(positions=node)
        return MeshView(block).construct()

    def segmentize(self) -> MeshView:
        """生成区间剖分网格."""
        node, cell = self.initialize()

        block = MeshBlock(positions=node)
        block.add_sector(
            EntitySector(
                id="edge",
                schema=EdgeSchema(),
                indices=cell,
            ),
            root=True,
        )
        return MeshView(block).construct()


@dataclass(slots=True)
class Box2d:
    """矩形上的结构网格生成器.

    Parameters
    ----------
    box : list of float, optional
        区域 ``[x0, x1, y0, y1]``, 默认单位正方形.
    nx, ny : int, optional
        各方向的剖分数, 默认 10.
    device : optional
        设备.
    """
    box: list[float] = field(default_factory=list)
    nx: int = 10
    ny: int = 10
    device: Any | None = None
    _cache: BoxCache | None = field(default=None, init=False)

    def __post_init__(self):
        if not self.box:
            self.box = [0, 1, 0, 1]

    def initialize(self):
        """返回节点与按张量积顺序编号的内部单元 (x 方向变化最快), 结果会缓存."""
        if self._cache is not None:
            return self._cache.node, self._cache.cell

        box = self.box
        nx = self.nx
        ny = self.ny

        NN = (nx + 1) * (ny + 1)
        x = bm.linspace(box[0], box[1], nx + 1, dtype=bm.float64, device=self.device)
        y = bm.linspace(box[2], box[3], ny + 1, dtype=bm.float64, device=self.device)
        X, Y = bm.meshgrid(x, y, indexing="ij")

        node = bm.concat(
            (
                bm.reshape(X, (-1, 1)),
                bm.reshape(Y, (-1, 1)),
            ),
            axis=1,
        )
        idx = bm.reshape(bm.arange(NN, dtype=bm.int32, device=self.device), (nx + 1, ny + 1))

        cell0 = idx[:-1, :-1] # type: ignore
        cell1 = idx[1:, :-1] # type: ignore
        cell2 = idx[:-1, 1:] # type: ignore
        cell3 = idx[1:, 1:] # type: ignore
        cell = bm.concat(
            (
                bm.reshape(cell0, (-1, 1)),
                bm.reshape(cell1, (-1, 1)),
                bm.reshape(cell2, (-1, 1)),
                bm.reshape(cell3, (-1, 1)),
            ),
            axis=1,
        )
        self._cache = BoxCache(node=node, cell=cell)
        return node, cell

    def clear(self) -> None:
        """清除缓存."""
        self._cache = None

    def nodalize(self) -> MeshView:
        """生成只含节点坐标、没有单元的网格."""
        node, _ = self.initialize()

        block = MeshBlock(positions=node)
        return MeshView(block).construct()

    def triangulate(self) -> MeshView:
        """生成三角形网格: 每个矩形沿 (0, 0)--(1, 1) 对角线分成两个三角形."""
        node, cell = self.initialize()
        local_cell = bm.asarray([
            [0, 1, 3],
            [0, 3, 2],
        ], dtype=bm.int32)
        cell = bm.reshape(cell[:, local_cell], (-1, 3)) # type: ignore

        block = MeshBlock(positions=node)
        block.add_sector(
            EntitySector(
                id="tri",
                schema=TriangleSchema(),
                indices=cell,
            ),
            root=True,
        )
        return MeshView(block).construct()

    def quadrangulate(self) -> MeshView:
        """生成四边形网格, 单元顶点按逆时针循环顺序排列."""
        node, cell = self.initialize()
        tensor_to_cyclic = bm.asarray(
            [0, 1, 3, 2],
            dtype=bm.int32,
            device=bm.get_device(cell),
        )
        cell = bm.reshape(cell[:, tensor_to_cyclic], (-1, 4))

        block = MeshBlock(positions=node)
        block.add_sector(
            EntitySector(
                id="quad",
                schema=QuadrilateralSchema(),
                indices=cell,
            ),
            root=True,
        )
        return MeshView(block).construct()


@dataclass(slots=True)
class Box3d:
    """长方体上的结构网格生成器.

    Parameters
    ----------
    box : list of float, optional
        区域 ``[x0, x1, y0, y1, z0, z1]``, 默认单位立方体.
    nx, ny, nz : int, optional
        各方向的剖分数, 默认 10.
    device : optional
        设备.
    """
    box: list[float] = field(default_factory=list)
    nx: int = 10
    ny: int = 10
    nz: int = 10
    device: Any | None = None
    _cache: BoxCache | None = field(default=None, init=False)

    def __post_init__(self):
        if not self.box:
            self.box = [0, 1, 0, 1, 0, 1]

    def initialize(self):
        """返回节点与按张量积顺序编号的内部单元 (x 方向变化最快), 结果会缓存."""
        if self._cache is not None:
            return self._cache.node, self._cache.cell

        box = self.box
        nx = self.nx
        ny = self.ny
        nz = self.nz

        NN = (nx + 1) * (ny + 1) * (nz + 1)
        x = bm.linspace(box[0], box[1], nx + 1, dtype=bm.float64, device=self.device)
        y = bm.linspace(box[2], box[3], ny + 1, dtype=bm.float64, device=self.device)
        z = bm.linspace(box[4], box[5], nz + 1, dtype=bm.float64, device=self.device)
        X, Y, Z = bm.meshgrid(x, y, z, indexing="ij")

        node = bm.concat(
            (
                bm.reshape(X, (-1, 1)),
                bm.reshape(Y, (-1, 1)),
                bm.reshape(Z, (-1, 1)),
            ),
            axis=1,
        )
        idx = bm.reshape(bm.arange(NN, dtype=bm.int32, device=self.device), (nx + 1, ny + 1, nz + 1))

        nyz = (ny + 1) * (nz + 1)
        cell0 = idx[:-1, :-1, :-1] # type: ignore
        cell1 = cell0 + nyz
        cell2 = cell1 + nz + 1
        cell3 = cell0 + nz + 1
        cell4 = cell0 + 1
        cell5 = cell4 + nyz
        cell6 = cell5 + nz + 1
        cell7 = cell4 + nz + 1
        cell = bm.concat(
            (
                bm.reshape(cell0, (-1, 1)),
                bm.reshape(cell1, (-1, 1)),
                bm.reshape(cell3, (-1, 1)),
                bm.reshape(cell2, (-1, 1)),
                bm.reshape(cell4, (-1, 1)),
                bm.reshape(cell5, (-1, 1)),
                bm.reshape(cell7, (-1, 1)),
                bm.reshape(cell6, (-1, 1)),
            ),
            axis=1,
        )
        self._cache = BoxCache(node=node, cell=cell)
        return node, cell

    def clear(self) -> None:
        """清除缓存."""
        self._cache = None

    def nodalize(self) -> MeshView:
        """生成只含节点坐标、没有单元的网格."""
        node, _ = self.initialize()

        block = MeshBlock(positions=node)
        return MeshView(block).construct()

    def tetrahedralize(self) -> MeshView:
        """生成四面体网格: 每个小长方体分成 6 个四面体."""
        node, cell = self.initialize()
        local_cell = bm.asarray([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [2, 1, 3, 7],
            [1, 5, 7, 6],
            [2, 7, 6, 1],
        ], dtype=bm.int32)
        cell = bm.reshape(cell[:, local_cell], (-1, 4)) # type: ignore

        block = MeshBlock(positions=node)
        block.add_sector(
            EntitySector(
                id="tet",
                schema=TetrahedronSchema(),
                indices=cell,
            ),
            root=True,
        )
        return MeshView(block).construct()

    def prismatize(self) -> MeshView:
        """生成三棱柱网格: 每个小长方体分成 2 个三棱柱."""
        node, cell = self.initialize()
        local_cell = bm.asarray([
            [0, 1, 2, 4, 5, 6],
            [2, 1, 3, 6, 5, 7],
        ], dtype=bm.int32)
        cell = bm.reshape(cell[:, local_cell], (-1, 6)) # type: ignore

        block = MeshBlock(positions=node)
        block.add_sector(
            EntitySector(
                id="prism",
                schema=PrismSchema(),
                indices=cell,
            ),
            root=True,
        )
        return MeshView(block).construct()

    def pyramidalize(self) -> MeshView:
        """生成四棱锥网格: 每个小长方体分成 3 个四棱锥."""
        node, cell = self.initialize()
        local_cell = bm.asarray([
            [0, 1, 2, 3, 6],
            [0, 1, 4, 5, 6],
            [0, 2, 4, 7, 6],
        ], dtype=bm.int32)
        cell = bm.reshape(cell[:, local_cell], (-1, 5)) # type: ignore

        block = MeshBlock(positions=node)
        block.add_sector(
            EntitySector(
                id="pyramid",
                schema=PyramidSchema(),
                indices=cell,
            ),
            root=True,
        )
        return MeshView(block).construct()

    def hexahedralize(self) -> MeshView:
        """生成六面体网格, 单元顶点按经典六面体的循环顺序排列."""
        node, cell = self.initialize()
        tensor_to_cyclic = bm.asarray(
            [0, 1, 3, 2, 4, 5, 7, 6],
            dtype=bm.int32,
            device=bm.get_device(cell),
        )
        cell = bm.reshape(cell[:, tensor_to_cyclic], (-1, 8))

        block = MeshBlock(positions=node)
        block.add_sector(
            EntitySector(
                id="hex",
                schema=HexahedronSchema(),
                indices=cell,
            ),
            root=True,
        )
        return MeshView(block).construct()
