"""长方形/长方体区域上的结构化网格生成器, 附带平移类约定.

本模块由 soptx 自行实现, 只经 FEALPy 的公开构造器 ``TriangleMesh(node, cell)`` 等
生成网格. 输出的节点坐标与单元顶点编号与 FEALPy ``*.from_box`` 逐位一致 (由单元测试
保证), 因此可以直接替换 ``from_box``, 已有结果仍可逐位比较.

与 ``from_box`` 不同的是, 单元编号约定在这里是 soptx 的公开契约, 而不是生成器的内部
实现: 区域先按轴向划成 ``nx x ny (x nz)`` 个格子, 格子按 x 最慢、z (二维时 y) 最快的
字典序排列; 每个格子再按固定的剖分模板切成 N_k 个单元, 连续编号. 于是

    单元 e 位于第 e // N_k 个格子, 采用第 e % N_k 种剖分.

同一种剖分在不同格子里的单元只差平移, 局部顶点顺序也相同, 构成一个平移类; 类归属
k(e) = e % N_k 直接由编号给出, 不需要比对几何. 各网格的 N_k 为: 四边形、六面体 1,
三角形 2, 四面体 6.

Notes
-----
节点坐标由逐轴 ``linspace`` 生成, 各格子的边长只在舍入意义下相等, 因此同类单元的相对
顶点坐标只差舍入, 由它们积出的单元矩阵也只差舍入.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Any, NamedTuple, Optional, Sequence, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh import HexahedronMesh, QuadrangleMesh, TetrahedronMesh, TriangleMesh
from fealpy.typing import TensorLike


# 各网格的剖分模板: 每种剖分列出其顶点相对格子最小角点的偏移 (以格子数计),
# 顺序即单元的局部顶点顺序
_TEMPLATES = {
    'tri': (TriangleMesh, 2, (
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (1, 1), (0, 1)),
    )),
    'quad': (QuadrangleMesh, 2, (
        ((0, 0), (1, 0), (1, 1), (0, 1)),
    )),
    'tet': (TetrahedronMesh, 3, (
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 1)),
        ((0, 0, 0), (1, 0, 1), (1, 0, 0), (0, 1, 1)),
        ((0, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)),
        ((0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
        ((1, 0, 0), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
        ((0, 1, 0), (1, 1, 1), (0, 1, 1), (1, 0, 0)),
    )),
    'hex': (HexahedronMesh, 3, (
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
    )),
}

MESH_TYPES = tuple(_TEMPLATES)


@dataclass(frozen=True)
class BoxTranslationClasses:
    """结构化网格的平移类划分, 由单元编号约定直接给出.

    Parameters
    ----------
    num_classes : 平移类数 N_k, 即每个格子剖出的单元数.
    num_cells : 单元总数 NC, 为 N_k 的整数倍.
    device : 生成索引数组所在的设备; 可为 None.
    """

    num_classes: int
    num_cells: int
    device: Any = None

    def representatives(self) -> TensorLike:
        """各平移类的代表单元, 取第一个格子里的 N_k 个单元.

        Returns
        -------
        reps : (N_k, ) 的单元编号, 即 ``arange(N_k)``.
        """
        return bm.arange(self.num_classes, dtype=bm.int32, device=self.device)

    def class_index(self) -> TensorLike:
        """每个单元的类归属 k(e).

        Returns
        -------
        cls : (NC, ) 的类编号, 即 ``arange(NC) % N_k``.
        """
        cell = bm.arange(self.num_cells, dtype=bm.int32, device=self.device)

        return cell % self.num_classes


class BoxMesh(NamedTuple):
    """``create_box_mesh`` 的返回值: 网格及其平移类划分"""

    mesh: Any
    classes: BoxTranslationClasses


def create_box_mesh(
    mesh_type: str,
    box: Sequence[float],
    nx: int,
    ny: int,
    nz: Optional[int] = None,
    *,
    device=None,
) -> BoxMesh:
    """在长方形/长方体区域上生成结构化网格, 并给出其平移类.

    Parameters
    ----------
    mesh_type : 网格类型, 取 ``'tri'``、``'quad'``、``'tet'``、``'hex'`` 之一.
    box : 区域范围, 二维为 ``[x0, x1, y0, y1]``, 三维再加 ``[z0, z1]``.
    nx, ny, nz : 各轴向的格子数, 均为正整数; ``nz`` 只在三维时给出.
    device : 网格数组所在的设备; 可为 None.

    Returns
    -------
    BoxMesh
        ``mesh`` 与对应 FEALPy ``*.from_box(box, nx, ny[, nz])`` 的节点、单元逐位一致;
        ``classes`` 为按单元编号约定给出的平移类.

    Raises
    ------
    ValueError
        网格类型未知, 或 ``box``、``nz`` 与网格维数不符.
    TypeError
        格子数不是正整数.
    """
    if mesh_type not in _TEMPLATES:
        raise ValueError(f"未知的网格类型 {mesh_type!r}, 可选 {MESH_TYPES}")
    mesh_class, dim, templates = _TEMPLATES[mesh_type]

    shape = (nx, ny) if dim == 2 else (nx, ny, nz)
    if dim == 2 and nz is not None:
        raise ValueError(f"二维网格 {mesh_type!r} 不接受 nz")
    for name, value in zip(('nx', 'ny', 'nz'), shape):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise TypeError(f"{name} 须为正整数, 实际为 {value!r}")
    shape = tuple(int(n) for n in shape)
    if len(box) != 2 * dim:
        raise ValueError(f"{mesh_type!r} 网格的 box 须有 {2 * dim} 个数, 实际为 {len(box)}")

    node = _grid_nodes(box, shape, device)
    cell = _grid_cells(shape, templates, device)
    mesh = mesh_class(node, cell)

    classes = BoxTranslationClasses(num_classes=len(templates),
                                    num_cells=int(cell.shape[0]),
                                    device=device)

    return BoxMesh(mesh=mesh, classes=classes)


def _grid_nodes(box: Sequence[float], shape: Tuple[int, ...], device) -> TensorLike:
    """生成格点坐标, x 最慢、最后一轴最快.

    Returns
    -------
    node : (prod(n + 1), GD) 的 float64 坐标.
    """
    axes = [bm.linspace(box[2 * d], box[2 * d + 1], n + 1, dtype=bm.float64, device=device)
            for d, n in enumerate(shape)]
    grids = bm.meshgrid(*axes, indexing='ij')

    return bm.stack([bm.reshape(g, (-1, )) for g in grids], axis=1)


def _grid_cells(shape: Tuple[int, ...],
                templates: Tuple[Tuple[Tuple[int, ...], ...], ...],
                device) -> TensorLike:
    """按剖分模板生成单元, 同一格子的 N_k 个单元连续编号.

    Returns
    -------
    cell : (N_k * prod(n), NV) 的 int32 顶点编号.
    """
    # 格点编号的逐轴步长, 最后一轴为 1
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * (shape[d + 1] + 1)

    # 每个格子最小角点的格点编号, 格子按字典序排列
    num_nodes = strides[0] * (shape[0] + 1)
    grid = bm.reshape(bm.arange(num_nodes, dtype=bm.int32, device=device),
                      tuple(n + 1 for n in shape))
    corner = bm.reshape(grid[tuple(slice(0, n) for n in shape)], (-1, 1, 1))

    # 模板顶点相对最小角点的编号偏移, (N_k, NV)
    local = bm.asarray([[sum(o * s for o, s in zip(offset, strides)) for offset in template]
                        for template in templates], dtype=bm.int32, device=device)

    return bm.reshape(corner + local[None, :, :], (-1, local.shape[1]))
