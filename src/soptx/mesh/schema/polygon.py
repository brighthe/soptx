# 移植自 brighthe/fealpy ``fealpy/mesh/schema/polygon.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""顶点数可变的多边形实体 Schema."""

from dataclasses import dataclass

from ...backend import bm, Index, Tensor
from ..storage import EntityContext
from .entity_schema import EntitySchema

__all__ = ["PolygonSchema"]


@dataclass(frozen=True, slots=True)
class PolygonSchema(EntitySchema):
    """顶点环大小可变的线性多边形.

    多边形没有单一固定的参考单元, 也没有固定的局部节点布局, 因此其分区连接使用展平的
    顶点编号张量加 ``indptr`` 张量. 参考基函数与参考映射有意不支持; 拓扑与物理几何直接
    使用具体的变长分区.
    """

    type_id = "polygon"
    schema_version = 1
    descriptor_parameter_names = ()
    name = "poly"
    top_dim = 2
    orientation = ()

    def __hash__(self) -> int:
        return hash((type(self), self.descriptor))

    def validate_connectivity_counts(self, counts: Tensor) -> None:
        """要求每个多边形单元至少有三个顶点."""
        if bool(bm.any(counts < 3)):
            raise ValueError("polygon cells must contain at least three vertices")

    def number_of_vertices(self) -> int:
        """多边形族没有固定的顶点数, 调用即报错."""
        raise NotImplementedError("PolygonSchema has variable vertex cardinality")

    def number_of_nodes(self) -> int:
        """多边形族没有固定的连接宽度, 调用即报错."""
        raise NotImplementedError("PolygonSchema has variable node cardinality")

    def local_vertices(self) -> tuple[int, ...]:
        """多边形族没有固定的局部顶点布局, 调用即报错."""
        raise NotImplementedError("PolygonSchema has a dynamic local vertex layout")

    def local_entity_groups(self, top_dim: int):
        """可变多边形没有固定的局部实体表, 调用即报错."""
        if type(top_dim) is not int:
            raise TypeError("top_dim must be a plain integer")
        if top_dim < 0 or top_dim > self.top_dim:
            raise ValueError(f"top_dim must be in [0, {self.top_dim}], got {top_dim}")
        raise NotImplementedError(
            "PolygonSchema local entities depend on sector connectivity"
        )

    def vertex_permutations(self) -> tuple[tuple[int, ...], ...]:
        """多边形没有定宽的顶点置换, 调用即报错."""
        raise NotImplementedError("PolygonSchema permutations are entity-dependent")

    def node_permutation(self, vertex_permutation: tuple[int, ...]) -> tuple[int, ...]:
        """多边形没有定宽的节点置换, 调用即报错."""
        raise NotImplementedError("PolygonSchema permutations are entity-dependent")

    @classmethod
    def local_entity(cls, tgt_name: str, /, indexing="o"):
        """可变多边形没有固定的局部实体表, 调用即报错."""
        raise NotImplementedError(
            "PolygonSchema local entities depend on sector connectivity"
        )

    @classmethod
    def size(cls, ctx: EntityContext) -> int:
        """变长分区中多边形单元的个数."""
        if ctx.sector.indptr is None:
            raise ValueError("PolygonSchema requires sector.indptr")
        return int(ctx.sector.indptr.shape[0]) - 1

    @classmethod
    def _cell_ids(cls, ctx: EntityContext) -> tuple[Tensor, Tensor]:
        indptr = ctx.sector.indptr
        if indptr is None:
            raise ValueError("PolygonSchema requires sector.indptr")
        counts = indptr[1:] - indptr[:-1]
        cell_ids = bm.repeat(
            bm.arange(
                counts.shape[0],
                dtype=ctx.sector.indices.dtype,
                device=bm.get_device(ctx.sector.indices),
            ),
            counts,
        )
        return counts, cell_ids

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """各多边形顶点的算术平均."""
        counts, cell_ids = cls._cell_ids(ctx)
        points = ctx.block.positions[ctx.sector.indices]
        result = bm.zeros(
            (counts.shape[0], points.shape[-1]),
            dtype=points.dtype,
            device=bm.get_device(points),
        )
        result = bm.index_add(result, cell_ids, points)
        result = result / bm.astype(counts[:, None], points.dtype)
        return result if index is None else result[index]

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """按鞋带公式计算的多边形面积 (无符号)."""
        if int(ctx.block.positions.shape[1]) != 2:
            raise ValueError("PolygonSchema area currently requires geo_dimension == 2")
        indptr = ctx.sector.indptr
        if indptr is None:
            raise ValueError("PolygonSchema requires sector.indptr")
        counts, cell_ids = cls._cell_ids(ctx)
        vertices = ctx.sector.indices
        successor = bm.concat([vertices[1:], vertices[:1]], axis=0)
        if int(vertices.shape[0]) > 0:
            successor = bm.set_at(
                successor,
                indptr[1:] - 1,
                vertices[indptr[:-1]],
            )
        p0 = ctx.block.positions[vertices]
        p1 = ctx.block.positions[successor]
        cross = p0[:, 0] * p1[:, 1] - p0[:, 1] * p1[:, 0]
        area = bm.zeros(
            (counts.shape[0],),
            dtype=p0.dtype,
            device=bm.get_device(p0),
        )
        area = bm.index_add(area, cell_ids, cross)
        area = bm.abs(area) * 0.5
        return area if index is None else area[index]

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        """物理嵌入空间的维数."""
        return int(ctx.block.positions.shape[1])

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        """多边形扇形剖分的子三角形上所用的三角形积分公式."""
        if qtype != "legendre":
            raise ValueError(f"unsupported quadrature type: {qtype}")
        from ...quadrature import TriangleQuadrature

        return TriangleQuadrature(q, device=device)

    @classmethod
    def barycentric(cls, ctx, func, index):
        """不支持: 多边形没有规范的重心坐标, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError("polygons have no canonical barycentric coordinates")

    @classmethod
    def bc_to_point(cls, ctx, bcs, index):
        """不支持: 多边形没有规范的参考映射, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError("polygons have no canonical reference map")

    @classmethod
    def integral(cls, ctx, func, q, index):
        """不支持: 组合积分应使用多边形网格的 ``integral``, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError("use PolygonMesh.integral for composite integration")

    @classmethod
    def normal(cls, ctx, index):
        """不支持: 二维版本未定义多边形单元的法向, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError("polygon cell normals are not defined in the 2D release")

    @classmethod
    def tangent(cls, ctx, index):
        """不支持: 多边形没有固定的切标架, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError("polygons have no fixed tangent frame")

    def shape_function(self, bcs):
        """不支持: 多边形没有规范的几何基函数, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError("polygons have no canonical geometry basis")

    def grad_shape_function_barycentric(self, bcs):
        """不支持: 多边形没有规范的几何基函数, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError("polygons have no canonical geometry basis")

    def grad_shape_function_reference(self, bcs):
        """不支持: 多边形没有规范的几何基函数, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError("polygons have no canonical geometry basis")

    def lagrange_basis_function(self, bcs, p):
        """不支持: 多边形没有规范的 Lagrange 基函数, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError("polygons have no canonical Lagrange basis")

    def grad_lagrange_basis_function_barycentric(self, bcs, p):
        """不支持: 多边形没有规范的 Lagrange 基函数, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError("polygons have no canonical Lagrange basis")

    def grad_lagrange_basis_function_reference(self, bcs, p):
        """不支持: 多边形没有规范的 Lagrange 基函数, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError("polygons have no canonical Lagrange basis")
