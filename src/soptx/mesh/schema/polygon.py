# 移植自 brighthe/fealpy ``fealpy/mesh/schema/polygon.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Variable-cardinality polygon entity Schema."""

from dataclasses import dataclass

from ...backend import bm, Index, Tensor
from ..storage import EntityContext
from .entity_schema import EntitySchema

__all__ = ["PolygonSchema"]


@dataclass(frozen=True, slots=True)
class PolygonSchema(EntitySchema):
    """Describe a linear polygon with a variable-size cyclic vertex ring.

    A polygon has no single fixed reference cell or fixed local-node layout.
    Its sector connectivity therefore uses a flat vertex-index tensor and an
    ``indptr`` tensor. Reference basis and reference-map operations are
    intentionally unsupported; topology and physical geometry use the
    concrete ragged sector directly.
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
        """Require at least three vertices in every polygon cell."""
        if bool(bm.any(counts < 3)):
            raise ValueError("polygon cells must contain at least three vertices")

    def number_of_vertices(self) -> int:
        """Reject a fixed vertex count for the polygon family."""
        raise NotImplementedError("PolygonSchema has variable vertex cardinality")

    def number_of_nodes(self) -> int:
        """Reject a fixed connectivity width for the polygon family."""
        raise NotImplementedError("PolygonSchema has variable node cardinality")

    def local_vertices(self) -> tuple[int, ...]:
        """Reject a fixed local vertex layout for the polygon family."""
        raise NotImplementedError("PolygonSchema has a dynamic local vertex layout")

    def local_entity_groups(self, top_dim: int):
        """Reject fixed local-entity tables for variable polygons."""
        if type(top_dim) is not int:
            raise TypeError("top_dim must be a plain integer")
        if top_dim < 0 or top_dim > self.top_dim:
            raise ValueError(f"top_dim must be in [0, {self.top_dim}], got {top_dim}")
        raise NotImplementedError(
            "PolygonSchema local entities depend on sector connectivity"
        )

    def vertex_permutations(self) -> tuple[tuple[int, ...], ...]:
        """Reject fixed-width polygon permutations."""
        raise NotImplementedError("PolygonSchema permutations are entity-dependent")

    def node_permutation(self, vertex_permutation: tuple[int, ...]) -> tuple[int, ...]:
        """Reject fixed-width polygon node permutations."""
        raise NotImplementedError("PolygonSchema permutations are entity-dependent")

    @classmethod
    def local_entity(cls, tgt_name: str, /, indexing="o"):
        """Reject fixed local-entity tables for variable polygons."""
        raise NotImplementedError(
            "PolygonSchema local entities depend on sector connectivity"
        )

    @classmethod
    def size(cls, ctx: EntityContext) -> int:
        """Return the number of polygon cells in the ragged sector."""
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
        """Return the arithmetic mean of each polygon's vertices."""
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
        """Return unsigned polygon areas by the shoelace formula."""
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
        """Return the physical embedding dimension."""
        return int(ctx.block.positions.shape[1])

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        """Return the triangle rule used on polygon fan subtriangles."""
        if qtype != "legendre":
            raise ValueError(f"unsupported quadrature type: {qtype}")
        from ...quadrature import TriangleQuadrature

        return TriangleQuadrature(q, device=device)

    @classmethod
    def barycentric(cls, ctx, func, index):
        raise NotImplementedError("polygons have no canonical barycentric coordinates")

    @classmethod
    def bc_to_point(cls, ctx, bcs, index):
        raise NotImplementedError("polygons have no canonical reference map")

    @classmethod
    def integral(cls, ctx, func, q, index):
        raise NotImplementedError("use PolygonMesh.integral for composite integration")

    @classmethod
    def normal(cls, ctx, index):
        raise NotImplementedError("polygon cell normals are not defined in the 2D release")

    @classmethod
    def tangent(cls, ctx, index):
        raise NotImplementedError("polygons have no fixed tangent frame")

    def shape_function(self, bcs):
        raise NotImplementedError("polygons have no canonical geometry basis")

    def grad_shape_function_barycentric(self, bcs):
        raise NotImplementedError("polygons have no canonical geometry basis")

    def grad_shape_function_reference(self, bcs):
        raise NotImplementedError("polygons have no canonical geometry basis")

    def lagrange_basis_function(self, bcs, p):
        raise NotImplementedError("polygons have no canonical Lagrange basis")

    def grad_lagrange_basis_function_barycentric(self, bcs, p):
        raise NotImplementedError("polygons have no canonical Lagrange basis")

    def grad_lagrange_basis_function_reference(self, bcs, p):
        raise NotImplementedError("polygons have no canonical Lagrange basis")
