# 移植自 brighthe/fealpy ``fealpy/mesh/schema/classic/tetrahedron.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

from ....backend import bm
from ....backend import Index, Tensor
from ...ipoints import MultiIndex as _MI, multi_index_tensorprod
from ..entity_schema import _freeze_local_entities
from .base import (
    EntityContext,
    _ScalarOrderSchema,
    _simplex_node_keys,
    _simplex_vertex_permutations,
    _require_bcs_tuple,
    _require_order_tuple,
)

__all__ = ["LagrangeTetrahedronSchema", "TetrahedronSchema"]


class LagrangeTetrahedronSchema(_ScalarOrderSchema):
    """Represent an immutable Lagrange tetrahedron of geometry order ``p``.

    ``p`` is a positive integer.  Reference input uses barycentric order
    ``(lambda_0, lambda_1, lambda_2, lambda_3)``; reference derivatives use
    the last three coordinates with ``lambda_0`` dependent.
    """

    __slots__ = ()
    type_id = "lagrange_tetrahedron"
    schema_version = 1
    descriptor_parameter_names = ("p",)
    name = "tet"
    top_dim = 3
    _vertex_count = 4
    OFace = _freeze_local_entities({
        "tri": [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
        "edge": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        "node": [[0], [1], [2], [3]],
    })
    SFace = _freeze_local_entities({
        "tri": [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        "edge": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        "node": [[0], [1], [2], [3]],
    })
    ref_measure = 1 / 6

    def _raw_node_keys(self):
        return _simplex_node_keys(self.p, self._vertex_count)

    def _layout_entity_definitions(self):
        from .edge import LagrangeEdgeSchema
        from .triangle import LagrangeTriangleSchema

        return (
            (
                LagrangeEdgeSchema(self.p),
                ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
            ),
            (LagrangeTriangleSchema(self.p), self.OFace["tri"]),
        )

    def _local_entity_definitions(self, top_dim: int):
        if top_dim == 1:
            from .edge import LagrangeEdgeSchema

            return ((LagrangeEdgeSchema(self.p), self.OFace["edge"]),)
        if top_dim == 2:
            from .triangle import LagrangeTriangleSchema

            return ((LagrangeTriangleSchema(self.p), self.OFace["tri"]),)
        return ()

    def _candidate_vertex_permutations(self):
        return _simplex_vertex_permutations(self._vertex_count)

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))
        points = ctx.block.positions[tet]
        return bm.mean(points, axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "tetrahedron bc_to_point", 1)
        if bcs[0].shape[-1] != 4:
            raise ValueError(f"tetrahedron barycentric coordinates expect last dimension 4, got {bcs[0].shape[-1]}")

        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))
        points = ctx.block.positions[tet]
        return bm.einsum("...j,cjd->c...d", bcs[0], points)

    @classmethod
    def jacobi_matrix(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None,
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "tetrahedron jacobi_matrix", 1)
        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))

        gphi = cls().grad_shape_function_reference(bcs)
        return bm.einsum("cim,qin->cqmn", ctx.block.positions[tet], gphi)

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))
        if ref:
            grad = bm.broadcast_to(
                bm.eye(4, dtype=ctx.block.positions.dtype)[None, :, :],
                (tet.shape[0], 4, 4),
            )
        else:
            gd = cls.geo_dimension(ctx)
            if gd != 3:
                raise ValueError(f"tetrahedron geometry requires GD == 3, got {gd}")
            points = ctx.block.positions[tet]
            v1 = points[:, 1, :] - points[:, 0, :]
            v2 = points[:, 2, :] - points[:, 0, :]
            v3 = points[:, 3, :] - points[:, 0, :]
            jac = bm.stack([v1, v2, v3], axis=-1)
            inv_jac = bm.linalg.inv(jac)
            g1 = inv_jac[:, 0, :]
            g2 = inv_jac[:, 1, :]
            g3 = inv_jac[:, 2, :]
            g0 = -g1 - g2 - g3
            grad = bm.stack([g0, g1, g2, g3], axis=1)
        if bcs is None:
            return grad
        bcs = _require_bcs_tuple(bcs, "tetrahedron grad_lambda", 1)
        nq = int(bcs[0].shape[0])
        return bm.broadcast_to(grad[:, None, :, :], (tet.shape[0], nq, grad.shape[1], grad.shape[2]))

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"tetrahedron geometry requires GD == 3, got {gd}")

        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))
        points = ctx.block.positions[tet]
        v1 = points[:, 1, :] - points[:, 0, :]
        v2 = points[:, 2, :] - points[:, 0, :]
        v3 = points[:, 3, :] - points[:, 0, :]
        jac = bm.stack([v1, v2, v3], axis=-1)
        return bm.abs(bm.linalg.det(jac)) / 6.0

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        p = _require_order_tuple(order, "tetrahedron multi_index", 1)[0]
        if internal:
            mi = _MI.multi_index_inner(p, 4)
        else:
            mi = _MI.multi_index_matrix(p, 4)
        if tensorprod:
            return multi_index_tensorprod(mi)
        return mi

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"tetrahedron geometry requires GD == 3, got {gd}")

        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))
        return bm.zeros((tet.shape[0], 0, 3), dtype=ctx.block.positions.dtype)

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported tetrahedron quadrature type: {qtype!r}")
        if q > 7:
            from ....quadrature.stroud_quadrature import StroudQuadrature
            return StroudQuadrature(3, q)
        from ....quadrature import TetrahedronQuadrature
        return TetrahedronQuadrature(q, device=device)

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"tetrahedron geometry requires GD == 3, got {gd}")

        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))
        points = ctx.block.positions[tet]
        v1 = points[:, 1, :] - points[:, 0, :]
        v2 = points[:, 2, :] - points[:, 0, :]
        v3 = points[:, 3, :] - points[:, 0, :]
        return bm.stack([v1, v2, v3], axis=1)


TetrahedronSchema = LagrangeTetrahedronSchema
