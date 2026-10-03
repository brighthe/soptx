# 移植自 brighthe/fealpy ``fealpy/mesh/schema/classic/pyramid.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

from ....backend import bm
from ....backend import Index, Tensor
from ...ipoints import MultiIndex as _MI, multi_index_tensorprod
from ..entity_schema import _freeze_local_entities
from .base import (
    EntityContext,
    _ScalarOrderSchema,
    _group_entity_definitions,
    _quadrilateral_vertex_permutations,
    _simplex_node_keys,
    _require_bcs_tuple,
    _require_order_tuple,
)

__all__ = ["LagrangePyramidSchema", "PyramidSchema"]


class LagrangePyramidSchema(_ScalarOrderSchema):
    """Immutable linear Lagrange pyramid schema.

    Only the five-node geometry order ``p=1`` is supported.  Reference input
    is ``(bc_u, bc_v, bc_w)`` and geometry reference gradients are returned in
    ``(u, v, w)`` order.  Arbitrary-order Lagrange basis evaluation, including
    a request with ``p=1``, is deliberately unsupported and raises
    ``NotImplementedError``; the approved geometry formula is not promoted to
    a finite-element reference-basis contract.
    """

    __slots__ = ()
    type_id = "lagrange_pyramid"
    schema_version = 1
    descriptor_parameter_names = ("p",)
    name = "pyramid"
    top_dim = 3
    _vertex_count = 5
    OFace = _freeze_local_entities({
        "quad": [[0, 3, 2, 1]],
        "tri": [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        "edge": [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [0, 4], [1, 4], [2, 4], [3, 4],
        ],
        "node": [[0], [1], [2], [3], [4]],
    })
    SFace = _freeze_local_entities({
        "quad": [[0, 1, 2, 3]],
        "tri": [[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]],
        "edge": [
            [0, 1], [1, 2], [2, 3], [0, 3],
            [0, 4], [1, 4], [2, 4], [3, 4],
        ],
        "node": [[0], [1], [2], [3], [4]],
    })

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.p != 1:
            raise NotImplementedError(
                "LagrangePyramidSchema currently supports only geometry p=1"
            )

    def _raw_node_keys(self):
        return _simplex_node_keys(1, self._vertex_count)

    def _edge_definitions(self):
        from .edge import LagrangeEdgeSchema

        schema = LagrangeEdgeSchema(1)
        return tuple((schema, vertices) for vertices in self.OFace["edge"])

    def _face_definitions(self):
        from .quadrilateral import LagrangeQuadrilateralSchema
        from .triangle import LagrangeTriangleSchema

        return (
            (LagrangeQuadrilateralSchema(1), self.OFace["quad"][0]),
            *tuple(
                (LagrangeTriangleSchema(1), vertices)
                for vertices in self.OFace["tri"]
            ),
        )

    def _layout_entity_definitions(self):
        definitions = self._edge_definitions() + self._face_definitions()
        return tuple((schema, (vertices,)) for schema, vertices in definitions)

    def _local_entity_definitions(self, top_dim: int):
        if top_dim == 1:
            return _group_entity_definitions(self._edge_definitions())
        if top_dim == 2:
            return _group_entity_definitions(self._face_definitions())
        return ()

    def _candidate_vertex_permutations(self):
        return tuple(
            permutation + (4,)
            for permutation in _quadrilateral_vertex_permutations()
        )

    @staticmethod
    def _split_bcs(bcs: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, ...]:
        bcs = _require_bcs_tuple(bcs, "pyramid bcs", 3)
        if bcs[0].shape[-1] != 2 or bcs[1].shape[-1] != 2 or bcs[2].shape[-1] != 2:
            raise ValueError("each pyramid barycentric tensor must have shape (..., 2)")

        bcu, bcv, bcw = bcs
        return (
            bcu[..., 0], bcu[..., 1],
            bcv[..., 0], bcv[..., 1],
            bcw[..., 0], bcw[..., 1],
        )

    @staticmethod
    def _product(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        return bm.einsum("i,j,k->ijk", a, b, c).reshape(-1)

    @classmethod
    def geometry_shape_function(cls, bcs: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        lu0, lu1, lv0, lv1, lw0, lw1 = cls._split_bcs(bcs)
        phi0 = cls._product(lu0, lv0, lw0)
        phi1 = cls._product(lu1, lv0, lw0)
        # The schema uses cyclic base order (u0v0, u1v0, u1v1, u0v1).
        phi2 = cls._product(lu1, lv1, lw0)
        phi3 = cls._product(lu0, lv1, lw0)
        phi4 = cls._product(bm.ones_like(lu0), bm.ones_like(lv0), lw1)
        return bm.stack([phi0, phi1, phi2, phi3, phi4], axis=-1)

    @classmethod
    def geometry_grad_shape_function(cls, bcs: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        lu0, lu1, lv0, lv1, lw0, _ = cls._split_bcs(bcs)
        z = cls._product(lu0, lv0, bm.zeros_like(lw0))
        o = cls._product(bm.ones_like(lu0), bm.ones_like(lv0), bm.ones_like(lw0))

        g0 = bm.stack([
            -cls._product(bm.ones_like(lu0), lv0, lw0),
            -cls._product(lu0, bm.ones_like(lv0), lw0),
            -cls._product(lu0, lv0, bm.ones_like(lw0)),
        ], axis=-1)
        g1 = bm.stack([
            cls._product(bm.ones_like(lu1), lv0, lw0),
            -cls._product(lu1, bm.ones_like(lv0), lw0),
            -cls._product(lu1, lv0, bm.ones_like(lw0)),
        ], axis=-1)
        g2 = bm.stack([
            cls._product(bm.ones_like(lu1), lv1, lw0),
            cls._product(lu1, bm.ones_like(lv1), lw0),
            -cls._product(lu1, lv1, bm.ones_like(lw0)),
        ], axis=-1)
        g3 = bm.stack([
            -cls._product(bm.ones_like(lu0), lv1, lw0),
            cls._product(lu0, bm.ones_like(lv1), lw0),
            -cls._product(lu0, lv1, bm.ones_like(lw0)),
        ], axis=-1)
        g4 = bm.stack([z, z, o], axis=-1)
        return bm.stack([g0, g1, g2, g3, g4], axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...],
                    index: Index | None) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "pyramid bc_to_point", 3)
        pyramid = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[pyramid]
        phi = cls().shape_function(bcs)
        return bm.einsum("qi,cid->cqd", phi, points)

    def shape_function(self, bcs: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Evaluate the five-node geometry basis on product points.

        The result has shape ``(Qu * Qv * Qw, 5)`` and columns follow the
        complete local-node order.
        """
        if not isinstance(bcs, tuple):
            raise TypeError(
                "pyramid shape_function expects a tuple of barycentric tensors"
            )
        return type(self).geometry_shape_function(bcs)

    def grad_shape_function_barycentric(
        self,
        bcs: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """Report the unsupported independent-barycentric gradient."""
        raise NotImplementedError(
            "pyramid geometry does not expose independent barycentric gradients"
        )

    def grad_shape_function_reference(
        self,
        bcs: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """Return geometry gradients with shape ``(Q, 5, 3)``.

        The final axis follows public reference order ``(u, v, w)``.
        """
        if not isinstance(bcs, tuple):
            raise TypeError(
                "pyramid grad_shape_function_reference expects a tuple of tensors"
            )
        return type(self).geometry_grad_shape_function(bcs)

    def lagrange_basis_function(
        self,
        bcs: tuple[Tensor, Tensor, Tensor],
        p: int,
    ) -> Tensor:
        """Report that arbitrary-order pyramid Lagrange bases are unsupported."""
        raise NotImplementedError(
            "LagrangePyramidSchema does not yet support an arbitrary-order "
            "reference Lagrange basis"
        )

    def grad_lagrange_basis_function_barycentric(
        self,
        bcs: tuple[Tensor, Tensor, Tensor],
        p: int,
    ) -> Tensor:
        """Report that pyramid barycentric basis gradients are unsupported."""
        raise NotImplementedError(
            "LagrangePyramidSchema does not yet support arbitrary-order "
            "barycentric basis gradients"
        )

    def grad_lagrange_basis_function_reference(
        self,
        bcs: tuple[Tensor, Tensor, Tensor],
        p: int,
    ) -> Tensor:
        """Report that pyramid reference basis gradients are unsupported."""
        raise NotImplementedError(
            "LagrangePyramidSchema does not yet support arbitrary-order "
            "reference basis gradients"
        )

    @classmethod
    def jacobi_matrix(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None
    ) -> Tensor:
        pyramid = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[pyramid]
        gphi = cls().grad_shape_function_reference(bcs)
        return bm.einsum("cid,qik->cqdk", points, gphi)

    @classmethod
    def transform_grad(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, Tensor, Tensor],
        ref_grad: Tensor,
        index: Index | None
    ) -> Tensor:
        J = cls.jacobi_matrix(ctx, bcs, index)
        metric = bm.einsum("cqdk,cqdl->cqkl", J, J)
        metric_inv = bm.linalg.inv(metric)
        return bm.einsum("cqdk,cqkl,qil->cqid", J, metric_inv, ref_grad)

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, Tensor, Tensor] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        pyramid = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(pyramid.shape) == 1:
            pyramid = bm.reshape(pyramid, (1, -1))
        if bcs is None:
            bcs = (
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
            )
            squeeze_q = True
        else:
            bcs = _require_bcs_tuple(bcs, "pyramid grad_lambda", 3)
            squeeze_q = False
        ref3 = cls().grad_shape_function_reference(bcs)
        z = bm.zeros((ref3.shape[0], ref3.shape[1], 3), dtype=ref3.dtype)
        ref6 = bm.concatenate([ref3[:, :, 0:1], z[:, :, 0:1], ref3[:, :, 1:2], z[:, :, 1:2], ref3[:, :, 2:3], z[:, :, 2:3]], axis=-1)
        if ref:
            grad = bm.broadcast_to(ref6[None, :, :, :], (pyramid.shape[0], ref6.shape[0], 5, 6))
            return grad[:, 0, :, :] if squeeze_q else grad
        grad = cls.transform_grad(ctx, bcs, ref3, index)
        return grad[:, 0, :, :] if squeeze_q else grad

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        p = _require_order_tuple(order, "pyramid multi_index", 1)[0]
        if internal:
            mi = _MI.multi_index_inner(p, 5)
        else:
            mi = _MI.multi_index_matrix(p, 5) # TODO: not correct
        if tensorprod:
            return multi_index_tensorprod(mi)
        return mi

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported pyramid quadrature type: {qtype!r}")
        from ....quadrature import GaussLegendreQuadrature, TensorProductQuadrature

        qf_uv = GaussLegendreQuadrature(q)
        qf_w = GaussLegendreQuadrature(max(q, 2))
        return TensorProductQuadrature((qf_uv, qf_uv, qf_w))

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        pyramid = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[pyramid]
        return bm.mean(points, axis=1)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        pyramid = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[pyramid]

        def tet_volume(i: int, j: int, k: int, m: int) -> Tensor:
            vectors = bm.stack([
                points[:, j, :] - points[:, i, :],
                points[:, k, :] - points[:, i, :],
                points[:, m, :] - points[:, i, :],
            ], axis=1)
            gram = bm.einsum("cig,cjg->cij", vectors, vectors)
            return bm.sqrt(bm.abs(bm.linalg.det(gram))) / 6.0

        return tet_volume(0, 1, 3, 4) + tet_volume(0, 3, 2, 4)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        pyramid = ctx.sector.indices if index is None else ctx.sector.indices[index]
        GD = cls.geo_dimension(ctx)
        normal_count = GD - cls.top_dim
        if normal_count < 0:
            raise ValueError(
                f"geometric dimension ({GD}) must be greater than or equal to "
                f"topological dimension ({cls.top_dim})"
            )

        if normal_count == 0:
            return bm.zeros(
                (pyramid.shape[0], 0, GD),
                dtype=ctx.block.positions.dtype,
                device=bm.get_device(ctx.block.positions),
            )

        _, _, vh = bm.linalg.svd(cls.tangent(ctx, index), full_matrices=True)
        return vh[:, cls.top_dim:, :]

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        pyramid = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[pyramid]
        # Representative local frame; Jacobian-based tangents depend on reference points.
        return bm.stack([
            points[:, 1, :] - points[:, 0, :],
            points[:, 2, :] - points[:, 0, :],
            points[:, 4, :] - points[:, 0, :],
        ], axis=1)


PyramidSchema = LagrangePyramidSchema
