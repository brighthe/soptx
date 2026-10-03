# 移植自 brighthe/fealpy ``fealpy/mesh/schema/classic/edge.py`` @ f474a5775.
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

__all__ = ["LagrangeEdgeSchema", "EdgeSchema"]


class LagrangeEdgeSchema(_ScalarOrderSchema):
    """Represent an immutable Lagrange edge of geometry order ``p``.

    ``p`` is a positive integer.  Reference input uses the barycentric order
    ``(lambda_0, lambda_1)`` and positive-order basis columns follow the
    complete topology-first node layout.
    """

    __slots__ = ()

    type_id = "lagrange_edge"
    schema_version = 1
    descriptor_parameter_names = ("p",)
    name = "edge"
    top_dim = 1
    _vertex_count = 2
    OFace = _freeze_local_entities({
        "node": [[0], [1]]
    })
    SFace = _freeze_local_entities({
        "node": [[0], [1]]
    })
    orientation = ((0, 1), (1, 0))

    def _raw_node_keys(self):
        return _simplex_node_keys(self.p, self._vertex_count)

    def _candidate_vertex_permutations(self):
        return _simplex_vertex_permutations(self._vertex_count)

    @classmethod
    def _entity(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Return the selected edge connectivity with an explicit entity axis.

        Parameters:
            ctx (EntityContext): The mesh block and edge sector.
            index (Index | None): Selected edge entities, or ``None`` for all edges.

        Returns:
            Tensor: The edge connectivity tensor with shape ``(NC, 2)``.
        """
        edge = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(edge.shape) == 1:
            edge = bm.reshape(edge, (1, -1))
        return edge

    @classmethod
    def _points(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Return the endpoint coordinates of the selected edges.

        Parameters:
            ctx (EntityContext): The mesh block and edge sector.
            index (Index | None): Selected edge entities, or ``None`` for all edges.

        Returns:
            Tensor: Endpoint coordinates with shape ``(NC, 2, GD)``.
        """
        return ctx.block.positions[cls._entity(ctx, index)]

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        """Return interpolation multi-indices on the reference edge.

        Parameters:
            order (tuple[int, ...]): Polynomial degree on the edge.
            internal (bool, optional): If ``True``, return only interior multi-indices.
            tensorprod (bool, optional): If ``True``, convert the simplex ordering to the
                tensor-product-compatible ordering used by interpolation-point utilities.

        Returns:
            Tensor: The multi-index tensor with one column per edge endpoint.
        """
        p = _require_order_tuple(order, "edge multi_index", 1)[0]
        if internal:
            mi = _MI.multi_index_inner(p, 2)
        else:
            mi = _MI.multi_index_matrix(p, 2)
        if tensorprod:
            return multi_index_tensorprod(mi)
        return mi

    @classmethod
    def num_multi_index(cls, order: tuple[int, ...], *, internal: bool = False) -> int:
        """Return the number of interpolation multi-indices on the edge.

        Parameters:
            order (tuple[int, ...]): Polynomial degree on the edge.
            internal (bool, optional): If ``True``, count only interior multi-indices.

        Returns:
            int: The number of local interpolation multi-indices.
        """
        p = _require_order_tuple(order, "edge num_multi_index", 1)[0]
        if internal:
            return p - 1 if p > 1 else 0
        return p + 1

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute edge barycenters as the average of the two endpoints.

        Parameters:
            ctx (EntityContext): The mesh block and edge sector.
            index (Index | None): Selected edge entities, or ``None`` for all edges.

        Returns:
            Tensor: Edge barycenters with shape ``(NC, GD)``.
        """
        points = cls._points(ctx, index)
        return bm.mean(points, axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """Map edge barycentric coordinates to physical points.

        Parameters:
            ctx (EntityContext): The mesh block and edge sector.
            bcs (tuple[Tensor, ...]): Edge barycentric coordinates with one tensor of
                shape ``(NQ, 2)``.
            index (Index | None): Selected edge entities, or ``None`` for all edges.

        Returns:
            Tensor: Physical points with shape ``(NC, NQ, GD)``.
        """
        bcs = _require_bcs_tuple(bcs, "edge bc_to_point", 1)
        if bcs[0].shape[-1] != 2:
            raise ValueError(f"edge barycentric coordinates expect last dimension 2, got {bcs[0].shape[-1]}")

        points = cls._points(ctx, index)
        return bm.einsum("...j,cjd->c...d", bcs[0], points)

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        """Return gradients of the edge barycentric coordinates.

        Parameters:
            ctx (EntityContext): The mesh block and edge sector.
            index (Index | None): Selected edge entities, or ``None`` for all edges.
            bcs (tuple[Tensor, ...] | None, optional): Evaluation points in barycentric
                form. When provided, the result is broadcast to shape
                ``(NC, NQ, 2, GD_or_2)``.
            ref (bool, optional): If ``True``, return gradients with respect to the two
                barycentric coordinates on the reference edge. If ``False``, return
                physical gradients with respect to cartesian coordinates.

        Returns:
            Tensor: Gradients of ``lambda_0`` and ``lambda_1``. Without ``bcs``, the
            shape is ``(NC, 2, GD_or_2)``. With ``bcs``, the gradients are broadcast
            along the quadrature axis.
        """
        points = cls._points(ctx, index)
        nc = int(points.shape[0])
        if ref:
            grad = bm.broadcast_to(
                bm.eye(2, dtype=ctx.block.positions.dtype)[None, :, :],
                (nc, 2, 2),
            )
        else:
            tangent = points[:, 1, :] - points[:, 0, :]
            sqnorm = bm.sum(tangent * tangent, axis=1, keepdims=True)
            g1 = tangent / sqnorm
            g0 = -g1
            grad = bm.stack([g0, g1], axis=1)
        if bcs is None:
            return grad
        bcs = _require_bcs_tuple(bcs, "edge grad_lambda", 1)
        nq = int(bcs[0].shape[0])
        return bm.broadcast_to(grad[:, None, :, :], (nc, nq, grad.shape[1], grad.shape[2]))

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        """Return a 1D Gauss-Legendre quadrature rule on the reference edge.

        Parameters:
            q (int): Quadrature order.
            qtype (str | None, optional): Quadrature family. Only ``"legendre"`` is
                supported for edges.
            device (optional): Backend device on which the quadrature data is created.

        Returns:
            Quadrature: The quadrature object for the reference edge.
        """
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported edge quadrature type: {qtype!r}")
        from ....quadrature import GaussLegendreQuadrature
        return GaussLegendreQuadrature(q, device=device)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Return the physical lengths of the selected edges.

        Parameters:
            ctx (EntityContext): The mesh block and edge sector.
            index (Index | None): Selected edge entities, or ``None`` for all edges.

        Returns:
            Tensor: Edge lengths with shape ``(NC,)``.
        """
        points = cls._points(ctx, index)
        tangent = points[:, 1, :] - points[:, 0, :]
        return bm.linalg.norm(tangent, axis=1)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Return normal directions orthogonal to each edge tangent.

        In 2D, one normal direction is returned for each edge. In 3D, two
        mutually orthogonal normal directions are constructed.

        Parameters:
            ctx (EntityContext): The mesh block and edge sector.
            index (Index | None): Selected edge entities, or ``None`` for all edges.

        Returns:
            Tensor: Normal directions with shape ``(NC, GD - 1, GD)`` for ``GD <= 3``.
        """
        points = cls._points(ctx, index)
        tangent = points[:, 1, :] - points[:, 0, :]
        gd = points.shape[-1]
        sqnorm = bm.sum(tangent * tangent, axis=1)

        if bm.any(sqnorm == 0):
            raise ValueError("degenerate edge has no well-defined normal directions")

        if gd == 1:
            return bm.zeros((points.shape[0], 0, gd), dtype=ctx.block.positions.dtype)

        if gd == 2:
            normal = bm.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
            return normal[:, None, :]

        if gd == 3:
            axis = bm.argmin(bm.abs(tangent), axis=1)
            ref_basis = bm.eye(gd, dtype=ctx.block.positions.dtype, device=bm.get_device(tangent))
            ref = ref_basis[axis]
            n1 = bm.cross(tangent, ref, axis=-1)
            n2 = bm.cross(tangent, n1, axis=-1)
            return bm.stack([n1, n2], axis=1)

        raise NotImplementedError(f"edge normal is only implemented for GD <= 3, got {gd}")

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Return the non-unit physical tangent vectors of the selected edges.

        Parameters:
            ctx (EntityContext): The mesh block and edge sector.
            index (Index | None): Selected edge entities, or ``None`` for all edges.

        Returns:
            Tensor: Tangent vectors with shape ``(NC, 1, GD)``.
        """
        points = cls._points(ctx, index)
        return (points[:, 1, :] - points[:, 0, :])[:, None, :]

    @classmethod
    def transform(cls, ctx: EntityContext, func, kind: str = "value"):
        """Wrap a cartesian function so it can be evaluated on edge barycentric points.

        Parameters:
            ctx (EntityContext): The mesh block and edge sector.
            func: A callable defined on physical points.
            kind (str, optional): Transformation kind. Only ``"value"`` is supported.

        Returns:
            Callable: A wrapper that accepts edge barycentric coordinates.
        """
        points = ctx.block.positions[ctx.sector.indices]

        def wrapper(bc: Tensor) -> Tensor:
            x = bm.einsum("...j,cjd->c...d", bc, points)
            value = func(x)
            if kind == "value":
                return value
            raise NotImplementedError(f"Unsupported edge transform kind: {kind!r}")

        return wrapper

    @classmethod
    def jacobi_matrix(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None,
    ) -> Tensor:
        """Compute the Jacobian of the reference-to-physical edge map.

        Parameters:
            ctx (EntityContext): The mesh block and edge sector.
            bcs (tuple[Tensor, ...]): Barycentric evaluation points on the reference
                edge.
            index (Index | None): Selected edge entities, or ``None`` for all edges.

        Returns:
            Tensor: The Jacobian tensor with shape ``(NC, NQ, GD, 1)``. Its last axis
            stores the derivative of the physical map with respect to the reference
            coordinate ``u``.
        """
        bcs = _require_bcs_tuple(bcs, "edge jacobi_matrix", 1)
        points = cls._points(ctx, index)
        gphi = cls().grad_shape_function_reference(bcs)
        return bm.einsum("cid,qin->cqdn", points, gphi)


EdgeSchema = LagrangeEdgeSchema
