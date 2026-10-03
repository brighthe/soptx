# 移植自 brighthe/fealpy ``fealpy/mesh/schema/classic/node.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

from dataclasses import dataclass

from ....backend import bm
from ....backend import Index, Tensor
from ..entity_schema import _freeze_local_entities
from .base import (
    EntityContext,
    ShapedEntitySchema,
    _simplex_node_keys,
    _simplex_vertex_permutations,
    _require_bcs_tuple,
    _require_order_tuple,
    _normalize_scalar_basis_order,
)


__all__ = ["NodeSchema", "NodeQuadrature"]


class NodeQuadrature:
    def __init__(self, *, dtype=None):
        dtype = bm.float64 if dtype is None else dtype
        self.quadpts = (bm.asarray([[1.0]], dtype=dtype),)
        self.weights = bm.asarray([1.0], dtype=dtype)

    def number_of_quadrature_points(self) -> int:
        return int(self.weights.shape[0])

    def get_quadrature_points_and_weights(self) -> tuple[tuple[Tensor, ...], Tensor]:
        return self.quadpts, self.weights

    def get_quadrature_point_and_weight(self, i: int) -> tuple[tuple[Tensor, ...], Tensor]:
        return tuple(qp[i:i + 1] for qp in self.quadpts), self.weights[i]

    def __len__(self) -> int:
        return self.number_of_quadrature_points()

    def __getitem__(self, i: int) -> tuple[tuple[Tensor, ...], Tensor]:
        return self.get_quadrature_point_and_weight(i)


@dataclass(frozen=True, slots=True)
class NodeSchema(ShapedEntitySchema):
    """Represent the immutable zero-dimensional point Schema.

    The sole geometry and reference basis value is one.  Reference gradients
    have zero reference dimension, while independent-barycentric gradients
    retain one zero component.  Input is a ``(Q, 1)`` tensor of ones or a
    one-item tuple containing it.
    """

    type_id = "node"
    schema_version = 1
    descriptor_parameter_names = ()
    name = "node"
    top_dim = 0
    _vertex_count = 1
    OFace = _freeze_local_entities({})
    SFace = _freeze_local_entities({})
    orientation = ((0,),)

    def __hash__(self) -> int:
        return hash((type(self), self.descriptor))

    def _raw_node_keys(self):
        return _simplex_node_keys(1, 1)

    def _candidate_vertex_permutations(self):
        return _simplex_vertex_permutations(1)

    @classmethod
    def _indices(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        point = ctx.sector.indices if index is None else ctx.sector.indices[index]
        return bm.reshape(point, (-1,))

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        return ctx.block.positions[cls._indices(ctx, index)]

    @staticmethod
    def _point_bcs(bcs: Tensor | tuple[Tensor, ...], name: str) -> Tensor:
        if isinstance(bcs, tuple):
            if len(bcs) != 1:
                raise ValueError(
                    f"{name} expects one barycentric tensor, got {len(bcs)}"
                )
            bcs = bcs[0]
        if len(bcs.shape) != 2 or int(bcs.shape[-1]) != 1:
            raise ValueError(f"{name} expects shape (num_points, 1)")
        return bcs

    def shape_function(self, bcs: Tensor | tuple[Tensor, ...]) -> Tensor:
        """Evaluate the unique constant geometry function on a point."""
        bc = self._point_bcs(bcs, "point shape_function")
        return bm.ones((bc.shape[0], 1), dtype=bc.dtype, device=bm.get_device(bc))

    def grad_shape_function_barycentric(
        self,
        bcs: Tensor | tuple[Tensor, ...],
    ) -> Tensor:
        """Return the zero independent-barycentric gradient on a point."""
        return self.grad_lagrange_basis_function_barycentric(bcs, 0)

    def grad_shape_function_reference(
        self,
        bcs: Tensor | tuple[Tensor, ...],
    ) -> Tensor:
        """Return the zero-dimensional reference gradient on a point."""
        return self.grad_lagrange_basis_function_reference(bcs, 0)

    def lagrange_basis_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int,
    ) -> Tensor:
        """Evaluate the unique point basis for any non-negative order."""
        _normalize_scalar_basis_order(p, type(self).__name__)
        return self.shape_function(bcs)

    def grad_lagrange_basis_function_barycentric(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int,
    ) -> Tensor:
        """Return zeros with shape ``(Q, 1, 1)``."""
        _normalize_scalar_basis_order(p, type(self).__name__)
        bc = self._point_bcs(bcs, "point basis barycentric gradient")
        return bm.zeros(
            (bc.shape[0], 1, 1),
            dtype=bc.dtype,
            device=bm.get_device(bc),
        )

    def grad_lagrange_basis_function_reference(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int,
    ) -> Tensor:
        """Return zeros with shape ``(Q, 1, 0)``."""
        _normalize_scalar_basis_order(p, type(self).__name__)
        bc = self._point_bcs(bcs, "point basis reference gradient")
        return bm.zeros(
            (bc.shape[0], 1, 0),
            dtype=bc.dtype,
            device=bm.get_device(bc),
        )

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        points = cls.barycenter(ctx, index)
        if not isinstance(bcs, tuple):
            raise TypeError(f"point barycentric coordinates expect a tuple, got {type(bcs).__name__}")
        if len(bcs) != 1:
            raise ValueError(f"point barycentric coordinates expect one tensor, got {len(bcs)}")
        if bcs[0].shape[-1] != 1:
            raise ValueError(f"point barycentric coordinates expect last dimension 1, got {bcs[0].shape[-1]}")
        expected = bm.ones(bcs[0].shape, dtype=bcs[0].dtype)
        if not bm.allclose(bcs[0], expected):
            raise ValueError("point barycentric coordinates must be identically 1")
        return bm.einsum("...j,cjd->c...d", bcs[0], points[:, None, :])

    @classmethod
    def jacobi_matrix(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """Return the unit Jacobian for a zero-dimensional entity."""
        points = cls._indices(ctx, index)
        nq = int(bcs[0].shape[0])
        return bm.ones((points.shape[0], nq, 1, 1), dtype=ctx.block.positions.dtype)

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        point = cls._indices(ctx, index)
        dim = 1 if ref else cls.geo_dimension(ctx)
        grad = bm.zeros((point.shape[0], 1, dim), dtype=ctx.block.positions.dtype)
        if bcs is None:
            return grad
        bcs = _require_bcs_tuple(bcs, "point grad_lambda", 1)
        nq = int(bcs[0].shape[0])
        return bm.broadcast_to(grad[:, None, :, :], (point.shape[0], nq, 1, dim))

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported point quadrature type: {qtype!r}")
        if q < 1:
            raise ValueError(f"point quadrature order must be positive, got {q}")
        return NodeQuadrature()

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        point = cls._indices(ctx, index)
        return bm.ones((point.shape[0],), dtype=ctx.block.positions.dtype)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        point = cls._indices(ctx, index)
        gd = cls.geo_dimension(ctx)
        basis = bm.eye(gd, dtype=ctx.block.positions.dtype)
        return bm.broadcast_to(basis, (point.shape[0], gd, gd))

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        point = cls._indices(ctx, index)
        gd = cls.geo_dimension(ctx)
        return bm.zeros((point.shape[0], 0, gd), dtype=ctx.block.positions.dtype)

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        p = _require_order_tuple(order, "point multi_index", 1)[0]

        mi = bm.asarray([[p]], dtype=bm.int32)
        if tensorprod:
            from ...ipoints import multi_index_tensorprod
            return multi_index_tensorprod(mi)
        return mi
