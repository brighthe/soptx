# 移植自 brighthe/fealpy ``fealpy/mesh/schema/classic/prism.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""参考三棱柱 (三角形乘区间) 的 Lagrange Schema."""

from fractions import Fraction

from ....backend import bm
from ....backend import Index, Tensor
from ...ipoints import MultiIndex as _MI, multi_index_tensorprod
from ..entity_schema import _freeze_local_entities
from .base import (
    EntityContext,
    _TensorProductOrderSchema,
    _group_entity_definitions,
    _prism_node_keys,
    _simplex_node_keys,
    _prism_vertex_permutations,
    _require_bcs_tuple,
    _require_order_tuple,
)

__all__ = ["LagrangePrismSchema", "PrismSchema"]


class LagrangePrismSchema(_TensorProductOrderSchema):
    """不可变的三角形乘区间 Lagrange 三棱柱.

    标量 ``p`` 对两个因子重复使用, 元组表示 ``(三角形次数, 区间次数)``. 参考输入为
    ``(triangle_bcs, interval_bcs)``, 梯度分量保持这一公开的因子顺序. 混合的三角形面与
    四边形面以各自的 :class:`~soptx.mesh.schema.LocalEntityGroup` 给出.
    """

    __slots__ = ()
    factor_count = 2
    type_id = "lagrange_prism"
    schema_version = 1
    descriptor_parameter_names = ("p",)
    name = "prism"
    top_dim = 3
    _vertex_count = 6
    OFace = _freeze_local_entities({
        "tri": [[0, 2, 1], [3, 4, 5]],
        "quad": [[0, 1, 4, 3], [1, 2, 5, 4], [0, 3, 5, 2]],
        "edge": [
            [0, 1], [1, 2], [0, 2],
            [0, 3], [1, 4], [2, 5],
            [3, 4], [4, 5], [3, 5],
        ],
        "node": [[0], [1], [2], [3], [4], [5]],
    })
    SFace = _freeze_local_entities({
        "tri": [[0, 1, 2], [3, 4, 5]],
        "quad": [[0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5]],
        "edge": [
            [0, 1], [1, 2], [0, 2],
            [0, 3], [1, 4], [2, 5],
            [3, 4], [4, 5], [3, 5],
        ],
        "node": [[0], [1], [2], [3], [4], [5]],
    })
    ref_measure = 0.5

    def _raw_node_keys(self):
        return _prism_node_keys(self.p)

    def _lagrange_kernel_node_keys(
        self,
    ) -> tuple[tuple[Fraction, ...], ...]:
        triangle_order, interval_order = self.p
        result: list[tuple[Fraction, ...]] = []
        for a, b, c in _simplex_node_keys(triangle_order, 3):
            for iz in range(interval_order + 1):
                z = Fraction(iz, interval_order)
                result.append(
                    (
                        a * (1 - z),
                        b * (1 - z),
                        c * (1 - z),
                        a * z,
                        b * z,
                        c * z,
                    )
                )
        return tuple(result)

    def _edge_definitions(self):
        from .edge import LagrangeEdgeSchema

        triangle_order, interval_order = self.p
        triangle_schema = LagrangeEdgeSchema(triangle_order)
        interval_schema = LagrangeEdgeSchema(interval_order)
        return (
            (triangle_schema, (0, 1)),
            (triangle_schema, (1, 2)),
            (triangle_schema, (2, 0)),
            (triangle_schema, (3, 4)),
            (triangle_schema, (4, 5)),
            (triangle_schema, (5, 3)),
            (interval_schema, (0, 3)),
            (interval_schema, (1, 4)),
            (interval_schema, (2, 5)),
        )

    def _face_definitions(self):
        from .quadrilateral import LagrangeQuadrilateralSchema
        from .triangle import LagrangeTriangleSchema

        triangle_order, interval_order = self.p
        triangle_schema = LagrangeTriangleSchema(triangle_order)
        return (
            (triangle_schema, (0, 2, 1)),
            (triangle_schema, (3, 4, 5)),
            (
                LagrangeQuadrilateralSchema((triangle_order, interval_order)),
                (0, 1, 4, 3),
            ),
            (
                LagrangeQuadrilateralSchema((triangle_order, interval_order)),
                (1, 2, 5, 4),
            ),
            (
                LagrangeQuadrilateralSchema((interval_order, triangle_order)),
                (0, 3, 5, 2),
            ),
        )

    def _local_edge_definitions(self):
        from .edge import LagrangeEdgeSchema

        triangle_order, interval_order = self.p
        edges = self.OFace["edge"]
        return (
            (
                LagrangeEdgeSchema(triangle_order),
                tuple(edges[index] for index in (0, 1, 2, 6, 7, 8)),
            ),
            (
                LagrangeEdgeSchema(interval_order),
                tuple(edges[index] for index in (3, 4, 5)),
            ),
        )

    def _layout_entity_definitions(self):
        definitions = self._edge_definitions() + self._face_definitions()
        return tuple((schema, (vertices,)) for schema, vertices in definitions)

    def _local_entity_definitions(self, top_dim: int):
        if top_dim == 1:
            return self._local_edge_definitions()
        if top_dim == 2:
            return _group_entity_definitions(self._face_definitions())
        return ()

    def _candidate_vertex_permutations(self):
        return _prism_vertex_permutations()

    @classmethod
    def _entity(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        entity = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(entity.shape) == 1:
            entity = entity[None, :]
        return entity

    @classmethod
    def _points(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        prism = cls._entity(ctx, index)
        return ctx.block.positions[prism]

    @classmethod
    def _tp_points(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        prism = cls._entity(ctx, index)
        # ``shape_function`` 把六个节点排为底层的三角形顶点, 接着顶层对应的顶点;
        # 这也是本 Schema 的约定顺序.
        return ctx.block.positions[prism]

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """三棱柱的重心 (顶点平均), 形状 ``(NC, GD)``."""
        points = cls._points(ctx, index)
        return bm.mean(points, axis=1)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """三棱柱的体积.

        ``∫_K dx = ∫_{K̂} sqrt(det(G)) dξ``, 其中 ``G = J^T J``.
        """
        qf = cls.quadrature_formula(2)
        bcs, ws = qf.get_quadrature_points_and_weights()
        G = cls.first_fundamental_form(ctx, bcs, index=index)
        l = bm.sqrt(bm.linalg.det(G))
        return 0.5 * bm.einsum("q,cq->c", ws, l)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """三维中三棱柱体单元没有法方向, 返回空张量."""
        prism = cls._entity(ctx, index)
        GD = cls.geo_dimension(ctx)
        return bm.zeros((prism.shape[0], 0, GD), dtype=ctx.block.positions.dtype)

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """三棱柱的切方向: 以顶点 0 出发到顶点 1、2、3 的向量, 形状 ``(NC, 3, GD)``."""
        points = cls._points(ctx, index)
        return points[:, [1, 2, 3], :] - points[:, [0], :]

    # 积分
    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        """参考三棱柱上三角形公式与 Gauss--Legendre 公式的张量积."""
        from ....quadrature import (
            GaussLegendreQuadrature,
            TensorProductQuadrature,
            TriangleQuadrature,
        )

        qf0 = TriangleQuadrature(q, device=device)
        qf1 = GaussLegendreQuadrature(q, device=device)
        return TensorProductQuadrature((qf0, qf1))

    @classmethod
    def jacobi_matrix(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None
    ) -> Tensor:
        """参考三棱柱到物理三棱柱映射的 Jacobi 矩阵, 形状 ``(NC, NQ, GD, 3)``."""
        bcs = _require_bcs_tuple(bcs, "prism jacobi_matrix", 2)

        points = cls._tp_points(ctx, index)
        gphi = cls().grad_shape_function_reference(bcs)

        return bm.einsum("cim,qin->cqmn", points, gphi)

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None = None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        """参考坐标的物理梯度; 不给 ``bcs`` 时在单元中心求值."""
        prism = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(prism.shape) == 1:
            prism = bm.reshape(prism, (1, -1))
        if bcs is None:
            bcs = (
                bm.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=ctx.block.positions.dtype),
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
            )
            squeeze_q = True
        else:
            bcs = _require_bcs_tuple(bcs, "prism grad_lambda", 2)
            squeeze_q = False
        ref3 = cls().grad_shape_function_reference(bcs)
        z3 = bm.zeros((ref3.shape[0], ref3.shape[1], 2), dtype=ref3.dtype)
        ref5 = bm.concatenate([ref3[:, :, 0:2], z3[:, :, 0:1], ref3[:, :, 2:3], z3[:, :, 1:2]], axis=-1)
        if ref:
            grad = bm.broadcast_to(ref5[None, :, :, :], (prism.shape[0], ref5.shape[0], 6, 5))
            return grad[:, 0, :, :] if squeeze_q else grad
        G, J = cls.first_fundamental_form(ctx, bcs, index=index, return_jacobi=True)
        Ginv = bm.linalg.inv(G)
        grad = bm.einsum("cqdk,cqkl,qil->cqid", J, Ginv, ref3)
        return grad[:, 0, :, :] if squeeze_q else grad

    # 插值点
    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        """参考三棱柱上的多重指标: 三角形与区间多重指标的张量积."""
        p0, p1 = _require_order_tuple(order, "prism multi_index", 2)

        if internal:
            mi0 = _MI.multi_index_inner(p0, 3)
            mi1 = _MI.multi_index_inner(p1, 2)
        else:
            mi0 = _MI.multi_index_matrix(p0, 3)
            mi1 = _MI.multi_index_matrix(p1, 2)

        shape = (mi1.shape[0], mi0.shape[0])
        mi0 = bm.broadcast_to(mi0[None, :, :], shape + (3,))
        mi1 = bm.broadcast_to(mi1[:, None, :], shape + (2,))

        mi = bm.concat([mi0, mi1], axis=-1).reshape(-1, 5)
        if tensorprod:
            return multi_index_tensorprod(mi, (3,))
        return mi

    @classmethod
    def bc_to_point(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None
    ) -> Tensor:
        """把重心坐标转换为直角坐标: 物理三棱柱上 ``x = sum_i phi_i x_i``."""
        phi = cls().shape_function(bcs)
        points = cls._tp_points(ctx, index)
        return bm.einsum("cim,qi->cqm", points, phi)

    @classmethod
    def first_fundamental_form(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None = None,
        return_jacobi: bool = False,
        return_grad: bool = False
    ):
        """Lagrange 三棱柱的第一基本形式 ``G = J^T J``, ``J`` 为参考到物理映射的 Jacobi 矩阵."""
        J = cls.jacobi_matrix(ctx, bcs, index=index)
        gphi = cls().grad_shape_function_reference(bcs)
        TD = J.shape[-1]
        shape = J.shape[0:-2] + (TD, TD)
        data = [[0 for _ in range(TD)] for _ in range(TD)]

        for i in range(TD):
            data[i][i] = bm.einsum("...d,...d->...", J[..., i], J[..., i])
            for j in range(i + 1, TD):
                data[i][j] = bm.einsum("...d,...d->...", J[..., i], J[..., j])
                data[j][i] = data[i][j]

        data = [val.reshape(val.shape + (1,)) for row in data for val in row]
        G = bm.concatenate(data, axis=-1).reshape(shape)

        if not return_jacobi and not return_grad:
            return G
        if return_jacobi and not return_grad:
            return G, J
        if not return_jacobi and return_grad:
            return G, gphi
        return G, J, gphi


PrismSchema = LagrangePrismSchema
