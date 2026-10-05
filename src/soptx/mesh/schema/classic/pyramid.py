# 移植自 brighthe/fealpy ``fealpy/mesh/schema/classic/pyramid.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""参考四棱锥的线性 Lagrange Schema (只支持五节点几何)."""

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
    """不可变的线性 Lagrange 四棱锥 Schema.

    只支持五节点的几何次数 ``p=1``. 参考输入为 ``(bc_u, bc_v, bc_w)``, 几何参考梯度按
    ``(u, v, w)`` 顺序返回. 任意次 Lagrange 基函数 (包括 ``p=1`` 的请求) 有意不支持, 抛
    ``NotImplementedError``; 已认可的几何公式不升格为有限元参考基函数约定.
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
        """五节点几何基函数在三个方向乘积点上的值."""
        lu0, lu1, lv0, lv1, lw0, lw1 = cls._split_bcs(bcs)
        phi0 = cls._product(lu0, lv0, lw0)
        phi1 = cls._product(lu1, lv0, lw0)
        # 本 Schema 的底面使用循环顺序 (u0v0, u1v0, u1v1, u0v1).
        phi2 = cls._product(lu1, lv1, lw0)
        phi3 = cls._product(lu0, lv1, lw0)
        phi4 = cls._product(bm.ones_like(lu0), bm.ones_like(lv0), lw1)
        return bm.stack([phi0, phi1, phi2, phi3, phi4], axis=-1)

    @classmethod
    def geometry_grad_shape_function(cls, bcs: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """五节点几何基函数对 ``(u, v, w)`` 的参考梯度."""
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
        """把 ``(bc_u, bc_v, bc_w)`` 重心坐标映射为物理点."""
        bcs = _require_bcs_tuple(bcs, "pyramid bc_to_point", 3)
        pyramid = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[pyramid]
        phi = cls().shape_function(bcs)
        return bm.einsum("qi,cid->cqd", phi, points)

    def shape_function(self, bcs: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """在乘积点上计算五节点几何基函数.

        结果形状为 ``(Qu * Qv * Qw, 5)``, 列按完整的局部节点顺序.
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
        """不支持独立重心坐标梯度, 调用即报错."""
        raise NotImplementedError(
            "pyramid geometry does not expose independent barycentric gradients"
        )

    def grad_shape_function_reference(
        self,
        bcs: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """几何基函数的参考梯度, 形状 ``(Q, 5, 3)``, 末轴按公开的参考顺序 ``(u, v, w)``."""
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
        """不支持任意次四棱锥 Lagrange 基函数, 调用即报错."""
        raise NotImplementedError(
            "LagrangePyramidSchema does not yet support an arbitrary-order "
            "reference Lagrange basis"
        )

    def grad_lagrange_basis_function_barycentric(
        self,
        bcs: tuple[Tensor, Tensor, Tensor],
        p: int,
    ) -> Tensor:
        """不支持四棱锥基函数的重心坐标梯度, 调用即报错."""
        raise NotImplementedError(
            "LagrangePyramidSchema does not yet support arbitrary-order "
            "barycentric basis gradients"
        )

    def grad_lagrange_basis_function_reference(
        self,
        bcs: tuple[Tensor, Tensor, Tensor],
        p: int,
    ) -> Tensor:
        """不支持四棱锥基函数的参考梯度, 调用即报错."""
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
        """参考四棱锥到物理四棱锥映射的 Jacobi 矩阵, 形状 ``(NC, NQ, GD, 3)``."""
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
        """把参考梯度变换为物理梯度: ``J (J^T J)^{-1}`` 作用于参考梯度."""
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
        """参考坐标的物理梯度; 不给 ``bcs`` 时在参考中心求值."""
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
        """四棱锥上插值点的多重指标.

        Notes
        -----
        非内部情形借用五顶点单纯形的多重指标, 原代码即注明不正确.
        """
        p = _require_order_tuple(order, "pyramid multi_index", 1)[0]
        if internal:
            mi = _MI.multi_index_inner(p, 5)
        else:
            mi = _MI.multi_index_matrix(p, 5) # TODO: 不正确
        if tensorprod:
            return multi_index_tensorprod(mi)
        return mi

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        """参考四棱锥上的张量积 Gauss--Legendre 公式, ``w`` 方向至少 2 阶.

        Raises
        ------
        ValueError
            ``qtype`` 不是 ``"legendre"`` 或 None.
        """
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported pyramid quadrature type: {qtype!r}")
        from ....quadrature import GaussLegendreQuadrature, TensorProductQuadrature

        qf_uv = GaussLegendreQuadrature(q)
        qf_w = GaussLegendreQuadrature(max(q, 2))
        return TensorProductQuadrature((qf_uv, qf_uv, qf_w))

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """四棱锥五个顶点坐标的平均."""
        pyramid = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[pyramid]
        return bm.mean(points, axis=1)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """四棱锥体积: 用每方向 2 点的张量积 Gauss 公式对 Jacobi 行列式积分.

        几何映射是顶面收缩为一点的三线性映射, Jacobi 行列式在每个参考方向上至多 2 次,
        2 点 Gauss 公式即精确; 底面不必是平面或平行四边形.
        """
        qf = cls.quadrature_formula(2)
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = cls.jacobi_matrix(ctx, bcs, index)
        return bm.einsum("q,cq->c", ws, bm.abs(bm.linalg.det(J)))

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """法方向: 几何维数等于拓扑维数时返回空张量.

        Raises
        ------
        ValueError
            几何维数小于拓扑维数.
        """
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
        """以顶点 0 出发到顶点 1、2、4 的代表性切向量, 形状 ``(NC, 3, GD)``."""
        pyramid = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[pyramid]
        # 代表性的局部标架; 基于 Jacobi 矩阵的切向依赖参考点.
        return bm.stack([
            points[:, 1, :] - points[:, 0, :],
            points[:, 2, :] - points[:, 0, :],
            points[:, 4, :] - points[:, 0, :],
        ], axis=1)


PyramidSchema = LagrangePyramidSchema
