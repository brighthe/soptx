# 移植自 brighthe/fealpy ``fealpy/mesh/schema/classic/triangle.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""参考三角形的 Lagrange Schema."""

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

__all__ = ["LagrangeTriangleSchema", "TriangleSchema"]


class LagrangeTriangleSchema(_ScalarOrderSchema):
    """几何次数为 ``p`` 的不可变 Lagrange 三角形.

    ``p`` 为正整数. 参考输入按重心坐标顺序 ``(lambda_0, lambda_1, lambda_2)``; 参考导数取
    ``(lambda_1, lambda_2)``, ``lambda_0 = 1 - lambda_1 - lambda_2``.
    """

    __slots__ = ()
    type_id = "lagrange_triangle"
    schema_version = 1
    descriptor_parameter_names = ("p",)
    name = "tri"
    top_dim = 2
    _vertex_count = 3
    OFace = _freeze_local_entities({
        "edge": [[1, 2], [2, 0], [0, 1]],
        "node": [[0], [1], [2]],
    })
    SFace = _freeze_local_entities({
        "edge": [[1, 2], [0, 2], [0, 1]],
        "node": [[0], [1], [2]],
    })
    orientation = (
        (0, 1, 2), (1, 2, 0), (2, 0, 1),
        (0, 2, 1), (2, 1, 0), (1, 0, 2),
    )
    ref_measure = 0.5

    def _raw_node_keys(self):
        return _simplex_node_keys(self.p, self._vertex_count)

    def _layout_entity_definitions(self):
        from .edge import LagrangeEdgeSchema

        return (
            (
                LagrangeEdgeSchema(self.p),
                ((0, 1), (1, 2), (2, 0)),
            ),
        )

    def _local_entity_definitions(self, top_dim: int):
        if top_dim != 1:
            return ()
        from .edge import LagrangeEdgeSchema

        return ((LagrangeEdgeSchema(self.p), self.OFace["edge"]),)

    def _candidate_vertex_permutations(self):
        return _simplex_vertex_permutations(self._vertex_count)

    @classmethod
    def _selected_triangles(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tri.shape) == 1:
            tri = bm.reshape(tri, (1, -1))
        return tri

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        """三角形上插值点的多重指标, 每个顶点一列; ``internal`` 为 True 时只取内部点, ``tensorprod`` 为 True 时转为插值点工具所用的张量积兼容顺序."""
        p = _require_order_tuple(order, "triangle multi_index", 1)[0]
        if internal:
            mi = _MI.multi_index_inner(p, 3)
        else:
            mi = _MI.multi_index_matrix(p, 3)
        if tensorprod:
            return multi_index_tensorprod(mi)
        return mi

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """选定三角形顶点坐标的平均, 形状 ``(NC, GD)``."""
        tri = cls._selected_triangles(ctx, index)
        points = ctx.block.positions[tri]
        return bm.mean(points, axis=1)

    @classmethod
    def jacobi_matrix(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None,
    ) -> Tensor:
        """参考三角形到物理三角形映射的 Jacobi 矩阵, 形状 ``(NC, NQ, GD, TD)``."""
        bcs = _require_bcs_tuple(bcs, "triangle jacobi_matrix", 1)
        tri = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tri.shape) == 1:
            tri = bm.reshape(tri, (1, -1))

        gphi = cls().grad_shape_function_reference(bcs)
        return bm.einsum("cim,qin->cqmn", ctx.block.positions[tri], gphi)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """把三角形上的重心坐标映射为物理点, 形状 ``(NC, NQ, GD)``."""
        bcs = _require_bcs_tuple(bcs, "triangle bc_to_point", 1)
        bc = bcs[0]
        tri = cls._selected_triangles(ctx, index)
        points = ctx.block.positions[tri]
        return bm.einsum("...j,cjd->c...d", bc, points)

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        """三个重心坐标的梯度: ``ref`` 为 True 时为单位阵 ``(NC, 3, 3)``, 否则为物理梯度 ``(NC, 3, GD)``; 给出 ``bcs`` 时沿积分点轴广播."""
        tri = cls._selected_triangles(ctx, index)
        node = ctx.block.positions
        if ref:
            grad = bm.broadcast_to(
                bm.eye(3, dtype=node.dtype)[None, :, :],
                (tri.shape[0], 3, 3),
            )
        else:
            gd = int(node.shape[1])
            if gd == 2:
                grad = bm.triangle_grad_lambda_2d(tri, node)
            elif gd == 3:
                grad = bm.triangle_grad_lambda_3d(tri, node)
            else:
                raise ValueError(f"unsupported geometric dimension: {gd}")
        if bcs is None:
            return grad
        bcs = _require_bcs_tuple(bcs, "triangle grad_lambda", 1)
        nq = int(bcs[0].shape[0])
        return bm.broadcast_to(grad[:, None, :, :], (tri.shape[0], nq, grad.shape[1], grad.shape[2]))

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        """参考三角形上的积分公式: ``q <= 9`` 用对称三角形公式, 更高阶用 Stroud 锥积公式.

        Raises
        ------
        ValueError
            ``qtype`` 不是 ``"legendre"``.
        """
        if qtype != "legendre":
            raise ValueError(f"unsupported quadrature type: {qtype}")
        if q > 9:
            from ....quadrature.stroud_quadrature import StroudQuadrature
            return StroudQuadrature(2, q)
        from ....quadrature import TriangleQuadrature
        return TriangleQuadrature(q, device=device)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """三角形面积: 二维为有向测度, 三维为叉积模长的一半."""
        tri = cls._selected_triangles(ctx, index)
        node = ctx.block.positions
        gd = int(node.shape[1])
        if gd == 2:
            return bm.simplex_measure(tri, node)
        if gd == 3:
            points = node[tri]
            v1 = points[:, 1, :] - points[:, 0, :]
            v2 = points[:, 2, :] - points[:, 0, :]
            normal = bm.cross(v1, v2)
            return bm.linalg.vector_norm(normal, axis=1) * 0.5
        raise ValueError(f"unsupported geometric dimension: {gd}")

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """二维时没有法方向, 返回 ``(NC, 0, 2)`` 的空张量; 三维时为两条边的叉积方向."""
        tri = cls._selected_triangles(ctx, index)
        points = ctx.block.positions[tri]
        gd = int(points.shape[2])
        if gd == 2:
            return bm.zeros((points.shape[0], 0, gd), **bm.context(points))
        if gd == 3:
            v1 = points[:, 1, :] - points[:, 0, :]
            v2 = points[:, 2, :] - points[:, 0, :]
            normal = bm.cross(v1, v2)
            return bm.expand_dims(normal, axis=1)
        raise ValueError(f"unsupported geometric dimension: {gd}")

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """以顶点 0 出发的两条边 ``(v1 - v0, v2 - v0)`` 作为切向量, 形状 ``(NC, 2, GD)``."""
        tri = cls._selected_triangles(ctx, index)
        points = ctx.block.positions[tri]
        t0 = points[:, 1, :] - points[:, 0, :]
        t1 = points[:, 2, :] - points[:, 0, :]
        return bm.stack([t0, t1], axis=1)


TriangleSchema = LagrangeTriangleSchema
