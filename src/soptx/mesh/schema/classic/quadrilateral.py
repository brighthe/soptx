# 移植自 brighthe/fealpy ``fealpy/mesh/schema/classic/quadrilateral.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""参考四边形 (区间乘区间) 的 Lagrange Schema."""

from ....backend import bm
from ....backend import Index, Tensor
from ...ipoints import MultiIndex as _MI, multi_index_tensorprod
from ..entity_schema import _freeze_local_entities
from .base import (
    EntityContext,
    _TensorProductOrderSchema,
    _group_entity_definitions,
    _quadrilateral_node_keys,
    _quadrilateral_vertex_permutations,
    _require_bcs_tuple,
    _require_order_tuple,
)

__all__ = ["LagrangeQuadrilateralSchema", "QuadrilateralSchema"]


class LagrangeQuadrilateralSchema(_TensorProductOrderSchema):
    """不可变的区间乘区间 Lagrange 四边形.

    标量 ``p`` 规范化为 ``(p, p)``, 元组表示 ``(px, py)``. 参考输入为 ``(bc_x, bc_y)``. 为笛卡尔
    积点排序所做的内核因子翻转在基函数与梯度的轴上不可见, 它们总按公开的 ``(x, y)``
    顺序返回.

    公开的连接使用循环顶点顺序 (左下, 右下, 右上, 左上); 张量积顺序只用于参考内核, 在进入
    网格连接之前即已转换.
    """

    __slots__ = ()
    factor_count = 2
    type_id = "lagrange_quadrilateral"
    schema_version = 1
    descriptor_parameter_names = ("p",)
    name = "quad"
    top_dim = 2
    _vertex_count = 4
    OFace = _freeze_local_entities({
        "edge": [[0, 1], [1, 2], [2, 3], [3, 0]],
        "node": [[0], [1], [2], [3]],
    })
    SFace = _freeze_local_entities({
        "edge": [[0, 1], [1, 2], [2, 3], [0, 3]],
        "node": [[0], [1], [2], [3]],
    })
    # 旧式插值点元数据置换的是张量因子的重心坐标列;
    # 公开的拓扑定向来自 vertex_permutations().
    orientation = (
        (0, 1, 2, 3), (2, 0, 3, 1), (3, 2, 1, 0), (1, 3, 0, 2),
        (2, 3, 0, 1), (0, 2, 1, 3), (1, 0, 3, 2), (3, 1, 2, 0),
    )
    ccw = (0, 1, 2, 3)
    _tp_to_contract = (0, 1, 3, 2)

    def _raw_node_keys(self):
        return _quadrilateral_node_keys(self.p)

    def _lagrange_kernel_factor_order(self) -> tuple[int, ...]:
        # 保持既定的积分点顺序: y 变化慢, x 变化快.
        return (1, 0)

    def _edge_definitions(self):
        from .edge import LagrangeEdgeSchema

        px, py = self.p
        return (
            (LagrangeEdgeSchema(px), (0, 1)),
            (LagrangeEdgeSchema(py), (1, 2)),
            (LagrangeEdgeSchema(px), (2, 3)),
            (LagrangeEdgeSchema(py), (3, 0)),
        )

    def _layout_entity_definitions(self):
        return tuple(
            (schema, (vertices,))
            for schema, vertices in self._edge_definitions()
        )

    def _local_entity_definitions(self, top_dim: int):
        if top_dim != 1:
            return ()
        return _group_entity_definitions(self._edge_definitions())

    def _candidate_vertex_permutations(self):
        return _quadrilateral_vertex_permutations()

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        """四边形上插值点的多重指标, 每个顶点一列; ``internal`` 为 True 时只取内部点, ``tensorprod`` 为 True 时转为插值点工具所用的张量积兼容顺序."""
        px, py = _require_order_tuple(order, "quadrilateral multi_index", 2)

        if internal:
            ix = _MI.multi_index_inner(px, 2)
            iy = _MI.multi_index_inner(py, 2)
        else:
            ix = _MI.multi_index_matrix(px, 2)
            iy = _MI.multi_index_matrix(py, 2)

        shape = (iy.shape[0], ix.shape[0], 2)
        multi_index0 = bm.broadcast_to(ix[None, :, :], shape).reshape(-1, 2)
        multi_index1 = bm.broadcast_to(iy[:, None, :], shape).reshape(-1, 2)
        mi = bm.concat([multi_index0, multi_index1], axis=1)
        if tensorprod:
            return multi_index_tensorprod(mi, (2,))
        return mi

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """选定四边形顶点坐标的平均, 形状 ``(NC, GD)``."""
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad]
        return bm.mean(points, axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """把四边形上的重心坐标映射为物理点, 形状 ``(NC, NQ, GD)``."""
        bcs = _require_bcs_tuple(bcs, "quadrilateral bc_to_point", 2)
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad[:, [0, 1, 3, 2]]]
        bc0 = bcs[0].reshape(-1, 2)
        bc1 = bcs[1].reshape(-1, 2)
        bc = bm.einsum("im,jn->ijmn", bc1, bc0).reshape(-1, 4)
        return bm.einsum("qj,cjd->cqd", bc, points)

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        """参考四边形上两个方向同阶的张量积 Gauss--Legendre 公式."""
        from ....quadrature import GaussLegendreQuadrature, TensorProductQuadrature
        qf = GaussLegendreQuadrature(q, device=device)
        return TensorProductQuadrature((qf, qf))

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        """两个方向局部坐标的梯度; 不给 ``bcs`` 时用以中点差分构造的平均 Jacobi 矩阵计算."""
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(quad.shape) == 1:
            quad = bm.reshape(quad, (1, -1))
        if bcs is None:
            if not ref:
                points = ctx.block.positions[quad]
                vr = 0.5 * ((points[:, 1, :] - points[:, 0, :]) + (points[:, 2, :] - points[:, 3, :]))
                vs = 0.5 * ((points[:, 3, :] - points[:, 0, :]) + (points[:, 2, :] - points[:, 1, :]))
                jac = bm.stack([vr, vs], axis=-1)
                jac_t = bm.einsum("nij->nji", jac)
                metric = bm.einsum("nik,nkj->nij", jac_t, jac)
                metric_inv = bm.linalg.inv(metric)
                grads = bm.einsum("nik,nkj->nij", metric_inv, jac_t)
                ref_grads = bm.asarray([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=points.dtype)
                return bm.einsum("ld, ndg->nlg", ref_grads, grads)
            bcs = (
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
            )
            squeeze_q = True
        else:
            bcs = _require_bcs_tuple(bcs, "quadrilateral grad_lambda", 2)
            squeeze_q = False
        u, v = bcs
        u0, u1 = u[:, 0], u[:, 1]
        v0 = bm.broadcast_to(v[:, 0], u0.shape)
        v1 = bm.broadcast_to(v[:, 1], u0.shape)
        z_u = bm.zeros_like(u0)
        z_v = bm.zeros_like(v0)
        ref_u = bm.stack([
            bm.stack([v0, z_u, u0, z_u], axis=-1),
            bm.stack([z_u, v0, u1, z_u], axis=-1),
            bm.stack([v1, z_u, z_u, u0], axis=-1),
            bm.stack([z_u, v1, z_u, u1], axis=-1),
        ], axis=1)
        if ref:
            grad = bm.broadcast_to(ref_u[None, :, :, :], (quad.shape[0], ref_u.shape[0], 4, 4))
            return grad[:, 0, :, :] if squeeze_q else grad

        dphi_duv = bm.stack([
            bm.stack([-v0, -u0], axis=-1),
            bm.stack([ v0, -u1], axis=-1),
            bm.stack([-v1,  u0], axis=-1),
            bm.stack([ v1,  u1], axis=-1),
        ], axis=1)
        points = ctx.block.positions[quad]
        J = bm.einsum("qit,cid->cqtd", dphi_duv, points)
        Jt = bm.einsum("cqtd->cqdt", J)
        metric = bm.einsum("cqtd,cqsd->cqts", J, J)
        metric_inv = bm.linalg.inv(metric)
        grad = bm.einsum("cqdt,cqts,qis->cqid", Jt, metric_inv, dphi_duv)
        return grad[:, 0, :, :] if squeeze_q else grad

    @classmethod
    def jacobi_matrix(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None,
    ) -> Tensor:
        """参考四边形到物理四边形映射的 Jacobi 矩阵, 形状 ``(NC, NQ, GD, TD)``."""
        bcs = _require_bcs_tuple(bcs, "quadrilateral jacobi_matrix", 2)
        node = ctx.block.positions
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        # (NQ, num_shape, ref_dim)
        gphi = cls().grad_shape_function_reference(bcs)
        J = bm.einsum('cim, qin -> cqmn', node[cell], gphi) # (NC, NQ, GD, ref_dim)

        return J

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """四边形面积: 沿对角线 (0, 2) 分成两个三角形求和."""
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad]
        v0 = points[:, 1, :] - points[:, 0, :]
        v1 = points[:, 2, :] - points[:, 0, :]
        v2 = points[:, 3, :] - points[:, 0, :]
        if points.shape[-1] == 2:
            cross01 = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
            cross12 = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            return 0.5 * (bm.abs(cross01) + bm.abs(cross12))
        cross0 = bm.cross(v0, v1)
        cross1 = bm.cross(v1, v2)
        area0 = bm.sqrt(bm.sum(cross0 * cross0, axis=1))
        area1 = bm.sqrt(bm.sum(cross1 * cross1, axis=1))
        return 0.5 * (area0 + area1)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """二维时没有法方向, 返回 ``(NC, 0, 2)`` 的空张量; 三维时由边向量的叉积构造."""
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad]
        gd = int(points.shape[2])
        if gd == 2:
            return bm.zeros((points.shape[0], 0, gd), **bm.context(points))
        if gd == 3:
            v1 = points[:, 1, :] - points[:, 0, :]
            v2 = points[:, 2, :] - points[:, 0, :]
            v3 = points[:, 3, :] - points[:, 0, :]
            normal = bm.cross(v1, v2) + bm.cross(v2, v3)
            return bm.expand_dims(normal, axis=1)
        raise ValueError(f"unsupported geometric dimension: {gd}")

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """两个方向的平均边向量 ``(vr, vs)`` 作为切向量, 形状 ``(NC, 2, GD)``."""
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad]
        vr = 0.5 * ((points[:, 1, :] - points[:, 0, :]) + (points[:, 2, :] - points[:, 3, :]))
        vs = 0.5 * ((points[:, 3, :] - points[:, 0, :]) + (points[:, 2, :] - points[:, 1, :]))
        return bm.stack([vr, vs], axis=1)


QuadrilateralSchema = LagrangeQuadrilateralSchema
