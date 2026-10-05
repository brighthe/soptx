# 移植自 brighthe/fealpy ``fealpy/mesh/schema/classic/hexahedron.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""参考六面体 (三个区间之积) 的 Lagrange Schema."""

from ....backend import bm
from ....backend import Index, Tensor
from ...ipoints import MultiIndex as _MI, multi_index_tensorprod
from ..entity_schema import _freeze_local_entities
from .base import (
    EntityContext,
    _TensorProductOrderSchema,
    _group_entity_definitions,
    _hexahedron_node_keys,
    _hexahedron_vertex_permutations,
    _require_bcs_tuple,
    _require_order_tuple,
)

__all__ = ["HexahedronSchema", "LagrangeHexahedronSchema"]


class LagrangeHexahedronSchema(_TensorProductOrderSchema):
    """不可变的三区间之积 Lagrange 六面体.

    标量 ``p`` 规范化为 ``(p, p, p)``, 元组表示 ``(px, py, pz)``. 参考输入与返回的梯度分量都按
    公开的 ``(x, y, z)`` 因子顺序, 尽管共享内核为保持既定的笛卡尔积点顺序而翻转了因子.
    """

    __slots__ = ()
    factor_count = 3
    type_id = "lagrange_hexahedron"
    schema_version = 1
    descriptor_parameter_names = ("p",)
    name = "hex"
    top_dim = 3
    _vertex_count = 8
    OFace = _freeze_local_entities({
        "quad": [
            [0, 3, 2, 1], [4, 5, 6, 7],
            [0, 1, 5, 4], [2, 3, 7, 6],
            [0, 4, 7, 3], [1, 2, 6, 5],
        ],
        "edge": [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [0, 4], [1, 5], [2, 6], [3, 7],
            [4, 5], [5, 6], [6, 7], [7, 4],
        ],
        "node": [[0], [1], [2], [3], [4], [5], [6], [7]],
    })
    SFace = _freeze_local_entities({
        "quad": [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 4, 5], [2, 3, 6, 7],
            [0, 3, 4, 7], [1, 2, 5, 6],
        ],
        "edge": [
            [0, 1], [1, 2], [2, 3], [0, 3],
            [0, 4], [1, 5], [2, 6], [3, 7],
            [4, 5], [5, 6], [6, 7], [4, 7],
        ],
        "node": [[0], [1], [2], [3], [4], [5], [6], [7]],
    })
    _tp_to_contract = (0, 1, 3, 2, 4, 5, 7, 6)

    def _raw_node_keys(self):
        return _hexahedron_node_keys(self.p)

    def _lagrange_kernel_factor_order(self) -> tuple[int, ...]:
        # 保持既定的积分点顺序: 依次为 z, y, x.
        return (2, 1, 0)

    def _edge_definitions(self):
        from .edge import LagrangeEdgeSchema

        px, py, pz = self.p
        x_schema = LagrangeEdgeSchema(px)
        y_schema = LagrangeEdgeSchema(py)
        z_schema = LagrangeEdgeSchema(pz)
        return (
            (x_schema, (0, 1)),
            (y_schema, (1, 2)),
            (x_schema, (2, 3)),
            (y_schema, (3, 0)),
            (z_schema, (0, 4)),
            (z_schema, (1, 5)),
            (z_schema, (2, 6)),
            (z_schema, (3, 7)),
            (x_schema, (4, 5)),
            (y_schema, (5, 6)),
            (x_schema, (6, 7)),
            (y_schema, (7, 4)),
        )

    def _face_definitions(self):
        from .quadrilateral import LagrangeQuadrilateralSchema

        px, py, pz = self.p
        return (
            (LagrangeQuadrilateralSchema((py, px)), (0, 3, 2, 1)),
            (LagrangeQuadrilateralSchema((px, py)), (4, 5, 6, 7)),
            (LagrangeQuadrilateralSchema((px, pz)), (0, 1, 5, 4)),
            (LagrangeQuadrilateralSchema((px, pz)), (2, 3, 7, 6)),
            (LagrangeQuadrilateralSchema((pz, py)), (0, 4, 7, 3)),
            (LagrangeQuadrilateralSchema((py, pz)), (1, 2, 6, 5)),
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
        return _hexahedron_vertex_permutations()

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """选定六面体顶点坐标的平均, 形状 ``(NC, GD)``."""
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        return bm.mean(ctx.block.positions[cell], axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """把三个方向的重心坐标映射为物理点; 连接按上下两层的循环顺序, 收缩前先换成张量积顺序."""
        bcs = _require_bcs_tuple(bcs, "hexahedron bc_to_point", 3)
        for bc in bcs:
            if bc.shape[-1] != 2:
                raise ValueError(
                    f"hexahedron barycentric coordinate tensors expect last dimension 2, got {bc.shape[-1]}"
                )

        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        # 连接按上下两层的循环顺序排列, 与张量积顺序不同.
        points = ctx.block.positions[cell[:, [0, 1, 3, 2, 4, 5, 7, 6]]]
        points = bm.reshape(points, (-1, 2, 2, 2, cls.geo_dimension(ctx)))
        u, v, w = bcs
        result = bm.einsum("ia,jb,kc,ncbae->nkjie", u, v, w, points)
        return bm.reshape(result, (result.shape[0], -1, result.shape[-1]))

    @classmethod
    def jacobi_matrix(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None,
    ) -> Tensor:
        """参考六面体到物理六面体映射的 Jacobi 矩阵, 形状 ``(NC, NQ, GD, 3)``."""
        bcs = _require_bcs_tuple(bcs, "hexahedron jacobi_matrix", 3)
        node = ctx.block.positions
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        gphi = cls().grad_shape_function_reference(bcs)
        return bm.einsum("cim,qin->cqmn", node[cell], gphi)

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        """六面体上插值点的多重指标 (三个区间多重指标的张量积); ``internal`` 为 True 时只取内部点."""
        order = _require_order_tuple(order, "hexahedron multi_index", 3)
        px, py, pz = order

        if internal:
            ix = _MI.multi_index_inner(px, 2)
            iy = _MI.multi_index_inner(py, 2)
            iz = _MI.multi_index_inner(pz, 2)
        else:
            ix = _MI.multi_index_matrix(px, 2)
            iy = _MI.multi_index_matrix(py, 2)
            iz = _MI.multi_index_matrix(pz, 2)
        shape = (iz.shape[0], iy.shape[0], ix.shape[0], 2)
        multi_index0 = bm.broadcast_to(ix[None, None, :, :], shape).reshape(-1, 2)
        multi_index1 = bm.broadcast_to(iy[None, :, None, :], shape).reshape(-1, 2)
        multi_index2 = bm.broadcast_to(iz[:, None, None, :], shape).reshape(-1, 2)
        mi = bm.concat([multi_index0, multi_index1, multi_index2], axis=-1)
        if tensorprod:
            return multi_index_tensorprod(mi, (2, 4))
        return mi

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        """三个方向局部坐标的梯度; 不给 ``bcs`` 时在单元中心求值."""
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        if bcs is None:
            bcs = (
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
            )
            squeeze_q = True
        else:
            bcs = _require_bcs_tuple(bcs, "hexahedron grad_lambda", 3)
            squeeze_q = False
        u, v, w = bcs
        u0, u1 = u[:, 0], u[:, 1]
        v0, v1 = v[:, 0], v[:, 1]
        w0, w1 = w[:, 0], w[:, 1]
        z = bm.zeros_like(u0)

        def ref_row(ua, ub, va, vb, wa, wb):
            """把六个分量拼成参考梯度的一行."""
            return bm.stack([ua, ub, va, vb, wa, wb], axis=-1)

        ref_grad = bm.stack([
            ref_row(v0*w0, z, u0*w0, z, u0*v0, z),
            ref_row(z, v0*w0, u1*w0, z, u1*v0, z),
            ref_row(v1*w0, z, z, u0*w0, u0*v1, z),
            ref_row(z, v1*w0, z, u1*w0, u1*v1, z),
            ref_row(v0*w1, z, u0*w1, z, z, u0*v0),
            ref_row(z, v0*w1, u1*w1, z, z, u1*v0),
            ref_row(v1*w1, z, z, u0*w1, z, u0*v1),
            ref_row(z, v1*w1, z, u1*w1, z, u1*v1),
        ], axis=1)
        if ref:
            ref_grad = ref_grad[..., cls._tp_to_contract, :]
            grad = bm.broadcast_to(ref_grad[None, :, :, :], (cell.shape[0], ref_grad.shape[0], 8, 6))
            return grad[:, 0, :, :] if squeeze_q else grad

        dphi = bm.stack([
            bm.stack([-v0*w0, -u0*w0, -u0*v0], axis=-1),
            bm.stack([ v0*w0, -u1*w0, -u1*v0], axis=-1),
            bm.stack([-v1*w0,  u0*w0, -u0*v1], axis=-1),
            bm.stack([ v1*w0,  u1*w0, -u1*v1], axis=-1),
            bm.stack([-v0*w1, -u0*w1,  u0*v0], axis=-1),
            bm.stack([ v0*w1, -u1*w1,  u1*v0], axis=-1),
            bm.stack([-v1*w1,  u0*w1,  u0*v1], axis=-1),
            bm.stack([ v1*w1,  u1*w1,  u1*v1], axis=-1),
        ], axis=1)
        dphi = dphi[:, cls._tp_to_contract, :]
        points = ctx.block.positions[cell]
        J = bm.einsum("qit,cid->cqtd", dphi, points)
        Jt = bm.einsum("cqtd->cqdt", J)
        metric = bm.einsum("cqtd,cqsd->cqts", J, J)
        metric_inv = bm.linalg.inv(metric)
        grad = bm.einsum("cqdt,cqts,qis->cqid", Jt, metric_inv, dphi)
        return grad[:, 0, :, :] if squeeze_q else grad

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """六面体体积, 用每方向 2 点的张量积 Gauss 公式对 Jacobi 行列式积分.

        Raises
        ------
        ValueError
            几何维数不是 3.
        """
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"hexahedron geometry requires GD == 3, got {gd}")

        qf = cls.quadrature_formula(2)
        bcs, ws = qf.get_quadrature_points_and_weights()
        u, v, w = bcs
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        points = ctx.block.positions[cell[:, cls._tp_to_contract]]
        points = bm.reshape(points, (-1, 2, 2, 2, 3))
        du = bm.broadcast_to(bm.asarray([-1.0, 1.0], dtype=ctx.block.positions.dtype)[None, :], u.shape)
        dv = bm.broadcast_to(bm.asarray([-1.0, 1.0], dtype=ctx.block.positions.dtype)[None, :], v.shape)
        dw = bm.broadcast_to(bm.asarray([-1.0, 1.0], dtype=ctx.block.positions.dtype)[None, :], w.shape)

        ju = bm.einsum("ia,jb,kc,ncbae->nijke", du, v, w, points)
        jv = bm.einsum("ia,jb,kc,ncbae->nijke", u, dv, w, points)
        jw = bm.einsum("ia,jb,kc,ncbae->nijke", u, v, dw, points)
        jac = bm.stack([ju, jv, jw], axis=-1)
        det = bm.abs(bm.linalg.det(jac))
        weight = bm.reshape(ws, (2, 2, 2))
        return bm.einsum("ijk,nijk->n", weight, det)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """三维体单元没有法方向, 返回 ``(NC, 0, 3)`` 的空张量.

        Raises
        ------
        ValueError
            几何维数不是 3.
        """
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"hexahedron geometry requires GD == 3, got {gd}")
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        return bm.zeros((cell.shape[0], 0, 3), dtype=ctx.block.positions.dtype)

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        """参考六面体上三个方向同阶的张量积 Gauss--Legendre 公式.

        Raises
        ------
        ValueError
            ``qtype`` 不是 ``"legendre"`` 或 None.
        """
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported hexahedron quadrature type: {qtype!r}")
        from ....quadrature import GaussLegendreQuadrature, TensorProductQuadrature

        qf = GaussLegendreQuadrature(q, device=device)
        return TensorProductQuadrature((qf, qf, qf))

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """以顶点 0 出发的三条棱作为切向量, 形状 ``(NC, 3, 3)``.

        Raises
        ------
        ValueError
            几何维数不是 3.
        """
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"hexahedron geometry requires GD == 3, got {gd}")
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        points = ctx.block.positions[cell]
        t0 = points[:, 1, :] - points[:, 0, :]
        t1 = points[:, 3, :] - points[:, 0, :]
        t2 = points[:, 4, :] - points[:, 0, :]
        return bm.stack([t0, t1, t2], axis=1)


HexahedronSchema = LagrangeHexahedronSchema
