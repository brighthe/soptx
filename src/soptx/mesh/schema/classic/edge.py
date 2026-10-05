# 移植自 brighthe/fealpy ``fealpy/mesh/schema/classic/edge.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""参考区间 (边) 的 Lagrange Schema."""

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
    """几何次数为 ``p`` 的不可变 Lagrange 边.

    ``p`` 为正整数. 参考输入按重心坐标顺序 ``(lambda_0, lambda_1)``, 正次数基函数的列遵循
    拓扑优先的完整节点布局.
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
        """返回选定边的连接数组, 保留显式的实体轴.

        Parameters
        ----------
        ctx : EntityContext
            网格块与边分区.
        index : Index or None
            选定的边, None 表示全部边.

        Returns
        -------
        Tensor
            边的连接张量, 形状 ``(NC, 2)``.
        """
        edge = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(edge.shape) == 1:
            edge = bm.reshape(edge, (1, -1))
        return edge

    @classmethod
    def _points(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """选定边的端点坐标.

        Parameters
        ----------
        ctx : EntityContext
            网格块与边分区.
        index : Index or None
            选定的边, None 表示全部边.

        Returns
        -------
        Tensor
            端点坐标, 形状 ``(NC, 2, GD)``.
        """
        return ctx.block.positions[cls._entity(ctx, index)]

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        """参考边上插值点的多重指标.

        Parameters
        ----------
        order : tuple of int
            参考边上的多项式次数.
        internal : bool, optional
            为 True 时只返回内部多重指标.
        tensorprod : bool, optional
            为 True 时把单纯形顺序转换为插值点工具所用的张量积兼容顺序.

        Returns
        -------
        Tensor
            多重指标张量, 每个端点一列.
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
        """边上插值多重指标的个数.

        Parameters
        ----------
        order : tuple of int
            边上的多项式次数.
        internal : bool, optional
            为 True 时只计内部多重指标.

        Returns
        -------
        int
            局部插值多重指标的个数.
        """
        p = _require_order_tuple(order, "edge num_multi_index", 1)[0]
        if internal:
            return p - 1 if p > 1 else 0
        return p + 1

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """边的重心, 即两端点的平均.

        Parameters
        ----------
        ctx : EntityContext
            网格块与边分区.
        index : Index or None
            选定的边, None 表示全部边.

        Returns
        -------
        Tensor
            边的重心, 形状 ``(NC, GD)``.
        """
        points = cls._points(ctx, index)
        return bm.mean(points, axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """把边上的重心坐标映射为物理点.

        Parameters
        ----------
        ctx : EntityContext
            网格块与边分区.
        bcs : tuple of Tensor
            边的重心坐标, 一个形状为 ``(NQ, 2)`` 的张量.
        index : Index or None
            选定的边, None 表示全部边.

        Returns
        -------
        Tensor
            物理点, 形状 ``(NC, NQ, GD)``.
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
        """边上重心坐标的梯度.

        Parameters
        ----------
        ctx : EntityContext
            网格块与边分区.
        index : Index or None
            选定的边, None 表示全部边.
        bcs : tuple of Tensor, optional
            求值点的重心坐标; 给出时结果广播为 ``(NC, NQ, 2, GD_or_2)``.
        ref : bool, optional
            为 True 时返回参考边上对两个重心坐标的梯度, 否则返回对直角坐标的物理梯度.

        Returns
        -------
        Tensor
            ``lambda_0`` 与 ``lambda_1`` 的梯度; 不给 ``bcs`` 时形状为 ``(NC, 2, GD_or_2)``, 给出
            时沿积分点轴广播.
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
        """参考边上的一维 Gauss--Legendre 积分公式.

        Parameters
        ----------
        q : int
            积分阶.
        qtype : str or None, optional
            积分公式族, 边上只支持 ``"legendre"``.
        device : optional
            积分数据所在的后端设备.

        Returns
        -------
        Quadrature
            参考边上的积分公式对象.
        """
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported edge quadrature type: {qtype!r}")
        from ....quadrature import GaussLegendreQuadrature
        return GaussLegendreQuadrature(q, device=device)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """选定边的物理长度.

        Parameters
        ----------
        ctx : EntityContext
            网格块与边分区.
        index : Index or None
            选定的边, None 表示全部边.

        Returns
        -------
        Tensor
            边长, 形状 ``(NC,)``.
        """
        points = cls._points(ctx, index)
        tangent = points[:, 1, :] - points[:, 0, :]
        return bm.linalg.norm(tangent, axis=1)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """与各边切向正交的法方向.

        二维时每条边返回一个法方向, 三维时构造两个相互正交的法方向.

        Parameters
        ----------
        ctx : EntityContext
            网格块与边分区.
        index : Index or None
            选定的边, None 表示全部边.

        Returns
        -------
        Tensor
            法方向, ``GD <= 3`` 时形状为 ``(NC, GD - 1, GD)``.
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
        """选定边的物理切向量 (非单位).

        Parameters
        ----------
        ctx : EntityContext
            网格块与边分区.
        index : Index or None
            选定的边, None 表示全部边.

        Returns
        -------
        Tensor
            切向量, 形状 ``(NC, 1, GD)``.
        """
        points = cls._points(ctx, index)
        return (points[:, 1, :] - points[:, 0, :])[:, None, :]

    @classmethod
    def transform(cls, ctx: EntityContext, func, kind: str = "value"):
        """把直角坐标函数包装为可在边的重心坐标点上求值的函数.

        Parameters
        ----------
        ctx : EntityContext
            网格块与边分区.
        func : callable
            定义在物理点上的函数.
        kind : str, optional
            变换类型, 只支持 ``"value"``.

        Returns
        -------
        callable
            接受边重心坐标的包装函数.
        """
        points = ctx.block.positions[ctx.sector.indices]

        def wrapper(bc: Tensor) -> Tensor:
            """把重心坐标映射为物理点后求函数值."""
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
        """参考边到物理边映射的 Jacobi 矩阵.

        Parameters
        ----------
        ctx : EntityContext
            网格块与边分区.
        bcs : tuple of Tensor
            参考边上求值点的重心坐标.
        index : Index or None
            选定的边, None 表示全部边.

        Returns
        -------
        Tensor
            Jacobi 张量, 形状 ``(NC, NQ, GD, 1)``, 末轴为物理映射对参考坐标 ``u`` 的导数.
        """
        bcs = _require_bcs_tuple(bcs, "edge jacobi_matrix", 1)
        points = cls._points(ctx, index)
        gphi = cls().grad_shape_function_reference(bcs)
        return bm.einsum("cid,qin->cqdn", points, gphi)


EdgeSchema = LagrangeEdgeSchema
