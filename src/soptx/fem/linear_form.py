# 部分移植自 brighthe/fealpy ``fealpy/fem/linear_form.py`` @ f474a5775 (coalesce 装配路线与
# 形状检查), 与 SOPTX 原子类合并为单个类.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""SOPTX 有限元线性型 (LinearForm) 模块.

``LinearForm`` 是全装配 (FA) 层级的右端项门面, 与 ``soptx.fem.bilinear_form`` 的
``BilinearForm`` 对称: 积分子容器与 ``add_integrator`` / ``__lshift__`` 等接口来自
``soptx.fem.form.Form``. ``assembly()`` 提供两条装配路线:

scatter (默认)
    逐组直接 ``bm.index_add`` 到稠密向量, 不建中间稀疏对象. 多列右端项按 soptx 求解器
    的规范布局产出 (gdof, B), 批量维在后 (见 ``ElementRestriction`` 的 Notes).
coalesce
    逐组展开成 ``COOTensor`` 后相加, 再合并或转稠密. 多列右端项为 (B, gdof), 批量维在前.

矩阵自由各层级里 E 向量到 L 向量的散加由 ``soptx.fem.kernels.ElementRestriction.scatter_add``
负责, 不经过本类: 那一步在 Krylov 迭代里每次 MatVec 都要走, 且要对 EA / PA / UA 三层都可用,
不能依赖 Form 层.
"""

from __future__ import annotations

import logging
from typing import Literal, Union, overload

from soptx.backend import backend_manager as bm
from soptx.fem.form import Form
from soptx.fem.integrator import LinearInt
from soptx.sparse import COOTensor
from soptx.typing import TensorLike

logger = logging.getLogger(__name__)


class LinearForm(Form[LinearInt]):
    """有限元线性型.

    Parameters
    ----------
    space : FunctionSpace
        检验函数空间.
    batch_size : int, optional
        多列右端项的列数, 0 表示单列. 默认为 0.

    Notes
    -----
    ``batch_size > 0`` 时默认路线返回 (gdof, B), coalesce 路线与 ``format='coo'`` 返回
    (B, gdof). 两种布局并存是刻意为之: 前者对齐求解器, 后者是积分子的原生布局.
    """

    _V = None

    def _get_sparse_shape(self):
        """返回全局向量形状 ``(gdof, )``."""
        spaces = self._spaces
        ugdof = spaces[0].number_of_global_dofs()
        return (ugdof,)

    def check_local_shape(self, entity_to_global: TensorLike, local_tensor: TensorLike):
        """检查积分子给出的局部张量与实体到全局自由度映射是否匹配.

        Parameters
        ----------
        entity_to_global : TensorLike
            ``(NE, ldof)`` 的实体到全局自由度映射.
        local_tensor : TensorLike
            ``(NE, ldof)`` 或带批量维的 ``(B, NE, ldof)`` 局部张量.

        Raises
        ------
        ValueError
            维数或实体数不匹配.
        """
        if entity_to_global.ndim != 2:
            raise ValueError("entity-to-global relationship should be a 2D tensor, "
                             f"but got shape {tuple(entity_to_global.shape)}.")
        if entity_to_global.shape[0] != local_tensor.shape[0]:
            raise ValueError(f"entity_to_global.shape[0] != local_tensor.shape[0]")
        if local_tensor.ndim not in (2, 3):
            raise ValueError("Output of operator integrators should be 3D "
                             "(or 4D with batch in the first dimension), "
                             f"but got shape {tuple(local_tensor.shape)}.")

    def check_space(self):
        """检查只有一个空间.

        Raises
        ------
        ValueError
            空间个数不是 1.
        """
        if len(self._spaces) != 1:
            raise ValueError("LinearForm should have only one space.")

    @overload
    def assembly(self) -> TensorLike: ...
    @overload
    def assembly(self, *, format: Literal['dense'], method: str = 'scatter') -> TensorLike: ...
    @overload
    def assembly(self, *, format: Literal['coo'], method: str = 'scatter') -> COOTensor: ...
    def assembly(
        self,
        *,
        format: Literal['dense', 'coo'] = 'dense',
        method: str = 'scatter',
    ) -> Union[TensorLike, COOTensor]:
        """装配全局右端项向量.

        Parameters
        ----------
        format : {'dense', 'coo'}, optional
            产出格式, 默认 'dense'.
        method : {'scatter', 'coalesce'}, optional
            装配路线, 默认直接散加的 'scatter'.

        Returns
        -------
        TensorLike or COOTensor
            ``format='dense'`` 时是稠密向量: scatter 路线为 (gdof, ) 或 (gdof, B),
            coalesce 路线为 (gdof, ) 或 (B, gdof). ``format='coo'`` 时是 ``COOTensor``.

        Raises
        ------
        ValueError
            ``method`` 或 ``format`` 取值不在支持范围内.
        RuntimeError
            scatter 路线下未添加任何积分子.

        Notes
        -----
        ``format='coo'`` 一律走 coalesce 路线, 批量布局为 (B, gdof).
        """
        if method == 'coalesce':
            return self._coalesce_assembly(format)

        if method != 'scatter':
            raise ValueError(f"不支持的装配路线: {method!r}, 只能是 'scatter' 或 'coalesce'")

        if format == 'coo':
            return self._coalesce_assembly('coo')

        if format != 'dense':
            raise ValueError(f"不支持的产出格式: {format!r}, 只能是 'dense' 或 'coo'")

        self._V = self._scatter_assembly()

        return self._V

    def _coalesce_assembly(self, format: str) -> Union[TensorLike, COOTensor]:
        """coalesce 路线: 逐组展开成 COO 后相加, 再转稠密或合并.

        Parameters
        ----------
        format : {'dense', 'coo'}
            产出格式.

        Returns
        -------
        TensorLike or COOTensor
            (gdof, ) 或 (B, gdof) 的稠密向量, 或合并后的 ``COOTensor``.

        Raises
        ------
        ValueError
            ``format`` 取值不在支持范围内.
        """
        V = self._scalar_assembly()

        if format == 'dense':
            self._V = V.to_dense()
        elif format == 'coo':
            self._V = V.coalesce()
        else:
            raise ValueError(f"Unsupported format {format}.")
        logger.info(f"Linear form vector constructed, with shape {list(V.shape)}.")

        return self._V

    def _scalar_assembly(self) -> COOTensor:
        """把全部积分子的局部张量展开成一个未合并的 ``COOTensor``."""
        self.check_space()
        space = self._spaces[0]
        batch_size = self.batch_size
        gdof = space.number_of_global_dofs()
        init_value_shape = (0,) if (batch_size == 0) else (batch_size, 0)
        sparse_shape = (gdof, )

        M = COOTensor(
            indices = bm.empty((1, 0), dtype=space.itype, device=bm.get_device(space)),
            values = bm.empty(init_value_shape, dtype=space.ftype, device=bm.get_device(space)),
            spshape = sparse_shape
        )

        for group_tensor, e2dofs_tuple in self.assembly_local_iterative():
            if (batch_size > 0) and (group_tensor.ndim == 2):
                group_tensor = bm.stack([group_tensor]*batch_size, axis=0)

            indices = e2dofs_tuple[0].reshape(1, -1)
            group_tensor = bm.reshape(group_tensor, self._values_ravel_shape)
            M = M.add(COOTensor(indices, group_tensor, sparse_shape))

        return M

    def _scatter_assembly(self) -> TensorLike:
        """scatter 路线: 直接散加装配, 不经过稀疏中间对象.

        Returns
        -------
        TensorLike
            (gdof, ) 或 (gdof, B) 的稠密右端项.

        Raises
        ------
        RuntimeError
            未添加任何积分子.

        Notes
        -----
        逐组累加而不是先把各组拼起来再散加: 后者要 ``bm.concat``, 前者只是对同一块
        目标内存多调几次 ``bm.index_add``, 组数通常个位数.
        """
        self.check_space()
        space = self._spaces[0]
        batch_size = self.batch_size
        gdof = space.number_of_global_dofs()

        shape = (gdof, ) if batch_size == 0 else (gdof, batch_size)
        y = bm.zeros(shape, dtype=space.ftype, device=bm.get_device(space))

        empty = True

        for group_tensor, e2dofs_tuple in self.assembly_local_iterative():
            empty = False
            entity_to_global = e2dofs_tuple[0]
            self.check_local_shape(entity_to_global, group_tensor)

            index = bm.reshape(entity_to_global, (-1, ))
            src = self._to_batch_last(group_tensor, batch_size)
            y = bm.index_add(y, index, src)

        if empty:
            raise RuntimeError("LinearForm 中未添加任何有效的积分子 (Integrator)")

        return y

    @staticmethod
    def _to_batch_last(group_tensor: TensorLike, batch_size: int) -> TensorLike:
        """把积分子给出的局部张量摊平成散加的源数组.

        Parameters
        ----------
        group_tensor : TensorLike
            (NC, ldof) 或 (B, NC, ldof) 的局部张量. 积分子把批量维放在最前.
        batch_size : int
            多列右端项的列数, 0 表示单列.

        Returns
        -------
        TensorLike
            (NC * ldof, ) 或 (NC * ldof, B) 的源数组, 批量维在后.

        Notes
        -----
        ``batch_size > 0`` 而局部张量只有两维时, 说明该积分子与列无关, 按 coalesce 路线
        的做法向各列广播.
        """
        if batch_size == 0:
            return bm.reshape(group_tensor, (-1, ))

        if group_tensor.ndim == 2:
            src = bm.broadcast_to(group_tensor[..., None],
                                group_tensor.shape + (batch_size, ))
        else:
            src = bm.permute_dims(group_tensor, (1, 2, 0))

        return bm.reshape(src, (-1, batch_size))
