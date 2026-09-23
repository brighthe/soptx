# -*- coding: utf-8 -*-
"""SOPTX 线性型 (LinearForm) 模块.

本模块提供 soptx 自己的右端项门面 ``LinearForm``, 与 ``soptx.fem.bilinear_form``
的 ``BilinearForm`` 对称: 两者都继承 FEALPy 的同名类, 只替换 ``assembly()`` 的装配
路线, 积分子容器与 ``add_integrator`` / ``__lshift__`` 等接口原样沿用.

与 FEALPy 实现的两点差别
------------------------
装配路线
    FEALPy 走 "建 COOTensor -> 逐组 concat -> to_dense" 三步, 最后仍落到一次
    ``bm.index_add``. 本类默认直接做那一次 ``index_add``, 省掉中间的稀疏对象与
    拼接.
批量布局
    FEALPy 的多列右端项是 (batch, gdof), 批量维在前; soptx 求解器的规范布局是
    (gdof, B), 批量维在后 (见 ``ElementRestriction`` 的 Notes: FEALPy
    ``BilinearForm.__matmul__`` 在这一点上与自己的散加轴不自洽). 本类默认按后者
    产出.

本类只负责 FA (全装配) 层级的右端项. 矩阵自由各层级里 E 向量到 L 向量的散加由
``soptx.fem.kernels.ElementRestriction.scatter_add`` 负责, 不经过本类: 那一步在
Krylov 迭代里每次 MatVec 都要走, 且要对 EA / PA / UA 三层都可用, 不能依赖 Form 层.
"""

from __future__ import annotations

from typing import Literal, Union, overload

from fealpy.backend import backend_manager as bm
from fealpy.fem.linear_form import LinearForm as FEALPyLinearForm
from fealpy.sparse import COOTensor
from fealpy.typing import TensorLike


class LinearForm(FEALPyLinearForm):
    """SOPTX 有限元线性型门面.

    Parameters
    ----------
    space : 检验函数空间, 与 FEALPy ``LinearForm`` 一致.
    batch_size : 多列右端项的列数, 0 表示单列. 默认为 0.

    Notes
    -----
    ``batch_size > 0`` 时 ``assembly(format='dense')`` 返回 (gdof, B), 与父类的
    (batch, gdof) 相反. 这是刻意为之的布局对齐, 不是疏漏; 需要父类布局时用
    ``method='coalesce'`` 或 ``format='coo'``, 两者都原样委托给父类.
    """

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
        format : 产出格式 ('dense' | 'coo'). 默认为 'dense'.
        method : 装配路线 ('scatter' | 'coalesce'). 默认为直接散加的 'scatter'.

        Returns
        -------
        global_vector : ``format='dense'`` 时是 (gdof, ) 或 (gdof, B) 的稠密向量,
            ``format='coo'`` 时是 FEALPy ``COOTensor``.

        Raises
        ------
        ValueError
            ``method`` 或 ``format`` 取值不在支持范围内.
        RuntimeError
            未添加任何积分子.

        Notes
        -----
        ``format='coo'`` 与 ``method='coalesce'`` 都委托父类, 因此批量布局仍是父类
        的 (batch, gdof); 只有默认路线产出 (gdof, B).
        """
        if method == 'coalesce':
            return super().assembly(format=format)

        if method != 'scatter':
            raise ValueError(f"不支持的装配路线: {method!r}, 只能是 'scatter' 或 'coalesce'")

        if format == 'coo':
            return super().assembly(format='coo')

        if format != 'dense':
            raise ValueError(f"不支持的产出格式: {format!r}, 只能是 'dense' 或 'coo'")

        self._V = self._scatter_assembly()

        return self._V

    def _scatter_assembly(self) -> TensorLike:
        """直接散加装配, 不经过稀疏中间对象.

        Returns
        -------
        y : (gdof, ) 或 (gdof, B) 的稠密右端项.

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
        group_tensor : (NC, ldof) 或 (B, NC, ldof) 的局部张量. 积分子按 FEALPy 约定
            把批量维放在最前.
        batch_size : 多列右端项的列数, 0 表示单列.

        Returns
        -------
        src : (NC * ldof, ) 或 (NC * ldof, B) 的源数组, 批量维在后.

        Notes
        -----
        ``batch_size > 0`` 而局部张量只有两维时, 说明该积分子与列无关, 按父类的做法
        向各列广播.
        """
        if batch_size == 0:
            return bm.reshape(group_tensor, (-1, ))

        if group_tensor.ndim == 2:
            src = bm.broadcast_to(group_tensor[..., None],
                                group_tensor.shape + (batch_size, ))
        else:
            src = bm.permute_dims(group_tensor, (1, 2, 0))

        return bm.reshape(src, (-1, batch_size))
