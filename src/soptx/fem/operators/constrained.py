# -*- coding: utf-8 -*-
"""本质边界条件算子 (constrained operator).

把 Dirichlet 约束施加成算子的一层包装, 而不是改写矩阵:

    A = Pi_I K Pi_I + Pi_D

其中 Pi_D 是 Dirichlet 自由度上的投影, Pi_I = I - Pi_D. 作用一次的做法是先把
输入在 Dirichlet 自由度上置零, 作用内层算子, 再把这些自由度还原成输入原值, 与
'fa' 的对称消元定义同一个离散系统.

与 FA 的对称消元相比, 这一层不需要矩阵元, 因此 EA 及以下的所有层级共用它; 也因
此它可以套在 ``OverlapOperator`` 外面 —— 包装次序必须是
``ConstrainedOperator(OverlapOperator(level))``: 跨 rank 归约要先把局部作用拼成
完整的算子作用, 再谈边界自由度.

本类替代 FEALPy 的 ``DirichletBCOperator``: 数学定义与后者逐位一致, 多出来的是
``diagonal()`` —— 有了它, Jacobi 类预条件不必再回到 analyzer 里按算子层级取对角.
"""

from typing import Any, Callable, Optional, Tuple, Union

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

CoefLike = Union[float, int, TensorLike, Callable[..., TensorLike]]


class ConstrainedOperator:
    """施加了 Dirichlet 约束的系统算子.

    Parameters
    ----------
    form : 内层算子, 需支持 ``@``, 并提供 ``shape`` 与 ``spaces``; 串行下是装配
        层级对象, 分布式下是包住层级的 ``OverlapOperator``.
    gd : Dirichlet 边界值, 可以是常数, 数组或可调用对象; 只有 ``init_solution``
        用得上.
    isDDof : (gdof, ) 的布尔标记, 为 None 时由函数空间按 ``threshold`` 现算.
    threshold : 仅在 ``isDDof`` 为 None 时用于挑边界自由度.
    """

    def __init__(self,
                form: Any,
                gd: Optional[CoefLike] = None,
                *,
                threshold: Optional[Callable] = None,
                isDDof: Optional[TensorLike] = None,
            ) -> None:
        self.form = form
        self.gd = gd

        if isDDof is None:
            isDDof = form.spaces[0].is_boundary_dof(threshold=threshold)
        self.is_boundary_dof = isDDof

        self.boundary_dof_index = bm.nonzero(isDDof)[0]
        self.shape: Tuple[int, int] = form.shape

    @property
    def spaces(self) -> Tuple:
        """内层算子所在的函数空间"""
        return self.form.spaces

    def init_solution(self, dtype=None) -> TensorLike:
        """造出满足 Dirichlet 条件的初值: 边界自由度取给定值, 其余为零.

        Returns
        -------
        uh : (gdof, ) 的解向量初值.
        """
        space = self.form.spaces[0]
        uh = bm.zeros(self.shape[1], dtype=dtype if dtype else space.ftype)
        space.boundary_interpolate(self.gd, uh, threshold=self.is_boundary_dof)

        return uh

    def apply(self, F: TensorLike, uh: TensorLike) -> TensorLike:
        """把边界值对右端项的贡献移到右端.

        Parameters
        ----------
        F : 未施加边界条件的右端项.
        uh : 边界自由度取给定值, 内部自由度为零的基准向量.

        Returns
        -------
        F : 边界自由度上取给定值, 其余减去 K uh 的右端项.
        """
        F = F - self.form @ uh
        F = bm.set_at(F, self.is_boundary_dof, uh[self.is_boundary_dof])

        return F

    def __matmul__(self, u: TensorLike) -> TensorLike:
        """算子作用 v = (Pi_I K Pi_I + Pi_D) u.

        Parameters
        ----------
        u : (gdof, ) 的输入向量.

        Returns
        -------
        v : 与 u 同形状; Dirichlet 自由度上原样返回 u, 其余是内部作用的结果.
        """
        v = bm.copy(u)
        val = v[self.is_boundary_dof]
        v = bm.set_at(v, self.is_boundary_dof, 0.0)
        v = self.form @ v
        v = bm.set_at(v, self.is_boundary_dof, val)

        return v

    def diagonal(self) -> TensorLike:
        """取算子对角.

        Dirichlet 自由度上恒为 1 (那里的方程就是 u_D = g), 其余取内层算子的对角.
        内层算子负责把对角交成与自己的 ``@`` 同口径的向量: 串行下装配层级给的是
        本 rank 的对角, 分布式下 ``OverlapOperator`` 给的是已跨 rank 求和的对角.

        Returns
        -------
        diag : (gdof, ) 的算子对角, SPD 系统下逐元严格为正.
        """
        diag = self.form.diagonal()

        return bm.set_at(diag, self.is_boundary_dof, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, form={self.form!r})"
