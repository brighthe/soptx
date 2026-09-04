"""代数多重网格: FA 旁路.

AMG 从矩阵非零结构做粗化与插值, 因此必须拿到显式稀疏矩阵 --
``requires = {CAP_MATRIX}``. 'ea' 层级的 matrix-free 算子给不出矩阵, 在其上
构造 AMG 会在 :meth:`LinearSolver.setup` 处抛
:class:`OperatorCapabilityError`, 这是设计意图而非缺陷: 3D 主线用几何多重网格
(:mod:`soptx.solvers.multigrid`), AMG 只服务 2D 与中小规模 FA 算例, 以及做
几何多重网格的对照基线.

.. todo:: 占位模块, 尚无实现.
"""

from __future__ import annotations

from typing import Any, Optional

from fealpy.backend import TensorLike

from .base import CAP_MATRIX, LinearSolver, SolveInfo


class AMGSolver(LinearSolver):
    """代数多重网格, 可作求解器也可作预条件子.

    Parameters
    ----------
    max_levels : int
        最大层数.
    theta : float
        强连接阈值.
    cycle : str, default 'V'
        循环类型.
    coarse_solver : LinearSolver, optional
        最粗层求解器.

    Notes
    -----
    :meth:`setup` 里做粗化与建层次 (代价可观), :meth:`_solve` 只跑循环. 密度
    更新只改元素值不改稀疏结构时, 层次可跨优化迭代复用 -- 对应 PETSc 的
    ``KSPSetReusePreconditioner``.
    """

    requires = frozenset({CAP_MATRIX})

    def __init__(
        self,
        *,
        max_levels: int = 10,
        theta: float = 0.25,
        cycle: str = "V",
        coarse_solver: Optional[LinearSolver] = None,
    ) -> None:
        raise NotImplementedError

    def setup(self, op: Any) -> "AMGSolver":
        """能力协商后取出矩阵并建层次."""
        raise NotImplementedError

    def _solve(
        self, b: TensorLike, x0: Optional[TensorLike] = None
    ) -> "tuple[TensorLike, SolveInfo]":
        raise NotImplementedError
