"""MINRES: 对称不定系统的 Krylov 解法.

存在的理由是 Hu--Zhang 混合元产生的鞍点系统

.. math::

    \\begin{pmatrix} A & B^{\\mathsf T} \\\\ B & 0 \\end{pmatrix}

对称但不定, CG 的收敛理论在其上不成立 (会在负曲率处 breakdown 或给出无意义
的解). ``huzhang_mfem_analyzer`` 目前那条无预条件 CG 的分支正是这个问题.

与 CG 一样对算子无要求 (``requires`` 为空), 因此 'fa' 与 'ea' 两个层级都能用.

.. todo:: 占位模块, 尚无实现.
"""

from __future__ import annotations

from typing import Optional

from fealpy.backend import TensorLike

from .base import LinearSolver, SolveInfo


class MINRESSolver(LinearSolver):
    """极小残差法, 用于对称不定系统.

    ``requires`` 保持为空: 只需算子支持 ``@``.

    Parameters
    ----------
    M : LinearSolver, optional
        预条件子; 对不定系统必须自身正定, 否则 MINRES 的极小化性质不成立.
    atol, rtol : float
        绝对与相对收敛容差.
    maxit : int, optional
        最大迭代数.
    """

    requires = frozenset()

    def __init__(
        self,
        M: Optional[LinearSolver] = None,
        *,
        atol: float = 1e-12,
        rtol: float = 1e-8,
        maxit: Optional[int] = 10000,
    ) -> None:
        raise NotImplementedError

    def _solve(
        self, b: TensorLike, x0: Optional[TensorLike] = None
    ) -> "tuple[TensorLike, SolveInfo]":
        raise NotImplementedError
