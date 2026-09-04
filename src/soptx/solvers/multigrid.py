"""几何多重网格: 3D matrix-free 主线的预条件子.

3D 大规模下多重网格是唯一能同时满足两项要求的预条件子: 迭代数不随网格加密
增长, 且全程只需 matvec 与限制/延拓, 不需要显式全局矩阵. 代数多重网格
(:mod:`soptx.solvers.amg`) 拿不到非零结构, 在 'ea' 层级下不可用.

层次不从算子推导, 由 fem 侧构造后传入 -- 网格序列、各层空间与延拓算子都是
离散侧的东西, 求解层没有 Mesh 也没有 FunctionSpace. 这与 MFEM 的
``MultigridBase::AddLevel(Operator*, Solver*, ...)`` 同构: 层次外部装配,
``SetOperator`` 基本为空.

因此 ``requires`` 为空而不是 ``{CAP_HIERARCHY}``: 最细层算子只需支持 ``@``.
``CAP_HIERARCHY`` 保留给将来能自报层次的算子.

.. todo:: 占位模块, 尚无实现.
"""

from __future__ import annotations

from typing import Any, List, Optional

from fealpy.backend import TensorLike

from .base import LinearSolver, SolveInfo


class MultigridLevel:
    """多重网格一层所需的全部对象.

    Parameters
    ----------
    op : Any
        本层算子, 支持 ``@``; 'ea' 层级下是 matrix-free 包装.
    smoother : LinearSolver
        本层光滑子 (Jacobi 或 Chebyshev).
    prolongation : Any, optional
        本层到更细一层的延拓算子 ``P``; 限制取 ``P`` 的转置作用.
        最粗层为 None.
    """

    def __init__(
        self,
        op: Any,
        smoother: LinearSolver,
        prolongation: Optional[Any] = None,
    ) -> None:
        raise NotImplementedError


class Multigrid(LinearSolver):
    """多重网格循环.

    既可当独立求解器 (``solve``), 也可当 Krylov 的预条件子 (``M=`` 位) --
    两者是同一个类型, 这正是统一 ``LinearSolver`` 的意义. 3D 主线装配是
    ``CGSolver(M=Multigrid(...))``, 外层 Krylov 吸收多重网格残余的非光滑
    误差分量.

    Parameters
    ----------
    levels : list of MultigridLevel
        由粗到细排列的层次; 调用方用 :meth:`add_level` 逐层添加.
    cycle : str, default 'V'
        循环类型, 'V' 或 'W'.
    coarse_solver : LinearSolver, optional
        最粗层求解器, 通常是 ``DirectSolver``; 最粗层规模应小到能直接分解.
    n_pre, n_post : int
        前光滑与后光滑步数.

    Notes
    -----
    作预条件子用时 ``__matmul__`` 走单次零初值循环, 且必须是**线性**算子 --
    循环内部不能带自适应终止, 否则 CG 的搜索方向共轭性被破坏.
    """

    requires = frozenset()

    def __init__(
        self,
        levels: Optional[List[MultigridLevel]] = None,
        *,
        cycle: str = "V",
        coarse_solver: Optional[LinearSolver] = None,
        n_pre: int = 1,
        n_post: int = 1,
    ) -> None:
        raise NotImplementedError

    def add_level(
        self,
        op: Any,
        smoother: LinearSolver,
        prolongation: Optional[Any] = None,
    ) -> "Multigrid":
        """自粗至细追加一层, 返回 self 以便链式调用."""
        raise NotImplementedError

    def _cycle(self, level: int, b: TensorLike, x: TensorLike) -> TensorLike:
        """在第 ``level`` 层递归执行一次循环, 返回修正后的解."""
        raise NotImplementedError

    def _solve(
        self, b: TensorLike, x0: Optional[TensorLike] = None
    ) -> "tuple[TensorLike, SolveInfo]":
        raise NotImplementedError
