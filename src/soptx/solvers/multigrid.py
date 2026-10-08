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
"""

from __future__ import annotations

from typing import Any, List, Optional

from soptx.backend import TensorLike

from .base import LinearSolver, SolveInfo


class MultigridLevel:
    """多重网格一层所需的全部对象.

    Parameters
    ----------
    op : 本层算子, 支持 ``@``; 'ea' 层级下是 matrix-free 包装. 最细层可为 None,
        由 :meth:`Multigrid.setup` 填入.
    smoother : 本层光滑子; 最粗层为 None, 由 ``coarse_solver`` 求解.
    prolongation : 由更粗一层到本层的延拓算子 ``P``, 支持 ``@`` 与 ``.T``; 限制取
        ``P`` 的转置作用. 最粗层为 None.
    """

    def __init__(
        self,
        op: Any = None,
        smoother: Optional[LinearSolver] = None,
        prolongation: Optional[Any] = None,
    ) -> None:
        self.op = op
        self.smoother = smoother
        self.prolongation = prolongation


class Multigrid(LinearSolver):
    """多重网格循环.

    既可当独立求解器 (``solve``), 也可当 Krylov 的预条件子 (``M=`` 位) --
    两者是同一个类型, 这正是统一 ``LinearSolver`` 的意义. 3D 主线装配是
    ``CGSolver(M=Multigrid(...))``, 外层 Krylov 吸收多重网格残余的非光滑
    误差分量.

    Parameters
    ----------
    levels : 由粗到细排列的层次; 也可用 :meth:`add_level` 逐层添加.
    cycle : 循环类型, 目前只实现 ``'V'``.
    coarse_solver : 最粗层求解器, 通常是 ``DirectSolver``; 最粗层规模应小到能直接分解.
    n_pre, n_post : 前光滑与后光滑次数, 须相等且为正整数.

    Raises
    ------
    ValueError
        ``cycle`` 不是 ``'V'``, 或 ``n_pre``、``n_post`` 不相等或不为正.

    Notes
    -----
    作预条件子用时 ``__matmul__`` 走单次零初值循环, 且必须是**线性**算子 --
    循环内部不能带自适应终止, 否则 CG 的搜索方向共轭性被破坏.

    V 循环对称 (PCG 的前提) 要求前后光滑次数相同, 且光滑子关于 :math:`A` 内积自伴
    (如加权 Jacobi); 最粗层须精确求解.
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
        super().__init__()
        if cycle != "V":
            raise ValueError(f"目前只实现 V 循环, 得到 cycle={cycle!r}")
        if n_pre != n_post or n_pre < 1:
            raise ValueError(
                f"n_pre 与 n_post 须相等且为正整数, V 循环才对称; 得到 {n_pre}, {n_post}"
            )
        self.levels: List[MultigridLevel] = list(levels) if levels is not None else []
        self.cycle = cycle
        self.coarse_solver = coarse_solver
        self.n_pre = int(n_pre)
        self.n_post = int(n_post)
        self._restrictions: List[Any] = []

    def add_level(
        self,
        op: Any = None,
        smoother: Optional[LinearSolver] = None,
        prolongation: Optional[Any] = None,
    ) -> "Multigrid":
        """自粗至细追加一层, 返回 self 以便链式调用."""
        self.levels.append(MultigridLevel(op, smoother, prolongation))
        return self

    def setup(self, op: Any) -> "Multigrid":
        """以 ``op`` 作最细层算子, 并绑定各层光滑子与最粗层求解器.

        Parameters
        ----------
        op : 最细层算子, 覆盖最细层原有的 ``op``.

        Returns
        -------
        Multigrid
            返回 self.

        Raises
        ------
        ValueError
            层次为空, 缺 ``coarse_solver``, 或某层的延拓、光滑子与其位置不符.
        """
        super().setup(op)
        if not self.levels:
            raise ValueError("Multigrid 没有任何层次, 请先传入 levels 或 add_level")
        if self.coarse_solver is None:
            raise ValueError("Multigrid 缺少最粗层求解器 coarse_solver")
        if self.levels[0].prolongation is not None:
            raise ValueError("最粗层不应有延拓算子")
        for index, level in enumerate(self.levels[1:], start=1):
            if level.prolongation is None or level.smoother is None:
                raise ValueError(f"第 {index} 层 (由粗到细计) 缺少延拓算子或光滑子")

        self.levels[-1].op = op
        for level in self.levels[1:]:
            if level.op is None:
                raise ValueError("非最细层的算子须由调用方给出")
            level.smoother.setup(level.op)
        self.coarse_solver.setup(self.levels[0].op)
        # 限制算子 P^T 每个循环都要用, 转置只做一次
        self._restrictions = [None] + [level.prolongation.T for level in self.levels[1:]]

        return self

    def __matmul__(self, r: TensorLike) -> TensorLike:
        # 预条件子快速路径: 零初值一次循环, 不绕 solve 的 info 构造与校验
        if not self.is_setup:
            raise RuntimeError("Multigrid 尚未 setup, 请先调用 setup(op)")
        return self._cycle(len(self.levels) - 1, r, None)

    def _cycle(self, level: int, b: TensorLike, x: Optional[TensorLike]) -> TensorLike:
        """在第 ``level`` 层 (由粗到细计, 0 为最粗) 递归执行一次 V 循环, 返回修正后的解.

        Parameters
        ----------
        level : 层号.
        b : 本层右端项.
        x : 本层初值; None 表示零初值.

        Returns
        -------
        x : 本层的近似解.
        """
        if level == 0:
            # 最粗层精确求解, 与初值无关
            return self.coarse_solver @ b
        current = self.levels[level]
        for _ in range(self.n_pre):
            x, _ = current.smoother.solve(b, x)
        residual = b - current.op @ x
        correction = self._cycle(level - 1, self._restrictions[level] @ residual, None)
        x = x + current.prolongation @ correction
        for _ in range(self.n_post):
            x, _ = current.smoother.solve(b, x)

        return x

    def _solve(
        self, b: TensorLike, x0: Optional[TensorLike] = None
    ) -> "tuple[TensorLike, SolveInfo]":
        # 残差修正形式的单次循环; 不判断收敛, 迭代交给外层 Krylov
        if x0 is None:
            x = self @ b
        else:
            x = x0 + self @ (b - self.op @ x0)

        return x, {"niter": 1, "relres": None, "converged": False}
