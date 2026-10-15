"""多重网格循环: Krylov 方法的多重网格预条件子.

多重网格的迭代数不随网格加密增长, 是 3D 大规模问题的主力预条件子. 本模块只实现
循环本身: 最细层算子只需支持 ``@``, 可以是 'ea' 层级的 matrix-free 包装; 粗层
算子与最粗层求解器的形式由层次提供方决定, 如 :mod:`soptx.fem.multigrid` 在粗层
装配稀疏矩阵, 最粗层直接分解. 代数多重网格 (尚未实现) 要从矩阵非零结构粗化,
在 'ea' 层级下不可用.

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
    n_pre, n_post : 前光滑与后光滑时调用光滑子的次数, 须相等且为正整数.

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

    光滑子自身也可能一次走多步 (如 ``JacobiSmoother`` 的 ``sweeps``), 实际光滑步数为
    ``n_pre`` 与其之积. :meth:`soptx.fem.multigrid.StructuredHexHierarchy.build_multigrid`
    经光滑子的 ``sweeps`` 调节, ``n_pre`` 与 ``n_post`` 保持 1.
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
        """自粗至细追加一层.

        Parameters
        ----------
        op : 本层算子; 最细层可为 None, 由 :meth:`setup` 填入.
        smoother : 本层光滑子; 最粗层为 None.
        prolongation : 由更粗一层到本层的延拓算子; 最粗层为 None.

        Returns
        -------
        Multigrid
            返回 self, 便于链式调用.
        """
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
            层次为空, 缺 ``coarse_solver``, 某层的延拓、光滑子与其位置不符, 或非最细层
            缺算子.

        Notes
        -----
        先校验层次, 再绑定各层, 最后才绑定 ``op``: 中途出错时对象仍处于未 setup 状态,
        ``@`` 报 "尚未 setup", 不会带着残缺的层次进入循环.
        """
        if not self.levels:
            raise ValueError("Multigrid 没有任何层次, 请先传入 levels 或 add_level")
        if self.coarse_solver is None:
            raise ValueError("Multigrid 缺少最粗层求解器 coarse_solver")
        if self.levels[0].prolongation is not None or self.levels[0].smoother is not None:
            raise ValueError("最粗层不应有延拓算子或光滑子, 它由 coarse_solver 求解")
        for index, level in enumerate(self.levels[1:], start=1):
            if level.prolongation is None or level.smoother is None:
                raise ValueError(f"第 {index} 层 (由粗到细计) 缺少延拓算子或光滑子")
        if any(level.op is None for level in self.levels[:-1]):
            raise ValueError("非最细层的算子须由调用方给出")

        self.levels[-1].op = op
        for level in self.levels[1:]:
            level.smoother.setup(level.op)
        self.coarse_solver.setup(self.levels[0].op)
        # 限制算子 P^T 每个循环都要用, 转置只做一次
        self._restrictions = [None] + [level.prolongation.T for level in self.levels[1:]]
        super().setup(op)

        return self

    def __matmul__(self, r: TensorLike) -> TensorLike:
        """预条件子快速路径: 零初值做一次循环, 不经 solve 的 info 构造与校验."""
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
