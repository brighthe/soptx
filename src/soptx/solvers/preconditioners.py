"""迭代解法的预条件子.

预条件子与求解器是同一个类型 :class:`~soptx.solvers.base.LinearSolver`:
``M @ r`` 返回 :math:`M^{-1} r` 意义下的修正残差, 可直接作 ``cg(..., M=...)``
传入; 同一个对象也能当求解器用 (对角预条件子即一步精确的对角求解).

本模块只放纯代数的预条件子 -- 不关心对角来自 FA 全局矩阵还是 EA 单元矩阵.
取对角这件事本身经 :func:`~soptx.solvers.base.operator_diagonal` 走能力协商,
按算子实际能提供什么分派, 不再由 analyzer 按 ``operator_level`` 分支决定.
"""

from __future__ import annotations

from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

from .base import CAP_DIAGONAL, LinearSolver, SolveInfo, operator_diagonal


class DiagonalPreconditioner(LinearSolver):
    """Jacobi (对角) 预条件子: ``M @ r = r / diag``.

    对角有两种来源, ``requires`` 随之取不同的值:

    - 构造时显式给 ``diag``: 对角已经在手, 对算子没有任何要求, ``requires``
      退成空集, 只支持 ``@`` 的 matrix-free 算子也能绑;
    - 构造时不给: ``setup`` 时向算子要, ``requires`` 为 ``{CAP_DIAGONAL}``,
      算子给不出就在 setup 处拒绝, 而不是等到 CG 里以 breakdown 暴露.

    后者是 analyzer 走的路: 它把算子交过来, 由本类按算子实际能提供什么取对角,
    不必再自己按 ``operator_level`` 分支。

    Parameters
    ----------
    diag : TensorLike, optional
        系统矩阵的对角线, 一维张量; 对 SPD 系统必须严格为正. 缺省时在
        ``setup`` 中经 :func:`~soptx.solvers.base.operator_diagonal` 取.
        校验在拿到对角的那一刻做 (显式传入即构造时, 否则 setup 时), 避免把
        奇异预条件子静默传进 CG 后以 breakdown 的形式暴露.
    """

    requires = frozenset({CAP_DIAGONAL})

    def __init__(self, diag: Optional[TensorLike] = None) -> None:
        super().__init__()
        self._inv_diag: Optional[TensorLike] = None
        if diag is not None:
            # 对角已在手, 对算子无所求; 实例上覆盖类属性
            self.requires = frozenset()
            self._accept_diag(diag)

    def _accept_diag(self, diag: TensorLike) -> None:
        """校验并存下对角的倒数"""
        if diag.ndim != 1:
            raise ValueError(f"diag 必须是一维张量, 得到 ndim={diag.ndim}")
        if float(bm.min(diag)) <= 0.0:
            raise ValueError(
                "diag 含非正元素, 不能作 SPD 系统的 Jacobi 预条件子: "
                f"min(diag)={float(bm.min(diag)):.16e}"
            )
        self._inv_diag = 1.0 / diag

    def setup(self, op) -> "DiagonalPreconditioner":
        """绑定算子; 构造时没给对角就在这里向算子要"""
        super().setup(op)
        if self._inv_diag is None:
            self._accept_diag(operator_diagonal(op))

        return self

    def __matmul__(self, other: TensorLike) -> TensorLike:
        # 快速路径: 预条件子在 Krylov 每步迭代都被调用一次, 不绕 solve 的
        # info 构造与校验
        if self._inv_diag is None:
            raise RuntimeError(
                "DiagonalPreconditioner 构造时未给对角, 请先 setup(op) 让它"
                "从算子取"
            )
        if other.ndim == 1:
            return other * self._inv_diag
        # 2D 右端: 第一维是自由度维 (与 cg 内部 batch_first=False 布局一致)
        return other * self._inv_diag[:, None]

    def _solve(
        self, b: TensorLike, x0: Optional[TensorLike] = None
    ) -> "tuple[TensorLike, SolveInfo]":
        # 对角系统一步精确求解, 与初值无关, 故忽略 x0.
        # relres 要算就得做一次 matvec, 而预条件子模式下 op 未必 setup 过,
        # 因此留 None 表示"未计算", 而不是填 0.0 冒充已收敛.
        return self @ b, {"niter": 1, "relres": None, "converged": True}


def estimate_lambda_max(
    op, *, n_iter: int = 10, seed: Optional[int] = None
) -> float:
    """幂迭代估计算子最大特征值.

    Chebyshev 需要谱区间上界, 而 'ea' 层级下拿不到矩阵, 只能靠 matvec 估计.
    实践中取估计值乘一个安全系数 (1.1 左右) 作上界: 低估会让高频分量不被
    衰减, 多重网格随之失效; 高估只是收敛慢一点.

    .. todo:: 占位, 尚无实现.
    """
    raise NotImplementedError


class ChebyshevSmoother(LinearSolver):
    """Chebyshev 多项式光滑子.

    多重网格在 matrix-free 下的默认光滑子: 只用 matvec 与对角, 不需要矩阵的
    非零结构, 也不像 Gauss--Seidel 那样依赖串行的行序遍历 -- 后者在 GPU 上
    无法并行, 这是 3D 主线选 Chebyshev 而非 GS 的原因.

    在 :math:`[\\lambda_{\\min}, \\lambda_{\\max}]` 的高频段构造衰减多项式,
    ``lambda_min`` 通常取 ``lambda_max / 30`` 量级 (只需压高频, 低频交给粗层).

    Parameters
    ----------
    diag : TensorLike
        算子对角线, 用于内部的对角缩放.
    lambda_max : float, optional
        谱上界; 缺省时 :meth:`setup` 用 :func:`estimate_lambda_max` 估计.
    degree : int, default 2
        多项式次数, 即每次光滑的 matvec 次数.

    Notes
    -----
    作多重网格光滑子时必须是**线性**算子: 固定次数, 不带自适应终止, 否则外层
    CG 的共轭性被破坏.

    .. todo:: 占位, 尚无实现.
    """

    requires = frozenset()

    def __init__(
        self,
        diag: TensorLike,
        *,
        lambda_max: Optional[float] = None,
        degree: int = 2,
    ) -> None:
        raise NotImplementedError

    def setup(self, op) -> "ChebyshevSmoother":
        """绑定算子; ``lambda_max`` 未给时在此估计."""
        raise NotImplementedError

    def _solve(
        self, b: TensorLike, x0: Optional[TensorLike] = None
    ) -> "tuple[TensorLike, SolveInfo]":
        raise NotImplementedError
