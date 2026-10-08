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

from soptx.backend import backend_manager as bm
from soptx.backend import TensorLike

from .base import CAP_DIAGONAL, LinearSolver, SolveInfo, operator_diagonal


def _inverse_diagonal(diag: TensorLike, owner: str) -> TensorLike:
    """校验对角并返回其倒数.

    Parameters
    ----------
    diag : 系统矩阵的主对角, 一维张量.
    owner : 报错信息里的类名.

    Returns
    -------
    inv_diag : (n, ) 的对角倒数.

    Raises
    ------
    ValueError
        对角不是一维张量, 或含非正元素 (不能用于 SPD 系统).
    """
    if diag.ndim != 1:
        raise ValueError(f"diag 必须是一维张量, 得到 ndim={diag.ndim}")
    if float(bm.min(diag)) <= 0.0:
        raise ValueError(
            f"diag 含非正元素, 不能作 SPD 系统的 {owner}: "
            f"min(diag)={float(bm.min(diag)):.16e}"
        )
    return 1.0 / diag


def _scale_rows(values: TensorLike, weights: TensorLike) -> TensorLike:
    """按自由度维逐行缩放; 2D 右端的第一维是自由度维 (与 cg 的 batch_first=False 一致)."""
    if values.ndim == 1:
        return values * weights
    return values * weights[:, None]


class DiagonalPreconditioner(LinearSolver):
    """Jacobi (对角) 预条件子: ``M @ r = r / diag``.

    对角有两种来源, ``requires`` 随之取不同的值:

    - 构造时显式给 ``diag``: 对角已经在手, 对算子没有任何要求, ``requires``
      退成空集, 只支持 ``@`` 的 matrix-free 算子也能绑;
    - 构造时不给: ``setup`` 时向算子要, ``requires`` 为 ``{CAP_DIAGONAL}``,
      算子给不出就在 setup 处拒绝, 而不是等到 CG 里以 breakdown 暴露.

    后者是 analyzer 走的路: 它把算子交过来, 由本类按算子实际能提供什么取对角,
    不必再自己按 ``operator_level`` 分支.

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
        self._inv_diag = _inverse_diagonal(diag, "Jacobi 预条件子")

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
        return _scale_rows(other, self._inv_diag)

    def _solve(
        self, b: TensorLike, x0: Optional[TensorLike] = None
    ) -> "tuple[TensorLike, SolveInfo]":
        # 对角系统一步精确求解, 与初值无关, 故忽略 x0.
        # relres 要算就得做一次 matvec, 而预条件子模式下 op 未必 setup 过,
        # 因此留 None 表示"未计算", 而不是填 0.0 冒充已收敛.
        return self @ b, {"niter": 1, "relres": None, "converged": True}


class JacobiSmoother(LinearSolver):
    """加权 Jacobi 光滑子: 重复 ``sweeps`` 次 :math:`x \\leftarrow x + \\omega D^{-1}(b - A x)`.

    多重网格的默认光滑子, 只用 matvec 与对角. 对 SPD 的 :math:`A`, 误差传播算子
    :math:`I - \\omega D^{-1} A` 关于 :math:`A` 内积自伴, 因此前后光滑用同一个对象、
    同样次数时 V 循环是对称的, 可作 PCG 的预条件子.

    Parameters
    ----------
    omega : 松弛因子, 须满足 :math:`0 < \\omega < 2 / \\lambda_{\\max}(D^{-1} A)` 才收敛.
        缺省 (None) 时在 ``setup`` 中取 :math:`\\omega = 4 / (3 \\cdot 1.1\\, \\hat\\lambda)`,
        :math:`\\hat\\lambda` 由 :func:`estimate_lambda_max` 给出.
    sweeps : 每次调用的扫描次数, 正整数.
    diag : 系统矩阵的主对角, 一维张量; 缺省时在 ``setup`` 中向算子要. 给出时
        ``requires`` 退成空集, 与 :class:`DiagonalPreconditioner` 相同.
    power_iterations : 自动取 ``omega`` 时幂迭代的 matvec 次数.

    Raises
    ------
    ValueError
        ``omega`` 不为正, ``sweeps`` 不是正整数, 或对角不合法.

    Notes
    -----
    固定的 ``omega`` 不稳健: 三维 Q1 线弹性在均匀材料下 :math:`\\lambda_{\\max}(D^{-1} A)`
    约为 3, 但单元系数剧烈跳变时可达 4.6 以上, 此时 :math:`\\omega = 0.6` 已使光滑发散,
    作 PCG 预条件子会失去正定性. 自动取值的 :math:`4/3` 是加权 Jacobi 光滑的常用最优
    系数, 1.1 抵消幂迭代对 :math:`\\lambda_{\\max}` 的低估.

    从零初值出发时第一次扫描化为 :math:`x = \\omega D^{-1} b`, 省一次 matvec.
    ``info`` 中 ``converged`` 恒为 False: 光滑子只做固定次数的扫描, 不判断收敛.
    """

    requires = frozenset({CAP_DIAGONAL})

    def __init__(self, *, omega: Optional[float] = None, sweeps: int = 1,
                 diag: Optional[TensorLike] = None, power_iterations: int = 15) -> None:
        super().__init__()
        if omega is not None and not omega > 0.0:
            raise ValueError(f"omega 须为正数, 得到 {omega!r}")
        if isinstance(sweeps, bool) or int(sweeps) != sweeps or sweeps < 1:
            raise ValueError(f"sweeps 须为正整数, 得到 {sweeps!r}")
        self._given_omega = None if omega is None else float(omega)
        self.omega: Optional[float] = self._given_omega
        self.sweeps = int(sweeps)
        self.power_iterations = int(power_iterations)
        self._diag: Optional[TensorLike] = None
        self._weights: Optional[TensorLike] = None
        if diag is not None:
            self.requires = frozenset()
            _inverse_diagonal(diag, "Jacobi 光滑子")
            self._diag = diag

    def setup(self, op) -> "JacobiSmoother":
        """绑定算子; 构造时没给对角就在这里向算子要, 没给 ``omega`` 就在这里估计"""
        super().setup(op)
        diag = self._diag if self._diag is not None else operator_diagonal(op)
        inverse = _inverse_diagonal(diag, "Jacobi 光滑子")
        if self._given_omega is None:
            lam = estimate_lambda_max(op, diag=diag, n_iter=self.power_iterations)
            self.omega = 4.0 / (3.0 * 1.1 * lam)
        self._weights = self.omega * inverse

        return self

    def _solve(
        self, b: TensorLike, x0: Optional[TensorLike] = None
    ) -> "tuple[TensorLike, SolveInfo]":
        op = self.op
        if x0 is None:
            x = _scale_rows(b, self._weights)
            remaining = self.sweeps - 1
        else:
            x = x0
            remaining = self.sweeps
        for _ in range(remaining):
            x = x + _scale_rows(b - op @ x, self._weights)

        return x, {"niter": self.sweeps, "relres": None, "converged": False}


def estimate_lambda_max(
    op, *, diag: Optional[TensorLike] = None, n_iter: int = 15
) -> float:
    """幂迭代估计 :math:`D^{-1} A` 的最大特征值, :math:`D` 缺省为单位阵.

    Chebyshev 与加权 Jacobi 都需要谱上界, 而 'ea' 层级下拿不到矩阵, 只能靠 matvec
    估计. 实践中取估计值乘一个安全系数 (1.1 左右) 作上界: 低估会让高频分量不被
    衰减甚至放大, 多重网格随之失效; 高估只是收敛慢一点.

    Parameters
    ----------
    op : 对称正定算子, 支持 ``@`` 与 ``shape``.
    diag : (n, ) 的正对角缩放; 给出时估计 :math:`D^{-1} A` 的最大特征值.
    n_iter : 幂迭代次数, 即 matvec 次数.

    Returns
    -------
    lam : 最后一步的 Rayleigh 商 :math:`v^{\\mathsf T} A v / v^{\\mathsf T} D v`. 它是
        :math:`D^{-1/2} A D^{-1/2}` 的 Rayleigh 商, 因此不超过真值, 不会因 :math:`D^{-1} A`
        非正规而高估.

    Notes
    -----
    初值取确定性的伪随机向量 :math:`\\mathrm{frac}(43758.5453 \\sin(12.9898\\, i)) - 1/2`,
    含各频率分量且不依赖随机数发生器, 同一输入的估计逐位可重复.
    """
    if n_iter < 1:
        raise ValueError(f"n_iter 须为正整数, 得到 {n_iter!r}")
    n = op.shape[0]
    context = bm.context(diag) if diag is not None else dict(dtype=bm.float64)
    index = bm.arange(n, **context)
    v = 43758.5453 * bm.sin(12.9898 * index)
    v = v - bm.floor(v) - 0.5
    lam = 0.0
    for _ in range(n_iter):
        w = op @ v
        scaled = v if diag is None else diag * v
        lam = float(bm.sum(v * w) / bm.sum(v * scaled))
        z = w if diag is None else w / diag
        v = z / bm.linalg.norm(z)

    return lam


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
