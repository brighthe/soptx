# 移植自 brighthe/fealpy ``fealpy/solver/direct.py`` @ 30ca15599 的 scipy 与
# MUMPS 两条路径 (cupy 路径 soptx 不用, 未移植); sym 对称性标志为 fork 扩展,
# 上游 suanhaitech/fealpy 没有. 此后以 SOPTX 本文件为准演化.
# 相对 fealpy 版的行为差异: (1) A 已是 scipy 稀疏矩阵时直接使用, 不再强求
# ``to_scipy()`` (substructure 接口系统传入的就是 scipy csr); (2) scipy 路径
# 内部统一转 CSR 并复制, 使 SuperLU 的原地改写不会破坏调用方持有的矩阵.

from __future__ import annotations

from typing import Optional

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

from .base import CAP_MATRIX, LinearSolver, SolveInfo
from .registry import register


def _as_scipy(A):
    """FEALPy 稀疏张量转 scipy; 已是 scipy 稀疏矩阵时原样返回."""
    return A.to_scipy() if hasattr(A, "to_scipy") else A


def _mumps_solve(A, b, sym: int = 0):
    """用 MUMPS 求解线性方程组.

    Parameters
    ----------
    A : COOTensor, CSRTensor or scipy sparse matrix
        系数矩阵, 须在 CPU 上.
    b : TensorLike
        右端项.
    sym : int, default 0
        MUMPS 对称性标志: 0 一般非对称, 1 对称正定, 2 一般对称.
        取 1 或 2 时只把下三角部分传给 MUMPS, 分解的存储与浮点开销
        约减半; 默认 0 保持既有调用方行为不变.

    Returns
    -------
    numpy.ndarray
        解向量.

    Notes
    -----
    对实际不对称的矩阵传 ``sym`` 1 或 2 会静默解另一个方程组:
    严格上三角部分被直接丢弃, 不做校验.

    依赖 ``mumps`` 包::

        sudo apt install libmumps64-scotch-dev
        pip3 install PyMUMPS
    """
    from mumps import DMumpsContext
    A = _as_scipy(A)
    x = bm.to_numpy(b).copy()

    if sym not in (0, 1, 2):
        raise ValueError(f"MUMPS sym must be 0, 1 or 2, got {sym}.")
    if sym != 0:
        # MUMPS 在 SYM=1/2 下只接受下三角部分, 上三角若一并传入会被重复计入
        from scipy.sparse import tril
        A = tril(A, format='coo')

    ctx = DMumpsContext(sym=sym)
    ctx.set_silent()
    ctx.set_centralized_sparse(A)

    ctx.set_rhs(x)
    ctx.run(job=6)
    ctx.destroy()
    return x


def _scipy_solve(A, b):
    """用 scipy 的稀疏直接解法 (SuperLU) 求解线性方程组.

    Notes
    -----
    SuperLU 会对输入矩阵做原地的列置换与缩放, 而 ``to_scipy()`` 返回的是与
    FEALPy 张量共享内存的视图; 对已经是 CSR 的 scipy 矩阵, ``tocsr()`` 默认
    也不复制. 两者叠加会让调用方持有的 K 在求解后被破坏 -- 拓扑优化里同一个
    K 还要用于伴随求解. 因此转 CSR 与保护性复制统一在本函数内完成, 调用方不
    必再自行 ``to_scipy().tocsr()`` 或预先缓存副本.
    """
    from scipy.sparse.linalg import spsolve as spsol

    A = _as_scipy(A)
    if hasattr(A, "tocsr"):
        # copy=True: A 已是 CSR 时 tocsr() 默认返回自身, 必须显式要求复制
        A = A.tocsr(copy=True)
    b = bm.to_numpy(b)
    return spsol(A, b)


def spsolve(A, b, solver: str = "scipy", sym: int = 0):
    """用直接法求解线性方程组.

    Parameters
    ----------
    A : COOTensor, CSRTensor or scipy sparse matrix
        系数矩阵.
    b : TensorLike
        右端项.
    solver : {'scipy', 'mumps'}, default 'scipy'
        直接法后端.
    sym : int, default 0
        MUMPS 对称性标志, 见 :func:`_mumps_solve`; scipy 后端忽略.

    Returns
    -------
    TensorLike
        解向量.
    """
    if solver == "mumps":
        return bm.tensor(_mumps_solve(A, b, sym=sym))
    elif solver == "scipy":
        return bm.tensor(_scipy_solve(A, b))
    else:
        raise ValueError(f"Unknown solver: {solver}")


@register("mumps", backend="mumps")
@register("scipy", backend="scipy")
class DirectSolver(LinearSolver):
    """直接稀疏解法 (scipy / MUMPS) 的 LinearSolver 包装.

    ``requires = {CAP_MATRIX}``: 直接法要做符号分解与数值分解, 必须拿到显式
    稀疏矩阵. 'ea' 层级的 matrix-free 算子在 :meth:`setup` 处即被拒绝, 而不
    是等到求解时才以难懂的属性错误暴露.

    在 3D 主线里本类只出现在多重网格最粗层: 3D 嵌套剖分的因子存储是
    :math:`\\mathcal{O}(n^{4/3})`, 直接分解细网格不可行.

    相对 :func:`spsolve` 的实质区别是**分解被缓存**: :meth:`setup` 做分解,
    :meth:`solve` 只做回代. 拓扑优化每步要用同一个 K 解状态方程和伴随方程,
    走 :func:`spsolve` 会分解两次. 密度更新后调用方须重新 :meth:`setup`.

    MUMPS 路径自行调用 ``ensure_mpi_initialized``, 调用方不必再各自准备 MPI
    上下文.

    Parameters
    ----------
    backend : {'scipy', 'mumps'}, default 'scipy'
        直接法后端.
    sym : int, default 0
        MUMPS 对称性标志, 见 :func:`_mumps_solve`; scipy 后端忽略.
    residual_tol : float, default 1e-8
        真残差判据阈值, 见 Notes.

    Notes
    -----
    直接法没有迭代收敛的概念, 但"分解成功"不等于"解可信": 接近奇异的系统下
    SuperLU 与 MUMPS 都可能不报错地给出垃圾解. 因此每次 :meth:`solve` 后按
    :math:`\\|b - Ax\\| / \\|b\\|` 算一次真残差填进 ``relres``, 并以
    ``residual_tol`` 判定 ``converged``. 这一次 matvec 相对分解开销可忽略.

    本类持有 MUMPS 上下文这一非 Python 资源. 重复 :meth:`setup`、:meth:`close`
    与析构都会释放它; 也可用 ``with`` 语句管理生命周期.
    """

    requires = frozenset({CAP_MATRIX})

    def __init__(
        self,
        backend: str = "scipy",
        *,
        sym: int = 0,
        residual_tol: float = 1e-8,
    ) -> None:
        super().__init__()
        if backend not in ("scipy", "mumps"):
            raise ValueError(
                f"未知的直接法后端: {backend!r}; 可选 'scipy' 或 'mumps'"
            )
        if sym not in (0, 1, 2):
            raise ValueError(f"MUMPS sym must be 0, 1 or 2, got {sym}.")
        self._backend = backend
        self._sym = sym
        self._residual_tol = residual_tol
        # 全矩阵引用, 只用于算真残差; sym != 0 时传给 MUMPS 的是下三角,
        # 不能拿它当 A 用.
        self._A = None
        self._lu = None      # scipy SuperLU 分解
        self._ctx = None     # MUMPS 上下文

    @property
    def backend(self) -> str:
        return self._backend

    def setup(self, op) -> "DirectSolver":
        """能力协商后取出矩阵并做分解.

        能力协商失败时旧分解保持不变: 先让基类校验通过, 再释放旧资源.
        """
        super().setup(op)
        self._release()

        A = _as_scipy(op)
        self._A = A

        if self._backend == "scipy":
            from scipy.sparse.linalg import splu

            # SuperLU 要 CSC; copy=True 同 _scipy_solve, 防止分解过程原地
            # 改写调用方持有的矩阵 (to_scipy() 返回的是共享内存的视图).
            self._lu = splu(A.tocsc(copy=True))
        else:
            from mumps import DMumpsContext

            from soptx.core.mpi_runtime import ensure_mpi_initialized

            ensure_mpi_initialized()

            if self._sym != 0:
                # MUMPS 在 SYM=1/2 下只接受下三角部分
                from scipy.sparse import tril

                A_in = tril(A, format="coo")
            else:
                A_in = A

            ctx = DMumpsContext(sym=self._sym)
            try:
                ctx.set_silent()
                ctx.set_centralized_sparse(A_in)
                ctx.run(job=4)  # 分析 + 数值分解, 不解
            except BaseException:
                # 分解失败也必须释放上下文, 否则 PyMUMPS 析构时告警,
                # 且 MUMPS 侧内存不回收.
                ctx.destroy()
                raise
            self._ctx = ctx

        return self

    def _solve(
        self, b: TensorLike, x0: Optional[TensorLike] = None
    ) -> "tuple[TensorLike, SolveInfo]":
        # 直接法一步给出解, 与初值无关, 故忽略 x0.
        if self._lu is None and self._ctx is None:
            raise RuntimeError(
                f"{type(self).__name__} 尚未 setup, 请先调用 setup(op)"
            )

        b_np = bm.to_numpy(b)

        if self._backend == "scipy":
            x = self._lu.solve(b_np)
        else:
            if b_np.ndim != 1:
                raise NotImplementedError(
                    "MUMPS 后端只支持一维右端项; 多右端请逐列调用 solve, "
                    "分解已缓存, 不会重复分解"
                )
            # set_rhs 原地改写并把解写回同一块内存, 必须给一份连续的
            # float64 副本, 不能直接把调用方的 b 交出去.
            x = np.array(b_np, dtype=np.float64, order="C")
            self._ctx.set_rhs(x)
            self._ctx.run(job=3)  # 只回代

        relres = self._relative_residual(b_np, x)
        info = {
            "niter": 1,
            "relres": relres,
            "converged": bool(relres <= self._residual_tol),
        }
        return bm.tensor(x), info

    def _relative_residual(self, b_np, x) -> float:
        """真残差 ``||b - A x|| / ||b||``; 2D 右端按 Frobenius 范数."""
        b_norm = float(np.linalg.norm(b_np))
        if b_norm == 0.0:
            return 0.0
        return float(np.linalg.norm(b_np - self._A @ x) / b_norm)

    def _release(self) -> None:
        """释放已缓存的分解与 MUMPS 上下文."""
        if self._ctx is not None:
            self._ctx.destroy()
            self._ctx = None
        self._lu = None

    def close(self) -> None:
        """显式释放分解占用的资源; 之后须重新 :meth:`setup` 才能求解."""
        self._release()
        self._A = None
        self._op = None

    def __enter__(self) -> "DirectSolver":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self):
        # 解释器退出时相关模块可能已被拆掉, 析构里不让异常逃逸.
        try:
            self._release()
        except BaseException:
            pass
