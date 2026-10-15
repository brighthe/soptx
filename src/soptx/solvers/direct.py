"""稀疏直接法: 一次性求解的 :func:`spsolve` 与缓存分解的 :class:`DirectSolver`.

两者都在主机上用 scipy (SuperLU) 或 MUMPS 分解. 系数矩阵与右端项可以在任意设备上:
入口处经 ``to_scipy`` 与 ``bm.to_numpy`` 转到主机, 解放回右端项所在的设备. 3D 主线里
:class:`DirectSolver` 主要作几何多重网格的最粗层求解器, 规模小, 主机与设备间的往返
开销可以忽略.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from soptx.backend import backend_manager as bm
from soptx.backend import TensorLike

from .base import CAP_MATRIX, LinearSolver, SolveInfo
from .registry import register

# MUMPS 以 INFOG(1) = -8 / -9 报整型 / 实型工作数组不足, 两者都由 ICNTL(14)
# (工作空间相对分析期估计的放大百分比, MUMPS 默认 20) 控制.
_MUMPS_WORKSPACE_ERRORS = (-8, -9)
# 依次尝试的 ICNTL(14); 首次用 MUMPS 默认值 (None), 只在工作空间不足时才放大,
# 使原本能分解的系统保持原有主元顺序与数值.
_MUMPS_ICNTL14_SCHEDULE = (None, 100, 400, 1600)


def _as_scipy(A):
    """soptx.sparse 稀疏张量转 scipy; 已是 scipy 稀疏矩阵时原样返回."""
    return A.to_scipy() if hasattr(A, "to_scipy") else A


def _import_dmumps_context():
    """导入 PyMUMPS 的 ``DMumpsContext``; 未安装时给出安装与替代提示.

    Returns
    -------
    type
        ``mumps.DMumpsContext``.

    Raises
    ------
    ImportError
        未安装 PyMUMPS, 或其依赖的 MUMPS 共享库加载失败.
    """
    try:
        from mumps import DMumpsContext
    except ImportError as exc:
        raise ImportError(
            "MUMPS 后端需要 PyMUMPS 与 MUMPS 库: 推荐 "
            "`conda install -c conda-forge pymumps`, 或 `pip install -e \".[mumps]\"` "
            "(需预装系统 MUMPS); 也可改用 'scipy' 后端."
        ) from exc
    return DMumpsContext


def _mumps_solve(A, b, sym: int = 0):
    """用 MUMPS 求解线性方程组.

    Parameters
    ----------
    A : COOTensor, CSRTensor or scipy sparse matrix
        系数矩阵; 在设备上时经 ``to_scipy`` 转到主机.
    b : TensorLike
        右端项.
    sym : int, default 0
        MUMPS 对称性标志: 0 一般非对称, 1 对称正定, 2 一般对称.
        取 1 或 2 时只把下三角部分传给 MUMPS, 分解的存储与浮点开销
        约减半.

    Returns
    -------
    numpy.ndarray
        解向量.

    Raises
    ------
    ValueError
        ``sym`` 不是 0, 1, 2 之一.
    ImportError
        未安装 PyMUMPS, 见 :func:`_import_dmumps_context`.

    Notes
    -----
    对实际不对称的矩阵传 ``sym`` 1 或 2 会静默解另一个方程组:
    严格上三角部分被直接丢弃, 不做校验.

    分析, 分解与回代以 ``job=6`` 一次做完, 不缓存分解. 安装方式见
    ``docs/solvers/direct.md`` 的"依赖"一节.
    """
    DMumpsContext = _import_dmumps_context()
    A = _as_scipy(A)
    # set_rhs 原地改写并把解写回同一块内存, 须给一份连续的 float64 副本
    x = np.array(bm.to_numpy(b), dtype=np.float64, order="C")

    if sym not in (0, 1, 2):
        raise ValueError(f"MUMPS 的 sym 须为 0, 1 或 2, 得到 {sym!r}")
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
    soptx.sparse 张量共享内存的视图; 对已经是 CSR 的 scipy 矩阵, ``tocsr()`` 默认
    也不复制. 两者叠加会让调用方持有的 K 在求解后被破坏, 而调用方之后往往还要
    用它 (算真残差, 再次求解等). 因此转 CSR 与保护性复制统一在本函数内完成, 调用方
    不必再自行 ``to_scipy().tocsr()`` 或预先缓存副本.

    每次调用都重新分解; 同一矩阵多次求解应改用 :class:`DirectSolver`.
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
        解向量, 与 ``b`` 在同一设备上.

    Raises
    ------
    ValueError
        ``solver`` 不是 'scipy' 或 'mumps'.
    ImportError
        ``solver='mumps'`` 而未安装 PyMUMPS.

    Notes
    -----
    每次调用都重新分解, 适合一次性求解; 同一矩阵多次求解应改用 :class:`DirectSolver`.
    """
    if solver == "mumps":
        x = _mumps_solve(A, b, sym=sym)
    elif solver == "scipy":
        x = _scipy_solve(A, b)
    else:
        raise ValueError(f"未知的直接法后端: {solver!r}; 可选 'scipy' 或 'mumps'")

    return bm.tensor(x, device=bm.get_device(b))


@register("mumps", backend="mumps")
@register("scipy", backend="scipy")
class DirectSolver(LinearSolver):
    """直接稀疏解法 (scipy / MUMPS) 的 LinearSolver 包装.

    ``requires = {CAP_MATRIX}``: 直接法要做符号分解与数值分解, 必须拿到显式
    稀疏矩阵. 'ea' 层级的 matrix-free 算子在 :meth:`setup` 处即被拒绝, 而不
    是等到求解时才以难懂的属性错误暴露.

    用途: 'fa' 层级的中小规模算例, Hu--Zhang 鞍点系统, 子结构接口系统, 以及
    3D 主线里几何多重网格的最粗层. 3D 嵌套剖分的因子存储是
    :math:`\\mathcal{O}(n^{4/3})`, 大规模细网格不宜直接分解, 走 MGCG.

    相对 :func:`spsolve` 的实质区别是**分解被缓存**: :meth:`setup` 做分解,
    :meth:`solve` 只做回代. 同一矩阵要反复回代时这是必需的: 作多重网格最粗层
    时每个 V 循环回代一次, 作 CG 的预条件子时每步回代一次. 矩阵数值改变 (如
    密度更新) 后调用方须重新 :meth:`setup`.

    两个后端的能力不对称: scipy (SuperLU) 支持二维的多右端项; MUMPS 只支持
    一维右端项, 多右端须逐列调用 :meth:`solve`. MUMPS 的 :meth:`setup` 每次都以
    ``job=4`` 重做符号分析与数值分解, 稀疏结构不变时也不复用符号分析, 见
    ``docs/solvers/direct.md``.

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

    MUMPS 分解报工作空间不足 (INFOG(1) = -8/-9) 时, 按 ICNTL(14) = 100, 400,
    1600 依次放大重试, 用尽仍失败才抛出. 零对角块的鞍点系统 (如 Hu--Zhang
    原生格式) 分解时主元大量延迟, 分析期的工作空间估计在粗网格上可能偏小.
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
            raise ValueError(f"MUMPS 的 sym 须为 0, 1 或 2, 得到 {sym!r}")
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
        """直接法后端, ``'scipy'`` 或 ``'mumps'``."""
        return self._backend

    def setup(self, op) -> "DirectSolver":
        """能力协商后取出矩阵并做分解.

        能力协商失败时旧分解保持不变: 先让基类校验通过, 再释放旧资源.

        Parameters
        ----------
        op : Any
            系数算子, 须能转成稀疏矩阵 (soptx.sparse 张量或 scipy 稀疏矩阵).

        Returns
        -------
        DirectSolver
            返回 self.

        Raises
        ------
        OperatorCapabilityError
            算子给不出显式矩阵, 如 'ea' 层级的 matrix-free 算子.
        ImportError
            MUMPS 后端而未安装 PyMUMPS.
        RuntimeError
            MUMPS 分解失败, 见 :meth:`_mumps_factorize`.
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
            from soptx.core.mpi_runtime import ensure_mpi_initialized

            ensure_mpi_initialized()

            if self._sym != 0:
                # MUMPS 在 SYM=1/2 下只接受下三角部分
                from scipy.sparse import tril

                A_in = tril(A, format="coo")
            else:
                A_in = A

            self._ctx = self._mumps_factorize(A_in)

        return self

    def _mumps_factorize(self, A_in):
        """建 MUMPS 上下文并做分析与数值分解, 工作空间不足时放大 ICNTL(14) 重试.

        Parameters
        ----------
        A_in : scipy sparse matrix
            传给 MUMPS 的矩阵; ``sym != 0`` 时已是下三角部分.

        Returns
        -------
        mumps.DMumpsContext
            已完成分解的上下文.

        Raises
        ------
        ImportError
            未安装 PyMUMPS.
        RuntimeError
            MUMPS 报非工作空间类错误, 或按 ``_MUMPS_ICNTL14_SCHEDULE`` 放大到最后
            一档仍工作空间不足.
        """
        DMumpsContext = _import_dmumps_context()

        last = len(_MUMPS_ICNTL14_SCHEDULE) - 1
        for attempt, icntl14 in enumerate(_MUMPS_ICNTL14_SCHEDULE):
            ctx = DMumpsContext(sym=self._sym)
            try:
                ctx.set_silent()
                if icntl14 is not None:
                    ctx.set_icntl(14, icntl14)
                ctx.set_centralized_sparse(A_in)
                ctx.run(job=4)  # 分析 + 数值分解, 不解
            except BaseException as error:
                # 分解失败也必须释放上下文, 否则 PyMUMPS 析构时告警,
                # 且 MUMPS 侧内存不回收.
                code = int(ctx.id.infog[0]) if isinstance(error, RuntimeError) else 0
                ctx.destroy()
                if code in _MUMPS_WORKSPACE_ERRORS and attempt < last:
                    continue
                raise
            return ctx

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
        # 分解与回代在主机上, 解放回右端项所在的设备, 免得作多重网格最粗层时与 GPU 上的向量混用
        return bm.tensor(x, device=bm.get_device(b)), info

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
