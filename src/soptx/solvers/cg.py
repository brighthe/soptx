# 移植自 brighthe/fealpy ``fealpy/solver/cg.py`` @ 40016dc56, 含上游
# suanhaitech/fealpy 没有的 dot_product / residual_refresh 扩展.
# 此后以 SOPTX 本文件为准演化, 不再跟随 fealpy.solver.cg.
#
# 停机判据已对齐 MFEM ``CGSolver::Mult`` (linalg/solvers.cpp): 参照量取初始
# 残差而非右端项, 判据两边同在迭代内积下度量, 批量右端项逐列独立判定.
# 上游 fealpy 的 ``rtol * ||b||_2`` 口径不再沿用, 原因见 :func:`cg` 的 Notes.
#
# 语义模型借自 PETSc: ``norm_type`` 对应 ``KSPSetNormType``, ``divtol`` 与
# 退出原因 ``ConvergedReason`` 对应 ``KSPConvergedReason``; ``monitor`` 与
# ``print_level`` 对应 MFEM 的 ``IterativeSolverMonitor`` 与 ``PrintLevel``.
# 借的是语义不是代码, 本文件不依赖 PETSc/MFEM.

from itertools import count
from typing import Optional, Callable

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy import logger

# SupportsMatmul 原定义在本文件, 已提到 soptx.protocols (层间边界) 并由
# .base 重导出; 此处再导出一次, 使
# ``from soptx.solvers.cg import SupportsMatmul`` 不断.
from .base import ConvergedReason, LinearSolver, SolveInfo, SupportsMatmul
from .base import reason_text
from .registry import register

#: ``norm_type`` 的合法取值, 命名对齐 PETSc ``KSPNormType``.
NORM_TYPES = ("natural", "unpreconditioned", "preconditioned")


def cg(A: SupportsMatmul, b: TensorLike, x0: Optional[TensorLike]=None,
       M: Optional[SupportsMatmul] = None, *,
       batch_first: bool=False,
       atol: float=1e-12, rtol: float=1e-8,
       maxit: Optional[int]=10000,
       returninfo: bool=False,
       # 借自 PETSc: 判据在哪个范数下度量 + 发散阈值
       norm_type: str='natural',
       divtol: float=1e5,
       # 借自 MFEM IterativeSolver: 残差监视器与打印档位
       monitor: Optional[Callable[[int, float, TensorLike, bool], None]]=None,
       print_level: int=1,
       # 相对 fealpy 上游的扩展: 可插拔内积 + 真残差刷新
       dot_product: Optional[Callable[[TensorLike, TensorLike], float]] = None,
       residual_refresh: int = 0,
       ) -> TensorLike:
    r"""用共轭梯度法 (CG) 求解线性方程组 Ax = b.

    Parameters
    ----------
    A : SupportsMatmul
        线性方程组的系数算子, 假定对称正定 (SPD).
    b : TensorLike
        右端项, 1D 或 2D 稠密张量.
    x0 : TensorLike, optional
        初值, 形状须与 ``b`` 一致; 缺省为零向量.
    M : SupportsMatmul, optional
        预条件子, ``M @ r`` 返回 :math:`M^{-1} r` 意义下的修正残差.
    batch_first : bool, default False
        ``b`` 为 2D 时 batch 维是否在第一维; ``b`` 为 1D 时忽略.
    atol : float, default 1e-12
        绝对收敛容差.
    rtol : float, default 1e-8
        相对收敛容差; 参照量是初始残差 ``||r0||`` 而不是 ``||b||``, 且与
        判据左边同在迭代内积 (有预条件时即 M^-1 内积) 下度量. 见 Notes.
    maxit : int, optional
        最大迭代数, 默认 10000; 传 ``None`` 则迭代到满足任一容差为止.
    returninfo : bool, default False
        为 True 时同时返回收敛信息字典.
    norm_type : {'natural', 'unpreconditioned', 'preconditioned'}, default 'natural'
        停机判据里残差在哪个范数下度量, 命名对齐 PETSc ``KSPNormType``:

        - ``'natural'``: :math:`\sqrt{r^{T} M^{-1} r}`, 即 CG 递推自带的
          量, 每步零额外开销, 与 MFEM ``CGSolver`` 一致 (默认);
        - ``'unpreconditioned'``: :math:`\|r\|_2`;
        - ``'preconditioned'``: :math:`\|M^{-1} r\|_2 = \|z\|_2`.

        后两者每步多一次归约, 不多做 matvec. ``M`` 为 ``None`` 时三者数值
        相同. 该选择同时决定 ``reference_norm`` 与 ``residual`` 的口径.
    divtol : float, default 1e5
        发散阈值 (PETSc ``KSPSetTolerances`` 的 ``divtol``): 判据量相对初值
        放大超过该倍数即判定发散并停机.
    monitor : callable, optional
        残差监视器 ``monitor(iteration, norm, residual, final)``, 签名照
        MFEM ``IterativeSolverMonitor::MonitorResidual``: ``iteration`` 从
        0 (初始残差) 开始, ``norm`` 是 ``norm_type`` 下的报告标量 (批量取
        最难收敛的那列), ``residual`` 是当前残差向量 ``r`` (未经预条件),
        ``final`` 标记停机后的最后一次回调. 缺省 ``None``, 零开销.

        3D 大规模跑一次几十分钟, 没有它就只能等结束才知道收敛曲线.
    print_level : int, default 1
        日志档位, 对应 MFEM ``IterativeSolver::PrintLevel``: ``0`` 全静默,
        ``1`` 只在停机时打一条摘要 (原有行为), ``2`` 另打每步残差.
    dot_product : callable, optional
        自定义内积 ``dot(x, y) -> float``. 缺省用
        ``bm.sum(bm.conj(x) * y, axis=0)``; MPI 分布式求解时传入
        overlap 修正内积 (如 ``entity_mpi.dot(local_size)[0]``).
        仅支持 1D 右端项: 这类内积只返回单个标量, 无法区分 batch
        的各列.
    residual_refresh : int, default 0
        大于 0 时每隔该迭代数 (以及最后一次迭代) 重算真残差
        ``||b - A @ x||`` 并参与收敛判定; 为 0 时只用递推残差
        (标准 CG 行为).

    Returns
    -------
    TensorLike or tuple of (TensorLike, dict)
        近似解; ``returninfo`` 为 True 时附带信息字典, 键含
        ``'residual'``, ``'niter'``, ``'breakdown'``, ``'converged'``,
        ``'true_residual'``, ``'recursive_residual'``, ``'reference_norm'``,
        以及退出原因 ``'reason'`` (:class:`~soptx.solvers.base.ConvergedReason`,
        约定 ``reason > 0`` 等价于 ``converged`` 为 True).

    Notes
    -----
    **停机判据 (对齐 MFEM)**. 判据为

    ``||r_n||_{M^-1}^2 <= max(rtol^2 * ||r_0||_{M^-1}^2, atol^2)``

    逐列成立, 与 MFEM ``CGSolver::Mult`` 的
    ``r0 = max(nom0*rel_tol*rel_tol, abs_tol*abs_tol)`` 一一对应
    (``nom0 = (B r_0, r_0)``). 相对上游 fealpy 有三点不同:

    1. 参照量取初始残差 ``||r_0||`` 而非右端项 ``||b||``. 二者只在
       ``x0 = 0`` 时相等; 热启动 (如 'ea' 路径用 prescribed 作初值) 下
       用 ``||b||`` 会让 rtol 失去标定意义.
    2. 判据两边同范数. 上游左边是 M^-1 内积下的递推残差, 右边是
       ``rtol * ||b||_2``; 开 Jacobi 后 ``diag^-1`` 可达 1e6 量级, 左边
       被放大约三个数量级, 同一个 rtol 在有无预条件时含义不同.
    3. 批量右端项逐列独立判定, 每一列都达到各自的容差才停机, 等价于
       N 次独立的 MFEM 求解; 上游按 Frobenius 聚合成单个标量, 各列
       共用一个以 ``||b||_F`` 为参照的阈值.

    判据在哪个范数下度量由 ``norm_type`` 选择; 默认 ``'natural'`` 与上式
    一致, 行为与 MFEM 相同.

    **退出语义 (对齐 PETSc)**. ``info['reason']`` 给出一个
    :class:`~soptx.solvers.base.ConvergedReason`, 取值与整数编码都照
    ``KSPConvergedReason``: 非正曲率记 ``DIVERGED_INDEFINITE_MAT``, 判据量
    出现 NaN/Inf 记 ``DIVERGED_NANORINF``, 超过 ``divtol`` 记
    ``DIVERGED_DTOL``, maxit 耗尽记 ``DIVERGED_ITS``. 其中非正曲率一律停机,
    这一点跟随 PETSc 而非 MFEM -- MFEM 在 ``den <= 0`` 时只告警, 仅
    ``den == 0`` 才返回; 停机是为了把误用在鞍点系统上的情形暴露出来.

    ``dot_product`` 与 ``residual_refresh`` 是 SOPTX 相对上游
    suanhaitech/fealpy 的扩展, 服务于分布式 overlap 修正内积与
    周期性真残差刷新, MFEM 与 PETSc 的 CG 都没有对应机制.
    """

    assert isinstance(b, TensorLike), "b must be a Tensor"
    if x0 is not None:
        assert isinstance(x0, TensorLike), "x0 must be a Tensor if not None"
    single_vector = b.ndim == 1

    if b.ndim not in {1, 2}:
        raise ValueError("b must be a 1D or 2D dense tensor")

    if x0 is None:
        x0 = bm.zeros_like(b)
    else:
        if x0.shape != b.shape:
            raise ValueError("x0 and b must have the same shape")

    if (dot_product is not None) and (not single_vector):
        raise NotImplementedError(
            "dot_product collapses a batch to one scalar and cannot drive "
            "per-column steps; use a 1-D right-hand side"
        )

    if norm_type not in NORM_TYPES:
        raise ValueError(
            f"norm_type 必须是 {NORM_TYPES} 之一, 得到 {norm_type!r}"
        )

    if divtol <= 0.0:
        raise ValueError(f"divtol 必须为正, 得到 {divtol}")

    if print_level < 0:
        raise ValueError(f"print_level 必须非负, 得到 {print_level}")

    if (not single_vector) and batch_first:
        b = bm.swapaxes(b, 0, 1)
        x0 = bm.swapaxes(x0, 0, 1)

    sol, info = _cg_impl(
        A, b, x0, M,
        atol=atol, rtol=rtol, maxit=maxit,
        dot_product=dot_product,
        residual_refresh=residual_refresh,
        norm_type=norm_type,
        divtol=divtol,
        monitor=monitor,
        print_level=print_level,
    )

    if (not single_vector) and batch_first:
        sol = bm.swapaxes(sol, 0, 1)
    if returninfo is True:
        return sol, info
    else:
        return sol


def _collapse(value) -> float:
    """把逐列量按 Frobenius 聚合成单个标量.

    ``_dot`` 只归约 ``axis=0``, 2D 右端项会留下 ``(batch,)`` 张量.
    这是**报告用**口径, 只服务于 ``relres`` -- 它要与 DirectSolver 的
    ``||b - Ax||_F / ||b||_F`` 可比. 停机判定不走这里, 见
    :func:`_column_squares`.
    """
    if getattr(value, 'ndim', 0) > 0:
        return float(bm.sum(value))
    return float(value)


def _column_squares(value):
    """把 ``_dot`` 的结果规整成逐列的非负平方量.

    MFEM 一次只解一个右端项, 每列有各自的参照量与阈值. 这里保留列维,
    把批量当成 N 次独立的 MFEM 求解: :func:`_cg_impl` 逐列判定、逐列冻
    结, 而不是先聚合成一个标量再比一次.

    ``dot_product`` 只支持一维右端项且返回单个标量, 走 0 维分支, 行为
    与改造前一致.
    """
    v = bm.asarray(value)
    return bm.maximum(v, bm.zeros_like(v))


def _tolerance_squares(reference_squares, atol: float, rtol: float):
    """逐列阈值 ``max(rtol^2 * ||r0||^2, atol^2)``.

    与 MFEM 的 ``r0 = std::max(nom*rel_tol*rel_tol, abs_tol*abs_tol)``
    同构: 全程比较平方量, 不开方.
    """
    floor = bm.zeros_like(reference_squares) + atol * atol
    return bm.maximum((rtol * rtol) * reference_squares, floor)


def _report_norm(squares) -> float:
    """报告用标量: 逐列范数里最大的那个 (最难收敛的列)."""
    return float(bm.max(squares)) ** 0.5


def _cg_impl(A, b, x0, M, atol, rtol, maxit,
             dot_product=None, residual_refresh=0,
             norm_type='natural', divtol=1e5,
             monitor=None, print_level=1):
    """CG 主循环: 逐列独立停机与冻结, 支持自定义内积、真残差刷新与判据范数.

    批量右端项按 N 次独立求解处理: 每列各有自己的阈值、退出原因与冻结时
    刻. 达标或失效的列把 ``alpha`` / ``beta`` 置零, 其 ``x`` 与 ``r`` 从此
    不再变化, 也不再参与曲率与发散判定 -- 否则已收敛列下溢的曲率会把整批
    误判成 breakdown.
    """

    info: dict = {
        'residual': 0.0,         # 递推残差 (向后兼容的键名)
        'niter': 0,
        'breakdown': None,
        'converged': False,
        'true_residual': None,   # 仅 residual_refresh > 0 时填充
        'recursive_residual': 0.0,
        # rtol 的参照量 ||r0||, 供下游门禁与判据对齐; x0 = 0 时等于 ||b||
        'reference_norm': 0.0,
        # 退出原因, 约定 reason > 0 等价于 converged 为 True
        'reason': ConvergedReason.ITERATING,
        # 批量右端项的逐列退出原因; 一维右端项为 None
        'column_reasons': None,
    }

    if dot_product is None:
        def _dot(x, y):
            return bm.sum(bm.conj(x) * y, axis=0)
    else:
        def _dot(x, y):
            return dot_product(x, y)

    def _squares(x):
        return _column_squares(_dot(x, x))

    def _criterion(r, z, rz):
        """当前残差在 ``norm_type`` 指定范数下的逐列平方.

        判据量与 CG 递推自带的 ``rhssq = (r, M^-1 r)`` 是两件事: 后者是算
        法本身要的量, 不可替换; 前者只决定"在哪个范数下比阈值".
        ``natural`` 直接复用 ``rz``, 另两者各多一次归约, 都不多做 matvec.
        """
        if norm_type == 'natural':
            return _column_squares(rz)
        if norm_type == 'unpreconditioned':
            return _squares(r)
        return _squares(z)

    def _criterion_of(r):
        """从残差本身算判据量, 供真残差刷新使用.

        与 :func:`_criterion` 同口径, 因此真残差与递推残差共用同一个阈值
        ``tol_sq`` -- 真残差刷新防的是递推漂移, 与判据取哪个范数正交, 不该
        再引入第二套阈值.
        """
        if norm_type == 'unpreconditioned':
            return _squares(r)
        z = M @ r if M is not None else r
        if norm_type == 'preconditioned':
            return _squares(z)
        return _column_squares(_dot(r, z))

    def _notify(iteration, norm, residual, final):
        """每步残差的出口: print_level 走日志, monitor 走调用方回调.

        对应 MFEM ``IterativeSolverMonitor::MonitorResidual``. ``final``
        的那次不再打日志 -- 摘要由 :func:`_finalize` 负责, 免得末步重复.
        """
        if print_level >= 2 and not final:
            logger.info(f"CG iteration {iteration:4d}: ||r||={norm:.6e}")
        if monitor is not None:
            monitor(iteration, norm, residual, final)

    if _report_norm(_squares(b)) < 1e-15:
        info['converged'] = True
        info['reason'] = ConvergedReason.CONVERGED_ATOL
        solution = bm.zeros_like(b)
        _notify(0, 0.0, solution, True)
        return solution, info

    x = x0
    r = b - A @ x
    z = M @ r if M is not None else r
    p = z
    n_iter = 0
    rhssq = _dot(r, z)
    true_norm = None

    recursive_sq = _criterion(r, z, rhssq)
    reference_sq = recursive_sq
    tol_sq = _tolerance_squares(reference_sq, atol, rtol)
    # divtol 的基准与 PETSc 一致: 判据量相对初值放大 divtol 倍即判发散.
    div_sq = (divtol * divtol) * reference_sq
    recursive_norm = _report_norm(recursive_sq)
    info['reference_norm'] = recursive_norm
    # 监视器从 0 起算, 与 MFEM 一样先报初始残差.
    residual_vector = r
    _notify(0, recursive_norm, residual_vector, False)

    # 逐列状态. reason 码全是小整数, 直接放在与判据量同 dtype 的张量里,
    # 免去各后端整型 dtype 的差异; 0 (ITERATING) 即"仍活跃".
    zeros = bm.zeros_like(reference_sq)
    reasons = zeros + 0.0
    active = reasons == 0.0
    # 收敛列区分 rtol 与 atol: 看 max(rtol^2*ref, atol^2) 里哪一项生效.
    converged_codes = bm.where(
        (rtol * rtol) * reference_sq >= atol * atol,
        zeros + float(ConvergedReason.CONVERGED_RTOL),
        zeros + float(ConvergedReason.CONVERGED_ATOL),
    )

    def _freeze(mask, code, message=None):
        """把 ``mask`` 命中的活跃列冻结并记下退出原因.

        ``code`` 可以是标量码, 也可以是逐列码张量 (收敛列要逐列区分
        rtol / atol). 已冻结的列不会被后来的判定改写.
        """
        nonlocal active, reasons
        hit = active & mask
        if not bool(bm.any(hit)):
            return False
        reasons = bm.where(hit, zeros + code, reasons)
        active = active & (~hit)
        if message is not None and info['breakdown'] is None:
            info['breakdown'] = message
        return True

    def _finalize(sol):
        """把逐列 reason 聚合成 info, 顺带记日志."""
        # 循环跑完仍活跃的列 = maxit 耗尽.
        _freeze(active, float(ConvergedReason.DIVERGED_ITS))

        codes = [ConvergedReason(int(v))
                 for v in bm.reshape(reasons, (-1,))]
        info['converged'] = all(int(c) > 0 for c in codes)
        # min 是一条统一规则: 负码 (失败) 天然排在正码前, 因此聚合值优先
        # 暴露失败; 全部成功时 RTOL(2) 排在 ATOL(3) 前.
        info['reason'] = ConvergedReason(min(int(c) for c in codes))
        if getattr(reasons, 'ndim', 0) > 0:
            info['column_reasons'] = tuple(codes)
        if info['breakdown'] is None and int(info['reason']) < 0:
            info['breakdown'] = reason_text(info['reason'])

        info['residual'] = float(recursive_norm)
        info['recursive_residual'] = float(recursive_norm)
        info['niter'] = n_iter
        if true_norm is not None:
            info['true_residual'] = float(true_norm)

        if print_level >= 1:
            if info['converged']:
                logger.info(f"CG: converged in {n_iter} iterations, "
                            f"||r||={recursive_norm:.6e} <= "
                            f"max(rtol*||r0||, atol).")
            elif info['reason'] == ConvergedReason.DIVERGED_ITS:
                logger.info(f"CG: failed, stopped by maxit ({maxit}).")
            else:
                logger.info(f"CG: stopped at iteration {n_iter}, "
                            f"reason={info['reason'].name}.")

        _notify(n_iter, recursive_norm, residual_vector, True)

        return sol, info

    _freeze(~bm.isfinite(recursive_sq),
            float(ConvergedReason.DIVERGED_NANORINF))
    _freeze(recursive_sq <= tol_sq, converged_codes)

    iterations = count(1) if maxit is None else range(1, maxit + 1)

    if bool(bm.any(active)):
        for n_iter in iterations:
            Ap = A @ p
            curvature = bm.asarray(_dot(p, Ap))

            _freeze(~bm.isfinite(curvature),
                    float(ConvergedReason.DIVERGED_NANORINF))

            negative = active & (curvature <= 0.0)
            if bool(bm.any(negative)):
                worst = float(bm.min(bm.where(negative, curvature,
                                              bm.zeros_like(curvature))))
                _freeze(negative,
                        float(ConvergedReason.DIVERGED_INDEFINITE_MAT),
                        f"CG encountered non-positive curvature: "
                        f"{worst:.16e}")

            if not bool(bm.any(active)):
                break

            # 冻结列的 alpha 置 0, 其 x 与 r 不再变化; 除法先用 1 挡住已
            # 冻结列可能为零或为负的曲率.
            safe_curvature = bm.where(active, curvature,
                                      bm.ones_like(curvature))
            alpha = bm.where(active, rhssq / safe_curvature,
                             bm.zeros_like(curvature))

            x = x + alpha * p
            r_new = r - alpha * Ap
            z_new = M @ r_new if M is not None else r_new
            rhssq_new = _dot(r_new, z_new)

            recursive_sq = _criterion(r_new, z_new, rhssq_new)
            recursive_norm = _report_norm(recursive_sq)
            residual_vector = r_new
            _notify(n_iter, recursive_norm, residual_vector, False)

            _freeze(~bm.isfinite(recursive_sq),
                    float(ConvergedReason.DIVERGED_NANORINF))
            _freeze(recursive_sq > div_sq,
                    float(ConvergedReason.DIVERGED_DTOL),
                    f"CG diverged: residual grew beyond divtol "
                    f"({divtol:.3e}) times the initial residual")

            reached = recursive_sq <= tol_sq

            # 真残差刷新: 递推残差在长迭代下会漂移, 定期用 b - A @ x 校正.
            # 与递推残差同范数, 因此共用 tol_sq.
            refresh = (residual_refresh > 0 and
                       (n_iter % residual_refresh == 0 or
                        bool(bm.any(active & reached)) or
                        n_iter == maxit))

            if refresh:
                true_sq = _criterion_of(b - A @ x)
                true_norm = _report_norm(true_sq)
                info['true_residual'] = float(true_norm)
                # 真残差只能额外促成收敛, 不能否决递推残差已给出的判定.
                reached = reached | (bm.isfinite(true_sq) &
                                     (true_sq <= tol_sq))

            _freeze(reached, converged_codes)

            if not bool(bm.any(active)):
                break

            # 非正的 rhssq 会让 beta = rhssq_new / rhssq 失去意义.
            _freeze(bm.asarray(rhssq) <= 0.0,
                    float(ConvergedReason.DIVERGED_BREAKDOWN),
                    "CG encountered non-positive residual norm")

            if not bool(bm.any(active)):
                break

            safe_rhssq = bm.where(active, bm.asarray(rhssq),
                                  bm.ones_like(curvature))
            beta = bm.where(active, rhssq_new / safe_rhssq,
                            bm.zeros_like(curvature))

            p = z_new + beta * p
            r, z, rhssq = r_new, z_new, rhssq_new

    return _finalize(x)



@register("cg")
class CGSolver(LinearSolver):
    """共轭梯度法的 LinearSolver 包装, 3D 主线的外层 Krylov.

    ``requires`` 为空: 只需算子支持 ``@``, 因此 'fa' 的稀疏矩阵与 'ea' 的
    matrix-free 算子都能用 -- 这是 3D 大规模主线选它作外层的根本原因.

    ``M`` 位接任意 :class:`LinearSolver` 或任何支持 ``@`` 的对象: 对角预条件
    子, 多重网格, 乃至另一个 Krylov. 求解器位与预条件子位是两个独立自由度,
    组合合法性由调用点负责, 包内不预设.

    要求算子对称正定. 鞍点系统请用 :class:`~soptx.solvers.minres.MINRESSolver`.

    Parameters
    ----------
    M : SupportsMatmul, optional
        预条件子, ``M @ r`` 返回 :math:`M^{-1} r`.
    atol, rtol : float
        绝对与相对收敛容差.
    maxit : int, optional
        最大迭代数; ``None`` 表示不设上限.
    batch_first : bool, default False
        二维右端项的 batch 维是否在第一维.
    norm_type : {'natural', 'unpreconditioned', 'preconditioned'}, default 'natural'
        停机判据里残差在哪个范数下度量; 见 :func:`cg`.
    divtol : float, default 1e5
        发散阈值; 见 :func:`cg`.
    monitor : callable, optional
        残差监视器, 签名照 MFEM ``MonitorResidual``; 见 :func:`cg`.
    print_level : int, default 1
        日志档位 (0 静默 / 1 摘要 / 2 每步); 见 :func:`cg`.
    dot_product : callable, optional
        自定义内积, 分布式求解时传 overlap 修正内积; 见 :func:`cg`.
        只支持一维右端项.
    residual_refresh : int, default 0
        真残差刷新间隔; 见 :func:`cg`.

    Notes
    -----
    ``_solve`` 转调 :func:`cg`, 保留它的全部诊断键 (``breakdown``,
    ``true_residual``, ``recursive_residual`` 等), 另填基类要求的 ``relres``.

    ``relres`` 按真残差 :math:`\\|b - Ax\\| / \\|b\\|` 现算, 而不是拿 ``cg``
    的 ``residual`` 换算 -- 后者在有预条件时是 :math:`M^{-1}` 内积意义下的
    量, 与 :class:`~soptx.solvers.direct.DirectSolver` 的 ``relres`` 不可比.
    代价是一次额外 matvec, 相对整个迭代过程可忽略.

    注意 ``relres`` 是**报告量**, 与停机判据不是同一个口径: 判据按 MFEM
    逐列比 :math:`\\|r\\|_{M^{-1}}` 与 :math:`\\mathrm{rtol}\\|r_0\\|_{M^{-1}}`,
    ``relres`` 则按 Frobenius 聚合并以 :math:`\\|b\\|` 为分母, 以保持与
    ``DirectSolver`` 可比. 参照量 :math:`\\|r_0\\|` 另由 ``reference_norm``
    给出.
    """

    requires = frozenset()

    def __init__(
        self,
        M: Optional[SupportsMatmul] = None,
        *,
        atol: float = 1e-12,
        rtol: float = 1e-8,
        maxit: Optional[int] = 10000,
        batch_first: bool = False,
        norm_type: str = 'natural',
        divtol: float = 1e5,
        monitor: Optional[
            Callable[[int, float, TensorLike, bool], None]] = None,
        print_level: int = 1,
        dot_product: Optional[Callable[[TensorLike, TensorLike], float]] = None,
        residual_refresh: int = 0,
    ) -> None:
        super().__init__()
        if residual_refresh < 0:
            raise ValueError(
                f"residual_refresh 必须非负, 得到 {residual_refresh}"
            )
        if norm_type not in NORM_TYPES:
            raise ValueError(
                f"norm_type 必须是 {NORM_TYPES} 之一, 得到 {norm_type!r}"
            )
        if divtol <= 0.0:
            raise ValueError(f"divtol 必须为正, 得到 {divtol}")
        if print_level < 0:
            raise ValueError(f"print_level 必须非负, 得到 {print_level}")
        self._M = M
        self._atol = atol
        self._rtol = rtol
        self._maxit = maxit
        self._batch_first = batch_first
        self._norm_type = norm_type
        self._divtol = divtol
        self._monitor = monitor
        self._print_level = print_level
        self._dot_product = dot_product
        self._residual_refresh = residual_refresh

    @property
    def M(self) -> Optional[SupportsMatmul]:
        return self._M

    def setup(self, op) -> "CGSolver":
        """绑定算子, 并把尚未 setup 的预条件子一并绑到同一个算子上.

        已经 setup 过的预条件子不动: 多重网格这类预条件子的层次由 fem 侧
        建好后传入, 它绑的算子未必是外层 Krylov 的这一个.
        """
        super().setup(op)
        if isinstance(self._M, LinearSolver) and not self._M.is_setup:
            self._M.setup(op)
        return self

    def _solve(
        self, b: TensorLike, x0: Optional[TensorLike] = None
    ) -> "tuple[TensorLike, SolveInfo]":
        x, info = cg(
            self.op, b, x0,
            M=self._M,
            batch_first=self._batch_first,
            atol=self._atol,
            rtol=self._rtol,
            maxit=self._maxit,
            returninfo=True,
            norm_type=self._norm_type,
            divtol=self._divtol,
            monitor=self._monitor,
            print_level=self._print_level,
            dot_product=self._dot_product,
            residual_refresh=self._residual_refresh,
        )
        info = dict(info)
        info['relres'] = self._relative_residual(b, x)
        return x, info

    def _relative_residual(self, b: TensorLike, x: TensorLike) -> float:
        """真相对残差, 用与迭代同一个内积度量."""
        if self._dot_product is None:
            def _dot(u, v):
                return bm.sum(bm.conj(u) * v, axis=0)
        else:
            def _dot(u, v):
                return self._dot_product(u, v)

        def _norm(u) -> float:
            return max(_collapse(_dot(u, u)), 0.0) ** 0.5

        # 算子按自由度维在前的布局作用, batch_first 下须先转回来.
        if b.ndim == 2 and self._batch_first:
            b = bm.swapaxes(b, 0, 1)
            x = bm.swapaxes(x, 0, 1)

        b_norm = _norm(b)
        # 与 cg 内部同一判据: 右端为零时直接返回零解, 相对残差记 0.
        if b_norm < 1e-15:
            return 0.0
        return float(_norm(b - self.op @ x) / b_norm)
