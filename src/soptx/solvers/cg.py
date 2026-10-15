"""共轭梯度法: 函数式 :func:`cg` 与 :class:`~soptx.solvers.base.LinearSolver` 包装 :class:`CGSolver`.

默认停机判据对齐 MFEM ``CGSolver::Mult``: 参照量取初始残差;
显式设置 rtol_reference='rhs' 时改以实际右端项为参照,
判据两边同在迭代内积下度量, 批量右端项逐列独立判定. 上游 fealpy 的 ``rtol * ||b||_2``
混合范数口径不再沿用, 原因见 :func:`cg` 的 Notes.

语义模型借自 PETSc: ``norm_type`` 对应 ``KSPSetNormType``, ``divtol`` 与退出原因
``ConvergedReason`` 对应 ``KSPConvergedReason``; ``monitor`` 与 ``print_level`` 对应
MFEM 的 ``IterativeSolverMonitor`` 与 ``PrintLevel``. 借的是语义不是代码, 本文件不依赖
PETSc/MFEM.
"""

from itertools import count
from math import isfinite
import logging
from typing import Optional, Callable

from soptx.backend import backend_manager as bm
from soptx.backend import TensorLike

# SupportsMatmul 原定义在本文件, 已提到 soptx.protocols (层间边界) 并由
# .base 重导出; 此处再导出一次, 使
# ``from soptx.solvers.cg import SupportsMatmul`` 不断.
from .base import ConvergedReason, LinearSolver, SolveInfo, SupportsMatmul
from .base import reason_text
from .registry import register

logger = logging.getLogger(__name__)

#: ``norm_type`` 的合法取值, 命名对齐 PETSc ``KSPNormType``.
NORM_TYPES = ("natural", "unpreconditioned", "preconditioned")
RTOL_REFERENCES = ("initial_residual", "rhs")

_INDEFINITE_PC_MESSAGE = "CG encountered an indefinite preconditioner: (r, M^-1 r) < 0"


def cg(A: SupportsMatmul, b: TensorLike, x0: Optional[TensorLike] = None,
       M: Optional[SupportsMatmul] = None, *,
       batch_first: bool = False,
       atol: float = 1e-12, rtol: float = 1e-8,
       maxit: Optional[int] = 10000,
       returninfo: bool = False,
       # 借自 PETSc: 判据在哪个范数下度量 + 发散阈值
       norm_type: str = 'natural',
       rtol_reference: str = 'initial_residual',
       divtol: float = 1e5,
       # 借自 MFEM IterativeSolver: 残差监视器与打印档位
       monitor: Optional[Callable[[int, float, TensorLike, bool], None]] = None,
       print_level: int = 1,
       # 相对 fealpy 上游的扩展: 收敛时用真残差验证
       verify_residual: bool = False,
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
        相对收敛容差; 默认参照量是初始残差 ``||r0||``, 设置 rtol_reference='rhs' 时取 ``||b||``, 且与
        判据左边在同一范数下度量. 热启动时 ``||r0||`` 已经很小, 想要以
        ``||b||`` 为参照的精度目标时设置 ``rtol_reference='rhs'``.
        见 Notes.
    rtol_reference : {'initial_residual', 'rhs'}, default 'initial_residual'
        相对容差参照量. rhs 模式逐列取右端项在 norm_type 下的范数,
        停机精度不随热启动改变; divtol 仍以初始残差为参照.
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
    verify_residual : bool, default False
        为 True 时, 递推残差达到判据或到达最后一次迭代时, 把活跃列的递推残差
        换成真残差 ``b - A @ x`` 并重算预条件残差后再判定; 递推漂移造成的虚假
        收敛因此被否决, 未达标的列以替换后的残差继续迭代. 每次验证多一次 matvec
        与一次预条件子作用. 为 False 时只用递推残差 (标准 CG 行为). 见 Notes.

    Returns
    -------
    TensorLike or tuple of (TensorLike, dict)
        近似解; ``returninfo`` 为 True 时附带信息字典, 键含
        ``'residual'``, ``'niter'``, ``'breakdown'``, ``'converged'``,
        ``'true_residual'``, ``'recursive_residual'``, ``'reference_norm'``,
        以及退出原因 ``'reason'`` (:class:`~soptx.solvers.base.ConvergedReason`,
        约定 ``reason > 0`` 等价于 ``converged`` 为 True) 与批量右端项的逐列退出
        原因 ``'column_reasons'`` (一维右端项为 None).

    Raises
    ------
    TypeError
        ``b`` 或 ``x0`` 不是张量.
    ValueError
        ``b`` 不是一维或二维, ``x0`` 与 ``b`` 形状不同, 或 ``norm_type``、
        ``divtol``、``print_level``、``verify_residual`` 取值不合法.

    Notes
    -----
    **默认停机判据 (对齐 MFEM)**. 判据为

    ``||r_n||_{M^-1}^2 <= max(rtol^2 * ||r_0||_{M^-1}^2, atol^2)``

    逐列成立, 与 MFEM ``CGSolver::Mult`` 的
    ``r0 = max(nom0*rel_tol*rel_tol, abs_tol*abs_tol)`` 一一对应
    (``nom0 = (B r_0, r_0)``). 相对上游 fealpy 有三点不同:

    1. 参照量取初始残差 ``||r_0||`` 而非右端项 ``||b||``. 二者只在
       ``x0 = 0`` 时相等. ``||r_0||`` 参照下 ``rtol`` 表示残差的下降倍数;
       热启动时 ``||r_0||`` 已经很小, 同一个 ``rtol`` 会比以 ``||b||`` 为
       参照更严. 需要与初值无关的精度目标时, 设置 ``rtol_reference='rhs'``.
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
    ``DIVERGED_DTOL``, maxit 耗尽记 ``DIVERGED_ITS``, 预条件子不正定
    (:math:`(r, M^{-1} r) < 0`) 记 ``DIVERGED_INDEFINITE_PC``. 后者须在收敛判定之前
    拦下: ``natural`` 范数下判据量就是 :math:`(r, M^{-1} r)`, 负值截断为零后会被
    误判为收敛. 非正曲率一律停机,
    这一点跟随 PETSc 而非 MFEM -- MFEM 在 ``den <= 0`` 时只告警, 仅
    ``den == 0`` 才返回; 停机是为了把误用在鞍点系统上的情形暴露出来.

    **收敛时的真残差验证**. ``verify_residual`` 是 SOPTX 相对上游
    suanhaitech/fealpy 的扩展, MFEM 与 PETSc 的 CG 都没有对应机制. 它只在
    递推残差宣布达标时 (以及最后一次迭代) 做一次残差替换, 不做周期替换:
    周期性地用真残差替换递推残差会破坏 CG 递推的局部正交性, 病态问题上
    迭代数可增加 15%--30%; 只在达标时替换, 算子精确时迭代轨迹与标准 CG
    完全一致. 代价是不提高可达精度, 只保证报告收敛时真残差确实达标.
    """

    if not isinstance(b, TensorLike):
        raise TypeError(f"b 须为张量, 得到 {type(b).__name__}")
    if x0 is not None and not isinstance(x0, TensorLike):
        raise TypeError(f"x0 须为张量或 None, 得到 {type(x0).__name__}")
    single_vector = b.ndim == 1

    if b.ndim not in {1, 2}:
        raise ValueError(f"b 须为一维或二维稠密张量, 得到 ndim={b.ndim}")

    if x0 is None:
        x0 = bm.zeros_like(b)
    else:
        if x0.shape != b.shape:
            raise ValueError(f"x0 与 b 的形状须一致, 得到 {tuple(x0.shape)} 与 {tuple(b.shape)}")

    if norm_type not in NORM_TYPES:
        raise ValueError(
            f"norm_type 必须是 {NORM_TYPES} 之一, 得到 {norm_type!r}"
        )

    if rtol_reference not in RTOL_REFERENCES:
        raise ValueError(f"rtol_reference 须为 {RTOL_REFERENCES}, 得到 {rtol_reference!r}")
    if not all(isfinite(v) and v >= 0.0 for v in (atol, rtol)):
        raise ValueError("atol 与 rtol 须为有限非负数.")

    if divtol <= 0.0:
        raise ValueError(f"divtol 必须为正, 得到 {divtol}")

    if print_level < 0:
        raise ValueError(f"print_level 必须非负, 得到 {print_level}")

    if not isinstance(verify_residual, bool):
        raise ValueError(f"verify_residual 必须是 bool, 得到 {verify_residual!r}")

    if (not single_vector) and batch_first:
        b = bm.swapaxes(b, 0, 1)
        x0 = bm.swapaxes(x0, 0, 1)

    sol, info = _cg_impl(
        A, b, x0, M,
        atol=atol, rtol=rtol, maxit=maxit,
        verify_residual=verify_residual,
        norm_type=norm_type,
        rtol_reference=rtol_reference,
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
    """
    v = bm.asarray(value)
    return bm.maximum(v, bm.zeros_like(v))


def _tolerance_squares(reference_squares, atol: float, rtol: float):
    """逐列阈值 ``max(rtol^2 * reference_squares, atol^2)``.

    与 MFEM 的 ``r0 = std::max(nom*rel_tol*rel_tol, abs_tol*abs_tol)``
    同构: 全程比较平方量, 不开方.
    """
    floor = bm.zeros_like(reference_squares) + atol * atol
    return bm.maximum((rtol * rtol) * reference_squares, floor)


def _report_norm(squares) -> float:
    """报告用标量: 逐列范数里最大的那个 (最难收敛的列)."""
    return float(bm.max(squares)) ** 0.5


def _cg_impl(A, b, x0, M, atol, rtol, maxit,
             verify_residual=False,
             norm_type='natural', rtol_reference='initial_residual', divtol=1e5,
             monitor=None, print_level=1):
    """CG 主循环: 逐列独立停机与冻结, 支持收敛时的真残差验证与判据范数选择.

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
        'true_residual': None,   # 仅 verify_residual 为 True 且做过验证时填充
        'recursive_residual': 0.0,
        # rtol 的参照量, 由 rtol_reference 和 norm_type 共同决定.
        'reference_norm': 0.0,
        'rtol_reference': rtol_reference,
        'norm_type': norm_type,
        'rhs_norm': 0.0,
        'stopping_threshold': float(atol),
        'initial_residual_norm': 0.0,
        # 退出原因, 约定 reason > 0 等价于 converged 为 True
        'reason': ConvergedReason.ITERATING,
        # 批量右端项的逐列退出原因; 一维右端项为 None
        'column_reasons': None,
    }

    def _dot(x, y):
        return bm.sum(bm.conj(x) * y, axis=0)

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

    def _notify(iteration, norm, residual, final):
        """每步残差的出口: print_level 走日志, monitor 走调用方回调.

        对应 MFEM ``IterativeSolverMonitor::MonitorResidual``. ``final``
        的那次不再打日志 -- 摘要由 :func:`_finalize` 负责, 免得末步重复.
        """
        if print_level >= 2 and not final:
            logger.info(f"CG iteration {iteration:4d}: ||r||={norm:.6e}")
        if monitor is not None:
            monitor(iteration, norm, residual, final)

    rhs_norm = _report_norm(_squares(b))
    info['rhs_norm'] = rhs_norm
    # rhs 模式仅对严格零右端返回零解, 不将极小但非零的载荷误判为零.
    zero_rhs = (not bool(bm.any(b != 0.0)) if rtol_reference == 'rhs'
                else rhs_norm < 1e-15)
    if zero_rhs:
        info['converged'] = True
        if verify_residual:
            info['true_residual'] = 0.0
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
    initial_sq = recursive_sq
    reference_sq = initial_sq
    if rtol_reference == 'rhs':
        if norm_type == 'unpreconditioned' or M is None:
            reference_sq = _squares(b)
        else:
            z_rhs = M @ b
            rhs_product = _dot(b, z_rhs)
            if bool(bm.any(bm.asarray(rhs_product) < 0.0)):
                raise ValueError(_INDEFINITE_PC_MESSAGE)
            reference_sq = _criterion(b, z_rhs, rhs_product)
    tol_sq = _tolerance_squares(reference_sq, atol, rtol)
    # 发散判据始终相对初始残差, 不随相对收敛容差的参考量改变.
    div_sq = (divtol * divtol) * initial_sq
    recursive_norm = _report_norm(recursive_sq)
    info['initial_residual_norm'] = recursive_norm
    info['reference_norm'] = _report_norm(reference_sq)
    info['stopping_threshold'] = _report_norm(tol_sq)
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

        if verify_residual and info['converged'] and true_norm is None:
            # 初值已经通过判据时, r 本身就是直接计算的真残差.
            info['true_residual'] = float(recursive_norm)
        info['residual'] = float(recursive_norm)
        info['recursive_residual'] = float(recursive_norm)
        info['niter'] = n_iter
        if true_norm is not None:
            info['true_residual'] = float(true_norm)

        if print_level >= 1:
            if info['converged']:
                logger.info(f"CG: converged in {n_iter} iterations, "
                            f"||r||={recursive_norm:.6e} <= "
                            f"max(rtol*reference_norm, atol), reference={rtol_reference}.")
            elif info['reason'] == ConvergedReason.DIVERGED_ITS:
                logger.info(f"CG: failed, stopped by maxit ({maxit}).")
            else:
                logger.info(f"CG: stopped at iteration {n_iter}, "
                            f"reason={info['reason'].name}.")

        _notify(n_iter, recursive_norm, residual_vector, True)

        return sol, info

    _freeze(~bm.isfinite(recursive_sq) | ~bm.isfinite(reference_sq),
            float(ConvergedReason.DIVERGED_NANORINF))
    _freeze(bm.asarray(rhssq) < 0.0, float(ConvergedReason.DIVERGED_INDEFINITE_PC),
            _INDEFINITE_PC_MESSAGE)
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

            _freeze(~bm.isfinite(recursive_sq),
                    float(ConvergedReason.DIVERGED_NANORINF))
            _freeze(recursive_sq > div_sq,
                    float(ConvergedReason.DIVERGED_DTOL),
                    f"CG diverged: residual grew beyond divtol "
                    f"({divtol:.3e}) times the initial residual")

            reached = recursive_sq <= tol_sq

            # 收敛时的真残差验证: 递推残差可能与真残差漂开, 有列宣布达标 (或到最后
            # 一次迭代) 时把活跃列的 r 换成 b - A x, 并重算 z 与 (r, z), 再判定.
            # 替换后判据量就是真残差, 虚假达标随之被否决, 未达标的列以替换后的残差
            # 继续迭代. 搜索方向 p 不动; 冻结列的 r 与 z 保持原值. 不做周期替换:
            # 那会破坏递推的局部正交性, 拖慢收敛.
            verify = (verify_residual and
                      (bool(bm.any(active & reached)) or n_iter == maxit))

            if verify:
                r_true = b - A @ x
                z_true = M @ r_true if M is not None else r_true
                rz_true = _dot(r_true, z_true)
                true_norm = _report_norm(_criterion(r_true, z_true, rz_true))
                info['true_residual'] = float(true_norm)
                r_new = bm.where(active, r_true, r_new)
                z_new = bm.where(active, z_true, z_new)
                rhssq_new = _dot(r_new, z_new)
                recursive_sq = _criterion(r_new, z_new, rhssq_new)
                reached = recursive_sq <= tol_sq

            recursive_norm = _report_norm(recursive_sq)
            residual_vector = r_new
            _notify(n_iter, recursive_norm, residual_vector, False)

            # 先拦不正定的预条件子, 再判收敛: natural 范数下负的 (r, M^-1 r) 被截断为零,
            # 会被误判为达标
            _freeze(bm.asarray(rhssq_new) < 0.0,
                    float(ConvergedReason.DIVERGED_INDEFINITE_PC),
                    _INDEFINITE_PC_MESSAGE)
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

    ``M`` 位接任意 :class:`LinearSolver` 或任何支持 ``@`` 的对象, 例如对角
    预条件子、多重网格与直接法. PCG 要求它是固定、对称、正定的线性算子: 按容差
    停止的内层迭代与 Krylov 求解器的输出对右端项是非线性的, 不能填进来, 嵌套
    Krylov 需要柔性 CG (FCG), 本包没有. 组合是否合法由调用点负责, 包内不检查.

    要求算子对称正定. 鞍点系统需要 MINRES 一类的方法, 本包尚未实现.

    Parameters
    ----------
    M : SupportsMatmul, optional
        预条件子, ``M @ r`` 返回 :math:`M^{-1} r`.
    atol, rtol : float
        绝对与相对收敛容差.
    rtol_reference : {'initial_residual', 'rhs'}, default 'initial_residual'
        相对容差参照量. rhs 模式逐列取右端项在 norm_type 下的范数,
        停机精度不随热启动改变; divtol 仍以初始残差为参照.
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
    verify_residual : bool, default False
        收敛时是否用真残差验证; 见 :func:`cg`.

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
    ``DirectSolver`` 可比. 所选参照量另由 ``reference_norm``
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
        rtol_reference: str = 'initial_residual',
        divtol: float = 1e5,
        monitor: Optional[
            Callable[[int, float, TensorLike, bool], None]] = None,
        print_level: int = 1,
        verify_residual: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(verify_residual, bool):
            raise ValueError(
                f"verify_residual 必须是 bool, 得到 {verify_residual!r}"
            )
        if norm_type not in NORM_TYPES:
            raise ValueError(f"norm_type 须为 {NORM_TYPES}, 得到 {norm_type!r}")
        if rtol_reference not in RTOL_REFERENCES:
            raise ValueError(f"rtol_reference 须为 {RTOL_REFERENCES}, 得到 {rtol_reference!r}")
        if not all(isfinite(v) and v >= 0.0 for v in (atol, rtol)):
            raise ValueError("atol 与 rtol 须为有限非负数.")
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
        self._rtol_reference = rtol_reference
        self._divtol = divtol
        self._monitor = monitor
        self._print_level = print_level
        self._verify_residual = verify_residual

    @property
    def M(self) -> Optional[SupportsMatmul]:
        """预条件子; 未给出时为 None."""
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
            rtol_reference=self._rtol_reference,
            divtol=self._divtol,
            monitor=self._monitor,
            print_level=self._print_level,
            verify_residual=self._verify_residual,
        )
        info = dict(info)
        info['relres'] = self._relative_residual(b, x)
        return x, info

    def _relative_residual(self, b: TensorLike, x: TensorLike) -> float:
        """真相对残差 ``||b - A x|| / ||b||``; 2D 右端按 Frobenius 范数."""
        def _dot(u, v):
            return bm.sum(bm.conj(u) * v, axis=0)

        def _norm(u) -> float:
            return max(_collapse(_dot(u, u)), 0.0) ** 0.5

        # 算子按自由度维在前的布局作用, batch_first 下须先转回来.
        if b.ndim == 2 and self._batch_first:
            b = bm.swapaxes(b, 0, 1)
            x = bm.swapaxes(x, 0, 1)

        b_norm = _norm(b)
        # 与 cg 内部同一判据: 右端为零时直接返回零解, 相对残差记 0.
        zero_rhs = (not bool(bm.any(b != 0.0)) if self._rtol_reference == 'rhs'
                    else b_norm < 1e-15)
        if zero_rhs:
            return 0.0
        return float(_norm(b - self.op @ x) / b_norm)
