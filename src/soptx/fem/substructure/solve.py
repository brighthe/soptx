"""接口系统的约束施加与求解.

直接法支持一般线性约束. CG 支持可等价为齐次固定自由度的约束,
沿用所选张量后端在 CPU 上求解接口系统并验收真实残差.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.linalg import qr
from scipy.sparse import bmat, csr_matrix

from soptx.backend import backend_manager as bm
from soptx.solvers import CGSolver, DiagonalPreconditioner, create
from soptx.sparse import CSRTensor

from .assembler import InterfaceSystem

#: 支持一般线性约束的直接法后端.
DIRECT_BACKENDS = ("scipy", "mumps")


@dataclass(frozen=True)
class ConstrainedSolveResult:
    """一般线性约束接口系统的求解结果."""

    displacement: Any
    constraint_rank: int
    equilibrium_relative_residual: float
    constraint_relative_residual: float
    mode: str
    iterations: Optional[int] = None
    converged: bool = True


def solve_interface_system(
    system: InterfaceSystem,
    load: Any,
    fixed_dofs: Optional[Any] = None,
    *,
    prescribed: Optional[Any] = None,
    solver: str = 'scipy',
) -> Any:
    """在接口系统上施加位移约束并用 soptx.solvers 的 DirectSolver 求解.

    参数:
        system: 已装配的接口系统 (持有 soptx.sparse 的 ``CSRTensor``).
        load: 接口自由度上的右端项, 形状 ``(n_interface,)``. 可由
            ``GlobalAssembler.project_global_vector`` 从全局载荷投影得到.
        fixed_dofs: 受约束的接口自由度编号. 可由
            ``GlobalAssembler.project_global_dofs`` 从全局固定自由度投影得到.
            为 ``None`` 时不施加任何约束.
        prescribed: 接口自由度上的给定位移, 形状 ``(n_interface,)``. 只有
            ``fixed_dofs`` 位置上的分量被采用, 其余分量被忽略. 为 ``None`` 时
            视为齐次约束.
        solver: 底层直接法后端, 取 ``DIRECT_BACKENDS`` 之一, 缺省为 ``'scipy'``.

    返回:
        u: 接口自由度上的位移, 形状 ``(n_interface,)``. 约束自由度取给定值,
            其余自由度为求解结果.

    异常:
        ValueError: 当 ``load`` 或 ``prescribed`` 的长度与接口自由度数不一致时
            抛出.

    说明:
        非齐次约束按 ``K_ff u_f = f_f - (K u_c)_f`` 缩减, 其中 ``u_c`` 是只在约束
        自由度上取给定值, 其余为零的向量.
    """
    n_interface = int(len(system.global_dofs))

    f: Any = bm.asarray(load, dtype=bm.float64)
    if len(f) != n_interface:
        raise ValueError(
            f"load 的长度必须等于接口自由度数 {n_interface}; 当前为 {len(f)}."
        )

    u: Any = bm.zeros((n_interface,), dtype=bm.float64)
    if fixed_dofs is None:
        fixed: Any = bm.zeros((0,), dtype=bm.int64)
    else:
        fixed = bm.unique(bm.asarray(fixed_dofs, dtype=bm.int64))

    if prescribed is not None:
        u_c = bm.asarray(prescribed, dtype=bm.float64)
        if len(u_c) != n_interface:
            raise ValueError(
                f"prescribed 的长度必须等于接口自由度数 {n_interface}; "
                f"当前为 {len(u_c)}."
            )
        u = bm.set_at(u, fixed, u_c[fixed])
        # 给定位移在未约束自由度上产生的反力, 移到右端项.
        f = f - (system.stiffness @ u)

    all_dofs: Any = bm.arange(n_interface, dtype=bm.int64)
    free = all_dofs[bm.isin(all_dofs, fixed, invert=True)]
    if len(free) == 0:
        return u

    free_np = bm.to_numpy(free)
    f_np = bm.to_numpy(f)

    # 提取标准的 CSR 主子矩阵
    if hasattr(system.stiffness, "to_scipy"):
        K_scipy = system.stiffness.to_scipy()
    else:
        K_scipy = system.stiffness

    K_free_scipy = K_scipy[free_np, :][:, free_np].tocsr()

    if solver not in DIRECT_BACKENDS:
        raise ValueError(
            f"未知的直接法后端: {solver!r}; 可选 {DIRECT_BACKENDS}."
        )
    # 接口系统在本函数内只解一次, 分解不跨调用复用; MUMPS 上下文用完即释放,
    # MPI 初始化由 DirectSolver 自己负责, 调用方不必再准备.
    linear_solver = create(solver)
    try:
        u_free, _ = linear_solver.setup(K_free_scipy).solve(f_np[free_np])
    finally:
        linear_solver.close()
    u_free_np = bm.to_numpy(u_free)

    return bm.set_at(u, free, bm.asarray(u_free_np, dtype=bm.float64))



class _FixedDofOperator:
    """通过行列掩码施加齐次支承, 不复制自由子矩阵."""

    def __init__(self, stiffness, fixed):
        self.stiffness = stiffness
        self.fixed = fixed
        self.shape = stiffness.shape

    def __matmul__(self, vector):
        free_vector = bm.copy(vector)
        free_vector[self.fixed] = 0.0
        result = self.stiffness @ free_vector
        result[self.fixed] = vector[self.fixed]
        return result


def _solve_fixed_cg(stiffness, force, matrix, values, *, cg_tol, cg_maxiter,
                    precond, x0):
    """对可等价为齐次固定自由度的约束执行 CPU CG.

    Parameters
    ----------
    stiffness : scipy.sparse.csr_matrix
        对称刚度矩阵, 自由子空间上须正定.
    force : numpy.ndarray
        接口载荷.
    matrix : scipy.sparse.csr_matrix
        约束矩阵.
    values : numpy.ndarray
        约束右端, 须全零.
    cg_tol : float
        相对于自由载荷二范数的真实残差容差.
    cg_maxiter : int
        最大迭代次数.
    precond : str
        "none" 或 "jacobi".
    x0 : TensorLike or None
        上一轮接口位移, 在受约束位置清零.

    Returns
    -------
    ConstrainedSolveResult
        经真实残差验收的位移与迭代信息.
    """
    if not np.isfinite(cg_tol) or not 0.0 < cg_tol < 1.0:
        raise ValueError("cg_tol 必须为 (0, 1) 内的有限数.")
    if cg_maxiter < 1 or int(cg_maxiter) != cg_maxiter:
        raise ValueError("cg_maxiter 必须为正整数.")
    if precond not in ("none", "jacobi"):
        raise ValueError("precond 必须为 none 或 jacobi.")
    if np.any(values != 0.0):
        raise ValueError("CG 目前仅支持齐次固定自由度约束, 非齐次约束请使用直接法.")
    counts = np.diff(matrix.indptr)
    fixed = np.unique(matrix.indices[matrix.indptr[:-1][counts == 1]])
    if not np.all(np.isin(matrix.indices, fixed)):
        raise ValueError("约束不能由单自由度固定行覆盖, 请使用直接法处理一般线性约束.")
    rhs = force.copy()
    rhs[fixed] = 0.0
    scale = float(np.linalg.norm(rhs))
    guess = np.zeros_like(rhs) if x0 is None else np.asarray(bm.to_numpy(x0), dtype=np.float64).copy()
    if guess.shape != rhs.shape or not np.all(np.isfinite(guess)):
        raise ValueError("CG 初值形状错误或含非有限值.")
    guess[fixed] = 0.0
    # from_numpy 保留当前张量后端, 接口系统及迭代向量驻留 CPU.
    operator = _FixedDofOperator(CSRTensor.from_scipy(stiffness), bm.from_numpy(fixed))
    if scale == 0.0:
        q = np.zeros_like(rhs)
        info = dict(niter=0, converged=True)
    else:
        diagonal = stiffness.diagonal().copy()
        diagonal[fixed] = 1.0
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
            raise ValueError("CG 要求消元后的刚度对角线有限且为正.")
        preconditioner = (DiagonalPreconditioner(diag=bm.from_numpy(diagonal))
                          if precond == "jacobi" else None)
        linear_solver = CGSolver(M=preconditioner, rtol=0.0, atol=cg_tol * scale,
                                 maxit=cg_maxiter, norm_type="unpreconditioned")
        displacement, info = linear_solver.setup(operator).solve(
            bm.from_numpy(rhs), x0=bm.from_numpy(guess))
        q = np.asarray(bm.to_numpy(displacement), dtype=np.float64)
    residual = stiffness @ q - force
    residual[fixed] = 0.0
    relative = float(np.linalg.norm(residual)) / max(scale, np.finfo(float).tiny)
    if not info["converged"] or not np.all(np.isfinite(q)) or not np.isfinite(relative) or relative > cg_tol:
        raise RuntimeError(f"CG 未通过真实残差验收: iterations={info['niter']}, "
                           f"relative_residual={relative:.3e}, tolerance={cg_tol:.3e}.")
    constraint_error = float(np.linalg.norm(matrix @ q)) / max(float(np.linalg.norm(q)), np.finfo(float).tiny)
    return ConstrainedSolveResult(
        displacement=bm.from_numpy(q), constraint_rank=len(fixed),
        equilibrium_relative_residual=relative,
        constraint_relative_residual=constraint_error,
        mode="齐次固定自由度 / CG", iterations=int(info["niter"]), converged=True,
    )


def solve_constrained_system(
    system: InterfaceSystem,
    load: Any,
    constraints: Any,
    *,
    prescribed: Optional[Any] = None,
    solver: str = "scipy",
    cg_tol: float = 1.0e-6,
    cg_maxiter: int = 20000,
    precond: str = "jacobi",
    x0: Optional[Any] = None,
) -> ConstrainedSolveResult:
    """求解满足 C u = d 的接口系统.

    Parameters
    ----------
    system : InterfaceSystem
        显式接口刚度矩阵及自由度映射.
    load : TensorLike
        接口载荷, 形状 (n,).
    constraints : sparse matrix
        约束矩阵, 形状 (m, n).
    prescribed : TensorLike, optional
        约束右端, 形状 (m,). 缺省为零.
    solver : str
        scipy, mumps 或 cg.
    cg_tol : float
        CG 相对于自由载荷二范数的真实残差容差.
    cg_maxiter : int
        CG 最大迭代次数.
    precond : str
        CG 预条件子, none 或 jacobi.
    x0 : TensorLike, optional
        CG 初始接口位移, 直接法忽略.

    Returns
    -------
    ConstrainedSolveResult
        位移与残差诊断.

    Notes
    -----
    直接法在检查约束一致性后约化相关行, 一般约束使用 Lagrange 乘子系统.
    CG 仅支持可由单自由度固定行覆盖的齐次约束, 使用对称行列消元.
    CG 不收敛或真实残差未满足容差时抛出 RuntimeError.
    """
    if solver not in (*DIRECT_BACKENDS, "cg"):
        raise ValueError(
            f"未知求解器: {solver!r}; 可选 scipy, mumps, cg."
        )

    n_dofs = int(len(system.global_dofs))
    force = np.asarray(
        bm.to_numpy(bm.asarray(load, dtype=bm.float64)), dtype=np.float64
    )
    if force.ndim != 1 or len(force) != n_dofs:
        raise ValueError(f"load 的形状必须为 ({n_dofs},); 当前为 {force.shape}.")
    if not np.all(np.isfinite(force)):
        raise ValueError("load 必须全部为有限值.")

    if hasattr(constraints, "to_scipy"):
        matrix = constraints.to_scipy().tocsr(copy=True)
    else:
        matrix = csr_matrix(constraints, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != n_dofs:
        raise ValueError(
            f"constraints 的形状必须为 (m, {n_dofs}); 当前为 {matrix.shape}."
        )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.data.size and not np.all(np.isfinite(matrix.data)):
        raise ValueError("constraints 必须全部为有限值.")

    n_constraints = matrix.shape[0]
    if prescribed is None:
        values = np.zeros(n_constraints, dtype=np.float64)
    else:
        values = np.asarray(
            bm.to_numpy(bm.asarray(prescribed, dtype=bm.float64)), dtype=np.float64
        )
        if values.ndim != 1 or len(values) != n_constraints:
            raise ValueError(
                f"prescribed 的形状必须为 ({n_constraints},); 当前为 {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("prescribed 必须全部为有限值.")

    row_counts = np.diff(matrix.indptr)
    zero_rows = row_counts == 0
    if np.any(zero_rows & ~np.isclose(values, 0.0, rtol=0.0, atol=1.0e-12)):
        raise ValueError("零约束行不能对应非零给定值.")
    if np.any(zero_rows):
        keep = ~zero_rows
        matrix = matrix[keep]
        values = values[keep]
        row_counts = row_counts[keep]

    stiffness = (
        system.stiffness.to_scipy().tocsr()
        if hasattr(system.stiffness, "to_scipy")
        else system.stiffness.tocsr()
    )
    if stiffness.shape != (n_dofs, n_dofs):
        raise ValueError(
            f"system.stiffness 的形状必须为 ({n_dofs}, {n_dofs}); "
            f"当前为 {stiffness.shape}."
        )

    if solver == "cg":
        return _solve_fixed_cg(stiffness, force, matrix, values, cg_tol=cg_tol,
                               cg_maxiter=cg_maxiter, precond=precond, x0=x0)

    if matrix.shape[0] == 0:
        displacement = solve_interface_system(
            system, bm.asarray(force, dtype=bm.float64), solver=solver
        )
        q = np.asarray(bm.to_numpy(displacement), dtype=np.float64)
        residual = stiffness @ q - force
        scale = max(float(np.linalg.norm(force)), np.finfo(float).tiny)
        return ConstrainedSolveResult(
            displacement=displacement,
            constraint_rank=0,
            equilibrium_relative_residual=float(np.linalg.norm(residual)) / scale,
            constraint_relative_residual=0.0,
            mode="无约束",
        )

    active = np.unique(matrix.indices)
    dense_active = matrix[:, active].toarray()
    _, triangular, pivots = qr(dense_active.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    threshold = (
        max(matrix.shape[0], len(active))
        * np.finfo(float).eps
        * (diagonal.max() if diagonal.size else 0.0)
    )
    rank = int(np.count_nonzero(diagonal > threshold))
    independent_rows = np.asarray(pivots[:rank], dtype=np.int64)
    independent = matrix[independent_rows]
    independent_values = values[independent_rows]

    coefficients, _, _, _ = np.linalg.lstsq(
        independent[:, active].toarray().T, dense_active.T, rcond=None
    )
    reconstructed_values = coefficients.T @ independent_values
    value_scale = max(float(np.linalg.norm(values)), 1.0)
    if float(np.linalg.norm(reconstructed_values - values)) > 1.0e-10 * value_scale:
        raise ValueError("线性相关约束对应了不一致的给定值.")

    if np.all(row_counts == 1):
        fixed = np.unique(independent.indices)
        prescribed_dofs = np.zeros(n_dofs, dtype=np.float64)
        for row_index in independent_rows:
            start, end = matrix.indptr[row_index:row_index + 2]
            dof = matrix.indices[start]
            prescribed_dofs[dof] = values[row_index] / matrix.data[start]
        displacement = solve_interface_system(
            system,
            bm.asarray(force, dtype=bm.float64),
            bm.asarray(fixed, dtype=bm.int64),
            prescribed=bm.asarray(prescribed_dofs, dtype=bm.float64),
            solver=solver,
        )
        q = np.asarray(bm.to_numpy(displacement), dtype=np.float64)
        free = np.setdiff1d(np.arange(n_dofs), fixed)
        free_internal_force = (stiffness @ q)[free]
        residual = free_internal_force - force[free]
        force_scale = max(
            float(np.linalg.norm(force[free])),
            float(np.linalg.norm((stiffness @ prescribed_dofs)[free])),
            np.finfo(float).tiny,
        )
        mode = "固定角点消元"
    else:
        saddle = bmat(
            [[stiffness, independent.T], [independent, None]], format="csc"
        )
        rhs = np.concatenate((force, independent_values))
        linear_solver = create(solver)
        try:
            solved, _ = linear_solver.setup(saddle).solve(rhs)
        finally:
            linear_solver.close()
        solved_np = np.asarray(bm.to_numpy(solved), dtype=np.float64)
        if not np.all(np.isfinite(solved_np)):
            raise AssertionError("一般线性约束系统产生非有限解.")
        q = solved_np[:n_dofs]
        multipliers = solved_np[n_dofs:]
        internal_force = stiffness @ q
        reaction = independent.T @ multipliers
        residual = internal_force + reaction - force
        force_scale = max(
            float(np.linalg.norm(force)),
            float(np.linalg.norm(internal_force)),
            float(np.linalg.norm(reaction)),
            np.finfo(float).tiny,
        )
        displacement = bm.asarray(q, dtype=bm.float64)
        mode = "一般线性约束 / 稀疏乘子系统"

    if not np.all(np.isfinite(q)):
        raise AssertionError("约束系统产生非有限位移.")
    constraint_residual = matrix @ q - values
    constraint_scale = max(
        float(np.linalg.norm(values)), float(np.linalg.norm(q)), np.finfo(float).tiny
    )
    return ConstrainedSolveResult(
        displacement=displacement,
        constraint_rank=rank,
        equilibrium_relative_residual=float(np.linalg.norm(residual)) / force_scale,
        constraint_relative_residual=(
            float(np.linalg.norm(constraint_residual)) / constraint_scale
        ),
        mode=mode,
    )
