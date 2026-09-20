"""接口系统的约束施加与直接求解.

``GlobalAssembler`` 只产出接口刚度矩阵与自由度映射, 不携带求解策略. 本模块提供
一个与装配器解耦的自由函数, 把 "施加位移约束 + 稀疏直接求解" 这一步固定下来,
避免各算例脚本各写一遍自由度取补集与子矩阵切片.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.linalg import qr
from scipy.sparse import bmat, csr_matrix

from fealpy.backend import backend_manager as bm
from soptx.solvers import create

from .assembler import InterfaceSystem

#: 本函数支持的直接法后端。CG 一类迭代法不在此列: 接口系统规模小且要求一次
#: 给准, 迭代法在这里没有收益; 旧实现对未知名字会静默回落到 scipy, 现在报错。
DIRECT_BACKENDS = ("scipy", "mumps")


@dataclass(frozen=True)
class ConstrainedSolveResult:
    """一般线性约束接口系统的求解结果."""

    displacement: Any
    constraint_rank: int
    equilibrium_relative_residual: float
    constraint_relative_residual: float
    mode: str


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
        system: 已装配的接口系统 (持有 FEALPy ``CSRTensor``).
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
    # MPI 初始化由 DirectSolver 自己负责, 调用方不必再准备。
    linear_solver = create(solver)
    try:
        u_free, _ = linear_solver.setup(K_free_scipy).solve(f_np[free_np])
    finally:
        linear_solver.close()
    u_free_np = bm.to_numpy(u_free)

    return bm.set_at(u, free, bm.asarray(u_free_np, dtype=bm.float64))


def solve_constrained_system(
    system: InterfaceSystem,
    load: Any,
    constraints: Any,
    *,
    prescribed: Optional[Any] = None,
    solver: str = "scipy",
) -> ConstrainedSolveResult:
    """求解满足 ``C u = d`` 的接口系统.

    零行且右端为零的约束被忽略; 重复或一般线性相关行在检查给定值一致后
    约化为独立约束. 坐标选择约束复用 ``solve_interface_system`` 消元,
    混合约束通过稀疏 Lagrange 乘子系统求解. 两条路径均使用 SOPTX 注册的
    ``DirectSolver``.

    参数:
        system: 显式接口刚度矩阵及其自由度映射.
        load: 接口载荷, 形状 ``(n,)``.
        constraints: 约束矩阵 ``C``, 形状 ``(m, n)``.
        prescribed: 约束右端 ``d``, 形状 ``(m,)``; ``None`` 表示齐次约束.
        solver: 直接法后端, 取 ``DIRECT_BACKENDS`` 之一.
    """
    if solver not in DIRECT_BACKENDS:
        raise ValueError(
            f"未知的直接法后端: {solver!r}; 可选 {DIRECT_BACKENDS}."
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
