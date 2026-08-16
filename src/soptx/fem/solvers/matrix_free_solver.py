"""分布式重叠加权 Krylov 求解器与求解诊断模块.

针对重叠自由度空间 (Overlapping-DOF Spaces) 上的无矩阵算子线性系统, 提供基于重叠加权
内积的共轭梯度法 (CG) 迭代求解器, 以及真残差范数与边界误差后验诊断功能.
所有内积运算均通过 ``dof_comm.dot`` 过滤权重, 使得多 rank 共享的交界面自由度在代数上仅被
精确计数一次; 在单 rank 下该权重退化为 1, 串行与并行复用同一套代码路径.

默认容差与数值界由 :mod:`soptx.numerics` 提供.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from fealpy.backend import backend_manager as bm
from fealpy.solver.cg import cg
from fealpy.typing import TensorLike

from soptx.numerics import (
    DEFAULT_ATOL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RTOL,
    NORM_FLOOR,
    RESIDUAL_REFRESH,
)


@dataclass
class PreparedLinearSystem:
    """施加边界条件后的无矩阵线性系统封装结构体.

    属性:
        operator (Any): 施加边界条件后的系统刚度算子 (支持 ``@`` 矩阵乘法).
        load (TensorLike): 施加边界条件修正后的等效右端载荷向量.
        prescribed (TensorLike): Dirichlet 边界上的指定位移真解向量 (通常用于初始化 x0).
        boundary_dofs (TensorLike): Dirichlet 边界自由度的一维布尔掩码向量.
    """

    operator: Any
    load: TensorLike
    prescribed: TensorLike
    boundary_dofs: TensorLike


def weighted_norm(vector: TensorLike, dof_comm: Any) -> float:
    """计算分布式重叠自由度空间上的消除重复计数加权 2-范数.

    参数:
        vector (TensorLike): 当前 rank 的局部解或残差张量.
        dof_comm (EntityMPI | None): 自由度跨进程通信器. 若包含 ``dot`` 接口则走分布式加权,
            否则直接调用单机张量范数.

    返回:
        float: 全局无重复计数的加权欧氏范数 :math:`\\|\\mathbf{v}\\|_w = \\sqrt{(\\mathbf{v}, \\mathbf{v})_w}`.
    """
    if hasattr(dof_comm, "dot"):
        _dot, norm_fn = dof_comm.dot(vector.shape[0])
        return norm_fn(vector)
    return float(bm.linalg.norm(vector))


def solver_diagnostics(
    system: PreparedLinearSystem,
    solution: TensorLike,
    dof_comm: Any,
    cg_info: dict[str, Any],
) -> dict[str, Any]:
    """计算解的真实代数残差范数、相对残差以及 Dirichlet 边界误差指标.

    参数:
        system (PreparedLinearSystem): 待求解的线性系统对象.
        solution (TensorLike): 求解器输出的局部自由度解向量.
        dof_comm (EntityMPI | None): 自由度跨进程通信器.
        cg_info (dict[str, Any]): CG 迭代求解器返回的状态信息字典.

    返回:
        dict[str, Any]: 包含迭代步数、真残差、相对残差、边界绝对/相对误差等收敛诊断字典.
    """
    residual = system.operator @ solution - system.load
    residual_norm = weighted_norm(residual, dof_comm)
    load_norm = weighted_norm(system.load, dof_comm)

    boundary_error = bm.where(
        system.boundary_dofs,
        solution - system.prescribed,
        bm.zeros_like(solution),
    )
    boundary_reference = bm.where(
        system.boundary_dofs,
        system.prescribed,
        bm.zeros_like(system.prescribed),
    )
    boundary_absolute = weighted_norm(boundary_error, dof_comm)
    boundary_reference_norm = weighted_norm(boundary_reference, dof_comm)

    return {
        "name": "matrix-free-weighted-cg",
        "converged": bool(cg_info["converged"]),
        "iterations": int(cg_info["niter"]),
        "reported_residual": float(
            cg_info.get("true_residual") or cg_info.get("residual", 0.0)
        ),
        "recursive_residual": float(cg_info.get("recursive_residual", 0.0)),
        "true_absolute_residual": residual_norm,
        "rhs_norm": load_norm,
        "true_relative_residual": (
            residual_norm / max(load_norm, NORM_FLOOR)
        ),
        "boundary_absolute_error": boundary_absolute,
        "boundary_relative_error": (
            boundary_absolute / max(boundary_reference_norm, NORM_FLOOR)
        ),
        "breakdown": cg_info.get("breakdown"),
    }


def weighted_cg(
    operator: Any,
    load: TensorLike,
    *,
    dof_comm: Any,
    x0: Optional[TensorLike] = None,
    maxiter: int = DEFAULT_MAX_ITERATIONS,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    residual_refresh: int = RESIDUAL_REFRESH,
) -> tuple[TensorLike, dict[str, Any]]:
    """基于重叠加权内积的分布式共轭梯度法 (PCG/CG) 迭代求解器.

    参数:
        operator (Any): 线性刚度算子 (如 ``ElasticityEAOperator`` 或 ``DirichletBCOperator``).
        load (TensorLike): 局部右端项载荷向量.
        dof_comm (EntityMPI | None): 自由度通信器 (提供重叠加权点积 ``dot``).
        x0 (TensorLike | None, 可选): 初始猜测解向量. 默认值为 None (全零或由 CG 初始化).
        maxiter (int, 可选): 最大允许 CG 迭代步数. 默认采用 ``DEFAULT_MAX_ITERATIONS``.
        rtol (float, 可选): 相对残差停机容差. 默认采用 ``DEFAULT_RTOL``.
        atol (float, 可选): 绝对残差停机容差. 默认采用 ``DEFAULT_ATOL``.
        residual_refresh (int, 可选): 定期重新计算真实代数残差以抑制舍入累积误差的步数间隔.

    返回:
        tuple[TensorLike, dict[str, Any]]: 包含解向量 ``solution`` 与 CG 运行状态字典 ``info`` 的二元组.
    """
    if hasattr(dof_comm, "dot"):
        dot_fn, _ = dof_comm.dot(int(load.shape[0]))
    else:
        def dot_fn(x: TensorLike, y: TensorLike) -> float:
            return float(bm.sum(x * y))

    if x0 is not None:
        x0 = bm.asarray(x0)

    solution, info = cg(
        operator,
        load,
        x0=x0,
        dot_product=dot_fn,
        residual_refresh=residual_refresh,
        atol=atol,
        rtol=rtol,
        maxit=maxiter,
        returninfo=True,
    )
    return solution, info


def solve_matrix_free_system(
    system: PreparedLinearSystem,
    dof_comm: Any,
    *,
    maxiter: int = DEFAULT_MAX_ITERATIONS,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> tuple[TensorLike, dict[str, Any]]:
    """求解已准备好的无矩阵线性系统, 并一键返回局部解向量与全套收敛诊断报告.

    参数:
        system (PreparedLinearSystem): 待求解的无矩阵系统结构体 (包含刚度算子、载荷、指定位移与边界标记).
        dof_comm (EntityMPI | None): 自由度通信器.
        maxiter (int, 可选): 最大允许迭代步数.
        rtol (float, 可选): 相对残差容差.
        atol (float, 可选): 绝对残差容差.

    返回:
        tuple[TensorLike, dict[str, Any]]: 包含局部解向量 ``solution`` 与诊断字典 ``diagnostics`` 的二元组.
    """
    solution, info = weighted_cg(
        system.operator,
        system.load,
        dof_comm=dof_comm,
        x0=system.prescribed,
        maxiter=maxiter,
        rtol=rtol,
        atol=atol,
    )
    diagnostics = solver_diagnostics(system, solution, dof_comm, info)
    return solution, diagnostics
