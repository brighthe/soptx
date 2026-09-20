"""重叠自由度空间 (Overlapping-DOF Spaces) 上的加权内积求解适配层.

多 rank 共享的交界面自由度在每个 rank 上各有一份副本, 普通内积会重复计数.
本模块把 ``dof_comm.dot`` 提供的重叠加权内积注入
:class:`~soptx.solvers.cg.CGSolver` 的 ``dot_product`` 扩展点, 使迭代内积与
``info['relres']`` 的残差口径都变成加权残差.

``dof_comm`` 是鸭子类型: 只要求有 ``dot`` 方法; 传 ``None`` (串行) 时退化为普通
2-范数内积, 串行与并行复用同一套代码路径. 因此本模块**不导入 mpi4py**, 这也是
它落在 :mod:`soptx.solvers` 而不是 :mod:`soptx.fem.distributed` 的原因 -- 后者
顶层依赖 mpi4py, 会把串行路径一并绑架.

默认容差与数值界由 :mod:`soptx.core.numerics` 提供.
"""

from __future__ import annotations

from typing import Any, Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.core.numerics import (
    DEFAULT_ATOL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RTOL,
    RESIDUAL_REFRESH,
)

from .cg import CGSolver


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
    norm_type: str = "natural",
) -> tuple[TensorLike, dict[str, Any]]:
    """基于重叠加权内积的分布式共轭梯度法 (PCG/CG) 迭代求解器.

    参数:
        operator (Any): 线性刚度算子 (如 ``ElasticityEAOperator`` 或 ``ConstrainedOperator``).
        load (TensorLike): 局部右端项载荷向量.
        dof_comm (EntityMPI | None): 自由度通信器 (提供重叠加权点积 ``dot``).
        x0 (TensorLike | None, 可选): 初始猜测解向量. 默认值为 None (全零或由 CG 初始化).
        maxiter (int, 可选): 最大允许 CG 迭代步数. 默认采用 ``DEFAULT_MAX_ITERATIONS``.
        rtol (float, 可选): 相对残差停机容差. 默认采用 ``DEFAULT_RTOL``.
        atol (float, 可选): 绝对残差停机容差. 默认采用 ``DEFAULT_ATOL``.
        residual_refresh (int, 可选): 定期重新计算真实代数残差以抑制舍入累积误差的步数间隔.
        norm_type (str, 可选): 停机判据在哪个范数下度量, 取值见 :func:`~soptx.solvers.cg.cg`.
            默认 ``"natural"``. 本函数不注入预条件子 (``M`` 为 ``None``), 三个取值在
            数值上重合, 该参数只为与 ``CGSolver`` 保持同一套口径, 供上层显式声明判据.

    返回:
        tuple[TensorLike, dict[str, Any]]: 包含解向量 ``solution`` 与 CG 运行状态字典 ``info`` 的二元组.

    说明:
        本函数是 ``CGSolver`` 在分布式布局下的装配点: 唯一的差异是把重叠加权点积
        ``dof_comm.dot`` 注入 ``dot_product``, 使迭代内积与 ``info['relres']`` 的残差
        口径都变成加权残差; 单进程 (无 ``dot``) 时退化为普通 2-范数内积.
        ``info`` 在原有 ``converged/niter/residual/recursive_residual/true_residual/
        breakdown`` 之外多出 ``LinearSolver`` 契约要求的 ``relres``.
    """
    if hasattr(dof_comm, "dot"):
        dot_fn, _ = dof_comm.dot(int(load.shape[0]))
    else:
        def dot_fn(x: TensorLike, y: TensorLike) -> float:
            return float(bm.sum(x * y))

    if x0 is not None:
        x0 = bm.asarray(x0)

    solver = CGSolver(
        atol=atol,
        rtol=rtol,
        maxit=maxiter,
        dot_product=dot_fn,
        residual_refresh=residual_refresh,
        norm_type=norm_type,
    ).setup(operator)
    solution, info = solver.solve(load, x0)
    return solution, info


__all__ = [
    "weighted_cg",
    "weighted_norm",
]
