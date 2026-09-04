"""SOPTX 自有的线性求解层.

直接法, Krylov 迭代, 预条件子与多重网格都是同一个类型
:class:`~soptx.solvers.base.LinearSolver`, 区别只在 ``setup`` 需要算子提供
什么, 不在接口形状. 3D 大规模是主线, 故默认装配全程 matrix-free:
``CGSolver(M=Multigrid(...))``, 只有多重网格最粗层才落到直接法.

SOPTX 依赖的求解器扩展 (CG 的可插拔内积与真残差刷新, MUMPS 的 sym 对称性
标志) 上游 fealpy 都没有, 故整层由 SOPTX 拥有并演化, 不再依赖 fealpy fork.

对外只暴露本文件导出的名字; 包内文件布局按需重组, 不影响调用方. 各后端模块
在此被导入, 从而完成向 :mod:`soptx.solvers.registry` 的注册.
"""

from soptx.protocols import SupportsMatmul

from .base import (
    CAP_DIAGONAL,
    CAP_HIERARCHY,
    CAP_MATRIX,
    ConvergedReason,
    LinearSolver,
    OperatorCapabilityError,
    SolveInfo,
    as_tensor,
    operator_capabilities,
    reason_text,
)
from .amg import AMGSolver
from .cg import CGSolver, cg
from .direct import DirectSolver, spsolve
from .minres import MINRESSolver
from .multigrid import Multigrid, MultigridLevel
from .overlap import weighted_cg, weighted_norm
from .preconditioners import (
    ChebyshevSmoother,
    DiagonalPreconditioner,
    estimate_lambda_max,
)
from .registry import available, create, register

__all__ = [
    # 契约层
    "CAP_DIAGONAL",
    "CAP_HIERARCHY",
    "CAP_MATRIX",
    "ConvergedReason",
    "LinearSolver",
    "OperatorCapabilityError",
    "SolveInfo",
    "SupportsMatmul",
    "as_tensor",
    "operator_capabilities",
    "reason_text",
    # 注册表
    "available",
    "create",
    "register",
    # 直接法
    "DirectSolver",
    "spsolve",
    # Krylov
    "CGSolver",
    "MINRESSolver",
    "cg",
    "weighted_cg",
    "weighted_norm",
    # 预条件子与光滑子
    "ChebyshevSmoother",
    "DiagonalPreconditioner",
    "estimate_lambda_max",
    # 多重网格
    "AMGSolver",
    "Multigrid",
    "MultigridLevel",
]
