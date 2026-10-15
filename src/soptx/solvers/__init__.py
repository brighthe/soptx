"""SOPTX 自有的线性求解层.

直接法, Krylov 迭代, 预条件子与多重网格都是同一个类型
:class:`~soptx.solvers.base.LinearSolver`, 区别只在 ``setup`` 需要算子提供
什么, 不在接口形状. 3D 大规模主线的装配是全程 matrix-free 的
``CGSolver(M=Multigrid(...))``, 只有多重网格最粗层才落到直接法. 这是推荐配置而非
默认值: 各分析器的默认求解器见其 ``solve_method``, 如 ``LagrangeFEMAnalyzer`` 默认
``'mumps'``.

SOPTX 依赖的求解器扩展 (CG 收敛时的真残差验证, MUMPS 的 sym 对称性标志) 上游
fealpy 都没有, 故整层由 SOPTX 拥有并演化, 不再依赖 fealpy fork.

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
    operator_diagonal,
    reason_text,
)
from .cg import CGSolver, cg
from .direct import DirectSolver, spsolve
from .multigrid import Multigrid, MultigridLevel
from .preconditioners import (
    DiagonalPreconditioner,
    JacobiSmoother,
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
    "operator_diagonal",
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
    "cg",
    # 预条件子与光滑子
    "DiagonalPreconditioner",
    "JacobiSmoother",
    "estimate_lambda_max",
    # 多重网格
    "Multigrid",
    "MultigridLevel",
]
