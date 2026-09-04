"""求解层的公共契约: 求解器基类, 能力协商与 info 约定.

本模块只定义类型与约定, 不含任何数值实现, 因此可被 solvers 包内所有模块
安全导入, 不会引入循环依赖.

算子侧的结构化协议 ``SupportsMatmul`` 是层间边界 (analyzer 按它标注传给求
解层的算子), 因此定义在 :mod:`soptx.protocols.operators`; 本模块重导出它,
使包内模块无需跨包写 import.

分层的依据是 ``setup`` 需要算子提供什么:

- 什么都不需要 (CG, MINRES): 任何支持 ``@`` 的算子都能用;
- 需要对角线 (Jacobi, Chebyshev): 算子须能给出自己的对角;
- 需要显式矩阵 (scipy, MUMPS, AMG): 算子须能转成稀疏矩阵, 'ea' 层级下
  的 matrix-free 算子给不出, 该类后端在其上不可用;
- 需要外部层次 (几何多重网格): 层次由 fem 侧构造后传入, 不从算子推导.

``setup`` 拿不到所需输入即抛 :class:`OperatorCapabilityError`. 这取代了在
analyzer 里按 ``operator_level`` 硬编码的分支判断 -- 判据从"调用方声明了哪
种层级"变成"这个算子实际能提供什么", 加新后端不必再回头改 analyzer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, Dict, Optional

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

# 重导出: 本模块内不使用, 供 cg / direct / preconditioners 从 .base 取用.
from soptx.protocols import SupportsMatmul as SupportsMatmul

# ``solve`` 返回的诊断字典. 三个必需键由 LinearSolver.solve 统一校验:
#   niter     : int, 迭代次数; 直接法等一步解法记 1
#   relres    : float | None, 相对残差; 无法廉价得到时为 None (不是 0.0,
#               否则调用方会把"没算"误读成"已收敛到机器精度")
#   converged : bool, 是否按收敛判据正常退出; maxit 耗尽或 breakdown 为 False
# 各实现可自由附加诊断键 (如 CG 的 true_residual, breakdown, reason).
SolveInfo = Dict[str, Any]

REQUIRED_INFO_KEYS = ("niter", "relres", "converged")

# 算子能力标签. 求解器用 ``requires`` 声明自己需要哪些, setup 时比对.
CAP_MATRIX = "matrix"
CAP_DIAGONAL = "diagonal"
CAP_HIERARCHY = "hierarchy"


class ConvergedReason(IntEnum):
    """迭代求解器的退出原因, 取值沿用 PETSc ``KSPConvergedReason``.

    ``converged: bool`` 只说成没成, 说不出为什么. 调用方 (门禁脚本,
    analyzer, 实验驱动) 想区分"达到 rtol"与"maxit 耗尽"就只能去 parse
    ``breakdown`` 字符串, 那是靠不住的. 本枚举把退出原因升为一等诊断.

    整数取值与 PETSc 保持一致, 便于直接对照上游文档; 约定
    ``reason > 0`` 等价于收敛, 由 :func:`_finalize_info` 强制它与
    ``info['converged']`` 不漂移.

    批量右端项按 N 次独立求解处理, 每列各有自己的 reason: 逐列结果放在
    ``info['column_reasons']``, ``info['reason']`` 给聚合后的单值.
    """

    ITERATING = 0

    CONVERGED_RTOL = 2
    CONVERGED_ATOL = 3
    #: 迭代数达到上限即视为完成 (预条件子按固定步数使用时的正常退出).
    CONVERGED_ITS = 4

    #: maxit 耗尽而未达判据.
    DIVERGED_ITS = -3
    #: 残差相对初值放大超过 divtol.
    DIVERGED_DTOL = -4
    #: 残差内积非正, 迭代无法继续.
    DIVERGED_BREAKDOWN = -5
    #: 出现非正曲率, 算子在该方向上不是正定的.
    DIVERGED_INDEFINITE_MAT = -8
    #: 判据量出现 NaN 或 Inf.
    DIVERGED_NANORINF = -9


_REASON_TEXT = {
    ConvergedReason.ITERATING: "仍在迭代",
    ConvergedReason.CONVERGED_RTOL: "达到相对容差 rtol",
    ConvergedReason.CONVERGED_ATOL: "达到绝对容差 atol",
    ConvergedReason.CONVERGED_ITS: "达到指定迭代数",
    ConvergedReason.DIVERGED_ITS: "maxit 耗尽而未收敛",
    ConvergedReason.DIVERGED_DTOL: "残差相对初值放大超过 divtol",
    ConvergedReason.DIVERGED_BREAKDOWN: "残差内积非正, 迭代中断",
    ConvergedReason.DIVERGED_INDEFINITE_MAT: "遇到非正曲率, 算子非正定",
    ConvergedReason.DIVERGED_NANORINF: "判据量出现 NaN 或 Inf",
}


def reason_text(reason: "ConvergedReason | int") -> str:
    """``ConvergedReason`` 的中文说明, 供日志与诊断消息使用."""
    try:
        return _REASON_TEXT[ConvergedReason(int(reason))]
    except (ValueError, KeyError):
        return f"未知退出原因 ({int(reason)})"


def as_tensor(value: Any) -> Any:
    """把调用方传入的向量归一成后端张量, ``None`` 原样返回.

    两家参考实现都在向量层做这件事 (PETSc 的 ``VecPlaceArray``, MFEM 的
    ``mfem::Vector``), 而不是在算法里放宽类型断言 -- 所以归一化落在
    :meth:`LinearSolver.solve` 入口, 各求解器内部仍可假定拿到的是
    ``TensorLike``.

    显式处理 :class:`fealpy.functionspace.function.Function`: 它的 MRO 是
    ``(Function, Generic, object)``, **不是** ``TensorLike`` 的注册子类,
    但持有 ``.array``. 'ea' 层级的 ``ElasticityEAOperator.assemble()`` 正
    是返回它, 不接就整条 matrix-free 路径进不来.
    """
    if value is None or isinstance(value, TensorLike):
        return value
    # fealpy Function: 'array' 走 object.__getattribute__, 不会被
    # Function.__getattr__ 转发, 拿到的是真正的底层张量.
    array = getattr(value, "array", None)
    if isinstance(array, TensorLike):
        return array
    return bm.asarray(value)


class OperatorCapabilityError(TypeError):
    """算子提供不了该求解器 setup 所需的输入."""


def operator_capabilities(op: Any) -> frozenset:
    """探测算子能提供哪些能力.

    按方法是否存在判断而不是按类型判断: 'fa' 侧是 fealpy 的 CSRTensor/
    COOTensor, substructure 侧直接传 scipy 稀疏矩阵, 'ea' 侧是 SOPTX 自有
    的算子包装, 三者没有共同基类.

    Parameters
    ----------
    op : Any
        待探测的算子.

    Returns
    -------
    frozenset
        该算子提供的能力标签集合.
    """
    caps = set()
    # to_scipy: fealpy 稀疏张量; tocsr: 已经是 scipy 稀疏矩阵
    if hasattr(op, "to_scipy") or hasattr(op, "tocsr"):
        caps.add(CAP_MATRIX)
    # diags: fealpy 稀疏张量的对角; diagonal: EA 算子将来自报对角的入口
    if hasattr(op, "diagonal") or hasattr(op, "diags"):
        caps.add(CAP_DIAGONAL)
    return frozenset(caps)


def _finalize_info(info: SolveInfo, owner: str) -> SolveInfo:
    """校验 ``_solve`` 返回的 info 含全部必需键, 且 reason 与 converged 一致.

    契约在基类强制而不是靠各实现自觉: 移植过来的 Krylov 实现在这一点上
    本来就不一致 (fealpy 各方法都不给 converged), 不拦就会漏进来.

    ``reason`` 是可选键 -- 直接法与预条件子不填, 不必为此改它们. 但一旦
    填了就必须与 ``converged`` 同向, 否则两个诊断会各说各话, 下游按哪个
    判都不安全.
    """
    missing = [k for k in REQUIRED_INFO_KEYS if k not in info]
    if missing:
        raise KeyError(
            f"{owner}._solve 返回的 info 缺少必需键 {missing}; "
            f"必需键为 {list(REQUIRED_INFO_KEYS)}, 实得 {sorted(info)}"
        )
    reason = info.get("reason", None)
    if reason is not None and (int(reason) > 0) != bool(info["converged"]):
        raise ValueError(
            f"{owner}._solve 返回的 info 自相矛盾: "
            f"reason={ConvergedReason(int(reason)).name} "
            f"({reason_text(reason)}), 但 converged={info['converged']}; "
            f"约定 reason > 0 等价于 converged 为 True"
        )
    return info


class LinearSolver(ABC):
    """求解器与预条件子的统一类型.

    直接法, Krylov 迭代法, 预条件子与多重网格都是本类型的实现 -- 三者的
    区别只在 ``setup`` 需要什么, 不在接口形状. 统一之后 ``M=`` 位与
    ``solver=`` 位可以互换填入, 组合逻辑留在调用点, 包内不预设哪种搭配
    合法.

    子类实现 :meth:`_solve` 即可; :meth:`solve` 由基类提供, 负责校验 info
    契约. 需要快速路径的预条件子应覆盖 :meth:`__matmul__`.

    Attributes
    ----------
    requires : frozenset
        本求解器 ``setup`` 需要算子提供的能力标签, 见模块级 ``CAP_*``.
        默认为空, 即对算子无要求.
    """

    requires: frozenset = frozenset()

    def __init__(self) -> None:
        self._op: Any = None

    @property
    def op(self) -> Any:
        """已绑定的算子; 未 setup 时访问是使用错误, 直接报错而非返回 None."""
        if self._op is None:
            raise RuntimeError(
                f"{type(self).__name__} 尚未 setup, 请先调用 setup(op)"
            )
        return self._op

    @property
    def is_setup(self) -> bool:
        return self._op is not None

    def setup(self, op: Any) -> "LinearSolver":
        """绑定算子并做能力协商.

        需要预计算的子类覆盖本方法, 先 ``super().setup(op)`` 再做自己的
        准备工作 (分解, 建层次, 提对角等).

        Parameters
        ----------
        op : Any
            系数算子.

        Returns
        -------
        LinearSolver
            返回 self, 便于 ``solver.setup(op).solve(b)`` 连写.

        Raises
        ------
        OperatorCapabilityError
            算子提供不了 ``requires`` 声明的能力.
        """
        available = operator_capabilities(op)
        missing = self.requires - available
        if missing:
            raise OperatorCapabilityError(
                f"{type(self).__name__} 需要算子提供 {sorted(missing)}, "
                f"但 {type(op).__name__} 只提供 {sorted(available)}. "
                f"matrix-free 算子没有可分解的全局矩阵, "
                f"请改用对算子无此要求的求解器."
            )
        self._op = op
        return self

    @abstractmethod
    def _solve(
        self, b: TensorLike, x0: Optional[TensorLike]
    ) -> "tuple[TensorLike, SolveInfo]":
        """实际求解; 返回 ``(x, info)``, info 须含 ``REQUIRED_INFO_KEYS``."""

    def solve(
        self, b: TensorLike, x0: Optional[TensorLike] = None
    ) -> "tuple[TensorLike, SolveInfo]":
        """求解 ``op x = b``, 无条件返回 ``(x, info)``.

        不给 ``returninfo`` 开关: "可能没收敛"是迭代法的固有状态, 让调用方
        每次都接住 info, 比让它可选安全.

        入参先经 :func:`as_tensor` 归一化, 因此 fealpy ``Function`` 与
        list 之类都能直接传入; 返回的 ``x`` 一律是后端张量.
        """
        x, info = self._solve(as_tensor(b), as_tensor(x0))
        return x, _finalize_info(info, type(self).__name__)

    def __matmul__(self, r: TensorLike) -> TensorLike:
        """预条件子模式: 零初值求解一次, 丢弃 info.

        默认走 :meth:`solve`; 每步 Krylov 迭代都会调用, 开销敏感的实现
        (如对角预条件子) 应覆盖本方法走直算路径.
        """
        x, _ = self.solve(r)
        return x

    def apply(self, r: TensorLike) -> TensorLike:
        """``__matmul__`` 的别名, 对齐 fealpy IterativeSolverManager 的命名."""
        return self @ r
