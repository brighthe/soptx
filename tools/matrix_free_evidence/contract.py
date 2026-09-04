"""Stage-1 numeric contract for the Matrix-Free elasticity baseline.

Every tolerance, default value and supported range used by this pipeline and by
``examples/matrix_free_elasticity``'s demo scripts is defined here exactly once,
so a tightened gate can never be applied on one side only.

Three modules could plausibly own these numbers; two of them must not:

- :mod:`soptx.core.numerics` owns *solver defaults*.  Its docstring is explicit that
  a number encoding an **acceptance gate** belongs to the study defining that
  gate, not to the solver, so the tolerances below stay out of it.  The solver
  defaults are re-exported here instead, so every consumer reads them from one
  place and the gate helpers can state a convergence criterion in the same
  numbers the solver actually used.
- ``schema`` owns the *shape* of the summary these numbers end up in, plus
  ``SCHEMA_VERSION``.  Values live here, layout lives there.

That leaves this package, which is where the gate is enforced.  The numbers sit
next to ``validate.py`` rather than in the example, because the example is two
demo scripts and this pipeline is also the fealpy fork's pre-merge gate.

This module must stay free of FEALPy and mpi4py imports: the evidence tooling
has to run on machines without an MPI runtime.  :mod:`soptx.core.numerics` is safe
precisely because it carries no such imports either.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from soptx.core.numerics import (
    DEFAULT_ATOL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RTOL,
    NORM_FLOOR,
    RESIDUAL_REFRESH,
)


STAGE = "soptx/matrix-free-elasticity/stage-1"

SUPPORTED_DIMENSIONS = (2, 3)
SUPPORTED_DEGREES = (1,)
OPERATOR_LEVELS = ("ea", "fa")

OPERATOR_STORAGE = {
    "ea": "cached-element-matrices",
    "fa": "global-csr",
}
DISTRIBUTED_REPRESENTATION = "equal-status-overlapping-copies"

DEFAULT_DIMENSION = 3
DEFAULT_DEGREE = 1
DEFAULT_RESOLUTION = 4
REFERENCE_RANDOM_SEED = 20260727

# Single-run gates, checked by both run.py and validate.py.
BOUNDARY_ABSOLUTE_TOL = 1.0e-12
MATVEC_RELATIVE_TOL = 1.0e-12
EXPLICIT_SOLUTION_RELATIVE_TOL = 1.0e-8

# Cross-run gates, checked by validate.py.
PARALLEL_SOLUTION_RELATIVE_TOL = 1.0e-9
EA_FA_SOLUTION_RELATIVE_TOL = 1.0e-9
PARALLEL_L2_DIFFERENCE_TOL = 1.0e-10
MINIMUM_FINAL_L2_ORDER = 1.5

# 三档加密, 依次对应 layout.EA_EVIDENCE_ROLES 的 coarse/medium/fine, 由
# validate.py 驱动。
#
# 两个维度取同一组剖分数, 这样 EA 与 FA 的比对档位在 2D/3D 上是同一个网格尺寸。
#
# 这里不承担"收敛阶趋势"的举证: 那条链由
# ``examples/lagrange_elasticity/manufactured_convergence_demo.py`` 的两条五档
# 显式组装结果给出(见该目录 ``results_analysis.md``)。本工具回答"EA 与 FA
# 是不是同一个离散": FA 与 EA 同档 (见 ``layout.SERIAL_VALIDATION_CASES``),
# 三档逐档比对 —— 收敛阶达标是门禁, 不是展示对象。
#
# 上界停在 32 已不再是硬约束: 3D 的 n=64 (823,875 自由度) 现在跑得动。原先的
# OOM 有两个叠加原因, 均已解除 —— ``linear_elastic_integrator`` 的 ``standard``
# 分支把被求和掉的积分点轴物化成九个 ``(NC, NQ, 4, 4)`` 临时张量(单块 3.75 GiB,
# 九块同时存活), 以及本机 WSL2 内存上限只有缺省的 31.2 GiB。改走
# ``assembly_method='fast'`` 并把上限抬到 48 GB 后, 该档实测峰值 RSS 17.38 GiB、
# 单档 150.3 s(FA + MUMPS ``sym=1``), 对照见
# ``examples/lagrange_elasticity/results_analysis.md`` §4.4。
#
# 之所以仍停在 32, 是因为把这里改成 (8, 16, 64) 之类要重跑并重新冻结整条 stage-1
# 证据链, 且本工具的职责("EA 与 FA 是不是同一个离散")三档已经足够。EA 侧的峰值
# 内存已另行测得, 走的是 ``examples/matrix_free_elasticity/benchmark_cpu_ea.py``
# 的 ``--mode serial-peak-rss``(一个进程只建一个层级), 与本链无关, 因此这里不必
# 因为要补那格数据而动 —— 对照见该目录 ``results_analysis.md`` §3.5。
REFINEMENTS = {
    2: (8, 16, 32),
    3: (8, 16, 32),
}


def residual_limit(
    rhs_norm: float,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> float:
    """Absolute residual a converged run must reach for the given RHS."""

    return max(atol, rtol * rhs_norm)


def matvec_reference_gates(matvec: dict) -> dict[str, bool]:
    """把 ``soptx.fem.verification.serial_references`` 的产出逐项对上阈值.

    判据本身也只写一次, 不只是阈值: ``compare_lagrange.py`` 与 ``report.py`` 调
    同一个函数, 就不会出现"两边阈值相同但一边漏了正定性探针"这种漂移. 入参是
    纯 dict, 所以本模块仍然不碰 FEALPy.

    前两条是这个脚本的正题 —— EA 与 FA 在裸算子和施加边界条件后是否给出同一个
    结果, 两条走的是不同代码路径, 不能并成一条. 第三条是唯一一条不以 "FA 是对的"
    为前提的检查: 它问这个离散系统本身是否退化. 曾经并列的对称性判据已删除, 因为
    ``dirichlet_matvec`` 通过就蕴含了它 (FA 严格对称), 且它从来只在 FA 存在时才评估.
    """

    return {
        "raw_matvec": matvec["raw_relative_error"] <= MATVEC_RELATIVE_TOL,
        "dirichlet_matvec": (
            matvec["dirichlet_relative_error"] <= MATVEC_RELATIVE_TOL
        ),
        "positive_definite": matvec["random_vector_energy"] > 0.0,
    }


def explicit_solution_gate(relative_error: float) -> bool:
    """CG 解与 FA 直接解的相对差是否落在门禁内."""

    return relative_error <= EXPLICIT_SOLUTION_RELATIVE_TOL


@dataclass(frozen=True)
class RunConfig:
    """One fully resolved single-run specification."""

    dimension: int
    degree: int
    resolution: tuple[int, ...]
    operator_level: str
    benchmark: bool
    max_iterations: int
    rtol: float
    atol: float
    output_path: Path
    summary_path: Path
    solution_path: Path

    @property
    def operator_storage(self) -> str:
        return OPERATOR_STORAGE[self.operator_level]

    def residual_limit(self, rhs_norm: float) -> float:
        return residual_limit(rhs_norm, rtol=self.rtol, atol=self.atol)
