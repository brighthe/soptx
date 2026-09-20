"""CGSolver 的正确性验证入口.

被验证的对象是 ``soptx/solvers/cg.py`` 里的 ``CGSolver`` (及其转调的函数式
``cg``), 连同经 ``M=`` 位接入的预条件子, 以及 ``LagrangeFEMAnalyzer.
assemble_operator_diagonal`` 在 'fa' / 'ea' 两个层级下取出的对角. 有限元装配只是
为了造出一个真实的 SPD 弹性刚度系统, 不是被验证的对象; 问题构造借自同目录的
``verify_direct_solvers.py``, 两边在同一个矩阵上说话.

case 轴是预条件子 (``--case``), 与 verify_direct_solvers 的"一个 case 一个后端"
同构:

    none    无预条件, M=None
    jacobi  DiagonalPreconditioner(diag), diag 由 assemble_operator_diagonal 取出

尚未实现的预条件子 (Chebyshev, AMG, 多重网格) 不在注册表里; 落地后加一条注册项
即可, 可用性由 ``case_available`` 探测.

同一个离散算子以两种形态进入 CG:

    fa   apply_bc 对称消元后的全局稀疏矩阵, 按 COO 作用 (与 analyzer 同路)
    ea   DirichletBCOperator 包住的 matrix-free 算子, 只支持 @

算子形态轴 (``--operator-level`` / ``-L``):

    fa   默认, 仅跑完全装配稀疏矩阵形态
    ea   仅跑无矩阵算子形态
    both 同步跑 'fa' 与 'ea' 并做两层级等价性与收敛对照

每个 case 下跑八项检查 (``--checks``):

consistency  'fa' 上 CG 解与 DirectSolver('scipy') 解一致, 对完整算子算的真残差
             在阈值内.
level        同一参数、同一初值下 'fa' 与 'ea' 同解, 迭代数相差不超过 1.
norm         三档 norm_type 都收敛且同解; 无预条件时三档迭代数相同.
refresh      residual_refresh=0 与 =20 同解; info['relres'] 与脚本独立算的
             ||b - A x|| / ||b|| 一致.
batch        两列尺度相差 1e3 的右端项批量求解, 逐列与单独求解一致;
             batch_first 两种布局同解.
contract     converged 与 reason > 0 一致; 初值已是解时 niter == 0 且 reason 为
             CONVERGED_ATOL; maxit 耗尽时 reason 为 DIVERGED_ITS; 非法构造参数与
             非正对角均抛 ValueError.
precond      只对 M 非空的 case: 两层级取出的对角一致; 接入预条件子后各层级的
             迭代数均不劣于无预条件.
solution     加密序列 (--levels, --base) 上逐层求解: 两层级都收敛, 'fa' 真残差在
             阈值内, 制造解 L2 误差的末档观测阶不低于门禁.

通过时只打印横幅、cases 行与每个 case 的 solution 逐层表 (n, gdof, 两层级 niter,
relres, 真残差, L2 误差, 观测阶); 其余检查通过时不打印, 失败时把整组结果打出来.

除 solution 外的检查里两个层级的初值都取 'ea' 的 Dirichlet 基准向量
(``prescribed_solution``): 'ea' 的 apply_bc 不改写右端项上的 Dirichlet 分量, 而是
把算子包成 A = Pi_I K Pi_I + Pi_D, Dirichlet 值靠初值携带; 'fa' 用同一初值后两个
系统在内部自由度上完全相同, 迭代数才可比.

使用方法::

    python examples/linear_solvers/verify_cg_solver.py --list
    python examples/linear_solvers/verify_cg_solver.py
    python examples/linear_solvers/verify_cg_solver.py --case jacobi -L fa
    python examples/linear_solvers/verify_cg_solver.py --case jacobi --checks precond solution
    python examples/linear_solvers/verify_cg_solver.py --dim 3 --n 16 --levels 3 --json

判定阈值属于本示例, 不属于 ``soptx``; 见本文件的 ``TOLERANCES``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from fealpy.backend import backend_manager as bm

from soptx.fem.verification import relative_difference, solution_error
from soptx.solvers import (
    CGSolver,
    ConvergedReason,
    DiagonalPreconditioner,
    spsolve,
)
from soptx.solvers.cg import NORM_TYPES


def _load_direct_module():
    """按路径加载同目录的 verify_direct_solvers.py, 借用它的问题构造与排版工具.

    examples/ 不是包, 用 importlib 按文件路径加载, 不往 sys.path 里塞脚本目录.
    """
    path = _SCRIPT_DIR / "verify_direct_solvers.py"
    spec = importlib.util.spec_from_file_location("verify_direct_solvers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_direct = _load_direct_module()
build_analyzer = _direct.build_analyzer
DEFAULT_MODELS = _direct.DEFAULT_MODELS
DEFAULT_MESH_TYPES = _direct.DEFAULT_MESH_TYPES
MESH_DIMENSIONS = _direct.MESH_DIMENSIONS
PROBLEM_FACTORIES = _direct.PROBLEM_FACTORIES
SYSTEM_ANALYZERS = _direct.SYSTEM_ANALYZERS
DEFAULT_SYSTEM = _direct.DEFAULT_SYSTEM
display_width = _direct.display_width
pad = _direct.pad
verdict = _direct.verdict
level_orders = _direct.level_orders
convergence_module = _direct.convergence_module

DEFAULT_OPERATOR_LEVEL = "fa"

# 本脚本一次最多验证两个槽位: 一个 'fa', 一个矩阵自由层级. 'both' 固定取 'ea';
# 三个层级放在一张表里横比不在这里做, 那是
# experiments/assembly_level_capability 的事.
MATRIX_FREE_LEVELS = ("ea", "pa")


def matrix_free_level(operator_level: str) -> str | None:
    """给定 --operator-level, 返回要建的矩阵自由层级; 'fa' 下没有"""
    if operator_level in MATRIX_FREE_LEVELS:
        return operator_level
    if operator_level == "both":
        return "ea"

    return None

# --------------------------------------------------------------------------
# case 注册表: 一个 case 一个预条件子
# --------------------------------------------------------------------------
# factory(entry) 接收 build_level 返回的层级字典, 返回交给 CG ``M=`` 位的对象;
# None 即无预条件. 每个层级各造一个实例: 'fa' 与 'ea' 的对角各自取出, precond
# 一项正是要比较它们.
CASE_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "id": "none",
        "title": "M=None",
        "levels": DEFAULT_OPERATOR_LEVEL,
        "factory": lambda entry: None,
    },
    {
        "id": "jacobi",
        "title": "DiagonalPreconditioner(diag)",
        "levels": DEFAULT_OPERATOR_LEVEL,
        "factory": lambda entry: DiagonalPreconditioner(entry["diag"]),
    },
)
CASES_BY_ID = {case["id"]: case for case in CASE_REGISTRY}
CASE_IDS = tuple(CASES_BY_ID)

# 八项检查: 名字即 --checks 的取值; 顺序即执行顺序, contract 的扫描要排在其它
# 求解之后, solution 最后打表.
CHECK_TITLES = {
    "consistency": "与直接法一致",
    "level": "算子层级无关",
    "norm": "判据范数",
    "refresh": "真残差口径",
    "batch": "批量右端项逐列判定",
    "contract": "info 契约与守卫",
    "precond": "预条件子收益",
    "solution": "制造解逐层收敛",
}
CHECK_IDS = tuple(CHECK_TITLES)

# 判定阈值. solution_rel_diff 留出 kappa(A) * rtol 的放大余量 (n=40, p=1 时
# kappa(A) 约 1e4); true_relres 与 verify_direct_solvers.TOLERANCES 的
# residual_relative 同口径.
TOLERANCES = {
    "solution_rel_diff": 1.0e-6,
    "true_relres": 1.0e-10,
    "niter_gap": 1,
    "diag_rel_diff": 1.0e-12,
    "relres_rel_diff": 1.0e-6,
}

# 求解参数, 与 LagrangeFEMAnalyzer 的 'cg' 默认值同口径.
CG_PARAMETERS = {
    "atol": 1.0e-12,
    "rtol": 1.0e-12,
    "maxit": 10000,
    "print_level": 0,
}

# batch 的第二列: 在第一列上叠加 10% 幅值的确定性噪声后再缩小 1e3 倍.
BATCH_SCALE = 1.0e-3
BATCH_NOISE = 0.1
BATCH_SEED = 0

# refresh 的真残差刷新间隔.
REFRESH_INTERVAL = 20

NORM_FLOOR = 1.0e-300


def case_available(case: dict[str, Any]) -> tuple[bool, str]:
    """预条件子在当前代码里是否已实现, 及不可用的原因.

    占位类 (Chebyshev, AMG, 多重网格) 在 ``__init__`` 就抛 NotImplementedError,
    用一个极小的假层级探一下即可, 不必装配真实系统.
    """
    probe = {"diag": bm.ones(2, dtype=bm.float64), "operator": None}
    try:
        case["factory"](probe)
    except NotImplementedError:
        return False, "尚无实现"
    return True, ""


def resolve_cases(case_ids: list[str]) -> list[dict[str, Any]] | None:
    """把 --case 给的 id 解析成注册表条目; 任一不可用即判本次验证失败.

    与 verify_direct_solvers 同理: 用户点名了哪个 case, 就是要求验证哪个预条件子,
    代码里没实现就是这次验证没做成, 不能静默跳过后再以退出码 0 报通过.
    """
    cases: list[dict[str, Any]] = []
    blocked: list[str] = []
    for case_id in case_ids:
        case = CASES_BY_ID[case_id]
        usable, reason = case_available(case)
        if usable:
            if case not in cases:
                cases.append(case)
        else:
            blocked.append(f"case {case_id} 不可用: {reason}")
    if blocked:
        for line in blocked:
            print(line)
        return None
    return cases


# --------------------------------------------------------------------------
# 两个层级的系统
# --------------------------------------------------------------------------
def build_level(
    dimension: int, resolution: int, order: int, model: str, mesh_type: str, level: str
) -> dict[str, Any]:
    """装配一个层级的带 Dirichlet 边界的系统, 连同 CG 需要的伴随量.

    Returns
    -------
    dict
        ``operator`` 是交给 CG 的算子 ('fa' 下照 ``LagrangeFEMAnalyzer.
        _as_iterative_operator`` 转成 COO 作用, 'ea' 下就是 DirichletBCOperator);
        ``matrix`` 只在 'fa' 下有值, 给直接法用; ``prescribed`` 是 Dirichlet
        基准向量; ``diag`` 是 ``assemble_operator_diagonal`` 取出的算子对角;
        ``mesh`` 与 ``problem`` 供 solution 一项算制造解误差.
    """
    analyzer, mesh, space, problem = build_analyzer(
        dimension, resolution, order, model, mesh_type, level
    )
    stiffness = analyzer.assemble_stiff_matrix()
    operator, load = analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector())
    diag = analyzer.assemble_operator_diagonal(operator)
    return {
        "level": level,
        "analyzer": analyzer,
        "mesh": mesh,
        "space": space,
        "problem": problem,
        "matrix": operator if level == "fa" else None,
        "operator": operator.tocoo() if level == "fa" else operator,
        "load": load,
        "prescribed": analyzer.prescribed_solution,
        "diag": diag,
    }


def preconditioner(case: dict[str, Any], entry: dict[str, Any]):
    """按 case 为一个层级造预条件子; 无预条件返回 None."""
    return case["factory"](entry)


def solve_cg(operator, load, x0=None, **options):
    """按 CG_PARAMETERS 造一个 CGSolver, ``options`` 覆盖其中的项, 返回 (x, info)."""
    settings = {**CG_PARAMETERS, **options}
    solver = CGSolver(**settings).setup(operator)
    return solver.solve(load, x0)


def true_relative_residual(operator, load, x) -> float:
    """脚本自己算的 ||b - A x|| / ||b||, 2D 右端按 Frobenius 范数."""
    residual = bm.to_numpy(load - operator @ x)
    reference = float(np.linalg.norm(bm.to_numpy(load)))
    return float(np.linalg.norm(residual) / max(reference, NORM_FLOOR))


def summarize(info: dict[str, Any]) -> dict[str, Any]:
    """info 里进报告的几个键, 转成可 JSON 化的纯量."""
    reason = info.get("reason")
    return {
        "niter": int(info["niter"]),
        "converged": bool(info["converged"]),
        "reason": ConvergedReason(int(reason)).name if reason is not None else None,
        "relres": None if info["relres"] is None else float(info["relres"]),
    }


def remember(context: dict[str, Any], label: str, info: dict[str, Any]) -> None:
    """把一次求解的 info 记入本 case 的上下文, 供 contract 的契约扫描."""
    context["infos"].append((label, info))


def reason_name(info: dict[str, Any]) -> str:
    return ConvergedReason(int(info["reason"])).name


def report_failure(name: str, lines: list[str]) -> None:
    """检查未过时把整组结果打出来; 通过时各检查什么都不打."""
    print(f"[{name}] {CHECK_TITLES[name]}")
    for line in lines:
        print(line)
    print("")


# --------------------------------------------------------------------------
# consistency
# --------------------------------------------------------------------------
def check_consistency(context: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """当前主层级上 CG 对 DirectSolver('scipy') 的解与真残差."""
    primary = context["primary"]
    x0 = primary["prescribed"]
    x, info = solve_cg(primary["operator"], primary["load"], x0, M=preconditioner(case, primary))
    remember(context, f"consistency {primary['level']}", info)

    _, diff = relative_difference(x, context["x_direct"])
    relres = true_relative_residual(primary["operator"], primary["load"], x)
    ok_diff = diff <= TOLERANCES["solution_rel_diff"]
    ok_res = relres <= TOLERANCES["true_relres"]
    ok_conv = bool(info["converged"])
    passed = ok_diff and ok_res and ok_conv

    if not passed:
        report_failure(
            "consistency",
            [
                f"  [{primary['level']}] niter {info['niter']}, reason {reason_name(info)}  [{verdict(ok_conv)}]",
                f"  ||x_cg - x_d|| / ||x_d||  {diff:.3e}"
                f"  (阈值 {TOLERANCES['solution_rel_diff']:.0e})  [{verdict(ok_diff)}]",
                f"  ||b - A x_cg|| / ||b||    {relres:.3e}"
                f"  (阈值 {TOLERANCES['true_relres']:.0e})  [{verdict(ok_res)}]",
            ],
        )
    return {
        "passed": passed,
        "entries": {
            "level": primary["level"],
            "solve": summarize(info),
            "solution_rel_diff": diff,
            "true_relres": relres,
        },
    }


# --------------------------------------------------------------------------
# level
# --------------------------------------------------------------------------
def check_level(context: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """同一参数、同一初值下 'fa' 与 'ea' 同解, 迭代数相差不超过 niter_gap."""
    if context["operator_level"] != "both":
        return {
            "passed": True,
            "skipped": True,
            "reason": f"仅验证单形态 {context['operator_level']!r}, 跳过跨形态一致性校验",
        }
    fa, ea = context["fa"], context["ea"]
    x0 = ea["prescribed"]

    _, prescribed_diff = relative_difference(fa["prescribed"], x0)
    ok_prescribed = prescribed_diff <= TOLERANCES["diag_rel_diff"]

    x_fa, info_fa = solve_cg(fa["operator"], fa["load"], x0, M=preconditioner(case, fa))
    x_ea, info_ea = solve_cg(ea["operator"], ea["load"], x0, M=preconditioner(case, ea))
    remember(context, "level fa", info_fa)
    remember(context, "level ea", info_ea)

    _, diff_levels = relative_difference(x_ea, x_fa)
    _, diff_ea_direct = relative_difference(x_ea, context["x_direct"])
    gap = abs(int(info_ea["niter"]) - int(info_fa["niter"]))
    ok_diff = diff_levels <= TOLERANCES["solution_rel_diff"]
    ok_direct = diff_ea_direct <= TOLERANCES["solution_rel_diff"]
    ok_gap = gap <= TOLERANCES["niter_gap"]
    ok_conv = bool(info_fa["converged"]) and bool(info_ea["converged"])
    passed = ok_prescribed and ok_diff and ok_direct and ok_gap and ok_conv

    if not passed:
        report_failure(
            "level",
            [
                f"  两层级 Dirichlet 基准向量相对差  {prescribed_diff:.3e}"
                f"  (阈值 {TOLERANCES['diag_rel_diff']:.0e})  [{verdict(ok_prescribed)}]",
                f"  fa: niter {info_fa['niter']}, reason {reason_name(info_fa)}",
                f"  ea: niter {info_ea['niter']}, reason {reason_name(info_ea)}",
                f"  ||x_ea - x_fa|| / ||x_fa||  {diff_levels:.3e}"
                f"  (阈值 {TOLERANCES['solution_rel_diff']:.0e})  [{verdict(ok_diff)}]",
                f"  ||x_ea - x_d|| / ||x_d||    {diff_ea_direct:.3e}"
                f"  (阈值 {TOLERANCES['solution_rel_diff']:.0e})  [{verdict(ok_direct)}]",
                f"  |niter_ea - niter_fa|       {gap}"
                f"  (阈值 {TOLERANCES['niter_gap']})  [{verdict(ok_gap)}]",
            ],
        )
    return {
        "passed": passed,
        "entries": {
            "prescribed_rel_diff": prescribed_diff,
            "fa": summarize(info_fa),
            "ea": summarize(info_ea),
            "solution_rel_diff": diff_levels,
            "ea_direct_rel_diff": diff_ea_direct,
            "niter_gap": gap,
        },
    }


# --------------------------------------------------------------------------
# norm
# --------------------------------------------------------------------------
def check_norm(context: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """三档 norm_type 都收敛且同解; 无预条件时三档迭代数相同."""
    primary = context["primary"]
    x0 = primary["prescribed"]
    width = max(display_width(name) for name in NORM_TYPES)

    entries: dict[str, Any] = {}
    lines: list[str] = []
    niters: dict[str, int] = {}
    passed = True
    for norm_type in NORM_TYPES:
        x, info = solve_cg(
            primary["operator"], primary["load"], x0,
            M=preconditioner(case, primary), norm_type=norm_type,
        )
        remember(context, f"norm {norm_type}", info)
        _, diff = relative_difference(x, context["x_direct"])
        ok = bool(info["converged"]) and diff <= TOLERANCES["solution_rel_diff"]
        passed = passed and ok
        niters[norm_type] = int(info["niter"])
        entries[norm_type] = {"solve": summarize(info), "solution_rel_diff": diff}
        lines.append(
            f"  {pad(norm_type, width)}  niter {info['niter']:5d}"
            f"  ||x - x_d|| / ||x_d|| {diff:.3e}  [{verdict(ok)}]"
        )
    # 无预条件时三档判据量是同一个数, 迭代数必须精确相同; 带预条件时三档
    # 度量不同, 迭代数允许不同.
    if preconditioner(case, primary) is None:
        ok_same = len(set(niters.values())) == 1
        passed = passed and ok_same
        entries["same_niter"] = ok_same
        lines.append(f"  三档迭代数相同  [{verdict(ok_same)}]")

    if not passed:
        report_failure(
            "norm",
            lines + [f"  阈值 {TOLERANCES['solution_rel_diff']:.0e} -> {verdict(passed)}"],
        )
    return {"passed": passed, "entries": entries}


# --------------------------------------------------------------------------
# refresh
# --------------------------------------------------------------------------
def check_refresh(context: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """residual_refresh 不改变解; info['relres'] 与脚本独立算的真残差一致."""
    entries: dict[str, Any] = {}
    lines: list[str] = []
    passed = True
    active_entries = [entry for entry in (context["fa"], context["ea"]) if entry is not None]
    for entry in active_entries:
        x0 = entry["prescribed"]
        level = entry["level"]
        solutions = {}
        group: dict[str, Any] = {}
        for refresh in (0, REFRESH_INTERVAL):
            x, info = solve_cg(
                entry["operator"], entry["load"], x0,
                M=preconditioner(case, entry), residual_refresh=refresh,
            )
            remember(context, f"refresh {level} refresh={refresh}", info)
            own = true_relative_residual(entry["operator"], entry["load"], x)
            reported = float(info["relres"])
            relres_diff = abs(reported - own) / max(own, NORM_FLOOR)
            ok = bool(info["converged"]) and relres_diff <= TOLERANCES["relres_rel_diff"]
            passed = passed and ok
            solutions[refresh] = x
            group[f"refresh={refresh}"] = {
                "solve": summarize(info),
                "own_relres": own,
                "relres_rel_diff": relres_diff,
            }
            lines.append(
                f"  {level} refresh={refresh:2d}: niter {info['niter']:5d}"
                f"  relres 报告 {reported:.3e} / 独立 {own:.3e}"
                f"  相对差 {relres_diff:.1e}  [{verdict(ok)}]"
            )
        _, diff = relative_difference(solutions[REFRESH_INTERVAL], solutions[0])
        ok_diff = diff <= TOLERANCES["solution_rel_diff"]
        passed = passed and ok_diff
        group["solution_rel_diff"] = diff
        lines.append(
            f"  {level}: ||x_{REFRESH_INTERVAL} - x_0|| / ||x_0||  {diff:.3e}"
            f"  (阈值 {TOLERANCES['solution_rel_diff']:.0e})  [{verdict(ok_diff)}]"
        )
        entries[level] = group

    if not passed:
        report_failure("refresh", lines)
    return {"passed": passed, "entries": entries}


# --------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------
def check_batch(context: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """两列尺度相差 1e3 的右端项批量求解, 逐列与单独求解一致; 两种布局同解.

    两列尺度刻意拉开, 用来抓"按整体范数判定"这类错误: 若判据不是逐列独立,
    小尺度那一列会在未收敛时被大尺度列的收敛带过.
    """
    primary = context["primary"]
    operator = primary["operator"]
    first = primary["load"]
    first_np = bm.to_numpy(first)
    rng = np.random.default_rng(BATCH_SEED)
    noise = rng.standard_normal(first_np.shape) * (BATCH_NOISE * np.max(np.abs(first_np)))
    second = bm.tensor(BATCH_SCALE * (first_np + noise))

    def solve(load, **options):
        return solve_cg(operator, load, M=preconditioner(case, primary), **options)

    x_first, info_first = solve(first)
    x_second, info_second = solve(second)
    remember(context, "batch single first", info_first)
    remember(context, "batch single second", info_second)

    batch = bm.stack([first, second], axis=1)
    x_batch, info_batch = solve(batch)
    remember(context, "batch dof-first", info_batch)

    batch_first = bm.stack([first, second], axis=0)
    x_batch_first, info_batch_first = solve(batch_first, batch_first=True)
    remember(context, "batch batch_first", info_batch_first)

    _, diff_first = relative_difference(x_batch[:, 0], x_first)
    _, diff_second = relative_difference(x_batch[:, 1], x_second)
    _, diff_layout_first = relative_difference(x_batch_first[0], x_batch[:, 0])
    _, diff_layout_second = relative_difference(x_batch_first[1], x_batch[:, 1])
    threshold = TOLERANCES["solution_rel_diff"]
    ok_columns = diff_first <= threshold and diff_second <= threshold
    ok_layout = diff_layout_first <= threshold and diff_layout_second <= threshold
    ok_conv = bool(info_batch["converged"]) and bool(info_batch_first["converged"])
    passed = ok_columns and ok_layout and ok_conv

    column_reasons = tuple(
        ConvergedReason(int(code)).name for code in (info_batch["column_reasons"] or ())
    )
    if not passed:
        report_failure(
            "batch",
            [
                f"  单独求解: niter 第一列 {info_first['niter']}, 第二列 {info_second['niter']}",
                f"  批量求解: niter {info_batch['niter']}, reason {reason_name(info_batch)},"
                f" 逐列 {' '.join(column_reasons)}  [{verdict(ok_conv)}]",
                f"  逐列对单独求解  第一列 {diff_first:.3e}  第二列 {diff_second:.3e}"
                f"  [{verdict(ok_columns)}]",
                f"  batch_first 对 dof 在前  第一列 {diff_layout_first:.3e}"
                f"  第二列 {diff_layout_second:.3e}  [{verdict(ok_layout)}]",
                f"  阈值 {threshold:.0e} -> {verdict(passed)}",
            ],
        )
    return {
        "passed": passed,
        "entries": {
            "single_first": summarize(info_first),
            "single_second": summarize(info_second),
            "batch": summarize(info_batch),
            "batch_first": summarize(info_batch_first),
            "column_reasons": column_reasons,
            "column_rel_diff": [diff_first, diff_second],
            "layout_rel_diff": [diff_layout_first, diff_layout_second],
        },
    }


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------
def check_contract(context: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """info 契约: reason 与 converged 同向, 热启动与 maxit 的退出原因, 参数守卫.

    守卫探针与 M 无关, 但代价可忽略, 每个 case 下都跑一遍, 不另设特例.
    """
    primary = context["primary"]
    M = preconditioner(case, primary)
    entries: dict[str, Any] = {}
    lines: list[str] = []
    passed = True

    # 1. 本 case 里记下的每一次求解, reason > 0 都必须与 converged 一致.
    mismatched = [
        label
        for label, info in context["infos"]
        if (int(info["reason"]) > 0) != bool(info["converged"])
    ]
    ok_sweep = not mismatched
    passed = passed and ok_sweep
    entries["reason_matches_converged"] = {
        "scanned": len(context["infos"]),
        "mismatched": mismatched,
    }
    lines.append(
        f"  已记录 {len(context['infos'])} 次求解, reason > 0 与 converged 一致"
        f"  [{verdict(ok_sweep)}]"
        + ("" if ok_sweep else "; 不一致: " + " | ".join(mismatched))
    )

    # 2. 初值已是解: 进入循环前即判定收敛, niter == 0, 走 atol 分支.
    _, info_warm = solve_cg(
        primary["operator"], primary["load"], context["x_direct"], M=M, atol=1.0e-8
    )
    ok_warm = (
        int(info_warm["niter"]) == 0
        and info_warm["reason"] == ConvergedReason.CONVERGED_ATOL
    )
    passed = passed and ok_warm
    entries["warm_start"] = summarize(info_warm)
    lines.append(
        f"  x0 = x_d, atol=1e-8: niter {info_warm['niter']}, reason {reason_name(info_warm)}"
        f"  [{verdict(ok_warm)}]"
    )

    # 3. maxit 耗尽: converged 为 False, reason 为 DIVERGED_ITS.
    _, info_maxit = solve_cg(primary["operator"], primary["load"], M=M, maxit=3)
    ok_maxit = (
        not bool(info_maxit["converged"])
        and info_maxit["reason"] == ConvergedReason.DIVERGED_ITS
        and int(info_maxit["niter"]) == 3
    )
    passed = passed and ok_maxit
    entries["maxit"] = summarize(info_maxit)
    lines.append(
        f"  maxit=3: niter {info_maxit['niter']}, converged {info_maxit['converged']},"
        f" reason {reason_name(info_maxit)}  [{verdict(ok_maxit)}]"
    )

    # 4. 守卫: 非法参数与非正对角都抛 ValueError.
    diag = primary["diag"]
    diag_with_zero = bm.set_at(bm.copy(diag), 0, 0.0)
    diag_with_negative = bm.set_at(bm.copy(diag), 0, -1.0)
    probes = [
        ("diag 含 0", lambda: DiagonalPreconditioner(diag_with_zero)),
        ("diag 含负元素", lambda: DiagonalPreconditioner(diag_with_negative)),
        ("diag 非一维", lambda: DiagonalPreconditioner(bm.reshape(diag, (-1, 1)))),
        ("非法 norm_type", lambda: CGSolver(norm_type="euclidean")),
        ("residual_refresh=-1", lambda: CGSolver(residual_refresh=-1)),
        ("divtol=0", lambda: CGSolver(divtol=0.0)),
    ]
    guard_entries: dict[str, Any] = {}
    ok_guard = True
    for description, probe in probes:
        try:
            probe()
        except ValueError as error:
            guard_entries[description] = {"raised": "ValueError", "message": str(error)}
            lines.append(f"  守卫 {description}: 抛出 ValueError  [OK]")
            continue
        except Exception as error:  # noqa: BLE001 - 抛错类型不对也是失败, 要记下来
            ok_guard = False
            guard_entries[description] = {
                "raised": type(error).__name__,
                "message": str(error),
            }
            lines.append(
                f"  守卫 {description}: 抛出 {type(error).__name__} 而非 ValueError  [FAIL]"
            )
            continue
        ok_guard = False
        guard_entries[description] = {"raised": None, "message": "未抛异常"}
        lines.append(f"  守卫 {description}: 未抛异常  [FAIL]")
    passed = passed and ok_guard
    entries["guard"] = guard_entries

    if not passed:
        report_failure("contract", lines)
    return {"passed": passed, "entries": entries}


# --------------------------------------------------------------------------
# precond
# --------------------------------------------------------------------------
def check_precond(context: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """两层级对角一致; 接入预条件子后各层级的迭代数均不劣于无预条件.

    只对 M 非空的 case 有意义; 无预条件 case 记 skipped, 不算失败. 迭代数用
    ``norm_type='unpreconditioned'`` 比, 两次求解才在同一个度量下停机.
    """
    primary = context["primary"]
    if preconditioner(case, primary) is None:
        return {"passed": True, "skipped": True, "entries": {}}

    entries: dict[str, Any] = {}
    lines: list[str] = []
    passed = True

    if context["operator_level"] == "both":
        fa, ea = context["fa"], context["ea"]
        _, diag_diff = relative_difference(ea["diag"], fa["diag"])
        ok_diag = diag_diff <= TOLERANCES["diag_rel_diff"]
        lines.append(
            f"  ||diag_ea - diag_fa|| / ||diag_fa||  {diag_diff:.3e}"
            f"  (阈值 {TOLERANCES['diag_rel_diff']:.0e})  [{verdict(ok_diag)}]"
        )
        entries["diag_rel_diff"] = diag_diff
        passed = passed and ok_diag
        active_entries = (fa, ea)
    else:
        active_entries = (primary,)

    for entry in active_entries:
        x0 = entry["prescribed"]
        level = entry["level"]
        x_plain, info_plain = solve_cg(
            entry["operator"], entry["load"], x0, norm_type="unpreconditioned"
        )
        x_pre, info_pre = solve_cg(
            entry["operator"], entry["load"], x0,
            M=preconditioner(case, entry), norm_type="unpreconditioned",
        )
        remember(context, f"precond {level} plain", info_plain)
        remember(context, f"precond {level} preconditioned", info_pre)

        _, diff = relative_difference(x_pre, context["x_direct"])
        ok_diff = diff <= TOLERANCES["solution_rel_diff"]
        ok_fewer = int(info_pre["niter"]) <= int(info_plain["niter"])
        ok_conv = bool(info_pre["converged"])
        passed = passed and ok_diff and ok_fewer and ok_conv
        entries[level] = {
            "plain": summarize(info_plain),
            "preconditioned": summarize(info_pre),
            "solution_rel_diff": diff,
        }
        lines.append(
            f"  {level}: niter 无预条件 {info_plain['niter']} -> {case['id']} {info_pre['niter']}"
            f"  [{verdict(ok_fewer)}]"
        )
        lines.append(
            f"  {level}: ||x_M - x_d|| / ||x_d||  {diff:.3e}"
            f"  (阈值 {TOLERANCES['solution_rel_diff']:.0e})  [{verdict(ok_diff and ok_conv)}]"
        )

    if not passed:
        report_failure("precond", lines)
    return {"passed": passed, "entries": entries}


# --------------------------------------------------------------------------
# solution: 加密序列上的逐层表
# --------------------------------------------------------------------------
def solve_level_pair(
    context: dict[str, Any], case: dict[str, Any], subdivisions: int
) -> dict[str, Any]:
    """在一层网格上按 operator_level 装配并求解, 返回该层的一行."""
    op_level = context["operator_level"]
    arguments = (
        context["dimension"], subdivisions, context["order"],
        context["model"], context["mesh_type"],
    )
    mf_level = matrix_free_level(op_level)
    fa = build_level(*arguments, "fa") if op_level in ("both", "fa") else None
    ea = build_level(*arguments, mf_level) if mf_level is not None else None
    primary = fa if fa is not None else ea
    x0 = primary["prescribed"]

    started = time.perf_counter()
    x_fa, info_fa = (None, None)
    if fa is not None:
        x_fa, info_fa = solve_cg(fa["operator"], fa["load"], x0, M=preconditioner(case, fa))
    elapsed = time.perf_counter() - started

    x_ea, info_ea = (None, None)
    if ea is not None:
        x_ea, info_ea = solve_cg(ea["operator"], ea["load"], x0, M=preconditioner(case, ea))

    x_sol = x_fa if x_fa is not None else x_ea
    info_sol = info_fa if info_fa is not None else info_ea

    uh = primary["space"].function()
    uh[:] = x_sol
    l2_error, l2_relative = solution_error(primary["mesh"], uh, primary["problem"], context["order"])
    level_diff = None
    if fa is not None and ea is not None:
        _, level_diff = relative_difference(x_ea, x_fa)

    converged = True
    if info_fa is not None:
        converged = converged and bool(info_fa["converged"])
    if info_ea is not None:
        converged = converged and bool(info_ea["converged"])

    return {
        "subdivisions": subdivisions,
        "mesh_size": 1.0 / subdivisions,
        "dofs": int(primary["space"].number_of_global_dofs()),
        "niter_fa": int(info_fa["niter"]) if info_fa is not None else None,
        "niter_ea": int(info_ea["niter"]) if info_ea is not None else None,
        "converged": converged,
        "reason_fa": reason_name(info_fa) if info_fa is not None else None,
        "reason_ea": reason_name(info_ea) if info_ea is not None else None,
        "relres": float(info_sol["relres"]),
        "true_relres": true_relative_residual(primary["operator"], primary["load"], x_sol),
        "level_rel_diff": level_diff,
        "l2_error": l2_error,
        "l2_relative": l2_relative,
        "seconds": float(elapsed),
    }


def print_iteration_table(
    rows: list[dict[str, Any]],
    orders: list[float | None],
    operator_level: str = "both",
) -> None:
    """逐层表: 分辨率、自由度、迭代数、报告残差、真残差、L2 误差、观测阶."""
    if operator_level == "both":
        header = (
            f"{'n':>5} {'gdof':>9} {'niter(fa)':>9} {'niter(ea)':>9} {'relres':>10}"
            f" {'true relres':>11} {'||u-u_h||_0':>13} {'order':>7}"
        )
    elif operator_level == "fa":
        header = (
            f"{'n':>5} {'gdof':>9} {'niter(fa)':>9} {'relres':>10}"
            f" {'true relres':>11} {'||u-u_h||_0':>13} {'order':>7}"
        )
    else:  # 矩阵自由层级, 'ea' 或 'pa'
        niter_column = f"niter({operator_level})"
        header = (
            f"{'n':>5} {'gdof':>9} {niter_column:>9} {'relres':>10}"
            f" {'true relres':>11} {'||u-u_h||_0':>13} {'order':>7}"
        )
    print(header)
    print("-" * len(header))
    for row, order in zip(rows, orders):
        order_text = "—" if order is None else f"{order:.3f}"
        if operator_level == "both":
            print(
                f"{row['subdivisions']:>5} {row['dofs']:>9} {row['niter_fa']:>9}"
                f" {row['niter_ea']:>9} {row['relres']:>10.2e} {row['true_relres']:>11.2e}"
                f" {row['l2_error']:>13.4e} {order_text:>7}"
            )
        elif operator_level == "fa":
            print(
                f"{row['subdivisions']:>5} {row['dofs']:>9} {row['niter_fa']:>9}"
                f" {row['relres']:>10.2e} {row['true_relres']:>11.2e}"
                f" {row['l2_error']:>13.4e} {order_text:>7}"
            )
        else:
            print(
                f"{row['subdivisions']:>5} {row['dofs']:>9} {row['niter_ea']:>9}"
                f" {row['relres']:>10.2e} {row['true_relres']:>11.2e}"
                f" {row['l2_error']:>13.4e} {order_text:>7}"
            )


def check_solution(context: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """加密序列上逐层求解: 各层级均收敛, 真残差在阈值内, 末档观测阶达门禁.

    这一项自带加密序列 (``--levels`` 与 ``--base``), 不使用 ``--n``. 通过时的全部
    输出就是这张表: 迭代数随 n 的增长与两个 case 的表对照, 就是预条件子的收益;
    L2 误差与观测阶两列在各 case 间相同, 说明预条件子只改变迭代数不改变解.
    """
    module = convergence_module()
    base = context["base"] or module.BASE_SUBDIVISIONS[context["dimension"]]
    gate = module.MINIMUM_L2_ORDER
    theoretical = float(context["order"] + 1)

    rows = [
        solve_level_pair(context, case, base * 2 ** level)
        for level in range(context["levels"])
    ]
    orders = level_orders(rows)
    final_order = orders[-1] if len(orders) > 1 else None

    ok_conv = all(row["converged"] for row in rows)
    ok_res = all(row["true_relres"] <= TOLERANCES["true_relres"] for row in rows)
    ok_order = final_order is not None and final_order >= gate
    passed = ok_conv and ok_res and ok_order

    print_iteration_table(rows, orders, context["operator_level"])
    if not passed:
        order_text = "—" if final_order is None else f"{final_order:.3f}"
        level_desc = "每层均收敛" if context["operator_level"] != "both" else "每层两层级均收敛"
        report_failure(
            "solution",
            [
                f"  {level_desc}  [{verdict(ok_conv)}]",
                f"  每层真残差 <= {TOLERANCES['true_relres']:.0e}  [{verdict(ok_res)}]",
                f"  末档观测阶 {order_text} >= 门禁 {gate:.2f} (理论 {theoretical:.0f})"
                f"  [{verdict(ok_order)}]",
            ],
        )
    return {
        "passed": passed,
        "entries": {
            "levels": [
                {**row, "l2_order": order} for row, order in zip(rows, orders)
            ],
            "final_l2_order": final_order,
            "gate": gate,
            "theoretical_order": theoretical,
        },
    }


CHECK_FUNCTIONS: dict[str, Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]] = {
    "consistency": check_consistency,
    "level": check_level,
    "norm": check_norm,
    "refresh": check_refresh,
    "batch": check_batch,
    "contract": check_contract,
    "precond": check_precond,
    "solution": check_solution,
}


def run_case(
    context: dict[str, Any], case: dict[str, Any], checks: list[str]
) -> dict[str, Any]:
    """在一个 case (一种预条件子) 下跑选中的检查."""
    # 通过时的输出只有 solution 的逐层表. 单个 case 时表无需署名; 多个 case
    # 时用一行 case=... 区分各自的表.
    if context["multiple_cases"]:
        print("")
        print(f"case={case['id']}")

    context["infos"] = []
    results: dict[str, Any] = {}
    passed = True
    for check in checks:
        outcome = CHECK_FUNCTIONS[check](context, case)
        results[check] = outcome
        passed = passed and outcome["passed"]
    return {"passed": passed, "checks": results}


# --------------------------------------------------------------------------
def list_cases(
    dimension: int,
    model: str,
    mesh_type: str,
    resolution: int,
    order: int,
    operator_level: str = DEFAULT_OPERATOR_LEVEL,
) -> int:
    """打印各 case 裸跑会跑出什么组合, 与 verify_direct_solvers 的 --list 同构."""
    problem_cls = PROBLEM_FACTORIES.get(dimension, {}).get(model)
    problem_text = (
        problem_cls.__name__ if problem_cls is not None else f"{model} {dimension}D"
    )
    mesh_text = f"{mesh_type} " + "x".join([str(resolution)] * dimension)
    analyzer_text = f"{SYSTEM_ANALYZERS[DEFAULT_SYSTEM]} p={order}"
    header = ("case-id", "preconditioner", "levels", "problem", "mesh", "analyzer")
    level_text = "fa ea" if operator_level == "both" else operator_level
    rows = [
        (
            case["id"],
            case["title"],
            level_text,
            problem_text,
            mesh_text,
            analyzer_text,
        )
        for case in CASE_REGISTRY
    ]
    widths = [
        max(display_width(row[index]) for row in (header, *rows))
        for index in range(len(header))
    ]
    for row in (header, *rows):
        print(
            "  ".join(pad(value, widths[index]) for index, value in enumerate(row)).rstrip()
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CGSolver 的正确性验证")
    parser.add_argument(
        "--list", action="store_true", help="列出各 case 及缺省组合, 不做任何求解"
    )
    parser.add_argument(
        "--case",
        nargs="+",
        default=list(CASE_IDS),
        choices=CASE_IDS,
        help="要验证的预条件子 case, 默认全部",
    )
    parser.add_argument(
        "--operator-level",
        "-L",
        dest="operator_level",
        choices=("fa", "ea", "pa", "both"),
        default=DEFAULT_OPERATOR_LEVEL,
        help="要验证的算子形态: 'fa' 完全装配, 'ea' 单元装配, 'pa' 部分装配, "
        "'both' 同时验证 fa 与 ea, 默认 fa",
    )
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3), help="空间维数, 默认 2")
    parser.add_argument(
        "--n", type=int, default=40, help="每方向单元数 (solution 以外的检查), 默认 40"
    )
    parser.add_argument("--order", type=int, default=1, help="有限元阶数, 默认 1")
    parser.add_argument(
        "--model", default=None, help="制造解模型, 默认按维数取 " + str(DEFAULT_MODELS)
    )
    parser.add_argument(
        "--mesh",
        default=None,
        choices=sorted(MESH_DIMENSIONS),
        help="网格类型, 默认按维数取 " + str(DEFAULT_MESH_TYPES),
    )
    parser.add_argument(
        "--checks",
        nargs="+",
        default=list(CHECK_IDS),
        choices=CHECK_IDS,
        help="要跑的检查, 默认全部; contract 的契约扫描只覆盖本 case 实际跑过的求解",
    )
    parser.add_argument(
        "--levels", type=int, default=5, help="solution 一项的加密层数, 默认 5"
    )
    parser.add_argument(
        "--base",
        type=int,
        default=None,
        help="solution 一项最粗一层的每方向单元数, 默认取制造解收敛脚本的 BASE_SUBDIVISIONS",
    )
    parser.add_argument("--json", action="store_true", help="把结果写入 outputs/")
    arguments = parser.parse_args()

    dimension = arguments.dim
    model = arguments.model or DEFAULT_MODELS[dimension]
    mesh_type = arguments.mesh or DEFAULT_MESH_TYPES[dimension]
    if MESH_DIMENSIONS[mesh_type] != dimension:
        print(f"网格 {mesh_type!r} 是 {MESH_DIMENSIONS[mesh_type]}D, 与 --dim {dimension} 不符")
        return 1

    if arguments.list:
        return list_cases(
            dimension, model, mesh_type, arguments.n, arguments.order, arguments.operator_level
        )

    cases = resolve_cases(arguments.case)
    if cases is None:
        return 1
    checks = [check for check in CHECK_IDS if check in arguments.checks]

    # 横幅格式同 verify_direct_solvers.
    problem_cls = PROBLEM_FACTORIES.get(dimension, {}).get(model)
    problem_name = problem_cls.__name__ if problem_cls is not None else model
    grid = ", ".join([str(arguments.n)] * dimension)
    print(
        f"CG 验证: problem={problem_name}, grid={grid}, mesh={mesh_type},"
        f" order={arguments.order}"
    )
    print("cases: " + " ".join(case["id"] for case in cases))

    op_level = arguments.operator_level
    mf_level = matrix_free_level(op_level)
    fa = (
        build_level(dimension, arguments.n, arguments.order, model, mesh_type, "fa")
        if op_level in ("both", "fa")
        else None
    )
    ea = (
        build_level(dimension, arguments.n, arguments.order, model, mesh_type, mf_level)
        if mf_level is not None
        else None
    )

    if fa is not None:
        primary = fa
        x_direct = spsolve(fa["matrix"], fa["load"], solver="scipy")
    else:
        primary = ea
        fa_ref = build_level(dimension, arguments.n, arguments.order, model, mesh_type, "fa")
        x_direct = spsolve(fa_ref["matrix"], fa_ref["load"], solver="scipy")

    number_of_dofs = int(primary["space"].number_of_global_dofs())

    context: dict[str, Any] = {
        "dimension": dimension,
        "model": model,
        "mesh_type": mesh_type,
        "order": arguments.order,
        "levels": arguments.levels,
        "operator_level": op_level,
        "base": arguments.base,
        "multiple_cases": len(cases) > 1,
        "primary": primary,
        "fa": fa,
        "ea": ea,
        "x_direct": x_direct,
        "infos": [],
    }

    case_results: dict[str, Any] = {}
    for case in cases:
        case_results[case["id"]] = run_case(context, case, checks)

    report = {
        "dimension": dimension,
        "model": model,
        "mesh_type": mesh_type,
        "resolution": arguments.n,
        "order": arguments.order,
        "levels": arguments.levels,
        "operator_level": op_level,
        "base": arguments.base,
        "number_of_dofs": number_of_dofs,
        "operator_types": {
            k: type(v["operator"]).__name__
            for k, v in [("fa", fa), (mf_level, ea)]
            if v is not None
        },
        "reference_true_relres": true_relative_residual(
            primary["operator"], primary["load"], x_direct
        ),
        "tolerances": TOLERANCES,
        "cg_parameters": CG_PARAMETERS,
        "checks": checks,
        "cases": case_results,
    }

    if arguments.json:
        output_directory = Path(__file__).resolve().parent / "outputs"
        output_directory.mkdir(exist_ok=True)
        target = output_directory / (
            f"verify_cg_solver_{dimension}d_{mesh_type}_{model}"
            f"_n{arguments.n}_p{arguments.order}.json"
        )
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # 通过时不打结论: 退出码 0 就是结论. 失败时点名哪些 case 没过.
    failed = [name for name, entry in case_results.items() if not entry["passed"]]
    if failed:
        print("")
        print("判定: FAIL  (" + " ".join(failed) + ")")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
