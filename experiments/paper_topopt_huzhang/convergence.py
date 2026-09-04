"""前向制造解收敛阶验证脚本 (对应论文第 5.1 节 / 表 5.1 与表 5.2).

本脚本计算任意次胡张混合有限元在混合边界制造解下的收敛误差与观测阶:
- 高阶原生格式: k = 3, 4
- 低阶跳量稳定化格式: k = 1, 2
算例参数取自 cases.toml 中 role = "convergence-verification" 的条目:
- manufactured-native     -> 表 5.1 (k = 3, 4, 原生格式)
- manufactured-stabilized -> 表 5.2 (k = 1, 2, 矩阵跳量稳定化)

产物按阶次增量写入 outputs/manufactured_convergence/summary.json, 供 ``run.py table``
(report.py) 一键生成论文表格: 单独重算某个 k 只覆盖该 k 的记录, 不会丢掉其余阶次的既有结果.

增量写入的代价是同一份 summary.json 可能横跨多次运行: 若代码在两次运行之间变动,
各阶次其实来自不同版本的代码, 而收敛阶是比值、对整体常数因子免疫, 表面上看不出来.
因此每次写入都给本次重算的阶次盖上 revision 与时间戳, 存于 provenance_by_degree.
"""

from __future__ import annotations

import argparse
import json
from math import log2
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import provenance
from config import (
    CASES_FILE,
    OUTPUT_DIR,
    ConfigurationError,
    bootstrap_source_path,
    load_cases,
)

bootstrap_source_path()

from fealpy.backend import backend_manager as bm  # noqa: E402
from fealpy.typing import TensorLike  # noqa: E402

from soptx.fem import (  # noqa: E402
    HuZhangMFEMAnalyzer,
    create_huzhang_checkerboard_mesh,
)
from soptx.materials import IsotropicLinearElasticMaterial  # noqa: E402
from soptx.problems import MixedBoundarySinusoidalElasticity2D  # noqa: E402

DIRECT_SOLVERS = ("scipy", "mumps")
SolverName = Literal["scipy", "mumps"]


class _ErrorEvaluableMesh(Protocol):
    def error(self, u: Any, v: Any, *, q: int) -> TensorLike: ...


class _DofCountableSpace(Protocol):
    def number_of_global_dofs(self) -> int: ...


def _as_float(value: TensorLike | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(bm.to_numpy(value).reshape(-1)[0])


def create_mesh(domain: tuple[float, ...], subdivisions: int) -> Any:
    return create_huzhang_checkerboard_mesh(
        box=domain,
        nx=subdivisions,
        ny=subdivisions,
    )


def solve_one_level(
    problem: MixedBoundarySinusoidalElasticity2D,
    material: IsotropicLinearElasticMaterial,
    degree: int,
    subdivisions: int,
    integration_order: int,
    use_relaxation: bool,
    solver: SolverName,
    stabilization: str,
) -> dict[str, Any]:
    mesh = create_mesh(problem.domain, subdivisions)

    analyzer = HuZhangMFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        interpolation_scheme=None,
        space_degree=degree,
        integration_order=integration_order,
        use_relaxation=use_relaxation,
        solve_method=solver,
        topopt_algorithm=None,
        stabilization=stabilization,
    )

    state = analyzer.solve_state(rho_val=None)
    sigmah, uh = state["stress"], state["displacement"]

    error_mesh = cast(_ErrorEvaluableMesh, mesh)
    disp_error = error_mesh.error(
        uh, problem.disp_solution, q=integration_order
    )
    stress_error = error_mesh.error(
        sigmah, problem.stress_solution, q=integration_order
    )
    div_stress_error = error_mesh.error(
        sigmah.div_value, problem.div_stress_solution, q=integration_order
    )
    stress_hdiv_error = bm.sqrt(
        bm.add(
            bm.multiply(stress_error, stress_error),
            bm.multiply(div_stress_error, div_stress_error),
        )
    )

    stress_space = cast(_DofCountableSpace, analyzer.huzhang_space)
    displacement_space = cast(_DofCountableSpace, analyzer.tensor_space)
    stress_dofs = stress_space.number_of_global_dofs()
    disp_dofs = displacement_space.number_of_global_dofs()

    return {
        "nx": subdivisions,
        "mesh_size": 1.0 / subdivisions,
        "total_dofs": int(stress_dofs + disp_dofs),
        "stress_dofs": int(stress_dofs),
        "disp_dofs": int(disp_dofs),
        "disp_l2_error": _as_float(disp_error),
        "stress_l2_error": _as_float(stress_error),
        "div_stress_l2_error": _as_float(div_stress_error),
        "stress_hdiv_error": _as_float(stress_hdiv_error),
        "relative_residual": analyzer.relative_state_residual(),
        "symmetry_error": analyzer.state_matrix_symmetry_error(),
    }


def compute_observed_orders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["disp_l2_error", "stress_l2_error", "stress_hdiv_error"]
    for i, row in enumerate(rows):
        if i == 0:
            for k in keys:
                row[f"{k}_order"] = None
        else:
            prev = rows[i - 1]
            for k in keys:
                if prev[k] > 0.0 and row[k] > 0.0:
                    row[f"{k}_order"] = log2(prev[k] / row[k])
                else:
                    row[f"{k}_order"] = None
    return rows


def print_convergence_table(degree: int, rows: list[dict[str, Any]]) -> None:
    """在终端格式化输出紧凑收敛阶数据表 (适配 80 列标准控制台)."""
    print(f"\n==================== Hu-Zhang (k = {degree}) 收敛阶实测统计 ====================")
    header = f"{'nx':>3} | {'DOF':>7} | {'h':>6} |   {'|u-uh|_0 (阶)':^16} |   {'|s-sh|_0 (阶)':^16} |  {'|s|_Hdiv (阶)':^15}"
    print(header)
    print("-" * len(header))
    for r in rows:
        u_ord = f"({r['disp_l2_error_order']:.2f})" if r["disp_l2_error_order"] is not None else "( — )"
        s_ord = f"({r['stress_l2_error_order']:.2f})" if r["stress_l2_error_order"] is not None else "( — )"
        h_ord = f"({r['stress_hdiv_error_order']:.2f})" if r["stress_hdiv_error_order"] is not None else "( — )"
        u_str = f"{r['disp_l2_error']:.4e} {u_ord:>6}"
        s_str = f"{r['stress_l2_error']:.4e} {s_ord:>6}"
        h_str = f"{r['stress_hdiv_error']:.4e} {h_ord:>6}"
        print(f"{r['nx']:3d} | {r['total_dofs']:7d} | {r['mesh_size']:6.4f} | {u_str:^18} | {s_str:^18} | {h_str:^16}")
    print("=" * len(header))


CONVERGENCE_ROLE = "convergence-verification"

# HuZhangMFEMAnalyzer 的装配分支 (huzhang_mfem_analyzer.py):
#   p >= GD + 1 -> 原生鞍点装配, stabilization 被忽略;
#   p <= GD     -> 按 stabilization 追加跳量稳定化项, 'none' 表示做消融不加.
# cases.toml 的 stabilization 字段直接驱动该参数 (不再是镜像), 断言只拒绝会被忽略的取值.
GEOMETRIC_DIMENSION = 2


STABILIZATION_CHOICES = ("none", "matrix_jump", "vector_jump")


def load_convergence_cases() -> tuple[dict[str, Any], ...]:
    selected = tuple(
        case for case in load_cases(CASES_FILE) if case.get("role") == CONVERGENCE_ROLE
    )
    if not selected:
        raise ConfigurationError(
            f"cases.toml 中没有 role = {CONVERGENCE_ROLE!r} 的算例."
        )
    return selected


def select_convergence_cases(identifiers: list[str] | None) -> tuple[dict[str, Any], ...]:
    available = load_convergence_cases()
    if not identifiers:
        return available
    index = {case["id"]: case for case in available}
    unknown = [name for name in identifiers if name not in index]
    if unknown:
        raise ConfigurationError(
            f"未知的收敛验证 case id: {', '.join(unknown)}; "
            f"可用: {', '.join(index)}."
        )
    return tuple(index[name] for name in identifiers)


def assert_stabilization_applicable(
    case: dict[str, Any], degrees: list[int], effective: str
) -> None:
    """拒绝会被静默忽略的稳定化取值.

    p >= GD + 1 时原生格式本身稳定, 分析器忽略 stabilization; 此时给出非 'none'
    的取值只会让人误以为跑的是加了稳定化的格式, 直接报错而不是算出一组名不副实的数.
    p <= GD 时三种取值都合法, 'none' 即低阶失稳消融.
    """
    if effective not in STABILIZATION_CHOICES:
        raise ConfigurationError(
            f"{case['id']}: 未知的 stabilization 取值 {effective!r}; "
            f"可选 {', '.join(STABILIZATION_CHOICES)}."
        )
    for degree in degrees:
        if degree >= GEOMETRIC_DIMENSION + 1 and effective != "none":
            raise ConfigurationError(
                f"{case['id']}: k = {degree} >= GD + 1, 原生格式本身稳定, "
                f"stabilization = {effective!r} 会被分析器忽略."
            )


def run_convergence_suite(
    case: dict[str, Any],
    degrees: list[int] | None = None,
    levels: int | None = None,
    solver: SolverName | None = None,
    use_relaxation: bool | None = None,
    stabilization: str | None = None,
    full: bool = False,
) -> dict[str, Any]:
    discretization = case["discretization"]
    parameters = case["model"]["parameters"]

    declared_degrees = [int(k) for k in discretization["comparison_orders"]]
    # 与优化侧同口径: 缺省只跑最小阶次 (一次运行), full 才展开注册表声明的整组。
    # summary.json 按阶次为键增量合并, 因此分两次单跑与一次全跑得到的论文表一致。
    if not degrees:
        degrees = declared_degrees if full else [min(declared_degrees)]
    levels = levels if levels is not None else int(discretization["levels"])
    solver = cast(SolverName, solver or discretization["solve_method"])
    if use_relaxation is None:
        use_relaxation = bool(discretization["use_relaxation"])
    base_nx = int(discretization["base_nx"])

    declared = discretization.get("stabilization")
    if declared is None:
        raise ConfigurationError(f"{case['id']}: discretization 缺少 stabilization 声明.")
    effective = stabilization if stabilization is not None else declared
    # 消融 = 生效值偏离注册表声明; 此时产物另开文件, 绝不覆盖论文表所依赖的 summary.json
    is_ablation = effective != declared
    assert_stabilization_applicable(case, degrees, effective)

    problem = MixedBoundarySinusoidalElasticity2D(
        lame_lambda=float(parameters["lame_lambda"]),
        shear_modulus=float(parameters["lame_mu"]),
    )
    material = IsotropicLinearElasticMaterial(
        hypothesis=problem.plane_type,
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        enable_logging=False,
    )

    all_results: dict[str, Any] = {}
    subdivisions_list = [base_nx * (2 ** i) for i in range(levels)]

    print(f"\n===== case: {case['id']} (论文表 {case.get('paper_table', '—')}) =====")
    print(f"      格式: {effective} | 阶次: {degrees} | 求解器: {solver}")
    if is_ablation:
        print(f"      [消融] 注册表声明为 {declared!r}, 本次覆盖为 {effective!r}, 产物另行落盘")

    for deg in degrees:
        print(f"\n>>> 正在计算 Hu-Zhang 空间次数 k = {deg} (网格序列: {subdivisions_list}) ...")
        q = 2 * deg + 2
        rows = []
        for subs in subdivisions_list:
            res = solve_one_level(
                problem=problem,
                material=material,
                degree=deg,
                subdivisions=subs,
                integration_order=q,
                use_relaxation=use_relaxation,
                solver=solver,
                stabilization=effective,
            )
            rows.append(res)
            print(
                f"  [nx={res['nx']:2d}] DOF={res['total_dofs']:7d} | "
                f"u_err={res['disp_l2_error']:.3e} | s_err={res['stress_l2_error']:.3e} | "
                f"s_Hdiv={res['stress_hdiv_error']:.3e}"
            )

        rows = compute_observed_orders(rows)
        print_convergence_table(deg, rows)
        all_results[str(deg)] = rows

    settings = {
        "case_id": case["id"],
        "paper_table": case.get("paper_table"),
        "stabilization": effective,
        "stabilization_declared": declared,
        "is_ablation": is_ablation,
        "gamma_0": discretization.get("gamma_0"),
        "base_nx": base_nx,
        "levels": levels,
        "subdivisions": subdivisions_list,
        "solver": solver,
        "use_relaxation": use_relaxation,
        "lame_lambda": problem.lam,
        "shear_modulus": problem.mu,
        "plane_type": problem.plane_type,
    }
    summary_path = write_summary(all_results, settings, is_ablation=is_ablation)
    print(f"\n[OK] 实测数据已保存至: {summary_path}")
    return all_results


PROVENANCE_KEY = "provenance_by_degree"


def _run_stamp(settings: dict[str, Any]) -> dict[str, Any]:
    """本次运行的溯源戳记: 代码版本 + 生效参数, 逐阶次记录以便识别跨版本的合并结果."""
    stamp = provenance.run_stamp()
    stamp.update(settings)
    return stamp


def write_summary(
    results: dict[str, Any], settings: dict[str, Any], is_ablation: bool = False
) -> Path:
    """按阶次合并写入汇总文件, 保留本次未重算的其余阶次记录并盖上运行戳记.

    消融运行 (稳定化取值偏离注册表声明) 写入独立的 ablation_<方法>.json:
    summary.json 是 run.py table 生成论文表 5.1 / 5.2 的唯一数据源, 按阶次为键
    增量合并, 若让消融结果落进去会静默替换掉同阶次的论文数值.
    """
    directory = OUTPUT_DIR / "manufactured_convergence"
    name = f"ablation_{settings['stabilization']}.json" if is_ablation else "summary.json"
    summary_path = directory / name
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[str, Any] = {}
    stamps: dict[str, Any] = {}
    if summary_path.is_file():
        stored = json.loads(summary_path.read_text(encoding="utf-8"))
        stamps.update(stored.pop(PROVENANCE_KEY, {}))
        merged.update(stored)
    merged.update(results)

    stamp = _run_stamp(settings)
    stamps.update({key: stamp for key in results})
    # 只为本次实际写入的阶次留戳记, 丢弃已被删除阶次的孤立记录
    stamps = {key: stamps[key] for key in sorted(stamps, key=int) if key in merged}

    ordered: dict[str, Any] = {key: merged[key] for key in sorted(merged, key=int)}
    ordered[PROVENANCE_KEY] = stamps
    summary_path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    return summary_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="制造解前向收敛阶实验")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="收敛验证 case id (可多次指定); 省略时跑 cases.toml 中全部该类算例",
    )
    parser.add_argument("--list", action="store_true", help="列出可用的收敛验证算例后退出")
    parser.add_argument(
        "--degree",
        type=int,
        action="append",
        dest="degrees",
        help="覆盖 case 的阶次 (仅允许配合单个 --case 使用); 缺省只跑其中最小的一个",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="跑完 comparison_orders 声明的全部阶次, 即论文表的完整一组.",
    )
    parser.add_argument("--levels", type=int, default=None, help="覆盖 case 的 levels")
    parser.add_argument(
        "--solver", choices=DIRECT_SOLVERS, default=None, help="覆盖 case 的 solve_method"
    )
    parser.add_argument(
        "--stabilization",
        choices=STABILIZATION_CHOICES,
        default=None,
        help="覆盖 case 的 stabilization; 偏离声明值即消融运行, 产物写 ablation_<方法>.json",
    )
    parser.add_argument(
        "--relaxation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 case 的 use_relaxation (角点松弛)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        for case in load_convergence_cases():
            discretization = case["discretization"]
            print(
                f"{case['id']:30s} 表 {case.get('paper_table', '—'):3s} "
                f"k = {discretization['comparison_orders']} "
                f"格式 = {discretization['stabilization']}"
            )
        return 0

    cases = select_convergence_cases(args.cases)
    if args.degrees and len(cases) > 1:
        raise ConfigurationError("--degree 只能配合单个 --case 使用, 否则阶次归属不明确.")
    if args.stabilization and len(cases) > 1:
        raise ConfigurationError(
            "--stabilization 只能配合单个 --case 使用, 否则消融与论文口径会混在一次运行里."
        )

    for case in cases:
        run_convergence_suite(
            case=case,
            degrees=args.degrees,
            levels=args.levels,
            solver=args.solver,
            use_relaxation=args.relaxation,
            stabilization=args.stabilization,
            full=args.full,
        )
    return 0
