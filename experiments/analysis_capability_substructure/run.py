"""精确子结构分析实验统一入口.

用法:
    python run.py --list
    python run.py --case full_trace_convergence_2d
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import AnalysisCase, ConfigError, OUTPUT_DIR, load  # noqa: E402


def _format_shape(values: tuple[int, ...] | None) -> str:
    return "--" if values is None else "x".join(str(value) for value in values)


def _case_grid_labels(case: AnalysisCase) -> tuple[str, str]:
    """返回工况的子结构划分范围与块内网格说明."""
    if case.task in ("full_trace_convergence", "linear_corner_convergence"):
        # 与两种解析解验证的网格规则一致.
        base_sub = 2
        last_sub = base_sub * 2 ** (case.levels - 1)
        first = _format_shape((base_sub,) * case.dim)
        last = _format_shape((last_sub,) * case.dim)
        return f"{first} -> {last}", _format_shape((2,) * case.dim)
    if case.task == "density_update_consistency":
        return (
            "待确定" if case.n_sub is None else _format_shape(case.n_sub),
            "待确定" if case.n_fine is None else _format_shape(case.n_fine),
        )
    return _format_shape(case.n_sub), _format_shape(case.n_fine)

def _print_cases(cases: tuple[AnalysisCase, ...]) -> None:
    """列出工况及其已接入的验证任务."""
    header = ("case-id", "task", "problem", "n_sub", "n_fine")
    rows = [
        (
            case.id,
            case.task,
            case.problem,
            *_case_grid_labels(case),
        )
        for case in cases
    ]
    widths = [
        max(len(row[index]) for row in (header, *rows))
        for index in range(len(header))
    ]
    for row in (header, *rows):
        print(
            "  ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            ).rstrip()
        )


def _new_run_dir(output_root: Path, case_id: str) -> Path:
    """为单次运行创建不会覆盖历史证据的时间戳目录."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = output_root / case_id / stamp
    target.mkdir(parents=True, exist_ok=False)
    return target


def run_case(
    case: AnalysisCase,
    *,
    meta: dict[str, Any],
    output_root: Path,
    monitor: bool = False,
) -> dict[str, Any]:
    """调用已有验证实现执行一个注册工况."""
    if (
        case.task in ("density_update_consistency", "route_cost")
        and (case.n_sub is None or case.n_fine is None)
    ):
        raise ConfigError(f"{case.id}: 必须先设置 n_sub 和 n_fine")

    output_dir = _new_run_dir(output_root, case.id)
    snapshot = {
        "meta": meta,
        "case": asdict(case),
        "runtime": {
            "monitor": monitor,
            "monitor_interval": case.monitor_interval,
        },
    }
    if case.task == "density_update_consistency":
        from experiments.analysis_capability_substructure._density_update import (
            density_update_config,
        )

        snapshot["density_update"] = density_update_config()
    if case.task == "route_cost":
        from experiments.analysis_capability_substructure._cost_measurement import (
            cost_measurement_config,
        )

        snapshot["cost_measurement"] = cost_measurement_config(case.routes[0])
    (output_dir / "run_config.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[run] {case.id}: {case.summary}")
    if case.task == "full_trace_convergence":
        print(f"[run] degree={case.degree}")
    print(f"[run] 结果目录: {output_dir}")

    if case.task == "route_cost":
        from experiments.analysis_capability_substructure._cost_measurement import (
            run_route_cost_2d,
        )

        assert case.n_sub is not None
        assert case.n_fine is not None
        assert case.warmup is not None
        assert case.repeat is not None
        return run_route_cost_2d(
            str(output_dir),
            route=case.routes[0],
            n_sub=case.n_sub,
            n_fine=case.n_fine,
            warmup=case.warmup,
            repeat=case.repeat,
            solve_method=case.solve_method,
            monitor=monitor,
            monitor_interval=case.monitor_interval,
            expected_displacement_sha256=case.expected_displacement_sha256,
            expected_strain_energy=case.expected_strain_energy,
        )
    if case.task == "density_update_consistency":
        from experiments.analysis_capability_substructure._density_update import (
            run_density_update_consistency,
        )

        assert case.n_sub is not None
        assert case.n_fine is not None
        return run_density_update_consistency(
            case.dim,
            str(output_dir),
            n_sub=case.n_sub,
            n_fine=case.n_fine,
            route=case.routes[0],
            solve_method=case.solve_method,
            monitor=monitor,
            monitor_interval=case.monitor_interval,
        )
    if case.task == "full_trace_convergence":
        from examples.substructure_elasticity.verify_full_trace_convergence import (
            run_convergence_benchmark,
        )

        return run_convergence_benchmark(
            dim=case.dim,
            model="harmonic-poly",
            degree=case.degree,
            levels=case.levels,
            output_dir=str(output_dir),
            solve_method=case.solve_method,
            compare_fa=True,
        )

    if case.task == "linear_corner_convergence":
        from experiments.analysis_capability_substructure._corner_convergence import (
            run_linear_corner_convergence,
        )

        return run_linear_corner_convergence(
            case.dim,
            levels=case.levels,
            output_dir=str(output_dir),
            solve_method=case.solve_method,
        )

    raise ConfigError(f"{case.id}: 未支持的 task {case.task}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="精确子结构分析的正确性、收敛性与计算成本统一入口",
        allow_abbrev=False,
    )
    parser.add_argument("--case", help="工况 id 或 all")
    parser.add_argument("--list", action="store_true", help="只列出注册工况")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="输出根目录，默认 experiments/analysis_capability_substructure/outputs",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="密度更新或性能工况运行时显示独立 Worker 的 CPU 与内存看板",
    )
    arguments = parser.parse_args()

    try:
        meta, cases = load()
    except ConfigError as error:
        print(f"[error] {error}")
        return 2
    if arguments.list:
        _print_cases(cases)
        return 0
    if arguments.case is None:
        parser.print_help()
        print("\n请使用 --case <id> 选择工况，或先用 --list 查看注册表。")
        return 2

    selected = cases if arguments.case == "all" else tuple(
        case for case in cases if case.id == arguments.case
    )
    if not selected:
        print(f"[error] 未找到工况: {arguments.case}")
        return 2
    if arguments.monitor and any(
        case.task not in ("density_update_consistency", "route_cost")
        for case in selected
    ):
        print("[error] --monitor 只适用于密度更新或性能工况")
        return 2

    incomplete = [
        case.id
        for case in selected
        if (
            case.task in ("density_update_consistency", "route_cost")
            and (case.n_sub is None or case.n_fine is None)
        )
    ]
    if incomplete:
        print(
            "[error] 以下工况尚未设置 n_sub 和 n_fine: "
            + ", ".join(incomplete)
        )
        return 2

    for case in selected:
        run_case(
            case,
            meta=meta,
            output_root=arguments.output_dir,
            monitor=arguments.monitor,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
