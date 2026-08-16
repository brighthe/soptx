"""Hu--Zhang 拓扑优化投稿论文实验主 Runner.

本脚本在同一物理问题、同一材料插值 (MSIMP) 与统一网格下比较两条求解路径:
1. ``LagrangeFEMAnalyzer``: 经典 Lagrange 位移有限元 (LFEM) 求解链;
2. ``HuZhangMFEMAnalyzer``: 对称弱形式胡--张混合有限元 (Hu--Zhang) 求解链.

受控比较协议:
- 给定阶次 ``k``, LFEM 采用位移阶 ``p=k``; Hu--Zhang 采用应力阶 ``k`` (对应位移阶 ``k-1``);
- 统一高斯积分阶 ``q=2k+2``;
- 载荷统一通过 ``project_patch_traction_to_p1_trace`` 投影至底边连续 P1 迹空间, 消除强施加与弱积分的几何不对齐误差;
- 比较阶次限定为 ``k=2, 3, 4`` (排除 ``k=1``, 因 P0 常数位移空间缺失刚体旋转模态, 会在拓扑演化中引发人工刚度硬化, 见 docs/fem/huzhang-mixed-fem-implementation.md).

运行模式:
- ``--mode optimization`` (默认): 执行完整 OC 拓扑优化迭代, 产物写入 ``outputs/<case-id>/<method>-k<order>/``,
  包含最终密度场 ``density_final.vtu``、收敛历史 ``history.json`` 与运行摘要 ``summary.json``;
- ``--mode state-compare``: 在固定初始密度 (rho=0.4) 下执行单次状态前向分析, 输出相对柔顺度差异与能量恒等式诊断.

使用方法:
    # 1. 运行固定梁单次状态对比 (k=2,3)
    python experiments/huzhang_topopt_paper/run.py --case compliance-fixed-fixed --method all --order 2 --order 3 --mode state-compare --solver scipy

    # 2. 运行胡张元 (k=2) 拓扑优化 (冒烟测试 3 步)
    python experiments/huzhang_topopt_paper/run.py --case compliance-fixed-fixed --method huzhang --order 2 --max-iterations 3 --solver scipy

    # 3. 运行全量论文矩阵对比 (LFEM 与 Hu--Zhang, k=2,3,4)
    python experiments/huzhang_topopt_paper/run.py --case compliance-fixed-fixed --method all --solver scipy
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Callable, cast

from fealpy.backend import backend_manager as bm

EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_ROOT / "cases.toml"
SOURCE_ROOT = EXPERIMENT_ROOT.parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from config import (
    ConfigurationError,
    UnsupportedModelError,
    configuration_summary,
    flatten_parameters,
    load_cases,
    resolve_runs,
    select_cases,
)
from diagnostics import (
    energy_identity_diagnostics,
    print_state_comparison,
    relative_residual,
    write_optimization_result,
)


def parse_arguments() -> argparse.Namespace:
    """解析算例选择、离散方法和临时覆盖参数.

    返回:
        命令行参数命名空间对象.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML 配置文件路径.")
    parser.add_argument("--list", action="store_true", help="列出已注册的论文算例.")
    parser.add_argument(
        "--case", action="append", help="算例 id, 可重复指定; 使用 all 选择所有 ready 算例."
    )
    parser.add_argument(
        "--method", choices=("lfem", "huzhang", "all"), help="运行 LFEM 基线、Hu--Zhang 或全部方法."
    )
    parser.add_argument(
        "--order", type=int, action="append", help="受控比较空间有限元次数 k, 可重复传入."
    )
    parser.add_argument("--solver", choices=("scipy", "mumps"), help="线性求解器后端.")
    parser.add_argument("--nx", type=int, help="临时覆盖横向网格剖分数.")
    parser.add_argument("--ny", type=int, help="临时覆盖纵向网格剖分数.")
    parser.add_argument("--max-iterations", type=int, help="临时覆盖最大 OC 迭代次数.")
    parser.add_argument("--check-only", action="store_true", help="仅校验配置和运行组合.")
    parser.add_argument(
        "--mode",
        choices=("optimization", "state-compare"),
        default="optimization",
        help="运行 OC 拓扑优化, 或仅执行相同初始密度下的单次状态对比.",
    )
    parser.add_argument(
        "--output", type=Path, default=EXPERIMENT_ROOT / "outputs", help="运行产物输出根目录."
    )
    return parser.parse_args()


def _fixed_fixed_builder(
    case: dict[str, Any],
    method: str,
    order: int,
    overrides: argparse.Namespace,
    *,
    analysis_only: bool = False,
) -> tuple[Any, Any]:
    """组装 FixedFixedBeamCenterLoad2d 的分析链或优化链."""
    from fixed_fixed_beam import MethodName, build_analysis_pipeline, build_config, build_pipeline

    params = flatten_parameters(case)
    config = build_config(params)
    changes = {
        field: val
        for field, val in (
            ("nx", overrides.nx),
            ("ny", overrides.ny),
            ("max_iterations", overrides.max_iterations),
            ("solve_method", overrides.solver),
        )
        if val is not None
    }
    config = replace(config, **changes)
    if config.nx <= 0 or config.ny <= 0 or config.nx % 2 or config.ny % 2:
        raise ConfigurationError("覆盖后的 nx 和 ny 必须为正偶数.")
    if config.max_iterations <= 0:
        raise ConfigurationError("覆盖后的最大迭代次数必须为正数.")

    factory = build_analysis_pipeline if analysis_only else build_pipeline
    return factory(config, params, cast(MethodName, method), order), config


MODEL_BUILDERS: dict[str, Callable[..., tuple[Any, Any]]] = {
    "FixedFixedBeamCenterLoad2d": _fixed_fixed_builder,
}


def run_one(
    case: dict[str, Any],
    method: str,
    order: int,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """执行单一阶次与离散方法的完整 OC 拓扑优化管线."""
    model_name = case["model"]["name"]
    if model_name not in MODEL_BUILDERS:
        raise UnsupportedModelError(f"{case['id']}: 模型 {model_name} 尚未注册 Runner.")

    pipeline, config = MODEL_BUILDERS[model_name](case, method, order, arguments)
    if pipeline.optimizer is None:
        raise RuntimeError("优化模式要求已创建 OCOptimizer.")

    density, history = pipeline.optimizer.optimize(
        design_variable=pipeline.design_variable,
        density_distribution=pipeline.density_distribution,
    )
    state = pipeline.analyzer.solve_state(rho_val=density)
    compliance = pipeline.objective.fun(density=density, state=state)
    volume_fraction = config.volume_fraction + float(pipeline.constraint.fun(density))

    label = f"{method}-k{order}"
    summary = {
        "case_id": case["id"],
        "model": model_name,
        "method": method,
        "order": order,
        "integration_order": 2 * order + 2,
        "compliance": float(compliance),
        "volume_fraction": volume_fraction,
        "relative_equilibrium_residual": relative_residual(pipeline, state),
        "optimization_iterations": len(history.iter_indices),
        "converged": bool(history.changes and history.changes[-1] <= config.change_tolerance),
        "solver": config.solve_method,
    }
    write_optimization_result(
        arguments.output / case["id"] / label, pipeline, density, history, summary
    )
    print(f"{case['id']}/{label}: compliance={summary['compliance']:.8e}, volfrac={volume_fraction:.6f}")
    return summary


def run_state_comparison(
    case: dict[str, Any],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """在固定初始密度场下求解单次状态方程并收集对比指标."""
    model_name = case["model"]["name"]
    if model_name not in MODEL_BUILDERS:
        raise UnsupportedModelError(f"{case['id']}: 模型 {model_name} 尚未注册 Runner.")

    rows: list[dict[str, Any]] = []
    config: Any = None
    for method, order in resolve_runs(case, arguments):
        pipeline, config = MODEL_BUILDERS[model_name](
            case, method, order, arguments, analysis_only=True
        )
        state = pipeline.analyzer.solve_state(rho_val=pipeline.density_distribution)
        compliance = pipeline.objective.fun(
            density=pipeline.density_distribution,
            state=state,
        )
        volume_fraction = config.volume_fraction + float(
            pipeline.constraint.fun(pipeline.density_distribution)
        )
        rows.append(
            {
                "method": method,
                "order": order,
                "integration_order": 2 * order + 2,
                "compliance": float(compliance),
                "volume_fraction": volume_fraction,
                "relative_equilibrium_residual": relative_residual(pipeline, state),
                "energy_diagnostics": energy_identity_diagnostics(pipeline, state),
                "cells": int(pipeline.mesh.number_of_cells()),
            }
        )

    if config is None:
        raise ConfigurationError(f"{case['id']}: state-compare 模式没有可执行的方法与阶次组合.")

    payload = {
        "case_id": case["id"],
        "model": model_name,
        "mode": "state-compare",
        "initial_density": config.volume_fraction,
        "comparison_protocol": "LFEM p=k versus Hu--Zhang stress order k, q=2k+2",
        "rows": rows,
    }
    output = arguments.output / case["id"] / "state-comparison"
    output.mkdir(parents=True, exist_ok=True)
    (output / "state_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_state_comparison(payload)
    print(f"\n{case['id']}/state-compare: 已完成 {len(rows)} 组单次状态分析.")
    return payload


def main() -> int:
    """列出、校验或执行论文拓扑优化算例."""
    arguments = parse_arguments()
    try:
        cases = load_cases(arguments.config)
        if arguments.list:
            for case in cases:
                print(f"{case['id']}\t{case['model']['name']}\t{case['status']}")
            return 0

        selected = select_cases(cases, arguments.case)
        prepared = []
        for case in selected:
            model_name = case["model"]["name"]
            if model_name not in MODEL_BUILDERS:
                raise UnsupportedModelError(f"{case['id']}: 模型 {model_name} 尚未注册 Runner.")

            runs = resolve_runs(case, arguments)
            if arguments.check_only:
                _, config = MODEL_BUILDERS[model_name](
                    case, "lfem", int(case["discretization"]["comparison_orders"][0]),
                    arguments, analysis_only=True,
                )
                prepared.append(configuration_summary(case, config, runs))
            elif arguments.mode == "state-compare":
                prepared.append(run_state_comparison(case, arguments))
            else:
                prepared.extend(run_one(case, method, order, arguments) for method, order in runs)

    except (ConfigurationError, UnsupportedModelError, KeyError, TypeError, ValueError) as error:
        print(f"配置错误: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    if arguments.check_only:
        print(json.dumps(prepared, ensure_ascii=False, indent=2))
    else:
        arguments.output.mkdir(parents=True, exist_ok=True)
        (arguments.output / "manifest.json").write_text(
            json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())