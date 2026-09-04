# -*- coding: utf-8 -*-
"""子结构缩聚变密度拓扑优化执行器 (精确缩聚, 接口迹可选; 2D/3D MBB).

优化循环 (modified SIMP + 灵敏度锥形滤波 + OC) 与
experiments/piml_substructure_topopt/run.py 的 FEA 基线逐参数一致
(move=0.2, damping=0.5, initial_lambda=1e9, bisection_tol=1e-4,
design_variable_min=1e-3, 收敛判据为连续 5 步 |dC|/C < tol_change 且 it >= 10),
因此同工况的柔度历史可以直接与该目录及 experiments/topopt_simp_fa 对照。

正问题 (装配 / 缩聚 / 求解 / 恢复 / 单元应变能) 全部在 pipeline.py 中,
本文件只负责优化状态、落盘与命令行。

用法:
    python run.py --list
    python run.py --case mbb_2d_full_trace
    python run.py --case all --max-iter 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fealpy.backend import backend_manager as bm

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from soptx.postprocess.vtk_export import write_vtu  # noqa: E402
from soptx.topology.filters import (  # noqa: E402
    apply_structured_density_filter,
    apply_structured_density_filter_adjoint,
    apply_structured_sensitivity_filter,
)
from soptx.topology.optimizers import OCOptimizer  # noqa: E402

import provenance  # noqa: E402
from config import CASES_FILE, OUTPUT_DIR, ConfigError, TopOptCase, load  # noqa: E402
from pipeline import build_components, compliance_sensitivity, solve_forward  # noqa: E402

OC_OPTIONS = {
    "move_limit": 0.2,
    "damping_coef": 0.5,
    "initial_lambda": 1.0e9,
    "bisection_tol": 1.0e-4,
    "design_variable_min": 1.0e-3,
}
CONVERGENCE_WINDOW = 5
CONVERGENCE_MIN_ITER = 10


def _filter_layout(ctx: Any, values: Any) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """把扁平单元场转换为过滤器使用的三维结构化布局."""
    values_np = np.asarray(bm.to_numpy(values), dtype=np.float64)
    if ctx.case.dim == 2:
        grid = values_np.reshape((*ctx.n_elem_grid, 1))
        spacing = (*ctx.h_grid, max(ctx.domain_size) * 10.0)
    else:
        grid = values_np.reshape(ctx.n_elem_grid)
        spacing = ctx.h_grid
    return grid, spacing


def _filter_sensitivity(ctx: Any, dc: Any, rho: Any) -> Any:
    """结构化锥形灵敏度滤波 (2D 需补一维厚度方向)."""
    dc_grid, spacing = _filter_layout(ctx, dc)
    rho_grid, _ = _filter_layout(ctx, rho)
    filtered = apply_structured_sensitivity_filter(
        sensitivity=dc_grid,
        density=rho_grid,
        rmin=ctx.case.filter_radius,
        spacing=spacing,
        kind="cone",
    )
    return bm.asarray(filtered.flatten(), dtype=bm.float64)


def _filter_density(ctx: Any, rho: Any) -> Any:
    """设计密度到物理密度的结构化锥形过滤."""
    rho_grid, spacing = _filter_layout(ctx, rho)
    filtered = apply_structured_density_filter(
        density=rho_grid,
        rmin=ctx.case.filter_radius,
        spacing=spacing,
    )
    return bm.asarray(filtered.flatten(), dtype=bm.float64)


def _backprop_density_gradient(ctx: Any, gradient: Any) -> Any:
    """把物理密度梯度经密度过滤 Jacobian 转置回传至设计变量."""
    gradient_grid, spacing = _filter_layout(ctx, gradient)
    design_gradient = apply_structured_density_filter_adjoint(
        gradient=gradient_grid,
        rmin=ctx.case.filter_radius,
        spacing=spacing,
    )
    return bm.asarray(design_gradient.flatten(), dtype=bm.float64)


def _save_topology_png(
    ctx: Any,
    rho_np: Any,
    history: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case = ctx.case
    rho_plot = rho_np.reshape(ctx.n_elem_grid).T  # 转置以符合 2D (y, x) 图像显示
    plt.figure(figsize=(10, 2.5), dpi=300)
    plt.imshow(
        1.0 - rho_plot,
        cmap="gray",
        origin="lower",
        extent=[case.domain[0], case.domain[1], case.domain[2], case.domain[3]],
    )
    plt.title(
        f"{case.id} ({case.trace}, iter {len(history)}, "
        f"C={history[-1]['compliance']:.4f})"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    figure_path = output_dir / "topology.png"
    plt.savefig(figure_path)
    plt.close()
    print(f"[run] 拓扑云图: {figure_path}")


def run_case(
    case: TopOptCase,
    *,
    output_root: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    write_vtk: bool = True,
) -> Dict[str, Any]:
    """执行单个工况的完整拓扑优化闭环, 返回 summary 字典."""
    output_dir = (output_root or OUTPUT_DIR) / case.id
    output_dir.mkdir(parents=True, exist_ok=True)
    vtu_dir = output_dir / "vtu"
    if write_vtk:
        vtu_dir.mkdir(parents=True, exist_ok=True)
        for stale in vtu_dir.glob("density_iter_*.vtu"):
            stale.unlink()

    print(f"[run] {case.id}: {case.summary}")
    t_setup = time.perf_counter()
    ctx = build_components(case)
    setup_time = time.perf_counter() - t_setup
    print(
        f"[run] trace={case.trace} reduction={case.reduction} "
        f"子结构 {ctx.n_sub_total} 个, 单元 {ctx.n_elem_total} 个, "
        f"全局迹自由度 {ctx.n_global_trace_dofs}, 组装耗时 {setup_time:.2f} s"
    )
    header = (
        f"{'iter':>5} | {'compliance':>12} | {'volfrac':>8} | "
        f"{'change':>10} | {'solve/ms':>10}"
    )
    print(header)
    print("-" * len(header))

    rho = bm.full((ctx.n_elem_total,), case.volfrac, dtype=bm.float64)
    if case.filter_type == "density":
        volume_gradient = _backprop_density_gradient(
            ctx,
            bm.ones(rho.shape, dtype=bm.float64),
        )
        constraint_gradient = volume_gradient
        constraint_function = lambda candidate: (
            bm.mean(volume_gradient * candidate) - case.volfrac
        )
    else:
        constraint_gradient = bm.ones(rho.shape, dtype=bm.float64)
        constraint_function = lambda candidate: bm.mean(candidate) - case.volfrac

    history: List[Dict[str, Any]] = []
    recent_relative_changes: List[float] = []
    converged = False
    t_total = time.perf_counter()

    for it in range(1, case.max_iter + 1):
        t_step = time.perf_counter()
        rho_physical = _filter_density(ctx, rho) if case.filter_type == "density" else rho
        forward = solve_forward(ctx, rho_physical)
        dc = compliance_sensitivity(ctx, rho_physical, forward.energy)
        dc_design = (
            _backprop_density_gradient(ctx, dc)
            if case.filter_type == "density"
            else _filter_sensitivity(ctx, dc, rho)
        )

        if write_vtk:
            rho_np = np.asarray(bm.to_numpy(rho_physical), dtype=np.float64)
            write_vtu(
                ctx.assembler.full_mesh,
                cell_data={"density": rho_np},
                filepath=str(vtu_dir / f"density_iter_{it:04d}"),
            )

        rho_new = OCOptimizer.update_design_variable(
            design_variable=rho,
            objective_gradient=dc_design,
            constraint_gradient=constraint_gradient,
            constraint_function=constraint_function,
            **OC_OPTIONS,
        )

        change = float(bm.max(bm.abs(rho_new - rho)))
        rho_new_physical = (
            _filter_density(ctx, rho_new)
            if case.filter_type == "density"
            else rho_new
        )
        volfrac_current = float(bm.mean(rho_new_physical))
        history.append(
            {
                "iter": it,
                "compliance": float(forward.compliance),
                "volfrac": volfrac_current,
                "change": change,
                "iteration_time": time.perf_counter() - t_step,
                "solve_time_ms": float(forward.solve_time_ms),
            }
        )

        if len(history) >= 2:
            previous = history[-2]["compliance"]
            recent_relative_changes.append(
                abs(forward.compliance - previous) / abs(previous)
            )
            if len(recent_relative_changes) > CONVERGENCE_WINDOW:
                recent_relative_changes.pop(0)

        if it % 5 == 0 or it == 1 or it == case.max_iter:
            print(
                f"{it:5d} | {forward.compliance:12.4f} | {volfrac_current:8.4f} | "
                f"{change:10.5f} | {forward.solve_time_ms:10.2f}"
            )

        rho = bm.copy(rho_new)

        if (
            len(recent_relative_changes) == CONVERGENCE_WINDOW
            and all(value < case.tol_change for value in recent_relative_changes)
            and it >= CONVERGENCE_MIN_ITER
        ):
            converged = True
            print(
                f"[run] 满足收敛判据 (连续 {CONVERGENCE_WINDOW} 步 "
                f"|dC|/C < {case.tol_change}), 迭代结束于第 {it} 步"
            )
            break

    total_time = time.perf_counter() - t_total
    rho_final = _filter_density(ctx, rho) if case.filter_type == "density" else rho
    rho_final_np = np.asarray(bm.to_numpy(rho_final), dtype=np.float64)
    np.save(output_dir / "density_final.npy", rho_final_np)
    if case.filter_type == "density":
        np.save(
            output_dir / "design_density_final.npy",
            np.asarray(bm.to_numpy(rho), dtype=np.float64),
        )
    if write_vtk:
        write_vtu(
            ctx.assembler.full_mesh,
            cell_data={"density": rho_final_np},
            filepath=str(output_dir / "density_final"),
        )
    if case.dim == 2:
        _save_topology_png(ctx, rho_final_np, history, output_dir)

    last = history[-1]
    summary = {
        "case_id": case.id,
        "method": "substructure-condensation-SIMP",
        "trace": case.trace,
        "reduction": case.reduction,
        "backend": "fealpy-numpy",
        "dimension": case.dim,
        "problem": "full_mbb_beam",
        "domain": list(case.domain),
        "n_sub": list(case.n_sub),
        "n_fine": list(case.n_fine),
        "integration_order": case.integration_order,
        "chunk_size": case.chunk_size,
        "n_substructures": ctx.n_sub_total,
        "grid": list(ctx.n_elem_grid),
        "n_cells": int(rho_final_np.size),
        "n_global_trace_dofs": ctx.n_global_trace_dofs,
        "n_boundary_dofs_per_sub": int(ctx.trace_basis.n_boundary_dofs),
        "n_trace_dofs_per_sub": int(ctx.trace_basis.n_trace_dofs),
        "volfrac": case.volfrac,
        "penal": case.simp_penalty,
        "emin": case.emin,
        "emax": case.emax,
        "nu": case.nu,
        "p_load": case.p_load,
        "filter": case.filter_type,
        "filter_radius": case.filter_radius,
        "vtk_enabled": write_vtk,
        "optimizer": "oc",
        "optimizer_options": dict(OC_OPTIONS),
        "max_iter": case.max_iter,
        "tol_change": case.tol_change,
        "seed": case.seed,
        "fa_reference": case.fa_reference or None,
        "overrides": overrides or None,
        "iterations": len(history),
        "converged": converged,
        "final_compliance": float(last["compliance"]),
        "final_volume_fraction": float(last["volfrac"]),
        "final_change": float(last["change"]),
        "setup_time": setup_time,
        "total_time": total_time,
        "mean_iteration_time": float(
            np.mean([record["iteration_time"] for record in history])
        ),
        "provenance": provenance.capture((CASES_FILE,)),
    }
    for name, payload in (("history.json", history), ("summary.json", summary)):
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"[run] 完成: 迭代 {len(history)} 步, 最终柔度 "
        f"{last['compliance']:.6f}, 体积分数 {last['volfrac']:.4f}, "
        f"总耗时 {total_time:.2f} s -> {output_dir}"
    )
    return summary


def _apply_overrides(
    case: TopOptCase, arguments: argparse.Namespace
) -> Tuple[TopOptCase, Dict[str, Any]]:
    overrides: Dict[str, Any] = {}
    if arguments.max_iter is not None:
        overrides["max_iter"] = int(arguments.max_iter)
    if arguments.volfrac is not None:
        overrides["volfrac"] = float(arguments.volfrac)
    return (replace(case, **overrides) if overrides else case), overrides


def _format_shape(shape: Tuple[int, ...]) -> str:
    """把各方向数量格式化为紧凑的 ``nx x ny [x nz]`` 文本."""
    return "x".join(str(value) for value in shape)


def _print_case_table(cases: Tuple[TopOptCase, ...]) -> None:
    """按问题规模、分析链与算法轴列出注册工况."""
    header = (
        "id",
        "mesh",
        "n_sub",
        "n_fine",
        "trace",
        "analyzer",
        "optimizer",
        "role",
    )
    rows = []
    for case in cases:
        global_mesh = tuple(
            case.n_sub[d] * case.n_fine[d] for d in range(case.dim)
        )
        cell_type = "quad" if case.dim == 2 else "hex"
        rows.append(
            (
                case.id,
                f"{cell_type} {_format_shape(global_mesh)}",
                _format_shape(case.n_sub),
                _format_shape(case.n_fine),
                case.trace,
                "lfem-p1",
                "oc",
                case.role,
            )
        )
    widths = [
        max(len(row[index]) for row in (header, *rows))
        for index in range(len(header))
    ]
    for row in (header, *rows):
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="子结构缩聚变密度拓扑优化执行器 (2D/3D)"
    )
    parser.add_argument("--case", type=str, default="all", help="工况 ID 或 all")
    parser.add_argument("--list", action="store_true", help="只列出注册工况")
    parser.add_argument("--max-iter", type=int, default=None, help="覆盖最大迭代步数")
    parser.add_argument("--volfrac", type=float, default=None, help="覆盖目标体积分数")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="覆盖输出根目录 (默认 outputs/)"
    )
    parser.add_argument(
        "--no-vtk",
        action="store_true",
        help="不写逐步/最终 VTU，避免论文原规模下构造完整可视化网格",
    )
    arguments = parser.parse_args()

    try:
        meta, cases = load()
    except ConfigError as error:
        print(f"[error] {error}")
        return 2

    if arguments.list:
        _print_case_table(cases)
        return 0

    selected = (
        cases
        if arguments.case == "all"
        else tuple(case for case in cases if case.id == arguments.case)
    )
    if not selected:
        print(f"[error] 未找到工况: {arguments.case}")
        return 2

    for case in selected:
        resolved, overrides = _apply_overrides(case, arguments)
        run_case(
            resolved,
            output_root=arguments.output_dir,
            overrides=overrides,
            write_vtk=not arguments.no_vtk,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
