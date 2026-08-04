"""Native executors for the Hu–Zhang common evidence matrix."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from time import perf_counter
import tracemalloc
from typing import Any, Callable

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from soptx.fem import HuZhangMFEMAnalyzer, create_huzhang_checkerboard_mesh
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import MixedBoundarySinusoidalElasticity2D


ERROR_ORDER_OFFSETS = {
    "displacement_l2_error": "displacement_l2_error_order_offset",
    "stress_l2_error": "stress_l2_error_order_offset",
    "div_stress_l2_error": "div_stress_l2_error_order_offset",
    "stress_hdiv_error": "stress_hdiv_error_order_offset",
}


def _as_float(value: Any) -> float:
    array = bm.to_numpy(value)
    return float(array.reshape(-1)[0])


def _uniform_mesh(level: int) -> TriangleMesh:
    subdivisions = 2**level
    return create_huzhang_checkerboard_mesh(
        box=(0.0, 1.0, 0.0, 1.0), nx=subdivisions, ny=subdivisions
    )


def _irregular_star_mesh(level: int) -> TriangleMesh:
    """Recursively refine a four-triangle, nonuniform star mesh."""
    node = bm.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.43, 0.57]],
        dtype=bm.float64,
    )
    cell = bm.array(
        [[4, 0, 1], [4, 1, 3], [4, 3, 2], [4, 2, 0]], dtype=bm.int32
    )
    mesh = TriangleMesh(node, cell)
    for _ in range(max(level - 1, 0)):
        mesh.uniform_refine()
    return mesh


MESH_FACTORIES: dict[str, Callable[[int], TriangleMesh]] = {
    "uniform-tri": _uniform_mesh,
    "irregular-star-refined": _irregular_star_mesh,
}


def _mesh_size(mesh: TriangleMesh) -> float:
    edge_measure = bm.to_numpy(mesh.entity_measure("edge"))
    return float(edge_measure.max())


def _observed_rates(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["mesh_family"]), int(row["order"]))].append(row)
    for records in groups.values():
        records.sort(key=lambda item: int(item["level"]))
        previous: dict[str, Any] | None = None
        for record in records:
            for name in ERROR_ORDER_OFFSETS:
                record[f"{name}_rate"] = None
            if previous is not None:
                h_ratio = float(previous["mesh_size"]) / float(record["mesh_size"])
                if h_ratio > 1.0:
                    for name in ERROR_ORDER_OFFSETS:
                        old_error = float(previous[name])
                        new_error = float(record[name])
                        if old_error > 0.0 and new_error > 0.0:
                            record[f"{name}_rate"] = math.log(
                                old_error / new_error
                            ) / math.log(h_ratio)
            previous = record


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _evaluate_manufactured_acceptance(
    rows: list[dict[str, Any]], case, acceptance: dict[str, float]
) -> dict[str, Any]:
    """Evaluate all formal diagnostics and the high-order rate gates."""
    expected_rows = len(case.mesh_families) * len(case.orders) * len(case.levels)
    expected = case.expected_orders
    high_orders = {int(order) for order in expected["high_order_gate_orders"]}
    fine_count = int(expected["last_fine_grid_count"])
    failures: list[str] = []
    if len(rows) != expected_rows:
        failures.append(f"matrix row count {len(rows)} != {expected_rows}")

    diagnostics = {
        "relative_equilibrium_residual": acceptance["relative_equilibrium_residual_max"],
        "normalized_normal_trace_jump": acceptance["normalized_normal_trace_jump_max"],
        "state_matrix_symmetry_error": acceptance["state_matrix_symmetry_error_max"],
    }
    maxima: dict[str, float | None] = {}
    for name, threshold in diagnostics.items():
        values = [row.get(name) for row in rows]
        if not values or not all(_is_finite(value) for value in values):
            failures.append(f"{name} contains a missing or non-finite value")
            maxima[name] = None
            continue
        maximum = max(float(value) for value in values)
        maxima[name] = maximum
        if maximum > threshold:
            failures.append(f"{name} maximum {maximum:.3e} exceeds {threshold:.3e}")

    for row in rows:
        for name in ERROR_ORDER_OFFSETS:
            if not _is_finite(row.get(name)):
                failures.append(f"{name} contains a missing or non-finite value")
                break
        if row.get("solver_status") != "completed":
            failures.append("a linear solve did not complete")

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["mesh_family"]), int(row["order"]))].append(row)
    rate_gates: list[dict[str, Any]] = []
    for (family, order), records in sorted(groups.items()):
        records.sort(key=lambda item: int(item["level"]))
        if order not in high_orders:
            rate_gates.append(
                {
                    "mesh_family": family,
                    "order": order,
                    "status": str(expected["low_order_policy"]),
                    "passed": None,
                }
            )
            continue
        fine_records = records[-fine_count:]
        for metric, offset_key in ERROR_ORDER_OFFSETS.items():
            theory_order = order + int(expected[offset_key])
            rates = [record.get(f"{metric}_rate") for record in fine_records]
            rate_values = [float(rate) for rate in rates if _is_finite(rate)]
            passed = (
                len(fine_records) == fine_count
                and len(rate_values) == fine_count
                and min(rate_values) >= theory_order - acceptance["high_order_rate_deficit"]
            )
            rate_gates.append(
                {
                    "mesh_family": family,
                    "order": order,
                    "metric": metric,
                    "theory_order": theory_order,
                    "observed_rates": rate_values,
                    "status": "gated",
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(f"{family}, k={order}, {metric} failed its rate gate")

    return {
        "passed": not failures,
        "failures": failures,
        "diagnostic_maxima": maxima,
        "rate_gates": rate_gates,
    }


def _write_manufactured_figure(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure_directory = output / "figures"
    figure_directory.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["mesh_family"]), int(row["order"])), []).append(row)
    for (family, order), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: int(item["total_dofs"]))
        axis.loglog(
            [int(item["total_dofs"]) for item in ordered],
            [float(item["stress_hdiv_error"]) for item in ordered],
            marker="o",
            label=f"{family}, k={order}",
        )
    axis.set_xlabel("total degrees of freedom")
    axis.set_ylabel(r"stress $H(\mathrm{div})$ error")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize="small", ncols=2)
    figure.savefig(figure_directory / "manufactured-error-vs-dofs.png", dpi=180)
    plt.close(figure)


def execute_manufactured(*, case, acceptance, output: Path) -> dict[str, Any]:
    """Run the 40-row mixed-boundary sinusoidal manufactured-solution matrix."""
    bm.set_backend("numpy")
    parameters = case.parameters
    problem = MixedBoundarySinusoidalElasticity2D(
        lame_lambda=float(parameters["lame_lambda"]),
        shear_modulus=float(parameters["shear_modulus"]),
    )
    material = IsotropicLinearElasticMaterial(
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        hypothesis=str(parameters["plane_type"]),
        enable_logging=False,
    )
    rows: list[dict[str, Any]] = []
    for family in case.mesh_families:
        try:
            mesh_factory = MESH_FACTORIES[family]
        except KeyError as error:
            raise ValueError(f"unsupported mesh family: {family}") from error
        for order in case.orders:
            for level in case.levels:
                mesh = mesh_factory(level)
                integration_order = 2 * order + int(parameters["integration_order_offset"])
                tracemalloc.start()
                started = perf_counter()
                analyzer = HuZhangMFEMAnalyzer(
                    disp_mesh=mesh,
                    pde=problem,
                    material=material,
                    interpolation_scheme=None,
                    space_degree=order,
                    integration_order=integration_order,
                    use_relaxation=bool(parameters["use_relaxation"]),
                    solve_method=str(parameters["solver"]),
                    topopt_algorithm=None,
                )
                state = analyzer.solve_state(rho_val=None)
                wall_time = perf_counter() - started
                _, peak_python_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                stress = state["stress"]
                displacement = state["displacement"]
                displacement_error = mesh.error(
                    displacement, problem.disp_solution, q=integration_order
                )
                stress_error = mesh.error(
                    stress, problem.stress_solution, q=integration_order
                )
                div_stress_error = mesh.error(
                    stress.div_value, problem.div_stress_solution, q=integration_order
                )
                stress_hdiv_error = bm.sqrt(stress_error**2 + div_stress_error**2)
                stress_dofs = analyzer.huzhang_space.number_of_global_dofs()
                displacement_dofs = analyzer.tensor_space.number_of_global_dofs()
                rows.append(
                    {
                        "case_id": case.identifier,
                        "mesh_family": family,
                        "order": order,
                        "level": level,
                        "mesh_size": _mesh_size(mesh),
                        "cells": int(mesh.number_of_cells()),
                        "stress_dofs": int(stress_dofs),
                        "displacement_dofs": int(displacement_dofs),
                        "total_dofs": int(stress_dofs + displacement_dofs),
                        "displacement_l2_error": _as_float(displacement_error),
                        "stress_l2_error": _as_float(stress_error),
                        "div_stress_l2_error": _as_float(div_stress_error),
                        "stress_hdiv_error": _as_float(stress_hdiv_error),
                        "relative_equilibrium_residual": analyzer.relative_state_residual(),
                        "state_matrix_symmetry_error": analyzer.state_matrix_symmetry_error(),
                        "normalized_normal_trace_jump": analyzer.normalized_normal_trace_jump(
                            stress, integration_order=integration_order
                        ),
                        "direct_solver_iterations": "N/A",
                        "wall_time_seconds": wall_time,
                        "peak_python_bytes": int(peak_python_bytes),
                        "solver_status": "completed",
                    }
                )
    _observed_rates(rows)
    _write_manufactured_figure(rows, output)
    evaluation = _evaluate_manufactured_acceptance(rows, case, acceptance)
    maxima = evaluation["diagnostic_maxima"]
    return {
        "summary": {
            "case_id": case.identifier,
            "status": "passed" if evaluation["passed"] else "failed",
            "matrix_rows": len(rows),
            "relative_equilibrium_residual_max": maxima["relative_equilibrium_residual"],
            "normalized_normal_trace_jump_max": maxima["normalized_normal_trace_jump"],
            "state_matrix_symmetry_error_max": maxima["state_matrix_symmetry_error"],
            "convergence_rate_status": "passed" if not any(
                gate.get("status") == "gated" and not gate.get("passed")
                for gate in evaluation["rate_gates"]
            ) else "failed",
            "rate_gates": evaluation["rate_gates"],
            "failures": evaluation["failures"],
            "acceptance_status": "passed" if evaluation["passed"] else "failed",
            "note": "peak_python_bytes is the Python traced peak, not process RSS.",
        },
        "metrics": rows,
        "history": [],
    }


EXECUTORS: dict[str, Callable[..., dict[str, Any]]] = {
    "manufactured": execute_manufactured,
}
