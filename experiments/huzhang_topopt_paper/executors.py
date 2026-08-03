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

from soptx.fem import (
    HuZhangMFEMAnalyzer,
    create_huzhang_checkerboard_mesh,
)
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import MixedBoundaryExponentialSineElasticity2D


def _as_float(value: Any) -> float:
    array = bm.to_numpy(value)
    return float(array.reshape(-1)[0])


def _uniform_mesh(level: int) -> TriangleMesh:
    subdivisions = 2**level
    return create_huzhang_checkerboard_mesh(
        box=(0.0, 1.0, 0.0, 1.0),
        nx=subdivisions,
        ny=subdivisions,
    )


def _irregular_star_mesh(level: int) -> TriangleMesh:
    node = bm.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.43, 0.57]],
        dtype=bm.float64,
    )
    cell = bm.array(
        [[4, 0, 1], [4, 1, 3], [4, 3, 2], [4, 2, 0]],
        dtype=bm.int32,
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
    error_names = (
        "displacement_l2_error",
        "stress_l2_error",
        "div_stress_l2_error",
        "stress_hdiv_error",
    )
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["mesh_family"], int(row["order"]))].append(row)
    for records in groups.values():
        records.sort(key=lambda item: int(item["level"]))
        previous: dict[str, Any] | None = None
        for record in records:
            for name in error_names:
                record[f"{name}_rate"] = None
            if previous is not None:
                h_ratio = float(previous["mesh_size"]) / float(record["mesh_size"])
                if h_ratio > 1.0:
                    for name in error_names:
                        old_error = float(previous[name])
                        new_error = float(record[name])
                        if old_error > 0.0 and new_error > 0.0:
                            record[f"{name}_rate"] = math.log(
                                old_error / new_error
                            ) / math.log(h_ratio)
            previous = record


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
    """Run the maintained mixed-boundary manufactured-solution matrix."""
    bm.set_backend("numpy")
    parameters = case.parameters
    problem = MixedBoundaryExponentialSineElasticity2D(
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
                integration_order = 2 * order + int(
                    parameters["integration_order_offset"]
                )
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
                    u=displacement,
                    v=problem.disp_solution,
                    q=integration_order,
                )
                stress_error = mesh.error(
                    u=stress,
                    v=problem.stress_solution,
                    q=integration_order,
                )
                div_stress_error = mesh.error(
                    u=stress.div_value,
                    v=problem.div_stress_solution,
                    q=integration_order,
                )
                stress_hdiv_error = bm.sqrt(
                    stress_error**2 + div_stress_error**2
                )
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
                        "normalized_normal_trace_jump": None,
                        "normal_trace_jump_status": "native-diagnostic-not-yet-implemented",
                        "wall_time_seconds": wall_time,
                        "peak_python_bytes": int(peak_python_bytes),
                        "solver_status": "completed",
                    }
                )
    _observed_rates(rows)
    _write_manufactured_figure(rows, output)
    residual_max = max(float(row["relative_equilibrium_residual"]) for row in rows)
    symmetry_max = max(float(row["state_matrix_symmetry_error"]) for row in rows)
    residual_passed = residual_max <= acceptance["relative_equilibrium_residual_max"]
    return {
        "summary": {
            "case_id": case.identifier,
            "status": "partial",
            "matrix_rows": len(rows),
            "relative_equilibrium_residual_max": residual_max,
            "relative_equilibrium_residual_passed": residual_passed,
            "state_matrix_symmetry_error_max": symmetry_max,
            "normal_trace_jump_status": "incomplete",
            "convergence_rate_status": "pending-theory-audit",
            "acceptance_status": "incomplete",
            "note": (
                "The numerical adapter is executable, but formal acceptance remains "
                "blocked by the normal-trace diagnostic and theory-backed expected orders."
            ),
        },
        "metrics": rows,
        "history": [],
    }


EXECUTORS: dict[str, Callable[..., dict[str, Any]]] = {
    "manufactured": execute_manufactured,
}
