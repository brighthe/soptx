"""HarmonicPoly 问题上的角点线性迹一致性与收敛验证."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import norm as sparse_norm

from fealpy.backend import backend_manager as bm

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.substructure import (
    FEAStaticCondensation,
    GlobalAssembler,
    InterfaceSystem,
    LinearCornerTraceBasis,
    build_substructures,
    solve_constrained_system,
    solve_interface_system,
)
from soptx.problems.elasticity import HarmonicPoly2D, HarmonicPoly3D


CONSISTENCY_TOLERANCE = 1.0e-9
FREE_RESIDUAL_TOLERANCE = 1.0e-9
DEFAULT_LEVELS = {2: 4, 3: 3}
CONSISTENCY_KEYS = (
    "reference_stiffness_relative_error",
    "reference_macro_displacement_relative_error",
    "reference_full_displacement_relative_error",
    "reference_energy_relative_error",
    "equilibrium_relative_residual",
    "constraint_relative_residual",
    "reference_equilibrium_relative_residual",
    "reference_constraint_relative_residual",
    "internal_relative_residual",
)


def _problem(dim: int) -> Any:
    """构造指定维度的调和多项式制造解问题."""
    if dim == 2:
        return HarmonicPoly2D(domain=(0.0, 1.0, 0.0, 1.0))
    if dim == 3:
        return HarmonicPoly3D(domain=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    raise ValueError("dim 必须为 2 或 3.")


def _as_scipy(matrix: Any) -> csr_matrix:
    """将 FEALPy 或 SciPy 稀疏矩阵规整为 CSR."""
    return (
        matrix.to_scipy().tocsr()
        if hasattr(matrix, "to_scipy")
        else matrix.tocsr()
    )


def _relative_error(value: Any, reference: Any) -> float:
    """计算数组相对二范数误差."""
    value = np.asarray(value)
    reference = np.asarray(reference)
    scale = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(value - reference)) / scale


def _free_residual(
    stiffness: csr_matrix,
    displacement: Any,
    prescribed: Any,
    fixed_mask: Any,
) -> float:
    """按非齐次 Dirichlet 驱动量归一化自由自由度平衡残差."""
    u = np.asarray(bm.to_numpy(displacement), dtype=np.float64)
    fixed = np.asarray(bm.to_numpy(fixed_mask), dtype=bool)
    u_c = np.zeros_like(u)
    prescribed_np = np.asarray(bm.to_numpy(prescribed), dtype=np.float64)
    u_c[fixed] = prescribed_np[fixed]
    residual = (stiffness @ u)[~fixed]
    driving = (stiffness @ u_c)[~fixed]
    scale = max(float(np.linalg.norm(driving)), np.finfo(float).tiny)
    return float(np.linalg.norm(residual)) / scale


def _build_full_extension(
    assembler: GlobalAssembler,
    sub_meshes: Sequence[Any],
    condensor: FEAStaticCondensation,
    trace_basis: LinearCornerTraceBasis,
) -> csr_matrix:
    """构造宏观角点到全细网格的延拓, 并检查共享行一致性."""
    if condensor.N is None:
        raise RuntimeError("构造全场延拓前必须先完成 Exact Schur 缩聚.")
    trace = np.asarray(bm.to_numpy(trace_basis.matrix), dtype=np.float64)
    recovery = np.asarray(bm.to_numpy(condensor.N), dtype=np.float64)
    n_batch = len(sub_meshes)
    n_local = int(sub_meshes[0].n_total_dofs)
    local = np.zeros((n_batch, n_local, trace_basis.n_trace_dofs))
    b_dofs = np.asarray(bm.to_numpy(condensor.b_dofs), dtype=np.int64)
    i_dofs = np.asarray(bm.to_numpy(condensor.i_dofs), dtype=np.int64)
    local[:, b_dofs] = trace
    local[:, i_dofs] = recovery @ trace

    batches, rows, cols = np.nonzero(local)
    corners = np.asarray(
        bm.to_numpy(assembler.macro_corner_indices(sub_meshes)), dtype=np.int64
    )
    candidates = coo_matrix(
        (
            local[batches, rows, cols],
            (batches * n_local + rows, corners[batches, cols]),
        ),
        shape=(n_batch * n_local, assembler.total_macro_dofs),
    ).tocsr()
    positions = assembler.substructure_positions(sub_meshes)
    global_rows = np.stack(
        [
            np.asarray(
                bm.to_numpy(assembler.get_substructure_global_dofs(pos, mesh)),
                dtype=np.int64,
            )
            for pos, mesh in zip(positions, sub_meshes, strict=True)
        ]
    ).ravel()
    unique_rows, first = np.unique(global_rows, return_index=True)
    if not np.array_equal(unique_rows, np.arange(assembler.total_full_dofs)):
        raise ValueError("全场延拓未覆盖全部细网格自由度.")
    extension = candidates[first].tocsr()
    difference = candidates - extension[global_rows]
    if difference.nnz and np.max(np.abs(difference.data)) > 1.0e-12:
        raise ValueError("相邻子结构在共享自由度上的延拓不一致.")
    return extension


def _macro_constraints(
    assembler: GlobalAssembler,
    pde: Any,
) -> tuple[csr_matrix, np.ndarray]:
    """在外边界宏观角点上构造解析位移约束."""
    coordinates = np.asarray(
        bm.to_numpy(assembler.macro_node_coordinates()), dtype=np.float64
    )
    prescribed = np.asarray(
        bm.to_numpy(pde.dirichlet_bc(bm.asarray(coordinates))), dtype=np.float64
    ).reshape(-1)
    boundary_nodes = np.any(
        np.isclose(coordinates, 0.0, rtol=0.0, atol=1.0e-12)
        | np.isclose(
            coordinates, np.asarray(assembler.domain_size),
            rtol=0.0, atol=1.0e-12,
        ),
        axis=1,
    )
    fixed = np.flatnonzero(np.repeat(boundary_nodes, assembler.dim))
    constraints = coo_matrix(
        (np.ones(len(fixed)), (np.arange(len(fixed)), fixed)),
        shape=(len(fixed), assembler.total_macro_dofs),
    ).tocsr()
    return constraints, prescribed[fixed]


def _solve_one_level(
    dim: int,
    n_sub: tuple[int, ...],
    n_fine: tuple[int, ...],
    solve_method: str,
) -> dict[str, Any]:
    """求解一层角点迹问题并与独立装配的同迹空间参照比较."""
    pde = _problem(dim)
    assembler = GlobalAssembler(
        (1.0,) * dim, n_sub, n_fine, degree=1, E_base=pde.E, nu=pde.nu
    )
    prototype, sub_meshes, _ = build_substructures(
        assembler, integration_order=4
    )
    density = bm.ones(
        (len(sub_meshes),) + tuple(assembler.n_fine), dtype=bm.float64
    )
    local_stiffness = prototype.assemble_local_stiffness_batch(density)
    condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    condensor.condense(local_stiffness)
    trace_basis = LinearCornerTraceBasis.from_prototype(prototype)

    macro = assembler.assemble_trace_system(
        sub_meshes, condensor, trace_basis=trace_basis
    )
    interface_dofs = assembler.build_interface_dofs(sub_meshes)
    interface_view = InterfaceSystem(macro.stiffness, interface_dofs)
    interface_projection = assembler.build_linear_corner_projection(
        sub_meshes, interface_view, trace_basis
    )
    constraints, values = _macro_constraints(assembler, pde)
    zero_force = np.zeros(assembler.total_macro_dofs)
    solved = solve_constrained_system(
        macro, zero_force, constraints,
        prescribed=values, solver=solve_method,
    )
    macro_u = np.asarray(bm.to_numpy(solved.displacement))
    interface_u = np.asarray(interface_projection @ macro_u)
    displacement = np.asarray(
        bm.to_numpy(assembler.recover_full_displacement(
            sub_meshes, condensor, interface_view, bm.asarray(interface_u)
        ))
    )

    extension = _build_full_extension(
        assembler, sub_meshes, condensor, trace_basis
    )
    fa = LagrangeFEMAnalyzer(
        disp_mesh=assembler.full_mesh,
        pde=pde,
        material=prototype.material,
        space_degree=1,
        integration_order=4,
        assembly_method="standard",
        operator_level="fa",
        solve_method=solve_method,
        tensor_space=assembler.space_full,
        enable_logging=False,
    )
    full_stiffness = _as_scipy(fa.assemble_stiff_matrix())
    reference_stiffness = (extension.T @ full_stiffness @ extension).tocsr()
    reference_system = InterfaceSystem(
        reference_stiffness,
        bm.arange(assembler.total_macro_dofs, dtype=bm.int64),
    )
    reference_solved = solve_constrained_system(
        reference_system, zero_force, constraints,
        prescribed=values, solver=solve_method,
    )
    reference_macro_u = np.asarray(bm.to_numpy(reference_solved.displacement))
    reference_displacement = np.asarray(extension @ reference_macro_u)

    exact_boundary, boundary_mask = assembler.space_full.boundary_interpolate(
        gd=pde.dirichlet_bc,
        threshold=pde.is_dirichlet_boundary(),
        method="interp",
    )
    boundary = np.asarray(bm.to_numpy(boundary_mask), dtype=bool)
    exact = np.asarray(bm.to_numpy(exact_boundary), dtype=np.float64)
    fixed_global = np.flatnonzero(boundary)
    full_system = InterfaceSystem(
        full_stiffness,
        bm.arange(assembler.total_full_dofs, dtype=bm.int64),
    )
    fa_u = np.asarray(bm.to_numpy(solve_interface_system(
        full_system,
        np.zeros(assembler.total_full_dofs),
        fixed_global,
        prescribed=exact,
        solver=solve_method,
    )), dtype=np.float64)

    uh = assembler.space_full.function()
    uh[:] = bm.asarray(displacement)
    l2_error = float(assembler.full_mesh.error(pde.disp_solution, uh, q=4))
    h1_error = float(
        assembler.full_mesh.error(pde.grad_disp_solution, uh.grad_value, q=4)
    )
    if not math.isfinite(l2_error) or l2_error <= 0.0:
        raise AssertionError("linear_corner 位移 L2 误差必须为有限正数.")
    if not math.isfinite(h1_error) or h1_error <= 0.0:
        raise AssertionError("linear_corner 位移 H1 半范误差必须为有限正数.")

    fa_uh = assembler.space_full.function()
    fa_uh[:] = bm.asarray(fa_u)
    fa_l2_error = float(
        assembler.full_mesh.error(pde.disp_solution, fa_uh, q=4)
    )
    fa_h1_error = float(
        assembler.full_mesh.error(pde.grad_disp_solution, fa_uh.grad_value, q=4)
    )
    for name, error in (("L2", fa_l2_error), ("H1 半范", fa_h1_error)):
        if not math.isfinite(error) or error <= 0.0:
            raise AssertionError(f"FA {name} 误差必须为有限正数.")

    macro_stiffness = _as_scipy(macro.stiffness)
    stiffness_scale = max(
        float(sparse_norm(reference_stiffness)), np.finfo(float).tiny
    )
    macro_energy = 0.5 * float(macro_u @ (macro_stiffness @ macro_u))
    reference_energy = 0.5 * float(
        reference_macro_u @ (reference_stiffness @ reference_macro_u)
    )
    fa_energy = 0.5 * float(fa_u @ (full_stiffness @ fa_u))
    fa_displacement_relative_error = _relative_error(displacement, fa_u)
    fa_energy_relative_error = abs(macro_energy - fa_energy) / max(
        abs(fa_energy), np.finfo(float).tiny
    )
    fa_free_residual = _free_residual(
        full_stiffness, fa_u, exact, boundary
    )
    for name, value, positive in (
        ("macro_energy", macro_energy, True),
        ("fa_energy", fa_energy, True),
        ("fa_displacement_relative_error", fa_displacement_relative_error, False),
        ("fa_energy_relative_error", fa_energy_relative_error, False),
    ):
        if not math.isfinite(value) or (positive and value <= 0.0):
            raise AssertionError(f"linear_corner {name} 不是有效有限值.")
    if (
        not math.isfinite(fa_free_residual)
        or fa_free_residual > FREE_RESIDUAL_TOLERANCE
    ):
        raise AssertionError(
            f"FA free_residual={fa_free_residual:.4e} 超过阈值 "
            f"{FREE_RESIDUAL_TOLERANCE:.1e}."
        )
    boundary_error = _relative_error(
        displacement[boundary], exact[boundary]
    )

    boundary_indices = np.asarray(
        bm.to_numpy(assembler.interface_indices(sub_meshes, interface_dofs)),
        dtype=np.int64,
    )
    local_boundary_u = interface_u[boundary_indices]
    local_u = np.zeros((len(sub_meshes), prototype.n_total_dofs))
    b_dofs = np.asarray(bm.to_numpy(prototype.b_dofs), dtype=np.int64)
    i_dofs = np.asarray(bm.to_numpy(prototype.i_dofs), dtype=np.int64)
    local_u[:, b_dofs] = local_boundary_u
    local_u[:, i_dofs] = np.asarray(
        bm.to_numpy(condensor.recover(bm.asarray(local_boundary_u)))
    )
    local_force = np.einsum(
        "bij,bj->bi", np.asarray(bm.to_numpy(local_stiffness)), local_u
    )
    internal_residual = float(np.linalg.norm(local_force[:, i_dofs])) / max(
        float(np.linalg.norm(local_force)), np.finfo(float).tiny
    )

    consistency = {
        "reference_stiffness_relative_error": float(
            sparse_norm(macro_stiffness - reference_stiffness)
        ) / stiffness_scale,
        "reference_macro_displacement_relative_error": _relative_error(
            macro_u, reference_macro_u
        ),
        "reference_full_displacement_relative_error": _relative_error(
            displacement, reference_displacement
        ),
        "reference_energy_relative_error": (
            abs(macro_energy - reference_energy)
            / max(abs(reference_energy), np.finfo(float).tiny)
        ),
        "equilibrium_relative_residual": solved.equilibrium_relative_residual,
        "constraint_relative_residual": solved.constraint_relative_residual,
        "reference_equilibrium_relative_residual": (
            reference_solved.equilibrium_relative_residual
        ),
        "reference_constraint_relative_residual": (
            reference_solved.constraint_relative_residual
        ),
        "internal_relative_residual": internal_residual,
    }
    for key in CONSISTENCY_KEYS:
        value = consistency[key]
        if not math.isfinite(value) or value > CONSISTENCY_TOLERANCE:
            raise AssertionError(
                f"linear_corner {key}={value:.4e} 超过一致性阈值 "
                f"{CONSISTENCY_TOLERANCE:.1e}."
            )

    total_fine = tuple(n_sub[d] * n_fine[d] for d in range(dim))
    return {
        "n_sub": list(n_sub),
        "n_fine": list(n_fine),
        "total_fine": list(total_fine),
        "mesh_size": max(1.0 / value for value in total_fine),
        "full_dofs": assembler.total_full_dofs,
        "macro_dofs": assembler.total_macro_dofs,
        "interface_dofs": len(interface_dofs),
        "linear_corner": {
            "l2_error": l2_error,
            "h1_semi_error": h1_error,
            "boundary_interpolation_relative_error": boundary_error,
            "macro_energy": macro_energy,
            "reference_energy": reference_energy,
            "fa_displacement_relative_error": fa_displacement_relative_error,
            "fa_energy_relative_error": fa_energy_relative_error,
            **consistency,
        },
        "fa": {
            "l2_error": fa_l2_error,
            "h1_semi_error": fa_h1_error,
            "strain_energy": fa_energy,
            "free_residual": fa_free_residual,
        },
    }


def run_linear_corner_convergence(
    dim: int,
    *,
    levels: int | None = None,
    output_dir: str | None = None,
    solve_method: str = "scipy",
) -> dict[str, Any]:
    """运行制造解上的角点线性迹网格加密验证."""
    if dim not in (2, 3):
        raise ValueError("dim 必须为 2 或 3.")
    if levels is None:
        levels = DEFAULT_LEVELS[dim]
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 2:
        raise ValueError("levels 必须为不小于 2 的整数.")
    if solve_method not in ("scipy", "mumps"):
        raise ValueError("solve_method 必须为 scipy 或 mumps.")

    bm.set_backend("numpy")
    pde = _problem(dim)
    n_fine = (2,) * dim
    print("verification linear_corner harmonic convergence and consistency")
    print(f"problem  {type(pde).__name__}, Q1, rho=1")
    print("boundary 宏观外边界角点取解析位移, 其余边界由线性迹插值")
    print(
        "层级  网格         corner L2    阶    FA L2        阶    "
        "相对FA位移    相对FA能量    同迹刚度误差"
    )
    results = []
    for level in range(levels):
        count = 2 * 2**level
        result = _solve_one_level(
            dim, (count,) * dim, n_fine, solve_method
        )
        for route in ("linear_corner", "fa"):
            for metric in ("l2", "h1_semi"):
                order = None
                if results:
                    previous = results[-1]
                    error_ratio = (
                        previous[route][f"{metric}_error"]
                        / result[route][f"{metric}_error"]
                    )
                    mesh_ratio = previous["mesh_size"] / result["mesh_size"]
                    order = math.log(error_ratio) / math.log(mesh_ratio)
                    if not math.isfinite(order):
                        raise AssertionError(
                            f"{route} {metric} 观测阶不是有限数."
                        )
                result[route][f"{metric}_order"] = order
        results.append(result)
        values = result["linear_corner"]
        fa_values = result["fa"]
        grid = "x".join(map(str, result["total_fine"]))
        l2_order = "--" if values["l2_order"] is None else f"{values['l2_order']:.2f}"
        fa_l2_order = (
            "--" if fa_values["l2_order"] is None
            else f"{fa_values['l2_order']:.2f}"
        )
        print(
            f"{level + 1:<4}  {grid:<11}  {values['l2_error']:.4e}  "
            f"{l2_order:<4}  {fa_values['l2_error']:.4e}  {fa_l2_order:<4}  "
            f"{values['fa_displacement_relative_error']:.2e}      "
            f"{values['fa_energy_relative_error']:.2e}      "
            f"{values['reference_stiffness_relative_error']:.2e}",
            flush=True,
        )

    summary = {
        "schema_version": "linear-corner-convergence-fa-reference-v2",
        "dimension": f"{dim}D",
        "problem": type(pde).__name__,
        "domain": list(pde.domain),
        "space_degree": 1,
        "youngs_modulus": pde.E,
        "poisson_ratio": pde.nu,
        "material_hypothesis": "plane_stress" if dim == 2 else "3D",
        "backend": "numpy",
        "integration_order": 4,
        "trace_basis": "linear_corner",
        "boundary_condition": (
            "exact_macro_corner_values_with_linear_trace_interpolation"
        ),
        "fa_reference": {
            "discretization": "same_mesh_same_q1_material_density_and_quadrature",
            "boundary_condition": "exact_displacement_on_all_fine_boundary_dofs",
            "comparison_interpretation": (
                "includes_outer_boundary_trace_interpolation_and_internal_"
                "interface_reduction_error"
            ),
            "strain_energy_definition": "0.5 * u.T @ K @ u",
            "free_residual_tolerance": FREE_RESIDUAL_TOLERANCE,
            "free_residual_normalization": "norm(K_fc @ u_c)",
            "linear_corner_displacement_energy": "reported_without_gate",
        },
        "density": 1.0,
        "base_subdivisions": 2,
        "refinement_levels": levels,
        "n_fine": list(n_fine),
        "solver": solve_method,
        "consistency_tolerance": CONSISTENCY_TOLERANCE,
        "free_residual_tolerance": FREE_RESIDUAL_TOLERANCE,
        "reference": (
            "full_fine_grid_fa_stiffness_projected_by_shared_exact_schur_"
            "full_field_extension"
        ),
        "levels": results,
        "validation": {
            "linear_corner_consistency": "PASS",
            "linear_corner_convergence": "REPORTED",
            "fa_convergence": "REPORTED",
            "fa_free_residual": "PASS",
            "linear_corner_relative_fa_error": "REPORTED",
        },
        "passed": True,
    }

    final = results[-1]["linear_corner"]
    print(
        "PASS, 同迹空间一致性指标均不超过 "
        f"{CONSISTENCY_TOLERANCE:.1e}; 解析误差与观测阶只报告."
    )
    print(
        f"最后一层 L2 阶={final['l2_order']:.4f}, "
        f"H1 半范阶={final['h1_semi_order']:.4f}"
    )
    final_fa = results[-1]["fa"]
    print(
        "同网格 FA 参照完成, "
        f"FA L2 阶={final_fa['l2_order']:.4f}, "
        f"H1 半范阶={final_fa['h1_semi_order']:.4f}, "
        f"自由残差={final_fa['free_residual']:.2e}; "
        "linear_corner 相对 FA 位移与能量误差只报告."
    )

    if output_dir is not None:
        target = Path(output_dir) / (
            f"linear_corner_convergence_{dim}d_harmonic-poly_p1_"
            f"levels{levels}_solver-{solve_method}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        print(f"结果: {target.resolve()}")
    return summary