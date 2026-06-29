"""Benchmark SOPTX matrix-free elasticity correctness and interface closure.

This script is intentionally a small, reproducible NumPy benchmark.  It records
assembled-vs-matrix-free correctness, CG closure, basic timings, and memory
estimates for presentation/report tables.  It is not a GPU or MPI performance
benchmark.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.fem import BilinearForm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import TetrahedronMesh, TriangleMesh
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.sparse.linalg import spsolve

from soptx.analysis.integrators.linear_elastic_integrator import LinearElasticIntegrator
from soptx.analysis.matrix_free import MatrixFreeCGSolver, MatrixFreeElasticityOperator
from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial


@dataclass
class BenchmarkRow:
    case: str
    dim: int
    backend: str
    ncell: int
    ndof: int
    nnz: int
    assembly_time_s: float
    assembled_matvec_time_s: float
    matrix_free_matvec_time_s: float
    rel_matvec_error: float
    cg_converged: bool
    cg_iterations: int
    cg_final_residual: float
    matrix_free_solve_time_s: float
    solve_rel_error: float
    assembled_memory_mb: float
    matrix_free_memory_est_mb: float


def _to_numpy(x):
    if hasattr(bm, "to_numpy"):
        return bm.to_numpy(x)
    return np.asarray(x)


def _time_average(fn: Callable[[], object], repeat: int) -> tuple[object, float]:
    result = None
    start = perf_counter()
    for _ in range(repeat):
        result = fn()
    elapsed = (perf_counter() - start) / repeat
    return result, elapsed


def _sparse_memory_mb(matrix) -> tuple[int, float]:
    scipy_matrix = matrix.to_scipy().tocoo()
    bytes_used = (
        scipy_matrix.data.nbytes
        + scipy_matrix.row.nbytes
        + scipy_matrix.col.nbytes
    )
    return scipy_matrix.nnz, bytes_used / 1024**2


def _matrix_free_memory_est_mb(space, coef) -> float:
    """Estimate matrix-free operator data used by this Python prototype."""
    cell2dof = space.cell_to_dof()
    bytes_used = _to_numpy(cell2dof).nbytes
    if coef is not None:
        bytes_used += _to_numpy(coef).nbytes
    return bytes_used / 1024**2


def _build_case(dim: int, n: int):
    if dim == 2:
        mesh = TriangleMesh.from_box(
            box=[0.0, 1.0, 0.0, 1.0],
            nx=n,
            ny=n,
        )
        plane_type = "plane_stress"
        fixed_threshold = lambda p: bm.abs(p[..., 0]) < 1.0e-12
        q = 4
    elif dim == 3:
        mesh = TetrahedronMesh.from_box(
            box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            nx=n,
            ny=n,
            nz=n,
        )
        plane_type = "3d"
        fixed_threshold = lambda p: bm.abs(p[..., 0]) < 1.0e-12
        q = 4
    else:
        raise ValueError(f"Unsupported dim={dim}")

    scalar_space = LagrangeFESpace(mesh, p=1, ctype="C")
    tensor_space = TensorFunctionSpace(
        scalar_space,
        shape=(-1, mesh.geo_dimension()),
    )
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0,
        poisson_ratio=0.3,
        plane_type=plane_type,
    )
    integrator = LinearElasticIntegrator(
        material=material,
        q=q,
        method="standard",
    )
    coef = bm.linspace(0.4, 1.0, mesh.number_of_cells())
    integrator.coef = coef
    return mesh, tensor_space, integrator, coef, fixed_threshold


def run_case(dim: int, n: int, repeat: int, cg_tol: float, cg_maxiter: int) -> BenchmarkRow:
    mesh, tensor_space, integrator, coef, fixed_threshold = _build_case(dim, n)
    case = f"{dim}d_n{n}"

    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator)

    assembly_start = perf_counter()
    K = bform.assembly(format="coo")
    assembly_time = perf_counter() - assembly_start

    nnz, assembled_memory_mb = _sparse_memory_mb(K)
    mf_memory_mb = _matrix_free_memory_est_mb(tensor_space, coef)

    # From here on, matrix-free action must use contraction, not local Ke @ xe.
    integrator._disable_action_assembly_fallback = True

    op = MatrixFreeElasticityOperator(
        space=tensor_space,
        integrator=integrator,
        rho=coef,
    )

    ndof = tensor_space.number_of_global_dofs()
    x = bm.linspace(0.1, 1.0, ndof)

    y_ref, assembled_matvec_time = _time_average(lambda: K.matmul(x), repeat)
    y_mf, mf_matvec_time = _time_average(lambda: op.matvec(x), repeat)
    rel_matvec_error = float(bm.linalg.norm(y_ref - y_mf) / bm.linalg.norm(y_ref))

    fixed_dofs = bm.where(
        tensor_space.is_boundary_dof(
            threshold=fixed_threshold,
            method="interp",
        )
    )[0]
    rhs = bm.linspace(0.2, 1.1, ndof)
    rhs = bm.set_at(rhs, fixed_dofs, 0.0)

    K_bc = K.to_scipy().tolil()
    fixed_dofs_np = np.asarray(_to_numpy(fixed_dofs), dtype=int)
    K_bc[fixed_dofs_np, :] = 0.0
    K_bc[:, fixed_dofs_np] = 0.0
    K_bc[fixed_dofs_np, fixed_dofs_np] = 1.0
    u_ref = spsolve(K_bc.tocsr(), np.asarray(_to_numpy(rhs)))

    op_bc = MatrixFreeElasticityOperator(
        space=tensor_space,
        integrator=integrator,
        rho=coef,
        dirichlet_dofs=fixed_dofs,
    )
    solver = MatrixFreeCGSolver(tol=cg_tol, maxiter=cg_maxiter)
    solve_start = perf_counter()
    u_mf, info = solver.solve(op_bc, rhs)
    solve_time = perf_counter() - solve_start
    solve_rel_error = float(bm.linalg.norm(u_ref - u_mf) / bm.linalg.norm(u_ref))

    return BenchmarkRow(
        case=case,
        dim=dim,
        backend="numpy",
        ncell=mesh.number_of_cells(),
        ndof=ndof,
        nnz=nnz,
        assembly_time_s=assembly_time,
        assembled_matvec_time_s=assembled_matvec_time,
        matrix_free_matvec_time_s=mf_matvec_time,
        rel_matvec_error=rel_matvec_error,
        cg_converged=info.converged,
        cg_iterations=info.iterations,
        cg_final_residual=info.final_residual,
        matrix_free_solve_time_s=solve_time,
        solve_rel_error=solve_rel_error,
        assembled_memory_mb=assembled_memory_mb,
        matrix_free_memory_est_mb=mf_memory_mb,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SOPTX matrix-free elasticity NumPy benchmark.",
    )
    parser.add_argument(
        "--output",
        default="outputs/matrix_free_elasticity_benchmark.csv",
        help="CSV output path relative to the repository root.",
    )
    parser.add_argument(
        "--xlsx-output",
        default=None,
        help="Formatted XLSX output path. Defaults to the CSV path with .xlsx suffix.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of matvec repeats for average timing.",
    )
    parser.add_argument(
        "--cases-2d",
        default="4,8,16",
        help="Comma-separated 2D mesh sizes.",
    )
    parser.add_argument(
        "--cases-3d",
        default="1,2",
        help="Comma-separated 3D mesh sizes.",
    )
    parser.add_argument("--cg-tol", type=float, default=1.0e-10)
    parser.add_argument("--cg-maxiter", type=int, default=500)
    return parser.parse_args()


def _parse_sizes(text: str) -> list[int]:
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _write_csv(rows: list[BenchmarkRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_xlsx(rows: list[BenchmarkRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(row) for row in rows]
    headers = list(data[0].keys())

    wb = Workbook()
    ws = wb.active
    ws.title = "matrix_free_benchmark"

    ws.append(headers)
    for item in data:
        ws.append([item[key] for key in headers])

    header_fill = PatternFill(fill_type="solid", fgColor="D9EAF7")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    scientific_cols = {
        "rel_matvec_error",
        "cg_final_residual",
        "solve_rel_error",
    }
    time_cols = {
        "assembly_time_s",
        "assembled_matvec_time_s",
        "matrix_free_matvec_time_s",
        "matrix_free_solve_time_s",
    }
    memory_cols = {
        "assembled_memory_mb",
        "matrix_free_memory_est_mb",
    }
    integer_cols = {"dim", "ncell", "ndof", "nnz", "cg_iterations"}

    for col_idx, header in enumerate(headers, start=1):
        col_letter = get_column_letter(col_idx)
        max_len = len(header)
        for row_idx in range(2, ws.max_row + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            max_len = max(max_len, len(str(cell.value)))
            if header in scientific_cols:
                cell.number_format = "0.000E+00"
            elif header in time_cols:
                cell.number_format = "0.000000"
            elif header in memory_cols:
                cell.number_format = "0.000"
            elif header in integer_cols:
                cell.number_format = "0"
            elif header == "cg_converged":
                cell.alignment = Alignment(horizontal="center")

        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 24)

    wb.save(output)


def main() -> None:
    args = parse_args()
    bm.set_backend("numpy")

    rows = []
    for n in _parse_sizes(args.cases_2d):
        rows.append(run_case(2, n, args.repeat, args.cg_tol, args.cg_maxiter))
    for n in _parse_sizes(args.cases_3d):
        rows.append(run_case(3, n, args.repeat, args.cg_tol, args.cg_maxiter))

    csv_output = Path(args.output)
    xlsx_output = Path(args.xlsx_output) if args.xlsx_output else csv_output.with_suffix(".xlsx")
    _write_csv(rows, csv_output)
    _write_xlsx(rows, xlsx_output)

    print(f"Wrote {len(rows)} rows to {csv_output}")
    print(f"Wrote formatted workbook to {xlsx_output}")
    for row in rows:
        print(
            f"{row.case}: ndof={row.ndof}, rel_matvec={row.rel_matvec_error:.3e}, "
            f"solve_rel={row.solve_rel_error:.3e}, cg={row.cg_iterations}, "
            f"K_mem={row.assembled_memory_mb:.3f} MB, MF_est={row.matrix_free_memory_est_mb:.3f} MB"
        )


if __name__ == "__main__":
    main()
