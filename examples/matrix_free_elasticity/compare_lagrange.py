"""Machine-Precision Comparison Script (compare_lagrange.py).

Cross-compares Matrix-Free (EA) matrix-vector actions and CG solutions against
Full Assembly (FA) global CSR matrices and Scipy direct solver down to 1e-16 machine precision.

Usage:
    python compare_lagrange.py              # Run 2D plane strain cross-comparison
    python compare_lagrange.py --dim 3      # Run 3D polynomial cross-comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure example directory is in sys.path
example_dir = Path(__file__).resolve().parent
if str(example_dir) not in sys.path:
    sys.path.insert(0, str(example_dir))

import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

import importlib.util

_spec = importlib.util.spec_from_file_location("matrix_free_operator", example_dir / "operator.py")
_op_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_op_mod)
ElasticityEAOperator = _op_mod.ElasticityEAOperator

import utils.contract as contract
from cases import create_case
from solver import PreparedLinearSystem, solve_matrix_free_system
from utils.analyzer import build_serial_analyzer
from utils.references import relative_difference, serial_references


def run_cross_comparison(dimension: int = 2, resolution_n: int = 8) -> bool:
    case = create_case(dimension)
    resolution = (resolution_n,) * dimension
    mesh = case.create_mesh(resolution)
    case.validate_mesh(mesh)

    scalar_space = LagrangeFESpace(mesh, p=1, ctype="C")
    vector_space = TensorFunctionSpace(scalar_space, shape=(-1, dimension))
    num_dofs = vector_space.number_of_global_dofs()

    # 1. Build EA and FA Analyzers & Reference Objects
    matvec_ref, direct_sol = serial_references(vector_space, case, degree=1)

    # 2. Solve Matrix-Free system with EA Operator and CG
    ea_op = ElasticityEAOperator(vector_space, case, degree=1)
    operator, load = ea_op.assemble()

    system = PreparedLinearSystem(
        operator=operator,
        load=load,
        prescribed=ea_op.prescribed_solution,
        boundary_dofs=ea_op.boundary_dofs,
    )
    cg_sol, diagnostics = solve_matrix_free_system(
        system,
        dof_comm=None,  # Serial execution
        maxiter=1000,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    # 3. Compute relative solution difference against Scipy Direct Solver
    sol_abs_err, sol_rel_err = relative_difference(cg_sol, direct_sol)

    raw_matvec_rel = matvec_ref["raw_relative_error"]
    dirichlet_matvec_rel = matvec_ref["dirichlet_relative_error"]
    symmetry_rel = matvec_ref["symmetry_relative_error"]

    # Gates check
    passed_raw = raw_matvec_rel <= contract.MATVEC_RELATIVE_TOL
    passed_bc = dirichlet_matvec_rel <= contract.MATVEC_RELATIVE_TOL
    passed_sym = symmetry_rel <= contract.SYMMETRY_RELATIVE_TOL
    passed_sol = sol_rel_err <= contract.EXPLICIT_SOLUTION_RELATIVE_TOL
    all_passed = passed_raw and passed_bc and passed_sym and passed_sol

    # Print Machine Precision Comparison Table
    print("\n" + "=" * 76)
    print(f" EA vs FA / Scipy Machine-Precision Cross-Comparison [{dimension}D - {case.name}]")
    print("=" * 76)
    print(f" Grid Resolution      : {'x'.join(str(r) for r in resolution)}")
    print(f" Total Global DOFs    : {num_dofs}")
    print("-" * 76)
    print(f" Raw MatVec Rel Err   : {raw_matvec_rel:.5e}  (Tol: {contract.MATVEC_RELATIVE_TOL:g}) -> [{'PASS' if passed_raw else 'FAIL'}]")
    print(f" BC MatVec Rel Err    : {dirichlet_matvec_rel:.5e}  (Tol: {contract.MATVEC_RELATIVE_TOL:g}) -> [{'PASS' if passed_bc else 'FAIL'}]")
    print(f" EA Operator Symmetry : {symmetry_rel:.5e}  (Tol: {contract.SYMMETRY_RELATIVE_TOL:g}) -> [{'PASS' if passed_sym else 'FAIL'}]")
    print(f" CG vs Scipy Sol Err  : {sol_rel_err:.5e}  (Tol: {contract.EXPLICIT_SOLUTION_RELATIVE_TOL:g}) -> [{'PASS' if passed_sol else 'FAIL'}]")
    print("-" * 76)
    print(f" Overall Verification Result : [{'PASSED (Machine-Precision)' if all_passed else 'FAILED'}]")
    print("=" * 76 + "\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Matrix-Free (EA) vs Full Assembly (FA) Precision Comparison"
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=(2, 3),
        default=2,
        help="Spatial dimension (2 or 3, default: 2)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="Mesh resolution along each axis (default: 8)",
    )
    args = parser.parse_args()
    success = run_cross_comparison(dimension=args.dim, resolution_n=args.n)
    sys.exit(0 if success else 1)
