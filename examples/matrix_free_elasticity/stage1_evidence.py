# -*- coding: utf-8 -*-
"""CPU 串行 stage-1 EA/FA 证据生成器.

本脚本只跑 stage-1 的 1a 串行范围: 2D/3D 各六个单 rank 算例
(EA/FA × coarse/medium/fine, 档位 ``n = 8/16/32``), 产出逐档相对 L2 误差、
同档 EA/FA 解相对差与观测阶, 写成汇总 JSON, 供
``experiments/matrix_free_capability`` 的表 a-2 使用。

图 2 的证据链因此不再经过 ``tools/`` 的验证管线; 阈值仍从
``tools/matrix_free_evidence/contract.py`` 读取, 与 ``verify_ea_correctness.py``
同一方式, 保持阈值单一来源。

用法:
    python examples/matrix_free_elasticity/stage1_evidence.py --dim all \\
        --output-dir experiments/matrix_free_capability/outputs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import TetrahedronMesh, TriangleMesh

from soptx.fem.analyzers import build_serial_analyzer
from soptx.fem.verification import (
    relative_difference,
    serial_references,
    solution_error,
)
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems.elasticity import (
    DivergenceFreePolynomialElasticity3D,
    SinusoidalPlaneStrainElasticity2D,
)
from tools.matrix_free_evidence import contract

# 维数决定这三样, 其余一律向制造解对象要, 防止制造解与算子参数漂移.
PROBLEM_FACTORIES = {
    2: SinusoidalPlaneStrainElasticity2D,
    3: DivergenceFreePolynomialElasticity3D,
}
MESH_FACTORIES = {2: TriangleMesh, 3: TetrahedronMesh}
MATERIAL_HYPOTHESES = {2: "plane_strain", 3: "3D"}
OPERATOR_STORAGE = {"ea": "cached-element-matrices", "fa": "global-csr"}


def case_name(role: str, operator_level: str) -> str:
    """算例名, 与 stage-1 既有命名一致."""
    return f"{operator_level}-{role}-1rank"


def run_case(
    dimension: int,
    role: str,
    operator_level: str,
    refinement: int,
) -> dict | None:
    """跑一个 CPU 串行算例并返回证据载荷.

    参数:
        dimension: 空间维度.
        role: 档位角色, 取 ``coarse``/``medium``/``fine``.
        operator_level: 算子层级, 取 ``ea`` 或 ``fa``.
        refinement: 每轴剖分数.

    返回:
        payload: 含解向量、误差、显式参考与 matvec 核对的字典;
            求解或门禁失败时记录失败并返回 ``None``.
    """
    problem = PROBLEM_FACTORIES[dimension]()
    mesh = MESH_FACTORIES[dimension].from_box(
        list(problem.domain),
        **dict(zip(("nx", "ny", "nz"), (refinement,) * dimension)),
    )
    scalar_space = LagrangeFESpace(mesh, p=contract.DEFAULT_DEGREE, ctype="C")
    vector_space = TensorFunctionSpace(
        scalar_space,
        shape=(-1, dimension),
    )
    material = IsotropicLinearElasticMaterial(
        hypothesis=MATERIAL_HYPOTHESES[dimension],
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        device=bm.get_device(mesh),
    )
    analyzer = build_serial_analyzer(
        vector_space,
        problem,
        material,
        contract.DEFAULT_DEGREE,
        operator_level,
    )

    # 黄金参考: EA/FA matvec 核对 + FA 显式组装 spsolve 直接解。
    matvec_reference, direct_solution = serial_references(
        vector_space,
        problem,
        material,
        contract.DEFAULT_DEGREE,
        seed=contract.REFERENCE_RANDOM_SEED,
    )

    operator, load = analyzer.apply_bc(
        analyzer.assemble_stiff_matrix(),
        analyzer.assemble_body_force_vector(),
    )
    solution = bm.zeros_like(load)
    # fealpy 的 cg 要求初值为后端张量; prescribed_solution 在 apply_bc 后非空。
    x0 = bm.asarray(analyzer.prescribed_solution, dtype=bm.float64)
    _, cg_info = analyzer.solve_system(
        operator,
        load,
        solution,
        x0=x0,
        solver="cg",
        maxiter=contract.DEFAULT_MAX_ITERATIONS,
        rtol=contract.DEFAULT_RTOL,
        atol=contract.DEFAULT_ATOL,
    )

    solution_function = vector_space.function(dtype=bm.float64)
    solution_function[:] = solution
    explicit_absolute, explicit_relative = relative_difference(
        solution_function,
        direct_solution,
    )
    l2_absolute, l2_relative = solution_error(
        mesh,
        solution_function,
        problem,
        contract.DEFAULT_DEGREE,
    )

    return {
        "solution": np.asarray(solution_function),
        "parameters": {
            "dimension": dimension,
            "case": type(problem).__name__,
            "domain": list(problem.domain),
            "resolution": [refinement] * dimension,
            "lagrange_p": contract.DEFAULT_DEGREE,
            "maxit": contract.DEFAULT_MAX_ITERATIONS,
            "rtol": contract.DEFAULT_RTOL,
            "atol": contract.DEFAULT_ATOL,
            "operator_level": operator_level,
            "operator_storage": OPERATOR_STORAGE[operator_level],
            "benchmark": False,
            "reference_random_seed": contract.REFERENCE_RANDOM_SEED,
        },
        "mpi_size": 1,
        "operator": {
            "level": operator_level,
            "storage": OPERATOR_STORAGE[operator_level],
        },
        "solver": {
            "name": "cg",
            "converged": bool(cg_info.get("converged", False)),
            "iterations": int(cg_info.get("niter", 0)),
        },
        "error": {
            "l2_absolute": float(l2_absolute),
            "l2_relative": float(l2_relative),
        },
        "explicit_solution_reference": {
            "absolute_error": float(explicit_absolute),
            "relative_error": float(explicit_relative),
        },
        "matvec_reference": {
            key: float(matvec_reference[key])
            for key in (
                "raw_absolute_error",
                "raw_relative_error",
                "dirichlet_absolute_error",
                "dirichlet_relative_error",
                "random_vector_energy",
            )
        },
    }


def _solution_relative_difference(left: dict, right: dict) -> float:
    """两个算例解向量的相对差, 口径同 stage-1."""
    a = left["solution"]
    b = right["solution"]
    return float(
        np.linalg.norm(a - b)
        / max(np.linalg.norm(a), contract.NORM_FLOOR)
    )


def compare_cases(
    dimension: int,
    results: dict[str, dict],
) -> tuple[dict, list[str]]:
    """逐档计算 EA 误差链、EA/FA 解相对差与观测阶, 并判定门禁.

    参数:
        dimension: 空间维度.
        results: 全部六个算例的载荷字典, 键为 ``case_name``.

    返回:
        comparison: 与 stage-1 同构的比对结果字典.
        failures: 未通过的门禁项.
    """
    failures: list[str] = []
    ea_names = [
        case_name(role, "ea") for role in ("coarse", "medium", "fine")
    ]

    differences: dict[str, float] = {}
    for role in ("coarse", "medium", "fine"):
        difference = _solution_relative_difference(
            results[case_name(role, "ea")],
            results[case_name(role, "fa")],
        )
        differences[role] = difference
        if difference > contract.EA_FA_SOLUTION_RELATIVE_TOL:
            failures.append(
                f"{dimension}d: {role} EA/FA solution difference "
                f"{difference:.16e} > {contract.EA_FA_SOLUTION_RELATIVE_TOL:g}"
            )

    errors = [
        float(results[name]["error"]["l2_relative"]) for name in ea_names
    ]
    orders = [
        math.log2(previous / current)
        for previous, current in zip(errors, errors[1:])
    ]
    if not all(previous > current for previous, current in zip(errors, errors[1:])):
        failures.append(
            f"{dimension}d: relative L2 error did not decrease: "
            + ", ".join(f"{value:.16e}" for value in errors)
        )
    if orders[-1] < contract.MINIMUM_FINAL_L2_ORDER:
        failures.append(
            f"{dimension}d: final relative L2 order "
            f"{orders[-1]:.8f} < {contract.MINIMUM_FINAL_L2_ORDER}"
        )

    comparison = {
        "stage": "1a",
        "coarse_solution_ea_fa_relative_difference": differences["coarse"],
        "ea_fa_solution_relative_differences": differences,
        "relative_l2_errors": dict(zip(ea_names, errors)),
        "observed_relative_l2_orders": orders,
        "gated_relative_l2_order": orders[-1],
        "minimum_gated_relative_l2_order": contract.MINIMUM_FINAL_L2_ORDER,
    }
    return comparison, failures


def _check_local_gates(
    dimension: int,
    name: str,
    payload: dict,
    failures: list[str],
) -> None:
    """单算例门禁: matvec 一致性、正定性、CG/spsolve 显式核对、CG 收敛."""
    label = f"{dimension}d/{name}"
    matvec = payload["matvec_reference"]
    if matvec["raw_relative_error"] > contract.MATVEC_RELATIVE_TOL:
        failures.append(
            f"{label}: raw EA/FA MatVec error "
            f"{matvec['raw_relative_error']:.16e} > {contract.MATVEC_RELATIVE_TOL:g}"
        )
    if matvec["dirichlet_relative_error"] > contract.MATVEC_RELATIVE_TOL:
        failures.append(
            f"{label}: Dirichlet EA/FA MatVec error "
            f"{matvec['dirichlet_relative_error']:.16e} > {contract.MATVEC_RELATIVE_TOL:g}"
        )
    if matvec["random_vector_energy"] <= 0.0:
        failures.append(f"{label}: random-vector energy is not positive")
    explicit = payload["explicit_solution_reference"]
    if explicit["relative_error"] > contract.EXPLICIT_SOLUTION_RELATIVE_TOL:
        failures.append(
            f"{label}: CG/assembled solution error "
            f"{explicit['relative_error']:.16e} > "
            f"{contract.EXPLICIT_SOLUTION_RELATIVE_TOL:g}"
        )
    if not payload["solver"]["converged"]:
        failures.append(f"{label}: CG did not converge")


def _strip_solution(payload: dict) -> dict:
    """从载荷中剔除解向量, 得到可序列化的算例记录."""
    return {key: value for key, value in payload.items() if key != "solution"}


def parse_arguments() -> argparse.Namespace:
    """命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", choices=("2", "3", "all"), default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


def main(argv: list[str] | None = None) -> int:
    """运行全部选中的维度并写汇总 JSON."""
    arguments = parse_arguments()
    dimensions = (2, 3) if arguments.dim == "all" else (int(arguments.dim),)

    failures: list[str] = []
    dimensions_block: dict[str, dict] = {}
    for dimension in dimensions:
        results: dict[str, dict] = {}
        for role, refinement in zip(
            ("coarse", "medium", "fine"),
            contract.REFINEMENTS[dimension],
        ):
            for operator_level in ("ea", "fa"):
                name = case_name(role, operator_level)
                print(f"[{dimension}d/{name}] n={refinement}", flush=True)
                payload = run_case(dimension, role, operator_level, refinement)
                if payload is None:
                    failures.append(f"{dimension}d/{name}: run failed")
                    continue
                _check_local_gates(dimension, name, payload, failures)
                results[name] = payload

        if len(results) != 6:
            failures.append(
                f"{dimension}d: only {len(results)}/6 cases completed"
            )
            dimensions_block[str(dimension)] = {
                "passed": False,
                "cases": {name: _strip_solution(p) for name, p in results.items()},
                "comparison": None,
            }
            continue

        comparison, comparison_failures = compare_cases(dimension, results)
        failures.extend(comparison_failures)
        dimensions_block[str(dimension)] = {
            "passed": not comparison_failures,
            "cases": {name: _strip_solution(p) for name, p in results.items()},
            "comparison": comparison,
        }

    evidence = {
        "schema_version": 1,
        "stage": contract.STAGE,
        "substage": "1a",
        "selected_dimensions": list(dimensions),
        "dimensions": dimensions_block,
        "passed": not failures,
        "failures": failures,
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    path = arguments.output_dir / "stage1_evidence_validation.json"
    path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Evidence: {path}", flush=True)

    if failures:
        for failure in failures:
            print(f"  - {failure}", flush=True)
        print("Stage 1a elasticity evidence: FAILED", flush=True)
        return 1
    print("Stage 1a elasticity evidence: PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
