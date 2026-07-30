from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI

import contract
import layout
from cases import ElasticityCase
from contract import RunConfig


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def git_value(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=layout.REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def environment_record() -> dict[str, Any]:
    status = git_value("status", "--porcelain")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": package_version("scipy"),
        "sympy": package_version("sympy"),
        "fealpy": package_version("fealpy"),
        "soptx": package_version("soptx"),
        "mpi4py": package_version("mpi4py"),
        "mpi_library": MPI.Get_library_version().strip(),
        "git_revision": git_value("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def local_gates(
    result: dict,
    config: RunConfig,
    mpi_size: int,
) -> dict[str, bool]:
    """Evaluate the single-run gates recorded in the summary.

    The serial EA/FA references only exist for a non-benchmark single-rank
    run; the gates that consume them are reported as passing elsewhere so the
    summary keeps one stable field set across every run mode.
    """

    solver = result["solver"]
    matvec = result["matvec_reference"]
    direct = result["explicit_solution_reference"]
    referenced = not (config.benchmark or mpi_size != 1)
    return {
        "converged": bool(solver["converged"]),
        "true_residual": (
            solver["true_absolute_residual"]
            <= config.residual_limit(solver["rhs_norm"])
        ),
        "boundary_dofs": (
            solver["boundary_absolute_error"]
            <= contract.BOUNDARY_ABSOLUTE_TOL
        ),
        "raw_matvec": not referenced or (
            matvec is not None
            and matvec["raw_relative_error"] <= contract.MATVEC_RELATIVE_TOL
        ),
        "dirichlet_matvec": not referenced or (
            matvec is not None
            and matvec["dirichlet_relative_error"]
            <= contract.MATVEC_RELATIVE_TOL
        ),
        "operator_symmetry": not referenced or (
            matvec is not None
            and matvec["symmetry_relative_error"]
            <= contract.SYMMETRY_RELATIVE_TOL
            and matvec["random_vector_energy"] > 0.0
        ),
        "explicit_solution": not referenced or (
            direct is not None
            and direct["relative_error"]
            <= contract.EXPLICIT_SOLUTION_RELATIVE_TOL
        ),
    }


def build_payload(
    result: dict,
    config: RunConfig,
    case: ElasticityCase,
    mpi_size: int,
    gates: dict[str, bool],
) -> dict:
    """Assemble the schema-versioned single-run summary."""

    return {
        "schema_version": contract.SCHEMA_VERSION,
        "stage": contract.STAGE,
        "command": [sys.executable, *sys.argv],
        "environment": environment_record(),
        "parameters": {
            "dimension": case.dimension,
            "case": case.name,
            "domain": list(case.domain),
            "resolution": list(config.resolution),
            "material": case.material.as_dict(),
            "lagrange_p": config.degree,
            "maxit": config.max_iterations,
            "rtol": config.rtol,
            "atol": config.atol,
            "preconditioner": None,
            "operator_level": config.operator_level,
            "operator_storage": config.operator_storage,
            "benchmark": config.benchmark,
            "reference_random_seed": contract.REFERENCE_RANDOM_SEED,
            "distributed_representation": (
                contract.DISTRIBUTED_REPRESENTATION
            ),
        },
        "mpi_size": mpi_size,
        "verification": (
            "skipped-for-benchmark" if config.benchmark else "full"
        ),
        **result,
        "artifacts": {
            "vtu": str(config.output_path.resolve()),
            "solution_npy": str(config.solution_path.resolve()),
            "summary_json": str(config.summary_path.resolve()),
        },
        "local_gates": gates,
        "local_passed": all(gates.values()),
    }


def print_summary(payload: dict) -> None:
    print(
        json.dumps(
            {
                "converged": payload["solver"]["converged"],
                "iterations": payload["solver"]["iterations"],
                "true_relative_residual": payload["solver"][
                    "true_relative_residual"
                ],
                "boundary_absolute_error": payload["solver"][
                    "boundary_absolute_error"
                ],
                "l2_relative": payload["error"]["l2_relative"],
                "timing": payload["timing"],
                "summary": str(
                    payload["artifacts"]["summary_json"]
                ),
                "local_passed": payload["local_passed"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
