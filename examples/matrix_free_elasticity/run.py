from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from mpi4py import MPI
import numpy as np

import layout

if str(layout.SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(layout.SOURCE_ROOT))

from fealpy.backend import backend_manager as bm
from fealpy.distributed import distribute_mesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import Mesh

import contract
import report
from cases import ElasticityCase, create_case
from contract import RunConfig
from distributed import (
    DistributedVectorSpace,
    distribute_vector_space,
    partition_cells,
    partition_strategy_label,
)
from analyzer import build_distributed_analyzer
from postprocess import solution_error, write_solution
from references import relative_difference, serial_references
from schema import RunResult
from solve import PreparedLinearSystem, solver_diagnostics


@dataclass(frozen=True)
class GlobalContext:
    """Objects that exist only on the root rank.

    Every field is ``None`` off the root: the global mesh, the undistributed
    spaces and the serial references are all built before distribution and
    are never replicated.
    """

    mesh: Mesh | None
    scalar_space: LagrangeFESpace | None
    vector_space: TensorFunctionSpace | None
    cell_masks: list | None
    global_cells: int | None
    global_dofs: int | None
    matvec_reference: dict | None
    direct_solution: Any | None


def measure_phase(name: str, callback):
    """Measure one selected-path phase without invoking validation work."""

    start = perf_counter()
    result = callback()
    return result, {"name": name, "seconds": perf_counter() - start}


def build_global_context(
    global_mesh: Mesh | None,
    case: ElasticityCase,
    config: RunConfig,
    *,
    rank: int,
    mpi_size: int,
) -> GlobalContext:
    """Build the root-rank global spaces, partition masks and references."""

    if rank != 0:
        return GlobalContext(
            mesh=None,
            scalar_space=None,
            vector_space=None,
            cell_masks=None,
            global_cells=None,
            global_dofs=None,
            matvec_reference=None,
            direct_solution=None,
        )
    if global_mesh is None:
        raise ValueError("root rank requires the global mesh")

    case.validate_mesh(global_mesh)
    dimension = int(global_mesh.geo_dimension())
    scalar_space = LagrangeFESpace(global_mesh, p=config.degree, ctype="C")
    vector_space = TensorFunctionSpace(scalar_space, shape=(-1, dimension))
    cell_masks = partition_cells(
        global_mesh,
        mpi_size,
        split_coordinate=case.partition_split_coordinate(),
    )
    if mpi_size == 1 and not config.benchmark:
        matvec_reference, direct_solution = serial_references(
            vector_space,
            case,
            config.degree,
        )
    else:
        matvec_reference = None
        direct_solution = None

    return GlobalContext(
        mesh=global_mesh,
        scalar_space=scalar_space,
        vector_space=vector_space,
        cell_masks=cell_masks,
        global_cells=global_mesh.number_of_cells(),
        global_dofs=vector_space.number_of_global_dofs(),
        matvec_reference=matvec_reference,
        direct_solution=direct_solution,
    )


def distribute(
    context: GlobalContext,
    case: ElasticityCase,
    comm,
) -> tuple[DistributedVectorSpace, list | None, Any]:
    """Distribute the mesh and vector space, and gather a partition report.

    The per-DOF reference counts are returned alongside: the partition report
    and the final ``gather_add`` both need them, and they are the same counts.
    """

    distributed_mesh = distribute_mesh(
        context.mesh,
        context.cell_masks,
        comm=comm,
    )
    distributed_space = distribute_vector_space(
        context.scalar_space,
        context.vector_space,
        distributed_mesh,
        context.cell_masks,
        components=case.dimension,
        root=0,
        comm=comm,
    )

    local_size = distributed_space.space.number_of_global_dofs()
    references = distributed_space.dof_comm.refs(local_size)
    partition_report = comm.gather(
        {
            "rank": comm.Get_rank(),
            "local_cells": int(
                distributed_space.space.mesh.number_of_cells()
            ),
            "local_dofs": int(local_size),
            "shared_local_dofs": int(
                np.count_nonzero(np.asarray(references) > 1)
            ),
        },
        root=0,
    )
    return distributed_space, partition_report, references


def run_solver(
    analyzer,
    distributed_space: DistributedVectorSpace,
    problem,
    config: RunConfig,
):
    """Assemble, apply boundary conditions and solve.

    Setup and solve stay behind closures so the benchmark path can time them
    separately while both paths run exactly the same code.
    """

    def prepare_system() -> PreparedLinearSystem:
        operator, load = analyzer.apply_bc(
            analyzer.assemble_stiff_matrix(),
            analyzer.assemble_body_force_vector(),
        )
        return PreparedLinearSystem(
            operator=operator,
            load=load,
            prescribed=analyzer.prescribed_solution,
            boundary_dofs=distributed_space.space.is_boundary_dof(
                threshold=problem.is_dirichlet_boundary(),
                method="interp",
            ),
        )

    def solve(system: PreparedLinearSystem):
        # 必须是裸数组: Function 持有 DistLagrangeFESpace 引用, gather 时无法 pickle
        solution = bm.zeros_like(system.load)
        _, cg_info = analyzer.solve_system(
            system.operator,
            system.load,
            solution,
            x0=system.prescribed,
            maxiter=config.max_iterations,
            rtol=config.rtol,
            atol=config.atol,
        )
        return solution, solver_diagnostics(
            system,
            solution,
            distributed_space.dof_comm,
            cg_info,
        )

    if not config.benchmark:
        system = prepare_system()
        local_solution, solver = solve(system)
        return local_solution, solver, None

    system, setup_timing = measure_phase("operator_setup", prepare_system)
    (local_solution, solver), solve_timing = measure_phase(
        "cg_solve",
        lambda: solve(system),
    )
    return local_solution, solver, {
        "mode": "selected-path-only",
        "phases": [setup_timing, solve_timing],
    }


def finalize(
    context: GlobalContext,
    global_solution,
    case: ElasticityCase,
    config: RunConfig,
    *,
    solver: dict,
    timing: dict | None,
    partition_report: list | None,
    mpi_size: int,
) -> RunResult:
    """Compute errors, write artifacts and assemble the run result."""

    solution = context.vector_space.function(dtype=bm.float64)
    solution[:] = global_solution
    l2_absolute, l2_relative = solution_error(
        context.mesh,
        solution,
        case.problem,
        config.degree,
    )
    if context.direct_solution is None:
        direct_reference = None
    else:
        direct_absolute, direct_relative = relative_difference(
            solution,
            context.direct_solution,
        )
        direct_reference = {
            "absolute_error": direct_absolute,
            "relative_error": direct_relative,
        }

    config.solution_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.solution_path, np.asarray(solution))
    write_solution(
        config.output_path,
        context.mesh,
        context.vector_space,
        solution,
        case,
    )

    return RunResult(
        global_cells=int(context.global_cells),
        global_dofs=int(context.global_dofs),
        operator={
            "level": config.operator_level,
            "storage": config.operator_storage,
        },
        partition={
            "strategy": partition_strategy_label(
                mpi_size,
                case.partition_split_coordinate(),
            ),
            "ranks": partition_report,
        },
        solver=solver,
        timing=timing,
        error={
            "l2_absolute": l2_absolute,
            "l2_relative": l2_relative,
        },
        matvec_reference=context.matvec_reference,
        explicit_solution_reference=direct_reference,
    )


def check_rank_support(config: RunConfig, mpi_size: int) -> None:
    """Reject rank counts stage 1 does not support for this run mode."""

    if mpi_size not in contract.SUPPORTED_RANKS:
        raise ValueError(
            f"stage 1 supports only {contract.SUPPORTED_RANKS} MPI ranks"
        )
    if config.operator_level == "fa" and mpi_size != 1:
        raise ValueError("the FA operator level currently supports one MPI rank")
    if config.benchmark and mpi_size != 1:
        raise ValueError("benchmark mode currently supports one MPI rank")


def execute(
    global_mesh: Mesh | None,
    case: ElasticityCase,
    config: RunConfig,
) -> RunResult | None:
    """Run one case end to end; only the root rank returns a result."""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    check_rank_support(config, mpi_size)

    context = build_global_context(
        global_mesh,
        case,
        config,
        rank=rank,
        mpi_size=mpi_size,
    )
    distributed_space, partition_report, references = distribute(
        context,
        case,
        comm,
    )
    analyzer = build_distributed_analyzer(
        distributed_space.space,
        case,
        config.degree,
        config.operator_level,
        dof_comm=distributed_space.dof_comm,
    )
    local_solution, solver, timing = run_solver(
        analyzer,
        distributed_space,
        case.problem,
        config,
    )

    global_solution = distributed_space.dof_comm.gather_add(
        local_solution / references,
        root=0,
    )
    if rank != 0:
        return None

    return finalize(
        context,
        global_solution,
        case,
        config,
        solver=solver,
        timing=timing,
        partition_report=partition_report,
        mpi_size=mpi_size,
    )


def parse_arguments(mpi_size: int) -> tuple[ElasticityCase, RunConfig]:
    parser = argparse.ArgumentParser(
        description=(
            "SOPTX 2D/3D elasticity FA/EA benchmark with an "
            "overlapping-MPI EA baseline"
        )
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=contract.SUPPORTED_DIMENSIONS,
        default=contract.DEFAULT_DIMENSION,
    )
    parser.add_argument("--p", type=int, default=contract.DEFAULT_DEGREE)
    parser.add_argument("--nx", type=int, default=contract.DEFAULT_RESOLUTION)
    parser.add_argument("--ny", type=int, default=contract.DEFAULT_RESOLUTION)
    parser.add_argument("--nz", type=int)
    parser.add_argument(
        "--maxit",
        type=int,
        default=contract.DEFAULT_MAX_ITERATIONS,
    )
    parser.add_argument("--rtol", type=float, default=contract.DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=contract.DEFAULT_ATOL)
    parser.add_argument(
        "--operator-level",
        choices=contract.OPERATOR_LEVELS,
        default="ea",
        help=(
            "ea: cached element matrices (default); "
            "fa: assembled global CSR matrix, single rank only"
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "measure only the selected serial FA/EA path; "
            "skip spsolve and EA/FA reference construction"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    arguments = parser.parse_args()

    if arguments.p not in contract.SUPPORTED_DEGREES:
        parser.error(
            "stage 1 currently supports only --p "
            + ", ".join(str(value) for value in contract.SUPPORTED_DEGREES)
        )
    if arguments.maxit <= 0:
        parser.error("--maxit must be positive")
    if arguments.rtol < 0.0 or arguments.atol < 0.0:
        parser.error("--rtol and --atol must be non-negative")

    case = create_case(arguments.dim)
    try:
        resolution = case.resolution(
            nx=arguments.nx,
            ny=arguments.ny,
            nz=arguments.nz,
        )
    except ValueError as error:
        parser.error(str(error))
    if min(resolution) <= 0:
        parser.error("all mesh resolution values must be positive")

    artifact_keywords = {
        "dimension": case.dimension,
        "operator_level": arguments.operator_level,
        "degree": arguments.p,
        "resolution": resolution,
        "mpi_size": mpi_size,
    }
    summary_path = arguments.summary or layout.run_artifact_path(
        "json",
        **artifact_keywords,
    )
    output_path = arguments.output or layout.run_artifact_path(
        "vtu",
        **artifact_keywords,
    )
    config = RunConfig(
        dimension=case.dimension,
        degree=arguments.p,
        resolution=resolution,
        operator_level=arguments.operator_level,
        benchmark=arguments.benchmark,
        max_iterations=arguments.maxit,
        rtol=arguments.rtol,
        atol=arguments.atol,
        output_path=output_path,
        summary_path=summary_path,
        solution_path=summary_path.with_suffix(".npy"),
    )
    return case, config


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    case, config = parse_arguments(mpi_size)

    mesh = case.create_mesh(config.resolution) if rank == 0 else None
    result = execute(mesh, case, config)

    if rank == 0:
        gates = report.local_gates(result, config, mpi_size)
        payload = report.build_payload(
            result,
            config,
            case,
            mpi_size,
            gates,
        )
        report.write_json(config.summary_path, payload)
        report.print_summary(payload)
        passed = payload["local_passed"]
    else:
        passed = None

    passed = comm.bcast(passed, root=0)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
