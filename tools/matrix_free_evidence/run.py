from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from mpi4py import MPI
import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

# 导入本包会把示例目录与 src/ 放上 sys.path
from tools.matrix_free_evidence import layout, report, schema

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import (
    Mesh,
    TetrahedronMesh,
    TriangleMesh,
    write_mesh_to_vtu,
)
from soptx.fem.distributed import (
    SUPPORTED_RANKS,
    DistributedVectorSpace,
    distribute_mesh,
    distribute_vector_space,
    partition_cells,
    partition_strategy_label,
)
from soptx.fem.solvers import (
    PreparedLinearSystem,
    build_distributed_analyzer,
    solver_diagnostics,
)
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
from tools.matrix_free_evidence.contract import RunConfig
from tools.matrix_free_evidence.schema import RunResult


# 维数决定这三样, 其余一律向对象本身要: 区域与弹性常数由制造解自带, 网格实体名由
# 网格自带. 平面降维假设不属于制造解, 所以只能在这里定
PROBLEM_FACTORIES = {
    2: SinusoidalPlaneStrainElasticity2D,
    3: DivergenceFreePolynomialElasticity3D,
}
MESH_FACTORIES = {2: TriangleMesh, 3: TetrahedronMesh}
MATERIAL_HYPOTHESES = {2: "plane_strain", 3: "3D"}


def partition_split_coordinate(problem) -> float:
    """沿 x 轴对半切单元的坐标, 由制造解的区域决定"""

    return 0.5 * (problem.domain[0] + problem.domain[1])


def write_solution(
    filename: Path,
    mesh: Mesh,
    space: TensorFunctionSpace,
    solution,
    problem,
) -> None:
    """把重心处的位移与误差写成 VTU

    只有本脚本的证据产物需要它, 所以留在这里而不是上浮到 ``soptx``。
    """

    weight = 1.0 / (problem.dimension + 1)
    barycenter = np.array([(weight,) * (problem.dimension + 1)])
    numerical = np.asarray(
        space.value(solution, barycenter)
    )[:, 0, :]
    exact = np.asarray(
        problem.disp_solution(mesh.Entity("cell").barycenter())
    )
    mesh.Entity("cell").set_attribute("displacement", numerical)
    mesh.Entity("cell").set_attribute(
        "displacement_error",
        numerical - exact,
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    write_mesh_to_vtu(
        str(filename),
        mesh,
        entity_names=[mesh.Entity("cell").schema.name],
    )


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
    problem,
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

    scalar_space = LagrangeFESpace(global_mesh, p=config.degree, ctype="C")
    vector_space = TensorFunctionSpace(
        scalar_space,
        shape=(-1, config.dimension),
    )
    cell_masks = partition_cells(
        global_mesh,
        mpi_size,
        split_coordinate=partition_split_coordinate(problem),
    )
    if mpi_size == 1 and not config.benchmark:
        matvec_reference, direct_solution = serial_references(
            vector_space,
            problem,
            IsotropicLinearElasticMaterial(
                hypothesis=MATERIAL_HYPOTHESES[config.dimension],
                lame_lambda=problem.lam,
                shear_modulus=problem.mu,
                device=bm.get_device(global_mesh),
            ),
            config.degree,
            seed=contract.REFERENCE_RANDOM_SEED,
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
    dimension: int,
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
        components=dimension,
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
    problem,
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
        problem,
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
        problem,
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
                partition_split_coordinate(problem),
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

    if mpi_size not in SUPPORTED_RANKS:
        raise ValueError(
            f"stage 1 supports only {SUPPORTED_RANKS} MPI ranks"
        )
    if config.operator_level == "fa" and mpi_size != 1:
        raise ValueError("the FA operator level currently supports one MPI rank")
    if config.benchmark and mpi_size != 1:
        raise ValueError("benchmark mode currently supports one MPI rank")


def execute(
    global_mesh: Mesh | None,
    problem,
    config: RunConfig,
) -> RunResult | None:
    """Run one case end to end; only the root rank returns a result."""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    check_rank_support(config, mpi_size)

    context = build_global_context(
        global_mesh,
        problem,
        config,
        rank=rank,
        mpi_size=mpi_size,
    )
    distributed_space, partition_report, references = distribute(
        context,
        config.dimension,
        comm,
    )
    analyzer = build_distributed_analyzer(
        distributed_space.space,
        problem,
        IsotropicLinearElasticMaterial(
            hypothesis=MATERIAL_HYPOTHESES[config.dimension],
            lame_lambda=problem.lam,
            shear_modulus=problem.mu,
            device=bm.get_device(distributed_space.space.mesh),
        ),
        config.degree,
        config.operator_level,
        dof_comm=distributed_space.dof_comm,
    )
    local_solution, solver, timing = run_solver(
        analyzer,
        distributed_space,
        problem,
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
        problem,
        config,
        solver=solver,
        timing=timing,
        partition_report=partition_report,
        mpi_size=mpi_size,
    )


def parse_arguments(mpi_size: int):
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

    if arguments.dim == 2:
        if arguments.nz is not None:
            parser.error("--nz is only valid when --dim 3")
        resolution = (arguments.nx, arguments.ny)
    else:
        nz = arguments.nz
        if nz is None:
            nz = contract.DEFAULT_RESOLUTION
        resolution = (arguments.nx, arguments.ny, nz)
    if min(resolution) <= 0:
        parser.error("all mesh resolution values must be positive")

    problem = PROBLEM_FACTORIES[arguments.dim]()
    artifact_keywords = {
        "dimension": arguments.dim,
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
        dimension=arguments.dim,
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
    return problem, config


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    problem, config = parse_arguments(mpi_size)

    if rank == 0:
        mesh = MESH_FACTORIES[config.dimension].from_box(
            list(problem.domain),
            **dict(zip(("nx", "ny", "nz"), config.resolution)),
        )
    else:
        mesh = None
    result = execute(mesh, problem, config)

    if rank == 0:
        gates = report.local_gates(result, config, mpi_size)
        payload = report.build_payload(
            result,
            config,
            problem,
            MATERIAL_HYPOTHESES[config.dimension],
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
