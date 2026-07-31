from __future__ import annotations

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import TensorFunctionSpace

from soptx.fem.solvers import LagrangeFEMAnalyzer

import contract
from cases import ElasticityCase
from cg import solve_cg
from distributed import EntityMPI, OverlapOperator


class DistributedElasticityAnalyzer(LagrangeFEMAnalyzer):
    """LagrangeFEMAnalyzer with the overlapping-MPI pieces filled in.

    The analyzer owns the discretisation; everything parallel stays here.
    Three extension points are overridden:

    * ``reduce_load``   -- sum each rank's contribution on shared DOF copies
    * ``wrap_operator`` -- make one local matvec act as the global operator
    * ``solve_system``  -- CG whose inner products skip duplicated DOF copies

    The first two are reachable only from the EA path: ``apply_bc('fa')``
    rewrites an assembled global matrix and has nowhere to insert an overlap
    reduction, which is why the base class rejects multi-rank FA outright.
    ``solve_system`` is used whatever the operator level, including the
    single-rank FA reference, where every overlap reduction is the identity.
    """

    def __init__(self, *args, dof_comm: EntityMPI, **kwargs) -> None:
        super().__init__(*args, dof_comm=dof_comm, **kwargs)

    def reduce_load(self, F):
        return self.dof_comm.sync_add(F)

    def wrap_operator(self, form):
        return OverlapOperator(form, self.dof_comm)

    def solve_system(
        self,
        K,
        F,
        out,
        *,
        x0=None,
        maxiter: int = contract.DEFAULT_MAX_ITERATIONS,
        rtol: float = contract.DEFAULT_RTOL,
        atol: float = contract.DEFAULT_ATOL,
        **kwargs,
    ):
        """Solve with the overlap-weighted CG.

        The keyword vocabulary matches the base class (``x0``, ``maxiter``,
        ``rtol``, ``atol``) so that callers reaching this through
        ``solve_state`` are not silently ignored.
        """

        solution, info = solve_cg(
            K,
            F,
            dof_comm=self.dof_comm,
            initial=x0,
            max_iterations=maxiter,
            rtol=rtol,
            atol=atol,
            residual_refresh=contract.RESIDUAL_REFRESH,
        )
        out[:] = solution
        return out, info


def _analyzer_arguments(
    space: TensorFunctionSpace,
    case: ElasticityCase,
    degree: int,
    operator_level: str,
) -> dict:
    """Compose problem, material, mesh and space into constructor arguments."""

    mesh = space.mesh
    return dict(
        disp_mesh=mesh,
        pde=case.problem,
        material=case.material.create(device=bm.get_device(mesh)),
        space_degree=degree,
        integration_order=degree + 3,
        operator_level=operator_level,
        tensor_space=space,
        topopt_algorithm=None,
    )


def build_serial_analyzer(
    space: TensorFunctionSpace,
    case: ElasticityCase,
    degree: int,
    operator_level: str,
) -> LagrangeFEMAnalyzer:
    """Build the plain analyzer, with no overlap communication at all.

    Used by the single-rank EA/FA references, which are built on the global
    space before the mesh is distributed.
    """

    return LagrangeFEMAnalyzer(
        solve_method="scipy",
        **_analyzer_arguments(space, case, degree, operator_level),
    )


def build_distributed_analyzer(
    space: TensorFunctionSpace,
    case: ElasticityCase,
    degree: int,
    operator_level: str,
    dof_comm: EntityMPI,
) -> DistributedElasticityAnalyzer:
    """Build the analyzer used for the actual run, on one rank or two."""

    return DistributedElasticityAnalyzer(
        dof_comm=dof_comm,
        **_analyzer_arguments(space, case, degree, operator_level),
    )
