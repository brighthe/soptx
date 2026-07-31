from __future__ import annotations

from fealpy.functionspace import TensorFunctionSpace

from soptx.fem.solvers import LagrangeFEMAnalyzer

import contract
from cg import solve_cg
from distributed import OverlapOperator


class DistributedElasticityAnalyzer(LagrangeFEMAnalyzer):
    """LagrangeFEMAnalyzer with the overlapping-MPI pieces filled in.

    The analyzer owns the discretisation; everything parallel stays here. Three
    extension points are overridden:

    * ``reduce_load``   -- sum each rank's contribution on shared DOF copies
    * ``wrap_operator`` -- make one local matvec act as the global operator
    * ``solve_system``  -- CG whose inner products skip duplicated DOF copies

    Both are applied on the EA path only. FA assembles a global matrix and is
    restricted to a single rank by ``contract.OPERATOR_LEVELS`` usage in
    ``run.py``, where every overlap reduction is the identity anyway.
    """

    def __init__(self, *args, dof_comm=None, **kwargs) -> None:
        if dof_comm is None:
            raise ValueError("a distributed analyzer requires a dof_comm")
        super().__init__(*args, dof_comm=dof_comm, **kwargs)

    def reduce_load(self, F):
        return self.dof_comm.sync_add(F)

    def wrap_operator(self, form):
        return OverlapOperator(form, self.dof_comm)

    def solve_system(self, K, F, out, **kwargs):
        solution, info = solve_cg(
            K,
            F,
            dof_comm=self.dof_comm,
            initial=kwargs.get("x0"),
            max_iterations=kwargs.get("max_iterations", contract.DEFAULT_MAX_ITERATIONS),
            rtol=kwargs.get("rtol", contract.DEFAULT_RTOL),
            atol=kwargs.get("atol", contract.DEFAULT_ATOL),
            residual_refresh=contract.RESIDUAL_REFRESH,
        )
        out[:] = solution
        return out, info


def build_analyzer(
    space: TensorFunctionSpace,
    case,
    degree: int,
    operator_level: str,
    dof_comm=None,
) -> LagrangeFEMAnalyzer:
    """Compose problem, material, mesh and space into one analyzer.

    Without a ``dof_comm`` this yields the plain serial analyzer, which is what
    the single-rank EA/FA references in ``references.py`` need: they are built
    on the global space before the mesh is distributed.
    """

    from fealpy.backend import backend_manager as bm

    mesh = space.mesh
    arguments = dict(
        disp_mesh=mesh,
        pde=case.problem,
        material=case.material.create(device=bm.get_device(mesh)),
        space_degree=degree,
        integration_order=degree + 3,
        operator_level=operator_level,
        tensor_space=space,
        topopt_algorithm=None,
    )
    if dof_comm is None:
        return LagrangeFEMAnalyzer(solve_method="scipy", **arguments)

    return DistributedElasticityAnalyzer(dof_comm=dof_comm, **arguments)
