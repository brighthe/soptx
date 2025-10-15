from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from fealpy.functionspace import FunctionSpace
from fealpy.fem.integrator import (LinearInt, OpInt, CellInt, enable_cache)

class HuZhangMixIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, q = None) -> None:
        super().__init__()
        self.q = q

    @enable_cache
    def to_global_dof(self, space: FunctionSpace) -> TensorLike:
        c2d0  = space[0].cell_to_dof() 
        c2d1  = space[1].cell_to_dof() 

        return (c2d0, c2d1)

    @enable_cache
    def fetch(self, space: FunctionSpace):
        space0 = space[0]
        space1 = space[1]

        p = space0.p
        q = self.q if self.q else p+3
        mesh = space1.mesh
        qf = mesh.quadrature_formula(q, 'cell')
        cm = mesh.entity_measure('cell')

        bcs, ws = qf.get_quadrature_points_and_weights()

        psi = space0.basis(bcs)
        div_phi = space1.div_basis(bcs)

        return cm, ws, div_phi, psi
    
    @enable_cache
    def assembly(self, space: FunctionSpace) -> TensorLike:
        assert space[0].mesh == space[1].mesh, "The mesh should be same for two space "

        cm, ws, div_phi, psi = self.fetch(space)
        res = bm.einsum('q, c, cqld, cqmd -> clm', ws, cm, div_phi, psi)

        return res



