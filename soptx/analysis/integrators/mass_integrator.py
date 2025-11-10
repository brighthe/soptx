
from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike

from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.utils import process_coef_func
from fealpy.functional import bilinear_integral
from fealpy.fem.integrator import (
                                LinearInt, OpInt, CellInt,
                                enable_cache
                            )

class MassIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, 
                index: Index=_S) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.index = index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The MassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)
        
        return bcs, ws, phi, cm, index

    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)

        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)

        return bilinear_integral(phi, phi, ws, cm, val)


