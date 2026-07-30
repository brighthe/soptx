from typing import Optional

from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.backend import backend_manager as bm
from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.utils import process_coef_func
from fealpy.functional import linear_integral
from fealpy.fem.integrator import LinearInt, SrcInt, CellInt, enable_cache

class SourceIntegrator(LinearInt, SrcInt, CellInt):
    def __init__(self, source: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False) -> None:
        super().__init__()
        self.source = source
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The SourceIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)

        return bcs, ws, phi, cm, index

    def assembly(self, space: _FS) -> TensorLike:
        f = self.source
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)
 
        val = process_coef_func(f, bcs=bcs, mesh=mesh, etype='cell', index=index)

        return linear_integral(phi, ws, cm, val, batched=self.batched)
