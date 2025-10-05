from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, SourceLike, Threshold
from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.functional import linear_integral
from fealpy.fem.integrator import LinearInt, SrcInt, FaceInt, enable_cache
from fealpy.utils import process_coef_func


class _FaceSourceIntegrator(LinearInt, SrcInt, FaceInt):
    def __init__(self, 
                source: Optional[SourceLike] = None, 
                q: Optional[int] = None, *,
                threshold: Optional[Threshold] = None,
                batched: bool = False):
        super().__init__()
        self.source = source 
        self.q = q
        self.threshold = threshold
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space) -> TensorLike:
        index = self.make_index(space)
        return space.face_to_dof(index=index)

    @enable_cache
    def fetch(self, space: _FS) -> TensorLike:
        index = self.make_index(space)
        mesh = space.mesh

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarSourceIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomogeneousMesh.")

        facemeasure = mesh.entity_measure('face', index=index)

        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.face_basis(bcs, index=index) # (1, NQ_face, LDOF_face, GD)

        return bcs, ws, phi, facemeasure, index
    
    def assembly(self, space):
        source = self.source
        bcs, ws, phi, fm, index = self.fetch(space) 
        mesh = getattr(space, 'mesh', None)

        ps = mesh.bc_to_point(bcs, index=index)
        val = source(ps) # (NF_bd, NQ_face, GD)

        F = bm.einsum('f, q, qld, fqd -> fl', fm, ws, phi[0], val) # (NF_bd, LDOF_face)

        return F

class InterFaceSourceIntegrator(_FaceSourceIntegrator):
    def make_index(self, space: _FS):
        index = self.threshold
        return index

class BoundaryFaceSourceIntegrator_lfem(_FaceSourceIntegrator): 
    def make_index(self, space: _FS):
        threshold = self.threshold

        if isinstance(threshold, TensorLike):
            index = threshold
        else:
            mesh = space.mesh
            index = mesh.boundary_face_index()
            if callable(threshold):
                bc = mesh.entity_barycenter('face', index=index)
                index = index[threshold(bc)]
        return index
