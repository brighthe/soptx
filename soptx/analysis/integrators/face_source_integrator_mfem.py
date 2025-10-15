from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, SourceLike, Threshold
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.decorator import variantmethod
from fealpy.fem.integrator import LinearInt, SrcInt, FaceInt, enable_cache


class _FaceSourceIntegrator(LinearInt, SrcInt, FaceInt):
    def __init__(self, 
                source: Optional[SourceLike] = None, 
                q: Optional[int] = None, *,
                threshold: Optional[Threshold] = None,
                method: Optional[str]=None
            ) -> None:
        super().__init__()
        self.source = source 
        self.q = q
        self.threshold = threshold

        self.assembly.set(method)


    @enable_cache
    def to_global_dof(self, space) -> TensorLike:
        # 边界面的全局索引
        index = self.make_index(space)

        mesh = space.mesh
        face2cell = mesh.face_to_cell(index)
        # 每个边界面所邻接的单元的全局索引
        cell_index = face2cell[:, 0] # (NF_bd, ) 
        # 单元到全局自由度的映射
        result = space.cell_to_dof(index=cell_index) # (NF_bd, LDOF)

        return result

    @enable_cache
    def fetch_dirichlet(self, space: _FS) -> TensorLike:
        index = self.make_index(space)
        mesh = space.mesh

        facemeasure = mesh.entity_measure('face', index=index)
        n = mesh.face_unit_normal(index=index)

        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = space.cell_basis_on_face(bcs, index) # (NF_bd, NQ, LDOF, 3)

        return bcs, ws, phi, facemeasure, index, n

    @variantmethod('dirichlet')
    def assembly(self, space: _FS) -> TensorLike:
        source = self.source
        bcs, ws, phi, fm, index, n = self.fetch_dirichlet(space) 
        mesh = getattr(space, 'mesh', None)

        NF_bd, NQ, LDOF, _ = phi.shape
        GD = mesh.geo_dimension()
        tau = bm.zeros((NF_bd, NQ, LDOF, GD, GD), dtype=phi.dtype)
        tau[..., 0, 0] = phi[..., 0] # σ_xx
        tau[..., 0, 1] = phi[..., 1] # σ_xy
        tau[..., 1, 0] = phi[..., 1] # σ_yx
        tau[..., 1, 1] = phi[..., 2] # σ_yy
        tau_n = bm.einsum('fqlij, fj -> fqli', tau, n) # (NF_bd, NQ, LDOF, GD)

        ps = mesh.bc_to_point(bcs, index=index)
        val = source(ps) # (NF_bd, NQ, GD)

        F = bm.einsum('f, q, fqli, fqd -> fl', fm, ws, tau_n, val) # (NF_bd, LDOF)

        return F
    
    @enable_cache
    def fetch_neumann(self, space: _FS) -> TensorLike:
        index = self.make_index(space)
        mesh = space.mesh

        facemeasure = mesh.entity_measure('face', index=index)

        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = space.cell_basis_on_face(bcs, index) # (NF_bd, NQ, LDOF, GD)

        return bcs, ws, phi, facemeasure, index

    @assembly.register('neumann')
    def assembly(self, space: _FS) -> TensorLike:
        source = self.source
        bcs, ws, phi, fm, index = self.fetch_neumann(space)
        mesh = getattr(space, 'mesh', None)

        ps = mesh.bc_to_point(bcs, index=index)
        val = source(ps) # (NF_bd, NQ, GD)

        F = bm.einsum('f, q, fqld, fqd -> fl', fm, ws, phi, val) # (NF_bd, LDOF)

        return F

class InterFaceSourceIntegrator(_FaceSourceIntegrator):
    def make_index(self, space: _FS):
        index = self.threshold
        return index

class BoundaryFaceSourceIntegrator_mfem(_FaceSourceIntegrator): 
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
