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
                use_cell_basis: bool = False,
                batched: bool = False):
        super().__init__()
        self.source = source 
        self.q = q
        self.threshold = threshold
        self.use_cell_basis = use_cell_basis
        self.batched = batched


    @enable_cache
    def to_global_dof(self, space) -> TensorLike:
        # 边界面的全局索引
        index = self.make_index(space)

        if self.use_cell_basis:
            mesh = space.mesh
            face2cell = mesh.face_to_cell(index)
            # 每个边界面所邻接的单元的全局索引
            cell_index = face2cell[:, 0] # (NFb, ) 
            # 单元到全局自由度的映射
            result = space.cell_to_dof(index=cell_index) # (NFb, LDOF)

            return result
        
        else:
            # TODO 删除
            return space.face_to_dof(index=index)

    @enable_cache
    def fetch(self, space: _FS) -> TensorLike:
        index = self.make_index(space)
        mesh = space.mesh

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarSourceIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        facemeasure = mesh.entity_measure('face', index=index)
        n = mesh.face_unit_normal(index=index)

        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if self.use_cell_basis:
            phi = space.cell_basis_on_face(bcs, index=index) # (NF_bd, NQ, LDOF, 3)
        else:
            phi = space.face_basis(bcs, index=index) # (1, NQ_face, LDOF, GD)

        return bcs, ws, phi, facemeasure, index, n

    # def assembly(self, space):
    #     source = self.source
    #     bcs, ws, phi, fm, index, n = self.fetch(space) 
    #     mesh = getattr(space, 'mesh', None)

    #     NFb, NQ, LDOF, _ = phi.shape
    #     GD = mesh.geo_dimension()
    #     tau = bm.zeros((NFb, NQ, LDOF, GD, GD), dtype=phi.dtype)
    #     tau[..., 0, 0] = phi[..., 0] # σ_xx
    #     tau[..., 0, 1] = phi[..., 1] # σ_xy
    #     tau[..., 1, 0] = phi[..., 1] # σ_yx
    #     tau[..., 1, 1] = phi[..., 2] # σ_yy
    #     tau_n = bm.einsum('fqlij, fj -> fqli', tau, n) # (NFb, NQ, LDOF, GD)

    #     ps = mesh.bc_to_point(bcs, index=index)
    #     val = source(ps) # (NFb, NQ, GD)

    #     F = linear_integral(tau_n, ws, fm, val, self.batched) # (NFb, LDOF)

    #     return F
    
    def assembly(self, space):
        source = self.source
        bcs, ws, phi, fm, index, n = self.fetch(space) 
        mesh = getattr(space, 'mesh', None)

        # NFb, NQ, LDOF, _ = phi.shape
        # GD = mesh.geo_dimension()
        # tau = bm.zeros((NFb, NQ, LDOF, GD, GD), dtype=phi.dtype)
        # tau[..., 0, 0] = phi[..., 0] # σ_xx
        # tau[..., 0, 1] = phi[..., 1] # σ_xy
        # tau[..., 1, 0] = phi[..., 1] # σ_yx
        # tau[..., 1, 1] = phi[..., 2] # σ_yy
        # tau_n = bm.einsum('fqlij, fj -> fqli', tau, n) # (NFb, NQ, LDOF, GD)

        ps = mesh.bc_to_point(bcs, index=index)
        val = source(ps) # (NFb, NQ, GD)

        F = linear_integral(phi, ws, fm, val, self.batched) # (NFb, LDOF)

        return F
    

class InterFaceSourceIntegrator(_FaceSourceIntegrator):
    def make_index(self, space: _FS):
        index = self.threshold
        return index

class BoundaryFaceSourceIntegrator(_FaceSourceIntegrator): 
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
