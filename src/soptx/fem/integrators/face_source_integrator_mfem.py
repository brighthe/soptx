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

        phi = space.cell_basis_on_face(bcs, index) # (NF_bd, NQ, LDOF_face, NS)

        return bcs, ws, phi, facemeasure, index, n

    @variantmethod('dirichlet')
    def assembly(self, space: _FS) -> TensorLike:
        source = self.source
        bcs, ws, phi, fm, index, n = self.fetch_dirichlet(space) 
        mesh = getattr(space, 'mesh', None)

        NF_bd, NQ, LDOF, NS = phi.shape
        GD = mesh.geo_dimension()
        assert NS == GD*(GD+1)//2, f"phi last dim {NS} != GD(GD+1)/2 for GD={GD}"

        # 构造对称应力张量 (NF_bd, NQ, LDOF, GD, GD)
        tau = bm.zeros((NF_bd, NQ, LDOF, GD, GD), dtype=phi.dtype)

        if GD == 2:
            tau[..., 0, 0] = phi[..., 0] # σ_xx
            tau[..., 0, 1] = phi[..., 1] # σ_xy
            tau[..., 1, 0] = phi[..., 1] # σ_yx
            tau[..., 1, 1] = phi[..., 2] # σ_yy
        elif GD == 3:
            tau[..., 0, 0] = phi[..., 0] # σ_xx
            tau[..., 0, 1] = phi[..., 1] # σ_xy
            tau[..., 0, 2] = phi[..., 2] # σ_xz
            tau[..., 1, 0] = phi[..., 1] # σ_yx
            tau[..., 1, 1] = phi[..., 3] # σ_yy
            tau[..., 1, 2] = phi[..., 4] # σ_yz
            tau[..., 2, 0] = phi[..., 2] # σ_zx
            tau[..., 2, 1] = phi[..., 4] # σ_zy
            tau[..., 2, 2] = phi[..., 5] # σ_zz

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

        phi = space.cell_basis_on_face(bcs, index) # (NF_bd, NQ, LDOF, NS)

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
            if isinstance(threshold, (tuple, list)):
                # threshold 是元组或列表, 包含多个边界判断函数
                bc = mesh.entity_barycenter('face', index=index)
                flags = []
                for thresh_func in threshold:
                    if callable(thresh_func):
                        flags.append(thresh_func(bc))
                if flags:
                    combined_flag = flags[0]
                    for flag in flags[1:]:
                        combined_flag = combined_flag | flag
                    index = index[combined_flag]
            elif callable(threshold):
                # threshold 是单个 callable 函数
                bc = mesh.entity_barycenter('face', index=index)
                index = index[threshold(bc)]

        return index
