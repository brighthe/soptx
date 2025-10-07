from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S
from fealpy.mesh import HomogeneousMesh, SimplexMesh, StructuredMesh
from fealpy.functionspace.space import FunctionSpace, Function
from fealpy.functionspace.tensor_space import TensorFunctionSpace
from fealpy.decorator.variantmethod import variantmethod
from fealpy.fem.integrator import (LinearInt, OpInt, CellInt, enable_cache)
from fealpy.fem.utils import LinearSymbolicIntegration

from ...interpolation.linear_elastic_material import LinearElasticMaterial

class LinearElasticIntegrator(LinearInt, OpInt, CellInt):
    """The linear elastic integrator for function spaces based on homogeneous meshes."""
    def __init__(self, 
                material: LinearElasticMaterial,
                coef: Optional[TensorLike]=None,
                q: Optional[int]=None, 
                *,
                index: Index=_S,
                method: Optional[str]=None
            ) -> None:
        super().__init__()

        self._material = material
        self._coef = coef
        self._q = q
        self._index = index
        
        self.assembly.set(method)


    @enable_cache
    def to_global_dof(self, space: FunctionSpace) -> TensorLike:
        return space.cell_to_dof()[self._index]
    

    ########################################################################################
    # 变体方法
    ########################################################################################

    @enable_cache
    def fetch_assembly(self, space: TensorFunctionSpace):
        index = self._index
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
    
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The LinearElasticIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
    
        cm = mesh.entity_measure('cell', index=index)
        q = scalar_space.p+3 if self._q is None else self._q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        
        gphi = scalar_space.grad_basis(bcs, index=index, variable='x')

        if isinstance(mesh, SimplexMesh):
            J = None
            detJ = None
        else:
            J = mesh.jacobi_matrix(bcs)
            detJ = bm.abs(bm.linalg.det(J))

        return cm, bcs, ws, gphi, detJ

    @variantmethod('standard')
    def assembly(self, space: TensorFunctionSpace) -> TensorLike:
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
        cm, bcs, ws, gphi, detJ = self.fetch_assembly(space)

        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        NQ = len(ws)
        D0 = self._material.elastic_matrix()  # 2D: (1, 1, 3, 3); 3D: (1, 1, 6, 6)
        
        # 不考虑相对密度: None; 相对单元密度: (NC, ); 相对节点密度: (NC, NQ)      
        coef = self._coef

        if coef is None:
            D = D0[0, 0] # 2D: (3, 3); 3D: (6, 6)
        elif coef.shape == (NC, ):
            D = bm.einsum('c, kl -> ckl', coef, D0[0, 0])  # 2D: (NC, 3, 3); 3D: (NC, 6, 6)
        elif coef.shape == (NC, NQ):
            D = bm.einsum('cq, cqkl -> cqkl', coef, D0) # 2D: (NC, NQ, 3, 3); 3D: (NC, NQ, 6, 6)
            
        if isinstance(mesh, SimplexMesh):
            A_xx = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 0], gphi[..., 0], cm)
            A_yy = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 1], gphi[..., 1], cm)
            A_xy = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 0], gphi[..., 1], cm)
            A_yx = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 1], gphi[..., 0], cm)
        else:
            A_xx = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 0], gphi[..., 0], detJ)
            A_yy = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 1], gphi[..., 1], detJ)
            A_xy = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 0], gphi[..., 1], detJ)
            A_yx = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 1], gphi[..., 0], detJ)

        GD = mesh.geo_dimension()
        if GD == 3:
            if isinstance(mesh, SimplexMesh):
                A_xz = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 0], gphi[..., 2], cm)
                A_zx = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 2], gphi[..., 0], cm)
                A_yz = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 1], gphi[..., 2], cm)
                A_zy = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 2], gphi[..., 1], cm)
                A_zz = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 2], gphi[..., 2], cm)
            else:
                A_xz = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 0], gphi[..., 2], detJ)
                A_zx = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 2], gphi[..., 0], detJ)
                A_yz = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 1], gphi[..., 2], detJ)
                A_zy = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 2], gphi[..., 1], detJ)
                A_zz = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 2], gphi[..., 2], detJ)

        ldof = scalar_space.number_of_local_dofs()
        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64, device=mesh.device)

        # 区域内的相对密度恒定都为 1, D 为全局常数矩阵
        if coef is None:
            if GD == 2:
                D00 = D[0, 0] # 2D: E/(1-ν²) 或 2μ+λ
                D01 = D[0, 1] # 2D: νE/(1-ν²) 或 λ
                D22 = D[2, 2] # 2D: E/2(1+ν) 或 μ
                KK_11 = D00 * bm.einsum('cqij -> cij', A_xx) + D22 * bm.einsum('cqij -> cij', A_yy)
                KK_22 = D00 * bm.einsum('cqij -> cij', A_yy) + D22 * bm.einsum('cqij -> cij', A_xx)
                KK_12 = D01 * bm.einsum('cqij -> cij', A_xy) + D22 * bm.einsum('cqij -> cij', A_yx)
                KK_21 = D01 * bm.einsum('cqij -> cij', A_yx) + D22 * bm.einsum('cqij -> cij', A_xy)
            else: 
                D00 = D[0, 0]  # 2μ + λ
                D01 = D[0, 1]  # λ
                D55 = D[5, 5]  # μ
                KK_11 = D00 * bm.einsum('cqij -> cij', A_xx) + D55 * bm.einsum('cqij -> cij', A_yy + A_zz)
                KK_22 = D00 * bm.einsum('cqij -> cij', A_yy) + D55 * bm.einsum('cqij -> cij', A_xx + A_zz)
                KK_33 = D00 * bm.einsum('cqij -> cij', A_zz) + D55 * bm.einsum('cqij -> cij', A_xx + A_yy)
                KK_12 = D01 * bm.einsum('cqij -> cij', A_xy) + D55 * bm.einsum('cqij -> cij', A_yx)
                KK_13 = D01 * bm.einsum('cqij -> cij', A_xz) + D55 * bm.einsum('cqij -> cij', A_zx)
                KK_21 = D01 * bm.einsum('cqij -> cij', A_yx) + D55 * bm.einsum('cqij -> cij', A_xy)
                KK_23 = D01 * bm.einsum('cqij -> cij', A_yz) + D55 * bm.einsum('cqij -> cij', A_zy)
                KK_31 = D01 * bm.einsum('cqij -> cij', A_zx) + D55 * bm.einsum('cqij -> cij', A_xz)
                KK_32 = D01 * bm.einsum('cqij -> cij', A_zy) + D55 * bm.einsum('cqij -> cij', A_yz)
        # 单元密度情况, D 为单元均匀矩阵
        elif coef.shape == (NC, ):
            if GD == 2:
                D00 = D[:, 0, 0] # 2D: E/(1-ν²) 或 2μ+λ
                D01 = D[:, 0, 1] # 2D: νE/(1-ν²) 或 λ
                D22 = D[:, 2, 2] # 2D: E/2(1+ν) 或 μ
                KK_11 = bm.einsum('c, cqij -> cij', D00, A_xx) + bm.einsum('c, cqij -> cij', D22, A_yy)
                KK_22 = bm.einsum('c, cqij -> cij', D00, A_yy) + bm.einsum('c, cqij -> cij', D22, A_xx)
                KK_12 = bm.einsum('c, cqij -> cij', D01, A_xy) + bm.einsum('c, cqij -> cij', D22, A_yx)
                KK_21 = bm.einsum('c, cqij -> cij', D01, A_yx) + bm.einsum('c, cqij -> cij', D22, A_xy)
            else:
                D00 = D[:, 0, 0] # 2μ + λ
                D01 = D[:, 0, 1] # λ
                D55 = D[:, 5, 5] # μ
                KK_11 = bm.einsum('c, cqij -> cij', D00, A_xx) + bm.einsum('c, cqij -> cij', D55, A_yy + A_zz)
                KK_22 = bm.einsum('c, cqij -> cij', D00, A_yy) + bm.einsum('c, cqij -> cij', D55, A_xx + A_zz)
                KK_33 = bm.einsum('c, cqij -> cij', D00, A_zz) + bm.einsum('c, cqij -> cij', D55, A_xx + A_yy)
                KK_12 = bm.einsum('c, cqij -> cij', D01, A_xy) + bm.einsum('c, cqij -> cij', D55, A_yx)
                KK_13 = bm.einsum('c, cqij -> cij', D01, A_xz) + bm.einsum('c, cqij -> cij', D55, A_zx)
                KK_21 = bm.einsum('c, cqij -> cij', D01, A_yx) + bm.einsum('c, cqij -> cij', D55, A_xy)
                KK_23 = bm.einsum('c, cqij -> cij', D01, A_yz) + bm.einsum('c, cqij -> cij', D55, A_zy)
                KK_31 = bm.einsum('c, cqij -> cij', D01, A_zx) + bm.einsum('c, cqij -> cij', D55, A_xz)
                KK_32 = bm.einsum('c, cqij -> cij', D01, A_zy) + bm.einsum('c, cqij -> cij', D55, A_yz)
        # 区域内的相对密度在单元内变化, D 为节点变化矩阵
        elif coef.shape == (NC, NQ):
            if GD == 2:
                D00 = D[..., 0, 0] # 2D: E/(1-ν²) 或 2μ+λ
                D01 = D[..., 0, 1] # 2D: νE/(1-ν²) 或 λ
                D22 = D[..., 2, 2] # 2D: E/2(1+ν) 或 μ
                KK_11 = bm.einsum('cq, cqij -> cij', D00, A_xx) + bm.einsum('cq, cqij -> cij', D22, A_yy)
                KK_22 = bm.einsum('cq, cqij -> cij', D00, A_yy) + bm.einsum('cq, cqij -> cij', D22, A_xx)
                KK_12 = bm.einsum('cq, cqij -> cij', D01, A_xy) + bm.einsum('cq, cqij -> cij', D22, A_yx)
                KK_21 = bm.einsum('cq, cqij -> cij', D01, A_yx) + bm.einsum('cq, cqij -> cij', D22, A_xy)
            else:
                D00 = D[..., 0, 0] # 2μ + λ
                D01 = D[..., 0, 1] # λ
                D22 = D[..., 2, 2] # μ
                KK_11 = bm.einsum('cq, cqij -> cij', D00, A_xx) + bm.einsum('cq, cqij -> cij', D55, A_yy + A_zz)
                KK_22 = bm.einsum('cq, cqij -> cij', D00, A_yy) + bm.einsum('cq, cqij -> cij', D55, A_xx + A_zz)
                KK_33 = bm.einsum('cq, cqij -> cij', D00, A_zz) + bm.einsum('cq, cqij -> cij', D55, A_xx + A_yy)
                KK_12 = bm.einsum('cq, cqij -> cij', D01, A_xy) + bm.einsum('cq, cqij -> cij', D55, A_yx)
                KK_13 = bm.einsum('cq, cqij -> cij', D01, A_xz) + bm.einsum('cq, cqij -> cij', D55, A_zx)
                KK_21 = bm.einsum('cq, cqij -> cij', D01, A_yx) + bm.einsum('cq, cqij -> cij', D55, A_xy)
                KK_23 = bm.einsum('cq, cqij -> cij', D01, A_yz) + bm.einsum('cq, cqij -> cij', D55, A_zy)
                KK_31 = bm.einsum('cq, cqij -> cij', D01, A_zx) + bm.einsum('cq, cqij -> cij', D55, A_xz)
                KK_32 = bm.einsum('cq, cqij -> cij', D01, A_zy) + bm.einsum('cq, cqij -> cij', D55, A_yz)

        if GD == 2:
            if space.dof_priority:
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(ldof, None)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, None)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(0, ldof)), KK_21)
            else:
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_21)
        else: 
            if space.dof_priority:
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(ldof, 2 * ldof)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(2 * ldof, None)), KK_33)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, 2 * ldof)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(2 * ldof, None)), KK_13)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(0, ldof)), KK_21)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(2 * ldof, None)), KK_23)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(0, ldof)), KK_31)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(ldof, 2 * ldof)), KK_32)
            else:
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_33)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_13)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_21)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_23)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_31)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_32)

        return KK
    
    @assembly.register('standard_multiresolution')
    def assembly(self, space: TensorFunctionSpace) -> TensorLike:
        index = self._index
        mesh_u = getattr(space, 'mesh', None)
        s_space_u = space.scalar_space
        GD = mesh_u.geo_dimension()
        q = s_space_u.p+3 if self._q is None else self._q
    
        # 单元密度多分辨率: (NC, n_sub); 节点密度多分辨率: (NC, n_sub, NQ)
        coef = self._coef
        NC, n_sub = coef.shape[0], coef.shape[1]
        
        # 计算位移单元积分点处的重心坐标
        qf_e = mesh_u.quadrature_formula(q)
        # bcs_e.shape = ( (NQ_x, GD), (NQ_y, GD) ), ws_e.shape = (NQ, )
        bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()
        NQ = ws_e.shape[0]

        # 把位移单元高斯积分点处的重心坐标映射到子密度单元 (子参考单元) 高斯积分点处的重心坐标 (仍表达在位移单元中)
        from soptx.analysis.utils import map_bcs_to_sub_elements
        # bcs_eg.shape = ( (n_sub, NQ_x, GD), (n_sub, NQ_y, GD) ), ws_e.shape = (NQ, )
        bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
        bcs_eg_x, bcs_eg_y = bcs_eg[0], bcs_eg[1]

        # 计算子密度单元内高斯积分点处的基函数梯度和 jacobi 矩阵
        LDOF = s_space_u.number_of_local_dofs()
        gphi_eg = bm.zeros((NC, n_sub, NQ, LDOF, GD)) # (NC, n_sub, NQ, LDOF, GD)
        detJ_eg = None

        if isinstance(mesh_u, SimplexMesh):
            cm = mesh_u.entity_measure('cell')
            cm_eg = bm.tile(cm.reshape(NC, 1), (1, n_sub)) # (NC, n_sub)
            
            for s_idx in range(n_sub):
                sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ_x, GD), (NQ_y, GD))
                gphi_sub = s_space_u.grad_basis(sub_bcs, index=index, variable='x')  # (NC, NQ, LDOF, GD)
                gphi_eg[:, s_idx, :, :, :] = gphi_sub

        else:
            detJ_eg = bm.zeros((NC, n_sub, NQ)) # (NC, n_sub, NQ)
            for s_idx in range(n_sub):
                sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ_x, GD), (NQ_y, GD))
                gphi_sub = s_space_u.grad_basis(sub_bcs, index=index, variable='x') # (NC, NQ, LDOF, GD)

                J_sub = mesh_u.jacobi_matrix(sub_bcs) # (NC, NQ, GD, GD)
                detJ_sub = bm.abs(bm.linalg.det(J_sub)) # (NC, NQ)

                gphi_eg[:, s_idx, :, :, :] = gphi_sub
                detJ_eg[:, s_idx, :] = detJ_sub

        if isinstance(mesh_u, SimplexMesh):
            A_xx_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 0], gphi_eg[..., 0], cm_eg)
            A_yy_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 1], gphi_eg[..., 1], cm_eg)
            A_xy_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 0], gphi_eg[..., 1], cm_eg)
            A_yx_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 1], gphi_eg[..., 0], cm_eg)
        else:
            A_xx_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 0], gphi_eg[..., 0], detJ_eg)
            A_yy_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 1], gphi_eg[..., 1], detJ_eg)
            A_xy_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 0], gphi_eg[..., 1], detJ_eg)
            A_yx_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 1], gphi_eg[..., 0], detJ_eg)

        if GD == 3:
            if isinstance(mesh_u, SimplexMesh):
                A_xz_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 0], gphi_eg[..., 2], cm_eg)
                A_zx_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 2], gphi_eg[..., 0], cm_eg)
                A_yz_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 1], gphi_eg[..., 2], cm_eg)
                A_zy_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 2], gphi_eg[..., 1], cm_eg)
                A_zz_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 2], gphi_eg[..., 2], cm_eg)
            else:
                A_xz_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 0], gphi_eg[..., 2], detJ_eg)
                A_zx_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 2], gphi_eg[..., 0], detJ_eg)
                A_yz_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 1], gphi_eg[..., 2], detJ_eg)
                A_zy_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 2], gphi_eg[..., 1], detJ_eg)
                A_zz_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 2], gphi_eg[..., 2], detJ_eg)

        # 位移单元 → 子密度单元的缩放
        J_g = 1 / n_sub

        # 基础材料的弹性矩阵
        D0 = self._material.elastic_matrix()[0, 0] # 2D: (3, 3); 3D: (6, 6)

        ldof = s_space_u.number_of_local_dofs()
        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64, device=mesh_u.device)

        # 区域内的相对密度恒定都为 1, D 为全局常数矩阵
        if coef is None:
            raise NotImplementedError("The global uniform density case is not implemented yet.")

        # 单元密度情况
        elif coef.shape == (NC, n_sub):
            if GD == 2:
                D00 = D0[0, 0] # 2D: E/(1-ν²) 或 2μ+λ
                D01 = D0[0, 1] # 2D: νE/(1-ν²) 或 λ
                D22 = D0[2, 2] # 2D: E/2(1+ν) 或 μ
                KK_11 = J_g * bm.einsum('cn, cnqij -> cij', coef * D00, A_xx_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D22, A_yy_eg)
                KK_22 = J_g * bm.einsum('cn, cnqij -> cij', coef * D00, A_yy_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D22, A_xx_eg)
                KK_12 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_xy_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D22, A_yx_eg)
                KK_21 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_yx_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D22, A_xy_eg)
            else: 
                D00 = D0[0, 0] # 2μ + λ
                D01 = D0[0, 1] # λ
                D55 = D0[5, 5] # μ
                KK_11 = J_g * bm.einsum('cn, cnqij -> cij', coef * D00, A_xx_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_yy_eg + A_zz_eg)
                KK_22 = J_g * bm.einsum('cn, cnqij -> cij', coef * D00, A_yy_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_xx_eg + A_zz_eg)
                KK_33 = J_g * bm.einsum('cn, cnqij -> cij', coef * D00, A_zz_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_xx_eg + A_yy_eg)
                KK_12 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_xy_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_yx_eg)
                KK_13 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_xz_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_zx_eg)
                KK_21 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_yx_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_xy_eg)
                KK_23 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_yz_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_zy_eg)
                KK_31 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_zx_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_xz_eg)
                KK_32 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_zy_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_yz_eg)
                    
        # 节点密度情况
        elif coef.shape == (NC, n_sub, NQ):
            if GD == 2:
                D00, D01, D22 = D0[0, 0], D0[0, 1], D0[2, 2]
                KK_11 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D00, A_xx_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D22, A_yy_eg)
                KK_22 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D00, A_yy_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D22, A_xx_eg)
                KK_12 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_xy_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D22, A_yx_eg)
                KK_21 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_yx_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D22, A_xy_eg)
            else: 
                D00, D01, D55 = D0[0, 0], D0[0, 1], D0[5, 5]
                KK_11 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D00, A_xx_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_yy_eg + A_zz_eg)
                KK_22 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D00, A_yy_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_xx_eg + A_zz_eg)
                KK_33 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D00, A_zz_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_xx_eg + A_yy_eg)
                KK_12 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_xy_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_yx_eg)
                KK_13 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_xz_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_zx_eg)
                KK_21 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_yx_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_xy_eg)
                KK_23 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_yz_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_zy_eg)
                KK_31 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_zx_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_xz_eg)
                KK_32 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_zy_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_yz_eg)

        if GD == 2:
            if space.dof_priority:
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(ldof, None)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, None)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(0, ldof)), KK_21)
            else:
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_21)
        else:  
            if space.dof_priority:
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(ldof, 2 * ldof)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(2 * ldof, None)), KK_33)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, 2 * ldof)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(2 * ldof, None)), KK_13)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(0, ldof)), KK_21)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(2 * ldof, None)), KK_23)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(0, ldof)), KK_31)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(ldof, 2 * ldof)), KK_32)
            else:
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_33)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_13)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_21)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_23)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_31)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_32)

        return KK

    @enable_cache
    def fetch_voigt_assembly(self, space: TensorFunctionSpace):
        index = self._index
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
    
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The LinearElasticIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
    
        cm = mesh.entity_measure('cell', index=index)
        q = scalar_space.p+3 if self._q is None else self._q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = scalar_space.grad_basis(bcs, index=index, variable='x')

        if isinstance(mesh, SimplexMesh):
            J = None
            detJ = None
        else:
            J = mesh.jacobi_matrix(bcs)
            detJ = bm.abs(bm.linalg.det(J))

        return cm, ws, bcs, gphi, detJ

    @assembly.register('voigt')
    def assembly(self, space: TensorFunctionSpace) -> TensorLike:
        mesh = getattr(space, 'mesh', None)
        cm, ws, bcs, gphi, detJ = self.fetch_voigt_assembly(space)

        NC = mesh.number_of_cells()
        NQ = gphi.shape[1]
        D0 = self._material.elastic_matrix() # 2D: (1, 1, NS, NS)
        B = self._material.strain_displacement_matrix(dof_priority=space.dof_priority, 
                                                    gphi=gphi) # (NC, NQ, NS, LDOF)

        # 单元密度: (NC, ); 节点密度: (NC, NQ)
        coef = self._coef

        if coef is None:

            D = D0[0, 0] # (NS, NS)

            if isinstance(mesh, SimplexMesh):
                KK = bm.einsum('q, c, cqki, kl, cqlj -> cij', ws, cm, B, D, B)
            else:
                KK = bm.einsum('q, cq, cqki, kl, cqlj -> cij', ws, detJ, B, D, B)
        
        # 单元密度的情况
        elif coef.shape == (NC, ):
            
            D_base = D0[0, 0] # (NS, NS)
            D = bm.einsum('c, kl -> ckl', coef, D_base) # (NC, NS, NS)
            
            if isinstance(mesh, SimplexMesh):
                KK = bm.einsum('q, c, cqki, ckl, cqlj -> cij', ws, cm, B, D, B)
            else:
                KK = bm.einsum('q, cq, cqki, ckl, cqlj -> cij', ws, detJ, B, D, B)
                    
        # 节点密度的情况
        elif coef.shape == (NC, NQ):
            
            D = bm.einsum('cq, ijkl -> cqkl', coef, D0) # (NC, NQ, NS, NS)
            
            if isinstance(mesh, SimplexMesh):
                KK = bm.einsum('q, c, cqki, cqkl, cqlj -> cij', ws, cm, B, D, B)
            else:
                KK = bm.einsum('q, cq, cqki, cqkl, cqlj -> cij', ws, detJ, B, D, B)
        
        else:


            raise NotImplementedError

        return KK

    @assembly.register('voigt_multiresolution')
    def assembly(self, space: TensorFunctionSpace) -> TensorLike:
        index = self._index
        mesh_u = getattr(space, 'mesh', None)
        s_space_u = space.scalar_space
        GD = mesh_u.geo_dimension()
        q = s_space_u.p+3 if self._q is None else self._q
       
        # 单元密度多分辨率: (NC, n_sub); 节点密度多分辨率: (NC, n_sub, NQ)
        coef = self._coef
        NC, n_sub = coef.shape[0], coef.shape[1]
        
        # 计算位移单元积分点处的重心坐标
        qf_e = mesh_u.quadrature_formula(q)
        # bcs_e.shape = ( (NQ_x, GD), (NQ_y, GD) ), ws_e.shape = (NQ, )
        bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()
        NQ = ws_e.shape[0]

        # 把位移单元高斯积分点处的重心坐标映射到子密度单元 (子参考单元) 高斯积分点处的重心坐标 (仍表达在位移单元中)
        from soptx.analysis.utils import map_bcs_to_sub_elements
        # bcs_eg.shape = ( (n_sub, NQ_x, GD), (n_sub, NQ_y, GD) ), ws_e.shape = (NQ, )
        bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
        bcs_eg_x, bcs_eg_y = bcs_eg[0], bcs_eg[1]

        # 计算子密度单元内高斯积分点处的基函数梯度和 jacobi 矩阵
        LDOF = s_space_u.number_of_local_dofs()
        gphi_eg = bm.zeros((NC, n_sub, NQ, LDOF, GD)) # (NC, n_sub, NQ, LDOF, GD)
        detJ_eg = None

        if isinstance(mesh_u, SimplexMesh):
            for s_idx in range(n_sub):
                sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ_x, GD), (NQ_y, GD))
                gphi_sub = s_space_u.grad_basis(sub_bcs, index=index, variable='x')  # (NC, NQ, LDOF, GD)
                gphi_eg[:, s_idx, :, :, :] = gphi_sub

        else:
            detJ_eg = bm.zeros((NC, n_sub, NQ)) # (NC, n_sub, NQ)
            for s_idx in range(n_sub):
                sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ_x, GD), (NQ_y, GD))
                gphi_sub = s_space_u.grad_basis(sub_bcs, index=index, variable='x') # (NC, NQ, LDOF, GD)

                J_sub = mesh_u.jacobi_matrix(sub_bcs) # (NC, NQ, GD, GD)
                detJ_sub = bm.abs(bm.linalg.det(J_sub)) # (NC, NQ)

                gphi_eg[:, s_idx, :, :, :] = gphi_sub
                detJ_eg[:, s_idx, :] = detJ_sub

        # 计算 B 矩阵
        from soptx.analysis.utils import reshape_multiresolution_data, reshape_multiresolution_data_inverse
        nx_u, ny_u = mesh_u.meshdata['nx'], mesh_u.meshdata['ny']
        gphi_eg_reshaped = reshape_multiresolution_data(nx=nx_u, ny=ny_u, data=gphi_eg) # (NC*n_sub, NQ, LDOF, GD)
        B_eg_reshaped = self._material.strain_displacement_matrix(
                                            dof_priority=space.dof_priority, 
                                            gphi=gphi_eg_reshaped
                                        ) # 2D: (NC*n_sub, NQ, 3, TLDOF), 3D: (NC*n_sub, NQ, 6, TLDOF)
        B_eg = reshape_multiresolution_data_inverse(nx=nx_u, 
                                                    ny=ny_u, 
                                                    data_flat=B_eg_reshaped, 
                                                    n_sub=n_sub) # (NC, n_sub, NQ, 3, TLDOF) or (NC, n_sub, NQ, 6, TLDOF)

        # 位移单元 → 子密度单元的缩放
        J_g = 1 / n_sub

        # 基础材料的弹性矩阵
        D0 = self._material.elastic_matrix()[0, 0] # 2D: (3, 3); 3D: (6, 6)

        if coef is None:
            raise NotImplementedError("The global constant density is not implemented"
                                      " in the multiresolution assembly.")

        # 单元密度
        if coef.shape == (NC, n_sub):

            D_g = bm.einsum('kl, cn -> cnkl', D0, coef) # 2D: (NC, n_sub, 3, 3); 3D: (NC, n_sub, 6, 6)
            if isinstance(mesh_u, SimplexMesh):
                cm = mesh_u.entity_measure('cell')
                cm_eg = bm.tile(cm.reshape(NC, 1), (1, n_sub)) # (NC, n_sub)
                KK = J_g * bm.einsum('q, cn, cnqki, cnkl, cnqlj -> cij',
                                    ws_e, cm_eg, B_eg, D_g, B_eg)
            else:
                KK = J_g * bm.einsum('q, cnq, cnqki, cnkl, cnqlj -> cij',
                                    ws_e, detJ_eg, B_eg, D_g, B_eg)
                
            return KK

        # 节点密度
        elif coef.shape == (NC, n_sub, NQ):

            D_g = bm.einsum('ijkl, cnq -> cnqkl', D0, coef) # 2D: (NC, n_sub, NQ, 3, 3); 3D: (NC, n_sub, NQ, 6, 6)
            if isinstance(mesh_u, SimplexMesh):
                KK = J_g * bm.einsum('q, cn, cnqki, cnqkl, cnqlj -> cij',
                                    ws_e, cm_eg, B_eg, D_g, B_eg)
            else:
                KK = J_g * bm.einsum('q, cnq, cnqki, cnqkl, cnqlj -> cij',
                                    ws_e, detJ_eg, B_eg, D_g, B_eg)
                
            return KK


    @enable_cache
    def fetch_fast_assembly(self, space: TensorFunctionSpace):
        index = self.index
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
    
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The LinearElasticIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
    
        cm = mesh.entity_measure('cell', index=index)
        q = scalar_space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi_lambda = scalar_space.grad_basis(bcs, index=index, variable='u')    # (NQ, LDOF, BC)

        if isinstance(mesh, SimplexMesh):
            glambda_x = mesh.grad_lambda()   # (NC, LDOF, GD)
            S = bm.einsum('q, qik, qjl -> ijkl', ws, gphi_lambda, gphi_lambda)  # (LDOF, LDOF, BC, BC)
            return cm, bcs, glambda_x, S
        
        elif isinstance(mesh, StructuredMesh):
            J = mesh.jacobi_matrix(bcs)[:, 0, ...]         # (NC, GD, GD)
            G = mesh.first_fundamental_form(J)             # (NC, GD, GD)
            G = bm.linalg.inv(G)                           # (NC, GD, GD)
            JG = bm.einsum('ckm, cmn -> ckn', J, G)        # (NC, GD, GD)
            S = bm.einsum('qim, qjn, q -> ijmn', gphi_lambda, gphi_lambda, ws)  # (LDOF, LDOF, BC, BC)
            return cm, bcs, JG, S
        
        else:
            J = mesh.jacobi_matrix(bcs)                   # (NC, NQ, GD, GD)
            detJ = bm.linalg.det(J)                       # (NC, NQ)
            G = mesh.first_fundamental_form(J)            # (NC, NQ, GD, GD)
            G = bm.linalg.inv(G)                          # (NC, NQ, GD, GD)
            JG = bm.einsum('cqkm, cqmn -> cqkn', J, G)    # (NC, NQ, GD, GD)
            S = bm.einsum('qim, qjn, q -> ijmnq', gphi_lambda, gphi_lambda, ws)  # (LDOF, LDOF, GD, GD, NQ)
        
            return cm, bcs, detJ, JG, S

    @assembly.register('fast')
    def assembly(self, space: TensorFunctionSpace) -> TensorLike:
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)

        D = self.material.elastic_matrix(bcs)
        if D.shape[1] != 1:
            raise ValueError("assembly currently only supports elastic matrices "
                            f"with shape (NC, 1, {2*GD}, {2*GD}) or (1, 1, {2*GD}, {2*GD}).")
                
        NC = mesh.number_of_cells()

        if isinstance(mesh, SimplexMesh):
            cm, bcs, glambda_x, S = self.fetch_fast_assembly(space)

            A_xx = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 0], glambda_x[..., 0], cm) # (NC, LDOF, LDOF)
            A_yy = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 1], glambda_x[..., 1], cm)
            A_xy = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 0], glambda_x[..., 1], cm)
            A_yx = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 1], glambda_x[..., 0], cm)

        elif isinstance(mesh, StructuredMesh):
            cm, bcs, JG, S = self.fetch_fast_assembly(space)

            A_xx = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 0], JG[..., 0], cm)  # (NC, LDOF, LDOF)
            A_yy = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 1], JG[..., 1], cm) 
            A_xy = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 0], JG[..., 1], cm)  
            A_yx = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 1], JG[..., 0], cm)  
        
        else:
            cm, bcs, detJ, JG, S = self.fetch_fast_assembly(space)

            A_xx = bm.einsum('ijmnq, cqm, cqn, cq -> cij', S, JG[..., 0, :], JG[..., 0, :], detJ) # (NC, LDOF, LDOF)
            A_yy = bm.einsum('ijmnq, cqm, cqn, cq -> cij', S, JG[..., 1, :], JG[..., 1, :], detJ) 
            A_xy = bm.einsum('ijmnq, cqm, cqn, cq -> cij', S, JG[..., 0, :], JG[..., 1, :], detJ) 
            A_yx = bm.einsum('ijmnq, cqm, cqn, cq -> cij', S, JG[..., 1, :], JG[..., 0, :], detJ) 
        
        GD = mesh.geo_dimension()
        if GD == 3:

            if isinstance(mesh, SimplexMesh):
                A_zz = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 2], glambda_x[..., 2], cm)
                A_xz = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 0], glambda_x[..., 2], cm)
                A_yz = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 1], glambda_x[..., 2], cm)
                A_zx = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 2], glambda_x[..., 0], cm)
                A_zy = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 2], glambda_x[..., 1], cm)
            
            elif isinstance(mesh, StructuredMesh):
                A_zz = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 2], JG[..., 2], cm)
                A_xz = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 0], JG[..., 2], cm)
                A_yz = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 1], JG[..., 2], cm)
                A_zx = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 2], JG[..., 0], cm)
                A_zy = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 2], JG[..., 1], cm)
            
            else:
                A_zz = bm.einsum('ijmnq, cqm, cqn, cq -> cij', S, JG[..., 2, :], JG[..., 2, :], detJ)
                A_xz = bm.einsum('ijmnq, cqm, cqn, cq -> cij', S, JG[..., 0, :], JG[..., 2, :], detJ)
                A_yz = bm.einsum('ijmnq, cqm, cqn, cq -> cij', S, JG[..., 1, :], JG[..., 2, :], detJ)
                A_zx = bm.einsum('ijmnq, cqm, cqn, cq -> cij', S, JG[..., 2, :], JG[..., 0, :], detJ)
                A_zy = bm.einsum('ijmnq, cqm, cqn, cq -> cij', S, JG[..., 2, :], JG[..., 1, :], detJ)

        ldof = scalar_space.number_of_local_dofs()
        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64, device=mesh.device)

        if GD == 2:
            D00 = D[..., 0, 0, None]  # E / (1-\nu^2) * 1         or 2*\mu + \lambda
            D01 = D[..., 0, 1, None]  # E / (1-\nu^2) * \nu       or \lambda
            D22 = D[..., 2, 2, None]  # E / (1-\nu^2) * (1-nu)/2  or \mu

            if space.dof_priority:
                # Fill the diagonal part
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), 
                                D00 * A_xx + D22 * A_yy)
                KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1]), slice(ldof, KK.shape[1])), 
                                D00 * A_yy + D22 * A_xx)

                # Fill the off-diagonal part
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, KK.shape[1])), 
                            D01 * A_xy + D22 * A_yx)
                KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1]), slice(0, ldof)), 
                            D01 * A_yx + D22 * A_xy)
            else:
                # Fill the diagonal part
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                            D00 * A_xx + D22 * A_yy)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                            D00 * A_yy + D22 * A_xx)

                # Fill the off-diagonal part
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                            D01 * A_xy + D22 * A_yx)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                            D01 * A_yx + D22 * A_xy)
        else:
            D00 = D[..., 0, 0, None]  # 2μ + λ
            D01 = D[..., 0, 1, None]  # λ
            D55 = D[..., 5, 5, None]  # μ

            if space.dof_priority:
                # Fill the diagonal part
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), 
                                D00 * A_xx + D55 * A_yy + D55 * A_zz)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(ldof, 2 * ldof)), 
                                D00 * A_yy + D55 * A_xx + D55 * A_zz)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(2 * ldof, None)), 
                                D00 * A_zz + D55 * A_xx + D55 * A_yy)

                # Fill the off-diagonal part
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, 2 * ldof)), 
                                D01 * A_xy + D55 * A_yx)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(2 * ldof, None)), 
                                D01 * A_xz + D55 * A_zx)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(0, ldof)), 
                                D01 * A_yx + D55 * A_xy)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(2 * ldof, None)), 
                                D01 * A_yz + D55 * A_zy)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(0, ldof)), 
                                D01 * A_zx + D55 * A_xz)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(ldof, 2 * ldof)), 
                                D01 * A_zy + D55 * A_yz)
            else:
                # Fill the diagonal part
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                                (2 * D55 + D01) * A_xx + D55 * (A_yy + A_zz))
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                                (2 * D55 + D01) * A_yy + D55 * (A_xx + A_zz))
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(2, KK.shape[2], GD)), 
                                (2 * D55 + D01) * A_zz + D55 * (A_xx + A_yy))

                # Fill the off-diagonal
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                                D01 * A_xy + D55 * A_yx)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(2, KK.shape[2], GD)), 
                                D01 * A_xz + D55 * A_zx)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                                D01 * A_yx + D55 * A_xy)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(2, KK.shape[2], GD)), 
                                D01 * A_yz + D55 * A_zy)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                                D01 * A_zx + D55 * A_xz)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                                D01 * A_zy + D55 * A_yz)

        return KK

    @enable_cache
    def fetch_symbolic_assembly(self, space: TensorFunctionSpace) -> TensorLike:
        index = self.index
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
    
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The LinearElasticIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
    
        cm = mesh.entity_measure('cell', index=index)
        q = scalar_space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        cell_vertices = node[cell]

        symbolic_int = LinearSymbolicIntegration(space1=scalar_space, space2=scalar_space)
        kwargs = bm.context(node)

        S = bm.tensor(symbolic_int.gphi_gphi_matrix(), **kwargs)  # (LDOF1, LDOF1, BC, BC)

        if isinstance(mesh, SimplexMesh):   
            glambda_x = mesh.grad_lambda()  # (NC, LDOF, GD)
            return cm, bcs, glambda_x, S
        elif isinstance(mesh, StructuredMesh):
            JG = symbolic_int.compute_mapping(vertices=cell_vertices)  # (NC, GD, GD)
            return cm, bcs, JG, S
        else:
            raise NotImplementedError("symbolic assembly for general meshes is not implemented yet.")

    @assembly.register('symbolic')
    def assembly(self, space: TensorFunctionSpace) -> TensorLike:
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
        
        if isinstance(mesh, SimplexMesh):
            cm, bcs, glambda_x, S = self.fetch_symbolic_assembly(space)   
            # 计算各方向的矩阵
            A_xx = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 0], glambda_x[..., 0], cm)
            A_yy = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 1], glambda_x[..., 1], cm)
            A_xy = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 0], glambda_x[..., 1], cm)
            A_yx = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 1], glambda_x[..., 0], cm)
        elif isinstance(mesh, StructuredMesh):
            cm, bcs, JG, S = self.fetch_symbolic_assembly(space)
            A_xx = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 0], JG[..., 0], cm)
            A_yy = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 1], JG[..., 1], cm)
            A_xy = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 0], JG[..., 1], cm)
            A_yx = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 1], JG[..., 0], cm)   
        else:
            raise NotImplementedError("symbolic assembly for general meshes is not implemented yet.")
        
        GD = mesh.geo_dimension()
        if GD == 3:
            if isinstance(mesh, SimplexMesh):
                A_zz = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 2], glambda_x[..., 2], cm)
                A_xz = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 0], glambda_x[..., 2], cm)
                A_yz = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 1], glambda_x[..., 2], cm)
                A_zx = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 2], glambda_x[..., 0], cm)
                A_zy = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 2], glambda_x[..., 1], cm)
            elif isinstance(mesh, StructuredMesh):
                A_zz = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 2], JG[..., 2], cm)
                A_xz = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 0], JG[..., 2], cm)
                A_yz = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 1], JG[..., 2], cm)
                A_zx = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 2], JG[..., 0], cm)
                A_zy = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 2], JG[..., 1], cm)
            else:
                raise NotImplementedError("symbolic assembly for general meshes is not implemented yet.")

        D = self.material.elastic_matrix(bcs)
        if D.shape[1] != 1:
            raise ValueError("assembly currently only supports elastic matrices "
                            f"with shape (NC, 1, {2*GD}, {2*GD}) or (1, 1, {2*GD}, {2*GD}).")
        
        NC = mesh.number_of_cells()
        ldof = scalar_space.number_of_local_dofs()
        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64, device=mesh.device)

        if GD == 2:
            D00 = D[..., 0, 0, None]  # E / (1-\nu^2) * 1         or 2*\mu + \lambda
            D01 = D[..., 0, 1, None]  # E / (1-\nu^2) * \nu       or \lambda
            D22 = D[..., 2, 2, None]  # E / (1-\nu^2) * (1-nu)/2  or \mu

            if space.dof_priority:
                # 填充对角块
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), 
                            D00 * A_xx + D22 * A_yy)
                KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1]), slice(ldof, KK.shape[1])), 
                            D00 * A_yy + D22 * A_xx)
                
                # 填充非对角块
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, KK.shape[1])), 
                            D01 * A_xy + D22 * A_yx)
                KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1]), slice(0, ldof)), 
                            D01 * A_yx + D22 * A_xy)
            else:
                # 类似的填充方式，但使用不同的索引方式
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                            D00 * A_xx + D22 * A_yy)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                            D00 * A_yy + D22 * A_xx)
                
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                            D01 * A_xy + D22 * A_yx)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                            D01 * A_yx + D22 * A_xy)
        else:
            D00 = D[..., 0, 0, None]  # 2μ + λ
            D01 = D[..., 0, 1, None]  # λ
            D55 = D[..., 5, 5, None]  # μ

            if space.dof_priority:
                # Fill the diagonal part
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), 
                                D00 * A_xx + D55 * A_yy + D55 * A_zz)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(ldof, 2 * ldof)), 
                                D00 * A_yy + D55 * A_xx + D55 * A_zz)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(2 * ldof, None)), 
                                D00 * A_zz + D55 * A_xx + D55 * A_yy)

                # Fill the off-diagonal part
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, 2 * ldof)), 
                                D01 * A_xy + D55 * A_yx)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(2 * ldof, None)), 
                                D01 * A_xz + D55 * A_zx)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(0, ldof)), 
                                D01 * A_yx + D55 * A_xy)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(2 * ldof, None)), 
                                D01 * A_yz + D55 * A_zy)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(0, ldof)), 
                                D01 * A_zx + D55 * A_xz)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(ldof, 2 * ldof)), 
                                D01 * A_zy + D55 * A_yz)
            else:
                # Fill the diagonal part
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                                (2 * D55 + D01) * A_xx + D55 * (A_yy + A_zz))
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                                (2 * D55 + D01) * A_yy + D55 * (A_xx + A_zz))
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(2, KK.shape[2], GD)), 
                                (2 * D55 + D01) * A_zz + D55 * (A_xx + A_yy))

                # Fill the off-diagonal
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                                D01 * A_xy + D55 * A_yx)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(2, KK.shape[2], GD)), 
                                D01 * A_xz + D55 * A_zx)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                                D01 * A_yx + D55 * A_xy)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(2, KK.shape[2], GD)), 
                                D01 * A_yz + D55 * A_zy)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                                D01 * A_zx + D55 * A_xz)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                                D01 * A_zy + D55 * A_yz)

        return KK
    
