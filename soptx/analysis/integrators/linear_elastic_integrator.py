from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S
from fealpy.mesh import HomogeneousMesh, SimplexMesh, StructuredMesh
from fealpy.functionspace.space import FunctionSpace
from fealpy.functionspace.tensor_space import TensorFunctionSpace
from fealpy.decorator.variantmethod import variantmethod
from fealpy.fem.integrator import (LinearInt, OpInt, CellInt, enable_cache)
from fealpy.fem.utils import LinearSymbolicIntegration

from ...interpolation.linear_elastic_material import LinearElasticMaterial

class LinearElasticIntegrator(LinearInt, OpInt, CellInt):
    """The linear elastic integrator for function spaces based on homogeneous meshes."""
    def __init__(self, 
                material: LinearElasticMaterial,
                coef: TensorLike,
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
            detJ = bm.linalg.det(J)

        return cm, ws, gphi, detJ

    @variantmethod('standard')
    def assembly(self, space: TensorFunctionSpace) -> TensorLike:
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
        cm, ws, gphi, detJ = self.fetch_assembly(space)

        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        NQ = len(ws)
        D0 = self._material.elastic_matrix()
        coef = self._coef

        if coef is None:
            D = D0
        elif coef.shape == (NC, ):
            D = bm.einsum('c, ijkl -> cjkl', coef, D0)
        elif coef.shape == (NC, NQ):
            D = bm.einsum('cq, ijkl -> cqkl', coef, D0)

        if not (D.shape[0] == 1 and D.shape[1] == 1) and \
            not (D.shape[0] == NC and D.shape[1] == 1) and \
            not (D.shape[0] == NC and D.shape[1] == NQ):
            raise ValueError(f"assembly currently only supports elastic matrices "
                    f"with shape (1, 1, {GD*(GD+1)//2}, {GD*(GD+1)//2}) or "
                    f"({NC}, 1, {GD*(GD+1)//2}, {GD*(GD+1)//2}) or "
                    f"({NC}, {NQ}, {GD*(GD+1)//2}, {GD*(GD+1)//2}), "
                    f"got {D.shape}.")
            
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

        if D.shape[0] == 1 and D.shape[1] == 1:
            # (1, 1, :, :) - 全局均匀相对密度
            if GD == 2:
                D00, D01, D22 = D[0, 0, 0, 0], D[0, 0, 0, 1], D[0, 0, 2, 2]
                KK_11 = D00 * bm.einsum('cqij -> cij', A_xx) + D22 * bm.einsum('cqij -> cij', A_yy)
                KK_22 = D00 * bm.einsum('cqij -> cij', A_yy) + D22 * bm.einsum('cqij -> cij', A_xx)
                KK_12 = D01 * bm.einsum('cqij -> cij', A_xy) + D22 * bm.einsum('cqij -> cij', A_yx)
                KK_21 = D01 * bm.einsum('cqij -> cij', A_yx) + D22 * bm.einsum('cqij -> cij', A_xy)
            else: 
                D00, D01, D55 = D[0, 0, 0, 0], D[0, 0, 0, 1], D[0, 0, 5, 5]
                KK_11 = D00 * bm.einsum('cqij -> cij', A_xx) + D55 * bm.einsum('cqij -> cij', A_yy + A_zz)
                KK_22 = D00 * bm.einsum('cqij -> cij', A_yy) + D55 * bm.einsum('cqij -> cij', A_xx + A_zz)
                KK_33 = D00 * bm.einsum('cqij -> cij', A_zz) + D55 * bm.einsum('cqij -> cij', A_xx + A_yy)
                KK_12 = D01 * bm.einsum('cqij -> cij', A_xy) + D55 * bm.einsum('cqij -> cij', A_yx)
                KK_13 = D01 * bm.einsum('cqij -> cij', A_xz) + D55 * bm.einsum('cqij -> cij', A_zx)
                KK_21 = D01 * bm.einsum('cqij -> cij', A_yx) + D55 * bm.einsum('cqij -> cij', A_xy)
                KK_23 = D01 * bm.einsum('cqij -> cij', A_yz) + D55 * bm.einsum('cqij -> cij', A_zy)
                KK_31 = D01 * bm.einsum('cqij -> cij', A_zx) + D55 * bm.einsum('cqij -> cij', A_xz)
                KK_32 = D01 * bm.einsum('cqij -> cij', A_zy) + D55 * bm.einsum('cqij -> cij', A_yz)
                
        elif D.shape[1] == 1:
            # (NC, 1, :, :) - 单元均匀相对密度
            if GD == 2:
                D00, D01, D22 = D[:, 0, 0, 0], D[:, 0, 0, 1], D[:, 0, 2, 2]
                KK_11 = bm.einsum('c, cqij -> cij', D00, A_xx) + bm.einsum('c, cqij -> cij', D22, A_yy)
                KK_22 = bm.einsum('c, cqij -> cij', D00, A_yy) + bm.einsum('c, cqij -> cij', D22, A_xx)
                KK_12 = bm.einsum('c, cqij -> cij', D01, A_xy) + bm.einsum('c, cqij -> cij', D22, A_yx)
                KK_21 = bm.einsum('c, cqij -> cij', D01, A_yx) + bm.einsum('c, cqij -> cij', D22, A_xy)
            else:  # GD == 3
                D00, D01, D55 = D[:, 0, 0, 0], D[:, 0, 0, 1], D[:, 0, 5, 5]
                KK_11 = bm.einsum('c, cqij -> cij', D00, A_xx) + bm.einsum('c, cqij -> cij', D55, A_yy + A_zz)
                KK_22 = bm.einsum('c, cqij -> cij', D00, A_yy) + bm.einsum('c, cqij -> cij', D55, A_xx + A_zz)
                KK_33 = bm.einsum('c, cqij -> cij', D00, A_zz) + bm.einsum('c, cqij -> cij', D55, A_xx + A_yy)
                KK_12 = bm.einsum('c, cqij -> cij', D01, A_xy) + bm.einsum('c, cqij -> cij', D55, A_yx)
                KK_13 = bm.einsum('c, cqij -> cij', D01, A_xz) + bm.einsum('c, cqij -> cij', D55, A_zx)
                KK_21 = bm.einsum('c, cqij -> cij', D01, A_yx) + bm.einsum('c, cqij -> cij', D55, A_xy)
                KK_23 = bm.einsum('c, cqij -> cij', D01, A_yz) + bm.einsum('c, cqij -> cij', D55, A_zy)
                KK_31 = bm.einsum('c, cqij -> cij', D01, A_zx) + bm.einsum('c, cqij -> cij', D55, A_xz)
                KK_32 = bm.einsum('c, cqij -> cij', D01, A_zy) + bm.einsum('c, cqij -> cij', D55, A_yz)
                
        else:
            # (NC, NQ, :, :) - 单元高斯积分点相对密度
            if GD == 2:
                D00, D01, D22 = D[..., 0, 0], D[..., 0, 1], D[..., 2, 2]
                KK_11 = bm.einsum('cq, cqij -> cij', D00, A_xx) + bm.einsum('cq, cqij -> cij', D22, A_yy)
                KK_22 = bm.einsum('cq, cqij -> cij', D00, A_yy) + bm.einsum('cq, cqij -> cij', D22, A_xx)
                KK_12 = bm.einsum('cq, cqij -> cij', D01, A_xy) + bm.einsum('cq, cqij -> cij', D22, A_yx)
                KK_21 = bm.einsum('cq, cqij -> cij', D01, A_yx) + bm.einsum('cq, cqij -> cij', D22, A_xy)
            else:  # GD == 3
                D00, D01, D55 = D[..., 0, 0], D[..., 0, 1], D[..., 5, 5]
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

    @enable_cache
    def fetch_voigt_assembly(self, space: TensorFunctionSpace):
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
        gphi = scalar_space.grad_basis(bcs, index=index, variable='x')

        if isinstance(mesh, SimplexMesh):
            J = None
            detJ = None
        else:
            J = mesh.jacobi_matrix(bcs)
            detJ = bm.linalg.det(J)

        D = self.material.elastic_matrix(bcs)
        B = self.material.strain_matrix(dof_priority=space.dof_priority, 
                                        gphi=gphi)

        return cm, ws, detJ, D, B
            
    @assembly.register('voigt')
    def assembly(self, space: TensorFunctionSpace) -> TensorLike:
        mesh = getattr(space, 'mesh', None)
        cm, ws, detJ, D, B = self.fetch_voigt_assembly(space)

        if isinstance(mesh, SimplexMesh):
            KK = bm.einsum('q, c, cqki, cqkl, cqlj -> cij',
                            ws, cm, B, D, B)
        else:
            KK = bm.einsum('q, cq, cqki, cqkl, cqlj -> cij',
                            ws, detJ, B, D, B)
            
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
    
