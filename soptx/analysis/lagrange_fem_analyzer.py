from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import VectorSourceIntegrator
from fealpy.decorator.variantmethod import variantmethod
from fealpy.sparse import CSRTensor

from ..interpolation.linear_elastic_material import LinearElasticMaterial
from .integrators.linear_elastic_integrator import LinearElasticIntegrator

class LagrangeFEMAnalyzer:
    def __init__(self,
                pde, 
                material: LinearElasticMaterial,
                space_degree: int = 1,
                assembly_method: str = 'standard',
                solve_method: str = 'mumps',
            ) -> None:
        self.pde = pde
        self.mesh =self.pde.mesh
        self.material = material
        self.space_degree = space_degree
        self.assembly_method = assembly_method
        self.solve.set(solve_method)

        self.scalar_space = LagrangeFESpace(self.mesh, p=space_degree, ctype='C')
        GD = self.mesh.geo_dimension()
        self.tensor_space = TensorFunctionSpace(scalar_space=self.scalar_space, shape=(GD, -1))

    def assemble_stiff_matrix(self) -> CSRTensor:
        """组装刚度矩阵"""
        integrator = LinearElasticIntegrator(material=self.material, 
                                            q=self.space_degree+3,
                                            method=self.assembly_method)
        bform = BilinearForm(self.tensor_space)
        bform.add_integrator(integrator)
        K = bform.assembly(format='csr')

        return K
    
    def assemble_force_vector(self) -> CSRTensor:
        """组装载荷向量"""
        body_force = self.pde.body_force
        force_type = self.pde.force_type

        if force_type == 'concentrated':
            F = self.tensor_space.interpolate(body_force)
        elif force_type == 'continuous':
            integrator = VectorSourceIntegrator(source=body_force, q=self.space_degree+3)
            lform = LinearForm(self.tensor_space)
            lform.add_integrator(integrator)
            F = lform.assembly(format='csr')
        else:
            raise ValueError(f"Unsupported force type: {force_type}")

        return F
    
    def _apply_matrix(self, A: CSRTensor, isDDof: TensorLike) -> CSRTensor:
        """
        FEALPy 中的 apply_matrix 使用了 D0@A@D0, 
        不同后端下 @ 会使用大量的 for 循环, 这在 GPU 下非常缓慢 
        """
        isIDof = bm.logical_not(isDDof)
        crow = A.crow
        col = A.col
        indices_context = bm.context(col)
        ZERO = bm.array([0], **indices_context)

        nnz_per_row = crow[1:] - crow[:-1]
        remain_flag = bm.repeat(isIDof, nnz_per_row) & isIDof[col] # 保留行列均为内部自由度的非零元素
        rm_cumsum = bm.concat([ZERO, bm.cumsum(remain_flag, axis=0)], axis=0) # 被保留的非零元素数量累积
        nnz_per_row = rm_cumsum[crow[1:]] - rm_cumsum[crow[:-1]] + isDDof # 计算每行的非零元素数量

        new_crow = bm.cumsum(bm.concat([ZERO, nnz_per_row], axis=0), axis=0)

        NNZ = new_crow[-1]
        non_diag = bm.ones((NNZ,), dtype=bm.bool, device=bm.get_device(isDDof)) # Field: non-zero elements
        loc_flag = bm.logical_and(new_crow[:-1] < NNZ, isDDof)
        non_diag = bm.set_at(non_diag, new_crow[:-1][loc_flag], False)

        # 修复：只选取适当数量的值对应设置
        # 找出所有边界DOF对应的行索引
        bd_rows = bm.where(loc_flag)[0]
        new_col = bm.empty((NNZ,), **indices_context)
        # 设置为相应行的边界 DOF 位置
        new_col = bm.set_at(new_col, new_crow[:-1][loc_flag], bd_rows)
        # 设置非对角元素的列索引
        new_col = bm.set_at(new_col, non_diag, col[remain_flag])

        new_values = bm.empty((NNZ,), **A.values_context())
        new_values = bm.set_at(new_values, new_crow[:-1][loc_flag], 1.)
        new_values = bm.set_at(new_values, non_diag, A.values[remain_flag])

        return CSRTensor(new_crow, new_col, new_values, A.sparse_shape)

    def apply_bc(self, K: CSRTensor, F: CSRTensor) -> tuple[CSRTensor, CSRTensor]:
        """应用边界条件"""
        boundary_type = self.pde.boundary_type
        gdof = self.tensor_space.number_of_global_dofs()

        if boundary_type == 'dirichlet':
            uh_bd = bm.zeros(gdof, dtype=bm.float64, device=self.tensor_space.device)
            uh_bd, isBdDof = self.tensor_space.boundary_interpolate(
                                    gd=self.pde.dirichlet, 
                                    threshold=self.pde.threshold(), 
                                    method='interp'
                                )
            F = F - K.matmul(uh_bd[:])
            F[isBdDof] = uh_bd[isBdDof]

            K = self._apply_matrix(A=K, isDDof=isBdDof)

        elif boundary_type == 'neumann':
            pass

        else:
            raise ValueError(f"Unsupported boundary type: {boundary_type}")

    @variantmethod('mumps')
    def solve(self, **kwargs) -> TensorLike:
        from fealpy.solver import spsolve
        K0 = self.assemble_stiff_matrix()
        F0 = self.assemble_force_vector()
        K, F = self.apply_bc(K0, F0)

        solver_type = kwargs.get('solver', 'mumps')
        uh = self.tensor_space.function()
        uh[:] = spsolve(K, F[:], solver=solver_type)

        return uh
    
    @solve.register('cg')
    def solve(self, **kwargs) -> TensorLike:
        from fealpy.solver import cg
        K0 = self.assemble_stiff_matrix()
        F0 = self.assemble_force_vector()
        K, F = self.apply_bc(K0, F0)

        maxiter = kwargs.get('maxiter', 5000)
        atol = kwargs.get('atol', 1e-12)
        rtol = kwargs.get('rtol', 1e-12)
        x0 = kwargs.get('x0', None)
        uh = self.tensor_space.function()
        uh[:], info = cg(K, F[:], x0=x0,
                        batch_first=True, 
                        atol=atol, rtol=rtol, 
                        maxit=maxiter, returninfo=True)

        return uh


