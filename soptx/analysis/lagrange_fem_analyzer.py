from typing import Optional, Union, Literal

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import SimplexMesh, HomogeneousMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace, Function
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import VectorSourceIntegrator
from fealpy.decorator.variantmethod import variantmethod
from fealpy.sparse import CSRTensor, COOTensor

from ..interpolation.linear_elastic_material import LinearElasticMaterial
from .integrators.linear_elastic_integrator import LinearElasticIntegrator
from ..pde.pde_base import PDEBase
from ..utils.base_logged import BaseLogged


class LagrangeFEMAnalyzer(BaseLogged):
    def __init__(self,
                mesh: HomogeneousMesh,
                pde: PDEBase, 
                material: LinearElasticMaterial,
                space_degree: int = 1,
                assembly_method: Literal['standard', 'fast'] = 'standard',
                solve_method: Literal['mumps', 'cg'] = 'mumps',
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        """
        1. 求解变体方法示例
        lfa = LagrangeFEMAnalyzer(pde=pde, material=material, solve_method='mumps')

        2. 直接使用默认方法求解位移
        uh = lfa.solve()
        
        3. 切换到其他求解方法
        lfa.solve.set('cg')     # 设置变体 (返回 None)
        uh = lfa.solve(maxiter=5000, atol=1e-12, rtol=1e-12)
        
        注意: 
        - solve.set() 只设置变体，不执行方法，返回 None
        - 需要分别调用 set() 和 solve() 来求解位移
        - 每次 set() 后，后续的 solve() 调用都使用新设置的变体
        """
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        # 私有属性 (不建议外部直接访问)
        self._mesh = mesh
        self._pde = pde
        self._material = material
        self._space_degree = space_degree
        self._assembly_method = assembly_method
        
        self.solve.set(solve_method)

        self._setup_function_spaces()

        self._K = None
        self._F = None


    ##############################################################################################
    # 访问器
    ##############################################################################################
    @property
    def mesh(self) -> SimplexMesh:
        """获取当前的网格对象"""
        return self._mesh
    
    @property
    def scalar_space(self) -> LagrangeFESpace:
        """获取当前的标量函数空间"""
        return self._scalar_space
    
    @property
    def tensor_space(self) -> TensorFunctionSpace:
        """获取当前的张量函数空间"""
        return self._tensor_space
    
    @property
    def material(self) -> LinearElasticMaterial:
        """获取当前的材料类"""
        return self._material
    
    @property
    def assembly_method(self) -> str:
        """获取当前的组装方法"""
        return self._assembly_method

    @property
    def stiffness_matrix(self) -> Union[CSRTensor, COOTensor]:
        """获取当前的刚度矩阵"""
        if self._K is None:
            self._K = self.assemble_stiff_matrix()

        return self._K
    
    @property
    def force_vector(self) -> Union[TensorLike, COOTensor]:
        """获取当前的载荷向量"""
        if self._F is None:
            self._F = self.assemble_force_vector()

        return self._F
    

    ######################################################################################################
    # 辅助方法
    ######################################################################################################

    def get_stiffness_matrix__derivative(self) -> TensorLike:
        """获取局部刚度矩阵的梯度"""
        
        top_material = self._material
        base_material = top_material.base_material
        density_location = top_material.density_location
        interpolation_scheme = top_material.interpolation_scheme
        density_distribution = top_material.density_distribution

        interpolate_derivative = interpolation_scheme.interpolate_derivative(
                                                            base_material=base_material, 
                                                            density_distribution=density_distribution[:]
                                                        )
        if density_location == 'element':
            LEA = LinearElasticIntegrator(material=base_material, 
                                        q=top_material.quadrature_order,
                                        method=self._assembly_method)
            ke0 = LEA.assembly(space=self.tensor_space)
            diff_ke = bm.einsum('c, cij -> cij', interpolate_derivative, ke0)

        elif density_location == 'element_gauss_integrate_point':
            mesh = self._mesh
            qf = mesh.quadrature_formula(top_material.quadrature_order)
            bcs, ws = qf.get_quadrature_points_and_weights()

            D0 = base_material.elastic_matrix()[0, 0]
            dof_priority = self.tensor_space.dof_priority
            scalar_space = self.tensor_space.scalar_space
            gphi = scalar_space.grad_basis(bcs, variable='x')
            B = base_material.strain_displacement_matrix(dof_priority=dof_priority, gphi=gphi)

            if isinstance(mesh, SimplexMesh):
                cm = mesh.entity_measure('cell')
                diff_ke = bm.einsum('q, c, cq, cqki, kl, cqlj -> cqij', ws, cm, interpolate_derivative, B, D0, B)

            else:
                J = mesh.jacobi_matrix(bcs)
                detJ = bm.linalg.det(J)
                diff_ke = bm.einsum('q, cq, cq, cqki, kl, cqlj -> cqij', ws, detJ, interpolate_derivative, B, D0, B)

        return diff_ke

    
    ##############################################################################################
    # 核心方法
    ##############################################################################################

    def assemble_stiff_matrix(self) -> Union[CSRTensor, COOTensor]:
        """组装刚度矩阵"""
        integrator = LinearElasticIntegrator(material=self._material, 
                                            q=self._space_degree+3,
                                            method=self._assembly_method)
        bform = BilinearForm(self._tensor_space)
        bform.add_integrator(integrator)
        K = bform.assembly(format='csr')

        self._K = K

        return K

    def assemble_force_vector(self) -> Union[TensorLike, COOTensor]:
        """组装载荷向量"""
        body_force = self._pde.body_force
        force_type = self._pde.force_type

        if force_type == 'concentrated':
            # NOTE F.dtype == TensorLike
            F = self._tensor_space.interpolate(body_force)
        elif force_type == 'continuous':
            # NOTE F.dtype == COOTensor or TensorLike
            integrator = VectorSourceIntegrator(source=body_force, q=self._space_degree+3)
            lform = LinearForm(self.tensor_space)
            lform.add_integrator(integrator)
            F = lform.assembly(format='dense')
        else:
            error_msg = f"Unsupported force type: {force_type}"
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        self._F = F

        return F

    def apply_bc(self, K: Union[CSRTensor, COOTensor], F: CSRTensor) -> tuple[CSRTensor, CSRTensor]:
        """应用边界条件"""
        boundary_type = self._pde.boundary_type
        gdof = self._tensor_space.number_of_global_dofs()

        gd = self._pde.dirichlet_bc
        threshold = self._pde.is_dirichlet_boundary()

        if boundary_type == 'dirichlet':
            uh_bd = bm.zeros(gdof, dtype=bm.float64, device=self._tensor_space.device)
            uh_bd, isBdDof = self._tensor_space.boundary_interpolate(
                                    gd=gd,
                                    threshold=threshold,
                                    method='interp'
                                )
            F = F - K.matmul(uh_bd[:])
            F[isBdDof] = uh_bd[isBdDof]

            K = self._apply_matrix(A=K, isDDof=isBdDof)

            return K, F

        elif boundary_type == 'neumann':
            pass

        else:
            error_msg = f"Unsupported boundary type: {boundary_type}"
            self._log_error(error_msg)
            raise ValueError(error_msg)

    @variantmethod('mumps')
    def solve(self, **kwargs) -> Function:
        from fealpy.solver import spsolve
        K0 = self.assemble_stiff_matrix()
        F0 = self.assemble_force_vector()
        K, F = self.apply_bc(K0, F0)

        solver_type = kwargs.get('solver', 'mumps')
        uh = self._tensor_space.function()
        uh[:] = spsolve(K, F[:], solver=solver_type)

        gdof = self._tensor_space.number_of_global_dofs()
        self._log_info(f"Solving linear system with {gdof} displacement DOFs with MUMPS solver.")

        return uh
    
    @solve.register('cg')
    def solve(self, **kwargs) -> Function:
        from fealpy.solver import cg
        K0 = self.assemble_stiff_matrix()
        F0 = self.assemble_force_vector()
        K, F = self.apply_bc(K0, F0)

        maxiter = kwargs.get('maxiter', 5000)
        atol = kwargs.get('atol', 1e-12)
        rtol = kwargs.get('rtol', 1e-12)
        x0 = kwargs.get('x0', None)
        uh = self._tensor_space.function()
        uh[:], info = cg(K, F[:], x0=x0,
                        batch_first=True, 
                        atol=atol, rtol=rtol, 
                        maxit=maxiter, returninfo=True)
        
        gdof = self._tensor_space.number_of_global_dofs()
        self._log_info(f"Solving linear system with {gdof} displacement DOFs with CG solver.")

        return uh

    
    ##############################################################################################
    # 内部方法
    ##############################################################################################

    def _setup_function_spaces(self):
        """设置函数空间"""
        self._scalar_space = LagrangeFESpace(self._mesh, p=self._space_degree, ctype='C')
        GD = self._mesh.geo_dimension()
        self._tensor_space = TensorFunctionSpace(scalar_space=self._scalar_space, shape=(GD, -1))

        self._log_info(f"Tensor space DOF ordering: dof_priority")
    
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

