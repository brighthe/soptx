from typing import Optional, Union, Literal, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace, Function, HuZhangFESpace
from fealpy.fem import BlockForm, BilinearForm, LinearForm, VectorSourceIntegrator
from fealpy.decorator.variantmethod import variantmethod
from fealpy.sparse import CSRTensor, COOTensor

from soptx.analysis.integrators.huzhang_stress_integrator import HuZhangStressIntegrator
from soptx.analysis.integrators.huzhang_mix_integrator import HuZhangMixIntegrator
from soptx.model.pde_base import PDEBase
from soptx.material.linear_elastic_material import LinearElasticMaterial
from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
from soptx.utils.base_logged import BaseLogged

class HuZhangMFEMAnalyzer(BaseLogged):
    def __init__(self,
                mesh: HomogeneousMesh,
                pde: PDEBase, 
                material: LinearElasticMaterial,
                space_degree: int = 1,
                integration_order: int = 4,
                assembly_method: Literal['standard', 'fast'] = 'standard',
                solve_method: Literal['mumps', 'cg'] = 'mumps',
                topopt_algorithm: Literal[None, 'density_based', 'level_set'] = None,
                interpolation_scheme: Optional[MaterialInterpolationScheme] = None,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        """初始化拉格朗日有限元分析器"""

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        # 验证拓扑优化算法与插值方案的匹配性
        self._validate_topopt_config(topopt_algorithm, interpolation_scheme)

        
        # 私有属性（建议通过属性访问器访问，不要直接修改）
        self._mesh = mesh
        self._pde = pde
        self._material = material
        self._interpolation_scheme = interpolation_scheme

        self._space_degree = space_degree
        self._integration_order = integration_order
        self._assembly_method = assembly_method

        self._topopt_algorithm = topopt_algorithm

        # 设置默认求解方法
        self.solve_displacement.set(solve_method)

        self._GD = self._mesh.geo_dimension()
        self._setup_function_spaces(mesh=self._mesh, 
                                    p=self._space_degree, 
                                    shape=(self._GD, -1))

        # 缓存的矩阵和向量
        self._K = None
        self._F = None

        self._log_info(f"Mesh Information: NC: {self._mesh.number_of_cells()}, ")

    def assemble_stiff_matrix(self, 
                        density_distribution: Optional[Function] = None
                    ) -> Union[CSRTensor, COOTensor]:
        
        if self._topopt_algorithm is None:
            if density_distribution is not None:
                self._log_warning("标准有限元分析模式下忽略相对密度分布参数 rho")
            
            space0 = self._huzhang_space
            space1 = self._tensor_space

            lambda0, lambda1 = self._pde.stress_matrix_coefficient()

            bform1 = BilinearForm(space0)
            bform1.add_integrator(HuZhangStressIntegrator(lambda0=lambda0, lambda1=lambda1))

            bform2 = BilinearForm((space1, space0))
            bform2.add_integrator(HuZhangMixIntegrator())

            bform = BlockForm([[bform1,   bform2],
                               [bform2.T, None]])
            
            K = bform.assembly(format='csr')

        return K

    def assemble_force_vector(self) -> Union[TensorLike, COOTensor]:
        """组装全局载荷向量"""
        body_force = self._pde.body_force
        force_type = self._pde.force_type

        space0 = self._huzhang_space
        space1 = self._tensor_space

        gdof0 = space0.number_of_global_dofs()
        gdof1 = space1.number_of_global_dofs()

        if force_type == 'concentrated':
            # NOTE F.dtype == TensorLike
            F_u = self._tensor_space.interpolate(body_force)
        
        elif force_type == 'distribution':
            # NOTE F.dtype == COOTensor or TensorLike
            integrator = VectorSourceIntegrator(source=body_force, q=self._integration_order)
            lform = LinearForm(space1)
            lform.add_integrator(integrator)
            F_u = lform.assembly(format='dense')
        
        else:
            error_msg = f"Unsupported force type: {force_type}"
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        F = bm.zeros(gdof0 + gdof1, dtype=F_u.dtype)
        F[gdof0:] = -F_u
        
        return F
    
    def apply_bc(self, K: Union[CSRTensor, COOTensor], F: CSRTensor) -> tuple[CSRTensor, CSRTensor]:
        """应用边界条件"""
        boundary_type = self._pde.boundary_type
        gdof = self._tensor_space.number_of_global_dofs()

        gd = self._pde.dirichlet_bc
        threshold = self._pde.is_dirichlet_boundary()

        if boundary_type == 'dirichlet':
            # uh_bd = bm.zeros(gdof, dtype=bm.float64, device=self._tensor_space.device)
            # uh_bd, isBdDof = self._tensor_space.boundary_interpolate(
            #                         gd=gd,
            #                         threshold=threshold,
            #                         method='interp'
            #                     )
            # F = F - K.matmul(uh_bd[:])
            # F[isBdDof] = uh_bd[isBdDof]

            # K = self._apply_matrix(A=K, isDDof=isBdDof)

            return K, F

        elif boundary_type == 'neumann':
            pass

        else:
            error_msg = f"Unsupported boundary type: {boundary_type}"
            self._log_error(error_msg)
            raise ValueError(error_msg)
    

    ##########################################################################################################
    # 变体方法
    ##########################################################################################################

    @variantmethod('mumps')
    def solve_displacement(self, 
            density_distribution: Optional[Function] = None, 
            **kwargs
            ) -> Tuple[Function, Function]:
        from fealpy.solver import spsolve

        if self._topopt_algorithm is None:
            if density_distribution is not None:
                self._log_warning("标准有限元分析模式下忽略密度分布参数 rho")
        
        elif self._topopt_algorithm in ['density_based', 'level_set']:
            if density_distribution is None:
                error_msg = f"拓扑优化算法 '{self._topopt_algorithm}' 需要提供密度分布参数 rho"
                self._log_error(error_msg)
                raise ValueError(error_msg)
    
        K0 = self.assemble_stiff_matrix(density_distribution=density_distribution)
        F0 = self.assemble_force_vector()
        K, F = self.apply_bc(K0, F0)

        solver_type = kwargs.get('solver', 'mumps')

        X = spsolve(K, F, solver=solver_type)

        space0 = self._huzhang_space
        space1 = self._tensor_space
        gdof0 = space0.number_of_global_dofs()

        sigmaval = X[:gdof0]
        uval = X[gdof0:]

        sigmah = space0.function()
        sigmah[:] = sigmaval

        uh = space1.function()
        uh[:] = uval

        return sigmah, uh
    

    ##############################################################################################
    # 内部方法
    ##############################################################################################

    def _validate_topopt_config(self, 
                            topopt_algorithm: Literal[None, 'density_based', 'level_set'], 
                            interpolation_scheme: Optional[MaterialInterpolationScheme]
                        ) -> None:
        """验证拓扑优化算法与插值方案的匹配性"""
        
        if topopt_algorithm is None:

            if interpolation_scheme is not None:
                error_msg = ("当 topopt_algorithm=None 时, interpolation_scheme 必须为 None."
                        "标准有限元分析不需要插值方案.")
                self._log_error(error_msg)
                raise ValueError(error_msg)
            
            self._log_info("使用标准有限元分析模式（无拓扑优化）")
                
        elif topopt_algorithm == 'density_based':

            if interpolation_scheme is None:
                error_msg = "当 topopt_algorithm='density_based' 时，必须提供 MaterialInterpolationScheme"
                self._log_error(error_msg)
                raise ValueError(error_msg)
            
            self._log_info(f"使用基于密度的拓扑优化, 插值方法：{interpolation_scheme.interpolation_method}")
                
        elif topopt_algorithm == 'level_set':
            
            raise NotImplementedError("Level set topology optimization is not yet implemented.")
                
        else:
            error_msg = f"不支持的拓扑优化算法: {topopt_algorithm}"
            self._log_error(error_msg)
            raise ValueError(error_msg)
    
    def _setup_function_spaces(self, 
                            mesh: HomogeneousMesh, 
                            p: int, 
                            shape : tuple[int, int]
                        ) -> None:
        """设置函数空间"""
        huzhang_space = HuZhangFESpace(mesh, p=p)
        self._huzhang_space = huzhang_space

        scalar_space = LagrangeFESpace(mesh, p=p-1, ctype='D')
        self._scalar_space = scalar_space

        tensor_space = TensorFunctionSpace(scalar_space=scalar_space, shape=shape)
        self._tensor_space = tensor_space

        self._log_info(f"Tensor space DOF ordering: dof_priority")


if __name__ == "__main__":

    from soptx.model.linear_elasticity_2d import BoxTriHuZhangData2d
    pde = BoxTriHuZhangData2d(lam=1, mu=0.5)
    # TODO 支持四边形网格
    pde.init_mesh.set('uniform_tri')
    analysis_mesh = pde.init_mesh(nx=2, ny=2)
    # TODO 支持 3 次以下
    space_degree = 3

    # from soptx.model.linear_elasticity_3d import BoxPolyHuZhangData3d
    # pde = BoxPolyHuZhangData3d(lam=1, mu=0.5)
    # # TODO 支持六面体网格
    # pde.init_mesh.set('uniform_tet')
    # analysis_mesh = pde.init_mesh(nx=2, ny=2, nz=2)
    # # TODO 支持 3 次以下
    # space_degree = 4

    integration_order = space_degree+3

    from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
    material = IsotropicLinearElasticMaterial(
                                        lame_lambda=pde.lam, 
                                        shear_modulus=pde.mu, 
                                        plane_type=pde.plane_type,
                                        enable_logging=False
                                    )
    maxit = 5
    errorType = [
                '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega,0}$',
                '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{\\Omega,0}$',
                ]
    errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
    NDof = bm.zeros(maxit, dtype=bm.int32)
    h = bm.zeros(maxit, dtype=bm.float64)

    for i in range(maxit):
        N = 2**(i+1) 

        huzhang_mfem_analyzer = HuZhangMFEMAnalyzer(
                            mesh=analysis_mesh,
                            pde=pde,
                            material=material,
                            space_degree=space_degree,
                            integration_order=integration_order,
                            assembly_method='standard',
                            solve_method='mumps',
                            topopt_algorithm=None,
                            interpolation_scheme=None,
                        )
        
        sigmah, uh = huzhang_mfem_analyzer.solve_displacement(density_distribution=None)
        
        e0 = analysis_mesh.error(uh, pde.disp_solution) 
        e1 = analysis_mesh.error(sigmah, pde.stress_solution)

        h[i] = 1/N
        errorMatrix[0, i] = e0
        errorMatrix[1, i] = e1 

        uh_dof = huzhang_mfem_analyzer._tensor_space.number_of_global_dofs()
        sigma_dof = huzhang_mfem_analyzer._huzhang_space.number_of_global_dofs()
        NDof[i] = uh_dof + sigma_dof

        analysis_mesh.uniform_refine()

    import matplotlib.pyplot as plt
    from soptx.utils.show import showmultirate, show_error_table

    show_error_table(h, errorType, errorMatrix)
    showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
    plt.show()
    print('------------------')