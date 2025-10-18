from typing import Optional, Union, Literal, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace, Function
from fealpy.fem import BlockForm, BilinearForm, LinearForm
from fealpy.decorator.variantmethod import variantmethod
from fealpy.sparse import CSRTensor, COOTensor

from soptx.functionspace.huzhang_fe_space import HuZhangFESpace
from soptx.analysis.integrators.huzhang_stress_integrator import HuZhangStressIntegrator
from soptx.analysis.integrators.huzhang_mix_integrator import HuZhangMixIntegrator
from soptx.analysis.integrators.jump_penalty_integrator import JumpPenaltyIntegrator
from soptx.analysis.integrators.source_integrator import SourceIntegrator
from soptx.model.pde_base import PDEBase
from soptx.interpolation.linear_elastic_material import LinearElasticMaterial
from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
from soptx.utils.base_logged import BaseLogged
from soptx.utils import timer

class HuZhangMFEMAnalyzer(BaseLogged):
    def __init__(self,
                mesh: HomogeneousMesh,
                pde: PDEBase, 
                material: LinearElasticMaterial,
                space_degree: int = 1,
                integration_order: int = 4,
                solve_method: Literal['mumps', 'cg'] = 'mumps',
                topopt_algorithm: Literal[None, 'density_based', 'level_set'] = None,
                interpolation_scheme: Optional[MaterialInterpolationScheme] = None,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        """初始化胡张混合有限元分析器"""

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

        self._topopt_algorithm = topopt_algorithm
        self._interpolation_scheme = interpolation_scheme

        # 设置默认求解方法
        self.solve_displacement.set(solve_method)

        self._GD = self._mesh.geo_dimension()
        self._setup_function_spaces(mesh=self._mesh, 
                                    p=self._space_degree, 
                                    shape=(-1, self._GD))

        # 缓存的矩阵和向量
        self._K = None
        self._F = None

        self._log_info(f"Mesh Information: NC: {self._mesh.number_of_cells()}, ")

    
    ##############################################################################################
    # 属性访问器 - 获取内部状态
    ##############################################################################################
    
    @property
    def mesh(self) -> HomogeneousMesh:
        """获取当前的网格对象"""
        return self._mesh
    
    @property
    def huzhang_space(self) -> HuZhangFESpace:
        """获取当前的 Hu-Zhang 混合有限元空间"""
        return self._huzhang_space
    
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
    def topopt_algorithm(self) -> Optional[str]:
        """获取当前的拓扑优化算法"""
        return self._topopt_algorithm

    @property
    def stiffness_matrix(self) -> Union[CSRTensor, COOTensor]:
        """刚度矩阵"""
        return self._K
    
    @property
    def load_vector(self) -> Union[TensorLike, COOTensor]:
        """载荷向量"""
        return self._F
    

    ##############################################################################################
    # 属性修改器 - 修改内部状态
    ##############################################################################################

    @huzhang_space.setter
    def huzhang_space(self, space: HuZhangFESpace) -> None:
        """设置 Hu-Zhang 混合有限元空间"""
        self._huzhang_space = space

    @scalar_space.setter
    def scalar_space(self, space: LagrangeFESpace) -> None:
        """设置标量函数空间"""
        self._scalar_space = space

    @tensor_space.setter
    def tensor_space(self, space: TensorFunctionSpace) -> None:
        """设置张量函数空间"""
        self._tensor_space = space

    
    ##################################################################################################
    # 核心方法
    ##################################################################################################

    def assemble_stiff_matrix(self, 
                        rho_val: Optional[Union[Function, TensorLike]] = None,
                        enable_timing: bool = False,
                    ) -> Union[CSRTensor, COOTensor]:
        """组装全局刚度矩阵

        Parameters
        ----------
        rho_val : 密度值
            - 单元密度 
                - 单分辨率 - (NC, )
                - 多分辨率 - (NC, n_sub)
            - 节点密度 
                - 单分辨率 - (NC, NQ)
                - 多分辨率 - (NC, n_sub, NQ)

        Returns
        -------
        Union[CSRTensor, COOTensor]
            组装后的刚度矩阵
        """
                
        t = None
        if enable_timing:
            t = timer(f"组装刚度矩阵")

        p = self._space_degree
        mesh = self._mesh
        GD = mesh.geo_dimension()

        space_sigma = self._huzhang_space
        space_u = self._tensor_space

        if self._topopt_algorithm is None:
            if rho_val is not None:
                self._log_warning("标准混合有限元分析模式下忽略相对密度分布参数 rho")
            coef = None
        elif self._topopt_algorithm in ['density_based']:
            E_rho = self._interpolation_scheme.interpolate_map(
                                            material=self._material,
                                            rho_val=rho_val,
                                            integration_order=self._integration_order
                                        )
            E0 = self.material.youngs_modulus
            coef = E0 / E_rho
        else:
            error_msg = f"不支持的拓扑优化算法: {self._topopt_algorithm}"
            self._log_error(error_msg)

        lambda0, lambda1 = self._stress_matrix_coefficient()

        bform1 = BilinearForm(space_sigma)
        hzs_integrator = HuZhangStressIntegrator(lambda0=lambda0, lambda1=lambda1, 
                                                q=self._integration_order, coef=coef)
        bform1.add_integrator(hzs_integrator)

        bform2 = BilinearForm((space_u, space_sigma)) # 应力-位移矩阵 B_σu
        hzm_integrator = HuZhangMixIntegrator()
        bform2.add_integrator(hzm_integrator)

        if p >= GD + 1:
            bform = BlockForm([[bform1,   bform2],
                               [bform2.T, None]])

        elif p <= GD:
            bform3 = BilinearForm(space_u)
            jpi_integrator = JumpPenaltyIntegrator(q=self._integration_order, method='vector_jump')
            # jpi_integrator = JumpPenaltyIntegrator(q=self._integration_order, method='matrix_jump')
            bform3.add_integrator(jpi_integrator)

            bform = BlockForm([[bform1,   bform2],
                               [bform2.T, bform3]])

        if enable_timing:
            t.send('准备时间')

        K = bform.assembly(format='csr')

        if enable_timing:
            t.send('组装时间')
            t.send(None)

        return K
    
    def get_stress_matrix(self, 
                      rho_val: Optional[Union[Function, TensorLike]] = None
                     ) -> Union[CSRTensor, COOTensor]:
        """获取应力矩阵 A_σσ """
        
        space0 = self._huzhang_space
        
        if self._topopt_algorithm is None:
            coef = None
        elif self._topopt_algorithm in ['density_based']:
            # 目前仅支持插值杨氏模量 E 
            E_rho = self._interpolation_scheme.interpolate_map(
                                            material=self._material,
                                            rho_val=rho_val,
                                            integration_order=self._integration_order
                                        )
            E0 = self._material.youngs_modulus
            coef = E0 / E_rho
        else:
            error_msg = f"不支持的拓扑优化算法: {self._topopt_algorithm}"
            self._log_error(error_msg)
        
        lambda0, lambda1 = self._stress_matrix_coefficient()
        
        bform = BilinearForm(space0)
        hzs_integrator = HuZhangStressIntegrator(lambda0=lambda0, lambda1=lambda1, 
                                                q=self._integration_order, coef=coef)
        bform.add_integrator(hzs_integrator)
        
        stress_matrix = bform.assembly(format='csr')

        return stress_matrix
    
    def get_mix_matrix(self) -> Union[CSRTensor, COOTensor]:
        """获取应力-位移耦合矩阵 B_σu"""
        
        space_sigma = self._huzhang_space
        space_u = self._tensor_space

        bform = BilinearForm((space_u, space_sigma)) # 应力-位移矩阵 B_σu
        hzm_integrator = HuZhangMixIntegrator()
        bform.add_integrator(hzm_integrator)

        B_sigma_u = bform.assembly(format='csr') # (GDOF_sigma, GDOF_u)

        return B_sigma_u

    def assemble_displacement_load_vector(self,
                                        enable_timing: bool = False,
                                    ) -> Union[TensorLike, COOTensor]:
        """组装载荷向量的位移分量 f_u"""
        t = None
        if enable_timing:
            t = timer(f"组装 f_u")

        body_force = self._pde.body_force
        space_uh = self._tensor_space

        # NOTE F.dtype == COOTensor or TensorLike
        integrator = SourceIntegrator(source=body_force, q=self._integration_order)
        lform = LinearForm(space_uh)
        lform.add_integrator(integrator)
        F_v = lform.assembly(format='dense')

        load_type = self._pde.load_type

        if load_type == 'concentrated':
            # Neumann 边界条件处理
            gd_sigmah = self._pde.neumann_bc
            threshold_sigmah = self._pde.is_neumann_boundary()
            # 集中载荷 (点力) - 等效节点力方法 - 弱形式施加
            #! 点力必须定义在网格节点上
            isBdTDof = space_uh.is_boundary_dof(threshold=threshold_sigmah, method='interp')
            isBdSDof = space_uh.scalar_space.is_boundary_dof(threshold=threshold_sigmah, method='interp')
            ipoints_uh = space_uh.interpolation_points()
            gd_sigmah_val = gd_sigmah(ipoints_uh[isBdSDof])

            F_uh = space_uh.function()
            if space_uh.dof_priority:
                F_uh[:] = bm.set_at(F_uh[:], isBdTDof, gd_sigmah_val.T.reshape(-1))
            else:
                F_uh[:] = bm.set_at(F_uh[:], isBdTDof, gd_sigmah_val.reshape(-1))

            F_u = -F_v + F_uh

        else:
            F_u = -F_v

        if enable_timing:
            t.send('组装时间')
            t.send(None)

        return F_u

    def assemble_stress_load_vector(self, 
                                enable_timing: bool = False
                            ) -> Union[TensorLike, COOTensor]:
        """组装载荷向量的应力分量 f_sigma"""
        t = None
        if enable_timing:
            t = timer(f"组装 f_sigma")

        space_sigmah = self._huzhang_space
        boundary_type = self._pde.boundary_type

        if boundary_type == 'neumann':
            F_sigma = space_sigmah.function()
        
        else:
            # Dirichlet 边界条件处理 - 弱形式施加
            gd_uh = self._pde.dirichlet_bc
            threshold_uh = self._pde.is_dirichlet_boundary()
            
            from soptx.analysis.integrators.face_source_integrator_mfem import BoundaryFaceSourceIntegrator_mfem
            integrator_uh = BoundaryFaceSourceIntegrator_mfem(source=gd_uh, 
                                                            q=self._integration_order, 
                                                            threshold=threshold_uh,
                                                            method='dirichlet')
            lform_sigmah = LinearForm(space_sigmah)
            lform_sigmah.add_integrator(integrator_uh)
            F_sigma = lform_sigmah.assembly(format='dense')

        if enable_timing:
            t.send('组装时间')
            t.send(None)

        return F_sigma

    def apply_neumann_bc(self, K: Union[CSRTensor, COOTensor], F: CSRTensor) -> tuple[CSRTensor, CSRTensor]:
        """应用 Neumann 边界条件"""
        boundary_type = self._pde.boundary_type
        load_type = self._pde.load_type

        space_sigmah = self._huzhang_space
        space_uh = self._tensor_space

        gdof_sigmah = space_sigmah.number_of_global_dofs()
        gdof_uh = space_uh.number_of_global_dofs()
        
        gd_sigmah = self._pde.neumann_bc
        threshold_sigmah = self._pde.is_neumann_boundary()

        # Neumann 边界条件处理
        # 分布载荷 (面力) - 强形式施加
        if load_type == 'distributed':
            A = K[:gdof_sigmah, :gdof_sigmah]

            isBdDof_sigmah = space_sigmah.is_boundary_dof(threshold=threshold_sigmah, method='barycenter')
            ipoints = space_sigmah.interpolation_points()
            gd_sigmah_val = gd_sigmah(ipoints)

            sigmah_bd = bm.zeros(gdof_sigmah, dtype=bm.float64, device=space_sigmah.device)
            sigmah_bd[:] = bm.set_at(sigmah_bd[:], isBdDof_sigmah, gd_sigmah_val.reshape(-1)[isBdDof_sigmah])

            load_bd = bm.zeros(gdof_sigmah + gdof_uh, dtype=bm.float64, device=space_sigmah.device)
            load_bd[:gdof_sigmah] = sigmah_bd

            F = F - K.matmul(load_bd[:])
            F[:gdof_sigmah][isBdDof_sigmah] = sigmah_bd[isBdDof_sigmah]

            isBdDof = bm.zeros(gdof_sigmah + gdof_uh, dtype=bm.bool, device=space_sigmah.device)
            isBdDof[:gdof_sigmah] = isBdDof_sigmah 
            
            K = self._apply_matrix(A=A, isDDof=isBdDof)
            print("---------------------")

        # if boundary_type == 'mixed':
        #     # 1. Dirichlet 边界条件处理 - 弱形式施加
        #     gd_uh = self._pde.dirichlet_bc
        #     threshold_uh = self._pde.is_dirichlet_boundary()
        #     integrator_uh = BoundaryFaceSourceIntegrator_mfem(source=gd_uh, 
        #                                                     q=self._integration_order, 
        #                                                     threshold=threshold_uh,
        #                                                     method='dirichlet')
        #     lform_sigmah = LinearForm(space_sigmah)
        #     lform_sigmah.add_integrator(integrator_uh)
        #     F_sigmah = lform_sigmah.assembly(format='dense')

        #     F[:gdof_sigmah] += F_sigmah

        #     # 2. Neumann 边界条件处理 - 强形式施加
        #     gd_sigmah = self._pde.neumann_bc
        #     threshold_sigmah = self._pde.is_neumann_boundary()
        #     # if load_type == 'concentrated':
        #     #     # 集中载荷 (点力) - 等效节点力方法 - 转换为弱形式施加
        #     #     #! 点力必须定义在网格节点上
        #     #     isBdTDof = space_uh.is_boundary_dof(threshold=threshold_sigmah, method='interp')
        #     #     isBdSDof = space_uh.scalar_space.is_boundary_dof(threshold=threshold_sigmah, method='interp')
        #     #     ipoints_uh = space_uh.interpolation_points()
        #     #     gd_sigmah_val = gd_sigmah(ipoints_uh[isBdSDof])

        #     #     F_uh = space_uh.function()
        #     #     if space_uh.dof_priority:
        #     #         F_uh[:] = bm.set_at(F_uh[:], isBdTDof, gd_sigmah_val.T.reshape(-1))
        #     #     else:
        #     #         F_uh[:] = bm.set_at(F_uh[:], isBdTDof, gd_sigmah_val.reshape(-1))

        #     #     F[gdof_sigmah:] += F_uh
        #     #     self._F = F

        #     if load_type == 'distributed':
        #         # 分布载荷 (面力) - 强形式施加
        #         p = space_sigmah.p
        #         isBdDof_sigmah = space_sigmah.is_boundary_dof(threshold=threshold_sigmah, method='barycenter')
        #         # if p == 1:
        #         #     #TODO 检查正确性
        #         #     integrator_sigmah = BoundaryFaceSourceIntegrator_mfem(source=gd_sigmah, 
        #         #                                                         q=self._integration_order, 
        #         #                                                         threshold=threshold_sigmah,
        #         #                                                         method='neumann')
        #         #     lform_uh = LinearForm(space_uh)
        #         #     lform_uh.add_integrator(integrator_sigmah)
        #         #     F_u = lform_uh.assembly(format='dense')

        #         #     F[gdof_sigmah:] += F_u       
        #         #     self._F = F

        #         # elif p >= 2:
        #         #     #TODO 如何实现
        #         #     self._F = F
        #         #     isBdEDof_sigmah = space_sigmah.is_boundary_edge_dof(threshold=threshold_sigmah, method='barycenter')

        #         #     # ? 边界上的精确解 gd_sigmah_val 如何计算 ?
        #         #     ipoints = space_sigmah.interpolation_points()
        #         #     gd_sigmah_val = gd_sigmah(ipoints)

        #         #     sigmah_bde = bm.zeros(gdof_sigmah, dtype=bm.float64, device=space_sigmah.device)
        #         #     sigmah_bde[:] = bm.set_at(sigmah_bde[:], isBdEDof_sigmah, gd_sigmah_val.reshape(-1)[isBdEDof_sigmah])
    
        #         #     source_bd = bm.zeros(gdof_sigmah + gdof_uh, dtype=bm.float64, device=space_sigmah.device)
        #         #     source_bd[:gdof_sigmah] = sigmah_bde

        #         #     F = F - K.matmul(source_bd[:])
        #         #     F[:gdof_sigmah][isBdEDof_sigmah] = sigmah_bde[isBdEDof_sigmah]

        #         #     isBdDof = bm.zeros(gdof_sigmah + gdof_uh, dtype=bm.bool, device=space_sigmah.device)
        #         #     isBdDof[:gdof_sigmah] = isBdEDof_sigmah 

        #             # K = self._apply_matrix(A=K, isDDof=isBdDof)

        #     else:
        #         raise NotImplementedError(f"不支持的载荷类型: {load_type}")

        #     return K, F

        # elif boundary_type == 'dirichlet':
        #     # 弱形式施加
        #     gd_uh = self._pde.dirichlet_bc
        #     threshold_uh = self._pde.is_dirichlet_boundary()
        #     integrator = BoundaryFaceSourceIntegrator_mfem(source=gd_uh, 
        #                                             q=self._integration_order, 
        #                                             threshold=threshold_uh,
        #                                             method='dirichlet')
        #     lform = LinearForm(space_sigmah)
        #     lform.add_integrator(integrator)
        #     F_sigma = lform.assembly(format='dense')

        #     F[:gdof_sigmah] += F_sigma

        #     self._F = F

        #     return K, F

        # elif boundary_type == 'neumann':
        #     pass

        else:
            error_msg = f"Unsupported boundary type: {boundary_type}"
            self._log_error(error_msg)
    

    ##########################################################################################################
    # 变体方法
    ##########################################################################################################

    @variantmethod('mumps')
    def solve_displacement(self, 
                        rho_val: Optional[Union[TensorLike, Function]] = None, 
                        enable_timing: bool = False, 
                        **kwargs
                    ) -> Tuple[Function, Function]:
        
        t = None
        if enable_timing:
            t = timer(f"分析阶段时间")
            next(t)
        
        from fealpy.solver import spsolve

        if self._topopt_algorithm is None:
            if rho_val is not None:
                self._log_warning("标准胡张混合有限元分析模式下忽略密度分布参数 rho")
        
        elif self._topopt_algorithm in ['density_based', 'level_set']:
            if rho_val is None:
                error_msg = f"拓扑优化算法 '{self._topopt_algorithm}' 需要提供密度分布参数 rho"
                self._log_error(error_msg)
    
        K0 = self.assemble_stiff_matrix(rho_val=rho_val)

        if enable_timing:
            t.send('刚度矩阵组装时间')

        space_sigmah = self._huzhang_space
        space_uh = self._tensor_space
        gdof_sigmah = space_sigmah.number_of_global_dofs()
        gdof_uh = space_uh.number_of_global_dofs()
        gdof = gdof_sigmah + gdof_uh
        F0 = bm.zeros(gdof, dtype=bm.float64, device=space_uh.device)

        F_uh = self.assemble_displacement_load_vector()
        F0[gdof_sigmah:] = F_uh

        F_sigmah = self.assemble_stress_load_vector()
        F0[:gdof_sigmah] = F_sigmah
        
        if enable_timing:
            t.send('载荷向量组装时间')

        boundary_type = self._pde.boundary_type
        if boundary_type == 'dirichlet':
            K, F = K0, F0
        else:
            K, F = self.apply_neumann_bc(K0, F0)

        if enable_timing:
            t.send('应用边界条件时间')
            
        solver_type = kwargs.get('solver', 'mumps')

        X = spsolve(K, F, solver=solver_type)

        if enable_timing:
            t.send('求解线性系统时间')

        space0 = self._huzhang_space
        space1 = self._tensor_space
        gdof0 = space0.number_of_global_dofs()

        sigmaval = X[:gdof0]
        uval = X[gdof0:]

        sigmah = space0.function()
        sigmah[:] = sigmaval

        uh = space1.function()
        uh[:] = uval

        if enable_timing:
            t.send('结果赋值时间')
            t.send(None)

        return sigmah, uh
    
    @solve_displacement.register('cg')
    def solve_displacement(self, 
            density_distribution: Optional[Function] = None, 
            **kwargs
            ) -> Function:
        from fealpy.solver import cg

        if self._topopt_algorithm is None:
            if density_distribution is not None:
                self._log_warning("标准有限元分析模式下忽略密度分布参数 rho")
        
        elif self._topopt_algorithm in ['density_based', 'level_set']:
            if density_distribution is None:
                error_msg = f"拓扑优化算法 '{self._topopt_algorithm}' 需要提供密度分布参数 rho"
                self._log_error(error_msg)
            
        K0 = self.assemble_stiff_matrix(density_distribution=density_distribution)
        F0 = self.assemble_force_vector()
        K, F = self.apply_bc(K0, F0)

        maxiter = kwargs.get('maxiter', 5000)
        atol = kwargs.get('atol', 1e-12)
        rtol = kwargs.get('rtol', 1e-12)
        x0 = kwargs.get('x0', None)

        X, info = cg(K, F[:], x0=x0,
            batch_first=True, 
            atol=atol, rtol=rtol, 
            maxit=maxiter, returninfo=True)
        
        space0 = self._huzhang_space
        space1 = self._tensor_space
        gdof0 = space0.number_of_global_dofs()

        sigmaval = X[:gdof0]
        uval = X[gdof0:]

        sigmah = space0.function()
        sigmah[:] = sigmaval

        uh = space1.function()
        uh[:] = uval
        
        
        gdof = self._tensor_space.number_of_global_dofs()
        self._log_info(f"Solving linear system with {gdof} displacement DOFs with CG solver.")

        return uh


    ###############################################################################################
    # 外部方法
    ###############################################################################################
    def get_local_stress_matrix_derivative(self, rho_val: Union[TensorLike, Function]) -> TensorLike:
        """计算局部应力矩阵 A 关于物理密度的导数（灵敏度）"""
        
        density_location = self._interpolation_scheme.density_location

        # TODO 目前仅支持插值杨氏模量 E
        E0 = self._material.youngs_modulus
        E_rho =  self._interpolation_scheme.interpolate_map(
                                                material=self._material, 
                                                rho_val=rho_val,
                                                integration_order=self._integration_order,
                                            )
        dE_rho = self._interpolation_scheme.interpolate_map_derivative(
                                                material=self._material, 
                                                rho_val=rho_val,
                                                integration_order=self._integration_order,
                                            ) 
        space0 = self._huzhang_space

        if density_location in ['element']:
            # rho_val: (NC, )
            diff_coef_element = - E0 * dE_rho / (E_rho**2) # (NC, )

            # TODO A0 的计算可以缓存下来
            lambda0, lambda1 = self._stress_matrix_coefficient()
            hzs_integrator = HuZhangStressIntegrator(lambda0=lambda0, lambda1=lambda1, 
                                                    q=self._integration_order, coef=None)
            AE0 = hzs_integrator.assembly(space=space0)

            diff_AE = bm.einsum('c, cij -> cij', diff_coef_element, AE0) # (NC, TLDOF, TLDOF)

        elif density_location in ['node']:
            # rho_val: (NN, )
            diff_coef_q = - E0 * dE_rho / (E_rho**2) # (NC, NQ)

            raise NotImplementedError("节点密度尚未实现")
        
        elif density_location in ['element_multiresolution']:
            raise NotImplementedError("多分辨率单元密度尚未实现")
        
        elif density_location in ['node_multiresolution']:
            raise NotImplementedError("多分辨率节点密度尚未实现")

        return diff_AE


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

    def _stress_matrix_coefficient(self) -> tuple[float, float]:
        """
        材料为均匀各向同性线弹性体时, 计算应力块矩阵的系数 lambda0 和 lambda1
        
        Returns
        -------
        lambda0: float
            1/(2μ)
        lambda1: float
            λ/(2μ(dλ+2μ)), 其中 d=2 为空间维数
        """
        d = self._GD  # 使用类中已有的几何维度
        
        # 从材料对象获取 Lamé 参数
        lam = self._material.lame_lambda
        mu = self._material.shear_modulus
        
        lambda0 = 1.0 / (2 * mu)
        lambda1 = lam / (2 * mu * (d * lam + 2 * mu))
        
        return lambda0, lambda1

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