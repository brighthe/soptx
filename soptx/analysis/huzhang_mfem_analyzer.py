from typing import Optional, Union, Literal, Tuple, Dict
from time import time

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace, Function
from fealpy.fem import BlockForm, BilinearForm, LinearForm
from fealpy.decorator.variantmethod import variantmethod
from fealpy.sparse import CSRTensor, COOTensor 
from fealpy.sparse.ops import bmat, spdiags

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
                disp_mesh: HomogeneousMesh,
                pde: PDEBase, 
                material: LinearElasticMaterial,
                interpolation_scheme: Optional[MaterialInterpolationScheme] = None,
                space_degree: int = 1,
                integration_order: int = 4,
                use_relaxation: bool = True,
                solve_method: Literal['mumps', 'cg'] = 'mumps',
                topopt_algorithm: Literal[None, 'density_based', 'level_set'] = None,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        """初始化胡张混合有限元分析器"""

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        # 私有属性（建议通过属性访问器访问，不要直接修改）
        self._mesh = disp_mesh
        self._pde = pde
        self._material = material
        self._interpolation_scheme = interpolation_scheme

        self._space_degree = space_degree
        self._integration_order = integration_order
        self._use_relaxation = use_relaxation

        self._topopt_algorithm = topopt_algorithm
        self._interpolation_scheme = interpolation_scheme

        self._solve_method = solve_method

        node = self._mesh.entity('node')
        self._mesh.meshdata['corner'] = self._pde.mark_corners(node)

        self._GD = self._mesh.geo_dimension()
        self._huzhang_space = HuZhangFESpace(mesh=self._mesh, p=self._space_degree, use_relaxation=self._use_relaxation)
        self._scalar_space = LagrangeFESpace(mesh=self._mesh, p=self._space_degree-1, ctype='D')
        self._tensor_space = TensorFunctionSpace(scalar_space=self._scalar_space, shape=(-1, self._GD))

        E0 = self._material.youngs_modulus
        nu0 = self._material.poisson_ratio
        lambda0, lambda1 = self._compute_compliance_coefficients(E0, nu0)
        self._hzs_integrator = HuZhangStressIntegrator(
                                            lambda0=lambda0, 
                                            lambda1=lambda1, 
                                            q=self._integration_order, 
                                            method='fast'
                                        )
        self._hzs_integrator.keep_data(True)
        self._cached_stress_matrix = None

        self._cached_mix_matrix = self._calculate_mix_matrix()

        self._E_rho = None
        self._nu_rho = None
        self._lambda0_rho = None
        self._lambda1_rho = None


    ##############################################################################################
    # 属性相关函数
    ##############################################################################################

    @property
    def pde(self) -> PDEBase:
        """获取当前的 PDE 对象"""
        return self._pde

    @property
    def disp_mesh(self) -> HomogeneousMesh:
        """获取当前的位移网格对象"""
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
        """获取当前的位移有限元空间"""
        return self._tensor_space
    
    @property
    def integration_order(self) -> int:
        """获取当前的数值积分阶次"""
        return self._integration_order
    
    @property
    def material(self) -> LinearElasticMaterial:
        """获取当前的材料类"""
        return self._material
    
    @property
    def interpolation_scheme(self) -> MaterialInterpolationScheme:
        """获取当前的材料插值方案"""
        return self._interpolation_scheme
    
    @property
    def mix_matrix(self) -> Union[CSRTensor, COOTensor]:
        """获取应力-位移耦合矩阵 B_σu"""
        return self._cached_mix_matrix
    
    def get_stress_matrix(self, 
                        rho_val: Optional[Union[Function, TensorLike]] = None
                        ) -> Union[CSRTensor, COOTensor]:
        """获取应力矩阵 A_σσ """
        if self._cached_stress_matrix is not None:
            return self._cached_stress_matrix

        else:
            return self._calculate_stress_matrix(rho_val=rho_val, enable_timing=False)

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

    def _calculate_stress_matrix(self, 
                        rho_val: Optional[Union[Function, TensorLike]] = None,
                        enable_timing: bool = True,
                    ) -> Union[CSRTensor, COOTensor]:
        """组装应力矩阵 A_σσ"""
        space_sigma = self._huzhang_space

        if self._topopt_algorithm in ['density_based']:
            material_params = self._interpolation_scheme.interpolate_material(
                                            material=self._material,
                                            rho_val=rho_val,
                                            integration_order=self._integration_order,
                                            displacement_mesh=self._mesh,
                                    )
            
            if isinstance(material_params, tuple):
                E_rho, nu_rho = material_params
            else:
                E_rho = material_params
                nu_rho = self._material.poisson_ratio
        
            lambda0_rho, lambda1_rho = self._compute_compliance_coefficients(E_rho, nu_rho)

            self._E_rho = E_rho
            self._nu_rho = nu_rho
            self._lambda0_rho = lambda0_rho
            self._lambda1_rho = lambda1_rho

            # 更新密度系数
            self._hzs_integrator.lambda0 = lambda0_rho
            self._hzs_integrator.lambda1 = lambda1_rho

        bform1 = BilinearForm(space_sigma)
        bform1.add_integrator(self._hzs_integrator)
        A = bform1.assembly(format='csr')

        #TODO 角点松弛
        if space_sigma.use_relaxation == True:
            TM = space_sigma.TM
            A = TM.T @ A @ TM

        # 缓存应力矩阵
        self._cached_stress_matrix = A

        return A
    
    def _calculate_mix_matrix(self, enable_timing: bool=True) -> Union[CSRTensor, COOTensor]:
        """组装应力-位移耦合矩阵 B_σu"""
        start_time = time()
        
        space_sigma = self._huzhang_space
        space_u = self._tensor_space

        bform = BilinearForm((space_u, space_sigma)) 
        hzm_integrator = HuZhangMixIntegrator()
        bform.add_integrator(hzm_integrator)

        B = bform.assembly(format='csr') # (GDOF_sigma, GDOF_u)

        #TODO 角点松弛
        if space_sigma.use_relaxation == True:
            TM = space_sigma.TM
            B = TM.T @ B
        
        iteration_time = time() - start_time

        if enable_timing:
            print(f"初始化应力-位移耦合矩阵 B_σu 时间: {iteration_time:.6f} 秒")

        return B

    def assemble_stiff_matrix(self, 
                        rho_val: Optional[Union[Function, TensorLike]] = None,
                        enable_timing: bool = False,
                    ) -> Union[CSRTensor, COOTensor]:
        """组装全局刚度矩阵"""
        t = None
        if enable_timing:
            t = timer(f"组装刚度矩阵")
            next(t)

        p = self._space_degree
        mesh = self._mesh
        GD = mesh.geo_dimension()

        space_u = self._tensor_space

        # 应力矩阵
        A = self._calculate_stress_matrix(rho_val=rho_val)

        if enable_timing:
            t.send('组装应力矩阵时间')

        # 应力-位移耦合矩阵
        B = self._cached_mix_matrix

        if enable_timing:
            t.send('组装混合矩阵时间')

        if p >= GD + 1:
            K = bmat([[A,   B   ],
                      [B.T, None]], format='csr')

        elif p <= GD:
            bform3 = BilinearForm(space_u)
            jpi_integrator = JumpPenaltyIntegrator(q=self._integration_order, threshold=None, method='vector_jump')
            # jpi_integrator = JumpPenaltyIntegrator(q=self._integration_order, threshold=None, method='matrix_jump')
            bform3.add_integrator(jpi_integrator)
            J = bform3.assembly(format='csr')
            K = bmat([[A,   B],
                      [B.T, J]], format='csr')

        if enable_timing:
            t.send('组装时间')
            t.send(None)

        return K
    
    def assemble_body_force_vector(self, 
                                   enable_timing: bool = False
                                ) -> TensorLike:
        """组装体力源项向量 (f, v) - 位移空间"""
        t = None
        if enable_timing:
            t = timer(f"组装 f_body")
        
        body_force = self._pde.body_force
        space_uh = self._tensor_space

        integrator = SourceIntegrator(source=body_force, q=self._integration_order)
        lform = LinearForm(space_uh)
        lform.add_integrator(integrator)

        F_body = lform.assembly(format='dense')

        if enable_timing:
            t.send('组装时间')
            t.send(None)

        return F_body
    
    def assemble_displacement_bc_vector_backup(self, enable_timing: bool = False):
        """组装位移边界条件产生的载荷向量 (自然边界条件) <u_D, (tau · n)>_Γ_D
        
        当前所有位移边界均为齐次边界条件 (u_D = 0), 故直接返回零向量.

        NOTE 角点松弛: 若将来支持非齐次位移边界 (u_D ≠ 0), 需在积分完成后
        对载荷向量施加松弛变换: F_natural = TM.T @ F_natural
        参考 boundary_interpolate 中的处理方式.
        """
        space = self._huzhang_space
        gdof = space.number_of_global_dofs()

        return bm.zeros(gdof, dtype=bm.float64, device=space.device)
    
    def apply_traction_boundary_condition(self, 
                                        K: CSRTensor, 
                                        F: TensorLike, 
                                        space_sigma: HuZhangFESpace,
                                    ) -> Tuple[CSRTensor, TensorLike]:
        """施加本质边界条件  σ·n = g_N

        NOTE 角点松弛: set_dirichlet_bc (即 boundary_interpolate) 内部已完成
        松弛变换 uh_val = TM.T @ uh_val, 此处无需额外处理.
        若替换为其他方式获取边界 DOF 值, 需确保同样完成该变换.
        """
        gd_traction = getattr(self._pde, "traction_bc", None)
        if gd_traction is None or (not callable(gd_traction)):
                self._log_error("PDE 对象缺少牵引边界函数 traction_bc")

        # 根据网格设置点载荷作用区域
        if hasattr(self._pde, 'set_load_region'):
            self._pde.set_load_region(self._mesh)
        
        # 计算边界自由度的值
        uh_val, is_bd_dof = space_sigma.set_dirichlet_bc(gd_traction)

        gdof_total = K.shape[0]
        gdof_sigma = space_sigma.number_of_global_dofs()
        
        U_fixed = bm.zeros(gdof_total, dtype=F.dtype)
        U_fixed[:gdof_sigma] = uh_val
        
        is_fixed_dof = bm.zeros(gdof_total, dtype=bm.bool)
        is_fixed_dof[:gdof_sigma] = is_bd_dof

        #* 修改线性系统 (标准的置 1 置 0 法)
        # 移项: 将已知非零值的贡献移到右端项
        F = F - K @ U_fixed
        
        # 强加边界值: 在右端项直接填入已知值
        F[is_fixed_dof] = U_fixed[is_fixed_dof]
        
        # 修改矩阵：行列清零，对角置1        
        fixed_idx = bm.zeros(gdof_total, dtype=bm.int32)
        fixed_idx[is_fixed_dof] = 1
        
        I_bd = spdiags(fixed_idx, 0, gdof_total, gdof_total)
        I_in = spdiags(1 - fixed_idx, 0, gdof_total, gdof_total)
        
        K = I_in @ K @ I_in + I_bd
        
        return K, F
    
    def solve_state(self, 
                    rho_val: Optional[Union[TensorLike, Function]] = None, 
                    enable_timing: bool = False, 
                    **kwargs
                ) -> Dict[str, Function]:
        t = None
        if enable_timing:
            t = timer(f"分析阶段时间")
            next(t)

        mesh = self._mesh
        pde = self._pde
        bc = mesh.entity_barycenter('edge')

        mesh.edgedata['essential_bc'] = pde.is_traction_boundary(bc)      # σ·n = t (强施加)
        mesh.edgedata['natural_bc']   = pde.is_displacement_boundary(bc)  # u = u_D (弱施加)

        K0 = self.assemble_stiff_matrix(rho_val=rho_val)

        if enable_timing:
            t.send('矩阵组装时间')

        space_sigmah = self._huzhang_space
        space_uh = self._tensor_space
        gdof_sigmah = space_sigmah.number_of_global_dofs()
        gdof_uh = space_uh.number_of_global_dofs()
        gdof = gdof_sigmah + gdof_uh

        F0 = bm.zeros(gdof, dtype=bm.float64, device=space_uh.device)

        # 组装体力源项 -> 对应位移测试函数 v
        F_body = self.assemble_body_force_vector() 
        F0[gdof_sigmah:] = -F_body 

        # 自然边界条件处理 (位移边界 u = u_D)
        F_natural = self.assemble_displacement_bc_vector()
        F0[:gdof_sigmah] = F_natural

        if enable_timing:
            t.send('源项处理时间')

        has_essential = ('essential_bc' in mesh.edgedata) and bool(mesh.edgedata['essential_bc'].any())
        if has_essential:
            K, F = self.apply_traction_boundary_condition(K0, F0, space_sigmah)
        else:
            K, F = K0, F0

        if enable_timing:
            t.send('本质边界条件处理时间')

        solver_type = kwargs.get('solver', self._solve_method)
        
        if solver_type in ['mumps', 'scipy']:
            from fealpy.solver import spsolve
            X = spsolve(K, F, solver=solver_type)

        elif solver_type in ['cg']: 
            pass

        else:
            self._log_error(f"未知的求解器类型: {solver_type}")

        if enable_timing:
            t.send('求解时间')

        sigmaval = X[:gdof_sigmah]
        uval = X[gdof_sigmah:]

        sigmah = space_sigmah.function()
        sigmah[:] = sigmaval

        uh = space_uh.function()
        uh[:] = uval

        if enable_timing:
            t.send('结果赋值时间')
            t.send(None)

        return {'stress': sigmah, 'displacement': uh}
    
    def solve_adjoint(self, 
                    rhs: TensorLike,
                    rho_val: Optional[Union[TensorLike, Function]] = None,
                    **kwargs
                ) -> TensorLike:
        """
        求解伴随方程 K @ λ = rhs
        
        Parameters
        ----------
        rhs : (n_gdof,) 或 (n_gdof, n_rhs) 伴随载荷向量 (支持批量求解)
        rho_val : 密度场
        
        Returns
        -------
        adjoint_lambda : (n_gdof, n_rhs) 伴随变量
        """
        # 组装刚度矩阵
        K0 = self.assemble_stiff_matrix(rho_val=rho_val)

        # 施加齐次本质边界条件
        space_sigma = self._huzhang_space
        pde = self._pde
        
        gdof_sigma = space_sigma.number_of_global_dofs()
        gdof_total = K0.shape[0]

        # 根据网格设置点载荷作用区域
        if hasattr(pde, 'set_load_region'):
            pde.set_load_region(self._mesh)

        # 获取边界自由度位置
        gd_traction = pde.traction_bc
        _, is_bd_dof = space_sigma.set_dirichlet_bc(gd_traction)

        # 扩展到全局自由度
        is_fixed_dof = bm.zeros(gdof_total, dtype=bm.bool)
        is_fixed_dof[:gdof_sigma] = is_bd_dof

        # 齐次边界条件：载荷在边界自由度上置零
        rhs = bm.set_at(rhs, is_fixed_dof, 0.0)

        # 修改矩阵：行列清零，对角置 1
        fixed_idx = bm.zeros(gdof_total, dtype=bm.int32)
        fixed_idx[is_fixed_dof] = 1
        
        I_bd = spdiags(fixed_idx, 0, gdof_total, gdof_total)
        I_in = spdiags(1 - fixed_idx, 0, gdof_total, gdof_total)
        
        K = I_in @ K0 @ I_in + I_bd

        # 初始化结果
        adjoint_lambda = bm.zeros_like(rhs)
        
        # 求解
        solver_type = kwargs.get('solver', self._solve_method)
        
        if solver_type in ['mumps', 'scipy']:
            from fealpy.solver import spsolve
            adjoint_lambda[:] = spsolve(K, rhs, solver=solver_type)
            
        elif solver_type in ['cg']:
            from fealpy.solver import cg
            maxiter = kwargs.get('maxiter', 5000)
            atol = kwargs.get('atol', 1e-12)
            rtol = kwargs.get('rtol', 1e-12)
            
            adjoint_lambda[:], _ = cg(K.tocoo(), rhs, 
                                    batch_first=False,
                                    atol=atol, rtol=rtol, 
                                    maxit=maxiter, returninfo=True)
        
        return adjoint_lambda
    
    def compute_stress_state(self, 
                        state: dict,
                        rho_val: Union[TensorLike, Function] = None,
                        integration_order: Optional[int] = None,
                    ) -> Dict[str, TensorLike]:
        """
        计算应力场
        
        Parameters
        ----------
        state : 状态字典，包含位移场等信息
        rho_val : 密度场（用于应力惩罚，拓扑优化时需要）
        integration_order : 积分阶次
        
        Returns
        -------
        dict : 包含以下键值：
            - 'stress_solid': 实体应力 (NC, NQ, NS) 或 (NC, n_sub, NQ, NS)
            - 'von_mises': von Mises 等效应力 (NC, NQ) 或 (NC, n_sub, NQ)
            - 'von_mises_max': 每个单元的最大 von Mises 应力 (NC,)
        """
        if integration_order is None:
            # integration_order = self._integration_order
            # TODO 测试
            integration_order = 1
        
        if state is None:
            state = self.solve_state(rho_val=rho_val)
        
        stress_dof = state['stress']  
        
        #TODO 转换为积分点应力
        stress_at_quad = self.extract_stress_at_quadrature_points(
                                                        stress_dof=stress_dof, 
                                                        integration_order=integration_order
                                                    )  # (NC, NQ, NS)
        
        result = {'stress_solid': stress_at_quad}
        
        # 计算 von Mises 应力
        von_mises = self._material.calculate_von_mises_stress(stress_vector=stress_at_quad)
        result['von_mises'] = von_mises
        
        if von_mises.ndim == 2:
            von_mises_max = bm.max(von_mises, axis=1)
        elif von_mises.ndim == 3:
            von_mises_max = bm.max(von_mises.reshape(von_mises.shape[0], -1), axis=1)
        else:
            self._log_error(f"意外的 von Mises 应力维度: {von_mises.ndim}")
        
        result['von_mises_max'] = von_mises_max
        
        return result


    ###############################################################################################
    # 外部方法
    ###############################################################################################

    def compute_local_stress_matrix_derivative(self, rho_val: Union[TensorLike, Function]) -> TensorLike:
        """计算局部应力矩阵 A 关于物理密度的导数（灵敏度）"""
        if self._lambda0_rho is None or self._lambda1_rho is None:
            material_vals = self._interpolation_scheme.interpolate_material(
                                            material=self._material, 
                                            rho_val=rho_val,
                                            integration_order=self._integration_order,
                                            displacement_mesh=self._mesh,
                                        )
            if isinstance(material_vals, tuple):
                E_rho, nu_rho = material_vals
            else:
                E_rho = material_vals
                nu_rho = bm.full_like(E_rho, self._material.poisson_ratio)

            lambda0_rho, lambda1_rho = self._compute_compliance_coefficients(E_rho, nu_rho)

        else:
            E_rho = self._E_rho
            nu_rho = self._nu_rho
            lambda0_rho = self._lambda0_rho
            lambda1_rho = self._lambda1_rho

        material_derivs = self._interpolation_scheme.interpolate_material_derivative(
                                        material=self._material, 
                                        rho_val=rho_val,
                                        integration_order=self._integration_order,
                                    )
        if isinstance(material_derivs, tuple):
            dE_rho, dnu_rho = material_derivs
        else:
            dE_rho = material_derivs
            dnu_rho = bm.zeros_like(dE_rho) 

        d = self._GD

        dlambda0 = (1.0 / E_rho) * dnu_rho - (lambda0_rho / E_rho) * dE_rho

        denominator_factor = 1.0 + (d - 2.0) * nu_rho
        numerator_deriv = 1.0 + 2.0 * nu_rho + (d - 2.0) * nu_rho**2
        g_prime = numerator_deriv / (denominator_factor**2)
        dlambda1 = (1.0 / E_rho) * g_prime * dnu_rho - (lambda1_rho / E_rho) * dE_rho

        space_sigma = self._huzhang_space
        
        # 调用 fetch_fast 获取缓存的几何矩阵 (M0, M1)
        M0, M1 = self._hzs_integrator.fetch_fast(space_sigma)

        if dlambda0.ndim == 1:
            dlambda0 = dlambda0[:, None, None]
        if dlambda1.ndim == 1:
            dlambda1 = dlambda1[:, None, None]

        diff_AE = dlambda0 * M0 - dlambda1 * M1

        return diff_AE

    def extract_stress_at_quadrature_points(self, 
                                        stress_dof: TensorLike, 
                                        integration_order: int = None
                                    ) -> TensorLike:
        """
        将全局应力自由度转换为单元积分点上的应力向量
            注意: 胡张元空间基函数输出的应力分量顺序为 [xx, xy, yy]
                  本函数会自动将其重排为标准 Voigt 顺序 [xx, yy, xy] 以适配后续计算
        
        Parameters
        ----------
        stress_dof : TensorLike, shape (gdof,)
            全局应力自由度向量
        integration_order : int, optional
            积分阶数，默认使用分析器的积分阶数
            
        Returns
        -------
        stress_vector : TensorLike, shape (NC, NQ, NS)
            单元积分点上的应力向量
        """
        space = self._huzhang_space
        mesh = space.mesh
        
        if integration_order is None:
            integration_order = self._integration_order
        
        qf = mesh.quadrature_formula(integration_order, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
                
        phi = space.basis(bcs) # (NC, NQ, LDOF, NS)
        
        cell2dof = space.cell_to_dof()  # (NC, LDOF)
        stress_cell = stress_dof[cell2dof]  # (NC, LDOF)
        
        stress_vector = bm.einsum('cqls, cl -> cqs', phi, stress_cell) # (NC, NQ, NS)

        if stress_vector.shape[-1] == 3:
            perm_indices = [0, 2, 1]
            stress_vector = stress_vector[..., perm_indices]
        else:
            raise NotImplementedError("仅支持二维问题的应力分量重排")
        
        return stress_vector


    ##############################################################################################
    # 内部方法
    ##############################################################################################

    def _compute_compliance_coefficients(self, 
                                       E: TensorLike, 
                                       nu: TensorLike
                                    ) -> tuple[TensorLike, TensorLike]:
        """
        材料为均匀各向同性线弹性体时, 
        根据杨氏模量场 E 和泊松比场 nu 计算柔度矩阵系数场 lambda0 和 lambda1
        """
        d = self._GD  

        lambda0 = (1.0 + nu) / E

        numerator = nu * (1.0 + nu)
        denominator_factor = 1.0 + (d - 2.0) * nu
        denominator = E * denominator_factor
        
        lambda1 = numerator / denominator
        
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
    

    def assemble_displacement_bc_vector(self, enable_timing: bool = False):
        """组装位移边界条件产生的载荷向量 (自然边界条件) <u_D, (tau · n)>_Γ_D"""
        t = None
        if enable_timing:
            t = timer("组装 F_disp_bc")

        space = self._huzhang_space
        mesh = space.mesh

        gdof = space.number_of_global_dofs()
        ldof = space.number_of_local_dofs()

        bd_edge_flag = mesh.boundary_edge_flag()

        # 1. 获取自然边界标记 (Natural BC)
        if 'natural_bc' in mesh.edgedata:
            disp_edge_flag = mesh.edgedata['natural_bc']
        else:
            bc_edge = mesh.entity_barycenter('edge')
            disp_edge_flag = self._pde.is_displacement_boundary(bc_edge)

        # 简化逻辑：不再检查 ndim==2，默认全模型下的标记是 1D 布尔数组
        bdedge = bd_edge_flag & disp_edge_flag

        # 2. 排除本质边界 (Essential BC) 的干扰
        # 虽然 PDE 定义中通常互斥，但保留此检查更健壮
        if 'essential_bc' in mesh.edgedata:
            essential_bc = mesh.edgedata['essential_bc']
            # 如果 essential_bc 意外是 2D 的 (旧代码遗留)，将其压缩为 1D
            if essential_bc.ndim == 2:
                essential_bc_1d = bm.any(essential_bc, axis=1)
            else:
                essential_bc_1d = essential_bc
            
            bdedge = bdedge & (~essential_bc_1d)

        NBF = int(bdedge.sum())
        if NBF == 0:
            return bm.zeros(gdof, dtype=bm.float64, device=space.device)

        # 3. 准备积分数据
        e2c = mesh.edge_to_cell()[bdedge] 
        en  = mesh.edge_unit_normal()[bdedge]
        edge_measure  = mesh.entity_measure('edge')[bdedge]

        qf = mesh.quadrature_formula(self._integration_order, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(bcs)

        # 将边积分点映射到单元内部
        bcsi = [bm.insert(bcs, i, 0, axis=-1) for i in range(3)]

        # 计算基函数投影
        symidx = [[0, 1], [1, 2]]
        phi_n = bm.zeros((NBF, NQ, ldof, 2), dtype=bm.float64, device=space.device)
        gd_val = bm.zeros((NBF, NQ, 2), dtype=bm.float64, device=space.device)

        # 获取位移边界函数
        gd = getattr(self._pde, "displacement_bc", None)
        if gd is None or (not callable(gd)):
            # 如果没有定义 displacement_bc，默认位移为 0 (齐次)，直接返回零向量
            # (通常底部固支 u=0，此时该积分为0，可以直接返回，节省计算)
            return bm.zeros(gdof, dtype=bm.float64, device=space.device)
        
        # 4. 循环计算
        for i in range(3):
            # 找到局部编号为 i 的面
            flag = (e2c[:, 2] == i)
            if not bm.any(flag):
                continue
                
            # 计算基函数
            current_cells = e2c[flag, 0]
            phi = space.basis(bcsi[i], index=current_cells)

            # 计算 tau · n (测试函数应力 在法向上的投影)
            # phi 的形状通常是 (..., 3) 对应 Voigt [xx, yy, xy] 或 [xx, xy, yy]
            # 这里假设 space.basis 返回的是标准 Voigt 顺序，需根据具体空间确认 symidx 映射
            en_curr = en[flag, None, None, :]
            
            # 投影逻辑：假设 phi 输出为 [tau_xx, tau_xy, tau_yy]
            # (tau . n)_x = tau_xx * n_x + tau_xy * n_y
            # (tau . n)_y = tau_xy * n_x + tau_yy * n_y
            phi_n[flag, ..., 0] = bm.sum(phi[..., symidx[0]] * en_curr, axis=-1)
            phi_n[flag, ..., 1] = bm.sum(phi[..., symidx[1]] * en_curr, axis=-1)

            # 计算边界函数值 u_D
            points = mesh.bc_to_point(bcsi[i], index=current_cells)
            gd_val[flag] = gd(points)
        
        # 5. 积分组装 (简化版)
        # 直接计算点积 <phi_n, u_D>，无需 mask
        val = bm.einsum('q, c, cqld, cqd -> cl', ws, edge_measure, phi_n, gd_val)
        
        cell2dof = space.cell_to_dof()[e2c[:, 0]]
        F_vec = bm.zeros(gdof, dtype=bm.float64, device=space.device)
        bm.add.at(F_vec, cell2dof, val)

        #TODO 角点松弛变换
        if hasattr(space, 'use_relaxation') and space.use_relaxation:
            F_vec = space.TM.T @ F_vec

        if enable_timing:
            t.send("组装完成")
            t.send(None)

        return F_vec