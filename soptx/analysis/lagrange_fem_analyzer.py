from typing import Optional, Union, Literal, Dict

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import SimplexMesh, HomogeneousMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace, Function
from fealpy.fem import BilinearForm, LinearForm
from fealpy.decorator import variantmethod
from fealpy.sparse import CSRTensor, COOTensor

from soptx.interpolation.linear_elastic_material import LinearElasticMaterial
from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
from soptx.analysis.integrators.linear_elastic_integrator import LinearElasticIntegrator
from soptx.analysis.integrators.source_integrator import SourceIntegrator
from soptx.model.pde_base import PDEBase
from soptx.utils.base_logged import BaseLogged
from soptx.utils import timer

class LagrangeFEMAnalyzer(BaseLogged):
    def __init__(self,
                disp_mesh: HomogeneousMesh,
                pde: PDEBase, 
                material: LinearElasticMaterial,
                space_degree: int = 1,
                integration_order: int = 4,
                assembly_method: Literal['standard', 'voigt', 'fast'] = 'standard',
                solve_method: Literal['mumps', 'cg'] = 'mumps',
                topopt_algorithm: Literal[None, 'density_based', 'level_set'] = None,
                interpolation_scheme: Optional[MaterialInterpolationScheme] = None,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        """初始化拉格朗日有限元分析器"""

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        # 私有属性（建议通过属性访问器访问，不要直接修改）
        self._mesh = disp_mesh
        self._pde = pde
        self._material = material
       
        self._space_degree = space_degree
        self._integration_order = integration_order
        self._assembly_method = assembly_method

        self._topopt_algorithm = topopt_algorithm
        self._interpolation_scheme = interpolation_scheme

        self._solve_method = solve_method

        self._GD = self._mesh.geo_dimension()

        #* (GD, -1): dof_priority (x0, ..., xn, y0, ..., yn)
        #* (-1, GD): gd_priority (x0, y0, ..., xn, yn)
        self._scalar_space = LagrangeFESpace(self._mesh, p=self._space_degree, ctype='C')
        self._tensor_space = TensorFunctionSpace(scalar_space=self._scalar_space, shape=(-1, self._GD))

        # 缓存的矩阵和向量
        self._K = None
        self._F = None

        self._integrator = LinearElasticIntegrator(material=self._material,
                                                q=self._integration_order,
                                                method=self._assembly_method)
        self._integrator.keep_data(True)

        self._cached_ke0 = None

    ##############################################################################################
    # 属性相关函数
    ##############################################################################################
    
    @property
    def disp_mesh(self) -> HomogeneousMesh:
        """获取当前的位移网格对象"""
        return self._mesh
    
    @property
    def pde(self) -> PDEBase:
        """获取当前的 PDE 对象"""
        return self._pde
    
    @property
    def scalar_space(self) -> LagrangeFESpace:
        """获取当前的标量函数空间"""
        return self._scalar_space
    
    @property
    def tensor_space(self) -> TensorFunctionSpace:
        """获取当前的张量函数空间"""
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
    def assembly_method(self) -> str:
        """获取当前的组装方法"""
        return self._assembly_method
    
    @property
    def topopt_algorithm(self) -> Optional[str]:
        """获取当前的拓扑优化算法"""
        return self._topopt_algorithm
    
    @property
    def stiffness_matrix(self) -> Union[CSRTensor, COOTensor]:
        """获取当前的刚度矩阵"""
        return self._K
    
    @property
    def force_vector(self) -> Union[TensorLike, COOTensor]:
        """获取当前的载荷向量"""
        return self._F

    @scalar_space.setter
    def scalar_space(self, space: LagrangeFESpace) -> None:
        """设置标量函数空间"""
        self._scalar_space = space

    @tensor_space.setter
    def tensor_space(self, space: TensorFunctionSpace) -> None:
        """设置张量函数空间"""
        self._tensor_space = space

    
    ##############################################################################################
    # 核心方法
    ##############################################################################################

    def assemble_stiff_matrix(self, 
                            rho_val: Optional[Union[Function, TensorLike]] = None,
                            enable_timing: bool = False, 
                        ) -> Union[CSRTensor, COOTensor]:
        """组装全局刚度矩阵

        Parameters
        ----------
        rho_val : 密度值
        - 单元密度 - TensorLike
            - 单分辨率 - (NC, )
            - 多分辨率 - (NC, n_sub)
        - 节点密度 - Fucntion
            - 单分辨率 - (NN, )
            - 多分辨率 - (NN, )
        """
        t = None
        if enable_timing:
            t = timer(f"双线性型组装内部")
            next(t)

        if self._topopt_algorithm is None:
            if rho_val is not None:
                self._log_warning("标准有限元分析模式下忽略相对密度 rho")
            
            coef = None
        
        elif self._topopt_algorithm == 'density_based':
            if rho_val is None:
                self._log_error("基于密度的拓扑优化算法需要提供相对密度 rho")

            # TODO 目前仅支持插值杨氏模量 E 
            E_rho = self._interpolation_scheme.interpolate_material(
                                            material=self._material,
                                            rho_val=rho_val,
                                            integration_order=self._integration_order,
                                            displacement_mesh=self._mesh,
                                        )
            E0 = self._material.youngs_modulus
            coef = E_rho / E0
        
        else:
            error_msg = f"不支持的拓扑优化算法: {self._topopt_algorithm}"
            self._log_error(error_msg)

        if enable_timing:
            t.send('预备')

        # TODO 这里的 coef 也和材料有关, 可能需要进一步处理,
        # TODO coef 是应该在 LinearElasticIntegrator 中, 还是在 MaterialInterpolationScheme 中处理 ?
        # 更新密度系数
        self._integrator.coef = coef

        bform = BilinearForm(self._tensor_space)
        bform.add_integrator(self._integrator)

        K = bform.assembly(format='csr')

        self._K = K

        if enable_timing:
            t.send('组装')
            t.send(None)

        return K
    
    def assemble_spring_stiff_matrix(self):
        """组装弹簧刚度矩阵"""
        tspace = self._tensor_space
        TGDOF = tspace.number_of_global_dofs()

        k_in = self._pde.k_in
        k_out = self._pde.k_out
        threshold_spring = self._pde.is_spring_boundary()
        isBdDof = tspace.is_boundary_dof(threshold=threshold_spring, method='interp')
        spring_dofs = bm.where(isBdDof)[0]
        indices = bm.stack([spring_dofs, spring_dofs], axis=0)
        values = bm.tensor([k_in, k_out], dtype=bm.float64, device=tspace.device)
        spshape = (TGDOF, TGDOF)

        K_coo = COOTensor(indices=indices, values=values, spshape=spshape)

        K = K_coo.tocsr()

        return K

    def assemble_body_force_vector(self) -> Union[TensorLike, COOTensor]:
        """组装体力对应的体积分"""
        body_force = self._pde.body_force

        # NOTE F.dtype == COOTensor or TensorLike
        integrator = SourceIntegrator(source=body_force, q=self._integration_order)
        lform = LinearForm(self._tensor_space)
        lform.add_integrator(integrator)
        F = lform.assembly(format='dense')
        
        return F

    def apply_bc(self, 
                K: Union[CSRTensor, COOTensor], 
                F: CSRTensor,
                adjoint: str = False
            ) -> tuple[CSRTensor, CSRTensor]:
        """应用边界条件"""        
        boundary_type = self._pde.boundary_type
        load_type = self._pde.load_type

        space_uh = self._tensor_space
        gdof = space_uh.number_of_global_dofs()

        if boundary_type == 'mixed':
            #* 1. Neumann 边界条件处理 - 弱形式施加 *#
            # 集中载荷 (点力) - 等效节点力方法
            if load_type == 'concentrated':
                gd_sigmah = self._pde.concentrate_load_bc
                threshold_sigmah = self._pde.is_concentrate_load_boundary()
        
                # 点力必须定义在网格节点上
                isBdTDof = space_uh.is_boundary_dof(threshold=threshold_sigmah, method='interp')
                isBdSDof = space_uh.scalar_space.is_boundary_dof(threshold=threshold_sigmah, method='interp')
                ipoints_uh = space_uh.interpolation_points()
                gd_sigmah_val = gd_sigmah(ipoints_uh[isBdSDof])

                # 动态计算节点数量, 将总力平均分配
                num_load_nodes = bm.sum(isBdSDof)
                if num_load_nodes > 0:
                    gd_sigmah_val = gd_sigmah_val / num_load_nodes

                F_sigmah = space_uh.function()
                if space_uh.dof_priority:
                    F_sigmah[:] = bm.set_at(F_sigmah[:], isBdTDof, gd_sigmah_val.T.reshape(-1))
                else:
                    F_sigmah[:] = bm.set_at(F_sigmah[:], isBdTDof, gd_sigmah_val.reshape(-1))
            # 分布载荷 (面力)
            elif load_type == 'distributed':
                gd_sigmah = self._pde.neumann_bc
                threshold_sigmah = self._pde.is_neumann_boundary()

                from soptx.analysis.integrators.face_source_integrator_lfem import BoundaryFaceSourceIntegrator_lfem
                integrator = BoundaryFaceSourceIntegrator_lfem(source=gd_sigmah, q=self._integration_order, threshold=threshold_sigmah)
                lform = LinearForm(self._tensor_space)
                lform.add_integrator(integrator)
                F_sigmah = lform.assembly(format='dense')
                
            else:
                raise NotImplementedError(f"不支持的载荷类型: {load_type}")
            
            if adjoint:
                gd_adjoint = self._pde.adjoint_load_bc
                threshold_adjoint = self._pde.is_adjoint_load_boundary()

                isBdTDof = space_uh.is_boundary_dof(threshold=threshold_adjoint, method='interp')
                isBdSDof = space_uh.scalar_space.is_boundary_dof(threshold=threshold_adjoint, method='interp')
                ipoints_uh = space_uh.interpolation_points()
                gd_adjoint_val = gd_adjoint(ipoints_uh[isBdSDof])

                F_adjoint = space_uh.function()
                if space_uh.dof_priority:
                    F_adjoint[:] = bm.set_at(F_adjoint[:], isBdTDof, gd_adjoint_val.T.reshape(-1))
                else:
                    F_adjoint[:] = bm.set_at(F_adjoint[:], isBdTDof, gd_adjoint_val.reshape(-1))

                F += bm.stack([F_sigmah, F_adjoint], axis=1)
            else:
                F += F_sigmah

            self._F = F

            #* 2. Dirichlet 边界条件处理 - 强形式施加 *#
            gd_uh = self._pde.dirichlet_bc
            threshold_uh = self._pde.is_dirichlet_boundary()

            uh_bd = bm.zeros(gdof, dtype=bm.float64, device=space_uh.device)
            uh_bd, isBdDof = space_uh.boundary_interpolate(
                                                        gd=gd_uh,
                                                        threshold=threshold_uh,
                                                        method='interp'
                                                    )
            
            if adjoint:
                uh_bd = bm.repeat(uh_bd.reshape(-1, 1), 2, axis=1)
                F = F - K.matmul(uh_bd[:])
                F[isBdDof, :] = uh_bd[isBdDof, :]

            else: 
                #? matmul 函数下 K 必须是 COO 格式, 不能是 CSR 格式, 否则 GPU 下 device_put 函数会出错
                F = F - K.tocoo().matmul(uh_bd[:])
                F[isBdDof] = uh_bd[isBdDof]
            
            K = self._apply_matrix(A=K, isDDof=isBdDof)

            return K, F

        elif boundary_type == 'dirichlet':
            # 强形式施加
            self._F = F

            gd = self._pde.dirichlet_bc
            threshold = self._pde.is_dirichlet_boundary()
        
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
    
    def solve_state(self, 
                    rho_val: Optional[Union[TensorLike, Function]] = None,
                    adjoint: bool = False,
                    enable_timing: bool = True, 
                    **kwargs
                ) -> Dict[str, Function]:
        t = None
        if enable_timing:
            t = timer(f"分析求解位移阶段")
            next(t)

        if self._topopt_algorithm is None:
            if rho_val is not None:
                self._log_warning("标准有限元分析模式下忽略密度分布参数 rho")
        
        elif self._topopt_algorithm in ['density_based', 'level_set']:
            if rho_val is None:
                error_msg = f"拓扑优化算法 '{self._topopt_algorithm}' 需要提供密度分布参数 rho"
                self._log_error(error_msg)

        if adjoint:
            K_struct = self.assemble_stiff_matrix(rho_val=rho_val)
            K_spring = self.assemble_spring_stiff_matrix()
            K0 = K_struct + K_spring
            F0_struct = self.assemble_body_force_vector()
            F0_spring = bm.zeros_like(F0_struct)
            F0 = bm.stack([F0_struct, F0_spring], axis=1)

            K, F = self.apply_bc(K0, F0, adjoint)

            uh = bm.zeros(F.shape, dtype=bm.float64, device=F.device)
        
        else:
            K0 = self.assemble_stiff_matrix(rho_val=rho_val)
            if enable_timing:
                t.send('双线性型组装')

            F0 = self.assemble_body_force_vector()
            if enable_timing:
                t.send('线性型组装')

            K, F = self.apply_bc(K0, F0)
            if enable_timing:
                t.send('边界条件处理')

            uh = self._tensor_space.function()

        solver_type = kwargs.get('solver', self._solve_method)

        if solver_type in ['mumps', 'scipy']:
            from fealpy.solver import spsolve

            uh[:] = spsolve(K, F, solver=solver_type)

        elif solver_type in ['cg']:
            from fealpy.solver import cg

            maxiter = kwargs.get('maxiter', 5000)
            atol = kwargs.get('atol', 1e-12)
            rtol = kwargs.get('rtol', 1e-12)
            x0 = kwargs.get('x0', None)
            
            # cg 支持批量求解, batch_first 为 False 时, 表示第一个维度为自由度维度
            #? matmul 函数下 K 必须是 COO 格式, 不能是 CSR 格式, 否则 GPU 下 device_put 函数会出错
            uh[:], info = cg(K.tocoo(), F[:], x0=x0,
                            batch_first=False, 
                            atol=atol, rtol=rtol, 
                            maxit=maxiter, returninfo=True)

        else:
            self._log_error(f"未知的求解器类型: {solver_type}")
        
        if enable_timing:
            t.send('求解')
            t.send(None)

        return {'displacement': uh}
    
    def solve_adjoint(self, 
                    rhs: TensorLike,
                    rho_val: Optional[Union[TensorLike, Function]] = None,
                    **kwargs
                ) -> TensorLike:
        """
        求解伴随方程 K @ λ = rhs
        v
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

        # 获取 Dirichlet 边界自由度
        gd = self._pde.dirichlet_bc
        threshold = self._pde.is_dirichlet_boundary()
        _, isBdDof = self._tensor_space.boundary_interpolate(
                                        gd=gd,
                                        threshold=threshold,
                                        method='interp'
                                    )
        
        # 先处理右端项 (伴随问题边界条件为齐次, λ = 0)
        rhs_bc = bm.copy(rhs)
        rhs_bc[isBdDof, :] = 0.0

        # 再处理刚度矩阵
        K = self._apply_matrix(A=K0, isDDof=isBdDof)
        
        # 初始化结果
        adjoint_lambda = bm.zeros_like(rhs_bc)
        
        # 求解
        solver_type = kwargs.get('solver', self._solve_method)
        
        if solver_type in ['mumps', 'scipy']:
            from fealpy.solver import spsolve
            adjoint_lambda[:] = spsolve(K, rhs_bc, solver=solver_type)
            
        elif solver_type in ['cg']:
            from fealpy.solver import cg
            maxiter = kwargs.get('maxiter', 5000)
            atol = kwargs.get('atol', 1e-12)
            rtol = kwargs.get('rtol', 1e-12)
            
            adjoint_lambda[:], _ = cg(K.tocoo(), rhs_bc, 
                                    batch_first=False,
                                    atol=atol, rtol=rtol, 
                                    maxit=maxiter, returninfo=True)
        
        return adjoint_lambda

    
    ###############################################################################################
    # 外部方法
    ###############################################################################################

    def compute_solid_stiffness_matrix(self):
        """计算实体材料的刚度矩阵"""
        lea = LinearElasticIntegrator(material=self._material,
                            coef=None,
                            q=self._integration_order,
                            method=self._assembly_method)
        ke0 = lea.assembly(space=self.tensor_space)

        self._cached_ke0 = ke0

        return ke0

    def compute_stiffness_matrix_derivative(self, rho_val: Union[TensorLike, Function]) -> TensorLike:
        """计算局部刚度矩阵关于物理密度的导数 (灵敏度)"""
        density_location = self._interpolation_scheme.density_location

        # TODO 目前仅支持插值杨氏模量 E 
        dE_rho = self._interpolation_scheme.interpolate_material_derivative(
                                                material=self._material, 
                                                rho_val=rho_val,
                                                integration_order=self._integration_order,
                                            ) 
        
        if density_location in ['element']:
            # rho_val.shape = (NC, )
            diff_coef_element = dE_rho / self._material.youngs_modulus # (NC, )

            if self._cached_ke0 is None:
                ke0 = self.compute_solid_stiffness_matrix()
            
            ke0 = self._cached_ke0

            diff_ke = bm.einsum('c, cij -> cij', diff_coef_element, ke0) # (NC, TLDOF, TLDOF)
 
            return diff_ke
        
        elif density_location in ['element_multiresolution']:
            # rho_val.shape = (NC, n_sub)
            diff_coef_sub_element = dE_rho / self._material.youngs_modulus # (NC, n_sub)

            mesh_u = self._mesh
            s_space_u = self._scalar_space
            q = self._integration_order
            NC, n_sub = rho_val.shape
            GD = mesh_u.geo_dimension()

            # 计算位移单元 (父参考单元) 高斯积分点处的重心坐标
            qf_e = mesh_u.quadrature_formula(q)
            # bcs_e.shape = ( (NQ_x, GD), (NQ_y, GD) ), ws_e.shape = (NQ, )
            bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()
            NQ = ws_e.shape[0]

            # 把位移单元高斯积分点处的重心坐标映射到子密度单元 (子参考单元) 高斯积分点处的重心坐标 (仍表达在位移单元中)
            from soptx.analysis.utils import map_bcs_to_sub_elements
            # bcs_eg.shape = ( (n_sub, NQ_x, GD), (n_sub, NQ_y, GD) ), ws_e.shape = (NQ, )
            bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
            bcs_eg_x, bcs_eg_y = bcs_eg

            # 计算子密度单元内高斯积分点处的基函数梯度和 jacobi 矩阵
            LDOF = s_space_u.number_of_local_dofs()
            gphi_eg = bm.zeros((NC, n_sub, NQ, LDOF, GD)) # (NC, n_sub, NQ, LDOF, GD)
            detJ_eg = None

            if isinstance(mesh_u, SimplexMesh):
                for s_idx in range(n_sub):
                    sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ_x, GD), (NQ_y, GD))
                    gphi_sub = s_space_u.grad_basis(sub_bcs, variable='x')    # (NC, NQ, LDOF, GD)
                    gphi_eg[:, s_idx, :, :, :] = gphi_sub

            else:
                detJ_eg = bm.zeros((NC, n_sub, NQ)) # (NC, n_sub, NQ)
                for s_idx in range(n_sub):
                    sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ_x, GD), (NQ_y, GD))
                    gphi_sub = s_space_u.grad_basis(sub_bcs, variable='x') # (NC, NQ, LDOF, GD)
                    J_sub = mesh_u.jacobi_matrix(sub_bcs) # (NC, NQ, GD, GD)
                    detJ_sub = bm.abs(bm.linalg.det(J_sub)) # (NC, NQ)
                    gphi_eg[:, s_idx, :, :, :] = gphi_sub
                    detJ_eg[:, s_idx, :] = detJ_sub

            # 计算 B 矩阵
            from soptx.analysis.utils import reshape_multiresolution_data, reshape_multiresolution_data_inverse
            nx_u, ny_u = mesh_u.meshdata['nx'], mesh_u.meshdata['ny']
            gphi_eg_reshaped = reshape_multiresolution_data(nx=nx_u, ny=ny_u, data=gphi_eg) # (NC*n_sub, NQ, LDOF, GD)
            B_eg_reshaped = self._material.strain_displacement_matrix(
                                                dof_priority=self._tensor_space.dof_priority, 
                                                gphi=gphi_eg_reshaped
                                            ) # (NC*n_sub, NQ, NS, TLDOF)
            B_eg = reshape_multiresolution_data_inverse(nx=nx_u, 
                                                        ny=ny_u, 
                                                        data_flat=B_eg_reshaped, 
                                                        n_sub=n_sub) # (NC, n_sub, NQ, NS, TLDOF)

            # 位移单元 → 子密度单元的缩放
            J_g = 1 / n_sub

            # 计算 D 矩阵的导数
            D0 = self._material.elastic_matrix()[0, 0] # (NS, NS)
            diff_D_g = bm.einsum('kl, cn -> cnkl', D0, diff_coef_sub_element) # (NC, n_sub, NS, NS)

            # 数值积分
            # diff_ke - (NC, n_sub, TLDOF, TLDOF)
            if isinstance(mesh_u, SimplexMesh):
                cm = mesh_u.entity_measure('cell')
                cm_eg = bm.tile(cm.reshape(NC, 1), (1, n_sub)) / n_sub # (NC, n_sub)
                diff_ke = J_g * bm.einsum('q, cn, cnqki, cnkl, cnqlj -> cnij',
                                    ws_e, cm_eg, B_eg, diff_D_g, B_eg)
            else:
                diff_ke = J_g * bm.einsum('q, cnq, cnqki, cnkl, cnqlj -> cnij',
                                    ws_e, detJ_eg, B_eg, diff_D_g, B_eg)

            return diff_ke
        
        elif density_location in ['node']:
            # rho_val.shape = (NN, )
            diff_coef_q = dE_rho / self._material.youngs_modulus # (NC, NQ)
            mesh = self._mesh
            qf = mesh.quadrature_formula(q=self._integration_order)
            # bcs_e.shape = ( (NQ_x, GD), (NQ_y, GD) ), ws_e.shape = (NQ, )
            bcs, ws = qf.get_quadrature_points_and_weights()
            
            rho_space = rho_val.space
            u_space = self._scalar_space

            # 高斯积分点处的基函数
            phi = rho_space.basis(bcs)[0] # (NQ, NCN)

            D0 = self._material.elastic_matrix()[0, 0] # (NS, NS)
            dof_priority = self._tensor_space.dof_priority
            gphi = u_space.grad_basis(bcs, variable='x') # (NC, NQ, LDOF, GD)
            B = self._material.strain_displacement_matrix(
                                    dof_priority=dof_priority, 
                                    gphi=gphi
                                ) # (NC, NQ, NS, TLDOF)
            BDB = bm.einsum('cqki, kl, cqlj -> cqij', B, D0, B) # (NC, NQ, TLDOF, TLDOF)

            if isinstance(mesh, SimplexMesh):
                cm = mesh.entity_measure('cell')
                kernel = bm.einsum('q, c, cq, cqij -> cqij', ws, cm, diff_coef_q, BDB)
            else:
                J = mesh.jacobi_matrix(bcs)
                detJ = bm.abs(bm.linalg.det(J))
                kernel = bm.einsum('q, cq, cq, cqij -> cqij', ws, detJ, diff_coef_q, BDB)

            diff_ke = bm.einsum('cqij, ql -> clij', kernel, phi) # (NC, NCN, TLDOF, TLDOF)

            return diff_ke
        
    def compute_strain_displacement_matrix(self, integration_order: Optional[int] = None) -> TensorLike:
        """
        计算应变-位移矩阵 B
        
        Parameters
        ----------
        integration_order : 积分阶次，默认使用分析器的积分阶次
        
        Returns
        -------
        B : 应变-位移矩阵
            - 单分辨率: (NC, NQ, NS, TLDOF)
            - 多分辨率: (NC, n_sub, NQ, NS, TLDOF)
        """
        if integration_order is None:
            integration_order = self._integration_order
        
        density_location = self._interpolation_scheme.density_location
        
        if density_location in ['element']:
            qf = self._mesh.quadrature_formula(integration_order)
            bcs, _ = qf.get_quadrature_points_and_weights()
            gphi = self._scalar_space.grad_basis(bcs, variable='x')  # (NC, NQ, LDOF, GD)
            B = self._material.strain_displacement_matrix(
                                                dof_priority=self._tensor_space.dof_priority, 
                                                gphi=gphi
                                            )  # (NC, NQ, NS, TLDOF)
            
        elif density_location in ['element_multiresolution']:
            nx_u = self._mesh.meshdata['nx']
            ny_u = self._mesh.meshdata['ny']
            n_sub = 4
            
            from soptx.interpolation.utils import (
                                        calculate_multiresolution_gphi_eg, 
                                        reshape_multiresolution_data_inverse
                                    )
            gphi_eg_reshaped = calculate_multiresolution_gphi_eg(
                                                        s_space_u=self._scalar_space,
                                                        q=integration_order,
                                                        n_sub=n_sub
                                                    )  # (NC*n_sub, NQ, LDOF, GD)
            
            B_reshaped = self._material.strain_displacement_matrix(
                                                        dof_priority=self._tensor_space.dof_priority, 
                                                        gphi=gphi_eg_reshaped
                                                    )  # (NC*n_sub, NQ, NS, TLDOF)
                                                    
            B = reshape_multiresolution_data_inverse(
                                                nx=nx_u, ny=ny_u, 
                                                data_flat=B_reshaped, 
                                                n_sub=n_sub
                                            )  # (NC, n_sub, NQ, NS, TLDOF)
            
        else:
            self._log_error(f"不支持的密度位置类型: {density_location}")
        
        return B


    def compute_stress_state(self, 
                            state: dict,
                            rho_val: Optional[Union[TensorLike, Function]] = None,
                            integration_order: Optional[int] = None
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
            - 'stress_penalized': 惩罚后应力
            - 'von_mises': von Mises 等效应力 (NC, NQ) 或 (NC, n_sub, NQ)
            - 'von_mises_max': 每个单元的最大 von Mises 应力 (NC,)
        """        
        if integration_order is None:
            integration_order = self._integration_order

        if state is None:
            state = self.solve_state(rho_val=rho_val)
        
        uh = state['displacement']
        cell2dof = self._tensor_space.cell_to_dof()
        uh_e = uh[cell2dof]  # (NC, TLDOF)

        B = self.compute_strain_displacement_matrix(integration_order)
        
        # 计算实体应力 (与密度无关)
        stress_solid = self._material.calculate_stress_vector(B, uh_e)
        
        result = {'stress_solid': stress_solid}
        
        # 应力惩罚（拓扑优化场景）
        if self._topopt_algorithm == 'density_based':
            if rho_val is None:
                self._log_warning("拓扑优化模式下建议提供 rho_val 以计算惩罚后应力")
                stress_for_vm = stress_solid
            else:
                penalty_data = self._interpolation_scheme.interpolate_stress(
                                                                        stress_solid=stress_solid,
                                                                        rho_val=rho_val,
                                                                        return_stress_penalty=True,
                                                                    )
                stress_for_vm = penalty_data['stress_penalized']
                result['stress_penalized'] = stress_for_vm
                result['eta_sigma'] = penalty_data['eta_sigma']
        else:
            stress_for_vm = stress_solid
        
        # 计算 von Mises 应力
        von_mises = self._material.calculate_von_mises_stress(stress_vector=stress_for_vm)
        result['von_mises'] = von_mises
        
        # 每个单元取最大值
        if von_mises.ndim == 2:
            # 单分辨率: (NC, NQ) -> (NC,)
            von_mises_max = bm.max(von_mises, axis=1)
        elif von_mises.ndim == 3:
            # 多分辨率: (NC, n_sub, NQ) -> (NC,)
            von_mises_max = bm.max(von_mises.reshape(von_mises.shape[0], -1), axis=1)
        else:
            self._log_error(f"意外的 von Mises 应力维度: {von_mises.ndim}")
        
        result['von_mises_max'] = von_mises_max
        
        return result
    
    ##############################################################################################
    # 内部方法
    ##############################################################################################


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

        # 找出所有边界DOF对应的行索引
        bd_rows = bm.where(loc_flag)[0]
        bd_rows = bm.astype(bd_rows, col.dtype)
        new_col = bm.empty((NNZ,), **indices_context)
        # 设置为相应行的边界 DOF 位置
        new_col = bm.set_at(new_col, new_crow[:-1][loc_flag], bd_rows)
        # 设置非对角元素的列索引
        new_col = bm.set_at(new_col, non_diag, col[remain_flag])

        new_values = bm.empty((NNZ,), **A.values_context())
        new_values = bm.set_at(new_values, new_crow[:-1][loc_flag], 1.)
        new_values = bm.set_at(new_values, non_diag, A.values[remain_flag])

        return CSRTensor(new_crow, new_col, new_values, A.sparse_shape)