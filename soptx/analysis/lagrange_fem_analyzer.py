from typing import Optional, Union, Literal

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import SimplexMesh, HomogeneousMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace, Function
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import VectorSourceIntegrator
from fealpy.decorator import variantmethod
from fealpy.sparse import CSRTensor, COOTensor

from ..interpolation.linear_elastic_material import LinearElasticMaterial
from ..interpolation.interpolation_scheme import MaterialInterpolationScheme
from .integrators.linear_elastic_integrator import LinearElasticIntegrator
from soptx.model.pde_base import PDEBase
from ..utils.base_logged import BaseLogged

class LagrangeFEMAnalyzer(BaseLogged):
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
       
        self._space_degree = space_degree
        self._integration_order = integration_order
        self._assembly_method = assembly_method

        self._topopt_algorithm = topopt_algorithm
        self._interpolation_scheme = interpolation_scheme

        # 设置默认求解方法
        self.solve_displacement.set(solve_method)

        self._GD = self._mesh.geo_dimension()

        #* (GD, -1): dof_priority (x0, ..., xn, y0, ..., yn)
        # self._setup_function_spaces(mesh=self._mesh, 
        #                             p=self._space_degree, 
        #                             shape=(self._GD, -1))
        #* (-1, GD): gd_priority (x0, y0, ..., xn, yn)
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
    def material(self) -> LinearElasticMaterial:
        """获取当前的材料类"""
        return self._material
    
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
    

    ##############################################################################################
    # 属性修改器 - 修改内部状态
    ##############################################################################################

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

        Returns
        -------
        Union[CSRTensor, COOTensor]
            组装后的刚度矩阵
        """

        if self._topopt_algorithm is None:
            if rho_val is not None:
                self._log_warning("标准有限元分析模式下忽略相对密度 rho")
            
            coef = None
        
        elif self._topopt_algorithm == 'density_based':
            if rho_val is None:
                self._log_error("基于密度的拓扑优化算法需要提供相对密度 rho")

            # TODO 目前仅支持插值杨氏模量 E 
            E_rho = self._interpolation_scheme.interpolate_map(
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

        # TODO 这里的 coef 也和材料有关, 可能需要进一步处理,
        # TODO coef 是应该在 LinearElasticIntegrator 中, 还是在 MaterialInterpolationScheme 中处理 ?
        integrator = LinearElasticIntegrator(material=self._material,
                                            coef=coef,
                                            q=self._integration_order,
                                            method=self._assembly_method)
        bform = BilinearForm(self._tensor_space)
        bform.add_integrator(integrator)
        K = bform.assembly(format='csr')

        self._K = K

        return K
    
    def assemble_spring_stiff_matrix(self):
        """组装弹簧刚度矩阵"""
        tspace = self._tensor_space
        TGDOF = tspace.number_of_global_dofs()

        k_in = self._pde.k_in
        k_out = self._pde.k_out

        isBdDof = tspace.is_boundary_dof(threshold=self._pde.is_spring_boundary(), method='interp')
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
        integrator = VectorSourceIntegrator(source=body_force, q=self._integration_order)
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
                gd_adjoint = self._pde.adjoint_bc
                threshold_adjoint = self._pde.is_adjoint_boundary()

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

            # 2. Dirichlet 边界条件处理 - 强形式施加
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
                F = F - K.matmul(uh_bd[:])
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
        
    
    ###############################################################################################
    # 外部方法
    ###############################################################################################

    def get_stiffness_matrix_derivative(self, rho_val: Union[TensorLike, Function]) -> TensorLike:
        """计算局部刚度矩阵关于物理密度的导数（灵敏度）"""
        
        density_location = self._interpolation_scheme.density_location

        # TODO 目前仅支持插值杨氏模量 E 
        dE_rho = self._interpolation_scheme.interpolate_map_derivative(
                                                material=self._material, 
                                                rho_val=rho_val,
                                                integration_order=self._integration_order,
                                            ) 
        
        if density_location in ['element']:
            # rho_val.shape = (NC, )
            diff_coef_element = dE_rho / self._material.youngs_modulus # (NC, )

            lea = LinearElasticIntegrator(material=self._material,
                                        coef=None,
                                        q=self._integration_order,
                                        method=self._assembly_method)
            ke0 = lea.assembly(space=self.tensor_space)
            diff_ke = bm.einsum('c, cij -> cij', diff_coef_element, ke0) # (NC, TLDOF, TLDOF)
 
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
        
        elif density_location in ['node_multiresolution']:
            # TODO 存在疑问
            rho_q_eg = rho_val  # (NC, n_sub, NQ)
            diff_coef = self._interpolation_scheme.interpolate_derivative(
                                                material=self._material, 
                                                density_distribution=rho_q_eg
                                            ) # (NC, n_sub, NQ)

            mesh_u = self._mesh
            s_space_u = self._scalar_space
            q = self._integration_order
            NC, n_sub, NQ = rho_q_eg.shape
            GD = mesh_u.geo_dimension()

            # 计算位移单元 (父参考单元) 高斯积分点处的重心坐标
            qf_e = mesh_u.quadrature_formula(q)
            # bcs_e.shape = ( (NQ, GD), (NQ, GD) ), ws_e.shape = (NQ, )
            bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()

            # 把位移单元高斯积分点处的重心坐标映射到子密度单元 (子参考单元) 高斯积分点处的重心坐标 (仍表达在位移单元中)
            from soptx.analysis.utils import map_bcs_to_sub_elements
            # bcs_eg.shape = ( (n_sub, NQ, GD), (n_sub, NQ, GD) ), ws_e.shape = (NQ, )
            bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
            bcs_eg_x, bcs_eg_y = bcs_eg

            # 计算子密度单元内高斯积分点处的基函数梯度和 jacobi 矩阵
            LDOF = s_space_u.number_of_local_dofs()
            gphi_eg = bm.zeros((NC, n_sub, NQ, LDOF, GD)) # (NC, n_sub, NQ, LDOF, GD)
            detJ_eg = None

            if isinstance(mesh_u, SimplexMesh):
                for s_idx in range(n_sub):
                    sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ, GD), (NQ, GD))
                    gphi_sub = s_space_u.grad_basis(sub_bcs, variable='x')      # (NC, NQ, LDOF, GD)
                    gphi_eg[:, s_idx, :, :, :] = gphi_sub

            else:
                detJ_eg = bm.zeros((NC, n_sub, NQ)) # (NC, n_sub, NQ)
                for s_idx in range(n_sub):
                    sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ, GD), (NQ, GD))

                    gphi_sub = s_space_u.grad_basis(sub_bcs, variable='x') # (NC, NQ, LDOF, GD)

                    J_sub = mesh_u.jacobi_matrix(sub_bcs) # (NC, NQ, GD, GD)
                    detJ_sub = bm.abs(bm.linalg.det(J_sub)) # (NC, NQ)

                    gphi_eg[:, s_idx, :, :, :] = gphi_sub
                    detJ_eg[:, s_idx, :] = detJ_sub

            # 计算 B 矩阵
            gphi_eg_reshaped = gphi_eg.reshape(NC * n_sub, NQ, LDOF, GD)
            B_eg_reshaped = self._material.strain_displacement_matrix(dof_priority=self.tensor_space.dof_priority, 
                                                            gphi=gphi_eg_reshaped) # 2D: (NC*n_sub, NQ, 3, TLDOF), 3D: (NC*n_sub, NQ, 6, TLDOF)
            B_eg = B_eg_reshaped.reshape(NC, n_sub, NQ, B_eg_reshaped.shape[-2], B_eg_reshaped.shape[-1])

            # 位移单元 → 子密度单元的缩放
            J_g = 1 / n_sub

            D0 = self._material.elastic_matrix()[0, 0] # 2D: (3, 3); 3D: (6, 6)
            BDB_eg = bm.einsum('cnqki, kl, cnqlj -> cnqij', B_eg, D0, B_eg) # (NC, n_sub, NQ, TLDOF, TLDOF)

            # 计算子密度单元内高斯积分点处基函数的值
            NCN = int(mesh_u.number_of_vertices_of_cells())
            phi_eg = bm.zeros((n_sub, NQ, NCN))

            for s_idx in range(n_sub):
                sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])

                phi_sub = s_space_u.basis(sub_bcs)[0] # (NQ, NCN)
                phi_eg[s_idx, :, :] = phi_sub 

            # 验证密度值范围
            if (bm.any(bm.isnan(rho_q_eg[:])) or bm.any(bm.isinf(rho_q_eg[:])) or 
                bm.any(rho_q_eg[:] < -1e-12) or bm.any(rho_q_eg[:] > 1 + 1e-12)):
                self._log_error(f"子单元高斯积分点处的节点密度值超出范围 [0, 1]: "
                                f"range=[{bm.min(rho_q_eg):.2e}, {bm.max(rho_q_eg):.2e}]")

            # 验证形函数范围
            if (bm.any(bm.isnan(phi_eg[:])) or bm.any(bm.isinf(phi_eg[:])) or 
                bm.any(phi_eg[:] < -1e-12) or bm.any(phi_eg[:] > 1 + 1e-12)):
                self._log_error(f"子单元的密度形函数超出范围 [0, 1]: "
                                f"range=[{bm.min(phi_eg):.2e}, {bm.max(phi_eg):.2e}]")

            diff_coef_q_eg = self._interpolation_scheme.interpolate_derivative(
                                                                material=self._material, 
                                                                density_distribution=rho_q_eg) # (NC, n_sub, NQ)
            
            # 数值积分
            if isinstance(mesh_u, SimplexMesh):
                cm = mesh_u.entity_measure('cell')
                cm_eg = bm.tile(cm.reshape(NC, 1), (1, n_sub)) / n_sub # (NC, n_sub)
                kernel = bm.einsum('q, cn, cnq, cnqij -> cnqij', ws_e, cm_eg, diff_coef_q_eg, BDB_eg)
            else:
                kernel = bm.einsum('q, cnq, cnq, cnqij -> cnqij', ws_e, detJ_eg, diff_coef_q_eg, BDB_eg)
            
            # 应用缩放因子
            kernel = J_g * kernel
            
            # 乘以密度形函数得到对每个节点的灵敏度
            diff_ke_sub = bm.einsum('cnqij, nql -> cnlij', kernel, phi_eg) # (NC, n_sub, NCN, TLDOF, TLDOF)
            diff_ke = bm.sum(diff_ke_sub, axis=1) # (NC, NCN, TLDOF, TLDOF)

            return diff_ke


    ###############################################################################################
    # 变体方法
    ###############################################################################################

    @variantmethod('mumps')
    def solve_displacement(self, 
                        rho_val: Optional[Union[TensorLike, Function]] = None,
                        adjoint: bool = False, 
                        **kwargs
                    ) -> Function:
        
        from fealpy.solver import spsolve

        if self._topopt_algorithm is None:
            if rho_val is not None:
                self._log_warning("标准有限元分析模式下忽略密度分布参数 rho")
        
        elif self._topopt_algorithm in ['density_based', 'level_set']:
            if rho_val is None:
                error_msg = f"拓扑优化算法 '{self._topopt_algorithm}' 需要提供密度分布参数 rho"
                self._log_error(error_msg)

        solver_type = kwargs.get('solver', 'mumps')

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
            F0 = self.assemble_body_force_vector()
            K, F = self.apply_bc(K0, F0)

            uh = self._tensor_space.function()
        
        uh[:] = spsolve(K, F, solver=solver_type)

        gdof = self._tensor_space.number_of_global_dofs()
        self._log_info(f"Solving linear system with {gdof} displacement DOFs with MUMPS solver.")

        return uh
    
    @solve_displacement.register('cg')
    def solve_displacement(self, 
                        rho_val: Optional[Union[TensorLike, Function]] = None,
                        adjoint: bool = False, 
                        **kwargs
                    ) -> Function:
        
        from fealpy.solver import cg

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

            uh = self._tensor_space.function()
        
        else:
            K0 = self.assemble_stiff_matrix(rho_val=rho_val)
            F0 = self.assemble_body_force_vector()
            K, F = self.apply_bc(K0, F0)

            uh = bm.zeros(F.shape, dtype=bm.float64, device=F.device)

        maxiter = kwargs.get('maxiter', 5000)
        atol = kwargs.get('atol', 1e-12)
        rtol = kwargs.get('rtol', 1e-12)
        x0 = kwargs.get('x0', None)
        
        #! cg 支持批量求解, batch_first 为 False 时, 表示第一个维度为自由度维度
        uh[:], info = cg(K, F, x0=x0,
                        batch_first=False, 
                        atol=atol, rtol=rtol, 
                        maxit=maxiter, returninfo=True)
        
        gdof = self._tensor_space.number_of_global_dofs()
        self._log_info(f"Solving linear system with {gdof} displacement DOFs with CG solver.")

        return uh

    
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
        scalar_space = LagrangeFESpace(mesh, p=p, ctype='C')
        self._scalar_space = scalar_space

        tensor_space = TensorFunctionSpace(scalar_space=scalar_space, shape=shape)
        self._tensor_space = tensor_space

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