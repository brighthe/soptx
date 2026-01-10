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
                mesh: HomogeneousMesh,
                pde: PDEBase, 
                material: LinearElasticMaterial,
                space_degree: int = 1,
                integration_order: int = 4,
                use_relaxation: bool = True,
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
        self._use_relaxation = use_relaxation

        self._topopt_algorithm = topopt_algorithm
        self._interpolation_scheme = interpolation_scheme

        self._solve_method = solve_method

        self._GD = self._mesh.geo_dimension()
        self._huzhang_space = HuZhangFESpace(mesh=self._mesh, p=self._space_degree, use_relaxation=self._use_relaxation)
        self._scalar_space = LagrangeFESpace(mesh=self._mesh, p=self._space_degree-1, ctype='D')
        self._tensor_space = TensorFunctionSpace(scalar_space=self._scalar_space, shape=(-1, self._GD))

        lambda0, lambda1 = self._stress_matrix_coefficient()
        self._hzs_integrator = HuZhangStressIntegrator(
                                            lambda0=lambda0, 
                                            lambda1=lambda1, 
                                            q=self._integration_order, 
                                            method='fast'
                                        )
        self._hzs_integrator.keep_data(True)
        self._cached_stress_matrix = None

        self._cached_mix_matrix = self._calculate_mix_matrix()


    ##############################################################################################
    # 属性相关函数
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
        """获取当前的位移有限元空间"""
        return self._tensor_space
    
    @property
    def material(self) -> LinearElasticMaterial:
        """获取当前的材料类"""
        return self._material
    
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

        if self._topopt_algorithm is None:
            if rho_val is not None:
                self._log_warning("标准混合有限元分析模式下忽略相对密度分布参数 rho")
            coef = None

        elif self._topopt_algorithm in ['density_based']:
            E_rho = self._interpolation_scheme.interpolate_material(
                                            material=self._material,
                                            rho_val=rho_val,
                                            integration_order=self._integration_order
                                        )
            E0 = self.material.youngs_modulus
            coef = E0 / E_rho

        else:
            error_msg = f"不支持的拓扑优化算法: {self._topopt_algorithm}"
            self._log_error(error_msg)

        # 更新密度系数
        self._hzs_integrator.coef = coef

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
            # jpi_integrator = JumpPenaltyIntegrator(q=self._integration_order, threshold=None, method='vector_jump')
            jpi_integrator = JumpPenaltyIntegrator(q=self._integration_order, threshold=None, method='matrix_jump')
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
        """组装体力源项向量 (f, v)"""
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

        if 'natural_bc' in mesh.edgedata:
            disp_edge_flag = mesh.edgedata['natural_bc']
        else:
            bc_edge = mesh.entity_barycenter('edge')
            disp_edge_flag = self._pde.is_displacement_boundary(bc_edge)
        
        bdedge = bd_edge_flag & disp_edge_flag

        if 'essential_bc' in mesh.edgedata:
            bdedge = bdedge & (~mesh.edgedata['essential_bc'])

        NBF = int(bdedge.sum())
        if NBF == 0:
            return bm.zeros(gdof, dtype=bm.float64, device=space.device)

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
            self._log_error("PDE 对象缺少位移边界函数 displacement_bc")
        gd = self._pde.displacement_bc

        for i in range(3):
            # 找到局部编号为 i 的面
            flag = (e2c[:, 2] == i)
            if not bm.any(flag):
                continue
                
            # 计算基函数
            current_cells = e2c[flag, 0]
            phi = space.basis(bcsi[i], index=current_cells)

            # 计算 tau · n
            en_curr = en[flag, None, None, :]
            phi_n[flag, ..., 0] = bm.sum(phi[..., symidx[0]] * en_curr, axis=-1)
            phi_n[flag, ..., 1] = bm.sum(phi[..., symidx[1]] * en_curr, axis=-1)

            # 计算边界函数值 u_D
            points = mesh.bc_to_point(bcsi[i], index=current_cells)
            gd_val[flag] = gd(points)
        
        # 组装
        val = bm.einsum('q, c, cqld, cqd -> cl', ws, edge_measure, phi_n, gd_val)
        
        cell2dof = space.cell_to_dof()[e2c[:, 0]]
        F_vec = bm.zeros(gdof, dtype=bm.float64, device=space.device)
        bm.add.at(F_vec, cell2dof, val)

        # TODO 角点松弛变换
        if space.use_relaxation == True:
            F_vec = space.TM.T @ F_vec

        if enable_timing:
            t.send("组装完成")
            t.send(None)

        return F_vec
    
    def apply_traction_boundary_condition(self, K: CSRTensor, F: TensorLike, space_sigma):
        """施加本质边界条件  σ·n = g_N"""
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

        # TODO 注释
        # mesh.edgedata['dirichlet'] = pde.is_traction_boundary(bc)   # 应力是本质边界条件
        # mesh.edgedata['neumann'] = pde.is_displacement_boundary(bc) # 位移是自然边界条件
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

        # 本质边界条件处理 (应力边界 σ·n = g_N)
        # if self._pde.boundary_type == 'neumann':
        #     K, F = K0, F0
        # else:
        #     K, F = self.apply_traction_boundary_condition(K0, F0, space_sigmah)
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


    ###############################################################################################
    # 外部方法
    ###############################################################################################

    def compute_local_stress_matrix_derivative(self, rho_val: Union[TensorLike, Function]) -> TensorLike:
        """计算局部应力矩阵 A 关于物理密度的导数（灵敏度）"""
        density_location = self._interpolation_scheme.density_location

        # TODO 目前仅支持插值杨氏模量 E
        E0 = self._material.youngs_modulus
        E_rho =  self._interpolation_scheme.interpolate_material(
                                                material=self._material, 
                                                rho_val=rho_val,
                                                integration_order=self._integration_order,
                                            )
        dE_rho = self._interpolation_scheme.interpolate_material_derivative(
                                                material=self._material, 
                                                rho_val=rho_val,
                                                integration_order=self._integration_order,
                                            ) 
        space_sigma = self._huzhang_space

        if density_location in ['element']:
            # rho_val: (NC, )
            diff_coef_element = - E0 * dE_rho / (E_rho**2) # (NC, )

            AE0 = self._hzs_integrator.fetch_fast(space_sigma)

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