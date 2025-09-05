from typing import Optional, Dict, Any, Literal, List, Union

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod, cartesian
from fealpy.typing import TensorLike, Tuple
from fealpy.functionspace import Function, LagrangeFESpace, BernsteinFESpace
from fealpy.mesh import HomogeneousMesh

from .linear_elastic_material import LinearElasticMaterial
from ..utils.base_logged import BaseLogged

class MaterialInterpolationScheme(BaseLogged):
    """材料插值方案类"""
    def __init__(self,
                density_location: Literal['element', 'element_coscos', 
                                          'gauss_integration_point', 
                                          'lagrange_interpolation_point'] = 'element',
                interpolation_method: Literal['simp', 'msimp', 'ramp'] = 'simp',
                options: Optional[dict] = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        """
        材料插值方法变体示例
        """
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._density_location = density_location
        self._interpolation_method = interpolation_method

        self._options = options or {}
        self._set_default_options()

        self.setup_density_distribution.set(density_location)
        self.interpolate_map.set(interpolation_method)

        self._log_info(f"Material interpolation scheme initialized: "
                      f"density_location={density_location}, "
                      f"interpolation_method={interpolation_method}, ")
        
    
    #########################################################################################
    # 属性访问器
    #########################################################################################

    @property
    def density_location(self) -> Optional[str]:
        """获取当前的密度位置"""
        return self._density_location

    @property
    def interpolation_method(self) -> Optional[str]:
        """获取当前的插值方法"""
        return self._interpolation_method

    
    #########################################################################################
    # 变体方法
    #########################################################################################

    @variantmethod('element')
    def setup_density_distribution(self, 
                                design_variable_mesh: HomogeneousMesh,
                                displacement_mesh: HomogeneousMesh,
                                relative_density: float = 1.0,
                                **kwargs,
                            ) -> Tuple[TensorLike, Function]:
        """
        单元密度-单分辨率 (SRTO), 设计变量就是单元密度, 自由度位于密度单元中心
        
        Returns:
        --------
        design_variable : (NC, )
        density_distribution : (NC, )
        """

        NC_design_variable = design_variable_mesh.number_of_cells()
        design_variable = bm.full((NC_design_variable, ), relative_density, dtype=bm.float64, device=design_variable_mesh.device)

        NC = displacement_mesh.number_of_cells()
        density_val = bm.full((NC, ), relative_density, dtype=bm.float64, device=displacement_mesh.device)
        space = LagrangeFESpace(displacement_mesh, p=0, ctype='D')
        density_distribution = space.function(density_val)

        return design_variable, density_distribution

    @setup_density_distribution.register('element_multiresolution')
    def setup_density_distribution(self, 
                            design_variable_mesh: HomogeneousMesh,
                            displacement_mesh: HomogeneousMesh,
                            relative_density: float = 1.0,
                            sub_density_element: int = 4,
                            **kwargs,
                        ) -> Tuple[TensorLike, Function]:
        """
        单元密度-多分辨率 (MRTO), 设计变量独立于有限元网格, 自由度位于子密度单元中心
        
        Returns:
        --------
        design_variable : (NC_design_variable, )
        density_distribution : (NC, n_sub)
        """

        NC_design_variable = design_variable_mesh.number_of_cells()
        NC = displacement_mesh.number_of_cells()
        n_sub = sub_density_element

        if NC_design_variable != NC * n_sub:
            error_msg = (f"Currently only support Nd = Ne * sub_density_element. "
                          f"Got Nd={NC_design_variable}, Ne={NC}, sub_density_element={n_sub}")
            self._log_error(error_msg)

        design_variable = bm.full((NC_design_variable, ), relative_density, 
                                dtype=bm.float64, device=design_variable_mesh.device)
        
        density_val = bm.full((NC, n_sub), relative_density, dtype=bm.float64, device=design_variable_mesh.device)
        space = LagrangeFESpace(displacement_mesh, p=0, ctype='D')
        density_distribution = space.function(density_val)

        return design_variable, density_distribution

    @setup_density_distribution.register('node')
    def setup_density_distribution(self,
                                   design_variable_mesh: HomogeneousMesh,
                                   displacement_mesh: HomogeneousMesh,
                                   relative_density: float = 1.0,
                                   integration_order: int = 3,
                                   **kwargs,
                                ) -> Tuple[TensorLike, TensorLike]:
        """
        节点密度-单分辨率 (SRTO), 设计变量就是节点密度, 自由度位于密度节点处
        
        Returns:
        --------
        design_variable : (NN, )
        density_distribution : (NC, NQ)
        """

        NN_design_variable = design_variable_mesh.number_of_nodes()
        design_variable = bm.full((NN_design_variable, ), relative_density, 
                                dtype=bm.float64, device=design_variable_mesh.device)

        NN = displacement_mesh.number_of_nodes()
        qf = displacement_mesh.quadrature_formula(q=integration_order)
        # bcs_e.shape = ( (NQ, GD), (NQ, GD) ), ws_e.shape = (NQ, )
        bcs, ws = qf.get_quadrature_points_and_weights()

        density_val = bm.full((NN, ), relative_density, dtype=bm.float64, device=design_variable_mesh.device)
        space = LagrangeFESpace(displacement_mesh, p=1, ctype='C')
        density = space.function(density_val)
        density_distribution = density(bcs) # (NC, NQ)

        return design_variable, density_distribution

    @setup_density_distribution.register('node_multiresolution')
    def setup_density_distribution(self, 
                            design_variable_mesh: HomogeneousMesh,
                            displacement_mesh: HomogeneousMesh,
                            relative_density: float = 1.0,
                            sub_density_element: int = 4,
                            integration_order: int = 3,
                            **kwargs,
                        ) -> Tuple[TensorLike, TensorLike]:
        """
        节点密度-多分辨率 (MRTO), 设计变量独立于有限元网格, 自由度位于子密度单元节点处
        
        Returns:
        --------
        design_variable : (NN_design_variable, )
        density_distribution : (NC, n_sub, NQ)
        """

        NN_design_variable = design_variable_mesh.number_of_nodes()
        design_variable = bm.full((NN_design_variable, ), relative_density, 
                                dtype=bm.float64, device=design_variable_mesh.device)
        
        # NC = displacement_mesh.number_of_cells()
        # n_sub = sub_density_element
        # qf_e = displacement_mesh.quadrature_formula(q=integration_order)
        # # bcs_e.shape = ( (NQ, GD), (NQ, GD) ), ws_e.shape = (NQ, )
        # bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()
        # NQ = ws_e.shape[0]
        # density_distribution = bm.full((NC, n_sub, NQ), 
        #                                relative_density, 
        #                                dtype=bm.float64, 
        #                                device=design_variable_mesh.device)

        # return design_variable, density_distribution

        s_space_u = LagrangeFESpace(displacement_mesh, p=1, ctype='C')
        NC = displacement_mesh.number_of_cells()
        qf_e = displacement_mesh.quadrature_formula(q=integration_order)
        # bcs_e.shape = ( (NQ, GD), (NQ, GD) ), ws_e.shape = (NQ, )
        bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()

        NN = displacement_mesh.number_of_nodes()
        density_val = bm.full((NN, ), relative_density, dtype=bm.float64, device=design_variable_mesh.device)
        density = s_space_u.function(density_val)
        n_sub = sub_density_element

        # 把位移单元高斯积分点处的重心坐标映射到子密度单元 (子参考单元) 高斯积分点处的重心坐标 (仍表达在位移单元中)
        from soptx.analysis.utils import map_bcs_to_sub_elements
        # bcs_eg.shape = ( (n_sub, NQ, GD), (n_sub, NQ, GD) ), ws_e.shape = (NQ, )
        bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
        bcs_eg_x, bcs_eg_y = bcs_eg

        NQ = ws_e.shape[0]
        density_distribution = bm.full((NC, n_sub, NQ), relative_density, dtype=bm.float64, device=design_variable_mesh.device)
        for s_idx in range(n_sub):
            sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])
            density_q_sub = density(sub_bcs) # (NC, NQ)
            density_distribution[:, s_idx, :] = density_q_sub

        return design_variable, density_distribution
    
    @setup_density_distribution.register('gauss_integration_point')
    def setup_density_distribution(self, 
                                mesh: HomogeneousMesh,
                                relative_density: float = 1.0,
                                integration_order: int = 3,
                                interpolation_order: int = None,
                                **kwargs,
                            ) -> TensorLike:
        """高斯积分点密度分布"""

        if integration_order is None:
            error_msg = "'gauss_integration_point' density distribution requires 'integration_order' parameter"
            self._log_error(error_msg)
        
        if interpolation_order is not None:
            warn_msg = f"'gauss_integration_point' density distribution does not require 'interpolation_order', provided interpolation_order={interpolation_order} will be ignored"
            self._log_warning(warn_msg)

        qf = mesh.quadrature_formula(integration_order)
        bcs, ws = qf.get_quadrature_points_and_weights()

        NC = mesh.number_of_cells()
        NQ = ws.shape[0]

        density_dist = bm.full((NC, NQ), relative_density, 
                            dtype=bm.float64, device=mesh.device)

        self._log_info(f"Element-Gauss density: shape={density_dist.shape}, value={relative_density}, q={integration_order}")

        return density_dist
    
    @setup_density_distribution.register('density_subelement_gauss_point')
    def setup_density_distribution(self, 
                                mesh: HomogeneousMesh,
                                relative_density: float = 1.0,
                                integration_order: int = 3,
                                subcells: int = None,
                                **kwargs,
                            ) -> TensorLike:
        """密度子单元（高斯点采样）分布"""
        
        qf = mesh.quadrature_formula(integration_order)
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ_gauss = ws.shape[0]
        
        # 检查子单元数量
        if subcells is None:
            subcells = NQ_gauss
            self._log_info(f"subcells not specified, using {subcells} (= number of Gauss points)")
        elif subcells != NQ_gauss:
            error_msg = (f"Currently only support subcells = number of Gauss points. "
                        f"Got subcells={subcells}, but integration_order={integration_order} "
                        f"has {NQ_gauss} Gauss points")
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        NC = mesh.number_of_cells()
        
        # 初始化密度分布：每个单元有 subcells 个子单元
        density_dist = bm.full((NC, subcells), relative_density, 
                            dtype=bm.float64, device=mesh.device)
        
        self._log_info(f"Density subelement (Gauss sampled): shape={density_dist.shape}, "
                    f"value={relative_density}, subcells={subcells}, q={integration_order}")
        
        return density_dist

    
    @setup_density_distribution.register('lagrange_interpolation_point')
    def setup_density_distribution(self,
                                   mesh: HomogeneousMesh,
                                   relative_density: float = 1.0,
                                   integration_order: int = None,
                                   interpolation_order: int = None,
                                   **kwargs,
                                ) -> Function:
        "节点密度 (Lagrange 插值)"
        if interpolation_order is None:
            error_msg = "'interpolation_point' density distribution requires 'interpolation_order' parameter"
            self._log_error(error_msg)
        
        if integration_order is not None:
            warn_msg = f"Interpolation point density distribution does not require 'integration_order', provided integration_order={integration_order} will be ignored"
            self._log_warning(warn_msg)

        space = LagrangeFESpace(mesh, p=interpolation_order, ctype='C')
        gdof = space.number_of_global_dofs()
        density_tensor = bm.full((gdof, ), relative_density, dtype=bm.float64, device=mesh.device)

        density_dist = space.function(density_tensor)

        self._log_info(f"Continuous density: shape={density_dist.shape}, value={relative_density}, p={interpolation_order}")
        
        return density_dist
    
    @setup_density_distribution.register('berstein_interpolation_point')
    def setup_density_distribution(self,
                                   mesh: HomogeneousMesh,
                                   relative_density: float = 1.0,
                                   integration_order: int = None,
                                   interpolation_order: int = None,
                                   **kwargs,
                                ) -> Function:
        "节点密度 (Berstein 插值)"
        if interpolation_order is None:
            error_msg = "'interpolation_point' density distribution requires 'interpolation_order' parameter"
            self._log_error(error_msg)
        
        if integration_order is not None:
            warn_msg = f"Interpolation point density distribution does not require 'integration_order', provided integration_order={integration_order} will be ignored"
            self._log_warning(warn_msg)

        space = BernsteinFESpace(mesh, p=interpolation_order, ctype='C')
        gdof = space.number_of_global_dofs()
        density_tensor = bm.full((gdof, ), relative_density, dtype=bm.float64, device=mesh.device)

        density_dist = space.function(density_tensor)

        self._log_info(f"Continuous density: shape={density_dist.shape}, value={relative_density}, p={interpolation_order}")
        
        return density_dist
    
    @setup_density_distribution.register('shepard_interpolation_point')
    def setup_density_distribution(self,
                                   mesh: HomogeneousMesh,
                                   relative_density: float = 1.0,
                                   integration_order: int = None,
                                   interpolation_order: int = 1,
                                   **kwargs,
                                ) -> Function:
        "节点密度 (Shepard 插值)"
        if interpolation_order is None:
            error_msg = "'interpolation_point' density distribution requires 'interpolation_order' parameter"
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        if integration_order is not None:
            warn_msg = f"Interpolation point density distribution does not require 'integration_order', provided integration_order={integration_order} will be ignored"
            self._log_warning(warn_msg)

        from soptx.interpolation.space import ShepardFESpace
        shepard_space = ShepardFESpace(mesh, p=interpolation_order, power=2.0)

        gdof = shepard_space.number_of_global_dofs()
        density_values = bm.full((gdof, ), relative_density, dtype=bm.float64, device=mesh.device)

        density_dist = shepard_space.function(density_values)
        
        return density_dist

    @variantmethod('simp')
    def interpolate_map(self, 
                    material: LinearElasticMaterial, 
                    density_distribution: Union[Function, TensorLike],
                ) -> TensorLike:
        """SIMP 插值: E(ρ) = ρ^p * E0"""

        penalty_factor = self._options['penalty_factor']
        target_variables = self._options['target_variables']

        if target_variables == ['E']:
            E0 = material.youngs_modulus
            simp_map = density_distribution[:] ** penalty_factor * E0 / E0

            if self._density_location in ['lagrange_interpolation_point', 
                                           'berstein_interpolation_point', 
                                           'shepard_interpolation_point']:
                # 如果是插值点密度分布, 则需要将结果转换为 Function
                density_space = density_distribution.space
                msimp_map = density_space.function(msimp_map)
            
            return simp_map
    
    @interpolate_map.register('msimp')
    def interpolate_map(self,
                    material: LinearElasticMaterial, 
                    rho_val: Union[Function, TensorLike],
                ) -> TensorLike:
        """修正 SIMP 插值: E(ρ) = Emin + ρ^p * (E0 - Emin)"""

        required_options = ['penalty_factor', 'target_variables', 'void_youngs_modulus']
        for option in required_options:
            if option not in self._options:
                error_msg = f"Missing required option '{option}' for msimp interpolation"
                self._log_error(error_msg)

        penalty_factor = self._options['penalty_factor']
        target_variables = self._options['target_variables']
        void_youngs_modulus = self._options['void_youngs_modulus']

        if target_variables == ['E']:

            E0 = material.youngs_modulus
            Emin = void_youngs_modulus
            msimp_map = (Emin + rho_val[:] ** penalty_factor * (E0 - Emin)) / E0

            # density_space = rho_val.space
            # msimp_map = density_space.function(msimp_map)

            # if self._density_location in ['lagrange_interpolation_point', 
            #                               'berstein_interpolation_point', 
            #                               'shepard_interpolation_point']:
            #     # 如果是插值点密度分布, 则需要将结果转换为 Function
            #     density_space = rho_val.space
            #     msimp_map = density_space.function(msimp_map)

            return msimp_map

    @interpolate_map.register('simp_double')
    def interpolate_map(self) -> TensorLike:
        """双指数 SIMP 插值"""
        pass

    @interpolate_map.register('ramp')
    def interpolate_map(self) -> TensorLike:
        """RAMP 插值"""
        pass


    ###########################################################################################################
    # 核心方法
    ###########################################################################################################

    def interpolate_derivative(self,
                        material: LinearElasticMaterial, 
                        density_distribution: Union[Function, TensorLike],
                    ) -> TensorLike:
        """获取当前插值方法的标量系数相对于物理密度的导数"""

        method = self.interpolation_method
        p = self._options['penalty_factor']

        if self._density_location in ['element']:

            rho_element = density_distribution[:] # (NC, )
            
            if method == 'simp':    
                dval = p * rho_element[:] ** (p - 1)
                return dval
            elif method == 'msimp':
                E0 = material.youngs_modulus
                Emin = self._options['void_youngs_modulus']
                dval = p * rho_element[:] ** (p - 1) * (E0 - Emin) / E0
                return dval
            
        elif self._density_location in ['element_multiresolution']:
            
            rho_sub_element = density_distribution[:] # (NC, n_sub)

            if method == 'simp':
                dval = p * rho_sub_element[:] ** (p - 1)
                return dval
            elif method == 'msimp':
                E0 = material.youngs_modulus
                Emin = self._options['void_youngs_modulus']
                dval = p * rho_sub_element[:] ** (p - 1) * (E0 - Emin) / E0
                return dval
            
        elif self._density_location in ['node']:
        
            rho_q = density_distribution[:] # (NC, NQ)

            if method == 'simp':
                dval = p * rho_q[:] ** (p - 1)
                return dval
            elif method == 'msimp':
                E0 = material.youngs_modulus
                Emin = self._options['void_youngs_modulus']
                dval = p * rho_q[:] ** (p - 1) * (E0 - Emin) / E0
                return dval
        
        elif self._density_location in ['node_multiresolution']:
            
            rho_sub_q = density_distribution[:] # (NC, n_sub, NQ)

            if method == 'simp':
                dval = p * rho_sub_q[:] ** (p - 1)
                return dval
            elif method == 'msimp':
                E0 = material.youngs_modulus
                Emin = self._options['void_youngs_modulus']
                dval = p * rho_sub_q[:] ** (p - 1) * (E0 - Emin) / E0
                return dval
            
        elif self._density_location in ['lagrange_interpolation_point', 
                                        'berstein_interpolation_point', 
                                        'shepard_interpolation_point']:

            rho_q = density_distribution[:]
            if method == 'simp':
                dval = p * rho_q[:] ** (p - 1)
                return dval
            elif method == 'msimp':
                E0 = material.youngs_modulus
                Emin = self._options['void_youngs_modulus']
                dval = p * rho_q[:] ** (p - 1) * (E0 - Emin) / E0
                return dval
            
        elif self._density_location in ['gauss_integration_point',]:

            rho_q = density_distribution[:]
            if method == 'simp':
                dval = p * rho_q[:] ** (p - 1)
                return dval
            elif method == 'msimp':
                E0 = material.youngs_modulus
                Emin = self._options['void_youngs_modulus']
                dval = p * rho_q[:] ** (p - 1) * (E0 - Emin) / E0
                return dval


    ###########################################################################################################
    # 内部方法
    ###########################################################################################################

    def _set_default_options(self) -> None:
        """设置默认选项"""
        defaults = {
            'penalty_factor': 3.0,
            'void_youngs_modulus': 1e-9,
            'target_variables': ['E']
        }
        
        for key, default_value in defaults.items():
            if key not in self._options:
                self._options[key] = default_value
    
    @cartesian
    def _density_distribution_coscos(self, points: TensorLike) -> TensorLike:
        """
        周期性密度函数, 基于二维余弦函数和双曲正切变换

        数学表达式：
            base = bm.cos(scale * x) * bm.cos(scale * y)
            density = 0.5 * (1 + bm.tanh(s * (base + 2 * r - 1)))
        """
        x, y = points[..., 0], points[..., 1]

        scale = 2
        base_pattern = bm.cos(scale * x) * bm.cos(scale * y)

        r = 0.5
        smoothness = 2.0 
        val = 0.5 * (1 + bm.tanh(smoothness * (base_pattern + 2 * r - 1)))

        return val