from typing import Optional, Dict, Any, Literal, List, Union

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod, cartesian
from fealpy.typing import TensorLike, Tuple
from fealpy.functionspace import Function, LagrangeFESpace
from fealpy.mesh import HomogeneousMesh

from .linear_elastic_material import LinearElasticMaterial
from ..utils.base_logged import BaseLogged

from dataclasses import dataclass
@dataclass
class DensityDistribution:
    """
    密度分布信息容器
    
    Attributes
    ----------
    function : Function
        密度有限元函数
    sub_density_element : int
        子密度单元数
    """
    function: Function
    sub_density_element: int
    
    def __call__(self, points):
        """在给定点处插值密度"""
        return self.function(points)
    
    def update(self, design_variable):
        """更新密度值"""
        self.function[:] = design_variable

    def __getitem__(self, key):
        """支持下标访问"""
        return self.function[key]
    
    def __setitem__(self, key, value):
        """支持下标赋值"""
        self.function[key] = value
    
    @property
    def array(self):
        """返回密度数组"""
        return self.function.array

    @property
    def shape(self):
        """返回形状"""
        return self.function.shape

class MaterialInterpolationScheme(BaseLogged):
    """材料插值方案类"""
    def __init__(self,
                density_location: Literal['element', 'element_multiresolution', 
                                          'node', 'node_multiresolution'] = 'element',
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
        self.interpolate_map_derivative.set(interpolation_method)

    
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
    
    @property
    def penalty_factor(self) -> float:
        """获取当前的惩罚因子"""
        return self._options['penalty_factor']
    

    #########################################################################################
    # 属性修改器
    #########################################################################################
    @penalty_factor.setter
    def penalty_factor(self, penalty_factor: float) -> None:
        """更新惩罚因子"""
        if penalty_factor <= 0:
            self._log_error(f"penalty_factor must be positive, got {penalty_factor}")
            
        self._options['penalty_factor'] = penalty_factor
    

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
        
        Returns
        -------
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
    
    @setup_density_distribution.register('node')
    def setup_density_distribution(self,
                                   design_variable_mesh: HomogeneousMesh,
                                   displacement_mesh: HomogeneousMesh,
                                   relative_density: float = 1.0,
                                   **kwargs,
                                ) -> Tuple[TensorLike, Function]:
        """
        节点密度-单分辨率 (SRTO), 设计变量就是节点密度, 自由度位于密度节点处
        
        Returns
        -------
        design_variable : TensorLike (NN, )
        density_distribution : Function (NN, )
        """

        NN_design_variable = design_variable_mesh.number_of_nodes()
        design_variable = bm.full((NN_design_variable, ), relative_density, 
                                dtype=bm.float64, device=design_variable_mesh.device) # (NN, )

        NN = displacement_mesh.number_of_nodes()
        density_val = bm.full((NN, ), relative_density, dtype=bm.float64, device=design_variable_mesh.device)
        space = LagrangeFESpace(displacement_mesh, p=1, ctype='C')
        density_distribution = space.function(density_val) # (NN, )

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
        
        Returns
        -------
        design_variable : (NC_design_variable, )
        density_distribution : (NC_density, n_sub)
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

    @setup_density_distribution.register('node_multiresolution')
    def setup_density_distribution(self, 
                            design_variable_mesh: HomogeneousMesh,
                            displacement_mesh: HomogeneousMesh,
                            relative_density: float = 1.0,
                            sub_density_element: int = 4,
                            integration_order: int = 3,
                            **kwargs,
                        ) -> Tuple[TensorLike, DensityDistribution]:
        """
        节点密度-多分辨率 (MRTO), 设计变量独立于有限元网格, 自由度位于子密度节点处
        
        Returns
        -------
        design_variable : TensorLike (NN_design_variable, )
        density_distribution : Function (NN_density, )
        """

        NN_design_variable = design_variable_mesh.number_of_nodes()
        design_variable = bm.full((NN_design_variable, ), relative_density, 
                                dtype=bm.float64, device=design_variable_mesh.device) # (NN_design_variable, )

        NN_density = displacement_mesh.number_of_nodes()
        density_val = bm.full((NN_density, ), relative_density, dtype=bm.float64, device=displacement_mesh.device)
        space = LagrangeFESpace(displacement_mesh, p=1, ctype='C')
        density_func = space.function(density_val) # (NN_density, )

        density_distribution = DensityDistribution(function=density_func, sub_density_element=sub_density_element)

        return design_variable, density_distribution
    

    @variantmethod('simp')
    def interpolate_map(self, 
                    material: LinearElasticMaterial, 
                    rho_val: Union[Function, TensorLike],
                    integration_order: Optional[int] = None,
                    displacement_mesh: Optional[HomogeneousMesh] = None,
                ) -> TensorLike:
        """SIMP 插值: E(ρ) = ρ^p * E0"""

        penalty_factor = self._options['penalty_factor']
        target_variables = self._options['target_variables']
        
        if target_variables == ['E']:
            E0 = material.youngs_modulus

            if self._density_location in ['element']:
                # rho_val.shape = (NC, )
                rho_element = rho_val[:]
                E_rho = rho_element[:] ** penalty_factor * E0

            elif self._density_location in ['node']:
                # rho_val.shape = (NN, )
                qf = self._mesh.quadrature_formula(q=integration_order)
                bcs, ws = qf.get_quadrature_points_and_weights()
                rho_q = rho_val(bcs) # (NC, NQ)
                E_rho = rho_q[:] ** penalty_factor * E0

            elif self._density_location in ['element_multiresolution']:
                # rho_val.shape = (NC, n_sub)
                rho_sub_element = rho_val[:] # (NC, n_sub)
                E_rho = rho_sub_element[:] ** penalty_factor * E0

            elif self._density_location in ['node_multiresolution']:
                pass

            return E_rho
        
    @interpolate_map.register('msimp')
    def interpolate_map(self,
                    material: LinearElasticMaterial, 
                    rho_val: Union[Function, TensorLike, DensityDistribution],
                    integration_order: Optional[int] = None,
                    displacement_mesh: Optional[HomogeneousMesh] = None,
                ) -> TensorLike:
        """修正 SIMP 插值: E(ρ) = Emin + ρ^p * (E0 - Emin)"""

        penalty_factor = self._options['penalty_factor']
        target_variables = self._options['target_variables']
        void_youngs_modulus = self._options['void_youngs_modulus']

        if target_variables == ['E']:

            E0 = material.youngs_modulus
            Emin = void_youngs_modulus

            if self._density_location in ['element']:
                # rho_val.shape = (NC, )
                rho_element = rho_val[:]
                E_rho = Emin + rho_element[:] ** penalty_factor * (E0 - Emin)

            elif self._density_location in ['node']:
                # rho_val.shape = (NN, )
                density_mesh = rho_val.space.mesh
                qf = density_mesh.quadrature_formula(q=integration_order)
                bcs, ws = qf.get_quadrature_points_and_weights()
                rho_q = rho_val(bcs) # (NC, NQ)
                E_rho = Emin + rho_q[:] ** penalty_factor * (E0 - Emin)

            elif self._density_location in ['element_multiresolution']:
                # rho_val.shape = (NC, n_sub)
                rho_sub_element = rho_val[:] # (NC, n_sub)
                E_rho = Emin + rho_sub_element[:] ** penalty_factor * (E0 - Emin)

            elif self._density_location in ['node_multiresolution']:
                # rho_val.shape = (NN, )
                NC = displacement_mesh.number_of_cells()
                qf_e = displacement_mesh.quadrature_formula(q=integration_order)
                # bcs_e.shape = ( (NQ_x, GD), (NQ_y, GD) ), ws_e.shape = (NQ, )
                bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()

                n_sub = rho_val.sub_density_element

                # 把位移单元高斯积分点处的重心坐标映射到子密度单元 (子参考单元) 高斯积分点处的重心坐标 (仍表达在位移单元中)
                from soptx.analysis.utils import map_bcs_to_sub_elements
                # bcs_eg.shape = ( (n_sub, NQ_x, GD), (n_sub, NQ_y, GD) ), ws_e.shape = (NQ, )
                bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
                bcs_eg_x, bcs_eg_y = bcs_eg

                NQ = ws_e.shape[0]
                rho_q = bm.zeros((NC, n_sub, NQ), dtype=bm.float64  , device=displacement_mesh.device)
                for s_idx in range(n_sub):
                    sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])
                    rho_q_sub = rho_val(sub_bcs) # (NC, NQ)
                    rho_q[:, s_idx, :] = rho_q_sub

                E_rho = Emin + rho_q[:] ** penalty_factor * (E0 - Emin)
            
            return E_rho

    @interpolate_map.register('simp_double')
    def interpolate_map(self) -> TensorLike:
        """双指数 SIMP 插值"""
        pass

    @interpolate_map.register('ramp')
    def interpolate_map(self) -> TensorLike:
        """RAMP 插值"""
        pass

    @variantmethod('simp')
    def interpolate_map_derivative(self, 
                        material: LinearElasticMaterial, 
                        rho_val: Union[Function, TensorLike],
                        integration_order: Optional[int] = None,
                    ) -> TensorLike:
        """SIMP 插值求导: dE(ρ) = pρ^{p-1} * E0"""
        p = self._options['penalty_factor']
        target_variables = self._options['target_variables']
        
        if target_variables == ['E']:

            E0 = material.youngs_modulus
            if self._density_location in ['element']:
                rho_element = rho_val[:] # (NC, )
                dE_rho = p * rho_element[:] ** (p - 1) * E0

            elif self._density_location in ['node']:
                rho_q = rho_val[:] # (NC, NQ)
                dE_rho = p * rho_q[:] ** (p - 1) * E0
            
            elif self._density_location in ['element_multiresolution']:
                rho_sub_element = rho_val[:] # (NC, n_sub)
                dE_rho = p * rho_sub_element[:] ** (p - 1) * E0

            elif self._density_location in ['node_multiresolution']:
                rho_sub_q = rho_val[:] # (NC, n_sub, NQ)
                dE_rho = p * rho_sub_q[:] ** (p - 1) * E0
            
            return dE_rho
        
        else:
            error_msg = f"Unknown target_variables: {target_variables}"
            self._log_error(error_msg)

    @interpolate_map_derivative.register('msimp')
    def interpolate_map_derivative(self, 
                        material: LinearElasticMaterial, 
                        rho_val: Union[Function, TensorLike],
                        integration_order: Optional[int] = None,
                    ) -> TensorLike:
        """修正 SIMP 插值求导: dE(ρ) = pρ^{p-1} * (E0 - Emin)"""
        p = self._options['penalty_factor']
        target_variables = self._options['target_variables']

        if target_variables == ['E']:

            E0 = material.youngs_modulus
            Emin = self._options['void_youngs_modulus']

            if self._density_location in ['element']:
                rho_element = rho_val[:] # (NC, )
                dE_rho = p * rho_element[:] ** (p - 1) * (E0 - Emin)

            elif self._density_location in ['node']:
                # rho_val.shape = (NN, )
                density_mesh = rho_val.space.mesh
                qf = density_mesh.quadrature_formula(q=integration_order)
                bcs, ws = qf.get_quadrature_points_and_weights()
                rho_q = rho_val(bcs) # (NC, NQ)
                dE_rho = p * rho_q[:] ** (p - 1) * (E0 - Emin)

            elif self._density_location in ['element_multiresolution']:
                rho_sub_element = rho_val[:] # (NC, n_sub)
                dE_rho = p * rho_sub_element[:] ** (p - 1) * (E0 - Emin)

            elif self._density_location in ['node_multiresolution']:
                rho_sub_q = rho_val[:] # (NC, n_sub, NQ)
                dE_rho = p * rho_sub_q[:] ** (p - 1) * (E0 - Emin)
            
            return dE_rho

        else:

            error_msg = f"Unknown target_variables: {target_variables}"
            self._log_error(error_msg)



    # ###########################################################################################################
    # # 核心方法
    # ###########################################################################################################

    # def interpolate_derivative(self,
    #                     material: LinearElasticMaterial, 
    #                     density_distribution: Union[Function, TensorLike],
    #                 ) -> TensorLike:
    #     """获取当前插值方法的标量系数相对于物理密度的导数"""

    #     method = self.interpolation_method
    #     p = self._options['penalty_factor']

    #     if self._density_location in ['element']:

    #         rho_element = density_distribution[:] # (NC, )
            
    #         if method == 'simp':    
    #             E0 = material.youngs_modulus
    #             dE_rho = p * rho_element[:] ** (p - 1) * E0
    #             return dE_rho
    #         elif method == 'msimp':
    #             E0 = material.youngs_modulus
    #             Emin = self._options['void_youngs_modulus']
    #             dE_rho = p * rho_element[:] ** (p - 1) * (E0 - Emin)
    #             return dE_rho
            
    #     elif self._density_location in ['element_multiresolution']:
            
    #         rho_sub_element = density_distribution[:] # (NC, n_sub)

    #         if method == 'simp':
    #             E0 = material.youngs_modulus
    #             dE_rho = p * rho_sub_element[:] ** (p - 1) * E0
    #             return dE_rho
    #         elif method == 'msimp':
    #             E0 = material.youngs_modulus
    #             Emin = self._options['void_youngs_modulus']
    #             dE_rho = p * rho_sub_element[:] ** (p - 1) * (E0 - Emin)
    #             return dE_rho
            
    #     elif self._density_location in ['node']:
        
    #         rho_q = density_distribution[:] # (NC, NQ)

    #         if method == 'simp':
    #             E0 = material.youngs_modulus
    #             dE_rho = p * rho_q[:] ** (p - 1) * E0
    #             return dE_rho
    #         elif method == 'msimp':
    #             E0 = material.youngs_modulus
    #             Emin = self._options['void_youngs_modulus']
    #             dE_rho = p * rho_q[:] ** (p - 1) * (E0 - Emin)
    #             return dE_rho
        
    #     elif self._density_location in ['node_multiresolution']:
            
    #         rho_sub_q = density_distribution[:] # (NC, n_sub, NQ)

    #         if method == 'simp':
    #             E0 = material.youngs_modulus
    #             dE_rho = p * rho_sub_q[:] ** (p - 1) * E0
    #             return dE_rho
    #         elif method == 'msimp':
    #             E0 = material.youngs_modulus
    #             Emin = self._options['void_youngs_modulus']
    #             dE_rho = p * rho_sub_q[:] ** (p - 1) * (E0 - Emin)
    #             return dE_rho


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