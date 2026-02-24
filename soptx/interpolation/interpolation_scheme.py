from typing import Optional, Dict, Any, Literal, List, Union

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike, Tuple
from fealpy.functionspace import Function, LagrangeFESpace
from fealpy.mesh import HomogeneousMesh

from .linear_elastic_material import LinearElasticMaterial
from ..utils.base_logged import BaseLogged

class MaterialInterpolationScheme(BaseLogged):
    """材料插值方案类"""
    def __init__(self,
                density_location: Literal['element', 'element_multiresolution', 
                                          'node', ] = 'element',
                interpolation_method: Literal['simp', 'msimp', 'ramp'] = 'simp',
                stress_interpolation_method: Literal['power_law'] = 'power_law',
                options: Optional[dict] = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:
        """材料插值方法变体示例"""
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._density_location = density_location
        self._interpolation_method = interpolation_method
        self._stress_interpolation_method = stress_interpolation_method

        self._options = options or {}
        self._set_default_options()

        self._target_variables_validated = False

        # 注册密度分布方法
        self.setup_density_distribution.set(density_location)
        
        # 注册材料插值方法
        self.interpolate_material.set(interpolation_method)
        self.interpolate_material_derivative.set(interpolation_method)

        # 注册应力惩罚方法
        self.interpolate_stress.set(stress_interpolation_method)

    
    #########################################################################################
    # 属性访问器
    #########################################################################################

    @property
    def density_location(self) -> Optional[str]:
        """获取当前的密度位置"""
        return self._density_location

    @property
    def interpolation_method(self) -> Optional[str]:
        """获取当前的材料插值方法"""
        return self._interpolation_method
    
    @property
    def stress_interpolation_method(self) -> Optional[str]:
        """获取当前的应力插值方法"""
        return self._stress_interpolation_method
    
    @property
    def n_sub(self) -> int:
        """获取子密度单元数量（仅多分辨率时有效）"""
        return getattr(self, '_n_sub', None)
    
    @property
    def penalty_factor(self) -> float:
        """获取当前的惩罚因子"""
        return self._options['penalty_factor']
    
    @property
    def stress_penalty_factor(self) -> float:
        """获取当前的应力惩罚因子"""
        return self._options['stress_penalty_factor']
    
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

        self._n_sub = n_sub

        design_variable = bm.full((NC_design_variable, ), relative_density, 
                                dtype=bm.float64, device=design_variable_mesh.device)
        
        density_val = bm.full((NC, n_sub), relative_density, dtype=bm.float64, device=design_variable_mesh.device)
        space = LagrangeFESpace(displacement_mesh, p=0, ctype='D')
        density_distribution = space.function(density_val)

        return design_variable, density_distribution

    # @setup_density_distribution.register('node_multiresolution')
    # def setup_density_distribution(self, 
    #                         design_variable_mesh: HomogeneousMesh,
    #                         displacement_mesh: HomogeneousMesh,
    #                         relative_density: float = 1.0,
    #                         sub_density_element: int = 4,
    #                         integration_order: int = 3,
    #                         **kwargs,
    #                     ) -> Tuple[TensorLike, DensityDistribution]:
    #     """
    #     节点密度-多分辨率 (MRTO), 设计变量独立于有限元网格, 自由度位于子密度节点处
        
    #     Returns
    #     -------
    #     design_variable : TensorLike (NN_design_variable, )
    #     density_distribution : Function (NN_density, )
    #     """

    #     NN_design_variable = design_variable_mesh.number_of_nodes()
    #     design_variable = bm.full((NN_design_variable, ), relative_density, 
    #                             dtype=bm.float64, device=design_variable_mesh.device) # (NN_design_variable, )

    #     NN_density = displacement_mesh.number_of_nodes()
    #     density_val = bm.full((NN_density, ), relative_density, dtype=bm.float64, device=displacement_mesh.device)
    #     space = LagrangeFESpace(displacement_mesh, p=1, ctype='C')
    #     density_func = space.function(density_val) # (NN_density, )

    #     density_distribution = DensityDistribution(function=density_func, sub_density_element=sub_density_element)

    #     return design_variable, density_distribution
    

    @variantmethod('simp')
    def interpolate_material(self, 
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
        
    @interpolate_material.register('msimp')
    def interpolate_material(self,
                    material: LinearElasticMaterial, 
                    rho_val: Union[Function, TensorLike],
                    integration_order: Optional[int] = None,
                    displacement_mesh: Optional[HomogeneousMesh] = None,
                ) -> TensorLike:
        """修正 SIMP 插值"""
        target_variables = self._options['target_variables']

        if not self._target_variables_validated:
            if material.is_incompressible and 'nu' not in target_variables:
                self._log_warning(
                    f"材料为不可压缩 (ν={material.poisson_ratio:.4f}), "
                    f"建议将 'nu' 添加到 target_variables 中以避免体积闭锁。"
                )
            
            if not material.is_incompressible and 'nu' in target_variables:
                self._log_info(
                    f"材料为可压缩 (ν={material.poisson_ratio:.4f}), "
                    f"无需对泊松比进行插值，'nu' 配置将被忽略。"
                )
            
            self._target_variables_validated = True

        results = []

        rho_interp = None

        if self._density_location in ['element']:
            # rho_val.shape = (NC, )
            if hasattr(rho_val, 'ndim') and rho_val.ndim == 0:
                rho_interp = rho_val
            else:
                rho_interp = rho_val[:]

        elif self._density_location in ['node']:
            # rho_val.shape = (NN, )
            density_mesh = rho_val.space.mesh
            qf = density_mesh.quadrature_formula(q=integration_order)
            bcs, ws = qf.get_quadrature_points_and_weights()
            rho_interp = rho_val(bcs)

        elif self._density_location in ['element_multiresolution']:
            # rho_val.shape = (NC, n_sub)
            rho_interp = rho_val[:]

        else:
            raise NotImplementedError(f"Unknown density_location: {self._density_location}")

        # elif self._density_location in ['node_multiresolution']:
        #     # rho_val.shape = (NN, )
        #     pass

        if 'E' in target_variables:
            p = self._options['penalty_factor']
            E0 = material.youngs_modulus
            Emin = self._options['void_youngs_modulus']
            
            E_rho = Emin + rho_interp ** p * (E0 - Emin)
            results.append(E_rho)
        
        if 'nu' in target_variables and material.is_incompressible:
            p_nu = self._options.get('nu_penalty_factor', 1.0) # 默认为 1.0
            nu0 = material.poisson_ratio  # 强材料泊松比 (例如 0.5)
            nu_void = self._options.get('void_poisson_ratio', 0.3) # 弱材料泊松比 (例如 0.3)
            
            # 插值公式: nu(rho) = nu_void + rho^p_nu * (nu_solid - nu_void)
            nu_rho = nu_void + rho_interp ** p_nu * (nu0 - nu_void)
            results.append(nu_rho)

        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)

    @variantmethod('simp')
    def interpolate_material_derivative(self, 
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

    @interpolate_material_derivative.register('msimp')
    def interpolate_material_derivative(self, 
                        material: LinearElasticMaterial, 
                        rho_val: Union[Function, TensorLike],
                        integration_order: Optional[int] = None,
                        displacement_mesh: Optional[HomogeneousMesh] = None,
                    ) -> TensorLike:
        """修正 SIMP 插值求导"""
        target_variables = self._options['target_variables']

        results = []

        rho_interp = None

        if self._density_location in ['element']:
            # rho_val.shape = (NC, )
            if hasattr(rho_val, 'ndim') and rho_val.ndim == 0:
                rho_interp = rho_val
            else:
                rho_interp = rho_val[:]

        elif self._density_location in ['node']:
            # rho_val.shape = (NN, )
            density_mesh = rho_val.space.mesh
            qf = density_mesh.quadrature_formula(q=integration_order)
            bcs, ws = qf.get_quadrature_points_and_weights()
            rho_interp = rho_val(bcs)

        elif self._density_location in ['element_multiresolution']:
            # rho_val.shape = (NC, n_sub)
            rho_interp = rho_val[:]

        elif self._density_location in ['node_multiresolution']:
            # rho_val.shape = (NN, )
            NC = displacement_mesh.number_of_cells()
            qf_e = displacement_mesh.quadrature_formula(q=integration_order)
            bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()
            n_sub = rho_val.sub_density_element
            from soptx.analysis.utils import map_bcs_to_sub_elements
            bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
            bcs_eg_x, bcs_eg_y = bcs_eg
            NQ = ws_e.shape[0]
            rho_interp = bm.zeros((NC, n_sub, NQ), dtype=bm.float64, device=displacement_mesh.device)
            for s_idx in range(n_sub):
                sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])
                rho_interp[:, s_idx, :] = rho_val(sub_bcs)

        if 'E' in target_variables:
            p = self._options['penalty_factor']
            E0 = material.youngs_modulus
            Emin = self._options['void_youngs_modulus']
            
            dE_rho = p * rho_interp ** (p - 1) * (E0 - Emin)
            results.append(dE_rho)

        # if 'nu' in target_variables and material.is_incompressible:
        if 'nu' in target_variables:
            p_nu = self._options.get('nu_penalty_factor', 1.0) # 默认为 1.0
            nu0 = material.poisson_ratio
            nu_void = self._options.get('void_poisson_ratio', 0.3)
            
            # 插值公式: dnu = p_nu * rho^(p_nu-1) * (nu_solid - nu_void)
            if abs(p_nu - 1.0) < 1e-6:
                # 线性插值特例 (p=1): 导数为常数 (nu0 - nu_void)
                dnu_rho = (nu0 - nu_void) * (rho_interp ** 0.0)
            else:
                dnu_rho = p_nu * rho_interp ** (p_nu - 1) * (nu0 - nu_void)
            
            results.append(dnu_rho)

        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)

    @variantmethod('power_law') 
    def interpolate_stress(self, 
                        stress_solid: TensorLike, 
                        rho_val: Union[Function, TensorLike],
                        return_stress_penalty: bool = True
                    ) -> TensorLike:
        """应力惩罚"""
        q = self._options['stress_penalty_factor']
        
        if self._density_location in ['element']:
            # rho_val.shape = (NC, )
            # stress_solid.shape = (NC, NQ, NS)
            rho_element = rho_val[:]
            eta_sigma = rho_element ** q

            stress_penalized = bm.einsum('c, cqs -> cqs', eta_sigma, stress_solid)

        elif self._density_location in ['node']:
            # rho_val.shape = (NN, )
            pass

        elif self._density_location in ['element_multiresolution']:
            # rho_val.shape = (NC, n_sub)
            # stress_solid.shape = (NC, n_sub, NQ, NS)
            rho_sub_element = rho_val[:] # (NC, n_sub)
            stress_penalized = bm.einsum('cn, cnqs -> cnqs', rho_sub_element ** q, stress_solid)
        
        if return_stress_penalty:
            return {
                'eta_sigma': eta_sigma,               # (NC, )
                'stress_penalized': stress_penalized, # (NC, NQ, NS)
            }
        else:
            return stress_penalized
        
    @variantmethod('power_law') 
    def interpolate_stress_derivative(self, 
                        rho_val: Union[Function, TensorLike],
                        stress_solid: Optional[TensorLike] = None,
                    ) -> Dict:
        """应力惩罚的导数"""
        q = self._options['stress_penalty_factor']
        
        result = {}
        
        if self._density_location == 'element':
            # rho_val.shape = (NC,)
            rho_element = rho_val[:]
            deta_sigma_drho = q * rho_element ** (q - 1)  # (NC,)
            
            result['deta_sigma_drho'] = deta_sigma_drho
            
            if stress_solid is not None:
                # stress_solid.shape = (NC, NQ, NS)
                # dstress_penalized_drho.shape = (NC, NQ, NS)
                dstress_penalized_drho = bm.einsum('c, cqs -> cqs', deta_sigma_drho, stress_solid)
                result['dstress_penalized_drho'] = dstress_penalized_drho

        elif self._density_location == 'node':
            # rho_val.shape = (NN,)
            pass

        elif self._density_location == 'element_multiresolution':
            # rho_val.shape = (NC, n_sub)
            rho_sub_element = rho_val[:]
            deta_sigma_drho = q * rho_sub_element ** (q - 1)  # (NC, n_sub)
            
            result['deta_sigma_drho'] = deta_sigma_drho
            
            if stress_solid is not None:
                # stress_solid.shape = (NC, n_sub, NQ, NS)
                # dstress_penalized_drho.shape = (NC, n_sub, NQ, NS)
                dstress_penalized_drho = bm.einsum('cn, cnqs -> cnqs', deta_sigma_drho, stress_solid)
                result['dstress_penalized_drho'] = dstress_penalized_drho
        
        return result
    

    ###########################################################################################################
    # 内部方法
    ###########################################################################################################

    def _set_default_options(self) -> None:
        """设置默认选项"""
        defaults = {
            'penalty_factor': 3.0,       # 杨氏模量惩罚因子 
            'void_youngs_modulus': 1e-9, # 孔洞杨氏模量
            'target_variables': ['E'],   # 插值变量

            'nu_penalty_factor': 1.0,    # 泊松比惩罚因子
            'void_poisson_ratio': 0.3,   # 孔洞泊松比

            'stress_penalty_factor': 0.5 # 应力惩罚因子 
        }
        
        for key, default_value in defaults.items():
            if key not in self._options:
                self._options[key] = default_value