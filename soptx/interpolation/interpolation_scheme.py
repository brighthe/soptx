from typing import Optional, Dict, Any, Literal, List, Union

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod, cartesian
from fealpy.typing import TensorLike
from fealpy.functionspace import Function, LagrangeFESpace
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
                                mesh: HomogeneousMesh,
                                relative_density: float = 1.0,
                                integration_order : int = None,
                                interpolation_order: int = None,
                                **kwargs,
                            ) -> Function:
        """单元密度分布"""
        if integration_order is not None:
            warn_msg = f"'element' density distribution does not require 'integration_order', provided integration_order={integration_order} will be ignored"
            self._log_warning(warn_msg)
        
        if interpolation_order is not None:
            warn_msg = f"'element' density distribution does not require 'interpolation_order', provided interpolation_order={interpolation_order} will be ignored"
            self._log_warning(warn_msg)

        NC = mesh.number_of_cells()
        density_tensor = bm.full((NC,), relative_density, dtype=bm.float64, device=mesh.device)

        space = LagrangeFESpace(mesh, p=0, ctype='D')
        density_dist = space.function(density_tensor)

        self._log_info(f"Element density: shape={density_dist.shape}, value={relative_density}")

        return density_dist
    
    @setup_density_distribution.register('gauss_integration_point')
    def setup_density_distribution(self, 
                                mesh: HomogeneousMesh,
                                relative_density: float = 1.0,
                                integrator_order: int = 3,
                                interpolation_order: int = None,
                                **kwargs,
                            ) -> TensorLike:
        """高斯积分点密度分布"""

        if integrator_order is None:
            error_msg = "'gauss_integration_point' density distribution requires 'integrator_order' parameter"
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        if interpolation_order is not None:
            warn_msg = f"'gauss_integration_point' density distribution does not require 'interpolation_order', provided interpolation_order={interpolation_order} will be ignored"
            self._log_warning(warn_msg)
        
        qf = mesh.quadrature_formula(integrator_order)
        bcs, ws = qf.get_quadrature_points_and_weights()

        NC = mesh.number_of_cells()
        density_tensor = bm.full((NC, ), relative_density, dtype=bm.float64, device=mesh.device)

        space = LagrangeFESpace(mesh, p=0, ctype='D')
        density_dist = space.function(density_tensor)
        density_dist = density_dist(bcs)

        self._log_info(f"Element-Gauss density: shape={density_dist.shape}, value={relative_density}, q={integrator_order}")

        return density_dist
    
    @setup_density_distribution.register('dual_mesh')
    def setup_density_distribution(self,
                                mesh: HomogeneousMesh,
                                relative_density: float = 1.0,
                                integrator_order: int = 3,
                                subcells: tuple = (3, 3),   # (nsx, nsy)
                                design_density: TensorLike = None,
                                **kwargs) -> TensorLike:
        """双网格（对齐细分）下的密度分布:
        - 设计变量定义在每个分析单元内的等分子单元上 (NC, Ns)
        - 返回在高斯点采样得到的密度 (NC, NQ)
        """
        if integrator_order is None:
            msg = "'dual_mesh' density distribution requires 'integrator_order'"
            self._log_error(msg); raise ValueError(msg)

        nsx, nsy = subcells
        Ns = nsx * nsy

        # 高斯点（参考单元坐标）与权重
        qf = mesh.quadrature_formula(integrator_order)
        bcs, ws = qf.get_quadrature_points_and_weights()   # bcs: (NQ, 2) for quad
        ws = ws.shape[0]

        NC = mesh.number_of_cells()

        # 设计变量 d: (NC, Ns)
        if design_density is None:
            d = bm.full((NC, Ns), relative_density, dtype=bm.float64, device=mesh.device)
        else:
            d = design_density
            if d.shape != (NC, Ns):
                msg = f"'design_density' must have shape {(NC, Ns)}, got {d.shape}"
                self._log_error(msg); raise ValueError(msg)

        # 将参考区间 [-1,1] 映射到子单元索引 [0..nsx-1], [0..nsy-1]
        # 对齐细分: 子单元边界在参考坐标均匀划分
        xi  = bcs[:, 0]           # (-1, 1)
        eta = bcs[:, 1]           # (-1, 1)
        ui  = 0.5 * (xi  + 1.0)   # (0, 1)
        vj  = 0.5 * (eta + 1.0)   # (0, 1)

        ii = bm.floor(ui * nsx).astype(bm.int32)
        jj = bm.floor(vj * nsy).astype(bm.int32)
        ii = bm.minimum(ii, nsx - 1)
        jj = bm.minimum(jj, nsy - 1)

        s_idx_local = jj * nsx + ii            # (NQ,) in [0, Ns-1]

        # 采样: ρ_{e,g} = d_{e, s_idx_local[g]}
        # 利用广播/按轴gather得到 (NC, NQ)
        row = bm.arange(NC).reshape(-1, 1)
        col = s_idx_local.reshape(1, -1)
        density_dist = d[row, col]             # (NC, NQ)

        # 可选：把映射保存起来，供灵敏度回传用（segment-sum）
        self._dualmesh_subcell_index = s_idx_local  # (NQ,)
        self._dualmesh_subcells = subcells

        self._log_info(f"Dual-mesh density: shape={density_dist.shape}, "
                    f"design_shape={d.shape}, q={integrator_order}, subcells={subcells}")

        return density_dist

    
    @setup_density_distribution.register('lagrange_interpolation_point')
    def setup_density_distribution(self,
                                   mesh: HomogeneousMesh,
                                   relative_density: float = 1.0,
                                   integration_order: int = None,
                                   interpolation_order: int = 1,
                                   **kwargs,
                                ) -> Function:
        "节点密度 (Lagrange 插值)"
        if interpolation_order is None:
            error_msg = "'interpolation_point' density distribution requires 'interpolation_order' parameter"
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        if integration_order is not None:
            warn_msg = f"Interpolation point density distribution does not require 'integration_order', provided integration_order={integration_order} will be ignored"
            self._log_warning(warn_msg)

        space = LagrangeFESpace(mesh, p=interpolation_order, ctype='C')
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

    @setup_density_distribution.register('element_coscos')
    def setup_density_distribution(self, 
                                mesh: HomogeneousMesh,
                                relative_density: float = None,
                                integrator_order: int = None,
                                **kwargs,
                            ) -> Function:
        """单元密度分布 (变体1)"""
        element_space = LagrangeFESpace(mesh, p=0, ctype='D')
        density_dist = element_space.interpolate(u=self._density_distribution_coscos)

        self._log_info(f"此时的密度是分片常数函数, ")

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

            if self._density_location == 'interpolation_point':
                # 如果是插值点密度分布, 则需要将结果转换为 Function
                density_space = density_distribution.space
                msimp_map = density_space.function(msimp_map)
            
            return simp_map
    
    @interpolate_map.register('msimp')
    def interpolate_map(self,
                    material: LinearElasticMaterial, 
                    density_distribution: Union[Function, TensorLike],
                ) -> TensorLike:
        """修正 SIMP 插值: E(ρ) = Emin + ρ^p * (E0 - Emin)"""

        required_options = ['penalty_factor', 'target_variables', 'void_youngs_modulus']
        for option in required_options:
            if option not in self._options:
                error_msg = f"Missing required option '{option}' for msimp interpolation"
                self._log_error(error_msg)
                raise ValueError(error_msg)

        penalty_factor = self._options['penalty_factor']
        target_variables = self._options['target_variables']
        void_youngs_modulus = self._options['void_youngs_modulus']

        if target_variables == ['E']:
            E0 = material.youngs_modulus
            Emin = void_youngs_modulus
            msimp_map = (Emin + density_distribution[:] ** penalty_factor * (E0 - Emin)) / E0

            if self._density_location == 'interpolation_point':
                # 如果是插值点密度分布, 则需要将结果转换为 Function
                density_space = density_distribution.space
                msimp_map = density_space.function(msimp_map)

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
                        density_distribution: Function,
                    ) -> TensorLike:
        """获取当前插值方法的导数对应的系数"""

        method = self.interpolation_method
        p = self._options['penalty_factor'] 

        if method == 'simp':
            dval = p * density_distribution[:] ** (p - 1)

            return dval

        elif method == 'msimp':
            E0 = material.youngs_modulus
            Emin = self._options['void_youngs_modulus']
            dval = p * density_distribution[:] ** (p - 1) * (E0 - Emin) / E0
            
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