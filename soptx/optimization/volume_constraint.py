from typing import Optional, Literal, Union, Dict, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import SimplexMesh, TensorMesh
from fealpy.functionspace import Function

from ..analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from ..utils.base_logged import BaseLogged
from soptx.optimization.utils import compute_volume

class VolumeConstraint(BaseLogged):
    def __init__(self,
                analyzer: LagrangeFEMAnalyzer,
                volume_fraction: float,
                diff_mode: Literal["auto", "manual"] = "manual",
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._analyzer = analyzer
        self._volume_fraction = volume_fraction
        self._diff_mode = diff_mode

        self._mesh = self._analyzer._mesh
        self._interpolation_scheme = self._analyzer._interpolation_scheme
        self._integration_order = self._analyzer._integration_order
        self._density_location = self._interpolation_scheme._density_location

        self._cell_measure = self._mesh.entity_measure('cell')
        self._v0 = bm.sum(self._cell_measure)
        self._v = None

    def fun(self, 
            density: Union[Function, TensorLike], 
            state: Optional[Dict] = None,
            **kwargs 
        ) -> float:
        """计算体积分数约束值 - (归一化)"""
        v = compute_volume(density=density, 
                           mesh=self._mesh,
                           density_location=self._density_location,
                           integration_order=self._integration_order)
        self._v = v

        v0 = self._v0
        gneq = v / v0 - self._volume_fraction

        return gneq
    
    def jac(self, 
            density: Union[Function, TensorLike], 
            state: Optional[dict] = None,
            diff_mode: Optional[Literal["auto", "manual"]] = None,
            **kwargs
        ) -> TensorLike:
        """计算体积约束函数相对于物理密度的灵敏度 - (归一化)"""
        mode = diff_mode if diff_mode is not None else self._diff_mode

        if mode == "manual":
            return self._manual_differentiation(density=density, state=state, **kwargs)
        
        elif mode == "auto":  
            return self._auto_differentiation(density=density, state=state, **kwargs)
        
        else:
            error_msg = f"Unknown diff_mode: {diff_mode}"
            self._log_error(error_msg)

    def _manual_differentiation(self, 
                            density: Union[Function, TensorLike],
                            state: Optional[dict] = None, 
                            enable_timing: bool = False, 
                            **kwargs
                        ) -> TensorLike:
        """手动计算体积约束函数相对于物理密度的灵敏度 (归一化)"""
        if self._density_location in ['element']:
            dg = bm.copy(self._cell_measure) / self._v0

            return dg
        
        elif self._density_location in ['element_multiresolution']:
            NC, n_sub = density.shape
            cell_measure = self._mesh.entity_measure('cell')
            sub_cm = bm.tile(cell_measure.reshape(NC, 1), (1, n_sub)) / (n_sub * self._v0)
            
            dg = bm.copy(sub_cm) # (NC, n_sub)

            return dg
        
        elif self._density_location in ['node']:
            #* 标准节点密度表征下的体积分数梯度计算
            # mesh = self._mesh
            density_space = density.space

            qf = self._mesh.quadrature_formula(self._integration_order)
            bcs, ws = qf.get_quadrature_points_and_weights()

            phi = density_space.basis(bcs)[0] # (NQ, NCN)

            if isinstance(self._mesh, SimplexMesh):
                cell_measure = self._mesh.entity_measure('cell')
                dg_e = bm.einsum('q, c, ql -> cl', ws, cell_measure, phi) / self._v0 # (NC, NCN)
            elif isinstance(self._mesh, TensorMesh):
                J = self._mesh.jacobi_matrix(bcs)    # (NC, NQ, GD, GD)
                detJ = bm.abs(bm.linalg.det(J))      # (NC, NQ)
                dg_e = bm.einsum('q, cq, ql -> cl', ws, detJ, phi) / self._v0    # (NC, NCN)

            NN = self._mesh.number_of_nodes()
            cell2node = self._mesh.cell_to_node() # (NC, NCN)

            dg = bm.zeros((NN, ), dtype=bm.float64, device=self._mesh.device) # (NN, )
            dg = bm.add_at(dg, cell2node.reshape(-1), dg_e.reshape(-1)) # (NN, )
            
            #* 简化节点密度表征下体积分数梯度计算
            # mesh = self._mesh
            # NN = mesh.number_of_nodes()

            # dg = bm.full((NN, ), 1.0 / NN, dtype=bm.float64, device=mesh.device)

            return dg

        elif self._density_location in ['node_multiresolution']:
            raise NotImplementedError(f"暂时不考虑节点多分辨率密度分布")

        else:
            raise NotImplementedError(f"暂时不考虑其它密度分布")
        
    def _auto_differentiation(self, 
                            density: Union[Function, TensorLike],
                            state: Optional[dict] = None, 
                            enable_timing: bool = False, 
                            **kwargs
                        ) -> TensorLike:
        """使用自动微分技术计算体积约束函数关于物理密度的梯度 (归一化)"""
        if bm.backend_name not in ['pytorch', 'jax']:
            self._log_error(f"自动微分仅在 pytorch 或者 jax 后端下有效")

        if self._density_location == 'element':
            # density.shape = (NC, )
            cell_measure = self._mesh.entity_measure('cell')

            # 定义核函数: 计算单个单元的体积贡献
            def vol_kernel(rho_i, cm_i):
                return rho_i * cm_i
            
            # 向量化求导
            grad_func = bm.vmap(bm.grad(vol_kernel), in_axes=(0, 0))
            
            dg = grad_func(density[:], cell_measure) / self._v0  

            return dg
        
        else:
            raise NotImplementedError(f"暂时不考虑其它密度分布")

