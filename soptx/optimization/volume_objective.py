from typing import Optional, Literal, Union, Dict

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from soptx.optimization.utils import compute_volume
from soptx.utils.base_logged import BaseLogged

class VolumeObjective(BaseLogged):
    def __init__(self,
                analyzer: Union[LagrangeFEMAnalyzer, HuZhangMFEMAnalyzer],
                diff_mode: Literal["auto", "manual"] = "manual",
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._analyzer = analyzer
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
        """计算体积目标函数值 (归一化)"""
        v = compute_volume(density=density, 
                           mesh=self._mesh,
                           density_location=self._density_location,
                           integration_order=self._integration_order)
        
        self._v = v

        v0 = self._v0
        
        return v / v0

    def jac(self, 
            density: Union[Function, TensorLike], 
            state: Optional[dict] = None,
            diff_mode: Optional[Literal["auto", "manual"]] = None,
            **kwargs
        ) -> TensorLike:
        """计算目标函数关于物理密度的灵敏度 (归一化)"""
        mode = diff_mode if diff_mode is not None else self._diff_mode
        
        if mode == "manual":
            return self._manual_differentiation(density=density, state=state, **kwargs)

        elif mode == "auto":
            raise NotImplementedError("自动微分尚未实现")
            
        else:
            raise ValueError(f"Unsupported diff_mode: {self._diff_mode}")
        
    def _manual_differentiation(self, 
                            density: Union[Function, TensorLike],
                            state: Optional[dict] = None, 
                            enable_timing: bool = False, 
                            **kwargs
                        ) -> TensorLike:
        """手动计算体积约束函数相对于物理密度的灵敏度 (归一化)"""
        if self._density_location in ['element']:
            dv = self._mesh.entity_measure('cell') / self._v0

            return dv
        
        elif self._density_location in ['element_multiresolution']:
            NC, n_sub = density.shape
            cell_measure = self._mesh.entity_measure('cell')
            sub_cm = bm.tile(cell_measure.reshape(NC, 1), (1, n_sub)) / (n_sub * self._v0)
            
            dv = bm.copy(sub_cm) # (NC, n_sub)

            return dv

        else:
            raise NotImplementedError(f"暂时不考虑其它密度分布")