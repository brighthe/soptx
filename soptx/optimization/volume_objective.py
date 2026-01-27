from typing import Optional, Literal, Union, Dict

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from soptx.utils.base_logged import BaseLogged
from soptx.utils import timer

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
        self._cell_measure = self._mesh.entity_measure('cell')

    def fun(self, 
            density: Union[Function, TensorLike],
            state: Optional[Dict] = None, 
            **kwargs
        ) -> float:
        """计算体积目标函数值"""
        cell_measure = self._cell_measure
        current_volume = bm.einsum('c, c -> ', cell_measure, density[:])

        return current_volume

    def jac(self, 
            density: Union[Function, TensorLike], 
            state: Optional[dict] = None,
            diff_mode: Optional[Literal["auto", "manual"]] = None,
            **kwargs
        ) -> TensorLike:
        """计算目标函数关于物理密度的灵敏度"""
        mode = diff_mode if diff_mode is not None else self._diff_mode
        
        if mode == "manual":
            return self._cell_measure
            
        elif mode == "auto":
            pass
            
        else:
            raise ValueError(f"Unsupported diff_mode: {self._diff_mode}")