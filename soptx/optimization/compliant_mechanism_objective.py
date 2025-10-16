from typing import Optional, Union, Literal
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function
from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from ..utils.base_logged import BaseLogged

class CompliantMechanismObjective(BaseLogged):
    def __init__(self,
                analyzer: Union[LagrangeFEMAnalyzer, HuZhangMFEMAnalyzer],
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._analyzer = analyzer
        self._pde = analyzer.pde
        self._space = analyzer.tensor_space

    def fun(self, 
            density: Union[Function, TensorLike], 
            displacement: Optional[TensorLike] = None,
           ) -> float:
        """
        计算柔顺机构的目标函数值 (即输出点的位移 u_out).
        """
        if displacement is None:
            U = self._analyzer.solve_displacement(rho_val=density, adjoint=True) 
        else:
            U = displacement

        threshold_dout = self._pde.is_dout_boundary()
        isBdTDof = self._space.is_boundary_dof(threshold=threshold_dout, method='interp')

        uh_real = U[:, 0]
        u_out = uh_real[isBdTDof]
        

        
        # 目标函数只关心真实物理载荷下的位移，即第一列
        uh_real = U[:, 0]

        # 2. 定位输出点的自由度 (DOF)
        pde = self._analyzer._pde
        tspace = self._analyzer.tensor_space

        # 从 PDE 定义中获取输出点的边界条件函数
        # is_output_boundary() 应该返回 (callable_for_x, None)
        output_boundary_func = pde.is_output_boundary()
        
        # 找到对应的自由度布尔掩码
        is_output_dof_mask = tspace.is_boundary_dof(threshold=output_boundary_func)
        
        # 将布尔掩码转换为索引
        output_dof_idx = bm.where(is_output_dof_mask)[0]

        if output_dof_idx.shape[0] != 1:
            raise ValueError("未能精确定位到唯一的输出自由度。")
        
        output_dof_idx = output_dof_idx[0]

        # 3. 提取目标函数值
        # 目标函数值就是真实位移向量在输出自由度上的分量
        u_out = uh_real[output_dof_idx]

        # 优化器将最小化这个值。由于 u_out 是负数，
        # 最小化它就等同于最大化其绝对值。
        return u_out
