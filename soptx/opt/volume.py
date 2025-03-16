from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Literal

from typing import Dict, Optional, Any
from dataclasses import dataclass

from soptx.solver import ElasticFEMSolver
from soptx.opt import ConstraintBase

@dataclass
class VolumeConfig:
    """Configuration for volume constraint computation"""
    diff_mode: Literal["auto", "manual"] = "manual"

class VolumeConstraint(ConstraintBase):
    """不等式体积约束"""
    def __init__(self,
                solver: ElasticFEMSolver,
                volume_fraction: float,
                config: Optional[VolumeConfig] = None):
        """
        Parameters
        - solver : 有限元求解器
        - volume_fraction : 目标体积分数
        - config : 体积约束计算的配置参数, 如果为 None 则使用默认配置
        """
        self.solver = solver
        self.volume_fraction = volume_fraction
        self.mesh = solver.tensor_space.mesh

        self.config = config if config is not None else VolumeConfig()

    #---------------------------------------------------------------------------
    # 体积计算相关方法
    #---------------------------------------------------------------------------
    def get_volume_fraction(self, rho: TensorLike) -> float:
        """计算当前设计的体积分数"""
        cell_measure = self.mesh.entity_measure('cell')
        current_volume = bm.einsum('c, c -> ', cell_measure, rho)
        total_volume = bm.sum(cell_measure)
        volume_fraction = current_volume / total_volume
        return volume_fraction

    #---------------------------------------------------------------------------
    # 内部方法
    #---------------------------------------------------------------------------
    def _compute_gradient_manual(self, rho: TensorLike) -> TensorLike:
        """使用解析方法计算梯度"""
        cell_measure = self.mesh.entity_measure('cell')
        dg = bm.copy(cell_measure)

        return dg
    
    def _compute_gradient_auto(self, rho: TensorLike) -> TensorLike:
        """使用自动微分计算梯度"""
        cell_measure = self.mesh.entity_measure('cell')
        
        def volume_contribution(rho_i: float, measure_i: float) -> float:
            """计算单个单元的体积贡献
            
            Parameters
            - rho_i : 单个单元的密度值
            - measure_i : 单个单元的测度
            """
            g_i = measure_i * rho_i
            return g_i
        
        # 创建向量化的梯度计算函数
        vmap_grad = bm.vmap(lambda r, m: 
                        bm.jacrev(lambda x: volume_contribution(x, m))
                            (r))
        
        # 并行计算所有单元的梯度
        dg = vmap_grad(rho, cell_measure)
        return dg
        
    #---------------------------------------------------------------------------
    # 优化相关方法
    #---------------------------------------------------------------------------
    def fun(self, 
            rho: TensorLike, 
            u: Optional[TensorLike] = None) -> float:
        """计算体积约束函数值"""
        cell_measure = self.mesh.entity_measure('cell')
        gneq = bm.einsum('c, c -> ', cell_measure, rho) - \
                self.volume_fraction * bm.sum(cell_measure)
        # gneq = bm.einsum('c, c -> ', cell_measure, rho) / \
        #         (self.volume_fraction * bm.sum(cell_measure)) - 1 # float
         
        return gneq
        
    def jac(self,
            rho: TensorLike,
            u: Optional[TensorLike] = None,
            diff_mode: Optional[Literal["auto", "manual"]] = None
            ) -> TensorLike:
        """计算体积约束的梯度
        
        Parameters
        - rho : 密度场
        - u : 位移场（体积约束不需要，但为了接口一致）
        - diff_mode : 梯度计算方式, 如果为 None 则使用配置中的默认值
            - "manual": 使用解析推导的梯度公式
            - "auto": 使用自动微分技术
        """
        if diff_mode is None:
            diff_mode = self.config.diff_mode

        if diff_mode == "manual":
            dg = self._compute_gradient_manual(rho)
        elif diff_mode == "auto":
            dg = self._compute_gradient_auto(rho)
        else:
            raise ValueError(f"Unknown diff_mode: {diff_mode}")
            
        return dg
        
    def hess(self, rho: TensorLike, lambda_: Dict[str, Any]) -> TensorLike:
        """计算体积约束 Hessian 矩阵 (未实现)"""
        pass

    @property
    def constraint_type(self) -> str:
        """返回约束类型 - 拓扑优化中体积约束一定是不等式约束"""
        return "inequality"