from typing import Optional, Literal, Union, Tuple, Dict
import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from ..utils.base_logged import BaseLogged

class StressConstraint(BaseLogged):
    """
    应力约束类

    实现基于聚类 P-norm 的全局应力聚合约束
    """
    def __init__(self,
                analyzer: Union[LagrangeFEMAnalyzer, HuZhangMFEMAnalyzer],
                stress_limit: float,
                p_norm_factor: float = 8.0,
                n_clusters: int = 10,
                recluster_freq: int = 1,          
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._analyzer = analyzer
        self._stress_limit = stress_limit

        self._p_norm_factor = p_norm_factor
        self._n_clusters = n_clusters
        self._recluster_freq = recluster_freq
        
        # 聚类相关变量
        self._clustering_map = None
        self._cluster_weight_sums = None

        self._cached_stress_state = None

    def _perform_clustering_logic(self, sigma_vm: TensorLike, weights: TensorLike) -> None:
        """
        执行基于 'Stress Level' 的聚类逻辑 
        
        策略：
        1. 获取所有应力点的 von Mises 应力值
        2. 将应力值降序排列 (从大到小)
        3. 将排序后的点均匀分配到 n_clusters 个簇中
           - Cluster 0: 应力最高的 N/nc 个点
           - Cluster 1: 应力次高的 N/nc 个点
           - ...
        4. 预计算每个簇的总权重 (用于后续 P-norm 的分母)，提高效率

        Parameters
        ----------
        sigma_vm : (NC, NQ) von Mises 应力
        weights : (NC, NQ) 积分权重 (detJ * w_q)
        """
        vals = sigma_vm.flatten()
        ws = weights.flatten()
        n_points = vals.shape[0]

        # 降序排序 (Stress Level) 
        sorted_indices = bm.argsort(vals, axis=0)[::-1]

        # 初始化聚类映射数组
        # map[i] 表示第 i 个原始应力点属于哪个 cluster
        new_map = bm.zeros((n_points,), dtype=bm.int64)

        # 计算分块大小
        block_size = n_points // self._n_clusters
        remainder = n_points % self._n_clusters

        current_start = 0
        
        # 用于缓存每个 Cluster 的总权重 (Sum of Weights)，避免在 fun 中重复计算
        cluster_weight_sums = bm.zeros((self._n_clusters,), dtype=vals.dtype)

        for i in range(self._n_clusters):
            # 计算当前 cluster 的大小
            # 如果有余数，前 remainder 个 cluster 多分 1 个点
            current_count = block_size + (1 if i < remainder else 0)
            
            # 确定在 sorted_indices 中的切片范围
            current_end = current_start + current_count
            
            # 获取属于当前 cluster 的原始点索引
            indices_in_this_cluster = sorted_indices[current_start : current_end]
            
            # 更新映射表：将这些点的 cluster id 设为 i
            new_map[indices_in_this_cluster] = i
            
            # 预计算：当前 Cluster 的总积分权重 (Sum W_i)
            cluster_weight_sums[i] = bm.sum(ws[indices_in_this_cluster])

            # 更新游标
            current_start = current_end

        # 更新类内部状态
        self._clustering_map = new_map
        self._cluster_weight_sums = cluster_weight_sums  # 缓存分母
        

    def update_clustering(self, iter_idx: int, state: Dict, density: TensorLike) -> None:
        """根据当前迭代步数和物理状态更新聚类"""
        should_update = (self._recluster_freq > 0) and (iter_idx % self._recluster_freq == 0)
        
        if self._clustering_map is None:
            should_update = True
        
        if should_update:
            stress_state = self._analyzer.compute_stress_state(state=state, rho_val=density)
            sigma_vm = stress_state['von_mises'] # (NC, NQ)
            weights = stress_state['weights'] # 获取积分权重 - (NC, NQ)

            # 执行核心聚类算法 (排序、切分等)
            self._perform_clustering_logic(sigma_vm, weights)

            self._cached_stress_state = stress_state
        
        else:
            self._cached_stress_state = None
        
    def _compute_clustered_pnorm(self, 
                                sigma_vm: TensorLike,
                                weights: TensorLike
                            ) -> float:
        """计算聚类 P-norm 应力约束值

        Parameters
        ----------
        sigma_vm: von Mises 应力场
            - STOP: (NC, NQ)
            - MTOP: (NC, n_sub, NQ)
        weigths: 积分权重 (NC, NQ)
        """
        vals = sigma_vm.flatten()
        ws = weights.flatten()
        
        if self._clustering_map is None or self._cluster_weight_sums is None:
             self._perform_clustering_logic(sigma_vm, weights)
        
        # 归一化应力
        normalized_stress = vals / self._stress_limit

        # 加权项
        base_term = (bm.maximum(normalized_stress, 0.0) + 1e-12) ** self._p_norm_factor
        weighted_term = ws * base_term
        
        # 聚类求和
        aggregated_numerator = bm.zeros((self._n_clusters,), dtype=vals.dtype)
        bm.add_at(aggregated_numerator, self._clustering_map, weighted_term)
        
        # 平均化修正
        mean_term = aggregated_numerator / (self._cluster_weight_sums + 1e-12)        
        
        # 开 P 次方
        pnorm_values = mean_term ** (1.0 / self._p_norm_factor)
        
        # 约束值
        return pnorm_values - 1.0

        
    def fun(self, 
            density: Union[Function, TensorLike], 
            state: Optional[Dict] = None,
            **kwargs
        ) -> TensorLike:
        """计算应力约束函数值"""
        sigma_vm = None
        weights = None
        
        if hasattr(self, '_cached_stress_state') and self._cached_stress_state is not None:
            sigma_vm = self._cached_stress_state['von_mises']
            weights = self._cached_stress_state['weights']

        if sigma_vm is None:
            stress_state = self._analyzer.compute_stress_state(state=state, rho_val=density)
            sigma_vm = stress_state['von_mises']
            weights = stress_state['weights']

        # 计算 P-norm
        val = self._compute_clustered_pnorm(sigma_vm, weights) # (N_clusterm, )
        
        return val
    
    def jac(self, 
            density: Function, 
            state: dict,
            **kwargs
        ) -> TensorLike:
        """计算应力约束的灵敏度"""
        pass