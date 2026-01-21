from typing import Optional, Literal, Union, Tuple, Dict
import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function
from fealpy.mesh import SimplexMesh

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
                diff_mode: Literal["auto", "manual"] = "manual",      
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self._analyzer = analyzer
        self._stress_limit = stress_limit

        self._p_norm_factor = p_norm_factor
        self._n_clusters = n_clusters
        self._recluster_freq = recluster_freq

        self._diff_mode = diff_mode
        
        # 聚类相关变量
        self._clustering_map = None
        self._cluster_weight_sums = None

        self._cached_stress_state = None

        # P-norm 计算的缓存
        self._cached_normalized_stress = None
        self._cached_pnorm_normalized = None

        # 积分权重缓存
        self._integration_weights = None

    def fun(self, 
            density: Union[Function, TensorLike], 
            state: Optional[Dict] = None,
            iter_idx: Optional[int] = None,
            **kwargs
        ) -> TensorLike:
        """计算应力约束函数值
        
        Parameters
        ----------
        density : 物理密度场
        state : 状态变量（位移场等）
        iter_idx : 当前迭代次数，用于控制聚类更新频率
        """
        self._update_clustering(iter_idx=iter_idx, state=state, density=density)
        
        if self._cached_stress_state is not None:
            sigma_vm = self._cached_stress_state['von_mises']
        else:
            stress_state = self._analyzer.compute_stress_state(state=state, rho_val=density)
            sigma_vm = stress_state['von_mises']

            self._cached_stress_state = stress_state

        weights = self._get_integration_weights()
        
        val = self._compute_clustered_pnorm(sigma_vm, weights)
        
        return val
    
    def _update_clustering(self, 
                        iter_idx: int, 
                        state: Dict, 
                        density: TensorLike
                    ) -> None:
        """根据当前迭代步数和物理状态更新聚类"""
        # 如果没有传入 iter_idx，默认不更新
        if iter_idx is None:
            if self._clustering_map is None:
                iter_idx = 0  
            else:
                return  
        
        should_update = (self._recluster_freq > 0) and (iter_idx % self._recluster_freq == 0)
        
        if self._clustering_map is None:
            should_update = True
        
        if should_update:
            stress_state = self._analyzer.compute_stress_state(state=state, rho_val=density)
            sigma_vm = stress_state['von_mises'] # (NC, NQ)

            weights = self._get_integration_weights() # (NC, NQ)

            # 执行核心聚类算法 (排序、切分等)
            self._perform_clustering_logic(sigma_vm, weights)

            self._cached_stress_state = stress_state
        
        else:
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
    
    def _get_integration_weights(self) -> TensorLike:
        """计算 P-norm 聚合所需的积分权重 detJ * w_q"""
        if self._integration_weights is not None:
            return self._integration_weights
    
        mesh = self._analyzer.disp_mesh
        interpolation_scheme = self._analyzer.interpolation_scheme
        integration_order = self._analyzer.integration_order

        density_location = interpolation_scheme.density_location
        if density_location == 'element':
            qf = mesh.quadrature_formula(integration_order)
            bcs, ws = qf.get_quadrature_points_and_weights() 
            
            if isinstance(mesh, SimplexMesh):
                cm = mesh.entity_measure('cell')
                weights = cm[:, None] * ws  # (NC, NQ)
            else:
                J = mesh.jacobi_matrix(bcs)
                detJ = bm.abs(bm.linalg.det(J))  # (NC, NQ)
                weights = detJ * ws # (NC, NQ) 
            
        elif density_location == 'element_multiresolution':
            pass

        self._integration_weights = weights
        
        return weights
        
    def _compute_clustered_pnorm(self, 
                                sigma_vm: TensorLike,
                                weights: TensorLike
                            ) -> TensorLike:
        """计算归一化聚类 P-norm 应力约束值

        Parameters
        ----------
        sigma_vm: von Mises 应力场
            - STOP: (NC, NQ)
            - MTOP: (NC, n_sub, NQ)
        weights: 积分权重 (NC, NQ)

        Returns
        -------
        constraint_values: (n_clusters, ) 约束函数值
        """
        P = self._p_norm_factor
        sigma_lim = self._stress_limit
        
        vals = sigma_vm.flatten()
        ws = weights.flatten()
        
        if self._clustering_map is None or self._cluster_weight_sums is None:
            self._perform_clustering_logic(sigma_vm, weights)
        
        # 归一化应力
        normalized_stress = vals / sigma_lim

        # 加权项
        base_term = (bm.maximum(normalized_stress, 0.0) + 1e-12) ** P
        weighted_term = ws * base_term
        
        # 聚类求和
        aggregated_numerator = bm.zeros((self._n_clusters,), dtype=vals.dtype)
        bm.add_at(aggregated_numerator, self._clustering_map, weighted_term)
        
        # 平均化修正
        mean_term = aggregated_numerator / (self._cluster_weight_sums + 1e-12)        
        
        # 开 P 次方
        pnorm_normalized = mean_term ** (1.0 / P)
        
        self._cached_normalized_stress = normalized_stress
        self._cached_pnorm_normalized = pnorm_normalized
        
        # 约束值
        return pnorm_normalized - 1.0
       
    
    def jac(self, 
            density: Union[Function, TensorLike], 
            state: Optional[dict] = None,
            diff_mode: Optional[Literal["auto", "manual"]] = None,
            **kwargs
        ) -> TensorLike:
        """计算应力约束相对于物理密度的灵敏度"""
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
        """手动计算应力约束相对于物理密度的灵敏度"""
        if self._cached_stress_state is not None:
            stress_state = self._cached_stress_state
        else:
            stress_state = self._analyzer.compute_stress_state(state=state, rho_val=density)

        weights = self._get_integration_weights()  # (NC, NQ)

        NC, NQ = weights.shape

        eta_sigma = stress_state['eta_sigma']                # (NC, )
        stress_solid = stress_state['stress_solid']          # (NC, NQ, NS)
        stress_penalized = stress_state['stress_penalized']  # (NC, NQ, NS)
        sigma_vm = stress_state['von_mises']                 # (NC, NQ)

        # 确保聚类已初始化
        if self._clustering_map is None:
            self._perform_clustering_logic(sigma_vm, weights)

        # ===================== 计算链式法则各项偏导 =====================
        # ∂g_m/∂σ^vM: P-norm 对 von Mises 应力的偏导
        dg_dsigma_vm = self._compute_pnorm_derivative(sigma_vm, weights)  # (n_clusters, n_points)
        
        # ∂σ^vM/∂σ: von Mises 对应力张量的偏导
        dvm_dsigma = self._compute_vm_stress_derivative(stress_penalized, sigma_vm)  # (NC, NQ, NS)

        # ===================== 组装伴随载荷并求解伴随方程 =====================
        # 组装伴随载荷向量 L_m
        L = self._assemble_adjoint_load(
                            dg_dsigma_vm=dg_dsigma_vm,
                            dvm_dsigma=dvm_dsigma,
                            eta_sigma=eta_sigma,
                        )  # (n_clusters, n_gdof)
        
        # 求解伴随方程 K @ λ_m = L_m
        L_transposed = L.T  # (n_gdof, n_clusters)
        adjoint_lambda_transposed = self._analyzer.solve_adjoint(L_transposed, rho_val=density)
        adjoint_lambda = adjoint_lambda_transposed.T  # (n_clusters, n_gdof)
        
        # ===================== 计算灵敏度两项 =====================
        # 显式密度项 (∂σ^vM/∂σ)^T · σ̂ solid (∂η_σ/∂ρ)
        interpolation_scheme = self._analyzer.interpolation_scheme
        stress_deriv = interpolation_scheme.interpolate_stress_derivative(
                                                                rho_val=density,
                                                            )
        deta_sigma_drho = stress_deriv['deta_sigma_drho']  # (NC, )
        # (∂σ^vM/∂σ) · σ^solid
        vm_dot_sigma_hat = bm.einsum('eqi, eqi -> eq', dvm_dsigma, stress_solid)  # (NC, NQ)
        explicit_common  = deta_sigma_drho[:, None]  * vm_dot_sigma_hat  # (NC, NQ)
        explicit_flat = explicit_common.flatten()       # (n_points, )
        
        # Term I = ∂g_m/∂σ^vM · explicit_common
        term_I_full = dg_dsigma_vm * explicit_flat[None, :]  # (n_clusters, n_points)
        # 按单元聚合（对积分点求和）
        term_I_cell = term_I_full.reshape(self._n_clusters, NC, NQ).sum(axis=2)  # (n_clusters, NC)

        # 隐式伴随项 λ_m^T · (∂K_e/∂ρ) · U_e
        # 获取单元位移
        tensor_space = self._analyzer.tensor_space
        cell2dof = tensor_space.cell_to_dof()  # (NC, n_ldof)
        uh = state['displacement']   # (n_gdof,)
        uh_e = uh[cell2dof]          # (NC, n_ldof)
        
        # 计算刚度矩阵的导数
        dKE_drho = self._analyzer.compute_stiffness_matrix_derivative(rho_val=density) # (NC, n_ldof, n_ldof)
        dKE_uh  = bm.einsum('eij, ej -> ei', dKE_drho, uh_e)  # (NC, n_ldof)

        # 提取各聚类的单元伴随向量
        lambda_e_all = adjoint_lambda[:, cell2dof]  # (n_clusters, NC, n_ldof)
        # Term II = λ_e^T · (∂K_e/∂ρ · U_e)
        term_II_cell = bm.einsum('mci, ci -> mc', lambda_e_all, dKE_uh)  # (n_clusters, NC)

        # 组合两项得到最终局部灵敏度
        local_sensitivity = term_I_cell - term_II_cell  # (n_clusters, NC)
        
        return local_sensitivity

    def _compute_pnorm_derivative(self, 
                                sigma_vm: TensorLike, 
                                weights: TensorLike
                            ) -> TensorLike:
        """计算归一化 P-norm 聚合函数对 von Mises 应力的偏导数
        
        Parameters
        ----------
        sigma_vm : (NC, NQ) von Mises 应力
        weights : (NC, NQ) 积分权重
        
        Returns
        -------
        dg_dsigma_vm : (n_clusters, n_points) 每个聚类对每个应力点的偏导数
        """
        P = self._p_norm_factor
        sigma_lim = self._stress_limit

        if hasattr(self, '_cached_normalized_stress') and self._cached_normalized_stress is not None:
            normalized_stress = self._cached_normalized_stress
            pnorm_normalized = self._cached_pnorm_normalized
        else:
            self._compute_clustered_pnorm(sigma_vm, weights)
            normalized_stress = self._cached_normalized_stress
            pnorm_normalized = self._cached_pnorm_normalized

        n_points = normalized_stress.shape[0]

        ws = weights.flatten()   

        # 计算 (σ_m^PN / σ_lim)^(1-P)
        pnorm_factor = (pnorm_normalized + 1e-12) ** (1.0 - P)  

        # 计算 (σ^vM_{e,i} / σ_lim)^(P-1)
        stress_factor = (bm.maximum(normalized_stress, 0.0) + 1e-12) ** (P - 1.0)  

        # 构建完整的导数矩阵 (n_clusters, n_points)
        dg_dsigma_vm = bm.zeros((self._n_clusters, n_points), dtype=normalized_stress.dtype)

        for m in range(self._n_clusters):
            # 找到属于聚类 m 的点
            mask = (self._clustering_map == m)
            # 计算该聚类的导数系数
            coeff = pnorm_factor[m] / (self._cluster_weight_sums[m] * sigma_lim + 1e-12)
            # 对属于该聚类的点，乘以各点的 stress_factor 和权重
            dg_dsigma_vm[m, mask] = coeff * ws[mask] * stress_factor[mask]
        
        return dg_dsigma_vm
    
    def _compute_vm_stress_derivative(self, 
                                    stress_penalized: TensorLike,
                                    sigma_vm: TensorLike,
                                ) -> TensorLike:
        """
        计算 von Mises 应力对应力张量分量的偏导数
        
        Parameters
        ----------
        stress_penalized : (NC, NQ, NS) 惩罚后的应力张量分量
            - 2D: NS = 3, [σ_xx, σ_yy, τ_xy]
            - 3D: NS = 6, [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_zx]
        sigma_vm : (NC, NQ) von Mises 应力
        
        Returns
        -------
        dvm_dsigma : (NC, NQ, NS) von Mises 应力对各应力分量的偏导
        """
        NS = stress_penalized.shape[-1]

        vm_safe = sigma_vm + 1e-12
        
        if NS == 3:
            # 2D 平面应力: σ = [σ_x, σ_y, τ_xy]
            sigma_x = stress_penalized[..., 0]
            sigma_y = stress_penalized[..., 1]
            tau_xy = stress_penalized[..., 2]

            dvm_dsigma = bm.stack([
                                (2.0 * sigma_x - sigma_y) / (2.0 * vm_safe),
                                (2.0 * sigma_y - sigma_x) / (2.0 * vm_safe),
                                (6.0 * tau_xy) / (2.0 * vm_safe)
                            ], axis=-1) 
            
        elif NS == 6:
            # 3D: σ = [σ_x, σ_y, σ_z, τ_xy, τ_yz, τ_zx]
            sigma_x = stress_penalized[..., 0]
            sigma_y = stress_penalized[..., 1]
            sigma_z = stress_penalized[..., 2]
            tau_xy = stress_penalized[..., 3]
            tau_yz = stress_penalized[..., 4]
            tau_zx = stress_penalized[..., 5]
                        
            dvm_dsigma = bm.stack([
                                (2.0 * sigma_x - sigma_y - sigma_z) / (2.0 * vm_safe),
                                (2.0 * sigma_y - sigma_x - sigma_z) / (2.0 * vm_safe),
                                (2.0 * sigma_z - sigma_x - sigma_y) / (2.0 * vm_safe),
                                (6.0 * tau_xy) / (2.0 * vm_safe),
                                (6.0 * tau_yz) / (2.0 * vm_safe),
                                (6.0 * tau_zx) / (2.0 * vm_safe)
                            ], axis=-1)  
        
        else:
            error_msg = f"Unsupported stress component number NS={NS} for von Mises derivative."
            self._log_error(error_msg)
        
        return dvm_dsigma
    
    def _assemble_adjoint_load(self,
                           dg_dsigma_vm: TensorLike,
                           dvm_dsigma: TensorLike,
                           eta_sigma: TensorLike,
                        ) -> TensorLike:
        """
        组装伴随载荷向量
        
        L_m = Σ_{(e,i)∈Ω_m} (∂g_m/∂σ^vM) * η_σ * B^T * D_0 * (∂σ^vM/∂σ)
        
        Parameters
        ----------
        dg_dsigma_vm : (n_clusters, n_points) P-norm 对 von Mises 应力的偏导
        dvm_dsigma : (NC, NQ, NS) von Mises 对应力分量的偏导
        eta_sigma : (NC, ) 应力惩罚因子
        
        Returns
        -------
        L : (n_clusters, n_gdof) 伴随载荷向量
        """
        material = self._analyzer.material
        tensor_space = self._analyzer.tensor_space

        D0 = material.elastic_matrix()[0, 0]  # (NS, NS)
        B = self._analyzer.compute_strain_displacement_matrix()  # (NC, NQ, NS, n_ldof)

        n_points = dg_dsigma_vm.shape[-1]
        NQ = dvm_dsigma.shape[1]
        n_gdof = tensor_space.number_of_global_dofs()

        cell2dof = tensor_space.cell_to_dof()  # (NC, n_ldof)

        dvm_dsigma_flat = dvm_dsigma.reshape(n_points, -1)  # (n_points, NS)
        B_flat = B.reshape(n_points, B.shape[-2], -1)  # (n_points, NS, n_ldof)

        # 计算 D_0 @ (∂σ^vM/∂σ)
        D_dvm = bm.einsum('ij, pj -> pi', D0, dvm_dsigma_flat)  # (n_points, NS)

        # 将 eta_sigma 从 (NC,) 扩展到 (n_points, ), 同一单元的所有积分点共享相同的惩罚因子
        eta_expanded = bm.repeat(eta_sigma, NQ)  # (NC, ) -> (NC*NQ, ) = (n_points, )

        # 计算 B^T @ D_0 @ (∂σ^vM/∂σ)
        BT_D_dvm = bm.einsum('pij, pi -> pj', B_flat, D_dvm)  # (n_points, n_ldof)

        # 乘以应力惩罚因子 η_σ * B^T @ D_0 @ (∂σ^vM/∂σ)
        local_contrib = eta_expanded[:, None] * BT_D_dvm  # (n_points, n_ldof)

        # 扩展 cell2dof：每个积分点对应其所在单元的自由度
        cell2dof_expanded = bm.repeat(cell2dof, NQ, axis=0)  # (n_points, n_ldof)

        # 初始化伴随载荷向量
        L = bm.zeros((self._n_clusters, n_gdof), dtype=dg_dsigma_vm.dtype)

        # 组装各聚类的伴随载荷
        for m in range(self._n_clusters):
            dg_m = dg_dsigma_vm[m, :]  # (n_points,)
            weighted_contrib = dg_m[:, None] * local_contrib  # (n_points, n_ldof)
            
            indices = cell2dof_expanded.flatten()
            values = weighted_contrib.flatten()
            bm.add_at(L[m], indices, values)

        return L

    def normalize(self, 
              con_val: TensorLike, 
              con_grad: TensorLike
          ) -> Tuple[TensorLike, TensorLike]:
        """标准化应力约束值和梯度"""

        return con_val, con_grad




        