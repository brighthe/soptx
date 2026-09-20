import math
from typing import Optional, Dict
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.core import BaseLogged
from soptx.fem.analyzers import LagrangeFEMAnalyzer

from .exemption import EXEMPT_CONSTRAINT_VALUE, apply_exemption, validate_exemption_mask
from .stress_formulation import StressRelaxationFormulation

class LagrangeStressConstraint(BaseLogged):
    """LFEM 局部应力约束的公共离散适配器.

    本类负责实体应力状态、伴随载荷和隐式灵敏度. 约束值、显式偏导和
    验收语义由注入的 formulation 提供, 因此新增松弛公式不需要修改 AL 层.

    Parameters
    ----------
    analyzer : LagrangeFEMAnalyzer
        位移有限元分析器.
    stress_limit : float
        材料许用应力.
    formulation : StressRelaxationFormulation
        局部应力松弛公式.
    exemption_mask : TensorLike, optional
        形状 ``(NC,)`` 的布尔张量, True 表示该单元的应力评价点被移出约束集合.
        用于剔除载荷贴片端部等无法由设计消除的奇异点邻域; 详见
        ``soptx.topology.constraints.exemption``.

    Notes
    -----
    豁免只改变约束集合的成员, 不改变 AL 的归一化基数: ``fun`` 返回的张量形状
    不变, 上层仍按全部评价点数做平均. 这样带豁免与不带豁免的两次运行, 罚项量级
    严格可比. 豁免点的约束值取严格可行的常值, 对目标与灵敏度的贡献恒为 0.
    """
    def __init__(
        self,
        analyzer: LagrangeFEMAnalyzer,
        stress_limit: float,
        formulation: StressRelaxationFormulation,
        exemption_mask: Optional[TensorLike] = None,
        enable_logging: bool = False,
        logger_name: Optional[str] = None,
    ) -> None:
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        if not math.isfinite(stress_limit) or stress_limit <= 0:
            raise ValueError("stress_limit 必须为有限正数")

        self._analyzer = analyzer
        self._stress_limit = stress_limit
        self._formulation = formulation
        self._exemption_mask = validate_exemption_mask(
            exemption_mask, analyzer.disp_mesh.number_of_cells()
        )

        self._interpolation_scheme = self._analyzer.interpolation_scheme
        self._n_sub = self._interpolation_scheme.n_sub if self._interpolation_scheme.n_sub is not None else 1
        self._is_multiresolution = (self._n_sub > 1)

    @property
    def analyzer(self) -> LagrangeFEMAnalyzer:
        """获取当前的分析器"""
        return self._analyzer

    @property
    def formulation(self) -> StressRelaxationFormulation:
        """获取当前局部应力松弛公式."""
        return self._formulation

    @property
    def exemption_mask(self) -> Optional[TensorLike]:
        """获取应力约束豁免单元掩码, 未豁免任何单元时为 None."""
        return self._exemption_mask

    @property
    def discretization_name(self) -> str:
        """返回灵敏度离散路径名称."""
        return "位移元"

    def _prepare_stress_state(
        self,
        density: TensorLike,
        state: Dict,
    ) -> None:
        """准备 LFEM 约束公式共享的求解器状态."""
        if 'stress_solid' not in state:
            state.update(self._analyzer.compute_stress_state(state))

        if getattr(self._analyzer, 'poisson_ratio_interpolated', False):
            raise RuntimeError(
                "LFEM 应力约束的隐式项假定 K_e = E(rho)/E0 * K_e0, "
                "不支持泊松比随密度插值的分析器"
            )

        if 'stiffness_ratio' not in state:
            cached = self._analyzer._cached_stiffness_relative
            if cached is None:
                raise RuntimeError(
                    "stiffness_ratio 未缓存: 请确保在调用 fun() 前已完成有限元分析, "
                    "使得 analyzer._cached_stiffness_relative 已被计算."
                )
            state['stiffness_ratio'] = cached

        if 'von_mises' not in state:
            state['von_mises'] = self._analyzer.material.calculate_von_mises_stress(
                state['stress_solid']
            )

    def _stress_ratio(self, state: Dict) -> TensorLike:
        return state['von_mises'] / self._stress_limit

    def compute_unexempted_constraint(
        self,
        density: TensorLike,
        state: Optional[Dict] = None,
        **kwargs,
    ) -> TensorLike:
        """计算未施加豁免的局部约束值.

        与 ``fun`` 的唯一差别是不把豁免单元换成 ``EXEMPT_CONSTRAINT_VALUE``,
        因此垫片内部的真实超限量可被诊断到. 优化过程不应调用本方法: 豁免区
        的约束值发散, 进入 AL 会重现被豁免掉的病态.

        Parameters
        ----------
        density : TensorLike
            物理密度场.
        state : dict, optional
            求解状态缓存; 为 None 时新建并就地填充.

        Returns
        -------
        TensorLike
            形状与 ``fun`` 一致的约束值, 豁免单元取其原始值.
        """
        if state is None:
            state = {}

        self._prepare_stress_state(density=density, state=state)

        threshold = self._formulation.threshold(state['stiffness_ratio'])
        if threshold is not None:
            state['eta_threshold'] = threshold

        return self._formulation.constraint_value(
            stress_ratio=self._stress_ratio(state),
            stiffness_ratio=state['stiffness_ratio'],
            stress_representation="solid",
        )

    def fun(
        self,
        density: TensorLike,
        state: Optional[Dict] = None,
        **kwargs,
    ) -> TensorLike:
        """计算当前松弛公式的局部约束值."""
        constraint_value = self.compute_unexempted_constraint(
            density=density, state=state, **kwargs
        )
        return apply_exemption(
            constraint_value, self._exemption_mask, EXEMPT_CONSTRAINT_VALUE
        )

    def compute_partial_gradient_wrt_mE(self, state: Dict) -> TensorLike:
        """计算固定 LFEM 状态时约束对相对刚度的显式偏导数."""
        partial = self._formulation.partial_wrt_stiffness_ratio(
            stress_ratio=self._stress_ratio(state),
            stiffness_ratio=state['stiffness_ratio'],
            stress_representation="solid",
        )
        return apply_exemption(partial, self._exemption_mask, 0.0)

    def compute_gradient_wrt_von_mises(self, state: Dict) -> TensorLike:
        """计算约束对实体 von Mises 应力的偏导数."""
        derivative = self._formulation.gradient_wrt_stress_ratio(
            stress_ratio=self._stress_ratio(state),
            stiffness_ratio=state['stiffness_ratio'],
            stress_representation="solid",
        )
        derivative = apply_exemption(derivative, self._exemption_mask, 0.0)
        return derivative / self._stress_limit

    def compute_adjoint_load(self, dPenaldVM: TensorLike, state: Dict) -> TensorLike:
        """计算伴随方程的右端项.
            K_T ξ = -Σ_j [λ_j + μ h_j] · ∂h_j/∂U

          其中 ∂h_j/∂U 通过链式法则展开:
               ∂h_j/∂U = (∂g_j/∂σ^v_j) · (∂σ^v_j/∂σ) · (∂σ/∂U)

        各项分别为:
        - ∂g_j/∂σ^v_j: 约束关于 von Mises 应力的导数 (由 dPenaldVM 加权传入)
        - ∂σ^v_j/∂σ = V₀σ / σ^v
        - ∂σ/∂U = D·B

        Parameters
        ----------
        dPenaldVM : TensorLike, shape (NC, NQ)
            罚函数关于 von Mises 应力的加权导数:
            dPenaldVM = (λ + μ·h) · ∂g/∂σ^v
            由 AugmentedLagrangianObjective 计算后传入.
        state : dict
            必须包含:
            - 'stress_solid': 实体 Cauchy 应力向量, shape (NC, NQ, NS)
            - 'von_mises': von Mises 应力标量, shape (NC, NQ)

        Returns
        -------
        adjoint_load : TensorLike, shape (gdofs,)
            组装后的全局伴随载荷向量.
        """
        material = self._analyzer.material
        disp_space = self._analyzer.tensor_space

        # --- 获取应变位移矩阵 B, 刚度矩阵 D, 和 von Mises 投影矩阵 M ---
        # 单分辨率: (NC, NQ, NS, LDOF) | 多分辨率: (NC, n_sub, NQ, NS, LDOF)
        B = self._analyzer.compute_strain_matrix(integration_order=1)
        D = material.elastic_matrix()[0, 0]    # (NS, NS)
        M = material.von_mises_matrix()        # (NS, NS)

        # --- 计算 dVM / dSigma (von Mises 应力相对于 Cauchy 应力张量的导数)---
        stress_vector = state['stress_solid'] # 单分辨率: (NC, NQ, NS) | 多分辨率: (NC, n_sub, NQ, NS)
        vm_val = state['von_mises']           # 单分辨率: (NC, NQ)     | 多分辨率: (NC, n_sub, NQ)
        vm_safe = bm.where(vm_val < 1e-12, 1.0, vm_val)

        if self._is_multiresolution:
            # (NC, n_sub, NQ, NS)
            M_sigma = bm.einsum('ij, csnj -> csni', M, stress_vector)
            dVM_dSigma = M_sigma / vm_safe[..., None]

            term1_eps = bm.einsum('kl, csnk -> csnl', D, dVM_dSigma) # # (NC, n_sub, NQ, NS)

            element_sens = bm.einsum('csnkl, csnk -> csnl', B, term1_eps) # (NC, n_sub, NQ, LDOF)

            weights = dPenaldVM[..., None]                      # (NC, n_sub, NQ, 1)
            element_loads = -1.0 * element_sens * weights       # (NC, n_sub, NQ, LDOF)
            element_loads = bm.sum(element_loads, axis=(1, 2))  # (NC, LDOF)

        else:
            # (NC, NQ, NS)
            M_sigma = bm.einsum('ij, cqj -> cqi', M, stress_vector)
            dVM_dSigma = M_sigma / vm_safe[..., None]

            # --- 应力 -> 应变 ( D^T * dVM/dSigma ) ---
            term1_eps = bm.einsum('kl, cqk -> cql', D, dVM_dSigma) # (NC, NQ, NS)

            # --- 应变 -> 位移 ( B^T * term1_eps ) ---
            element_sens = bm.einsum('cqkl, cqk -> cql', B, term1_eps) # (NC, NQ, LDOF)

            # --- 应用罚函数权重 dPenaldVM ---
            weights = dPenaldVM[..., None] # (NC, NQ)

            # 单元级伴随载荷 (来自伴随方程定义 F_adj = - dP/dU)
            element_loads = -1.0 * element_sens * weights # (NC, NQ, LDOF)

            element_loads = bm.sum(element_loads, axis=1) # (NC, LDOF)

        # --- 全局组装 ---
        cell2dof = disp_space.cell_to_dof() # (NC, LDOF)
        indices = cell2dof.flatten()
        values = element_loads.flatten()

        # 初始化全局载荷向量
        gdofs = disp_space.number_of_global_dofs()
        adjoint_load = bm.zeros((gdofs, ), dtype=bm.float64)

        # 累加组装
        bm.add_at(adjoint_load, indices, values)

        return adjoint_load

    def compute_implicit_sensitivity_term(self,
                                          adjoint_vector: TensorLike,
                                          state: Dict
                                        ) -> TensorLike:
        """计算伴随法隐式灵敏度项
            ξ^T · (∂F_int/∂E_ℓ)"""
        # 提取位移向量 U 和伴随向量 ψ
        uh = state['displacement']

        # 确保拉平为 1D 以便切片
        uh_flat = uh.reshape(-1)
        psi_flat = adjoint_vector.reshape(-1)

        # 提取单元级的局部向量 (NC, TLDOF)
        cell2dof = self._analyzer.tensor_space.cell_to_dof()
        uh_e = uh_flat[cell2dof]
        psi_e = psi_flat[cell2dof]

        if not self._is_multiresolution:
            if self._analyzer._cached_ke0 is None:
                K0 = self._analyzer.compute_solid_stiffness_matrix()
            else:
                K0 = self._analyzer._cached_ke0  # (NC, LDOF, LDOF)

            implicit_term = bm.einsum('ci, cij, cj -> c', psi_e, K0, uh_e) # (NC, )

        else:
            if self._analyzer._cached_ke0_sub is None:
                ke0_sub = self._analyzer.compute_sub_element_stiffness_matrix()
            else:
                ke0_sub = self._analyzer._cached_ke0_sub  # (NC, n_sub, TLDOF, TLDOF)

            implicit_term = bm.einsum('ci, csij, cj -> cs', psi_e, ke0_sub, uh_e)  # (NC, n_sub)

        return implicit_term

    def compute_stress_measure(self, rho: TensorLike, state: Dict) -> TensorLike:
        """返回当前公式用于展示的无量纲应力测度."""
        return self._formulation.stress_measure(
            stress_ratio=self._stress_ratio(state),
            stiffness_ratio=state['stiffness_ratio'],
            stress_representation="solid",
        )

    def compute_solid_stress_ratio(self, rho: TensorLike, state: Dict) -> TensorLike:
        """返回未加权的实体材料应力比, 仅用于物理诊断."""
        return self._stress_ratio(state)

    def compute_relative_violation(self, rho: TensorLike, state: Dict) -> TensorLike:
        """返回当前公式定义的验收超限量.

        对多项式消失约束, 该量沿用历史加权口径
        m_E * sigma_vm_solid / sigma_lim - 1, 不等于 AL 使用的多项式约束值 g.
        对 epsilon 松弛约束, 该量就是 AL 约束值 g.
        豁免单元返回严格可行的常值, 因此不进入 C2 验收; 这些单元的真实应力
        水平由 ``compute_solid_stress_ratio`` 如实报告, 不被掩盖.
        """
        stiffness_ratio = state['stiffness_ratio']
        if not bool(bm.all(bm.isfinite(stiffness_ratio) & (stiffness_ratio > 0))):
            raise ValueError("应力阈值或相对刚度必须有限且为正")

        result = self._formulation.acceptance_violation(
            stress_ratio=self._stress_ratio(state),
            stiffness_ratio=stiffness_ratio,
            stress_representation="solid",
        )
        result = apply_exemption(
            result, self._exemption_mask, EXEMPT_CONSTRAINT_VALUE
        )
        if not bool(bm.all(bm.isfinite(result))):
            raise ValueError("相对约束超限量包含非有限值")
        return result
