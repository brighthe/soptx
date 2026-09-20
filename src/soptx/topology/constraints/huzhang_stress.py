"""Hu-Zhang 局部应力约束的公共离散适配器."""

import math
from typing import Optional, Dict

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.core import BaseLogged
from soptx.fem.analyzers import HuZhangMFEMAnalyzer

from .exemption import EXEMPT_CONSTRAINT_VALUE, apply_exemption, validate_exemption_mask
from .stress_formulation import StressRelaxationFormulation


class HuZhangStressConstraint(BaseLogged):
    """组合 Hu-Zhang 原生表观应力与可替换的局部松弛公式.

    Parameters
    ----------
    analyzer : HuZhangMFEMAnalyzer
        提供表观应力恢复和伴随求解的混合有限元分析器.
    stress_limit : float
        材料许用应力, 必须为有限正数.
    formulation : StressRelaxationFormulation
        支持 apparent 原生应力表示的局部松弛公式.
    exemption_mask : TensorLike, optional
        形状 ``(NC,)`` 的布尔张量, True 表示该单元的应力评价点被移出约束集合.
        用于剔除载荷贴片端部等无法由设计消除的奇异点邻域; 详见
        ``soptx.topology.constraints.exemption``.
    enable_logging : bool, optional
        是否启用日志.
    logger_name : str, optional
        日志记录器名称.

    Notes
    -----
    本类负责有限元状态, 伴随载荷和隐式灵敏度. 约束值, 显式偏导及
    验收尺度由 formulation 提供, 不预设具体松弛模型.

    豁免只改变约束集合的成员, 不改变 AL 的归一化基数: ``fun`` 返回的张量形状
    不变, 上层仍按全部评价点数做平均, 使带豁免与不带豁免的两次运行罚项量级
    严格可比. 豁免点的约束值取严格可行的常值, 对目标与灵敏度的贡献恒为 0.
    """

    def __init__(
        self,
        analyzer: HuZhangMFEMAnalyzer,
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

    @property
    def analyzer(self) -> HuZhangMFEMAnalyzer:
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
        return "混合元"

    def _stress_ratio(self, state: Dict) -> TensorLike:
        return state['von_mises'] / self._stress_limit

    def compute_unexempted_constraint(self,
            density: TensorLike,
            state: Optional[Dict] = None,
            **kwargs
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

        # 1. 获取表观应力场 Σ
        if 'stress_apparent' not in state:
            state.update(self._analyzer.compute_stress_state(state=state, rho_val=density))

        # 2. 获取材料刚度插值系数 m_E(ρ)
        if 'stiffness_ratio' not in state:
            E_rho_cached = self._analyzer._E_rho
            if E_rho_cached is None:
                raise RuntimeError("stiffness_ratio 未缓存，请确保已执行分析器正向计算。")
            E0 = self._analyzer.material.youngs_modulus
            state['stiffness_ratio'] = E_rho_cached / E0

        m_E = state['stiffness_ratio'] # (NC,)

        # 3. 计算 von Mises 应力
        if 'von_mises' not in state:
            state['von_mises'] = self._analyzer.material.calculate_von_mises_stress(state['stress_apparent'])

        # 4. 由公式策略计算阈值与约束值.
        threshold = self._formulation.threshold(m_E)
        if threshold is not None:
            state['eta_threshold'] = threshold

        return self._formulation.constraint_value(
            stress_ratio=self._stress_ratio(state),
            stiffness_ratio=m_E,
            stress_representation="apparent",
        )

    def fun(self,
            density: TensorLike,
            state: Optional[Dict] = None,
            **kwargs
        ) -> TensorLike:
        """计算注入公式的局部约束值."""
        constraint_value = self.compute_unexempted_constraint(
            density=density, state=state, **kwargs
        )
        return apply_exemption(
            constraint_value, self._exemption_mask, EXEMPT_CONSTRAINT_VALUE
        )

    def compute_partial_gradient_wrt_mE(self, state: Dict) -> TensorLike:
        """计算固定混合元状态时约束对相对刚度的显式偏导数."""
        partial = self._formulation.partial_wrt_stiffness_ratio(
            stress_ratio=self._stress_ratio(state),
            stiffness_ratio=state['stiffness_ratio'],
            stress_representation="apparent",
        )
        return apply_exemption(partial, self._exemption_mask, 0.0)

    def compute_gradient_wrt_von_mises(self, state: Dict) -> TensorLike:
        """计算约束对表观 von Mises 应力的偏导数."""
        derivative = self._formulation.gradient_wrt_stress_ratio(
            stress_ratio=self._stress_ratio(state),
            stiffness_ratio=state['stiffness_ratio'],
            stress_representation="apparent",
        )
        derivative = apply_exemption(derivative, self._exemption_mask, 0.0)
        return derivative / self._stress_limit

    def compute_adjoint_load(self, dPenaldVM: TensorLike, state: Dict) -> TensorLike:
        """
        计算伴随载荷 (右端项).

        对于混合元，伴随载荷直接作用在应力自由度 Σ 上:
            F_adj = -(∂σ^v/∂Σ)^T * dPenaldVM
        """
        material = self._analyzer.material
        stress_vector = state['stress_apparent']  # (NC, NQ, NS)
        vm_val = state['von_mises']               # (NC, NQ)

        # 除零由 calculate_von_mises_stress 的平方量下限 1e-12 兜住, 即 vm >= 1e-6,
        # 此处不再另设阈值 (原有的 vm_val < 1e-12 判据永不成立, 已删除).

        # von Mises 投影矩阵 M
        M = material.von_mises_matrix() # (NS, NS)

        # 1. 计算 ∂σ^v / ∂Σ = (M * Σ) / σ^v
        M_sigma = bm.einsum('ij, ...j -> ...i', M, stress_vector) # (NC, NQ, NS)
        dVM_dSigma = M_sigma / vm_val[..., None]                  # (NC, NQ, NS)

        # 2. 获取应力基函数矩阵 Ψ
        stress_space = self._analyzer.huzhang_space
        disp_mesh = self._analyzer.disp_mesh
        integration_order = 1
        qf = disp_mesh.quadrature_formula(integration_order, 'cell')
        bcs, _ = qf.get_quadrature_points_and_weights()
        phi = stress_space.basis(bcs) # (NC, NQ, LDOF, NS) HuZhang 原生 [xx, xy, yy]
        phi = phi[..., [0, 2, 1]]     # 标准 Voigt [xx, yy, xy]

        # 3. 通过基函数将 ∂σ^v/∂σ_vec 映射到自由度空间
        element_sens = bm.einsum('cqls, cqs -> cql', phi, dVM_dSigma)  # (NC, NQ, LDOF)

        # 4. 应用罚函数权重并对积分点求和
        weights = dPenaldVM[..., None]                 # (NC, NQ, 1)
        element_loads = 1.0 * element_sens * weights   # (NC, NQ, LDOF)
        element_loads = bm.sum(element_loads, axis=1)  # (NC, LDOF)

        # 5. 全局组装至应力自由度
        gdofs = stress_space.number_of_global_dofs()
        adjoint_load = bm.zeros((gdofs,), dtype=bm.float64)

        cell2dof = stress_space.cell_to_dof()  # (NC, LDOF)
        indices = cell2dof.flatten()
        values = element_loads.flatten()

        bm.add_at(adjoint_load, indices, values)

        # sigma_basis = TM @ sigma_solve, 故对求解系数的导数为 TM.T @ rhs_basis.
        if stress_space.use_relaxation and stress_space.NCP > 0:
            adjoint_load = stress_space.TM.T @ adjoint_load
        return adjoint_load

    def compute_implicit_sensitivity_term(self, adjoint_vector: TensorLike, state: Dict) -> TensorLike:
        """
        计算隐式灵敏度项（预除 m_E²）: (1/m_E²) * λ_σ,e^T * A⁰_e * Σ_e
        通过物理掩码（Mask）防范孔洞区域除零溢出。
        """
        # --- 1. 提取基础数据 ---
        A0 = self._analyzer._cached_Ae0
        huzhang_space = self._analyzer.huzhang_space
        cell2dof = huzhang_space.cell_to_dof()
        gdof_sigma = huzhang_space.number_of_global_dofs()
        m_E = state['stiffness_ratio']  # (NC, )

        # --- 2. 提取完整的局部应力与伴随应力 (绝对不要在这里除以 m_E) ---
        sigma_coeff = state['stress'][:]
        lambda_coeff = adjoint_vector[:gdof_sigma]
        # Ae0 在原始基函数坐标中装配, 双线性型两侧均需转换.
        if huzhang_space.use_relaxation and huzhang_space.NCP > 0:
            sigma_coeff = huzhang_space.TM @ sigma_coeff
            lambda_coeff = huzhang_space.TM @ lambda_coeff
        sigma_e = sigma_coeff[cell2dof]
        lambda_sigma_e = lambda_coeff[cell2dof]

        # --- 3. 全局计算双线性型 W_e = λ_σ,e^T * A⁰_e * Σ_e ---
        # 这里只有纯乘法，即使在孔洞区遇到 1e-10 的数值噪声，乘完依然是极小数，绝对安全
        W_e = bm.einsum('ci, cij, cj -> c', lambda_sigma_e, A0, sigma_e)

        # --- 4. 物理掩码截断（核心防爆震逻辑） ---
        term = bm.zeros_like(m_E)    # 默认全域敏度为 0
        active = m_E > 1e-4          # 划定红线：只认有实质刚度的单元

        # --- 5. 安全除法 ---
        # 只有在非孔洞区域，才执行除以 m_E^2 的操作
        if bm.any(active):
            term[active] = W_e[active] / (m_E[active] ** 2)

        return term

    def compute_stress_measure(self, rho: TensorLike, state: Dict) -> TensorLike:
        """返回当前公式用于展示的无量纲应力测度."""
        return self._formulation.stress_measure(
            stress_ratio=self._stress_ratio(state),
            stiffness_ratio=state['stiffness_ratio'],
            stress_representation="apparent",
        )

    def compute_solid_stress_ratio(self, rho: TensorLike, state: Dict) -> TensorLike:
        """未加权的实体材料应力比 sigma^solid_vm / sigma_lim.

        本路径求解的是表观应力 sigma^app = m_E sigma^solid, 故实体应力比由表观
        量除以 m_E 还原, 使两条路径报告同一个物理量。只作诊断报告, 不参与任何
        停机判据; 验收量见 compute_relative_violation。
        依据: papers/huzhang-topopt/stress-constrained-topopt-models-and-algorithms.md
        (dut-postdoc) 第 6.4 节。

        还原次序: 先对应力张量除以 m_E 再取 von Mises, 不能反过来。von Mises 本身
        是一次正齐次的, 两种次序在精确算术下等价; 但 calculate_von_mises_stress 对
        平方量施加了 1e-12 下限 (即 vm >= 1e-6), 该下限破坏齐次性。空洞单元的表观
        应力恒低于此下限, 先取 vm 会读回常数 1e-6, 再除以 m_E = 1e-9 便放大成
        1e-6 / (1e-9 * sigma_lim) 的定值 (sigma_lim = 180 时恒为 5.5556), 与应力、
        阶次、设计全都无关。先做除法则下限落在实体应力上, 相对量级 1e-6 / sigma_lim,
        与 LFEM 路径 (直接对 stress_solid 取 vm) 口径一致。

        Parameters
        ----------
        rho : TensorLike
            当前物理密度, 本路径不用到, 保留以与 LFEM 路径同签名.
        state : dict
            当前密度的状态, 须先调用 fun 刷新应力与阈值缓存.

        Returns
        -------
        TensorLike
            各评价点的未加权实体应力比.
        """
        m_E = state['stiffness_ratio']
        if not bool(bm.all(bm.isfinite(m_E) & (m_E > 0))):
            raise ValueError('相对刚度必须有限且为正')

        stress_apparent = state['stress_apparent']              # (NC, NQ, NS)
        stress_solid = stress_apparent / m_E[:, None, None]     # (NC, NQ, NS)
        von_mises_solid = self._analyzer.material.calculate_von_mises_stress(
            stress_solid
        )                                                        # (NC, NQ)

        return von_mises_solid / self._stress_limit

    def compute_relative_violation(self, rho: TensorLike, state: Dict) -> TensorLike:
        """返回当前公式定义的验收超限量.

        具体尺度由注入公式定义, 不根据有限元类型推断. 豁免单元返回严格可行的
        常值, 因此不进入 C2 验收; 这些单元的真实应力水平由
        ``compute_solid_stress_ratio`` 如实报告, 不被掩盖.
        """
        stiffness_ratio = state['stiffness_ratio']
        if not bool(bm.all(bm.isfinite(stiffness_ratio) & (stiffness_ratio > 0))):
            raise ValueError("应力阈值或相对刚度必须有限且为正")

        result = self._formulation.acceptance_violation(
            stress_ratio=self._stress_ratio(state),
            stiffness_ratio=stiffness_ratio,
            stress_representation="apparent",
        )
        result = apply_exemption(
            result, self._exemption_mask, EXEMPT_CONSTRAINT_VALUE
        )
        if not bool(bm.all(bm.isfinite(result))):
            raise ValueError("相对约束超限量包含非有限值")
        return result
