"""
应力约束拓扑优化后处理模块

对应 MATLAB PolyScript_quad.m 中优化完成后的后处理部分:
  1. 应力约束检查 (归一化应力测度 SM = E · σ^v / σ_lim)
  2. 实体单元统计 (ρ > 阈值)
  3. von Mises 屈服面可视化

使用示例:
    from soptx.postprocessing.stress_post import StressPostProcessor

    post = StressPostProcessor(
        optimizer=optimizer,
        analyzer=analyzer,
        stress_limit=100.0,
    )

    # 运行完整后处理
    results = post.run(rho_phys=rho_opt, design_variable=d)

    # 或分步调用
    results = post.check_stress_constraints(rho_phys=rho_opt)
    post.print_summary(results)
    post.plot_yield_surface(results)
    post.plot_density_and_stress(results)
"""

from typing import Dict, Optional, Union
from dataclasses import dataclass, field

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

import numpy as np


@dataclass
class StressPostResults:
    """后处理结果数据容器"""
    # 全局统计
    max_SM: float = 0.0                         # 最大归一化应力 max(SM)
    mean_SM: float = 0.0                        # 平均归一化应力
    num_violated: int = 0                       # 违反约束的单元数
    num_total: int = 0                          # 总单元数
    max_violation_pct: float = 0.0              # 最大违反百分比

    # 实体单元统计 (ρ > solid_threshold)
    num_solid: int = 0                          # 实体单元数
    max_SM_solid: float = 0.0                   # 实体单元最大归一化应力
    mean_SM_solid: float = 0.0                  # 实体单元平均归一化应力
    num_violated_solid: int = 0                 # 违反约束的实体单元数

    # 体积分数
    volume_fraction: float = 0.0

    # 字段数据 (用于可视化)
    SM: Optional[TensorLike] = field(default=None, repr=False)          # (NC,) 归一化应力
    V: Optional[TensorLike] = field(default=None, repr=False)           # (NC,) 物理密度
    E: Optional[TensorLike] = field(default=None, repr=False)           # (NC,) 刚度插值
    sig_1_norm: Optional[TensorLike] = field(default=None, repr=False)  # (NC,) 归一化主应力 1
    sig_2_norm: Optional[TensorLike] = field(default=None, repr=False)  # (NC,) 归一化主应力 2
    solid_mask: Optional[TensorLike] = field(default=None, repr=False)  # (NC,) bool


class StressPostProcessor:
    """应力约束拓扑优化后处理器.
    
    Parameters
    ----------
    analyzer : LagrangeFEMAnalyzer or HuZhangMFEMAnalyzer
        有限元分析器.
    stress_limit : float
        材料应力极限 σ_lim.
    solid_threshold : float
        实体单元判定阈值, 默认 0.5.
    constraint_tolerance : float
        约束违反判定容差, 默认 0.01 (1%).
    """
    def __init__(self,
                 analyzer,
                 stress_limit: float,
                 solid_threshold: float = 0.5,
                 constraint_tolerance: float = 0.01,
                ) -> None:
        self._analyzer = analyzer
        self._stress_limit = stress_limit
        self._solid_threshold = solid_threshold
        self._constraint_tolerance = constraint_tolerance

    def check_stress_constraints(self, 
                                  rho_phys: TensorLike,
                                  state: Optional[Dict] = None,
                                ) -> StressPostResults:
        """计算归一化应力测度并进行约束检查.
        
        对应 MATLAB:
            SM = E .* fem.VM_Stress0 / fem.SLim;
            
        Parameters
        ----------
        rho_phys : TensorLike
            最终优化的物理密度场.
        state : dict, optional
            如果已有求解结果可直接传入, 避免重复求解.
            
        Returns
        -------
        StressPostResults
            包含所有统计量和字段数据的结果对象.
        """
        analyzer = self._analyzer
        slim = self._stress_limit
        tol = self._constraint_tolerance

        # --- 1. 求解或获取状态 ---
        if state is None:
            state = analyzer.solve_state(rho_val=rho_phys)

        # --- 2. 计算材料刚度插值系数 E(ρ) / E0 ---
        E_rho = analyzer.interpolation_scheme.interpolate_material(
            material=analyzer.material,
            rho_val=rho_phys,
            integration_order=analyzer.integration_order,
        )
        E0 = analyzer.material.youngs_modulus
        E = E_rho / E0  # (NC,) 相对刚度

        # --- 3. 计算 von Mises 应力 ---
        if 'von_mises' not in state:
            if 'stress_solid' not in state:
                solid_stress_dict = analyzer.compute_stress_state(state)
                state.update(solid_stress_dict)
            state['von_mises'] = analyzer.material.calculate_von_mises_stress(
                state['stress_solid']
            )
        
        vm = state['von_mises']  # (NC,) 或 (NC, NQ)

        # 如果有多个积分点, 取最大值
        if vm.ndim > 1:
            vm_scalar = bm.max(vm, axis=-1)  # (NC,)
        else:
            vm_scalar = vm

        # --- 4. 归一化应力测度 SM = E · σ^v / σ_lim ---
        SM = E * vm_scalar / slim  # (NC,)

        # --- 5. 物理密度 V ---
        if hasattr(rho_phys, '__len__'):
            V = bm.array(rho_phys).reshape(-1)
        else:
            V = rho_phys
        
        # 如果 rho_phys 是插值后的密度对象, 提取单元密度
        if hasattr(rho_phys, 'cell_values'):
            V = rho_phys.cell_values()

        NC = len(SM)

        # --- 6. 全局统计 ---
        max_SM = float(bm.max(SM))
        mean_SM = float(bm.mean(SM))
        violated_mask = SM > 1.0 + tol
        num_violated = int(bm.sum(violated_mask))
        max_violation_pct = max((max_SM - 1.0) * 100, 0.0)

        # --- 7. 实体单元统计 ---
        solid_mask = V > self._solid_threshold
        num_solid = int(bm.sum(solid_mask))

        if num_solid > 0:
            SM_solid = SM[solid_mask]
            max_SM_solid = float(bm.max(SM_solid))
            mean_SM_solid = float(bm.mean(SM_solid))
            num_violated_solid = int(bm.sum(SM_solid > 1.0 + tol))
        else:
            max_SM_solid = 0.0
            mean_SM_solid = 0.0
            num_violated_solid = 0

        # --- 8. 体积分数 ---
        cell_measure = analyzer._mesh.entity_measure('cell')
        volume_fraction = float(bm.sum(cell_measure * V) / bm.sum(cell_measure))

        # --- 9. 计算主应力 (用于屈服面绘制) ---
        sig_1_norm, sig_2_norm = self._compute_principal_stresses(
            state=state, E=E, slim=slim
        )

        # --- 10. 组装结果 ---
        results = StressPostResults(
            max_SM=max_SM,
            mean_SM=mean_SM,
            num_violated=num_violated,
            num_total=NC,
            max_violation_pct=max_violation_pct,
            num_solid=num_solid,
            max_SM_solid=max_SM_solid,
            mean_SM_solid=mean_SM_solid,
            num_violated_solid=num_violated_solid,
            volume_fraction=volume_fraction,
            SM=SM,
            V=V,
            E=E,
            sig_1_norm=sig_1_norm,
            sig_2_norm=sig_2_norm,
            solid_mask=solid_mask,
        )

        return results

    def _compute_principal_stresses(self, 
                                     state: Dict,
                                     E: TensorLike,
                                     slim: float,
                                    ) -> tuple:
        """计算归一化主应力 σ_1/σ_lim 和 σ_2/σ_lim.
        
        对应 MATLAB:
            Cauchy_S = Cauchy_S .* E';
            sig_1 = center + radius;
            sig_2 = center - radius;
        """
        stress = state['stress_solid']  # (NC, NQ, NS) 或 (NC, NS)

        # 处理多积分点: 取第一个积分点
        if stress.ndim == 3:
            stress = stress[:, 0, :]  # (NC, NS)

        # 应力乘以材料插值 (得到"真实"应力)
        # 对应 MATLAB: Cauchy_S = Cauchy_S .* repmat(E', 3, 1);
        cauchy = stress * E[:, None]  # (NC, NS)

        # 2D 平面应力/应变: [σ11, σ22, σ12]
        sig_11 = cauchy[:, 0]
        sig_22 = cauchy[:, 1]
        tau_12 = cauchy[:, 2]

        # 主应力
        center = (sig_11 + sig_22) / 2.0
        radius = bm.sqrt(((sig_11 - sig_22) / 2.0)**2 + tau_12**2)
        sig_1 = center + radius
        sig_2 = center - radius

        # 归一化
        sig_1_norm = sig_1 / slim
        sig_2_norm = sig_2 / slim

        return sig_1_norm, sig_2_norm

    def print_summary(self, results: StressPostResults) -> None:
        """打印后处理统计摘要.
        
        对应 MATLAB 中的 fprintf 输出.
        """
        print("\n========== 应力约束检查结果 ==========")
        print(f"应力限制 (σ_lim): {self._stress_limit:.1f}")
        print(f"最大归一化应力 (max(SM)): {results.max_SM:.4f}")
        print(f"平均归一化应力: {results.mean_SM:.4f}")
        print(f"体积分数: {results.volume_fraction:.4f}")
        print(f"违反应力约束的单元数量: "
              f"{results.num_violated} / {results.num_total}")

        if results.num_violated > 0:
            print(f"⚠  警告: 存在应力约束违反!")
            print(f"最大违反程度: {results.max_violation_pct:.2f}%")
        else:
            print(f"✓ 所有应力约束均满足!")

        print(f"\n========== 实体单元 (ρ>{self._solid_threshold}) 应力统计 ==========")
        print(f"实体单元数量: {results.num_solid}")
        if results.num_solid > 0:
            print(f"实体单元最大归一化应力: {results.max_SM_solid:.4f}")
            print(f"实体单元平均归一化应力: {results.mean_SM_solid:.4f}")
            print(f"违反约束的实体单元: {results.num_violated_solid}")

    def plot_density_and_stress(self, 
                                 results: StressPostResults,
                                 mesh=None,
                                 save_path: Optional[str] = None,
                                ) -> None:
        """绘制单元密度分布和归一化 von Mises 应力分布.
        
        对应 MATLAB 中 InitialPlot 的 subplot(1,2,1) 和 subplot(1,2,2).
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection

        if mesh is None:
            mesh = self._analyzer._mesh

        node = bm.to_numpy(mesh.entity('node'))
        cell = bm.to_numpy(mesh.entity('cell'))

        V = bm.to_numpy(results.V)
        SM = bm.to_numpy(results.SM)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # --- 左图: 单元密度 ---
        verts1 = [node[c] for c in cell]
        pc1 = PolyCollection(verts1, array=1.0 - V, 
                              cmap='gray', edgecolors='none')
        pc1.set_clim(0, 1)
        ax1.add_collection(pc1)
        ax1.autoscale_view()
        ax1.set_aspect('equal')
        ax1.set_title('Element Densities')
        ax1.axis('off')
        plt.colorbar(pc1, ax=ax1)

        # --- 右图: 归一化 von Mises 应力 ---
        verts2 = [node[c] for c in cell]
        pc2 = PolyCollection(verts2, array=SM, 
                              cmap='jet', edgecolors='none')
        pc2.set_clim(0, max(1.2, float(bm.max(results.SM))))
        ax2.add_collection(pc2)
        ax2.autoscale_view()
        ax2.set_aspect('equal')
        ax2.set_title('Normalized von Mises Stress')
        ax2.axis('off')
        plt.colorbar(pc2, ax=ax2)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_yield_surface(self, 
                            results: StressPostResults,
                            save_path: Optional[str] = None,
                           ) -> None:
        """绘制 von Mises 屈服面图.
        
        对应 MATLAB PlotYieldSurface 函数:
        将所有实体单元的主应力点绘制在归一化的 σ1-σ2 平面上,
        与 von Mises 屈服椭圆对比.
        """
        import matplotlib.pyplot as plt

        if results.sig_1_norm is None or results.num_solid == 0:
            print("无法绘制屈服面: 无实体单元或主应力未计算.")
            return

        solid = bm.to_numpy(results.solid_mask)
        s1 = bm.to_numpy(results.sig_1_norm)[solid]
        s2 = bm.to_numpy(results.sig_2_norm)[solid]
        SM_solid = bm.to_numpy(results.SM)[solid]

        # --- von Mises 屈服椭圆 ---
        # σ1² - σ1·σ2 + σ2² = σ_lim²
        # 参数化: σ1 = cos(θ) + 0.5·sin(θ), σ2 = cos(θ) - 0.5·sin(θ) (旋转)
        theta = np.linspace(0, 2 * np.pi, 300)
        # 精确参数化 von Mises 椭圆
        r1 = np.cos(theta)
        r2 = np.sin(theta)
        # 变换到 σ1-σ2 坐标
        s1_ellipse = r1 * np.sqrt(2/3) + r2 * np.sqrt(2) / np.sqrt(3) / np.sqrt(2)
        s2_ellipse = r1 * np.sqrt(2/3) - r2 * np.sqrt(2) / np.sqrt(3) / np.sqrt(2)
        
        # 更简洁的参数化方法
        t = np.linspace(0, 2 * np.pi, 500)
        s1_ellipse = (2.0 / np.sqrt(3.0)) * np.cos(t + np.pi / 6)
        s2_ellipse = (2.0 / np.sqrt(3.0)) * np.cos(t - np.pi / 6)

        fig, ax = plt.subplots(figsize=(8, 7))

        # 屈服椭圆
        ax.plot(s1_ellipse, s2_ellipse, 'r-', linewidth=2.5, 
                label='von Mises yield surface')

        # 应力点
        sc = ax.scatter(s1, s2, c=SM_solid, s=20, cmap='jet', 
                        alpha=0.6, edgecolors='none',
                        label='Stress evaluation points')

        # 格式
        vm_actual = np.sqrt(s1**2 - s1 * s2 + s2**2)
        max_ratio = float(np.max(vm_actual)) if len(vm_actual) > 0 else 0.0

        ax.set_xlabel(r'$\sigma_1 / \sigma_{\mathrm{lim}}$', fontsize=14)
        ax.set_ylabel(r'$\sigma_2 / \sigma_{\mathrm{lim}}$', fontsize=14)
        ax.set_title(f'Yield Surface Check (max stress ratio: {max_ratio:.3f})',
                     fontsize=13)
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Normalized stress measure', fontsize=11)
        cbar.mappable.set_clim(0, max(1.2, max_ratio))

        # 统计信息文本框
        text_str = (f"Solid elements: {results.num_solid}\n"
                    f"Max SM: {results.max_SM_solid:.3f}\n"
                    f"Mean SM: {results.mean_SM_solid:.3f}")
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', 
                          edgecolor='black', alpha=0.8))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print("\n✓ 屈服面图已生成")

    def run(self, 
            rho_phys: TensorLike,
            state: Optional[Dict] = None,
            save_prefix: Optional[str] = None,
           ) -> StressPostResults:
        """运行完整后处理流程: 统计 + 打印 + 绘图.
        
        Parameters
        ----------
        rho_phys : TensorLike
            优化后的物理密度.
        state : dict, optional
            FEA 求解结果, 如不传入则自动求解.
        save_prefix : str, optional
            如提供, 将图片保存为 {prefix}_density_stress.png 
            和 {prefix}_yield_surface.png.
        """
        results = self.check_stress_constraints(rho_phys, state)
        self.print_summary(results)

        density_path = f"{save_prefix}_density_stress.png" if save_prefix else None
        yield_path = f"{save_prefix}_yield_surface.png" if save_prefix else None

        self.plot_density_and_stress(results, save_path=density_path)
        self.plot_yield_surface(results, save_path=yield_path)

        return results