from typing import Optional, Literal, Union, Dict

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
from soptx.utils.base_logged import BaseLogged
from soptx.utils import timer

class ComplianceObjective(BaseLogged):
    def __init__(self,
                analyzer: Union[LagrangeFEMAnalyzer, HuZhangMFEMAnalyzer],
                state_variable: Literal['u', 'sigma'] = 'u',
                diff_mode: Literal["auto", "manual"] = "manual",
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._analyzer = analyzer
        self._state_variable = state_variable
        self._diff_mode = diff_mode
        self._interpolation_scheme = analyzer._interpolation_scheme

        self._material = self._analyzer.material

        self._cached_grad_func = None

    @staticmethod
    def _element_compliance_kernel(rho, ue, ke0, material, interpolation_scheme):
        """[静态纯函数] 单元级柔顺度核函数"""
        E_rho = interpolation_scheme.interpolate_material(
                                material=material, 
                                rho_val=rho
                            )
        strain_energy = bm.einsum('i, ij, j ->', ue, ke0, ue)
        ce = -E_rho * strain_energy

        return ce
    
    def _build_grad_operator(self, ke0: TensorLike) -> TensorLike:
        """构建并编译梯度算子"""
        import functools

        if ke0.ndim == 3:
            # Case A: ke0 是 (NC, ldof, ldof), 每个单元有独立的刚度矩阵
            # 我们需要沿 axis 0 进行批处理映射
            k_axis = 0
        elif ke0.ndim == 2:
            # Case B: ke0 是 (ldof, ldof), 所有单元共用一个基础刚度矩阵
            # 我们不需要映射 (Broadcast)
            k_axis = None
        else:
            self._log_error(f"ke0 维度异常: {ke0.shape}, 期望为 2 维或 3 维")

        kernel_partial = functools.partial(
                                self._element_compliance_kernel, 
                                material=self._material,
                                interpolation_scheme=self._interpolation_scheme
                            )

        grad_func = bm.vmap(bm.grad(kernel_partial, argnums=0), in_axes=(0, 0, k_axis))
        
        return grad_func

    def fun(self, 
            density: Union[Function, TensorLike],
            state: Optional[Dict] = None, 
            **kwargs
        ) -> float:
        """计算柔顺度目标函数值"""
        if isinstance(self._analyzer, LagrangeFEMAnalyzer):
            #* 拉格朗日位移有限元 *#
            uh = state['displacement']
        
            F = self._analyzer.force_vector
            c = bm.einsum('i, i ->', uh[:], F[:])

            print(f"位移场范围: [{bm.min(uh):.4f}, {bm.max(uh):.4f}]")
            print(f"载荷向量和: {bm.sum(F):.2f}")
            print(f"柔顺度目标函数:{c}")

            # K = self._analyzer.stiffness_matrix
            # Ku = K.matmul(uh[:])
            # c = bm.einsum('i, i ->', uh[:], Ku)

        elif isinstance(self._analyzer, HuZhangMFEMAnalyzer):
            #* 胡张应力位移混合有限元 *#
            sigmah, uh = state['stress'], state['displacement']
                
            if self._state_variable == 'u':
                from fealpy.solver import spsolve

                B_sigma_u = self._analyzer.get_mix_matrix()
                B_u_sigma = B_sigma_u.T
                A = self._analyzer.get_stress_matrix(rho_val=density)
                x = spsolve(A, B_sigma_u @ uh[:], solver='mumps')
                Ku = B_u_sigma @ x
                c = bm.einsum('i, i ->', uh[:], Ku)

            elif self._state_variable == 'sigma':
                A = self._analyzer.get_stress_matrix(rho_val=density)
                Asigmah = A.matmul(sigmah[:])
                c = bm.einsum('i, i ->', sigmah[:], Asigmah)

            else:
                error_msg = f"Unknown state_variable: {self._state_variable}"
                self._log_error(error_msg)

        return c
    
    def jac(self, 
            density: Union[Function, TensorLike], 
            state: Optional[dict] = None,
            diff_mode: Optional[Literal["auto", "manual"]] = None,
            **kwargs
        ) -> TensorLike:
        """计算柔顺度目标函数相对于物理密度的灵敏度"""
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
        """手动计算柔顺度目标函数相对于物理密度的灵敏度"""
        density_location = self._interpolation_scheme.density_location

        if self._analyzer.__class__ in [LagrangeFEMAnalyzer]:
            #* 拉格朗日位移有限元 *#
            if state is None:
                state = self._analyzer.solve_state(rho_val=density)
            
            uh = state.get('displacement')
            
            space_uh = self._analyzer._tensor_space
            cell2dof = space_uh.cell_to_dof()
            uhe = uh[cell2dof]

            diff_KE = self._analyzer.compute_stiffness_matrix_derivative(rho_val=density)

            if density_location in ['element']:
                dc = -bm.einsum('ci, cij, cj -> c', uhe, diff_KE, uhe) # (NC, )

                return dc[:]
            
            elif density_location in ['element_multiresolution']:
                dc = -bm.einsum('ci, cnij, cj -> cn', uhe, diff_KE, uhe) # (NC, n_sub)

                return dc[:]
            
            elif density_location in ['node']:
                dc_e = -bm.einsum('ci, clij, cj -> cl', uhe, diff_KE, uhe) # (NC, NCN)

                mesh = space_uh.mesh
                cell2node = mesh.cell_to_node() # (NC, NCN)
                NN = mesh.number_of_nodes()

                dc = bm.zeros((NN, ), dtype=uhe.dtype, device=uhe.device) # (NN, )
                dc = bm.add_at(dc, cell2node.reshape(-1), dc_e.reshape(-1))

                return dc[:]
            
            elif density_location in ['node_multiresolution']:
                dc_e = -bm.einsum('ci, clij, cj -> cl', uhe, diff_KE, uhe) # (NC, NCN)

                density_mesh = density.space.mesh
                cell2node = density_mesh.cell_to_node() # (NC, NCN)
                NN = density_mesh.number_of_nodes()

                dc = bm.zeros((NN, ), dtype=uhe.dtype, device=uhe.device) # (NN, )
                dc = bm.add_at(dc, cell2node.reshape(-1), dc_e.reshape(-1))

                return dc[:]
            
            else:
                error_msg = f"Unknown density location: {density_location}"
                self._log_error(error_msg)

        elif isinstance(self._analyzer, HuZhangMFEMAnalyzer):
            #* 胡张应力位移混合有限元 *#
            t = None
            if enable_timing:
                t = timer(f"灵敏度时间")
                next(t)
            
            if self._state_variable == 'u':
                raise NotImplementedError(
                    "基于位移的灵敏度分析暂未实现。请使用 'sigma' 状态变量以利用胡张混合元的高效互补能列式。"
                )

            elif self._state_variable == 'sigma':
                sigmah = state.get('stress')

                if sigmah is None:
                    self._log_error(f"胡张混合元的灵敏度分析需要应力状态变量，但未提供")

                space_sigmah = self._analyzer.huzhang_space
                cell2dof = space_sigmah.cell_to_dof()
                sigmah_e = sigmah[cell2dof] # (NC, TLDOF_sigma)

                diff_AE = self._analyzer.compute_local_stress_matrix_derivative(rho_val=density) # (NC, TLDOF_sigma, TLDOF_sigma)

                if enable_timing:
                    t.send('矩阵求导时间')

                if density_location in ['element']:
                    dc = bm.einsum('ci, cij, cj -> c', sigmah_e, diff_AE, sigmah_e) # (NC, )

                    if enable_timing:
                        t.send('einsum 时间')
                        t.send(None)
                    
                    return dc[:]
            
                elif density_location in ['node']:
                    raise NotImplementedError("节点密度尚未实现")
                
                elif density_location in ['element_multiresolution']:
                    raise NotImplementedError("多分辨率单元密度尚未实现")
                
                elif density_location in ['node_multiresolution']:
                    raise NotImplementedError("多分辨率节点密度尚未实现")
                
            else:
                error_msg = f"Unknown state_variable: {self._state_variable}"
                self._log_error(error_msg)
            
    
    def _auto_differentiation(self, 
                                density: Union[Function, TensorLike],
                                state: Optional[dict] = None, 
                                enable_timing: bool = False, 
                                **kwargs
                            ) -> TensorLike:
        """使用自动微分技术计算目标函数关于物理密度的梯度"""
        if bm.backend_name not in ['pytorch', 'jax']:
            self._log_error(f"自动微分仅在 pytorch 或者 jax 后端下有效")

        density_location = self._interpolation_scheme.density_location

        if self._analyzer.__class__ in [LagrangeFEMAnalyzer]:
            #* 拉格朗日位移有限元 *#
            if state is None:
                state = self._analyzer.solve_state(rho_val=density)
            
            uh = state.get('displacement')

            if density_location in ['element']:
                cell2dof = self._analyzer.tensor_space.cell_to_dof()
                
                if self._analyzer._cached_ke0 is None:
                    ke0 = self._analyzer.compute_solid_stiffness_matrix()
                else:
                    ke0 = self._analyzer._cached_ke0

                uhe = uh[cell2dof]

                if self._cached_grad_func is None:
                    self._cached_grad_func = self._build_grad_operator(ke0)
                
                dc = self._cached_grad_func(density[:], uhe, ke0)

                # # 定义自动微分核函数
                # def compliance_kernel(rho_i, ue_i, ke0_i):
                #     E_rho = self._interpolation_scheme.interpolate_material(
                #                                     material=self._material, 
                #                                     rho_val=rho_i
                #                                 )
                #     strain_energy = bm.einsum('i, ij, j ->', ue_i, ke0_i, ue_i)
                #     ce = -E_rho * strain_energy
                    
                #     return ce
                
                # if ke0.ndim == 3:
                #     # Case A: ke0 是 (NC, ldof, ldof), 每个单元有独立的刚度矩阵
                #     # 我们需要沿 axis 0 进行批处理映射
                #     k_axis = 0
                # elif ke0.ndim == 2:
                #     # Case B: ke0 是 (ldof, ldof), 所有单元共用一个基础刚度矩阵
                #     # 我们不需要映射 (Broadcast)
                #     k_axis = None
                # else:
                #     self._log_error(f"ke0 维度异常: {ke0.shape}, 期望为 2 维或 3 维")
                
                # grad_func = bm.vmap(bm.grad(compliance_kernel), in_axes=(0, 0, k_axis))

                # dc = grad_func(density[:], uhe, ke0)

                return dc

            else:
                raise NotImplementedError(f"暂时不考虑其它密度分布")

        elif isinstance(self._analyzer, HuZhangMFEMAnalyzer):
            #* 胡张应力位移混合有限元 *#
            raise NotImplementedError("暂时不考虑胡张混合有限元")