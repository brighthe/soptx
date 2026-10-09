"""精确子结构分析器: 以 ``LagrangeFEMAnalyzer`` 协议封装静力缩聚方法.

细网格与位移空间由 ``StructuredSubstructureLayout`` 给出, 与基类自建的空间同编号;
刚度算子是接口 Schur 系统而不是细网格矩阵, 求解在接口未知量上进行, 再逐批恢复
全场位移. 材料插值、载荷、Dirichlet 数据与单元能量导数全部复用基类, 插值公式只在
``MaterialInterpolationScheme`` 中定义一次, 并注入参考子结构的局部装配.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Union

from soptx.backend import backend_manager as bm
from soptx.backend.base import TensorLike
from soptx.decorator import variantmethod
from soptx.fem.analyzers.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.fem.kernels import ElementRestriction
from soptx.fem.levels import SharedReferenceElementAssembly
from soptx.fem.operators import ConstrainedOperator
from soptx.functionspace import Function
from soptx.fem.substructure import (
    GlobalAssembler,
    InterfaceSpace,
    InterfaceSystem,
    StructuredSubstructureLayout,
    SubstructurePrototype,
    build_interface_space,
    build_substructures,
    solve_constrained_system,
)
from soptx.fem.substructure.implicit import RestrictedPreconditioner
from soptx.fem.substructure.recovery import recover_full_displacement_batches
from soptx.fem.substructure.streaming import (
    assemble_exact_interface_system,
    iter_exact_internal_displacement_batches,
)

TraceName = Literal['full_trace', 'linear_corner']
TRACES: tuple[str, ...] = ('full_trace', 'linear_corner')


class SubstructureAnalyzer(LagrangeFEMAnalyzer):
    """结构化布局上的精确子结构静力分析器.

    与 ``LagrangeFEMAnalyzer`` 同协议: ``solve_state`` 依次调用 ``assemble_stiff_matrix``,
    ``assemble_body_force_vector``, ``apply_bc`` 与 ``solve_system``; ``ComplianceObjective``,
    ``VolumeConstraint`` 与 ``Filter`` 不加改动即可使用. 三步的含义改为:

    - ``assemble_stiff_matrix``: 逐批局部装配、精确 Schur 缩聚、迹降阶并散加成全局接口
      刚度 ``K_Q``, 返回 ``InterfaceSystem``;
    - ``apply_bc``: 细网格载荷照基类装配并记为 ``force_vector``, 供目标函数计算
      ``F^T u``; 返回的右端是投影到接口未知量 ``Q`` 的载荷, 支承约束留在实例上;
    - ``solve_system``: 在约束接口系统上求 ``Q``, 再逐批恢复内部位移写入全场向量.

    Parameters
    ----------
    全部参数仅限关键字传递.

    layout : StructuredSubstructureLayout
        结构化子结构布局, 提供细网格、位移空间与局部到全局的自由度映射.
    pde : Any
        满足 soptx 弹性问题契约的问题对象, 只支持齐次 Dirichlet 条件.
    material : LinearElasticMaterial
        实体材料, 其杨氏模量与泊松比须与 ``layout`` 一致.
    trace : {'full_trace', 'linear_corner'}
        接口迹空间: ``full_trace`` 保留全部接口自由度, 与细网格有限元代数等价;
        ``linear_corner`` 以角点位移线性插值接口位移.
    chunk_size : int
        局部装配、缩聚与恢复时每批处理的子结构数, 只影响峰值内存.
    space_degree, integration_order : int
        位移插值次数与数值积分阶, 前者须与 ``layout.degree`` 一致.
    solve_method : {'cg', 'mumps', 'scipy'}
        接口系统求解器.
    solver_options : dict, optional
        迭代解法参数, 词汇与基类相同: ``precond`` (``'jacobi'``, ``'mg'``, ``'none'`` 或 None),
        ``maxiter``, ``atol``, ``rtol``. 接口求解器的停机判据是自由接口上的真实残差
        相对接口载荷二范数, 由 ``atol / ||F_Q||_2`` 给出, ``atol`` 缺省时取 ``rtol``.
    topopt_algorithm, interpolation_scheme, enable_logging, logger_name
        同基类.

    Notes
    -----
    局部刚度写成 ``coef(rho) * KE_unit``, ``coef = E(rho) / E_0`` 由 ``interpolation_scheme``
    给出并注入 ``SubstructurePrototype.coefficient_function``; 参考子结构自身的
    ``penal`` 与 ``rho_min`` 路径不参与. 泊松比随密度插值时局部刚度不再是单位刚度的
    倍数, 不支持.
    """

    def __init__(
        self,
        *,
        layout: StructuredSubstructureLayout,
        pde: Any,
        material: Any,
        trace: TraceName = 'full_trace',
        chunk_size: int = 256,
        space_degree: int = 1,
        integration_order: int = 2,
        solve_method: Literal['mumps', 'scipy', 'cg'] = 'cg',
        solver_options: Optional[Dict] = None,
        topopt_algorithm: Literal[None, 'density_based'] = 'density_based',
        interpolation_scheme: Optional[Any] = None,
        enable_logging: bool = False,
        logger_name: Optional[str] = None,
    ) -> None:
        if trace not in TRACES:
            raise ValueError(f"trace 须为 {TRACES} 之一, 得到 {trace!r}")
        if int(chunk_size) < 1:
            raise ValueError(f"chunk_size 须为正整数, 得到 {chunk_size!r}")
        if int(layout.degree) != int(space_degree):
            raise ValueError(f"space_degree={space_degree} 与 layout.degree={layout.degree} 不一致")
        if float(layout.E_base) != float(material.youngs_modulus) or float(layout.nu) != float(material.poisson_ratio):
            raise ValueError(
                f"layout 的材料 (E={layout.E_base}, nu={layout.nu}) 须与 material "
                f"(E={material.youngs_modulus}, nu={material.poisson_ratio}) 一致"
            )
        if topopt_algorithm not in (None, 'density_based'):
            raise ValueError(f"子结构分析器只支持 topopt_algorithm=None 或 'density_based', 得到 {topopt_algorithm!r}")
        # 布局的细网格与张量空间交给基类, 编号与 layout 的局部到全局映射一致;
        # 接口系统是显式 CSR, 对应基类的 'fa' 层级; 结构化网格只需一份参考单元刚度
        super().__init__(
            disp_mesh=layout.full_mesh, pde=pde, material=material,
            space_degree=space_degree, integration_order=integration_order,
            assembly_method='fast', operator_level='fa', solve_method=solve_method,
            solver_options=solver_options, tensor_space=layout.space_full,
            topopt_algorithm=topopt_algorithm, interpolation_scheme=interpolation_scheme,
            enable_logging=enable_logging, logger_name=logger_name, reference_classes=1,
        )
        self._layout = layout
        self._trace = trace
        self._chunk_size = int(chunk_size)
        self._assembler = GlobalAssembler(layout)
        prototype, sub_meshes, _ = build_substructures(self._assembler, integration_order=integration_order)
        prototype.coefficient_function = self._relative_stiffness
        self._prototype: SubstructurePrototype = prototype
        self._sub_meshes: tuple = tuple(sub_meshes)
        self._interface_space: InterfaceSpace = build_interface_space(
            kind=trace, assembler=self._assembler, sub_meshes=self._sub_meshes, prototype=prototype,
        )
        # 载荷与支承先投影到完整接口, 再经全局迹映射转到 Q; 只依赖问题, 构造时算一次
        self._interface_load, self._constraints = self._interface_space.constrained_conditions(pde)
        self._density_cells: Optional[TensorLike] = None
        self._interface_displacement: Optional[TensorLike] = None
        # 接口 MG 预条件子所需的细网格单元限制算子, 只依赖拓扑, 首次构造后缓存
        self._fine_restriction: Optional[ElementRestriction] = None

    ##############################################################################################
    # 属性
    ##############################################################################################

    @property
    def layout(self) -> StructuredSubstructureLayout:
        """结构化子结构布局"""
        return self._layout

    @property
    def trace(self) -> str:
        """接口迹空间名"""
        return self._trace

    @property
    def chunk_size(self) -> int:
        """每批处理的子结构数"""
        return self._chunk_size

    @property
    def prototype(self) -> SubstructurePrototype:
        """共享的参考子结构"""
        return self._prototype

    @property
    def sub_meshes(self) -> Sequence[Any]:
        """全部子结构, 顺序与布局一致"""
        return self._sub_meshes

    @property
    def interface_space(self) -> InterfaceSpace:
        """接口空间"""
        return self._interface_space

    @property
    def interface_displacement(self) -> Optional[TensorLike]:
        """最近一次求解得到的接口未知量 Q, 可作下一次 CG 的初值; 求解前为 None"""
        return self._interface_displacement

    ##############################################################################################
    # 材料插值
    ##############################################################################################

    def _relative_stiffness(self, rho: TensorLike) -> TensorLike:
        """相对刚度系数 E(rho) / E_0, 由插值方案给出; 标准有限元分析下恒为 1.

        Parameters
        ----------
        rho : (..., NC_sub) 的子结构单元密度.

        Returns
        -------
        coef : 与 ``rho`` 同形状的相对刚度系数.
        """
        scheme = self._interpolation_scheme
        if self._topopt_algorithm is None or scheme is None:
            return bm.ones_like(rho)
        material_params = scheme.interpolate_material(
            material=self._material, rho_val=rho,
            integration_order=self._integration_order, displacement_mesh=self._mesh,
        )
        if isinstance(material_params, tuple):
            self._log_error("子结构分析器要求局部刚度为单位刚度的倍数, 不支持泊松比随密度插值")
        # 构造时已校验 layout.E_base 与 material.youngs_modulus 相等
        return material_params / self._layout.E_base

    def _to_density_cells(self, rho_val: Optional[Union[Function, TensorLike]]) -> TensorLike:
        """把细网格单元密度 (NC, ) 切分为按子结构与参考单元编号排列的 (M, NC_sub).

        标准有限元分析下密度恒为 1; 拓扑优化下 ``rho_val`` 缺失已由 ``_update_density_coefficient`` 报错.
        """
        n_cells = int(self._mesh.number_of_cells())
        if self._topopt_algorithm is None or rho_val is None:
            rho = bm.ones((n_cells, ), dtype=bm.float64)
        else:
            rho = bm.reshape(bm.asarray(rho_val[:], dtype=bm.float64), (-1, ))
            if int(rho.shape[0]) != n_cells:
                self._log_error(f"rho_val 须含 {n_cells} 个单元密度, 得到形状 {tuple(rho.shape)}")
        grid = self._layout.split_global_cell_field(rho)
        return self._prototype.grid_to_cell_field(grid)

    ##############################################################################################
    # 接口系统的多重网格预条件子
    ##############################################################################################

    def _interface_multigrid_preconditioner(self, **kwargs) -> RestrictedPreconditioner:
        """细网格几何多重网格限制到接口自由度的预条件子 ``R B R^T``.

        Parameters
        ----------
        **kwargs : 求解调用的选项, 透传给基类 ``_multigrid_preconditioner`` (``mg_*`` 项).

        Returns
        -------
        RestrictedPreconditioner
            作用于完整接口向量 ``Q`` 的预条件子: 把接口残差零延拓到全场, 做一次细网格 MG
            V 循环, 再提取接口分量. 对自由细网格刚度有 ``R K^{-1} R^T = S^{-1}``, 以 MG
            近似 ``K^{-1}`` 即得接口 Schur 系统的预条件子.

        Notes
        -----
        只支持 ``full_trace``: 其接口未知量就是细网格的接口自由度, ``R`` 为提取矩阵.
        最细层算子取 ``Pi_I K Pi_I + Pi_D`` 形式的细网格 EA 算子 (共享参考单元矩阵乘以
        当前相对刚度系数), 与基类 'ea' 层级下的最细层相同; 受约束接口分量落在 Dirichlet
        自由度上, MG 在其上为恒等, 与 CG 对固定分量的处理一致. 层级拓扑由基类构造, 粗层
        算子由基类按本方法构造的最细层算子所带系数更新, 粗细两层出自同一组系数.
        """
        if self._trace != 'full_trace':
            self._log_error("precond='mg' 目前只支持 trace='full_trace': 角点接口的未知量不是细网格自由度")
        if self._fine_restriction is None:
            self._fine_restriction = ElementRestriction.from_integrator(
                self._integrator, self._tensor_space, layout='flat')
        fine_level = SharedReferenceElementAssembly(
            self._tensor_space, restriction=self._fine_restriction,
            reference_matrices=self._reference_stiffness_matrices(), scale=self._integrator.coef)
        mg = self._multigrid_preconditioner(fine_level, **kwargs)
        _, is_dirichlet, _ = self._dirichlet_data()
        mg.setup(ConstrainedOperator(fine_level, gd=self._pde.dirichlet_bc, isDDof=is_dirichlet))
        return RestrictedPreconditioner(
            mg, self._interface_space.global_dofs, int(self._tensor_space.number_of_global_dofs()))

    ##############################################################################################
    # 分析四步: 缩聚装配, 载荷, 边界条件, 求解与恢复
    ##############################################################################################

    def assemble_stiff_matrix(self,
                            rho_val: Optional[Union[Function, TensorLike]] = None,
                        ) -> Any:
        """逐批局部装配、精确 Schur 缩聚、迹降阶并散加成全局接口刚度.

        Parameters
        ----------
        rho_val : (NC, ) 的细网格单元物理密度; 标准有限元分析下忽略.

        Returns
        -------
        InterfaceSystem
            接口刚度 ``K_Q`` 的 CSR 及其自由度编号. 类型标注放宽为 Any 以兼容基类协议,
            基类返回的是细网格刚度算子.
        """
        self._update_density_coefficient(rho_val)
        self._density_cells = self._to_density_cells(rho_val)
        return assemble_exact_interface_system(
            self._prototype, self._density_cells, self._interface_space, chunk_size=self._chunk_size,
        )

    # 基类的 apply_bc 是按 operator_level 分派的变体方法, 元类按名合并变体; 本类以同键 'fa'
    # 注册覆盖基类的 'fa' 变体, 基类构造函数里的 apply_bc.set('fa') 便分派到这里
    @variantmethod('fa')
    def apply_bc(self,
                K: Any,
                F: TensorLike,
                adjoint: bool = False
            ) -> tuple[Any, TensorLike]:
        """装配细网格载荷并记为 ``force_vector``, 返回投影到接口未知量的载荷.

        Parameters
        ----------
        K : InterfaceSystem
            ``assemble_stiff_matrix`` 返回的接口系统, 原样返回.
        F : 细网格体力向量, 本方法把非体力载荷加到它上面.
        adjoint : 不支持, 为 True 时报错.

        Returns
        -------
        K : InterfaceSystem
            输入的接口系统.
        F_Q : TensorLike
            接口载荷, 形状 ``(N_q,)``.

        Notes
        -----
        子结构缩聚不缩聚载荷, 载荷与支承须落在接口自由度上, 由接口空间的投影检查.
        Dirichlet 数据只支持零位移; 基准向量照基类记录, 作为位移的 Dirichlet 分量参照.
        """
        if adjoint:
            self._log_error("子结构分析器不支持伴随双列右端项")
        F_non_body = self._non_body_loads_by_boundary_type(adjoint=False)
        if F_non_body is not None:
            F = F + F_non_body
        self._F = self.reduce_load(F)
        uh_bd, _, bd_nonzero = self._dirichlet_data()
        if bd_nonzero:
            self._log_error("子结构分析器只支持齐次 Dirichlet 条件")
        self._prescribed_solution = bm.copy(uh_bd[:])
        return K, self._interface_load

    def solve_system(self, K: Any, F: TensorLike, out: Any, **kwargs) -> tuple[Any, Dict]:
        """在约束接口系统上求解 ``Q``, 再逐批恢复全场位移写入 ``out``.

        Parameters
        ----------
        K : InterfaceSystem
            接口刚度.
        F : (N_q, ) 的接口载荷.
        out : 全场位移向量 (有限元函数或张量), 就地写入.
        **kwargs
            ``x0``: CG 的初值, 为接口未知量 ``Q`` 而非全场位移, 通常取上一次的
            ``interface_displacement``; ``solver``: 覆盖构造时的求解器名; ``mg_*``: 多重网格选项.

        Returns
        -------
        out : 全场位移.
        info : 求解诊断, 含 'name', 'niter', 'relres', 'converged', 'constraint_relres' 与 'mode';
            'relres' 为自由接口上的真实残差相对接口载荷二范数, 直接法下 'niter' 为 0.
        """
        if self._density_cells is None:
            self._log_error("solve_system 之前须先调用 assemble_stiff_matrix")
        solver_type = kwargs.get('solver', self._solve_method)
        options = dict(self._solver_options)
        cg_kwargs: Dict[str, Any] = {}
        if solver_type == 'cg':
            load_norm = float(bm.linalg.norm(F))
            atol = options.get('atol', None)
            rtol = options.get('rtol', None)
            cg_tol: Optional[float] = None
            if atol is not None and atol > 0 and load_norm > 0:
                cg_tol = float(atol) / load_norm
            elif rtol is not None and rtol > 0:
                cg_tol = float(rtol)
            if cg_tol is None:
                raise ValueError("solver_options 须给出正的 atol 或 rtol 作为 CG 停机容差")
            precond = options.get('precond', 'jacobi')
            if precond == 'mg':
                precond = self._interface_multigrid_preconditioner(**kwargs)
            cg_kwargs = dict(cg_tol=cg_tol, cg_maxiter=int(options.get('maxiter', 20000)),
                             precond='none' if precond is None else precond, x0=kwargs.get('x0', None))
        result = solve_constrained_system(
            system=K, load=F, constraints=self._constraints, solver=solver_type, **cg_kwargs,
        )
        self._interface_displacement = result.displacement
        # 各子结构的迹位移 q^j = A_q^j Q, 逐批 u_b = Psi q, u_i = T u_b 后写回全场
        q_local = bm.asarray(result.displacement)[self._interface_space.local_dofs]
        batches = iter_exact_internal_displacement_batches(
            self._prototype, self._density_cells, q_local, self._interface_space.trace_basis,
            chunk_size=self._chunk_size,
        )
        out[:] = recover_full_displacement_batches(self._layout, self._sub_meshes, batches)
        info = {'name': solver_type,
                'niter': 0 if result.iterations is None else int(result.iterations),
                'relres': float(result.equilibrium_relative_residual),
                'converged': bool(result.converged),
                'constraint_relres': float(result.constraint_relative_residual),
                'mode': result.mode}
        return out, info

    def solve_state(self,
                    rho_val: Optional[Union[TensorLike, Function]] = None,
                    adjoint: bool = False,
                    enable_timing: bool = False,
                    **kwargs
                ) -> Dict[str, Any]:
        """缩聚装配、载荷、边界条件与求解恢复, 比基类多返回接口未知量.

        Returns
        -------
        dict
            基类的 ``'displacement'`` 与 ``'solver'`` 之外另含 ``'interface_displacement'``,
            即接口未知量 ``Q``, 可作下一次 ``solve_state(x0=...)`` 的热启动初值.
        """
        if adjoint:
            self._log_error("子结构分析器不支持伴随求解")
        state = super().solve_state(rho_val=rho_val, enable_timing=enable_timing, **kwargs)
        return {**state, 'interface_displacement': self._interface_displacement}


__all__ = ['SubstructureAnalyzer', 'TRACES']
