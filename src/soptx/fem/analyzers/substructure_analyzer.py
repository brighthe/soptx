"""精确子结构分析器: 以 ``LagrangeFEMAnalyzer`` 协议封装静力缩聚方法.

细网格与位移空间由 ``StructuredSubstructureLayout`` 给出, 与基类自建的空间同编号;
刚度算子是接口 Schur 系统而不是细网格矩阵, 求解在接口未知量上进行, 再逐批恢复
全场位移. 材料插值、载荷、Dirichlet 数据与单元能量导数全部复用基类, 插值公式只在
``MaterialInterpolationScheme`` 中定义一次. FEM 层产生细单元刚度, 独立局部组装
内核仅按编号散加, 缩聚与位移恢复消费矩阵批次.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Literal, Optional, Sequence, Union

import numpy as np

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
)
from soptx.fem.substructure.implicit import RestrictedPreconditioner
from soptx.fem.substructure.recovery import recover_full_displacement_batches
from soptx.fem.substructure.local_assembly import assemble_local_stiffness, build_local_assembly_map
from soptx.fem.substructure.solve import (
    FixedDofConstraints,
    PreparedFixedSystem,
    PreparedDirectSystem,
    fixed_dofs_from_constraints,
    prepare_constrained_system,
    solve_prepared_constrained_system,
)
from soptx.fem.substructure.streaming import (
    assemble_corner_from_stiffness_batches,
    iter_condensation_from_stiffness_batches,
    iter_trace_stiffness_from_local_batches,
    iter_displacement_from_recovery_batches,
)

TraceName = Literal['full_trace', 'linear_corner']
TRACES: tuple[str, ...] = ('full_trace', 'linear_corner')
OperatorLevel = Literal['fa', 'ea']
OPERATOR_LEVELS: tuple[str, ...] = ('fa', 'ea')


class SubstructureAnalyzer(LagrangeFEMAnalyzer):
    """结构化布局上的精确子结构静力分析器.

    与 ``LagrangeFEMAnalyzer`` 同协议: ``solve_state`` 依次调用 ``assemble_stiff_matrix``,
    ``assemble_body_force_vector``, ``apply_bc`` 与 ``solve_system``; ``ComplianceObjective``,
    ``VolumeConstraint`` 与 ``Filter`` 不加改动即可使用. 三步的含义改为:

    - ``assemble_stiff_matrix``: 逐批局部装配、精确 Schur 缩聚、迹降阶并散加成全局接口
      刚度 ``K_Q``, 返回 ``InterfaceSystem``;
    - ``apply_bc``: 细网格载荷照基类装配并记为 ``force_vector``, 供目标函数计算
      ``F^T u``; 准备 CG 掩码算子或直接法的消元/乘子系统与右端;
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
    operator_level : {'fa', 'ea'}
        接口刚度 ``K_Q = sum_j (A_q^j)^T K_r^j A_q^j`` 的存储与作用方式, 两者对应同一个
        离散算子, 只差在以什么形式常驻:

        - 'fa': 散加为全局接口 CSR 矩阵, 支持迭代解法与直接解法;
        - 'ea': 只保留各子结构的缩聚刚度块 ``K_r^j``, matvec 时 gather-批量 GEMV-scatter,
          不形成全局矩阵, 只能用 ``solve_method='cg'``. 接口已确定, 实现待补, 当前构造时报错.

        细网格层级与此无关, 基类内部固定为 'fa', 只用于单元刚度与 GMG 预条件子的细网格层.
    solve_method : {'cg', 'mumps', 'scipy'}
        接口系统求解器.
    solver_options : dict, optional
        迭代解法参数, 词汇与基类相同: ``precond`` (``'jacobi'``, ``'gmg'``, ``'none'`` 或 None),
        ``maxiter``, ``atol``, ``rtol``. 接口求解器的停机判据是自由接口上的真实残差
        按 ``norm_type`` 度量; ``rtol_reference`` 选择初始残差或施加支承后的右端项.
    topopt_algorithm, interpolation_scheme
        同基类.
    corner_condensation : callable, optional
        替换 ``linear_corner`` 的精确角点缩聚, 调用协议同
        ``assemble_corner_from_stiffness_batches``, 另接收 ``modulus_lookup(start, end)``
        取该批子结构的归一化模量; 其 ``report`` 属性经 ``condensation_report`` 暴露.
        只支持 ``trace='linear_corner'``; 见 ``PIMLShapeCornerCondensation``.

    Notes
    -----
    每轮材料系数由基类更新一次. ``element_stiffness`` 复用 FEM 参考矩阵或按批积分,
    局部组装只消费 KE 与拓扑映射. prototype 上的材料回调仅为旧密度调用方保留,
    不参与正式装配或恢复. 当前仍只支持逐单元标量材料系数, 不扩展泊松比插值.
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
        operator_level: OperatorLevel = 'fa',
        solve_method: Literal['mumps', 'scipy', 'cg'] = 'cg',
        solver_options: Optional[Dict] = None,
        topopt_algorithm: Literal[None, 'density_based'] = 'density_based',
        interpolation_scheme: Optional[Any] = None,
        corner_condensation: Optional[Any] = None,
    ) -> None:
        if trace not in TRACES:
            raise ValueError(f"trace 须为 {TRACES} 之一, 得到 {trace!r}")
        if corner_condensation is not None and trace != 'linear_corner':
            raise ValueError(f"corner_condensation 只支持 trace='linear_corner', 得到 {trace!r}")
        if int(chunk_size) < 1:
            raise ValueError(f"chunk_size 须为正整数, 得到 {chunk_size!r}")
        if operator_level not in OPERATOR_LEVELS:
            raise ValueError(f"operator_level 须为 {OPERATOR_LEVELS} 之一, 得到 {operator_level!r}")
        if operator_level == 'ea' and solve_method != 'cg':
            raise ValueError(f"operator_level='ea' 不形成全局矩阵, 只支持 solve_method='cg', "
                             f"得到 {solve_method!r}")
        if operator_level == 'ea':
            raise NotImplementedError("operator_level='ea' 的接口已确定, 实现待补; 当前只支持 'fa'")
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
        # 基类的细网格层级固定为 'fa', 只供单元刚度与 GMG 细网格层使用, 接口层级另存于 _interface_level;
        # 结构化细网格由基类判定为单一平移类, 只存一份参考单元刚度
        super().__init__(
            disp_mesh=layout.full_mesh, pde=pde, material=material,
            space_degree=space_degree, integration_order=integration_order,
            assembly_method='fast', operator_level='fa', solve_method=solve_method,
            solver_options=solver_options, tensor_space=layout.space_full,
            topopt_algorithm=topopt_algorithm, interpolation_scheme=interpolation_scheme,
        )
        self._layout = layout
        self._trace = trace
        self._corner_condensation = corner_condensation
        self._interface_level = operator_level
        self._chunk_size = int(chunk_size)
        self._assembler = GlobalAssembler(layout)
        prototype, sub_meshes, _ = build_substructures(self._assembler, integration_order=integration_order)
        # 兼容历史密度入口; 正式路径不调用此回调或 prototype 的刚度生成方法.
        prototype.coefficient_function = self._relative_stiffness
        self._prototype: SubstructurePrototype = prototype
        # 参考子结构总在主机上构造; 散加编号每轮与细网格设备上的单元刚度相加, 须先移到同一设备
        self._local_assembly_map = build_local_assembly_map(
            bm.device_put(prototype.cell2dof, self._mesh.device), prototype.n_total_dofs)
        self._sub_meshes: tuple = tuple(sub_meshes)
        self._interface_space: InterfaceSpace = build_interface_space(
            kind=trace, assembler=self._assembler, sub_meshes=self._sub_meshes, prototype=prototype,
        )
        # 载荷与支承先投影到完整接口, 再经全局迹映射转到 Q; 只依赖问题, 构造时算一次.
        # 接口载荷与接口编号同设备; 约束矩阵留在主机, CG 用到时再解析为固定自由度
        self._interface_load, self._constraints = self._interface_space.constrained_conditions(pde)
        self._fixed_constraints: Optional[FixedDofConstraints] = None
        self._density_cells: Optional[TensorLike] = None
        self._corner_recovery_batches = None
        self._assembly_ready = False
        self._material_state_ready = False
        self._assembled_coefficient = None
        self._assembled_interface_system = None
        self._prepared_systems = []
        self.assembly_timings = {}
        self.boundary_timings = {}
        self._interface_displacement: Optional[TensorLike] = None
        # 本轮接口求解是否已完成, 回代只接受本轮的接口位移
        self._interface_solved = False
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
    def operator_level(self) -> str:
        """接口刚度的算子层级 ('fa' 或 'ea'), 不是基类内部的细网格层级"""
        return self._interface_level

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
    def condensation_report(self) -> Optional[dict]:
        """注入的角点缩聚器最近一轮的诊断记录; 未注入时为 None."""
        if self._corner_condensation is None:
            return None
        return getattr(self._corner_condensation, 'report', None)

    @property
    def interface_displacement(self) -> Optional[TensorLike]:
        """最近一次求解得到的接口未知量 Q, 可作下一次 CG 的初值; 求解前为 None"""
        return self._interface_displacement

    ##############################################################################################
    # 材料插值
    ##############################################################################################

    def _update_density_coefficient(self, rho_val=None):
        """更新材料状态并失效上轮系统, 正式路径每轮仅调用一次.

        Parameters
        ----------
        rho_val : TensorLike or None
            全局细单元物理密度.

        Returns
        -------
        TensorLike or None
            当前逐单元标量系数, 实体分析为 None.
        """
        self._assembly_ready = False
        self._material_state_ready = False
        self._assembled_coefficient = None
        self._assembled_interface_system = None
        self._prepared_systems = []
        self._corner_recovery_batches = None
        self._density_cells = None
        self._interface_solved = False
        coef = super()._update_density_coefficient(rho_val)
        if coef is not None and tuple(coef.shape) != (int(self._mesh.number_of_cells()),):
            raise ValueError("子结构分析器只支持全局逐单元标量材料系数.")
        self._material_state_ready = True
        return coef

    def _iter_local_stiffness_batches(self):
        """由 FEM 单元矩阵与局部拓扑映射逐批产生子结构刚度.

        Yields
        ------
        tuple
            (start, end, K_local), 每次调用返回新的生成器, 可供完整接口恢复重算.

        Raises
        ------
        RuntimeError
            当前材料状态尚未准备成功.
        """
        if not self._material_state_ready:
            raise RuntimeError("生成局部刚度前须先更新材料状态.")
        mapping = self._local_assembly_map
        for start in range(0, len(self._sub_meshes), self._chunk_size):
            end = min(start + self._chunk_size, len(self._sub_meshes))
            cell_ids = self._layout.substructure_cell_ids(start, end, prototype=self._prototype)
            matrices = self.element_stiffness(bm.reshape(cell_ids, (-1,)))
            matrices = bm.reshape(matrices, (
                end - start, mapping.n_cells, mapping.n_element_dofs, mapping.n_element_dofs,
            ))
            local = assemble_local_stiffness(matrices, mapping)
            del matrices, cell_ids
            yield start, end, local
            del local

    def _assemble_interface_from_local_batches(self, stiffness_batches):
        """将局部矩阵批次缩聚并组装接口系统.

        Parameters
        ----------
        stiffness_batches : iterable
            连续的 (start, end, K_local) 批次.

        Returns
        -------
        tuple
            (InterfaceSystem, recovery), 完整接口不保存恢复矩阵, recovery 为 None.
        """
        prototype = self._prototype
        if self._trace == 'linear_corner':
            if self._corner_condensation is not None:
                return self._corner_condensation(
                    stiffness_batches, self._interface_space,
                    i_dofs=prototype.i_dofs, b_dofs=prototype.b_dofs,
                    modulus_lookup=self._local_modulus_batch,
                    timings=self.assembly_timings,
                )
            return assemble_corner_from_stiffness_batches(
                stiffness_batches, self._interface_space,
                i_dofs=prototype.i_dofs, b_dofs=prototype.b_dofs,
                timings=self.assembly_timings,
            )
        batches = iter_trace_stiffness_from_local_batches(
            stiffness_batches, self._interface_space.trace_basis,
            i_dofs=prototype.i_dofs, b_dofs=prototype.b_dofs,
        )
        return self._interface_space.assemble(batches), None

    def _local_modulus_batch(self, start: int, end: int) -> Any:
        """取一批子结构按参考单元顺序排列的归一化模量 ``E(rho)/E0``.

        Parameters
        ----------
        start, end : int
            子结构编号的半开区间, 与局部刚度批次相同.

        Returns
        -------
        numpy.ndarray
            形状 ``(end - start, n_cells)``, 即积分子当前的相对刚度系数按子结构重排;
            实体分析 (系数为 None) 时全为 1.
        """
        if not self._material_state_ready:
            raise RuntimeError("取局部模量前须先更新材料状态.")
        cell_ids = self._layout.substructure_cell_ids(start, end, prototype=self._prototype)
        coef = self._integrator.coef
        if coef is None:
            return np.ones((end - start, int(self._prototype.n_cells)), dtype=np.float64)
        values = bm.to_numpy(bm.asarray(coef)[bm.reshape(cell_ids, (-1, ))])
        return np.asarray(values, dtype=np.float64).reshape(end - start, -1)

    def _fixed_dof_constraints(self) -> FixedDofConstraints:
        """CG 路径的固定自由度, 首次使用时由约束矩阵解析一次并放到细网格设备.

        Returns
        -------
        FixedDofConstraints
            固定自由度编号、自由分量掩码与约束矩阵, 只依赖问题, 各轮共用.
        """
        if self._fixed_constraints is None:
            self._fixed_constraints = fixed_dofs_from_constraints(
                self._constraints, self._interface_space.n_global, device=self._mesh.device,
            )
        return self._fixed_constraints

    def _require_assembled_state(self):
        """拒绝未成功装配或材料状态已被替换的系统."""
        if (not self._assembly_ready
                or self._integrator.coef is not self._assembled_coefficient):
            raise RuntimeError("须先成功调用 assemble_stiff_matrix, 且随后不得改变材料状态.")

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

    def _interface_gmg_preconditioner(self, **kwargs) -> RestrictedPreconditioner:
        """细网格几何多重网格限制到接口自由度的预条件子 ``R B R^T``.

        Parameters
        ----------
        **kwargs : 求解调用的选项, 透传给基类 ``_gmg_preconditioner`` (``gmg_*`` 项).

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
            self._log_error("precond='gmg' 目前只支持 trace='full_trace': 角点接口的未知量不是细网格自由度")
        if self._fine_restriction is None:
            self._fine_restriction = ElementRestriction.from_integrator(
                self._integrator, self._tensor_space, layout='flat')
        fine_level = SharedReferenceElementAssembly(
            self._tensor_space, restriction=self._fine_restriction,
            reference_matrices=self._reference_stiffness_matrices(), scale=self._integrator.coef)
        mg = self._gmg_preconditioner(fine_level, **kwargs)
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
        self.assembly_timings = {}
        self._update_density_coefficient(rho_val)
        # 仅保留给历史对照调用方, 正式矩阵生成和恢复不读取此密度副本.
        density_cells = self._to_density_cells(rho_val)
        system, recovery = self._assemble_interface_from_local_batches(
            self._iter_local_stiffness_batches(),
        )
        self._density_cells = density_cells
        self._corner_recovery_batches = recovery
        self._assembled_coefficient = self._integrator.coef
        self._assembled_interface_system = system
        self._assembly_ready = True
        return system

    # 基类的 apply_bc 是按 operator_level 分派的变体方法, 元类按名合并变体; 本类以同键 'fa'
    # 注册覆盖基类的 'fa' 变体, 基类构造函数里的 apply_bc.set('fa') 便分派到这里
    @variantmethod('fa')
    def apply_bc(self,
                K: Any,
                F: TensorLike,
                *, solver: Optional[str] = None,
            ) -> tuple[Any, TensorLike]:
        """装配细网格载荷, 并准备所有求解路径的受约束系统.

        Parameters
        ----------
        K : InterfaceSystem
            ``assemble_stiff_matrix`` 返回的原接口系统.
        F : 细网格体力向量, 本方法把非体力载荷加到它上面.
        solver : str, optional
            目标求解器, 缺省使用构造时的配置.

        Returns
        -------
        K : PreparedFixedSystem or PreparedDirectSystem
            CG 掩码系统, 或已消元/组装乘子矩阵的直接法系统.
        F_Q : TensorLike
            受约束系统的右端项, 维数随消元或乘子系统变化.

        Notes
        -----
        子结构缩聚不缩聚载荷, 载荷与支承须落在接口自由度上, 由接口空间的投影检查.
        Dirichlet 数据只支持零位移; 基准向量照基类记录, 作为位移的 Dirichlet 分量参照.
        """
        self._require_assembled_state()
        if K is not self._assembled_interface_system:
            raise ValueError("apply_bc 须使用本轮成功装配的接口系统.")
        self.boundary_timings = {}
        started = perf_counter()
        load_cached = self._non_body_cache is not None
        F_non_body = self._non_body_loads_by_boundary_type()
        if F_non_body is not None:
            F = F + F_non_body
        self._F = F
        self.boundary_timings["non_body_load"] = perf_counter() - started
        self.boundary_timings["load_cache_present_before"] = load_cached
        started = perf_counter()
        dirichlet_cached = self._dirichlet_cache is not None
        uh_bd, _, bd_nonzero = self._dirichlet_data()
        if bd_nonzero:
            self._log_error("子结构分析器只支持齐次 Dirichlet 条件")
        self._prescribed_solution = bm.copy(uh_bd[:])
        self.boundary_timings["dirichlet_data"] = perf_counter() - started
        self.boundary_timings["dirichlet_cache_present_before"] = dirichlet_cached
        started = perf_counter()
        solver = self._solve_method if solver is None else solver
        if solver == 'cg':
            # CG 在接口刚度所在设备上准备: 复用已解析的固定自由度, 对角按缓存的槽位取值
            prepared = prepare_constrained_system(
                K, self._interface_load, self._constraints, solver=solver,
                fixed=self._fixed_dof_constraints(),
                diagonal=self._interface_space.diagonal(K.stiffness),
            )
        else:
            prepared = prepare_constrained_system(
                K, self._interface_load, self._constraints, solver=solver,
            )
        rhs = bm.copy(bm.asarray(prepared.rhs, dtype=bm.float64))
        self.boundary_timings["prepare_system"] = perf_counter() - started
        self._prepared_systems.append(prepared)
        return prepared, rhs

    def _apply_state_bc(self, K, F, **kwargs):
        """将状态求解的 solver 选项传给边界条件准备阶段.

        Parameters
        ----------
        K, F : Any
            原接口刚度与细网格载荷.
        **kwargs
            状态求解选项.

        Returns
        -------
        tuple
            按目标求解器准备的受约束系统与右端项.
        """
        return self.apply_bc(K, F, solver=kwargs.get('solver', self._solve_method))

    def solve_interface(self, K: Any, F: TensorLike, **kwargs) -> Dict:
        """在约束接口系统上求解接口未知量 ``Q``, 不恢复全场位移.

        Parameters
        ----------
        K : PreparedFixedSystem or PreparedDirectSystem
            ``apply_bc`` 返回的接口系统.
        F : TensorLike
            apply_bc 返回的受约束右端项.
        **kwargs
            ``x0``: CG 的初值, 为接口未知量 ``Q`` 而非全场位移, 通常取上一次的
            ``interface_displacement``; ``solver``: 覆盖构造时的求解器名; ``gmg_*``: 几何多重网格选项.

        Returns
        -------
        info : 求解诊断, 含 'name', 'niter', 'relres', 'converged', 'constraint_relres' 与 'mode';
            'relres' 为自由接口上的真实残差相对接口载荷二范数, 直接法下 'niter' 为 0.

        Notes
        -----
        解存为 ``interface_displacement``, 与细网格同设备; 随后由 ``recover_displacement``
        恢复全场位移. 两步分开调用便于分别计量, ``solve_system`` 依次调用二者.
        """
        self._require_assembled_state()
        solver_type = kwargs.get('solver', self._solve_method)
        if not isinstance(K, (PreparedFixedSystem, PreparedDirectSystem)):
            raise TypeError("solve_interface 需要受约束系统, 请先调用 apply_bc.")
        if not any(K is prepared for prepared in self._prepared_systems):
            raise ValueError("solve_interface 须使用本轮 apply_bc 准备的系统.")
        options = self._merged_solver_options(kwargs)
        cg_kwargs: Dict[str, Any] = {}
        if solver_type == 'cg':
            precond = options.get('precond', 'jacobi')
            if precond == 'gmg':
                precond = self._interface_gmg_preconditioner(**kwargs)
            cg_kwargs = dict(atol=options.get('atol', 1e-12),
                             rtol=options.get('rtol', 1e-12),
                             rtol_reference=options.get('rtol_reference', 'initial_residual'),
                             norm_type=options.get('norm_type', 'unpreconditioned'),
                             verify_residual=options.get('verify_residual', True),
                             cg_maxiter=options.get('maxiter', 20000),
                             precond='none' if precond is None else precond, x0=kwargs.get('x0', None))
        result = solve_prepared_constrained_system(K, solver=solver_type, load=F, **cg_kwargs)
        # CG 的解已在细网格设备上; 直接法在主机上求解, 解移回细网格设备, 与设备上的恢复矩阵一同回代
        self._interface_displacement = bm.device_put(result.displacement, self._mesh.device)
        self._interface_solved = True
        info = {'name': solver_type,
                'niter': 0 if result.iterations is None else int(result.iterations),
                'relres': float(result.equilibrium_relative_residual),
                'converged': bool(result.converged),
                'constraint_relres': float(result.constraint_relative_residual),
                'mode': result.mode}
        if result.solver_info is not None:
            for key in ('rtol_reference', 'norm_type', 'rhs_norm', 'reference_norm',
                        'initial_residual_norm', 'stopping_threshold',
                        'recursive_residual', 'true_residual'):
                info[key] = result.solver_info.get(key)
            info['reason'] = result.solver_info['reason'].name
        return info

    def recover_displacement(self, out: Any) -> Any:
        """由本轮的接口未知量逐批恢复内部位移, 拼成全场位移写入 ``out``.

        Parameters
        ----------
        out : 全场位移向量 (有限元函数或张量), 就地写入.

        Returns
        -------
        out : 全场位移.

        Raises
        ------
        RuntimeError
            本轮尚未调用 ``solve_interface``, 或其后材料状态已被替换.

        Notes
        -----
        角点接口用组装时缓存的恢复矩阵回代; 完整接口不缓存恢复矩阵, 逐批重做局部组装与
        精确缩聚后回代.
        """
        self._require_assembled_state()
        if not self._interface_solved:
            raise RuntimeError("须先调用本轮的 solve_interface, 再恢复全场位移.")
        q_local = bm.asarray(self._interface_displacement)[self._interface_space.local_dofs]
        recovery_on_trace = self._corner_recovery_batches is not None
        recovery_batches = self._corner_recovery_batches
        if recovery_batches is None:
            # 完整接口恢复重新消费同一材料状态的矩阵, 不重新进行材料插值.
            recovery_batches = (
                (batch.start, batch.end, batch.recovery)
                for batch in iter_condensation_from_stiffness_batches(
                    self._iter_local_stiffness_batches(),
                    i_dofs=self._prototype.i_dofs, b_dofs=self._prototype.b_dofs,
                )
            )
        batches = iter_displacement_from_recovery_batches(
            recovery_batches, q_local, self._interface_space.trace_basis,
            n_internal=len(self._prototype.i_dofs), recovery_on_trace=recovery_on_trace,
        )
        out[:] = recover_full_displacement_batches(self._layout, self._sub_meshes, batches)
        return out

    def solve_system(self, K: Any, F: TensorLike, out: Any, **kwargs) -> tuple[Any, Dict]:
        """在约束接口系统上求解 ``Q``, 再逐批恢复全场位移写入 ``out``.

        Parameters
        ----------
        K : PreparedFixedSystem or PreparedDirectSystem
            ``apply_bc`` 返回的接口系统.
        F : TensorLike
            apply_bc 返回的受约束右端项.
        out : 全场位移向量 (有限元函数或张量), 就地写入.
        **kwargs
            同 ``solve_interface``.

        Returns
        -------
        out : 全场位移.
        info : 求解诊断, 同 ``solve_interface``.

        Notes
        -----
        依次调用 ``solve_interface`` 与 ``recover_displacement``.
        """
        info = self.solve_interface(K, F, **kwargs)
        return self.recover_displacement(out), info

    def solve_state(self,
                    rho_val: Optional[Union[TensorLike, Function]] = None,
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
        state = super().solve_state(rho_val=rho_val, enable_timing=enable_timing, **kwargs)
        return {**state, 'interface_displacement': self._interface_displacement}


__all__ = ['OPERATOR_LEVELS', 'SubstructureAnalyzer', 'TRACES']
