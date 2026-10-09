"""Lagrange 位移有限元分析器.

在 Lagrange 张量空间上组装线弹性刚度算子与外载荷, 施加 Dirichlet 边界条件并
求解状态与伴随方程. 算子层级 ``'fa'``, ``'ea'``, ``'pa'``, ``'ua'`` 共用同一套
离散; ``reduce_load`` 与 ``wrap_operator`` 是串行下为恒等操作的分布式扩展点.
"""

from typing import Optional, Union, Literal, Dict

from soptx.backend import backend_manager as bm
from soptx.typing import TensorLike
from soptx.mesh import SimplexMesh, HomogeneousMesh
from soptx.functionspace import LagrangeFESpace, TensorFunctionSpace, Function
from soptx.decorator import variantmethod
from soptx.sparse import CSRTensor, COOTensor

from soptx.core import BaseLogged, timer
from soptx.protocols import (
    BodyForce,
    DirichletElasticityProblem,
    MaterialInterpolation,
    PointForce,
)
from soptx.fem.integrators import LinearElasticIntegrator
from soptx.fem.kernels import ElementRestriction
from soptx.fem.levels import (
    AssemblyLevelExtension,
    FullAssembly,
    SharedReferenceElementAssembly,
    available_levels,
    create_level,
)
from soptx.fem.load_assembly import assemble_body_forces, assemble_non_body_loads
from soptx.fem.matrix import SymmetricElimination, assemble_csr, build_csr_pattern
from soptx.fem.utils import multiresolution_sub_element_matrices
from soptx.fem.operators import ConstrainedOperator
from soptx.materials import (
    IsotropicLinearElasticMaterial,
    elastic_matrices,
    lame_basis_matrices,
    lame_parameter_derivatives,
)

class LagrangeFEMAnalyzer(BaseLogged):
    """Lagrange 位移有限元的线弹性分析器.

    消费满足 ``DirichletElasticityProblem`` 的 Problem, 提供刚度组装, 边界条件
    施加, 状态与伴随求解, 应力计算及刚度对密度的导数, 满足 ``AnalysisStage``
    协议. 设置 ``topopt_algorithm`` 时按 ``interpolation_scheme`` 插值材料参数.
    """

    def __init__(self,
                disp_mesh: HomogeneousMesh,
                pde: DirichletElasticityProblem,
                material: IsotropicLinearElasticMaterial,
                space_degree: int = 1,
                integration_order: int = 4,
                assembly_method: Literal['standard', 'voigt', 'fast'] = 'standard',
                operator_level: Literal['fa', 'ea', 'pa', 'ua'] = 'fa',
                preconditioner_level: Optional[Literal['fa', 'ea', 'pa', 'ua']] = None,
                solve_method: Literal['mumps', 'scipy', 'cg'] = 'mumps',
                solver_options: Optional[Dict] = None,
                tensor_space: Optional[TensorFunctionSpace] = None,
                dof_comm: Optional[object] = None,
                topopt_algorithm: Literal[None, 'density_based', 'level_set'] = None,
                interpolation_scheme: Optional[MaterialInterpolation] = None,
                enable_logging: bool = False,
                logger_name: Optional[str] = None,
                reference_classes: Optional[int] = None,
            ) -> None:
        """初始化拉格朗日有限元分析器.

        Parameters
        ----------
        disp_mesh : 位移有限元网格.
        pde : 弹性力学边值问题或制造解对象.
        material : 各向同性线弹性材料; 密度插值与灵敏度要用到其杨氏模量与假设类型.
        space_degree : 有限元基函数多项式阶数.
        integration_order : 数值求积公式的代数精度阶数.
        assembly_method : 单元刚度矩阵缩并算法.
        operator_level : 离散算子的存储与作用方式.

            - 'fa': 装配全局稀疏矩阵 (full assembly), 支持直接解法与伴随求解;
            - 'ea': 只保留单元矩阵 (element assembly), matvec 时 gather-作用-scatter,
              不形成全局矩阵, 只能用迭代解法; 单元密度下只常驻 K_e^0 (与敏度共用, 给出
              ``reference_classes`` 时按平移类各一份) 与逐单元标量 s_e, 不另存 K_e;
            - 'pa': 只保留积分点上的几何与材料数据 (partial assembly), matvec 时
              gather-B-D-B^T-scatter, 高阶下比 'ea' 省内存, 只能用迭代解法;
            - 'ua': 什么都不常驻 (unassembled), 每次 matvec 现算几何与材料, 只用于取证.

            四者对应同一个离散算子 K = sum_e R_e^T K_e R_e, 只差在以什么形式常驻.
        preconditioner_level : 预条件子所用的装配层级.

            - None: 预条件子绑主算子本身, 不另建层级;
            - 其余: 另建一个该层级的算子, 只供预条件子使用.
        solve_method : 求解方式.
        solver_options : 迭代解法的默认参数 (maxiter, atol, rtol, precond); precond='mg' 时另读
            mg_* 选项, 见 ``_multigrid_preconditioner``.
        tensor_space : 外部构造的张量函数空间; 为 None 时内部自动构造.
        dof_comm : 分布式重叠自由度通信器.
        topopt_algorithm : 拓扑优化算法类型.
        interpolation_scheme : 密度插值方案.
        enable_logging : 是否开启日志记录.
        logger_name : 日志记录器名称.
        reference_classes : 单元按 ``create_box_mesh`` / ``from_box`` 的编号约定排列时的平移类数
            N_k (六面体、四边形为 1, 三角形 2, 四面体 6). 给出时单元密度下的 K_e^0 只按类各存
            一份 (N_k, TLDOF, TLDOF), 单元 e 取第 e % N_k 份, 供 FA 缩放装配、'ea' 层级、单元
            能量导数与多重网格共用; 缺省 (None) 时逐单元各存一份. 约定由调用方保证, 首次计算
            时只抽查少数单元, 见 ``_reference_stiffness_matrices``.
        """

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        # 私有属性 (建议通过属性访问器访问, 不要直接修改)
        self._mesh = disp_mesh
        self._pde = pde
        self._material = material

        self._space_degree = space_degree
        self._integration_order = integration_order
        self._assembly_method = assembly_method

        self._topopt_algorithm = topopt_algorithm
        self._interpolation_scheme = interpolation_scheme
        if self._topopt_algorithm in ['density_based', 'level_set'] and self._interpolation_scheme is None:
            self._log_error(
                f"拓扑优化算法 '{self._topopt_algorithm}' 需要显式提供 interpolation_scheme: "
                f"fem (layer 2) 不构造 topology (layer 3) 的插值方案, 请由调用方传入, "
                f"例如 soptx.topology.interpolation.MaterialInterpolationScheme"
            )

        self._solve_method = solve_method
        self._solver_options = dict(solver_options) if solver_options else {}
        self._dof_comm = dof_comm

        if operator_level not in available_levels():
            self._log_error(
                f"不支持的算子层级: {operator_level}, "
                f"可选 {available_levels()}"
            )
        self._operator_level = operator_level

        if preconditioner_level is not None:
            if preconditioner_level not in available_levels():
                self._log_error(
                    f"不支持的预条件子层级: {preconditioner_level}, "
                    f"可选 {available_levels()}"
                )
            # 与 apply_bc('fa') 的多 rank 拒绝同因: 对称消元没有重叠归约的插入点
            if (preconditioner_level == 'fa'
                    and self._dof_comm is not None
                    and self._dof_comm.mpi_size > 1):
                self._log_error(
                    f"preconditioner_level='fa' 只支持单 rank, 当前为 "
                    f"{self._dof_comm.mpi_size} 个 rank: 全局矩阵的对称消元没有重叠"
                    f"归约的插入点。多 rank 请使用 preconditioner_level='ea' 或 'pa'"
                )
        self._preconditioner_level = preconditioner_level

        self._GD = self._mesh.geo_dimension()

        #* (GD, -1): dof_priority (x0, ..., xn, y0, ..., yn)
        #* (-1, GD): gd_priority (x0, y0, ..., xn, yn)
        if tensor_space is None:
            self._scalar_space = LagrangeFESpace(self._mesh, p=self._space_degree, ctype='C')
            self._tensor_space = TensorFunctionSpace(scalar_space=self._scalar_space, shape=(-1, self._GD))
        else:
            if tensor_space.mesh is not self._mesh:
                self._log_error("外部传入的张量空间必须建立在 disp_mesh 上")
            self._tensor_space = tensor_space
            self._scalar_space = tensor_space.scalar_space

        # 注册算子层级变体: 'fa' 直接改写矩阵, 其余层级都只是把算子包起来,
        # 与常驻形式无关, 因此共用同一个变体
        self.apply_bc.set('fa' if self._operator_level == 'fa' else 'matrix_free')

        # 缓存的矩阵和向量
        self._K = None
        self._F = None
        self._prescribed_solution = None  # 满足 Dirichlet 值、内部为零的基准向量
        self._csr_pattern = None  # 模式先行 (Pattern-First) 静态拓扑骨架缓存
        # Dirichlet 数据与对称消元的槽位只依赖空间、问题与 CSR 骨架, 与密度无关, 各算一次
        self._dirichlet_cache = None
        self._elimination = SymmetricElimination()
        # 几何多重网格层次的拓扑部分 (网格序列, 延拓, 粗层骨架), 只依赖空间与约束
        self._mg_hierarchy = None
        # 纯集中力的非体力载荷: (载荷指纹, 载荷向量), 见 _non_body_loads_by_boundary_type
        self._non_body_cache = None
        # 当前的装配层级对象, self._operator_level 是它的名字; 换空间时由 tensor_space 置空
        self._level: Optional[AssemblyLevelExtension] = None

        self._integrator = LinearElasticIntegrator(material=self._material,
                                                q=self._integration_order,
                                                method=self._assembly_method)
        self._integrator.keep_data(True)

        self._cached_ke0 = None
        self._cached_ke0_sub = None
        # 平移类参考单元矩阵 (N_k, TLDOF, TLDOF); 只在给出 reference_classes 时使用
        if reference_classes is not None:
            n_cells = int(self._mesh.number_of_cells())
            if (isinstance(reference_classes, bool) or int(reference_classes) != reference_classes
                    or reference_classes < 1 or n_cells % int(reference_classes) != 0):
                self._log_error(f"reference_classes 须为整除单元数 {n_cells} 的正整数, 得到 {reference_classes!r}")
            reference_classes = int(reference_classes)
        self._reference_classes = reference_classes
        self._reference_ke0 = None

        self._cached_stiffness_absolute = None # 绝对刚度 (带量纲)
        self._cached_stiffness_relative = None # 相对刚度 (无量纲)

        # 泊松比随密度插值 (近不可压缩算例) 时的缓存: 逐单元泊松比, 以及
        # K_e = λ*_e K_e^λ + μ_e K_e^μ 中与设计无关的两个基矩阵
        self._cached_nu_rho = None
        self._cached_ke_lambda = None
        self._cached_ke_mu = None

    ##############################################################################################
    # 属性相关函数
    ##############################################################################################
    
    @property
    def disp_mesh(self) -> HomogeneousMesh:
        """获取当前的位移网格对象"""
        return self._mesh
    
    @property
    def pde(self) -> DirichletElasticityProblem:
        """获取当前的 PDE 对象"""
        return self._pde
    
    @property
    def scalar_space(self) -> LagrangeFESpace:
        """获取当前的标量函数空间"""
        return self._scalar_space
    
    @property
    def tensor_space(self) -> TensorFunctionSpace:
        """获取当前的张量函数空间"""
        return self._tensor_space
    
    @property
    def integration_order(self) -> int:
        """获取当前的数值积分阶次"""
        return self._integration_order
    
    @property
    def material(self) -> IsotropicLinearElasticMaterial:
        """获取当前的材料类"""
        return self._material
    
    @property
    def interpolation_scheme(self) -> MaterialInterpolation:
        """当前的材料插值方案; 标准有限元分析 (topopt_algorithm=None) 下没有, 访问即报错"""
        scheme = self._interpolation_scheme
        if scheme is None:
            self._log_error("当前分析器没有材料插值方案: topopt_algorithm 为 None 时不做材料插值")
        return scheme
    
    @property
    def assembly_method(self) -> str:
        """获取当前的组装方法"""
        return self._assembly_method

    @property
    def operator_level(self) -> str:
        """获取当前的算子层级 ('fa' 或 'ea')"""
        return self._operator_level

    @property
    def preconditioner_level(self) -> Optional[str]:
        """预条件子所用的装配层级; None 表示绑主算子本身"""
        return self._preconditioner_level

    @property
    def topopt_algorithm(self) -> Optional[str]:
        """获取当前的拓扑优化算法"""
        return self._topopt_algorithm

    @property
    def poisson_ratio_interpolated(self) -> bool:
        """最近一次装配中泊松比是否随密度插值

        为 True 时 K_e 不再是实体单元刚度的标量倍, 依赖 K_e = E(ρ)/E0 · K_e^0
        的外部路径 (自动微分、应力约束的隐式项) 不再成立.
        """
        return self._cached_nu_rho is not None
    
    @property
    def stiffness_matrix(self) -> Union[CSRTensor, COOTensor, AssemblyLevelExtension]:
        """最近一次 assemble_stiff_matrix 构造的刚度算子: 'fa' 下为未施加边界条件的全局稀疏矩阵, 其余层级下为装配层级对象"""
        return self._K
    
    @property
    def assembly_level(self) -> Optional[AssemblyLevelExtension]:
        """最近一次 assemble_stiff_matrix 构造出的装配层级对象

        'fa' 下它持有全局稀疏矩阵与 CSR 骨架, 'ea' 下它就是刚度算子本身 (与
        stiffness_matrix 同一个对象). assemble_stiff_matrix 之前为 None.
        """
        return self._level

    @property
    def force_vector(self) -> Union[TensorLike, COOTensor]:
        """获取当前的载荷向量"""
        return self._F

    @property
    def dof_comm(self):
        """获取分布式重叠自由度通信器, 串行下为 None"""
        return self._dof_comm

    @property
    def prescribed_solution(self) -> Optional[TensorLike]:
        """最近一次 apply_bc 得到的 Dirichlet 基准向量

        边界自由度取给定值, 内部自由度为零. 可用作迭代解法的初值, 以及边界误差
        的比较基准. apply_bc 之前为 None.
        """
        return self._prescribed_solution

    @scalar_space.setter
    def scalar_space(self, space: LagrangeFESpace) -> None:
        """设置标量函数空间"""
        self._scalar_space = space

    @tensor_space.setter
    def tensor_space(self, space: TensorFunctionSpace) -> None:
        """设置张量函数空间"""
        self._tensor_space = space
        self._csr_pattern = None
        # 层级与 K_e^0 都依赖空间, 换空间后作废, 下次装配重建
        self._level = None
        self._cached_ke0 = None
        self._reference_ke0 = None
        self._dirichlet_cache = None
        self._elimination = SymmetricElimination()
        self._mg_hierarchy = None
        self._non_body_cache = None

    
    ##############################################################################################
    # 核心方法
    ##############################################################################################

    def _update_density_coefficient(self,
                            rho_val: Optional[Union[Function, TensorLike]] = None,
                        ) -> Optional[TensorLike]:
        """按拓扑优化算法更新积分子的相对刚度系数

        Parameters
        ----------
        rho_val : 密度值
        - 单元密度 - TensorLike
            - 单分辨率 - (NC, )
            - 多分辨率 - (NC, n_sub)
        - 节点密度 - Fucntion
            - 单分辨率 - (NN, )
            - 多分辨率 - (NN, )

        Returns
        -------
        coef : 写入积分子的系数. 标准有限元分析下为 None; 单元密度只插值 E 时为 (NC, ) 的
            相对刚度; 多分辨率、节点密度或泊松比插值时为更高维的数组.
        """
        if self._topopt_algorithm is None:
            if rho_val is not None:
                self._log_warning("标准有限元分析模式下忽略相对密度 rho")

            # 标准有限元分析不做材料插值, 积分子直接使用实体材料本构
            coef = None

            self._cached_stiffness_absolute = None
            self._cached_stiffness_relative = None
            self._cached_nu_rho = None
        
        elif self._topopt_algorithm == 'density_based':
            if rho_val is None:
                self._log_error("基于密度的拓扑优化算法需要提供相对密度 rho")

            material_params = self.interpolation_scheme.interpolate_material(
                                            material=self._material,
                                            rho_val=rho_val,
                                            integration_order=self._integration_order,
                                            displacement_mesh=self._mesh,
                                        )
            # interpolate_material 只在材料近不可压缩且 target_variables 含 'nu' 时
            # 返回 (E_rho, nu_rho) 二元组, 否则只返回 E_rho
            if isinstance(material_params, tuple):
                E_rho, nu_rho = material_params[0], material_params[1]
            else:
                E_rho, nu_rho = material_params, None
            E0 = self._material.youngs_modulus
            relative_stiffness = E_rho / E0

            self._cached_stiffness_absolute = E_rho               # 绝对刚度 (带量纲)
            self._cached_stiffness_relative = relative_stiffness  # 相对刚度 (无量纲)
            self._cached_nu_rho = nu_rho

            if nu_rho is None:
                # 只插值 E: D_e = E(ρ)/E0 · D0, 积分子按标量系数处理
                coef = relative_stiffness
            else:
                # E 与 ν 同时插值: D_e 不再是 D0 的标量倍, 把逐单元本构矩阵交给积分子
                self._check_poisson_interpolation_support()
                coef = elastic_matrices(E_rho, nu_rho, self._material.hypothesis,
                                        device=self._mesh.device)   # (NC, NS, NS)
        
        else:
            error_msg = f"不支持的拓扑优化算法: {self._topopt_algorithm}"
            self._log_error(error_msg)

        # 更新积分子的材料系数, 形状约定见 LinearElasticIntegrator.assembly('standard')
        self._integrator.coef = coef

        return coef

    def assemble_stiff_matrix(self,
                            rho_val: Optional[Union[Function, TensorLike]] = None,
                        ) -> Union[CSRTensor, COOTensor, AssemblyLevelExtension]:
        """按当前算子层级构造刚度算子.

        Parameters
        ----------
        rho_val : 材料物理密度场分布, 用于变密度拓扑优化中的刚度矩阵插值; 为 None 时
            取基准实体刚度. 形状约定见 ``_update_density_coefficient``.

        Returns
        -------
        operator : 'fa' 层级下为全局稀疏矩阵 K, 其余层级下为对应的
            ``AssemblyLevelExtension`` 算子, 其 ``@`` 运算与 'fa' 对应同一个离散算子.
        """
        coef = self._update_density_coefficient(rho_val)
        level = self._level
        scaled = (coef is not None and coef.ndim == 1
                  and self._operator_level in ('fa', 'ea'))

        if (level is not None and self._operator_level != 'fa'
                and (level.scale is not None) == scaled):
            level.update(coef)
        elif scaled and self._operator_level == 'fa':
            if self._csr_pattern is None:
                self._csr_pattern = build_csr_pattern(self._tensor_space)
            matrix = assemble_csr(self._reference_stiffness_matrices(), self._csr_pattern, scale=coef)
            level = FullAssembly(self._tensor_space, matrix, pattern=self._csr_pattern, scale=coef)
        elif scaled:
            # 只常驻参考单元矩阵与 s_e, 不另存 K_e; update 只换 s_e
            restriction = ElementRestriction.from_integrator(self._integrator,
                                                            self._tensor_space,
                                                            layout='flat')
            level = SharedReferenceElementAssembly(self._tensor_space,
                                                restriction=restriction,
                                                reference_matrices=self._reference_stiffness_matrices(),
                                                scale=coef)
        else:
            level = create_level(self._operator_level,
                                space=self._tensor_space,
                                integrator=self._integrator,
                                pattern=self._csr_pattern)
            # 'fa' 下层级把首次装配建好的 CSR 骨架交回来供下次复用; 其余层级没有骨架
            pattern = getattr(level, 'pattern', None)
            if pattern is not None:
                self._csr_pattern = pattern

        self._level = level
        self._K = level.operator

        return self._K

    def assemble_spring_stiff_matrix(self):
        """组装弹簧刚度矩阵"""
        tspace = self._tensor_space
        TGDOF = tspace.number_of_global_dofs()

        k_in = self._pde.k_in
        k_out = self._pde.k_out
        threshold_spring = self._pde.is_spring_boundary()
        isBdDof = tspace.is_boundary_dof(threshold=threshold_spring, method='interp')
        spring_dofs = bm.where(isBdDof)[0]
        indices = bm.stack([spring_dofs, spring_dofs], axis=0)
        values = bm.tensor([k_in, k_out], dtype=bm.float64, device=tspace.device)
        spshape = (TGDOF, TGDOF)

        K = COOTensor(indices=indices, values=values, spshape=spshape)

        return K

    def assemble_body_force_vector(self) -> TensorLike:
        """组装 ``Problem.loads()`` 中全部体力的体积分."""
        return assemble_body_forces(self._tensor_space, self._pde.loads(), q=self._integration_order)

    def assemble_external_load(self, adjoint: bool = False) -> TensorLike:
        """组装施加 Dirichlet 条件之前的全局外载向量.

        参数:
            adjoint: 为 ``True`` 时返回结构载荷与伴随载荷堆叠成的两列右端项.

        返回:
            F: 全部物理载荷的等效节点力之和, 形状 ``(n_dof,)``; ``adjoint`` 为
                ``True`` 时形状 ``(n_dof, 2)``.

        说明:
            该向量与 ``apply_bc`` 存入 ``force_vector`` 的是同一个量, 但不需要先
            装配刚度矩阵. 子结构缩聚等只消费外载, 不求解全尺度系统的调用方应使用
            本方法: ``force_vector`` 在 ``apply_bc`` 之前为 ``None``.

            本方法不写入 ``self._F``, 因此不会干扰分析器自身的求解状态.
        """
        F = self.assemble_body_force_vector()
        if adjoint:
            F = bm.stack([F, bm.zeros_like(F)], axis=1)

        F_non_body = self._non_body_loads_by_boundary_type(adjoint)
        if F_non_body is not None:
            F = F + F_non_body

        # 载荷必须在施加边界条件之前完成跨 rank 归约; 串行下为恒等操作.
        return self.reduce_load(F)

    def _non_body_loads_by_boundary_type(
                self,
                adjoint: bool = False
            ) -> Optional[TensorLike]:
        """按 pde.boundary_type 决定是否装配自然边界载荷

        assemble_external_load 与两个 apply_bc 变体都经过本方法, 保证「只消费外载
        的调用方」(子结构缩聚等) 与「求解全尺度系统的调用方」看到的是同一个载荷.

        - 'mixed'     : 装配点力、线载荷与边界牵引 (以及伴随载荷列);
        - 'dirichlet' : 全边界都是本质边界, 自然边界项不进入右端. 此时 pde 若仍
                        给出非体力载荷, 说明边界类型与载荷契约矛盾, 显式报错而不是
                        静默丢弃;
        - 其他        : 尚未定义装配语义, 报错.

        非伴随且非体力载荷全是集中力时, 载荷向量首次装配后缓存, 以各集中力的作用点与力向量
        为指纹: 指纹不变即返回缓存的副本 (逐位相同), 载荷被改动则重新装配; 换空间时作废.
        线载荷、边界牵引由可调用对象给出, 无从比较取值, 不缓存.
        """
        boundary_type = self._pde.boundary_type

        if boundary_type == 'mixed':
            loads = [load for load in self._pde.loads() if not isinstance(load, BodyForce)]
            fingerprint = None
            if not adjoint and loads and all(isinstance(load, PointForce) for load in loads):
                fingerprint = tuple((tuple(float(v) for v in load.point),
                                     tuple(float(v) for v in load.force())) for load in loads)
                cache = self._non_body_cache
                if cache is not None and cache[0] == fingerprint:
                    return self._tensor_space.function(bm.copy(cache[1]))
            F_non_body = self._assemble_non_body_loads(adjoint)
            if fingerprint is not None:
                self._non_body_cache = (fingerprint, bm.copy(F_non_body[:]))
            return F_non_body

        if boundary_type == 'dirichlet':
            unused = [
                type(load).__name__
                for load in self._pde.loads()
                if not isinstance(load, BodyForce)
            ]
            if unused:
                self._log_error(
                    f"boundary_type='dirichlet' 的全边界本质边界问题不接受自然边界"
                    f"载荷, 但 pde.loads() 给出了 {unused}"
                )
            return None

        self._log_error(f"Unsupported boundary type: {boundary_type}")

    def _assemble_non_body_loads(self, adjoint: bool = False) -> TensorLike:
        """组装点力、线载荷和边界牵引, 并保留现有伴随载荷入口."""
        space_uh = self._tensor_space
        F_physical = assemble_non_body_loads(space_uh, self._pde.loads(), q=self._integration_order)

        if not adjoint:
            return F_physical

        gd_adjoint = self._pde.adjoint_load_bc
        threshold_adjoint = self._pde.is_adjoint_load_boundary()

        isBdTDof = space_uh.is_boundary_dof(
            threshold=threshold_adjoint,
            method='interp',
        )
        isBdSDof = space_uh.scalar_space.is_boundary_dof(
            threshold=threshold_adjoint,
            method='interp',
        )
        ipoints_uh = space_uh.interpolation_points()
        gd_adjoint_val = gd_adjoint(ipoints_uh[isBdSDof])

        F_adjoint = space_uh.function()
        if space_uh.dof_priority:
            F_adjoint[:] = bm.set_at(
                F_adjoint[:],
                isBdTDof,
                gd_adjoint_val.T.reshape(-1),
            )
        else:
            F_adjoint[:] = bm.set_at(
                F_adjoint[:],
                isBdTDof,
                gd_adjoint_val.reshape(-1),
            )
        return bm.stack([F_physical, F_adjoint], axis=1)

    @variantmethod('fa')
    def apply_bc(self,
                K: Union[CSRTensor, COOTensor],
                F: TensorLike,
                adjoint: bool = False
            ) -> tuple[Union[CSRTensor, COOTensor], TensorLike]:
        """在全局矩阵上施加边界条件 (对称消元)

        Note
        ----
        本分支不经过 reduce_load / wrap_operator: 对称消元直接改写一个已经装配好
        的全局矩阵, 没有可供插入重叠归约的位置. 因此 'fa' 只能在单 rank 上用——
        多 rank 若放行, 各 rank 会在自己的局部矩阵上求解, 不报错但结果是错的.
        单 rank 下重叠归约本身是恒等操作, 带 dof_comm 也是安全的.
        """
        if self._dof_comm is not None and self._dof_comm.mpi_size > 1:
            self._log_error(
                f"operator_level='fa' 只支持单 rank, 当前为 "
                f"{self._dof_comm.mpi_size} 个 rank: 全局矩阵的对称消元没有重叠归约"
                f"的插入点。多 rank 请使用 operator_level='ea'"
            )

        space_uh = self._tensor_space

        #* 1. 非体力载荷处理 (边界类型分派由 _non_body_loads_by_boundary_type 统一负责) *#
        F_non_body = self._non_body_loads_by_boundary_type(adjoint)
        if F_non_body is not None:
            F = F + F_non_body
        self._F = F

        #* 2. Dirichlet 边界条件处理 - 强形式施加 *#
        # u_D 与自由度掩码只依赖空间与问题, 首次计算后缓存; 交出副本, 免得下游原地改写缓存
        uh_bd, isBdDof, bd_nonzero = self._dirichlet_data()
        uh_bd = bm.copy(uh_bd[:])
        self._prescribed_solution = uh_bd

        if adjoint:
            uh_bd = bm.repeat(uh_bd.reshape(-1, 1), 2, axis=1)
            # u_D 恒为零 (齐次约束) 时 K u_D = 0, 跳过; 大规模下这次乘法很昂贵
            if bd_nonzero:
                F = F - K.matmul(uh_bd[:])
            F = bm.set_at(F, (isBdDof, slice(None)), uh_bd[isBdDof, :])
        else:
            if bd_nonzero:
                F = F - K.matmul(uh_bd[:])
            F = bm.set_at(F, isBdDof, uh_bd[isBdDof])

        # 保结构消元, 不改写原矩阵 (self._K 仍为未施加边界条件的刚度矩阵)
        K = self._elimination.apply(K, isBdDof)

        return K, F

    @apply_bc.register('matrix_free')
    def apply_bc(self,
                K: AssemblyLevelExtension,
                F: TensorLike,
                adjoint: bool = False
            ) -> tuple[ConstrainedOperator, TensorLike]:
        """在矩阵自由算子上施加边界条件

        不改写任何矩阵, 而是把算子包进 ConstrainedOperator: matvec 时先把
        Dirichlet 自由度置零, 作用后再还原, 等价于 'fa' 的对称消元系统.

        本变体只用到 AssemblyLevelExtension 的接口, 不碰常驻形式, 因此 'ea' 与
        'pa' 共用它.
        """
        if adjoint:
            self._log_error(
                f"operator_level={self._operator_level!r} 不支持伴随双列右端项, "
                "请改用 operator_level='fa'"
            )

        F_non_body = self._non_body_loads_by_boundary_type(adjoint=False)
        if F_non_body is not None:
            F = F + F_non_body

        # 载荷必须在施加边界条件之前完成跨 rank 归约
        F = self.reduce_load(F)
        self._F = F

        # 边界自由度取给定值、内部自由度取零的基准向量与自由度掩码只依赖空间与问题, 与 'fa'
        # 共用首次计算后的缓存; 交出副本, 免得下游原地改写缓存
        uh_bd, isBdDof, bd_nonzero = self._dirichlet_data()
        uh_bd = bm.copy(uh_bd[:])

        operator = ConstrainedOperator(self.wrap_operator(K),
                                    gd=self._pde.dirichlet_bc,
                                    isDDof=isBdDof)

        # u_D 恒为零时 K u_D = 0, 只需把边界行置为 u_D, 与完整的 apply 逐位相同 (x - 0.0 = x).
        # 分布式下算子作用含跨 rank 的重叠归约, bd_nonzero 又是各 rank 自己判定的, 若只有部分
        # rank 跳过会使集体通信失配, 故只在串行时跳过
        if bd_nonzero or self._dof_comm is not None:
            F = operator.apply(F, uh_bd)
        else:
            F = bm.set_at(F, isBdDof, uh_bd[isBdDof])

        self._prescribed_solution = uh_bd

        return operator, F


    ##############################################################################################
    # 分布式扩展点 (串行下均为恒等操作)
    ##############################################################################################

    def wrap_operator(self, form: AssemblyLevelExtension):
        """在施加边界条件之前对单元级算子做一层包装

        串行下原样返回. 分布式实现覆盖本方法, 返回一个把 matvec 结果在重叠自由度
        上求和的包装 (示例中的 `distributed.OverlapOperator`), 之后的边界条件处理
        和求解都不需要知道它的存在.

        Note
        ----
        包装必须发生在 ConstrainedOperator 之前: 先跨 rank 组装出完整的算子作用,
        再在其上消去 Dirichlet 自由度. 顺序反过来会把边界行的置换也带进通信.
        """
        return form

    def reduce_load(self, F: TensorLike) -> TensorLike:
        """把按自由度分布的右端项在重叠自由度上归约

        串行下原样返回. 分布式实现覆盖本方法, 通常是 `dof_comm.sync_add(F)`.
        必须在施加 Dirichlet 边界条件之前调用, 否则边界行会被重复累加.
        """
        return F

    ##############################################################################################
    # 求解
    ##############################################################################################

    def solve_state(self,
                    rho_val: Optional[Union[TensorLike, Function]] = None,
                    adjoint: bool = False,
                    enable_timing: bool = False, 
                    **kwargs
                ) -> Dict[str, Function]:
        """组装刚度与载荷, 施加边界条件后求解位移.

        Parameters
        ----------
        rho_val : TensorLike or Function, optional
            密度场. 拓扑优化模式下必须提供; 标准有限元分析下被忽略并告警.
        adjoint : bool, optional
            为 ``True`` 时在刚度中叠加弹簧刚度, 右端项取物理载荷与伴随载荷两列,
            一次求解两个右端. 仅 ``operator_level='fa'`` 支持.
        enable_timing : bool, optional
            是否打印各阶段耗时.
        **kwargs
            传给 ``solve_system`` 的选项. 非 ``'fa'`` 层级下若未给出 ``x0``,
            以 ``apply_bc`` 得到的 Dirichlet 基准向量为初值.

        Returns
        -------
        dict
            ``'displacement'`` 为位移解: ``adjoint=False`` 时是有限元函数,
            ``adjoint=True`` 时是形状 ``(gdof, 2)`` 的张量, 第 0 列为物理解.
            ``'solver'`` 为 ``solve_system`` 返回的求解诊断.
        """
        t = None
        if enable_timing:
            t = timer(f"分析求解位移阶段")
            next(t)

        if self._topopt_algorithm is None:
            if rho_val is not None:
                self._log_warning("标准有限元分析模式下忽略密度分布参数 rho")
        
        elif self._topopt_algorithm in ['density_based', 'level_set']:
            if rho_val is None:
                error_msg = f"拓扑优化算法 '{self._topopt_algorithm}' 需要提供密度分布参数 rho"
                self._log_error(error_msg)

        if adjoint:
            K_struct = self.assemble_stiff_matrix(rho_val=rho_val)
            K_spring = self.assemble_spring_stiff_matrix()
            K0 = K_struct + K_spring
            F0_struct = self.assemble_body_force_vector()
            F0_spring = bm.zeros_like(F0_struct)
            F0 = bm.stack([F0_struct, F0_spring], axis=1)

            K, F = self.apply_bc(K0, F0, adjoint)

            uh = bm.zeros(F.shape, dtype=bm.float64, device=F.device)
        
        else:
            K0 = self.assemble_stiff_matrix(rho_val=rho_val)
            if t is not None:
                t.send('双线性型组装')

            F0 = self.assemble_body_force_vector()
            if t is not None:
                t.send('线性型组装')

            K, F = self.apply_bc(K0, F0)
            if t is not None:
                t.send('边界条件处理')

            uh = self._tensor_space.function()

        # 矩阵自由层级从刚刚由 apply_bc 得到的 Dirichlet 基准向量起步; 显式传入
        # 而非让 solve_system 去读实例状态
        if self._operator_level != 'fa':
            kwargs.setdefault('x0', self._prescribed_solution)

        _, solver_info = self.solve_system(K, F, uh, **kwargs)

        if t is not None:
            t.send('求解')
            t.send(None)

        return {
            'displacement': uh,
            'solver': solver_info,
            }

    def solve_adjoint(self, 
                    rhs: TensorLike,
                    rho_val: Optional[Union[TensorLike, Function]] = None,
                    **kwargs
                ) -> TensorLike:
        """求解伴随方程 K @ λ = rhs, 伴随问题的 Dirichlet 条件为齐次.

        Raises
        ------
        NotImplementedError
            ``operator_level`` 不是 ``'fa'``: 矩阵自由层级的算子不支持按行列施加
            边界条件.
        """
        if self._operator_level != 'fa':
            raise NotImplementedError(
                f"solve_adjoint 只支持 operator_level='fa', 当前为 {self._operator_level!r}."
            )
        # 组装刚度矩阵
        K0 = self.assemble_stiff_matrix(rho_val=rho_val)

        # 获取 Dirichlet 边界自由度
        gd = self._pde.dirichlet_bc
        threshold = self._pde.is_dirichlet_boundary()
        _, isBdDof = self._tensor_space.boundary_interpolate(
                                        gd=gd,
                                        threshold=threshold,
                                        method='interp'
                                    )
        
        # 先处理右端项 (伴随问题边界条件为齐次, λ = 0)
        rhs_bc = bm.copy(rhs)
        rhs_bc[isBdDof] = 0.0

        # 再处理刚度矩阵
        K = self._elimination.apply(K0, isBdDof)
        
        # 初始化结果并求解
        adjoint_lambda = bm.zeros_like(rhs_bc)
        self.solve_system(K, rhs_bc, adjoint_lambda, **kwargs)

        return adjoint_lambda


    def _as_iterative_operator(self, K):
        """把刚度算子转成迭代解法可以直接作用的形式

        矩阵自由层级下 K 本身就支持 @ 运算; 'fa' 下 PyTorch 后端需要绕开 FEALPy
        的 CSRTensor, 其余后端直接用 CSR: numpy 后端的 csr_spmm 就是 scipy 的
        csr_matvec, 转 COO 只会多出 (2, nnz) 的 int64 索引 (首档约 6 GiB).
        """
        if self._operator_level != 'fa':
            return K

        if bm.backend_name == 'pytorch':
            #? 需要使用 PyTorch 原始的稀疏矩阵, FEALPy 中的 CSRTensor 存在问题
            import torch
            K_coo_torch = torch.sparse_coo_tensor(
                                            indices=bm.stack([K.row, K.col]),
                                            values=bm.tensor(K.data),
                                            size=K.shape,
                                            device=K.data.device
                                        )
            #? matmul 函数下 K 必须是 COO 格式, 不能是 CSR 格式, 否则 GPU 下 device_put 函数会出错
            K._values = bm.copy(K._values)

            return K_coo_torch.to_sparse_csr()

        return K

    def assemble_operator_diagonal(self, K) -> TensorLike:
        """取已施加 Dirichlet 条件的系统算子对角, 供 Jacobi 类预条件使用

        取对角已下沉到求解层的 ``soptx.solvers.operator_diagonal``, 按算子实际
        能提供什么分派: 'ea' 下 ConstrainedOperator 自报 (Dirichlet 自由度上恒
        为 1, 其余转发内层算子, 跨 rank 归约由 OverlapOperator 完成), 'fa' 下扫
        对称消元后稀疏矩阵的 COO 取主对角.

        analyzer 内部已不再调用本方法: ``_build_solver`` 把算子直接交给
        ``DiagonalPreconditioner``, 由它在 setup 时取. 本方法保留为一层转发,
        供 examples 与 experiments 里的既有脚本沿用.

        Parameters
        ----------
        K : 'fa' 下为全局稀疏矩阵, 'ea' 下为 ConstrainedOperator

        Returns
        -------
        diag : (TGDOF, ) 的算子对角, SPD 系统下逐元严格为正
        """
        from soptx.solvers import operator_diagonal

        return operator_diagonal(K)


    def _preconditioner_operator(self):
        """按 preconditioner_level 另建一个算子, 供预条件子使用

        主算子在 solve_system 拿到时已经施加过边界条件 (solve_state 里
        ``K, F = self.apply_bc(K0, F0)``), 预条件层级不走一遍同样的处理就是在给
        奇异矩阵做分解, 因此本方法负责补上矩阵侧的边界条件.

        不复用 ``apply_bc``: 它是按 ``operator_level`` 定死变体的 variantmethod,
        预条件层级可能属于另一个变体; 它还同时做载荷侧的事 (累加非体力载荷, 跨
        rank 归约, 写 ``_F`` 与 ``_prescribed_solution``), 二次调用会重复加载荷并
        覆盖状态. 这里只取两个变体的矩阵侧, 各自都已经是现成的单句.

        Returns
        -------
        'fa' 下为对称消元后的全局稀疏矩阵, 其余层级下为 ConstrainedOperator

        Notes
        -----
        必须在 ``assemble_stiff_matrix`` 之后调用: 层级从 ``self._integrator``
        构造, 而密度系数是 ``_update_density_coefficient`` 在装配时写进积分子的,
        提前调用会读到上一步的密度. ``_build_solver`` 的调用点天然满足这一点.

        结果刻意不缓存: 拓扑优化每步都改密度, 缓存必然读到陈旧的刚度. 代价是每次
        求解多一次装配 —— 这是第一版的取舍, 把失效管理与本轴解耦.
        """
        space_uh = self._tensor_space
        threshold_uh = self._pde.is_dirichlet_boundary()
        isBdDof = space_uh.is_boundary_dof(threshold=threshold_uh, method='interp')

        # 不传 pattern: self._csr_pattern 是主算子的 CSR 骨架缓存,
        # assemble_stiff_matrix 会回写它, 共用会让两根轴互相干扰
        level = create_level(self._preconditioner_level,
                            space=space_uh,
                            integrator=self._integrator)

        if self._preconditioner_level == 'fa':
            # 每次另装配一份骨架, 槽位缓存必然失效; 用临时实例, 不挤掉主算子的缓存
            return SymmetricElimination().apply(level.operator, isBdDof)

        return ConstrainedOperator(self.wrap_operator(level.operator),
                                gd=self._pde.dirichlet_bc,
                                isDDof=isBdDof)

    def _build_solver(self, solver_type, K, **kwargs):
        """按名字造出求解器, 并选定它要绑定的算子

        名字到类的分派由 soptx.solvers.registry 完成, 本方法只剩两件本地的事:
        各后端读哪些选项, 以及 'fa'/'ea' 下算子形态的差异.

        Returns
        -------
        solver : LinearSolver
            尚未 setup 的求解器实例
        op     : 该求解器要绑定的算子
        extra  : 并入返回 info 的后端专属诊断键
        tol    : 迭代解法的 (atol, rtol), 直接法为 None
        """
        from soptx.solvers import available, create

        if solver_type not in available():
            self._log_error(
                f"未知的求解器类型: {solver_type}; "
                f"可用: {', '.join(available())}"
            )

        if solver_type == 'cg':
            # 优先级: 调用方 kwargs > 构造时的 solver_options > 硬编码默认
            maxiter = kwargs.get('maxiter', self._solver_options.get('maxiter', 5000))
            atol = kwargs.get('atol', self._solver_options.get('atol', 1e-12))
            rtol = kwargs.get('rtol', self._solver_options.get('rtol', 1e-12))
            precond = kwargs.get('precond',
                                self._solver_options.get('precond', None))
            residual_refresh = int(kwargs.get('residual_refresh',
                                self._solver_options.get('residual_refresh', 0)))

            M = None
            # 无预条件子时三个 norm_type 数值等价, 取默认的 natural
            norm_type = 'natural'
            if precond is not None:
                from soptx.solvers import (
                    DiagonalPreconditioner,
                    OperatorCapabilityError,
                )

                # 预条件子的算子源: 默认就是主算子本身, 给了 preconditioner_level
                # 就另建一个. 两者都是已施加边界条件的算子, 对预条件子而言等价.
                #
                # 先在 fem 侧的算子上 setup 再交给 CG (CG 只对未 setup 的预条件子
                # 做级联): 一是 _as_iterative_operator 的产物在 pytorch 后端是原生
                # torch 稀疏张量, 取对角要另说; 二是级联用的是主算子, 那样
                # preconditioner_level 就白设了
                pc_level = self._preconditioner_level or self._operator_level
                if self._preconditioner_level is None:
                    K_pc = K
                else:
                    K_pc = self._preconditioner_operator()

                if precond in ('jacobi', 'diagonal'):
                    M = DiagonalPreconditioner()
                elif precond == 'mg':
                    M = self._multigrid_preconditioner(**kwargs)
                elif precond in ('scipy', 'mumps'):
                    # 直接法当预条件子: LinearSolver.__matmul__ 本就是"零初值解一
                    # 次"的预条件子模式, 不需要适配层. 它是精确逆, CG 应一步收敛,
                    # 因此主要用途是验证两个层级确实是同一个离散算子
                    M = create(precond)
                else:
                    self._log_error(
                        f"未知的预条件子类型: {precond}; "
                        f"可选 'jacobi'/'diagonal', 'mg', 'scipy', 'mumps'"
                    )

                try:
                    M.setup(K_pc)
                except OperatorCapabilityError as exc:
                    self._log_error(
                        f"预条件子 {precond!r} 无法绑定到 {pc_level!r} 层级的算子 "
                        f"(operator_level={self._operator_level!r}, "
                        f"preconditioner_level={self._preconditioner_level!r}): "
                        f"{exc} 请把 preconditioner_level 设为 'fa'"
                    )

                # 判据范数与下游口径对齐: cg 默认在 natural 范数
                # sqrt(r^T M^-1 r) 下停机, 而本方法返回的 relres 是 2-范数,
                # Jacobi 的 diag^-1 可达 1e6 量级, 两个口径能差几个数量级.
                # 显式选 unpreconditioned 让停机判据也用 ||r||_2, 不多做 matvec
                norm_type = 'unpreconditioned'
                # 与范数选择正交: 递推残差在长迭代下会漂移, 周期性用 b - A x
                # 校正. 撤掉它需要单独的数值证据, 故与 M 绑定保持开启
                if residual_refresh <= 0:
                    residual_refresh = 50

            # cg 支持批量求解, batch_first 为 False 时, 表示第一个维度为自由度维度
            solver = create('cg', M=M, atol=atol, rtol=rtol, maxit=maxiter,
                            batch_first=False,
                            norm_type=norm_type,
                            residual_refresh=residual_refresh)

            return (solver, self._as_iterative_operator(K),
                    {'maxit': maxiter, 'precond': precond}, (atol, rtol))

        if solver_type == 'mumps':
            # sym=0 按一般非对称矩阵分解; 位移元刚度阵经对称消元后仍是对称正定,
            # 传 1 (正定) 或 2 (一般对称) 只让 MUMPS 读下三角, 因子存储与运算量
            # 大致减半. 默认保持 0, 由调用方显式开启
            mumps_sym = int(kwargs.get('sym', 0))

            return create('mumps', sym=mumps_sym), K, {'sym': mumps_sym}, None

        return create(solver_type), K, {}, None

    def _multigrid_preconditioner(self, level: Optional[AssemblyLevelExtension] = None, **kwargs):
        """几何多重网格预条件子, 粗层按层级对象所带的单元系数重建.

        Parameters
        ----------
        level : 提供粗层系数的层级对象, 须与最细层算子出自同一次装配. 缺省取最近一次
            ``assemble_stiff_matrix`` 产出的 ``self._level``; 最细层算子不经
            ``assemble_stiff_matrix`` 产出的子类须显式传入.
        **kwargs : 求解调用的选项, 优先于构造时的 ``solver_options``. 读取
            ``mg_omega`` (默认 None, 各层自动取值), ``mg_sweeps`` (默认 1),
            ``mg_coarse_solver`` (默认 'scipy') 与 ``mg_coarse_max_dofs`` (默认 20000,
            只在首次构造层次时生效).

        Returns
        -------
        mg : 尚未 setup 的 ``Multigrid``; 最细层算子由调用方 setup 时给出.

        Notes
        -----
        粗层系数取自层级对象的 ``scale``; 只有由单元系数缩放参考刚度装配的层级 ('fa' 与 'ea' 下
        单元密度的拓扑优化) 带这组系数, 其余层级下报错. 层次的拓扑部分首次构造后缓存, 换空间
        时作废. 网格须为结构化六面体网格. 两个层级下
        施加边界条件后的最细层算子都是 Pi_I K Pi_I + Pi_D ('fa' 为保结构消元的 CSR 矩阵,
        'ea' 为 ``ConstrainedOperator``), 粗层只由单元系数、K^0 与 Dirichlet 掩码构造, 与最细层
        的常驻形式无关.
        """
        from soptx.fem.multigrid import StructuredHexHierarchy

        if level is None:
            level = self._level
        scale = None if level is None else level.scale
        if scale is None:
            self._log_error(
                "precond='mg' 需要由单元系数缩放参考刚度装配的层级, 即 'fa' 或 'ea' 下单元密度的"
                f"拓扑优化; 当前 operator_level={self._operator_level!r}, "
                f"topopt_algorithm={self._topopt_algorithm!r}, 层级 {level!r}"
            )

        def option(name, default):
            return kwargs.get(name, self._solver_options.get(name, default))

        if self._mg_hierarchy is None:
            _, isBdDof, _ = self._dirichlet_data()
            try:
                self._mg_hierarchy = StructuredHexHierarchy(
                    self._tensor_space, isBdDof, self._reference_stiffness_matrices()[0],
                    coarse_max_dofs=int(option('mg_coarse_max_dofs', 20000)))
            except (ValueError, NotImplementedError) as exc:
                self._log_error(f"precond='mg' 无法构造多重网格层次: {exc}")
        self._mg_hierarchy.update(scale)

        return self._mg_hierarchy.build_multigrid(omega=option('mg_omega', None),
                                                  sweeps=int(option('mg_sweeps', 1)),
                                                  coarse_solver=option('mg_coarse_solver', 'scipy'))

    def solve_system(self, K, F, out, **kwargs):
        """在给定算子上求解线性系统, 解就地写入 out

        Parameters
        ----------
        K   : 'fa' 下为全局稀疏矩阵, 'ea' 下为支持 @ 运算的算子
        F   : 右端项, (TGDOF, ) 或批量的 (TGDOF, nrhs)
        out : 就地写入的解向量

        Returns
        -------
        out  : 就地写入的解向量
        info : 求解诊断, 含 'name'、'niter'、'relres' 和 'converged'; 直接法
               另含 'sym', 迭代解法另含 'maxit'、'precond'、
               'recursive_residual' 和 'true_residual'

        Note
        ----
        本方法不读取任何由 apply_bc 留下的状态. 迭代解法的初值必须由调用方通过
        kwargs['x0'] 显式给出——对 'ea' 而言通常就是 apply_bc 产生的
        prescribed_solution, 它已满足 Dirichlet 值.

        直接法在 'ea' 下不可用一事不在此处硬编码判断: DirectSolver 声明自己需要
        显式矩阵, matrix-free 算子给不出, setup 时即抛 OperatorCapabilityError,
        本方法只负责把它转成与其他使用错误一致的 RuntimeError.

        分解不跨调用复用: 求解完即释放, MUMPS 上下文的生命周期与改造前逐次
        建销一致. 状态解与伴随解共用一个分解要改 analyzer 的状态管理, 单独一步做.

        这是分布式求解唯一的注入点: 并行只需在此处把 fealpy 的 cg 换成带
        overlap 加权内积的版本, 上层的组装与边界条件处理不受影响. 覆盖本方法的
        实现负责自行处理 dof_comm.
        """
        if self._dof_comm is not None:
            raise NotImplementedError(
                "串行求解器不能用于分布式系统。请覆盖 solve_system, "
                "提供带 overlap 加权内积的 CG, 参考 examples/matrix_free_elasticity"
            )

        from soptx.solvers import OperatorCapabilityError

        solver_type = kwargs.get('solver', self._solve_method)
        solver, op, extra, tol = self._build_solver(solver_type, K, **kwargs)

        try:
            try:
                solver.setup(op)
            except OperatorCapabilityError as exc:
                self._log_error(
                    f"operator_level={self._operator_level!r} 下无法使用求解器 "
                    f"'{solver_type}': {exc} 请改用 solver='cg'; "
                    f"若是想要显式矩阵上的预条件, 可保留 solver='cg' 并设 "
                    f"preconditioner_level='fa'"
                )

            out[:], raw = solver.solve(F[:], kwargs.get('x0', None))
        finally:
            # 直接法持有 SuperLU 分解或 MUMPS 上下文, 用完即释放. 预条件子位上的
            # 直接法 (preconditioner_level 配 precond='scipy'/'mumps') 持有的是
            # 另一份, 一并释放, 否则 MUMPS 侧的内存不回收
            for owner in (solver, getattr(solver, 'M', None)):
                close = getattr(owner, 'close', None)
                if close is not None:
                    close()

        info = {'name': solver_type, **extra,
                'niter': int(raw['niter']),
                'relres': float(raw['relres']),
                'converged': bool(raw['converged'])}

        if tol is not None:
            # 收敛与否以求解器自己的退出原因为准, 不在这里重判. 原来那段用
            # max(atol, rtol * ||F||_2) 重算是错的: solve_system 支持传 x0,
            # 热启动时 rtol 的参照量是 ||r0|| 而非 ||F||. 判据口径的对齐已在
            # 求解器内部完成 (见上面 _build_solver 的 norm_type)
            reason = raw.get('reason', None)
            if reason is not None:
                info['reason'] = reason.name
            info['reference_norm'] = float(raw.get('reference_norm', 0.0))
            info['recursive_residual'] = float(raw['residual'])
            true_residual = raw.get('true_residual', None)
            info['true_residual'] = (None if true_residual is None
                                     else float(true_residual))

        return out, info


    ###############################################################################################
    # 外部方法
    ###############################################################################################

    def compute_solid_stiffness_matrix(self):
        """计算并缓存实体材料的逐单元刚度矩阵 K_e^0.

        Returns
        -------
        ke0 : (NC, TLDOF, TLDOF) 的逐单元刚度矩阵.

        Notes
        -----
        装配方法取分析器的 ``assembly_method``, 与全局刚度矩阵一致, 使灵敏度与刚度在舍入
        意义上也自洽. 不固定用 ``'standard'``: 它带积分点维的中间数组, 大规模下一次性峰值
        约为结果的 10 倍.
        """
        lea = LinearElasticIntegrator(material=self._material,
                            coef=None,
                            q=self._integration_order,
                            method=self._assembly_method)
        ke0 = lea.assembly(space=self.tensor_space)

        self._cached_ke0 = ke0

        return ke0

    def _reference_stiffness_matrices(self) -> TensorLike:
        """单元密度下 K_e = s_e K^0_{k(e)} 所用的参考单元矩阵, 首次计算后缓存.

        Returns
        -------
        reference : (N_k, TLDOF, TLDOF) 的参考单元矩阵, 单元 e 取第 e % N_k 份. 未给
            ``reference_classes`` 时 N_k = NC, 即 ``compute_solid_stiffness_matrix`` 的逐单元缓存.

        Notes
        -----
        给出 ``reference_classes`` 时只积分前 N_k 个代表单元, 不形成逐单元的 (NC, TLDOF, TLDOF).
        另积分中间与最后一个单元, 与其所属类的代表比较, 相对偏差超过 1e-10 即报错; 这只能
        拦住重编号、非等距之类的明显误用, 不能证明每个单元都满足平移类约定.
        """
        if self._reference_classes is None:
            if self._cached_ke0 is None:
                self.compute_solid_stiffness_matrix()
            return self._cached_ke0
        if self._reference_ke0 is None:
            n_classes = self._reference_classes
            n_cells = int(self._mesh.number_of_cells())

            def integrate(index):
                integrator = LinearElasticIntegrator(material=self._material, coef=None,
                                                     q=self._integration_order, index=index,
                                                     method=self._assembly_method)
                return integrator.assembly(space=self.tensor_space)

            reference = integrate(slice(0, n_classes))
            sample = bm.tensor(sorted({n_cells // 2, n_cells - 1}), dtype=bm.int64)
            sampled = integrate(sample)
            expected = reference[sample % n_classes]
            scale = float(bm.max(bm.abs(reference)))
            if float(bm.max(bm.abs(sampled - expected))) > 1e-10 * scale:
                self._log_error(
                    f"reference_classes={n_classes} 与网格不符: 抽查单元的 K_e^0 与其平移类代表不一致, "
                    f"网格须按 create_box_mesh / from_box 的编号约定生成且等距"
                )
            self._reference_ke0 = reference

        return self._reference_ke0

    # ------------------------------------------------------------------
    # 泊松比随密度插值 (近不可压缩算例)
    #
    # K_e = λ*_e K_e^λ + μ_e K_e^μ, 两个基矩阵与设计无关; λ*、μ 的定义与导数见
    # soptx.materials.lame_split, 这里只负责基矩阵的积分与链式法则.
    # ------------------------------------------------------------------

    def _check_poisson_interpolation_support(self) -> None:
        """泊松比插值只在逐单元本构矩阵能进入装配的组合下允许, 其余组合直接报错"""
        density_location = self.interpolation_scheme.density_location
        if density_location != 'element':
            self._log_error(
                f"泊松比随密度插值只支持 density_location='element', "
                f"当前为 '{density_location}'"
            )
        if self._assembly_method != 'standard':
            self._log_error(
                f"泊松比随密度插值只支持 assembly_method='standard' (只有它接受"
                f"逐单元本构矩阵), 当前为 '{self._assembly_method}'"
            )
        if 'pa' in (self._operator_level, self._preconditioner_level):
            self._log_error(
                "泊松比随密度插值不支持 'pa' 层级: partial assembly 的 QFunction "
                "假定 coef 为实体本构矩阵的标量倍"
            )

    def compute_lame_basis_matrices(self) -> tuple:
        """计算与设计无关的单元基矩阵 (K_e^λ, K_e^μ), 满足 K_e = λ*_e K_e^λ + μ_e K_e^μ

        Returns
        -------
        (ke_lambda, ke_mu) : 各为 (NC, TLDOF, TLDOF)
        """
        NC = self._mesh.number_of_cells()
        ones = bm.ones((NC, ), dtype=bm.float64, device=self._mesh.device)
        results = []
        for D_basis in lame_basis_matrices(self._material.hypothesis, device=self._mesh.device):
            lea = LinearElasticIntegrator(material=self._material,
                                coef=bm.einsum('c, kl -> ckl', ones, D_basis),
                                q=self._integration_order,
                                method='standard')
            results.append(lea.assembly(space=self.tensor_space))

        self._cached_ke_lambda, self._cached_ke_mu = results

        return tuple(results)

    def _stiffness_derivative_with_poisson(self,
                                        material_params: tuple,
                                        material_derivs: tuple,
                                    ) -> TensorLike:
        """E 与 ν 同时插值时的 ∂K_e/∂ρ_e (单元密度)

        ∂K_e/∂ρ = (∂λ*/∂E E' + ∂λ*/∂ν ν') K_e^λ + (∂μ/∂E E' + ∂μ/∂ν ν') K_e^μ
        """
        E_rho, nu_rho = material_params[0], material_params[1]
        dE_rho, dnu_rho = material_derivs[0], material_derivs[1]

        dlam_dE, dlam_dnu, dmu_dE, dmu_dnu = lame_parameter_derivatives(E_rho, nu_rho,
                                                                        self._material.hypothesis)
        dlam = dlam_dE * dE_rho + dlam_dnu * dnu_rho   # (NC, )
        dmu = dmu_dE * dE_rho + dmu_dnu * dnu_rho      # (NC, )

        if self._cached_ke_lambda is None or self._cached_ke_mu is None:
            self.compute_lame_basis_matrices()

        return (bm.einsum('c, cij -> cij', dlam, self._cached_ke_lambda)
                + bm.einsum('c, cij -> cij', dmu, self._cached_ke_mu))
    
    def compute_sub_element_stiffness_matrix(self) -> TensorLike:
        """计算并缓存多分辨率下各子密度单元上的实体刚度.

        Returns
        -------
        ke0_sub : (NC, n_sub, TLDOF, TLDOF), 与 K_e^0 同口径 (含 E_0), 满足
            K_e = Σ_n (E(ρ_{e,n}) / E_0) ke0_sub[e, n], 因而
            ∂K_e/∂ρ_{e,n} = (E'(ρ_{e,n}) / E_0) ke0_sub[e, n].

        Notes
        -----
        积分阶按 n_sub 选取: 4 <= n_sub <= 9 取 3, n_sub >= 16 取 2, 其余取 p + 3.
        """
        n_sub = self.interpolation_scheme.n_sub
        if 4 <= n_sub <= 9:
            q = 3
        elif n_sub >= 16:
            q = 2
        else:
            q = self._scalar_space.p + 3

        ke0_sub = multiresolution_sub_element_matrices(self.tensor_space, self._material, q=q, n_sub=n_sub)
        self._cached_ke0_sub = ke0_sub

        return ke0_sub

    def compute_element_energy_derivative(self,
                                          rho_val: Union[TensorLike, Function],
                                          uhe: TensorLike,
                                        ) -> TensorLike:
        """计算单元密度下每个单元的 u_e^T (dK_e / drho_e) u_e.

        Parameters
        ----------
        rho_val : (NC, ) 的单元物理密度.
        uhe : (NC, TLDOF) 的单元位移.

        Returns
        -------
        energy : (NC, ) 的单元能量导数; 柔顺度灵敏度为其相反数.

        Notes
        -----
        泊松比不随密度插值时 dK_e / drho_e = (E'(rho_e) / E_0) K_e^0, 故按
        (E'(rho_e) / E_0) (u_e^T K_e^0 u_e) 计算, 不构造 (NC, TLDOF, TLDOF) 的导数矩阵;
        泊松比随密度插值时退回 ``compute_stiffness_matrix_derivative`` 后缩并.
        """
        material_params = self.interpolation_scheme.interpolate_material(
                                            material=self._material,
                                            rho_val=rho_val,
                                            integration_order=self._integration_order,
                                            displacement_mesh=self._mesh,
                                        )
        if isinstance(material_params, tuple):
            diff_ke = self.compute_stiffness_matrix_derivative(rho_val=rho_val)
            return bm.einsum('ci, cij, cj -> c', uhe, diff_ke, uhe)

        material_derivs = self.interpolation_scheme.interpolate_material_derivative(
                                                material=self._material,
                                                rho_val=rho_val,
                                                integration_order=self._integration_order,
                                            )
        dE_rho = material_derivs[0] if isinstance(material_derivs, tuple) else material_derivs
        ke0 = self._reference_stiffness_matrices()
        n_classes = int(ke0.shape[0])
        if n_classes == uhe.shape[0]:
            energy = bm.einsum('ci, cij, cj -> c', uhe, ke0, uhe)
        else:
            # 按平移类收缩: 单元 e = g N_k + k 取第 k 份参考矩阵
            grouped = bm.reshape(uhe, (-1, n_classes, uhe.shape[-1]))
            energy = bm.reshape(bm.einsum('gki, kij, gkj -> gk', grouped, ke0, grouped), (-1, ))
        return (dE_rho / self._material.youngs_modulus) * energy

    def compute_stiffness_matrix_derivative(self, rho_val: Union[TensorLike, Function]) -> TensorLike:
        """计算局部刚度矩阵关于物理密度的导数 (灵敏度)"""
        density_location = self.interpolation_scheme.density_location

        material_derivs = self.interpolation_scheme.interpolate_material_derivative(
                                                material=self._material, 
                                                rho_val=rho_val,
                                                integration_order=self._integration_order,
                                            ) 
        if isinstance(material_derivs, tuple):
            dE_rho = material_derivs[0]
        else:
            dE_rho = material_derivs

        # 泊松比是否随密度插值以 interpolate_material 的返回为准: 可压缩材料下
        # interpolate_material_derivative 仍会返回 dν, 但 ν 本身并未插值
        material_params = self.interpolation_scheme.interpolate_material(
                                            material=self._material,
                                            rho_val=rho_val,
                                            integration_order=self._integration_order,
                                            displacement_mesh=self._mesh,
                                        )
        nu_interpolated = isinstance(material_params, tuple)
        
        if density_location in ['element']:
            # rho_val.shape = (NC, )
            if nu_interpolated:
                return self._stiffness_derivative_with_poisson(material_params, material_derivs)

            diff_coef_element = dE_rho / self._material.youngs_modulus # (NC, )

            if self._cached_ke0 is None:
                self.compute_solid_stiffness_matrix()

            ke0 = self._cached_ke0

            diff_ke = bm.einsum('c, cij -> cij', diff_coef_element, ke0) # (NC, TLDOF, TLDOF)
 
            return diff_ke
        
        elif density_location in ['element_multiresolution']:
            # rho_val.shape = (NC, n_sub); 子单元上材料系数为常数, 提到积分外
            if nu_interpolated:
                self._log_error("泊松比随密度插值不支持 density_location='element_multiresolution'")

            diff_coef_sub_element = dE_rho / self._material.youngs_modulus # (NC, n_sub)
            ke0_sub = multiresolution_sub_element_matrices(self._tensor_space, self._material,
                                                        q=self._integration_order, n_sub=rho_val.shape[1])

            return bm.einsum('cn, cnij -> cnij', diff_coef_sub_element, ke0_sub) # (NC, n_sub, TLDOF, TLDOF)

        elif density_location in ['node']:
            # rho_val.shape = (NN, )
            diff_coef_q = dE_rho / self._material.youngs_modulus # (NC, NQ)
            mesh = self._mesh
            qf = mesh.quadrature_formula(q=self._integration_order)
            # bcs_e.shape = ( (NQ_x, GD), (NQ_y, GD) ), ws_e.shape = (NQ, )
            bcs, ws = qf.get_quadrature_points_and_weights()

            # 密度空间在高斯积分点处的基函数
            phi = rho_val.space.basis(bcs)[0] # (NQ, NCN)

            D0 = self._material.elastic_matrix()[0, 0] # (NS, NS)
            B = self.compute_strain_matrix(self._integration_order) # (NC, NQ, NS, TLDOF)
            BDB = bm.einsum('cqki, kl, cqlj -> cqij', B, D0, B) # (NC, NQ, TLDOF, TLDOF)

            if isinstance(mesh, SimplexMesh):
                cm = mesh.entity_measure('cell')
                kernel = bm.einsum('q, c, cq, cqij -> cqij', ws, cm, diff_coef_q, BDB)
            else:
                J = mesh.entity_view('cell').jacobi_matrix(bcs)
                detJ = bm.abs(bm.linalg.det(J))
                kernel = bm.einsum('q, cq, cq, cqij -> cqij', ws, detJ, diff_coef_q, BDB)

            diff_ke = bm.einsum('cqij, ql -> clij', kernel, phi) # (NC, NCN, TLDOF, TLDOF)

            return diff_ke
        
    def compute_strain_matrix(self, integration_order: Optional[int] = None) -> TensorLike:
        """
        计算应变-位移矩阵 B
        
        Parameters
        ----------
        integration_order : 积分阶次, 默认使用分析器的积分阶次
        
        Returns
        -------
        B : 应变-位移矩阵
            - 单分辨率: (NC, NQ, NS, TLDOF)
            - 多分辨率: (NC, n_sub, NQ, NS, TLDOF)

        Note
        ----
        B 只取决于位移离散和积分点位置, 与密度自由度住在哪里无关; 唯一的区别是
        多分辨率要在子密度单元的积分点上求值, 因此这里只按是否多分辨率分支.
        """
        if integration_order is None:
            integration_order = self._integration_order

        density_location = self.interpolation_scheme.density_location

        if density_location in ['element_multiresolution']:
            from soptx.fem.utils import (calculate_multiresolution_gphi_eg,
                                            reshape_multiresolution_data_inverse)
            n_sub = self.interpolation_scheme.n_sub
            gphi_eg_reshaped = calculate_multiresolution_gphi_eg(
                                        s_space_u=self._scalar_space,
                                        q=integration_order,
                                        n_sub=n_sub
                                    )  # (NC*n_sub, NQ, LDOF, GD)
            B_reshaped = self._material.strain_matrix(
                                            dof_priority=self._tensor_space.dof_priority,
                                            gphi=gphi_eg_reshaped
                                        )  # (NC*n_sub, NQ, NS, TLDOF)
            B = reshape_multiresolution_data_inverse(
                            mesh=self._mesh,
                            data_flat=B_reshaped,
                            n_sub=n_sub
                        )  # (NC, n_sub, NQ, NS, TLDOF)

        else:
            qf = self._mesh.quadrature_formula(integration_order)
            bcs, _ = qf.get_quadrature_points_and_weights()
            gphi = self._scalar_space.grad_basis(bcs, variable='x')  # (NC, NQ, LDOF, GD)
            B = self._material.strain_matrix(
                                                dof_priority=self._tensor_space.dof_priority,
                                                gphi=gphi
                                            )  # (NC, NQ, NS, TLDOF)

        return B
    
    def compute_stress_state(self, 
                            state: dict,
                            integration_order: Optional[int] = None
                        ) -> Dict[str, TensorLike]:
        """
        计算基础应力状态, 负责计算基于当前位移场的实体柯西应力

        Parameters
        ----------
        state : dict
            状态字典, 必须包含 'displacement' (位移场).
        integration_order : int, optional
            积分阶次. 默认为 1, 单纯形上即单元形心单点. 局部应力约束不传该
            参数, 故此默认值就是约束的评价位置, 属问题定义而非数值参数;
            考察胞内起伏应显式传高阶, 不要改默认值.

        Returns
        -------
        dict : 包含以下键值的字典
            - 'stress_solid': 实体柯西应力张量 (Voigt 向量形式)
              Shape: (NC, NQ, NS)
              
              !! 关键提示: 应力分量顺序 (Voigt Notation) !!
              -------------------------------------------
              Index 0: sigma_xx (正应力 X)
              Index 1: sigma_yy (正应力 Y)
              Index 2: tau_xy   (剪应力 XY)
              -------------------------------------------
        """        
        if integration_order is None:
            # 单点 = 单元形心; 这是应力约束的评价位置定义, 见上方 docstring.
            integration_order = 1

        if state is None:
            self._log_error("compute_stress_state 需要传入有效的 state 字典")
        
        uh = state['displacement']
        cell2dof = self._tensor_space.cell_to_dof()
        uh_e = uh[cell2dof]

        # 1. 计算应变-位移矩阵 B
        B = self.compute_strain_matrix(integration_order)
        
        # 2. 计算实体柯西应力
        stress_tensor = self._material.calculate_stress_vector(B, uh_e)
        
        # 返回最原始的应力张量 (或向量形式)
        return {'stress_solid': stress_tensor}
    
    ##############################################################################################
    # 内部方法
    ##############################################################################################

    def _dirichlet_data(self) -> tuple[TensorLike, TensorLike, bool]:
        """取 Dirichlet 基准向量 u_D, 自由度掩码及 u_D 是否非零, 首次计算后缓存.

        Returns
        -------
        uh_bd : (gdof, ) 的基准向量, Dirichlet 自由度取给定值, 其余为零; 调用方不得原地改写.
        isBdDof : (gdof, ) 的 Dirichlet 自由度布尔掩码.
        nonzero : u_D 是否含非零分量; 为 False 时 K u_D = 0, 右端项无需修正.

        Notes
        -----
        只依赖张量空间与问题, 与密度无关; 换空间时由 ``tensor_space`` 的 setter 作废.
        """
        space = self._tensor_space
        if self._dirichlet_cache is None or self._dirichlet_cache[0] is not space:
            uh_bd, isBdDof = space.boundary_interpolate(gd=self._pde.dirichlet_bc,
                                                        threshold=self._pde.is_dirichlet_boundary(),
                                                        method='interp')
            self._dirichlet_cache = (space, uh_bd, isBdDof, bool(bm.any(uh_bd[:] != 0)))
        return self._dirichlet_cache[1:]
