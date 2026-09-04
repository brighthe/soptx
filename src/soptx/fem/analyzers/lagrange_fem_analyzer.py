from typing import Optional, Union, Literal, Dict

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import SimplexMesh, HomogeneousMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace, Function
from fealpy.fem import DirichletBCOperator, LinearForm
from fealpy.decorator import variantmethod
from fealpy.sparse import CSRTensor, COOTensor

from soptx.core import BaseLogged, timer
from soptx.fem.bilinear_form import BilinearForm
from soptx.protocols import (
    BodyForce,
    BoundaryTraction,
    DirichletElasticityProblem,
    LineTraction,
    MaterialInterpolation,
    PointForce,
)
from soptx.fem.integrators import (
    LagrangeBoundarySourceIntegrator,
    LinearElasticIntegrator,
    SourceIntegrator,
)
from soptx.fem.load_projection import project_nodal_loads
from soptx.materials import LinearElasticMaterial

class LagrangeFEMAnalyzer(BaseLogged):
    def __init__(self,
                disp_mesh: HomogeneousMesh,
                pde: DirichletElasticityProblem,
                material: LinearElasticMaterial,
                space_degree: int = 1,
                integration_order: int = 4,
                assembly_method: Literal['standard', 'voigt', 'fast'] = 'standard',
                operator_level: Literal['fa', 'ea'] = 'fa',
                solve_method: Literal['mumps', 'scipy', 'cg'] = 'mumps',
                solver_options: Optional[Dict] = None,
                tensor_space: Optional[TensorFunctionSpace] = None,
                dof_comm: Optional[object] = None,
                topopt_algorithm: Literal[None, 'density_based', 'level_set'] = None,
                interpolation_scheme: Optional[MaterialInterpolation] = None,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        """初始化拉格朗日有限元分析器

        Parameters
        ----------
        operator_level : 离散算子的存储与作用方式
        - 'fa' : 装配全局稀疏矩阵 (full assembly), 支持直接解法与伴随求解
        - 'ea' : 只保留单元矩阵 (element assembly), matvec 时 gather-作用-scatter,
                 不形成全局矩阵, 只能用迭代解法
        两者对应同一个离散算子 K = Σ_e R_e^T K_e R_e
        tensor_space : 外部构造的张量函数空间; 为 None 时由 disp_mesh 内部构造
        solve_method : 求解方式。直接法支持 'scipy' 与 'mumps', 都经
                       soptx.solvers.registry 分派到对应后端; 'mumps' 需要环境
                       装有 PyMUMPS 包 (pip install pymumps) 与系统 MUMPS 库。
                       'cg' 是迭代解法 (soptx.solvers.cg, 本仓库自有实现),
                       'fa' 与 'ea' 层级都可用; 'ea' 只能用它。
        solver_options : 迭代解法的默认参数, 支持 'maxiter'、'atol'、'rtol' 键。
                       solve_system 的 kwargs 优先于这里的默认值; 两者都未给出时
                       落到内部硬编码默认 (maxiter=5000, atol=rtol=1e-12)。
        interpolation_scheme : 密度插值方案。topopt_algorithm 取 'density_based'
                       或 'level_set' 时必须由调用方显式传入: fem (layer 2) 不构造
                       topology (layer 3) 的插值方案, 缺省即报错, 不静默回落到 SIMP
        dof_comm : 分布式重叠自由度通信器。本类不提供分布式求解, 传入后必须同时
                   覆盖 wrap_operator、reduce_load 和 solve_system, 否则
                   solve_system 会拒绝执行
        """

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        # 私有属性（建议通过属性访问器访问，不要直接修改）
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

        if operator_level not in ('fa', 'ea'):
            self._log_error(f"不支持的算子层级: {operator_level}, 可选 'fa' 或 'ea'")
        self._operator_level = operator_level

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

        # 注册算子层级变体
        self.assemble_stiff_matrix.set(self._operator_level)
        self.apply_bc.set(self._operator_level)

        # 缓存的矩阵和向量
        self._K = None
        self._F = None
        self._prescribed_solution = None  # 满足 Dirichlet 值、内部为零的基准向量
        self._csr_pattern = None  # 模式先行 (Pattern-First) 静态拓扑骨架缓存
        # 'ea' 下持有 const 积分子, 单元矩阵与 cell2dof 都在其中, 供对角提取复用
        self._const_integrator = None

        self._integrator = LinearElasticIntegrator(material=self._material,
                                                q=self._integration_order,
                                                method=self._assembly_method)
        self._integrator.keep_data(True)

        self._cached_ke0 = None
        self._cached_ke0_sub = None

        self._cached_stiffness_absolute = None # 绝对刚度 (带量纲)
        self._cached_stiffness_relative = None # 相对刚度 (无量纲)

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
    def material(self) -> LinearElasticMaterial:
        """获取当前的材料类"""
        return self._material
    
    @property
    def interpolation_scheme(self) -> MaterialInterpolation:
        """获取当前的材料插值方案"""
        return self._interpolation_scheme
    
    @property
    def assembly_method(self) -> str:
        """获取当前的组装方法"""
        return self._assembly_method

    @property
    def operator_level(self) -> str:
        """获取当前的算子层级 ('fa' 或 'ea')"""
        return self._operator_level

    @property
    def topopt_algorithm(self) -> Optional[str]:
        """获取当前的拓扑优化算法"""
        return self._topopt_algorithm
    
    @property
    def stiffness_matrix(self) -> Union[CSRTensor, COOTensor]:
        """获取当前的刚度矩阵"""
        return self._K
    
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

        边界自由度取给定值, 内部自由度为零。可用作迭代解法的初值, 以及边界误差
        的比较基准。apply_bc 之前为 None。
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

    
    ##############################################################################################
    # 核心方法
    ##############################################################################################

    def _update_density_coefficient(self,
                            rho_val: Optional[Union[Function, TensorLike]] = None,
                        ) -> None:
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
        """
        if self._topopt_algorithm is None:
            if rho_val is not None:
                self._log_warning("标准有限元分析模式下忽略相对密度 rho")

            # 标准有限元分析不做材料插值, 积分子直接使用实体材料本构
            relative_stiffness = None

            self._cached_stiffness_absolute = None
            self._cached_stiffness_relative = None
        
        elif self._topopt_algorithm == 'density_based':
            if rho_val is None:
                self._log_error("基于密度的拓扑优化算法需要提供相对密度 rho")

            material_params = self._interpolation_scheme.interpolate_material(
                                            material=self._material,
                                            rho_val=rho_val,
                                            integration_order=self._integration_order,
                                            displacement_mesh=self._mesh,
                                        )
            if isinstance(material_params, tuple):
                E_rho = material_params[0]
            else:
                E_rho = material_params
            E0 = self._material.youngs_modulus
            relative_stiffness = E_rho / E0

            self._cached_stiffness_absolute = E_rho               # 绝对刚度 (带量纲)
            self._cached_stiffness_relative = relative_stiffness  # 相对刚度 (无量纲)
        
        else:
            error_msg = f"不支持的拓扑优化算法: {self._topopt_algorithm}"
            self._log_error(error_msg)

        # TODO 这里的 coef 也和材料有关, 可能需要进一步处理,
        # TODO coef 是应该在 LinearElasticIntegrator 中, 还是在 MaterialInterpolationScheme 中处理 ?
        # 更新密度系数
        self._integrator.coef = relative_stiffness

    @variantmethod('fa')
    def assemble_stiff_matrix(self,
                            rho_val: Optional[Union[Function, TensorLike]] = None,
                            enable_timing: bool = False,
                        ) -> Union[CSRTensor, COOTensor]:
        """装配全局刚度矩阵 (full assembly)

        rho_val 的形状约定见 `_update_density_coefficient`
        """
        t = None
        if enable_timing:
            t = timer(f"双线性型组装内部")
            next(t)

        self._update_density_coefficient(rho_val)

        if enable_timing:
            t.send('预备')

        bform = BilinearForm(self._tensor_space, pattern=self._csr_pattern)
        bform.add_integrator(self._integrator)

        K = bform.assembly(format='csr')
        self._csr_pattern = bform.pattern

        self._K = K

        if enable_timing:
            t.send('组装')
            t.send(None)

        return K

    @assemble_stiff_matrix.register('ea')
    def assemble_stiff_matrix(self,
                            rho_val: Optional[Union[Function, TensorLike]] = None,
                            enable_timing: bool = False,
                        ) -> BilinearForm:
        """构造单元级刚度算子 (element assembly)

        预先算出并缓存单元矩阵 {K_e}, 但不求和成全局矩阵。返回的 BilinearForm
        未调用 assembly, 其 `@` 运算走 gather-单元作用-scatter-add, 与 'fa'
        对应同一个离散算子。
        """
        t = None
        if enable_timing:
            t = timer(f"单元算子构造内部")
            next(t)

        self._update_density_coefficient(rho_val)

        if enable_timing:
            t.send('预备')

        # const 预先算出单元矩阵, 之后每次 matvec 不再重复积分
        const_integrator = self._integrator.const(self._tensor_space)
        self._const_integrator = const_integrator

        bform = BilinearForm(self._tensor_space)
        bform.dtype = bm.float64
        bform.add_integrator(const_integrator)

        self._K = bform

        if enable_timing:
            t.send('单元矩阵缓存')
            t.send(None)

        return bform


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
        F = self._tensor_space.function()
        for load in self._pde.loads():
            if not isinstance(load, BodyForce):
                continue
            integrator = SourceIntegrator(
                source=load.body_force,
                q=self._integration_order,
            )
            lform = LinearForm(self._tensor_space)
            lform.add_integrator(integrator)
            F = F + lform.assembly(format='dense')
        return F

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
        的调用方」(子结构缩聚等) 与「求解全尺度系统的调用方」看到的是同一个载荷。

        - 'mixed'     : 装配点力、线载荷与边界牵引 (以及伴随载荷列);
        - 'dirichlet' : 全边界都是本质边界, 自然边界项不进入右端。此时 pde 若仍
                        给出非体力载荷, 说明边界类型与载荷契约矛盾, 显式报错而不是
                        静默丢弃;
        - 其他        : 尚未定义装配语义, 报错。
        """
        boundary_type = self._pde.boundary_type

        if boundary_type == 'mixed':
            return self._assemble_non_body_loads(adjoint)

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
        """组装点力、线载荷和边界牵引，并保留现有伴随载荷入口."""
        space_uh = self._tensor_space
        F_physical = space_uh.function()
        nodal_loads = []

        for load in self._pde.loads():
            if isinstance(load, BodyForce):
                continue
            if isinstance(load, (PointForce, LineTraction)):
                nodal_loads.append(load)
                continue
            if isinstance(load, BoundaryTraction):
                integrator = LagrangeBoundarySourceIntegrator(
                    source=load.traction,
                    q=self._integration_order,
                    threshold=load.is_load_boundary,
                )
                lform = LinearForm(space_uh)
                lform.add_integrator(integrator)
                F_physical = F_physical + lform.assembly(format='dense')
                continue
            raise TypeError(
                "LFEM 不支持载荷对象 "
                f"{type(load).__name__}; 请提供已定义装配语义的 Load."
            )

        if nodal_loads:
            node_major = project_nodal_loads(
                nodal_loads,
                space_uh.interpolation_points(),
                self._GD,
                degree=self._scalar_space.p,
            )
            if space_uh.dof_priority:
                node_values = bm.reshape(node_major, (-1, self._GD))
                nodal_vector = bm.reshape(
                    bm.transpose(node_values, (1, 0)),
                    (-1,),
                )
            else:
                nodal_vector = node_major
            F_physical = F_physical + nodal_vector

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
        的全局矩阵, 没有可供插入重叠归约的位置。因此 'fa' 只能在单 rank 上用——
        多 rank 若放行, 各 rank 会在自己的局部矩阵上求解, 不报错但结果是错的。
        单 rank 下重叠归约本身是恒等操作, 带 dof_comm 也是安全的。
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
        gd_uh = self._pde.dirichlet_bc
        threshold_uh = self._pde.is_dirichlet_boundary()

        uh_bd, isBdDof = space_uh.boundary_interpolate(
                                                    gd=gd_uh,
                                                    threshold=threshold_uh,
                                                    method='interp'
                                                )
        self._prescribed_solution = uh_bd

        if adjoint:
            uh_bd = bm.repeat(uh_bd.reshape(-1, 1), 2, axis=1)
            #? matmul 函数下 K 必须是 COO 格式, 不能是 CSR 格式, 否则 GPU 下 device_put 函数会出错
            F = F - K.tocoo().matmul(uh_bd[:])
            F = bm.set_at(F, (isBdDof, slice(None)), uh_bd[isBdDof, :])
        else:
            F = F - K.tocoo().matmul(uh_bd[:])
            F = bm.set_at(F, isBdDof, uh_bd[isBdDof])

        K = self._apply_matrix(K, isDDof=isBdDof)

        return K, F

    @apply_bc.register('ea')
    def apply_bc(self,
                K: BilinearForm,
                F: TensorLike,
                adjoint: bool = False
            ) -> tuple[DirichletBCOperator, TensorLike]:
        """在单元级算子上施加边界条件

        不改写任何矩阵, 而是把算子包进 DirichletBCOperator: matvec 时先把
        Dirichlet 自由度置零, 作用后再还原, 等价于 'fa' 的对称消元系统。
        """
        if adjoint:
            self._log_error(
                "operator_level='ea' 不支持伴随双列右端项, 请改用 operator_level='fa'"
            )

        F_non_body = self._non_body_loads_by_boundary_type(adjoint=False)
        if F_non_body is not None:
            F = F + F_non_body

        # 载荷必须在施加边界条件之前完成跨 rank 归约
        F = self.reduce_load(F)
        self._F = F

        space_uh = self._tensor_space
        threshold_uh = self._pde.is_dirichlet_boundary()
        isBdDof = space_uh.is_boundary_dof(threshold=threshold_uh, method='interp')

        operator = DirichletBCOperator(self.wrap_operator(K),
                                    gd=self._pde.dirichlet_bc,
                                    isDDof=isBdDof)

        # 边界自由度取给定值, 内部自由度取零, 作为消去边界贡献的基准向量
        uh_bd = operator.init_solution(dtype=bm.float64)
        uh_bd = bm.set_at(uh_bd, ~isBdDof, 0.0)
        F = operator.apply(F, uh_bd)

        self._prescribed_solution = uh_bd

        return operator, F


    ##############################################################################################
    # 分布式扩展点 (串行下均为恒等操作)
    ##############################################################################################

    def wrap_operator(self, form: BilinearForm):
        """在施加边界条件之前对单元级算子做一层包装

        串行下原样返回。分布式实现覆盖本方法, 返回一个把 matvec 结果在重叠自由度
        上求和的包装 (示例中的 `distributed.OverlapOperator`), 之后的边界条件处理
        和求解都不需要知道它的存在。

        Note
        ----
        包装必须发生在 DirichletBCOperator 之前: 先跨 rank 组装出完整的算子作用,
        再在其上消去 Dirichlet 自由度。顺序反过来会把边界行的置换也带进通信。
        """
        return form

    def reduce_load(self, F: TensorLike) -> TensorLike:
        """把按自由度分布的右端项在重叠自由度上归约

        串行下原样返回。分布式实现覆盖本方法, 通常是 `dof_comm.sync_add(F)`。
        必须在施加 Dirichlet 边界条件之前调用, 否则边界行会被重复累加。
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
            if enable_timing:
                t.send('双线性型组装')

            F0 = self.assemble_body_force_vector()
            if enable_timing:
                t.send('线性型组装')

            K, F = self.apply_bc(K0, F0)
            if enable_timing:
                t.send('边界条件处理')

            uh = self._tensor_space.function()

        # 'ea' 从刚刚由 apply_bc 得到的 Dirichlet 基准向量起步; 显式传入而非让
        # solve_system 去读实例状态
        if self._operator_level == 'ea':
            kwargs.setdefault('x0', self._prescribed_solution)

        _, solver_info = self.solve_system(K, F, uh, **kwargs)

        if enable_timing:
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
        """求解伴随方程 K @ λ = rhs"""
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
        K = self._apply_matrix(K0, isDDof=isBdDof)
        
        # 初始化结果并求解
        adjoint_lambda = bm.zeros_like(rhs_bc)
        self.solve_system(K, rhs_bc, adjoint_lambda, **kwargs)

        return adjoint_lambda


    def _as_iterative_operator(self, K):
        """把刚度算子转成迭代解法可以直接作用的形式

        'ea' 下 K 本身就支持 @ 运算; 'fa' 下 PyTorch 后端需要绕开 FEALPy 的
        CSRTensor, 其余后端直接用 COO。
        """
        if self._operator_level == 'ea':
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

        return K.tocoo()

    def assemble_operator_diagonal(self, K) -> TensorLike:
        """取已施加 Dirichlet 条件的系统算子对角, 供 Jacobi 类预条件使用

        'ea' 下 apply_bc 给出的是 A = Pi_I K Pi_I + Pi_D, 故对角在 Dirichlet
        自由度上恒为 1, 其余为 diag(K): 逐单元取小矩阵对角再按 cell2dof 散加,
        与 matvec 共享同一套 gather/scatter 语义, 代价是一次批量取对角加一次
        散加, 不作用算子。
        'fa' 下 apply_bc 已完成对称消元, 直接从稀疏矩阵读对角。

        Parameters
        ----------
        K : 'fa' 下为全局稀疏矩阵, 'ea' 下为 DirichletBCOperator

        Returns
        -------
        diag : (TGDOF, ) 的算子对角, SPD 系统下逐元严格为正
        """
        if self._operator_level == 'ea':
            if self._const_integrator is None:
                self._log_error(
                    "取 'ea' 算子对角前必须先调用 assemble_stiff_matrix 缓存单元矩阵"
                )

            KE = self._const_integrator.value                      # (NC, ldof, ldof)
            cell2dof = self._const_integrator.to_global_dof(self._tensor_space)
            diag_e = bm.einsum('cii -> ci', KE)                     # (NC, ldof)

            gdof = self._tensor_space.number_of_global_dofs()
            diag = bm.zeros((gdof, ), **bm.context(KE))
            diag = bm.index_add(diag,
                                bm.reshape(cell2dof, (-1, )),
                                bm.reshape(diag_e, (-1, )))

            # 重叠自由度需跨 rank 求和; 对角与右端项同为可加的自由度向量, 复用
            # 同一个归约扩展点 (串行下恒等)
            diag = self.reduce_load(diag)

            return bm.set_at(diag, K.is_boundary_dof, 1.0)

        K_coo = K.tocoo()
        row, col = K_coo.row, K_coo.col
        values = K_coo.values
        on_diagonal = (row == col)

        diag = bm.zeros((K.shape[0], ), **bm.context(values))

        return bm.index_add(diag, row[on_diagonal], values[on_diagonal])


    def _build_solver(self, solver_type, K, **kwargs):
        """按名字造出求解器, 并选定它要绑定的算子

        名字到类的分派由 soptx.solvers.registry 完成, 本方法只剩两件本地的事:
        各后端读哪些选项, 以及 'fa'/'ea' 下算子形态的差异。

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
            if precond in ('jacobi', 'diagonal'):
                from soptx.solvers import DiagonalPreconditioner

                M = DiagonalPreconditioner(self.assemble_operator_diagonal(K))
                # 判据范数与下游口径对齐: cg 默认在 natural 范数
                # sqrt(r^T M^-1 r) 下停机, 而本方法返回的 relres 是 2-范数,
                # Jacobi 的 diag^-1 可达 1e6 量级, 两个口径能差几个数量级。
                # 显式选 unpreconditioned 让停机判据也用 ||r||_2, 不多做 matvec
                norm_type = 'unpreconditioned'
                # 与范数选择正交: 递推残差在长迭代下会漂移, 周期性用 b - A x
                # 校正。撤掉它需要单独的数值证据, 故与 M 绑定保持开启
                if residual_refresh <= 0:
                    residual_refresh = 50
            elif precond is not None:
                self._log_error(f"未知的预条件子类型: {precond}")

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
            # 大致减半。默认保持 0, 由调用方显式开启
            mumps_sym = int(kwargs.get('sym', 0))

            return create('mumps', sym=mumps_sym), K, {'sym': mumps_sym}, None

        return create(solver_type), K, {}, None

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
        本方法不读取任何由 apply_bc 留下的状态。迭代解法的初值必须由调用方通过
        kwargs['x0'] 显式给出——对 'ea' 而言通常就是 apply_bc 产生的
        prescribed_solution, 它已满足 Dirichlet 值。

        直接法在 'ea' 下不可用一事不在此处硬编码判断: DirectSolver 声明自己需要
        显式矩阵, matrix-free 算子给不出, setup 时即抛 OperatorCapabilityError,
        本方法只负责把它转成与其他使用错误一致的 RuntimeError。

        分解不跨调用复用: 求解完即释放, MUMPS 上下文的生命周期与改造前逐次
        建销一致。状态解与伴随解共用一个分解要改 analyzer 的状态管理, 单独一步做。

        这是分布式求解唯一的注入点: 并行只需在此处把 fealpy 的 cg 换成带
        overlap 加权内积的版本, 上层的组装与边界条件处理不受影响。覆盖本方法的
        实现负责自行处理 dof_comm。
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
                    f"'{solver_type}': {exc} 请改用 solver='cg'"
                )

            out[:], raw = solver.solve(F[:], kwargs.get('x0', None))
        finally:
            # 直接法持有 SuperLU 分解或 MUMPS 上下文, 用完即释放
            close = getattr(solver, 'close', None)
            if close is not None:
                close()

        info = {'name': solver_type, **extra,
                'niter': int(raw['niter']),
                'relres': float(raw['relres']),
                'converged': bool(raw['converged'])}

        if tol is not None:
            # 收敛与否以求解器自己的退出原因为准, 不在这里重判。原来那段用
            # max(atol, rtol * ||F||_2) 重算是错的: solve_system 支持传 x0,
            # 热启动时 rtol 的参照量是 ||r0|| 而非 ||F||。判据口径的对齐已在
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
        """计算实体材料的刚度矩阵"""
        lea = LinearElasticIntegrator(material=self._material,
                            coef=None,
                            q=self._integration_order,
                            method='standard')
        ke0 = lea.assembly(space=self.tensor_space)

        self._cached_ke0 = ke0

        return ke0
    
    def compute_sub_element_stiffness_matrix(self) -> TensorLike:
        """计算各子单元对位移单元刚度矩阵的贡献 (单位弹性模量 E=1)

        Returns
        -------
        ke0_sub : TensorLike, shape (NC, n_sub, TLDOF, TLDOF)
            满足: K_e = Σ_s E(ρ_{e,s}) · ke0_sub[c, s]
            因此: ∂K_e/∂ρ_{e,i} = E'(ρ_{e,i}) · ke0_sub[c, i]
        """
        space    = self.tensor_space
        s_space  = space.scalar_space
        mesh_u   = space.mesh
        GD       = mesh_u.geo_dimension()
        n_sub    = self._interpolation_scheme.n_sub
        NC       = mesh_u.number_of_cells()
        LDOF     = s_space.number_of_local_dofs()

        # --- 复用 voigt_multiresolution 前半段: 积分点、gphi、detJ ---
        if 4 <= n_sub <= 9:
            q = 3
        elif n_sub >= 16:
            q = 2
        else:
            q = s_space.p + 3

        qf_e = mesh_u.quadrature_formula(q)
        bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()  # ws_e: (NQ,)

        from soptx.fem.utils import map_bcs_to_sub_elements
        bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
        bcs_eg_x, bcs_eg_y = bcs_eg[0], bcs_eg[1]

        NQ = ws_e.shape[0]
        gphi_eg = bm.zeros((NC, n_sub, NQ, LDOF, GD))
        detJ_eg = bm.zeros((NC, n_sub, NQ))

        for s_idx in range(n_sub):
            sub_bcs = (bcs_eg_x[s_idx], bcs_eg_y[s_idx])
            gphi_eg[:, s_idx] = s_space.grad_basis(sub_bcs, variable='x')  # (NC, NQ, LDOF, GD)
            J_sub = mesh_u.entity_view('cell').jacobi_matrix(sub_bcs)           # (NC, NQ, GD, GD)
            detJ_eg[:, s_idx] = bm.abs(bm.linalg.det(J_sub))              # (NC, NQ)

        # --- 计算 B 矩阵, 与 voigt_multiresolution 完全一致 ---
        from soptx.fem.utils import (reshape_multiresolution_data,
                                        reshape_multiresolution_data_inverse)
        B_eg = reshape_multiresolution_data_inverse(
                    mesh_u,
                    self._material.strain_matrix(
                        dof_priority=space.dof_priority,
                        gphi=reshape_multiresolution_data(mesh_u, gphi_eg)  # (NC*n_sub, NQ, NS, TLDOF)
                    ),
                    n_sub=n_sub
            )  # (NC, n_sub, NQ, NS, TLDOF)

        # --- 核心改动: coef=1, 保留 n_sub 维度 ---
        J_g = 1.0 / n_sub
        D0  = self._material.elastic_matrix()[0, 0]  # (NS, NS)

        # voigt assembly:  'q, cnq, cnqki, cnkl, cnqlj -> cij'  (消掉 n)
        # 此处:            'q, cnq, cnqki,   kl, cnqlj -> cnij' (保留 n)
        ke0_sub = J_g * bm.einsum('q, cnq, cnqki, kl, cnqlj -> cnij',
                                ws_e, detJ_eg, B_eg, D0, B_eg)
        # shape: (NC, n_sub, TLDOF, TLDOF)

        self._cached_ke0_sub = ke0_sub

        return ke0_sub

    def compute_stiffness_matrix_derivative(self, rho_val: Union[TensorLike, Function]) -> TensorLike:
        """计算局部刚度矩阵关于物理密度的导数 (灵敏度)"""
        density_location = self._interpolation_scheme.density_location

        material_derivs = self._interpolation_scheme.interpolate_material_derivative(
                                                material=self._material, 
                                                rho_val=rho_val,
                                                integration_order=self._integration_order,
                                            ) 
        if isinstance(material_derivs, tuple):
            dE_rho = material_derivs[0]
        else:
            dE_rho = material_derivs
        
        if density_location in ['element']:
            # rho_val.shape = (NC, )
            diff_coef_element = dE_rho / self._material.youngs_modulus # (NC, )

            if self._cached_ke0 is None:
                self.compute_solid_stiffness_matrix()

            ke0 = self._cached_ke0

            diff_ke = bm.einsum('c, cij -> cij', diff_coef_element, ke0) # (NC, TLDOF, TLDOF)
 
            return diff_ke
        
        elif density_location in ['element_multiresolution']:
            # rho_val.shape = (NC, n_sub)
            diff_coef_sub_element = dE_rho / self._material.youngs_modulus # (NC, n_sub)

            mesh_u = self._mesh
            s_space_u = self._scalar_space
            q = self._integration_order
            NC, n_sub = rho_val.shape
            GD = mesh_u.geo_dimension()

            # 计算位移单元 (父参考单元) 高斯积分点处的重心坐标
            qf_e = mesh_u.quadrature_formula(q)
            # bcs_e.shape = ( (NQ_x, GD), (NQ_y, GD) ), ws_e.shape = (NQ, )
            bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()
            NQ = ws_e.shape[0]

            # 把位移单元高斯积分点处的重心坐标映射到子密度单元 (子参考单元) 高斯积分点处的重心坐标 (仍表达在位移单元中)
            from soptx.fem.utils import map_bcs_to_sub_elements
            # bcs_eg.shape = ( (n_sub, NQ_x, GD), (n_sub, NQ_y, GD) ), ws_e.shape = (NQ, )
            bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
            bcs_eg_x, bcs_eg_y = bcs_eg

            # 计算子密度单元内高斯积分点处的基函数梯度和 jacobi 矩阵
            LDOF = s_space_u.number_of_local_dofs()
            gphi_eg = bm.zeros((NC, n_sub, NQ, LDOF, GD)) # (NC, n_sub, NQ, LDOF, GD)
            detJ_eg = None

            if isinstance(mesh_u, SimplexMesh):
                for s_idx in range(n_sub):
                    sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ_x, GD), (NQ_y, GD))
                    gphi_sub = s_space_u.grad_basis(sub_bcs, variable='x')    # (NC, NQ, LDOF, GD)
                    gphi_eg[:, s_idx, :, :, :] = gphi_sub

            else:
                detJ_eg = bm.zeros((NC, n_sub, NQ)) # (NC, n_sub, NQ)
                for s_idx in range(n_sub):
                    sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ_x, GD), (NQ_y, GD))
                    gphi_sub = s_space_u.grad_basis(sub_bcs, variable='x') # (NC, NQ, LDOF, GD)
                    J_sub = mesh_u.entity_view('cell').jacobi_matrix(sub_bcs) # (NC, NQ, GD, GD)
                    detJ_sub = bm.abs(bm.linalg.det(J_sub)) # (NC, NQ)
                    gphi_eg[:, s_idx, :, :, :] = gphi_sub
                    detJ_eg[:, s_idx, :] = detJ_sub

            # 计算 B 矩阵
            from soptx.fem.utils import reshape_multiresolution_data, reshape_multiresolution_data_inverse
            gphi_eg_reshaped = reshape_multiresolution_data(mesh=mesh_u, data=gphi_eg) # (NC*n_sub, NQ, NS, TLDOF)
            B_eg_reshaped = self._material.strain_matrix(
                                                dof_priority=self._tensor_space.dof_priority,
                                                gphi=gphi_eg_reshaped
                                            ) # (NC*n_sub, NQ, NS, TLDOF)
            B_eg = reshape_multiresolution_data_inverse(mesh=mesh_u, data_flat=B_eg_reshaped, n_sub=n_sub) # (NC, n_sub, NQ, NS, TLDOF)

            # 位移单元 → 子密度单元的缩放
            J_g = 1 / n_sub

            # 计算 D 矩阵的导数
            D0 = self._material.elastic_matrix()[0, 0] # (NS, NS)
            diff_D_g = bm.einsum('kl, cn -> cnkl', D0, diff_coef_sub_element) # (NC, n_sub, NS, NS)

            # 数值积分
            # diff_ke - (NC, n_sub, TLDOF, TLDOF)
            if isinstance(mesh_u, SimplexMesh):
                cm = mesh_u.entity_measure('cell')
                cm_eg = bm.tile(cm.reshape(NC, 1), (1, n_sub)) / n_sub # (NC, n_sub)
                diff_ke = J_g * bm.einsum('q, cn, cnqki, cnkl, cnqlj -> cnij',
                                    ws_e, cm_eg, B_eg, diff_D_g, B_eg)
            else:
                diff_ke = J_g * bm.einsum('q, cnq, cnqki, cnkl, cnqlj -> cnij',
                                    ws_e, detJ_eg, B_eg, diff_D_g, B_eg)

            return diff_ke
        
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
        integration_order : 积分阶次，默认使用分析器的积分阶次
        
        Returns
        -------
        B : 应变-位移矩阵
            - 单分辨率: (NC, NQ, NS, TLDOF)
            - 多分辨率: (NC, n_sub, NQ, NS, TLDOF)

        Note
        ----
        B 只取决于位移离散和积分点位置, 与密度自由度住在哪里无关; 唯一的区别是
        多分辨率要在子密度单元的积分点上求值, 因此这里只按是否多分辨率分支。
        """
        if integration_order is None:
            integration_order = self._integration_order

        density_location = self._interpolation_scheme.density_location

        if density_location in ['element_multiresolution']:
            from soptx.fem.utils import (calculate_multiresolution_gphi_eg,
                                            reshape_multiresolution_data_inverse)
            n_sub = self._interpolation_scheme.n_sub
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
            积分阶次. 默认为 1 (中心点积分), 这对 Q4 单元足以避免棋盘格效应.

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
            integration_order = 1 # 默认使用中心点积分

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

    def _apply_matrix(self, matrix, isDDof, check=True):
        """Apply Dirichlet boundary condition to left-hand-size matrix only.

        Parameters:
            matrix (SparseTensor): The original left-hand-size sparse matrix\
                of the linear system.
            check (bool, optional): Whether to check the matrix. Defaults to True.

        Returns:
            SparseTensor: New adjusted left-hand-size matrix.
        """
        A = matrix
        kwargs = A.values_context()
        if isinstance(A, COOTensor):
            indices = A.indices
            remove_flag = bm.logical_or(
                isDDof[indices[0, :]], isDDof[indices[1, :]]
            )
            retain_flag = bm.logical_not(remove_flag)
            new_indices = indices[:, retain_flag]
            new_values = A.values[..., retain_flag]
            A = COOTensor(new_indices, new_values, A.sparse_shape)

            index = bm.nonzero(isDDof)[0]
            shape = new_values.shape[:-1] + (len(index), )
            one_values = bm.ones(shape, **kwargs)
            one_indices = bm.stack([index, index], axis=0)
            A1 = COOTensor(one_indices, one_values, A.sparse_shape)
            A = A.add(A1).coalesce()

        elif isinstance(A, CSRTensor):
            isIDof = bm.logical_not(isDDof)
            crow = A.crow
            col = A.col
            indices_context = bm.context(col)
            ZERO = bm.array([0], **indices_context)

            nnz_per_row = crow[1:] - crow[:-1]
            remain_flag = bm.repeat(isIDof, nnz_per_row) & isIDof[col] # 保留行列均为内部自由度的非零元素
            rm_cumsum = bm.concat([ZERO, bm.cumsum(remain_flag, axis=0)], axis=0) # 被保留的非零元素数量累积
            nnz_per_row = rm_cumsum[crow[1:]] - rm_cumsum[crow[:-1]] + isDDof # 计算每行的非零元素数量

            new_crow = bm.cumsum(bm.concat([ZERO, nnz_per_row], axis=0), axis=0)

            NNZ = new_crow[-1]
            non_diag = bm.ones((NNZ,), dtype=bm.bool, device=bm.get_device(isDDof)) # Field: non-zero elements
            loc_flag = bm.logical_and(new_crow[:-1] < NNZ, isDDof)
            non_diag = bm.set_at(non_diag, new_crow[:-1][loc_flag], False)

            bd_rows = bm.where(loc_flag)[0]
            new_col = bm.empty((NNZ,), **indices_context)
            new_col = bm.set_at(new_col, new_crow[:-1][loc_flag], bd_rows)
            new_col = bm.set_at(new_col, non_diag, col[remain_flag])

            new_values = bm.empty((NNZ,), **kwargs)
            new_values = bm.set_at(new_values, new_crow[:-1][loc_flag], 1.)
            new_values = bm.set_at(new_values, non_diag, A.values[remain_flag])

            A = CSRTensor(new_crow, new_col, new_values, A.sparse_shape)

        return A
