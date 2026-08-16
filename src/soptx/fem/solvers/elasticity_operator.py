"""线弹性力学有限元分析器构造入口与 EA 算子门面模块.

提供串行与分布式线弹性有限元分析器的标准化构造工厂 (``build_serial_analyzer`` / ``build_distributed_analyzer``),
以及无矩阵 (Matrix-Free EA) 模式下的懒装配算子门面 ``ElasticityEAOperator`` 与一步正向求解接口 ``solve_ea_system``.

本模块导入时不依赖 ``mpi4py``: 仅在构造分布式分析器时延迟导入分布式实现,
确保无 MPI 环境下依然可以安全导入本模块执行单机串行分析.
"""

from __future__ import annotations

from typing import Any, Optional

from fealpy.functionspace import TensorFunctionSpace
from fealpy.typing import TensorLike

from soptx.fem.solvers.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.fem.solvers.matrix_free_solver import (
    PreparedLinearSystem,
    solve_matrix_free_system,
)


def _analyzer_arguments(
    space: TensorFunctionSpace,
    pde: Any,
    material: Any,
    degree: int = 1,
    operator_level: str = "ea",
) -> dict[str, Any]:
    """将物理问题、材料本构与有限元空间组合为标准分析器构造参数字典."""
    return {
        "disp_mesh": space.mesh,
        "pde": pde,
        "material": material,
        "space_degree": degree,
        "integration_order": degree + 3,
        "operator_level": operator_level,
        "tensor_space": space,
    }


def build_serial_analyzer(
    space: TensorFunctionSpace,
    pde: Any,
    material: Any,
    degree: int = 1,
    operator_level: str = "ea",
) -> LagrangeFEMAnalyzer:
    """构造不含跨进程通信的串行拉格朗日有限元分析器.

    供单 Rank 下的 EA (无矩阵单元装配) 或 FA (全矩阵装配) 求解与基线对比使用,
    直接建立在未经分布式切分的全局空间上.

    参数:
        space (TensorFunctionSpace): 全局向量有限元空间.
        pde (Any): 物理模型或制造解对象 (提供几何区域、体力荷载与边界条件).
        material (Any): 弹性力学本构材料模型 (提供材料常数与计算设备 device).
        degree (int, 可选): 有限元插值多项式阶数. 默认值为 1.
        operator_level (str, 可选): 算子装配级别, 可选 "ea" (无矩阵单元装配) 或 "fa" (全局稀疏矩阵全装配). 默认值为 "ea".

    返回:
        LagrangeFEMAnalyzer: 初始化的串行有限元分析器实例.
    """
    return LagrangeFEMAnalyzer(
        solve_method="scipy",
        **_analyzer_arguments(space, pde, material, degree, operator_level),
    )


def build_distributed_analyzer(
    space: TensorFunctionSpace,
    pde: Any,
    material: Any,
    degree: int = 1,
    operator_level: str = "ea",
    *,
    dof_comm: Any,
) -> Any:
    """构造重叠副本布局下的分布式线弹性分析器 (支持单 Rank 与多 Rank).

    本函数内部对 ``soptx.fem.solvers.matrix_free_analyzer.DistributedElasticityAnalyzer``
    采用延迟导入 (Lazy Import), 确保在未安装 MPI 运行时的环境中仍能正常导入本模块的串行能力.

    参数:
        space (TensorFunctionSpace): 局部子域上的向量有限元空间.
        pde (Any): 物理模型或制造解对象.
        material (Any): 弹性力学本构材料模型.
        degree (int, 可选): 有限元插值多项式阶数. 默认值为 1.
        operator_level (str, 可选): 算子装配级别 ("ea" 或 "fa"). 默认值为 "ea".
        dof_comm (EntityMPI): 自由度跨进程通信器 (关键字参数).

    返回:
        DistributedElasticityAnalyzer: 初始化的分布式有限元分析器实例.
    """
    from soptx.fem.solvers.matrix_free_analyzer import (
        DistributedElasticityAnalyzer,
    )

    return DistributedElasticityAnalyzer(
        dof_comm=dof_comm,
        **_analyzer_arguments(space, pde, material, degree, operator_level),
    )


class ElasticityEAOperator:
    """线弹性 Matrix-Free EA 刚度算子与边界条件的懒装配 (Lazy-Assembly) 缓存门面.

    本类封装了线弹性分析器的装配生命周期与算子代数接口, 承担三大职责:
        1. 懒加载与结果缓存: 首次访问刚度算子或载荷向量时才触发装配, 之后自动复用, 避免重复装配开销;
        2. Dirichlet 自由度掩码推导: 自动提取并缓存边界自由度索引掩码 ``boundary_dofs``;
        3. 算子矩阵乘接口 (``@``): 实现 ``__matmul__``, 使其实例可作为线性算子直接传入 Krylov 求解器.

    串行与分布式统一:
        - 当 ``dof_comm=None`` 时, 底层自动调用 ``LagrangeFEMAnalyzer`` 串行分析器;
        - 当提供 ``dof_comm`` 时, 底层自动调用 ``DistributedElasticityAnalyzer`` 分布式分析器.
    """

    def __init__(
        self,
        space: TensorFunctionSpace,
        pde: Any,
        material: Any,
        degree: int = 1,
        dof_comm: Any = None,
    ) -> None:
        """初始化 EA 算子门面.

        参数:
            space (TensorFunctionSpace): 局部或全局交错布局的向量有限元空间.
            pde (Any): 物理工程问题或制造解对象.
            material (Any): 弹性力学本构材料模型.
            degree (int, 可选): 有限元基函数多项式次数. 默认值为 1.
            dof_comm (EntityMPI | None, 可选): 自由度跨进程通信器. 串行传入 None. 默认值为 None.
        """
        self.space = space
        self.pde = pde
        self.degree = degree
        self.dof_comm = dof_comm

        self.analyzer: LagrangeFEMAnalyzer
        if dof_comm is None:
            self.analyzer = build_serial_analyzer(
                space, pde, material, degree, "ea"
            )
        else:
            self.analyzer = build_distributed_analyzer(
                space, pde, material, degree, "ea", dof_comm=dof_comm
            )

        self._system_operator: Any = None
        self._load_vector: Optional[TensorLike] = None
        self._prescribed: Optional[TensorLike] = None
        self._boundary_dofs: Optional[TensorLike] = None

    def assemble(self) -> tuple[Any, TensorLike]:
        """执行 EA 单元刚度矩阵与体力向量装配, 并施加 Dirichlet 边界条件对角投影.

        计算流程说明:
            1. 刚度算子装配 (EA 多态分发): 调用 ``assemble_stiff_matrix()`` 预计算并缓存各单元的小稠密
               刚度矩阵集 :math:`\\{\\mathbf{K}_e\\}`, 不拼接全局大矩阵;
            2. 体力荷载装配 (全通用实现): 调用 ``assemble_body_force_vector()`` 组装一维稠密载荷向量 :math:`\\mathbf{f}` (与 FA 共享);
            3. 边界条件施加 (EA 投影多态): 调用 ``apply_bc()`` 构建无矩阵对角投影算子 :math:`\\mathbf{A} = \\boldsymbol{\\Pi}_I \\mathbf{K} \\boldsymbol{\\Pi}_I + \\boldsymbol{\\Pi}_D`
               (多进程下自动包含 ``OverlapOperator`` 重叠同步包装);
            4. 边界几何信息提取 (全通用实现): 提取指定位移值 ``prescribed_solution`` 与边界自由度掩码 ``boundary_dofs``.

        返回:
            tuple[Any, TensorLike]: 包含以下两项的二元组:
                - operator (DirichletBCOperator | OverlapOperator): 施加边界条件后的系统刚度算子.
                - load (TensorLike): 施加边界条件修正后的等效右端载荷向量.
        """
        # 1. 刚度算子: 走 EA 多态路径, 预计算单元矩阵集并封装为无矩阵 BilinearForm
        stiff_matrix = self.analyzer.assemble_stiff_matrix()
        # 2. 体力荷载: 走通用线性型积分, 组装稠密右端项载荷向量
        body_force = self.analyzer.assemble_body_force_vector()
        # 3. 边界条件: 走 EA 对角投影算子包装 (分布式下自动提升为 OverlapOperator)
        operator, load = self.analyzer.apply_bc(stiff_matrix, body_force)

        self._system_operator = operator
        self._load_vector = load
        self._prescribed = self.analyzer.prescribed_solution
        self._boundary_dofs = self.space.is_boundary_dof(
            threshold=self.pde.is_dirichlet_boundary(),
            method="interp",
        )
        return operator, load

    def _ensure_assembled(self) -> None:
        """内部辅助方法: 确保刚度算子与载荷向量已装配."""
        if self._system_operator is None:
            self.assemble()

    @property
    def system_operator(self) -> Any:
        """施加 Dirichlet 边界条件后的系统刚度算子 (懒装配属性)."""
        self._ensure_assembled()
        return self._system_operator

    @property
    def load_vector(self) -> TensorLike:
        """施加 Dirichlet 边界条件后的等效右端载荷向量 (懒装配属性)."""
        self._ensure_assembled()
        assert self._load_vector is not None
        return self._load_vector

    @property
    def prescribed_solution(self) -> TensorLike:
        """Dirichlet 边界上的指定位移真解向量 (懒装配属性)."""
        self._ensure_assembled()
        assert self._prescribed is not None
        return self._prescribed

    @property
    def boundary_dofs(self) -> TensorLike:
        """Dirichlet 边界自由度的布尔掩码向量 (懒装配属性)."""
        self._ensure_assembled()
        assert self._boundary_dofs is not None
        return self._boundary_dofs

    def __matmul__(self, vector: TensorLike) -> TensorLike:
        """执行系统刚度算子与向量的矩阵-向量乘积运算 (A @ v)."""
        return self.system_operator @ vector


def solve_ea_system(
    space: TensorFunctionSpace,
    pde: Any,
    material: Any,
    *,
    degree: int = 1,
    dof_comm: Any = None,
    **solver_options: Any,
) -> tuple[TensorLike, dict[str, Any]]:
    """装配 Matrix-Free EA 刚度算子与体力右端项, 施加 Dirichlet 边界条件, 并调用加权共轭梯度法 (CG) 一键求解.

    本函数是线弹性力学无矩阵 (Matrix-Free EA) 有限元正向分析的高阶门面 (Facade) 接口.
    它将「分析器调度 ➔ 单元小刚度矩阵装配 ➔ Dirichlet 边界条件对角投影 ➔ 线性系统封装 ➔ 重叠加权 CG 迭代求解」
    整条流水线封装为一步闭环调用.

    串行与分布式无分支统一:
        - 串行求解: 当 ``dof_comm=None`` 时, 内部自动调度基于全局空间的串行分析器与普通 CG;
        - 分布式求解: 当传入 ``dof_comm`` (如 ``dist_space.dof_comm``) 时, 内部自动调度分布式分析器
          并执行重叠加权内积与跨进程界面同步归约 (``sync_add``), 两者调用代码完全一致.

    参数:
        space (TensorFunctionSpace): 局部或全局交错布局的向量拉格朗日有限元空间 (位移未知量空间).
        pde (Any): 物理工程问题或制造解对象 (提供几何区域、解析体力载荷 ``source`` 与 Dirichlet 边界真解).
        material (Any): 弹性力学本构材料模型 (如 ``IsotropicLinearElasticMaterial``, 提供拉梅常数与剪切模量).
        degree (int, 可选): 有限元基函数多项式次数. 默认值为 1 (P1 线性元).
        dof_comm (EntityMPI | None, 可选): 自由度跨进程通信器. 串行运行时传入 None, 分布式并行运行时传入子空间的 ``dof_comm``. 默认值为 None.
        **solver_options (Any): 透传给 Krylov 求解器 (``weighted_cg``) 的可选参数, 包括:
            - atol (float): 绝对残差容差 (默认采用 ``soptx.numerics.DEFAULT_ATOL``).
            - rtol (float): 相对残差容差 (默认采用 ``soptx.numerics.DEFAULT_RTOL``).
            - max_iter (int): 最大允许 CG 迭代步数 (默认采用 ``soptx.numerics.DEFAULT_MAX_ITERATIONS``).

    返回:
        tuple[TensorLike, dict[str, Any]]: 包含以下两项的二元组:
            - solution (TensorLike): 求解得到的局部节点位移自由度解向量 (一维张量, 对应局部自由度索引).
            - diagnostics (dict[str, Any]): 求解器收敛诊断字典, 包含:
                - "iterations" (int): 实际迭代步数.
                - "converged" (bool): 是否成功收敛到指定容差.
                - "residual" (float): 最终相对残差范数.

    说明:
        本函数适用于一次性正向求解场景 (如 demo 运行、收敛阶证据收集或单元测试);
        若处于需要在优化迭代中反复复用刚度算子进行 MatVec 计算的场景 (如拓扑优化每一轮灵敏度分析),
        建议直接实例化 ``ElasticityEAOperator`` 以避免重复装配.
    """

    operator_facade = ElasticityEAOperator(
        space,
        pde,
        material,
        degree=degree,
        dof_comm=dof_comm,
    )
    operator, load = operator_facade.assemble()
    system = PreparedLinearSystem(
        operator=operator,
        load=load,
        prescribed=operator_facade.prescribed_solution,
        boundary_dofs=operator_facade.boundary_dofs,
    )
    return solve_matrix_free_system(system, dof_comm, **solver_options)


__all__ = [
    "ElasticityEAOperator",
    "build_distributed_analyzer",
    "build_serial_analyzer",
    "solve_ea_system",
]
