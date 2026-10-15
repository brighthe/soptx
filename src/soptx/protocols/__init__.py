"""公共结构化协议.

本模块描述 Problem、有限元分析器、拓扑优化层与线性求解层之间的接口边界, 不
规定具体实现. 物理载荷的语义及作用的几何实体由 :mod:`soptx.protocols.loads` 定义,
线性算子的最小语义由 :mod:`soptx.protocols.operators` 定义; 本模块只规定
Problem 如何提供这些载荷, 以及各消费方可以依赖哪些成员.

协议覆盖三个依赖方向:

1. ``Problem -> analyzer``: ``ElasticityProblem`` 及其细化协议规定分析器可读取
   的问题数据.
2. ``Analyzer -> topology``: ``AnalysisStage`` 规定拓扑优化目标与约束可调用的
   公共分析阶段接口.
3. ``Analyzer -> solvers``: ``SupportsMatmul`` 规定线性求解层可作用的算子形状,
   FA 的稀疏矩阵与 EA 的 matrix-free 算子都按它标注.

``runtime_checkable`` 只检查成员是否存在, 不检查方法签名. 分析器新增对 Problem
成员的依赖时, 必须同步更新相应协议; ``test_problem_protocol_conformance.py``
负责检查这项约束.

Problem 接口分层
----------------
工程基准问题类按"核心层 + 可选能力层"组织. 一个类实现哪些层由其载荷类型与用途
决定, 不实现的层不提供占位成员; 同一层内的成员同名同义. 层 1 至层 3 由下列协议
强制, 层 1 中的材料属性、层 4 与层 5 是约定, 由消费它们的算例层依赖.

1. 核心层 (所有问题): ``dimension``, ``domain``, ``E``, ``nu``, ``plane_type``,
   ``P``, ``loads()``; 边界与角点的几何判定用绝对容差
   ``soptx.core.numerics.GEOMETRIC_TOL``, 类内以 ``_eps`` 持有. 对应
   ``ElasticityProblem``.
2. Lagrange 层: ``boundary_type``, ``dirichlet_bc``, ``is_dirichlet_boundary()``
   及逐分量的 ``is_dirichlet_boundary_dof_*``. 对应 ``DirichletElasticityProblem``.
3. Hu--Zhang 层 (载荷可表示为面牵引的问题): ``is_displacement_boundary``,
   ``is_traction_boundary``, 可选 ``displacement_bc`` 与 ``is_symmetry_boundary``. 对应
   ``MixedBoundaryElasticityProblem``. 集中力与线载荷不能按应力迹强施加, 这类问题
   不实现本层.
4. 局部牵引描述层 (载荷为边界上一段局部牵引的问题): ``load_width``,
   ``traction_patch``, ``traction_axis``, ``traction_level``, ``traction_intensity``,
   ``uses_step_traction``, 以及构造参数 ``traction`` (注入 P1 迹载荷等受控牵引).
   未注入 ``traction`` 时按整边选取施加牵引, 载荷区端点须是网格节点, 由分析器构造时
   经 ``soptx.fem.check_traction_patch_alignment`` 检查.
5. 应力奇点层 (用于应力约束的问题): ``traction_patch_endpoints`` (载荷侧),
   ``support_corner_points`` (支承侧 Dirichlet--Neumann 角点).

数值处置不进入问题类: 被动实体区掩码经 ``Filter(passive_mask=...)`` 传入; Hu--Zhang
角点松弛的几何角点由网格检出 (``mesh.boundary_corner_index``), 有意只松弛某个子集时
经 ``HuZhangMFEMAnalyzer(relaxed_corners=...)`` 传入. 物理非设计区 (如
简支桥桥面) 是例外, 由 ``get_passive_element_mask(mesh)`` 给出. 三维问题由二维
问题沿第三方向拉伸得到时, 支承与载荷一并拉伸, 只补消除新增刚体运动所必需的约束.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

# ``TensorLike`` 仅用于类型注解.
if TYPE_CHECKING:
    from soptx.typing import TensorLike

from .loads import (
    BodyForce,
    BoundaryTraction,
    LineTraction,
    Load,
    LoadKind,
    LoadProvider,
    PointForce,
)
from .operators import SupportsMatmul


@runtime_checkable
class ElasticityProblem(Protocol):
    """所有弹性分析器共享的基础 Problem 契约.

    Problem 描述方程、区域、边界数据和物理载荷, 不创建离散网格, 也不持有
    FunctionSpace 或 Material.
    """

    dimension: int

    @property
    def domain(self) -> Sequence[float]:
        """轴对齐盒形计算区域的边界值.

        长度为 ``2 * dimension``, 按 ``(x_min, x_max, y_min, y_max[, z_min, z_max])``
        排列; 消费方以 ``domain[0::2]`` 作为物理域原点.
        """
        ...

    def loads(self) -> Sequence[Load]:
        """返回全部物理外载荷.

        Returns
        -------
        Sequence[Load]
            满足 ``Load`` 契约的载荷对象, 其 ``dimension`` 必须与 Problem 的
            ``dimension`` 一致. 分析器按 ``BodyForce``, ``BoundaryTraction``,
            ``PointForce``, ``LineTraction`` 分派装配.
        """
        ...


@runtime_checkable
class DirichletElasticityProblem(ElasticityProblem, Protocol):
    """``LagrangeFEMAnalyzer`` 消费的 Problem 契约.

    物理外载荷统一由 ``loads()`` 提供.
    """

    @property
    def boundary_type(self) -> str:
        """边界类型, 取 ``'mixed'`` 或 ``'dirichlet'``.

        ``'mixed'`` 时 ``LagrangeFEMAnalyzer`` 装配 ``loads()`` 中的非体力载荷;
        ``'dirichlet'`` 表示全边界为本质边界, 此时 ``loads()`` 只能给出
        ``BodyForce``, 否则分析器报错. 其他取值同样报错.
        """
        ...

    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """在给定点上计算 Dirichlet 位移值.

        Parameters
        ----------
        points : TensorLike
            形状 ``(..., GD)`` 的物理坐标.

        Returns
        -------
        TensorLike
            与 ``points`` 同形状的位移值; 由 ``boundary_interpolate`` 只在
            ``is_dirichlet_boundary()`` 标记的分量上取用.
        """
        ...

    def is_dirichlet_boundary(self) -> tuple[Callable[..., TensorLike], ...]:
        """返回按位移分量划分的 Dirichlet 边界判定函数.

        Returns
        -------
        tuple of Callable
            长度为 ``dimension``, 第 ``i`` 项判定第 ``i`` 个位移分量是否受约束:
            输入形状 ``(N, GD)`` 的坐标, 返回形状 ``(N,)`` 的布尔掩码. 子结构
            投影路径允许某项为 ``None``, 表示该分量不受约束.
        """
        ...


@runtime_checkable
class MixedBoundaryElasticityProblem(ElasticityProblem, Protocol):
    """``HuZhangMFEMAnalyzer`` 消费的 Problem 契约.

    Hu--Zhang 混合形式中, 位移边界数据按自然边界条件弱施加, 牵引边界数据
    作为应力法向迹的本质边界条件强施加. 两个边界标记必须明确划分边界;
    牵引值由 ``loads()`` 中的 ``BoundaryTraction`` 提供. 角点松弛所需的几何角点
    由分析器从网格检出, 不是 Problem 的成员.

    以下成员可选, 分析器按 ``getattr`` 探测, 不列入协议方法 (否则缺这些成员的
    问题类无法通过 ``isinstance`` 检查):

    - ``displacement_bc(points)``: 位移边界上弱施加的位移数据 :math:`u_D`;
      缺省时视为 :math:`u_D = 0`.
    - ``is_symmetry_boundary(points)``: 只强加切向牵引 :math:`\\sigma_{nt} = 0`
      的边界 (对称面或滚动支承), 法向位移为自然条件; 缺省时视为无此类边界.
    """

    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """判定点是否位于位移边界 (弱施加 ``u = u_D``).

        Parameters
        ----------
        points : TensorLike
            形状 ``(..., GD)`` 的坐标; 分析器传入边的重心.

        Returns
        -------
        TensorLike
            形状 ``points.shape[:-1]`` 的布尔掩码.
        """
        ...

    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """判定点是否位于牵引边界 (强施加 ``sigma n = t``).

        Parameters
        ----------
        points : TensorLike
            形状 ``(..., GD)`` 的坐标; 分析器传入边的重心.

        Returns
        -------
        TensorLike
            形状 ``points.shape[:-1]`` 的布尔掩码.
        """
        ...


@runtime_checkable
class MaterialInterpolation(Protocol):
    """有限元分析器消费的材料插值契约.

    各成员按只读属性声明, 与现有插值方案的 property 和 ``variantmethod`` 实现
    保持兼容. 两个插值入口使用 ``Any``, 因为未绑定的 ``variantmethod`` 描述符
    不是普通函数; 在实例上绑定后才可调用.
    """

    @property
    def density_location(self) -> str:
        """密度场的离散位置.

        取 ``'element'``, ``'node'``, ``'element_multiresolution'`` 或
        ``'node_multiresolution'``.
        """
        ...

    @property
    def n_sub(self) -> Optional[int]:
        """每个位移单元内的子密度单元数; 非多分辨率时为 None."""
        ...

    @property
    def interpolate_material(self) -> Any:
        """绑定后的材料插值入口.

        调用形式为 ``interpolate_material(material=..., rho_val=...,
        integration_order=..., displacement_mesh=...)``, 返回插值后的杨氏模量
        ``E_rho``; 同时插值泊松比时返回二元组 ``(E_rho, nu_rho)``.
        """
        ...

    @property
    def interpolate_material_derivative(self) -> Any:
        """绑定后的材料插值导数入口, 返回插值材料参数对密度的导数."""
        ...


@runtime_checkable
class AnalysisStage(Protocol):
    """拓扑优化分析阶段的公共契约.

    给定密度场 ``rho``, 分析阶段负责求解位移状态、应力状态和灵敏度所需的
    伴随变量. ``LagrangeFEMAnalyzer`` 与 ``HuZhangMFEMAnalyzer`` 都提供这些
    公共能力, 拓扑优化目标与约束可依赖本协议.

    两种分析器的内部离散和装配方式不同, 本协议只描述它们共有的分析阶段接口.
    ``compute_stress_state`` 的关键字参数目前仍存在方法相关差异.

    满足本协议的对象持有网格、材料、插值方案和分析求解器, 不持有优化循环、
    目标函数或约束函数.
    """

    @property
    def disp_mesh(self) -> Any:
        """位移有限元网格."""
        ...

    @property
    def material(self) -> Any:
        """实体线弹性材料, 提供 ``youngs_modulus`` 与 ``calculate_von_mises_stress`` 等."""
        ...

    @property
    def interpolation_scheme(self) -> MaterialInterpolation:
        """材料插值方案."""
        ...

    def solve_state(self, rho_val: Any = None, **kwargs) -> dict:
        """在密度场 ``rho_val`` 下求解状态方程.

        Parameters
        ----------
        rho_val : TensorLike or Function, optional
            密度场; 拓扑优化模式下必须提供.
        **kwargs
            分析器特有的选项, 例如 ``enable_timing``.

        Returns
        -------
        dict
            至少含 ``'displacement'``; ``HuZhangMFEMAnalyzer`` 另含 ``'stress'``.
        """
        ...

    def solve_adjoint(self, rhs: Any, rho_val: Any = None, **kwargs) -> Any:
        """以伴随载荷 ``rhs`` 求解齐次边界条件下的伴随方程.

        Parameters
        ----------
        rhs : TensorLike
            伴随载荷向量. ``HuZhangMFEMAnalyzer`` 只接受应力自由度部分.
        rho_val : TensorLike or Function, optional
            密度场, 用于组装伴随方程的左端算子.
        **kwargs
            传给线性求解器的选项.

        Returns
        -------
        TensorLike
            伴随变量向量. ``HuZhangMFEMAnalyzer`` 返回应力与位移自由度的完整向量.
        """
        ...

    def compute_stress_state(self, state: dict, **kwargs) -> dict:
        """由状态字典计算积分点应力.

        Parameters
        ----------
        state : dict
            ``solve_state`` 返回的状态字典.
        **kwargs
            方法相关的关键字参数, 例如 ``integration_order``;
            ``HuZhangMFEMAnalyzer`` 另接受 ``rho_val``.

        Returns
        -------
        dict
            应力字典, 键随方法而异: ``LagrangeFEMAnalyzer`` 给出
            ``'stress_solid'``, ``HuZhangMFEMAnalyzer`` 给出 ``'stress_apparent'``,
            形状均为 ``(NC, NQ, NS)``.
        """
        ...


__all__ = [
    "AnalysisStage",
    "BodyForce",
    "BoundaryTraction",
    "DirichletElasticityProblem",
    "ElasticityProblem",
    "LineTraction",
    "Load",
    "LoadKind",
    "LoadProvider",
    "MaterialInterpolation",
    "MixedBoundaryElasticityProblem",
    "PointForce",
    "SupportsMatmul",
]
