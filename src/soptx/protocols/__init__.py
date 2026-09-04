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
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    Sequence,
    runtime_checkable,
)

# ``TensorLike`` 仅用于类型注解.
if TYPE_CHECKING:
    from fealpy.typing import TensorLike

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
    def domain(self) -> Sequence[float]: ...

    def loads(self) -> Sequence[Load]: ...


@runtime_checkable
class DirichletElasticityProblem(ElasticityProblem, Protocol):
    """``LagrangeFEMAnalyzer`` 消费的 Problem 契约.

    物理外载荷统一由 ``loads()`` 提供.

    伴随右端项以及弹簧支承不是物理外载荷对象, 仍分别使用
    ``adjoint_load_bc`` / ``is_adjoint_load_boundary`` 和
    ``k_in`` / ``k_out`` / ``is_spring_boundary``.
    """

    @property
    def boundary_type(self) -> str: ...

    def dirichlet_bc(self, points: TensorLike) -> TensorLike: ...

    def is_dirichlet_boundary(self) -> tuple[Callable[..., TensorLike], ...]: ...


@runtime_checkable
class MixedBoundaryElasticityProblem(ElasticityProblem, Protocol):
    """``HuZhangMFEMAnalyzer`` 消费的 Problem 契约.

    Hu--Zhang 混合形式中, 位移边界数据按自然边界条件弱施加, 牵引边界数据
    作为应力法向迹的本质边界条件强施加. 两个边界标记必须明确划分边界;
    牵引值由 ``loads()`` 中的 ``BoundaryTraction`` 提供. ``mark_corners``
    为 Hu--Zhang 角点松弛提供几何角点.
    """

    def mark_corners(self, node: TensorLike) -> TensorLike: ...

    def is_displacement_boundary(self, points: TensorLike) -> TensorLike: ...

    def is_traction_boundary(self, points: TensorLike) -> TensorLike: ...


@runtime_checkable
class MaterialInterpolation(Protocol):
    """有限元分析器消费的材料插值契约.

    各成员按只读属性声明, 与现有插值方案的 property 和 ``variantmethod`` 实现
    保持兼容. 两个插值入口使用 ``Any``, 因为未绑定的 ``variantmethod`` 描述符
    不是普通函数; 在实例上绑定后才可调用.
    """

    @property
    def density_location(self) -> str: ...

    @property
    def n_sub(self) -> int: ...

    @property
    def interpolate_material(self) -> Any: ...

    @property
    def interpolate_material_derivative(self) -> Any: ...


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
    def disp_mesh(self) -> Any: ...

    @property
    def material(self) -> Any: ...

    @property
    def interpolation_scheme(self) -> MaterialInterpolation: ...

    def solve_state(self, rho_val: Any = None, **kwargs) -> dict: ...

    def solve_adjoint(self, rhs: Any, rho_val: Any = None, **kwargs) -> Any: ...

    def compute_stress_state(self, state: dict, **kwargs) -> dict: ...


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
