"""把本例的算例定义接到 SOPTX 的分析器上

分析器本身和重叠副本下的求解器都住在 ``soptx.fem``; 这里只有 demo 侧的粘合:
把 :class:`cases.ElasticityCase` 拆成构造参数, 以及给 demo 脚本用的懒装配门面。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fealpy.backend import backend_manager as bm
from fealpy.distributed import EntityMPI
from fealpy.functionspace import TensorFunctionSpace
from fealpy.typing import TensorLike

from soptx.fem.solvers import LagrangeFEMAnalyzer
from soptx.fem.solvers.matrix_free_analyzer import (
    DistributedElasticityAnalyzer,
    SupportsMatmul,
)

if TYPE_CHECKING:
    from cases import ElasticityCase


def _analyzer_arguments(
    space: TensorFunctionSpace,
    case: ElasticityCase,
    degree: int,
    operator_level: str,
) -> dict[str, Any]:
    """把 problem、material、mesh 和 space 组合成构造参数"""

    mesh = space.mesh
    return dict(
        disp_mesh=mesh,
        pde=case.problem,
        material=case.material.create(device=bm.get_device(mesh)),
        space_degree=degree,
        integration_order=degree + 3,
        operator_level=operator_level,
        tensor_space=space,
        topopt_algorithm=None,
    )


def build_serial_analyzer(
    space: TensorFunctionSpace,
    case: ElasticityCase,
    degree: int,
    operator_level: str,
) -> LagrangeFEMAnalyzer:
    """构造不含任何重叠通信的普通分析器

    供单 rank 的 EA/FA 参照使用: 它们建立在网格分布之前的全局空间上。
    """

    return LagrangeFEMAnalyzer(
        solve_method="scipy",
        **_analyzer_arguments(space, case, degree, operator_level),
    )


def build_distributed_analyzer(
    space: TensorFunctionSpace,
    case: ElasticityCase,
    degree: int,
    operator_level: str,
    dof_comm: EntityMPI,
) -> DistributedElasticityAnalyzer:
    """构造实际运行使用的分析器, 单 rank 和双 rank 都走这个"""

    return DistributedElasticityAnalyzer(
        dof_comm=dof_comm,
        **_analyzer_arguments(space, case, degree, operator_level),
    )


class ElasticityEAOperator:
    """EA 算例的懒装配缓存门面

    数值本身不在这里: :math:`\\mathbf K_e` 由 FEALPy 的 ``LinearElasticIntegrator``
    计算, :math:`\\sum_e \\mathbf R_e^\\top \\mathbf K_e \\mathbf R_e` 由
    ``BilinearForm.__matmul__`` 完成, 边界处理由分析器的 ``apply_bc`` 完成。本类
    只做三件调用方需要、而分析器不直接提供的事:

    1. 首次访问任一结果时才装配, 之后复用;
    2. 计算 Dirichlet 自由度掩码 ``boundary_dofs``;
    3. 提供 ``@``, 让 demo 脚本能把它当算子直接用。

    ``utils/run.py`` 的证据路径不经过本类, 而是直接驱动分析器。
    """

    def __init__(
        self,
        space: TensorFunctionSpace,
        case: ElasticityCase,
        degree: int = 1,
        dof_comm: EntityMPI | None = None,
    ) -> None:
        self.space = space
        self.case = case
        self.degree = degree
        self.dof_comm = dof_comm

        if dof_comm is None:
            self.analyzer = build_serial_analyzer(space, case, degree, "ea")
        else:
            self.analyzer = build_distributed_analyzer(
                space, case, degree, "ea", dof_comm=dof_comm
            )

        self._system_operator: Any = None
        self._load_vector: TensorLike | None = None
        self._prescribed: TensorLike | None = None
        self._boundary_dofs: TensorLike | None = None

    def assemble(self) -> tuple[SupportsMatmul, TensorLike]:
        """装配刚度算子与体力向量并施加边界条件"""

        operator, load = self.analyzer.apply_bc(
            self.analyzer.assemble_stiff_matrix(),
            self.analyzer.assemble_body_force_vector(),
        )
        self._system_operator = operator
        self._load_vector = load
        self._prescribed = self.analyzer.prescribed_solution
        self._boundary_dofs = self.space.is_boundary_dof(
            threshold=self.case.problem.is_dirichlet_boundary(),
            method="interp",
        )
        return operator, load

    def _ensure_assembled(self) -> None:
        if self._system_operator is None:
            self.assemble()

    @property
    def system_operator(self) -> SupportsMatmul:
        self._ensure_assembled()
        return self._system_operator

    @property
    def load_vector(self) -> TensorLike:
        self._ensure_assembled()
        return self._load_vector

    @property
    def prescribed_solution(self) -> TensorLike:
        self._ensure_assembled()
        return self._prescribed

    @property
    def boundary_dofs(self) -> TensorLike:
        self._ensure_assembled()
        return self._boundary_dofs

    def __matmul__(self, vector: TensorLike) -> TensorLike:
        return self.system_operator @ vector
