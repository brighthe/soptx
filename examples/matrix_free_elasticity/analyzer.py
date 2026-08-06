from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

from fealpy.backend import backend_manager as bm
from fealpy.distributed import EntityMPI
from fealpy.fem import BilinearForm
from fealpy.functionspace import TensorFunctionSpace
from fealpy.typing import TensorLike

from soptx.fem.solvers import LagrangeFEMAnalyzer

import contract
from cases import ElasticityCase
from distributed import OverlapOperator


class SupportsMatmul(Protocol):
    """能被迭代解法作用的算子

    'fa' 下是 CSRTensor/COOTensor, 'ea' 下是层层包装的 DirichletBCOperator。
    两者唯一的共性就是支持 ``@``, 所以按这个协议标注, 而不是枚举具体类型——
    枚举会随实现漂移。

    FEALPy 内部有同名协议, 但定义在 ``fealpy/solver/cg.py`` 里且未导出, 不宜依赖。
    """

    def __matmul__(self, other: TensorLike) -> TensorLike: ...


#: 求解器返回 (解, 诊断信息)
SolverResult = tuple[TensorLike, dict]

#: 登记表里求解器的统一签名。关键字随求解器而异, 所以参数列表用 ``...``。
SolverRoutine = Callable[..., SolverResult]


def _overlap_cg(
    K: SupportsMatmul,
    F: TensorLike,
    *,
    dof_comm: EntityMPI,
    x0: Optional[TensorLike],
    maxiter: int,
    rtol: float,
    atol: float,
    **options: Any,
) -> SolverResult:
    """用 fealpy.solver.cg 的 ``dot_product`` 注入重叠修正内积."""

    from fealpy.solver.cg import cg

    dot_fn, _norm_fn = dof_comm.dot(int(F.shape[0]))

    return cg(
        K,
        F,
        x0=x0,
        dot_product=dot_fn,
        residual_refresh=options.pop(
            "residual_refresh",
            contract.RESIDUAL_REFRESH,
        ),
        atol=atol,
        rtol=rtol,
        maxit=maxiter,
        returninfo=True,
    )


#: 重叠副本布局下可用的求解器。
#:
#: ``fealpy.solver.cg`` 现在接受 ``dot_product`` 参数注入重叠修正内积,
#: 因此可以直接使用, 不再需要本地 CG 副本。新增其他 Krylov 子空间方法同理。
DISTRIBUTED_SOLVERS: dict[str, SolverRoutine] = {
    "cg": _overlap_cg,
}


class DistributedElasticityAnalyzer(LagrangeFEMAnalyzer):
    """补上重叠 MPI 部分的 LagrangeFEMAnalyzer

    离散由分析器负责, 并行的东西全部留在这里。覆盖三个扩展点:

    * ``reduce_load``   -- 把各 rank 在共享自由度副本上的贡献求和
    * ``wrap_operator`` -- 让单个局部 matvec 表现为全局算子
    * ``solve_system``  -- 派发到 ``DISTRIBUTED_SOLVERS`` 里的加权内积求解器

    前两个只能从 EA 路径到达: ``apply_bc('fa')`` 改写的是已经装配好的全局矩阵,
    没有插入重叠归约的位置, 这也是基类直接拒绝多 rank FA 的原因。
    ``solve_system`` 则两个算子层级都用, 包括单 rank 的 FA 参照——那里所有重叠
    归约本身就是恒等操作。
    """

    def __init__(self, *args: Any, dof_comm: EntityMPI, **kwargs: Any) -> None:
        if dof_comm is None:
            raise ValueError(
                "分布式分析器必须提供 dof_comm; 串行请用 build_serial_analyzer"
            )
        super().__init__(*args, dof_comm=dof_comm, **kwargs)

    def reduce_load(self, F: TensorLike) -> TensorLike:
        return self.dof_comm.sync_add(F)

    def wrap_operator(self, form: BilinearForm) -> OverlapOperator:
        return OverlapOperator(form, self.dof_comm)

    def solve_system(
        self,
        K: SupportsMatmul,
        F: TensorLike,
        out: TensorLike,
        *,
        x0: Optional[TensorLike] = None,
        solver: str = "cg",
        maxiter: int = contract.DEFAULT_MAX_ITERATIONS,
        rtol: float = contract.DEFAULT_RTOL,
        atol: float = contract.DEFAULT_ATOL,
        **options: Any,
    ) -> SolverResult:
        """派发到重叠加权内积的迭代解法

        关键字沿用基类的词汇表 (``x0``、``solver``、``maxiter``、``rtol``、
        ``atol``), 这样经由 ``solve_state`` 传进来的参数不会被悄悄丢弃。多余的
        ``options`` 原样转交给具体求解器, 供其自有选项使用。
        """

        routine = DISTRIBUTED_SOLVERS.get(solver)
        if routine is None:
            raise NotImplementedError(
                f"求解器 {solver!r} 在重叠副本布局下没有可用实现; "
                f"当前已登记: {sorted(DISTRIBUTED_SOLVERS)}。"
                f"详见 DISTRIBUTED_SOLVERS 的说明。"
            )

        solution, info = routine(
            K,
            F,
            dof_comm=self.dof_comm,
            x0=x0,
            maxiter=maxiter,
            rtol=rtol,
            atol=atol,
            **options,
        )
        out[:] = solution
        return out, info


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
