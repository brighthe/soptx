"""线弹性 Matrix-Free EA 正向分析: 线性系统封装、求解诊断与一步求解门面.

把「分析器调度 ➔ 单元小刚度矩阵装配 ➔ Dirichlet 边界条件对角投影 ➔ 线性系统封装 ➔
重叠加权 CG 迭代求解」整条流水线封装为一步闭环调用. 求解策略留在本模块,
算子门面 ``ElasticityEAOperator`` 只负责装配生命周期与算子代数.

重叠加权内积本身与具体物理无关, 已下沉到 :mod:`soptx.solvers.overlap`; 本模块
只做线弹性专用的组合, 依赖方向为 ``solve`` → ``operator`` 与
``solve`` → ``soptx.solvers``.

默认容差与数值界由 :mod:`soptx.core.numerics` 提供.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import TensorFunctionSpace
from fealpy.typing import TensorLike

from soptx.core.numerics import (
    DEFAULT_ATOL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RTOL,
    NORM_FLOOR,
)
from soptx.solvers import weighted_cg, weighted_norm

from soptx.fem.matrix_free.operator import ElasticityEAOperator


@dataclass
class PreparedLinearSystem:
    """施加边界条件后的无矩阵线性系统封装结构体.

    属性:
        operator (Any): 施加边界条件后的系统刚度算子 (支持 ``@`` 矩阵乘法).
        load (TensorLike): 施加边界条件修正后的等效右端载荷向量.
        prescribed (TensorLike): Dirichlet 边界上的指定位移真解向量 (通常用于初始化 x0).
        boundary_dofs (TensorLike): Dirichlet 边界自由度的一维布尔掩码向量.
    """

    operator: Any
    load: TensorLike
    prescribed: TensorLike
    boundary_dofs: TensorLike


def solver_diagnostics(
    system: PreparedLinearSystem,
    solution: TensorLike,
    dof_comm: Any,
    cg_info: dict[str, Any],
) -> dict[str, Any]:
    """计算解的真实代数残差范数、相对残差以及 Dirichlet 边界误差指标.

    参数:
        system (PreparedLinearSystem): 待求解的线性系统对象.
        solution (TensorLike): 求解器输出的局部自由度解向量.
        dof_comm (EntityMPI | None): 自由度跨进程通信器.
        cg_info (dict[str, Any]): CG 迭代求解器返回的状态信息字典.

    返回:
        dict[str, Any]: 包含迭代步数、真残差、相对残差、边界绝对/相对误差等收敛诊断字典.

    说明:
        ``true_relative_residual`` 与 ``cg_info['relres']`` 是同一个量的两次独立计算,
        这里刻意保留: 本函数是求解器之外的事后核验 (还要一并给出边界误差), 用它自己的
        ``weighted_norm`` 度量, 不复用求解器内部结论.

        ``rhs_norm`` 与 ``reference_norm`` 是两个不同的量, 都要给: 前者是 ``||load||``,
        后者是 CG 里 rtol 真正的参照量 ``||r0||``. ``solve_matrix_free_system`` 传
        ``x0=system.prescribed`` 热启动, 此时 ``||r0|| < ||load||``, 按 ``rhs_norm``
        设门限会比求解器实际达到的判据宽。这里先把 ``reference_norm`` 生产出来,
        门禁口径的切换 (``tools/matrix_free_evidence/contract.py``) 要连同 stage-1
        证据链重跑, 是独立的一步.
    """
    residual = system.operator @ solution - system.load
    residual_norm = weighted_norm(residual, dof_comm)
    load_norm = weighted_norm(system.load, dof_comm)

    boundary_error = bm.where(
        system.boundary_dofs,
        solution - system.prescribed,
        bm.zeros_like(solution),
    )
    boundary_reference = bm.where(
        system.boundary_dofs,
        system.prescribed,
        bm.zeros_like(system.prescribed),
    )
    boundary_absolute = weighted_norm(boundary_error, dof_comm)
    boundary_reference_norm = weighted_norm(boundary_reference, dof_comm)

    return {
        "name": "matrix-free-weighted-cg",
        "converged": bool(cg_info["converged"]),
        "iterations": int(cg_info["niter"]),
        "reported_residual": float(
            cg_info.get("true_residual") or cg_info.get("residual", 0.0)
        ),
        "recursive_residual": float(cg_info.get("recursive_residual", 0.0)),
        "true_absolute_residual": residual_norm,
        "rhs_norm": load_norm,
        "reference_norm": float(cg_info.get("reference_norm", 0.0)),
        "true_relative_residual": (
            residual_norm / max(load_norm, NORM_FLOOR)
        ),
        "boundary_absolute_error": boundary_absolute,
        "boundary_relative_error": (
            boundary_absolute / max(boundary_reference_norm, NORM_FLOOR)
        ),
        "breakdown": cg_info.get("breakdown"),
    }


def solve_matrix_free_system(
    system: PreparedLinearSystem,
    dof_comm: Any,
    *,
    maxiter: int = DEFAULT_MAX_ITERATIONS,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> tuple[TensorLike, dict[str, Any]]:
    """求解已准备好的无矩阵线性系统, 并一键返回局部解向量与全套收敛诊断报告.

    参数:
        system (PreparedLinearSystem): 待求解的无矩阵系统结构体 (包含刚度算子、载荷、指定位移与边界标记).
        dof_comm (EntityMPI | None): 自由度通信器.
        maxiter (int, 可选): 最大允许迭代步数.
        rtol (float, 可选): 相对残差容差.
        atol (float, 可选): 绝对残差容差.

    返回:
        tuple[TensorLike, dict[str, Any]]: 包含局部解向量 ``solution`` 与诊断字典 ``diagnostics`` 的二元组.
    """
    solution, info = weighted_cg(
        system.operator,
        system.load,
        dof_comm=dof_comm,
        x0=system.prescribed,
        maxiter=maxiter,
        rtol=rtol,
        atol=atol,
    )
    diagnostics = solver_diagnostics(system, solution, dof_comm, info)
    return solution, diagnostics


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
            - atol (float): 绝对残差容差 (默认采用 ``soptx.core.numerics.DEFAULT_ATOL``).
            - rtol (float): 相对残差容差 (默认采用 ``soptx.core.numerics.DEFAULT_RTOL``).
            - max_iter (int): 最大允许 CG 迭代步数 (默认采用 ``soptx.core.numerics.DEFAULT_MAX_ITERATIONS``).

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
    "PreparedLinearSystem",
    "solve_ea_system",
    "solve_matrix_free_system",
    "solver_diagnostics",
]
