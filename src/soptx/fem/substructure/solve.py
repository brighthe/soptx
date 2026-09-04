"""接口系统的约束施加与直接求解.

``GlobalAssembler`` 只产出接口刚度矩阵与自由度映射, 不携带求解策略. 本模块提供
一个与装配器解耦的自由函数, 把 "施加位移约束 + 稀疏直接求解" 这一步固定下来,
避免各算例脚本各写一遍自由度取补集与子矩阵切片.
"""

from typing import Any, Optional

from fealpy.backend import backend_manager as bm
from soptx.solvers import create

from .assembler import InterfaceSystem

#: 本函数支持的直接法后端。CG 一类迭代法不在此列: 接口系统规模小且要求一次
#: 给准, 迭代法在这里没有收益; 旧实现对未知名字会静默回落到 scipy, 现在报错。
DIRECT_BACKENDS = ("scipy", "mumps")


def solve_interface_system(
    system: InterfaceSystem,
    load: Any,
    fixed_dofs: Optional[Any] = None,
    *,
    prescribed: Optional[Any] = None,
    solver: str = 'scipy',
) -> Any:
    """在接口系统上施加位移约束并用 soptx.solvers 的 DirectSolver 求解.

    参数:
        system: 已装配的接口系统 (持有 FEALPy ``CSRTensor``).
        load: 接口自由度上的右端项, 形状 ``(n_interface,)``. 可由
            ``GlobalAssembler.project_global_vector`` 从全局载荷投影得到.
        fixed_dofs: 受约束的接口自由度编号. 可由
            ``GlobalAssembler.project_global_dofs`` 从全局固定自由度投影得到.
            为 ``None`` 时不施加任何约束.
        prescribed: 接口自由度上的给定位移, 形状 ``(n_interface,)``. 只有
            ``fixed_dofs`` 位置上的分量被采用, 其余分量被忽略. 为 ``None`` 时
            视为齐次约束.
        solver: 底层直接法后端, 取 ``DIRECT_BACKENDS`` 之一, 缺省为 ``'scipy'``.

    返回:
        u: 接口自由度上的位移, 形状 ``(n_interface,)``. 约束自由度取给定值,
            其余自由度为求解结果.

    异常:
        ValueError: 当 ``load`` 或 ``prescribed`` 的长度与接口自由度数不一致时
            抛出.

    说明:
        非齐次约束按 ``K_ff u_f = f_f - (K u_c)_f`` 缩减, 其中 ``u_c`` 是只在约束
        自由度上取给定值, 其余为零的向量.
    """
    n_interface = int(len(system.global_dofs))

    f: Any = bm.asarray(load, dtype=bm.float64)
    if len(f) != n_interface:
        raise ValueError(
            f"load 的长度必须等于接口自由度数 {n_interface}; 当前为 {len(f)}."
        )

    u: Any = bm.zeros((n_interface,), dtype=bm.float64)
    if fixed_dofs is None:
        fixed: Any = bm.zeros((0,), dtype=bm.int64)
    else:
        fixed = bm.unique(bm.asarray(fixed_dofs, dtype=bm.int64))

    if prescribed is not None:
        u_c = bm.asarray(prescribed, dtype=bm.float64)
        if len(u_c) != n_interface:
            raise ValueError(
                f"prescribed 的长度必须等于接口自由度数 {n_interface}; "
                f"当前为 {len(u_c)}."
            )
        u = bm.set_at(u, fixed, u_c[fixed])
        # 给定位移在未约束自由度上产生的反力, 移到右端项.
        f = f - (system.stiffness @ u)

    all_dofs: Any = bm.arange(n_interface, dtype=bm.int64)
    free = all_dofs[bm.isin(all_dofs, fixed, invert=True)]
    if len(free) == 0:
        return u

    free_np = bm.to_numpy(free)
    f_np = bm.to_numpy(f)

    # 提取标准的 CSR 主子矩阵
    if hasattr(system.stiffness, "to_scipy"):
        K_scipy = system.stiffness.to_scipy()
    else:
        K_scipy = system.stiffness

    K_free_scipy = K_scipy[free_np, :][:, free_np].tocsr()

    if solver not in DIRECT_BACKENDS:
        raise ValueError(
            f"未知的直接法后端: {solver!r}; 可选 {DIRECT_BACKENDS}."
        )
    # 接口系统在本函数内只解一次, 分解不跨调用复用; MUMPS 上下文用完即释放,
    # MPI 初始化由 DirectSolver 自己负责, 调用方不必再准备。
    linear_solver = create(solver)
    try:
        u_free, _ = linear_solver.setup(K_free_scipy).solve(f_np[free_np])
    finally:
        linear_solver.close()
    u_free_np = bm.to_numpy(u_free)

    return bm.set_at(u, free, bm.asarray(u_free_np, dtype=bm.float64))
