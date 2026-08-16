"""接口系统的约束施加与直接求解.

``GlobalAssembler`` 只产出接口刚度矩阵与自由度映射, 不携带求解策略. 本模块提供
一个与装配器解耦的自由函数, 把 "施加位移约束 + 稀疏直接求解" 这一步固定下来,
避免各算例脚本各写一遍自由度取补集与子矩阵切片.
"""

from typing import Any, Optional

import scipy.sparse.linalg as spla

from fealpy.backend import backend_manager as bm

from .assembler import InterfaceSystem


def solve_interface_system(
    system: InterfaceSystem,
    load: Any,
    fixed_dofs: Optional[Any] = None,
    *,
    prescribed: Optional[Any] = None,
) -> Any:
    """在接口系统上施加位移约束并用稀疏直接法求解.

    参数:
        system: 已装配的接口系统.
        load: 接口自由度上的右端项, 形状 ``(n_interface,)``. 可由
            ``GlobalAssembler.project_global_vector`` 从全局载荷投影得到.
        fixed_dofs: 受约束的接口自由度编号. 可由
            ``GlobalAssembler.project_global_dofs`` 从全局固定自由度投影得到.
            为 ``None`` 时不施加任何约束.
        prescribed: 接口自由度上的给定位移, 形状 ``(n_interface,)``. 只有
            ``fixed_dofs`` 位置上的分量被采用, 其余分量被忽略. 为 ``None`` 时
            视为齐次约束.

    返回:
        u: 接口自由度上的位移, 形状 ``(n_interface,)``. 约束自由度取给定值,
            其余自由度为求解结果.

    异常:
        ValueError: 当 ``load`` 或 ``prescribed`` 的长度与接口自由度数不一致时
            抛出.

    说明:
        非齐次约束按 ``K_ff u_f = f_f - (K u_c)_f`` 缩减, 其中 ``u_c`` 是只在约束
        自由度上取给定值, 其余为零的向量.

        接口矩阵以 ``scipy`` 稀疏格式持有, 数值在此经 ``bm.to_numpy`` 转出到
        ``scipy.sparse.linalg``, 这是流程中与第三方求解库对接的边界.
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
        f = f - bm.asarray(system.stiffness @ bm.to_numpy(u), dtype=bm.float64)

    all_dofs: Any = bm.arange(n_interface, dtype=bm.int64)
    free = all_dofs[bm.isin(all_dofs, fixed, invert=True)]
    if len(free) == 0:
        return u

    free_np = bm.to_numpy(free)
    u_free = spla.spsolve(
        system.stiffness[free_np[:, None], free_np],
        bm.to_numpy(f[free]),
    )
    return bm.set_at(u, free, bm.asarray(u_free, dtype=bm.float64))
