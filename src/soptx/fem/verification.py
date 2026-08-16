"""交叉校核用的参考解构造与误差度量

EA 路径没有可以打印出来的全局矩阵, 正确性只能靠与 FA 及直接解的比对来钉住。
本模块提供这套比对所需的三样东西: 相对差、L2 位移误差, 以及一次性构造好全部
参照量的 :func:`serial_references`。

原先住在 ``examples/matrix_free_elasticity/utils/references.py`` 与
``utils/postprocess.py``, demo 脚本和证据工具都要用, 因此上浮到 ``soptx``。

判定阈值不在这里: 按 :mod:`soptx.numerics` 的约定, 验收门禁的数字属于定义该门禁
的示例或研究, 本模块只产出被判定的量。随机种子同理, 由调用方显式传入。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.linalg import spsolve as scipy_spsolve

from fealpy.backend import backend_manager as bm
from fealpy.fem import DirichletBCOperator
from fealpy.functionspace import TensorFunctionSpace
from fealpy.mesh import Mesh

from soptx.numerics import NORM_FLOOR

from .solvers.elasticity_operator import build_serial_analyzer


def relative_difference(left, right) -> tuple[float, float]:
    """两个向量的绝对差与以右侧范数为基准的相对差"""

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    absolute = float(np.linalg.norm(left_array - right_array))
    relative = absolute / max(float(np.linalg.norm(right_array)), NORM_FLOOR)
    return absolute, relative


def solution_error(
    mesh: Mesh,
    solution,
    pde: Any,
    degree: int,
) -> tuple[float, float]:
    """位移的绝对 L2 误差, 及以精确解范数为基准的相对 L2 误差"""

    absolute = float(
        mesh.error(
            pde.disp_solution,
            solution,
            q=degree + 3,
        )
    )

    def zero_field(points):
        return bm.zeros_like(pde.disp_solution(points))

    exact_norm = float(
        mesh.error(
            pde.disp_solution,
            zero_field,
            q=degree + 3,
        )
    )
    return absolute, absolute / max(exact_norm, NORM_FLOOR)


def serial_references(
    space: TensorFunctionSpace,
    pde: Any,
    material: Any,
    degree: int,
    *,
    seed: int,
):
    """构造只用于正确性判定的 EA/FA 与直接解参照

    仅限单 rank, 所以用普通串行分析器。两个算子层级出自同一个类, 因此下面任何
    差异都是两种装配策略之间的真实差异, 而不是两份独立实现之间的差异。
    """

    ea = build_serial_analyzer(space, pde, material, degree, "ea")
    fa = build_serial_analyzer(space, pde, material, degree, "fa")

    element_form = ea.assemble_stiff_matrix()
    fa_matrix = fa.assemble_stiff_matrix()
    fa_operator, fa_load = fa.apply_bc(fa_matrix, fa.assemble_body_force_vector())

    boundary_dofs = space.is_boundary_dof(
        threshold=pde.is_dirichlet_boundary(),
        method="interp",
    )

    random = np.random.default_rng(seed)
    first = bm.asarray(
        random.standard_normal(space.number_of_global_dofs()),
        dtype=bm.float64,
    )
    raw_absolute, raw_relative = relative_difference(
        element_form @ first,
        fa_matrix @ first,
    )

    element_boundary_operator = DirichletBCOperator(
        element_form,
        gd=pde.dirichlet_bc,
        isDDof=boundary_dofs,
    )
    boundary_absolute, boundary_relative = relative_difference(
        element_boundary_operator @ first,
        fa_operator @ first,
    )

    # 正定性探针. 这里不再另测对称性: EA 算子与 FA 在随机探针下差到 1e-16, 而 FA
    # 按对称双线性型装配再对称消元, 本身严格对称, 于是 EA 的对称性是上面那条比对
    # 的推论而非独立信息. 正定性不同, 它问的是这个离散系统本身有没有退化, 与 FA
    # 对不对无关, 而且复用已经算出的 first_action, 不额外花钱
    energy = float(bm.sum(first * (element_boundary_operator @ first)))

    # SciPy 的 ``spsolve`` 只高效接受 CSR/CSC. ``DirichletBCOperator`` 当前导出
    # 为 COO, 因此在黄金参考路径显式转成 CSR, 避免每次验证时隐式转换并发出警告.
    direct_solution = scipy_spsolve(
        fa_operator.to_scipy().tocsr(),
        bm.to_numpy(fa_load),
    )
    return (
        {
            "raw_absolute_error": raw_absolute,
            "raw_relative_error": raw_relative,
            "dirichlet_absolute_error": boundary_absolute,
            "dirichlet_relative_error": boundary_relative,
            "random_vector_energy": energy,
        },
        np.asarray(direct_solution),
    )


__all__ = [
    "relative_difference",
    "serial_references",
    "solution_error",
]
