"""完整接口 full_trace 的收敛验证实现, 由独立验证脚本调用.

本脚本通过在多层嵌套加密网格上求解线弹性问题, 验证子结构静力缩聚 (Schur 补缩聚)
离散解逼近真实连续解的收敛速度.

根据有限元先验误差估计理论, 采用双线性四边形 (Q1) 或三线性六面体单元 (Hexahedron)
离散时, 有限元解在 :math:`L_2` 范数下的理论收敛阶为 2.0 阶 (:math:`O(h^2)`).

物理模型采用无体力的调和多项式制造解 :class:`~soptx.problems.elasticity.HarmonicPoly2D`
与 :class:`~soptx.problems.elasticity.HarmonicPoly3D`, 严格满足
:math:`\\Delta u = 0, \\nabla \\cdot u = 0 \\implies b(x) \\equiv 0`, 由非齐次 Dirichlet
位移边界条件驱动. 该设定完全符合子结构静力缩聚内部自由度不受载 (:math:`f_i = 0`) 的建模假设,
能够纯粹、隔离地检验 Schur 补缩聚刚度算子 :math:`S` 与内部位移恢复算子 :math:`N` 的有限元逼近精度.
制造解的数学推导见 `制造解文档 <../../docs/problems/manufactured-elasticity.md>`__.

仅求解 full_trace, 计算其相对解析制造解的位移 L2 误差与 H1 半范误差.
最后一对网格检查 L2 阶不低于 p+1-0.2、H1 阶不低于 p-0.2.
该制造位移为三次多项式, 因此本入口仅检验 p=1、2; p>=3 已可精确表示该场,
不适合用误差比值检验先验收敛阶. 这不限制子结构核心的高阶单元能力.

二维统一使用原型当前实现的 plane_stress. 制造位移无散且调和, 对该本构仍满足
零体力平衡; 本脚本不调用问题类按 plane_strain 定义的应力函数.

使用方法:
    # 2D 调和多项式收敛阶验证 (默认 4 层网格加密).
    python examples/substructure_elasticity/verify_full_trace_convergence.py --problem HarmonicPoly2D

    # 3D 调和多项式收敛阶验证 (默认 3 层网格加密).
    python examples/substructure_elasticity/verify_full_trace_convergence.py --problem HarmonicPoly3D

``--output-dir`` 缺省为本脚本同级的 ``outputs/``, 按脚本位置解析.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm

from soptx.fem.integrators import SourceIntegrator
from soptx.protocols import BodyForce
from soptx.fem.substructure import (
    FEAStaticCondensation,
    GlobalAssembler,
    SubstructureMesh,
    SubstructurePrototype,
    solve_interface_system,
)
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.fem.substructure.solve import DIRECT_BACKENDS
from soptx.problems.elasticity import (
    HarmonicPoly2D,
    HarmonicPoly3D,
)

from examples.substructure_elasticity._common import (
    DEFAULT_LEVELS,
    ORDER_MARGIN,
    display_width,
)


### 子结构网格构建 ###

def build_substructures(
    assembler: GlobalAssembler,
) -> Tuple[SubstructurePrototype, List[SubstructureMesh]]:
    """依据装配器拓扑构建参考子结构原型与子结构网格集合."""
    dim = assembler.dim
    sub_size = tuple(assembler.domain_size[d] / assembler.n_sub[d] for d in range(dim))
    prototype = SubstructurePrototype(
        sub_size,
        assembler.n_fine,
        assembler.E_base,
        assembler.nu,
        degree=assembler.degree,
        integration_order=max(4, assembler.degree + 3),
    )
    grid = (
        [
            (x, y)
            for y in range(assembler.n_sub[1])
            for x in range(assembler.n_sub[0])
        ]
        if dim == 2
        else [
            (x, y, z)
            for z in range(assembler.n_sub[2])
            for y in range(assembler.n_sub[1])
            for x in range(assembler.n_sub[0])
        ]
    )
    sub_meshes: List[SubstructureMesh] = []
    for sub_id, pos in enumerate(grid):
        spans = tuple(
            (pos[d] * sub_size[d], (pos[d] + 1) * sub_size[d]) for d in range(dim)
        )
        sub_meshes.append(
            SubstructureMesh(
                sub_id, *spans, *assembler.n_fine,
                assembler.E_base, assembler.nu, prototype=prototype,
            )
        )
    return prototype, sub_meshes


### 单层求解与误差评估 ###

def solve_one_level(
    pde: Any,
    material: Any,
    n_sub: Tuple[int, ...],
    n_fine: Tuple[int, ...],
    degree: int = 1,
    solve_method: str = 'scipy',
) -> Dict[str, Any]:
    """求解 full_trace 并计算相对解析制造解的误差.

    Parameters
    ----------
    pde : object
        无体力、全 Dirichlet 制造解问题.
    material : object
        与子结构原型一致的各向同性材料.
    n_sub, n_fine : tuple of int
        各方向子结构数与每块细单元数.
    degree : int
        位移插值次数, 本制造解验证限于 1 或 2.
    solve_method : str
        接口直接求解后端, scipy 或 mumps.

    Returns
    -------
    dict
        网格信息与 full_trace 的 L2/H1 半范误差.
    """
    if solve_method not in DIRECT_BACKENDS:
        raise ValueError(f"未知求解方法 {solve_method!r}; 可选 {DIRECT_BACKENDS}.")
    dim = pde.dimension
    domain_size = tuple(
        pde.domain[2 * d + 1] - pde.domain[2 * d] for d in range(dim)
    )
    assembler = GlobalAssembler(
        domain_size, n_sub, n_fine, degree=degree, E_base=pde.E, nu=pde.nu
    )
    prototype, sub_meshes = build_substructures(assembler)
    if prototype.material.hypothesis != material.hypothesis:
        raise ValueError("配置材料与子结构原型的材料假设不同.")

    # 全尺度空间仅用于载荷检查、边界插值及恢复后的误差积分, 不装配 FA 刚度.
    density = bm.ones((len(sub_meshes),) + tuple(assembler.n_fine), dtype=bm.float64)
    space = assembler.space_full
    force = bm.zeros(assembler.total_full_dofs, dtype=bm.float64)
    for load in pde.loads():
        if not isinstance(load, BodyForce):
            raise ValueError("本全 Dirichlet 制造解不支持自然边界载荷.")
        form = LinearForm(space)
        form.add_integrator(SourceIntegrator(
            source=load.body_force, q=max(4, degree + 3)
        ))
        force = force + form.assembly(format='dense')
    if bool(bm.any(force != 0.0)):
        raise ValueError("本制造解验证仅支持零外载, 非零内部载荷需要额外缩聚右端项.")

    K_local_batch = prototype.assemble_local_stiffness_batch(density)
    condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    condensor.condense(K_local_batch)
    system = assembler.assemble_interface_system(sub_meshes, condensor)

    dofs_val, fixed_mask = space.boundary_interpolate(
        gd=pde.dirichlet_bc,
        threshold=cast(Any, pde.is_dirichlet_boundary()),
        method='interp',
    )
    fixed_global_dofs = bm.nonzero(fixed_mask)[0]
    if not bool(bm.all(bm.isin(fixed_global_dofs, system.global_dofs))):
        raise ValueError("存在未保留在接口系统的 Dirichlet 自由度.")
    fixed_interface_dofs = assembler.project_global_dofs(system, fixed_global_dofs)
    prescribed_interface = assembler.project_global_vector(system, dofs_val)

    # 非齐次边界由给定接口位移驱动, 不把边界反力当成外载.
    zero_load = bm.zeros((len(system.global_dofs),), dtype=bm.float64)
    u_interface = solve_interface_system(
        system,
        zero_load,
        fixed_interface_dofs,
        prescribed=prescribed_interface,
        solver=solve_method,
    )

    U_sub = assembler.recover_full_displacement(
        sub_meshes, condensor, system, u_interface
    )
    uh_sub = space.function()
    uh_sub[:] = bm.reshape(U_sub, (-1,))
    if not bool(bm.all(bm.isfinite(uh_sub[:]))):
        raise AssertionError("full_trace 位移含 NaN 或 Inf.")

    l2_error = float(assembler.full_mesh.error(
        pde.disp_solution, uh_sub, q=max(4, degree + 3)
    ))
    # 梯度按 (..., 位移分量, 空间方向) 排列, 对所有分量平方和积分.
    h1_error = float(assembler.full_mesh.error(
        pde.grad_disp_solution, uh_sub.grad_value, q=max(4, degree + 3)
    ))
    for name, error in (("L2", l2_error), ("H1 半范", h1_error)):
        if not math.isfinite(error) or error <= 0.0:
            raise AssertionError(f"full_trace {name} 误差必须为有限正数, 收到 {error}.")

    total_fine = tuple(n_sub[d] * n_fine[d] for d in range(dim))
    mesh_size = float(max(domain_size[d] / total_fine[d] for d in range(dim)))

    return {
        "n_sub": list(n_sub),
        "n_fine": list(n_fine),
        "total_fine": list(total_fine),
        "mesh_size": mesh_size,
        "full_dofs": int(assembler.total_full_dofs),
        "interface_dofs": int(len(system.global_dofs)),
        "full_trace": {"l2_error": l2_error, "h1_semi_error": h1_error},
    }


### 问题与材料工厂 ###

PROBLEM_FACTORIES = {
    2: {
        "harmonic-poly": lambda: (
            HarmonicPoly2D(domain=(0.0, 1.0, 0.0, 1.0)),
            "plane_stress",
        ),
    },
    3: {
        "harmonic-poly": lambda: (
            HarmonicPoly3D(domain=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)),
            "3D",
        ),
    },
}


def run_convergence_benchmark(
    dim: int,
    model: str = "harmonic-poly",
    degree: int = 1,
    levels: int | None = None,
    output_dir: str | None = None,
    solve_method: str = 'scipy',
) -> Dict[str, Any]:
    """检验 full_trace 相对解析制造解的 L2 与 H1 半范收敛阶.

    Parameters
    ----------
    dim : int
        计算维度, 2 或 3.
    model : str
        制造解名称, 当前支持 harmonic-poly.
    degree : int
        位移插值次数, 本制造解验证仅支持 1、2.
    levels : int or None
        网格层数, 至少为 2; 默认二维 4 层、三维 3 层.
    output_dir : str or None
        通过全部门禁后的 JSON 输出目录; None 表示不落盘.
    solve_method : str
        接口直接求解后端, scipy 或 mumps, 默认 scipy.

    Returns
    -------
    dict
        配置、每层 full_trace 误差和阶、验收结果.

    Raises
    ------
    ValueError
        维度、次数、层数或模型不支持.
    AssertionError
        有限性或最终收敛阶未通过门禁.
    """
    if isinstance(dim, bool) or not isinstance(dim, int) or dim not in (2, 3):
        raise ValueError("dim 必须为 2 或 3.")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree not in (1, 2):
        raise ValueError(
            "本三次多项式制造解仅用于 degree=1 或 2 的收敛阶验证; "
            "degree>=3 可精确表示该场, 不能以误差比值验证理论阶."
        )
    if levels is None:
        levels = DEFAULT_LEVELS[dim]
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 2:
        raise ValueError("levels 必须为不小于 2 的整数.")
    if model not in PROBLEM_FACTORIES[dim]:
        available = list(PROBLEM_FACTORIES[dim].keys())
        raise ValueError(
            f"{dim}D 维度下不支持模型 '{model}'; 可选模型为: {available}."
        )

    if solve_method not in DIRECT_BACKENDS:
        raise ValueError(f"未知求解方法 {solve_method!r}; 可选 {DIRECT_BACKENDS}.")

    bm.set_backend('numpy')
    factory = PROBLEM_FACTORIES[dim][model]
    pde, hypothesis = factory()
    material = IsotropicLinearElasticMaterial(
        hypothesis=hypothesis,
        youngs_modulus=pde.E,
        poisson_ratio=pde.nu,
        enable_logging=False,
    )

    base_sub = 2
    n_fine = (2, 2) if dim == 2 else (2, 2, 2)
    expected_orders = {"l2": float(degree + 1), "h1_semi": float(degree)}
    minimum_orders = {key: value - ORDER_MARGIN for key, value in expected_orders.items()}
    target = (
        Path(output_dir) / (
            f"full_trace_convergence_{dim}d_{model}_p{degree}_levels{levels}"
            f"_solver-{solve_method}.json"
        )
        if output_dir is not None else None
    )

    domain_text = "×".join(
        f"[{pde.domain[2*d]:g},{pde.domain[2*d+1]:g}]" for d in range(dim)
    )
    last_sub = base_sub * (2 ** (levels - 1))
    first_grid = "x".join([str(base_sub)] * dim)
    last_grid = "x".join([str(last_sub)] * dim)
    print(
        f"verification full_trace convergence, n-sub={first_grid} -> {last_grid} "
        f"({base_sub ** dim} -> {last_sub ** dim} 块)"
    )
    print(f"problem  {type(pde).__name__}, domain={domain_text}")
    print(f"model    {hypothesis}, E={pde.E:g}, nu={pde.nu:g}, rho=1")
    print(f"space    Q{degree}, shape=(-1,{dim}), float64, solver={solve_method}")
    print("boundary 零体力, 全外边界给定位移")

    headers = [
        "层级", "网格", "全自由度", "接口自由度", "位移L2误差", "阶", "位移H1半范误差", "阶",
    ]
    widths = [4, 12, 10, 10, 12, 4, 16, 4]

    def print_row(values: Sequence[str]) -> None:
        """逐层打印结果, 按中文显示宽度对齐并立即刷新."""
        print("  ".join(
            value + " " * max(0, width - display_width(value))
            for value, width in zip(values, widths, strict=True)
        ).rstrip(), flush=True)

    print()
    print_row(headers)

    level_results: List[Dict[str, Any]] = []
    for lvl in range(levels):
        sub_count = base_sub * (2**lvl)
        n_sub = tuple(sub_count for _ in range(dim))
        res = solve_one_level(
            pde, material, n_sub, n_fine, degree=degree, solve_method=solve_method
        )
        for metric in ("l2", "h1_semi"):
            order = None
            if lvl > 0:
                prev = level_results[-1]
                error_ratio = prev["full_trace"][f"{metric}_error"] / res["full_trace"][f"{metric}_error"]
                mesh_ratio = prev["mesh_size"] / res["mesh_size"]
                order = math.log(error_ratio) / math.log(mesh_ratio)
                if not math.isfinite(order):
                    raise AssertionError(f"full_trace {metric} 观测阶不是有限数.")
            res["full_trace"][f"{metric}_order"] = order
        level_results.append(res)
        values = res["full_trace"]
        print_row([
            str(lvl + 1),
            "x".join(map(str, res["total_fine"])),
            str(res["full_dofs"]),
            str(res["interface_dofs"]),
            f"{values['l2_error']:.4e}",
            "--" if values["l2_order"] is None else f"{values['l2_order']:.2f}",
            f"{values['h1_semi_error']:.4e}",
            "--" if values["h1_semi_order"] is None else f"{values['h1_semi_order']:.2f}",
        ])

    for metric, minimum in minimum_orders.items():
        final_order = level_results[-1]["full_trace"][f"{metric}_order"]
        if final_order is None or not math.isfinite(final_order) or final_order < minimum:
            raise AssertionError(
                f"full_trace 最细网格 {metric} 观测阶 {final_order} "
                f"低于门禁 {minimum:.2f} (理论 {expected_orders[metric]:.2f})."
            )

    summary = {
        "schema_version": "full-trace-convergence-v1",
        "dimension": f"{dim}D",
        "problem": type(pde).__name__,
        "domain": list(pde.domain),
        "model": model,
        "space_degree": degree,
        "material_hypothesis": material.hypothesis,
        "youngs_modulus": pde.E,
        "poisson_ratio": pde.nu,
        "base_subdivisions": base_sub,
        "refinement_levels": levels,
        "boundary_condition": "exact_displacement_interpolation_all_dirichlet",
        "density": 1.0,
        "assembly_method": "standard",
        "solver": solve_method,
        "error_quantity": "displacement",
        "backend": "numpy",
        "integration_order": max(4, degree + 3),
        "theoretical_orders": expected_orders,
        "minimum_order_gates": minimum_orders,
        "levels": level_results,
        "validation": {"full_trace_convergence": "PASS"},
        "passed": True,
    }

    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    final = level_results[-1]["full_trace"]
    print(
        f"\nPASS, 位移 L2 阶 {final['l2_order']:.4f}, "
        f"H1 半范阶 {final['h1_semi_order']:.4f}",
        flush=True,
    )
    if target is not None:
        default_dir = Path(__file__).resolve().parent / "outputs"
        display_path = (
            Path("outputs") / target.name
            if target.parent.resolve() == default_dir.resolve() else target.resolve()
        )
        print(f"结果：{display_path}", flush=True)

    return summary
