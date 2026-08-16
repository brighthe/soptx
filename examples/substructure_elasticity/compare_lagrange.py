"""子结构静力缩聚正确性验证入口.

本脚本在同一物理问题, 同一材料插值与同一密度场下比较两条求解路径: ``LagrangeFEMAnalyzer``
全装配直接求解全尺度细网格, 与子结构 Schur 补缩聚只求解接口系统再回代内部位移. 两条路径
的外载向量与约束自由度都取自同一个问题对象, 因此一致性是结构性的而非巧合; 缩聚在代数上
精确, 柔度与全节点位移的相对误差应落在机器精度内. 二维取 ``HalfMBBBeamRight2d`` 对称半梁,
三维取 ``FullMBBBeam3d`` 完整实体梁, 后者对齐 Huang 2023 第 4.1 节.

验收阈值 ``RELATIVE_ERROR_TOLERANCE`` 以运行时断言实现: 超出即抛异常且不写任何文件,
通过才落盘 JSON 证据. 契约与实测证据见同目录 ``results_analysis.md``.

使用方法:
    # 2D 对称半 MBB 梁.
    python examples/substructure_elasticity/compare_lagrange.py --dim 2

    # 3D 完整 MBB 梁.
    python examples/substructure_elasticity/compare_lagrange.py --dim 3

``--output-dir`` 缺省为本脚本同级的 ``outputs/``, 按脚本位置解析, 与从哪个目录发起命令无关;
传相对路径时按当前工作目录解析, 可能落到 ``.gitignore`` 覆盖范围之外.
"""

import json
import time
import argparse
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

from fealpy.backend import backend_manager as bm

from soptx.fem.solvers import LagrangeFEMAnalyzer
from soptx.problems.elasticity import HalfMBBBeamRight2d, FullMBBBeam3d
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.fem.substructure import (
    FEAStaticCondensation,
    GlobalAssembler,
    SubstructureMesh,
    SubstructurePrototype,
    solve_interface_system,
)


RELATIVE_ERROR_TOLERANCE = 1.0e-12


### 表格排版 ###

def display_width(s: str) -> int:
    """计算字符串在等宽终端下的显示宽度, 东亚全角字符按两列计."""
    width = 0
    for char in s:
        width += 2 if unicodedata.east_asian_width(char) in ('F', 'W') else 1
    return width


def format_table_row(
    col1: str, col2: str, col3: str, widths: Tuple[int, int, int]
) -> str:
    """按显示宽度对齐三列表格行."""
    return (
        f"{col1}{' ' * (widths[0] - display_width(col1))} | "
        f"{col2}{' ' * (widths[1] - display_width(col2))} | "
        f"{col3}{' ' * (widths[2] - display_width(col3))}"
    )


### 验收 ###

def validate_and_write_result(result: Dict[str, Any], output_dir: str | None) -> None:
    """以统一阈值验收两条求解路径, 并可选地落盘机器可读证据.

    参数:
        result: 单个算例的全部统计指标.
        output_dir: 证据输出目录; 为 ``None`` 时只验收不落盘.

    异常:
        AssertionError: 当柔度或位移的相对误差超出 ``RELATIVE_ERROR_TOLERANCE``
            时抛出.
    """
    for key, label in (
        ("compliance_relative_error", "compliance"),
        ("displacement_relative_error", "displacement"),
    ):
        if result[key] > RELATIVE_ERROR_TOLERANCE:
            raise AssertionError(
                f"{result['dimension']} {label} relative error "
                f"{result[key]:.4e} exceeds {RELATIVE_ERROR_TOLERANCE:.1e}"
            )

    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"lagrange_comparison_{result['dimension'].lower()}.json"
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[证据] 验收通过, 结果已写入: {target}")


### 子结构构造与密度场 ###

def build_substructures(
    assembler: GlobalAssembler,
) -> Tuple[SubstructurePrototype, List[SubstructureMesh], List[Tuple[int, ...]]]:
    """按装配器的布局铺开全部子结构, 共享同一个参考子结构.

    参数:
        assembler: 已构造的全局装配器, 提供求解域尺寸与子结构划分.

    返回:
        (prototype, sub_meshes, positions): 共享的参考子结构, 子结构列表, 以及
            各子结构在子结构网格中的整数位置, 三者按 x 优先的字典序同序排列.

    说明:
        字典序是 ``reconstruct_global_field`` 的次序契约; 涉及网格的装配接口则
        由 ``box_span`` 反解位置, 不依赖列表次序.
    """
    dim = assembler.dim
    sub_size = tuple(
        assembler.domain_size[d] / assembler.n_sub[d] for d in range(dim)
    )
    prototype = SubstructurePrototype(
        sub_size, assembler.n_fine, assembler.E_base, assembler.nu
    )

    grid = (
        [(sx, sy) for sx in range(assembler.n_sub[0]) for sy in range(assembler.n_sub[1])]
        if dim == 2
        else [
            (sx, sy, sz)
            for sx in range(assembler.n_sub[0])
            for sy in range(assembler.n_sub[1])
            for sz in range(assembler.n_sub[2])
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
    return prototype, sub_meshes, grid


def make_density_fields(
    sub_meshes: Sequence[SubstructureMesh],
    domain_size: Sequence[float],
) -> Any:
    """按子结构中心坐标生成一批平滑变化的局部密度场.

    参数:
        sub_meshes: 子结构列表.
        domain_size: 各方向的求解域尺寸.

    返回:
        density: 形状 ``(B, *n_fine)`` 的批量密度场, 前导维 ``B`` 与
            ``sub_meshes`` 同序; 每个子结构内部密度均匀.
    """
    dim = len(domain_size)
    centers = bm.asarray(
        [
            [(sm.box_span[d][0] + sm.box_span[d][1]) / 2.0 for d in range(dim)]
            for sm in sub_meshes
        ],
        dtype=bm.float64,
    )
    scaled = bm.pi * centers / bm.asarray(domain_size, dtype=bm.float64)
    modulation = bm.sin(scaled[:, 0]) * bm.cos(scaled[:, 1])
    if dim == 3:
        modulation = modulation * bm.sin(scaled[:, 2])
    rho = 0.7 + 0.3 * modulation

    n_fine = tuple(sub_meshes[0].n_fine)
    rho = bm.reshape(rho, (len(sub_meshes),) + (1,) * dim)
    return bm.broadcast_to(rho, (len(sub_meshes),) + n_fine)


### 基准主体 ###

def run_benchmark(dim: int, output_dir: str | None = None) -> Dict[str, Any]:
    """运行给定维数的两路径对比基准.

    参数:
        dim: 空间维数, 取 ``2`` 或 ``3``.
        output_dir: 证据输出目录; 为 ``None`` 时只验收不落盘.

    返回:
        result: 该算例的全部统计指标.

    异常:
        AssertionError: 当两条路径的相对误差超出验收阈值时抛出.
    """
    if dim == 2:
        pde: Any = HalfMBBBeamRight2d(
            domain=(0.0, 60.0, 0.0, 20.0), P=-1.0, E=1.0, nu=0.3
        )
        n_sub, n_fine = (6, 2), (5, 5)
    else:
        pde = FullMBBBeam3d(
            domain=(0.0, 6.0, 0.0, 1.0, 0.0, 1.0), P=-1.0, E=1.0, nu=0.3
        )
        n_sub, n_fine = (6, 2, 2), (4, 4, 4)

    domain_size = tuple(
        pde.domain[2 * d + 1] - pde.domain[2 * d] for d in range(dim)
    )
    assembler = GlobalAssembler(
        domain_size, n_sub, n_fine, E_base=pde.E, nu=pde.nu
    )

    # 子结构与密度场为两条路径共用: 全尺度密度由局部密度拼接而成, 保证两边
    # 求解的是同一个材料分布.
    prototype, sub_meshes, _ = build_substructures(assembler)
    density = make_density_fields(sub_meshes, domain_size)
    rho_global = assembler.reconstruct_global_field(density)

    ### 路径 A: 拉格朗日有限元全装配直接求解 ###
    t0 = time.time()

    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=assembler.full_mesh,
        pde=pde,
        material=assembler.material,
        space_degree=1,
        assembly_method='standard',
        operator_level='fa',
        solve_method='scipy',
        topopt_algorithm='density_based',
        interpolation_scheme=MaterialInterpolationScheme(
            density_location='element',
            interpolation_method='simp',
            options={'penalty_factor': 3.0, 'stress_penalty_factor': 1.0},
        ),
        enable_logging=False,
    )
    solution = analyzer.solve_state(rho_val=bm.reshape(rho_global, (-1,)))
    U_lagrange = bm.reshape(bm.asarray(solution['displacement']), (-1,))

    t_lagrange = time.time() - t0

    # 外载与约束都取自分析器: force_vector 是施加 Dirichlet 条件之前的外载向量,
    # 固定自由度由物理问题自己的边界判据插值得到. 两条路径据此共用同一份定义.
    F_ext = bm.asarray(analyzer.force_vector, dtype=bm.float64)
    _, fixed_mask = analyzer.tensor_space.boundary_interpolate(
        gd=pde.dirichlet_bc,
        threshold=cast(Any, pde.is_dirichlet_boundary()),
        method='interp',
    )
    fixed_global_dofs = bm.nonzero(fixed_mask)[0]
    n_free_lagrange = assembler.total_full_dofs - int(len(fixed_global_dofs))

    C_lagrange = float(bm.sum(F_ext * U_lagrange))

    ### 路径 B: 子结构静力缩聚 ###
    t0 = time.time()

    # 全部子结构共用一套离散结构, 局部刚度与 Schur 补各只需一次批量调用.
    K_local_batch = prototype.assemble_local_stiffness_batch(density)
    condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    condensor.condense(K_local_batch)

    system = assembler.assemble_interface_system(sub_meshes, condensor)
    interface_displacement = solve_interface_system(
        system,
        assembler.project_global_vector(system, F_ext),
        assembler.project_global_dofs(system, fixed_global_dofs),
    )
    U_sub = assembler.recover_full_displacement(
        sub_meshes, condensor, system, interface_displacement
    )

    t_sub = time.time() - t0

    n_interface = int(len(system.global_dofs))
    n_free_interface = n_interface - int(
        len(assembler.project_global_dofs(system, fixed_global_dofs))
    )
    C_sub = float(bm.sum(F_ext * U_sub))

    ### 交叉校验 ###
    rel_err_u = float(
        bm.linalg.norm(U_sub - U_lagrange) / bm.linalg.norm(U_lagrange)
    )
    rel_err_c = abs(C_sub - C_lagrange) / abs(C_lagrange)

    label = f"{dim}D {type(pde).__name__}"
    print(f"\n【{label}】子结构静力缩聚 VS 传统拉格朗日有限元")
    print("=" * 120)
    widths = (50, 25, 30)
    print(format_table_row(
        "对比评估指标", "拉格朗日有限元 (Lagrange)",
        "子结构静力缩聚 (Substructure)", widths,
    ))
    print("-" * 120)
    print(format_table_row("全网格自由度规模 (Global DOFs)",
                           str(assembler.total_full_dofs),
                           str(assembler.total_full_dofs), widths))
    print(format_table_row("求解自由度规模 (Solvable DOFs)",
                           str(n_free_lagrange), str(n_free_interface), widths))
    print(format_table_row("结构总柔度 (Compliance / Strain Energy)",
                           f"{C_lagrange:.8f}", f"{C_sub:.8f}", widths))
    print(format_table_row("柔度相对误差 (Compliance Rel Error)",
                           "--", f"{rel_err_c:.4e}", widths))
    print(format_table_row("位移场全节点相对误差 (Displacement Rel Error)",
                           "--", f"{rel_err_u:.4e}", widths))
    print(format_table_row("求解耗时 (Solve Time, s)",
                           f"{t_lagrange:.4f}", f"{t_sub:.4f}", widths))

    result = {
        "dimension": f"{dim}D",
        "problem": type(pde).__name__,
        "n_sub": list(n_sub),
        "n_fine": list(n_fine),
        "full_dofs": int(assembler.total_full_dofs),
        "lagrange_free_dofs": int(n_free_lagrange),
        "interface_dofs": n_interface,
        "condensed_interface_free_dofs": int(n_free_interface),
        "lagrange_compliance": C_lagrange,
        "condensed_compliance": C_sub,
        "compliance_relative_error": rel_err_c,
        "displacement_relative_error": rel_err_u,
        "lagrange_seconds": float(t_lagrange),
        "condensed_seconds": float(t_sub),
        "relative_error_tolerance": RELATIVE_ERROR_TOLERANCE,
    }
    validate_and_write_result(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="子结构静力缩聚 VS 传统拉格朗日有限元对比基准"
    )
    parser.add_argument(
        "--dim", type=int, choices=[2, 3], default=2, help="计算维度 (2D 或 3D)"
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).with_name("outputs")),
        help="通过验收后写入 JSON 结果的目录",
    )
    args = parser.parse_args()

    run_benchmark(args.dim, args.output_dir)


if __name__ == "__main__":
    main()
