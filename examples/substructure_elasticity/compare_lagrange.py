"""
子结构静态缩聚 vs 传统拉格朗日有限元 (Lagrange FEM) 对比分析验证
=======================================================================
用于全方位校验子结构静态缩聚算法 (Substructure Static Condensation)
与标准拉格朗日有限元全装配求解器 (Full-Assembly Lagrange FEM) 的正确性与性能。

算例对应关系：
  * 3D MBB 梁 (run_benchmark_3d) : 1-to-1 完全精确对齐论文 Huang2023 4.1 节的 3D 六面体 MBB 梁实体算例
  * 2D MBB 梁 (run_benchmark_2d) : 对接 SOPTX 原生 HalfMBBBeamRight2d 对称半梁模型

作者: Liang He (大连理工大学博士后) & Antigravity Assistant
日期: 2026-08-07
"""

import os
import time
import argparse
import unicodedata
from typing import Tuple, List, Any
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm


def display_width(s: str) -> int:
    """计算字符串的显示宽度（考虑中文字符）"""
    width = 0
    for char in s:
        if unicodedata.east_asian_width(char) in ('F', 'W'):
            width += 2
        else:
            width += 1
    return width


def format_table_row(col1: str, col2: str, col3: str, widths: Tuple[int, int, int]) -> str:
    """格式化表格行，正确处理中文字符对齐"""
    w1, w2, w3 = widths
    pad1 = w1 - display_width(col1)
    pad2 = w2 - display_width(col2)
    pad3 = w3 - display_width(col3)
    return f"{col1}{' ' * pad1} | {col2}{' ' * pad2} | {col3}{' ' * pad3}"
from fealpy.mesh import QuadrangleMesh, HexahedronMesh
from soptx.fem.solvers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.topology.interpolation import MaterialInterpolationScheme

from substructure import SubstructureMesh, FEAStaticCondensation
from assembler import GlobalAssembler

from soptx.problems.elasticity.mbb import HalfMBBBeamRight2d, HalfMBBBeamRight3d, FullMBBBeam3d


def run_benchmark_2d() -> None:
    # 实例化 SOPTX 原生 HalfMBBBeamRight2d 物理问题模型 (60 mm x 20 mm 对称半梁)
    pde = HalfMBBBeamRight2d(domain=(0.0, 60.0, 0.0, 20.0), P=-1.0, E=1.0, nu=0.3)
    Lx, Ly = pde.domain[1] - pde.domain[0], pde.domain[3] - pde.domain[2]
    E_base, nu = pde.E, pde.nu

    n_sub = (6, 2)
    n_fine = (5, 5)
    total_fine = (n_sub[0] * n_fine[0], n_sub[1] * n_fine[1])

    # 1. 构造子结构与材料场
    dx_sub, dy_sub = Lx / n_sub[0], Ly / n_sub[1]
    sub_meshes: List[SubstructureMesh] = []
    densities: List[Any] = []
    exact_condensors: List[FEAStaticCondensation] = []

    for sx in range(n_sub[0]):
        for sy in range(n_sub[1]):
            sub_id = sx * n_sub[1] + sy
            x_span = (sx * dx_sub, (sx + 1) * dx_sub)
            y_span = (sy * dy_sub, (sy + 1) * dy_sub)
            sub_mesh = SubstructureMesh(sub_id, x_span, y_span, n_fine[0], n_fine[1], E_base, nu)
            sub_meshes.append(sub_mesh)

            xc = (x_span[0] + x_span[1]) / 2.0
            yc = (y_span[0] + y_span[1]) / 2.0
            rho = 0.7 + 0.3 * bm.sin(bm.pi * xc / Lx) * bm.cos(bm.pi * yc / Ly)
            rho_field = bm.full(n_fine, rho, dtype=bm.float64)
            densities.append(rho_field)

    # SOPTX HalfMBBBeam 荷载定义：左上角 (0, Ly) 施加向下集中荷载 P = -1.0
    lt_node = total_fine[1]  # 节点索引: x=0, y=total_fine[1]
    load_dof = 2 * lt_node + 1
    load_val = pde.P

    # --------------------------------------------------------------------------
    # 路径 A: 使用 LagrangeFEMAnalyzer 求解 (Lagrange FEM Full-Assembly)
    # --------------------------------------------------------------------------
    t0_lagrange = time.time()

    mesh_lagrange = QuadrangleMesh.from_box(box=[0, Lx, 0, Ly], nx=total_fine[0], ny=total_fine[1])
    total_dofs_2d = mesh_lagrange.number_of_nodes() * 2

    # 构建全局密度分布
    full_density_grid_2d = (
        bm.asarray(densities)
        .reshape(n_sub[0], n_sub[1], n_fine[0], n_fine[1])
        .transpose(0, 2, 1, 3)
        .reshape(total_fine[0], total_fine[1])
    )
    rho_2d = bm.asarray(full_density_grid_2d.flatten(), dtype=bm.float64)

    # 计算固定 DOF（用于表格显示"求解自由度规模"）
    node_coords_2d = mesh_lagrange.entity('node')
    eps = 1e-7
    left_nodes = [idx for idx, pt in enumerate(node_coords_2d) if abs(pt[0]) < eps]
    rb_nodes = [idx for idx, pt in enumerate(node_coords_2d) if abs(pt[0] - Lx) < eps and abs(pt[1]) < eps]
    fixed_dofs_2d_list = [2 * n for n in left_nodes] + [2 * n + 1 for n in rb_nodes]
    fixed_dofs_2d = bm.sort(bm.array(fixed_dofs_2d_list, dtype=bm.int64))
    free_dofs_lagrange = bm.setdiff1d(bm.arange(total_dofs_2d), fixed_dofs_2d)

    mat_lagrange = IsotropicLinearElasticMaterial(youngs_modulus=E_base, poisson_ratio=nu, hypothesis=pde.plane_type)
    interp_scheme = MaterialInterpolationScheme(
        density_location='element',
        interpolation_method='simp',
        options={'penalty_factor': 3.0, 'stress_penalty_factor': 1.0}
    )

    analyzer_2d = LagrangeFEMAnalyzer(
        disp_mesh=mesh_lagrange,
        pde=pde,
        material=mat_lagrange,
        space_degree=1,
        assembly_method='standard',
        operator_level='fa',
        solve_method='scipy',
        topopt_algorithm='density_based',
        interpolation_scheme=interp_scheme
    )

    # 使用密度分布求解
    solution_dict = analyzer_2d.solve_state(rho_val=rho_2d)
    uh = solution_dict['displacement']
    U_lagrange = bm.to_numpy(uh.data)
    if U_lagrange.ndim > 1:
        U_lagrange = U_lagrange.flatten()

    # 计算柔度：C = F^T * u
    F_lagrange = bm.zeros(total_dofs_2d, dtype=bm.float64)
    F_lagrange[load_dof] = load_val
    C_lagrange = float(F_lagrange @ U_lagrange)

    t_lagrange = time.time() - t0_lagrange

    # --------------------------------------------------------------------------
    # 路径 B: 子结构静态缩聚有限元求解 (Substructure Static Condensation)
    # --------------------------------------------------------------------------
    t0_sub = time.time()

    assembler = GlobalAssembler(Lx, Ly, n_sub[0], n_sub[1], n_fine[0], n_fine[1], E_base, nu)
    for sub_mesh, rho_field in zip(sub_meshes, densities):
        condensor = FEAStaticCondensation(sub_mesh.i_dofs, sub_mesh.b_dofs)
        K_local = sub_mesh.assemble_local_stiffness(rho_field)
        condensor.condense(K_local)
        exact_condensors.append(condensor)

    # 真缩聚求解：接口系统装配与求解
    U_sub_full, interface_free_2d = assembler.solve_condensed_fea(
        densities, sub_meshes, exact_condensors,
        load_dof, load_val, bc_type="mbb"
    )

    t_sub = time.time() - t0_sub
    C_sub = float(F_lagrange @ U_sub_full)

    # --------------------------------------------------------------------------
    # 指标交叉校验统计
    # --------------------------------------------------------------------------
    rel_err_u = float(bm.linalg.norm(U_sub_full - U_lagrange) / bm.linalg.norm(U_lagrange))
    rel_err_c = abs(C_sub - C_lagrange) / abs(C_lagrange)

    print("\n【2D MBB 梁】子结构静态缩聚 VS 传统拉格朗日有限元")
    print("=" * 120)

    widths = (50, 25, 30)
    header = format_table_row("对比评估指标", "拉格朗日有限元 (Lagrange)", "子结构静态缩聚 (Substructure)", widths)
    print(header)
    print("-" * 120)

    print(format_table_row("全网格自由度规模 (Global DOFs)",
                         str(total_dofs_2d),
                         str(assembler.total_full_dofs), widths))
    print(format_table_row("求解自由度规模 (Solvable DOFs)",
                         str(len(free_dofs_lagrange)),
                         str(len(interface_free_2d)), widths))
    print(format_table_row("结构总柔度 (Compliance / Strain Energy)",
                         f"{C_lagrange:.8f}",
                         f"{C_sub:.8f}", widths))
    print(format_table_row("柔度相对误差 (Compliance Rel Error)",
                         "--",
                         f"{rel_err_c:.4e}", widths))
    print(format_table_row("位移场全节点相对误差 (Displacement Rel Error)",
                         "--",
                         f"{rel_err_u:.4e}", widths))


def run_benchmark_3d() -> None:
    # 实例化 SOPTX 原生 FullMBBBeam3d 物理问题模型 (6.0 x 1.0 x 1.0 完整实体梁，无对称简化)
    pde3d = FullMBBBeam3d(domain=(0.0, 6.0, 0.0, 1.0, 0.0, 1.0), P=-1.0, E=1.0, nu=0.3)
    Lx = pde3d.domain[1] - pde3d.domain[0]
    Ly = pde3d.domain[3] - pde3d.domain[2]
    Lz = pde3d.domain[5] - pde3d.domain[4]
    E_base, nu = pde3d.E, pde3d.nu

    n_sub = (6, 2, 2)
    n_fine = (4, 4, 4)
    total_fine = (n_sub[0] * n_fine[0], n_sub[1] * n_fine[1], n_sub[2] * n_fine[2])

    dx_sub, dy_sub, dz_sub = Lx / n_sub[0], Ly / n_sub[1], Lz / n_sub[2]
    sub_meshes: List[SubstructureMesh] = []
    densities: List[Any] = []
    exact_condensors: List[FEAStaticCondensation] = []

    for sx in range(n_sub[0]):
        for sy in range(n_sub[1]):
            for sz in range(n_sub[2]):
                sub_id = (sx * n_sub[1] + sy) * n_sub[2] + sz
                x_span = (sx * dx_sub, (sx + 1) * dx_sub)
                y_span = (sy * dy_sub, (sy + 1) * dy_sub)
                z_span = (sz * dz_sub, (sz + 1) * dz_sub)
                sub_mesh = SubstructureMesh(sub_id, x_span, y_span, z_span, n_fine[0], n_fine[1], n_fine[2], E_base, nu)
                sub_meshes.append(sub_mesh)

                xc = (x_span[0] + x_span[1]) / 2.0
                yc = (y_span[0] + y_span[1]) / 2.0
                zc = (z_span[0] + z_span[1]) / 2.0
                rho = 0.7 + 0.3 * bm.sin(bm.pi * xc / Lx) * bm.cos(bm.pi * yc / Ly) * bm.sin(bm.pi * zc / Lz)
                rho_field = bm.full(n_fine, rho, dtype=bm.float64)
                densities.append(rho_field)

    # --------------------------------------------------------------------------
    # 路径 A: 使用 LagrangeFEMAnalyzer 求解 (3D Lagrange FEM)
    # --------------------------------------------------------------------------
    t0_lagrange = time.time()

    mesh_lagrange_3d = HexahedronMesh.from_box(box=[0, Lx, 0, Ly, 0, Lz], nx=total_fine[0], ny=total_fine[1], nz=total_fine[2])
    total_dofs_3d = mesh_lagrange_3d.number_of_nodes() * 3

    # SOPTX 3D FullMBBBeam 荷载定义：顶面中心点 (Lx/2, Ly, Lz/2) 施加向下集中荷载 P = -1.0
    # 从网格节点中查找加载点
    node_coords_3d = mesh_lagrange_3d.entity('node')
    eps = 1e-7
    target_coord = bm.array([Lx / 2.0, Ly, Lz / 2.0])
    distances = bm.linalg.norm(node_coords_3d - target_coord, axis=1)
    top_center_node = bm.argmin(distances)
    load_dof = 3 * top_center_node + 1  # y 向向下集中载荷
    load_val = pde3d.P

    # 构建全局密度分布
    full_density_grid_3d = (
        bm.asarray(densities)
        .reshape(n_sub[0], n_sub[1], n_sub[2], n_fine[0], n_fine[1], n_fine[2])
        .transpose(0, 3, 1, 4, 2, 5)
        .reshape(total_fine[0], total_fine[1], total_fine[2])
    )
    rho_3d = bm.asarray(full_density_grid_3d.flatten(), dtype=bm.float64)

    # 计算固定 DOF（用于表格显示"求解自由度规模"）-- 与 FullMBBBeam3d 边界条件精确对齐:
    #   ux=0 at (x=0, y=0);  uy=0 at (y=0) & (x=0 | x=Lx);  uz=0 at (y=0, z=Lz/2)
    z_mid_3d = Lz / 2.0
    fixed_dofs_3d = bm.sort(bm.concat([
        bm.array([3 * n + 0 for n, pt in enumerate(node_coords_3d)
                  if abs(pt[0]) < eps and abs(pt[1]) < eps], dtype=bm.int64),
        bm.array([3 * n + 1 for n, pt in enumerate(node_coords_3d)
                  if (abs(pt[0]) < eps or abs(pt[0] - Lx) < eps) and abs(pt[1]) < eps], dtype=bm.int64),
        bm.array([3 * n + 2 for n, pt in enumerate(node_coords_3d)
                  if abs(pt[1]) < eps and abs(pt[2] - z_mid_3d) < eps], dtype=bm.int64),
    ]))
    free_dofs_lagrange_3d = bm.setdiff1d(bm.arange(total_dofs_3d), fixed_dofs_3d)

    mat_lagrange_3d = IsotropicLinearElasticMaterial(youngs_modulus=E_base, poisson_ratio=nu)
    interp_scheme_3d = MaterialInterpolationScheme(
        density_location='element',
        interpolation_method='simp',
        options={'penalty_factor': 3.0, 'stress_penalty_factor': 1.0}
    )

    analyzer_3d = LagrangeFEMAnalyzer(
        disp_mesh=mesh_lagrange_3d,
        pde=pde3d,
        material=mat_lagrange_3d,
        space_degree=1,
        assembly_method='standard',
        operator_level='fa',
        solve_method='scipy',
        topopt_algorithm='density_based',
        interpolation_scheme=interp_scheme_3d
    )

    # 使用密度分布求解
    solution_dict = analyzer_3d.solve_state(rho_val=rho_3d)
    uh = solution_dict['displacement']
    U_lagrange_3d = bm.to_numpy(uh.data)
    if U_lagrange_3d.ndim > 1:
        U_lagrange_3d = U_lagrange_3d.flatten()

    # 计算柔度：C = F^T * u
    F_lagrange_3d = bm.zeros(total_dofs_3d, dtype=bm.float64)
    F_lagrange_3d[load_dof] = load_val
    C_lagrange_3d = float(F_lagrange_3d @ U_lagrange_3d)

    t_lagrange_3d = time.time() - t0_lagrange

    # --------------------------------------------------------------------------
    # 路径 B: 子结构静态缩聚有限元求解 (3D Substructure Condensation)
    # --------------------------------------------------------------------------
    t0_sub_3d = time.time()

    assembler_3d = GlobalAssembler((Lx, Ly, Lz), n_sub, n_fine, E_base, nu)
    for sub_mesh, rho_field in zip(sub_meshes, densities):
        condensor = FEAStaticCondensation(sub_mesh.i_dofs, sub_mesh.b_dofs)
        K_local = sub_mesh.assemble_local_stiffness(rho_field)
        condensor.condense(K_local)
        exact_condensors.append(condensor)

    # 真缩聚求解：接口系统装配与求解
    U_sub_3d, interface_free_3d = assembler_3d.solve_condensed_fea(
        densities, sub_meshes, exact_condensors,
        load_dof, load_val, bc_type="mbb"
    )

    t_sub_3d = time.time() - t0_sub_3d
    C_sub_3d = float(F_lagrange_3d @ U_sub_3d)

    # --------------------------------------------------------------------------
    # 指标交叉校验统计
    # --------------------------------------------------------------------------
    rel_err_u_3d = float(bm.linalg.norm(U_sub_3d - U_lagrange_3d) / bm.linalg.norm(U_lagrange_3d))
    rel_err_c_3d = abs(C_sub_3d - C_lagrange_3d) / abs(C_lagrange_3d)

    print("\n【3D MBB 梁】子结构静态缩聚 VS 传统拉格朗日有限元")
    print("=" * 120)

    widths = (50, 25, 30)
    header = format_table_row("对比评估指标", "拉格朗日有限元 (Lagrange)", "子结构静态缩聚 (Substructure)", widths)
    print(header)
    print("-" * 120)

    print(format_table_row("全网格自由度规模 (Global DOFs)",
                         str(total_dofs_3d),
                         str(assembler_3d.total_full_dofs), widths))
    print(format_table_row("求解自由度规模 (Solvable DOFs)",
                         str(len(free_dofs_lagrange_3d)),
                         str(len(interface_free_3d)), widths))
    print(format_table_row("结构总柔度 (Compliance / Strain Energy)",
                         f"{C_lagrange_3d:.8f}",
                         f"{C_sub_3d:.8f}", widths))
    print(format_table_row("柔度相对误差 (Compliance Rel Error)",
                         "--",
                         f"{rel_err_c_3d:.4e}", widths))
    print(format_table_row("位移场全节点相对误差 (Displacement Rel Error)",
                         "--",
                         f"{rel_err_u_3d:.4e}", widths))



def main() -> None:
    parser = argparse.ArgumentParser(description="子结构静态缩聚 VS 传统拉格朗日有限元全方位对比基准")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2, help="计算维度 (2D 或 3D)")
    args = parser.parse_args()

    if args.dim == 2:
        run_benchmark_2d()
    else:
        run_benchmark_3d()


if __name__ == "__main__":
    main()
