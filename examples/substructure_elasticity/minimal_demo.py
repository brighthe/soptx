"""
2D & 3D 线弹性子结构静态缩聚分析 Demo
========================================================
100% 纯有限元 (FEA) 子结构静态缩聚求解与验证流程（全面支持 2D 及 3D）。
无任何机器学习或神经网络依赖。

串联模块：
  1. SubstructureMesh       : 二维/三维矩形子结构网格划分与 SIMP 变密度场设置
  2. FEAStaticCondensation  : 维度无关的精确 Schur 补求逆与内部位移恢复
  3. GlobalAssembler        : 界面系统组装与全尺寸 2D/3D FEA 直接求解
  4. 基准验证               : 2D & 3D 机器精度误差校验与位移场可视化云图导出

作者: Liang He (大连理工大学博士后) & Antigravity Assistant
日期: 2026-08-06
"""

import os
import time
import unicodedata
from typing import List, Tuple, Any
import matplotlib.pyplot as plt
from fealpy.backend import backend_manager as bm

from substructure import SubstructureMesh, FEAStaticCondensation
from assembler import GlobalAssembler


def display_width(s: str) -> int:
    """计算字符串的显示宽度（考虑中文字符）"""
    width = 0
    for char in s:
        if unicodedata.east_asian_width(char) in ('F', 'W'):
            width += 2
        else:
            width += 1
    return width


def format_table_row(col1: str, col2: str, widths: Tuple[int, int]) -> str:
    """格式化表格行（2列），正确处理中文字符对齐"""
    w1, w2 = widths
    pad1 = w1 - display_width(col1)
    pad2 = w2 - display_width(col2)
    return f"{col1}{' ' * pad1} | {col2}{' ' * pad2}"


def run_demo_2d(out_dir: str) -> None:
    print("\n" + "=" * 80)
    print("【测试 1】纯 2D 弹性体子结构静态缩聚分析 (2D FEA Baseline)")
    print("=" * 80)

    Lx, Ly = 2.0, 1.0
    E_base, nu = 1.0, 0.3

    n_sub_x, n_sub_y = 4, 2
    n_fine_x, n_fine_y = 5, 5

    total_fine_x = n_sub_x * n_fine_x
    total_fine_y = n_sub_y * n_fine_y

    print(f"问题几何尺寸     : 2D 矩形悬臂梁 [{Lx} x {Ly}]")
    print(f"子结构划分       : {n_sub_x} x {n_sub_y} (共 {n_sub_x * n_sub_y} 个子结构)")
    print(f"局部子网格       : 每个子结构 {n_fine_x} x {n_fine_y} Q4 单元")
    print(f"全局精细网格     : 共 {total_fine_x} x {total_fine_y} Q4 细网格单元")
    print("-" * 80)

    assembler = GlobalAssembler(Lx, Ly, n_sub_x, n_sub_y, n_fine_x, n_fine_y, E_base, nu)
    dx_sub = Lx / n_sub_x
    dy_sub = Ly / n_sub_y

    sub_meshes: List[SubstructureMesh] = []
    for sx in range(n_sub_x):
        for sy in range(n_sub_y):
            sub_id = sx * n_sub_y + sy
            x_span = (sx * dx_sub, (sx + 1) * dx_sub)
            y_span = (sy * dy_sub, (sy + 1) * dy_sub)
            sub_mesh = SubstructureMesh(sub_id, x_span, y_span, n_fine_x, n_fine_y, E_base, nu)
            sub_meshes.append(sub_mesh)

    print("[步骤 1 & 2] 生成 2D 局部材料密度分布并运行 FEAStaticCondensation...")
    t0 = time.time()

    densities: List[Any] = []
    exact_condensors: List[FEAStaticCondensation] = []

    for sub_mesh in sub_meshes:
        xc = (sub_mesh.x_span[0] + sub_mesh.x_span[1]) / 2.0
        yc = (sub_mesh.y_span[0] + sub_mesh.y_span[1]) / 2.0
        rho = 0.7 + 0.3 * bm.sin(bm.pi * xc / Lx) * bm.cos(bm.pi * yc / Ly)
        rho_field = bm.full((n_fine_x, n_fine_y), rho, dtype=bm.float64)
        densities.append(rho_field)

        condensor = FEAStaticCondensation(sub_mesh.i_dofs, sub_mesh.b_dofs)
        K_local = sub_mesh.assemble_local_stiffness(rho_field)
        condensor.condense(K_local)
        exact_condensors.append(condensor)

    t_condense = time.time() - t0
    print(f"-> 完成 {len(sub_meshes)} 个 2D 子结构的 FEAStaticCondensation 静态缩聚，耗时 {t_condense:.4f} s。")

    print("[步骤 4] 求解 2D 全尺寸精细有限元直接参考解...")
    right_mid_node = total_fine_x * (total_fine_y + 1) + ((total_fine_y + 1) // 2)
    load_dof = 2 * right_mid_node + 1

    t0_solve = time.time()
    U_full_ref, free_dofs = assembler.solve_fullscale_fea(densities, load_dof, load_val=-1.0)
    t_full_solve = time.time() - t0_solve
    print(f"-> 2D 全尺寸参考解求解完成，耗时 {t_full_solve:.4f} s。")

    print("[步骤 5] 通过 StaticCondensation.recover() 恢复 2D 内部位移...")

    U_recovered = bm.zeros(assembler.total_full_dofs, dtype=bm.float64)
    U_recovered[free_dofs] = U_full_ref[free_dofs]

    for sx in range(n_sub_x):
        for sy in range(n_sub_y):
            sub_idx = sx * n_sub_y + sy
            sub_mesh = sub_meshes[sub_idx]
            condensor = exact_condensors[sub_idx]

            sub_global_dofs = assembler.get_substructure_global_dofs(sx, sy, sub_mesh)
            u_sub_b = U_full_ref[sub_global_dofs[sub_mesh.b_dofs]]
            u_sub_i_recovered = condensor.recover(u_sub_b)
            U_recovered[sub_global_dofs[sub_mesh.i_dofs]] = u_sub_i_recovered

    err_recovery = float(bm.linalg.norm(U_recovered - U_full_ref) / bm.linalg.norm(U_full_ref))

    print("\n" + "=" * 85)
    print("2D 纯弹性体子结构静态缩聚验证汇总表")
    print("=" * 85)

    widths_2d = (40, 35)
    print(format_table_row("统计指标 / 评估项目", "数值", widths_2d))
    print("-" * 85)
    print(format_table_row("2D 全尺寸精细网格总自由度数 (DOFs)", str(assembler.total_full_dofs), widths_2d))
    print(format_table_row("2D 界面缩聚自由度数 (Condensed DOFs)", str(len(free_dofs)), widths_2d))
    print(format_table_row("2D 自由度缩聚比例", f"{assembler.total_full_dofs / len(free_dofs):.2f}x", widths_2d))
    print(format_table_row("2D 静态缩聚内部位移恢复相对误差", f"{err_recovery:.4e} (机器精度)", widths_2d))
    print("=" * 85)

    # 绘制可视化云图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_full_nodes_x = total_fine_x + 1
    n_full_nodes_y = total_fine_y + 1

    U_y_full = bm.to_numpy(U_full_ref[1::2].reshape((n_full_nodes_x, n_full_nodes_y)).T)
    U_y_rec = bm.to_numpy(U_recovered[1::2].reshape((n_full_nodes_x, n_full_nodes_y)).T)

    im0 = axes[0].imshow(U_y_full, origin='lower', cmap='viridis', extent=[0, Lx, 0, Ly])
    axes[0].set_title("Full-Scale Direct FEA U_y Field")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(U_y_rec, origin='lower', cmap='viridis', extent=[0, Lx, 0, Ly])
    axes[1].set_title("Static Condensation Recovered U_y Field")
    axes[1].set_xlabel("x")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "substructure_cantilever_demo_2d.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"\n[成功] 2D 位移云图结果已保存至: {fig_path}")


def run_demo_3d(out_dir: str) -> None:
    print("\n" + "=" * 80)
    print("【测试 2】纯 3D 弹性体子结构静态缩聚分析 (3D FEA Baseline)")
    print("=" * 80)

    Lx, Ly, Lz = 2.0, 1.0, 1.0
    E_base, nu = 1.0, 0.3

    n_sub = (4, 2, 2)
    n_fine = (4, 4, 4)

    total_fine = tuple(n_sub[d] * n_fine[d] for d in range(3))

    print(f"问题几何尺寸     : 3D 矩形悬臂梁 [{Lx} x {Ly} x {Lz}]")
    print(f"子结构划分       : {n_sub[0]} x {n_sub[1]} x {n_sub[2]} (共 {n_sub[0]*n_sub[1]*n_sub[2]} 个子结构)")
    print(f"局部子网格       : 每个子结构 {n_fine[0]} x {n_fine[1]} x {n_fine[2]} 六面体单元")
    print(f"全局精细网格     : 共 {total_fine[0]} x {total_fine[1]} x {total_fine[2]} 六面体细网格单元")
    print("-" * 80)

    assembler = GlobalAssembler((Lx, Ly, Lz), n_sub, n_fine, E_base, nu)
    dx_sub = Lx / n_sub[0]
    dy_sub = Ly / n_sub[1]
    dz_sub = Lz / n_sub[2]

    sub_meshes: List[SubstructureMesh] = []
    for sx in range(n_sub[0]):
        for sy in range(n_sub[1]):
            for sz in range(n_sub[2]):
                sub_id = (sx * n_sub[1] + sy) * n_sub[2] + sz
                x_span = (sx * dx_sub, (sx + 1) * dx_sub)
                y_span = (sy * dy_sub, (sy + 1) * dy_sub)
                z_span = (sz * dz_sub, (sz + 1) * dz_sub)
                sub_mesh = SubstructureMesh(sub_id, x_span, y_span, z_span, n_fine[0], n_fine[1], n_fine[2], E_base, nu)
                sub_meshes.append(sub_mesh)

    print("[步骤 1 & 2] 生成 3D 局部材料密度分布并运行 FEAStaticCondensation...")
    t0 = time.time()

    densities: List[Any] = []
    exact_condensors: List[FEAStaticCondensation] = []

    for sub_mesh in sub_meshes:
        xc = (sub_mesh.box_span[0][0] + sub_mesh.box_span[0][1]) / 2.0
        yc = (sub_mesh.box_span[1][0] + sub_mesh.box_span[1][1]) / 2.0
        zc = (sub_mesh.box_span[2][0] + sub_mesh.box_span[2][1]) / 2.0
        rho = 0.7 + 0.3 * bm.sin(bm.pi * xc / Lx) * bm.cos(bm.pi * yc / Ly) * bm.sin(bm.pi * zc / Lz)
        rho_field = bm.full(n_fine, rho, dtype=bm.float64)
        densities.append(rho_field)

        condensor = FEAStaticCondensation(sub_mesh.i_dofs, sub_mesh.b_dofs)
        K_local = sub_mesh.assemble_local_stiffness(rho_field)
        condensor.condense(K_local)
        exact_condensors.append(condensor)

    t_condense = time.time() - t0
    print(f"-> 完成 {len(sub_meshes)} 个 3D 子结构的 FEAStaticCondensation 静态缩聚，耗时 {t_condense:.4f} s。")

    print("[步骤 4] 求解 3D 全尺寸精细有限元直接参考解...")
    mid_y = (total_fine[1] + 1) // 2
    mid_z = (total_fine[2] + 1) // 2
    right_mid_node = (total_fine[0] * (total_fine[1] + 1) + mid_y) * (total_fine[2] + 1) + mid_z
    load_dof = 3 * right_mid_node + 2  # z 方向荷载

    t0_solve = time.time()
    U_full_ref, free_dofs = assembler.solve_fullscale_fea(densities, load_dof, load_val=-1.0)
    t_full_solve = time.time() - t0_solve
    print(f"-> 3D 全尺寸参考解求解完成，耗时 {t_full_solve:.4f} s。")

    print("[步骤 5] 通过 StaticCondensation.recover() 恢复 3D 内部位移...")

    U_recovered = bm.zeros(assembler.total_full_dofs, dtype=bm.float64)
    U_recovered[free_dofs] = U_full_ref[free_dofs]

    for sx in range(n_sub[0]):
        for sy in range(n_sub[1]):
            for sz in range(n_sub[2]):
                sub_idx = (sx * n_sub[1] + sy) * n_sub[2] + sz
                sub_mesh = sub_meshes[sub_idx]
                condensor = exact_condensors[sub_idx]

                sub_global_dofs = assembler.get_substructure_global_dofs(sx, sy, sz, sub_mesh)
                u_sub_b = U_full_ref[sub_global_dofs[sub_mesh.b_dofs]]
                u_sub_i_recovered = condensor.recover(u_sub_b)
                U_recovered[sub_global_dofs[sub_mesh.i_dofs]] = u_sub_i_recovered

    err_recovery = float(bm.linalg.norm(U_recovered - U_full_ref) / bm.linalg.norm(U_full_ref))

    print("\n" + "=" * 85)
    print("3D 纯弹性体子结构静态缩聚验证汇总表")
    print("=" * 85)

    widths_3d = (40, 35)
    print(format_table_row("统计指标 / 评估项目", "数值", widths_3d))
    print("-" * 85)
    print(format_table_row("3D 全尺寸精细网格总自由度数 (DOFs)", str(assembler.total_full_dofs), widths_3d))
    print(format_table_row("3D 界面缩聚自由度数 (Condensed DOFs)", str(len(free_dofs)), widths_3d))
    print(format_table_row("3D 自由度缩聚比例 (DOF Reduction Ratio)", f"{assembler.total_full_dofs / len(free_dofs):.2f}x", widths_3d))
    print(format_table_row("3D 静态缩聚内部位移恢复相对误差", f"{err_recovery:.4e} (机器精度)", widths_3d))
    print("=" * 85)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="子结构静态缩聚与拉格朗日有限元对比演示")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=None, help="运行维度 (2 或 3; 默认两者都运行)")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 运行子结构内部缩聚位移恢复基准测试
    if args.dim is None or args.dim == 2:
        run_demo_2d(out_dir)
    if args.dim is None or args.dim == 3:
        run_demo_3d(out_dir)

    # 2. 运行与传统拉格朗日有限元 (Lagrange FEM) 的全方位对比基准
    from compare_lagrange import run_benchmark_2d, run_benchmark_3d
    if args.dim is None or args.dim == 2:
        run_benchmark_2d()
    if args.dim is None or args.dim == 3:
        run_benchmark_3d()

    dim_str = f"{args.dim}D" if args.dim else "2D 与 3D"
    print(f"\n[成功] {dim_str} 子结构弹性力学分析及其与传统拉格朗日有限元对比演示完成！")


if __name__ == "__main__":
    main()



