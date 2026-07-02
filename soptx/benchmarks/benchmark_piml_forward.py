"""PIML 多尺度前向原型必跑产出 (回填 dut-postdoc 答辩帧 7).

对每个粗/细比 L (默认 5, 10) 在固定粗网格上产出:

1. V1 误差: K^cond (逐子结构静力缩聚组装) vs 全尺度全局 Schur 补,
   rel_F = ||K_cond - S||_F / ||S||_F  (目标 < 1e-10, 机器精度);
2. 接口求解残差: ||K_bc U_b - F_bc|| / ||F_bc||;
3. 接口解 vs 全尺度直解误差: ||U_b - u_full[iface]|| / ||u_full[iface]||
   及细尺度恢复误差 ||u_rec - u_full|| / ||u_full||;
4. 规模参数: 粗网格、全尺度 DOF vs 接口 DOF、降维比.

另附 MockPredictor (均匀缩放解析映射) 在同一非均匀密度场下的 V1 误差,
作为"预测器接口互换 + 非精确预测器误差可度量"的演示行.

算例: 单位厚度矩形悬臂 (左边固支, 右边中点竖向点载荷 P=-1),
E=1, nu=0.3, 平面应力; 密度为光滑非均匀场 (值域约 [0.3, 0.9]).

运行 (仓库根目录):

    .\\.venv\\Scripts\\python.exe -m soptx.benchmarks.benchmark_piml_forward \
        --coarse 8x8 --levels 5,10 --output outputs/piml_forward_prototype.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from fealpy.backend import backend_manager as bm

from soptx.analysis.multiscale import (
    CoarseFineMeshPair,
    ExactPredictor,
    InterfaceCondensedSystem,
    MockPredictor,
    SubstructureTemplate,
    assemble_fullscale_stiffness,
    fullscale_schur_complement,
    solve_with_dirichlet,
)
from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial


def smooth_rho(mesh_pair) -> np.ndarray:
    x0, _, y0, _ = mesh_pair.domain
    xc = x0 + (mesh_pair.cell_ix + 0.5) * mesh_pair.hx
    yc = y0 + (mesh_pair.cell_iy + 0.5) * mesh_pair.hy
    return 0.3 + 0.3 * (1.0 + np.sin(3.0 * xc + 1.0) * np.cos(2.0 * yc + 0.5))


def run_case(ncx: int, ncy: int, L: int, q: int = 4,
             schur_chunk: int = 512) -> dict:
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0, poisson_ratio=0.3, plane_type="plane_stress",
    )
    domain = (0.0, 2.0, 0.0, 1.0)
    mesh_pair = CoarseFineMeshPair(domain=domain, ncx=ncx, ncy=ncy, L=L)
    rho = smooth_rho(mesh_pair)

    template = SubstructureTemplate(mesh_pair, material, q=q)

    # -- 逐子结构缩聚 + 接口组装 (Exact) --
    t0 = time.perf_counter()
    system = InterfaceCondensedSystem(mesh_pair, ExactPredictor(template), rho)
    t_condense = time.perf_counter() - t0

    # -- V1: 全尺度全局 Schur 补对照 --
    K = assemble_fullscale_stiffness(mesh_pair, material, rho, q=q)
    t0 = time.perf_counter()
    S = fullscale_schur_complement(K, mesh_pair.interface_dofs,
                                   chunk=schur_chunk)
    t_schur = time.perf_counter() - t0
    norm_S = np.linalg.norm(S)
    v1_exact = np.linalg.norm(system.K_cond.toarray() - S) / norm_S

    # Mock 在同一非均匀密度场下的 V1 误差 (接口互换演示)
    sys_mock = InterfaceCondensedSystem(mesh_pair, MockPredictor(template), rho)
    v1_mock = np.linalg.norm(sys_mock.K_cond.toarray() - S) / norm_S

    # -- T5: 前向闭环 (悬臂: 左边固支, 右边中点 P=-1) --
    x0, x1, y0, y1 = domain
    fixed_dofs = mesh_pair.dofs_on_line_x(x0)
    F = np.zeros(mesh_pair.n_dofs)
    F[mesh_pair.dof_at_point(x1, 0.5 * (y0 + y1), comp=1)] = -1.0

    t0 = time.perf_counter()
    U_b, info = system.solve_interface(F, fixed_dofs)
    t_iface_solve = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_full = solve_with_dirichlet(K, F, fixed_dofs)
    t_full_solve = time.perf_counter() - t0

    u_full_b = u_full[mesh_pair.interface_dofs]
    rel_iface = np.linalg.norm(U_b - u_full_b) / np.linalg.norm(u_full_b)
    u_rec = system.recover_fine(U_b, F)
    rel_fine = np.linalg.norm(u_rec - u_full) / np.linalg.norm(u_full)

    return {
        "coarse": f"{ncx}x{ncy}",
        "L": L,
        "fine": f"{mesh_pair.nx}x{mesh_pair.ny}",
        "n_fine_dofs": mesh_pair.n_dofs,
        "n_interface_dofs": mesh_pair.n_interface_dofs,
        "dof_reduction": mesh_pair.n_dofs / mesh_pair.n_interface_dofs,
        "v1_rel_error_exact": v1_exact,
        "v1_rel_error_mock": v1_mock,
        "interface_residual": info["interface_residual"],
        "rel_err_interface_vs_full": rel_iface,
        "rel_err_fine_recovery": rel_fine,
        "t_condense_s": t_condense,
        "t_schur_ref_s": t_schur,
        "t_interface_solve_s": t_iface_solve,
        "t_full_solve_s": t_full_solve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", default="8x8",
                        help="粗网格 ncx x ncy, 如 8x8")
    parser.add_argument("--levels", default="5,10",
                        help="粗/细比 L 列表, 逗号分隔")
    parser.add_argument("--output", default="outputs/piml_forward_prototype.csv")
    parser.add_argument("--schur-chunk", type=int, default=512)
    args = parser.parse_args()

    bm.set_backend("numpy")
    ncx, ncy = (int(v) for v in args.coarse.lower().split("x"))
    levels = [int(v) for v in args.levels.split(",") if v.strip()]

    rows = []
    for L in levels:
        print(f"[run] coarse={ncx}x{ncy}, L={L} ...", flush=True)
        rows.append(run_case(ncx, ncy, L, schur_chunk=args.schur_chunk))

    header = list(rows[0].keys())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    fmt = {float: lambda v: f"{v:.3e}", int: str}
    print("\n=== PIML 多尺度前向原型 必跑产出 ===")
    for row in rows:
        print(f"\n粗网格 {row['coarse']}, L={row['L']} (细网格 {row['fine']}):")
        print(f"  ④ 全尺度 DOF = {row['n_fine_dofs']}, "
              f"接口 DOF = {row['n_interface_dofs']} "
              f"(降维 {row['dof_reduction']:.2f}x)")
        print(f"  ① V1 误差 (Exact K_cond vs 全尺度 Schur) = "
              f"{row['v1_rel_error_exact']:.3e}"
              f"   [Mock 对照 = {row['v1_rel_error_mock']:.3e}]")
        print(f"  ② 接口求解残差 = {row['interface_residual']:.3e}")
        print(f"  ③ 接口解 vs 全尺度直解 = "
              f"{row['rel_err_interface_vs_full']:.3e}, "
              f"细尺度恢复 = {row['rel_err_fine_recovery']:.3e}")
        print(f"  时间: 缩聚 {row['t_condense_s']:.3f}s, "
              f"接口解 {row['t_interface_solve_s']:.3f}s, "
              f"全尺度直解 {row['t_full_solve_s']:.3f}s, "
              f"Schur 参照 {row['t_schur_ref_s']:.3f}s")
    print(f"\nCSV -> {out}")


if __name__ == "__main__":
    main()
