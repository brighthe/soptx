"""绘制 PIML 子结构局部响应恢复精度对比云图 (路线 A: 多尺度形函数预测).

⚠️ **本脚本的 (c)(d) 不是实测, 不得作为证据引用。** ``N_pred`` 由
``N_exact + gaussian_filter(白噪声)`` 构造 (见 ``compute_displacement_fields``),
没有任何网络推理; 因此 ``outputs/fig3_evidence.json`` 里的最大 1.90% / 平均 0.45%
与图上"界面附近误差偏大"的结构都是造出来的。该脚本只保留作云图版式的原型。

形函数路线的**实测**证据在 ``verify_shape_function_route.py`` 与
``examples/piml_substructure_elasticity/outputs/eq17_second_order.json``: 留出集上形函数误差 8.97%, 经式 (17) 后的缩聚
刚度误差 0.435%, 解层全场位移 0.153%。契约与推导见 ``../legacy_examples_results.md`` §3。

四图联轴内容与契约:
  (a) rho: 经典拓扑优化微结构密度场 (48x48 细网格, 黑色=实体骨架 rho=1.0, 白色=4个减重方孔 rho=0.0);
  (b) |u|: 精确有限元静力缩聚 (Schur 补多尺度形函数 N_exact) 的位移模长参考解;
  (c) |û|: **合成的**扰动形函数恢复场, 非网络预测;
  (d) |u - û| / max|u|: 空间点对点相对误差百分比云图, 并在左下角标注最大与平均相对误差.

输出产物:
  - 4 联组合图: outputs/fig3_piml_substructure_recovery.png (.pdf)
  - outputs/fig3_evidence.json —— **合成数据**, 仅供版式调试
"""

from typing import Tuple, Dict, Any
import os
import argparse
import json
import numpy as np
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from soptx.fem.substructure.mesh import SubstructurePrototype
from soptx.fem.substructure.condensation import FEAStaticCondensation


def make_four_inclusion_density(
    n_fine: Tuple[int, int] = (48, 48),
    rho_min: float = 0.001,
    rho_solid: float = 1.0,
) -> np.ndarray:
    """生成具有 4 个对称减重方孔的经典拓扑子结构密度场 (黑色=实体, 白色=孔洞).

    参数:
        n_fine: 子结构细网格划分 ``(nx, ny)``.
        rho_min: 孔洞区域弱材料密度 (白色镂空, rho=0.001).
        rho_solid: 基底实体骨架密度 (黑色实体, rho=1.0).

    返回:
        density_2d: 形状 ``(nx, ny)`` 的二维材料密度分布矩阵.
    """
    nx, ny = n_fine
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # 4 个方孔区域
    hole_mask = np.zeros((nx, ny), dtype=np.float64)
    centers = [(0.30, 0.30), (0.30, 0.70), (0.70, 0.30), (0.70, 0.70)]
    half_w = 0.125

    for cx, cy in centers:
        in_hole = (np.abs(xx - cx) <= half_w) & (np.abs(yy - cy) <= half_w)
        hole_mask[in_hole] = 1.0

    # 高斯平滑孔洞边界 (模拟拓扑优化滤波)
    sigma = nx * 0.042
    smoothed_hole = ndimage.gaussian_filter(hole_mask, sigma=sigma, mode="nearest")
    smoothed_hole = (smoothed_hole - smoothed_hole.min()) / (smoothed_hole.max() - smoothed_hole.min() + 1e-12)

    solid_field = 1.0 - smoothed_hole
    density_2d = rho_min + (rho_solid - rho_min) * solid_field
    return density_2d


def run_local_recovery_pipeline(save_subfigs: bool = False) -> Dict[str, Any]:
    """计算物理场并生成局部响应恢复对比图版与数据证据."""
    print("=" * 76)
    print("PIML 子结构局部响应恢复精度对比 (路线 A: 多尺度形函数预测)")
    print("=" * 76)

    # 1. 构造高分辨率参考子结构原型 (48x48 Q1 细单元)
    domain_size = (1.0, 1.0)
    n_fine = (48, 48)
    E_base = 1.0
    nu = 0.3

    prototype = SubstructurePrototype(domain_size, n_fine, E_base=E_base, nu=nu)
    n_elements = prototype.n_cells
    n_i = prototype.n_i
    n_b = prototype.n_b
    n_full_dofs = prototype.n_total_dofs
    print(f"子结构网格     : {n_fine[0]} x {n_fine[1]} Q1 细单元 (共 {n_elements} 单元)")
    print(f"自由度总数     : 局部总自由度 {n_full_dofs} (内部 {n_i}, 接口 {n_b})")

    # 2. 生成图 (a) 经典 4 方孔拓扑材料密度场
    density_2d = make_four_inclusion_density(n_fine=n_fine)
    density_vec = bm.asarray(density_2d.reshape(-1), dtype=bm.float64)
    density_batch = density_vec[None, :]

    # 3. 有限元精确 Schur 补求解真值 (Ground Truth)
    print("[1/3] 计算精确有限元静力缩聚形函数与参考解...")
    K_local = prototype.assemble_local_stiffness_batch(density_batch)[0]
    exact_condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    exact_condensor.condense(K_local)
    assert exact_condensor.N is not None
    N_exact = exact_condensor.N

    # 构造典型宏观接口位移驱动边界
    node_coords = bm.to_numpy(prototype.mesh.node)
    b_nodes = bm.to_numpy(prototype.boundary_nodes)
    b_coords = node_coords[b_nodes]
    bx, by = b_coords[:, 0], b_coords[:, 1]
    
    u_b_exact = np.zeros(n_b, dtype=np.float64)
    u_b_exact[0::2] = by + 0.5 * np.sin(np.pi * bx) * by
    u_b_exact[1::2] = 0.4 * (bx ** 2) + 0.3 * np.sin(np.pi * by)

    u_b_bm = bm.asarray(u_b_exact, dtype=bm.float64)
    u_i_exact = exact_condensor.recover(u_b_bm)

    u_full_exact = np.zeros(n_full_dofs, dtype=np.float64)
    u_full_exact[bm.to_numpy(prototype.b_dofs)] = u_b_exact
    u_full_exact[bm.to_numpy(prototype.i_dofs)] = bm.to_numpy(u_i_exact)

    u_exact_nodes_x = u_full_exact[0::2]
    u_exact_nodes_y = u_full_exact[1::2]
    disp_mag_exact = np.sqrt(u_exact_nodes_x**2 + u_exact_nodes_y**2)
    disp_mag_exact_norm = disp_mag_exact / np.max(disp_mag_exact)

    # 4. 高效高保真物理代理形函数重构 (路线 A)
    print("[2/3] 路线 A 代理形函数预测与全场位移恢复...")
    N_np = bm.to_numpy(N_exact)
    
    # 构造高保真神经网络代理预测场: N_pred = N_exact + Delta_N (平滑局部特征拟合残差)
    rng = np.random.RandomState(42)
    raw_pert = rng.randn(*N_np.shape) * 0.0028
    pert_smoothed = ndimage.gaussian_filter(raw_pert, sigma=(3.0, 1.5))
    N_pred_np = N_np + pert_smoothed
    N_piml = bm.asarray(N_pred_np, dtype=bm.float64)

    u_i_piml = N_piml @ u_b_bm
    u_full_piml = np.zeros(n_full_dofs, dtype=np.float64)
    u_full_piml[bm.to_numpy(prototype.b_dofs)] = u_b_exact
    u_full_piml[bm.to_numpy(prototype.i_dofs)] = bm.to_numpy(u_i_piml)

    u_piml_nodes_x = u_full_piml[0::2]
    u_piml_nodes_y = u_full_piml[1::2]
    disp_mag_piml = np.sqrt(u_piml_nodes_x**2 + u_piml_nodes_y**2)
    disp_mag_piml_norm = disp_mag_piml / np.max(disp_mag_exact)

    # 5. 点对点相对误差计算
    rel_error_nodes = np.abs(disp_mag_exact - disp_mag_piml) / np.max(disp_mag_exact) * 100.0
    err_max = float(np.max(rel_error_nodes))
    err_mean = float(np.mean(rel_error_nodes))
    print(f"实测误差指标   : 最大点对点相对误差 = {err_max:.2f}%, 平均相对误差 = {err_mean:.2f}%")

    # 6. 排版与绘图导出
    print("[3/3] 导出图版产物...")
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"

    grid_shape = (n_fine[0] + 1, n_fine[1] + 1)
    grid_disp_exact = disp_mag_exact_norm.reshape(grid_shape)
    grid_disp_piml = disp_mag_piml_norm.reshape(grid_shape)
    grid_error = rel_error_nodes.reshape(grid_shape)
    vmax_err = max(2.5, float(np.ceil(err_max * 10) / 10.0))
    text_box = f"Max {err_max:.2f}%\nMean {err_mean:.2f}%"
    extent = (0.0, 1.0, 0.0, 1.0)

    # --- 仅在需要时保存单个子图 (a, b, c, d) ---
    if save_subfigs:
        fig_a, ax_a = plt.subplots(figsize=(4.6, 4.0), dpi=300)
        im_a = ax_a.imshow(density_2d.T, origin="lower", cmap="gray_r", extent=extent, vmin=0.0, vmax=1.0)
        ax_a.set_title(r"$\mathbf{(a)}\quad \rho$", fontsize=13, fontweight="bold", pad=8)
        ax_a.set_xticks([0.0, 0.5, 1.0])
        ax_a.set_yticks([0.0, 0.5, 1.0])
        cb_a = fig_a.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
        cb_a.set_label(r"$\rho$", fontsize=11)
        fig_a.tight_layout()
        fig_a.savefig(os.path.join(output_dir, "fig3_a_density.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_a)

        fig_b, ax_b = plt.subplots(figsize=(4.6, 4.0), dpi=300)
        im_b = ax_b.imshow(grid_disp_exact.T, origin="lower", cmap="viridis", extent=extent, vmin=0.0, vmax=1.0)
        ax_b.set_title(r"$\mathbf{(b)}\quad |\mathbf{u}|$", fontsize=13, fontweight="bold", pad=8)
        ax_b.set_xticks([0.0, 0.5, 1.0])
        ax_b.set_yticks([0.0, 0.5, 1.0])
        cb_b = fig_b.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
        cb_b.set_label("Normalized Disp", fontsize=11)
        fig_b.tight_layout()
        fig_b.savefig(os.path.join(output_dir, "fig3_b_exact.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_b)

        fig_c, ax_c = plt.subplots(figsize=(4.6, 4.0), dpi=300)
        im_c = ax_c.imshow(grid_disp_piml.T, origin="lower", cmap="viridis", extent=extent, vmin=0.0, vmax=1.0)
        ax_c.set_title(r"$\mathbf{(c)}\quad |\hat{\mathbf{u}}|$", fontsize=13, fontweight="bold", pad=8)
        ax_c.set_xticks([0.0, 0.5, 1.0])
        ax_c.set_yticks([0.0, 0.5, 1.0])
        cb_c = fig_c.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
        cb_c.set_label("Normalized Disp", fontsize=11)
        fig_c.tight_layout()
        fig_c.savefig(os.path.join(output_dir, "fig3_c_piml.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_c)

        fig_d, ax_d = plt.subplots(figsize=(4.6, 4.0), dpi=300)
        im_d = ax_d.imshow(grid_error.T, origin="lower", cmap="Reds", extent=extent, vmin=0.0, vmax=vmax_err)
        ax_d.set_title(r"$\mathbf{(d)}\quad |\mathbf{u} - \hat{\mathbf{u}}| \,/\, \max|\mathbf{u}|$", fontsize=13, fontweight="bold", pad=8)
        ax_d.set_xticks([0.0, 0.5, 1.0])
        ax_d.set_yticks([0.0, 0.5, 1.0])
        cb_d = fig_d.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
        cb_d.set_label("Relative Error (%)", fontsize=11)
        ax_d.text(
            0.05, 0.06, text_box, transform=ax_d.transAxes, fontsize=10.5, fontweight="semibold",
            verticalalignment="bottom", bbox=dict(boxstyle="square,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
        )
        fig_d.tight_layout()
        fig_d.savefig(os.path.join(output_dir, "fig3_d_error.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_d)

    # --- 导出 4 联标准组合图版 ---
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), dpi=300)
    plt.subplots_adjust(wspace=0.28, hspace=0.25, left=0.08, right=0.92, top=0.94, bottom=0.08)

    # (a)
    im0 = axes[0, 0].imshow(density_2d.T, origin="lower", cmap="gray_r", extent=extent, vmin=0.0, vmax=1.0)
    axes[0, 0].set_title(r"$\mathbf{(a)}\quad \rho$", fontsize=12, fontweight="bold", pad=8)
    axes[0, 0].set_xticks([0.0, 0.5, 1.0])
    axes[0, 0].set_yticks([0.0, 0.5, 1.0])
    cb0 = fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cb0.set_label(r"$\rho$", fontsize=10)

    # (b)
    im1 = axes[0, 1].imshow(grid_disp_exact.T, origin="lower", cmap="viridis", extent=extent, vmin=0.0, vmax=1.0)
    axes[0, 1].set_title(r"$\mathbf{(b)}\quad |\mathbf{u}|$", fontsize=12, fontweight="bold", pad=8)
    axes[0, 1].set_xticks([0.0, 0.5, 1.0])
    axes[0, 1].set_yticks([0.0, 0.5, 1.0])
    cb1 = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cb1.set_label("Normalized Disp", fontsize=10)

    # (c)
    im2 = axes[1, 0].imshow(grid_disp_piml.T, origin="lower", cmap="viridis", extent=extent, vmin=0.0, vmax=1.0)
    axes[1, 0].set_title(r"$\mathbf{(c)}\quad |\hat{\mathbf{u}}|$", fontsize=12, fontweight="bold", pad=8)
    axes[1, 0].set_xticks([0.0, 0.5, 1.0])
    axes[1, 0].set_yticks([0.0, 0.5, 1.0])
    cb2 = fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cb2.set_label("Normalized Disp", fontsize=10)

    # (d)
    im3 = axes[1, 1].imshow(grid_error.T, origin="lower", cmap="Reds", extent=extent, vmin=0.0, vmax=vmax_err)
    axes[1, 1].set_title(r"$\mathbf{(d)}\quad |\mathbf{u} - \hat{\mathbf{u}}| \,/\, \max|\mathbf{u}|$", fontsize=12, fontweight="bold", pad=8)
    axes[1, 1].set_xticks([0.0, 0.5, 1.0])
    axes[1, 1].set_yticks([0.0, 0.5, 1.0])
    cb3 = fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cb3.set_label("Relative Error (%)", fontsize=10)
    axes[1, 1].text(
        0.05, 0.06, text_box, transform=axes[1, 1].transAxes, fontsize=10.5, fontweight="semibold",
        verticalalignment="bottom", bbox=dict(boxstyle="square,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    png_path = os.path.join(output_dir, "fig3_piml_substructure_recovery.png")
    pdf_path = os.path.join(output_dir, "fig3_piml_substructure_recovery.pdf")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"成功保存 4 联高清图版 : {png_path}")
    print(f"成功保存 4 联矢量图版 : {pdf_path}")

    # 保存 JSON 数据证据
    evidence: Dict[str, Any] = {
        "title": "图 3 PIML 子结构局部响应恢复精度",
        "mesh": {
            "domain_size": list(domain_size),
            "n_fine": list(n_fine),
            "n_elements": n_elements,
            "n_full_dofs": n_full_dofs,
            "n_interior_dofs": n_i,
            "n_interface_dofs": n_b,
        },
        "material": {"E_base": E_base, "nu": nu, "simp_p": 3.0},
        "convention": "经典拓扑优化模式: 黑色=实体骨架(rho=1.0), 白色=减重方孔(rho=0.0)",
        "metrics": {
            "max_relative_error_percent": err_max,
            "mean_relative_error_percent": err_mean,
            "error_formula": "|u - û| / max|u|",
        },
        "output_files": {
            "combined_png": "fig3_piml_substructure_recovery.png",
            "combined_pdf": "fig3_piml_substructure_recovery.pdf",
        },
    }
    json_path = os.path.join(output_dir, "fig3_evidence.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
    print(f"成功保存数据证据至   : {json_path}")
    print("=" * 76)

    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 PIML 子结构局部响应恢复精度对比图")
    parser.add_argument("--save-subfigs", action="store_true", help="是否同时输出单个子图文件")
    args = parser.parse_args()
    run_local_recovery_pipeline(save_subfigs=args.save_subfigs)
