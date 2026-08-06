"""物理信息神经网络 (PINN) 求解线弹性问题的最小可运行算例.

目的是用尽量少的代码走通 PINN 求解线弹性方程的全流程:
构造坐标到位移的神经网络 -> 采样域内与边界配点 -> 自动微分 (autograd) 计算应变、应力与平衡残差
-> Adam 优化器极小化强形式 Loss -> 评估位移 L2 误差。

与本地多模块架构的关系:
本目录下的 contract.py / cases.py / operators.py / solve.py 等模块承担自动门禁与 evidence 导出;
而本 minimal_demo.py 属于完全自包含的单文件脚本, 不依赖本目录的任何本地模块。
想看工程级自动化校验, 去读那些模块; 想看 PINN 是怎么把弹性力学方程解出来的, 读这一个文件就够。

问题类和材料类直接取自 soptx.problems.elasticity:
- 2D: ExponentialSineManufacturedElasticity2D (平面应变, lambda=1.0, mu=0.5)
- 3D: DivergenceFreePolynomialElasticity3D (3D各向同性, lambda=1.0, mu=1.0)

各制造解的完整数学定义见
`制造解文档 <../../docs/problems/manufactured-elasticity.md>`__。

运行::

    python examples/pinn_elasticity/minimal_demo.py
    python examples/pinn_elasticity/minimal_demo.py --dim 3
    python examples/pinn_elasticity/minimal_demo.py --epochs 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn

from fealpy.backend import bm
from fealpy.mesh import TetrahedronMesh, TriangleMesh
from fealpy.ml.grad import gradient
from fealpy.ml.modules import Solution
from fealpy.ml.sampler import BoxBoundarySampler, ISampler

from soptx.problems.elasticity import (
    DivergenceFreePolynomialElasticity3D,
    ExponentialSineManufacturedElasticity2D,
)


class PINNElasticityNet(nn.Module):
    """坐标到位移的 MLP 神经网络 (d -> 32 -> 32 -> 16 -> d)."""

    def __init__(self, dim: int, hidden_size: tuple[int, ...] = (32, 32, 16)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        sizes = (dim,) + hidden_size + (dim,)
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], dtype=torch.float64))
            if i < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


def compute_pinn_residuals(
    net: nn.Module,
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    problem: object,
    lame_lambda: float,
    shear_modulus: float,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """使用 autograd 计算强形式平衡残差 -div(sigma) - b 与 Dirichlet 边界残差."""
    
    # 1. 域内位移预测 u
    u = net(interior_points)

    # 2. 计算位移梯度 grad(u)
    grad_u = torch.stack(
        [
            gradient(u[..., c : c + 1], interior_points, create_graph=True)
            for c in range(dim)
        ],
        dim=-2,
    )

    # 3. 对称应力张量 sigma = lambda * tr(eps) * I + 2 * mu * eps
    strain = 0.5 * (grad_u + grad_u.transpose(-1, -2))
    trace_strain = torch.diagonal(strain, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    identity = torch.eye(dim, dtype=torch.float64, device=interior_points.device)
    stress = (
        lame_lambda * trace_strain.unsqueeze(-1) * identity
        + 2.0 * shear_modulus * strain
    )

    # 4. 平衡方程散度 div(sigma)
    div_components = []
    total_derivatives = dim * dim
    deriv_count = 0
    for i in range(dim):
        div_i = torch.zeros_like(interior_points[..., 0])
        for j in range(dim):
            sig_ij = stress[..., i, j]
            is_last = deriv_count == total_derivatives - 1
            grad_sig = torch.autograd.grad(
                outputs=sig_ij,
                inputs=interior_points,
                grad_outputs=torch.ones_like(sig_ij),
                create_graph=True,
                retain_graph=True,
            )[0]
            div_i = div_i + grad_sig[..., j]
            deriv_count += 1
        div_components.append(div_i)

    divergence = torch.stack(div_components, dim=-1)
    body_force = problem.body_force(interior_points)
    eq_residual = -divergence - body_force

    # 5. 全位移 Dirichlet 边界残差
    u_bc = net(boundary_points)
    exact_bc = problem.dirichlet_bc(boundary_points)
    bc_residual = u_bc - exact_bc

    return eq_residual, bc_residual


def run_minimal_demo(
    dim: int = 2,
    epochs: int = 500,
    lr: float = 1e-3,
    save_model: bool = False,
    save_vtu: bool = False,
    plot: bool = False,
) -> None:
    print(f"=" * 65)
    print(f"PINN 线弹性求解最小算例 [{dim}D]")
    print(f"=" * 65)

    bm.set_backend("pytorch")
    device = torch.device("cpu")
    torch.manual_seed(0)

    # 1. 实例化 PDE 制造解模型与材料参数
    if dim == 2:
        domain = (0.0, 1.0, 0.0, 1.0)
        lame_lambda = 1.0
        shear_modulus = 0.5
        problem = ExponentialSineManufacturedElasticity2D(
            domain=domain, lame_lambda=lame_lambda, shear_modulus=shear_modulus
        )
        mesh = TriangleMesh.from_box(list(domain), nx=29, ny=29)
    elif dim == 3:
        domain = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        lame_lambda = 1.0
        shear_modulus = 1.0
        problem = DivergenceFreePolynomialElasticity3D(
            domain=domain, lame_lambda=lame_lambda, shear_modulus=shear_modulus
        )
        mesh = TetrahedronMesh.from_box(list(domain), nx=7, ny=7, nz=7)
    else:
        raise ValueError(f"只支持 2D 或 3D, 收到 dim={dim}")

    # 2. 神经网络与优化器
    net = Solution(PINNElasticityNet(dim=dim)).to(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # 3. 配点采样器
    sampler_opts = {
        "mode": "random",
        "dtype": bm.float64,
        "device": device,
        "requires_grad": True,
    }
    interior_sampler = ISampler(domain, **sampler_opts)
    boundary_sampler = BoxBoundarySampler(domain, **sampler_opts)

    npde = 400
    nbc = 100
    w_eq, w_bc = 1.0, 30.0

    print(f"问题参数: 维数={dim}D, 配点: 域内={npde}, 边界={nbc}, Loss权重=({w_eq}, {w_bc})")
    print(f"网络结构: {dim} -> 32 -> 32 -> 16 -> {dim}, 优化器=Adam(lr={lr})")
    print("-" * 65)

    # 4. 训练循环
    for epoch in range(1, epochs + 1):
        interior_pts = interior_sampler.run(npde)
        boundary_pts = boundary_sampler.run(nbc)
        interior_pts.grad = None
        boundary_pts.grad = None

        optimizer.zero_grad()
        eq_res, bc_res = compute_pinn_residuals(
            net, interior_pts, boundary_pts, problem, lame_lambda, shear_modulus, dim
        )
        eq_loss = torch.mean(eq_res**2)
        bc_loss = torch.mean(bc_res**2)
        total_loss = w_eq * eq_loss + w_bc * bc_loss

        total_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:4d}/{epochs:4d} | Total Loss: {total_loss.item():.6e} "
                f"| Eq Loss: {eq_loss.item():.6e} | BC Loss: {bc_loss.item():.6e}"
            )

    # 5. 评估位移全维度数值误差 (全局L2、分量L2、最大绝对误差L_inf)
    print("-" * 65)
    component_error = net.estimate_error(problem.disp_solution, mesh, coordtype="c").detach()
    combined_l2_error = float(torch.linalg.vector_norm(component_error).item())

    class ZeroNet(nn.Module):
        def forward(self, x): return torch.zeros_like(x)

    zero_net = Solution(ZeroNet()).to(device=device, dtype=torch.float64)
    exact_norm_components = zero_net.estimate_error(problem.disp_solution, mesh, coordtype="c").detach()
    exact_norm_combined = float(torch.linalg.vector_norm(exact_norm_components).item())
    rel_l2_error = combined_l2_error / exact_norm_combined

    # 计算分量 L2 相对误差
    comp_err_list = [float(v) for v in component_error.flatten()]
    comp_norm_list = [float(v) for v in exact_norm_components.flatten()]
    comp_rel_errors = [err / ref for err, ref in zip(comp_err_list, comp_norm_list)]

    # 计算全局最大绝对误差 (L_infinity 范数)
    eval_nodes = mesh.entity("node")
    with torch.no_grad():
        eval_nodes_tensor = torch.as_tensor(eval_nodes, dtype=torch.float64, device=device)
        u_pinn_eval = net(eval_nodes_tensor)
        u_exact_eval = problem.disp_solution(eval_nodes_tensor)
        diff_eval = torch.abs(u_pinn_eval - u_exact_eval)
        max_abs_error = float(torch.max(diff_eval).item())

    print(f"全面数值误差评估结果:")
    print(f"  1. 位移联合 L2 误差:")
    print(f"     - 绝对 L2 误差 : {combined_l2_error:.6e}")
    print(f"     - 相对 L2 误差 : {rel_l2_error:.6e} ({rel_l2_error * 100:.2f}%)")
    print(f"  2. 位移分量 L2 相对误差:")
    for idx, rel_err in enumerate(comp_rel_errors):
        axis_name = ("x", "y", "z")[idx]
        print(f"     - u_{axis_name} 相对 L2 误差: {rel_err:.6e} ({rel_err * 100:.2f}%)")
    print(f"  3. 全场最大绝对误差 (L_inf 范数): {max_abs_error:.6e}")
    print(f"=" * 65)

    # 6. 可视化绘图 (支持 2D/3D 位移场与误差云图)
    fig_dir = Path(__file__).resolve().parent / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"pinn_elasticity_{dim}d.png"

    try:
        import matplotlib.pyplot as plt

        if dim == 2:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            node_x = eval_nodes[:, 0]
            node_y = eval_nodes[:, 1]
            triangles = mesh.entity("cell")

            u_exact_norm = torch.linalg.vector_norm(u_exact_eval, dim=-1).cpu().numpy()
            u_pinn_norm = torch.linalg.vector_norm(u_pinn_eval, dim=-1).cpu().numpy()
            point_error_norm = torch.linalg.vector_norm(diff_eval, dim=-1).cpu().numpy()

            c1 = axes[0].tripcolor(node_x, node_y, triangles, u_exact_norm, cmap="viridis")
            axes[0].set_title("Exact Displacement ||u_exact||")
            axes[0].set_aspect("equal")
            fig.colorbar(c1, ax=axes[0])

            c2 = axes[1].tripcolor(node_x, node_y, triangles, u_pinn_norm, cmap="viridis")
            axes[1].set_title("PINN Prediction ||u_pinn||")
            axes[1].set_aspect("equal")
            fig.colorbar(c2, ax=axes[1])

            c3 = axes[2].tripcolor(node_x, node_y, triangles, point_error_norm, cmap="magma")
            axes[2].set_title("Absolute Error ||u_pinn - u_exact||")
            axes[2].set_aspect("equal")
            fig.colorbar(c3, ax=axes[2])

            plt.suptitle(f"PINN Elasticity 2D (Relative L2 Error: {rel_l2_error*100:.2f}%)")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            print(f"三场对比云图已自动生成并保存至:\n  {fig_path}")
            if plot:
                plt.show()
            plt.close(fig)
        elif dim == 3:
            print(f"3D 评估数值已计算完成。云图文件路径:\n  {fig_path}")
    except Exception as err:
        print(f"绘图跳过: {err}")

    # 7. 可选：导出 ParaView 格式的 .vtu 可视化文件 (使用 pyevtk 库机制)
    if save_vtu:
        try:
            from pyevtk.hl import unstructuredGridToVTK
            import numpy as np

            vtu_dir = Path(__file__).resolve().parent / "outputs" / "vtu"
            vtu_dir.mkdir(parents=True, exist_ok=True)
            vtu_stem = str(vtu_dir / f"pinn_elasticity_{dim}d")
            vtu_file_path = vtu_dir / f"pinn_elasticity_{dim}d.vtu"

            nodes = np.asarray(mesh.entity("node"), dtype=np.float64)
            cells = np.asarray(mesh.entity("cell"), dtype=np.int32)
            n_nodes = nodes.shape[0]
            n_cells = cells.shape[0]
            nodes_per_cell = cells.shape[1]

            if dim == 2:
                cell_type = 5  # VTK_TRIANGLE
                x = np.ascontiguousarray(nodes[:, 0])
                y = np.ascontiguousarray(nodes[:, 1])
                z = np.zeros(n_nodes, dtype=np.float64)
            else:
                cell_type = 10  # VTK_TETRA
                x = np.ascontiguousarray(nodes[:, 0])
                y = np.ascontiguousarray(nodes[:, 1])
                z = np.ascontiguousarray(nodes[:, 2])

            connectivity = np.ascontiguousarray(cells.flatten())
            offsets = np.arange(
                nodes_per_cell, n_cells * nodes_per_cell + 1, nodes_per_cell, dtype=np.int32
            )
            cell_types = np.full(n_cells, cell_type, dtype=np.int32)

            def _to_np(t):
                if hasattr(t, "detach"):
                    t = t.detach()
                if hasattr(t, "cpu"):
                    t = t.cpu()
                return np.ascontiguousarray(np.asarray(t, dtype=np.float64))

            pinn_np = _to_np(u_pinn_eval)
            exact_np = _to_np(u_exact_eval)
            diff_np = _to_np(diff_eval)

            point_data = {
                "u_pinn_mag": np.ascontiguousarray(np.linalg.norm(pinn_np, axis=-1)),
                "u_exact_mag": np.ascontiguousarray(np.linalg.norm(exact_np, axis=-1)),
                "u_diff_mag": np.ascontiguousarray(np.linalg.norm(diff_np, axis=-1)),
            }
            for d in range(dim):
                axis_name = ("x", "y", "z")[d]
                point_data[f"u_pinn_{axis_name}"] = np.ascontiguousarray(pinn_np[:, d])
                point_data[f"u_exact_{axis_name}"] = np.ascontiguousarray(exact_np[:, d])
                point_data[f"u_diff_{axis_name}"] = np.ascontiguousarray(diff_np[:, d])

            unstructuredGridToVTK(
                vtu_stem, x, y, z, connectivity, offsets, cell_types, pointData=point_data
            )
            print(f"已导出 ParaView 格式的可视化文件 (.vtu):\n  {vtu_file_path}")
            print(f"=" * 65)
        except Exception as err:
            print(f"VTU 导出跳过: {err}")

    # 8. 可选：保存训练好的神经网络模型
    if save_model:
        checkpoint_dir = Path(__file__).resolve().parent / "outputs" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoint_dir / f"pinn_elasticity_{dim}d.pt"
        torch.save(net.state_dict(), model_path)
        print(f"已将训练好的神经网络权重保存至特定输出目录:\n  {model_path}")
        print(f"=" * 65)


def main() -> int:
    parser = argparse.ArgumentParser(description="PINN 线弹性求解最小可运行算例.")
    parser.add_argument("--dim", type=int, choices=(2, 3), default=2, help="问题维数 (2 或 3)")
    parser.add_argument("--epochs", type=int, default=500, help="训练迭代步数 (默认 500)")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率 (默认 1e-3)")
    parser.add_argument("--save-model", action="store_true", help="训练结束后将神经网络权重保存至磁盘 .pt 文件")
    parser.add_argument("--save-vtu", action="store_true", help="训练结束后导出 ParaView 格式的 .vtu 文件")
    parser.add_argument("--plot", action="store_true", help="训练结束后自动弹出交互式云图")
    args = parser.parse_args()

    run_minimal_demo(
        dim=args.dim,
        epochs=args.epochs,
        lr=args.lr,
        save_model=args.save_model,
        save_vtu=args.save_vtu,
        plot=args.plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
