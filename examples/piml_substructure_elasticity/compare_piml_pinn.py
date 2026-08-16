"""
PIML 子结构静力缩聚 vs PINN 强形式求解器 跨范式横向对比
================================================================================
两条路径在同一个二维线弹性制造解算例
(`ExponentialSineManufacturedElasticity2D
<../../docs/problems/manufactured-elasticity.md>`_) 的同一块方域上运行, 对比:

1. 范式特征: 问题依赖性, 边界条件变更后的重训需求, 输入语义
2. 精度: PIML 比较缩聚算子 ``K_s``, PINN 比较全场位移
3. 效率: 离线训练与在线推理求解耗时

两条路径的精度指标为什么不是同一个量:
    PINN 直接学一个坐标到位移的映射, 它的自然误差是全场位移相对 ``L2`` 误差.
    PIML 学的是局部密度到 Schur 补 ``K_s`` 的映射, 是一个与外载, 边界条件都
    无关的算子, 它的自然误差是 ``K_s`` 的相对 Frobenius 误差. 这个差别本身就是
    两种范式的分界: 前者的产物绑定单个定解问题, 后者的产物可跨定解问题复用.

    本算例不比较 PIML 路径的位移场, 因为该制造解的位移在四条边上恒为零, 唯一的
    驱动是体力, 而缩聚路径按 Huang 2023 的建模假设只对内部自由度不受载的问题成立,
    在本算例上只能解出零位移场, 位移误差会退化成无意义的 ``0/0``. 带真实外载的
    位移场验证见同目录 ``compare_exact.py`` (MBB 梁集中载荷).

已知口径差异:
    子结构库的平面假设固定为 ``plane_stress``, 而本制造解是 plane strain. 由于
    PIML 路径只在自身离散算子内部与精确 Schur 补对比, 该差异不影响所报误差,
    但它使 PIML 路径的离散算子与 PINN 路径的连续问题并非同一个物理模型.

作者: 何亮 (大连理工大学博士后) & Antigravity Assistant
日期: 2026-08-16
"""

import os
import time
import argparse
import unicodedata
from typing import Any, Callable, List, Literal, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm

from soptx.problems.elasticity import ExponentialSineManufacturedElasticity2D
from soptx.fem.substructure import (
    SubstructureMesh,
    SubstructurePrototype,
    GlobalAssembler,
    FEAStaticCondensation,
    PIMLStaticCondensation,
    PIMLSurrogateNet,
)


TABLE_WIDTHS = (34, 30, 26)


def display_width(s: str) -> int:
    """计算字符串在等宽终端下的显示宽度, 东亚全角字符按两列计.

    参数:
        s: 待测量的字符串.

    返回:
        width: 终端显示所占的列数.
    """
    width = 0
    for char in s:
        width += 2 if unicodedata.east_asian_width(char) in ('F', 'W') else 1
    return width


def format_table_row(col1: str, col2: str, col3: str) -> str:
    """按显示宽度对齐三列表格行.

    参数:
        col1: 第一列文本.
        col2: 第二列文本.
        col3: 第三列文本.

    返回:
        row: 按 ``TABLE_WIDTHS`` 补齐空格后的整行文本.
    """
    return (
        f"{col1}{' ' * max(0, TABLE_WIDTHS[0] - display_width(col1))} | "
        f"{col2}{' ' * max(0, TABLE_WIDTHS[1] - display_width(col2))} | "
        f"{col3}{' ' * max(0, TABLE_WIDTHS[2] - display_width(col3))}"
    )


def gradient(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """对 ``x`` 求 ``y`` 之和的梯度.

    参数:
        y: 被求导的张量.
        x: 求导变量, 需要 ``requires_grad=True``.
        create_graph: 是否保留计算图以支持高阶导数.

    返回:
        grad: 与 ``x`` 同形状的梯度张量.
    """
    return torch.autograd.grad(y.sum(), x, create_graph=create_graph, retain_graph=True)[0]


def _to_torch_training_tensor(values: List[Any]) -> torch.Tensor:
    """将后端数据适配为 PyTorch 训练张量.

    参数:
        values: 一批同形状的后端数组.

    返回:
        tensor: 堆叠后的 ``float32`` 训练张量.

    说明:
        PyTorch 后端直接复用 ``bm.stack`` 的张量; 其他后端经 ``bm.to_numpy``
        规范转换, 使有限元数据生成与缩聚过程不依赖特定后端.
    """
    stacked = bm.stack(values)
    if isinstance(stacked, torch.Tensor):
        return stacked.to(dtype=torch.float32)
    return torch.from_numpy(bm.to_numpy(stacked)).to(dtype=torch.float32)


def build_substructures(
    assembler: GlobalAssembler,
) -> Tuple[SubstructurePrototype, List[SubstructureMesh]]:
    """按装配器的布局铺开全部子结构, 共享同一个参考子结构.

    参数:
        assembler: 已构造的全局装配器.

    返回:
        (prototype, sub_meshes): 共享的参考子结构与按 x 优先字典序排列的子结构
            列表.

    说明:
        全部子结构同构, 因此离散结构, 自由度划分与单位密度单元刚度只需构造一次,
        由 ``prototype`` 持有并被所有 ``SubstructureMesh`` 共享.
    """
    sub_size = tuple(
        assembler.domain_size[d] / assembler.n_sub[d] for d in range(assembler.dim)
    )
    prototype = SubstructurePrototype(
        sub_size, assembler.n_fine, assembler.E_base, assembler.nu
    )

    sub_meshes: List[SubstructureMesh] = []
    sub_id = 0
    for sx in range(assembler.n_sub[0]):
        for sy in range(assembler.n_sub[1]):
            spans = (
                (sx * sub_size[0], (sx + 1) * sub_size[0]),
                (sy * sub_size[1], (sy + 1) * sub_size[1]),
            )
            sub_meshes.append(
                SubstructureMesh(
                    sub_id, *spans, *assembler.n_fine,
                    assembler.E_base, assembler.nu, prototype=prototype,
                )
            )
            sub_id += 1
    return prototype, sub_meshes


def make_density_fields(
    sub_meshes: List[SubstructureMesh], n_fine: Tuple[int, int]
) -> Any:
    """按细单元重心生成平滑变化的局部密度分布.

    参数:
        sub_meshes: 子结构列表.
        n_fine: 单个子结构的细单元数, 形如 ``(n_fine_x, n_fine_y)``.

    返回:
        density: 形状 ``(B, n_fine_x, n_fine_y)`` 的密度场, ``B`` 为子结构数.

    说明:
        取值落在 ``[0.35, 0.95]``, 位于代理网络训练集的密度区间 ``[0.3, 1.0]``
        之内, 使评估不外推.
    """
    fields = []
    for sm in sub_meshes:
        nodes_x = bm.linspace(sm.x_span[0], sm.x_span[1], n_fine[0] + 1, dtype=bm.float64)
        nodes_y = bm.linspace(sm.y_span[0], sm.y_span[1], n_fine[1] + 1, dtype=bm.float64)
        center_x = 0.5 * (nodes_x[:-1] + nodes_x[1:])
        center_y = 0.5 * (nodes_y[:-1] + nodes_y[1:])
        grid_x, grid_y = bm.meshgrid(center_x, center_y, indexing="ij")
        fields.append(
            0.65 + 0.3 * bm.sin(2.0 * bm.pi * grid_x) * bm.cos(2.0 * bm.pi * grid_y)
        )
    return bm.stack(fields, axis=0)


def compute_pinn_loss(net, int_pts, bnd_pts, bnd_val, problem, lame_lambda, shear_modulus):
    """计算 PINN 强形式残差 Loss 与 Dirichlet 边界 Loss.

    参数:
        net: 坐标到位移的网络.
        int_pts: 域内配点, 形状 ``(n_int, 2)``, 需要 ``requires_grad=True``.
        bnd_pts: 边界配点, 形状 ``(n_bnd, 2)``.
        bnd_val: 边界配点上的位移真值, 形状 ``(n_bnd, 2)``.
        problem: 提供 ``body_force`` 的物理问题对象.
        lame_lambda: Lamé 第一参数.
        shear_modulus: 剪切模量.

    返回:
        (total_loss, loss_pde, loss_bnd): 加权总损失, 强形式残差损失与边界损失.
    """
    u = net(int_pts)
    grad_u = torch.stack(
        [
            gradient(u[..., c : c + 1], int_pts, create_graph=True)
            for c in range(2)
        ],
        dim=-2,
    )
    strain = 0.5 * (grad_u + grad_u.transpose(-1, -2))
    trace_strain = torch.diagonal(strain, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    identity = torch.eye(2, dtype=torch.float64, device=int_pts.device)
    stress = (
        lame_lambda * trace_strain.unsqueeze(-1) * identity
        + 2.0 * shear_modulus * strain
    )

    div_components = []
    for i in range(2):
        div_i = torch.zeros_like(int_pts[..., 0])
        for j in range(2):
            g = gradient(
                stress[..., i, j : j + 1], int_pts, create_graph=True
            )
            div_i = div_i + g[..., j]
        div_components.append(div_i)

    div_stress = torch.stack(div_components, dim=-1)

    int_pts_bm = bm.asarray(int_pts.detach().cpu().numpy(), dtype=bm.float64)
    body_force_bm = problem.body_force(int_pts_bm)
    body_force_t = torch.tensor(bm.to_numpy(body_force_bm), dtype=torch.float64, device=int_pts.device)

    pde_residual = div_stress + body_force_t
    loss_pde = torch.mean(pde_residual ** 2)

    u_bnd_pred = net(bnd_pts)
    loss_bnd = torch.mean((u_bnd_pred - bnd_val) ** 2)

    total_loss = loss_pde + 30.0 * loss_bnd
    return total_loss, loss_pde, loss_bnd


class PINNElasticityNet(nn.Module):
    """坐标到位移的 MLP 网络 (2 -> 32 -> 32 -> 16 -> 2)."""

    def __init__(self, dim: int = 2, hidden_size: Tuple[int, ...] = (32, 32, 16)) -> None:
        """构造 PINN 网络.

        参数:
            dim: 空间维数, 同时决定输入与输出宽度.
            hidden_size: 各隐藏层宽度.
        """
        super().__init__()
        layers: List[nn.Module] = []
        sizes = (dim,) + hidden_size + (dim,)
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], dtype=torch.float64))
            if i < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """由坐标预测位移.

        参数:
            points: 坐标点, 形状 ``(..., dim)``.

        返回:
            displacement: 位移预测, 形状 ``(..., dim)``.
        """
        return self.net(points)


def run_cross_paradigm_comparison(
    pinn_epochs: int = 400,
    backend: Literal["numpy", "pytorch"] = "numpy",
) -> None:
    """运行 PIML 与 PINN 跨范式对比.

    参数:
        pinn_epochs: PINN 训练轮数.
        backend: 本次示例运行使用的 ``bm`` 后端. 后端选择仅位于示例入口, 不传入
            子结构缩聚核心库.
    """
    bm.set_backend(backend)
    print("=" * 85)
    print("PIML 子结构静力缩聚 vs PINN 强形式求解器 跨范式横向对比")
    print("=" * 85)

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    problem = ExponentialSineManufacturedElasticity2D(
        domain=(0.0, 1.0, 0.0, 1.0), lame_lambda=1.0, shear_modulus=0.5
    )
    Lx, Ly = problem.domain[1], problem.domain[3]
    lame_lambda, shear_modulus = problem.lam, problem.mu
    # 由 Lamé 参数换算弹性模量与泊松比, 供子结构离散使用.
    E_base = shear_modulus * (3.0 * lame_lambda + 2.0 * shear_modulus) / (lame_lambda + shear_modulus)
    nu = lame_lambda / (2.0 * (lame_lambda + shear_modulus))

    print(f"物理域配置   : 2D 方域 [{Lx} x {Ly}], 制造解 plane strain, E={E_base:.4f}, nu={nu:.4f}")
    print(f"Lamé 参数    : lambda={lame_lambda:.4f}, mu={shear_modulus:.4f}\n")

    # =========================================================================
    # 路径 A: PIML 子结构静力缩聚 (Problem-Independent 局部算子)
    # =========================================================================
    print("-" * 73)
    print("[路径 A] PIML 子结构静力缩聚 (Problem-Independent 范式)")
    print("-" * 73)

    n_sub = (2, 2)
    n_fine = (5, 5)
    assembler = GlobalAssembler((Lx, Ly), n_sub, n_fine, E_base=E_base, nu=nu)
    prototype, sub_meshes = build_substructures(assembler)

    # 预测限制在刚体模态的正交补上: 自由漂浮子结构的 K_s 以刚体模态为精确零空间,
    # 在全部接口自由度上做 Cholesky 分解会给最软的方向注入伪刚度, 详见
    # ``PIMLStaticCondensation`` 的类说明.
    range_basis = prototype.deformation_basis
    n_reduced = int(range_basis.shape[1])
    tril_mask = bm.tril(bm.ones((n_reduced, n_reduced), dtype=bm.bool))
    n_tril = int(bm.sum(tril_mask))

    # -------------------------------------------------------------------------
    # 离线训练: 全部随机密度样本共用同一套离散结构, 局部刚度与缩聚各只调用一次
    # -------------------------------------------------------------------------
    t0_piml_train = time.time()
    piml_net = PIMLSurrogateNet(input_dim=n_fine[0] * n_fine[1], output_dim=n_tril)
    piml_opt = optim.Adam(piml_net.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    n_train = 250
    # 后端在运行时接受多个尺寸参数, 但静态类型声明仅暴露单个参数.
    random_rand = cast(Callable[..., Any], bm.random.rand)
    rand_rho = 0.3 + 0.7 * bm.asarray(
        random_rand(n_train, n_fine[0], n_fine[1]), dtype=bm.float64
    )
    K_train_batch = prototype.assemble_local_stiffness_batch(rand_rho)
    train_condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    K_s_train, _ = train_condensor.condense(K_train_batch)

    # 限制到变形子空间后算子严格正定, Cholesky 分解无需任何正则.
    L_train = bm.linalg.cholesky(range_basis.T @ K_s_train @ range_basis)

    X_train_t = _to_torch_training_tensor(
        [bm.reshape(rand_rho[i], (-1,)) for i in range(n_train)]
    )
    Y_train_t = _to_torch_training_tensor(
        [L_train[i][tril_mask] for i in range(n_train)]
    )

    piml_net.train()
    piml_final_loss = float("nan")
    for _ in range(300):
        piml_opt.zero_grad()
        loss = criterion(piml_net(X_train_t), Y_train_t)
        loss.backward()
        piml_opt.step()
        piml_final_loss = float(loss.item())
    piml_net.eval()
    t_piml_train = time.time() - t0_piml_train
    print(f"-> 代理网络离线训练完成 ({t_piml_train:.2f} s), 最终训练 MSE: {piml_final_loss:.6e}")

    # -------------------------------------------------------------------------
    # 在线推理: 精确路径批量缩聚一次; 代理网络只接受单个子结构密度, 故逐个推理
    # -------------------------------------------------------------------------
    density = make_density_fields(sub_meshes, n_fine)
    K_local_batch = prototype.assemble_local_stiffness_batch(density)

    t0_exact = time.time()
    exact_condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    K_s_exact, _ = exact_condensor.condense(K_local_batch)
    t_exact_condense = time.time() - t0_exact

    t0_piml_solve = time.time()
    piml_condensors = []
    for idx, sub_mesh in enumerate(sub_meshes):
        piml_condensor = PIMLStaticCondensation(
            sub_mesh.i_dofs, sub_mesh.b_dofs, model=piml_net, is_cholesky=True,
            range_basis=sub_mesh.deformation_basis,
        )
        piml_condensor.condense(K_local_batch[idx], density[idx])
        piml_condensors.append(piml_condensor)
    t_piml_solve = time.time() - t0_piml_solve

    K_s_piml = bm.stack([c.K_s for c in piml_condensors], axis=0)
    n_fallback = sum(1 for c in piml_condensors if c.used_fallback)

    # 缩聚算子的相对 Frobenius 误差, 逐子结构求后取最大值.
    diff_norm = bm.linalg.norm(K_s_piml - K_s_exact, axis=(-2, -1))
    ref_norm = bm.linalg.norm(K_s_exact, axis=(-2, -1))
    err_ks_each = diff_norm / ref_norm
    err_ks_piml = float(bm.max(err_ks_each))

    print(f"-> 在线缩聚: 精确批量 {t_exact_condense:.4f} s, PIML 逐个推理 {t_piml_solve:.4f} s")
    print(f"-> 缩聚算子 K_s 相对 Frobenius 误差 (逐子结构最大值): {err_ks_piml:.4e}")
    if n_fallback > 0:
        print(
            f"-> [警告] {n_fallback}/{len(piml_condensors)} 个子结构未通过正定性门禁, "
            f"已回退到精确缩聚; 上面的 K_s 误差不反映代理网络的真实精度."
        )

    # =========================================================================
    # 路径 B: PINN 强形式求解器 (Problem-Dependent 坐标映射)
    # =========================================================================
    print("\n" + "-" * 73)
    print(f"[路径 B] PINN 强形式求解器 (Problem-Dependent 范式, Epochs={pinn_epochs})")
    print("-" * 73)

    # 评估点直接取全局细网格节点, 按全局节点编号排列, 可直接交给 to_node_grid 重排.
    eval_pts = bm.asarray(assembler.full_mesh.entity("node"), dtype=bm.float64)
    u_exact_bm = problem.disp_solution(eval_pts)
    u_exact_flat = bm.reshape(u_exact_bm, (-1,))

    t0_pinn = time.time()
    pinn_net = PINNElasticityNet(dim=2, hidden_size=(32, 32, 16))
    pinn_opt = torch.optim.Adam(pinn_net.parameters(), lr=0.002)

    n_int, n_bnd = 400, 200
    int_pts_np = np.random.uniform(0, 1, (n_int, 2))
    n_per_edge = n_bnd // 4
    bnd_pts_np = np.vstack(
        [
            np.column_stack([np.zeros(n_per_edge), np.random.uniform(0, 1, n_per_edge)]),
            np.column_stack([np.ones(n_per_edge), np.random.uniform(0, 1, n_per_edge)]),
            np.column_stack([np.random.uniform(0, 1, n_per_edge), np.zeros(n_per_edge)]),
            np.column_stack([np.random.uniform(0, 1, n_per_edge), np.ones(n_per_edge)]),
        ]
    )

    int_pts_t = torch.tensor(int_pts_np, dtype=torch.float64, requires_grad=True)
    bnd_pts_t = torch.tensor(bnd_pts_np, dtype=torch.float64)
    bnd_val_bm = problem.disp_solution(bm.asarray(bnd_pts_np, dtype=bm.float64))
    bnd_val_t = torch.tensor(bm.to_numpy(bnd_val_bm), dtype=torch.float64)

    pinn_net.train()
    for _ in range(pinn_epochs):
        pinn_opt.zero_grad()
        tot_loss, pde_l, bnd_l = compute_pinn_loss(
            pinn_net, int_pts_t, bnd_pts_t, bnd_val_t, problem, lame_lambda, shear_modulus
        )
        tot_loss.backward()
        pinn_opt.step()
    t_pinn_solve = time.time() - t0_pinn

    pinn_net.eval()
    with torch.no_grad():
        eval_pts_t = torch.tensor(bm.to_numpy(eval_pts), dtype=torch.float64)
        u_pinn_pred_bm = bm.asarray(pinn_net(eval_pts_t).numpy(), dtype=bm.float64)

    u_pinn_flat = bm.reshape(u_pinn_pred_bm, (-1,))
    err_u_pinn = float(
        bm.linalg.norm(u_pinn_flat - u_exact_flat) / bm.linalg.norm(u_exact_flat)
    )

    print(f"-> 训练与求解完毕, 耗时 {t_pinn_solve:.2f} s")
    print(f"-> 最终 Loss: {tot_loss.item():.6e} (PDE: {pde_l.item():.6e}, Bnd: {bnd_l.item():.6e})")
    print(f"-> 全场位移相对 L2 误差 (vs 解析解): {err_u_pinn:.4e}")

    # =========================================================================
    # 跨范式对比表格
    # =========================================================================
    total_width = sum(TABLE_WIDTHS) + 6
    print("\n" + "=" * total_width)
    print("PIML SUBSTRUCTURE vs PINN CROSS-PARADIGM COMPARISON (2D Linear Elasticity)")
    print("=" * total_width)
    print(format_table_row("评估维度 (Metric / Aspect)", "PIML 子结构静力缩聚", "PINN 强形式求解器"))
    print("-" * total_width)
    print(format_table_row("范式类型", "Problem-Independent 局部算子", "Problem-Dependent 坐标映射"))
    print(format_table_row("边界条件/外载变更", "免重训, 直接复用", "必须整轮重训"))
    print(format_table_row("网络输入语义", "局部细网格密度 rho^j", "空间点坐标 (x, y)"))
    print(format_table_row("网络输出语义", "Schur 补 K_s 的 Cholesky 因子", "该点位移 (ux, uy)"))
    print(format_table_row(
        "可训练参数量",
        f"{sum(p.numel() for p in piml_net.parameters()):,}",
        f"{sum(p.numel() for p in pinn_net.parameters()):,}",
    ))
    print(format_table_row("精度指标", "K_s 相对 Frobenius 误差", "位移相对 L2 误差"))
    print(format_table_row("精度数值", f"{err_ks_piml:.4e}", f"{err_u_pinn:.4e}"))
    print(format_table_row("回退到精确缩聚的子结构数", f"{n_fallback} / {len(piml_condensors)}", "N/A"))
    print(format_table_row("离线训练耗时", f"{t_piml_train:.2f} s", "N/A (无离线模型)"))
    print(format_table_row("在线推理耗时", f"{t_piml_solve:.4f} s", f"{t_pinn_solve:.2f} s"))
    print("=" * total_width)

    # =========================================================================
    # 对比云图: 上排为 PINN 路径的场, 下排为 PIML 路径的算子
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # to_node_grid 依据节点坐标重排, 不假定网格生成器的节点编号次序.
    uy_exact = bm.to_numpy(assembler.to_node_grid(u_exact_bm[:, 1])).T
    uy_pinn = bm.to_numpy(assembler.to_node_grid(u_pinn_pred_bm[:, 1])).T

    im = axes[0, 0].imshow(uy_exact, origin="lower", cmap="viridis", extent=[0, Lx, 0, Ly])
    axes[0, 0].set_title("Manufactured Solution (Uy)")
    fig.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(uy_pinn, origin="lower", cmap="viridis", extent=[0, Lx, 0, Ly])
    axes[0, 1].set_title(f"PINN Strong-Form (Uy, Rel L2 {err_u_pinn:.2e})")
    fig.colorbar(im, ax=axes[0, 1])

    im = axes[1, 0].imshow(bm.to_numpy(K_s_exact[0]), cmap="coolwarm")
    axes[1, 0].set_title("Exact Schur Complement K_s (sub 0)")
    fig.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].imshow(bm.to_numpy(K_s_piml[0]), cmap="coolwarm")
    axes[1, 1].set_title(f"PIML Predicted K_s (sub 0, Rel Fro {float(err_ks_each[0]):.2e})")
    fig.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "piml_vs_pinn_comparison.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"\n[成功] 跨范式对比云图已保存至: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIML vs PINN 跨范式对比")
    parser.add_argument("--pinn-epochs", type=int, default=400, help="PINN 训练轮数")
    parser.add_argument(
        "--backend",
        choices=("numpy", "pytorch"),
        default="numpy",
        help="FEALPy bm 后端, 默认 numpy",
    )
    args = parser.parse_args()

    run_cross_paradigm_comparison(
        pinn_epochs=args.pinn_epochs,
        backend=cast(Literal["numpy", "pytorch"], args.backend),
    )
