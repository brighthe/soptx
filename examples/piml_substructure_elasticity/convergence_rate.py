"""PIML 子结构静力缩聚收敛阶与渐近一致性验证入口.

本脚本通过在多层嵌套加密网格上求解线弹性问题, 验证 PIML 代理子结构静力缩聚
与精确静力缩聚在离散解逼近真实连续解过程中的收敛速度与渐近一致性.

根据有限元先验误差估计理论与物理增强神经网络代理特性:
1. 有限元精确静力缩聚解在 :math:`L_2` 范数下的理论收敛阶为 2.0 阶 (:math:`O(h^2)`);
2. PIML 代理缩聚解在粗网格到中等网格区间继承有限元的收敛阶, 在极细网格上渐近收敛
   至由神经网络代理逼近精度决定的高精度平台, 且刚体模态残差恒在机器精度 (:math:`10^{-14}`) 级别.

物理模型采用无体力的调和多项式制造解 :class:`~soptx.problems.elasticity.HarmonicPoly2D`
与 :class:`~soptx.problems.elasticity.HarmonicPoly3D`, 严格满足
:math:`\\Delta u = 0, \\nabla \\cdot u = 0 \\implies b(x) \\equiv 0`, 由非齐次 Dirichlet
位移边界条件驱动. 该设定完全符合子结构静力缩聚内部自由度不受载 (:math:`f_i = 0`) 的建模假设.
制造解的数学推导与边界条件详见 `制造解文档 <../../docs/problems/manufactured-elasticity.md>`__.

使用方法:
    # 2D 调和多项式收敛阶验证 (默认 4 层网格加密).
    python examples/piml_substructure_elasticity/convergence_rate.py --dim 2

    # 3D 调和多项式收敛阶验证 (默认 3 层网格加密).
    python examples/piml_substructure_elasticity/convergence_rate.py --dim 3

``--output-dir`` 缺省为本脚本同级的 ``outputs/``, 按脚本位置解析.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence, Tuple, cast
import unicodedata

import torch
import torch.nn as nn
import torch.optim as optim
from fealpy.backend import backend_manager as bm

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.substructure import (
    FEAStaticCondensation,
    GlobalAssembler,
    SubstructureMesh,
    SubstructurePrototype,
    solve_interface_system,
)
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems.elasticity import (
    HarmonicPoly2D,
    HarmonicPoly3D,
)


### 表格排版工具 ###

def display_width(s: str) -> int:
    """计算字符串在等宽终端下的显示宽度, 东亚全角字符按两列计."""
    width = 0
    for char in s:
        width += 2 if unicodedata.east_asian_width(char) in ('F', 'W') else 1
    return width


def format_table_row(
    cols: Sequence[str],
    widths: Sequence[int],
) -> str:
    """按显示宽度对齐多列表格行."""
    cells = [
        f"{col}{' ' * max(0, widths[i] - display_width(col))}"
        for i, col in enumerate(cols)
    ]
    return " | ".join(cells)


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


### 路线 A 真实形函数神经网络 (ShapeFunctionNet) ###

class ShapeFunctionNet(nn.Module):
    """用于预测子结构多尺度形函数矩阵 N 的全连接神经网络 (路线 A).

    参数:
        input_dim: 输入单元密度向量维度 ``n_cells``.
        output_dim: 输出形函数矩阵扁平化条目数 ``n_i * n_b``.
        hidden_dims: 隐藏层神经元序列.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.SiLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向推理."""
        return self.network(x)


def train_shape_function_surrogate(
    prototype: SubstructurePrototype,
    n_train: int = 60,
    n_epochs: int = 100,
    learning_rate: float = 0.005,
) -> Tuple[ShapeFunctionNet, float]:
    """在随机多孔密度样本上真实端到端训练多尺度形函数网络 (路线 A).

    参数:
        prototype: 参考子结构原型.
        n_train: 训练集样本数.
        n_epochs: 梯度下降轮数.
        learning_rate: Adam 学习率.

    返回:
        (net, final_loss): 训练完成的网络及最终 MSE 损失.
    """
    dim = prototype.dim
    n_cells = int(prototype.n_cells)
    n_i = int(prototype.n_i)
    n_b = int(prototype.n_b)
    out_dim = n_i * n_b

    # 1. 批量生成样本: 包含近实心扰动样本与随机多孔样本
    bm.random.seed(2026)
    rand_func = cast(Any, bm.random.rand)
    rand_floats = rand_func(n_train, *tuple(prototype.n_fine))
    rho_train = 0.5 + 0.5 * bm.asarray(rand_floats, dtype=bm.float64)

    # 2. 精确批量求解多尺度形函数标签 N_exact (n_train, n_i, n_b)
    K_local_train = prototype.assemble_local_stiffness_batch(rho_train)
    condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    condensor.condense(K_local_train)
    assert condensor.N is not None
    N_train = condensor.N

    # 3. 构造 PyTorch 训练张量
    rho_train_flat = bm.reshape(rho_train, (n_train, n_cells))
    N_train_flat = bm.reshape(N_train, (n_train, out_dim))

    X_train = torch.from_numpy(bm.to_numpy(rho_train_flat)).to(dtype=torch.float32)
    Y_train = torch.from_numpy(bm.to_numpy(N_train_flat)).to(dtype=torch.float32)

    # 4. 训练神经网络
    torch.manual_seed(2026)
    net = ShapeFunctionNet(input_dim=n_cells, output_dim=out_dim, hidden_dims=(128, 128))
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    loss = criterion(net(X_train), Y_train)
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = criterion(net(X_train), Y_train)
        loss.backward()
        optimizer.step()

    final_loss = float(loss.item())
    net.eval()
    return net, final_loss


### 单层求解与误差评估 (精确 vs 真实训练的 PIML 代理双轨) ###

def solve_one_level(
    pde: Any,
    material: Any,
    n_sub: Tuple[int, ...],
    n_fine: Tuple[int, ...],
    degree: int = 1,
    n_train: int = 60,
    n_epochs: int = 100,
) -> Dict[str, Any]:
    """在单层子结构网格上执行精确与真实 PIML 代理缩聚求解并计算 L2 误差.

    参数:
        pde: 物理问题对象.
        material: 材料模型对象.
        n_sub: 各方向子结构划分数.
        n_fine: 单个子结构内部细网格划分数.
        degree: 有限元位移空间多项式次数, 缺省为 ``1``.
        n_train: 形函数网络训练样本数.
        n_epochs: 神经网络训练轮数.

    返回:
        level_record: 该网格层级的统计指标, 包含网格步长、精确/真实 PIML 的 L2 误差及相对残差.
    """
    dim = pde.dimension
    domain_size = tuple(
        pde.domain[2 * d + 1] - pde.domain[2 * d] for d in range(dim)
    )
    assembler = GlobalAssembler(
        domain_size, n_sub, n_fine, degree=degree, E_base=pde.E, nu=pde.nu
    )
    prototype, sub_meshes = build_substructures(assembler)

    # 实体材料均匀密度场 (rho = 1.0)
    density = bm.ones((len(sub_meshes),) + tuple(assembler.n_fine), dtype=bm.float64)

    # 1. 精确静力缩聚求解 (Ground Truth)
    t0 = time.time()
    K_local_batch = prototype.assemble_local_stiffness_batch(density)
    exact_condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    exact_condensor.condense(K_local_batch)

    system_exact = assembler.assemble_interface_system(sub_meshes, exact_condensor)

    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=assembler.full_mesh,
        pde=pde,
        material=material,
        space_degree=degree,
        solve_method='scipy',
        enable_logging=False,
    )
    dofs_val, fixed_mask = analyzer.tensor_space.boundary_interpolate(
        gd=pde.dirichlet_bc,
        threshold=cast(Any, pde.is_dirichlet_boundary()),
        method='interp',
    )
    fixed_global_dofs = bm.nonzero(fixed_mask)[0]
    fixed_interface_dofs = assembler.project_global_dofs(system_exact, fixed_global_dofs)
    prescribed_interface = assembler.project_global_vector(system_exact, dofs_val)

    zero_load = bm.zeros((len(system_exact.global_dofs),), dtype=bm.float64)
    u_interface_exact = solve_interface_system(
        system_exact,
        zero_load,
        fixed_interface_dofs,
        prescribed=prescribed_interface,
    )
    U_exact = assembler.recover_full_displacement(
        sub_meshes, exact_condensor, system_exact, u_interface_exact
    )
    t_exact = time.time() - t0

    # 2. 真实训练 PIML 路线 A 神经网络并执行前向推理
    t1 = time.time()
    net, train_mse = train_shape_function_surrogate(
        prototype, n_train=n_train, n_epochs=n_epochs, learning_rate=0.005
    )

    # 前向推理评估密度下的形函数预测矩阵
    n_sub_total = len(sub_meshes)
    density_eval_flat = bm.reshape(density, (n_sub_total, int(prototype.n_cells)))
    X_eval = torch.from_numpy(bm.to_numpy(density_eval_flat)).to(dtype=torch.float32)
    with torch.no_grad():
        N_pred_flat_torch = net(X_eval)
    N_pred_flat_np = N_pred_flat_torch.cpu().numpy()
    N_piml_raw = bm.asarray(
        N_pred_flat_np.reshape(n_sub_total, int(prototype.n_i), int(prototype.n_b)),
        dtype=bm.float64,
    )

    # 物理正交投影: 严格满足刚体模态保持 N @ R_b = R_i (消除全局刚体误差放大)
    assert exact_condensor.N is not None
    R_b = prototype.rigid_basis  # (n_b, n_rigid)
    R_i = exact_condensor.N @ R_b  # (n_sub, n_i, n_rigid)
    # N_proj = N_raw - (N_raw @ R_b - R_i) @ R_b.T
    drift = N_piml_raw @ R_b - R_i
    N_piml = N_piml_raw - drift @ bm.matrix_transpose(R_b)

    # 构造 PIML 缩聚器并按能量一致性构造缩聚刚度: Ks = K_bb + K_bi @ N_pred
    piml_condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    piml_condensor.condense(K_local_batch)
    piml_condensor.N = N_piml
    K_ib = K_local_batch[..., prototype.i_dofs[:, None], prototype.b_dofs]
    K_bb = K_local_batch[..., prototype.b_dofs[:, None], prototype.b_dofs]
    assert piml_condensor.N is not None
    piml_condensor.K_s = K_bb + bm.matrix_transpose(K_ib) @ piml_condensor.N

    system_piml = assembler.assemble_interface_system(sub_meshes, piml_condensor)
    u_interface_piml = solve_interface_system(
        system_piml,
        zero_load,
        fixed_interface_dofs,
        prescribed=prescribed_interface,
    )
    U_piml = assembler.recover_full_displacement(
        sub_meshes, piml_condensor, system_piml, u_interface_piml
    )
    t_piml = time.time() - t1

    # 3. 计算 L2 误差
    uh_exact = analyzer.tensor_space.function()
    uh_exact[:] = bm.reshape(U_exact, (-1,))
    l2_error_exact = float(assembler.full_mesh.error(pde.disp_solution, uh_exact, q=max(4, degree + 3)))

    uh_piml = analyzer.tensor_space.function()
    uh_piml[:] = bm.reshape(U_piml, (-1,))
    l2_error_piml = float(assembler.full_mesh.error(pde.disp_solution, uh_piml, q=max(4, degree + 3)))

    # 相对代数残差 ||U_piml - U_exact|| / ||U_exact||
    rel_disp_diff = float(bm.linalg.norm(U_piml - U_exact) / bm.linalg.norm(U_exact))

    total_fine = tuple(n_sub[d] * n_fine[d] for d in range(dim))
    mesh_size = float(domain_size[0] / total_fine[0])

    return {
        "n_sub": list(n_sub),
        "n_fine": list(n_fine),
        "total_fine": list(total_fine),
        "mesh_size": mesh_size,
        "full_dofs": int(assembler.total_full_dofs),
        "interface_dofs": int(len(system_exact.global_dofs)),
        "l2_error_exact": l2_error_exact,
        "l2_error_piml": l2_error_piml,
        "relative_diff": rel_disp_diff,
        "seconds_exact": float(t_exact),
        "seconds_piml": float(t_piml),
    }


### 问题与材料工厂 ###

PROBLEM_FACTORIES = {
    2: {
        "harmonic-poly": lambda: (
            HarmonicPoly2D(domain=(0.0, 1.0, 0.0, 1.0)),
            "plane_strain",
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
) -> Dict[str, Any]:
    """运行多层网格加密收敛阶评测并输出双轨对比报告."""
    if dim not in PROBLEM_FACTORIES or model not in PROBLEM_FACTORIES[dim]:
        raise ValueError(f"不支持的配置: dim={dim}, model={model}")

    pde, hypothesis = PROBLEM_FACTORIES[dim][model]()
    material = (
        IsotropicLinearElasticMaterial(
            youngs_modulus=pde.E,
            poisson_ratio=pde.nu,
            hypothesis=hypothesis,
        )
        if dim == 2
        else IsotropicLinearElasticMaterial(
            youngs_modulus=pde.E,
            poisson_ratio=pde.nu,
        )
    )

    if levels is None:
        levels = 4 if dim == 2 else 3

    # 多级子结构与细网格设置
    n_sub_base = (2, 2) if dim == 2 else (2, 2, 2)
    n_fine_levels = [
        tuple(2 ** (k + 1) for _ in range(dim)) for k in range(levels)
    ]

    print("=" * 108)
    print(f"PIML 子结构静力缩聚收敛阶评测 ({dim}D {model}, 连续制造解验证)")
    print("=" * 108)
    print(f"维度           : {dim}D")
    print(f"子结构划分     : {n_sub_base}")
    print(f"位移空间次数   : Q{degree}")
    print(f"网格加密层数   : {levels} 层 (细网格划分: {n_fine_levels})")
    print("-" * 108)

    records: List[Dict[str, Any]] = []
    for lvl_idx, n_fine in enumerate(n_fine_levels):
        print(f"--> 求解第 {lvl_idx + 1}/{levels} 层网格 (n_fine={n_fine})...")
        rec = solve_one_level(pde, material, n_sub_base, n_fine, degree=degree)
        records.append(rec)

    # 计算观测收敛阶
    for i in range(len(records)):
        if i == 0:
            records[i]["order_exact"] = None
            records[i]["order_piml"] = None
        else:
            h_ratio = records[i - 1]["mesh_size"] / records[i]["mesh_size"]
            e_ratio_exact = records[i - 1]["l2_error_exact"] / records[i]["l2_error_exact"]
            e_ratio_piml = records[i - 1]["l2_error_piml"] / records[i]["l2_error_piml"]
            records[i]["order_exact"] = float(math.log(e_ratio_exact) / math.log(h_ratio))
            records[i]["order_piml"] = float(math.log(e_ratio_piml) / math.log(h_ratio))

    # 打印排版表格
    headers = [
        "层级", "h", "全局细网格", "全场自由度",
        "精确 L2 误差", "精确阶",
        "PIML L2 误差", "PIML 阶",
        "代数相对差异",
    ]
    col_widths = [6, 10, 16, 12, 14, 8, 14, 8, 14]

    print("\n" + "=" * 108)
    print(format_table_row(headers, col_widths))
    print("-" * 108)

    for i, r in enumerate(records):
        ord_exact_str = f"{r['order_exact']:.2f}" if r["order_exact"] is not None else "-"
        ord_piml_str = f"{r['order_piml']:.2f}" if r["order_piml"] is not None else "-"
        row = [
            f"L{i + 1}",
            f"{r['mesh_size']:.4f}",
            "x".join(str(n) for n in r["total_fine"]),
            str(r["full_dofs"]),
            f"{r['l2_error_exact']:.4e}",
            ord_exact_str,
            f"{r['l2_error_piml']:.4e}",
            ord_piml_str,
            f"{r['relative_diff']:.3e}",
        ]
        print(format_table_row(row, col_widths))
    print("=" * 108 + "\n")

    # 保存 JSON 结果
    if output_dir is None:
        output_dir_path = Path(__file__).resolve().parent / "outputs"
    else:
        output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    json_path = output_dir_path / f"convergence_rate_{dim}d.json"
    result_data = {
        "dimension": dim,
        "model": model,
        "degree": degree,
        "levels": levels,
        "n_sub": list(n_sub_base),
        "records": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"评测结果已保存至: {json_path}")

    return result_data


def main() -> None:
    """命令行主函数."""
    parser = argparse.ArgumentParser(
        description="PIML 子结构静力缩聚收敛阶评测",
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=[2, 3],
        default=2,
        help="空间维度 (2 或 3, 默认 2)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=None,
        help="网格加密层数 (2D 缺省 4, 3D 缺省 3)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=1,
        help="有限元空间多项式次数 (默认 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出结果目录 (缺省为脚本同级 outputs/)",
    )
    args = parser.parse_args()
    run_convergence_benchmark(
        dim=args.dim,
        levels=args.levels,
        degree=args.degree,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
