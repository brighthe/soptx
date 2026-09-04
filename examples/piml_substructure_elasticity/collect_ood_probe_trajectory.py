# -*- coding: utf-8 -*-
"""采集真实拓扑优化的密度轨迹, 作为 PIML 代理模型的分布外 (OOD) 评估数据集.

背景
----
``verify_shape_function_route.py`` 与 ``verify_stiffness_route.py`` 的训练与留出
样本都取自 ``DENSITY_RANGE = (0.3, 1.0)`` 上各细单元**独立均匀**的随机采样, 因此
现有的留出误差与门禁触发统计只覆盖训练分布内部. 而拓扑优化在线部署时, 代理模型
面对的是 OC 迭代产生的密度场: 取值下界由 ``OCOptions._design_variable_min``
(默认 ``1e-9``) 决定, 远低于训练下界 ``0.3``; 空间上又被灵敏度/密度滤波强烈相关化,
与独立均匀采样正交.

本脚本负责产生这批真实工况数据, 本身不做任何 PIML 推理, 也不下 OOD 结论: 它只跑
一条**纯精确**的拓扑优化链 (直接法求解, 无代理模型), 把每次迭代每个子结构的局部
密度场落盘. 判定分布错配的严重程度由后续的重放评估脚本负责.

物理问题, 子结构划分与细网格规模全部对齐 ``verify_stiffness_route.py`` 的 24 子结构
装配系统 (Huang 2023 第 4.1 节 MBB 梁), 使轨迹数据可与已有的在役证据逐项对照.

产物
----
``outputs/ood_probe_trajectory_<tag>.npz``, 含:

- ``rho_sub``: ``(n_iter, n_sub_total, n_fine_x, n_fine_y)`` 逐迭代的子结构局部密度场,
  子结构按 x 优先字典序 ``sub_id = sx * n_sub_y + sy`` 排列, 与
  ``GlobalAssembler.reconstruct_global_field`` 的次序契约一致.
- ``rho_cell``: ``(n_iter, NC)`` 同一轨迹在全局网格单元编号下的原始排列, 供交叉核对.
- ``iter_indices``, ``changes``, ``compliance``, ``volfrac``: 逐迭代的标量历史.
- ``config``: 本次运行的完整配置 JSON 字符串.

用法
----
.. code-block:: bash

    # 默认配置: 密度滤波, rmin = 0.5 (本配置 hx = hy = 0.2, 合 2.5 个细单元).
    # --solve-method 默认为 mumps, 环境未装 pymumps 时须显式改用 scipy.
    python examples/piml_substructure_elasticity/collect_ood_probe_trajectory.py \
        --solve-method scipy

    # 加大滤波半径, 考察空间相关长度对分布错配的影响.
    # 注意 rmin 是物理长度而非单元个数: 梁厚只有 10 个细单元, rmin 取到 1.5 就相当于
    # 7.5 个单元, 整场被抹成灰度 (密度均值恰为 0.5, 两端各 0%), 那样的轨迹测不出 OOD.
    python examples/piml_substructure_elasticity/collect_ood_probe_trajectory.py \
        --rmin 1.0 --solve-method scipy --tag density_r10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.substructure import GlobalAssembler
from soptx.problems.elasticity import FullMBBBeam2d
from soptx.topology.constraints import VolumeConstraint
from soptx.topology.filters import Filter
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective
from soptx.topology.optimizers import OCOptimizer


# 问题配置的单一来源; 与 verify_stiffness_route.py / verify_shape_function_route.py
# 共用同一份投产参数 (12x2 子结构, 单个子结构 5x5 细单元, 合计 24 个子结构与 60x10
# 全局网格), 任何一方都不得在本地复制字面量.
# TRAINING_DENSITY_RANGE 只用于在配置中留痕, 本脚本不据此裁剪轨迹.
from deployment_config import (  # noqa: E402
    CELL_SIZE,
    DOMAIN,
    E_BASE,
    N_FINE,
    N_SUB,
    NU,
    OUTPUT_DIR,
    P_LOAD,
    TRAINING_DENSITY_RANGE,
)


def build_cell_to_grid_index(mesh: Any, assembler: GlobalAssembler) -> np.ndarray:
    """由单元重心反解结构化网格下标, 得到单元编号到网格线性下标的置换.

    参数:
        mesh: 全尺度结构化四边形网格, 即 ``assembler.full_mesh``.
        assembler: 提供求解域尺寸与各方向细单元总数的装配器.

    返回:
        grid_index: 形状 ``(NC,)`` 的整型数组, ``grid_index[c]`` 是第 ``c`` 号单元
            在行优先展平的 ``(total_fine_x, total_fine_y)`` 网格中的线性下标.

    异常:
        ValueError: 当反解结果不是一个双射时抛出, 说明网格并非预期的结构化划分.

    说明:
        不假定 ``QuadrangleMesh.from_box`` 的单元编号次序, 与
        ``GlobalAssembler.to_node_grid`` 由坐标反解节点下标的做法一致.
    """
    nx, ny = assembler.total_fine_x, assembler.total_fine_y
    hx = assembler.Lx / nx
    hy = assembler.Ly / ny

    bc = np.asarray(bm.to_numpy(mesh.entity_barycenter("cell")), dtype=np.float64)
    ix = np.clip((bc[:, 0] / hx).astype(np.int64), 0, nx - 1)
    iy = np.clip((bc[:, 1] / hy).astype(np.int64), 0, ny - 1)
    grid_index = ix * ny + iy

    if grid_index.shape[0] != nx * ny or np.unique(grid_index).size != nx * ny:
        raise ValueError(
            f"单元重心未能反解出 {nx}x{ny} 结构化网格的双射: "
            f"NC = {grid_index.shape[0]}, 不重复下标数 = {np.unique(grid_index).size}."
        )
    return grid_index


def split_into_substructures(
    rho_cell: np.ndarray, grid_index: np.ndarray, assembler: GlobalAssembler
) -> np.ndarray:
    """把全局单元密度切分为按子结构组织的局部密度场.

    参数:
        rho_cell: 形状 ``(n_iter, NC)`` 的逐迭代全局单元密度, 按网格单元编号排列.
        grid_index: ``build_cell_to_grid_index`` 给出的单元到网格线性下标的置换.
        assembler: 提供子结构与细单元布局的装配器.

    返回:
        rho_sub: 形状 ``(n_iter, n_sub_x * n_sub_y, n_fine_x, n_fine_y)`` 的局部密度场,
            子结构按 ``sub_id = sx * n_sub_y + sy`` 的 x 优先字典序排列.

    说明:
        切分是 ``GlobalAssembler.reconstruct_global_field`` 的严格逆运算: 后者把
        ``(n_sub_x, n_sub_y, n_fine_x, n_fine_y)`` 经 ``permute(0, 2, 1, 3)`` 展平为
        ``(total_fine_x, total_fine_y)``, 这里按相同的轴次序还原回去.
    """
    n_iter = rho_cell.shape[0]
    nsx, nsy = assembler.n_sub_x, assembler.n_sub_y
    nfx, nfy = assembler.n_fine_x, assembler.n_fine_y

    grid = np.empty(
        (n_iter, assembler.total_fine_x * assembler.total_fine_y), dtype=np.float64
    )
    grid[:, grid_index] = rho_cell
    grid = grid.reshape(n_iter, nsx, nfx, nsy, nfy)
    grid = np.transpose(grid, (0, 1, 3, 2, 4))
    return np.ascontiguousarray(grid.reshape(n_iter, nsx * nsy, nfx, nfy))


def check_roundtrip(
    rho_sub: np.ndarray,
    rho_cell: np.ndarray,
    grid_index: np.ndarray,
    assembler: GlobalAssembler,
) -> None:
    """用装配器自带的重构接口反向校验切分的正确性.

    参数:
        rho_sub: 切分得到的子结构局部密度场.
        rho_cell: 原始的全局单元密度.
        grid_index: 单元到网格线性下标的置换.
        assembler: 提供 ``reconstruct_global_field`` 的装配器.

    异常:
        ValueError: 当往返重构与原始密度不一致时抛出.

    说明:
        切分次序若与装配器的契约不符, 落盘的轨迹将是逐子结构错位的乱序数据, 而
        数值本身仍然"看起来正常", 因此这条校验必须在落盘前无条件执行.
    """
    for it in (0, rho_sub.shape[0] - 1):
        rebuilt = np.asarray(
            bm.to_numpy(assembler.reconstruct_global_field(bm.asarray(rho_sub[it])))
        ).reshape(-1)
        expected = np.empty_like(rebuilt)
        expected[grid_index] = rho_cell[it]
        max_dev = float(np.max(np.abs(rebuilt - expected)))
        if max_dev > 0.0:
            raise ValueError(
                f"第 {it} 次迭代的子结构切分未能通过往返重构校验, 最大偏差 {max_dev:.3e}; "
                f"请核对 n_sub / n_fine 布局与子结构排列次序."
            )


def run_trajectory(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """执行一次纯精确的拓扑优化并采集逐迭代密度轨迹.

    参数:
        args: 命令行参数命名空间.

    返回:
        (arrays, config): 待落盘的数组字典与本次运行的配置字典.

    说明:
        全链路不涉及任何代理模型: 状态方程由直接法求解, 局部刚度由精确单元装配得到,
        因此轨迹本身即为 PIML 侧的精确参照工况.
    """
    problem = FullMBBBeam2d(domain=DOMAIN, P=P_LOAD, E=E_BASE, nu=NU)
    domain_size = (problem.domain[1], problem.domain[3])
    assembler = GlobalAssembler(
        domain_size, N_SUB, N_FINE, E_base=problem.E, nu=problem.nu
    )
    mesh = assembler.full_mesh
    # GlobalAssembler 用 QuadrangleMesh.from_box 构造 full_mesh, 不带 soptx 过滤器
    # 依赖的 meshdata 元数据字典, 这里按全尺度结构化网格的真实参数补齐.
    xmin, xmax, ymin, ymax = problem.domain
    mesh.meshdata = {
        "domain": list(problem.domain),
        "mesh_type": "uniform_quad",
        "nx": assembler.total_fine_x,
        "ny": assembler.total_fine_y,
        "hx": (xmax - xmin) / assembler.total_fine_x,
        "hy": (ymax - ymin) / assembler.total_fine_y,
    }

    interpolation = MaterialInterpolationScheme(
        density_location="element",
        interpolation_method="simp",
        options={
            "penalty_factor": args.penalty_factor,
            "void_youngs_modulus": args.void_youngs_modulus,
            "target_variables": ["E"],
        },
        enable_logging=False,
    )
    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=assembler.material,
        space_degree=1,
        integration_order=4,
        assembly_method="standard",
        operator_level="fa",
        solve_method=args.solve_method,
        topopt_algorithm="density_based",
        interpolation_scheme=interpolation,
        enable_logging=False,
    )

    design_variable, density = interpolation.setup_density_distribution(
        design_variable_mesh=mesh,
        displacement_mesh=mesh,
        relative_density=args.volume_fraction,
    )
    objective = ComplianceObjective(
        analyzer=analyzer,
        state_variable="u",
        diff_mode="manual",
        enable_logging=False,
    )
    constraint = VolumeConstraint(
        analyzer=analyzer,
        volume_fraction=args.volume_fraction,
        diff_mode="manual",
        enable_logging=False,
    )
    density_filter = Filter(
        design_mesh=mesh,
        filter_type=args.filter_type,
        rmin=args.rmin,
        density_location="element",
        enable_logging=False,
    )
    optimizer = OCOptimizer(
        objective=objective,
        constraint=constraint,
        filter=density_filter,
        options={
            "max_iterations": args.max_iterations,
            "change_tolerance": args.change_tolerance,
        },
        enable_logging=True,
    )
    optimizer.options.set_advanced_options(
        move_limit=0.2,
        damping_coef=0.5,
        initial_lambda=1.0e9,
        bisection_tol=1.0e-3,
    )

    _, history = optimizer.optimize(
        design_variable=design_variable,
        density_distribution=density,
    )

    rho_cell = np.stack(
        [
            np.asarray(bm.to_numpy(rho[:]), dtype=np.float64)
            for rho in history.physical_densities
        ]
    )
    grid_index = build_cell_to_grid_index(mesh, assembler)
    rho_sub = split_into_substructures(rho_cell, grid_index, assembler)
    check_roundtrip(rho_sub, rho_cell, grid_index, assembler)

    arrays: Dict[str, Any] = {
        "rho_sub": rho_sub,
        "rho_cell": rho_cell,
        "grid_index": grid_index,
        "iter_indices": np.asarray(history.iter_indices, dtype=np.int64),
        "changes": np.asarray(history.changes, dtype=np.float64),
    }
    for key in ("compliance", "volfrac"):
        values = history.scalar_histories.get(key)
        if values is not None:
            arrays[key] = np.asarray(values, dtype=np.float64)

    config: Dict[str, Any] = {
        "problem": "FullMBBBeam2d",
        "domain": list(DOMAIN),
        "P": P_LOAD,
        "E": E_BASE,
        "nu": NU,
        "n_sub": list(N_SUB),
        "n_fine": list(N_FINE),
        "n_sub_total": int(N_SUB[0] * N_SUB[1]),
        "n_cells": int(mesh.number_of_cells()),
        "space_degree": 1,
        "solve_method": args.solve_method,
        "interpolation_method": "simp",
        "penalty_factor": args.penalty_factor,
        "void_youngs_modulus": args.void_youngs_modulus,
        "volume_fraction": args.volume_fraction,
        "filter_type": args.filter_type,
        "rmin": args.rmin,
        "max_iterations": args.max_iterations,
        "change_tolerance": args.change_tolerance,
        "design_variable_min": float(optimizer.options._design_variable_min),
        "training_density_range": list(TRAINING_DENSITY_RANGE),
        "n_iter_recorded": int(rho_sub.shape[0]),
    }
    return arrays, config


def summarize(arrays: Dict[str, Any], config: Dict[str, Any]) -> None:
    """打印轨迹与训练分布的粗粒度错配概览.

    参数:
        arrays: 落盘用的数组字典.
        config: 本次运行的配置字典.

    说明:
        这里只报告落在训练区间之外的密度占比等原始统计量, 供运行时快速核对数据是否
        合理; 分布错配的判定与归因由后续的重放评估脚本负责.
    """
    rho_sub = arrays["rho_sub"]
    lo, hi = config["training_density_range"]
    outside = np.mean((rho_sub < lo) | (rho_sub > hi), axis=(1, 2, 3))

    print()
    print("=" * 72)
    print(
        f"轨迹长度: {rho_sub.shape[0]} 次迭代, 子结构数 {rho_sub.shape[1]}, "
        f"单子结构细单元 {rho_sub.shape[2]}x{rho_sub.shape[3]}"
    )
    print(f"密度全局范围: [{rho_sub.min():.3e}, {rho_sub.max():.3e}]")
    print(f"训练区间: [{lo}, {hi}]")
    print(
        f"落在训练区间外的密度占比: 首次迭代 {outside[0]:.2%}, "
        f"末次迭代 {outside[-1]:.2%}, 全程均值 {outside.mean():.2%}"
    )
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        description="采集拓扑优化密度轨迹, 用于 PIML 代理模型的分布外评估.",
    )
    parser.add_argument(
        "--filter-type",
        default="density",
        choices=("none", "sensitivity", "density", "projection"),
        help="滤波类型; 直接影响密度场的空间相关结构.",
    )
    parser.add_argument(
        "--rmin",
        type=float,
        default=0.5,
        help=(
            "滤波半径, 物理长度, 与细单元尺寸同单位 "
            f"(本配置 hx = hy = {CELL_SIZE[0]:g}, 故 0.5 约合 2.5 个单元)."
        ),
    )
    parser.add_argument(
        "--volume-fraction", type=float, default=0.5, help="体积分数上限."
    )
    parser.add_argument(
        "--penalty-factor", type=float, default=3.0, help="SIMP 杨氏模量惩罚因子."
    )
    parser.add_argument(
        "--void-youngs-modulus", type=float, default=1e-9, help="孔洞杨氏模量."
    )
    parser.add_argument(
        "--max-iterations", type=int, default=200, help="OC 最大迭代次数."
    )
    parser.add_argument(
        "--change-tolerance",
        type=float,
        default=1e-2,
        help="设计变量无穷范数收敛阈值.",
    )
    parser.add_argument(
        "--solve-method",
        default="mumps",
        choices=("mumps", "scipy"),
        help="状态方程求解方式; 两者均为直接法, mumps 需环境装有 pymumps 与 MUMPS 库.",
    )
    parser.add_argument(
        "--tag", default=None, help="产物文件名后缀; 缺省时由滤波类型与半径自动生成."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="产物落盘目录."
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口: 跑一次优化, 校验切分, 落盘轨迹."""
    args = parse_args()
    tag = args.tag or f"{args.filter_type}_r{args.rmin:g}".replace(".", "p")

    arrays, config = run_trajectory(args)
    config["tag"] = tag
    summarize(arrays, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"ood_probe_trajectory_{tag}.npz"
    np.savez_compressed(
        out_path, config=json.dumps(config, ensure_ascii=False), **arrays
    )
    print(f"轨迹已落盘: {out_path}")


if __name__ == "__main__":
    main()
