"""对固定密度的三维 MBB 梁比较精确子结构与 shape 网络分析.

Notes
-----
两条路线共用 linear_corner 接口, 不执行过滤或优化迭代. 为检验网络本身,
禁用均匀子结构的精确替代. 本结果不等同于完整细网格有限元基线.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from soptx.backend import backend_manager as bm
from soptx.fem.substructure import (
    GlobalAssembler, IndependentPredictionDecoder, StructuredSubstructureLayout,
    build_interface_space, build_modulus_substructures,
    make_density_fields, recover_full_displacement_batches, solve_constrained_system,
)
from soptx.ml.substructure.independent_checkpoints import (
    decoder_metadata_matches, load_analysis_provider, load_independent_network,
)
from soptx.problems.elasticity import FullMBBBeam3d

DATA_ROOT = Path.home() / "codespace/data/soptx/piml_substructure"
LAYOUT = "independent_15_layer"
TRAINING = "mbb_linear_corner_m5_h1_nu0p3_emin1e-7_seed2026_trainseed2026"


def parse_args(argv=None):
    """解析并校验固定密度对照设置.

    Parameters
    ----------
    argv : sequence of str or None
        命令行参数, None 使用当前进程参数.

    Returns
    -------
    argparse.Namespace
        已校验的运行设置.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=
                        DATA_ROOT / LAYOUT / "training/shape" / TRAINING / "shape_best.pt")
    parser.add_argument("--n-sub", type=int, nargs=3, default=(12, 2, 2))
    fields = parser.add_mutually_exclusive_group()
    fields.add_argument("--density-pattern", choices=("uniform", "smooth"), default="uniform")
    fields.add_argument("--density-file", type=Path,
                        help="已过滤的物理密度 NPY, 形状须与全局细网格一致")
    parser.add_argument("--density", type=float, default=0.12,
                        help="uniform 模式的物理密度")
    parser.add_argument("--density-range", type=float, nargs=2, default=(0.04, 0.20),
                        help="smooth 模式的密度构造区间; 实际范围与均值保存至结果")
    parser.add_argument("--min-modulus", type=float, default=1e-7)
    parser.add_argument("--penal", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backend", choices=("numpy",), required=True,
                        help="有限元分析后端; 网络推理固定使用 PyTorch")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="新建结果目录, 拒绝覆盖已有目录")
    args = parser.parse_args(argv)
    if any(n <= 0 for n in args.n_sub) or args.batch_size <= 0:
        parser.error("子结构数和 batch size 必须为正整数")
    if not np.isfinite(args.density) or not 0 <= args.density <= 1:
        parser.error("density 必须为 [0, 1] 内的有限值")
    if not np.isfinite(args.min_modulus) or not 0 < args.min_modulus <= 1:
        parser.error("min-modulus 必须为 (0, 1] 内的有限值")
    if not np.isfinite(args.penal) or args.penal <= 0:
        parser.error("penal 必须为有限正数")
    lo, hi = args.density_range
    if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi <= 1:
        parser.error("density-range 须满足 0 <= lo < hi <= 1")
    return args


def relative_error(predicted, reference):
    """计算相对 Euclidean 或 Frobenius 范数误差.

    Parameters
    ----------
    predicted, reference : numpy.ndarray
        相同形状的预测值与非零参考值.

    Returns
    -------
    float
        范数差除以参考范数.
    """
    return float(np.linalg.norm(predicted - reference) / np.linalg.norm(reference))


def analyze(space, layout, meshes, load, constraints, stiffness, shapes, psi):
    """组装接口系统, 求解并恢复完整位移.

    Parameters
    ----------
    space : InterfaceSpace
        两条路线共用的协调接口空间.
    layout : StructuredSubstructureLayout
        全局细网格布局.
    meshes : sequence
        与局部矩阵同序的子结构网格.
    load : numpy.ndarray
        接口载荷.
    constraints : scipy.sparse.csr_matrix
        齐次支承约束矩阵.
    stiffness, shapes, psi : numpy.ndarray
        各子结构接口刚度, 内部形函数及共用迹矩阵.

    Returns
    -------
    tuple
        接口位移, 完整位移和求解诊断.
    """
    count = len(meshes)
    system = space.assemble([SimpleNamespace(start=0, end=count, stiffness=stiffness)])
    result = solve_constrained_system(system, load, constraints, solver="scipy")
    q = np.asarray(result.displacement)
    local_q = q[np.asarray(space.local_dofs)]
    batch = SimpleNamespace(
        start=0, end=count, boundary=local_q @ psi.T,
        internal=np.einsum("bij,bj->bi", shapes, local_q),
    )
    displacement = np.asarray(recover_full_displacement_batches(layout, meshes, [batch]))
    compliance = float(load @ q)
    if not np.isfinite(displacement).all() or not np.isfinite(compliance) or compliance <= 0:
        raise ValueError("位移或柔顺度无效")
    diagnostics = {
        "compliance": compliance,
        "equilibrium_relative_residual": result.equilibrium_relative_residual,
        "constraint_relative_residual": result.constraint_relative_residual,
        "constraint_rank": result.constraint_rank, "solve_mode": result.mode,
    }
    return q, displacement, diagnostics


def main(argv=None):
    """执行对照, 将设置, 位移与误差保存至新目录.

    Parameters
    ----------
    argv : sequence of str or None
        命令行参数, None 使用当前进程参数.
    """
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有目录: {args.output_dir}")
    started = perf_counter()
    bm.set_backend(args.backend)
    provider = load_analysis_provider(args.checkpoint.parent)
    metadata = provider.metadata()
    if metadata["spatial_dimension"] != 3 or metadata["trace"] != "linear_corner":
        raise ValueError("此入口仅支持三维 linear_corner 权重")
    network, source = load_independent_network(args.checkpoint, metadata, route="shape")
    domain = tuple(float(h) * n for h, n in zip(metadata["cell_size"], args.n_sub))
    grid = tuple(int(m) * n for m, n in zip(metadata["n_fine"], args.n_sub))
    layout = StructuredSubstructureLayout(
        domain_size=domain, n_sub=tuple(args.n_sub), n_fine=tuple(metadata["n_fine"]),
        E_base=1.0, nu=metadata["poisson_ratio"], hypothesis="3D",
    )
    prototype, meshes, _ = build_modulus_substructures(layout, integration_order=2)
    decoder = IndependentPredictionDecoder(prototype, trace_kind="linear_corner")
    if not decoder_metadata_matches(metadata, decoder.metadata()):
        raise ValueError("整体分析参考子结构与权重契约不相容")
    space = build_interface_space("linear_corner", GlobalAssembler(layout), meshes, prototype)
    psi = np.asarray(provider.trace.matrix)
    if not np.allclose(psi, np.asarray(space.trace_basis.matrix), rtol=1e-12, atol=1e-12):
        raise ValueError("整体接口迹与权重接口迹不一致")
    problem = FullMBBBeam3d(
        domain=(0, domain[0], 0, domain[1], 0, domain[2]), P=-1.0,
        E=1.0, nu=metadata["poisson_ratio"], support="end_lines",
        load_subdivisions=(grid[0], grid[2]),
    )
    load, constraints = space.constrained_conditions(problem)
    density_source = {"pattern": args.density_pattern}
    if args.density_file is not None:
        density = np.asarray(np.load(args.density_file, allow_pickle=False), dtype=np.float64)
        density_source = {
            "pattern": "file", "path": str(args.density_file.resolve()),
            "sha256": sha256(args.density_file.read_bytes()).hexdigest(),
        }
    elif args.density_pattern == "smooth":
        local_density = make_density_fields(meshes, domain, tuple(args.density_range))
        density = np.asarray(layout.merge_substructure_cell_field(local_density))
        density_source["construction_range"] = list(args.density_range)
    else:
        density = np.full(grid, args.density, dtype=np.float64)
        density_source["uniform_density"] = args.density
    if density.shape != grid:
        raise ValueError(f"物理密度形状 {density.shape} 须为 {grid}")
    if not np.isfinite(density).all() or np.any(density < 0) or np.any(density > 1):
        raise ValueError("物理密度须为 [0, 1] 内的有限值")
    density_statistics = {
        "min": float(density.min()), "max": float(density.max()),
        "mean": float(density.mean()), "standard_deviation": float(density.std()),
    }
    local_density = np.asarray(layout.split_global_cell_field(density)).reshape(len(meshes), -1)
    deviations = np.abs(local_density.max(axis=1) - local_density.mean(axis=1))
    uniform_count = int(np.count_nonzero(deviations < 1e-4))
    modulus = args.min_modulus + density ** args.penal * (1.0 - args.min_modulus)
    blocks = layout.split_global_cell_field(modulus)
    inputs = np.stack([prototype.grid_to_cell_field(block) for block in blocks])
    exact_shapes, predicted_shapes, exact_stiffness, predicted_stiffness = [], [], [], []
    print(f"[配置] 细网格 {grid}, 域 {domain}, 子结构 {len(meshes)}", flush=True)
    print(f"[密度] {density_source['pattern']}: {density_statistics}; "
          f"满足均匀判据 {uniform_count}/{len(meshes)}", flush=True)
    for start in range(0, len(meshes), args.batch_size):
        x = inputs[start:start + args.batch_size]
        exact = provider.exact_matrices(x)
        with torch.inference_mode():
            outputs = network(torch.from_numpy(x)).cpu().numpy()
        predicted = provider.shape_codec.decode(outputs)
        basis = np.empty((len(x), prototype.n_total_dofs, psi.shape[1]))
        basis[:, prototype.b_dofs, :] = psi
        basis[:, prototype.i_dofs, :] = predicted
        reduced = basis.swapaxes(1, 2) @ exact["local_stiffness"] @ basis
        exact_shapes.append(exact["shape"])
        predicted_shapes.append(predicted)
        exact_stiffness.append(exact["stiffness"])
        predicted_stiffness.append(reduced)
        print(f"[局部] {start + len(x)}/{len(meshes)}", flush=True)
    te, tp, ke, kp = map(np.concatenate,
                         (exact_shapes, predicted_shapes, exact_stiffness, predicted_stiffness))
    qe, ue, de = analyze(space, layout, meshes, load, constraints, ke, te, psi)
    qp, up, dp = analyze(space, layout, meshes, load, constraints, kp, tp, psi)
    summary = {
        "exact": de, "shape_network": dp,
        "density_statistics": density_statistics,
        "uniform_substructure_count": uniform_count,
        "substructure_count": len(meshes),
        "uniform_density_threshold": 1e-4,
        "relative_errors": {
            "shape": relative_error(tp, te), "local_stiffness": relative_error(kp, ke),
            "interface_displacement": relative_error(qp, qe),
            "full_displacement": relative_error(up, ue),
            "compliance": abs(dp["compliance"] / de["compliance"] - 1),
        },
        "elapsed_seconds": perf_counter() - started,
        "uniform_shortcut": "disabled_for_network_diagnostic",
        "reference": "exact_substructure_with_same_linear_corner_trace",
    }
    config = {
        "checkpoint": source, "n_sub": args.n_sub, "fine_grid": grid,
        "domain": domain, "provider": metadata, "density_source": density_source,
        "physical_density_statistics": density_statistics, "filter_applied": False,
        "min_modulus": args.min_modulus, "penal": args.penal,
        "backend": args.backend, "inference_backend": "pytorch", "device": args.device,
        "batch_size": args.batch_size, "support": "end_lines", "total_load_y": -1.0,
        "uniform_shortcut": summary["uniform_shortcut"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, values in (
        ("physical_density", density), ("normalized_modulus", modulus),
        ("exact_interface_displacement", qe), ("shape_interface_displacement", qp),
        ("exact_full_displacement", ue), ("shape_full_displacement", up),
    ):
        np.save(args.output_dir / f"{name}.npy", values)
    for name, values in (("run_config", config), ("summary", summary)):
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print(f"[完成] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
