"""生成三维 MBB 梁 PIML 子结构的离线样本与两类精确标签.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path

DOMAIN_SIZE = (6.0, 1.0, 1.0)
INTEGRATION_ORDER = 2
DATA_ROOT = Path.home() / "codespace" / "data" / "soptx" / "piml_substructure"
LAYOUT = "independent_15_layer"


def parse_args(argv=None):
    """解析局部配置, 采样参数与运行设置.

    Parameters
    ----------
    argv : sequence of str or None
        命令行参数, None 使用当前进程参数.

    Returns
    -------
    argparse.Namespace
        校验后的配置, cell_size 由 DOMAIN_SIZE 与 n_sub 推导.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    problem = parser.add_argument_group("三维 MBB 局部问题")
    problem.add_argument(
        "--n-sub", type=int, nargs=3, default=(78, 13, 13),
        metavar=("NX", "NY", "NZ"), help="各方向子结构数, 用于推导局部尺寸",
    )
    problem.add_argument("--n-fine", type=int, default=5,
                         help="子结构每方向细单元数 m, 至少为 2")
    problem.add_argument("--nu", type=float, default=0.3, help="泊松比")
    problem.add_argument(
        "--trace-kind", choices=("linear_corner", "full_trace"),
        default="linear_corner", help="局部接口类型",
    )
    sampling = parser.add_argument_group("样本生成参数")
    sampling.add_argument("--n-train", type=int, default=400_000, help="训练样本数")
    sampling.add_argument("--n-validation", type=int, default=40_000,
                          help="独立验证样本数")
    sampling.add_argument("--min-modulus", type=float, default=1e-7,
                          help="归一化杨氏模量独立均匀采样下界, 上界为 1")
    sampling.add_argument("--sample-seed", type=int, default=2026,
                          help="派生训练与验证独立随机流的主种子")
    runtime = parser.add_argument_group("运行设置")
    runtime.add_argument("--backend", choices=("numpy", "pytorch"), default="numpy",
                         help="批量局部计算后端")
    runtime.add_argument("--device", default="cpu",
                         help="计算设备: cpu, cuda 或 cuda:N, CUDA 需要 pytorch")
    runtime.add_argument("--generation-batch-size", type=int, default=32,
                         help="每批局部精确求解的样本数")
    runtime.add_argument("--outputs-root", type=Path, default=DATA_ROOT,
                         help="产物根目录, 相对路径相对于当前工作目录")
    args = parser.parse_args(argv)
    if args.device != "cpu" and args.device != "cuda" and not (
        args.device.startswith("cuda:") and args.device[5:].isascii()
        and args.device[5:].isdigit()
    ):
        parser.error("--device 须为 cpu, cuda 或 cuda:N")
    if args.backend == "numpy" and args.device != "cpu":
        parser.error("numpy 后端仅支持 --device cpu, CUDA 请使用 --backend pytorch")
    if min(args.n_sub) < 1:
        parser.error("--n-sub 各方向须为正整数")
    if args.n_fine < 2:
        parser.error("--n-fine 至少为 2, 以保留内部节点")
    if not isfinite(args.nu) or not -1 < args.nu < 0.5:
        parser.error("--nu 须为 (-1, 0.5) 内的有限数")
    if min(args.n_train, args.n_validation, args.generation_batch_size) < 1:
        parser.error("样本数与 --generation-batch-size 须为正整数")
    if not isfinite(args.min_modulus) or not 0 < args.min_modulus < 1:
        parser.error("--min-modulus 须为 (0, 1) 内的有限数")
    if args.sample_seed < 0:
        parser.error("--sample-seed 须为非负整数")
    args.cell_size = tuple(size / count for size, count in zip(DOMAIN_SIZE, args.n_sub))
    args.outputs_root = args.outputs_root.expanduser().resolve()
    return args


def main(argv=None):
    """生成训练集与验证集, 保存精确标签及运行配置.

    Parameters
    ----------
    argv : sequence of str or None
        命令行参数, None 使用当前进程参数.

    Returns
    -------
    pathlib.Path
        新生成数据集目录, 包含 manifest.json, 六个 NPY 文件与运行配置.

    Raises
    ------
    FileExistsError
        已存在相同局部配置, 采样设置与样本数量的完整数据集.
    """
    args = parse_args(argv)
    from soptx.backend import backend_manager as bm
    from soptx.fem.substructure.independent_targets import IndependentTargetProvider
    from soptx.ml.substructure.independent_data import (
        find_training_data, prepare_training_data,
    )

    # 参考子结构与标签编号始终由 NumPy 构建, 批量计算使用所选设备.
    bm.set_backend("numpy")
    provider = IndependentTargetProvider(
        cell_size=args.cell_size, n_fine=(args.n_fine,) * 3,
        nu=args.nu, trace_kind=args.trace_kind, integration_order=INTEGRATION_ORDER,
        backend=args.backend, device=args.device,
    )
    metadata = provider.metadata()
    sampling = dict(n_train=args.n_train, n_validation=args.n_validation,
                    min_modulus=args.min_modulus, seed=args.sample_seed)
    samples_root = args.outputs_root / LAYOUT / "samples"
    existing, skipped = find_training_data(samples_root, provider, **sampling)
    for directory in skipped:
        print(f"[样本] 跳过未完成或无法识别的目录: {directory}", flush=True)
    if existing is not None:
        raise FileExistsError(
            f"已有同配置样本: {existing}. 不重复生成, 如需独立副本请另设 --outputs-root."
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = samples_root / stamp
    print(f"[局部配置] 三维 Q1, 二阶高斯积分, 接口 {args.trace_kind}, "
          f"网格 {(args.n_fine,) * 3}, 尺寸 {args.cell_size}, nu={args.nu}", flush=True)
    print(f"[样本] 训练 {args.n_train:,}, 验证 {args.n_validation:,}, "
          f"独立均匀采样 [{args.min_modulus:g}, 1), seed={args.sample_seed}, float64",
          flush=True)
    print(f"[标签] 形函数独立条目 {metadata['n_shape_targets']}, "
          f"刚度独立条目 {metadata['n_stiffness_targets']}", flush=True)
    print(f"[计算] backend={args.backend}, device={provider.device}, "
          f"batch={args.generation_batch_size}", flush=True)
    print(f"[输出] {output_dir}", flush=True)
    output_dir = prepare_training_data(
        provider, output_dir, batch_size=args.generation_batch_size, **sampling,
    )
    config = {
        "domain_size": DOMAIN_SIZE, "n_sub": args.n_sub,
        "cell_size": args.cell_size, "n_fine": [args.n_fine] * 3,
        "trace_kind": args.trace_kind, "poisson_ratio": args.nu,
        "finite_element_degree": 1, "integration_order": INTEGRATION_ORDER,
        **sampling, "generation_batch_size": args.generation_batch_size,
        "backend": args.backend, "device": provider.device, "dtype": "float64",
    }
    (output_dir / "generation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"[完成] 样本目录: {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()
