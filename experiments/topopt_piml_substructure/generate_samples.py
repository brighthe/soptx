"""生成三维 MBB 梁 PIML 子结构的离线样本与两类精确标签.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import re
from math import isfinite
from pathlib import Path

FINE_CELL_SIZE = 1.0
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
        校验后的配置, cell_size 与 domain_size 由细单元尺寸及网格推导.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    problem = parser.add_argument_group("三维 MBB 局部问题")
    problem.add_argument(
        "--n-sub", type=int, nargs=3, default=(78, 13, 13),
        metavar=("NX", "NY", "NZ"), help="各方向子结构数, 与 m 和细单元尺寸共同确定整体域尺寸",
    )
    problem.add_argument("--n-fine", type=int, default=5,
                         help="子结构每方向细单元数 m, 至少为 2")
    problem.add_argument("--fine-cell-size", type=float, default=FINE_CELL_SIZE,
                         help="样本参考细单元边长 h, 默认采用 h=1 的复现长度约定")
    problem.add_argument("--nu", type=float, default=0.3, help="泊松比")
    problem.add_argument(
        "--trace-kind", choices=("linear_corner", "full_trace"),
        default="linear_corner", help="局部接口类型",
    )
    sampling = parser.add_argument_group("样本生成参数")
    sampling.add_argument("--sampling", choices=("independent_uniform", "mixed"),
                          default="independent_uniform", help="原始均匀采样或分布覆盖补充试验")
    sampling.add_argument("--n-train", type=int, help="训练样本数: 原始默认 400000, mixed 默认 20000")
    sampling.add_argument("--n-validation", type=int,
                          help="验证样本数: 原始默认 40000, mixed 默认 4000")
    sampling.add_argument("--n-test", type=int, help="mixed 独立测试数量, 默认 2000")
    sampling.add_argument("--replay-samples-dir", type=Path,
                          default=DATA_ROOT / LAYOUT / "samples" / "mbb_linear_corner_m5_h1_nu0p3_emin1e-7_seed2026",
                          help="mixed 仅回放此目录的原训练输入, 不读取验证或优化快照")
    sampling.add_argument("--material-scale-range", type=float, nargs=2, default=(1e-5, 1.),
                          help="mixed 缩放材料的 log-uniform 区间")
    sampling.add_argument("--filter-radius", type=float, default=3., help="mixed 密度过滤物理半径")
    sampling.add_argument("--penal", type=float, default=3., help="mixed 密度转换的 SIMP 指数")
    sampling.add_argument("--min-modulus", type=float, default=1e-7,
                          help="归一化杨氏模量独立均匀采样下界, 上界为 1")
    sampling.add_argument("--sample-seed", type=int,
                          help="独立随机流主种子: 原始默认 2026, mixed 默认 2030")
    runtime = parser.add_argument_group("运行设置")
    runtime.add_argument("--backend", choices=("numpy", "pytorch"), default="numpy",
                         help="参考子结构与批量局部计算使用的后端")
    runtime.add_argument("--device", default="cpu",
                         help="批量计算设备: cpu, cuda 或 cuda:N, CUDA 需要 pytorch")
    runtime.add_argument("--generation-batch-size", type=int, default=32,
                         help="每批局部精确求解的样本数")
    runtime.add_argument("--dataset-name", help="数据集目录名, 未指定时按配置生成, 不使用时间戳")
    runtime.add_argument("--outputs-root", type=Path, default=DATA_ROOT,
                         help="产物根目录, 相对路径相对于当前工作目录")
    args = parser.parse_args(argv)
    mixed = args.sampling == "mixed"
    if args.n_train is None:
        args.n_train = 20_000 if mixed else 400_000
    if args.n_validation is None:
        args.n_validation = 4_000 if mixed else 40_000
    if args.n_test is None:
        args.n_test = 2_000 if mixed else 0
    if args.sample_seed is None:
        args.sample_seed = 2030 if mixed else 2026
    if mixed:
        if any(n <= 0 or n % 10 for n in (args.n_train, args.n_validation, args.n_test)):
            parser.error("mixed 三个划分的样本数须为 10 的正整数倍")
    elif args.n_test != 0:
        parser.error("--n-test 仅适用于 --sampling mixed")
    lo, hi = args.material_scale_range
    if not isfinite(lo) or not isfinite(hi) or not 0 < lo <= hi <= 1:
        parser.error("material-scale-range 须满足 0 < lo <= hi <= 1")
    if not isfinite(args.filter_radius) or args.filter_radius <= 0 or not isfinite(args.penal) or args.penal <= 0:
        parser.error("filter-radius 和 penal 须为有限正数")
    args.replay_samples_dir = args.replay_samples_dir.expanduser().resolve()
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
    if not isfinite(args.fine_cell_size) or args.fine_cell_size <= 0:
        parser.error("--fine-cell-size 须为有限正数")
    args.grid = tuple(n * args.n_fine for n in args.n_sub)
    args.cell_size = (args.n_fine * args.fine_cell_size,) * 3
    args.domain_size = tuple(n * args.fine_cell_size for n in args.grid)
    if any(not isfinite(size) or size <= 0 for size in args.cell_size + args.domain_size):
        parser.error("派生的子结构与整体域尺寸须为有限正数")
    args.outputs_root = args.outputs_root.expanduser().resolve()
    if args.dataset_name is None:
        nu = str(args.nu).replace(".", "p")
        emin = str(args.min_modulus).replace("e-0", "e-").replace("e+0", "e+").replace(".", "p")
        grid = "x".join(str(n) for n in args.n_sub)
        h = format(args.fine_cell_size, ".17g").replace(".", "p")
        args.dataset_name = (
            f"mbb_{args.trace_kind}_m{args.n_fine}_h{h}_sub{grid}_nu{nu}_emin{emin}"
            f"_train{args.n_train}_val{args.n_validation}_seed{args.sample_seed}"
        )
        if mixed:
            scale = "to".join(format(v, ".6g").replace(".", "p") for v in args.material_scale_range)
            radius = format(args.filter_radius, ".6g").replace(".", "p")
            penal = format(args.penal, ".6g").replace(".", "p")
            args.dataset_name += f"_mixed_test{args.n_test}_scale{scale}_r{radius}_p{penal}"
    if len(args.dataset_name) > 200 or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._+-]*", args.dataset_name,
    ) is None:
        parser.error("--dataset-name 须为单个目录名, 以英文字母或数字开头, "
                     "仅包含英文字母, 数字, 点, 下划线, 加减号, 长度不超过 200")
    return args


def main(argv=None):
    """生成训练与验证集, mixed 另生成独立测试集, 保存标签及配置.

    Parameters
    ----------
    argv : sequence of str or None
        命令行参数, None 使用当前进程参数.

    Returns
    -------
    pathlib.Path
        新数据目录含 manifest.json, 双路线标签与运行配置.
        mixed 另含测试集及样本类型, 镜像家族和回放索引记录.

    Raises
    ------
    FileExistsError
        目标目录已存在, 或已有同配置完整数据集.
    """
    args = parse_args(argv)
    samples_root = args.outputs_root / LAYOUT / "samples"
    output_dir = samples_root / args.dataset_name
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"数据集目录已存在, 不覆盖或续写: {output_dir}. 请另设 --dataset-name.")
    from soptx.backend import backend_manager as bm
    from soptx.fem.substructure.independent_targets import IndependentTargetProvider
    from soptx.ml.substructure.independent_data import (
        find_training_data, prepare_training_data,
    )

    bm.set_backend(args.backend)
    reference_context = nullcontext()
    if args.backend == "pytorch":
        import torch

        # 参考结构使用所选后端的 CPU 张量, 批量计算设备由 provider 显式设置.
        reference_context = torch.device("cpu")
    with reference_context:
        provider = IndependentTargetProvider(
            cell_size=args.cell_size, n_fine=(args.n_fine,) * 3,
            nu=args.nu, trace_kind=args.trace_kind, integration_order=INTEGRATION_ORDER,
            backend=args.backend, device=args.device,
        )
    metadata = provider.metadata()
    sampling = dict(n_train=args.n_train, n_validation=args.n_validation,
                    min_modulus=args.min_modulus, seed=args.sample_seed)
    existing, skipped = (None, []) if args.sampling == "mixed" else find_training_data(
        samples_root, provider, **sampling,
    )
    for directory in skipped:
        print(f"[样本] 跳过未完成或无法识别的目录: {directory}", flush=True)
    if existing is not None:
        raise FileExistsError(
            f"已有同配置样本: {existing}. 不重复生成, 如需独立副本请另设 --outputs-root."
        )
    print(f"[参考尺度] h={args.fine_cell_size:g}, 整体域 {args.domain_size}, "
          f"全局细网格 {args.grid}", flush=True)
    print(f"[局部配置] 三维 Q1, 二阶高斯积分, 接口 {args.trace_kind}, "
          f"网格 {(args.n_fine,) * 3}, 尺寸 {args.cell_size}, nu={args.nu}", flush=True)
    print(f"[样本] 训练 {args.n_train:,}, 验证 {args.n_validation:,}, 测试 {args.n_test:,}, "
          f"采样={args.sampling}, seed={args.sample_seed}, float64",
          flush=True)
    print(f"[标签] 形函数独立条目 {metadata['n_shape_targets']}, "
          f"刚度独立条目 {metadata['n_stiffness_targets']}", flush=True)
    print(f"[计算] backend={args.backend}, device={provider.device}, "
          f"batch={args.generation_batch_size}", flush=True)
    print(f"[输出] {output_dir}", flush=True)
    if args.sampling == "mixed":
        from mixed_sampling import prepare_mixed_training_data

        output_dir = prepare_mixed_training_data(
            provider, output_dir, batch_size=args.generation_batch_size, **sampling,
            n_test=args.n_test, replay_dir=args.replay_samples_dir,
            scale_range=tuple(args.material_scale_range),
            filter_radius=args.filter_radius, penal=args.penal,
        )
    else:
        output_dir = prepare_training_data(
            provider, output_dir, batch_size=args.generation_batch_size, **sampling,
        )
    config = {
        "dataset_name": args.dataset_name,
        "domain_size": args.domain_size, "n_sub": args.n_sub,
        "fine_cell_size": args.fine_cell_size, "grid": args.grid,
        "length_convention": "explicit_fine_cell_size",
        "cell_size": args.cell_size, "n_fine": [args.n_fine] * 3,
        "trace_kind": args.trace_kind, "poisson_ratio": args.nu,
        "finite_element_degree": 1, "integration_order": INTEGRATION_ORDER,
        **sampling, "sampling": args.sampling, "n_test": args.n_test,
        "mixed_parameters": None if args.sampling != "mixed" else dict(
            material_scale_range=args.material_scale_range,
            filter_radius=args.filter_radius, penal=args.penal,
            replay_samples_dir=str(args.replay_samples_dir),
        ), "generation_batch_size": args.generation_batch_size,
        "backend": args.backend, "device": provider.device, "dtype": "float64",
    }
    (output_dir / "generation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"[完成] 样本目录: {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()
