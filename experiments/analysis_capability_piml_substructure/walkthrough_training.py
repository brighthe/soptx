"""PIML 子结构离线流程: 局部样本生成, 网络训练与权重保存.

以下三种模式中选择一种:

- --samples-only: 只生成样本, 不训练网络;
- --samples-dir DIR: 用已有样本训练, 从其 manifest.json 恢复局部问题配置;
- --from-scratch: 完整离线阶段, 生成样本后训练.

每次只训练 --route 所选的一个网络, 权重写入
<outputs-root>/independent_15_layer/training/<route>/<时间戳>/.
"""

import argparse
import json
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
# 正式样本与权重的存放根目录; 文件清单与 sha256 见该目录下 SHA256SUMS.
DATA_ROOT = Path.home() / "codespace" / "data" / "soptx" / "piml_substructure"
LAYOUT = "independent_15_layer"


def parse_args(argv=None):
    """解析离线样本生成、样本复用与网络训练参数."""
    parser = argparse.ArgumentParser(description="PIML 子结构离线训练")

    artifacts = parser.add_argument_group("产物与流程选择")
    mode = artifacts.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--samples-only", action="store_true",
        help="只生成样本, 不训练网络",
    )
    mode.add_argument(
        "--samples-dir", type=Path,
        help="用该已有样本目录训练, 从其 manifest.json 恢复局部问题配置",
    )
    mode.add_argument(
        "--from-scratch", action="store_true",
        help="完整离线阶段: 生成样本后训练",
    )
    artifacts.add_argument(
        "--outputs-root", type=Path, default=DATA_ROOT,
        help="新样本与权重的写入根目录, 均位于其下 independent_15_layer/",
    )

    problem = parser.add_argument_group(
        "局部问题定义",
        "仅用于 --samples-only 与 --from-scratch, 同时用于检查是否已有同配置样本.",
    )
    problem.add_argument("--dim", type=int, choices=[2, 3])
    problem.add_argument("--n-fine", type=int, help="子结构各方向细单元数")
    problem.add_argument(
        "--cell-size", type=float, nargs="+",
        help="子结构各方向尺寸, 数量须与 --dim 一致",
    )
    problem.add_argument("--nu", type=float, help="泊松比")
    problem.add_argument(
        "--hypothesis", choices=["plane_stress", "plane_strain"],
        help="二维材料假设; 三维不接受此参数",
    )
    problem.add_argument(
        "--trace-kind", choices=["linear_corner", "full_trace"], help="接口空间",
    )

    learning = parser.add_argument_group("学习目标选择")
    learning.add_argument(
        "--route", choices=["shape", "stiffness"], default="shape",
        help=(
            "学习目标; shape 只训练形函数网络, stiffness 只训练刚度网络; "
            "刚度路线恢复内部位移所需的形函数网络须另以 shape 训练"
        ),
    )

    samples = parser.add_argument_group(
        "离线样本生成",
        "仅用于 --samples-only 与 --from-scratch; 除 --generation-batch-size 外"
        "均参与同配置样本检查.",
    )
    samples.add_argument("--n-train", type=int, help="训练样本数")
    samples.add_argument("--n-validation", type=int, help="验证样本数")
    samples.add_argument(
        "--min-modulus", type=float, help="归一化杨氏模量的采样下界",
    )
    samples.add_argument(
        "--sample-seed", type=int,
        help="材料采样主种子, 派生训练与验证两条独立随机流",
    )
    samples.add_argument(
        "--generation-batch-size", type=int,
        help="每批局部精确求解的样本数",
    )

    training = parser.add_argument_group("离线训练", "指定 --samples-only 时忽略.")
    training.add_argument(
        "--num-networks", type=int, default=4,
        help=(
            "所选路线的输出拆分网络数量, 默认 4, 不得超过该路线的独立输出数; "
            "正式刚度权重使用 1"
        ),
    )
    training.add_argument(
        "--epochs", type=int, default=500,
        help="最大训练轮数; 默认与正式数据一致",
    )
    training.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="训练设备",
    )
    training.add_argument(
        "--train-seed", type=int, default=2026,
        help="网络初始化与每轮训练样本打乱的种子",
    )
    training.add_argument(
        "--batch-size", type=int, default=256,
        help="训练与验证的小批量样本数",
    )
    training.add_argument(
        "--optimizer", choices=["adam", "adamw", "sgd"], default="adam",
        help="训练优化器",
    )
    training.add_argument(
        "--lr", type=float, default=1e-3,
        help="初始学习率; 验证损失连续 10 轮未降时减半, 下限 1e-6",
    )
    training.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="权重衰减系数, 0 表示不启用",
    )
    training.add_argument(
        "--patience", type=int, default=40,
        help="验证损失连续未改善的提前停止轮数, 0 表示不启用",
    )
    args = parser.parse_args(argv)

    for name in ("samples_dir", "outputs_root"):
        path = getattr(args, name)
        if path is not None:
            path = path.expanduser()
            setattr(args, name, path if path.is_absolute() else CURRENT_DIR / path)

    problem_names = ("dim", "n_fine", "cell_size", "nu", "hypothesis", "trace_kind")
    sample_names = (
        "n_train", "n_validation", "min_modulus", "sample_seed",
        "generation_batch_size",
    )
    if args.samples_dir is not None:
        conflicts = [
            f"--{name.replace('_', '-')}"
            for name in problem_names + sample_names
            if getattr(args, name) is not None
        ]
        if conflicts:
            parser.error(
                "--samples-dir 从 manifest.json 恢复配置, 不接受以下参数 (生成样本请用 --samples-only 或 --from-scratch): "
                + ", ".join(conflicts)
            )
    else:
        defaults = {
            "dim": 3, "n_fine": 5, "nu": 0.3, "trace_kind": "linear_corner",
            "n_train": 400_000, "n_validation": 40_000,
            "min_modulus": 1e-6, "sample_seed": 2026,
            "generation_batch_size": 32,
        }
        for name, value in defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
        if args.cell_size is None:
            args.cell_size = [1.0] * args.dim
        if args.n_fine < 2:
            parser.error("--n-fine 至少为 2, 以保留内部自由度")
        if len(args.cell_size) != args.dim or any(
            not isfinite(size) or size <= 0 for size in args.cell_size
        ):
            parser.error("--cell-size 须包含与 --dim 同数量的有限正数")
        if not -1.0 < args.nu < 0.5:
            parser.error("--nu 须位于 (-1, 0.5)")
        if args.dim == 3 and args.hypothesis is not None:
            parser.error("--hypothesis 仅用于二维")
        if args.dim == 2 and args.hypothesis is None:
            args.hypothesis = "plane_stress"
        if min(args.n_train, args.n_validation, args.generation_batch_size) <= 0:
            parser.error("样本数与 --generation-batch-size 必须为正整数")
        if args.sample_seed < 0:
            parser.error("--sample-seed 不能为负数")
        if not 0 < args.min_modulus < 1:
            parser.error("--min-modulus 须位于 (0, 1)")

    if args.num_networks <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--num-networks, --epochs 与 --batch-size 必须为正整数")
    if args.train_seed < 0:
        parser.error("--train-seed 不能为负数")
    if not isfinite(args.lr) or args.lr < 1e-6:
        parser.error("--lr 须为有限数且不低于学习率调度下限 1e-6")
    if not isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay 须为有限非负数")
    if args.patience < 0:
        parser.error("--patience 不能为负数")
    return args


def main(argv=None):
    """按所选模式生成或恢复样本, 训练所选路线并打印结果目录."""
    args = parse_args(argv)

    from soptx.backend import backend_manager as bm
    from soptx.fem.substructure.independent_targets import IndependentTargetProvider
    from soptx.ml.substructure.independent_contract import (
        SCHEMA, build_network, provider_metadata_matches,
    )
    from soptx.ml.substructure.independent_data import (
        find_training_data, prepare_training_data,
    )
    from soptx.ml.substructure.independent_training import train_network
    from soptx.ml.substructure.training import TrainingConfig

    bm.set_backend("numpy")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    if args.samples_dir is None:
        provider = IndependentTargetProvider(
            cell_size=tuple(args.cell_size), n_fine=(args.n_fine,) * args.dim,
            nu=args.nu, trace_kind=args.trace_kind, hypothesis=args.hypothesis,
        )
        metadata = provider.metadata()
        samples_root = args.outputs_root / LAYOUT / "samples"
        sampling = {
            "n_train": args.n_train, "n_validation": args.n_validation,
            "min_modulus": args.min_modulus, "seed": args.sample_seed,
        }
        existing, skipped = find_training_data(samples_root, provider, **sampling)
        for path in skipped:
            print(f"[样本] 跳过未完成或无法识别的目录 {path}", flush=True)
        if existing is not None:
            raise FileExistsError(
                f"已有同配置样本 {existing}; 训练请用 --samples-dir {existing}; "
                "若旧样本已失效, 请先移走或改用其他 --outputs-root"
            )
        samples_dir = samples_root / stamp
        print(
            f"[样本] 生成训练 {args.n_train:,} / 验证 {args.n_validation:,}, "
            f"写入 {samples_dir}", flush=True,
        )
        samples_dir = prepare_training_data(
            provider, samples_dir, batch_size=args.generation_batch_size,
            **sampling,
        )
    else:
        samples_dir = args.samples_dir
        manifest_path = samples_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
                or manifest.get("complete") is not True
                or manifest.get("input_quantity") != "normalized_young_modulus"):
            raise ValueError("保存的数据集记录未完成或格式不支持")
        saved = manifest.get("provider")
        required = {
            "cell_size", "n_fine", "poisson_ratio", "trace", "spatial_dimension",
        }
        if not isinstance(saved, dict) or not required.issubset(saved):
            raise ValueError("保存的元数据缺少子结构配置")
        if saved["spatial_dimension"] == 2 and "material_hypothesis" not in saved:
            raise ValueError("二维数据集缺少 material_hypothesis")
        provider = IndependentTargetProvider(
            cell_size=tuple(saved["cell_size"]), n_fine=tuple(saved["n_fine"]),
            nu=saved["poisson_ratio"], trace_kind=saved["trace"],
            hypothesis=saved.get("material_hypothesis"),
        )
        metadata = provider.metadata()
        if not provider_metadata_matches(saved, metadata):
            raise ValueError("恢复的子结构配置或独立分量编号与数据集不一致")
        print(
            f"[样本] 恢复配置: trace={saved['trace']}, "
            f"cell_size={saved['cell_size']}, n_fine={saved['n_fine']}",
            flush=True,
        )

    print(
        f"[参考子结构] trace {metadata['trace']}, n_fine {tuple(metadata['n_fine'])}, "
        f"n_i {metadata['n_i']}, n_trace {metadata['n_trace']}, "
        f"n_rigid {metadata['n_rigid']}"
    )
    if args.samples_only:
        print(f"[完成] 样本目录: {samples_dir}")
        return
    training_dir = args.outputs_root / LAYOUT / "training" / args.route / stamp
    print(
        f"[权重] 训练 {args.route}, epochs {args.epochs}, "
        f"device {args.device}, 写入 {training_dir}", flush=True,
    )
    config = TrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size, optimizer=args.optimizer,
        optimizer_params={"lr": args.lr, "weight_decay": args.weight_decay},
        seed=args.train_seed, patience=args.patience,
    )
    network = build_network(
        metadata, route=args.route, seed=args.train_seed,
        num_networks=args.num_networks,
    )
    result = train_network(
        samples_dir, training_dir, route=args.route, network=network,
        provider=provider, device=args.device, config=config,
    )
    print(
        f"[完成] 最佳轮 {result['best_epoch']} / 共 {result['epochs_run']} 轮, "
        f"最佳验证损失 {result['best_validation_loss']:.6e}"
    )
    print(f"[完成] 权重: {result['checkpoint']}")


if __name__ == "__main__":
    main()
