"""二维/三维 PIML 子结构: 构建网络、准备样本、监督训练."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parents[1] / "src"))


def parse_args():
    """解析命令行参数并检查参数组合."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--all", dest="stage", action="store_const", const="all",
                         help="构建网络、生成样本并训练")
    actions.add_argument("--generate-samples", dest="stage", action="store_const",
                         const="samples", help="仅生成样本")
    actions.add_argument("--train", dest="stage", action="store_const",
                         const="train", help="使用已有样本训练")
    parser.add_argument("--dim", type=int, choices=(2, 3), default=3)
    parser.add_argument("--n-fine", type=int, default=5, help="每个方向的细单元数, 至少为 2")
    parser.add_argument("--dataset", type=Path, help="已有数据集目录")
    parser.add_argument("--output-dir", type=Path, default=CURRENT_DIR / "outputs")
    parser.add_argument("--n-train", type=int, default=400_000)
    parser.add_argument("--n-validation", type=int, default=40_000)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--min-modulus", type=float, default=1e-6)
    parser.add_argument("--route", choices=("shape", "stiffness", "both"), default="both")
    parser.add_argument("--num-networks", type=int,
                        help="每条所选路线的网络数量; 默认形函数 4, 刚度 1")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (args.stage == "train") != (args.dataset is not None):
        parser.error("--train 必须指定 --dataset, 其他入口不接受 --dataset")
    if min(args.n_train, args.n_validation, args.generation_batch_size,
           args.epochs, args.batch_size) <= 0:
        parser.error("样本数、批量大小与训练轮数必须为正数")
    if args.num_networks is not None and args.num_networks <= 0:
        parser.error("--num-networks 必须为正整数")
    if args.n_fine < 2:
        parser.error("--n-fine 至少为 2, 以保留内部自由度")
    if not 0 < args.min_modulus < 1:
        parser.error("--min-modulus 必须位于 (0, 1)")
    if not args.lr >= 1e-6 or args.patience < 0 or args.seed < 0:
        parser.error("lr 不得低于 1e-6, patience 与 seed 不能为负")
    return args


def main():
    """按网络、样本、训练的顺序调用各模块."""
    args = parse_args()
    from soptx.fem.substructure.independent_targets import IndependentTargetProvider
    from soptx.ml.substructure.independent_training import (
        build_network, generate_dataset, train_networks,
    )
    from soptx.ml.substructure.training import TrainingConfig

    output_root = args.output_dir / "independent_15_layer"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    provider = IndependentTargetProvider(
        cell_size=(1.0,) * args.dim, n_fine=(args.n_fine,) * args.dim,
    )
    metadata = provider.metadata()

    # 1. 构建网络.
    if args.stage != "samples":
        routes = ("shape", "stiffness") if args.route == "both" else (args.route,)
        networks = {
            route: build_network(
                metadata, route=route, seed=args.seed, num_networks=args.num_networks,
            )
            for route in routes
        }

    # 2. 生成训练与验证样本, 或使用已有数据集.
    dataset = args.dataset
    if args.stage != "train":
        dataset = generate_dataset(
            provider, output_root / "samples" / stamp,
            n_train=args.n_train, n_validation=args.n_validation,
            batch_size=args.generation_batch_size,
            min_modulus=args.min_modulus, seed=args.seed,
        )
        print(f"数据集: {dataset}")
    if args.stage == "samples":
        return

    # 3. 训练网络并保存最佳权重.
    config = TrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, seed=args.seed, patience=args.patience,
    )
    output = output_root / "training" / stamp
    train_networks(
        dataset, output, networks=networks, codecs=provider.codecs,
        provider_metadata=metadata, route=args.route,
        device=args.device, config=config,
    )
    print(f"模型与训练记录: {output}")


if __name__ == "__main__":
    main()
