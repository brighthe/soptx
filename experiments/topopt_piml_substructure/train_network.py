"""使用已有三维 MBB 子结构样本训练单条预测路线.

Notes
-----
复用独立条目监督训练接口, 从 manifest.json 恢复 codec, 不重新生成样本.
训练后端固定为 PyTorch, 数据与模型使用 float64. 尚未启用后期一致性损失.
"""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import re

DATA_ROOT = Path.home() / "codespace" / "data" / "soptx" / "piml_substructure"
LAYOUT = "independent_15_layer"
DATASET_NAME = "mbb_linear_corner_m5_h1_nu0p3_emin1e-7_seed2026"
STIFFNESS_HIDDEN_DIMS = (50, 60, 70, 80, 90, 100, 90, 80, 70, 60, 50)
STIFFNESS_ACTIVATIONS = (
    "tanh", "elu", "tanh", "elu", "tanh", "elu",
    "elu", "tanh", "elu", "tanh", "elu",
)


def parse_args(argv=None):
    """解析样本位置, 网络结构及训练运行参数.

    Parameters
    ----------
    argv : sequence of str or None
        命令行参数, None 使用当前进程参数.

    Returns
    -------
    argparse.Namespace
        已校验的训练设置, 相对路径以当前工作目录为基准.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples-dir", type=Path,
                        default=DATA_ROOT / LAYOUT / "samples" / DATASET_NAME,
                        help="已有完整样本目录, 局部配置由 manifest.json 恢复")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="从相同局部契约的已有权重微调, 恢复网络结构并新建优化器")
    parser.add_argument("--route", choices=("shape", "stiffness"), default="shape",
                        help="本次训练的预测路线")
    parser.add_argument("--input-normalization", choices=("none", "per_sample_max"),
                        help="shape 输入处理; 未指定时沿用初始化权重, 新网络默认 none")
    parser.add_argument("--num-networks", type=int,
                        help="拆分网络数, 角点 shape 默认 4, stiffness 默认 1")
    parser.add_argument("--hidden-dims", type=int, nargs="+",
                        help="各隐藏层宽度, 角点默认 shape 15 层或 stiffness 11 层")
    parser.add_argument("--activations", nargs="+", choices=("tanh", "elu", "relu"),
                        help="各隐藏层激活函数, 与 --hidden-dims 同时指定且等长")
    parser.add_argument("--stiffness-loss-weight", type=float, default=0.,
                        help="shape 相对重构刚度监督权重, 0 保持原 MSE")
    parser.add_argument("--train-stiffness-weights", type=Path,
                        help="训练专用刚度权重 npy 及同名 json 来源记录, 验证仍等权")
    parser.add_argument("--loss-chunk-size", type=int,
                        help="梯度累积计算块大小, 不改变 batch-size")
    parser.add_argument("--epochs", type=int, default=500, help="最大训练轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="训练与验证批量")
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd"),
                        default="adam", help="优化器")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=0., help="权重衰减")
    parser.add_argument("--patience", type=int, default=40,
                        help="验证损失连续未改善的停止轮数, 0 禁用提前停止")
    parser.add_argument("--train-seed", type=int, default=2026,
                        help="网络初始化与训练样本打乱的随机种子")
    parser.add_argument("--backend", choices=("pytorch",), required=True,
                        help="显式指定训练后端, 当前仅支持 pytorch")
    parser.add_argument("--device", default="cpu", help="训练设备: cpu, cuda 或 cuda:N")
    parser.add_argument("--outputs-root", type=Path, default=DATA_ROOT,
                        help="训练产物根目录")
    parser.add_argument("--output-dir", type=Path,
                        help="显式指定新训练目录, 默认按数据集名称与训练种子命名")
    args = parser.parse_args(argv)
    if args.route != "shape" and args.input_normalization == "per_sample_max":
        parser.error("per_sample_max 仅支持 shape 路线")
    if not isfinite(args.stiffness_loss_weight) or args.stiffness_loss_weight < 0:
        parser.error("--stiffness-loss-weight 须为有限非负数")
    if args.stiffness_loss_weight and args.route != "shape":
        parser.error("重构刚度监督仅支持 shape 路线")
    if args.train_stiffness_weights is not None and not args.stiffness_loss_weight:
        parser.error("--train-stiffness-weights 要求启用 shape 刚度监督")
    if args.loss_chunk_size is not None and args.loss_chunk_size <= 0:
        parser.error("--loss-chunk-size 必须为正整数")
    if min(args.epochs, args.batch_size) <= 0:
        parser.error("--epochs 与 --batch-size 必须为正整数")
    if args.num_networks is not None and args.num_networks <= 0:
        parser.error("--num-networks 必须为正整数")
    if (args.hidden_dims is None) != (args.activations is None):
        parser.error("--hidden-dims 与 --activations 须同时指定")
    if args.hidden_dims is not None and (
        min(args.hidden_dims) <= 0 or len(args.hidden_dims) != len(args.activations)
    ):
        parser.error("隐藏层宽度须为正整数, 且与激活函数数量一致")
    if not isfinite(args.lr) or args.lr < 1e-6:
        parser.error("--lr 必须为有限数且不低于调度下限 1e-6")
    if not isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay 必须为有限非负数")
    if min(args.patience, args.train_seed) < 0:
        parser.error("--patience 与 --train-seed 不能为负数")
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", args.device):
        parser.error("--device 须为 cpu, cuda 或 cuda:N")
    for name in ("samples_dir", "outputs_root", "output_dir", "init_checkpoint", "train_stiffness_weights"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())
    args.automatic_output = args.output_dir is None
    if args.output_dir is None:
        args.output_dir = (args.outputs_root / LAYOUT / "training" / args.route
                           / f"{args.samples_dir.name}_trainseed{args.train_seed}")
    return args


def main(argv=None):
    """恢复样本契约, 构建网络并保存验证损失最优的权重.

    Parameters
    ----------
    argv : sequence of str or None
        命令行参数, None 使用当前进程参数.

    Returns
    -------
    dict
        最佳权重位置, 最佳轮次及验证损失.
    """
    args = parse_args(argv)
    import torch
    from torch import nn
    from soptx.fem.substructure.independent_targets import IndependentPredictionDecoder
    from soptx.ml.substructure.independent_contract import (
        ACTIVATIONS, HIDDEN_DIMS, SCHEMA, build_network,
    )
    from soptx.ml.substructure.independent_training import train_network
    from soptx.ml.substructure.training import TrainingConfig

    manifest = json.loads((args.samples_dir / "manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
            or manifest.get("complete") is not True
            or manifest.get("input_quantity") != "normalized_young_modulus"
            or manifest.get("dtype") != "float64"):
        raise ValueError("样本未完成或数据格式不支持")
    metadata = manifest.get("provider")
    if not isinstance(metadata, dict) or metadata.get("spatial_dimension") != 3:
        raise ValueError("本入口仅用于三维子结构样本")
    if args.init_checkpoint is None and metadata["trace"] == "full_trace" and (
        args.num_networks is None or args.hidden_dims is None
    ):
        raise ValueError("三维 full_trace 须显式指定网络数量, 隐藏层与激活函数")
    provider = IndependentPredictionDecoder.from_metadata(metadata)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("当前环境无可用 CUDA")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("CUDA 设备编号越界")
    activation_classes = {"tanh": nn.Tanh, "elu": nn.ELU, "relu": nn.ReLU}
    if args.hidden_dims is not None:
        hidden_dims = tuple(args.hidden_dims)
        activation = tuple(activation_classes[name] for name in args.activations)
    elif args.route == "shape":
        hidden_dims, activation = HIDDEN_DIMS, ACTIVATIONS
    else:
        hidden_dims = STIFFNESS_HIDDEN_DIMS
        activation = tuple(activation_classes[name] for name in STIFFNESS_ACTIVATIONS)
    count = args.num_networks if args.num_networks is not None else (
        4 if args.route == "shape" else 1
    )
    config = TrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size, optimizer=args.optimizer,
        optimizer_params={"lr": args.lr, "weight_decay": args.weight_decay},
        seed=args.train_seed, patience=args.patience,
    )
    initialization = None
    if args.init_checkpoint is None:
        network = build_network(metadata, route=args.route, seed=args.train_seed,
                                num_networks=count, hidden_dims=hidden_dims, activation=activation,
                                input_normalization=args.input_normalization or "none")
    else:
        from soptx.ml.substructure.independent_checkpoints import load_independent_network

        network, initialization = load_independent_network(
            args.init_checkpoint, metadata, route=args.route,
        )
        if args.input_normalization is not None:
            network.input_normalization = args.input_normalization
        count = len(network.nets) if hasattr(network, "nets") else 1
        if args.num_networks is not None and args.num_networks != count:
            raise ValueError("--num-networks 与初始化权重不一致")
        if args.hidden_dims is not None:
            from soptx.ml.substructure.independent_contract import activation_names

            expected = tuple(activation_classes[name].__name__ for name in args.activations)
            if (tuple(args.hidden_dims) != tuple(network.hidden_dims)
                    or expected != tuple(activation_names(network))):
                raise ValueError("显式网络结构与初始化权重不一致")
        hidden_dims = tuple(network.hidden_dims)
        print(f"[初始化] {args.init_checkpoint}; 仅加载模型参数, 新建优化器", flush=True)
    print(f"[样本] {args.samples_dir}", flush=True)
    print(f"[配置] 接口={metadata['trace']}, 网格={metadata['n_fine']}, "
          f"尺寸={metadata['cell_size']}, nu={metadata['poisson_ratio']}", flush=True)
    print(f"[网络] 路线={args.route}, 网络数={count}, 隐藏层={hidden_dims}", flush=True)
    print(f"[训练] backend={args.backend}, device={device}, float64, epochs={args.epochs}, "
          f"batch={args.batch_size}, optimizer={args.optimizer}, lr={args.lr}, "
          f"seed={args.train_seed}, 一致性损失未启用", flush=True)
    print(f"[输入] input_normalization={network.input_normalization}", flush=True)
    if args.automatic_output and network.input_normalization != "none":
        args.output_dir = args.output_dir.with_name(
            f"{args.output_dir.name}_{network.input_normalization}")
    if args.automatic_output and args.stiffness_loss_weight:
        suffix = f"_kweight{args.stiffness_loss_weight:g}".replace(".", "p")
        args.output_dir = args.output_dir.with_name(args.output_dir.name + suffix)
    if args.automatic_output and args.train_stiffness_weights is not None:
        args.output_dir = args.output_dir.with_name(args.output_dir.name + "_hardweights")
    print(f"[损失] stiffness_loss_weight={args.stiffness_loss_weight}, "
          f"loss_chunk_size={args.loss_chunk_size}; 刚度教师为精确标签", flush=True)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"训练目录已存在, 不覆盖或续训: {args.output_dir}")
    print(f"[输出] {args.output_dir}", flush=True)
    result = train_network(args.samples_dir, args.output_dir, route=args.route,
                           network=network, provider=provider, device=device, config=config,
                           initialization=initialization,
                           stiffness_loss_weight=args.stiffness_loss_weight,
                           loss_chunk_size=args.loss_chunk_size,
                           train_stiffness_weights=args.train_stiffness_weights)
    print(f"[完成] 最佳轮={result['best_epoch']}, 总轮数={result['epochs_run']}, "
          f"验证损失={result['best_validation_loss']:.6e}", flush=True)
    print(f"[完成] 权重: {result['checkpoint']}", flush=True)
    return result


if __name__ == "__main__":
    main()
