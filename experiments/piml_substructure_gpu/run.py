#!/usr/bin/env python3
"""子结构 PIML GPU 实验的惰性 CLI。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
from training import ExperimentError, run_case, run_pair


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="子结构 PIML GPU 实验（当前仅实现 training）"
    )
    choice = result.add_mutually_exclusive_group(required=True)
    choice.add_argument("--list", action="store_true", help="列出已注册工况")
    choice.add_argument("--case", help="运行一个 CPU 或 CUDA training 工况")
    choice.add_argument("--pair", help="运行同一数据/初值的 CPU 与 CUDA pair")
    result.add_argument("--samples", type=int, help="覆盖 pair 两端的样本数")
    result.add_argument("--epochs", type=int, help="覆盖 pair 两端的 epoch 数")
    result.add_argument("--batch-size", type=int, help="覆盖 pair 两端的 batch size")
    result.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    return result


def show(cases: tuple[config.TrainingCase, ...]) -> None:
    print("工况列表")
    id_width = max([len("case-id")] + [len(case.id) for case in cases])
    pair_width = max([len("pair-id")] + [len(case.pair_id) for case in cases])
    print(f"{'case-id':<{id_width}}  {'stage':<8}  {'device':<6}  {'pair-id':<{pair_width}}  description")
    for case in cases:
        print(f"{case.id:<{id_width}}  {'training':<8}  {case.device:<6}  {case.pair_id:<{pair_width}}  {case.summary}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        _, cases = config.load()
        if args.list:
            show(cases)
            return 0
        if args.case:
            selected = tuple(case for case in cases if case.id == args.case)
            if len(selected) != 1:
                raise config.ConfigError(f"未找到唯一工况: {args.case}")
            case = selected[0].with_overrides(args.samples, args.epochs, args.batch_size)
            path = run_case(case, args.output_dir)
        else:
            cpu, cuda = config.pair(cases, args.pair)
            cpu = cpu.with_overrides(args.samples, args.epochs, args.batch_size)
            cuda = cuda.with_overrides(args.samples, args.epochs, args.batch_size)
            path = run_pair(cpu, cuda, args.output_dir)
        print(f"结果已写入: {path}")
        return 0
    except (config.ConfigError, ExperimentError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
