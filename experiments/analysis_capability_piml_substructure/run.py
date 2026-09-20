# -*- coding: utf-8 -*-
"""PIML 子结构分析的工况调度与图件快照采集入口.

用法:
    python experiments/analysis_capability_piml_substructure/run.py --list
    python experiments/analysis_capability_piml_substructure/run.py --all
    python experiments/analysis_capability_piml_substructure/run.py --task <task>
    python experiments/analysis_capability_piml_substructure/run.py --case <case_id>
    python experiments/analysis_capability_piml_substructure/run.py --panel <a|b|c>
    python experiments/analysis_capability_piml_substructure/run.py --collect

选择器中 --task 是主入口, 按研究任务选工况; --panel 按图件分格选, 供图件工作流
使用。声明了 source 的工况计算归其他实验目录, 不会被执行。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import collect
import config


def _trace_label(case: config.AnalysisCase) -> str:
    return case.trace_basis or "-"


def _new_run_dir(output_root: Path, case_id: str) -> Path:
    """为单次运行创建不会覆盖历史证据的时间戳目录."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = output_root / case_id / stamp
    target.mkdir(parents=True, exist_ok=False)
    return target


def command_list(cases: Sequence[config.AnalysisCase], output_root: Path = config.OUTPUT_DIR) -> int:
    """列出已注册工况及其产物状态."""
    header = (
        f"{'Task':<30} {'Case ID':<34} {'Trace':<14} "
        f"{'Artifact':<38} {'Status'}"
    )
    print(header)
    print("-" * len(header))
    for c in cases:
        artifact_file = c.find_latest_artifact(output_root)
        if not c.runnable:
            status = "EXTERNAL" if (artifact_file and artifact_file.is_file()) else "MISSING"
        else:
            status = "READY" if (artifact_file and artifact_file.is_file()) else "MISSING"
        print(
            f"{c.task:<30} {c.id:<34} {_trace_label(c):<14} "
            f"{c.artifact_name:<38} {status}"
        )
    external = [c for c in cases if not c.runnable]
    if external:
        print("\nEXTERNAL 表示计算归其他实验目录, 本目录只引用其产物:")
        for c in external:
            print(f"  {c.id:<34} <- {c.source}")
    return 0


def run_case(
    case: config.AnalysisCase,
    *,
    output_root: Path,
    skip_train: bool = False,
    strict: bool = False,
) -> int:
    """执行单个已注册工况."""
    if not case.runnable:
        print(f"[跳过] {case.id}: 计算归 {case.source}, 本目录只引用其产物")
        return 0

    output_dir = _new_run_dir(output_root, case.id)
    snapshot = {
        "case": asdict(case),
        "timestamp": output_dir.name,
        "runtime": {
            "skip_train": skip_train,
            "strict": strict,
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[run] {case.id}: {case.summary}")
    print(f"[run] 结果目录: {output_dir}")

    if case.task == "shape_function_route":
        from examples.piml_substructure_elasticity.verify_shape_function_route import (
            build_parser,
            run_verification,
        )

        argv = [
            "--trace-basis", str(case.trace_basis),
            "--n-train", str(case.n_train),
            "--n-eval", str(case.n_eval),
            "--epochs", str(case.epochs),
            "--lr", str(case.lr),
            "--hidden-dim", str(case.hidden_dim),
            "--seed", str(case.seed),
            "--output-dir", str(output_dir),
        ]
        if skip_train:
            argv.append("--skip-train")
        args = build_parser().parse_args(argv)
        try:
            run_verification(args)
            print(f"[OK] 产物已生成至: {output_dir}")
            return 0
        except Exception as err:
            print(f"[FAIL] case {case.id} 执行失败: {err}")
            if strict:
                raise
            return 1

    elif case.task == "reduced_stiffness_route":
        from examples.piml_substructure_elasticity.verify_stiffness_route import (
            run_comparison,
        )

        try:
            run_comparison(
                n_train=case.n_train,
                n_epochs=case.epochs,
                learning_rate=case.lr,
                n_val=case.n_eval,
                seed=case.seed,
                output_dir=str(output_dir),
                backend=case.backend or "numpy",
                strict=strict,
            )
            print(f"[OK] 产物已生成至: {output_dir}")
            return 0
        except Exception as err:
            print(f"[FAIL] case {case.id} 执行失败: {err}")
            if strict:
                raise
            return 1

    print(f"[FAIL] 未支持的 task: {case.task}")
    return 1


def command_run(
    cases: Sequence[config.AnalysisCase],
    output_root: Path = config.OUTPUT_DIR,
    skip_train: bool = False,
    strict: bool = False,
) -> int:
    """按注册配置逐个执行工况."""
    runnable = [c for c in cases if c.runnable]
    skipped = [c for c in cases if not c.runnable]
    for c in skipped:
        print(f"[跳过] {c.id}: 计算归 {c.source}, 本目录只引用其产物")

    total = len(runnable)
    if total == 0:
        print("没有可执行的工况。")
        return 0

    print(f"准备运行 {total} 个工况...")
    for idx, c in enumerate(runnable, start=1):
        print(f"\n[{idx}/{total}] 调度 case: {c.id}")
        code = run_case(c, output_root=output_root, skip_train=skip_train, strict=strict)
        if code != 0 and strict:
            return code
    return 0


def command_collect(cases: Sequence[config.AnalysisCase], figure: dict) -> int:
    """读取既有产物生成图件快照并判定门禁."""
    try:
        snapshot = collect.build(tuple(cases), figure)
    except collect.CollectError as err:
        print(f"[错误] 采集失败: {err}")
        return 1

    path = collect.write(snapshot)
    record = snapshot["provenance"]

    print(f"快照已写入: {path}")
    print(f"  revision : {record['git_revision']}  (dirty={record['git_dirty']})")
    print(f"  采集时间 : {record['generated_at_utc']}")
    print(f"  Python   : {record['python']}")
    print(f"  PyTorch  : {record.get('torch')}")
    print(f"  CUDA 设备: {record.get('cuda_device') or 'CPU / 无 CUDA'}")

    p_a = snapshot["panels"]["a"]
    print("\n[Panel a: 真值代数等价性]")
    for lbl, val in zip(p_a["labels"], p_a["values"]):
        print(f"  {lbl:12s}: 相对差 = {val:.4e}  (门禁 <= {p_a['gate']:.1e})")

    p_b = snapshot["panels"]["b"]
    print("\n[Panel b: 二阶误差压缩机理]")
    print(f"  log-log 拟合斜率 : {p_b['slope']:.4f}  (理论值 2.00)")
    print(f"  形函数自身预测误差: {p_b['net_n']*100:.2f}%")
    print(f"  变分回推缩聚刚度误差: {p_b['net_k']*100:.2f}%")

    p_c = snapshot["panels"]["c"]
    print("\n[Panel c: 全系统求解保真度 (FullMBBBeam2d 24 子结构装配)]")
    for lbl, pval, dval in zip(p_c["labels"], p_c["piml_values"], p_c["direct_values"]):
        print(f"  {lbl:18s}: 预测形函数 = {pval*100:.2f}%  (预测刚度 = {dval*100:.2f}%)")

    p_d = snapshot["panels"]["d"]
    print("\n[Panel d: PIML 批量缩聚 GPU 加速]")
    print(f"  测试硬件: {p_d.get('device', 'NVIDIA GeForce RTX 5080')}")
    for n, tc, tg, sp in zip(p_d["n_subs"], p_d["t_cpu_ms"], p_d["t_gpu_ms"], p_d["speedup"]):
        print(f"  子结构数 = {n:3d} : 传统 CPU = {tc:7.2f} ms | PIML GPU = {tg:5.2f} ms | 加速比 = {sp:5.1f}x")

    failures = snapshot["gate_failures"]
    if failures:
        print(f"\n门禁未通过 ({len(failures)} 项):")
        for item in failures:
            print(f"  - {item}")
        return 1

    if not snapshot["reproducible"]:
        print("\n注意: 工作区 dirty 或无 git 溯源, 这批数字属开发证据, 不可复现。正式投递前须在 clean revision 上重跑。")

    print("\n门禁全部通过。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PIML 子结构分析工况的调度与图件快照采集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true", help="列出已注册工况及产物状态")
    selection.add_argument("--all", action="store_true", help="跑全部可执行工况")
    selection.add_argument("--task", choices=config.tasks(), help="只跑指定研究任务的工况")
    selection.add_argument("--case", help="只跑指定 case id")
    selection.add_argument("--panel", choices=config.PANELS, help="只跑指定图件分格的工况")
    selection.add_argument("--collect", action="store_true", help="读取既有产物生成快照并判定门禁")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR,
        help="输出根目录, 默认 experiments/analysis_capability_piml_substructure/outputs",
    )
    parser.add_argument("--skip-train", action="store_true", help="解析检查验证, 跳过耗时的神经网络训练")
    parser.add_argument("--strict", action="store_true", help="遇到非零退出码或异常立刻终止")

    args = parser.parse_args(argv)
    try:
        figure, cases = config.load_cases()
    except config.ConfigError as err:
        print(f"[错误] cases.toml 不合法: {err}")
        return 2

    if args.list:
        return command_list(cases, output_root=args.output_dir)

    if args.collect:
        return command_collect(cases, figure)

    if args.all:
        selected = cases
    else:
        try:
            selected = config.select(
                cases, case_id=args.case, task=args.task, panel=args.panel,
            )
        except config.ConfigError as err:
            print(f"[错误] {err}")
            return 2

    return command_run(
        selected,
        output_root=args.output_dir,
        skip_train=args.skip_train,
        strict=args.strict,
    )


if __name__ == "__main__":
    sys.exit(main())

