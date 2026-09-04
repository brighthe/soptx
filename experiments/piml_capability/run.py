# -*- coding: utf-8 -*-
"""PIML 能力验证实验调度与数据快照生成入口.

用法:
    python experiments/piml_capability/run.py --list
    python experiments/piml_capability/run.py --all
    python experiments/piml_capability/run.py --case <case_id>
    python experiments/piml_capability/run.py --panel <a|b|c|d>
    python experiments/piml_capability/run.py --collect
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import collect
import config


def command_list(cases: Sequence[config.Case]) -> int:
    header = f"{'Panel':<6} {'Case ID':<18} {'Target':<22} {'Artifact':<36} {'Status'}"
    print(header)
    print("-" * len(header))
    for c in cases:
        st = "READY" if c.artifact_path.is_file() else "MISSING"
        print(
            f"{c.panel:<6} {c.id:<18} {c.role:<22} "
            f"{c.artifact_path.name:<36} {st}"
        )
    return 0


def command_run(cases: Sequence[config.Case], strict: bool = False) -> int:
    import subprocess
    total = len(cases)
    print(f"准备运行 {total} 个数据点...")
    for idx, c in enumerate(cases, start=1):
        print(f"\n[{idx}/{total}] 运行 case: {c.id} ({c.summary})")
        print(f"  脚本: {c.script}")
        print(f"  参数: {' '.join(c.args)}")
        res = subprocess.run(c.command())
        code = res.returncode
        if code != 0:
            print(f"  [FAIL] case {c.id} 返回非零退出码: {code}")
            if strict:
                return code
        else:
            print(f"  [OK] 产物: {c.artifact_path}")
    return 0


def command_collect(cases: Sequence[config.Case], figure: dict) -> int:
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
    print(f"  变分回推缩聚刚度误差: {p_b['net_k']*100:.2f}%  (误差压缩 20 倍)")

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
        description="图 4 (PIML 局部力学表示与精确缩聚) 数据点的调度与快照采集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true", help="列出已注册的数据点及产物状态")
    selection.add_argument("--all", action="store_true", help="跑全部数据点")
    selection.add_argument("--case", help="只跑指定 case id")
    selection.add_argument("--panel", choices=config.PANELS, help="只跑指定图面分格的数据点")
    selection.add_argument("--collect", action="store_true", help="读取既有产物生成快照并判定门禁")
    parser.add_argument("--strict", action="store_true", help="遇到非零退出码立刻终止")

    args = parser.parse_args(argv)
    figure, cases = config.load_cases()

    if args.list:
        return command_list(cases)

    if args.collect:
        return command_collect(cases, figure)

    if args.all:
        selected = cases
    elif args.case:
        selected = tuple(c for c in cases if c.id == args.case)
        if not selected:
            print(f"错误: 未找到 case: {args.case}")
            return 2
    elif args.panel:
        selected = tuple(c for c in cases if c.panel == args.panel)
        if not selected:
            print(f"错误: 未找到 panel: {args.panel}")
            return 2
    else:
        parser.print_help()
        return 2

    return command_run(selected, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
