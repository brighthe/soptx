# -*- coding: utf-8 -*-
"""轴承装置全实体域的 h 收敛闭锁考察: 固定 nu, 逐级加密网格, 看各离散误差随 h 的走势.

固定网格扫 nu 只能说明该网格上谁闭锁; 本脚本走另一条轴: 固定 nu (0.3 与 0.4999
两档), 网格 30x10 -> 240x80 四级, 离散 lfem p=1 / lfem p=2 / Hu--Zhang k=2
(跳量稳定化) / Hu--Zhang k=3,4, 全实体域 rho == 1. 参考值取 Hu--Zhang
k=4 前三级按逐级差等比递减外推的极限 (Aitken delta^2, 不假定收敛阶; 240x80 上
k=4 求解过慢, 不跑). 目的是区分 "p=2 不闭锁" 与 "p=2 的闭锁被细网格掩盖": 若 p=2
在 nu=0.4999 的粗网格误差显著高于 nu=0.3 档且收敛率下降, 而 k=2 两档曲线重合,
则 p=2 的闭锁只是被 120x40 掩盖.

入口由 ``plot.py`` 派发, 本模块不直接执行::

    plot.py bearing-h-locking -> run()
    plot.py bearing-h-locking --analyzer huzhang --order 3

默认保留已有结果并只补缺项, 新增求解单独记录溯源. 全部补算完成后才原子写回.

分析链由 ``run_bearing.build`` 组装 (与优化运行同一份代码), 两档 nu 即其两组材料.
产出 ``results/bearing/postprocess/solid_h_sweep.json``, 终端打印
每档 nu 的绝对值表与相对参考值的偏差表.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_bearing
from config import OUTPUT_DIR

from . import provenance

CASE_ID = run_bearing.CASE_ID
# 两档 nu 与 run_bearing 的两组材料一一对应 (组名 -> 泊松比)
NU_GROUPS: dict[float, str] = {nu: group for group, (nu, _) in run_bearing.GROUPS.items()}
NU_LEVELS: tuple[float, ...] = tuple(NU_GROUPS)
MESH_LEVELS: tuple[tuple[int, int], ...] = ((30, 10), (60, 20), (120, 40), (240, 80))
ANALYSES: tuple[tuple[str, int], ...] = (("lfem", 1), ("lfem", 2), ("huzhang", 2), ("huzhang", 3), ("huzhang", 4))
REFERENCE = "huzhang-4"


def _label(method: str, order: int) -> str:
    return f"{method}-{order}"


def solid_compliance(nu: float, nx: int, ny: int, method: str, order: int) -> tuple[float, int, float]:
    parts = run_bearing.build(NU_GROUPS[nu], method, order, nx, ny)
    rho = parts["density"]
    rho[:] = 1.0
    started = time.perf_counter()
    state = parts["analyzer"].solve_state(rho_val=rho)
    compliance = float(parts["objective"].fun(density=rho, state=state))
    return compliance, int(rho.shape[0]), time.perf_counter() - started


def run(argv: list[str] | None = None) -> int:
    """保留已有结果, 按离散筛选补算缺项.

    Parameters
    ----------
    argv : list[str] or None, optional
        命令行参数.

    Returns
    -------
    int
        成功时返回 0.
    """
    parser = argparse.ArgumentParser(description="轴承全实体网格加密, 默认补算缺项")
    parser.add_argument("--analyzer", choices=("lfem", "huzhang"))
    parser.add_argument("--order", type=int, choices=(1, 2, 3, 4))
    args = parser.parse_args(argv)
    selected = [(m, k) for m, k in ANALYSES
                if (args.analyzer is None or m == args.analyzer)
                and (args.order is None or k == args.order)]
    if not selected:
        parser.error("没有匹配的离散组合")
    target = OUTPUT_DIR / CASE_ID / "postprocess" / "solid_h_sweep.json"
    source = provenance.file_digest(target) if target.exists() else None
    previous = json.loads(target.read_text(encoding="utf-8")) if source else None
    if previous:
        if previous.get("case_id") != CASE_ID or previous.get("design") != "solid":
            parser.error("已有结果不是轴承全实体数据")
        existing = {(r["poisson_ratio"], r["nx"], r["ny"]): r for r in previous["rows"]}
        expected = {(nu, nx, ny) for nu in NU_LEVELS for nx, ny in MESH_LEVELS}
        if set(existing) != expected or len(previous["rows"]) != len(expected):
            parser.error("已有结果网格或材料组不完整或重复")
        for nu in NU_LEVELS:
            if str(nu) not in previous.get("reference_values", {}):
                parser.error("缺少已有外推参考值")
            for nx, ny in MESH_LEVELS[:3]:
                if REFERENCE not in existing[nu, nx, ny]["compliance"]:
                    parser.error("缺少 k=4 参考数据")
    else:
        existing = {}
        if ("huzhang", 4) not in selected:
            parser.error("首次运行须包含 k=4, 请不带筛选参数运行以建立参考数据")
    stamp = provenance.run_stamp()
    rows: list[dict[str, Any]] = []
    added = []
    for nu in NU_LEVELS:
        for nx, ny in MESH_LEVELS:
            row: dict[str, Any] = {"poisson_ratio": nu, "nx": nx, "ny": ny, "compliance": {}, "elapsed": {}}
            row = existing.get((nu, nx, ny), row)
            for method, order in selected:
                label = _label(method, order)
                if (method, order) == ("huzhang", 4) and (nx, ny) == MESH_LEVELS[-1]:
                    continue  # k=4 在 240x80 上求解超过 20 min, 不作参考; 参考值改由前三级等比外推
                if label in row["compliance"]:
                    continue
                value, ncell, elapsed = solid_compliance(nu, nx, ny, method, order)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"无效柔顺度: {value}")
                if "cells" in row and row["cells"] != ncell:
                    raise ValueError("单元数与已有结果不一致")
                row["cells"] = ncell
                row["compliance"][label] = value
                row["elapsed"][label] = elapsed
                added.append({"poisson_ratio": nu, "nx": nx, "ny": ny, "analysis": label})
                print(f"[solid-h] nu={nu:<7} mesh={nx}x{ny} {label}: C={value:.6f} ({elapsed:.1f}s)", flush=True)
            rows.append(row)

    if not added:
        print("指定结果已存在, 未求解或改写文件.")
        return 0

    labels = [_label(m, o) for m, o in ANALYSES]
    references: dict[str, float] = {}
    for nu in NU_LEVELS:
        block = [r for r in rows if r["poisson_ratio"] == nu]
        # 参考值: k=4 前三级按逐级差等比递减外推的极限 (Aitken delta^2), 不假定收敛阶;
        # 增量运行沿用已有参考值, 保持比较基准不变.
        c1, c2, c3 = (block[i]["compliance"][REFERENCE] for i in range(3))
        ratio = (c2 - c3) / (c1 - c2)
        reference = (previous["reference_values"][str(nu)] if previous
                     else c3 - (c2 - c3) * ratio / (1.0 - ratio))
        references[str(nu)] = reference
        print(f"\nnu={nu}, 全实体域, 参考值 = {REFERENCE} 逐级差比 {ratio:.3f} 的外推值 {reference:.6f}")
        print("| mesh | " + " | ".join(labels) + " | " + " | ".join(f"{l} 偏差" for l in labels) + " |")
        print("|---|" + "---|" * (2 * len(labels)))
        for r in block:
            c = r["compliance"]
            cells = " | ".join(f"{c[l]:.4f}" if l in c else "-" for l in labels)
            devs = " | ".join(f"{100.0 * (c[l] / reference - 1.0):+.3f}%" if l in c else "-" for l in labels)
            print(f"| {r['nx']}x{r['ny']} | {cells} | {devs} |")

    target = OUTPUT_DIR / CASE_ID / "postprocess" / "solid_h_sweep.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": CASE_ID,
        "design": "solid",
        "reference": f"{REFERENCE} 前三级等比外推极限 (Aitken delta^2)",
        "reference_values": references,
        "analyses": [list(a) for a in ANALYSES],
        "provenance": stamp,
        "rows": rows,
    }
    if previous:
        payload = previous
        for m, k in selected:
            if [m, k] not in payload["analyses"]:
                payload["analyses"].append([m, k])
        payload.setdefault("incremental_runs", []).append({
            "source": source, "provenance": stamp, "added": added,
        })
        if provenance.file_digest(target) != source:
            raise RuntimeError("运行期间源数据发生变化, 拒绝覆盖")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    print(f"\n写入 {target}")
    if not stamp.get("reproducible"):
        print("注意: 工作区不干净, 本次数字不满足 provenance.reproducible, 不可直接引用.")
    return 0

