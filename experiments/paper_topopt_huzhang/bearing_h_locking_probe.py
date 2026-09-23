# -*- coding: utf-8 -*-
"""轴承装置全实体域的 h 收敛闭锁考察: 固定 nu, 逐级加密网格, 看各离散误差随 h 的走势.

bearing_reanalysis.solid_sweep 固定 120x40 网格扫 nu, 只能说明该网格上谁闭锁;
本脚本补另一条轴: 固定 nu (0.3 与 0.4999 两档), 网格 30x10 -> 240x80 四级, 离散
lfem p=1 / lfem p=2 / Hu--Zhang k=2 (跳量稳定化) / Hu--Zhang k=4, 全实体域 rho == 1,
无材料插值. 参考值取 Hu--Zhang k=4 前三级按逐级差等比递减外推的极限 (Aitken delta^2,
不假定收敛阶; 240x80 上 k=4 求解过慢, 不跑). 目的是区分 "p=2 不闭锁" 与
"p=2 的闭锁被细网格掩盖": 若 p=2 在 nu=0.4999 的粗网格误差显著高于 nu=0.3 档且
收敛率下降, 而 k=2 两档曲线重合, 则 p=2 的闭锁只是被 120x40 掩盖.

直接运行::

    python bearing_h_locking_probe.py

产出 ``outputs/bearing-incompressible/postprocess/solid_h_sweep.json``, 终端打印
每档 nu 的绝对值表与相对参考值的偏差表.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from config import OUTPUT_DIR, bootstrap_source_path

bootstrap_source_path()

from bearing_reanalysis import case_parameters  # noqa: E402
from pipeline import build_bearing_analysis_pipeline, build_bearing_config  # noqa: E402
import provenance  # noqa: E402

CASE_ID = "bearing-incompressible"
NU_LEVELS: tuple[float, ...] = (0.3, 0.4999)
MESH_LEVELS: tuple[tuple[int, int], ...] = ((30, 10), (60, 20), (120, 40), (240, 80))
ANALYSES: tuple[tuple[str, int], ...] = (("lfem", 1), ("lfem", 2), ("huzhang", 2), ("huzhang", 4))
REFERENCE = "huzhang-4"


def _label(method: str, order: int) -> str:
    return f"{method}-{order}"


def solid_compliance(nu: float, nx: int, ny: int, method: str, order: int) -> tuple[float, int, float]:
    parameters = case_parameters(CASE_ID)
    parameters["comparison_orders"] = [order]
    parameters["nx"] = nx
    parameters["ny"] = ny
    parameters["poisson_ratio"] = float(nu)
    parameters["interpolation_variables"] = "auto"
    config = build_bearing_config(parameters)
    pipeline = build_bearing_analysis_pipeline(config, parameters, method, order)
    rho = pipeline.density_distribution
    rho[:] = 1.0
    started = time.perf_counter()
    state = pipeline.analyzer.solve_state(rho_val=rho)
    compliance = float(pipeline.objective.fun(density=rho, state=state))
    return compliance, int(rho.shape[0]), time.perf_counter() - started


def run() -> int:
    stamp = provenance.run_stamp()
    rows: list[dict[str, Any]] = []
    for nu in NU_LEVELS:
        for nx, ny in MESH_LEVELS:
            row: dict[str, Any] = {"poisson_ratio": nu, "nx": nx, "ny": ny, "compliance": {}, "elapsed": {}}
            for method, order in ANALYSES:
                label = _label(method, order)
                if (method, order) == ("huzhang", 4) and (nx, ny) == MESH_LEVELS[-1]:
                    continue  # k=4 在 240x80 上求解超过 20 min, 不作参考; 参考值改由前三级等比外推
                value, ncell, elapsed = solid_compliance(nu, nx, ny, method, order)
                row["cells"] = ncell
                row["compliance"][label] = value
                row["elapsed"][label] = elapsed
                print(f"[solid-h] nu={nu:<7} mesh={nx}x{ny} {label}: C={value:.6f} ({elapsed:.1f}s)", flush=True)
            rows.append(row)

    labels = [_label(m, o) for m, o in ANALYSES]
    references: dict[str, float] = {}
    for nu in NU_LEVELS:
        block = [r for r in rows if r["poisson_ratio"] == nu]
        # 参考值: k=4 前三级按逐级差等比递减外推的极限 (Aitken delta^2), 不假定收敛阶;
        # 用其余序列外推, 参考值变化不超过 0.02%, 即其不确定度
        c1, c2, c3 = (block[i]["compliance"][REFERENCE] for i in range(3))
        ratio = (c2 - c3) / (c1 - c2)
        reference = c3 - (c2 - c3) * ratio / (1.0 - ratio)
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
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n写入 {target}")
    if not stamp.get("reproducible"):
        print("注意: 工作区不干净, 本次数字不满足 provenance.reproducible, 不可直接引用.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
