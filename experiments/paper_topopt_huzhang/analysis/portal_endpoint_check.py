# -*- coding: utf-8 -*-
"""门式框架全实体域的载荷端点应力有界性验证 (论文 5.2.4 节不设被动实体区的依据).

顶边载荷是法向区段牵引, 端点处应力只间断不奇异; 悬臂梁 (5.2.3 节) 的载荷是竖直边上的切向
区段牵引, 端点处有对数奇异, 故设被动实体区. 本模块在全实体域 (rho == 1) 上逐级加密网格,
比较各离散在载荷端点邻域内的单元形心实体 von Mises 应力比: 有界即随 h 收敛, 奇异即持续增长.
凹角顶点 (L/2, h_c) 与支座段内端 (b, 0) / (L - b, 0) 作奇异对照.

入口由 ``plot.py`` 派发, 本模块不直接执行::

    plot.py portal-endpoint-check -> run()

分析链由 ``discretization_probe.build_frozen(..., case='portal-frame-2d-stress', mesh_size=h)``
组装, 问题与网格取自 ``run_portal_frame.build_problem / build_mesh``, 与优化运行同一份代码.
产出 ``results/portal-frame-2d-stress/postprocess/endpoint_h_sweep.json``, 终端打印 Markdown 表.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from soptx.backend import backend_manager as bm

import run_portal_frame
from config import OUTPUT_DIR

from . import provenance
from .discretization_probe import build_frozen

CASE_ID = run_portal_frame.CASE_ID
# 网格尺寸须为 0.5/k 形式; 0.0625 即优化所用网格
MESH_LEVELS: tuple[float, ...] = (0.125, 0.0625, 0.03125)
# HZMFEM k=3 在 h = 0.03125 上约 360 万自由度, 不算
ANALYSES: tuple[tuple[str, int, tuple[float, ...]], ...] = (
    ("lfem", 1, MESH_LEVELS),
    ("lfem", 2, MESH_LEVELS),
    ("lfem", 3, MESH_LEVELS),
    ("huzhang", 2, MESH_LEVELS),
    ("huzhang", 3, MESH_LEVELS[:2]),
)
# 统计邻域半径 (mm), 取过滤半径
RADIUS = run_portal_frame.FILTER_RADIUS


def _label(method: str, order: int) -> str:
    return f"{'P' if method == 'lfem' else 'HZ'}{order}"


def solid_stress_ratio(method: str, order: int, h: float) -> tuple[np.ndarray, np.ndarray, int, float]:
    """全实体域上逐单元的形心实体 von Mises 应力比.

    Returns
    -------
    ratio : ndarray, shape (NC,)
        单元评价点上的实体应力比 (取单元内最大值).
    centers : ndarray, shape (NC, 2)
        单元形心.
    ndof : int
        自由度总数 (HZMFEM 为应力与位移之和).
    seconds : float
        求解与评价耗时.
    """
    pipeline = build_frozen(method, order, CASE_ID, mesh_size=h)
    analyzer, constraint, rho = pipeline.analyzer, pipeline.stress_constraint, pipeline.density
    rho[:] = 1.0
    ndof = int(analyzer.tensor_space.number_of_global_dofs())
    if method == "huzhang":
        ndof += int(analyzer.huzhang_space.number_of_global_dofs())
    started = time.perf_counter()
    state = analyzer.solve_state(rho_val=rho)
    constraint.fun(rho[:], state)  # 在 state 中写入 stiffness_ratio 等中间量
    ratio = np.asarray(bm.to_numpy(constraint.compute_solid_stress_ratio(rho[:], state)))
    ratio = ratio.reshape(ratio.shape[0], -1).max(axis=1)
    centers = np.asarray(bm.to_numpy(pipeline.mesh.entity_barycenter("cell")))
    return ratio, centers, ndof, time.perf_counter() - started


def neighborhood_max(ratio: np.ndarray, centers: np.ndarray, points, radius: float) -> float:
    """各点 ``radius`` 邻域 (按单元形心判定) 内的最大值."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    distance = np.min(np.linalg.norm(centers[:, None, :] - points[None, :, :], axis=-1), axis=1)
    return float(ratio[distance <= radius].max())


def run() -> int:
    stamp = provenance.run_stamp()
    problem = run_portal_frame.build_problem()
    endpoints = np.asarray(problem.traction_patch_endpoints, dtype=np.float64)
    apex = np.array([[problem.L / 2.0, problem.h_c]])
    support_ends = np.array([[problem.b, 0.0], [problem.L - problem.b, 0.0]])

    rows: list[dict[str, Any]] = []
    for method, order, levels in ANALYSES:
        for h in levels:
            ratio, centers, ndof, seconds = solid_stress_ratio(method, order, h)
            row = {
                "method": method, "order": order, "mesh_size": h,
                "cells": int(ratio.shape[0]), "ndof": ndof, "seconds": seconds,
                "load_endpoints_max": neighborhood_max(ratio, centers, endpoints, RADIUS),
                "apex_max": neighborhood_max(ratio, centers, apex, RADIUS),
                "support_ends_max": neighborhood_max(ratio, centers, support_ends, RADIUS),
            }
            rows.append(row)
            print(f"[endpoint] {_label(method, order)} h={h:<8g} ndof={ndof:>9,d}  "
                  f"载荷端点 {row['load_endpoints_max']:.4f}  凹角 {row['apex_max']:.4f}  "
                  f"支座内端 {row['support_ends_max']:.4f}  ({seconds:.0f}s)", flush=True)

    print(f"\n全实体域, 单元形心实体 von Mises 应力比在 r = {RADIUS:g} mm 邻域内的最大值 "
          f"(h = {' / '.join(f'{h:g}' for h in MESH_LEVELS)})")
    print("| 离散 | 载荷端点 | 凹角顶点 | 支座段内端 |")
    print("|---|---|---|---|")
    for method, order, levels in ANALYSES:
        block = [r for r in rows if (r["method"], r["order"]) == (method, order)]
        cell = lambda key: " / ".join(f"{r[key]:.3f}" for r in block)
        print(f"| {_label(method, order)} | {cell('load_endpoints_max')} | {cell('apex_max')} | "
              f"{cell('support_ends_max')} |")

    target = OUTPUT_DIR / CASE_ID / "postprocess" / "endpoint_h_sweep.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": CASE_ID,
        "design": "solid",
        "quantity": "单元形心实体 von Mises 应力比, 邻域内最大值",
        "radius": RADIUS,
        "points": {"load_endpoints": endpoints.tolist(), "apex": apex.tolist(),
                   "support_ends": support_ends.tolist()},
        "mesh_levels": list(MESH_LEVELS),
        "provenance": stamp,
        "rows": rows,
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n写入 {target}")
    if not stamp.get("reproducible"):
        print("注意: 工作区不干净, 本次数字不满足 provenance.reproducible, 提交后应重跑一次再引用.")
    return 0
