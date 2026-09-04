# -*- coding: utf-8 -*-
"""产物收集与数据快照组装 (TopOpt 平台能力验证).

从真实运行产物 ``outputs/<figure.base>/`` 组装申请书图 10 的数据快照 ``fig4_data.json``：
  - ``summary.json``           GPU 191.5 万自由度全收敛运行摘要
  - ``gpu_single_step.json``   均匀密度单步计时（GPU 张量化）
  - ``cpu_sparse_single_step.json``  scipy 稀疏单步计时（CPU 基线）
  - ``cpu_vs_gpu_compare.json`` 同一设计密度下 CPU/GPU 解对比
  - ``history.json`` / ``density_final.npy``  完整优化历程与最终构型

本模块**不含任何性能数字硬编码**；快照里所有耗时、加速比、迭代数与解误差均来自上述
真实产物。若产物缺失则直接报错，不静默回退。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config
import provenance


def _load(base: Path, name: str) -> dict[str, Any]:
    path = base / name
    if not path.is_file():
        raise FileNotFoundError(
            f"真实运行产物缺失: {path}。请先运行 "
            f"examples/topopt_platform/topopt_3d_simp_real.py 及其 CPU 基线/对比。")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_consistency(gpu: dict[str, Any], gpu_single: dict[str, Any],
                       cpu: dict[str, Any], cmp: dict[str, Any]) -> None:
    """核对四次运行的网格、设计域与求解容差是否同口径.

    单步加速比只有在两侧同网格、同容差、同精度时才有意义; 口径一旦分叉, 面板 (a)
    与 (b) 的数字会互相矛盾, 故在组装快照前直接报错而非静默出图.

    异常:
        ValueError: 任一口径不一致.
    """
    grid = gpu["grid"]
    for name, d in (("gpu_single", gpu_single), ("cpu", cpu), ("cmp", cmp)):
        if list(d["grid"]) != list(grid):
            raise ValueError(f"网格口径不一致: summary={grid}, {name}={d['grid']}")
    L = gpu["L"]
    for name, d in (("gpu_single", gpu_single), ("cpu", cpu), ("cmp", cmp)):
        if "L" in d and [float(x) for x in d["L"]] != [float(x) for x in L]:
            raise ValueError(f"设计域口径不一致: summary={L}, {name}={d['L']}")
    rtol = gpu["cg_rtol"]
    for name, d in (("gpu_single", gpu_single), ("cpu", cpu), ("cmp", cmp)):
        if "cg_rtol" in d and float(d["cg_rtol"]) != float(rtol):
            raise ValueError(f"求解容差不一致: summary={rtol}, {name}={d['cg_rtol']}")
    # CPU 侧 scipy 恒为 float64; GPU 侧若退回 float32 则单步对比不对等.
    for name, key, d in (("gpu_single", "dtype", gpu_single), ("cmp", "dtype_gpu", cmp)):
        if d.get(key) != "float64":
            raise ValueError(f"GPU 精度非 float64, 与 CPU 基线不对等: {name}={d.get(key)}")


def build_panels(base: Path) -> dict[str, Any]:
    gpu = _load(base, "summary.json")
    gpu_single = _load(base, "gpu_single_step.json")
    cpu = _load(base, "cpu_sparse_single_step.json")
    cmp = _load(base, "cpu_vs_gpu_compare.json")
    _check_consistency(gpu, gpu_single, cpu, cmp)
    history = json.loads((base / "history.json").read_text(encoding="utf-8"))

    gpu_times = gpu_single["times_mean"]
    cpu_times = [cpu["t_asm"], cpu["t_solve"], cpu["t_sens"], cpu["t_oc"]]
    gpu_times_l = [gpu_times["asm"], gpu_times["solve"], gpu_times["sens"],
                   gpu_times["oc"]]
    speedups = [c / g for c, g in zip(cpu_times, gpu_times_l)]
    total_cpu = cpu["t_total"]
    total_gpu = gpu_single["times_mean_total"]

    return {
        "n_dofs": gpu["n_dofs"],
        "n_cells": gpu["n_cells"],
        "grid": gpu["grid"],
        "domain": gpu["L"],
        "cell_size": gpu["h"],
        "volfrac": gpu["volfrac"],
        "penal": gpu["penal"],
        "rmin_phys": gpu["rmin_phys"],
        "filter": gpu["filter"],
        # ke0 已含完整等参映射, 故柔顺度即物理值; 保留标度因子供旧产物换算.
        "ke0_scale": gpu["ke0_scale"],
        "dtype_gpu": gpu["dtype"],
        "dtype_cpu": cpu["dtype"],
        "cg_rtol": gpu["cg_rtol"],
        "iterations": gpu["iterations"],
        "converged": gpu["converged"],
        # 收敛判据本身要进图 (面板 (d) 的阈值线), 不能让下游硬编码 0.01.
        "ctol": gpu["ctol"],
        "ctol_metric": gpu["ctol_metric"],
        "ctol_patience": gpu["ctol_patience"],
        "initial_compliance": gpu_single["final_compliance"],
        "final_compliance": gpu["final_compliance"],
        "compliance_reduction": gpu_single["final_compliance"] / gpu["final_compliance"],
        "final_volume_fraction": gpu["final_volume_fraction"],
        "time_per_iter_gpu": gpu["times_mean_total"],
        "time_total_gpu": gpu["times_mean_total"] * gpu["iterations"],
        "cg_iters_mean_converged": gpu["cg_iters_mean"],
        "peak_gpu_mb": gpu.get("peak_gpu_mb"),
        "history_iter": [h["iter"] for h in history],
        "history_compliance": [h["compliance"] for h in history],
        "history_change": [h["change"] for h in history],
        "history_volume_fraction": [h["volume_fraction"] for h in history],
        "labels_a": ["CPU 传统流程\n(NumPy / SciPy 稀疏)", "GPU 张量化平台\n(SOPTX PyTorch)"],
        "times_a": [round(total_cpu, 2), round(total_gpu, 2)],
        "speedup_a": round(total_cpu / total_gpu, 1),
        "stages_b": ["1. 刚度组装", "2. 平衡方程求解", "3. 伴随灵敏度滤波", "4. 设计变量更新"],
        "speedups_b": [round(s, 1) for s in speedups],
        "time_details_b": [
            f"{cpu['t_asm']:.2f}→{gpu_times_l[0]:.2f} s",
            f"{cpu['t_solve']:.2f}→{gpu_times_l[1]:.2f} s",
            f"{cpu['t_sens']:.2f}→{gpu_times_l[2]:.2f} s",
            f"{cpu['t_oc']:.2f}→{gpu_times_l[3]:.2f} s",
        ],
        "cg_iters_cpu": cpu["cg_iters"],
        "cg_iters_gpu": int(gpu_single["cg_iters_mean"]),
        "solution_rel_diff": cmp["solution_rel_diff"],
        "rel_res_cpu": cmp["rel_res_cpu"],
        "rel_res_gpu": cmp["rel_res_gpu"],
    }


def collect() -> dict[str, Any]:
    figure, cases = config.load()
    base = config.OUTPUT_DIR / figure["base"]
    panels = build_panels(base)

    prov = provenance.collect()
    repro = provenance.reproducible(prov)

    snapshot = {
        "figure": figure,
        "provenance": prov,
        "reproducible": repro,
        "panels": panels,
    }
    return snapshot


def write(snapshot: dict[str, Any], path: Path | None = None) -> Path:
    target = path or (config.FIGURE_DATA_DIR / "fig4_data.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


if __name__ == "__main__":
    out = write(collect())
    print(f"[collect] Fig4 snapshot written to {out}")
