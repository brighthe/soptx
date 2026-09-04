# -*- coding: utf-8 -*-
"""应力算例的后验校核与指标复算: 梯度校验 / 冻结重分析 / 插图场数据导出.

三段共用同一条链路 —— 都按 ``cases.toml`` 的权威参数经 ``pipeline`` 的悬臂梁
装配器重建分析管线, 故合并为一个模块 (原 ``check_gradients.py`` / ``frozen_metrics.py``
/ ``export_fig_data.py``, 2026-09-01 并入; 命名沿用
``experiments/elasticity_paradigm_comparison/metrics.py``)。

入口函数由 ``run.py`` 派发, 本模块不直接执行::

    run.py gradients   -> run_gradient_check()    有限差分校验伴随灵敏度
    run.py metrics     -> run_frozen_metrics()    冻结设计的论文口径指标
    run.py export      -> run_export(argv)        导出插图 npz (支持 --run/--check)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import numpy as np

from soptx.postprocess.vtk_export import read_vtu_cell_data

from config import (
    CASES_FILE,
    OUTPUT_DIR,
    bootstrap_source_path,
    flatten_parameters,
    load_cases,
)

bootstrap_source_path()

from pipeline import (  # noqa: E402
    build_stress_analysis_pipeline as build_analysis_pipeline,
    build_stress_config as build_config,
)
from soptx.postprocess.stress_report import StressPostProcessor  # noqa: E402


# ============================================ 一、伴随灵敏度的有限差分校验
# 验证迁移后的灵敏度链路 (伴随载荷符号 / Voigt 重排 / 隐式项重写) 对
# HuZhang (ApparentStressConstraint) 与 LFEM (VanishingStressConstraint) 两条路径.

def make_params(nx: int, ny: int) -> dict:
    return {
        "nx": nx,
        "ny": ny,
        "load": -400.0,
        "load_width": 6.0,
        "load_discretization": "patch",
        "youngs_modulus": 1.0,
        "poisson_ratio": 0.25,
        "plane_type": "plane_stress",
        "stress_limit": 180.0,
        "epsilon": 1.0e-4,
        "comparison_orders": [2],
        "optimizer": "al_mma",
        "filter_type": "projection",
        "filter_radius": 6.0,
        "interpolation_method": "msimp",
        "penalty_factor": 3.5,
        "void_youngs_modulus": 1.0e-9,
        "initial_density": 0.5,
        "max_al_iterations": 150,
        "mma_iters_per_al": 5,
        "change_tolerance": 2.0e-3,
        "stress_tolerance": 3.0e-3,
        "mu_0": 50.0,
        "mu_max": 10000.0,
        "alpha": 1.1,
        "lambda_0_init_val": 0.0,
        "move_limit": 0.15,
        "use_relaxation": True,
        "solve_method": "scipy",
    }


def check_method(method: str, order: int = 2, nx: int = 8, ny: int = 4) -> bool:
    print(f"\n===== method={method}, order={order}, mesh={nx}x{ny} =====")
    params = make_params(nx, ny)
    config = build_config(params)
    pipe = build_analysis_pipeline(config, params, method, order)

    rho = pipe.density_distribution
    NC = rho.shape[0]

    # 非均匀密度场 (远离 mask/投影死区: [0.3, 0.9])
    rng = np.random.default_rng(0)
    rho[:] = rng.uniform(0.3, 0.9, size=NC)

    al = pipe.al_objective
    # 设 lambda=1.0, mu=50 -> 阈值 -lambda/mu = -0.02, 让绝大多数单元约束激活
    al.lamb[:] = 1.0
    al.mu = 50.0

    def eval_state():
        # 与 ALMMMAOptimizer 一致: 先正向求解填充 state, 再喂给 AL 目标
        return dict(pipe.analyzer.solve_state(rho_val=rho))

    # 基准点: fun -> jac (jac 依赖 fun 缓存的 g/h 与 state)
    state = eval_state()
    J0 = al.fun(rho, state)
    grad = np.asarray(al.jac(rho, state))
    g_base = np.asarray(al._cache_g).reshape(NC)
    thresh = -1.0 / 50.0
    n_active = int(np.sum(g_base > thresh))
    print(f"J0 = {J0:.6e},  active constraints: {n_active}/{NC},  "
          f"g range [{g_base.min():.3e}, {g_base.max():.3e}]")

    # 抽取待检单元: 随机 6 个 + |grad| 最大的 2 个
    idx = list(rng.choice(NC, size=min(6, NC), replace=False))
    idx += list(np.argsort(-np.abs(grad))[:2])
    idx = sorted(set(int(i) for i in idx))

    print(f"{'elem':>5} {'g_e':>11} {'adjoint':>14} {'FD(1e-5)':>14} "
          f"{'FD(1e-6)':>14} {'relerr':>10}")
    ok = True
    for i in idx:
        # 靠近激活集 kink 的单元中心差分会跨越不可导点, 标记跳过
        if abs(g_base[i] - thresh) < 1e-4:
            print(f"{i:>5} {g_base[i]:>11.3e}  -- 距激活集 kink 过近, 跳过 --")
            continue
        fds = []
        for eps in (1e-5, 1e-6):
            orig = float(rho[i])
            rho[i] = orig + eps
            Jp = al.fun(rho, eval_state())
            rho[i] = orig - eps
            Jm = al.fun(rho, eval_state())
            rho[i] = orig
            fds.append((Jp - Jm) / (2.0 * eps))
        fd = fds[0]
        denom = max(abs(fd), abs(grad[i]), 1e-14)
        relerr = abs(grad[i] - fd) / denom
        flag = "" if relerr < 1e-4 else ("  <-- MISMATCH" if relerr > 1e-2 else "  (borderline)")
        if relerr > 1e-2:
            ok = False
        print(f"{i:>5} {g_base[i]:>11.3e} {grad[i]:>14.6e} {fds[0]:>14.6e} "
              f"{fds[1]:>14.6e} {relerr:>10.2e}{flag}")

    # 恢复基准缓存, 避免影响后续 (无后续, 仅卫生)
    al.fun(rho, eval_state())
    print(f"[{method}] gradient check {'PASSED' if ok else 'FAILED'}")
    return ok


def run_gradient_check() -> int:
    results = {}
    for method in ("huzhang", "lfem"):
        try:
            results[method] = check_method(method)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[method] = False
            print(f"[{method}] gradient check ERROR: {exc}")
    print("\n===== summary =====")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


# ============================================ 二、冻结设计的论文口径指标
# 对 outputs/cantilever-middle-2d-stress 下两个 density_final.vtu:
# 1. 用与优化相同的方法/阶次 (k=2) 冻结求解, 按博士论文口径报实体单元 (rho>0.5)
#    最大/平均归一化应力、实体单元数、全域最大及其位置;
# 2. 用 HZ k=3 / k=4 冻结求解做独立高阶重分析.

# 本段固定按注册默认网格 80x40 取产物 (frozen_eval 也写死 make_params(80, 40)).
OUT = OUTPUT_DIR / "cantilever-middle-2d-stress"
SIGMA_LIM = 180.0


def read_vtu_cell_density(path: Path | str) -> np.ndarray:
    """解析 VTKFile appended-raw 格式的 CellData density 数组."""
    return read_vtu_cell_data(path, "density")


def frozen_eval(method: str, order: int, rho_np: np.ndarray, label: str):
    params = make_params(80, 40)
    params["comparison_orders"] = [order]
    config = build_config(params)
    pipe = build_analysis_pipeline(config, params, method, order)
    rho = pipe.density_distribution
    assert rho.shape[0] == rho_np.shape[0], (rho.shape, rho_np.shape)
    rho[:] = rho_np

    state = dict(pipe.analyzer.solve_state(rho_val=rho))
    pipe.al_objective.fun(rho, state)
    SM = np.asarray(pipe.stress_constraint.compute_stress_measure(rho=rho, state=state))
    SM = SM.reshape(rho_np.shape[0], -1).max(axis=1)  # (NC,) NQ=1

    solid = rho_np > 0.5
    imax = int(np.argmax(SM))
    bc = np.asarray(pipe.mesh.entity_barycenter("cell"))
    print(f"[{label}] 全域 max = {SM.max():.4f} @elem {imax} "
          f"(rho={rho_np[imax]:.3f}, xy=({bc[imax][0]:.1f},{bc[imax][1]:.1f}))")
    print(f"[{label}] 实体单元数 = {int(solid.sum())}, "
          f"实体 max = {SM[solid].max():.4f}, 实体 mean = {SM[solid].mean():.4f}, "
          f"volfrac = {rho_np.mean():.4f}")

    # 高阶重分析时额外报更高积分阶的逐点最大 (仅 HZ)
    if method == "huzhang" and order >= 3:
        sq = pipe.analyzer.extract_stress_at_quadrature_points(
            stress_dof=state["stress"], integration_order=4)
        vm = np.asarray(pipe.analyzer.material.calculate_von_mises_stress(sq)) / SIGMA_LIM
        vm_e = vm.reshape(rho_np.shape[0], -1).max(axis=1)
        print(f"[{label}] (积分阶4) 全域 max = {vm_e.max():.4f}, "
              f"实体 max = {vm_e[solid].max():.4f}")
    return SM


def run_frozen_metrics() -> int:
    designs = {
        "lfem": read_vtu_cell_density(OUT / "analyzer-lfem__order-2" / "density_final.vtu"),
        "huzhang": read_vtu_cell_density(OUT / "analyzer-huzhang__order-2" / "density_final.vtu"),
    }

    print("========== 一、论文口径 (各自方法 k=2 冻结求解) ==========")
    print("博士论文参考: LFEM max=1.0008, 实体 2266, 实体 mean=0.6067, V*=0.3499")
    print("             HZ   max=0.9978, 实体 2549, 实体 mean=0.5509, V*=0.3877")
    frozen_eval("lfem", 2, designs["lfem"], "LFEM设计/LFEM-k2")
    frozen_eval("huzhang", 2, designs["huzhang"], "HZ设计/HZ-k2")

    print("\n========== 二、独立高阶重分析 (HZ k=3 / k=4) ==========")
    for design_name, rho_np in designs.items():
        for k in (3, 4):
            frozen_eval("huzhang", k, rho_np, f"{design_name}设计/HZ-k{k}")
    return 0


# ============================================ 三、插图场数据导出 (npz)
# 论文 5.2.3 节的三张插图不直接读优化历程, 而是读
# outputs/cantilever-middle-2d-stress/postprocess/fig_data_<run>.npz; 本段是其唯一来源.
# npz 属 outputs/ 下的中间产物, 不入版本控制; 数字的溯源依据是各 run 目录下
# summary.json 自带的运行戳记 (provenance.run_stamp).

CASE_ID = "cantilever-middle-2d-stress"
# 插图涉及的四次运行: LFEM k=2 基线 + Hu--Zhang k=2/3/4
RUNS: dict[str, tuple[str, int]] = {
    "lfem-k2": ("lfem", 2),
    "huzhang-k2": ("huzhang", 2),
    "huzhang-k3": ("huzhang", 3),
    "huzhang-k4": ("huzhang", 4),
}


def case_parameters(case_id: str = CASE_ID) -> dict[str, Any]:
    """从 cases.toml 取该算例的扁平参数, 保证与优化时同口径."""
    for case in load_cases(CASES_FILE):
        if case["id"] == case_id:
            return flatten_parameters(case)
    raise SystemExit(f"cases.toml 中没有算例 {case_id}.")


def export_run(name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """对单次运行做冻结重分析, 返回 npz 待写入的场量字典."""
    method, order = RUNS[name]
    # name 只用于 npz 文件名; run 目录第二层按 driver._run_label 的参数标签拼。
    run_dir = OUTPUT_DIR / CASE_ID / f"analyzer-{method}__order-{order}"
    density_file = run_dir / "density_final.vtu"
    if not density_file.is_file():
        raise FileNotFoundError(f"最终构型缺失: {density_file}. 请先运行该算例.")

    parameters = {**parameters, "comparison_orders": [order]}
    pipeline = build_analysis_pipeline(build_config(parameters), parameters, method, order)

    rho_final = read_vtu_cell_density(density_file)
    rho = pipeline.density_distribution
    if rho.shape[0] != rho_final.shape[0]:
        raise ValueError(f"{name}: 网格单元数 {rho.shape[0]} 与构型 {rho_final.shape[0]} 不符.")
    rho[:] = rho_final

    processor = StressPostProcessor(
        analyzer=pipeline.analyzer, stress_limit=float(parameters["stress_limit"])
    )
    results = processor.check_stress_constraints(rho)
    mesh = pipeline.mesh
    return {
        "node": np.asarray(mesh.entity("node"), dtype=np.float64),
        "cell": np.asarray(mesh.entity("cell"), dtype=np.int32),
        "rho": rho_final,
        "vm": np.asarray(results.SM, dtype=np.float64),
        "sig1": np.asarray(results.sig_1_norm, dtype=np.float64),
        "sig2": np.asarray(results.sig_2_norm, dtype=np.float64),
        "solid_mask": np.asarray(results.solid_mask, dtype=bool),
        "vol": np.float64(results.volume_fraction),
    }


def compare(fields: dict[str, Any], path: Path) -> list[str]:
    """与既有 npz 逐键比对, 返回不一致的键名列表."""
    if not path.is_file():
        return ["<文件不存在>"]
    with np.load(path) as reference:
        missing = set(fields) ^ set(reference.files)
        if missing:
            return [f"<键集合不一致: {sorted(missing)}>"]
        return [
            key
            for key, value in fields.items()
            if not np.allclose(reference[key], value, rtol=1e-10, atol=1e-12)
        ]


def run_export(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出/校验应力算例插图数据.")
    parser.add_argument("--run", choices=sorted(RUNS), action="append",
                        help="只处理指定运行, 可重复; 缺省处理全部.")
    parser.add_argument("--check", action="store_true",
                        help="只与现有 npz 比对, 不写盘.")
    arguments = parser.parse_args(argv)

    parameters = case_parameters()
    target_dir = OUTPUT_DIR / CASE_ID / "postprocess"
    target_dir.mkdir(parents=True, exist_ok=True)

    mismatched = 0
    for name in arguments.run or sorted(RUNS):
        fields = export_run(name, parameters)
        path = target_dir / f"fig_data_{name}.npz"
        if arguments.check:
            differences = compare(fields, path)
            mismatched += bool(differences)
            verdict = "一致" if not differences else f"不一致: {', '.join(differences)}"
            print(f"[check] {path.name}: {verdict}")
        else:
            np.savez(path, **fields)
            print(f"[export] {path} (max SM = {fields['vm'].max():.4f}, "
                  f"V = {float(fields['vol']):.4f})")
    return 1 if mismatched else 0
