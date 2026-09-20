#!/usr/bin/env python
"""跳量稳定化在高对比度密度场下的缩放核验探针.

背景
----
``soptx.fem.integrators.jump_penalty_integrator.JumpPenaltyIntegrator`` 的惩罚
系数取 ``alpha = mu / L0**2``, 其中 ``mu`` 来自基材 (``self.material`` 是
``interpolate_material`` 的输入, 未经密度插值), 因此 alpha 是全局常数、不含密度
依赖. 与之相对, 同一装配中的柔度块 ``A = _calculate_stress_matrix(rho_val=rho)``
量级为 ``O(1/E(rho))``. 在 SIMP 空区 ``E(rho) ~ 1e-9 * E0``, 按局部物理应取的惩罚
强度为 ``mu(rho)/L0**2``, 实际取的是 ``mu0/L0**2``, 相差约 ``1/void_youngs_modulus``
倍. 若该推断成立, 低阶稳定化格式 (二维即 ``k <= 2``) 会在空区产生虚假应力, 而
``k >= 3`` 走原生装配路径 (``p >= GD + 1``), 不受影响.

判据
----
本探针在**同一个冻结构型**上用不同离散各做一次前向求解, 按密度分带比较逐单元
``sigma^solid_vM / sigma_bar``. 关键在于偏差是否**随密度下降而放大**:

- 若 ``huzhang-2`` 相对 ``huzhang-3`` 的偏差集中在低密度带、实体带 (rho > 0.9) 基本
  一致, 支持"稳定化缩放未随密度插值"这一解释;
- 若各密度带偏差量级相当, 则只是阶次/精度差异, 不支持该解释;
- ``huzhang-2-nostab`` 消融臂关掉稳定化项: 若空区偏差随之消失, 即坐实惩罚项是来源.

注意: 消融臂通过写入 ``analyzer._stabilization`` 实现, 该字段为私有属性; 构造入参
未经 ``build_analyzer`` 暴露, 故只能如此注入. 关掉稳定化后低阶格式本身可能失稳,
求解失败属预期结果之一, 同样构成证据.

Examples
--------
默认臂 (4 次求解, 80x40 设计网格)::

    python stabilization_probe.py

指定冻结构型与臂::

    python stabilization_probe.py --design analyzer-lfem__lfem_constraint-apparent__load_pad_radius-1.5__order-2 \
        --arms lfem-2,huzhang-2,huzhang-3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from fealpy.backend import backend_manager as bm

from config import OUTPUT_DIR, bootstrap_source_path

bootstrap_source_path()

from pipeline import (  # noqa: E402
    build_stress_analysis_pipeline,
    build_stress_config,
)
from singularity_h_probe import (  # noqa: E402
    CASE_ID,
    _per_cell_max,
    case_parameters,
    load_frozen_design,
)

DEFAULT_DESIGN = (
    "analyzer-huzhang__lfem_constraint-apparent__load_pad_radius-1.5__order-3"
)

DEFAULT_ARMS: tuple[str, ...] = (
    "huzhang-3",          # 参照臂: p >= GD + 1, 原生格式, 不加稳定化
    "huzhang-2",          # 受检臂: matrix_jump 稳定化
    "huzhang-2-nostab",   # 消融臂: 同阶次关掉稳定化
    "lfem-2",             # 旁证臂: 独立方法
)

# 密度分带边界. 空区/近空区是判据的关键区间, 故在低端加密.
BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.01), (0.01, 0.05), (0.05, 0.1),
    (0.1, 0.3), (0.3, 0.5), (0.5, 0.9), (0.9, 1.0001),
)


def parse_arm(token: str) -> tuple[str, int, bool]:
    """解析臂标识.

    Parameters
    ----------
    token : str
        形如 ``huzhang-2``、``lfem-2`` 或 ``huzhang-2-nostab``.

    Returns
    -------
    tuple
        ``(method, order, disable_stabilization)``.
    """
    parts = token.strip().split("-")
    nostab = parts[-1] == "nostab"
    if nostab:
        parts = parts[:-1]
    if len(parts) != 2:
        raise SystemExit(f"无法解析臂 {token!r}, 期望 <method>-<order>[-nostab].")
    method, order = parts[0], int(parts[1])
    if method not in ("lfem", "huzhang"):
        raise SystemExit(f"未知离散 {method!r}.")
    if nostab and method != "huzhang":
        raise SystemExit("--arms: nostab 只对 huzhang 有意义.")
    return method, order, nostab


def solve_arm(
    parameters: dict[str, Any],
    nx: int,
    ny: int,
    method: str,
    order: int,
    disable_stabilization: bool,
    density: np.ndarray,
) -> dict[str, Any]:
    """在冻结密度场上做一次前向求解, 返回逐单元实体应力比.

    Parameters
    ----------
    density : ndarray
        逐单元物理密度, 长度须与网格单元数一致.
    disable_stabilization : bool
        置 True 时写入 ``analyzer._stabilization = 'none'``, 走原生装配路径.

    Returns
    -------
    dict
        含 ``solid_stress_ratio``、``density``、``barycenter``、``seconds``.
    """
    run_parameters = {
        **parameters,
        "nx": nx,
        "ny": ny,
        "comparison_orders": [order],
        # 本探针要看的正是垫片盖住的区域, 故一律关掉豁免与实体保留.
        "load_pad_radius": 0.0,
        "support_pad_radius": 0.0,
    }
    config = build_stress_config(run_parameters)
    pipeline = build_stress_analysis_pipeline(config, run_parameters, method, order)

    if disable_stabilization:
        analyzer = pipeline.analyzer
        if not hasattr(analyzer, "_stabilization"):
            raise SystemExit("当前 analyzer 无 _stabilization 字段, 消融臂不可用.")
        analyzer._stabilization = "none"

    rho = pipeline.density_distribution
    if rho.shape[0] != density.shape[0]:
        raise SystemExit(
            f"{method}-{order}: 网格单元数 {rho.shape[0]} 与密度 {density.shape[0]} 不符."
        )
    rho[:] = bm.asarray(density, dtype=rho.dtype)

    started = time.perf_counter()
    state = pipeline.analyzer.solve_state(rho_val=rho)
    constraint = pipeline.stress_constraint
    # 约束求值会就地填充 stress_solid / von_mises / stiffness_ratio,
    # compute_solid_stress_ratio 依赖这些键, 故须先走一遍.
    constraint.compute_unexempted_constraint(rho, state)
    ratio = constraint.compute_solid_stress_ratio(rho, state)
    return {
        "solid_stress_ratio": _per_cell_max(ratio),
        "density": np.asarray(bm.to_numpy(rho), dtype=np.float64).reshape(-1),
        "barycenter": np.asarray(
            bm.to_numpy(pipeline.mesh.entity_barycenter("cell")), dtype=np.float64
        ),
        "seconds": time.perf_counter() - started,
    }


def print_band_table(
    results: dict[str, dict[str, Any]], reference: str, density: np.ndarray
) -> None:
    """按密度分带打印各臂的实体应力比与相对参照臂的偏差."""
    ref = results[reference]["solid_stress_ratio"]
    arms = [k for k in results if k != reference]

    print("\n" + "=" * 86)
    print(f"按密度分带的实体应力比 sigma^solid_vM / sigma_bar   (参照臂: {reference})")
    print("  判据: 偏差是否随密度下降而放大. 放大 -> 支持稳定化缩放缺陷;")
    print("        各带量级相当 -> 只是阶次精度差异, 不支持.")
    print("=" * 86)
    header = f"  {'密度带':<14}{'单元数':>7}{'参照 max':>11}"
    for arm in arms:
        header += f"{arm + ' max':>18}"
    print(header)
    for low, high in BANDS:
        mask = (density >= low) & (density < high)
        n = int(mask.sum())
        if n == 0:
            continue
        line = f"  [{low:.2f},{high:.2f})".ljust(16) + f"{n:>7}" + f"{ref[mask].max():>11.4f}"
        for arm in arms:
            val = results[arm]["solid_stress_ratio"][mask].max()
            rel = (val - ref[mask].max()) / max(ref[mask].max(), 1e-30)
            line += f"{val:>11.4f} ({rel:+6.1%})"
        print(line)

    print("\n" + "-" * 86)
    print("各臂全域最大值及其所在单元")
    print("-" * 86)
    bary = results[reference]["barycenter"]
    for arm, entry in results.items():
        v = entry["solid_stress_ratio"]
        i = int(np.argmax(v))
        print(
            f"  {arm:<20} max = {v[i]:>10.4f}  @ ({bary[i, 0]:7.3f}, {bary[i, 1]:7.3f})"
            f"  rho = {density[i]:.4f}   {entry['seconds']:.1f}s"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--design", default=DEFAULT_DESIGN, metavar="<run-dir>",
        help=f"冻结构型的运行目录名; 默认 {DEFAULT_DESIGN}.",
    )
    parser.add_argument(
        "--arms", default=",".join(DEFAULT_ARMS), metavar="<a,b,...>",
        help=f"参与比较的臂, 逗号分隔; 默认 {','.join(DEFAULT_ARMS)}.",
    )
    parser.add_argument(
        "--reference", default="huzhang-3", metavar="<arm>",
        help="作为参照的臂; 默认 huzhang-3 (原生格式, 不加稳定化).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印将要执行的求解, 不实际计算.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    arms = [t for t in arguments.arms.split(",") if t.strip()]
    if arguments.reference not in arms:
        raise SystemExit(f"参照臂 {arguments.reference!r} 不在 --arms 中.")

    design_dir = OUTPUT_DIR / CASE_ID / arguments.design
    design, meta = load_frozen_design(design_dir)
    parameters = case_parameters()
    nx, ny = int(parameters["nx"]), int(parameters["ny"])

    print(f"[stab] 冻结构型 = {meta['run_dir']}")
    print(f"[stab]   converged={meta['converged']}  iterations={meta['iterations']}  "
          f"V={meta['volume_fraction']}")
    print(f"[stab] 设计网格 {nx}x{ny}, 单元数 {design.size}, 臂 = {', '.join(arms)}")
    if arguments.dry_run:
        for token in arms:
            method, order, nostab = parse_arm(token)
            print(f"  会求解: {method} order={order} stabilization="
                  f"{'none' if nostab else '(默认)'}")
        return 0

    results: dict[str, dict[str, Any]] = {}
    for token in arms:
        method, order, nostab = parse_arm(token)
        print(f"[stab] 求解 {token} ...", flush=True)
        try:
            results[token] = solve_arm(
                parameters, nx, ny, method, order, nostab, design
            )
        except Exception as exc:  # noqa: BLE001
            # 消融臂失稳导致求解失败本身即是证据, 不应中断整个探针.
            print(f"[stab]   {token} 求解失败: {type(exc).__name__}: {exc}")
            continue
        print(f"[stab]   完成, {results[token]['seconds']:.1f}s")

    if arguments.reference not in results:
        raise SystemExit("参照臂求解失败, 无法比较.")
    print_band_table(results, arguments.reference, design)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
