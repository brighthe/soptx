# -*- coding: utf-8 -*-
"""腹杆区探针: 同一份冻结构型下, 两条离散的应力评价与约束灵敏度逐区对照.

2026-09-16 的观察: 端点牵引修复后, Hu--Zhang k=2 的最终构型在 x in [41, 52] 的
上下两块长出一对灰腹杆 (平均 rho 0.20, 应力利用率 <= 0.22), 而 LFEM k=2 在同一
区域是纯空 (平均 rho 0.010); 两者体积差 0.0083 几乎全部由这两块贡献. 垫片已被
排除: LFEM 加垫片 (V 0.3519) 与不加垫片 (V 0.3476) 的拓扑逐杆一致.

剩下三种可能需要分开:

A. Hu--Zhang 在低密度区把实体应力比 ``s`` 算高了, 优化器留材料有 "理由";
B. ``s`` 一致但约束灵敏度 ``dg/drho`` 有偏, 出在导数而非函数值;
C. 两者评价都一致, 纯粹是 Hu--Zhang 的优化路径落进了另一个局部极小.

本模块把构型冻结, 对每条离散各做一次前向求解与一次伴随求解, 在指定的矩形探针区
内读四组量, 从而把 A 与 B 同 C 分开:

- 函数值: 实体应力比 ``s = sigma^sol_vM / sigma_bar`` 与约束值 ``g``;
- 可行性: 临界密度 ``rho_crit`` 与余量 ``eta = rho / rho_crit - 1`` (负值可行);
- 灵敏度: ``dG = d[(1/N) sum_j g_j] / drho``, 由 ``lamb`` 全置 1 后调用
  ``AugmentedLagrangianObjective.lagrangian_jac`` 再减去体积项得到. 该权重与
  离散无关, 两条路径可直接比较; 体积项 ``dV/drho`` 恒为 1/N, 作为标度参照.

判读: 若 Hu--Zhang 在腹杆区的 ``s`` 明显高于 LFEM, 则 A 成立; 若 ``s`` 一致而
``dG`` 显著更负 (加材料更能压低总约束), 则 B 成立; 两者都一致则归 C.

评价一律在不豁免 (``load_pad_radius = 0``) 的约束对象上做, 与
``stress_cross_evaluation`` 同口径; 构型由 ``rho[:] = design`` 整体覆写, 不经过
滤波链.

用法::

    python probe_web_members.py
    python probe_web_members.py --designs huzhang-2-pad,lfem-2-pad --methods huzhang-2,lfem-2
    python probe_web_members.py --box 自定义:41,52,24,32

产出写到 ``outputs/<case>/postprocess/probe_web_members.json``, 终端同时打印
Markdown 表.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fealpy.backend import backend_manager as bm

from config import OUTPUT_DIR

from stress_cross_evaluation import (
    CASE_ID,
    case_parameters,
    critical_density,
    evaluate,
    load_design,
)
import provenance


# 构型别名 -> outputs/<case>/ 下的运行目录名. driver 按字母序排标签, 因此垫片标签
# 落在 order 之前; stress_cross_evaluation.resolve_design_dir 的拼接顺序与此不符,
# 故这里直接给全名, 不走那条解析.
DESIGNS: dict[str, str] = {
    "huzhang-2-pad": "analyzer-huzhang__lfem_constraint-apparent__load_pad_radius-1.5__order-2",
    "lfem-2-pad": "analyzer-lfem__lfem_constraint-apparent__load_pad_radius-1.5__order-2",
    "lfem-2-nopad": "analyzer-lfem__lfem_constraint-apparent__order-2",
}
DEFAULT_DESIGNS = ("huzhang-2-pad", "lfem-2-pad")
DEFAULT_METHODS = ("huzhang-2", "lfem-2")

# 探针区 (x0, x1, y0, y1), 单元重心落在 [x0, x1) x [y0, y1) 内即计入.
# 前两块是争议中的腹杆; 后三块是对照, 用来确认差异是腹杆区独有还是全域偏移.
DEFAULT_BOXES: dict[str, tuple[float, float, float, float]] = {
    "上腹杆": (41.0, 52.0, 24.0, 32.0),
    "下腹杆": (41.0, 52.0, 8.0, 16.0),
    "中部灰带": (33.0, 45.0, 17.0, 23.0),
    "参照-右弦实体": (50.0, 64.0, 26.0, 34.0),
    "参照-中部空腔": (56.0, 72.0, 17.0, 23.0),
}


def _split(label: str) -> tuple[str, int]:
    method, _, order = label.rpartition("-")
    return method, int(order)


def parse_box(text: str) -> tuple[str, tuple[float, float, float, float]]:
    """解析 ``名称:x0,x1,y0,y1`` 形式的探针区."""
    name, _, numbers = text.partition(":")
    values = [float(v) for v in numbers.split(",")]
    if not name or len(values) != 4:
        raise SystemExit(f"探针区格式应为 名称:x0,x1,y0,y1, 收到 {text!r}")
    return name, (values[0], values[1], values[2], values[3])


def box_masks(
    barycenter: np.ndarray,
    boxes: dict[str, tuple[float, float, float, float]],
) -> dict[str, np.ndarray]:
    """把每个探针区化成单元布尔掩码."""
    masks = {}
    for name, (x0, x1, y0, y1) in boxes.items():
        masks[name] = (
            (barycenter[:, 0] >= x0) & (barycenter[:, 0] < x1)
            & (barycenter[:, 1] >= y0) & (barycenter[:, 1] < y1)
        )
    return masks


def constraint_sensitivity(pipeline: Any, state: dict, rho: Any) -> np.ndarray:
    """返回 ``d[(1/N) sum_j g_j] / drho`` 的逐单元值.

    ``lagrangian_jac`` 算的是 ``f_V + (1/N) sum_j lamb_j g_j`` 的梯度; 把 ``lamb``
    全置 1 并减去体积项, 剩下的就是与离散无关权重下的约束灵敏度. 该量为负表示
    在该单元加材料会压低总约束, 即优化器有保留材料的动机.

    Parameters
    ----------
    pipeline : Any
        已冻结构型的分析管线.
    state : dict
        与该构型一致的状态解.
    rho : Any
        物理密度 (已被构型覆写).

    Returns
    -------
    ndarray
        形状 ``(NC,)`` 的约束灵敏度.
    """
    objective = pipeline.al_objective
    objective.lamb = bm.ones_like(objective.lamb)
    total = objective.lagrangian_jac(density=rho, state=state)
    volume = pipeline.volume_objective.jac(density=rho, state=state)
    return np.asarray(bm.to_numpy(total - volume), dtype=np.float64).reshape(-1)


def volume_sensitivity(pipeline: Any, state: dict, rho: Any) -> np.ndarray:
    """返回体积目标的逐单元灵敏度 ``dV/drho``, 作为灵敏度量级的参照."""
    volume = pipeline.volume_objective.jac(density=rho, state=state)
    return np.asarray(bm.to_numpy(volume), dtype=np.float64).reshape(-1)


def probe_one(
    parameters: dict[str, Any],
    method: str,
    order: int,
    design: np.ndarray,
    boxes: dict[str, tuple[float, float, float, float]],
) -> dict[str, Any]:
    """在一份冻结构型上用一条离散求解, 返回各探针区的汇总量."""
    result = evaluate(parameters, method, order, design)
    pipeline = result["pipeline"]
    state = result["state"]
    rho = result["density"]

    stress_ratio = np.asarray(result["solid_stress_ratio"], dtype=np.float64).reshape(-1)
    constraint = np.asarray(result["constraint_value"], dtype=np.float64).reshape(-1)

    epsilon = float(parameters["epsilon"])
    penalty = float(parameters["penalty_factor"])
    void_ratio = float(parameters["void_youngs_modulus"]) / float(parameters["youngs_modulus"])
    rho_crit = critical_density(stress_ratio, epsilon, penalty, void_ratio)
    margin = np.where(rho_crit > 0.0, design / np.maximum(rho_crit, 1e-300) - 1.0, np.inf)

    d_constraint = constraint_sensitivity(pipeline, state, rho)
    d_volume = volume_sensitivity(pipeline, state, rho)

    barycenter = np.asarray(bm.to_numpy(pipeline.mesh.entity_barycenter("cell")), dtype=np.float64)
    masks = box_masks(barycenter, boxes)

    rows: dict[str, Any] = {}
    for name, mask in masks.items():
        if not mask.any():
            rows[name] = None
            continue
        rows[name] = {
            "cells": int(mask.sum()),
            "rho_mean": float(design[mask].mean()),
            "rho_max": float(design[mask].max()),
            "s_mean": float(stress_ratio[mask].mean()),
            "s_max": float(stress_ratio[mask].max()),
            "g_max": float(constraint[mask].max()),
            "rho_crit_min": float(rho_crit[mask].min()),
            "margin_max": float(margin[mask].max()),
            "dG_mean": float(d_constraint[mask].mean()),
            "dG_min": float(d_constraint[mask].min()),
        }
    return {
        "rows": rows,
        "dV_drho": float(np.median(d_volume)),
        "max_solid_stress_ratio": float(stress_ratio.max()),
        "seconds": result["seconds"],
    }


def print_markdown(payload: dict[str, Any]) -> None:
    """按构型分块打印函数值表与灵敏度表."""
    for design_label, block in payload["designs"].items():
        print(f"\n\n## 冻结构型: {design_label}  ({block['run_dir']})")
        print(f"V = {block['volume_fraction']:.4f}")

        print("\n(一) 函数值: 实体应力比 s 与约束 g")
        print("| 探针区 | 单元数 | 平均 rho | 最大 rho | 平均 s | 最大 s | 最大 g | 最大余量 eta |")
        print("|---|---|---|---|---|---|---|---|")
        for method_label, result in block["methods"].items():
            for name, row in result["rows"].items():
                if row is None:
                    continue
                print(
                    f"| {name} [{method_label}] | {row['cells']} | {row['rho_mean']:.3f} | "
                    f"{row['rho_max']:.3f} | {row['s_mean']:.3f} | {row['s_max']:.3f} | "
                    f"{row['g_max']:+.5f} | {row['margin_max']:+.1%} |"
                )

        print("\n(二) 约束灵敏度 dG = d[(1/N) sum g] / drho (负值 = 加材料压低总约束)")
        first = next(iter(block["methods"].values()))
        print(f"体积项参照 dV/drho = {first['dV_drho']:.3e}")
        print("| 探针区 | " + " | ".join(
            f"{label} 平均 dG | {label} 最负 dG" for label in block["methods"]
        ) + " |")
        print("|---" * (1 + 2 * len(block["methods"])) + "|")
        names = list(next(iter(block["methods"].values()))["rows"])
        for name in names:
            cells = []
            for result in block["methods"].values():
                row = result["rows"][name]
                cells.append("- | -" if row is None
                             else f"{row['dG_mean']:+.3e} | {row['dG_min']:+.3e}")
            print(f"| {name} | " + " | ".join(cells) + " |")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probe_web_members.py",
        description="腹杆区探针: 冻结构型下两条离散的应力评价与约束灵敏度对照.",
    )
    parser.add_argument(
        "--designs", default=",".join(DEFAULT_DESIGNS), metavar="<别名列表>",
        help=f"提供构型的运行别名, 逗号分隔. 可选: {', '.join(DESIGNS)}.",
    )
    parser.add_argument(
        "--methods", default=",".join(DEFAULT_METHODS), metavar="<method>-<order>",
        help="用于前向求解的离散组合, 逗号分隔.",
    )
    parser.add_argument(
        "--box", action="append", default=None, metavar="名称:x0,x1,y0,y1",
        help="自定义探针区, 可重复; 不给则用内置的五块.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    boxes = DEFAULT_BOXES if not arguments.box else dict(
        parse_box(text) for text in arguments.box
    )
    parameters = case_parameters()

    payload: dict[str, Any] = {"case_id": CASE_ID, "boxes": {
        name: list(values) for name, values in boxes.items()
    }, "designs": {}}

    for design_label in arguments.designs.split(","):
        design_label = design_label.strip()
        if design_label not in DESIGNS:
            raise SystemExit(f"未知构型别名 {design_label!r}; 可选 {', '.join(DESIGNS)}.")
        design_dir = OUTPUT_DIR / CASE_ID / DESIGNS[design_label]
        design, summary = load_design(design_dir)

        block: dict[str, Any] = {
            "run_dir": DESIGNS[design_label],
            "design_digest": provenance.file_digest(design_dir / "density_final.vtu"),
            "volume_fraction": float(summary.get("volume_fraction", float("nan"))),
            "methods": {},
        }
        for method_label in arguments.methods.split(","):
            method, order = _split(method_label.strip())
            print(f"[probe] {design_label} x {method_label} ...", flush=True)
            block["methods"][method_label.strip()] = probe_one(
                parameters, method, order, design, boxes
            )
        payload["designs"][design_label] = block

    payload["provenance"] = provenance.run_stamp()
    print_markdown(payload)

    target = OUTPUT_DIR / CASE_ID / "postprocess" / "probe_web_members.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[probe] {target.relative_to(OUTPUT_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
