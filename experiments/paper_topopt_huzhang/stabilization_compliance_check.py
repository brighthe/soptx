#!/usr/bin/env python
"""核验跳量稳定化缩放缺陷是否影响柔顺度算例 (5.2.1 / 5.2.2).

背景
----
`stabilization_probe.py` 已实测确认: `JumpPenaltyIntegrator` 的惩罚系数取自基材
剪切模量、不随密度插值, 使 HZMFEM k <= GD 在 SIMP 空区产生虚假应力 (rho < 0.01 带
上实体应力比虚高约 15 倍). 轴承算例的构型有 51% 单元落在 rho < 0.01, 前提条件同样
成立, 故虚假应力在 5.2.1 / 5.2.2 中也必然存在.

判据
----
柔顺度 `C = f^T u` 是全局泛函, 局部伪像应被真实载荷路径的权重压掉; 局部应力约束是
逐点的, 压不掉. 本脚本在**同一批冻结构型**上分别用 HZMFEM k=2 (受稳定化影响) 与
k=3 (p >= GD + 1, 走原生装配, 不受影响) 求解, 比较柔顺度:

- 两者相对差远小于表 5.3 中各方法间的差异 (可压组 < 4%, 近不可压组 27.1%)
  -> 柔顺度对该伪像不敏感, 5.2.1 / 5.2.2 的结论不受影响;
- 相对差与之同量级 -> 表 5.3 需要重做.

注意 k=3 只作分析列, 不需要 k=3 的设计.

Examples
--------
    python stabilization_compliance_check.py
    python stabilization_compliance_check.py --cases bearing-incompressible
"""

from __future__ import annotations

import argparse
from typing import Any

from bearing_reanalysis import (
    DISCRETIZATIONS,
    _label,
    build_pipeline,
    frozen_compliance,
    load_design,
)

# 受检臂与参照臂. k=2 走 matrix_jump 稳定化, k=3 走原生装配.
PROBE_ORDERS: tuple[tuple[str, int], ...] = (("huzhang", 2), ("huzhang", 3))

DEFAULT_CASES: tuple[str, ...] = ("bearing-compressible", "bearing-incompressible")


def check_case(case_id: str) -> None:
    """在一个算例的全部冻结构型上比较 k=2 与 k=3 的柔顺度."""
    print("\n" + "=" * 78)
    print(f"算例 {case_id}")
    print("=" * 78)

    pipelines: dict[str, Any] = {}
    for method, order in PROBE_ORDERS:
        pipelines[_label(method, order)] = build_pipeline(case_id, method, order)[0]

    probe_labels = [_label(m, o) for m, o in PROBE_ORDERS]
    reference = probe_labels[0]

    header = f"  {'冻结构型':<16}"
    for label in probe_labels:
        header += f"{label:>16}"
    header += f"{'相对差':>12}"
    print(header)

    for method, order in DISCRETIZATIONS:
        design_label = _label(method, order)
        try:
            rho, _summary = load_design(case_id, method, order)
        except SystemExit as exc:
            print(f"  {design_label:<16} 跳过: {exc}")
            continue
        values: dict[str, float] = {}
        for label in probe_labels:
            values[label], _ = frozen_compliance(pipelines[label], rho)
        line = f"  {design_label:<16}"
        for label in probe_labels:
            line += f"{values[label]:>16.6f}"
        base = values[reference]
        other = values[probe_labels[-1]]
        rel = (other - base) / abs(base) if base else float("nan")
        line += f"{rel:>11.3%}"
        print(line)

    print("\n  判据: 相对差 << 表 5.3 的方法间差异 (可压组 < 4%, 近不可压组 27.1%)")
    print("        -> 柔顺度对空区虚假应力不敏感, 5.2.1 / 5.2.2 结论不受影响.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cases", default=",".join(DEFAULT_CASES), metavar="<a,b>",
        help=f"待检算例, 逗号分隔; 默认 {','.join(DEFAULT_CASES)}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    for case_id in (c for c in arguments.cases.split(",") if c.strip()):
        check_case(case_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
