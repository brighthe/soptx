"""验证完整接口静力缩聚的有限元误差收敛阶."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from examples.substructure_elasticity._common import CONVERGENCE_SOLVERS, DEFAULT_LEVELS

PROBLEMS = {"HarmonicPoly2D": 2, "HarmonicPoly3D": 3}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证完整接口静力缩聚的位移误差收敛阶.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--problem", choices=tuple(PROBLEMS), default="HarmonicPoly2D",
        help="制造解 Problem; 同时确定计算维度. 默认 HarmonicPoly2D.",
    )
    parser.add_argument(
        "--degree", type=int, choices=(1, 2), default=1,
        help="张量积 Lagrange 有限元次数 p. 默认 1.",
    )
    parser.add_argument(
        "--levels", type=int,
        help="嵌套加密层数; 省略时 2D 为 4, 3D 为 3.",
    )
    parser.add_argument(
        "--solve-method", choices=CONVERGENCE_SOLVERS, default="scipy",
        help="线性系统直接求解方法. 默认 scipy.",
    )
    parser.add_argument(
        "--output-dir", default=str(_SCRIPT_DIR / "outputs"),
        help="JSON 结果目录. 默认脚本同级 outputs/.",
    )
    return parser


def load_runner() -> Any:
    return importlib.import_module(
        "examples.substructure_elasticity._convergence"
    ).run_convergence_benchmark


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dim = PROBLEMS[args.problem]
    levels = args.levels if args.levels is not None else DEFAULT_LEVELS[dim]
    if levels < 2:
        build_parser().error("--levels 必须 >= 2.")
    load_runner()(
        dim=dim, model="harmonic-poly", degree=args.degree, levels=levels,
        output_dir=args.output_dir, solve_method=args.solve_method,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
