"""验证角点线性迹投影及降阶系统的一致性."""

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

from examples.substructure_elasticity._common import DEFAULT_DENSITY

PROBLEMS = {"HalfMBBBeamRight2d": 2, "FullMBBBeam3d": 3}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证角点线性迹投影及降阶系统的内部一致性.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--problem", choices=tuple(PROBLEMS), default="HalfMBBBeamRight2d",
        help="MBB 梁 Problem; 同时确定计算维度. 默认 HalfMBBBeamRight2d.",
    )
    parser.add_argument(
        "--n-sub", type=int, nargs="+", metavar="N",
        help="各方向子结构数; 省略时 2D 为 6 2, 3D 为 6 2 2.",
    )
    parser.add_argument(
        "--n-fine", type=int, nargs="+", metavar="N",
        help="每个子结构各方向的有限元单元数; 省略时 2D 为 5 5, 3D 为 4 4 4.",
    )
    parser.add_argument(
        "--density", choices=("cell", "uniform"), default=DEFAULT_DENSITY,
        help="单元密度场类型. 默认 cell.",
    )
    parser.add_argument(
        "--output-dir", default=str(_SCRIPT_DIR / "outputs"),
        help="JSON 结果目录. 默认脚本同级 outputs/.",
    )
    return parser


def load_runner() -> Any:
    return importlib.import_module(
        "examples.substructure_elasticity._comparison"
    ).run_linear_corner_consistency


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_runner()(
        PROBLEMS[args.problem], args.output_dir,
        n_sub=args.n_sub, n_fine=args.n_fine, density_mode=args.density,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
