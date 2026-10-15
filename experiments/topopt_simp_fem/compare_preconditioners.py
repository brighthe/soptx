"""固定物理密度场下的 CG 预条件子对照框架.

对照 EA/FA 与 None/Jacobi/GMG, 不执行 OC 更新或密度过滤.
模型、材料、SIMP 和离散设置后续复用 run_fem_ea_vs_fa.py 的公共配置,
数值操作复用 LagrangeFEMAnalyzer, 计量复用 soptx.core.measure.

当前仅实现参数解析、任务清单与结果字段约定. --dry-run 打印清单,
不导入数值库、不读取 NPY、不创建输出目录; 实际求解入口尚未实现.

后续执行约定
------------
1. 每个任务在独立子进程执行, 避免 EA/FA 缓存与累计 RSS 相互污染.
2. 输入为单元物理密度, 不再过滤; 检查网格、单元顺序、形状、有限性与 [0, 1] 范围.
3. 所有配置使用相同载荷、初值与真实残差阈值, 每个任务复制初始位移.
4. 初始均匀密度与中期、末期密度分别调用脚本, 使用不同输出目录.
5. 计入预条件构造与应用成本; 不把完整 solve_system 耗时误记为纯 CG 耗时.
6. 未收敛和运行异常分别记录, 不据此生成有效的解一致性结论.
7. 输出目录存在时拒绝覆盖, 保存输入指纹、线程环境、配置及任务结果.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ComparisonCase:
    """描述一个独立执行的对照任务.

    Parameters
    ----------
    operator_level : str
        刚度算子层级, 取 ea 或 fa.
    preconditioner : str
        预条件子名称, none 在数值接口中映射为 None.
    """

    operator_level: Literal["ea", "fa"]
    preconditioner: Literal["none", "jacobi", "gmg"]


@dataclass
class CaseResult:
    """约定单任务结果字段, 未测得的量保持 None.

    Parameters
    ----------
    case : ComparisonCase
        对照任务.
    status : str
        结果状态, 区分收敛、预算内未收敛与运行异常.

    Notes
    -----
    total_solve_seconds 包含预条件构造、CG 与求解诊断.
    分项时间只有独立计量后才填写, 不从总时间猜测.
    场一致性应比较收敛解, 不以向量范数相同代替逐场差异检查.
    """

    case: ComparisonCase
    status: Literal["not_run", "converged", "not_converged", "error"] = "not_run"
    cg_iterations: int | None = None
    true_residual_norm: float | None = None
    relative_true_residual: float | None = None
    residual_threshold: float | None = None
    compliance: float | None = None
    setup_seconds: float | None = None
    assembly_seconds: float | None = None
    boundary_seconds: float | None = None
    preconditioner_setup_seconds: float | None = None
    krylov_seconds: float | None = None
    total_solve_seconds: float | None = None
    process_peak_rss_gib: float | None = None
    displacement_file: str | None = None
    error: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析任务配置, 不读取数值场或创建目录.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数, 默认读取进程参数.

    Returns
    -------
    argparse.Namespace
        经过基本有效性检查的配置.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, nargs=3, default=(390, 65, 65),
                        metavar=("NX", "NY", "NZ"))
    density = parser.add_mutually_exclusive_group()
    density.add_argument("--density-file", type=Path, help="单元物理密度 NPY 文件")
    density.add_argument("--uniform-density", type=float, default=None,
                         help="均匀物理密度; 未指定密度输入时沿用算例 VOLFRAC")
    parser.add_argument("--initial-displacement", type=Path,
                        help="相同网格和自由度顺序下的初始位移 NPY; 默认零初值")
    parser.add_argument("--operator-levels", nargs="+", choices=("ea", "fa"),
                        default=("ea", "fa"))
    parser.add_argument("--preconditioners", nargs="+", choices=("none", "jacobi", "gmg"),
                        default=("none", "jacobi", "gmg"))
    parser.add_argument("--cg-tol", type=float, default=1.0e-6,
                        help="真实残差阈值为 cg_tol * ||F||_2, 与优化脚本一致")
    parser.add_argument("--cg-maxiter", type=int, default=20000)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "preconditioners",
                        help="后续执行的独立结果目录, 框架阶段不写入")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印任务清单, 不加载场、不求解、不落盘")
    args = parser.parse_args(argv)
    nx, ny, nz = args.grid
    if min(args.grid) < 1 or nx != 6 * ny or ny != nz:
        parser.error("grid 须为正整数且满足 NX = 6 NY, NY = NZ")
    if not math.isfinite(args.cg_tol) or args.cg_tol <= 0 or args.cg_maxiter < 1:
        parser.error("cg-tol 须为有限正数, cg-maxiter 须为正整数")
    if args.uniform_density is not None:
        if not math.isfinite(args.uniform_density) or not 0 <= args.uniform_density <= 1:
            parser.error("uniform-density 须为 [0, 1] 内的有限数")
    args.grid = tuple(args.grid)
    args.operator_levels = tuple(dict.fromkeys(args.operator_levels))
    args.preconditioners = tuple(dict.fromkeys(args.preconditioners))
    return args


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """生成可审阅的执行计划, 不检查数值输入文件.

    Parameters
    ----------
    args : argparse.Namespace
        命令行配置.

    Returns
    -------
    dict
        输入、求解口径和任务组合.
    """
    cases = [ComparisonCase(level, precond)
             for level in args.operator_levels for precond in args.preconditioners]
    return {
        "implementation_status": "framework_only",
        "backend": "numpy",
        "device": "cpu",
        "grid": args.grid,
        "density": {
            "kind": "physical_npy" if args.density_file else "uniform_physical",
            "file": str(args.density_file) if args.density_file else None,
            "uniform_value": args.uniform_density,
            "default_value_source": "run_fem_ea_vs_fa.VOLFRAC",
        },
        "initial_displacement": (str(args.initial_displacement)
                                 if args.initial_displacement else "zero"),
        "cg": {"rtol": 0.0, "atol_rule": "cg_tol * ||F||_2",
               "cg_tol": args.cg_tol, "maxiter": args.cg_maxiter},
        "output_dir": str(args.output_dir),
        "cases": [asdict(case) for case in cases],
    }


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """预留数值执行入口, 当前不产生结果.

    Parameters
    ----------
    args : argparse.Namespace
        对照任务配置.

    Returns
    -------
    dict
        后续实现的配置、任务结果和收敛解一致性报告.

    Raises
    ------
    NotImplementedError
        数值执行尚未接入.
    """
    # 后续: 复用模型配置 -> 校验场 -> 独立进程逐任务执行 -> 比较收敛解 -> 保存 JSON.
    # 单任务: 构造分析器 -> assemble_stiff_matrix -> apply_bc -> solve_system.
    # 使用同一个输入状态; 禁止分别启动优化生成各配置自己的密度场.
    raise NotImplementedError("当前仅提供框架; 请用 --dry-run 查看任务清单")


def main(argv: list[str] | None = None) -> None:
    """执行框架入口.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数.
    """
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(build_plan(args), ensure_ascii=False, indent=2))
        return
    try:
        run_comparison(args)
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
