"""四范式线弹性受控对比实验主 Runner.

本实验在同一线弹性问题, 同一细网格与同一参考真解下比较四条求解路径, 它们恰好填满一张
2x2 定位表:

======================  ====================  ==========================
计算载体                精确数值              神经网络代理
======================  ====================  ==========================
全局解场 ``u(x)``       ``lagrange``          ``pinn``
缩聚算子 ``(N, K_s)``   ``substructure``      ``piml``
======================  ====================  ==========================

四条路径不是并列的竞争者, 每条边只变一个变量, 构成受控消融:

- ``lagrange`` 与 ``substructure``: 只变载体, 两者代数等价, 偏差应停留在机器精度,
  这条边是实现正确性自检而非精度比较;
- ``substructure`` 与 ``piml``: 只变是否代理, 载体相同, 给出纯代理误差;
- ``lagrange`` 与 ``pinn``: 只变是否代理, 载体相同;
- ``pinn`` 与 ``piml``: 只变载体, 两者均为代理, 回答 problem-dependent 与
  problem-independent 的复用边界差异.

四条路径的相对 ``L2`` 误差不构成同一张排名表: 精确路径在 ``1e-13`` 量级, 代理路径在
``1e-2`` 至 ``1e-4`` 量级, 排名不含信息. 结论应落在摊销盈亏点与结构保持退化上, 相应
指标见 ``metrics.breakeven_count``, ``metrics.min_eigenvalue`` 与
``metrics.rigid_mode_residual``.

使用方法::

    # 列出算例与四条路径的就绪状态
    python experiments/elasticity_paradigm_comparison/run.py --list

    # 只校验配置与受控比较契约, 不求解
    python experiments/elasticity_paradigm_comparison/run.py \
        --case tier-a-mbb-homogeneous --check-only

    # 运行指定算例的全部路径 (适配器就绪后)
    python experiments/elasticity_paradigm_comparison/run.py \
        --case tier-a-mbb-homogeneous --method all
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_ROOT / "cases.toml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs"
SOURCE_ROOT = EXPERIMENT_ROOT.parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from config import (  # noqa: E402
    METHODS,
    ConfigurationError,
    configuration_summary,
    find_case,
    fine_mesh_shape,
    load_config,
)
from solvers import SOLVER_REGISTRY, ExperimentContext  # noqa: E402


def build_context(case: dict[str, Any], protocol: dict[str, Any]) -> ExperimentContext:
    """构造四条路径共享的算例上下文.

    实现时按 ``case['model']`` 实例化 ``soptx.problems.elasticity`` 中的物理问题,
    构造 ``GlobalAssembler`` 与 ``SubstructurePrototype``, 生成 ``case['density']``
    指定的密度场, 并从 ``LagrangeFEMAnalyzer`` 取施加 Dirichlet 条件之前的全局外载与
    约束掩码. 该组装范式已存在于
    ``examples/piml_substructure_elasticity/verify_stiffness_route.py``, 下沉时直接复用,
    不在本目录重写.

    参数:
        case: 已校验的单个算例配置.
        protocol: 已校验的共享冻结项.

    返回:
        四条路径共享的上下文.

    异常:
        NotImplementedError: 当前恒抛出, 上下文组装尚未接线.
    """
    raise NotImplementedError(
        "上下文组装待接线: 复用 examples/piml_substructure_elasticity/verify_stiffness_route.py "
        "中的 GlobalAssembler 与外载提取范式, 下沉后在此调用."
    )


def print_registry() -> None:
    """打印四条路径的 2x2 象限归属与就绪状态."""
    print("=" * 88)
    print("四范式定位表与适配器就绪状态")
    print("=" * 88)
    header = f"{'路径':<14}{'计算载体':<20}{'求解性质':<12}{'就绪':<6}阻塞原因"
    print(header)
    print("-" * 88)
    for name in METHODS:
        solver = SOLVER_REGISTRY[name]
        ready = "是" if solver.READY else "否"
        blocker = "" if solver.READY else (solver.BLOCKER or "")
        print(f"{name:<14}{solver.CARRIER:<20}{solver.NATURE:<12}{ready:<6}{blocker}")
    print()


def print_cases(config: dict[str, Any]) -> None:
    """打印全部算例摘要.

    参数:
        config: ``load_config`` 的返回值.
    """
    print("=" * 88)
    print("算例注册表")
    print("=" * 88)
    for case in config["cases"]:
        print(configuration_summary(case))
        print("-" * 88)


def check_case(case: dict[str, Any], protocol: dict[str, Any]) -> None:
    """校验单个算例是否满足受控比较契约并打印结论.

    参数:
        case: 已校验的单个算例配置.
        protocol: 已校验的共享冻结项.
    """
    print(configuration_summary(case))
    print("-" * 88)
    print(f"参考真解     : {protocol['reference_method']}")
    print(f"共享细网格   : {'x'.join(str(v) for v in fine_mesh_shape(case))}")
    print(f"PINN 边界    : {protocol['pinn_boundary_mode']} (硬约束)")
    print(f"随机数种子   : {protocol['random_seed']}")
    print(f"精确后端     : {protocol['backend']}   代理后端: {protocol['ml_backend']}")
    print("-" * 88)

    pending = [m for m in case["methods"] if not SOLVER_REGISTRY[m].READY]
    if pending:
        print(f"配置合法; 尚未就绪的路径: {', '.join(pending)}")
    else:
        print("配置合法; 全部路径已就绪.")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数.

    参数:
        argv: 命令行参数列表; 为 ``None`` 时取 ``sys.argv[1:]``.

    返回:
        解析后的参数命名空间.
    """
    parser = argparse.ArgumentParser(
        description="四范式线弹性受控对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="cases.toml 路径")
    parser.add_argument("--case", type=str, default=None, help="算例 id")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=(*METHODS, "all"),
        help="求解路径; all 表示该算例登记的全部路径",
    )
    parser.add_argument("--list", action="store_true", help="列出算例与路径就绪状态后退出")
    parser.add_argument("--check-only", action="store_true", help="只校验配置, 不求解")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT, help="运行产物目录"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """实验入口.

    参数:
        argv: 命令行参数列表; 为 ``None`` 时取 ``sys.argv[1:]``.

    返回:
        进程退出码; ``0`` 表示成功, ``1`` 表示配置错误, ``2`` 表示路径尚未就绪.
    """
    args = parse_arguments(argv)

    try:
        config = load_config(args.config)
    except ConfigurationError as error:
        print(f"配置错误: {error}", file=sys.stderr)
        return 1

    if args.list:
        print_registry()
        print_cases(config)
        return 0

    if args.case is None:
        print("必须通过 --case 指定算例; 可用算例见 --list.", file=sys.stderr)
        return 1

    try:
        case = find_case(config, args.case)
    except ConfigurationError as error:
        print(f"配置错误: {error}", file=sys.stderr)
        return 1

    protocol = config["protocol"]
    methods = case["methods"] if args.method == "all" else [args.method]
    if args.method != "all" and args.method not in case["methods"]:
        print(f"算例 {case['id']} 未登记路径 {args.method}.", file=sys.stderr)
        return 1

    if args.check_only:
        check_case(case, protocol)
        return 0

    try:
        context = build_context(case, protocol)
        for name in methods:
            SOLVER_REGISTRY[name]().solve(context, case, protocol)
    except NotImplementedError as error:
        print(f"尚未就绪: {error}", file=sys.stderr)
        print("当前可用: --list 查看就绪状态, --check-only 校验配置.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
