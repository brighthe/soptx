"""Hu--Zhang 拓扑优化论文实验配置加载与校验模块."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tomllib
from typing import Any

# --- 目录常量 --------------------------------------------------------------
# 实验内所有脚本的路径都由此派生, 不再各自硬编码 /home/... 或 /mnt/c/... 绝对路径.
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_DIR = REPOSITORY_ROOT / "src"
CASES_FILE = EXPERIMENT_DIR / "cases.toml"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

# 论文插图目录属于另一个仓库 (dut-postdoc), 默认取 WSL 下的挂载路径; 可用环境变量
# HUZHANG_PAPER_FIGDIR 覆盖, 目录不存在时同步步骤自动跳过而非报错.
PAPER_FIGURE_DIR = Path(
    os.environ.get("HUZHANG_PAPER_FIGDIR", "/mnt/c/workspace/dut-postdoc/papers/figures")
)


def bootstrap_source_path() -> None:
    """把 soptx 源码目录与本实验目录加入 ``sys.path``, 供脚本按文件路径直接运行."""
    for path in (SOURCE_DIR, EXPERIMENT_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


class ConfigurationError(RuntimeError):
    """表示论文算例配置不完整或不合法."""


class UnsupportedModelError(RuntimeError):
    """表示配置选择了尚未接入 Runner 的物理模型."""


def load_cases(path: Path) -> tuple[dict[str, Any], ...]:
    """读取并校验多算例 TOML 配置.

    参数:
        path: ``cases.toml`` 文件路径.

    返回:
        已校验的算例配置字典元组.

    异常:
        ConfigurationError: 当 TOML 格式错误或关键字段缺失/重复时抛出.
    """
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"无法读取配置: {error}") from error

    if data.get("schema_version") != 1:
        raise ConfigurationError("schema_version 必须为 1.")
    if data.get("stage") != "soptx/huzhang-topopt-paper/v1":
        raise ConfigurationError("stage 与当前投稿实验不匹配.")

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ConfigurationError("配置必须包含非空 [[cases]] 列表.")

    identifiers: set[str] = set()
    validated: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ConfigurationError("每个 [[cases]] 项必须是 TOML 表.")
        identifier = case.get("id")
        model = case.get("model")
        discretization = case.get("discretization")
        optimization = case.get("optimization")

        if not isinstance(identifier, str) or not identifier:
            raise ConfigurationError("每个 case 必须具有非空 id.")
        if identifier in identifiers:
            raise ConfigurationError(f"case id 重复: {identifier}.")
        if not isinstance(model, dict) or not isinstance(model.get("name"), str):
            raise ConfigurationError(f"{identifier}: 必须指定 model.name.")
        if not isinstance(model.get("parameters"), dict):
            raise ConfigurationError(f"{identifier}: 必须指定 model.parameters.")
        if not isinstance(discretization, dict) or not isinstance(optimization, dict):
            raise ConfigurationError(f"{identifier}: 必须指定 discretization 和 optimization.")
        if case.get("status") not in ("ready", "planned"):
            raise ConfigurationError(f"{identifier}: status 只能为 ready 或 planned.")

        identifiers.add(identifier)
        validated.append(case)

    return tuple(validated)


def select_cases(
    cases: tuple[dict[str, Any], ...],
    requested: list[str] | None,
) -> tuple[dict[str, Any], ...]:
    """按命令行参数筛选算例列表.

    参数:
        cases: 全量候选算例元组.
        requested: 用户传入的算例 id 列表.

    返回:
        筛选后的算例元组.

    异常:
        ConfigurationError: 当算例未找到或未就绪时抛出.
    """
    if not requested:
        raise ConfigurationError("实际运行必须提供 --case; 使用 --list 查看可用 id.")

    by_id = {case["id"]: case for case in cases}
    if "all" in requested:
        if len(requested) != 1:
            raise ConfigurationError("--case all 不能与其他 case id 同时使用.")
        selected = tuple(case for case in cases if case["status"] == "ready")
    else:
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ConfigurationError(f"未知 case id: {unknown}.")
        selected = tuple(by_id[identifier] for identifier in requested)

    if not selected:
        raise ConfigurationError("没有可运行的 ready case.")
    planned = [case["id"] for case in selected if case["status"] != "ready"]
    if planned:
        raise ConfigurationError(f"以下 case 尚未实现: {planned}.")
    return selected


def flatten_parameters(case: dict[str, Any]) -> dict[str, Any]:
    """把模型、离散和优化参数合并为组装器的输入字典.

    参数:
        case: 原始算例字典.

    返回:
        扁平化后的全量参数字典.
    """
    return {
        **case["model"]["parameters"],
        **case["discretization"],
        **case["optimization"],
    }


# 缺省求解链取本文方法; 该 case 没注册 huzhang 时退回它声明的首项.
DEFAULT_METHOD = "huzhang"


def default_method(methods: tuple[str, ...]) -> str:
    """给出该 case 缺省使用的求解链, 供 resolve_runs 与 run.py --list 共用一个口径."""
    return DEFAULT_METHOD if DEFAULT_METHOD in methods else methods[0]


def resolve_runs(
    case: dict[str, Any],
    arguments: argparse.Namespace,
) -> list[tuple[str, int]]:
    """将方法和空间次数选择展开为确定的运行组合.

    ``--method`` 与 ``--order`` 省略时只跑一个默认组合 (见 ``default_method`` 与
    ``comparison_orders`` 的最小值); ``--full`` 才展开成 ``cases.toml`` 声明的完整
    对比组。注册表里的 ``methods`` / ``comparison_orders`` 同时是这两个选项的白名单,
    ``supplementary_orders`` 额外放宽 ``--order`` 的白名单但不参与任何自动展开.

    参数:
        case: 目标算例配置字典.
        arguments: 命令行解析对象.

    返回:
        由 ``(method, order)`` 二元组构成的执行任务列表.

    异常:
        ConfigurationError: 当指定了不支持的方法或未注册的阶次时抛出.
    """
    # 缺省只展开一个组合: 一条命令一次运行, 便于探索与调试; 论文那套方法/阶次
    # 对比是显式动作, 走 --full (或显式 --method all / --order k1 k2).
    full = bool(getattr(arguments, "full", False))

    discretization = case["discretization"]
    methods = tuple(case.get("methods", ()))
    if not methods:
        raise ConfigurationError(f"{case['id']}: 未声明 methods.")
    requested = arguments.method or ("all" if full else default_method(methods))
    if requested != "all" and requested not in methods:
        raise ConfigurationError(f"{case['id']}: 未配置方法 {requested}.")

    comparison_orders = tuple(int(o) for o in discretization["comparison_orders"])
    # 补充专题阶次只进白名单, 不进缺省也不进 --full: 如半域梁的 k=1 只服务 supp-k1
    # 失效专题, 混进论文对比组会跑出一批不该进表的数.
    supplementary_orders = tuple(int(o) for o in discretization.get("supplementary_orders", ()))
    if arguments.order:
        selected_orders = tuple(arguments.order)
    elif full:
        selected_orders = comparison_orders
    else:
        selected_orders = (min(comparison_orders),)
    invalid = sorted(set(selected_orders) - set(comparison_orders) - set(supplementary_orders))
    if invalid:
        raise ConfigurationError(f"{case['id']}: 比较阶次不在配置中: {invalid}.")

    runs: list[tuple[str, int]] = []
    if requested in ("lfem", "all") and "lfem" in methods:
        runs.extend(("lfem", o) for o in selected_orders)
    if requested in ("huzhang", "all") and "huzhang" in methods:
        runs.extend(("huzhang", o) for o in selected_orders)
    return runs


def configuration_summary(
    case: dict[str, Any],
    config: Any,
    runs: list[tuple[str, int]],
    domain: tuple[float, ...],
) -> dict[str, Any]:
    """构造无需启动有限元分析的静态配置摘要字典.

    参数:
        case: 算例配置字典.
        config: 组装器配置数据类.
        runs: 待执行的组合列表.
        domain: 实际物理问题计算域, 从已实例化问题的 ``domain`` 读取.

    返回:
        包含基本几何、网格、离散协议与求解器配置的摘要字典.
    """
    return {
        "case_id": case["id"],
        "title": case.get("title", ""),
        "model": case["model"]["name"],
        "status": case["status"],
        "domain": list(domain),
        "nx": config.nx,
        "ny": config.ny,
        "comparison_protocol": "LFEM p=k versus Hu--Zhang stress order k, q=2k+2",
        "selected_runs": [
            {"method": method, "order": order, "integration_order": 2 * order + 2}
            for method, order in runs
        ],
        "solver": config.solve_method,
        "max_iterations": config.max_iterations,
    }
