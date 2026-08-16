"""Hu--Zhang 拓扑优化论文实验配置加载与校验模块."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib
from typing import Any


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


def resolve_runs(
    case: dict[str, Any],
    arguments: argparse.Namespace,
) -> list[tuple[str, int]]:
    """将方法和空间次数选择展开为确定的运行组合.

    参数:
        case: 目标算例配置字典.
        arguments: 命令行解析对象.

    返回:
        由 ``(method, order)`` 二元组构成的执行任务列表.

    异常:
        ConfigurationError: 当指定了不支持的方法或未注册的阶次时抛出.
    """
    if arguments.method is None:
        if getattr(arguments, "check_only", False):
            return []
        raise ConfigurationError("实际运行必须提供 --method.")

    discretization = case["discretization"]
    methods = tuple(case.get("methods", ()))
    if arguments.method != "all" and arguments.method not in methods:
        raise ConfigurationError(f"{case['id']}: 未配置方法 {arguments.method}.")

    comparison_orders = tuple(int(o) for o in discretization["comparison_orders"])
    selected_orders = tuple(arguments.order) if arguments.order else comparison_orders
    invalid = sorted(set(selected_orders) - set(comparison_orders))
    if invalid:
        raise ConfigurationError(f"{case['id']}: 比较阶次不在配置中: {invalid}.")

    runs: list[tuple[str, int]] = []
    if arguments.method in ("lfem", "all") and "lfem" in methods:
        runs.extend(("lfem", o) for o in selected_orders)
    if arguments.method in ("huzhang", "all") and "huzhang" in methods:
        runs.extend(("huzhang", o) for o in selected_orders)
    return runs


def configuration_summary(
    case: dict[str, Any],
    config: Any,
    runs: list[tuple[str, int]],
) -> dict[str, Any]:
    """构造无需启动有限元分析的静态配置摘要字典.

    参数:
        case: 算例配置字典.
        config: 组装器配置数据类.
        runs: 待执行的组合列表.

    返回:
        包含基本几何、网格、离散协议与求解器配置的摘要字典.
    """
    return {
        "case_id": case["id"],
        "title": case.get("title", ""),
        "model": case["model"]["name"],
        "status": case["status"],
        "domain": [0.0, 160.0, 0.0, 20.0],
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
