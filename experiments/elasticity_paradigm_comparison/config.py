"""四范式线弹性对比实验的配置加载与校验模块."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any

STAGE = "soptx/elasticity-paradigm-comparison/v1"

# 四条求解路径的登记名. 顺序即 2x2 定位表的行列顺序: 先经典后 ML, 先全局后局部.
METHODS: tuple[str, ...] = ("lagrange", "pinn", "substructure", "piml")

# 各路径所属的 2x2 象限, 用于 run.py 的报告分组与 manifest 落盘.
METHOD_QUADRANT: dict[str, tuple[str, str]] = {
    "lagrange": ("global_field", "exact"),
    "pinn": ("global_field", "surrogate"),
    "substructure": ("condensed_operator", "exact"),
    "piml": ("condensed_operator", "surrogate"),
}


class ConfigurationError(RuntimeError):
    """表示对比实验配置不完整或不合法."""


class UnsupportedMethodError(RuntimeError):
    """表示配置选择了尚未接入 Runner 的求解路径."""


def load_config(path: Path) -> dict[str, Any]:
    """读取并校验对比实验 TOML 配置.

    参数:
        path: ``cases.toml`` 文件路径.

    返回:
        含 ``protocol`` 与 ``cases`` 两个键的配置字典, ``cases`` 为已校验的算例元组.

    异常:
        ConfigurationError: 当 TOML 格式错误, ``stage`` 不匹配或关键字段缺失时抛出.
    """
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"无法读取配置: {error}") from error

    if data.get("schema_version") != 1:
        raise ConfigurationError("schema_version 必须为 1.")
    if data.get("stage") != STAGE:
        raise ConfigurationError(f"stage 必须为 {STAGE}, 当前为 {data.get('stage')!r}.")

    protocol = _validate_protocol(data.get("protocol"))
    cases = _validate_cases(data.get("cases"), protocol)
    return {"protocol": protocol, "cases": cases}


def _validate_protocol(protocol: Any) -> dict[str, Any]:
    """校验 ``[protocol]`` 共享冻结项.

    参数:
        protocol: TOML 中的 ``[protocol]`` 表.

    返回:
        已校验的冻结项字典.

    异常:
        ConfigurationError: 当必填冻结项缺失或取值非法时抛出.
    """
    if not isinstance(protocol, dict):
        raise ConfigurationError("配置必须包含 [protocol] 表.")

    reference = protocol.get("reference_method")
    if reference not in METHODS:
        raise ConfigurationError(f"protocol.reference_method 必须取自 {METHODS}.")
    if protocol.get("shared_fine_mesh") is not True:
        raise ConfigurationError(
            "protocol.shared_fine_mesh 必须为 true: 四条路径不共享细网格时误差不可比."
        )
    if protocol.get("pinn_boundary_mode") != "hard":
        raise ConfigurationError(
            "protocol.pinn_boundary_mode 必须为 hard: 软约束会把范式对比退化成 loss 权重调参对比."
        )
    if not isinstance(protocol.get("random_seed"), int):
        raise ConfigurationError("protocol.random_seed 必须为整数.")
    for key in ("dtype", "backend", "ml_backend"):
        if not isinstance(protocol.get(key), str) or not protocol[key]:
            raise ConfigurationError(f"protocol.{key} 必须为非空字符串.")
    return protocol


def _validate_cases(cases: Any, protocol: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """校验 ``[[cases]]`` 列表.

    参数:
        cases: TOML 中的 ``[[cases]]`` 列表.
        protocol: 已校验的共享冻结项, 用于交叉检查参考路径是否在 ``methods`` 内.

    返回:
        已校验的算例配置元组.

    异常:
        ConfigurationError: 当 id 缺失/重复, 必填子表缺失或 methods 非法时抛出.
    """
    if not isinstance(cases, list) or not cases:
        raise ConfigurationError("配置必须包含非空 [[cases]] 列表.")

    identifiers: set[str] = set()
    validated: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ConfigurationError("每个 [[cases]] 项必须是 TOML 表.")

        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ConfigurationError("每个 case 必须具有非空 id.")
        if identifier in identifiers:
            raise ConfigurationError(f"case id 重复: {identifier}.")
        if case.get("status") not in ("ready", "planned"):
            raise ConfigurationError(f"{identifier}: status 只能为 ready 或 planned.")
        if case.get("tier") not in ("A", "B"):
            raise ConfigurationError(f"{identifier}: tier 只能为 A 或 B.")

        methods = case.get("methods")
        if not isinstance(methods, list) or not methods:
            raise ConfigurationError(f"{identifier}: 必须给出非空 methods 列表.")
        unknown = [m for m in methods if m not in METHODS]
        if unknown:
            raise ConfigurationError(f"{identifier}: 未知求解路径 {unknown}.")
        if protocol["reference_method"] not in methods:
            raise ConfigurationError(
                f"{identifier}: methods 必须包含参考路径 {protocol['reference_method']}."
            )

        for key in ("model", "discretization", "density"):
            if not isinstance(case.get(key), dict):
                raise ConfigurationError(f"{identifier}: 必须指定 [cases.{key}].")
        if not isinstance(case["model"].get("name"), str):
            raise ConfigurationError(f"{identifier}: 必须指定 model.name.")
        if not isinstance(case["model"].get("parameters"), dict):
            raise ConfigurationError(f"{identifier}: 必须指定 model.parameters.")

        _validate_discretization(identifier, case["discretization"])

        # 各范式的超参子表按 methods 条件必填: 不跑的路径不要求配置.
        if "pinn" in methods and not isinstance(case.get("pinn"), dict):
            raise ConfigurationError(f"{identifier}: methods 含 pinn 时必须指定 [cases.pinn].")
        if "piml" in methods and not isinstance(case.get("piml"), dict):
            raise ConfigurationError(f"{identifier}: methods 含 piml 时必须指定 [cases.piml].")

        identifiers.add(identifier)
        validated.append(case)

    return tuple(validated)


def _validate_discretization(identifier: str, discretization: dict[str, Any]) -> None:
    """校验单个算例的离散参数.

    参数:
        identifier: 算例 id, 仅用于错误信息.
        discretization: ``[cases.discretization]`` 表.

    异常:
        ConfigurationError: 当 ``n_sub`` 或 ``n_fine`` 缺失, 维数不一致或含非正整数时抛出.
    """
    n_sub = discretization.get("n_sub")
    n_fine = discretization.get("n_fine")
    for name, value in (("n_sub", n_sub), ("n_fine", n_fine)):
        if not isinstance(value, list) or len(value) not in (2, 3):
            raise ConfigurationError(f"{identifier}: discretization.{name} 必须是长度 2 或 3 的列表.")
        if any(not isinstance(v, int) or v <= 0 for v in value):
            raise ConfigurationError(f"{identifier}: discretization.{name} 各分量必须是正整数.")
    if len(n_sub) != len(n_fine):
        raise ConfigurationError(f"{identifier}: n_sub 与 n_fine 的维数必须一致.")


def fine_mesh_shape(case: dict[str, Any]) -> tuple[int, ...]:
    """返回算例的全局细网格单元数.

    四条路径共享的细网格由子结构划分与子结构内细剖分张成, 是全部误差度量的公共基准.

    参数:
        case: 已校验的单个算例配置.

    返回:
        各方向的全局细单元数.
    """
    discretization = case["discretization"]
    return tuple(
        s * f for s, f in zip(discretization["n_sub"], discretization["n_fine"])
    )


def find_case(config: dict[str, Any], identifier: str) -> dict[str, Any]:
    """按 id 查找算例配置.

    参数:
        config: ``load_config`` 的返回值.
        identifier: 算例 id.

    返回:
        对应的算例配置字典.

    异常:
        ConfigurationError: 当 id 不存在时抛出.
    """
    for case in config["cases"]:
        if case["id"] == identifier:
            return case
    available = ", ".join(case["id"] for case in config["cases"])
    raise ConfigurationError(f"未知 case id: {identifier}. 可用: {available}.")


def configuration_summary(case: dict[str, Any]) -> str:
    """把单个算例配置格式化为可打印摘要.

    参数:
        case: 已校验的单个算例配置.

    返回:
        多行摘要字符串.
    """
    shape = "x".join(str(v) for v in fine_mesh_shape(case))
    n_sub = "x".join(str(v) for v in case["discretization"]["n_sub"])
    n_fine = "x".join(str(v) for v in case["discretization"]["n_fine"])
    lines = [
        f"case id      : {case['id']}  (tier {case['tier']}, status {case['status']})",
        f"标题         : {case['title']}",
        f"物理模型     : {case['model']['name']}",
        f"子结构划分   : {n_sub}",
        f"子结构细网格 : {n_fine}",
        f"全局细网格   : {shape}",
        f"密度场       : {case['density']['kind']}",
        f"求解路径     : {', '.join(case['methods'])}",
        f"度量         : {', '.join(case['metrics'])}",
    ]
    return "\n".join(lines)
