# -*- coding: utf-8 -*-
"""FA 变密度拓扑优化工况加载与校验.

字段的单一事实来源是 ``TopOptCase`` 的声明: 每个字段用 ``_spec`` 同时给出
分类轴、解析函数与通用校验规则, 解析 (``_build_case``)、通用校验
(``_validate``) 与结果影响判据 (``GATE_FIELDS``) 全部由这份声明推导,
不再各写一份字段清单。新增参数只改 ``TopOptCase``; 忘记标注分类轴会在
import 时直接失败。

一条 ``[[cases]]`` 只登记一种模型 (问题 + 几何 + 载荷 + 单元核) 及其基准参数;
参数变化不进注册表, 一律走命令行 ``--override``。产物目录分两层: 第一层是工况
id, 第二层由规范化的 override 标签推导 (见 ``TopOptCase.run_id``), 与
experiments/topopt_simp_ea/ 用相同 override 跑就落在相同路径下, EA 侧 compare
据此配对。
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
CASES_FILE = EXPERIMENT_DIR / "cases.toml"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
FIGURE_DATA_DIR = EXPERIMENT_DIR / "figure_data"

# 非工况参数: collect.py 的事后验收阈值, 不进入求解, 因此不进 cases.toml。
# 所有影响求解的参数一律逐工况显式写在注册表里。
VOLUME_TOLERANCE = 5.0e-3
NONTRIVIAL_STD = 1.0e-3

VALID_CELL_TYPES = {2: {"tri", "quad"}, 3: {"tet", "hex"}}
VALID_PROBLEMS = {
    2: {
        "cantilever_corner",
        "cantilever_middle",
        "bearing_device",
        "half_mbb_beam_right",
        "simply_supported_bridge",
    },
    3: {"cantilever_right_bottom_edge"},
}
VALID_FILTERS = {"none", "sensitivity", "density", "projection"}
VALID_OPTIMIZERS = {"oc", "mma"}
VALID_INTERPOLATIONS = {"simp", "msimp", "ramp"}
VALID_ASSEMBLY_METHODS = {"standard", "voigt", "fast"}
VALID_SOLVERS = {"scipy", "mumps", "cg"}
# CG 的预条件子。"none" 即不加预条件; "jacobi" 走 DiagonalPreconditioner,
# 此时 LagrangeFEMAnalyzer 会自动把真残差刷新间隔绑成 50, 无需在此另开旋钮。
VALID_PRECONDS = {"none", "jacobi"}

# 分类轴 = "改了它, 变的是题目还是解题路径" (口径同 cases.toml 头部):
#   A 问题 / B 离散 / C 拓扑建模 / D 算法 —— 四者都影响结果, 进门禁与产物命名;
#   ledger 台账 —— 只做溯源, 不进求解, 不进门禁。
RESULT_AXES = ("A", "B", "C", "D")
LEDGER_AXIS = "ledger"

# id 与 override 标签直接拼成目录路径, 不做转义, 所以限定安全字符。
_NAME_RE = re.compile(r"[0-9A-Za-z_.\-]+")
_TAG_SEPARATOR = "__"
# 没有 override 的基准运行, 第二层目录名; 与参数标签同级, 不留空目录名。
_BASE_TAG = "base"


class ConfigError(RuntimeError):
    """工况注册表无效."""


def _as_str(value: Any) -> str:
    return str(value)


def _as_int(value: Any) -> int:
    return int(value)


def _as_float(value: Any) -> float:
    return float(value)


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


def _as_float_tuple(value: Any) -> tuple[float, ...]:
    return tuple(float(item) for item in value)


def _as_override_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(name), str(text)) for name, text in dict(value).items()))


def _spec(axis: str, parse: Any, **rules: Any):
    """声明一个工况字段.

    axis    分类轴, RESULT_AXES 之一或 LEDGER_AXIS;
    parse   toml 原值 -> 字段值的转换;
    missing 键缺失时的取值, 不给则该字段必填;
    choices 取值集合; positive 要求 > 0。跨字段与依赖维度的约束写在 _validate。
    """
    metadata = {"axis": axis, "parse": parse, **rules}
    if "missing" in rules:
        return field(default=rules["missing"], metadata=metadata)
    return field(metadata=metadata)


@dataclass(frozen=True)
class TopOptCase:
    """一次运行的全部参数; 分组与顺序同 cases.toml."""

    # -- 台账 (不进求解) --
    # id 是模型身份, 也是产物目录的第一层; FA/EA 配对靠它。
    id: str = _spec(LEDGER_AXIS, _as_str)
    # -- A 问题 (连续问题本身; 改了就是换一道题) --
    problem: str = _spec("A", _as_str)
    domain: tuple[float, ...] = _spec("A", _as_float_tuple)
    load: float = _spec("A", _as_float)
    emax: float = _spec("A", _as_float)
    nu: float = _spec("A", _as_float)
    volfrac: float = _spec("A", _as_float)
    # -- B 离散 (有限维近似; grid 同时决定设计空间分辨率) --
    dimension: int = _spec("B", _as_int)
    cell_type: str = _spec("B", _as_str)
    grid: tuple[int, ...] = _spec("B", _as_int_tuple)
    space_degree: int = _spec("B", _as_int, positive=True)
    integration_order: int = _spec("B", _as_int, positive=True)
    # -- C 拓扑优化建模 (0/1 松弛手段; 改了必然改变最优解) --
    interpolation_method: str = _spec("C", _as_str, choices=VALID_INTERPOLATIONS)
    simp_penalty: float = _spec("C", _as_float, positive=True)
    void_youngs_modulus: float = _spec("C", _as_float)
    filter_type: str = _spec("C", _as_str, choices=VALID_FILTERS)
    filter_radius: float = _spec("C", _as_float)
    projection_beta: float = _spec("C", _as_float)
    projection_eta: float = _spec("C", _as_float)
    projection_beta_max: float = _spec("C", _as_float)
    projection_continuation_iter: int = _spec("C", _as_int, positive=True)
    # -- D 算法 (求同一问题的不同路径; 收敛意义下应给出同一解) --
    optimizer: str = _spec("D", _as_str, choices=VALID_OPTIMIZERS)
    move: float = _spec("D", _as_float, positive=True)
    density_min: float = _spec("D", _as_float)
    max_iter: int = _spec("D", _as_int, positive=True)
    tol_change: float = _spec("D", _as_float, positive=True)
    assembly_method: str = _spec("D", _as_str, choices=VALID_ASSEMBLY_METHODS)
    solve_method: str = _spec("D", _as_str, choices=VALID_SOLVERS)
    # cg 参数只在 solve_method = "cg" 时生效并校验; 直接法工况可不写。
    cg_rtol: float = _spec("D", _as_float, missing=0.0)
    cg_atol: float = _spec("D", _as_float, missing=0.0)
    cg_maxiter: int = _spec("D", _as_int, missing=0)
    cg_precond: str = _spec("D", _as_str, choices=VALID_PRECONDS, missing="none")
    # -- 台账 (可选; 必须排在必填字段之后) --
    summary: str = _spec(LEDGER_AXIS, _as_str, missing="")
    # 这条工况在实验里承担的角色 (覆盖矩阵的哪一格 / 哪个对照组), 只进 --list 表,
    # 不进求解; 口径同 experiments/huzhang_topopt_paper 的 role 字段。
    role: str = _spec(LEDGER_AXIS, _as_str, missing="")
    # 命令行 override 的原文 (字段名, 文本), 按字段名排序; 空表示注册表基准运行。
    # 只做溯源与 collect 重放, 参数本身已经写进对应字段。
    overrides: tuple[tuple[str, str], ...] = _spec(
        LEDGER_AXIS, _as_override_pairs, missing=()
    )

    @property
    def run_id(self) -> str:
        """一次运行的唯一标识 = 产物目录相对 outputs/ 的路径 ``<id>/<参数标签>``.

        第一层是工况 id, 第二层是相对注册表基准的偏离: 按字段名顺序拼
        ``<字段>-<取值>`` 再用 ``__`` 连接, 基准运行没有偏离, 记 ``base``。
        取值取解析后的规范形式 (而非命令行原文), 所以 ``filter_radius=2.4``
        与 ``filter_radius=2.40`` 落在同一目录。
        标识与落盘位置同一个来源, 不会出现 "叫这个名字却写到别处"。
        两层的分法与 experiments/huzhang_topopt_paper 一致 (第一层认工况,
        第二层认参数), 三个实验的 outputs/ 用同一套读法。
        """
        tags = [
            f"{name}-{_tag_text(getattr(self, name))}" for name, _ in self.overrides
        ]
        return f"{self.id}/{_TAG_SEPARATOR.join(tags) if tags else _BASE_TAG}"

    @property
    def output_dir(self) -> Path:
        """产物目录 outputs/<run_id>/ = outputs/<工况 id>/<参数标签>/."""
        return OUTPUT_DIR / self.run_id


def _tag_text(value: Any) -> str:
    """字段取值 -> 目录名里的规范文本; 只含 _NAME_RE 允许的字符."""
    if isinstance(value, tuple):
        text = "x".join(_tag_text(item) for item in value)
    elif isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return re.sub(r"[^0-9A-Za-z_.\-]+", "-", text)


# 未标注分类轴的字段既不会进门禁也不会进产物命名 —— 那意味着改了它会静默覆盖
# 旧结果, 比漏写校验更危险, 所以在 import 期直接失败。
_UNCLASSIFIED = tuple(
    entry.name
    for entry in fields(TopOptCase)
    if entry.metadata.get("axis") not in (*RESULT_AXES, LEDGER_AXIS)
)
if _UNCLASSIFIED:
    raise ConfigError(
        "以下字段未标注分类轴 (RESULT_AXES 或 LEDGER_AXIS): "
        + ", ".join(_UNCLASSIFIED)
    )

# 影响结果的全部字段, 供产物落盘与 FA-EA 门禁共用。
GATE_FIELDS = tuple(
    entry.name
    for entry in fields(TopOptCase)
    if entry.metadata.get("axis") in RESULT_AXES
)
# override 只允许改影响结果的参数; 台账字段是工况身份, 不可覆盖。
NON_OVERRIDABLE_FIELDS = frozenset(
    entry.name
    for entry in fields(TopOptCase)
    if entry.metadata.get("axis") == LEDGER_AXIS
)
# 注册表里允许出现的键 = 全部字段减去只能由命令行产生的 overrides。
_REGISTRY_FIELDS = frozenset(
    entry.name for entry in fields(TopOptCase) if entry.name != "overrides"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def config_values(case: TopOptCase) -> dict[str, Any]:
    """影响结果的全部参数: 字段名 -> 可 JSON 序列化的取值.

    产物落盘与 FA-EA 门禁共用这一份, 谁都不重新起名或挑字段。
    """
    return {name: _jsonable(getattr(case, name)) for name in GATE_FIELDS}


def _build_case(values: dict[str, Any]) -> TopOptCase:
    """按字段声明逐项解析; 缺字段、多字段与类型错都在这里点名."""
    unknown = sorted(set(values) - _REGISTRY_FIELDS)
    if unknown:
        raise ConfigError(
            f"工况 {values.get('id')!r} 含未声明的字段: {', '.join(unknown)}."
        )
    kwargs: dict[str, Any] = {}
    for entry in fields(TopOptCase):
        metadata = entry.metadata
        if entry.name in values:
            try:
                kwargs[entry.name] = metadata["parse"](values[entry.name])
            except (TypeError, ValueError) as error:
                raise ConfigError(
                    f"工况字段 {entry.name} 取值非法: {values[entry.name]!r}"
                ) from error
        elif "missing" in metadata:
            kwargs[entry.name] = metadata["missing"]
        else:
            raise ConfigError(f"工况缺少字段: {entry.name}")
    case = TopOptCase(**kwargs)
    _validate(case)
    return case


def _validate(case: TopOptCase) -> None:
    """通用规则由字段声明推导, 这里只写跨字段与依赖维度的约束."""
    if not case.id:
        raise ConfigError("工况缺少 id.")
    if not _NAME_RE.fullmatch(case.id):
        # id 直接当目录名用, 分隔符和空格会把结果写到意料之外的位置。
        raise ConfigError(
            f"工况 {case.id!r} 的 id 只能是字母/数字/下划线/点/连字符."
        )
    for entry in fields(TopOptCase):
        value = getattr(case, entry.name)
        choices = entry.metadata.get("choices")
        if choices is not None and value not in choices:
            raise ConfigError(
                f"工况 {case.id} 的 {entry.name} 不受支持: {value} "
                f"(可选 {', '.join(sorted(choices))})"
            )
        if entry.metadata.get("positive") and not value > 0:
            raise ConfigError(f"工况 {case.id} 的 {entry.name} 必须为正数.")
    if case.dimension not in VALID_CELL_TYPES:
        raise ConfigError(f"工况 {case.id} 的维度必须为 2 或 3.")

    # -- A 问题 --
    if case.problem not in VALID_PROBLEMS[case.dimension]:
        raise ConfigError(f"工况 {case.id} 的问题类型与维度不匹配.")
    if len(case.domain) != case.dimension:
        raise ConfigError(f"工况 {case.id} 的 domain 维数不匹配.")
    if any(value <= 0 for value in case.domain):
        raise ConfigError(f"工况 {case.id} 的 domain 必须为正数.")
    if not 0.0 < case.volfrac <= 1.0:
        raise ConfigError(f"工况 {case.id} 的体积上限必须落在 (0, 1].")

    # -- B 离散 --
    if case.cell_type not in VALID_CELL_TYPES[case.dimension]:
        raise ConfigError(f"工况 {case.id} 的单元类型与维度不匹配.")
    if len(case.grid) != case.dimension:
        raise ConfigError(f"工况 {case.id} 的 grid 维数不匹配.")
    if any(value <= 0 for value in case.grid):
        raise ConfigError(f"工况 {case.id} 的 grid 必须为正数.")

    # -- C 拓扑优化建模 --
    if not 0.0 < case.void_youngs_modulus < case.emax:
        raise ConfigError(f"工况 {case.id} 的材料模量不合法.")
    if case.filter_type != "none" and case.filter_radius <= 0.0:
        raise ConfigError(f"工况 {case.id} 的过滤半径必须为正数.")
    if (
        case.projection_beta <= 0.0
        or case.projection_beta_max < case.projection_beta
    ):
        raise ConfigError(f"工况 {case.id} 的 projection beta 区间不合法.")
    if not 0.0 < case.projection_eta < 1.0:
        raise ConfigError(f"工况 {case.id} 的 projection eta 必须落在 (0, 1).")

    # -- D 算法 --
    if not 0.0 <= case.density_min < case.volfrac:
        raise ConfigError(f"工况 {case.id} 的设计变量下界不合法.")
    if case.solve_method == "cg" and (
        case.cg_rtol <= 0.0 or case.cg_atol <= 0.0 or case.cg_maxiter <= 0
    ):
        raise ConfigError(f"工况 {case.id} 的 cg 求解参数必须为正数.")
    if case.solve_method != "cg" and case.cg_precond != "none":
        raise ConfigError(
            f"工况 {case.id} 的 solve_method={case.solve_method} 是直接法, "
            f"不接受 cg_precond={case.cg_precond}."
        )


def parse_override_args(groups: list[list[str]]) -> dict[str, str]:
    """把 argparse (action=append, nargs=+) 收到的 KEY=VALUE 摊平成字典.

    同名键后写覆盖先写。与 EA 侧 run.py / compare.py 同一套解析, 保证同一串
    命令行在两侧解析出同一组 override。
    """
    overrides: dict[str, str] = {}
    for text in (item for group in groups for item in group):
        key, separator, value = text.partition("=")
        if not separator or not key.strip() or not value.strip():
            raise ConfigError(f"--override 需要 KEY=VALUE 形式: {text}")
        overrides[key.strip()] = value.strip()
    return overrides


def build_overridden_case(
    case: TopOptCase, overrides: dict[str, str]
) -> TopOptCase:
    """在注册表基准运行上应用命令行 override, 得到一次新的运行.

    override 是正式的参数维度, 不是探索区: 产物落在 outputs/<run_id>/,
    与基准运行同属一个工况目录, 通过验收即为证据。FA/EA 两侧用同一组 override 得到
    同一个 run_id, compare 据此配对, 参数是否真的一致仍由门禁逐项断言。
    """
    if not overrides:
        raise ConfigError("override 为空.")
    if case.overrides:
        raise ConfigError(f"运行 {case.run_id} 已带 override, 不能再叠加.")
    specs = {entry.name: entry for entry in fields(TopOptCase)}
    invalid = (set(overrides) & NON_OVERRIDABLE_FIELDS) | (
        set(overrides) - set(specs)
    )
    if invalid:
        raise ConfigError(f"不可覆盖的字段: {', '.join(sorted(invalid))}.")
    parsed: dict[str, Any] = {}
    for name, text in overrides.items():
        # 逗号分隔的值展开成列表, 类型转换仍走字段自己声明的 parse。
        raw: Any = text.split(",") if "," in text else text
        try:
            parsed[name] = specs[name].metadata["parse"](raw)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"override 取值非法: {name}={text}") from error
    updated = replace(case, **parsed, overrides=_as_override_pairs(overrides))
    _validate(updated)
    return updated


def load() -> tuple[dict[str, Any], tuple[TopOptCase, ...]]:
    """加载元数据和全部注册工况 (每条 [[cases]] 一次基准运行)."""
    try:
        raw = tomllib.loads(CASES_FILE.read_text(encoding="utf-8"))
    except Exception as error:
        raise ConfigError(f"解析 {CASES_FILE} 失败: {error}") from error
    items = raw.get("cases", [])
    identifiers = [item.get("id") for item in items]
    if not items or len(identifiers) != len(set(identifiers)):
        raise ConfigError("工况为空或存在重复 id.")
    cases = tuple(_build_case(item) for item in items)
    return raw.get("meta", {}), cases
