# -*- coding: utf-8 -*-
"""cases.toml 的加载、校验与模型转换.

本模块只负责 cases.toml 的数据结构反序列化与静态门禁检查,
不依赖 fealpy, 不执行任何数值计算。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = EXPERIMENT_DIR / "cases.toml"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

# 面板: 眼下只有 convergence 一个。曾另有一个 correctness 面板 (EA / PA 与 FA 对同一随机
# 向量的作用结果比相对误差), 在 convergence 只有 FA 一条链时是必要的; 三个层级各有四格
# 之后, 逐档比 convergence_<层级> 与 convergence_fa 就是同一件事且覆盖更广, 故整体删除,
# 理由详见 cases.toml 头部注意 0。元组保留而不写死成字符串: id 的命名规则是
# <panel>_<scheme>, 将来再开面板时不必改 id 格式。
PANELS: Tuple[str, ...] = ("convergence",)

# 本目录只收装配层级分类法里的层级 (FA/TA -> LA -> EA/EbE -> PA/QA -> UA/NONE), 一层级一
# 组工况。soptx 现有三级: create_level 注册了 full / element / partial, LagrangeFEMAnalyzer
# 的签名也是 Literal['fa','ea','pa']。LA 串行下退化为 FA, UA 尚无实现, 均见 cases.toml 末尾。
#
# _common 里另有 stored-b / shared-ke 两个对照实现, 但它们不是分类法里的层级, 故不在本目录
# 取证; 用到它们的是 ../assembly_level_capability/, 由那边自行负责其前提与正确性。
LEVELS: Tuple[str, ...] = ("fa", "ea", "pa", "ua")

# FA 是逐档比对时的参照层级: 它走直接解法, 没有迭代求解容差, 是三条链里唯一不含
# Krylov 误差的一条。ea / pa 的判读都以它为基准。
REFERENCE_LEVEL: str = "fa"

# 本目录的 run.py 只是调度器, 不再自产任何数值产物 (correctness worker 已随面板一起删除),
# 故没有一条工况可以把 script 指向它。这里留常量是为了在注册时把这种写法拦住并给出原因,
# 而不是等子进程 argparse 报一个看不懂的错。
OWN_SCRIPT: str = "experiments/assembly_level_consistency/run.py"

# 产物重定向方式: 现存工况全是外部脚本, 调度器只用 --output-dir 给目录, 文件名由上游
# 脚本按 (dim, mesh_type, model, degree, solver, assembly_method, operator_level) 自拼。
OUTPUT_MODES: Tuple[str, ...] = ("dir",)

# 全部工况统一的单线程口径: 子进程与直接启动的 worker 都注入这组环境变量。
THREAD_ENV: Dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}



class ConfigError(ValueError):
    """cases.toml 配置校验失败时抛出."""


@dataclass(frozen=True)
class Case:
    """单个工况的强类型声明."""

    id: str
    mesh: str
    panel: str
    summary: str
    script: str
    args: Tuple[str, ...]
    output_mode: str
    artifact: str
    artifact_path: Path
    role: str = ""
    mesh_type: str = "HexahedronMesh"
    grid: str = "16^3"
    problem: str = "LinearElasticity3D_UnitCube"
    scheme: str = "ea"
    device: str = "cpu"
    n: int = 16
    min_l2_order: float = 0.0

    @property
    def ref(self) -> str:
        """(id, 网格) 这一对的紧凑写法, 用于日志、选择与报错."""
        return f"{self.id}[{self.mesh}]"

    def to_command(self, repo_root: Path) -> List[str]:
        """将工况配置还原为可执行的子进程命令行列表.

        不接受任何运行期覆盖: 档位、层级、求解器一律写死在 ``args`` 里。要换参数就改
        cases.toml 或直接手工调上游脚本, 不从调度器这一层开口子 —— 开了口子, 产物名
        与注册表就对不上了 (文件名由上游脚本按参数自拼)。
        """
        return [sys.executable, str(repo_root / self.script), *self.args]

    def subprocess_env(self) -> Dict[str, str]:
        """子进程环境: 继承当前环境并强制单线程 BLAS/OpenMP."""
        env = dict(os.environ)
        env.update(THREAD_ENV)
        return env


def load_cases(path: Optional[Path] = None) -> Tuple[Dict[str, Any], Tuple[Case, ...]]:
    """解析并校验 cases.toml 文件."""
    cases_path = path or DEFAULT_CASES_PATH
    if not cases_path.is_file():
        raise ConfigError(f"未找到 cases.toml: {cases_path}")

    with cases_path.open("rb") as f:
        try:
            data = tomllib.load(f)
        except Exception as err:
            raise ConfigError(f"TOML 语法解析失败: {err}") from err

    figure = data.get("figure", {})
    raw_cases = data.get("cases", [])
    if not raw_cases:
        raise ConfigError("cases.toml 中未定义任何 [[cases]] 工况")

    cases: List[Case] = []
    known_keys = set()

    for idx, entry in enumerate(raw_cases):
        case_id = entry.get("id")
        if not case_id:
            raise ConfigError(f"第 {idx + 1} 个工况缺少必填项 'id'")

        panel = entry.get("panel", "")
        if panel not in PANELS:
            raise ConfigError(f"工况 '{case_id}' 的 panel '{panel}' 非法, 必须为 {PANELS} 之一")

        scheme = entry.get("scheme", "")
        if scheme not in LEVELS:
            raise ConfigError(
                f"工况 '{case_id}' 的 scheme '{scheme}' 非法, 必须为 {LEVELS} 之一。"
                f"本目录只收装配层级, 不收 _common 的对照实现 (stored-b / shared-ke)"
            )
        # id 恰为 "<panel>_<scheme>" 两段, 不含网格: 网格是工况的另一个坐标 (见下面的
        # [cases.mesh.*]), 由使用者在命令行上选, 不进 id。命名规则散在注释里管不住漂移,
        # 这里拦一道。
        expected = f"{panel}_{scheme}"
        if case_id != expected:
            raise ConfigError(
                f"工况 '{case_id}' 的 id 不合命名规则: 应为 '{expected}' "
                f"(格式 <panel>_<scheme>, 网格不进 id)"
            )

        mesh_table = entry.get("mesh")
        if not isinstance(mesh_table, dict) or not mesh_table:
            raise ConfigError(
                f"工况 '{case_id}' 至少要有一个 [cases.mesh.<网格>] 子表: "
                f"一个 (层级, 网格) 组合才是一个数据点"
            )

        shared = {k: v for k, v in entry.items() if k != "mesh"}
        for mesh_tag, variant in mesh_table.items():
            if not isinstance(variant, dict):
                raise ConfigError(f"工况 '{case_id}' 的 [cases.mesh.{mesh_tag}] 必须是子表")
            # 网格标签是 id 之外的单段坐标: 段内允许 "-" (hex-distorted), 但不许再含 "_",
            # 否则产物名 <id>_<网格>.json 会读不出边界。
            if "_" in mesh_tag:
                raise ConfigError(
                    f"工况 '{case_id}' 的网格标签 '{mesh_tag}' 不得含 '_' "
                    f"(它是单段, 段内请用 '-', 如 hex-distorted)"
                )
            key = (case_id, mesh_tag)
            if key in known_keys:
                raise ConfigError(f"工况重复注册: '{case_id}' 的网格 '{mesh_tag}'")
            known_keys.add(key)

            c: Dict[str, Any] = {**shared, **variant}
            ref = f"{case_id}[{mesh_tag}]"

            script = c.get("script", "")
            if not script:
                raise ConfigError(f"工况 '{ref}' 缺少必填项 'script'")
            if script == OWN_SCRIPT:
                raise ConfigError(
                    f"工况 '{ref}' 的 script 不能指向本目录的 run.py: 它只是调度器, "
                    f"不产出数值结果 (自产的 correctness worker 已随该面板一起删除)。"
                    f"数值代码一律放在 examples/ 或 src/ 下, 由这里以子进程调用"
                )

            output_mode = c.get("output_mode", "dir")
            if output_mode not in OUTPUT_MODES:
                raise ConfigError(
                    f"工况 '{ref}' 的 output_mode '{output_mode}' 非法, 必须为 {OUTPUT_MODES} 之一"
                )

            # 文件名由上游脚本按自己的参数拼, 本目录只能给它一个落盘目录, 推导不出来,
            # 故必须显式声明 artifact —— 否则调度器无从判断产物是否真的生成。
            artifact_name = c.get("artifact", "")
            if not artifact_name:
                raise ConfigError(
                    f"工况 '{ref}' 由外部脚本 '{script}' 产出, 必须显式声明 'artifact': "
                    f"文件名由上游脚本决定, 无法从 id 推导"
                )

            cases.append(Case(
                id=case_id,
                mesh=mesh_tag,
                panel=panel,
                summary=c.get("summary", ""),
                script=script,
                args=tuple(c.get("args", [])),
                output_mode=output_mode,
                artifact=artifact_name,
                artifact_path=OUTPUT_DIR / artifact_name,
                role=c.get("role", ""),
                mesh_type=c.get("mesh_type", "HexahedronMesh"),
                grid=c.get("grid", "16^3"),
                problem=c.get("problem", "LinearElasticity3D_UnitCube"),
                scheme=scheme,
                device=c.get("device", "cpu"),
                n=int(c.get("n", 16)),
                min_l2_order=float(c.get("min_l2_order", 0.0)),
            ))

    return figure, tuple(cases)


def select(
    cases: Tuple[Case, ...],
    case_ids: Optional[Sequence[str]] = None,
    panel: Optional[str] = None,
    meshes: Optional[Sequence[str]] = None,
) -> Tuple[Case, ...]:
    """根据过滤条件筛选工况子集.

    ``case_ids`` 给的是 ``<panel>_<scheme>`` 两段 id, 命中该 id 注册的全部网格;
    要落到单个数据点再加 ``meshes``。
    """
    selected = list(cases)
    if panel:
        selected = [c for c in selected if c.panel == panel]
    if case_ids:
        target_set = set(case_ids)
        unknown = target_set - {c.id for c in cases}
        if unknown:
            raise ConfigError(f"未注册的 case id: {sorted(unknown)}")
        selected = [c for c in selected if c.id in target_set]
    if meshes:
        # 网格既可用注册表里的短标签 (quad / hex), 也可用 --list 打印的类名
        # (QuadrangleMesh / HexahedronMesh), 免得看着表却猜不出该往 --mesh 填什么。
        wanted = {m.lower() for m in meshes}
        matched = {
            c.mesh for c in selected
            if c.mesh.lower() in wanted or c.mesh_type.lower() in wanted
        }
        unknown = wanted - {m.lower() for m in matched} - {
            c.mesh_type.lower() for c in selected if c.mesh in matched
        }
        if unknown:
            raise ConfigError(
                f"所选工况下没有网格 {sorted(unknown)}; "
                f"可选: {sorted({f'{c.mesh} ({c.mesh_type})' for c in selected})}"
            )
        selected = [c for c in selected if c.mesh in matched]
    return tuple(selected)


if __name__ == "__main__":
    fig, all_cases = load_cases()
    print(f"成功加载 figure: {fig.get('id')} ({len(all_cases)} cases)")
    for c in all_cases:
        print(f"  - [{c.panel}] {c.ref} -> {c.artifact}")
