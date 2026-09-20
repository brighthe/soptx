"""Hu--Zhang 拓扑优化投稿论文实验的后处理入口.

``run.py`` 负责执行算例, 把运行产物写进 ``outputs/<case>/<run>/``; 本文件只作用在
已有产物上, 把它们整理成论文里的图与数字::

    python compare.py --list                      # 列出产物 case
    python compare.py --case compliance-topology  # 整理出一件产物
    python compare.py table
    python compare.py export [--check]
    python compare.py gradients
    python compare.py metrics
    python compare.py bearing-reanalysis
    python compare.py stress-cross-eval

一件产物 = 一条 ``--case``, 与 ``run.py --case`` 同一个词: 那边一条 case 是一道要解的
题, 这边一条 case 是一件要整理出来的产物。产物 case 不另立注册表, 由 ``plots/`` 下声明
了 ``SOURCE_CASE`` 的模块自描述 (见 discover_cases) —— 图读哪个算例的哪几次运行, 本就
是绘图代码的事实, 存第二份必然漂移。case id 取模块文件名, 论文图号只留在各模块
docstring 首行的括注里, 排版改号不波及命令行。

其中 export/gradients/metrics 会按冻结设计重新组装并求解 (见 ``metrics.py``), 因此
不是纯读产物; 但它们的产出是论文校验数字而非运行产物, 不写 ``outputs/<case>/<run>/``,
故归在本入口而不是 ``run.py``. ``--help`` 用 ``[重分析]`` 标出这三个动词。
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import re
import sys

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from config import OUTPUT_DIR, bootstrap_source_path

bootstrap_source_path()

PLOTS_DIR = EXPERIMENT_DIR / "plots"


# 子命令 -> "模块:入口函数". metrics.py 一个模块承载三个动词, 故入口名显式给出.
COMMAND_MODULES: dict[str, str] = {
    "table": "report:main",
    "export": "metrics:run_export",
    "gradients": "metrics:run_gradient_check",
    "metrics": "metrics:run_frozen_metrics",
    "audit-final-stress": "metrics:run_audit_final_stress",
    "bearing-reanalysis": "bearing_reanalysis:run_bearing_reanalysis",
    "stress-cross-eval": "stress_cross_evaluation:run_stress_cross_evaluation",
    "discretization-probe": "discretization_probe:run_discretization_probe",
}

# 需要把子命令之后的参数透传下去的模块 (其余不接受参数)
FORWARDS_ARGV = {"export", "stress-cross-eval", "discretization-probe"}

# 会按冻结设计重新组装并求解的动词: 比纯读产物慢, --help 里标出来免得误当作秒回
REANALYSIS = {
    "export", "gradients", "metrics", "audit-final-stress",
    "bearing-reanalysis", "stress-cross-eval", "discretization-probe",
}

# 动词的说明; 产物 case 的说明取自各 plots 模块自己的 docstring, 不在此重复
DESCRIPTIONS: dict[str, str] = {
    "table": "由 summary.json 重算论文表 5.1 / 5.2",
    "export": "冻结重分析导出插图场数据 (npz)",
    "gradients": "伴随灵敏度的有限差分校验",
    "metrics": "冻结设计的论文口径指标复算",
    "audit-final-stress": "核查两组 k=2 最终密度的实际约束, 不覆盖结果",
    "bearing-reanalysis": "轴承算例冻结设计交叉再分析与 nu 扫描 (论文表 5.3 / 5.4)",
    "stress-cross-eval": "应力算例: 一份构型 x 七条离散的应力比与可行性余量交叉表",
    "discretization-probe": "应力算例: 冻结构型的离散敏感性探针 (散布/采样/牵引跳量)",
}


@dataclass(frozen=True)
class ProductCase:
    """一件可整理出来的产物: id, 来源算例, 依赖的产物, 一句话说明.

    ``source_cases`` 是元组而非单值: 图 5.5 要把可压缩基准组与近不可压实验组并排,
    两者按 cases.toml 的口径是两条 case (nu 属 A 问题层), 故一件产物可以跨 case。
    """

    id: str
    module: str
    source_cases: tuple[str, ...]
    required_runs: tuple[str, ...]
    summary: str

    def resolve(self, run: str) -> Path:
        """把 REQUIRED_RUNS 的一项解析成 outputs 下的绝对路径.

        首段是本产物的某条 source case 时按 ``<case>/<产物>`` 读全 —— 跨 case 的产物
        只能这么写; 否则整项都是 case 内的相对路径, 拼到唯一的 source case 上, 已迁
        的单 case 模块因此一个字都不用改。不按有没有 ``/`` 判断: 应力三图依赖的是
        ``postprocess/fig_data_*.npz``, 带 ``/`` 却仍是 case 内路径。
        """
        head, _, rest = run.partition("/")
        if rest and head in self.source_cases:
            return OUTPUT_DIR / head / rest
        return OUTPUT_DIR / self.source_cases[0] / run

    def missing_runs(self) -> tuple[str, ...]:
        """依赖里还没落盘的产物; 空元组表示可以直接整理.

        用 exists 而不是 is_dir: 依赖不都是运行目录, 三张应力图吃的是 postprocess/
        下的 npz 文件 (由 compare.py export 冻结重分析导出)。
        """
        return tuple(
            run for run in self.required_runs if not self.resolve(run).exists()
        )

    def run_commands(self) -> tuple[str, ...]:
        """补齐**尚缺**依赖要跑的命令; 已经齐的 source case 不出现在这里.

        --full 的判据仍看该 case 在 REQUIRED_RUNS 里的**全部**依赖数而不是缺失数:
        裸跑一条 case 只出注册表的缺省单条组合 (见 run.py 的 registered_defaults),
        缺的那条未必就是缺省的那条, 只剩一个缺口时照样要靠这个旗标补上。

        缺失里带扩展名的那些 (postprocess/*.npz) 不是 run.py 的落盘产物, 而是
        ``compare.py export`` 从 density_final.vtu 冻结重分析导出的, 故补一条 export;
        它同样要先有运行目录, 所以排在 run.py 之后。
        """
        missing = self.missing_runs()
        commands = []
        for case in self.source_cases:
            root = OUTPUT_DIR / case
            if not any(self.resolve(run).is_relative_to(root) for run in missing):
                continue
            total = sum(
                1
                for run in self.required_runs
                if self.resolve(run).is_relative_to(root)
            )
            suffix = " --full" if total > 1 else ""
            commands.append(f"run.py --case {case}{suffix}")
        if any(self.resolve(run).suffix for run in missing):
            commands.append("compare.py export")
        return tuple(commands)


def module_literal(tree: ast.Module, name: str):
    """取模块顶层的字面量常量; 没有这个名字或它不是字面量都返回 None."""
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        try:
            return ast.literal_eval(node.value)
        except ValueError:
            return None
    return None


def discover_cases() -> dict[str, ProductCase]:
    """扫 plots/ 下声明了 SOURCE_CASE 的模块, 每个即一条产物 case.

    元数据长在模块自己身上 (docstring 首行 + SOURCE_CASE / REQUIRED_RUNS 两个字面量
    常量), 于是没有第二份注册表可漂移。同样用 ast 静态解析而不 import: --list 只是
    列清单, 不该为此加载 matplotlib。没声明 SOURCE_CASE 的模块不算产物, 直接跳过。
    """
    cases: dict[str, ProductCase] = {}
    for path in sorted(PLOTS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            # utf-8-sig: 容忍 BOM, 免得带 BOM 的模块在 ast.parse 处报 U+FEFF
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (OSError, SyntaxError):
            continue
        source_case = module_literal(tree, "SOURCE_CASE")
        # 单值是跨 case 写法的简写: 一件产物多数只吃一条 case, 让那些模块照旧写字符串
        if isinstance(source_case, str):
            source_cases: tuple[str, ...] = (source_case,)
        elif isinstance(source_case, (tuple, list)) and all(
            isinstance(item, str) for item in source_case
        ):
            source_cases = tuple(source_case)
        else:
            continue
        module_name = f"plots.{path.stem}"
        identifier = path.stem.replace("_", "-")
        cases[identifier] = ProductCase(
            id=identifier,
            module=module_name,
            source_cases=source_cases,
            required_runs=tuple(module_literal(tree, "REQUIRED_RUNS") or ()),
            summary=figure_summary(module_name),
        )
    return cases


def command_help(command: str) -> str:
    """子命令的一句话说明; 重分析动词标出来, 免得被误当作秒回的读产物动作."""
    return ("[重分析] " if command in REANALYSIS else "") + DESCRIPTIONS[command]


def command_entry(command: str):
    """按 COMMAND_MODULES 取出子命令的入口函数."""
    module_name, _, function_name = COMMAND_MODULES[command].partition(":")
    return getattr(import_module(module_name), function_name)


def module_path(module_name: str) -> str:
    """把 ``plots.compliance_topology`` 这类模块名还原成相对路径, 供 --list 显示."""
    return module_name.replace(".", "/") + ".py"


def figure_summary(module_name: str) -> str:
    """取 plots 模块 docstring 首行作为说明.

    用 ast 静态解析而不 import: --list 只是列清单, 不该为此加载 matplotlib。
    迁移后的模块首行形如 ``<正文> (论文图 5.2)``, 未迁移的仍是 ``生成论文图 5.2:
    <正文>``; 两种写法的图号都不该进 --list, 分别由下面的冒号切分与括注剥除去掉。
    """
    path = EXPERIMENT_DIR / module_path(module_name)
    try:
        # utf-8-sig: 容忍 BOM, 免得带 BOM 的模块在 ast.parse 处报 U+FEFF
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError):
        return "(读不到 docstring)"
    doc = ast.get_docstring(tree) or ""
    if not doc:
        return "(无 docstring)"
    first = doc.splitlines()[0]
    if ":" in first:
        first = first.split(":", 1)[1]
    first = first.strip().removeprefix("生成")
    # 去掉句末的括注 (排版说明, 或与 command 列重复的图号) 与句号, 只留一句话正文
    return re.sub(r"\s*\([^()]*\)\s*[.。]?\s*$", "", first).rstrip(".。")


def list_targets() -> int:
    """列出全部产物 case.

    只列 case, 与 ``run.py --list`` 同一个口径。动词与尚未迁移的图号由 argparse 的
    ``--help`` 逐条列出, 在这里重印一遍只是同一份信息的第二个出口。
    """
    cases = discover_cases()
    header = ("case-id", "source-case", "说明")
    rows = [
        (case.id, " ".join(case.source_cases), case.summary)
        for case in cases.values()
    ]
    # 说明列在最末, 无需补齐; 前两列全是 ASCII, len() 即显示宽度, 直接 ljust
    widths = [max(len(row[i]) for row in (header, *rows)) for i in range(2)]
    for row in (header, *rows):
        print(f"{row[0].ljust(widths[0])}  {row[1].ljust(widths[1])}  {row[2]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="列出产物 case")
    parser.add_argument(
        "--case", metavar="<case-id>", help="整理出一件产物, 如 compliance-topology"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for command in COMMAND_MODULES:
        # export 自己解析后续参数 (--check), 关掉这一层的 -h 免得被抢先接管
        sub.add_parser(
            command,
            add_help=command not in FORWARDS_ARGV,
            help=command_help(command),
        )
    return parser


def run_case(identifier: str) -> int:
    """整理出一件产物; 依赖缺目录时先报缺哪几个, 不进到绘图里才炸."""
    cases = discover_cases()
    case = cases.get(identifier)
    if case is None:
        # 算例 id 与产物 id 是两套命名, 敲混了直接指路, 而不是甩一句未知 id
        related = sorted(
            item.id for item in cases.values() if identifier in item.source_cases
        )
        if related:
            print(
                f"{identifier} 是算例 id (run.py 那边的); 它的产物 case: "
                + "  ".join(related),
                file=sys.stderr,
            )
        else:
            print(f"未知产物 case: {identifier}", file=sys.stderr)
            print("可用: " + ("  ".join(sorted(cases)) or "(无)"), file=sys.stderr)
        return 1
    # 吃 postprocess/ npz 的 case 只准备自己那几组, 不触发其它阶次的批量导出.
    EXPORTS_BY_CASE = {
        "stress-topologies": ["lfem-k2", "huzhang-k2"],
        "stress-cubic-convergence": ["lfem-k3", "huzhang-k3"],
    }
    if identifier in EXPORTS_BY_CASE:
        from metrics import prepare_exports
        try:
            prepare_exports(EXPORTS_BY_CASE[identifier])
        except FileNotFoundError as error:
            print(str(error), file=sys.stderr)
            return 1
    missing = case.missing_runs()
    if missing:
        print(f"{identifier} 缺 {len(missing)} 个产物:", file=sys.stderr)
        for run in missing:
            print(f"  {case.resolve(run).relative_to(OUTPUT_DIR.parent)}", file=sys.stderr)
        print("请先运行:", file=sys.stderr)
        for command in case.run_commands():
            print(f"  {command}", file=sys.stderr)
        return 1
    print(f"[case] {identifier} -> {case.module}")
    import_module(case.module).main()
    return 0


def main() -> int:
    parser = build_parser()
    argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        return 1

    command = argv[0]
    if command in ("-h", "--help"):
        parser.print_help()
        return 0
    if command in FORWARDS_ARGV:
        return command_entry(command)(argv[1:]) or 0

    arguments = parser.parse_args(argv)
    if arguments.list:
        return list_targets()
    if arguments.case:
        return run_case(arguments.case)
    if arguments.command in COMMAND_MODULES:
        return command_entry(arguments.command)() or 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
