"""Fail when the stable SOPTX layers contain a reverse dependency.

Also fail when any maintained Python file imports ``fealpy``: the FEALPy code
SOPTX depends on has been ported into ``soptx`` (see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "soptx"

# backend / typing / decorator / sparse / quadrature / mesh / functionspace 移植自
# FEALPy (见 THIRD_PARTY_NOTICES.md); 同层互相导入允许, 只禁止导入更高层.
# ml 与 fem 同层: ml.substructure 为 fem.substructure 提供网络与训练组件, 二者互相导入.
LAYER = {
    "backend": 0,
    "core": 0,
    "decorator": 0,
    "protocols": 0,
    "quadrature": 0,
    "sparse": 0,
    "typing": 0,
    "functionspace": 1,
    "materials": 1,
    "mesh": 1,
    "problems": 1,
    "fem": 2,
    "ml": 2,
    "topology": 3,
    "postprocess": 4,
}
LEGACY_ROOTS = {
    "analysis",
    "demo",
    "interpolation",
    "model",
    "old",
    "optimization",
    "regularization",
    "tests",
    "utils",
}

# 主题目录隔离: 同一父目录下的各主题目录不得互相 import; 跨父目录只允许
# experiments/ 以包式 import 复用 examples/ 中已验证的流水线, 反向禁止.
TOPIC_PARENTS = ("examples", "experiments")
ALLOWED_CROSS_PARENT = {("experiments", "examples")}
TOPIC_SKIP_PARTS = {"__pycache__", "legacy", "old"}
# 已声明的外部依赖根, 不参与同名模块的越界判定.
THIRD_PARTY_ROOTS = {
    "soptx",
    "numpy",
    "scipy",
    "sympy",
    "matplotlib",
    "PIL",
    "mpi4py",
    "torch",
    "pytest",
    "vtk",
}

# 禁止导入 fealpy 的扫描范围.
FEALPY_SCAN_ROOTS = ("src", "tests", "examples", "experiments", "tools")
FEALPY_SKIP_PARTS = {"__pycache__"}


def imported_soptx_roots(
    tree: ast.AST,
    current_package: tuple[str, ...],
) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    names.append(node.module)
            else:
                parent_count = node.level - 1
                if parent_count > len(current_package):
                    continue
                base = current_package[: len(current_package) - parent_count]
                if node.module:
                    names.append(".".join((*base, *node.module.split("."))))
                else:
                    names.extend(
                        ".".join((*base, alias.name.split(".")[0]))
                        for alias in node.names
                    )
        for name in names:
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == "soptx":
                roots.add(parts[1])
    return roots


def absolute_imports(tree: ast.AST) -> set[str]:
    """收集绝对 import 的完整模块名."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module)
    return names


def topic_isolation_errors() -> list[str]:
    """主题目录之间不得互相 import: 检查跨主题的模块名越界.

    同一父目录下的主题互相禁止; ``experiments`` 可以包式 import ``examples``,
    ``examples`` 不得 import ``experiments``.
    """
    errors: list[str] = []
    for parent in TOPIC_PARENTS:
        base = REPOSITORY_ROOT / parent
        if not base.exists():
            continue
        topics = [
            entry
            for entry in sorted(base.iterdir())
            if entry.is_dir() and entry.name not in TOPIC_SKIP_PARTS
        ]
        modules_by_topic = {
            topic.name: {
                path.stem
                for path in topic.rglob("*.py")
                if TOPIC_SKIP_PARTS.isdisjoint(path.parts)
            }
            for topic in topics
        }
        for topic in topics:
            own_modules = modules_by_topic[topic.name]
            foreign_modules = {}
            for other_name, modules in modules_by_topic.items():
                if other_name == topic.name:
                    continue
                for module in modules - own_modules:
                    foreign_modules.setdefault(module, other_name)
            for path in sorted(topic.rglob("*.py")):
                if not TOPIC_SKIP_PARTS.isdisjoint(path.parts):
                    continue
                relative = path.relative_to(REPOSITORY_ROOT).as_posix()
                try:
                    tree = ast.parse(
                        path.read_text(encoding="utf-8"), filename=relative
                    )
                except SyntaxError:
                    continue
                for name in sorted(absolute_imports(tree)):
                    parts = name.split(".")
                    if parts[0] in TOPIC_PARENTS:
                        # 包式 import 允许指向本主题自身, 以及 ALLOWED_CROSS_PARENT
                        # 所列方向的另一父目录主题, 禁止指向同级其他主题.
                        if len(parts) < 2 or (
                            parts[0] == parent and parts[1] == topic.name
                        ) or (parent, parts[0]) in ALLOWED_CROSS_PARENT:
                            continue
                        errors.append(
                            f"{relative}: imports '{name}' from another "
                            "topic dir"
                        )
                    elif (
                        parts[0] in foreign_modules
                        and parts[0] not in THIRD_PARTY_ROOTS
                        and parts[0] not in sys.stdlib_module_names
                    ):
                        errors.append(
                            f"{relative}: imports module '{parts[0]}' that "
                            f"only exists in sibling topic "
                            f"{parent}/{foreign_modules[parts[0]]}"
                        )
    return errors


def _is_fealpy(name: str) -> bool:
    return name == "fealpy" or name.startswith("fealpy.")


def fealpy_import_errors() -> list[str]:
    """检查维护范围内的 Python 文件不再导入 fealpy.

    覆盖 ``import fealpy``、``from fealpy ... import`` 以及以字面量模块名调用的
    ``import_module("fealpy...")`` / ``__import__("fealpy...")``.

    Returns
    -------
    list of str
        违规位置与说明, 无违规时为空列表.
    """
    errors: list[str] = []
    for root in FEALPY_SCAN_ROOTS:
        base = REPOSITORY_ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if not FEALPY_SKIP_PARTS.isdisjoint(path.parts):
                continue
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Call):
                    func = node.func
                    called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                    if (called in {"import_module", "__import__"} and node.args
                            and isinstance(node.args[0], ast.Constant)
                            and isinstance(node.args[0].value, str)):
                        names = [node.args[0].value]
                for name in names:
                    if _is_fealpy(name):
                        errors.append(
                            f"{relative}:{node.lineno}: imports '{name}'; "
                            "fealpy code is ported into soptx"
                        )
    return errors


def _layer_files(source_root: str) -> list[Path]:
    """返回某个分层根 (子包目录或单文件模块) 下的全部 Python 文件."""
    directory = PACKAGE_ROOT / source_root
    if directory.is_dir():
        return sorted(directory.rglob("*.py"))
    module = PACKAGE_ROOT / f"{source_root}.py"
    return [module] if module.is_file() else []


def main() -> int:
    errors: list[str] = []
    errors.extend(topic_isolation_errors())
    errors.extend(fealpy_import_errors())
    for source_root, source_rank in LAYER.items():
        for path in _layer_files(source_root):
            relative_to_package = path.relative_to(PACKAGE_ROOT)
            current_package = (
                "soptx",
                *relative_to_package.with_suffix("").parts[:-1],
            )
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
            for target_root in imported_soptx_roots(tree, current_package):
                relative = path.relative_to(REPOSITORY_ROOT).as_posix()
                if target_root in LEGACY_ROOTS:
                    errors.append(
                        f"{relative}: stable layer imports legacy "
                        f"soptx.{target_root}"
                    )
                    continue
                target_rank = LAYER.get(target_root)
                if target_rank is not None and target_rank > source_rank:
                    errors.append(
                        f"{relative}: {source_root} imports higher layer "
                        f"{target_root}"
                    )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("SOPTX architecture dependency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
