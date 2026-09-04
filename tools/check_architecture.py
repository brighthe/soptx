"""Fail when the stable SOPTX layers contain a reverse dependency."""

from __future__ import annotations

import ast
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "soptx"

LAYER = {
    "core": 0,
    "ml": 0,
    "protocols": 0,
    "materials": 1,
    "problems": 1,
    "fem": 2,
    "topology": 3,
    "postprocess": 4,
}
LEGACY_ROOTS = {
    "analysis",
    "demo",
    "functionspace",
    "interpolation",
    "model",
    "old",
    "optimization",
    "regularization",
    "tests",
    "utils",
}

# 主题目录隔离: examples/ 与 experiments/ 的各主题目录不得互相 import.
TOPIC_PARENTS = ("examples", "experiments")
TOPIC_SKIP_PARTS = {"__pycache__", "legacy", "old"}
# 已声明的外部依赖根, 不参与同名模块的越界判定.
THIRD_PARTY_ROOTS = {
    "soptx",
    "fealpy",
    "numpy",
    "scipy",
    "sympy",
    "matplotlib",
    "PIL",
    "mpi4py",
    "torch",
    "pytest",
}


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
    """主题目录之间不得互相 import: 检查跨主题的模块名越界."""
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
                        # 包式 import 允许指向本主题自身, 禁止指向其他主题.
                        if len(parts) < 2 or (
                            parts[0] == parent and parts[1] == topic.name
                        ):
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


def main() -> int:
    errors: list[str] = []
    errors.extend(topic_isolation_errors())
    for source_root, source_rank in LAYER.items():
        directory = PACKAGE_ROOT / source_root
        for path in sorted(directory.rglob("*.py")):
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
