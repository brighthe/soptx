"""Fail when the stable SOPTX layers contain a reverse dependency."""

from __future__ import annotations

import ast
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "soptx"

LAYER = {
    "core": 0,
    "materials": 1,
    "problems": 1,
    "fem": 2,
    "topology": 3,
    "visualization": 4,
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


def main() -> int:
    errors: list[str] = []
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
