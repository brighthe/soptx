"""Parse maintained and compatibility Python sources without importing them."""

from __future__ import annotations

import ast
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOTS = ("src", "tests", "examples", "tools")
EXCLUDED_PARTS = {
    "__pycache__",
    "demo",
    "old",
    "tests",  # package-internal pre-v2 scripts under src/soptx/tests
}
EXCLUDED_SUFFIXES = ("_old.py", "_backup.py")


def is_maintained(path: Path) -> bool:
    relative = path.relative_to(REPOSITORY_ROOT)
    if relative.parts[0] == "src" and EXCLUDED_PARTS.intersection(
        relative.parts[2:]
    ):
        return False
    return not path.name.endswith(EXCLUDED_SUFFIXES)


def main() -> int:
    errors: list[str] = []
    count = 0
    for root_name in ROOTS:
        root = REPOSITORY_ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if not is_maintained(path):
                continue
            count += 1
            try:
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                )
            except (SyntaxError, UnicodeError) as error:
                relative = path.relative_to(REPOSITORY_ROOT).as_posix()
                errors.append(f"{relative}: {error}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Parsed {count} maintained Python files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
