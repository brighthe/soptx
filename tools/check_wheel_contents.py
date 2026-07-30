"""Verify that a built wheel contains only installable package code."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import sys
from zipfile import ZipFile


BANNED_PARTS = {
    "demo",
    "examples",
    "experiments",
    "old",
    "reference_code",
    "tests",
}
BANNED_FILE_SUFFIXES = ("_backup.py", "_old.py")


def main(arguments: list[str]) -> int:
    if len(arguments) != 1:
        print(
            "usage: python tools/check_wheel_contents.py DIST.whl",
            file=sys.stderr,
        )
        return 2
    wheel = Path(arguments[0])
    with ZipFile(wheel) as archive:
        names = archive.namelist()
    violations = [
        name
        for name in names
        if (
            BANNED_PARTS.intersection(PurePosixPath(name).parts)
            or PurePosixPath(name).name.endswith(BANNED_FILE_SUFFIXES)
        )
    ]
    if violations:
        print("wheel contains excluded content:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    if not any(name == "soptx/__init__.py" for name in names):
        print("wheel does not contain soptx/__init__.py", file=sys.stderr)
        return 1
    print(f"Wheel content check passed: {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
