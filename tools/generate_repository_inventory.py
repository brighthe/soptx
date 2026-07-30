"""Generate deterministic SHA-256 inventories for migration governance."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPOSITORY_ROOT / "reference_code"
REFERENCE_MANIFEST = (
    REPOSITORY_ROOT
    / "docs"
    / "references"
    / "reference-code-manifest.sha256"
)
PYTHON_MANIFEST = (
    REPOSITORY_ROOT
    / "docs"
    / "architecture"
    / "current-python-files.sha256"
)
PYTHON_ROOTS = ("src", "tests", "examples", "experiments", "tools")
EXCLUDED_DIRECTORY_NAMES = {
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "outputs",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def manifest(paths: list[Path]) -> str:
    lines = []
    for path in sorted(paths, key=lambda item: item.as_posix().lower()):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        lines.append(f"{digest(path)}  {relative}")
    return "\n".join(lines) + "\n"


def python_files_under(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, directory_names, file_names in os.walk(root):
        directory_names[:] = [
            name
            for name in directory_names
            if name not in EXCLUDED_DIRECTORY_NAMES
        ]
        current = Path(directory)
        paths.extend(
            current / name
            for name in file_names
            if name.endswith(".py")
        )
    return paths


def generated_payloads() -> dict[Path, str]:
    reference_files = [
        path for path in REFERENCE_ROOT.rglob("*") if path.is_file()
    ]
    python_files: list[Path] = []
    for root_name in PYTHON_ROOTS:
        root = REPOSITORY_ROOT / root_name
        if root.exists():
            python_files.extend(python_files_under(root))
    return {
        REFERENCE_MANIFEST: manifest(reference_files),
        PYTHON_MANIFEST: manifest(python_files),
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed manifests differ from generated content",
    )
    options = parser.parse_args(arguments)

    stale: list[Path] = []
    for path, content in generated_payloads().items():
        if options.check:
            if not path.exists() or path.read_text(
                encoding="utf-8"
            ) != content:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")

    if stale:
        for path in stale:
            print(
                f"stale inventory: "
                f"{path.relative_to(REPOSITORY_ROOT).as_posix()}",
                file=sys.stderr,
            )
        return 1
    action = "checked" if options.check else "generated"
    print(f"Repository SHA-256 inventories {action}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
