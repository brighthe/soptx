"""Generate a deterministic SHA-256 inventory for reference-code governance.

``reference_code/`` is unredistributable and will leave ``main`` once the
``archive/pre-v2`` tag is authorised, so its manifest is the only record of
what those files contained.  Repository Python files are deliberately *not*
inventoried here: git already stores a content hash for every file at every
commit, so a second manifest would be redundant bookkeeping.
"""

from __future__ import annotations

import argparse
import hashlib
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


def generated_payloads() -> dict[Path, str]:
    reference_files = [
        path for path in REFERENCE_ROOT.rglob("*") if path.is_file()
    ]
    return {REFERENCE_MANIFEST: manifest(reference_files)}


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
    print(f"Reference-code SHA-256 inventory {action}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
