"""Validate the Hu–Zhang legacy experiment inventory without execution."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tomllib
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_MATRIX = EXPERIMENT_ROOT / "matrix.toml"
EXPECTED_STAGE = "soptx/huzhang-topopt-paper/inventory-v1"


class InventoryError(RuntimeError):
    """Raised when the static experiment inventory is not self-consistent."""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Hu–Zhang legacy driver matrix without importing "
            "or executing any driver."
        )
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the dry-run summary as JSON.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def resolved_experiment_path(value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise InventoryError(f"unsafe relative path: {value!r}")
    path = (EXPERIMENT_ROOT / Path(*relative.parts)).resolve()
    try:
        path.relative_to(EXPERIMENT_ROOT)
    except ValueError as error:
        raise InventoryError(
            f"path escapes experiment root: {value!r}"
        ) from error
    return path


def selector_records(path: Path) -> set[tuple[str, str, int]]:
    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )
    records: set[tuple[str, str, int]] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for method in node.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            if method.name != "run":
                continue
            for decorator in method.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                if not decorator.args:
                    continue
                argument = decorator.args[0]
                if not (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                ):
                    continue
                records.add(
                    (argument.value, node.name, method.lineno)
                )
    return records


def require_list(
    payload: dict[str, Any],
    key: str,
) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise InventoryError(f"{key!r} must be an array of tables")
    return value


def load_matrix(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise InventoryError(f"cannot read matrix {path}: {error}") from error


def validate_matrix(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise InventoryError("expected schema_version=1")
    if payload.get("stage") != EXPECTED_STAGE:
        raise InventoryError(f"expected stage={EXPECTED_STAGE!r}")
    if payload.get("status") != "incubating":
        raise InventoryError("inventory status must remain 'incubating'")
    blockers = payload.get("default_blockers")
    if not isinstance(blockers, list) or not blockers:
        raise InventoryError("default_blockers must not be empty")

    sources = require_list(payload, "sources")
    excluded = require_list(payload, "excluded_sources")
    cases = require_list(payload, "cases")
    declared_sources: dict[str, Path] = {}
    discovered: set[tuple[str, str]] = set()
    source_hashes: dict[str, str] = {}

    for source in sources:
        relative = source.get("path")
        expected_hash = source.get("sha256")
        if not isinstance(relative, str) or not isinstance(
            expected_hash, str
        ):
            raise InventoryError("source path and sha256 must be strings")
        if relative in declared_sources:
            raise InventoryError(f"duplicate source: {relative}")
        path = resolved_experiment_path(relative)
        if not path.is_file():
            raise InventoryError(f"missing source: {relative}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise InventoryError(
                f"source hash drift: {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        declared_sources[relative] = path
        source_hashes[relative] = actual_hash
        discovered.update(
            (relative, selector)
            for selector, _, _ in selector_records(path)
        )

    excluded_paths: set[str] = set()
    for item in excluded:
        relative = item.get("path")
        reason = item.get("reason")
        if not isinstance(relative, str) or not isinstance(reason, str):
            raise InventoryError(
                "excluded source path and reason must be strings"
            )
        path = resolved_experiment_path(relative)
        if not path.is_file():
            raise InventoryError(f"missing excluded source: {relative}")
        excluded_paths.add(relative)

    actual_python_sources = {
        path.relative_to(EXPERIMENT_ROOT).as_posix()
        for path in (EXPERIMENT_ROOT / "legacy_drivers").glob("*.py")
    }
    accounted = set(declared_sources) | excluded_paths
    if actual_python_sources != accounted:
        raise InventoryError(
            "legacy source accounting mismatch: "
            f"missing={sorted(actual_python_sources - accounted)}, "
            f"extra={sorted(accounted - actual_python_sources)}"
        )

    ids: set[str] = set()
    outputs: set[str] = set()
    declared_cases: set[tuple[str, str]] = set()
    for case in cases:
        identifier = case.get("id")
        source = case.get("source")
        selector = case.get("selector")
        output = case.get("output_stem")
        if not all(
            isinstance(value, str)
            for value in (identifier, source, selector, output)
        ):
            raise InventoryError(
                "case id/source/selector/output_stem must be strings"
            )
        if identifier in ids:
            raise InventoryError(f"duplicate case id: {identifier}")
        ids.add(identifier)
        if output in outputs:
            raise InventoryError(f"duplicate output_stem: {output}")
        outputs.add(output)
        resolved_experiment_path(output)
        if source not in declared_sources:
            raise InventoryError(
                f"{identifier}: undeclared source {source!r}"
            )
        declared_cases.add((source, selector))
        if case.get("chapter") not in (3, 4, 5, 6):
            raise InventoryError(f"{identifier}: invalid chapter")
        if case.get("dimension") not in (2, 3):
            raise InventoryError(f"{identifier}: invalid dimension")
        if case.get("role") not in (
            "analysis-baseline",
            "optimization",
            "postprocess",
        ):
            raise InventoryError(f"{identifier}: invalid role")
        if not isinstance(case.get("method"), str):
            raise InventoryError(f"{identifier}: missing method")
        if case.get("execution_status") != "blocked":
            raise InventoryError(
                f"{identifier}: unvalidated case must remain blocked"
            )

    if declared_cases != discovered:
        raise InventoryError(
            "case matrix does not exactly cover driver selectors: "
            f"missing={sorted(discovered - declared_cases)}, "
            f"extra={sorted(declared_cases - discovered)}"
        )

    status = git_value("status", "--porcelain")
    return {
        "schema_version": payload["schema_version"],
        "stage": payload["stage"],
        "status": payload["status"],
        "inventory_base_revision": payload[
            "inventory_base_revision"
        ],
        "runtime_git_revision": git_value("rev-parse", "HEAD"),
        "runtime_git_dirty": None if status is None else bool(status),
        "source_count": len(sources),
        "excluded_source_count": len(excluded),
        "case_count": len(cases),
        "case_count_by_chapter": dict(
            sorted(Counter(case["chapter"] for case in cases).items())
        ),
        "case_count_by_dimension": dict(
            sorted(Counter(case["dimension"] for case in cases).items())
        ),
        "case_count_by_role": dict(
            sorted(Counter(case["role"] for case in cases).items())
        ),
        "source_sha256": source_hashes,
        "drivers_imported": False,
        "drivers_executed": False,
        "default_blockers": blockers,
        "passed": True,
    }


def main() -> int:
    arguments = parse_arguments()
    try:
        matrix_path = arguments.matrix.resolve()
        payload = load_matrix(matrix_path)
        summary = validate_matrix(payload)
    except (InventoryError, OSError, SyntaxError) as error:
        print(f"Hu–Zhang inventory dry-run: FAILED: {error}", file=sys.stderr)
        return 1

    if arguments.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("Hu–Zhang inventory dry-run: PASSED")
        print(
            f"sources={summary['source_count']} "
            f"excluded={summary['excluded_source_count']} "
            f"cases={summary['case_count']}"
        )
        print(
            "drivers_imported=false drivers_executed=false "
            f"status={summary['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
