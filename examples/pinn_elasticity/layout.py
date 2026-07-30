"""Filesystem layout and artifact-naming contract for the PINN example."""

from __future__ import annotations

from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_DIR.parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
OUTPUT_DIR = EXAMPLE_DIR / "outputs"
EVIDENCE_DIR = EXAMPLE_DIR / "evidence"
README_PATH = EXAMPLE_DIR / "README.md"
RUN_SCRIPT = EXAMPLE_DIR / "run.py"
SYNC_SCRIPT_NAME = "sync_results.py"

EVIDENCE_SCOPE = "cpu-float64-training-baseline"


def validation_summary_path(dimension: int) -> Path:
    return OUTPUT_DIR / f"stage1-validation-{dimension}.json"


def aggregate_validation_path(dim_argument: str) -> Path:
    return OUTPUT_DIR / f"stage1-validation-{dim_argument}.json"


def evidence_path(dimension: int) -> Path:
    return EVIDENCE_DIR / f"{EVIDENCE_SCOPE}-{dimension}d.json"


def readme_markers(dimension: int) -> tuple[str, str]:
    key = f"{EVIDENCE_SCOPE}-{dimension}d"
    return (
        f"<!-- BEGIN GENERATED: {key} -->",
        f"<!-- END GENERATED: {key} -->",
    )


def relative_to_example(path: Path) -> str:
    return path.relative_to(EXAMPLE_DIR).as_posix()
