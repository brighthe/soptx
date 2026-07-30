"""Filesystem layout and artifact-naming contract.

``validate.py`` writes the raw run artifacts and ``sync_results.py`` reads them
back to build committed evidence, so both sides must agree on every directory
and file name. Those names are derived here from a single case table instead of
being spelled out on each side.

Like :mod:`contract`, this module must not import FEALPy, SOPTX or mpi4py.
"""

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

# (role, operator_level, ranks) for every case validate.py runs per dimension.
VALIDATION_CASES = (
    ("coarse", "ea", 1),
    ("medium", "ea", 1),
    ("fine", "ea", 1),
    ("fine", "ea", 2),
    ("coarse", "fa", 1),
)

# EA roles whose summaries feed the committed evidence, in refinement order.
EA_EVIDENCE_ROLES = ("coarse", "medium", "fine")

EVIDENCE_SCOPE = "cpu-single-rank-fa-ea"


def case_name(role: str, operator_level: str, ranks: int) -> str:
    """Canonical artifact stem for one validation case."""

    return f"{operator_level}-{role}-{ranks}rank"


def dimension_output_dir(dimension: int) -> Path:
    return OUTPUT_DIR / f"{dimension}d"


def validation_artifact_paths(
    dimension: int,
    name: str,
) -> tuple[Path, Path, Path]:
    """Summary, solution and VTU paths for one validation case."""

    directory = dimension_output_dir(dimension)
    summary = directory / f"{name}.json"
    return summary, summary.with_suffix(".npy"), directory / f"{name}.vtu"


def validation_case_specs(
    refinements: tuple[int, int, int],
) -> tuple[tuple[str, int, int, str], ...]:
    """Pair every validation case with its refinement level."""

    coarse, medium, fine = refinements
    refinement_by_role = {
        "coarse": coarse,
        "medium": medium,
        "fine": fine,
    }
    return tuple(
        (
            case_name(role, operator_level, ranks),
            refinement_by_role[role],
            ranks,
            operator_level,
        )
        for role, operator_level, ranks in VALIDATION_CASES
    )


def ea_evidence_sources(dimension: int) -> tuple[tuple[str, Path], ...]:
    """Single-rank EA summaries consumed by the evidence builder."""

    return tuple(
        (
            role,
            validation_artifact_paths(
                dimension,
                case_name(role, "ea", 1),
            )[0],
        )
        for role in EA_EVIDENCE_ROLES
    )


def fa_evidence_source(dimension: int) -> Path:
    """Single-rank coarse FA summary consumed by the evidence builder."""

    return validation_artifact_paths(
        dimension,
        case_name("coarse", "fa", 1),
    )[0]


def run_artifact_path(
    suffix: str,
    *,
    dimension: int,
    operator_level: str,
    degree: int,
    resolution: tuple[int, ...],
    mpi_size: int,
) -> Path:
    """Default artifact path for a manually launched single run."""

    grid = "x".join(str(value) for value in resolution)
    return OUTPUT_DIR / (
        f"elasticity-{dimension}d-{operator_level}-p{degree}-{grid}-"
        f"{mpi_size}ranks.{suffix}"
    )


def validation_evidence_path(dim_argument: str) -> Path:
    """Cross-run evidence written by validate.py for ``--dim``."""

    return OUTPUT_DIR / f"stage1-validation-{dim_argument}.json"


def evidence_path(dimension: int) -> Path:
    """Committed per-dimension evidence written by sync_results.py."""

    return EVIDENCE_DIR / f"{EVIDENCE_SCOPE}-{dimension}d.json"


def readme_markers(dimension: int) -> tuple[str, str]:
    key = f"{EVIDENCE_SCOPE}-{dimension}d"
    return (
        f"<!-- BEGIN GENERATED: {key} -->",
        f"<!-- END GENERATED: {key} -->",
    )


def relative_to_example(path: Path) -> str:
    return path.relative_to(EXAMPLE_DIR).as_posix()
