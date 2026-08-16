"""Filesystem layout and artifact-naming contract.

``validate.py`` writes the raw run artifacts and ``sync_results.py`` reads them
back to build committed evidence, so both sides must agree on every directory
and file name. Those names are derived here from a single case table instead of
being spelled out on each side.

This harness lives outside the example it validates, so every path below is
anchored on the repository root rather than on ``__file__``'s own parent.

Like the example's ``contract`` module, this one must not import FEALPy, SOPTX
or mpi4py.
"""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
TOOL_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = REPOSITORY_ROOT / "examples" / "matrix_free_elasticity"
OUTPUT_DIR = EXAMPLE_DIR / "outputs"
EVIDENCE_DIR = EXAMPLE_DIR / "evidence"
RUN_SCRIPT = TOOL_DIR / "run.py"

# (role, operator_level, ranks) per dimension, split by推进阶段。
#
# 1a（CPU 串行 EA/FA）是当前的默认验证范围: 三档单 rank EA 加一档单 rank FA
# 参照, 不涉及任何 MPI 分区与重叠副本归约。
SERIAL_VALIDATION_CASES = (
    ("coarse", "ea", 1),
    ("medium", "ea", 1),
    ("fine", "ea", 1),
    ("coarse", "fa", 1),
)

# 1b（CPU 并行 EA）在 1a 之上追加的算例; 由 validate.py 的
# ``--include-parallel`` 打开, 对应 1/2-rank 一致性门禁。
PARALLEL_VALIDATION_CASES = (
    ("fine", "ea", 2),
)


def validation_cases(
    *,
    include_parallel: bool = False,
) -> tuple[tuple[str, str, int], ...]:
    """当前验证范围内的算例表。

    默认只返回 1a 的串行算例; ``include_parallel`` 追加 1b 的多 rank 算例。
    两个阶段共享同一份算例定义与同一份阈值, 因此 1b 的跨 rank 门禁始终与 1a
    的串行结果可比。
    """

    if include_parallel:
        return SERIAL_VALIDATION_CASES + PARALLEL_VALIDATION_CASES
    return SERIAL_VALIDATION_CASES

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
    *,
    include_parallel: bool = False,
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
        for role, operator_level, ranks in validation_cases(
            include_parallel=include_parallel,
        )
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


def relative_to_example(path: Path) -> str:
    return path.relative_to(EXAMPLE_DIR).as_posix()
