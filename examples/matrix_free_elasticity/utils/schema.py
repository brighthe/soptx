"""Structural contract for the Matrix-Free elasticity run summary.

``contract`` owns every number and ``layout`` owns every path; this module owns
the *shape* of the JSON summary that ``run.py`` writes and that ``validate.py``
and ``sync_results.py`` read back. Before it existed the three sides each knew
the field names independently, so renaming one key failed silently on the other
two while ``SCHEMA_VERSION`` kept claiming the format was unchanged.

Like :mod:`contract` and :mod:`layout`, this module must stay free of FEALPy,
SOPTX and mpi4py imports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


#: Bumped to 3 when ``local_gates`` values changed from ``bool`` to the
#: ``GATE_PASSED``/``GATE_SKIPPED`` vocabulary below.
SCHEMA_VERSION = 3

#: A gate that ran and succeeded.
GATE_PASSED = "passed"

#: A gate that ran and failed.
GATE_FAILED = "failed"

#: A gate whose inputs do not exist for this run mode, so it never ran. Only
#: benchmark and multi-rank runs produce it: both skip the serial EA/FA
#: references that the MatVec, symmetry and explicit-solution gates consume.
GATE_SKIPPED = "skipped"

GATE_STATUSES = (GATE_PASSED, GATE_FAILED, GATE_SKIPPED)


def gate_status(passed: bool) -> str:
    """Status for a gate that actually ran."""

    return GATE_PASSED if passed else GATE_FAILED


@dataclass(frozen=True)
class RunResult:
    """Numeric outcome of one run, as produced by ``run.execute``.

    These fields are spliced into the summary top level by
    ``report.build_payload``, which is why they are named here once instead of
    being an anonymous dict passed between the two modules.
    """

    global_cells: int
    global_dofs: int
    operator: dict[str, Any]
    partition: dict[str, Any]
    solver: dict[str, Any]
    timing: dict[str, Any] | None
    error: dict[str, Any]
    matvec_reference: dict[str, Any] | None
    explicit_solution_reference: dict[str, Any] | None

    def as_payload_fields(self) -> dict[str, Any]:
        """Summary fields contributed by the run itself."""

        return asdict(self)


#: Field names ``RunResult`` contributes to the summary top level.
RESULT_FIELDS = tuple(RunResult.__dataclass_fields__)

#: Fields ``report.build_payload`` adds around the ``RunResult`` ones.
ENVELOPE_FIELDS = (
    "schema_version",
    "stage",
    "command",
    "environment",
    "parameters",
    "mpi_size",
    "verification",
    "artifacts",
    "local_gates",
    "local_passed",
)

#: Every top-level key a well-formed summary must carry.
SUMMARY_TOP_LEVEL_FIELDS = ENVELOPE_FIELDS + RESULT_FIELDS


def gates_passed(gates: dict[str, str]) -> bool:
    """Whether every gate that actually ran succeeded.

    Skipped gates are excluded rather than counted as passing, so
    ``local_passed`` means the same thing in benchmark and full runs.
    """

    if not gates:
        return False
    unknown = sorted(
        f"{name}={status!r}"
        for name, status in gates.items()
        if status not in GATE_STATUSES
    )
    if unknown:
        # Schema 2 wrote booleans here, so an old summary lands in this
        # branch rather than being silently misread as all-passing.
        raise ValueError(
            "unknown gate status: " + ", ".join(unknown)
        )
    executed = [
        status for status in gates.values() if status != GATE_SKIPPED
    ]
    return bool(executed) and all(
        status == GATE_PASSED for status in executed
    )


def missing_summary_fields(payload: dict[str, Any]) -> list[str]:
    """Top-level keys absent from ``payload``, in contract order."""

    return [
        field
        for field in SUMMARY_TOP_LEVEL_FIELDS
        if field not in payload
    ]
