from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tools.matrix_free_evidence import contract, layout, schema


class EvidenceError(RuntimeError):
    """Raised when raw result artifacts violate the evidence contract."""


FORMAL_ENVIRONMENT_KEYS = (
    "python",
    "platform",
    "numpy",
    "scipy",
    "sympy",
    "fealpy",
    "soptx",
    "mpi4py",
    "mpi_library",
    "git_revision",
    "git_dirty",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize dimension-specific CPU FA/EA evidence from "
            "ignored raw JSON outputs."
        )
    )
    parser.add_argument(
        "--dim",
        choices=("2", "3", "all"),
        default="all",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Check for drift without writing evidence."
        ),
    )
    return parser.parse_args()


def selected_dimensions(value: str) -> tuple[int, ...]:
    if value == "all":
        return contract.SUPPORTED_DIMENSIONS
    return (int(value),)


def read_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"required source artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(
            f"cannot read JSON artifact {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise EvidenceError(
            f"JSON artifact must contain an object: {path}"
        )
    return payload


def require_mapping(
    payload: dict[str, Any],
    key: str,
    source: Path,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise EvidenceError(
            f"{source}: missing object field {key!r}"
        )
    return value


def require_number(
    mapping: dict[str, Any],
    key: str,
    source: Path,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(
            f"{source}: missing numeric field {key!r}"
        )
    return float(value)


def require_positive_int(
    mapping: dict[str, Any],
    key: str,
    source: Path,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvidenceError(
            f"{source}: field {key!r} must be a positive integer"
        )
    return value


def require_local_gates(
    payload: dict[str, Any],
    source: Path,
) -> dict[str, str]:
    """Require every gate to have run and passed.

    Formal evidence does not accept ``GATE_SKIPPED``: a run that skipped the
    MatVec, positive-definiteness or explicit-solution comparisons cannot back
    a correctness claim, however green its remaining gates are.
    """

    gates = require_mapping(payload, "local_gates", source)
    if not gates:
        raise EvidenceError(f"{source}: local_gates must not be empty")
    skipped = sorted(
        name
        for name, value in gates.items()
        if value == schema.GATE_SKIPPED
    )
    if skipped:
        raise EvidenceError(
            f"{source}: formal evidence rejects skipped gates: "
            + ", ".join(skipped)
        )
    failed = sorted(
        name
        for name, value in gates.items()
        if value != schema.GATE_PASSED
    )
    if failed:
        raise EvidenceError(
            f"{source}: local gates did not all pass: "
            + ", ".join(failed)
        )
    return {str(name): schema.GATE_PASSED for name in gates}


def require_formal_environment(
    payload: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    environment = require_mapping(payload, "environment", source)
    if not environment.get("git_revision"):
        raise EvidenceError(f"{source}: missing git revision")
    if not environment.get("generated_at_utc"):
        raise EvidenceError(f"{source}: missing generated_at_utc")
    if environment.get("git_dirty") is not False:
        raise EvidenceError(
            f"{source}: formal evidence requires git_dirty=false, got "
            f"{environment.get('git_dirty')!r}. A dirty worktree cannot be "
            "replayed, so the run is development evidence only: commit or "
            "stash the worktree, re-run validate.py, then sync again."
        )
    missing = [
        key
        for key in FORMAL_ENVIRONMENT_KEYS
        if key not in environment
    ]
    if missing:
        raise EvidenceError(
            f"{source}: missing environment fields: "
            + ", ".join(missing)
        )
    return environment


def validate_source(
    payload: dict[str, Any],
    source: Path,
    *,
    dimension: int,
    operator_level: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if payload.get("schema_version") != schema.SCHEMA_VERSION:
        raise EvidenceError(
            f"{source}: expected schema_version="
            f"{schema.SCHEMA_VERSION}"
        )
    missing = schema.missing_summary_fields(payload)
    if missing:
        raise EvidenceError(
            f"{source}: summary is missing top-level fields: "
            + ", ".join(missing)
        )
    if payload.get("mpi_size") != 1:
        raise EvidenceError(f"{source}: expected mpi_size=1")
    if payload.get("local_passed") is not True:
        raise EvidenceError(
            f"{source}: expected local_passed=true"
        )
    parameters = require_mapping(payload, "parameters", source)
    if parameters.get("dimension") != dimension:
        raise EvidenceError(
            f"{source}: expected dimension={dimension}"
        )
    if parameters.get("operator_level") != operator_level:
        raise EvidenceError(
            f"{source}: expected operator_level={operator_level!r}"
        )
    if (
        parameters.get("reference_random_seed")
        != contract.REFERENCE_RANDOM_SEED
    ):
        raise EvidenceError(
            f"{source}: unexpected reference random seed"
        )
    environment = require_formal_environment(payload, source)
    return parameters, environment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(
    role: str,
    path: Path,
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": role,
        "path": layout.relative_to_example(path),
        "sha256": sha256(path),
        "generated_at_utc": environment["generated_at_utc"],
        "git_revision": environment["git_revision"],
        "git_dirty": environment["git_dirty"],
    }


def grid_label(parameters: dict[str, Any], source: Path) -> str:
    resolution = parameters.get("resolution")
    if (
        not isinstance(resolution, list)
        or not resolution
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in resolution
        )
    ):
        raise EvidenceError(
            f"{source}: resolution must be a list of positive integers"
        )
    return "×".join(str(value) for value in resolution)


def extract_ea_case(
    dimension: int,
    role: str,
    source: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = read_payload(source)
    parameters, environment = validate_source(
        payload,
        source,
        dimension=dimension,
        operator_level="ea",
    )
    solver = require_mapping(payload, "solver", source)
    error = require_mapping(payload, "error", source)
    matvec = require_mapping(payload, "matvec_reference", source)
    direct = require_mapping(
        payload,
        "explicit_solution_reference",
        source,
    )
    case = {
        "role": role,
        "grid": grid_label(parameters, source),
        "operator_level": "ea",
        "mpi_size": 1,
        "local_passed": True,
        "solver": {
            "iterations": require_positive_int(
                solver,
                "iterations",
                source,
            ),
            "true_relative_residual": require_number(
                solver,
                "true_relative_residual",
                source,
            ),
            "boundary_absolute_error": require_number(
                solver,
                "boundary_absolute_error",
                source,
            ),
        },
        "error": {
            "l2_relative": require_number(
                error,
                "l2_relative",
                source,
            ),
        },
        "fa_references": {
            "raw_matvec_relative_error": require_number(
                matvec,
                "raw_relative_error",
                source,
            ),
            "dirichlet_matvec_relative_error": require_number(
                matvec,
                "dirichlet_relative_error",
                source,
            ),
            "explicit_solution_relative_error": require_number(
                direct,
                "relative_error",
                source,
            ),
        },
        "local_gates": require_local_gates(payload, source),
    }
    return (
        case,
        source_record(
            f"{dimension}d-ea-{role}-with-fa-references",
            source,
            environment,
        ),
        environment,
    )


def extract_fa_case(
    dimension: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = layout.fa_evidence_source(dimension)
    payload = read_payload(source)
    parameters, environment = validate_source(
        payload,
        source,
        dimension=dimension,
        operator_level="fa",
    )
    solver = require_mapping(payload, "solver", source)
    error = require_mapping(payload, "error", source)
    case = {
        "role": "coarse",
        "grid": grid_label(parameters, source),
        "operator_level": "fa",
        "mpi_size": 1,
        "local_passed": True,
        "solver": {
            "iterations": require_positive_int(
                solver,
                "iterations",
                source,
            ),
            "true_relative_residual": require_number(
                solver,
                "true_relative_residual",
                source,
            ),
            "boundary_absolute_error": require_number(
                solver,
                "boundary_absolute_error",
                source,
            ),
        },
        "error": {
            "l2_relative": require_number(
                error,
                "l2_relative",
                source,
            ),
        },
        "local_gates": require_local_gates(payload, source),
    }
    return (
        case,
        source_record(
            f"{dimension}d-independent-fa-coarse",
            source,
            environment,
        ),
        environment,
    )


def build_evidence(dimension: int) -> dict[str, Any]:
    ea_cases: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    environments: list[dict[str, Any]] = []
    for role, source in layout.ea_evidence_sources(dimension):
        case, source_entry, environment = extract_ea_case(
            dimension,
            role,
            source,
        )
        ea_cases.append(case)
        sources.append(source_entry)
        environments.append(environment)

    independent_fa, fa_source, fa_environment = extract_fa_case(
        dimension
    )
    sources.append(fa_source)
    environments.append(fa_environment)
    formal_environment = {
        key: environments[0][key]
        for key in FORMAL_ENVIRONMENT_KEYS
    }
    for environment in environments[1:]:
        candidate = {
            key: environment[key]
            for key in FORMAL_ENVIRONMENT_KEYS
        }
        if candidate != formal_environment:
            raise EvidenceError(
                f"{dimension}d source environments are inconsistent"
            )
    l2_errors = [case["error"]["l2_relative"] for case in ea_cases]
    if not l2_errors[0] > l2_errors[1] > l2_errors[2]:
        raise EvidenceError(
            f"{dimension}d relative L2 errors must strictly decrease"
        )
    orders = [
        math.log2(l2_errors[0] / l2_errors[1]),
        math.log2(l2_errors[1] / l2_errors[2]),
    ]
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "dimension": dimension,
        "scope": f"{layout.EVIDENCE_SCOPE}-correctness",
        "environment": formal_environment,
        "source_artifacts": sources,
        "ea_cases": ea_cases,
        "observed_relative_l2_orders": orders,
        "independent_fa_case": independent_fa,
    }


def serialized_evidence(evidence: dict[str, Any]) -> str:
    return json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"


def check_file(path: Path, expected: str) -> bool:
    if not path.is_file():
        print(f"OUT OF DATE: missing {path}", file=sys.stderr)
        return False
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        print(f"OUT OF DATE: {path}", file=sys.stderr)
        return False
    return True


def main() -> int:
    arguments = parse_arguments()
    try:
        serialized: dict[int, str] = {}
        for dimension in selected_dimensions(arguments.dim):
            evidence = build_evidence(dimension)
            serialized[dimension] = serialized_evidence(evidence)

        if arguments.check:
            passed = True
            for dimension, evidence_text in serialized.items():
                passed = (
                    check_file(
                        layout.evidence_path(dimension),
                        evidence_text,
                    )
                    and passed
                )
            if passed:
                print(
                    "Dimension-specific FA/EA evidence is in sync."
                )
                return 0
            return 1

        layout.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        for dimension, evidence_text in serialized.items():
            path = layout.evidence_path(dimension)
            path.write_text(evidence_text, encoding="utf-8")
            print(f"Wrote evidence: {path}")
        return 0
    except (EvidenceError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
