"""Run route-neutral Hu–Zhang paper evidence cases.

The runner deliberately keeps the historical ``matrix.toml`` and legacy
drivers out of the execution path.  Numerical adapters import only current
SOPTX namespaces.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePath
import platform
import re
import subprocess
import sys
import tomllib
from typing import Any, Iterable


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "cases.toml"
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_STAGE = "soptx/huzhang-topopt-paper/common-evidence-v1"
CASE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
ALLOWED_EXECUTION_STATUS = {"ready", "configured"}
REQUIRED_ACCEPTANCE_KEYS = {
    "high_order_rate_deficit",
    "relative_equilibrium_residual_max",
    "normalized_normal_trace_jump_max",
    "finite_difference_relative_error_max",
    "volume_fraction_absolute_error_max",
    "stress_constraint_violation_max",
    "frozen_design_relative_change_max",
}

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


class ConfigurationError(RuntimeError):
    """Raised when the common evidence configuration is invalid."""


class CaseBlocked(RuntimeError):
    """Raised when a configured case has no native numerical adapter yet."""


@dataclass(frozen=True)
class CaseDefinition:
    """Validated case metadata passed to a numerical executor."""

    identifier: str
    role: str
    problem: str
    methods: tuple[str, ...]
    mesh_families: tuple[str, ...]
    orders: tuple[int, ...]
    levels: tuple[int, ...]
    metrics: tuple[str, ...]
    execution_status: str
    executor: str | None
    blockers: tuple[str, ...]
    parameters: dict[str, Any]
    expected_orders: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "role": self.role,
            "problem": self.problem,
            "methods": list(self.methods),
            "mesh_families": list(self.mesh_families),
            "orders": list(self.orders),
            "levels": list(self.levels),
            "metrics": list(self.metrics),
            "execution_status": self.execution_status,
            "executor": self.executor,
            "blockers": list(self.blockers),
            "parameters": self.parameters,
            "expected_orders": self.expected_orders,
        }


@dataclass(frozen=True)
class ExperimentConfiguration:
    """Validated top-level common evidence configuration."""

    path: Path
    stage: str
    status: str
    provenance: dict[str, Any]
    acceptance: dict[str, float]
    outputs: dict[str, str]
    cases: tuple[CaseDefinition, ...]

    def by_id(self) -> dict[str, CaseDefinition]:
        return {case.identifier: case for case in self.cases}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or validate the Hu–Zhang common evidence matrix."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--case",
        default=None,
        help="Case id to run, or 'all-common'. Required unless --list or --check-only.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate schema, dependencies, case registry and output safety without execution.",
    )
    parser.add_argument("--list", action="store_true", help="List configured cases.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    return parser.parse_args()


def _require_table(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key!r} must be a TOML table")
    return value


def _require_string_list(case_id: str, raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ConfigurationError(f"{case_id}: {key!r} must be a non-empty string list")
    return tuple(value)


def _require_positive_int_list(
    case_id: str, raw: dict[str, Any], key: str
) -> tuple[int, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0
        for item in value
    ):
        raise ConfigurationError(f"{case_id}: {key!r} must be a positive integer list")
    return tuple(value)


def _safe_output_name(value: str) -> str:
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ConfigurationError(f"unsafe output filename: {value!r}")
    return value


def _case_definition(raw: dict[str, Any]) -> CaseDefinition:
    identifier = raw.get("id")
    if not isinstance(identifier, str) or not CASE_ID_PATTERN.fullmatch(identifier):
        raise ConfigurationError(f"invalid case id: {identifier!r}")
    role = raw.get("role")
    problem = raw.get("problem")
    if not isinstance(role, str) or not role:
        raise ConfigurationError(f"{identifier}: missing role")
    if not isinstance(problem, str) or not problem:
        raise ConfigurationError(f"{identifier}: missing problem")
    execution_status = raw.get("execution_status")
    if execution_status not in ALLOWED_EXECUTION_STATUS:
        raise ConfigurationError(
            f"{identifier}: execution_status must be one of "
            f"{sorted(ALLOWED_EXECUTION_STATUS)}"
        )
    executor = raw.get("executor")
    if executor is not None and not isinstance(executor, str):
        raise ConfigurationError(f"{identifier}: executor must be a string")
    blockers = raw.get("blockers", [])
    if not isinstance(blockers, list) or not all(
        isinstance(item, str) and item for item in blockers
    ):
        raise ConfigurationError(f"{identifier}: blockers must be a string list")
    if execution_status == "ready" and not executor:
        raise ConfigurationError(f"{identifier}: ready case requires an executor")
    if execution_status == "configured" and not blockers:
        raise ConfigurationError(f"{identifier}: configured case requires blockers")
    parameters = raw.get("parameters", {})
    expected_orders = raw.get("expected_orders", {})
    if not isinstance(parameters, dict) or not isinstance(expected_orders, dict):
        raise ConfigurationError(
            f"{identifier}: parameters and expected_orders must be TOML tables"
        )
    return CaseDefinition(
        identifier=identifier,
        role=role,
        problem=problem,
        methods=_require_string_list(identifier, raw, "methods"),
        mesh_families=_require_string_list(identifier, raw, "mesh_families"),
        orders=_require_positive_int_list(identifier, raw, "orders"),
        levels=_require_positive_int_list(identifier, raw, "levels"),
        metrics=_require_string_list(identifier, raw, "metrics"),
        execution_status=execution_status,
        executor=executor,
        blockers=tuple(blockers),
        parameters=parameters,
        expected_orders=expected_orders,
    )


def load_configuration(path: Path = DEFAULT_CONFIG) -> ExperimentConfiguration:
    resolved = path.resolve()
    try:
        payload = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"cannot read {resolved}: {error}") from error
    if payload.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ConfigurationError(f"expected schema_version={EXPECTED_SCHEMA_VERSION}")
    if payload.get("stage") != EXPECTED_STAGE:
        raise ConfigurationError(f"expected stage={EXPECTED_STAGE!r}")
    status = payload.get("status")
    if status != "configured":
        raise ConfigurationError("common evidence status must remain 'configured'")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases or not all(
        isinstance(item, dict) for item in raw_cases
    ):
        raise ConfigurationError("'cases' must be a non-empty array of tables")
    cases = tuple(_case_definition(item) for item in raw_cases)
    identifiers = [case.identifier for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ConfigurationError("case ids must be unique")
    provenance = _require_table(payload, "provenance")
    required_packages = provenance.get("required_packages")
    if not isinstance(required_packages, list) or not required_packages or not all(
        isinstance(item, str) and item for item in required_packages
    ):
        raise ConfigurationError(
            "provenance.required_packages must be a non-empty string list"
        )
    if provenance.get("formal_requires_clean_revision") is not True:
        raise ConfigurationError(
            "provenance.formal_requires_clean_revision must be true"
        )
    if provenance.get("hash_algorithm") != "sha256":
        raise ConfigurationError("provenance.hash_algorithm must be 'sha256'")
    random_seed = provenance.get("random_seed")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool) or random_seed < 0:
        raise ConfigurationError("provenance.random_seed must be a non-negative integer")
    acceptance_raw = _require_table(payload, "acceptance")
    if not acceptance_raw or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        for value in acceptance_raw.values()
    ):
        raise ConfigurationError("all acceptance thresholds must be positive numbers")
    if set(acceptance_raw) != REQUIRED_ACCEPTANCE_KEYS:
        raise ConfigurationError(
            f"acceptance must contain exactly {sorted(REQUIRED_ACCEPTANCE_KEYS)}"
        )
    outputs_raw = _require_table(payload, "outputs")
    outputs: dict[str, str] = {}
    for key, value in outputs_raw.items():
        if not isinstance(value, str):
            raise ConfigurationError(f"outputs.{key} must be a string")
        outputs[key] = _safe_output_name(value)
    required_outputs = {"manifest", "summary", "metrics", "history", "figures_directory"}
    if set(outputs) != required_outputs:
        raise ConfigurationError(
            f"outputs must contain exactly {sorted(required_outputs)}"
        )
    return ExperimentConfiguration(
        path=resolved,
        stage=EXPECTED_STAGE,
        status=status,
        provenance=provenance,
        acceptance={key: float(value) for key, value in acceptance_raw.items()},
        outputs=outputs,
        cases=cases,
    )


def _git_value(repository: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _repository_provenance(repository: Path) -> dict[str, Any]:
    status = _git_value(repository, "status", "--porcelain")
    return {
        "path": str(repository),
        "revision": _git_value(repository, "rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _fealpy_provenance() -> dict[str, Any]:
    try:
        import fealpy
    except ImportError as error:
        return {"version": None, "path": None, "revision": None, "dirty": None, "error": str(error)}
    module_path = Path(fealpy.__file__).resolve()
    repository = next(
        (candidate for candidate in module_path.parents if (candidate / ".git").exists()),
        None,
    )
    result: dict[str, Any] = {
        "version": _package_version("fealpy"),
        "path": str(module_path),
        "revision": None,
        "dirty": None,
    }
    if repository is not None:
        result.update(_repository_provenance(repository))
        result["module_path"] = str(module_path)
    return result


def dependency_report(configuration: ExperimentConfiguration) -> dict[str, Any]:
    packages = {
        name: _package_version(name)
        for name in configuration.provenance["required_packages"]
    }
    missing = sorted(name for name, version in packages.items() if version is None)
    import_errors: list[str] = []
    try:
        from soptx.fem import HuZhangMFEMAnalyzer  # noqa: F401
        from soptx.materials import IsotropicLinearElasticMaterial  # noqa: F401
        from soptx.topology.constraints import ApparentStressConstraint  # noqa: F401
        from soptx.topology.optimizers import ALMMMAOptimizer  # noqa: F401
    except ImportError as error:
        import_errors.append(str(error))
    return {
        "packages": packages,
        "missing_packages": missing,
        "current_namespace_import_errors": import_errors,
        "passed": not missing and not import_errors,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _csv_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    fieldnames = sorted({key for row in records for key in row}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _safe_case_output(base: Path, case_id: str, multi_case: bool) -> Path:
    resolved_base = base.resolve()
    target = (resolved_base / case_id).resolve() if multi_case else resolved_base
    try:
        target.relative_to(resolved_base)
    except ValueError as error:
        raise ConfigurationError(f"case output escapes base directory: {target}") from error
    return target


def _formal_evidence_allowed(provenance: dict[str, Any]) -> bool:
    soptx_dirty = provenance["repositories"]["soptx"]["dirty"]
    fealpy_dirty = provenance["repositories"]["fealpy"]["dirty"]
    return soptx_dirty is False and fealpy_dirty is False


def _base_provenance(configuration: ExperimentConfiguration) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "random_seed": configuration.provenance["random_seed"],
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
        },
        "packages": {
            name: _package_version(name)
            for name in configuration.provenance["required_packages"]
        },
        "repositories": {
            "soptx": _repository_provenance(REPOSITORY_ROOT),
            "fealpy": _fealpy_provenance(),
        },
        "configuration": {
            "path": str(configuration.path),
            "sha256": _sha256(configuration.path),
            "stage": configuration.stage,
        },
    }


def _execute_ready_case(
    case: CaseDefinition, acceptance: dict[str, float], output: Path
) -> dict[str, Any]:
    from executors import EXECUTORS

    executor = EXECUTORS.get(case.executor or "")
    if executor is None:
        raise ConfigurationError(
            f"{case.identifier}: unknown executor {case.executor!r}"
        )
    result = executor(case=case, acceptance=acceptance, output=output)
    if not isinstance(result, dict):
        raise RuntimeError(f"{case.identifier}: executor must return a dictionary")
    return result


def run_case(
    configuration: ExperimentConfiguration,
    case: CaseDefinition,
    output: Path,
) -> tuple[dict[str, Any], int]:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / configuration.outputs["figures_directory"]
    figures.mkdir(exist_ok=True)
    provenance = _base_provenance(configuration)
    exit_code = 0
    try:
        if case.execution_status != "ready":
            raise CaseBlocked(
                f"native adapter not ready: {', '.join(case.blockers)}"
            )
        result = _execute_ready_case(case, configuration.acceptance, output)
        summary = result.get("summary", {})
        metrics = result.get("metrics", [])
        history = result.get("history", [])
        status = "completed"
    except CaseBlocked as error:
        summary = {"status": "blocked", "reason": str(error)}
        metrics = [{"status": "blocked", "reason": str(error)}]
        history = []
        status = "blocked"
        exit_code = 2
    except Exception as error:
        summary = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        metrics = [{"status": "failed", "error": str(error)}]
        history = []
        status = "failed"
        exit_code = 1

    summary_path = output / configuration.outputs["summary"]
    metrics_path = output / configuration.outputs["metrics"]
    history_path = output / configuration.outputs["history"]
    _json_write(summary_path, summary)
    _csv_write(metrics_path, metrics)
    _csv_write(history_path, history)
    artifact_paths = [summary_path, metrics_path, history_path]
    artifact_paths.extend(path for path in figures.rglob("*") if path.is_file())
    artifact_hashes = {
        path.relative_to(output).as_posix(): _sha256(path) for path in artifact_paths
    }
    acceptance_passed = summary.get("acceptance_status") == "passed"
    provenance["evidence_level"] = (
        "formal"
        if status == "completed"
        and acceptance_passed
        and _formal_evidence_allowed(provenance)
        else "development"
    )
    manifest = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "stage": configuration.stage,
        "case": case.as_dict(),
        "acceptance": configuration.acceptance,
        "status": status,
        "provenance": provenance,
        "artifacts": artifact_hashes,
    }
    manifest_path = output / configuration.outputs["manifest"]
    _json_write(manifest_path, manifest)
    return manifest, exit_code


def check_only(configuration: ExperimentConfiguration, output: Path | None) -> dict[str, Any]:
    report = dependency_report(configuration)
    if output is not None:
        resolved = output.resolve()
        if resolved == REPOSITORY_ROOT.resolve():
            raise ConfigurationError("output must not be the repository root")
    report.update(
        {
            "schema_version": EXPECTED_SCHEMA_VERSION,
            "stage": configuration.stage,
            "status": configuration.status,
            "case_count": len(configuration.cases),
            "ready_cases": [
                case.identifier
                for case in configuration.cases
                if case.execution_status == "ready"
            ],
            "configured_cases": [
                case.identifier
                for case in configuration.cases
                if case.execution_status == "configured"
            ],
            "output": None if output is None else str(output.resolve()),
        }
    )
    report["passed"] = bool(report["passed"])
    return report


def main() -> int:
    arguments = parse_arguments()
    try:
        configuration = load_configuration(arguments.config)
        if arguments.list:
            payload = [case.as_dict() for case in configuration.cases]
            print(json.dumps(payload, ensure_ascii=False, indent=2) if arguments.json else "\n".join(case.identifier for case in configuration.cases))
            return 0
        if arguments.check_only:
            report = check_only(configuration, arguments.output)
            print(json.dumps(report, ensure_ascii=False, indent=2) if arguments.json else report)
            return 0 if report["passed"] else 1
        if not arguments.case:
            raise ConfigurationError("--case is required unless --list or --check-only is used")
        if arguments.output is None:
            raise ConfigurationError("--output is required for numerical execution")
        case_by_id = configuration.by_id()
        if arguments.case == "all-common":
            selected = list(configuration.cases)
        else:
            try:
                selected = [case_by_id[arguments.case]]
            except KeyError as error:
                raise ConfigurationError(f"unknown case: {arguments.case!r}") from error
        manifests: list[dict[str, Any]] = []
        final_code = 0
        for case in selected:
            output = _safe_case_output(
                arguments.output, case.identifier, multi_case=len(selected) > 1
            )
            manifest, code = run_case(configuration, case, output)
            manifests.append(manifest)
            final_code = max(final_code, code)
        payload: Any = manifests[0] if len(manifests) == 1 else manifests
        print(json.dumps(payload, ensure_ascii=False, indent=2) if arguments.json else payload)
        return final_code
    except ConfigurationError as error:
        print(f"Hu–Zhang common evidence runner: FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
