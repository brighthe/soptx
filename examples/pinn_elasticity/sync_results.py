from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import contract
import layout


class EvidenceError(RuntimeError):
    """Raised when raw validation output violates the evidence contract."""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize clean CPU float64 PINN evidence and README "
            "result blocks from ignored validation JSON."
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
        help="Check for drift without writing evidence or README files.",
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
        raise EvidenceError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise EvidenceError(f"JSON artifact must contain an object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def require_check(
    payload: dict[str, Any],
    name: str,
    source: Path,
) -> dict[str, Any]:
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise EvidenceError(f"{source}: missing checks object")
    check = checks.get(name)
    if not isinstance(check, dict) or check.get("passed") is not True:
        raise EvidenceError(f"{source}: check {name!r} did not pass")
    metrics = check.get("metrics")
    if not isinstance(metrics, dict):
        raise EvidenceError(f"{source}: check {name!r} has no metrics")
    return metrics


def build_evidence(dimension: int) -> dict[str, Any]:
    source = layout.validation_summary_path(dimension)
    payload = read_payload(source)
    if payload.get("schema_version") != contract.SCHEMA_VERSION:
        raise EvidenceError(
            f"{source}: expected schema_version={contract.SCHEMA_VERSION}"
        )
    if payload.get("stage") != contract.STAGE:
        raise EvidenceError(f"{source}: unexpected stage")
    if payload.get("dimension") != dimension:
        raise EvidenceError(f"{source}: dimension does not match its filename")
    if payload.get("status") != "passed":
        raise EvidenceError(f"{source}: validation status is not passed")
    if payload.get("official_baseline") is not True:
        raise EvidenceError(f"{source}: training config is not the baseline")

    environment = payload.get("environment")
    if not isinstance(environment, dict):
        raise EvidenceError(f"{source}: missing environment")
    if environment.get("git_dirty") is not False:
        raise EvidenceError(
            f"{source}: formal evidence requires git_dirty=false"
        )
    if not environment.get("git_revision"):
        raise EvidenceError(f"{source}: missing git revision")

    manufactured = require_check(
        payload,
        "manufactured_solution_consistency",
        source,
    )
    training = require_check(payload, "training_baseline", source)
    require_check(payload, "constructor_and_shape", source)
    require_check(payload, "unsupported_problem_guards", source)
    require_check(payload, "one_update_and_checkpoints", source)
    require_check(payload, "cli_smoke", source)

    l2 = training.get("best_checkpoint_displacement_l2")
    if not isinstance(l2, dict):
        raise EvidenceError(f"{source}: missing best-checkpoint L2 metrics")
    case = payload.get("case")
    config = payload.get("training_config")
    thresholds = payload.get("thresholds")
    if not all(isinstance(item, dict) for item in (case, config, thresholds)):
        raise EvidenceError(f"{source}: missing case/config/threshold contract")
    if config != contract.RunConfig().evidence_dict():
        raise EvidenceError(f"{source}: training config differs from baseline")
    if thresholds != contract.validation_thresholds(dimension):
        raise EvidenceError(f"{source}: validation thresholds differ from contract")

    return {
        "schema_version": contract.SCHEMA_VERSION,
        "stage": contract.STAGE,
        "dimension": dimension,
        "scope": f"{layout.EVIDENCE_SCOPE}-correctness",
        "source_artifact": {
            "path": layout.relative_to_example(source),
            "sha256": sha256(source),
        },
        "environment": {
            "git_revision": environment["git_revision"],
            "git_dirty": environment["git_dirty"],
            "python": environment.get("python"),
            "platform": environment.get("platform"),
            "torch": environment.get("torch"),
            "fealpy": environment.get("fealpy"),
            "soptx": environment.get("soptx"),
        },
        "case": case,
        "training_config": config,
        "thresholds": thresholds,
        "manufactured_solution": {
            "displacement_gradient_max_abs": manufactured[
                "displacement_gradient_max_abs"
            ],
            "equilibrium_residual_max_abs": manufactured[
                "equilibrium_residual_max_abs"
            ],
            "dirichlet_residual_max_abs": manufactured[
                "dirichlet_residual_max_abs"
            ],
        },
        "training": {
            "first_validation_loss": training["first_validation_loss"],
            "best_validation_loss": training["best_validation_loss"],
            "final_validation_loss": training["final_validation_loss"],
            "best_epoch": training["best_checkpoint_epoch"],
            "best_relative_displacement_l2": l2["relative_combined"],
            "fixed_equilibrium_residual_rms": training[
                "fixed_equilibrium_residual_rms"
            ],
            "fixed_boundary_error_max_abs": training[
                "fixed_boundary_error_max_abs"
            ],
            "elapsed_seconds": training["elapsed_seconds"],
        },
    }


def scientific(value: float) -> str:
    return f"{value:.5e}"


def render_readme_block(
    dimension: int,
    evidence: dict[str, Any],
) -> str:
    begin, end = layout.readme_markers(dimension)
    manufactured = evidence["manufactured_solution"]
    training = evidence["training"]
    evidence_name = layout.evidence_path(dimension).name
    return "\n".join(
        [
            begin,
            "",
            (
                f"本节由 `{layout.SYNC_SCRIPT_NAME} --dim {dimension}` 根据 "
                "clean revision 的原始 validation JSON 生成；精简证据见 "
                f"`evidence/{evidence_name}`。"
            ),
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
            (
                "| 精确位移梯度最大绝对误差 | "
                f"`{scientific(manufactured['displacement_gradient_max_abs'])}` |"
            ),
            (
                "| 精确平衡 residual 最大绝对值 | "
                f"`{scientific(manufactured['equilibrium_residual_max_abs'])}` |"
            ),
            (
                "| 精确 Dirichlet residual 最大绝对值 | "
                f"`{scientific(manufactured['dirichlet_residual_max_abs'])}` |"
            ),
            (
                "| first fixed-validation loss | "
                f"`{scientific(training['first_validation_loss'])}` |"
            ),
            (
                "| best fixed-validation loss | "
                f"`{scientific(training['best_validation_loss'])}` |"
            ),
            (
                "| best checkpoint relative displacement L2 | "
                f"`{scientific(training['best_relative_displacement_l2'])}` |"
            ),
            (
                "| 固定点平衡 residual RMS | "
                f"`{scientific(training['fixed_equilibrium_residual_rms'])}` |"
            ),
            (
                "| 最大边界位移误差 | "
                f"`{scientific(training['fixed_boundary_error_max_abs'])}` |"
            ),
            "",
            (
                "源 revision："
                f"`{evidence['environment']['git_revision']}`；"
                f"{contract.RunConfig.epochs} 次更新耗时 "
                f"`{training['elapsed_seconds']:.2f} s`。"
            ),
            "",
            end,
        ]
    )


def replace_generated_block(
    readme: str,
    dimension: int,
    generated: str,
) -> str:
    begin, end = layout.readme_markers(dimension)
    begin_index = readme.find(begin)
    end_index = readme.find(end)
    if begin_index < 0 or end_index < begin_index:
        raise EvidenceError(
            f"README {dimension}D result markers are missing or invalid"
        )
    end_index += len(end)
    return readme[:begin_index] + generated + readme[end_index:]


def serialized(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def check_file(path: Path, expected: str) -> bool:
    if not path.is_file():
        print(f"OUT OF DATE: missing {path}", file=sys.stderr)
        return False
    if path.read_text(encoding="utf-8") != expected:
        print(f"OUT OF DATE: {path}", file=sys.stderr)
        return False
    return True


def main() -> int:
    arguments = parse_arguments()
    try:
        readme = layout.README_PATH.read_text(encoding="utf-8")
        evidence_text: dict[int, str] = {}
        for dimension in selected_dimensions(arguments.dim):
            evidence = build_evidence(dimension)
            readme = replace_generated_block(
                readme,
                dimension,
                render_readme_block(dimension, evidence),
            )
            evidence_text[dimension] = serialized(evidence)

        if arguments.check:
            passed = all(
                check_file(layout.evidence_path(dimension), text)
                for dimension, text in evidence_text.items()
            )
            passed = check_file(layout.README_PATH, readme) and passed
            if passed:
                print("PINN evidence and README are in sync.")
                return 0
            return 1

        layout.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        for dimension, text in evidence_text.items():
            layout.evidence_path(dimension).write_text(
                text,
                encoding="utf-8",
            )
        layout.README_PATH.write_text(readme, encoding="utf-8")
        print("PINN evidence and README were synchronized.")
        return 0
    except EvidenceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
