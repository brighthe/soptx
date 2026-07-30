from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

import contract
import layout


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=layout.REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def environment_record() -> dict[str, Any]:
    status = git_value("status", "--porcelain")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "fealpy": package_version("fealpy"),
        "soptx": package_version("soptx"),
        "cuda_available": torch.cuda.is_available(),
        "git_revision": git_value("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def history_is_finite(history: dict[str, list[Any]]) -> bool:
    for key, values in history.items():
        if key == "epoch":
            continue
        for item in values:
            nested = item if isinstance(item, list) else [item]
            for value in nested:
                if value is not None and not math.isfinite(value):
                    return False
    return True


def local_gates(training, config: contract.RunConfig) -> dict[str, bool]:
    history = training.history
    checkpoints_expected = config.checkpoint_dir is not None
    if checkpoints_expected:
        directory = Path(config.checkpoint_dir)
        checkpoints_present = (
            (directory / "best.pt").is_file()
            and (directory / "last.pt").is_file()
        )
    else:
        checkpoints_present = True
    return {
        "history_recorded": bool(history.get("epoch")),
        "history_finite": history_is_finite(history),
        "best_state_recorded": (
            training.best_epoch >= 1
            and bool(training.best_model_state_dict)
        ),
        "requested_checkpoints_written": checkpoints_present,
    }


def build_run_payload(
    case,
    config: contract.RunConfig,
    training,
    gates: dict[str, bool],
    *,
    command: list[str] | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "stage": contract.STAGE,
        "command": [sys.executable, *sys.argv] if command is None else command,
        "environment": (
            environment_record() if environment is None else environment
        ),
        "parameters": {
            "dimension": case.dimension,
            "case": case.name,
            "domain": list(case.domain),
            "material": case.material.as_dict(),
            "config": config.as_dict(),
            "official_baseline": contract.is_official_baseline(config),
        },
        "training": {
            "history": training.history,
            "best_epoch": training.best_epoch,
            "best_validation_loss": training.best_validation_loss,
            "best_metrics": training.best_metrics,
            "last_metrics": training.last_metrics,
        },
        "local_gates": gates,
        "local_passed": all(gates.values()),
        "artifacts": {
            "summary_json": (
                None
                if config.summary_path is None
                else str(config.summary_path.resolve())
            ),
            "checkpoint_dir": (
                None
                if config.checkpoint_dir is None
                else str(config.checkpoint_dir.resolve())
            ),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_run_summary(payload: dict[str, Any]) -> None:
    training = payload["training"]
    print(
        json.dumps(
            {
                "dimension": payload["parameters"]["dimension"],
                "case": payload["parameters"]["case"],
                "best_epoch": training["best_epoch"],
                "best_validation_loss": training["best_validation_loss"],
                "last_validation_loss": training["last_metrics"].get(
                    "validation_loss"
                ),
                "summary": payload["artifacts"]["summary_json"],
                "local_passed": payload["local_passed"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
