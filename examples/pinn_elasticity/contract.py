"""Stable numeric and configuration contract for the elasticity PINN baseline.

This module intentionally has no FEALPy, SOPTX or PyTorch imports.  Reporting,
validation and evidence tooling must be able to inspect the public contract
without initializing the numerical runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 3
STAGE = "soptx/pinn-elasticity/stage-1"

SUPPORTED_DIMENSIONS = (2, 3)
DEFAULT_DIMENSION = 2
SUPPORTED_DTYPES = ("float64",)
SUPPORTED_SAMPLING_MODES = ("random", "linspace")
SUPPORTED_LOG_LEVELS = (
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "NOTSET",
)

EXACT_STRAIN_SYMMETRY_MAX_ABS = 1.0e-12
EXACT_GRADIENT_MAX_ABS = 1.0e-12
EXACT_EQUILIBRIUM_MAX_ABS = 1.0e-10
EXACT_BOUNDARY_MAX_ABS = 1.0e-12
BEST_VALIDATION_LOSS_MAX_2D = 5.0e-2
RELATIVE_DISPLACEMENT_L2_MAX_2D = 1.0e-1
RELATIVE_DISPLACEMENT_L2_MAX_3D = 1.0
NORM_FLOOR = 1.0e-30


@dataclass(frozen=True)
class RunConfig:
    """One fully resolved PINN training run."""

    hidden_size: tuple[int, ...] = (32, 32, 16)
    activation: str = "Tanh"
    optimizer: str = "Adam"
    lr: float = 1.0e-3
    step_size: int = 0
    gamma: float = 0.99
    epochs: int = 2000
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float64"
    sampling_mode: str = "random"
    npde: int = 400
    nbc: int = 100
    nval_pde: int = 400
    nval_bc: int = 100
    weights: tuple[float, float] = (1.0, 30.0)
    log_interval: int = 100
    checkpoint_dir: Path | None = None
    summary_path: Path | None = None
    diagnostic_mesh_size: int | None = None
    pbar_log: bool = False
    log_level: str = "INFO"

    def validate(self) -> None:
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"'dtype' must be one of {SUPPORTED_DTYPES}, "
                f"received {self.dtype!r}."
            )
        if not self.device.strip():
            raise ValueError("'device' must be a non-empty device name.")
        if self.epochs < 1:
            raise ValueError("'epochs' must be at least one parameter update.")
        counts = (self.npde, self.nbc, self.nval_pde, self.nval_bc)
        if min(counts) < 1:
            raise ValueError("All collocation-point counts must be positive.")
        if self.sampling_mode == "linspace" and min(counts) < 2:
            raise ValueError(
                "Linspace collocation counts are per-axis steps and must "
                "all be at least two."
            )
        if self.sampling_mode not in SUPPORTED_SAMPLING_MODES:
            raise ValueError(
                f"'sampling_mode' must be one of "
                f"{SUPPORTED_SAMPLING_MODES}."
            )
        if not self.hidden_size or min(self.hidden_size) < 1:
            raise ValueError("'hidden_size' must contain positive widths.")
        if not self.activation:
            raise ValueError("'activation' must be non-empty.")
        if not self.optimizer:
            raise ValueError("'optimizer' must be non-empty.")
        if not math.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError("'lr' must be finite and positive.")
        if self.step_size < 0:
            raise ValueError("'step_size' must be non-negative.")
        if (
            not math.isfinite(self.gamma)
            or self.gamma <= 0.0
            or self.gamma > 1.0
        ):
            raise ValueError("'gamma' must be finite and in (0, 1].")
        if self.seed < 0:
            raise ValueError("'seed' must be non-negative.")
        if len(self.weights) != 2:
            raise ValueError("'weights' must contain (equilibrium, Dirichlet).")
        if (
            any(not math.isfinite(value) or value < 0.0 for value in self.weights)
            or not any(value > 0.0 for value in self.weights)
        ):
            raise ValueError(
                "'weights' must be finite, non-negative and not both zero."
            )
        if self.log_interval < 1:
            raise ValueError("'log_interval' must be positive.")
        if (
            self.diagnostic_mesh_size is not None
            and self.diagnostic_mesh_size < 2
        ):
            raise ValueError("'diagnostic_mesh_size' must be at least two.")
        if self.log_level.upper() not in SUPPORTED_LOG_LEVELS:
            raise ValueError(
                f"Unknown logging level {self.log_level!r}."
            )

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["hidden_size"] = list(self.hidden_size)
        values["weights"] = list(self.weights)
        for key in ("checkpoint_dir", "summary_path"):
            value = values[key]
            values[key] = None if value is None else str(value)
        return values

    def evidence_dict(self) -> dict[str, Any]:
        """Return reproducibility-relevant values without artifact paths."""

        values = self.as_dict()
        for key in (
            "checkpoint_dir",
            "summary_path",
            "pbar_log",
            "log_level",
        ):
            values.pop(key)
        return values


def is_official_baseline(config: RunConfig) -> bool:
    """Return whether ``config`` matches the evidence-producing baseline."""

    return config.evidence_dict() == RunConfig().evidence_dict()


def validation_thresholds(dimension: int) -> dict[str, float]:
    if dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError(
            f"dimension must be one of {SUPPORTED_DIMENSIONS}, "
            f"received {dimension}"
        )
    values = {
        "exact_strain_symmetry_max_abs": EXACT_STRAIN_SYMMETRY_MAX_ABS,
        "exact_gradient_max_abs": EXACT_GRADIENT_MAX_ABS,
        "exact_equilibrium_max_abs": EXACT_EQUILIBRIUM_MAX_ABS,
        "exact_boundary_max_abs": EXACT_BOUNDARY_MAX_ABS,
        "relative_displacement_l2_max": (
            RELATIVE_DISPLACEMENT_L2_MAX_2D
            if dimension == 2
            else RELATIVE_DISPLACEMENT_L2_MAX_3D
        ),
    }
    if dimension == 2:
        values["best_validation_loss_max"] = BEST_VALIDATION_LOSS_MAX_2D
    return values
