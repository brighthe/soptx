from __future__ import annotations

import copy
from dataclasses import dataclass
import logging
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from fealpy.backend import bm
from fealpy.ml.modules import Solution
from fealpy.ml.sampler import BoxBoundarySampler, ISampler
from fealpy.ml.torch_mapping import activations, optimizers

import contract
from cases import ElasticityCase
from operators import PINNOperator
from postprocess import displacement_l2_error
from report import environment_record


@dataclass
class PreparedPINNProblem:
    case: ElasticityCase
    config: contract.RunConfig
    operator: PINNOperator
    optimizer: torch.optim.Optimizer
    scheduler: StepLR | None
    diagnostic_mesh: object
    dtype: torch.dtype
    device: torch.device
    logger: logging.Logger
    environment: dict[str, Any]


@dataclass
class TrainingResult:
    history: dict[str, list[Any]]
    best_epoch: int
    best_validation_loss: float
    best_metrics: dict[str, Any]
    best_model_state_dict: dict[str, torch.Tensor]
    last_metrics: dict[str, Any]


def _set_seed(seed: int, device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"Requested device '{device}' is unavailable.")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _make_logger(case: ElasticityCase, level: str) -> logging.Logger:
    logger = logging.getLogger(f"ElasticityPINN[{case.dimension}D]")
    logger.propagate = False
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def _make_network(
    case: ElasticityCase,
    config: contract.RunConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Solution:
    layers: list[nn.Module] = []
    sizes = (case.dimension,) + config.hidden_size + (case.dimension,)
    activation_type = activations[config.activation]
    for index in range(len(sizes) - 1):
        layers.append(
            nn.Linear(
                sizes[index],
                sizes[index + 1],
                dtype=dtype,
                device=device,
            )
        )
        if index < len(sizes) - 2:
            layers.append(activation_type())
    return Solution(nn.Sequential(*layers)).to(device=device, dtype=dtype)


def prepare_problem(
    case: ElasticityCase,
    config: contract.RunConfig | None = None,
    *,
    network: nn.Module | None = None,
) -> PreparedPINNProblem:
    if bm.backend_name != "pytorch":
        raise RuntimeError(
            "Elasticity PINN requires the PyTorch backend because its residual "
            "uses torch.autograd."
        )
    case.validate()
    config = contract.RunConfig() if config is None else config
    config.validate()
    if config.activation not in activations:
        raise ValueError(f"Unknown activation {config.activation!r}.")
    if config.optimizer not in optimizers:
        raise ValueError(f"Unknown optimizer {config.optimizer!r}.")

    dtype = torch.float64
    device = torch.device(config.device)
    _set_seed(config.seed, device)
    points = torch.zeros(
        (2, case.dimension),
        dtype=dtype,
        device=device,
    )
    case.validate_problem_values(points)

    if network is None:
        solution = _make_network(
            case,
            config,
            dtype=dtype,
            device=device,
        )
    elif isinstance(network, Solution):
        solution = network.to(device=device, dtype=dtype)
    else:
        solution = Solution(network).to(device=device, dtype=dtype)

    operator = PINNOperator(case, solution)
    optimizer_type = optimizers[config.optimizer]
    optimizer = optimizer_type(solution.parameters(), lr=config.lr)
    scheduler = (
        None
        if config.step_size == 0
        else StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    )
    diagnostic_mesh = case.create_diagnostic_mesh(
        config.diagnostic_mesh_size,
        device=config.device,
    )
    return PreparedPINNProblem(
        case=case,
        config=config,
        operator=operator,
        optimizer=optimizer,
        scheduler=scheduler,
        diagnostic_mesh=diagnostic_mesh,
        dtype=dtype,
        device=device,
        logger=_make_logger(case, config.log_level),
        environment=environment_record(),
    )


def loss_components(
    operator: PINNOperator,
    interior,
    boundary,
    *,
    weights: tuple[float, float],
    create_graph: bool,
):
    equilibrium = operator.equilibrium_residual(
        interior,
        create_graph=create_graph,
    )
    dirichlet = operator.dirichlet_residual(boundary)
    equilibrium_loss = torch.mean(equilibrium**2)
    dirichlet_loss = torch.mean(dirichlet**2)
    total_loss = (
        weights[0] * equilibrium_loss
        + weights[1] * dirichlet_loss
    )
    return equilibrium_loss, dirichlet_loss, total_loss


def _make_samplers(prepared: PreparedPINNProblem):
    options = {
        "mode": prepared.config.sampling_mode,
        "dtype": bm.float64,
        "device": prepared.device,
        "requires_grad": True,
    }
    return (
        ISampler(prepared.case.domain, **options),
        BoxBoundarySampler(prepared.case.domain, **options),
    )


def _sample_pair(
    interior_sampler: ISampler,
    boundary_sampler: BoxBoundarySampler,
    n_interior: int,
    n_boundary: int,
):
    return (
        interior_sampler.run(n_interior),
        boundary_sampler.run(n_boundary),
    )


def _empty_history() -> dict[str, list[Any]]:
    return {
        "epoch": [],
        "train_equilibrium_loss": [],
        "train_dirichlet_loss": [],
        "train_loss": [],
        "validation_equilibrium_loss": [],
        "validation_dirichlet_loss": [],
        "validation_loss": [],
        "learning_rate": [],
        "l2_error_components": [],
        "l2_error": [],
    }


def _clone_model_state(network: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else None
        ),
    }


def _checkpoint_payload(
    prepared: PreparedPINNProblem,
    history: dict[str, list[Any]],
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "stage": contract.STAGE,
        "dimension": prepared.case.dimension,
        "case": prepared.case.name,
        "domain": list(prepared.case.domain),
        "material": prepared.case.material.as_dict(),
        "environment": prepared.environment,
        "rng_state": _rng_state(prepared.device),
        "epoch": epoch,
        "model_state_dict": prepared.operator.network.state_dict(),
        "optimizer_state_dict": prepared.optimizer.state_dict(),
        "scheduler_state_dict": (
            None
            if prepared.scheduler is None
            else prepared.scheduler.state_dict()
        ),
        "options": prepared.config.as_dict(),
        "history": history,
        "metrics": metrics,
    }


def _save_checkpoint(
    prepared: PreparedPINNProblem,
    history: dict[str, list[Any]],
    name: str,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    if prepared.config.checkpoint_dir is None:
        return
    directory = prepared.config.checkpoint_dir
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(prepared, history, epoch, metrics),
        directory / name,
    )


def _record_diagnostics(
    prepared: PreparedPINNProblem,
    history: dict[str, list[Any]],
    epoch: int,
    learning_rate: float,
    train_points,
    validation_points,
) -> dict[str, Any]:
    for point in (*train_points, *validation_points):
        point.grad = None
    train_losses = loss_components(
        prepared.operator,
        *train_points,
        weights=prepared.config.weights,
        create_graph=False,
    )
    validation_losses = loss_components(
        prepared.operator,
        *validation_points,
        weights=prepared.config.weights,
        create_graph=False,
    )
    try:
        component_error, combined_error = displacement_l2_error(
            prepared.operator.network,
            prepared.case,
            prepared.diagnostic_mesh,
        )
        component_values = [
            float(value) for value in component_error.flatten().tolist()
        ]
    except (AttributeError, NotImplementedError, RuntimeError, ValueError) as error:
        prepared.logger.warning(
            "Unable to evaluate displacement L2 error: %s",
            error,
        )
        component_values = None
        combined_error = None

    metrics = {
        "train_equilibrium_loss": float(train_losses[0].detach().item()),
        "train_dirichlet_loss": float(train_losses[1].detach().item()),
        "train_loss": float(train_losses[2].detach().item()),
        "validation_equilibrium_loss": float(
            validation_losses[0].detach().item()
        ),
        "validation_dirichlet_loss": float(
            validation_losses[1].detach().item()
        ),
        "validation_loss": float(validation_losses[2].detach().item()),
        "learning_rate": learning_rate,
        "l2_error_components": component_values,
        "l2_error": combined_error,
    }
    history["epoch"].append(epoch)
    for key, value in metrics.items():
        history[key].append(value)
    prepared.logger.info(
        "epoch: %d, train loss: %.6e, validation loss: %.6e",
        epoch,
        metrics["train_loss"],
        metrics["validation_loss"],
    )
    return metrics


def _show_progress(epoch: int, total: int) -> None:
    width = 30
    completed = int(width * epoch / total)
    bar = "#" * completed + "-" * (width - completed)
    ending = "\n" if epoch == total else "\r"
    print(
        f"[{bar}] {epoch}/{total}",
        end=ending,
        flush=True,
    )


def train_prepared_problem(prepared: PreparedPINNProblem) -> TrainingResult:
    """Train for exactly ``config.epochs`` parameter updates."""

    config = prepared.config
    _set_seed(config.seed, prepared.device)
    history = _empty_history()
    best_epoch = 0
    best_validation_loss = float("inf")
    best_metrics: dict[str, Any] = {}
    best_model_state_dict: dict[str, torch.Tensor] = {}
    last_metrics: dict[str, Any] = {}

    train_interior, train_boundary = _make_samplers(prepared)
    validation_interior, validation_boundary = _make_samplers(prepared)
    validation_points = _sample_pair(
        validation_interior,
        validation_boundary,
        config.nval_pde,
        config.nval_bc,
    )

    train_points = None
    if config.sampling_mode == "linspace":
        train_points = _sample_pair(
            train_interior,
            train_boundary,
            config.npde,
            config.nbc,
        )

    for epoch in range(1, config.epochs + 1):
        if train_points is None:
            train_points = _sample_pair(
                train_interior,
                train_boundary,
                config.npde,
                config.nbc,
            )
        interior, boundary = train_points
        interior.grad = None
        boundary.grad = None
        learning_rate = float(prepared.optimizer.param_groups[0]["lr"])

        prepared.optimizer.zero_grad()
        losses = loss_components(
            prepared.operator,
            interior,
            boundary,
            weights=config.weights,
            create_graph=True,
        )
        losses[2].backward()
        prepared.optimizer.step()
        if prepared.scheduler is not None:
            prepared.scheduler.step()

        if (
            epoch == 1
            or epoch % config.log_interval == 0
            or epoch == config.epochs
        ):
            last_metrics = _record_diagnostics(
                prepared,
                history,
                epoch,
                learning_rate,
                train_points,
                validation_points,
            )
            if last_metrics["validation_loss"] < best_validation_loss:
                best_epoch = epoch
                best_validation_loss = last_metrics["validation_loss"]
                best_metrics = copy.deepcopy(last_metrics)
                best_model_state_dict = _clone_model_state(
                    prepared.operator.network
                )
                _save_checkpoint(
                    prepared,
                    history,
                    "best.pt",
                    epoch,
                    last_metrics,
                )

        if config.pbar_log:
            _show_progress(epoch, config.epochs)
        if config.sampling_mode == "random":
            train_points = None

    _save_checkpoint(
        prepared,
        history,
        "last.pt",
        config.epochs,
        last_metrics,
    )
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        best_metrics=best_metrics,
        best_model_state_dict=best_model_state_dict,
        last_metrics=last_metrics,
    )


def restore_best_state(
    prepared: PreparedPINNProblem,
    result: TrainingResult,
) -> None:
    if not result.best_model_state_dict:
        raise RuntimeError("Training did not record a best model state.")
    prepared.operator.network.load_state_dict(result.best_model_state_dict)
