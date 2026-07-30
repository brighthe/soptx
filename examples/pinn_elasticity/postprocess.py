from __future__ import annotations

from typing import Any

import torch

from fealpy.ml.modules import Solution

import contract
from cases import ElasticityCase


class ZeroDisplacement(torch.nn.Module):
    def forward(self, points):
        return torch.zeros_like(points)


def displacement_l2_error(
    network,
    case: ElasticityCase,
    mesh,
) -> tuple[torch.Tensor, float]:
    component_error = network.estimate_error(
        case.problem.disp_solution,
        mesh,
        coordtype="c",
    ).detach()
    combined_error = float(torch.linalg.vector_norm(component_error).item())
    return component_error, combined_error


def relative_l2_metrics(
    network,
    case: ElasticityCase,
    mesh,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    error_components, absolute_combined = displacement_l2_error(
        network,
        case,
        mesh,
    )
    zero = Solution(ZeroDisplacement()).to(device=device, dtype=dtype)
    exact_norm_components = zero.estimate_error(
        case.problem.disp_solution,
        mesh,
        coordtype="c",
    ).detach()
    exact_norm_combined = float(
        torch.linalg.vector_norm(exact_norm_components).item()
    )
    if exact_norm_combined <= contract.NORM_FLOOR:
        raise RuntimeError("The exact displacement has zero L2 norm.")

    error_values = [
        float(value) for value in error_components.flatten().tolist()
    ]
    norm_values = [
        float(value) for value in exact_norm_components.flatten().tolist()
    ]
    relative_components: list[float | None] = []
    for error, reference in zip(error_values, norm_values):
        relative_components.append(
            None
            if abs(reference) <= contract.NORM_FLOOR
            else error / reference
        )
    return {
        "absolute_components": error_values,
        "absolute_combined": absolute_combined,
        "exact_norm_components": norm_values,
        "exact_norm_combined": exact_norm_combined,
        "relative_components": relative_components,
        "relative_combined": absolute_combined / exact_norm_combined,
    }


def show_training_history(history: dict[str, list]) -> None:
    if not history.get("epoch"):
        raise RuntimeError("Run training before displaying diagnostics.")

    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(epochs, history["train_loss"], label="train loss")
    axes[0].semilogy(
        epochs,
        history["validation_loss"],
        label="fixed validation loss",
    )
    axes[0].set_xlabel("parameter updates")
    axes[0].set_ylabel("loss")
    axes[0].set_title("PINN training diagnostics")
    axes[0].grid(True)
    axes[0].legend()

    filtered = [
        (epoch, value)
        for epoch, value in zip(epochs, history["l2_error"])
        if value is not None
    ]
    if filtered:
        axes[1].semilogy(
            [item[0] for item in filtered],
            [item[1] for item in filtered],
        )
        axes[1].set_ylabel("combined displacement L2 error")
        axes[1].set_title("Exact-solution diagnostic")
        axes[1].grid(True)
        axes[1].set_xlabel("parameter updates")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No exact displacement diagnostic",
            ha="center",
            va="center",
        )
        axes[1].set_axis_off()
    figure.tight_layout()
    plt.show()
