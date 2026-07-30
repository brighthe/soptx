from __future__ import annotations

import torch
import torch.nn as nn

from fealpy.ml.grad import gradient
from fealpy.typing import TensorLike

from cases import ElasticityCase


class PINNOperator:
    """Dimension-independent strong-form linear-elasticity PINN operator."""

    def __init__(self, case: ElasticityCase, network: nn.Module) -> None:
        self.case = case
        self.network = network
        self.dimension = case.dimension

    def predict(self, points: TensorLike) -> TensorLike:
        value = self.network(points)
        expected = (*points.shape[:-1], self.dimension)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"Network must return shape {expected}, got {tuple(value.shape)}."
            )
        return value

    def displacement_gradient(self, points: TensorLike) -> TensorLike:
        """Return grad(u) with shape (..., dimension, dimension)."""

        displacement = self.predict(points)
        return torch.stack(
            [
                gradient(
                    displacement[..., component : component + 1],
                    points,
                    create_graph=True,
                )
                for component in range(self.dimension)
            ],
            dim=-2,
        )

    def strain(self, points: TensorLike) -> TensorLike:
        gradient_value = self.displacement_gradient(points)
        return 0.5 * (gradient_value + gradient_value.transpose(-1, -2))

    def stress(self, points: TensorLike) -> TensorLike:
        strain_value = self.strain(points)
        material = self.case.material
        identity = torch.eye(
            self.dimension,
            dtype=points.dtype,
            device=points.device,
        )
        trace = torch.diagonal(
            strain_value,
            dim1=-2,
            dim2=-1,
        ).sum(dim=-1, keepdim=True)
        return (
            material.lame_lambda * trace.unsqueeze(-1) * identity
            + 2.0 * material.shear_modulus * strain_value
        )

    def equilibrium_residual(
        self,
        points: TensorLike,
        *,
        create_graph: bool = True,
    ) -> TensorLike:
        """Return -div(sigma) - body_force with shape (..., dimension)."""

        stress_value = self.stress(points)
        divergence_components = []
        total_derivatives = self.dimension * self.dimension
        derivative_index = 0
        for equation_component in range(self.dimension):
            divergence_component = torch.zeros_like(points[..., 0])
            for coordinate_component in range(self.dimension):
                stress_component = stress_value[
                    ..., equation_component, coordinate_component
                ]
                is_last_derivative = derivative_index == total_derivatives - 1
                stress_gradient = torch.autograd.grad(
                    outputs=stress_component,
                    inputs=points,
                    grad_outputs=torch.ones_like(stress_component),
                    create_graph=create_graph,
                    retain_graph=create_graph or not is_last_derivative,
                )[0]
                divergence_component = (
                    divergence_component
                    + stress_gradient[..., coordinate_component]
                )
                derivative_index += 1
            divergence_components.append(divergence_component)

        divergence = torch.stack(divergence_components, dim=-1)
        body_force = self.case.problem.body_force(points)
        if tuple(body_force.shape) != tuple(divergence.shape):
            raise ValueError(
                f"body_force shape {tuple(body_force.shape)} does not match "
                f"equilibrium residual shape {tuple(divergence.shape)}."
            )
        return -divergence - body_force

    def dirichlet_component_mask(self, points: TensorLike) -> TensorLike:
        direct_predicate = getattr(
            self.case.problem,
            "is_displacement_boundary",
            None,
        )
        if direct_predicate is not None:
            mask = direct_predicate(points)
            if tuple(mask.shape) != tuple(points.shape[:-1]):
                raise ValueError("The displacement-boundary predicate has invalid shape.")
            return mask.unsqueeze(-1).expand(*mask.shape, self.dimension)

        predicates = self.case.problem.is_dirichlet_boundary()
        if not isinstance(predicates, (tuple, list)):
            mask = predicates(points)
            if tuple(mask.shape) != tuple(points.shape[:-1]):
                raise ValueError("The Dirichlet-boundary predicate has invalid shape.")
            return mask.unsqueeze(-1).expand(*mask.shape, self.dimension)
        if len(predicates) != self.dimension:
            raise ValueError(
                "The number of component Dirichlet predicates must match "
                f"dimension {self.dimension}."
            )

        masks = [predicate(points) for predicate in predicates]
        expected = tuple(points.shape[:-1])
        if any(tuple(mask.shape) != expected for mask in masks):
            raise ValueError("A component Dirichlet-boundary predicate has invalid shape.")
        return torch.stack(masks, dim=-1)

    def dirichlet_residual(self, points: TensorLike) -> TensorLike:
        mask = self.dirichlet_component_mask(points)
        if not bool(torch.all(mask)):
            raise ValueError(
                "This baseline requires every sampled boundary point and "
                "displacement component to be Dirichlet; traction and mixed "
                "boundary conditions are not implemented."
            )
        residual = self.predict(points) - self.case.problem.dirichlet_bc(points)
        if tuple(residual.shape) != tuple(points.shape):
            raise ValueError(
                f"Dirichlet residual shape {tuple(residual.shape)} does not match "
                f"point shape {tuple(points.shape)}."
            )
        return residual
