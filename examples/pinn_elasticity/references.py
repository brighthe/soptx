from __future__ import annotations

import torch

from fealpy.backend import bm
from fealpy.ml.modules import Solution
from fealpy.ml.sampler import BoxBoundarySampler

from cases import ElasticityCase
from operators import PINNOperator


class ExactDisplacement(torch.nn.Module):
    """Represent the SOPT-X manufactured displacement as a network."""

    def __init__(self, problem) -> None:
        super().__init__()
        self.problem = problem

    def forward(self, points):
        return self.problem.disp_solution(points)


def make_exact_operator(
    case: ElasticityCase,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> PINNOperator:
    network = Solution(ExactDisplacement(case.problem)).to(
        dtype=dtype,
        device=device,
    )
    return PINNOperator(case, network)


def fixed_diagnostic_points(
    case: ElasticityCase,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    interior_count = 32 if case.dimension == 2 else 10
    boundary_steps = 101 if case.dimension == 2 else 21
    axes = [
        torch.linspace(
            case.domain[2 * axis],
            case.domain[2 * axis + 1],
            interior_count + 2,
            dtype=dtype,
            device=device,
        )[1:-1]
        for axis in range(case.dimension)
    ]
    grids = torch.meshgrid(*axes, indexing="ij")
    interior = torch.stack(
        [grid.reshape(-1) for grid in grids],
        dim=-1,
    )
    interior.requires_grad_(True)

    boundary_sampler = BoxBoundarySampler(
        case.domain,
        mode="linspace",
        dtype=bm.float64,
        device=device,
        requires_grad=True,
    )
    boundary = boundary_sampler.run(boundary_steps)
    return interior, boundary


def manufactured_validation_points(
    case: ElasticityCase,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic interior and whole-boundary points for exact gates."""

    axes = [
        torch.linspace(
            case.domain[2 * axis],
            case.domain[2 * axis + 1],
            7,
            dtype=dtype,
            device=device,
        )[1:-1]
        for axis in range(case.dimension)
    ]
    grids = torch.meshgrid(*axes, indexing="ij")
    interior = torch.stack(
        [grid.reshape(-1) for grid in grids],
        dim=-1,
    )
    interior.requires_grad_(True)

    sampler = BoxBoundarySampler(
        case.domain,
        mode="linspace",
        dtype=bm.float64,
        device=device,
        requires_grad=True,
    )
    return interior, sampler.run(7)
