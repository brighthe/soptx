import pytest
import torch

from fealpy.backend import bm

import contract
from cases import create_case
from references import make_exact_operator, manufactured_validation_points


@pytest.mark.parametrize("dimension", (2, 3))
def test_exact_operator_matches_gradient_and_equilibrium(dimension):
    bm.set_backend("pytorch")
    case = create_case(dimension)
    device = torch.device("cpu")
    operator = make_exact_operator(
        case,
        dtype=torch.float64,
        device=device,
    )
    interior, boundary = manufactured_validation_points(
        case,
        dtype=torch.float64,
        device=device,
    )
    gradient_error = torch.max(
        torch.abs(
            operator.displacement_gradient(interior)
            - case.exact_displacement_gradient(interior)
        )
    )
    equilibrium = operator.equilibrium_residual(
        interior,
        create_graph=False,
    )
    dirichlet = operator.dirichlet_residual(boundary)
    assert gradient_error.detach().item() <= contract.EXACT_GRADIENT_MAX_ABS
    assert torch.max(torch.abs(equilibrium)).detach().item() <= (
        contract.EXACT_EQUILIBRIUM_MAX_ABS
    )
    assert torch.max(torch.abs(dirichlet)).detach().item() <= (
        contract.EXACT_BOUNDARY_MAX_ABS
    )


@pytest.mark.parametrize("dimension", (2, 3))
def test_component_boundary_mask_covers_every_sample(dimension):
    bm.set_backend("pytorch")
    case = create_case(dimension)
    device = torch.device("cpu")
    operator = make_exact_operator(
        case,
        dtype=torch.float64,
        device=device,
    )
    _, boundary = manufactured_validation_points(
        case,
        dtype=torch.float64,
        device=device,
    )
    mask = operator.dirichlet_component_mask(boundary)
    assert mask.shape == boundary.shape
    assert bool(torch.all(mask))
