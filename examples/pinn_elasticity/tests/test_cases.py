import pytest
import torch

from fealpy.backend import bm

from cases import create_case


@pytest.mark.parametrize("dimension", (2, 3))
def test_case_values_and_gradient_have_dimension_consistent_shapes(dimension):
    bm.set_backend("pytorch")
    case = create_case(dimension)
    points = torch.full(
        (4, dimension),
        0.25,
        dtype=torch.float64,
    )
    case.validate_problem_values(points)
    assert case.problem.disp_solution(points).shape == points.shape
    assert case.exact_displacement_gradient(points).shape == (
        4,
        dimension,
        dimension,
    )


def test_unsupported_dimension_is_rejected():
    with pytest.raises(ValueError, match="2 or 3"):
        create_case(4)


def test_dimension_specific_diagnostic_mesh_defaults():
    assert create_case(2).diagnostic_mesh_size == 30
    assert create_case(3).diagnostic_mesh_size == 8
