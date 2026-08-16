"""Tests for cell partitioning and its recorded strategy label."""

from __future__ import annotations

import numpy as np
import pytest

# soptx.fem.distributed 经由 fealpy.distributed 引入 mpi4py, 属可选 extra
pytest.importorskip("mpi4py")

from fealpy.mesh import TetrahedronMesh, TriangleMesh

from tools.matrix_free_evidence import contract
from soptx.fem.distributed import partition_cells, partition_strategy_label
from soptx.problems.elasticity import (
    DivergenceFreePolynomialElasticity3D,
    SinusoidalPlaneStrainElasticity2D,
)

PROBLEM_FACTORIES = {
    2: SinusoidalPlaneStrainElasticity2D,
    3: DivergenceFreePolynomialElasticity3D,
}
MESH_FACTORIES = {2: TriangleMesh, 3: TetrahedronMesh}


def coarse_mesh(dimension: int):
    """取该维数下最粗的一档网格, 并给出沿 x 轴对半切的坐标"""

    problem = PROBLEM_FACTORIES[dimension]()
    resolution = (contract.REFINEMENTS[dimension][0],) * dimension
    mesh = MESH_FACTORIES[dimension].from_box(
        list(problem.domain),
        **dict(zip(("nx", "ny", "nz"), resolution)),
    )
    split_coordinate = 0.5 * (problem.domain[0] + problem.domain[1])
    return mesh, split_coordinate


@pytest.mark.parametrize("dimension", contract.SUPPORTED_DIMENSIONS)
def test_single_rank_owns_every_cell(dimension):
    mesh, split_coordinate = coarse_mesh(dimension)

    masks = partition_cells(mesh, 1, split_coordinate=split_coordinate)

    assert len(masks) == 1
    assert np.all(np.asarray(masks[0]))


@pytest.mark.parametrize("dimension", contract.SUPPORTED_DIMENSIONS)
def test_two_ranks_are_disjoint_and_exhaustive(dimension):
    mesh, split_coordinate = coarse_mesh(dimension)

    masks = partition_cells(mesh, 2, split_coordinate=split_coordinate)

    assert len(masks) == 2
    coverage = sum(np.asarray(mask, dtype=np.int8) for mask in masks)
    assert np.all(coverage == 1)
    assert all(np.any(np.asarray(mask)) for mask in masks)


@pytest.mark.parametrize("dimension", contract.SUPPORTED_DIMENSIONS)
def test_four_rank_stripes_are_disjoint_and_exhaustive(dimension):
    mesh, split_coordinate = coarse_mesh(dimension)

    masks = partition_cells(mesh, 4, split_coordinate=split_coordinate)

    assert len(masks) == 4
    coverage = sum(np.asarray(mask, dtype=np.int8) for mask in masks)
    assert np.all(coverage == 1)
    assert all(np.any(np.asarray(mask)) for mask in masks)


def test_a_split_outside_the_domain_leaves_an_empty_partition():
    mesh, _ = coarse_mesh(2)

    with pytest.raises(ValueError, match="empty"):
        partition_cells(mesh, 2, split_coordinate=-1.0)


def test_split_coordinate_follows_the_domain():
    """驱动脚本里那句 ``0.5 * (domain[0] + domain[1])`` 确实落在区域中点上"""

    _, split_coordinate = coarse_mesh(2)
    assert split_coordinate == 0.5

    shifted = (2.0, 6.0, 0.0, 1.0)
    assert 0.5 * (shifted[0] + shifted[1]) == 4.0


def test_strategy_label_preserves_the_recorded_wording():
    assert partition_strategy_label(1, 0.5) == "all-cells"
    assert partition_strategy_label(2, 0.5) == (
        "non-overlapping-cells-split-at-x=0.5"
    )
    assert partition_strategy_label(2, 4.0, axis=1) == (
        "non-overlapping-cells-split-at-y=4"
    )
    assert partition_strategy_label(4, 0.5) == (
        "non-overlapping-cells-striped-along-x-into-4-parts"
    )
