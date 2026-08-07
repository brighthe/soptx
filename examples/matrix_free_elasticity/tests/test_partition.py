"""Tests for cell partitioning and its recorded strategy label."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import utils.contract as contract
from cases import create_case
from utils.distributed import partition_cells, partition_strategy_label


def coarse_mesh(dimension: int):
    case = create_case(dimension)
    resolution = (contract.REFINEMENTS[dimension][0],) * dimension
    return case, case.create_mesh(resolution)


@pytest.mark.parametrize("dimension", contract.SUPPORTED_DIMENSIONS)
def test_single_rank_owns_every_cell(dimension):
    case, mesh = coarse_mesh(dimension)

    masks = partition_cells(
        mesh,
        1,
        split_coordinate=case.partition_split_coordinate(),
    )

    assert len(masks) == 1
    assert np.all(masks[0])


@pytest.mark.parametrize("dimension", contract.SUPPORTED_DIMENSIONS)
def test_two_ranks_are_disjoint_and_exhaustive(dimension):
    case, mesh = coarse_mesh(dimension)

    masks = partition_cells(
        mesh,
        2,
        split_coordinate=case.partition_split_coordinate(),
    )

    assert len(masks) == 2
    coverage = sum(mask.astype(np.int8) for mask in masks)
    assert np.all(coverage == 1)
    assert all(np.any(mask) for mask in masks)


def test_unsupported_rank_count_is_rejected():
    case, mesh = coarse_mesh(2)

    with pytest.raises(ValueError, match="ranks"):
        partition_cells(
            mesh,
            3,
            split_coordinate=case.partition_split_coordinate(),
        )


def test_a_split_outside_the_domain_leaves_an_empty_partition():
    _, mesh = coarse_mesh(2)

    with pytest.raises(ValueError, match="empty"):
        partition_cells(mesh, 2, split_coordinate=-1.0)


def test_split_coordinate_follows_the_domain():
    case = create_case(2)
    assert case.partition_split_coordinate() == 0.5

    shifted = replace(case, domain=(2.0, 6.0, 0.0, 1.0))
    assert shifted.partition_split_coordinate() == 4.0
    assert shifted.partition_split_coordinate(axis=1) == 0.5

    with pytest.raises(ValueError, match="axis"):
        case.partition_split_coordinate(axis=2)


def test_strategy_label_preserves_the_recorded_wording():
    assert partition_strategy_label(1, 0.5) == "all-cells"
    assert partition_strategy_label(2, 0.5) == (
        "non-overlapping-cells-split-at-x=0.5"
    )
    assert partition_strategy_label(2, 4.0, axis=1) == (
        "non-overlapping-cells-split-at-y=4"
    )
