from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import Mesh, write_mesh_to_vtu

from utils import contract

if TYPE_CHECKING:
    from cases import ElasticityCase


def solution_error(
    mesh: Mesh,
    solution,
    problem,
    degree: int,
) -> tuple[float, float]:
    """Absolute and exact-norm-relative L2 displacement error."""

    absolute = float(
        mesh.error(
            problem.disp_solution,
            solution,
            q=degree + 3,
        )
    )

    def zero_field(points):
        return bm.zeros_like(problem.disp_solution(points))

    exact_norm = float(
        mesh.error(
            problem.disp_solution,
            zero_field,
            q=degree + 3,
        )
    )
    return absolute, absolute / max(exact_norm, contract.NORM_FLOOR)


def write_solution(
    filename: Path,
    mesh: Mesh,
    space,
    solution,
    case: ElasticityCase,
) -> None:
    """Write barycentric displacement and error to a VTU file."""

    barycenter = np.array([case.barycentric_coordinate()])
    numerical = np.asarray(
        space.value(solution, barycenter)
    )[:, 0, :]
    exact = np.asarray(
        case.problem.disp_solution(mesh.Entity("cell").barycenter())
    )
    mesh.Entity("cell").set_attribute("displacement", numerical)
    mesh.Entity("cell").set_attribute(
        "displacement_error",
        numerical - exact,
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    write_mesh_to_vtu(
        str(filename),
        mesh,
        entity_names=[case.mesh_entity_name],
    )
