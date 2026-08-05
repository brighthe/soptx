from __future__ import annotations

import subprocess
import sys
from textwrap import indent

import pytest


COMPATIBILITY_CASES = (
    (
        "analysis",
        """
from soptx.analysis.integrators.linear_elastic_integrator import (
    LinearElasticIntegrator as legacy,
)
from soptx.fem.integrators import LinearElasticIntegrator as current
assert legacy is current
""",
    ),
    (
        "functionspace",
        """
from soptx.functionspace.huzhang_fe_space_2d import (
    HuZhangFESpace2d as legacy,
)
from soptx.fem.spaces import HuZhangFESpace2d as current
assert legacy is current
""",
    ),
    (
        "interpolation",
        """
from soptx.interpolation.linear_elastic_material import (
    IsotropicLinearElasticMaterial as legacy,
)
from soptx.materials import IsotropicLinearElasticMaterial as current
assert legacy is current
""",
    ),
    (
        "optimization",
        """
from soptx.optimization.mma_optimizer import MMAOptimizer as legacy
from soptx.topology.optimizers import MMAOptimizer as current
assert legacy is current
""",
    ),
    (
        "regularization",
        """
from soptx.regularization.filter import Filter as legacy
from soptx.topology.filters import Filter as current
assert legacy is current
""",
    ),
    (
        "utils",
        """
from soptx.utils.base_logged import BaseLogged as legacy
from soptx.core import BaseLogged as current
assert legacy is current
""",
    ),
    (
        "model",
        """
from soptx.model.linear_elasticity_2d import (
    TriSolHomoDirHuZhang2d,
)
from soptx.problems import (
    ExponentialSineManufacturedElasticity2D,
)
assert TriSolHomoDirHuZhang2d.__name__ == "TriSolHomoDirHuZhang2d"
assert (
    ExponentialSineManufacturedElasticity2D.__name__
    == "ExponentialSineManufacturedElasticity2D"
)

from soptx.model.mbb_beam_2d_lfem import HalfMBBBeamRight2d
legacy_mbb = HalfMBBBeamRight2d()
mesh = legacy_mbb.init_mesh(nx=2, ny=1)
assert mesh.number_of_cells() == 2
""",
    ),
)


@pytest.mark.parametrize(("namespace", "imports"), COMPATIBILITY_CASES)
def test_public_compatibility_import_warns_once(
    namespace: str,
    imports: str,
) -> None:
    indented_imports = indent(imports.strip(), "    ")
    source = f"""
import warnings

with warnings.catch_warnings(record=True) as recorded:
    warnings.simplefilter("always", DeprecationWarning)
{indented_imports}

deprecations = [
    item for item in recorded
    if issubclass(item.category, DeprecationWarning)
]
assert len(deprecations) == 1, [
    str(item.message) for item in deprecations
]
assert "soptx.{namespace}" in str(deprecations[0].message)
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
