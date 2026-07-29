"""Problem selection for the 2D plane-strain PINN baseline.

The manufactured problem remains a SOPTX model so the example does not copy a
second implementation of its material parameters, body force, or boundary data.
"""

from soptx.model.linear_elasticity_2d import TriSolHomoDirHuZhang2d


def make_default_problem() -> TriSolHomoDirHuZhang2d:
    """Return the all-Dirichlet plane-strain manufactured problem for stage 1."""
    return TriSolHomoDirHuZhang2d()
