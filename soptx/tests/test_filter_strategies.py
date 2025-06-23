from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike
from fealpy.functionspace import LagrangeFESpace

from soptx.filter.filter_strategies import (
                                        NoneStrategy, 
                                        SensitivityStrategy, 
                                        DensityStrategy, 
                                        HeavisideDensityStrategy,
                                    )

bm.set_backend('pytorch')



def test_get_initial_density(rho: TensorLike) -> None:
    strategy = HeavisideDensityStrategy(H=None, cell_measure=None, normalize_factor=None)

    rho_phys = bm.zeros_like(rho)
    strategy.get_initial_density(x=rho, xPhys=rho_phys)
    print("-------------------------------")

if __name__ == "__main__":

    mesh = QuadrangleMesh.from_box(
                            box=[0, 1, 0, 1],
                            nx=10, ny=10,
                            device='cuda'
                        )
    space_D = LagrangeFESpace(mesh=mesh, p=0, ctype='D')
    node = mesh.entity('node')
    kwargs = bm.context(node)
    @cartesian
    def density_func(x: TensorLike):
        val = 1 * bm.ones(x.shape[0], **kwargs)
        return val
    rho = space_D.interpolate(u=density_func)
    test_get_initial_density(rho[:])