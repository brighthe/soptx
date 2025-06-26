from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike, Callable

class PolyDisp2dData:
    """多项式位移算例"""
    def __init__(self) -> None:
        pass

    def domain(self) -> list:
        return [0, 1, 0, 1]

    @cartesian
    def source(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        val = bm.zeros(points.shape, dtype=bm.float64)
        val[..., 0] = 35/13*y - 35/13*y**2 + 10/13*x - 10/13*x**2
        val[..., 1] = -25/26*(-1+2*y) * (-1+2*x)

        return val

    @cartesian
    def solution(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        val = bm.zeros(points.shape, dtype=bm.float64)
        val[..., 0] = x*(1-x)*y*(1-y)
        val[..., 1] = 0

        return val

    @cartesian
    def dirichlet(self, points: TensorLike) -> Callable[[TensorLike], TensorLike]:
        
        return self.solution(points)

    @cartesian
    def is_dirichlet_boundary(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        flag1 = bm.abs(x - self.domain()[0]) < 1e-13
        flag2 = bm.abs(x - self.domain()[1]) < 1e-13
        flagx = bm.logical_or(flag1, flag2)
        flag3 = bm.abs(y - self.domain()[2]) < 1e-13
        flag4 = bm.abs(y - self.domain()[3]) < 1e-13
        flagy = bm.logical_or(flag3, flag4)
        flag = bm.logical_or(flagx, flagy)

        return flag
    
class BoxTriData2d:
    def __init__(self, E: float = 1.0, nu: float = 0.3) -> None:
        self.E = E
        self.nu = nu
        self.eps = 1e-12
        self.plane_type = 'plane_strain'

    def domain(self) -> list:
        return [0, 1, 0, 1]

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi

        f_x = 22.5 * pi**2 / 13 * bm.sin(pi * x) * bm.sin(pi * y)
        f_y = -12.5 * pi**2 / 13 * bm.cos(pi * x) * bm.cos(pi * y)

        f = bm.stack([f_x, f_y], axis=-1)
        
        return f

        # val = bm.zeros(points.shape, dtype=points.dtype, device=bm.device_put(points))
        # val[..., 0] = 22.5*pi**2/13 * bm.sin(pi*x) * bm.sin(pi*y)
        # val[..., 1] = -12.5*pi**2/13 * bm.cos(pi*x) * bm.cos(pi*y)

        # return val

    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi

        u_x = bm.sin(pi * x) * bm.sin(pi * y)
        u_y = bm.zeros_like(x) 

        u = bm.stack([u_x, u_y], axis=-1)

        return u

        # val = bm.zeros(points.shape, dtype=points.dtype, device=bm.device_put(points))
        # val[..., 0] = bm.sin(pi*x) * bm.sin(pi*y)
        # val[..., 1] = 0

        # return val

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        val = self.disp_solution(points)
        
        return val

    @cartesian
    def is_dirichlet_boundary(self, points: TensorLike) -> TensorLike:
        domain = self.domain()
        x, y = points[..., 0], points[..., 1]

        flag_x0 = bm.abs(x - domain[0]) < self.eps
        flag_x1 = bm.abs(x - domain[1]) < self.eps
        flag_y0 = bm.abs(y - domain[2]) < self.eps
        flag_y1 = bm.abs(y - domain[3]) < self.eps
        
        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1
        
        return flag