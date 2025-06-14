from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from fealpy.decorator import cartesian

from typing import Tuple, Callable
from builtins import list

class HalfMBBBeam2dData1:
    '''模型来源论文: Efficient topology optimization in MATLAB using 88 lines of code'''
    def __init__(
            self, 
            xmin: float=0, xmax: float=60, 
            ymin: float=0, ymax: float=20,
            T: float = 1
        ) -> None:
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.T = T
        self.eps = 1e-12

    def domain(self) -> list:
        
        box = [self.xmin, self.xmax, self.ymin, self.ymax]

        return box
    
    @cartesian
    def force(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        coord = (
            (bm.abs(x - domain[0]) < self.eps) & 
            (bm.abs(y - domain[3]) < self.eps)
        )
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        # val[coord, 1] = self.T
        val = bm.set_at(val, (coord, 1), -self.T)

        return val
    
    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        coord = ((bm.abs(x - domain[1]) < self.eps) &
                 (bm.abs(y - domain[0]) < self.eps))
        
        return coord
    
    def threshold(self) -> Tuple[Callable, Callable]:
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    

class MBBBeam2dData2:
    '''模型来源论文: Topology optimization using the p-version of the finite element method'''
    def __init__(
            self, 
            xmin: float=0, xmax: float=60, 
            ymin: float=0, ymax: float=10,
            T: float = 1
        ) -> None:
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.T = T
        self.eps = 1e-12

    def domain(self) -> list:
        
        box = [self.xmin, self.xmax, self.ymin, self.ymax]

        return box
    
    @cartesian
    def force(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        coord = (bm.abs(x - domain[1] / 2) < self.eps) & (bm.abs(y - domain[3]) < self.eps)
        
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val = bm.set_at(val, (coord, 1), -self.T)

        return val
    
    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        coord = (bm.abs(x - domain[0]) < self.eps) & (bm.abs(y - domain[2]) < self.eps)
        
        return coord

    @cartesian  
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]
        
        left_support = (bm.abs(x - domain[0]) < self.eps) & (bm.abs(y - domain[2]) < self.eps)
        right_support = (bm.abs(x - domain[1]) < self.eps) & (bm.abs(y - domain[2]) < self.eps)
        
        coord = left_support | right_support

        return coord
    
    def threshold(self) -> Tuple[Callable, Callable]:
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    
class HalfMBBBeam2dData2:
    '''模型来源论文: Topology Optimization of Structures Using Higher Order Finite Elements in Analysis'''
    def __init__(
            self, 
            xmin: float=0, xmax: float=120, 
            ymin: float=0, ymax: float=40,
            T: float = 1
        ) -> None:
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.T = T
        self.eps = 1e-12

    def domain(self) -> list:
        
        box = [self.xmin, self.xmax, self.ymin, self.ymax]

        return box
    
    @cartesian
    def force(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        coord = (
            (bm.abs(x - domain[0]) < self.eps) & 
            (bm.abs(y - domain[3]) < self.eps)
        )
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        # val[coord, 1] = self.T
        val = bm.set_at(val, (coord, 1), -self.T)

        return val
    
    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        coord = (bm.abs(x - domain[0]) < self.eps) & \
                (bm.abs(y - domain[2]) < self.eps)

        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        coord1 = (bm.abs(x - domain[0]) < self.eps) & \
                (bm.abs(y - domain[2]) < self.eps)
        
        coord2 = ((bm.abs(x - domain[1]) < self.eps) &
                 (bm.abs(y - domain[0]) < self.eps))

        return coord1 | coord2

    def threshold(self) -> Tuple[Callable, Callable]:
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)

    
