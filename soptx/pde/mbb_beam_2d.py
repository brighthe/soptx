from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import Mesh, QuadrangleMesh, TriangleMesh

class HalfMBBBeam2dData1:
    '''模型来源论文: Efficient topology optimization in MATLAB using 88 lines of code'''
    def __init__(self,
                domain: List[float] = [0, 60, 0, 20],
                T: float = -1.0, # 负值代表方向向下
                E: float = 1.0,
                nu: float = 0.3,
            ) -> None:
        self.domain = domain
        self.T = T
        self.E = E
        self.nu = nu
        self.eps = 1e-12
        self.plane_type = 'plane_stress'
    
    @variantmethod('hex')
    def create_mesh(self,
                    mesh_type: str = 'triangle', 
                    nx: int = 60, ny: int = 20, 
                    threshold: Optional[Callable] = None, 
                    device: str = None
                ) -> Mesh:
        box = self.domain

        if mesh_type == 'quadrangle':
            mesh = QuadrangleMesh.from_box(box=box, nx=nx, ny=ny,
                                        threshold=threshold, device=device)
        elif mesh_type == 'triangle':
            mesh = TriangleMesh.from_box(box=box, nx=nx, ny=ny,
                                        threshold=threshold, device=device)
        else:
            raise ValueError(f"Unsupported mesh type: {mesh_type}")

        hx = (box[1] - box[0]) / nx
        hy = (box[3] - box[2]) / ny
        
        mesh.meshdata['domain'] = box
        mesh.meshdata['nx'] = nx
        mesh.meshdata['ny'] = ny
        mesh.meshdata['hx'] = hx
        mesh.meshdata['hy'] = hy
        mesh.meshdata['mesh_type'] = mesh_type
    
        return mesh
    
    @cartesian
    def force(self, points: TensorLike) -> TensorLike:
        domain = self.domain

        x = points[..., 0]
        y = points[..., 1]

        coord = (
            (bm.abs(x - domain[0]) < self.eps) & 
            (bm.abs(y - domain[3]) < self.eps)
        )
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val = bm.set_at(val, (coord, 1), self.T)

        return val
    
    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain

        x = points[..., 0]
        y = points[..., 1]

        coord = ((bm.abs(x - domain[1]) < self.eps) &
                 (bm.abs(y - domain[2]) < self.eps))
        
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

    
