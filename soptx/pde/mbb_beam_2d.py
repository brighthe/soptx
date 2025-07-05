from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

class HalfMBBBeam2dData1:
    '''
    模型来源:
    https://wnesm678i4.feishu.cn/wiki/Xi3dw6mzNi6V9ckNcoScfmAXnFg#part-TuQydJUDvojFG6x0NPzcH3stnTh
    '''
    def __init__(self,
                domain: List[float] = [0, 60, 0, 20],
                mesh_method:Optional[str] = None,
                T: float = -1.0, # 负值代表方向向下
                E: float = 1.0,
                nu: float = 0.3,
            ) -> None:
        """
        1. 实例化时设置默认网格变体方法
        hmb = HalfMBBBeam2dData1(mesh_method='uniform_quad')

        2. 直接使用默认方法生成网格
        mesh1 = hmb.init_mesh(nx=60, ny=20)  # 生成四边形网格

        3. 切换到其他网格方法
        hmb.init_mesh.set('uniform_tri')     # 设置变体 (返回 None)
        mesh2 = hmb.init_mesh(nx=30, ny=10)  # 生成三角形网格
        
        注意: 
        - init_mesh.set() 只设置变体，不执行方法，返回 None
        - 需要分别调用 set() 和 init_mesh() 来生成网格
        - 每次 set() 后，后续的 init_mesh() 调用都使用新设置的变体
        """
        self.domain = domain
        self.init_mesh.set(mesh_method)
        
        self.T = T
        self.E, self.nu = E, nu

        self.eps = 1e-12
        self.plane_type = 'plane_stress'
        self.force_type = 'concentrated'
        self.boundary_type = 'dirichlet'
    
    @variantmethod('uniform_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 20)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', None)

        mesh = TriangleMesh.from_box(box=self.domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)

        hx = (self.domain[1] - self.domain[0]) / nx
        hy = (self.domain[3] - self.domain[2]) / ny

        mesh.meshdata['domain'] = self.domain
        mesh.meshdata['nx'] = nx
        mesh.meshdata['ny'] = ny
        mesh.meshdata['hx'] = hx
        mesh.meshdata['hy'] = hy
        mesh.meshdata['mesh_type'] = self.init_mesh.vm.get_key(self)
    
        return mesh
    
    @init_mesh.register('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 20)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', None)

        mesh = QuadrangleMesh.from_box(box=self.domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)

        hx = (self.domain[1] - self.domain[0]) / nx
        hy = (self.domain[3] - self.domain[2]) / ny

        mesh.meshdata['domain'] = self.domain
        mesh.meshdata['nx'] = nx
        mesh.meshdata['ny'] = ny
        mesh.meshdata['hx'] = hx
        mesh.meshdata['hy'] = hy
        mesh.meshdata['mesh_type'] = self.init_mesh.vm.get_key(self)

        return mesh

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
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
    def displacement_solution(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_displacement_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    @cartesian
    def is_displacement_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain

        x = points[..., 0]
        y = points[..., 1]

        coord = ((bm.abs(x - domain[1]) < self.eps) &
                 (bm.abs(y - domain[2]) < self.eps))
        
        return coord
    
    def threshold(self) -> Tuple[Callable, Callable]:
        return (self.is_displacement_boundary_dof_x, 
                self.is_displacement_boundary_dof_y)


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

    
