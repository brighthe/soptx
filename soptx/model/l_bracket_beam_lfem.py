from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh, HomogeneousMesh
from fealpy.mesher import LshapeMesher

from soptx.model.pde_base import PDEBase  

class LBracketBeam2d(PDEBase):
    def __init__(self,
                domain: List[float] = [0, 1, 0, 1],
                hole_domain: List[float] = [0.4, 1, 0.4, 1],
                mesh_type: str = 'uniform_quad',
                P: float = -2.0, # N
                E: float = 7e4,  # MPa (N/mm^2)
                nu: float = 0.25,
                load_height_ratio: float = 0.85,  # 载荷施加位置参数
                plane_type: str = 'plane_stress', # 'plane_stress' or 'plane_strain'
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)

        # 几何参数
        self._domain = domain
        self._hole_domain = hole_domain
        
        self._P = P
        self._E, self._nu = E, nu
        self._plane_type = plane_type

        self._eps = 1e-8

        self._load_height_ratio = load_height_ratio 

        self._load_type = 'concentrated'     
        self._boundary_type = 'mixed'

    @property
    def E(self) -> float:
        """获取杨氏模量"""
        return self._E
    
    @property
    def nu(self) -> float:
        """获取泊松比"""
        return self._nu
    
    @property
    def P(self) -> float:
        """获取总载荷"""
        return self._P

    @variantmethod('tri_threshold')
    def init_mesh(self, nx: int = 10, ny: int = 10) -> TriangleMesh:
        
        big_box = self._domain
        small_box = self._hole_domain

        def threshold(p):
            x = p[..., 0]
            y = p[..., 1]
            return ((x>=small_box[0])
                   &(x<=small_box[1])
                   &(y>=small_box[2])
                   &(y<=small_box[3]))

        l_shape_mesh = TriangleMesh.from_box(big_box,
                                             nx=nx, ny=ny,
                                             threshold=threshold)
        
        self._save_meshdata(l_shape_mesh, 'tri_threshold', nx=nx, ny=ny)

        return l_shape_mesh
    
    @init_mesh.register('quad_threshold')
    def init_mesh(self, nx: int = 10, ny: int = 10) -> QuadrangleMesh:
        
        big_box = self._domain
        small_box = self._hole_domain
    
        def threshold(p):
            x = p[..., 0]
            y = p[..., 1]
            return ((x>=small_box[0])
                   &(x<=small_box[1])
                   &(y>=small_box[2])
                   &(y<=small_box[3]))

        l_shape_mesh = QuadrangleMesh.from_box(big_box,
                                            nx=nx, ny=ny,
                                            threshold=threshold)
                                             
        self._save_meshdata(l_shape_mesh, 'quad_threshold', nx=nx, ny=ny)

        return l_shape_mesh
    
    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        """体力密度 b(x, y)"""
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        
        return bm.abs(y - domain[3]) < self._eps
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        
        return bm.abs(y - domain[3]) < self._eps
    
    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x,
                self.is_dirichlet_boundary_dof_y)
    
    @cartesian
    def concentrate_load_bc(self, points: TensorLike) -> TensorLike:
        """节点载荷边界条件
    
        返回总载荷值，实际使用时会自动均分到所有满足条件的节点上"""
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val = bm.set_at(val, (..., 1), self._P)

        return val
    
    @cartesian
    def is_concentrate_load_boundary_dof(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        
        # 右边缘条件
        on_right_edge = bm.abs(x - domain[1]) < self._eps

        L = domain[1] - domain[0]
        
        # 载荷高度条件
        # L型下部矩形高度为 2/5 = 0.4
        lower_rect_height = (2.0 / 5.0) * L
        load_threshold = self._load_height_ratio * lower_rect_height
        
        # y > 0.34 (即在 y ∈ (0.34, 0.4] 区间)
        in_load_region = y > load_threshold
        
        return on_right_edge & in_load_region
    
    def is_concentrate_load_boundary(self) -> Callable:

        return self.is_concentrate_load_boundary_dof