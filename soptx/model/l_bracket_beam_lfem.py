from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh, HomogeneousMesh

from soptx.model.pde_base import PDEBase  

class LBracketMiddle2d(PDEBase):
    def __init__(self,
                domain: List[float] = [0, 1, 0, 1],
                hole_domain: List[float] = [0.4, 1, 0.4, 1],
                mesh_type: str = 'uniform_quad',
                P: float = -2.0, # N
                E: float = 7e4,  # MPa (N/mm^2)
                nu: float = 0.25,
                load_width: Optional[float] = None,  # 载荷分布宽度，None 时为单点载荷
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

        self._load_width = load_width 

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

    @variantmethod('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        device = kwargs.get('device', 'cpu')

        
        big_box = self._domain
        small_box = self._hole_domain

        def threshold(p):
            x = p[..., 0]
            y = p[..., 1]
            return ((x>=small_box[0])
                   &(x<=small_box[1])
                   &(y>=small_box[2])
                   &(y<=small_box[3]))

        l_shape_mesh = TriangleMesh.from_box_cross_mesh(big_box,
                                             nx=nx, ny=ny,
                                             threshold=threshold,
                                             device=device)
                                             
        
        self._save_meshdata(l_shape_mesh, 'tri_threshold', nx=nx, ny=ny)

        return l_shape_mesh
    
    @init_mesh.register('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        device = kwargs.get('device', 'cpu')
        
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
                                            threshold=threshold, 
                                            device=device)
                                             
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

        # 右侧边缘下部矩形的中点高度
        # 下部矩形 y 范围为 [domain[2], hole_domain[2]]，中点为二者均值
        middle_y = (domain[2] + self._hole_domain[2]) / 2.0
        on_right_edge = bm.abs(x - domain[1]) < self._eps

        if self._load_width is None:
            # 单点载荷模式：集中力施加在右侧边缘中点处
            coord = on_right_edge & (bm.abs(y - middle_y) < self._eps)
        else:
            # 分布载荷模式：以中点为中心，上下各扩展 load_width / 2
            # 闭区间 y ∈ [middle_y - load_width/2, middle_y + load_width/2]
            half_width = self._load_width / 2.0
            coord = (
                on_right_edge &
                (y >= middle_y - half_width - self._eps) &
                (y <= middle_y + half_width + self._eps)
            )

        return coord
    
    def is_concentrate_load_boundary(self) -> Callable:

        return self.is_concentrate_load_boundary_dof
    

class LBracketCorner2d(PDEBase):
    def __init__(self,
                domain: List[float] = [0, 1, 0, 1],
                hole_domain: List[float] = [0.4, 1, 0.4, 1],
                mesh_type: str = 'uniform_quad',
                P: float = -2.0, # N
                E: float = 7e4,  # MPa (N/mm^2)
                nu: float = 0.25,
                load_width: Optional[float] = None,  # 载荷分布宽度，None 时为单点载荷
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

        self._load_width = load_width 

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

    @variantmethod('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        device = kwargs.get('device', 'cpu')

        
        big_box = self._domain
        small_box = self._hole_domain

        def threshold(p):
            x = p[..., 0]
            y = p[..., 1]
            return ((x>=small_box[0])
                   &(x<=small_box[1])
                   &(y>=small_box[2])
                   &(y<=small_box[3]))

        l_shape_mesh = TriangleMesh.from_box_cross_mesh(big_box,
                                             nx=nx, ny=ny,
                                             threshold=threshold,
                                             device=device)
                                             
        
        self._save_meshdata(l_shape_mesh, 'tri_threshold', nx=nx, ny=ny)

        return l_shape_mesh
    
    @init_mesh.register('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        device = kwargs.get('device', 'cpu')
        
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
                                            threshold=threshold, 
                                            device=device)
                                             
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

        # 内角高度：hole_domain 的下边界，即 y = hole_domain[2]
        corner_y = self._hole_domain[2]   # 默认为 0.4
        on_right_edge = bm.abs(x - domain[1]) < self._eps

        if self._load_width is None:
            # 单点载荷模式：仅在内角处施加
            coord = on_right_edge & (bm.abs(y - corner_y) < self._eps)
        else:
            # 分布载荷模式：闭区间 y ∈ [corner_y - load_width, corner_y]
            # 总力 P 由上层代码均分到所有节点，load_width 决定覆盖范围
            coord = (
                on_right_edge &
                (y >= corner_y - self._load_width - self._eps) &
                (y <= corner_y + self._eps)
            )

        return coord
    
    def is_concentrate_load_boundary(self) -> Callable:

        return self.is_concentrate_load_boundary_dof