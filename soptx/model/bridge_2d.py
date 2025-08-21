from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.pde.pde_base import PDEBase  

class Bridge2dData(PDEBase):
    '''
    -∇·σ = b    in Ω
       u = 0    on ∂Ω_D (两侧固支边界)
    where:
    - σ is the stress tensor
    - ε = (∇u + ∇u^T)/2 is the strain tensor
    
    Material parameters:
        E = 1, nu = 0.35 (可压缩材料) 或 0.5 (不可压缩材料)

    For isotropic materials:
        σ = 2με + λtr(ε)I
    '''
    def __init__(self,
                domain: List[float] = [0, 8, 0, 4],
                mesh_type: str = 'uniform_quad',
                T: float = -2.0,  # 向下的集中载荷
                E: float = 1.0, 
                nu: float = 0.35,
                support_width: float = 0.5,
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)
        
        self._T = T
        self._E, self._nu = E, nu

        self._support_width = support_width

        self._eps = 1e-12
        self._plane_type = 'plane_strain'
        self._force_type = 'concentrated'
        self._boundary_type = 'dirichlet'

        self._log_info(f"Initialized Bridge2dData with domain={self._domain}, "
                f"mesh_type='{mesh_type}', force={T}, E={E}, nu={nu}, "
                f"plane_type='{self._plane_type}', "
                f"force_type='{self._force_type}', "
                f"boundary_type='{self._boundary_type}'")


    #######################################################################################################################
    # 访问器
    #######################################################################################################################

    @property
    def E(self) -> float:
        """获取杨氏模量"""
        return self._E
    
    @property
    def nu(self) -> float:
        """获取泊松比"""
        return self._nu
    
    @property
    def T(self) -> float:
        """获取集中力"""
        return self._T
    
    #######################################################################################################################
    # 变体方法
    #######################################################################################################################
    
    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 80) 
        ny = kwargs.get('ny', 40)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)

        return mesh
    
    @init_mesh.register('uniform_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 80)
        ny = kwargs.get('ny', 40)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)
        
        self._save_meshdata(mesh, 'uniform_tri', nx=nx, ny=ny)

        return mesh


    #######################################################################################################################
    # 核心方法
    #######################################################################################################################

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        """定义体力（集中载荷）"""
        domain = self.domain

        x, y = points[..., 0], points[..., 1]   

        mid_x = (domain[0] + domain[1]) / 2
        coord = (
            (bm.abs(x - mid_x) < self._eps) & 
            (bm.abs(y - domain[3]) < self._eps)
        )
        
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        # 在y方向施加向下的力
        val = bm.set_at(val, (coord, 1), self._T)
        
        return val
    
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)
        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        left_corner = (
            (x < domain[0] + self._support_width) & 
            (bm.abs(y - domain[2]) < self._eps)
        )
        
        right_corner = (
            (x > domain[1] - self._support_width) & 
            (bm.abs(y - domain[2]) < self._eps)
        )
        
        coord = left_corner | right_corner
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        left_corner = (
            (x < domain[0] + self._support_width) & 
            (bm.abs(y - domain[2]) < self._eps)
        )
        
        right_corner = (
            (x > domain[1] - self._support_width) & 
            (bm.abs(y - domain[2]) < self._eps)
        )
        
        coord = left_corner | right_corner
        
        return coord
    
    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)