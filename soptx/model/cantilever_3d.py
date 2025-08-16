from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import HexahedronMesh, TetrahedronMesh

from soptx.pde.pde_base import PDEBase

class CantileverBeam3dData(PDEBase):
    '''
    3D Cantilever Beam Problem
    
    -∇·σ = b    in Ω
       u = 0    on ∂Ω (left boundary fixed)
    where:
    - σ is the stress tensor
    - ε = (∇u + ∇u^T)/2 is the strain tensor
    
    Material parameters:
        E = 1, nu = 0.3
        
    For isotropic materials:
        σ = 2με + λtr(ε)I
        
    Geometry:
           3------- 7
         / |       /|
        1 ------- 5 |
        |  |      | |
        |  2------|-6
        | /       |/
        0 ------- 4
    '''
    def __init__(self,
                domain: List[float] = [0, 60, 0, 20, 0, 4],
                mesh_type: str = 'uniform_hex',
                T: float = -1.0,  # 负值代表方向向下
                E: float = 1.0, 
                nu: float = 0.3,
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:

        super().__init__(domain=domain, mesh_type=mesh_type, 
                        enable_logging=enable_logging, logger_name=logger_name)
        
        self._T = T
        self._E = E
        self._nu = nu
        
        self._eps = 1e-12
        self._plane_type = '3d'
        self._force_type = 'concentrated'
        self._boundary_type = 'dirichlet'

        self._log_info(f"Initialized CantileverBeam3dData with domain={self._domain}, "
                f"mesh_type='{mesh_type}', force={T}, E={E}, nu={nu}, "
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
    
    @variantmethod('uniform_hex')
    def init_mesh(self, **kwargs) -> HexahedronMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 20)
        nz = kwargs.get('nz', 4)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')
        
        mesh = HexahedronMesh.from_box(box=self._domain, nx=nx, ny=ny, nz=nz,
                                      threshold=threshold, device=device)
        
        self._save_meshdata(mesh, 'uniform_hex', nx=nx, ny=ny, nz=nz)
        
        return mesh
    
    @init_mesh.register('uniform_tet')
    def init_mesh(self, **kwargs) -> TetrahedronMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 20)
        nz = kwargs.get('nz', 4)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')
        
        mesh = TetrahedronMesh.from_box(box=self._domain, nx=nx, ny=ny, nz=nz,
                                       threshold=threshold, device=device)
        
        self._save_meshdata(mesh, 'uniform_tet', nx=nx, ny=ny, nz=nz)
        
        return mesh
    
    ###############################################################################################
    # 核心方法
    ###############################################################################################
    
    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        
        coord = (
            (bm.abs(x - self._domain[1]) < self._eps) & 
            (bm.abs(y - self._domain[0]) < self._eps)
        )
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        # 在 y 方向施加载荷
        val = bm.set_at(val, (coord, 1), self._T)
        
        return val
    
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain

        x = points[..., 0]
        
        coord = bm.abs(x - domain[0]) < self._eps
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain

        x = points[..., 0]
        
        coord = bm.abs(x - domain[0]) < self._eps
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_z(self, points: TensorLike) -> TensorLike:
        domain = self.domain

        x = points[..., 0]
        
        coord = bm.abs(x - domain[0]) < self._eps
        
        return coord
    
    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y,
                self.is_dirichlet_boundary_dof_z)