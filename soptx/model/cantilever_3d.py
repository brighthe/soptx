from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import HexahedronMesh, TetrahedronMesh

from soptx.model.pde_base import PDEBase

class CantileverBeam3d(PDEBase):
    '''
    3D 悬臂梁的 PDE 模型
    
    控制方程:
        -∇·σ = 0      in Ω
            u = u_D    on ∂F_D
          σ·n = t      on ∂F_N 
    
    设计域:
        - 全设计域: 60 mm x 20 mm x 4 mm
    
    边界条件:
        - 左端面 (x=0): 位移边界条件 u = 0
        - 右端面底边 (x=60, y=0): 施加向下的均布线载荷 p = -1 [N/mm]

    材料参数:
        E = 1 [MPa], nu = 0.3
        
    几何:
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
                p: float = -1.0,  # N
                E: float = 1.0,   # MPa
                nu: float = 0.3,
                plane_type: str = '3d',
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:

        super().__init__(domain=domain, mesh_type=mesh_type, 
                        enable_logging=enable_logging, logger_name=logger_name)
        
        self._p = p
        self._E, self._nu = E, nu
        self._plane_type = plane_type
        
        self._eps = 1e-12
        self._load_type = 'concentrated'
        self._boundary_type = 'mixed'
    

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
    def p(self) -> float:
        """获取点力"""
        return self._p
    
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
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
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
    
    @cartesian
    def concentrate_load_bc(self, points: TensorLike) -> TensorLike:
        """均布线载荷 (类似二维点力)"""
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val = bm.set_at(val, (..., 1), self._p)

        return val
    
    @cartesian
    def is_concentrate_load_boundary_dof(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x = points[..., 0]
        y = points[..., 1]

        on_right_boundary = bm.abs(x - domain[1]) < self._eps
        on_bottom_boundary = bm.abs(y - domain[0]) < self._eps
        
        return on_right_boundary & on_bottom_boundary
    
    def is_concentrate_load_boundary(self) -> Callable:

        return self.is_concentrate_load_boundary_dof