from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.pde.pde_base import PDEBase  

class HalfMBBBeam2dData(PDEBase):
    '''
    -∇·σ = b    in Ω
       u = 0    on ∂Ω (homogeneous Dirichlet)
    where:
    - σ is the stress tensor
    - ε = (∇u + ∇u^T)/2 is the strain tensor
    
    Material parameters:
        E = 1, nu = 0.3

    For isotropic materials:
        σ = 2με + λtr(ε)I
    '''
    def __init__(self,
                domain: List[float] = [0, 60, 0, 20],
                mesh_type: str = 'uniform_quad',
                T: float = -1.0, # 负值代表方向向下
                E: float = 1.0, nu: float = 0.3,
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)
        
        self._T = T
        self._E, self._nu = E, nu

        self._eps = 1e-12
        self._plane_type = 'plane_stress'
        self._force_type = 'concentrated'
        self._boundary_type = 'dirichlet'

        self._log_info(f"Initialized BoxTriLagrangeData2d with domain={self._domain}, "
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
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 20)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)

        return mesh
    
    @init_mesh.register('uniform_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 20)
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
        domain = self.domain

        x, y = points[..., 0], points[..., 1]   

        coord = (
            (bm.abs(x - domain[0]) < self._eps) & 
            (bm.abs(y - domain[3]) < self._eps)
        )
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val = bm.set_at(val, (coord, 1), self._T)
# 
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
        y = points[..., 1]

        coord = ((bm.abs(x - domain[1]) < self._eps) &
                 (bm.abs(y - domain[2]) < self._eps))
        
        return coord
    
    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)