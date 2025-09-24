from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.model.pde_base import PDEBase  

class ClampedBeam2D(PDEBase):
    '''
    Example from Castañar et al. (2022) paper, Section 5.1
    A clamped-clamped beam with a single-point load. This is not a bridge model.
    
    PDEs:
    -∇·σ = b   in Ω
      [cite_start]u = 0    on ∂Ω_D (fully clamped on the left and right sides) [cite: 617, 618]
      
    where:
    - σ is the stress tensor
    - ε = (∇u + ∇u^T)/2 is the strain tensor
    
    几何参数:
        矩形域, 左右两端完全固支, 底部中点施加向下的集中载荷。
        本实现考虑完整计算域，不利用对称性。
    
    Material parameters from paper:
        [cite_start]E_s = 30 Pa, nu is varied (e.g., 0.4 for compressible, 0.5 for incompressible) [cite: 620, 647]
        
    For isotropic materials:
        σ = 2με + λtr(ε)I
    '''
    def __init__(self,
                domain: List[float] = [0, 160, 0, 20], 
                mesh_type: str = 'uniform_tri', 
                T: float = -3.0,  # 负值代表方向向下 (单位 - N)
                E: float = 30.0,  # 杨氏模量 ( 单位 - Pa / N/m^2 )
                nu: float = 0.4,  
                plane_type: str = 'plane_strain', # 'plane_stress' or 'plane_strain' 
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                         enable_logging=enable_logging, logger_name=logger_name)
        
        self._T = T
        self._E, self._nu = E, nu
        self._plane_type = plane_type

        self._eps = 1e-12
        self._force_type = 'concentrated'
        self._boundary_type = 'dirichlet'

        self._log_info(f"Initialized ClampedBeam2DSingleLoad (from Castañar et al. 2022) with domain={self._domain}, "
                     f"mesh_type='{mesh_type}', force={T}, E={E}, nu={nu}, "
                     f"plane_type='{self._plane_type}', boundary='fully clamped'.")


    #######################################################################################################################
    # 访问器 (Accessors)
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
    # 变体方法 (Variant Methods)
    #######################################################################################################################
    
    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 320)
        ny = kwargs.get('ny', 40)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                      threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)

        return mesh
    
    @init_mesh.register('uniform_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 320)
        ny = kwargs.get('ny', 40)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                     threshold=threshold, device=device)
        
        self._save_meshdata(mesh, 'uniform_tri', nx=nx, ny=ny)

        return mesh


    #######################################################################################################################
    # 核心方法 (Core Methods)
    #######################################################################################################################

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        """
        定义体力 (以集中载荷形式施加)
        在底部中点施加向下的集中载荷 F = -3 (N)
        """
        domain = self.domain

        x, y = points[..., 0], points[..., 1]  

        mid_x = (domain[0] + domain[1]) / 2  
        coord = (
            (bm.abs(x - mid_x) < self._eps) & 
            (bm.abs(y - domain[2]) < self._eps)
        )
        
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        # 集中力施加在y方向
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
        
        # 判断条件：位于左右边界上
        on_left_boundary = bm.abs(x - domain[0]) < self._eps
        on_right_boundary = bm.abs(x - domain[1]) < self._eps
        
        return on_left_boundary | on_right_boundary
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        
        # 判断条件：位于左右边界上
        on_left_boundary = bm.abs(x - domain[0]) < self._eps
        on_right_boundary = bm.abs(x - domain[1]) < self._eps
        
        return on_left_boundary | on_right_boundary
    
    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    

class HalfClampedBeam2D(PDEBase):
    '''
    Symmetric half-domain model for the clamped-clamped beam example from 
    Castañar et al. (2022), Section 5.1.
    
    PDEs:
    -∇·σ = b   in Ω (left half-domain)
      u = 0    on ∂Ω_D (fully clamped on the left side, x=0)
      u_x = 0  on ∂Ω_S (symmetry boundary, x=80)
      
    几何参数:
        左半部分矩形域, 左端完全固支, 右端为对称边界, 右下角施加向下的集中载荷。
    
    Material parameters from paper:
        E_s = 30 Pa, nu is varied (e.g., 0.4 for compressible, 0.5 for incompressible)
    '''
    def __init__(self,
                 domain: List[float] = [0, 80, 0, 20],  # 几何尺寸为原始域的左半部分
                 mesh_type: str = 'uniform_tri', 
                 T: float = -1.5,  # 载荷为原始载荷的一半
                 E: float = 30.0, 
                 nu: float = 0.4, 
                 plane_type: str = 'plane_strain', # 'plane_stress' or 'plane_strain'
                 enable_logging: bool = False, 
                 logger_name: Optional[str] = None
                 ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                         enable_logging=enable_logging, logger_name=logger_name)
        
        self._T = T
        self._E, self._nu = E, nu
        self._plane_type = plane_type

        self._eps = 1e-12
        self._force_type = 'concentrated'
        self._boundary_type = 'dirichlet'

        self._log_info(f"Initialized HalfClampedBeam2D (Symmetric model from Castañar et al. 2022) with domain={self._domain}, "
                     f"mesh_type='{mesh_type}', force={T}, E={E}, nu={nu}, "
                     f"plane_type='{self._plane_type}', boundary='clamped-left, symmetric-right'.")

    #######################################################################################################################
    # 访问器 (Accessors)
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
    # 变体方法 (Variant Methods)
    #######################################################################################################################
    
    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        # 保持单元密度，x方向单元数为全模型一半
        nx = kwargs.get('nx', 160)
        ny = kwargs.get('ny', 40)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                      threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)

        return mesh
    
    @init_mesh.register('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 20)
        device = kwargs.get('device', 'cpu')
        node = bm.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0]], dtype=bm.float64, device=device) 
        
        cell = bm.array([[0, 1, 2, 3]], dtype=bm.int32, device=device)     
        
        qmesh = QuadrangleMesh(node, cell)
        qmesh = qmesh.from_box(box=self._domain, nx=nx, ny=ny)
        node = qmesh.entity('node')
        cell = qmesh.entity('cell')
        isLeftCell = bm.zeros((nx, ny), dtype=bm.bool)
        isLeftCell[0, 0::2] = True
        isLeftCell[1, 1::2] = True
        if nx > 2:
            isLeftCell[2::2, :] = isLeftCell[0, :]
        if ny > 3:
            isLeftCell[3::2, :] = isLeftCell[1, :]
        isLeftCell = isLeftCell.reshape(-1)
        lcell = cell[isLeftCell]
        rcell = cell[~isLeftCell]
        import numpy as np
        newCell = np.r_['0', 
                lcell[:, [1, 2, 0]], 
                lcell[:, [3, 0, 2]],
                rcell[:, [0, 1, 3]],
                rcell[:, [2, 3, 1]]]
        
        mesh = TriangleMesh(node, newCell)

        self._save_meshdata(mesh, 'uniform_crisscross_tri', nx=nx, ny=ny)

        return mesh

    #######################################################################################################################
    # 核心方法 (Core Methods)
    #######################################################################################################################

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        """
        定义体力 (以集中载荷形式施加)
        在右下角顶点施加向下的集中载荷 F = -1.5 (N)
        """
        domain = self.domain

        x, y = points[..., 0], points[..., 1]  

        # 载荷点位于右下角
        coord = (
            (bm.abs(x - domain[1]) < self._eps) & 
            (bm.abs(y - domain[2]) < self._eps)
        )
        
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        # 集中力施加在y方向
        val = bm.set_at(val, (coord, 1), self._T)
        
        return val
    
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """Dirichlet边界值为0"""
        kwargs = bm.context(points)
        return bm.zeros(points.shape, **kwargs)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """
        判断x方向的Dirichlet边界自由度。
        左边界(x=0)完全固支 -> u_x = 0
        右边界(x=80)对称 -> u_x = 0
        """
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        
        on_left_boundary = bm.abs(x - domain[0]) < self._eps
        on_right_boundary = bm.abs(x - domain[1]) < self._eps
        
        return on_left_boundary | on_right_boundary
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """
        判断y方向的Dirichlet边界自由度。
        左边界(x=0)完全固支 -> u_y = 0
        右边界(x=80)为自然边界条件，非Dirichlet
        """
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        
        on_left_boundary = bm.abs(x - domain[0]) < self._eps
        
        return on_left_boundary
    
    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:
        """返回判断x和y方向Dirichlet边界的函数元组"""
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)