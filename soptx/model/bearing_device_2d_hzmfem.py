from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.model.pde_base import PDEBase  

class BearingDeviceLeftHalf2d(PDEBase):
    '''
    轴承装置左半设计域的 PDE 模型 - 胡张混合元

    设计域:
        - 全设计域: 120 mm x 40 mm
        - 左半设计域: 60 mm x 40 mm

    边界条件:
        - 底部固支 (u_x = u_y = 0)
        - 右侧对称边界 (u_x = 0)

    载荷条件:
        - 顶部向下的均匀分布牵引载荷
    '''
    def __init__(self,
                domain: List[float] = [0, 60, 0, 40],  
                mesh_type: str = 'uniform_quad',
                t: float = -1.8e-2, # N/mm
                E: float = 1,       # MPa
                nu: float = 0.5,
                plane_type: str = 'plane_stress', # 'plane_stress' or 'plane_strain'
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type,
                         enable_logging=enable_logging, logger_name=logger_name)

        self._t = t
        self._E, self._nu = E, nu
        self._plane_type = plane_type

        self._eps = 1e-8
        self._load_type = 'distributed'
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
    def t(self) -> float:
        """获取面力"""
        return self._t
    
    @variantmethod('uniform_aligned_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 40)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_aligned_tri', nx=nx, ny=ny)

        return mesh

    @init_mesh.register('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 40)
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
    
    def mark_corners(self, node: TensorLike) -> TensorLike:
        """显示标记几何角点坐标"""
        x_min, x_max = self._domain[0], self._domain[1]
        y_min, y_max = self._domain[2], self._domain[3]

        is_x_bd = (bm.abs(node[:, 0] - x_min) < self._eps) | (bm.abs(node[:, 0] - x_max) < self._eps)
        is_y_bd = (bm.abs(node[:, 1] - y_min) < self._eps) | (bm.abs(node[:, 1] - y_max) < self._eps)
        is_corner = is_x_bd & is_y_bd
        corner_coords = node[is_corner]

        return corner_coords

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        """体力密度 b(x, y)"""
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def displacement_bc(self, points: TensorLike) -> TensorLike:
        """位移边界条件 u_D(x, y)"""
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)

    @cartesian
    def is_displacement_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        on_bottom_boundary = bm.abs(y - domain[2]) < self._eps
        on_right_boundary = bm.abs(x - domain[1]) < self._eps

        return on_bottom_boundary | on_right_boundary

    @cartesian
    def is_displacement_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        y = points[..., 1]
        
        on_bottom_boundary = bm.abs(y - domain[2]) < self._eps

        return on_bottom_boundary

    @cartesian
    def is_displacement_boundary_dof(self) -> Tuple[Callable, Callable]:
        """标记位移边界 - 按分量"""
        return (self.is_displacement_boundary_dof_x,
                self.is_displacement_boundary_dof_y)
    
    @cartesian
    def traction_bc(self, points: TensorLike) -> TensorLike:
        """牵引边界条件 g_N(x, y) - 牵引力分量 σ·n"""
        domain = self.domain
        y = points[..., 1]
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)  

        flag_top = bm.abs(y - domain[3]) < self._eps
        val = bm.set_at(val, (flag_top, 1), self._t)  

        return val
    
    @cartesian
    def is_traction_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """Whether (σn)_x is prescribed on this edge."""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        on_top  = bm.abs(y - domain[3]) < self._eps          # prescribed: 0
        on_left = bm.abs(x - domain[0]) < self._eps          # prescribed: 0
        # on_right (symmetry): (σn)_x is NOT prescribed
        return on_top | on_left

    @cartesian
    def is_traction_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """Whether (σn)_y is prescribed on this edge."""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        on_top   = bm.abs(y - domain[3]) < self._eps         # prescribed: t
        on_left  = bm.abs(x - domain[0]) < self._eps         # prescribed: 0
        on_right = bm.abs(x - domain[1]) < self._eps         # symmetry: prescribed 0 (shear traction)
        return on_top | on_left | on_right

    @cartesian
    def is_traction_boundary_dof(self):
        """标记牵引边界 - 按分量"""
        return (self.is_traction_boundary_dof_x,
                self.is_traction_boundary_dof_y)

class BearingDevice2d(PDEBase):
    '''
    轴承装置全设计域的 PDE 模型

    设计域:
        - 全设计域: 120 mm x 40 mm

    边界条件:
        - 底部固支 (u_x = u_y = 0)
    
    载荷条件:
        - 顶部向下的均匀分布牵引载荷 t = -1.8e-3 [N/mm]
    '''
    def __init__(self,
                domain: List[float] = [0, 120, 0, 40],  
                mesh_type: str = 'uniform_quad',
                t: float = -1.8e-2, # N/mm
                E: float = 1,       # MPa
                nu: float = 0.5,
                plane_type: str = 'plane_stress', # 'plane_stress' or 'plane_strain'
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type,
                         enable_logging=enable_logging, logger_name=logger_name)

        self._t = t
        self._E, self._nu = E, nu
        self._plane_type = plane_type

        self._eps = 1e-12
        self._load_type = 'distributed'
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
    def t(self) -> float:
        """获取面力"""
        return self._t

    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        # 根据几何调整默认单元数（完整区域是半区域的2倍宽）
        nx = kwargs.get('nx', 120)  
        ny = kwargs.get('ny', 40)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                       threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)

        return mesh
    
    @init_mesh.register('uniform_aligned_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 120)  
        ny = kwargs.get('ny', 40)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_aligned_tri', nx=nx, ny=ny)

        return mesh

    @init_mesh.register('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 120) 
        ny = kwargs.get('ny', 40)
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
    
    def mark_corners(self, node: TensorLike) -> TensorLike:
        """显示标记几何角点坐标"""
        x_min, x_max = self._domain[0], self._domain[1]
        y_min, y_max = self._domain[2], self._domain[3]

        is_x_bd = (bm.abs(node[:, 0] - x_min) < self._eps) | (bm.abs(node[:, 0] - x_max) < self._eps)
        is_y_bd = (bm.abs(node[:, 1] - y_min) < self._eps) | (bm.abs(node[:, 1] - y_max) < self._eps)
        is_corner = is_x_bd & is_y_bd
        corner_coords = node[is_corner]

        return corner_coords

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        """体力密度 b(x, y)"""
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def displacement_bc(self, points: TensorLike) -> TensorLike:
        """位移边界条件 u_D(x, y)"""
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """标记位移边界 - 按边集合"""
        domain = self.domain
        y = points[..., 1]
        on_bottom_boundary = bm.abs(y - domain[2]) < self._eps

        return on_bottom_boundary
    
    @cartesian
    def traction_bc(self, points: TensorLike) -> TensorLike:
        """牵引边界条件 g_N(x, y) - 牵引力分量 σ·n"""
        domain = self.domain
        y = points[..., 1]

        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)  

        flag_top = bm.abs(y - domain[3]) < self._eps
        val = bm.set_at(val, (flag_top, 1), self._t)  

        return val
    
    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记牵引边界 - 按边集合"""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        on_top  = bm.abs(y - domain[3]) < self._eps
        on_left = bm.abs(x - domain[0]) < self._eps
        on_right = bm.abs(x - domain[1]) < self._eps

        return on_top | on_left | on_right