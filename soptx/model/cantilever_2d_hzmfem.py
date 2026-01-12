from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.model.pde_base import PDEBase  
    
class CantileverBeam2d(PDEBase):
    '''
    二维悬臂梁设计域的 PDE 模型

    设计域:
        - 全设计域: 80 mm x 40 mm

    边界条件:
        - 左侧边界完全固支 (u_x = u_y = 0)

    载荷条件:
        - 右侧边界中点施加向下集中载荷 P = -1 N
    '''
    def __init__(self,
                domain: List[float] = [0, 80, 0, 40], 
                mesh_type: str = 'uniform_quad',
                p: float = -1.0,   # N，向下为负
                E: float = 1.0,    # MPa
                nu: float = 0.3,  
                plane_type: str = 'plane_stress',  # 'plane_stress' or 'plane_strain'
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)

        self._p = p
        self._E, self._nu = E, nu
        self._plane_type = plane_type

        self._eps = 1e-8        
        self._load_type = 'concentrated'
        self._boundary_type = 'mixed'

        # 点载荷作用区域的半宽度 (需要根据网格尺寸调整)
        self._load_region_half_width = None 

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
        """获取集中力"""
        return self._p
    
    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 160) 
        ny = kwargs.get('ny', 80)   
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)

        return mesh
    
    @init_mesh.register('uniform_aligned_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 160)
        ny = kwargs.get('ny', 80)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_aligned_tri', nx=nx, ny=ny)

        return mesh
    
    @init_mesh.register('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 160)
        ny = kwargs.get('ny', 80)
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
        """显式标记几何角点坐标"""
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
        """标记位移边界 - 左侧边界完全固支"""
        domain = self.domain
        x = points[..., 0]
        
        # 左侧边界：x = x_min
        on_left = bm.abs(x - domain[0]) < self._eps
        
        return on_left
    
    def set_load_region(self, mesh):
        """根据网格设置点载荷作用区域"""
        if self._load_region_half_width is not None:
            return
    
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        
        bd_edge_flag = mesh.boundary_edge_flag()
        bd_edges = edge[bd_edge_flag]
        edge_lengths = bm.linalg.norm(
                            node[bd_edges[:, 1]] - node[bd_edges[:, 0]], axis=-1
                        )
        
        self._load_region_half_width = float(bm.min(edge_lengths)) / 2 + self._eps
        
        # 计算等效牵引力强度
        load_region_width = 2 * self._load_region_half_width
        self._t = self._p / load_region_width  # N/mm
    
    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记牵引边界 - 排除位移边界后的所有外边界"""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        
        # 顶部边界
        on_top = bm.abs(y - domain[3]) < self._eps
        # 底部边界
        on_bottom = bm.abs(y - domain[2]) < self._eps
        # 右侧边界
        on_right = bm.abs(x - domain[1]) < self._eps
        
        # 牵引边界 = 顶部 + 底部 + 右侧（左侧是位移边界）
        return on_top | on_bottom | on_right

    @cartesian
    def traction_bc(self, points: TensorLike) -> TensorLike:
        """牵引边界条件 - 包含点载荷的等效分布牵引"""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        y_mid = (domain[2] + domain[3]) / 2  # 右侧边界中点的 y 坐标
        
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        
        if self._load_region_half_width is None:
            # 未设置载荷区域，返回零牵引
            return val
        
        hw = self._load_region_half_width
        
        # 右侧载荷区域：x = x_max, y ∈ [y_mid - hw, y_mid + hw]
        on_right = bm.abs(x - domain[1]) < self._eps
        in_load_region = on_right & (bm.abs(y - y_mid) <= hw)
        
        # 设置牵引力（向下为负 y 方向）
        val = bm.set_at(val, (in_load_region, 1), self._t)

        return val