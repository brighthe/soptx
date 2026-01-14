from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.model.pde_base import PDEBase  

class HalfMBBBeamRight2d(PDEBase):
    '''
    对称 MBB 梁右半设计域的 PDE 模型

    设计域:
        - 全设计域: 120 mm x 20 mm
        - 右半设计域: 60 mm x 20 mm

    边界条件:
        - 左侧对称约束 (u_x = 0)
        - 右下角滑移支座 (u_y = 0)
    
    载荷条件:
        - 左上角施加向下的集中载荷 P = 1 [N]

    材料参数:
        E = 1 [MPa], nu = 0.3
    ''' 
    def __init__(self,
                domain: List[float] = [0, 60, 0, 20],
                mesh_type: str = 'uniform_quad',
                P: float = -1.0, # N
                E: float = 1.0,  # MPa
                nu: float = 0.3,
                plane_type: str = 'plane_stress', # 'plane_stress' or 'plane_strain'
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)

        self._P = P
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
    def P(self) -> float:
        """获取点力"""
        return self._P

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
    
    @init_mesh.register('uniform_aligned_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 20)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_aligned_tri', nx=nx, ny=ny)

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
    
    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """标记位移边界 - 左侧对称约束 + 右下角滑移支座"""
        domain = self._domain
        x, y = points[..., 0], points[..., 1]
        
        # 左边界：x = x_min，约束 u_x = 0
        on_left = bm.abs(x - domain[0]) < self._eps
        
        # 右下角：x = x_max 且 y = y_min，约束 u_y = 0
        on_right_bottom = (bm.abs(x - domain[1]) < self._eps) & \
                          (bm.abs(y - domain[2]) < self._eps)
        
        # 构建分量边界标记
        bc_x = on_left                # 左边界约束 x 方向
        bc_y = on_right_bottom        # 右下角约束 y 方向
        
        return bm.stack([bc_x, bc_y], axis=-1)
    
    def set_load_region(self, mesh) -> None:
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
        self._t = self._P / load_region_width  # N/mm
    
    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记牵引边界 - 除位移边界外的所有外边界"""
        domain = self._domain
        x, y = points[..., 0], points[..., 1]
        
        # 顶部边界
        on_top = bm.abs(y - domain[3]) < self._eps
        
        # 底部边界（排除右下角点）
        on_bottom = bm.abs(y - domain[2]) < self._eps
        on_right_bottom = (bm.abs(x - domain[1]) < self._eps) & on_bottom
        on_bottom_traction = on_bottom & (~on_right_bottom)
        
        # 右边界（排除右下角点）
        on_right = bm.abs(x - domain[1]) < self._eps
        on_right_traction = on_right & (~on_right_bottom)
        
        return on_top | on_bottom_traction | on_right_traction

    @cartesian
    def traction_bc(self, points: TensorLike) -> TensorLike:
        """牵引边界条件 - 左上角施加集中力"""
        domain = self._domain
        x, y = points[..., 0], points[..., 1]
        
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        
        if self._load_region_half_width is None:
            return val
        
        hw = self._load_region_half_width
        
        # 左上角载荷区域：x = x_min, y ∈ [y_max - hw, y_max]
        on_top = bm.abs(y - domain[3]) < self._eps
        near_left = x <= domain[0] + hw
        in_load_region = on_top & near_left
        
        # 设置向下的牵引力
        val = bm.set_at(val, (in_load_region, 1), self._t)
        
        return val
    
    def get_passive_element_mask(self, 
                                nx: int, 
                                ny: int,
                                load_region: tuple = (3, 2),
                                support_region: tuple = (3, 3)
                            ) -> TensorLike:
        """
        生成被动单元掩码（适用于列主序的单元编号）
        
        Parameters
        ----------
        nx, ny : int
            网格在 x, y 方向的单元数
        load_region : tuple, optional
            载荷点区域尺寸 (宽度, 高度)，默认 (3, 2)
        support_region : tuple, optional
            支座点区域尺寸 (宽度, 高度)，默认 (3, 3)
            
        Returns
        -------
        mask : TensorLike, shape (nx * ny,)
            被动单元掩码，True 表示被动单元
        """
        n_elements = nx * ny
        
        el_indices = bm.arange(n_elements)
        ix = el_indices // ny  # 列号 (x 方向)
        iy = el_indices % ny   # 行号 (y 方向)
        
        # 载荷点区域（左上角）
        load_w, load_h = load_region
        mask_load = (ix < load_w) & (iy >= ny - load_h)
        
        # 支座点区域（右下角）
        support_w, support_h = support_region
        mask_support = (ix >= nx - support_w) & (iy < support_h)
        
        return mask_load | mask_support
