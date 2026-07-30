from typing import List, Optional, Tuple, Callable

from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh
from fealpy.backend import backend_manager as bm
from soptx.model.pde_base import PDEBase  

class FixedFixedBeamCenterLoad2d(PDEBase):
    '''
    两端固支梁底部中点受载的 PDE 模型 (用于混合元基准测试)

    设计域:
        - 全设计域: 160 m x 20 m (长宽比 8:1) [cite: 74]

    边界条件:
        - 左侧边界完全固支 (x = 0, u_x = u_y = 0)
        - 右侧边界完全固支 (x = 160, u_x = u_y = 0)

    载荷条件:
        - 底部边界中点 (x = 80, y = 0) 施加向下分布载荷 P = -3.0 N
        - 混合元框架下，载荷以 σ·n = t 的形式强施加在底部边界的应力自由度上
    '''
    def __init__(self,
                domain: List[float] = [0, 160, 0, 20], 
                mesh_type: str = 'uniform_aligned_tri',
                P: float = -3.0,   # N，向下为负
                E: float = 30.0,   # Pa
                nu: float = 0.3,   
                plane_type: str = 'plane_stress',
                load_width: Optional[float] = None, 
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)

        self._P = P
        self._E, self._nu = E, nu
        self._plane_type = plane_type
        self._load_width = load_width 

        self._eps = 1e-8        
        self._load_type = 'distributed'
        self._boundary_type = 'mixed'

        self._t = None  
        self._hx = None # 载荷在水平边，主要参考 x 方向步长

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
        """获取集中力"""
        return self._P
    
    @property
    def load_width(self) -> float:
        """获取载荷分布宽度"""
        return self._load_width
    
    @property
    def support_width(self) -> Optional[float]:
        """获取支撑宽度"""
        return self._support_width
    
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
        """标记位移边界 - 左右两侧完全固支"""
        x = points[..., 0]
        on_left = bm.abs(x - self._domain[0]) < self._eps
        on_right = bm.abs(x - self._domain[1]) < self._eps

        return on_left | on_right
    
    def set_load_region(self, mesh):
        """初始化载荷宽度"""
        if self._t is not None: return
        
        # 提取网格步长 hx
        self._hx = mesh.meshdata['hx']

        if self._load_width is None:
            self._load_width = self._hx  # 默认载荷宽度为一个单元
            
        self._t = self._P / self._load_width  # 初始估计的载荷强度

    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记牵引边界 - 顶部 + 底部 (排除左右位移边界)"""
        y = points[..., 1]
        on_top = bm.abs(y - self._domain[3]) < self._eps
        on_bottom = bm.abs(y - self._domain[2]) < self._eps
        
        # 只有上下边是牵引边界
        is_all_bd = on_top | on_bottom
        is_disp = self.is_displacement_boundary(points)
        
        return is_all_bd & (~is_disp)

    @cartesian
    def traction_bc(self, points: TensorLike) -> TensorLike:
        """牵引边界条件 - 底部边界中点区域施加等效分布牵引"""
        x_mid = (self._domain[0] + self._domain[1]) / 2
        y_min = self._domain[2]

        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)

        if self._t is None: return val

        # 边中心坐标
        edge_center = points.mean(axis=-2)
        x_c, y_c = edge_center[..., 0], edge_center[..., 1]

        # 锁定底部边界 y = 0
        on_bottom = bm.abs(y_c - y_min) < self._eps
        dist = bm.where(on_bottom, bm.abs(x_c - x_mid), bm.full_like(x_c, bm.inf))

        # 寻找载荷覆盖区域
        if self._load_width <= self._hx + self._eps:
            min_dist = bm.min(dist)
            in_load_region = on_bottom & (dist <= min_dist + self._eps)
        else:
            in_load_region = on_bottom & (dist <= self._load_width / 2 + self._eps)

        num_edges = bm.sum(in_load_region)
        actual_t = self._P / (num_edges * self._hx)

        # 构造载荷向量 [0, actual_t]
        mask = in_load_region[:, None, None]
        t_field = bm.zeros(points.shape, **kwargs)
        t_field = bm.set_at(t_field, (..., 1), actual_t)

        return bm.where(mask, t_field, val)