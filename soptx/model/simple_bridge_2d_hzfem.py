from typing import List, Optional, Tuple, Callable

from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh
from fealpy.backend import backend_manager as bm
from soptx.model.pde_base import PDEBase  

class SimpleBridge2d(PDEBase):
    '''
    二维简支桥梁设计域的 PDE 模型

    设计域:
        - 全设计域: 60 mm x 30 mm (比例 2:1)

    边界条件:
        - 底部左下角点 (x_min, y_min) 完全固定 (u_x = u_y = 0)
        - 底部右下角点 (x_max, y_min) 完全固定 (u_x = u_y = 0)

    载荷条件:
        - 底部边缘中点 (x_mid, y_min) 施加竖直向下的集中载荷 P
        - 混合元框架下以分布牵引 t = P / load_width 强施加在底部边界的应力自由度上
    '''
    def __init__(self,
                domain: List[float] = [0, 60, 0, 30], 
                mesh_type: str = 'uniform_quad',
                P: float = -1.0,   # N，向下为负
                E: float = 1.0,    # MPa
                nu: float = 0.3,  
                plane_type: str = 'plane_stress',
                load_width: Optional[float] = None, 
                support_width: Optional[float] = None,
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)

        self._P = P
        self._E, self._nu = E, nu
        self._plane_type = plane_type

        self._load_width = load_width 
        self._support_width = support_width

        self._eps = 1e-8        
        self._load_type = 'distributed'
        self._boundary_type = 'mixed'

        self._t = None  
        self._hx = None # 注意：此处载荷在水平边，主要参考 x 方向步长

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

    # @cartesian
    # def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
    #     """标记位移边界 - 底部左右两个角点"""
    #     x, y = points[..., 0], points[..., 1]
    #     x_min, x_max = self._domain[0], self._domain[1]
    #     y_min = self._domain[2]
        
    #     # 靠近左下角或右下角
    #     is_left_corner = (bm.abs(x - x_min) < self._eps) & (bm.abs(y - y_min) < self._eps)
    #     is_right_corner = (bm.abs(x - x_max) < self._eps) & (bm.abs(y - y_min) < self._eps)
        
    #     return is_left_corner | is_right_corner
    
    @cartesian
    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """标记位移边界 - 底部左右两端分布化的小段支撑区域"""
        x, y = points[..., 0], points[..., 1]
        x_min, x_max = self._domain[0], self._domain[1]
        y_min = self._domain[2]
        
        # 确保 support_width 已经初始化
        sw = self._support_width if self._support_width is not None else 1e-5
        
        # 位于底部边界
        on_bottom = bm.abs(y - y_min) < self._eps
        
        # 扩展为一个小段区域 (覆盖 support_width)
        is_left_support = on_bottom & (x <= x_min + sw + self._eps)
        is_right_support = on_bottom & (x >= x_max - sw - self._eps)
        
        return is_left_support | is_right_support
    
    def set_load_region(self, mesh):
        """初始化水平方向的牵引力强度 t = P / load_width
        
        混合元框架下载荷宽度由物理参数 load_width 直接决定，与网格无关.
        仅在首次调用时计算，后续调用直接返回."""
        if self._t is not None:
            return
        
        hx = mesh.meshdata['hx']
        self._hx = hx

        if self._load_width is None:
            self._load_width = hx   # 默认载荷宽度为一个单元
            
        if self._support_width is None:
            self._support_width = hx # 默认支撑宽度也为一个单元，逼近点支撑

        self._t = self._P / self._load_width

    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记牵引边界 - 简支桥梁中除了支撑点外的所有外边界区域"""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        
        # 边界判定逻辑
        on_left   = bm.abs(x - domain[0]) < self._eps
        on_right  = bm.abs(x - domain[1]) < self._eps
        on_top    = bm.abs(y - domain[3]) < self._eps
        on_bottom = bm.abs(y - domain[2]) < self._eps
        
        # 在简支桥梁中，除了支座（位移边界）外，所有外边缘均视为牵引边界
        # 实际上在 SOPTX 内部，Dirichlet BC 通常拥有更高的掩码优先级
        return on_left | on_right | on_top | on_bottom

    @cartesian
    def traction_bc(self, points: TensorLike) -> TensorLike:
        """牵引边界条件 - 底部边界中点区域"""
        domain = self.domain
        x_mid = (domain[0] + domain[1]) / 2
        y_min = domain[2]

        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        if self._t is None: return val

        # 边中心坐标
        edge_center = points.mean(axis=-2)
        x_c, y_c = edge_center[..., 0], edge_center[..., 1]

        # 仅考虑底部边界上的边 (y = y_min)
        on_bottom = bm.abs(y_c - y_min) < self._eps
        dist = bm.where(on_bottom, bm.abs(x_c - x_mid), bm.full_like(x_c, bm.inf))

        if self._load_width <= self._hx + self._eps:
            # 寻找所有等于最小距离的边（完美解决节点受力的对称性平局问题）
            min_dist = bm.min(dist)
            in_load_region = on_bottom & (dist <= min_dist + self._eps)
        else:
            # 显式分布载荷：区间选取覆盖 load_width 范围内的所有边
            search_hw = self._load_width / 2 + self._eps
            in_load_region = on_bottom & (dist <= search_hw)

        # 动态调整局部牵引力大小，确保宏观静力学严格等效
        # 若中位线落在某条边上，选中 1 条边；若中位线落在节点上，对称选中 2 条边
        num_edges = bm.sum(in_load_region)
        actual_t = self._P / (num_edges * self._hx)

        # 广播到插值点维度并赋值
        mask = in_load_region[:, None, None]  # (NEb, 1, 1)
        t_field = bm.zeros(points.shape, **kwargs)
        t_field = bm.set_at(t_field, (..., 1), actual_t)

        val = bm.where(mask, t_field, val)
        
        return val