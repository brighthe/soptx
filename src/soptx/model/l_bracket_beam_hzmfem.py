from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.model.pde_base import PDEBase  


class LBracketMiddle2d(PDEBase):
    '''
    二维 L 型支架设计域的 PDE 模型（混合有限元框架）

    设计域:
        - 全设计域: domain，通过切除 hole_domain 右上角得到 L 形区域

    边界条件:
        - 顶部边界完全固支 (u_x = u_y = 0)

    载荷条件:
        - 右侧边缘下部矩形中点施加向下分布载荷 P，载荷宽度为 load_width
        - 混合元框架下牵引力以 σ·n = t 的形式强施加在应力自由度上，
          不存在真正的点载荷，因此 load_width 为必须显式指定的物理参数。
    '''
    def __init__(self,
                domain: List[float] = [0, 1, 0, 1],
                hole_domain: List[float] = [0.4, 1, 0.4, 1],
                mesh_type: str = 'uniform_quad',
                P: float = -400.0,            # N，向下为负
                E: float = 7e4,             # MPa (N/mm^2)
                nu: float = 0.25,
                load_width: Optional[float] = None,  # 载荷分布宽度（必须显式指定）
                plane_type: str = 'plane_stress',    # 'plane_stress' or 'plane_strain'
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type,
                enable_logging=enable_logging, logger_name=logger_name)

        self._domain = domain
        self._hole_domain = hole_domain
        self._P = P
        self._E, self._nu = E, nu
        self._plane_type = plane_type
        self._load_width = load_width

        self._eps = 1e-8
        self._load_type = 'distributed'
        self._boundary_type = 'mixed'

        self._t = None    # 牵引力强度，由 set_load_region 初始化
        self._hy = None   # 单元尺寸（y 方向）

    @property
    def E(self) -> float:
        return self._E

    @property
    def nu(self) -> float:
        return self._nu

    @property
    def P(self) -> float:
        return self._P

    @property
    def load_width(self) -> float:
        return self._load_width

    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        device = kwargs.get('device', 'cpu')

        big_box = self._domain
        small_box = self._hole_domain

        def threshold(p):
            x = p[..., 0]
            y = p[..., 1]
            return ((x >= small_box[0])
                   &(x <= small_box[1])
                   &(y >= small_box[2])
                   &(y <= small_box[3]))

        mesh = QuadrangleMesh.from_box(big_box, nx=nx, ny=ny,
                                       threshold=threshold, device=device)
        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)

        return mesh

    @init_mesh.register('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        device = kwargs.get('device', 'cpu')

        big_box = self._domain
        small_box = self._hole_domain

        def threshold(p):
            x = p[..., 0]
            y = p[..., 1]
            return ((x >= small_box[0])
                   &(x <= small_box[1])
                   &(y >= small_box[2])
                   &(y <= small_box[3]))

        mesh = TriangleMesh.from_box_cross_mesh(big_box, nx=nx, ny=ny,
                                                threshold=threshold, device=device)
        self._save_meshdata(mesh, 'uniform_crisscross_tri', nx=nx, ny=ny)

        return mesh
    
    def mark_corners(self, node: TensorLike) -> TensorLike:
        """显式标记 L 形域的所有几何角点坐标

        L 形域共有 6 个角点：
            - 外角点（位于全局矩形框边界）：
                (x_min, y_min), (x_max, y_min),
                (x_min, y_max), (hole_x_min, y_max)  ← 注意最后一个不在 x 全局边界上
            - 内凹角点（切除区域引入）：
                (hole_x_min, hole_y_min), (x_max, hole_y_min)
        """
        x_min, x_max = self._domain[0], self._domain[1]
        y_min, y_max = self._domain[2], self._domain[3]
        hole_x_min = self._hole_domain[0]
        hole_y_min = self._hole_domain[2]

        # 标准外角点：x 在全局边界 & y 在全局边界
        is_x_outer = (bm.abs(node[:, 0] - x_min) < self._eps) | \
                    (bm.abs(node[:, 0] - x_max) < self._eps)
        is_y_outer = (bm.abs(node[:, 1] - y_min) < self._eps) | \
                    (bm.abs(node[:, 1] - y_max) < self._eps)
        is_outer_corner = is_x_outer & is_y_outer

        # 特殊外角点：(hole_x_min, y_max) —— x 不在全局边界，需单独处理
        is_top_notch = (bm.abs(node[:, 0] - hole_x_min) < self._eps) & \
                    (bm.abs(node[:, 1] - y_max) < self._eps)

        # 内凹角点1：(hole_x_min, hole_y_min) —— L 形内凹顶点
        is_inner_corner_1 = (bm.abs(node[:, 0] - hole_x_min) < self._eps) & \
                            (bm.abs(node[:, 1] - hole_y_min) < self._eps)

        # 内凹角点2：(x_max, hole_y_min) —— 右侧边缘与内凹水平边的交点
        is_inner_corner_2 = (bm.abs(node[:, 0] - x_max) < self._eps) & \
                            (bm.abs(node[:, 1] - hole_y_min) < self._eps)

        is_corner = is_outer_corner | is_top_notch | is_inner_corner_1 | is_inner_corner_2
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
        """标记位移边界 —— 顶部边界完全固支"""
        domain = self.domain
        y = points[..., 1]

        # 顶部边界：y = y_max
        return bm.abs(y - domain[3]) < self._eps

    def set_load_region(self, mesh):
        """初始化牵引力强度 t = P / load_width

        混合元框架下载荷宽度由物理参数 load_width 直接决定，与网格无关。
        仅在首次调用时计算，后续调用直接返回。
        """
        if self._t is not None:
            return

        hy = mesh.meshdata['hy']
        self._hy = hy

        if self._load_width is None:
            # 默认载荷宽度为单元尺寸级别，逼近点载荷
            self._load_width = hy

        self._t = self._P / self._load_width

    @cartesian
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记牵引边界 —— 除顶部固支边外所有自由边界

        L 形域的自由边界包括：
        底部、左侧、右侧下部、内凹水平边、内凹竖直边
        """
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        # 顶部为位移边界，其余所有边界均为牵引边界
        on_top = bm.abs(y - domain[3]) < self._eps

        return ~on_top

    @cartesian
    def traction_bc(self, points: TensorLike) -> TensorLike:
        """牵引边界条件

        仅在右侧边缘下部矩形中点区域施加等效分布牵引，其余自由边界牵引为零。
        """
        domain = self.domain

        # 右侧边缘下部矩形的中点高度
        # 下部矩形 y 范围为 [domain[2], hole_domain[2]]
        middle_y = (domain[2] + self._hole_domain[2]) / 2.0

        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)

        if self._t is None:
            return val

        # 边级中心坐标（NEb, 2）
        edge_center = points.mean(axis=-2)
        x_c = edge_center[..., 0]
        y_c = edge_center[..., 1]

        # 仅在右侧边缘（x = x_max）搜索载荷区域
        on_right = bm.abs(x_c - domain[1]) < self._eps
        dist = bm.where(on_right, bm.abs(y_c - middle_y), bm.full_like(y_c, bm.inf))

        if self._load_width <= self._hy + self._eps:
            # 单元级点载荷逼近：选取距中点最近的边
            min_dist = bm.min(dist)
            in_load_region = on_right & (dist <= min_dist + self._eps)
        else:
            # 显式分布载荷：覆盖 load_width 范围内所有边
            search_hw = self._load_width / 2.0 + self._eps
            in_load_region = on_right & (dist <= search_hw)

        # 动态调整牵引力大小，确保宏观静力学严格等效：∫t dy = P
        num_edges = bm.sum(in_load_region)
        actual_t = self._P / (num_edges * self._hy)

        # 广播赋值到插值点维度
        mask = in_load_region[:, None, None]   # (NEb, 1, 1)
        t_field = bm.zeros(points.shape, **kwargs)
        t_field = bm.set_at(t_field, (..., 1), actual_t)

        val = bm.where(mask, t_field, val)

        return val