from typing import List, Optional, Tuple, Callable

from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh, HomogeneousMesh
from fealpy.backend import backend_manager as bm
from soptx.model.pde_base import PDEBase  

class SimplySupportedBridge2d(PDEBase):
    """
    二维简支桥梁的 PDE 模型

    设计域: 120 mm x 40 mm

    边界条件:
    - 左下角固定铰支座 (u_x = u_y = 0)
    - 右下角滑动支座 (u_x = 0)

    载荷条件:
    - 上边界施加竖直向下的均布边界牵引载荷 t = 1 N/mm

    非设计域:
    - 顶部 4mm 区域为实体非设计域 (桥面)
    """
    def __init__(self,
                domain: List[float] = [0.0, 120.0, 0.0, 40.0],
                mesh_type: str = 'uniform_quad',
                t: float = 1.0,   # N/mm
                E: float = 1.0,   # MPa
                nu: float = 0.3,
                plane_type: str = 'plane_stress',  # 'plane_stress' or 'plane_strain'
                deck_height: float = 4.0,           # 桥面物理高度
                load_region: Optional[Tuple[int, int]] = None,    # (宽, 高) 网格层数
                support_region: Optional[Tuple[int, int]] = None, # (宽, 高) 网格层数
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:

        super().__init__(domain=domain, mesh_type=mesh_type,
                         enable_logging=enable_logging, logger_name=logger_name)

        self._t = t
        self._E, self._nu = E, nu
        self._plane_type = plane_type

        self.deck_height = deck_height
        self.load_region = load_region
        self.support_region = support_region

        self._eps = 1e-8
        self._load_type = 'distributed'
        self._boundary_type = 'mixed'

    @property
    def E(self) -> float:
        return self._E

    @property
    def nu(self) -> float:
        return self._nu

    @property
    def t(self) -> float:
        return self._t

    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> 'QuadrangleMesh':
        nx = kwargs.get('nx', 120)
        ny = kwargs.get('ny', 60)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                       threshold=threshold, device=device)
        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)
        return mesh

    @init_mesh.register('uniform_aligned_tri')
    def init_mesh(self, **kwargs) -> 'TriangleMesh':
        nx = kwargs.get('nx', 120)
        ny = kwargs.get('ny', 60)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                     threshold=threshold, device=device)
        self._save_meshdata(mesh, 'uniform_aligned_tri', nx=nx, ny=ny)
        return mesh

    @init_mesh.register('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> 'TriangleMesh':
        nx = kwargs.get('nx', 120)
        ny = kwargs.get('ny', 60)
        device = kwargs.get('device', 'cpu')

        node = bm.array([[0.0, 0.0],
                         [1.0, 0.0],
                         [1.0, 1.0],
                         [0.0, 1.0]], dtype=bm.float64, device=device)
        cell = bm.array([[0, 1, 2, 3]], dtype=bm.int32, device=device)
        qmesh = QuadrangleMesh(node, cell).from_box(box=self._domain, nx=nx, ny=ny)

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

    @cartesian
    def body_force(self, points: 'TensorLike') -> 'TensorLike':
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)

    @cartesian
    def dirichlet_bc(self, points: 'TensorLike') -> 'TensorLike':
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: 'TensorLike') -> 'TensorLike':
        domain = self._domain
        x, y = points[..., 0], points[..., 1]
        
        left_bottom = (bm.abs(x - domain[0]) < self._eps) & (bm.abs(y - domain[2]) < self._eps)
                
        return left_bottom

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: 'TensorLike') -> 'TensorLike':
        domain = self._domain
        x, y = points[..., 0], points[..., 1]
        left_bottom = (bm.abs(x - domain[0]) < self._eps) & (bm.abs(y - domain[2]) < self._eps)
        right_bottom = (bm.abs(x - domain[1]) < self._eps) & (bm.abs(y - domain[2]) < self._eps)

        return left_bottom | right_bottom

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x,
                self.is_dirichlet_boundary_dof_y)
    
    @cartesian
    def neumann_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val = bm.set_at(val, (..., 1), -self._t) 

        return val
    
    @cartesian
    def is_neumann_boundary_dof(self, points: TensorLike) -> TensorLike:
        domain = self._domain
        x, y = points[..., 0], points[..., 1]
        on_top = bm.abs(y - domain[3]) < self._eps

        return on_top

    def is_neumann_boundary(self) -> Callable:
        
        return self.is_neumann_boundary_dof
    
    def get_passive_element_mask(self, mesh: HomogeneousMesh) -> TensorLike:
        """生成被动单元掩码 - 节点密度表征专用"""
        # 1. 获取所有节点的物理坐标
        # 假设 mesh.nodes 是一个形状为 [N_nodes, 2] 的张量
        points = mesh.entity('node')
        x = points[..., 0]
        y = points[..., 1]

        # --- 智能参数解析 (与单元版保持逻辑一致) ---
        
        # A. 确定桥面高度阈值
        # 物理域总高度
        domain_min_y = self.domain[2]
        domain_max_y = self.domain[3]
        total_height = domain_max_y - domain_min_y
        
        # 计算桥面下沿的物理 Y 坐标
        # 逻辑：如果指定了 load_region (层数)，将其转为物理高度；否则直接用 deck_height
        if self.load_region is not None:
            # 如果用户按"单元层数"指定了桥面，我们需要反算物理高度
            # 注意：这需要保证 mesh 的均匀性
            ny = mesh.meshdata['ny']
            cell_h = total_height / ny
            deck_physical_height = self.load_region[1] * cell_h
        else:
            # 推荐：直接使用物理高度参数 (如 4mm)
            deck_physical_height = self.deck_height

        # 桥面区域判定阈值 (Y >= Top - Height)
        # 添加一个小 epsilon 防止浮点误差
        deck_y_threshold = domain_max_y - deck_physical_height - 1e-6

        # B. 确定支座区域 (可选，保持逻辑完整性)
        # 如果需要锁定支座处的节点为实体 (视具体物理需求而定，通常支座点位移为0，密度由于是实心也应为1)
        support_physical_width = 0.0
        if self.support_region is not None:
             nx = mesh.meshdata['nx']
             total_width = self.domain[1] - self.domain[0]
             cell_w = total_width / nx
             support_physical_width = self.support_region[0] * cell_w

        # -------------------

        # 2. 区域判定 (生成布尔掩码)

        # 判定 A: 桥面节点 (Top Deck)
        # 只要节点的 Y 坐标高于阈值，即为桥面节点
        mask_deck = (y >= deck_y_threshold)

        # 判定 B: 支座节点 (Supports) - 左下和右下
        # 仅当 support_physical_width > 0 时生效
        mask_support = bm.zeros_like(mask_deck, dtype=bm.bool)
        if support_physical_width > 0:
            domain_min_x = self.domain[0]
            domain_max_x = self.domain[1]
            support_y_threshold = domain_min_y + (self.support_region[1] * (total_height/mesh.meshdata['ny'])) # 简略计算高度
            
            # 左下角区域
            mask_left = (x <= domain_min_x + support_physical_width + 1e-6) & \
                        (y <= support_y_threshold + 1e-6)
            # 右下角区域
            mask_right = (x >= domain_max_x - support_physical_width - 1e-6) & \
                         (y <= support_y_threshold + 1e-6)
            
            mask_support = mask_left | mask_right

        # 3. 合并掩码
        return mask_deck | mask_support
    
    def _get_passive_element_mask(self, mesh: HomogeneousMesh) -> TensorLike:
        """生成被动单元掩码 - 单元密度表征专用"""
        # 1. 确定网格维度
        nx, ny = mesh.meshdata['nx'], mesh.meshdata['ny']

        # --- 智能参数解析 ---
        # 如果用户初始化时指定了 load_region，则使用它
        # 否则，根据 deck_height 自动计算全长桥面
        if self.load_region is not None:
            current_load_region = self.load_region
        else:
            # 自动计算: 宽度=nx(全长), 高度=物理高度对应的层数
            total_height = self.domain[3] - self.domain[2]
            cell_h = total_height / ny
            deck_layers = int(round(self.deck_height / cell_h))
            # 确保至少有一层
            deck_layers = max(1, deck_layers)
            current_load_region = (nx, deck_layers)
            
        # 支座区域，默认为空 (0,0)
        current_support_region = self.support_region if self.support_region is not None else (0, 0)
        # -------------------

        # 2. 网格索引映射 (保持原有逻辑)
        if isinstance(mesh, TriangleMesh):
            n_elements = 2 * nx * ny
            el_indices = bm.arange(n_elements)
            grid_cell_indices = el_indices // 2
        elif isinstance(mesh, QuadrangleMesh):
            n_elements = nx * ny
            el_indices = bm.arange(n_elements)
            grid_cell_indices = el_indices

        # 3. 计算 (ix, iy)
        # 注意：使用列主序逻辑适配之前的代码 ix = index // ny
        ix = grid_cell_indices // ny  # 列号
        iy = grid_cell_indices % ny   # 行号
        
        # 4. 区域判定
        
        # 处理 load_region (对应桥面)
        load_w, load_h = current_load_region
        limit_load_w = min(load_w, nx)
        mask_load = (ix < limit_load_w) & (iy >= ny - load_h)
        
        # 处理 support_region (对应底部支座增强)
        support_w, support_h = current_support_region
        limit_support_w = max(nx - support_w, 0)
        mask_support = (ix >= limit_support_w) & (iy < support_h)
        
        return mask_load | mask_support