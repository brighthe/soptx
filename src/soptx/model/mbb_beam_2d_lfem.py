from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh, HomogeneousMesh

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
                E: float = 1.0,  # MPa (N/mm^2)
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
        domain = self._domain
        x = points[..., 0]
        coord = bm.abs(x - domain[0]) < self._eps
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self._domain
        x = points[..., 0]
        y = points[..., 1]
        coord = ((bm.abs(x - domain[1]) < self._eps) &
                 (bm.abs(y - domain[2]) < self._eps))
        
        return coord
    
    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:
        
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    
    @cartesian
    def concentrate_load_bc(self, points: TensorLike) -> TensorLike:
        """集中载荷 (点力)"""
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val = bm.set_at(val, (..., 1), self._P) 
        
        return val
    
    @cartesian
    def is_concentrate_load_boundary_dof(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]  

        on_top_boundary = bm.abs(y - domain[3]) < self._eps
        on_left_boundary = bm.abs(x - domain[0]) < self._eps

        return on_top_boundary & on_left_boundary

    def is_concentrate_load_boundary(self) -> Callable:

        return self.is_concentrate_load_boundary_dof
    
    def get_passive_element_mask(self, 
                                mesh: HomogeneousMesh,
                                load_region: tuple = (3, 2),
                                support_region: tuple = (3, 3),
                            ) -> TensorLike:
        """生成被动单元掩码"""
        # 1. 确定总单元数和父方格索引映射
        nx , ny  = mesh.meshdata['nx'], mesh.meshdata['ny']

        if isinstance(mesh, TriangleMesh):
            n_elements = 2 * nx * ny
            # 生成所有三角形的索引
            el_indices = bm.arange(n_elements)
            # 核心修正：将三角形索引映射回它所在的“父方格”索引
            # 例如：三角形 0,1 -> 方格 0；三角形 2,3 -> 方格 1
            grid_cell_indices = el_indices // 2
        elif isinstance(mesh, QuadrangleMesh):
            n_elements = nx * ny
            el_indices = bm.arange(n_elements)
            # 四边形本身就是方格，索引不变
            grid_cell_indices = el_indices

        # 2. 基于父方格索引计算空间坐标 (ix, iy)
        # 注意：这里依然使用 ny，因为网格在几何上仍然是 nx 列 ny 行
        ix = grid_cell_indices // ny  # 列号
        iy = grid_cell_indices % ny   # 行号
        
        # 3. 区域判定 (逻辑与之前完全一致，因为是基于 ix, iy 判定的)
        
        # 载荷点区域（左上角）
        load_w, load_h = load_region
        # 增加边界保护
        limit_load_w = min(load_w, nx)
        mask_load = (ix < limit_load_w) & (iy >= ny - load_h)
        
        # 支座点区域（右下角）
        support_w, support_h = support_region
        # 增加边界保护
        limit_support_w = max(nx - support_w, 0)
        mask_support = (ix >= limit_support_w) & (iy < support_h)
        
        return mask_load | mask_support


class MBBBeam2d(PDEBase):
    """
    全区域 MBB 梁的 PDE 模型
    
    设计域:
        - 全设计域: 60 mm x 10 mm

    边界条件:
        - 左下角铰支座 (u_x = u_y = 0)
        - 右下角滑移支座 (u_y = 0)

    载荷条件:
        - 上中点施加向下的集中载荷 P = -2 [N]
    
    材料参数:
        E = 1 [MPa], nu = 0.3
    """

    def __init__(self,
                domain: List[float] = [0.0, 60.0, 0.0, 10.0],
                mesh_type: str = 'uniform_quad',
                P: float = -2.0,  # N
                E: float = 1.0,   # MPa
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

        self._eps = 1e-12
        self._load_type = 'concentrated'
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
    def P(self) -> float:
        """获取集中力"""
        return self._P

    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 10)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                       threshold=threshold, device=device)
        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)

        return mesh

    @init_mesh.register('uniform_aligned_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 10)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                     threshold=threshold, device=device)
        self._save_meshdata(mesh, 'uniform_aligned_tri', nx=nx, ny=ny)

        return mesh

    @init_mesh.register('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 10)
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
    def body_force(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self._domain
        x, y = points[..., 0], points[..., 1]
        temp = (bm.abs(x - domain[0]) < self._eps) & (bm.abs(y - domain[2]) < self._eps)

        return temp

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self._domain
        x, y = points[..., 0], points[..., 1]
        left = (bm.abs(x - domain[0]) < self._eps) & (bm.abs(y - domain[2]) < self._eps)
        right = (bm.abs(x - domain[1]) < self._eps) & (bm.abs(y - domain[2]) < self._eps)
        temp = left | right
    
        return temp

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x,
                self.is_dirichlet_boundary_dof_y)
    
    @cartesian
    def concentrate_load_bc(self, points: TensorLike) -> TensorLike:
        """集中载荷 (点力)"""
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val = bm.set_at(val, (..., 1), self._P) 
        
        return val
    
    @cartesian
    def is_concentrate_load_boundary_dof(self, points: TensorLike) -> TensorLike:
        domain = self._domain
        xm = 0.5 * (domain[0] + domain[1])
        x, y = points[..., 0], points[..., 1]   
        coord = (bm.abs(x - xm) < self._eps) & (bm.abs(y - domain[3]) < self._eps)
        
        return coord
    
    def is_concentrate_load_boundary(self) -> Callable:

        return self.is_concentrate_load_boundary_dof
