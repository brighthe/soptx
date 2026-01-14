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
    
    def get_passive_element_mask(self, nx: int, ny: int) -> TensorLike:
        """
        生成被动单元掩码 (适用于列主序的单元编号)
        
        区域定义：
        1. 载荷点 (左上角): 3x2 区域 
           防止点载荷引起的应力奇异性
        2. 支座点 (右下角): 3x3 区域
           防止点支撑引起的应力奇异性
        """        
        n_elements = nx * ny
        
        # === 修正点：使用列主序 (Column-Major) ===
        # 对应图片：编号先沿 Y 轴增加
        el_indices = bm.arange(n_elements)
        ix = el_indices // ny  # 整除 ny 得到列号 x
        iy = el_indices % ny   # 对 ny 取余得到行号 y
        
        # 区域 1: 载荷点 (左上角 3x2)
        # x < 3, y >= ny - 2
        load_region_w = 3
        load_region_h = 2 
        mask_load = (ix < load_region_w) & (iy >= (ny - load_region_h))
        
        # 区域 2: 支座点 (右下角 3x3)
        # x >= nx - 3, y < 3
        support_region_w = 3
        support_region_h = 3
        mask_support = (ix >= (nx - support_region_w)) & (iy < support_region_h)
        
        mask = mask_load | mask_support
        
        return mask

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
