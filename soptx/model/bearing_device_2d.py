from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.model.pde_base import PDEBase  


class HalfBearingDevice2D(PDEBase):
    '''
    Symmetric half-domain model for the bearing device example from
    Castañar et al. (2022), Section 5.2.

    PDEs:
    -∇·σ = 0      in Ω (左半区域)
      u = 0        on ∂Ω_bottom (clamped bottom boundary, y=0)
      u_x = 0      on ∂Ω_left (symmetry boundary, x=0)
      σ·n = t      on ∂Ω_top (distributed traction, y=0.4)

    几何参数:
        左半部分矩形域, 尺寸为 0.6m x 0.4m
        底部完全固支, 右侧为对称边界, 顶部施加向下的分布载荷
    
    载荷类型:
        分布载荷(面力) (单位 - N/m)

    材料参数:
        E_s = 100 Pa, nu_s = 0.5 (incompressible)
    '''
    def __init__(self,
                domain: List[float] = [0, 0.6, 0, 0.4],  
                mesh_type: str = 'uniform_tri',
                t: float = -1.8, 
                E: float = 100.0,
                nu: float = 0.5,
                plane_type: str = 'plane_strain', # 'plane_stress' or 'plane_strain'
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
    def t(self) -> float:
        """获取面力"""
        return self._t

    #######################################################################################################################
    # 变体方法 (Variant Methods) - 网格生成
    #######################################################################################################################

    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        # 根据几何调整默认单元数
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 40)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                       threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)

        return mesh
    
    @init_mesh.register('uniform_aligned_tri')
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

    #######################################################################################################################
    # 核心方法
    #######################################################################################################################

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def neumann_bc(self, points: TensorLike) -> TensorLike:
        """
        σ·n = 0 on Γ_N1
        σ·n = t on Γ_N2
        上边界 y=1, n=(0, 1):  t(x, 1) = [0, -1.8]^T
        左边界 x=0, n=(-1, 0): t(0, y) = [0, 0]^T
        """
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)

        # 优先处理非齐次边界(上边界)
        flag_top = bm.abs(y - domain[3]) < self._eps
        val = bm.set_at(val, (flag_top, 0), 0.0)  
        val = bm.set_at(val, (flag_top, 1), self._t) 

        # 处理齐次边界(左边界), 但要排除已作为上边界处理的角点
        flag_left = (bm.abs(x - domain[0]) < self._eps) & (~flag_top)
        val = bm.set_at(val, (flag_left, 0), 0.0)
        val = bm.set_at(val, (flag_left, 1), 0.0)

        # # 上边界 y = 1
        # flag_top = bm.abs(y - domain[3]) < self._eps
        # val = bm.set_at(val, (flag_top, 0), 0.0)  
        # val = bm.set_at(val, (flag_top, 1), self._t)

        # # 左边界 x = 0
        # flag_left = bm.abs(x - domain[0]) < self._eps
        # val = bm.set_at(val, (flag_left, 0), 0.0)
        # val = bm.set_at(val, (flag_left, 1), 0.0)

        return val
    
    @cartesian
    def neumann_bc_normal(self, points: TensorLike) -> TensorLike:
        """获取 Neumann 边界上的单位外法向量"""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        
        kwargs = bm.context(points)
        normals = bm.zeros((points.shape[0], 2), **kwargs)

        # 优先处理非齐次边界(上边界)
        flag_top = bm.abs(y - domain[3]) < self._eps
        normals = bm.set_at(normals, (flag_top, 1), 1.0)
        
        # 处理齐次边界(左边界), 但要排除已作为上边界处理的角点
        flag_left = (bm.abs(x - domain[0]) < self._eps) & (~flag_top)
        normals = bm.set_at(normals, (flag_left, 0), -1.0)

        # # 左边界 x = 0, n = (-1, 0)
        # flag_left = bm.abs(x - domain[0]) < self._eps
        # normals = bm.set_at(normals, (flag_left, 0), -1.0)
        
        # # 上边界 y = 1, n = (0, 1)
        # flag_top = bm.abs(y - domain[3]) < self._eps
        # normals = bm.set_at(normals, (flag_top, 1), 1.0)

        return normals
    
    @cartesian
    def is_neumann_boundary_dof_xx(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x = points[..., 0]

        # 左边界 σ_xx = 0
        on_left_boundary = bm.abs(x - domain[0]) < self._eps

        return on_left_boundary
    
    @cartesian
    def is_neumann_boundary_dof_xy(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x = points[..., 0]
        y = points[..., 1]

        # 左边界和上边界 σ_xy = 0
        on_left_boundary = bm.abs(x - domain[0]) < self._eps
        on_top_boundary = bm.abs(y - domain[3]) < self._eps

        return on_left_boundary | on_top_boundary
    
    @cartesian
    def is_neumann_boundary_dof_yy(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        y = points[..., 1]

        # 上边界 σ_yy = 0
        on_top_boundary = bm.abs(y - domain[3]) < self._eps

        return on_top_boundary
    
    def is_neumann_boundary(self) -> Tuple[Callable, Callable, Callable]:
        
        return (self.is_neumann_boundary_dof_xx,
                self.is_neumann_boundary_dof_xy,
                self.is_neumann_boundary_dof_yy)

    
    # @cartesian
    # def neumann_bc(self, points: TensorLike) -> TensorLike:
    #     kwargs = bm.context(points)
    #     val = bm.zeros(points.shape, **kwargs)
    #     val = bm.set_at(val, (..., 1), self._t) 
        
    #     return val
    
    # @cartesian
    # def is_neumann_boundary_dof(self, points: TensorLike) -> TensorLike:
    #     domain = self.domain
    #     y = points[..., 1]

    #     on_top_boundary = bm.abs(y - domain[3]) < self._eps

    #     return on_top_boundary

    # def is_neumann_boundary(self) -> Callable:
        
    #     return self.is_neumann_boundary_dof

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """
        判断 x 方向的 Dirichlet 边界自由度
        - 底部边界(y=0) 完全固支 -> u_x = 0
        - 右侧边界(x=0) 对称 -> u_x = 0
        """
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        on_bottom_boundary = bm.abs(y - domain[2]) < self._eps
        on_right_boundary = bm.abs(x - domain[1]) < self._eps

        return on_bottom_boundary | on_right_boundary

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """
        判断 y 方向的 Dirichlet 边界自由度
        - 底部边界(y=0) 完全固支 -> u_y = 0
        - 右侧对称边界 y 方向自由
        """
        domain = self.domain
        y = points[..., 1]
        
        on_bottom_boundary = bm.abs(y - domain[2]) < self._eps

        return on_bottom_boundary

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x,
                self.is_dirichlet_boundary_dof_y)


class BearingDevice2D(PDEBase):
    '''
    Full-domain model for the bearing device example from
    Castañar et al. (2022), Section 5.2.

    PDEs:
    -∇·σ = 0      in Ω (full domain)
      u = 0        on ∂Ω_bottom (clamped bottom boundary, y=0)
      σ·n = t      on ∂Ω_top (distributed traction, y=0.4)

    几何参数:
        完整矩形域, 尺寸为 1.2m x 0.4m
        底部完全固支, 顶部施加向下的分布载荷
    
    载荷类型:
        分布载荷(面力) (单位 - N/m)

    材料参数:
        E_s = 100 Pa, nu_s = 0.5 (incompressible)
    '''
    def __init__(self,
                domain: List[float] = [0, 120, 0, 40],  
                mesh_type: str = 'uniform_tri',
                t: float = -1.8, 
                E: float = 100.0,
                nu: float = 0.5,
                plane_type: str = 'plane_strain', # 'plane_stress' or 'plane_strain'
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
    def t(self) -> float:
        """获取面力"""
        return self._t

    #######################################################################################################################
    # 变体方法 (Variant Methods) - 网格生成
    #######################################################################################################################

    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        # 根据几何调整默认单元数（完整区域是半区域的2倍宽）
        nx = kwargs.get('nx', 120)  # 2 * 60
        ny = kwargs.get('ny', 40)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                       threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)

        return mesh
    
    @init_mesh.register('uniform_aligned_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 120)  # 2 * 60
        ny = kwargs.get('ny', 40)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_aligned_tri', nx=nx, ny=ny)

        return mesh

    @init_mesh.register('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 120)  # 2 * 60
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

    #######################################################################################################################
    # 核心方法
    #######################################################################################################################

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    # @cartesian
    # def neumann_bc(self, points: TensorLike) -> TensorLike:
    #     """
    #     σ·n = 0 on Γ_N1
    #     σ·n = t on Γ_N2
    #     上边界 y=1, n=(0, 1):  t(x, 1) = [0, -1.8]^T
    #     左边界 x=0, n=(-1, 0): t(0, y) = [0, 0]^T
    #     """
    #     domain = self.domain
    #     x, y = points[..., 0], points[..., 1]

    #     kwargs = bm.context(points)
    #     val = bm.zeros(points.shape, **kwargs)

    #     # 上边界 y = 1
    #     flag_top = bm.abs(y - domain[3]) < self._eps
    #     val = bm.set_at(val, (flag_top, 0), 0.0)  
    #     val = bm.set_at(val, (flag_top, 1), self._t)

    #     # 左边界 x = 0
    #     flag_left = bm.abs(x - domain[0]) < self._eps
    #     val = bm.set_at(val, (flag_left, 0), 0.0)
    #     val = bm.set_at(val, (flag_left, 1), 0.0)

    #     return val
    
    # @cartesian
    # def neumann_bc_normal(self, points: TensorLike) -> TensorLike:
    #     """获取 Neumann 边界上的单位外法向量"""
    #     domain = self.domain
    #     x, y = points[..., 0], points[..., 1]
        
    #     kwargs = bm.context(points)
    #     normals = bm.zeros((points.shape[0], 2), **kwargs)

    #     # 左边界 x = 0, n = (-1, 0)
    #     flag_left = bm.abs(x - domain[0]) < self._eps
    #     normals = bm.set_at(normals, (flag_left, 0), -1.0)
        
    #     # 上边界 y = 1, n = (0, 1)
    #     flag_top = bm.abs(y - domain[3]) < self._eps
    #     normals = bm.set_at(normals, (flag_top, 1), 1.0)

    #     return normals
    
    # @cartesian
    # def is_neumann_boundary_dof_xx(self, points: TensorLike) -> TensorLike:
    #     domain = self.domain
    #     x = points[..., 0]

    #     # 左边界 σ_xx = 0
    #     on_left_boundary = bm.abs(x - domain[0]) < self._eps

    #     return on_left_boundary
    
    # @cartesian
    # def is_neumann_boundary_dof_xy(self, points: TensorLike) -> TensorLike:
    #     domain = self.domain
    #     x = points[..., 0]
    #     y = points[..., 1]

    #     # 左边界和上边界 σ_xy = 0
    #     on_left_boundary = bm.abs(x - domain[0]) < self._eps
    #     on_top_boundary = bm.abs(y - domain[3]) < self._eps

    #     return on_left_boundary | on_top_boundary
    
    # @cartesian
    # def is_neumann_boundary_dof_yy(self, points: TensorLike) -> TensorLike:
    #     domain = self.domain
    #     y = points[..., 1]

    #     # 上边界 σ_yy = 0
    #     on_top_boundary = bm.abs(y - domain[3]) < self._eps

    #     return on_top_boundary
    
    # def is_neumann_boundary(self) -> Tuple[Callable, Callable, Callable]:
        
    #     return (self.is_neumann_boundary_dof_xx,
    #             self.is_neumann_boundary_dof_xy,
    #             self.is_neumann_boundary_dof_yy)

    
    @cartesian
    def is_neumann_boundary_dof(self, points: TensorLike) -> TensorLike:
        """判断 Neumann 边界: 顶部边界"""
        domain = self.domain
        y = points[..., 1]

        on_top_boundary = bm.abs(y - domain[3]) < self._eps

        return on_top_boundary

    def is_neumann_boundary(self) -> Callable:
        
        return self.is_neumann_boundary_dof

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """Dirichlet 边界条件: 底部完全固支, 位移为零"""
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """
        判断 x 方向的 Dirichlet 边界自由度
        - 底部边界(y=0) 完全固支 -> u_x = 0
        """
        domain = self.domain
        y = points[..., 1]

        on_bottom_boundary = bm.abs(y - domain[2]) < self._eps

        return on_bottom_boundary

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """
        判断 y 方向的 Dirichlet 边界自由度
        - 底部边界(y=0) 完全固支 -> u_y = 0
        """
        domain = self.domain
        y = points[..., 1]
        
        on_bottom_boundary = bm.abs(y - domain[2]) < self._eps

        return on_bottom_boundary

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x,
                self.is_dirichlet_boundary_dof_y)