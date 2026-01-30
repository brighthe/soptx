from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, TriangleMesh
from fealpy.decorator import cartesian, variantmethod
from fealpy.typing import TensorLike, Callable

from soptx.model.pde_base import PDEBase

class HZmfemGeneralShearDirichlet(PDEBase):
    """
    二维线弹性 (胡张混合元) —— 剪切应力 + 纯位移边界条件模型

    解析位移:
        u(x, y) = [ exp(x-y) * x(1-x)y(1-y),
                    sin(πx)sin(πy) ]^T

    解析应力:
        σ_xx = (λ+2μ) * exp(x-y) * y(1-y)(1 - x - x^2) + λ*π*sin(πx)sin(πy)
        σ_xy = μ * [ exp(x-y) * (1 - 3y + y^2) * x(1-x) + π*cos(πx)sin(πy) ]
        σ_yy = (λ+2μ) * π * sin(πx)sin(πy) + λ * exp(x-y) * x(1-x)(1 - 3y + y^2)

    体力密度: 
        b(x, y) = [ 2(x^2+3x)(y-y^2) - 0.5(x-x^2)(-y^2+5y-4) ] * exp(x-y) - 1.5π^2 * cos(πx)cos(πy),
                    -1.5(1-x-x^2)(1-3y+y^2) * E + 2.5π^2 * sin(πx)sin(πy) ]^T
    
    边界条件:
        全边界 (x=0, x=1, y=0, y=1) 均为本质位移边界条件 u = u_exact
    """
    def __init__(self, 
                domain: List[float] = [0, 1, 0, 1],
                mesh_type: str = 'uniform_crisscross_tri', 
                lam: float = 1.0, mu: float = 0.5,
                plane_type: str = 'plane_strain',       
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)
                
        self._lam, self._mu = lam, mu
        self._eps = 1e-12

        self._plane_type = plane_type
        self._load_type = None   
        self._boundary_type = 'neumann'

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def mu(self) -> float:
        return self._mu

    @variantmethod('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 10)
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
    
    @init_mesh.register('union_crisscross')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        node = bm.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=bm.float64)
        cell = bm.array([[4, 0, 1], [4, 1, 3], [4, 3, 2], [4, 2, 0]], dtype=bm.int32)
        mesh = TriangleMesh(node, cell)

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
    def body_force(self, points: TensorLike) -> 'TensorLike':
        """体力密度 b(x, y)"""
        x, y = points[..., 0], points[..., 1]
        exp_xy = bm.exp(x - y)

        b1_term1 = 2 * (x**2 + 3*x) * (y - y**2) * exp_xy
        b1_term2 = -0.5 * (x - x**2) * (-y**2 + 5*y - 4) * exp_xy
        b1_term3 = -1.5 * bm.pi**2 * bm.cos(bm.pi * x) * bm.cos(bm.pi * y)
        b1 = b1_term1 + b1_term2 + b1_term3

        b2_term1 = -1.5 * (1 - x - x**2) * (1 - 3*y + y**2) * exp_xy
        b2_term2 = 2.5 * bm.pi**2 * bm.sin(bm.pi * x) * bm.sin(bm.pi * y)
        b2 = b2_term1 + b2_term2

        val = bm.stack([b1, b2], axis=-1)

        return  val
    
    @cartesian
    def displacement_bc(self, points: TensorLike) -> TensorLike:
        """位移边界条件 u_D(x, y)"""
        return self.displacement_solution(points)
    
    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """标记位移边界"""
        x = points[..., 0]
        y = points[..., 1]
        
        is_x_bd = (bm.abs(x - self._domain[0]) < self._eps) | (bm.abs(x - self._domain[1]) < self._eps)
        is_y_bd = (bm.abs(y - self._domain[2]) < self._eps) | (bm.abs(y - self._domain[3]) < self._eps)

        return is_x_bd | is_y_bd
    
    def traction_bc(self, points: TensorLike) -> TensorLike:
        """牵引边界条件 g_N(x, y)"""
        val = bm.zeros(points.shape[:-1] + (2,), dtype=bm.float64)
        return val
    
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记牵引边界"""
        return bm.zeros(points.shape[:-1], dtype=bm.bool)
    
    @cartesian
    def displacement_solution(self, points: TensorLike) -> TensorLike:
        """位移解析解 u(x, y)"""
        x, y = points[..., 0], points[..., 1]
        exp_xy = bm.exp(x - y)

        u1 = exp_xy * x * (1 - x) * y * (1 - y)
        u2 = bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

        val = bm.stack([u1, u2], axis=-1)

        return val
    
    @cartesian
    def grad_displacement_solution(self, points: TensorLike) -> 'TensorLike':
        """位移解析解梯度 ∇u"""
        x, y = points[..., 0], points[..., 1]
        exp_xy = bm.exp(x - y)
        pi = bm.pi

        du1_dx = exp_xy * y * (1 - y) * (1 - x - x**2)
        du1_dy = exp_xy * x * (1 - x) * (1 - 3*y + y**2)

        du2_dx = pi * bm.cos(pi * x) * bm.sin(pi * y)
        du2_dy = pi * bm.sin(pi * x) * bm.cos(pi * y)
        
        grad = bm.stack([
                        bm.stack([du1_dx, du1_dy], axis=-1),  # [∂u1/∂x, ∂u1/∂y]
                        bm.stack([du2_dx, du2_dy], axis=-1)   # [∂u2/∂x, ∂u2/∂y]
                    ], axis=-2)
        
        return grad
    
    @cartesian
    def stress_solution(self, points: TensorLike) -> 'TensorLike':
        """应力解析解 σ = [σ_xx, σ_xy, σ_yy]"""
        x, y = points[..., 0], points[..., 1]
        exp_xy = bm.exp(x - y)
        pi = bm.pi
        
        du1_dx = exp_xy * y * (1 - y) * (1 - x - x**2) 
        du1_dy = exp_xy * x * (1 - x) * (1 - 3*y + y**2)
        du2_dx = pi * bm.cos(pi * x) * bm.sin(pi * y)
        du2_dy = pi * bm.sin(pi * x) * bm.cos(pi * y)
        
        eps_xx = du1_dx
        eps_yy = du2_dy
        eps_xy = 0.5 * (du1_dy + du2_dx)
        
        lam, mu = self.lam, self.mu
        sigma_xx = lam * (eps_xx + eps_yy) + 2 * mu * eps_xx
        sigma_yy = lam * (eps_xx + eps_yy) + 2 * mu * eps_yy
        sigma_xy = 2 * mu * eps_xy
        
        val = bm.stack([sigma_xx, sigma_xy, sigma_yy], axis=-1)
        
        return val
    
    @cartesian
    def div_stress_solution(self, points: TensorLike) -> 'TensorLike':
        """应力解析解散度 ∇·σ"""
        return -self.body_force(points)


class HZmfemZeroShearDirichlet(PDEBase):
    """
    二维线弹性 (胡张混合元) —— 零剪切应力 + 纯位移边界条件模型
 
    解析位移:
        u(x, y) = [ sin(πx)sin(πy) + x, 
                    cos(πx)cos(πy) + y ]^T

    解析应力:
        σ_xx = 2(λ+μ) + 2μπ cos(πx)sin(πy)
        σ_xy = 0
        σ_yy = 2(λ+μ) - 2μπ cos(πx)sin(πy)

    体力密度: 
        b(x, y) = [ 2μ * π^2 * sin(πx)sin(πy),
                    2μ * π^2 * cos(πx)cos(πy) ]^T
    
    边界条件:
        全边界 (x=0, x=1, y=0, y=1) 均为本质位移边界条件 u = u_exact
    """
    def __init__(self, 
                domain: List[float] = [0, 1, 0, 1],
                mesh_type: str = 'uniform_crisscross_tri', 
                lam: float = 0.125, mu: float = 0.125,
                plane_type: str = 'plane_strain',       
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)
                
        self._lam, self._mu = lam, mu
        self._eps = 1e-12

        self._plane_type = plane_type
        self._load_type = None   
        self._boundary_type = 'neumann'

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def mu(self) -> float:
        return self._mu

    @variantmethod('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 10)
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
    
    @init_mesh.register('union_crisscross')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        node = bm.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=bm.float64)
        cell = bm.array([[4, 0, 1], [4, 1, 3], [4, 3, 2], [4, 2, 0]], dtype=bm.int32)
        mesh = TriangleMesh(node, cell)

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
    def body_force(self, points: TensorLike) -> 'TensorLike':
        """体力密度 b(x, y)"""
        x = points[..., 0]
        y = points[..., 1]
        val = bm.zeros(points.shape[:-1] + (2, ), dtype=bm.float64)
        pi = bm.pi
        mu = self._mu
        
        factor = 2 * mu * (pi**2)
        val[..., 0] = factor * bm.sin(pi * x) * bm.sin(pi * y)
        val[..., 1] = factor * bm.cos(pi * x) * bm.cos(pi * y)
        
        return val
    
    @cartesian
    def displacement_bc(self, points: TensorLike) -> TensorLike:
        """位移边界条件 u_D(x, y)"""
        return self.displacement_solution(points)
    
    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """标记位移边界"""
        x = points[..., 0]
        y = points[..., 1]
        
        is_x_bd = (bm.abs(x - self._domain[0]) < self._eps) | (bm.abs(x - self._domain[1]) < self._eps)
        is_y_bd = (bm.abs(y - self._domain[2]) < self._eps) | (bm.abs(y - self._domain[3]) < self._eps)

        return is_x_bd | is_y_bd
    
    def traction_bc(self, points: TensorLike) -> TensorLike:
        """牵引边界条件 g_N(x, y)"""
        val = bm.zeros(points.shape[:-1] + (2,), dtype=bm.float64)
        return val
    
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记牵引边界"""
        return bm.zeros(points.shape[:-1], dtype=bm.bool)
    
    @cartesian
    def displacement_solution(self, points: TensorLike) -> TensorLike:
        """位移解析解 u(x, y)"""
        x = points[..., 0]
        y = points[..., 1]
        val = bm.zeros(points.shape[:-1] + (2, ), dtype=bm.float64)
        pi = bm.pi
        
        val[..., 0] = bm.sin(pi * x) * bm.sin(pi * y) + x
        val[..., 1] = bm.cos(pi * x) * bm.cos(pi * y) + y
        
        return val
    
    @cartesian
    def grad_displacement_solution(self, points: TensorLike) -> 'TensorLike':
        """位移解析解梯度 ∇u"""
        x = points[..., 0]
        y = points[..., 1]
        val = bm.zeros(points.shape[:-1] + (2, 2), dtype=bm.float64)
        pi = bm.pi
        
        sin_pix = bm.sin(pi * x)
        cos_pix = bm.cos(pi * x)
        sin_piy = bm.sin(pi * y)
        cos_piy = bm.cos(pi * y)
        
        val[..., 0, 0] = pi * cos_pix * sin_piy + 1.0
        val[..., 0, 1] = pi * sin_pix * cos_piy
        val[..., 1, 0] = -pi * sin_pix * cos_piy
        val[..., 1, 1] = -pi * cos_pix * sin_piy + 1.0
        
        return val
    
    @cartesian
    def stress_solution(self, points: TensorLike) -> 'TensorLike':
        """应力解析解 σ = [σ_xx, σ_xy, σ_yy]"""
        x = points[..., 0]
        y = points[..., 1]
        val = bm.zeros(points.shape[:-1] + (3, ), dtype=bm.float64)
        pi = bm.pi
        lam, mu = self._lam, self._mu
        
        const_term = 2 * (lam + mu)
        wave_term = 2 * mu * pi * bm.cos(pi * x) * bm.sin(pi * y)

        val[..., 0] = const_term + wave_term
        val[..., 1] = 0.0
        val[..., 2] = const_term - wave_term
        
        return val
    
    @cartesian
    def div_stress_solution(self, points: TensorLike) -> 'TensorLike':
        """应力解析解散度 ∇·σ"""
        return -self.body_force(points)
    

class HZmfemZeroShearMix(PDEBase):
    """
    二维线弹性 (胡张混合元) —— 零剪切应力 + 混合边界条件
    
    解析位移:
        u(x, y) = [ sin(πx)sin(πy) + x, 
                    cos(πx)cos(πy) + y ]^T
    
    解析应力:
        σ_xx = 2(λ+μ) + 2μπ cos(πx)sin(πy)
        σ_xy = 0
        σ_yy = 2(λ+μ) - 2μπ cos(πx)sin(πy)
    
    体力密度: 
        b(x, y) = [ 2μ * π^2 * sin(πx)sin(πy),
                    2μ * π^2 * cos(πx)cos(πy) ]^T
    
    边界条件:
    """
    def __init__(self, 
                domain: List[float] = [0, 1, 0, 1],
                mesh_type: str = 'uniform_crisscross_tri', 
                lam: float = 0.125, 
                mu: float = 0.125,
                plane_type: str = 'plane_strain', # 'plane_stress' or 'plane_strain'        
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)
                
        self._lam, self._mu = lam, mu
        self._eps = 1e-12

        self._plane_type = plane_type
        self._load_type = 'distributed'   
        self._boundary_type = 'mixed'

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def mu(self) -> float:
        return self._mu

    @variantmethod('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        # TODO 目前胡张元仅支持这种网格类型
        nx = kwargs.get('nx', 10)
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
    
    @init_mesh.register('union_crisscross')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        node = bm.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=bm.float64)
        cell = bm.array([[4, 0, 1], [4, 1, 3], [4, 3, 2], [4, 2, 0]], dtype=bm.int32)
        mesh = TriangleMesh(node, cell)

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
    def body_force(self, points: TensorLike) -> 'TensorLike':
        """体力密度 b(x, y)"""
        x = points[..., 0]
        y = points[..., 1]
        val = bm.zeros(points.shape[:-1] + (2, ), dtype=bm.float64)
        pi = bm.pi
        mu = self._mu
        
        factor = 2 * mu * (pi**2)
        val[..., 0] = factor * bm.sin(pi * x) * bm.sin(pi * y)
        val[..., 1] = factor * bm.cos(pi * x) * bm.cos(pi * y)
        
        return val
    
    @cartesian
    def displacement_bc(self, points: TensorLike) -> TensorLike:
        """位移边界条件 u_D(x, y)"""

        return self.displacement_solution(points)
    
    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """标记位移边界"""
        x = points[..., 0]
        y = points[..., 1]
        is_bottom_bd = bm.abs(y - self._domain[2]) < self._eps  
        is_top_bd    = bm.abs(y - self._domain[3]) < self._eps 
        is_left_bd   = bm.abs(x - self._domain[0]) < self._eps  

        return is_bottom_bd | is_top_bd | is_left_bd
    
    def traction_bc(self, points: TensorLike) -> TensorLike:
        """牵引边界条件 g_N(x, y) - 应力分量 σ"""
        return self.stress_solution(points)

    # def traction_bc(self, points: TensorLike) -> TensorLike:
    #     """牵引边界条件 g_N(x, y) - 牵引力分量 σ·n"""
    #     val = bm.zeros(points.shape[:-1] + (2,), dtype=bm.float64)
        
    #     x = points[..., 0]
        
    #     flag_right = bm.abs(x - self._domain[1]) < self._eps
        
    #     if bm.any(flag_right):
    #         sigma = self.stress_solution(points) 
            
    #         sigma_xx = sigma[..., 0]
    #         sigma_xy = sigma[..., 1]
            
    #         val[flag_right, 0] = sigma_xx[flag_right]
    #         val[flag_right, 1] = sigma_xy[flag_right]

    #     return val
    
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记牵引边界"""
        x = points[..., 0]
        is_right_bd = bm.abs(x - self._domain[1]) < self._eps

        return is_right_bd
    
    @cartesian
    def displacement_solution(self, points: TensorLike) -> TensorLike:
        """位移解析解 u(x, y)"""
        x = points[..., 0]
        y = points[..., 1]
        val = bm.zeros(points.shape[:-1] + (2, ), dtype=bm.float64)
        pi = bm.pi
        
        val[..., 0] = bm.sin(pi * x) * bm.sin(pi * y) + x
        val[..., 1] = bm.cos(pi * x) * bm.cos(pi * y) + y
        
        return val
    
    @cartesian
    def grad_displacement_solution(self, points: TensorLike) -> 'TensorLike':
        """位移解析解梯度 ∇u"""
        x = points[..., 0]
        y = points[..., 1]
        val = bm.zeros(points.shape[:-1] + (2, 2), dtype=bm.float64)
        pi = bm.pi
        
        sin_pix = bm.sin(pi * x)
        cos_pix = bm.cos(pi * x)
        sin_piy = bm.sin(pi * y)
        cos_piy = bm.cos(pi * y)
        
        val[..., 0, 0] = pi * cos_pix * sin_piy + 1.0
        val[..., 0, 1] = pi * sin_pix * cos_piy
        val[..., 1, 0] = -pi * sin_pix * cos_piy
        val[..., 1, 1] = -pi * cos_pix * sin_piy + 1.0
        
        return val
    
    @cartesian
    def stress_solution(self, points: TensorLike) -> 'TensorLike':
        """应力解析解 σ = [σ_xx, σ_xy, σ_yy]"""
        x = points[..., 0]
        y = points[..., 1]
        val = bm.zeros(points.shape[:-1] + (3, ), dtype=bm.float64)
        pi = bm.pi
        lam, mu = self._lam, self._mu
        
        const_term = 2 * (lam + mu)
        wave_term = 2 * mu * pi * bm.cos(pi * x) * bm.sin(pi * y)

        val[..., 0] = const_term + wave_term
        val[..., 1] = 0.0
        val[..., 2] = const_term - wave_term
        
        return val
    
    @cartesian
    def div_stress_solution(self, points: TensorLike) -> 'TensorLike':
        """应力解析解散度 ∇·σ"""
        return -self.body_force(points)
    
    
class HZmfemGeneralShearMix(PDEBase):
    """
    二维线弹性 (胡张混合元) —— 一般剪切应力 + 混合边界条件

    解析位移:
        u(x, y) = [ sin(πx/2) · sin(πy),
                   -2 sin(πx) · (sin(πy/2)-y) ]^T

    解析应力:
        

    体力密度: 
        b(x, y) = [ π^2( sin(πx/2) sin(πy) + (3/2) cos(πx) cos(πy/2) ) - 2π(λ+μ) cos(πx),
                   -(3/4)π^2 cos(πx/2) cos(πy) - 2π^2 sin(πx) sin(πy/2) + 2μ π^2 y sin(πx) ]^T

    位移边界:
        下边界  y = 0
        上边界  y = 1
        左边界  x = 0

    牵引力边界:
        右边界  x = 1,  n = ( 1, 0), t = (0,  1): g(1, y) = [ 0,
                                                            (π/2)cos(πy) + πsin(πy/2) - 2μπy ]^T
    """
    def __init__(self, 
            domain: List[float] = [0, 1, 0, 1],
            mesh_type: str = 'uniform_crisscross_tri', 
            lam: float = 1.0, 
            mu: float = 0.5,
            plane_type: str = 'plane_strain', # 'plane_stress' or 'plane_strain'                     
            enable_logging: bool = False, 
            logger_name: Optional[str] = None) -> None:
        
        super().__init__(domain=domain, mesh_type=mesh_type, 
                        enable_logging=enable_logging, logger_name=logger_name)
        self._lam, self._mu = lam, mu
        self._eps = 1e-12

        self._plane_type = plane_type
        self._load_type = 'distributed'   
        self._boundary_type = 'mixed'

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def mu(self) -> float:
        return self._mu

    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                       threshold=threshold, device=device)
        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)
        return mesh

    @init_mesh.register('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 10)
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
    
    @init_mesh.register('union_crisscross')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        node = bm.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=bm.float64)
        cell = bm.array([[4, 0, 1], [4, 1, 3], [4, 3, 2], [4, 2, 0]], dtype=bm.int32)
        mesh = TriangleMesh(node, cell)

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
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        lam, mu = self.lam, self.mu

        b1 = (pi**2) * (
                bm.sin(0.5 * pi * x) * bm.sin(pi * y)
            + 1.5 * bm.cos(pi * x) * bm.cos(0.5 * pi * y)
            ) - 2.0 * pi * (lam + mu) * bm.cos(pi * x)

        b2 = -(0.75) * (pi**2) * bm.cos(0.5 * pi * x) * bm.cos(pi * y) \
            - 2.0 * (pi**2) * bm.sin(pi * x) * bm.sin(0.5 * pi * y) \
            + 2.0 * mu * (pi**2) * y * bm.sin(pi * x)

        return bm.stack([b1, b2], axis=-1)
    
    @cartesian
    def displacement_bc(self, points: TensorLike) -> TensorLike:
        """位移边界条件 u_D(x, y)"""

        return self.displacement_solution(points)
        
    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        """标记位移边界"""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        is_left_bd = bm.abs(x - domain[0]) < self._eps 
        is_bottom_bd = bm.abs(y - domain[2]) < self._eps
        is_top_bd = bm.abs(y - domain[3]) < self._eps  

        return is_left_bd | is_bottom_bd | is_top_bd
    
    @cartesian
    def traction_bc(self, points: TensorLike) -> TensorLike:
        """牵引边界条件 g_N(x, y) - 应力分量 σ"""
        return self.stress_solution(points)
    
    # @cartesian
    # def traction_bc(self, points: TensorLike) -> TensorLike:
    #     """牵引边界条件 g_N(x, y) - 牵引力分量 σ·n"""
    #     domain = self.domain
    #     x, y = points[..., 0], points[..., 1]
    #     pi = bm.pi
    #     lam, mu = self.lam, self.mu  

    #     kwargs = bm.context(points)
    #     val = bm.zeros(points.shape, **kwargs)

    #     flag_right = bm.abs(x - domain[1]) < self._eps
    #     if bm.any(flag_right):
            
    #         g_y = mu * pi * (
    #             bm.cos(pi * y) + 
    #             2.0 * bm.sin(0.5 * pi * y) - 
    #             2.0 * y
    #         )
    #         val = bm.set_at(val, (flag_right, 1), g_y[flag_right])

    #     return val
    
    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        """标记牵引边界"""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        is_right_bd = bm.abs(x - domain[1]) < self._eps  

        return is_right_bd
    
    @cartesian
    def displacement_solution(self, points: TensorLike) -> TensorLike:
        """位移解析解 u(x, y)"""
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        u1 = bm.sin(0.5 * pi * x) * bm.sin(pi * y)
        u2 = -2.0 * bm.sin(pi * x) * (bm.sin(0.5 * pi * y) - y)

        return bm.stack([u1, u2], axis=-1)

    @cartesian
    def grad_displacement_solution(self, points: TensorLike) -> TensorLike:
        """位移解析解梯度 ∇u"""
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        du1_dx = 0.5 * pi * bm.cos(0.5 * pi * x) * bm.sin(pi * y)
        du1_dy =        pi * bm.sin(0.5 * pi * x) * bm.cos(pi * y)
        du2_dx =  2.0 * pi * bm.cos(pi * x) * (y - bm.sin(0.5 * pi * y))
        du2_dy = -      pi * bm.sin(pi * x) * bm.cos(0.5 * pi * y) + 2.0 * bm.sin(pi * x)

        return bm.stack([
                        bm.stack([du1_dx, du1_dy], axis=-1),
                        bm.stack([du2_dx, du2_dy], axis=-1)
                    ], axis=-2)

    @cartesian
    def stress_solution(self, points: TensorLike) -> TensorLike:
        """应力解析解 σ = [σ_xx, σ_xy, σ_yy]"""
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        lam, mu = self.lam, self.mu

        du1_dx = 0.5 * pi * bm.cos(0.5 * pi * x) * bm.sin(pi * y)
        du1_dy =        pi * bm.sin(0.5 * pi * x) * bm.cos(pi * y)
        du2_dx =  2.0 * pi * bm.cos(pi * x) * (y - bm.sin(0.5 * pi * y))
        du2_dy = -      pi * bm.sin(pi * x) * bm.cos(0.5 * pi * y) + 2.0 * bm.sin(pi * x)

        eps_xx = du1_dx
        eps_yy = du2_dy
        eps_xy = 0.5 * (du1_dy + du2_dx)
        tr_eps = eps_xx + eps_yy

        sigma_xx = 2 * mu * eps_xx + lam * tr_eps
        sigma_yy = 2 * mu * eps_yy + lam * tr_eps
        sigma_xy = 2 * mu * eps_xy

        return bm.stack([sigma_xx, sigma_xy, sigma_yy], axis=-1)
    
    @cartesian
    def div_stress_solution(self, points: TensorLike) -> TensorLike:
        """应力解析解散度 ∇·σ"""
        return -self.body_force(points)