from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, TriangleMesh
from fealpy.decorator import cartesian, variantmethod
from fealpy.typing import TensorLike, Callable

from soptx.model.pde_base import PDEBase

class TriSolDirHuZhangData(PDEBase):
    """
    胡张混合有限元
        带真解的二维线弹性算例 (纯齐次 Dirichlet 边界条件)

    -∇·σ = b    in Ω
      Aσ = ε(u) in Ω
       u = 0    on ∂Ω (homogeneous Dirichlet)
    where:
    - σ is the stress tensor
    - ε = (∇u + ∇u^T)/2 is the strain tensor
    - A is the compliance tensor
    
    Material parameters:
        lam = 1.0, mu = 0.5
    
    For isotropic materials:
        Aσ = (1/2μ)σ - (λ/(2μ(dλ+2μ)))tr(σ)I
    """
    def __init__(self, 
                domain: List[float] = [0, 1, 0, 1],
                mesh_type: str = 'uniform_aligned_tri', 
                lam: float = 1.0, mu: float = 0.5,
                plane_type: str = 'plane_strain', # 'plane_stress' or 'plane_strain'                
                enable_logging: bool = False, 
                logger_name: Optional[str] = None 
            ) -> None:
        
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)
        
        self._lam, self._mu = lam, mu
        self._eps = 1e-12

        self._plane_type = plane_type
        self._load_type = None
        self._boundary_type = 'dirichlet'
    

    #######################################################################################################################
    # 访问器
    #######################################################################################################################
    
    @property
    def lam(self) -> float:
        """获取拉梅常数 λ"""
        return self._lam
    
    @property
    def mu(self) -> float:
        """获取剪切模量 μ"""
        return self._mu
    

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
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        exp_xy = bm.exp(x - y)

        u1 = exp_xy * x * (1 - x) * y * (1 - y)
        u2 = bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

        val = bm.stack([u1, u2], axis=-1)

        return val
    
    @cartesian
    def grad_disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        exp_xy = bm.exp(x - y)
        pi = bm.pi
        
        # u1 = exp(x-y) * x * (1-x) * y * (1-y)
        # ∂u1/∂x = exp(x-y) * y * (1-y) * [x*(1-x) + (1-2x)]
        #        = exp(x-y) * y * (1-y) * (1 - x - x²)
        du1_dx = exp_xy * y * (1 - y) * (1 - x - x**2)
        
        # ∂u1/∂y = exp(x-y) * x * (1-x) * [-y*(1-y) + (1-2y)]  
        #        = exp(x-y) * x * (1-x) * (1 - 3y + y²)
        du1_dy = exp_xy * x * (1 - x) * (1 - 3*y + y**2)
        
        # u2 = sin(πx) * sin(πy)
        # ∂u2/∂x = π * cos(πx) * sin(πy)
        du2_dx = pi * bm.cos(pi * x) * bm.sin(pi * y)
        
        # ∂u2/∂y = π * sin(πx) * cos(πy)
        du2_dy = pi * bm.sin(pi * x) * bm.cos(pi * y)
        
        grad = bm.stack([
            bm.stack([du1_dx, du1_dy], axis=-1),  # [∂u1/∂x, ∂u1/∂y]
            bm.stack([du2_dx, du2_dy], axis=-1)   # [∂u2/∂x, ∂u2/∂y]
        ], axis=-2)
        
        return grad

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        val = self.disp_solution(points)
        
        return val
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps
        flag_y0 = bm.abs(y - domain[2]) < self._eps
        flag_y1 = bm.abs(y - domain[3]) < self._eps

        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1

        return flag

    @cartesian  
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps
        flag_y0 = bm.abs(y - domain[2]) < self._eps
        flag_y1 = bm.abs(y - domain[3]) < self._eps

        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1
        
        return flag

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:
        
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    
    @cartesian
    def stress_solution(self, points: TensorLike) -> TensorLike:
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
        
        # 按照 2D 对称张量的存储顺序返回 (σ_xx, σ_xy, σ_yy)
        val = bm.stack([sigma_xx, sigma_xy, sigma_yy], axis=-1)
        
        return val
    
    @cartesian
    def div_stress_solution(self, points: TensorLike) -> TensorLike:
        """
        计算应力张量的散度: div(σ) = [∂σ_xx/∂x + ∂σ_xy/∂y, ∂σ_xy/∂x + ∂σ_yy/∂y]
        也可以根据平衡方程: -∇·σ = b, 所以 ∇·σ = -b
        """
        x, y = points[..., 0], points[..., 1]
        exp_xy = bm.exp(x - y)
        pi = bm.pi
        lam, mu = self.lam, self.mu
        
        # 计算应变分量的导数
        # ε_xx = exp(x-y) * y * (1-y) * (1 - x - x²)
        # ∂ε_xx/∂x = exp(x-y) * y * (1-y) * [(1 - x - x²) + (-1 - 2x)]
        deps_xx_dx = exp_xy * y * (1 - y) * (-x**2 - 3*x)
        
        # ∂ε_xx/∂y = exp(x-y) * (1 - x - x²) * [-y*(1-y) + (1-2y)]
        deps_xx_dy = exp_xy * (1 - x - x**2) * (1 - 3*y + y**2)
        
        # ε_yy = π * sin(πx) * cos(πy)
        # ∂ε_yy/∂x = π² * cos(πx) * cos(πy)
        deps_yy_dx = pi**2 * bm.cos(pi * x) * bm.cos(pi * y)
        
        # ∂ε_yy/∂y = -π² * sin(πx) * sin(πy)
        deps_yy_dy = -pi**2 * bm.sin(pi * x) * bm.sin(pi * y)
        
        # ε_xy = 0.5 * [exp(x-y)*x*(1-x)*(1-3y+y²) + π*cos(πx)*sin(πy)]
        # ∂ε_xy/∂x = 0.5 * [exp(x-y)*(1-3y+y²)*(1-x-x²) - π²*sin(πx)*sin(πy)]
        deps_xy_dx = 0.5 * (exp_xy * (1 - 3*y + y**2) * (1 - x - x**2) 
                            - pi**2 * bm.sin(pi * x) * bm.sin(pi * y))
        
        # ∂ε_xy/∂y = 0.5 * [exp(x-y)*x*(1-x)*(-4+5y-y²) + π²*cos(πx)*cos(πy)]
        deps_xy_dy = 0.5 * (exp_xy * x * (1 - x) * (-4 + 5*y - y**2) 
                            + pi**2 * bm.cos(pi * x) * bm.cos(pi * y))
        
        # 计算应力分量的导数
        # σ_xx = (λ + 2μ) * ε_xx + λ * ε_yy
        # ∂σ_xx/∂x = (λ + 2μ) * ∂ε_xx/∂x + λ * ∂ε_yy/∂x
        dsigma_xx_dx = (lam + 2*mu) * deps_xx_dx + lam * deps_yy_dx
        
        # σ_xy = 2μ * ε_xy  
        # ∂σ_xy/∂y = 2μ * ∂ε_xy/∂y
        dsigma_xy_dy = 2 * mu * deps_xy_dy
        
        # ∂σ_xy/∂x = 2μ * ∂ε_xy/∂x
        dsigma_xy_dx = 2 * mu * deps_xy_dx
        
        # σ_yy = λ * ε_xx + (λ + 2μ) * ε_yy
        # ∂σ_yy/∂y = λ * ∂ε_xx/∂y + (λ + 2μ) * ∂ε_yy/∂y
        dsigma_yy_dy = lam * deps_xx_dy + (lam + 2*mu) * deps_yy_dy
        
        # 计算散度的两个分量
        div_x = dsigma_xx_dx + dsigma_xy_dy
        div_y = dsigma_xy_dx + dsigma_yy_dy
        
        val = bm.stack([div_x, div_y], axis=-1)
        
        return val


class PolySolDirHuZhangData(PDEBase):
    """
    胡张混合有限元
        带多项式真解的二维线弹性算例 (纯齐次 Dirichlet 边界条件)

    -∇·σ = b      in Ω = (-1, 1)^2
      Aσ = ε(u)  in Ω
       u = 0      on ∂Ω (homogeneous Dirichlet)
    where:
    - σ is the stress tensor
    - ε = (∇u + ∇u^T)/2 is the strain tensor
    - A is the compliance tensor
    
    Material parameters:
        lam = 0.3, mu = 0.35
    
    For isotropic materials:
        Aσ = (1/2μ)σ - (λ/(2μ(dλ+2μ)))tr(σ)I
    """
    def __init__(self, 
                domain: List[float] = [-1, 1, -1, 1],
                mesh_type: str = 'uniform_aligned_tri', 
                lam: float = 0.3, mu: float = 0.35,
                plane_type: str = 'plane_strain', # 'plane_stress' or 'plane_strain'            
                enable_logging: bool = False, 
                logger_name: Optional[str] = None 
            ) -> None:
        
        super().__init__(domain=domain, mesh_type=mesh_type, 
                         enable_logging=enable_logging, logger_name=logger_name)
        
        self._lam, self._mu = lam, mu
        self._eps = 1e-12

        self._plane_type = plane_type
        self._load_type = None
        self._boundary_type = 'dirichlet'


    #######################################################################################################################
    # 访问器
    #######################################################################################################################
    
    @property
    def lam(self) -> float:
        """获取拉梅常数 λ"""
        return self._lam
    
    @property
    def mu(self) -> float:
        """获取剪切模量 μ"""
        return self._mu
    

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

        # This part remains the same as it generates a mesh within a unit box
        # and then scales it to self._domain.
        qmesh = QuadrangleMesh.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny)
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
        
        # Scale nodes to the correct domain
        xmin, xmax, ymin, ymax = self._domain
        scaled_node = bm.zeros_like(node)
        scaled_node[:, 0] = xmin + node[:, 0] * (xmax - xmin)
        scaled_node[:, 1] = ymin + node[:, 1] * (ymax - ymin)
        
        mesh = TriangleMesh(scaled_node, bm.asarray(newCell, dtype=bm.int32))

        self._save_meshdata(mesh, 'uniform_crisscross_tri', nx=nx, ny=ny)
        return mesh
    
    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        
        b1_factor = -8 * (x + y)
        b1_term1 = (3*x*y - 2) * (x**2 + y**2)
        b1_term2 = 5 * (x*y - 1)**2
        b1_term3 = -2 * x**2 * y**2
        b1 = b1_factor * (b1_term1 + b1_term2 + b1_term3)

        b2_factor = -8 * (x - y)
        b2_term1 = (3*x*y + 2) * (x**2 + y**2)
        b2_term2 = -5 * (x*y + 1)**2
        b2_term3 = 2 * x**2 * y**2
        b2 = b2_factor * (b2_term1 + b2_term2 + b2_term3)
        
        val = bm.stack([b1, b2], axis=-1)

        return val

    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        
        c1 = 80.0 / 7.0
        c2 = -4.0
        
        term1_u1 = -c1 * y * (1 - y**2) * (1 - x**2)**2
        term2_u1 = c2 * x * (1 - x**2) * (1 - y**2)**2
        u1 = term1_u1 + term2_u1

        term1_u2 = c1 * x * (1 - x**2) * (1 - y**2)**2
        term2_u2 = c2 * y * (1 - y**2) * (1 - x**2)**2
        u2 = term1_u2 + term2_u2
        
        val = bm.stack([u1, u2], axis=-1)

        return val
    
    @cartesian
    def grad_disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        
        A = -80.0 / 7.0
        B = -4.0
        C = 80.0 / 7.0
        D = -4.0
        
        # ∂u1/∂x
        du1_dx_t1 = -4 * A * x * y * (1 - y**2) * (1 - x**2)
        du1_dx_t2 = B * (1 - 3*x**2) * (1 - y**2)**2
        du1_dx = du1_dx_t1 + du1_dx_t2
        
        # ∂u1/∂y
        du1_dy_t1 = A * (1 - 3*y**2) * (1 - x**2)**2
        du1_dy_t2 = -4 * B * x * y * (1 - x**2) * (1 - y**2)
        du1_dy = du1_dy_t1 + du1_dy_t2
        
        # ∂u2/∂x
        du2_dx_t1 = C * (1 - 3*x**2) * (1 - y**2)**2
        du2_dx_t2 = -4 * D * x * y * (1 - y**2) * (1 - x**2)
        du2_dx = du2_dx_t1 + du2_dx_t2
        
        # ∂u2/∂y
        du2_dy_t1 = -4 * C * x * y * (1 - x**2) * (1 - y**2)
        du2_dy_t2 = D * (1 - 3*y**2) * (1 - x**2)**2
        du2_dy = du2_dy_t1 + du2_dy_t2
        
        grad = bm.stack([
                bm.stack([du1_dx, du1_dy], axis=-1),  # [∂u1/∂x, ∂u1/∂y]
                bm.stack([du2_dx, du2_dy], axis=-1)   # [∂u2/∂x, ∂u2/∂y]
            ], axis=-2)
        
        return grad

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        val = self.disp_solution(points)
        
        return val
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps
        flag_y0 = bm.abs(y - domain[2]) < self._eps
        flag_y1 = bm.abs(y - domain[3]) < self._eps

        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1
        return flag

    @cartesian  
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps
        flag_y0 = bm.abs(y - domain[2]) < self._eps
        flag_y1 = bm.abs(y - domain[3]) < self._eps

        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1

        return flag

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    
    @cartesian
    def stress_solution(self, points: TensorLike) -> TensorLike:
        grad_u = self.grad_disp_solution(points)
        du1_dx = grad_u[..., 0, 0]
        du1_dy = grad_u[..., 0, 1]
        du2_dx = grad_u[..., 1, 0]
        du2_dy = grad_u[..., 1, 1]
        
        eps_xx = du1_dx
        eps_yy = du2_dy
        eps_xy = 0.5 * (du1_dy + du2_dx)
        
        lam, mu = self.lam, self.mu
        sigma_xx = lam * (eps_xx + eps_yy) + 2 * mu * eps_xx
        sigma_yy = lam * (eps_xx + eps_yy) + 2 * mu * eps_yy
        sigma_xy = 2 * mu * eps_xy
        
        # 按照 2D 对称张量的存储顺序返回 (σ_xx, σ_xy, σ_yy)
        val = bm.stack([sigma_xx, sigma_xy, sigma_yy], axis=-1)
        
        return val
    
    @cartesian
    def div_stress_solution(self, points: TensorLike) -> TensorLike:
        """
        根据平衡方程: -∇·σ = b, 所以 ∇·σ = -b
        """
        return -self.body_force(points)


class TriSolMixHuZhangData(PDEBase):
    r"""
    胡张混合有限元
        二维线弹性（平面应变）—— 解析解 + 混合边界条件 (左/下 Dirichlet, 右/上 Neumann)

    方程：
        Aσ = ε(u)                     in Ω
       -∇·σ = b                       in Ω
            u = 0                    on Γ_D = {x=0} ∪ {y=0}
          σ·n = t                    on Γ_N = {x=1} ∪ {y=1}

    材料参数：
        λ = 1.0, μ = 0.5  (⇒ 2μ = 1)

    本构（平面应变）：
        σ = 2μ ε + λ tr(ε) I
      ⇔  Aσ = ε,   A 为柔度算子 (与 λ, μ 对偶)

    解析位移 (三角函数):
        u(x,y) = [ sin(πx/2)·sin(πy),
                  -2 sin(πx) ·sin(πy/2) ]^T

      体力密度: 
        b(x,y) =
            [  π^2 ( sin(πx/2) sin(πy) + (3/2) cos(πx) cos(πy/2) ),
            - π^2 ( 2 sin(πx) sin(πy/2) + (3/4) cos(πx/2) cos(πy) ) ]^T

      Neumann:
        右边界 x=1, n=(1,0):  t(1,y) = [ 0,
                                         (π/2) cos(πy) + π sin(πy/2) ]^T
        上边界 y=1, n=(0,1):  t(x,1) = [ - (π/2) sin(πx/2) - π cos(πx),
                                          0 ]^T
    """
    def __init__(self, 
                domain: List[float] = [0, 1, 0, 1],
                mesh_type: str = 'uniform_aligned_tri', 
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

    # ----------------------------
    # 访问器
    # ----------------------------
    @property
    def lam(self) -> float:
        return self._lam

    @property
    def mu(self) -> float:
        return self._mu

    # ----------------------------
    # 网格变体
    # ----------------------------
    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
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
    def disp_solution(self, points: TensorLike) -> TensorLike:
        """解析解 u(x, y)"""
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        u1 = bm.sin(0.5 * pi * x) * bm.sin(pi * y)
        u2 = -2.0 * bm.sin(pi * x) * bm.sin(0.5 * pi * y)

        return bm.stack([u1, u2], axis=-1)

    @cartesian
    def grad_disp_solution(self, points: TensorLike) -> TensorLike:
        """解析解梯度 ∇u"""
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        du1_dx = 0.5 * pi * bm.cos(0.5 * pi * x) * bm.sin(pi * y)
        du1_dy =        pi * bm.sin(0.5 * pi * x) * bm.cos(pi * y)
        du2_dx = -2.0 * pi * bm.cos(pi * x) * bm.sin(0.5 * pi * y)
        du2_dy = -      pi * bm.sin(pi * x) * bm.cos(0.5 * pi * y)
        
        return bm.stack([
                            bm.stack([du1_dx, du1_dy], axis=-1),
                            bm.stack([du2_dx, du2_dy], axis=-1)
                        ], axis=-2)

    @cartesian
    def stress_solution(self, points: TensorLike) -> TensorLike:
        """σ=2μ ε + λ tr(ε) I, 返回顺序 (σ_xx, σ_xy, σ_yy)"""
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        lam, mu = self.lam, self.mu

        # 梯度
        du1_dx = 0.5 * pi * bm.cos(0.5 * pi * x) * bm.sin(pi * y)
        du1_dy =        pi * bm.sin(0.5 * pi * x) * bm.cos(pi * y)
        du2_dx = -2.0 * pi * bm.cos(pi * x) * bm.sin(0.5 * pi * y)
        du2_dy = -      pi * bm.sin(pi * x) * bm.cos(0.5 * pi * y)

        # 应变
        eps_xx = du1_dx
        eps_yy = du2_dy
        eps_xy = 0.5 * (du1_dy + du2_dx)
        tr_eps = eps_xx + eps_yy

        # 应力
        sigma_xx = 2 * mu * eps_xx + lam * tr_eps
        sigma_yy = 2 * mu * eps_yy + lam * tr_eps
        sigma_xy = 2 * mu * eps_xy

        return bm.stack([sigma_xx, sigma_xy, sigma_yy], axis=-1)

    @cartesian
    def div_stress_solution(self, points: TensorLike) -> TensorLike:
        """
        计算应力张量的散度: div(σ) = [∂σ_xx/∂x + ∂σ_xy/∂y, ∂σ_xy/∂x + ∂σ_yy/∂y]
        也可以根据平衡方程: -∇·σ = b, 所以 ∇·σ = -b
        """
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        lam, mu = self.lam, self.mu
        
        # 计算应变分量的导数
        # ε_xx = 0.5 * π * cos(0.5 * π * x) * sin(π * y)
        # ∂ε_xx/∂x = -0.25 * π² * sin(0.5 * π * x) * sin(π * y)
        deps_xx_dx = -0.25 * pi**2 * bm.sin(0.5 * pi * x) * bm.sin(pi * y)
        
        # ∂ε_xx/∂y = 0.5 * π² * cos(0.5 * π * x) * cos(π * y)
        deps_xx_dy = 0.5 * pi**2 * bm.cos(0.5 * pi * x) * bm.cos(pi * y)
        
        # ε_yy = -π * sin(π * x) * cos(0.5 * π * y)
        # ∂ε_yy/∂x = -π² * cos(π * x) * cos(0.5 * π * y)
        deps_yy_dx = -pi**2 * bm.cos(pi * x) * bm.cos(0.5 * pi * y)
        
        # ∂ε_yy/∂y = 0.5 * π² * sin(π * x) * sin(0.5 * π * y)
        deps_yy_dy = 0.5 * pi**2 * bm.sin(pi * x) * bm.sin(0.5 * pi * y)
        
        # ε_xy = 0.5 * [π * sin(0.5 * π * x) * cos(π * y) - 2π * cos(π * x) * sin(0.5 * π * y)]
        # ∂ε_xy/∂x = 0.25 * π² * cos(0.5 * π * x) * cos(π * y) + π² * sin(π * x) * sin(0.5 * π * y)
        deps_xy_dx = (0.25 * pi**2 * bm.cos(0.5 * pi * x) * bm.cos(pi * y) 
                    + pi**2 * bm.sin(pi * x) * bm.sin(0.5 * pi * y))
        
        # ∂ε_xy/∂y = -0.5 * π² * sin(0.5 * π * x) * sin(π * y) - 0.5 * π² * cos(π * x) * cos(0.5 * π * y)
        deps_xy_dy = (-0.5 * pi**2 * bm.sin(0.5 * pi * x) * bm.sin(pi * y) 
                    - 0.5 * pi**2 * bm.cos(pi * x) * bm.cos(0.5 * pi * y))
        
        # 计算应力分量的导数
        # σ_xx = (λ + 2μ) * ε_xx + λ * ε_yy
        # ∂σ_xx/∂x = (λ + 2μ) * ∂ε_xx/∂x + λ * ∂ε_yy/∂x
        dsigma_xx_dx = (lam + 2*mu) * deps_xx_dx + lam * deps_yy_dx
        
        # σ_xy = 2μ * ε_xy
        # ∂σ_xy/∂y = 2μ * ∂ε_xy/∂y
        dsigma_xy_dy = 2 * mu * deps_xy_dy
        
        # ∂σ_xy/∂x = 2μ * ∂ε_xy/∂x
        dsigma_xy_dx = 2 * mu * deps_xy_dx
        
        # σ_yy = λ * ε_xx + (λ + 2μ) * ε_yy
        # ∂σ_yy/∂y = λ * ∂ε_xx/∂y + (λ + 2μ) * ∂ε_yy/∂y
        dsigma_yy_dy = lam * deps_xx_dy + (lam + 2*mu) * deps_yy_dy
        
        # 计算散度的两个分量
        div_x = dsigma_xx_dx + dsigma_xy_dy
        div_y = dsigma_xy_dx + dsigma_yy_dy
        
        val = bm.stack([div_x, div_y], axis=-1)
        
        return val

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        """体力密度 b=-div σ"""
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        b1 =  (pi**2) * ( bm.sin(0.5 * pi * x) * bm.sin(pi * y)
                        + 1.5 * bm.cos(pi * x) * bm.cos(0.5 * pi * y) )
        b2 = -(pi**2) * ( 2.0 * bm.sin(pi * x) * bm.sin(0.5 * pi * y)
                        + 0.75 * bm.cos(0.5 * pi * x) * bm.cos(pi * y) )
        
        return bm.stack([b1, b2], axis=-1)

    # ----------------------------
    # Dirichlet: 左边 x=0 与 下边 y=0
    # ----------------------------
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        val = self.disp_solution(points)

        return val

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        flag_x0 = bm.abs(x - domain[0]) < self._eps  # x=0
        flag_y0 = bm.abs(y - domain[2]) < self._eps  # y=0

        return flag_x0 | flag_y0

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        flag_x0 = bm.abs(x - domain[0]) < self._eps  # x=0
        flag_y0 = bm.abs(y - domain[2]) < self._eps  # y=0
        
        return flag_x0 | flag_y0

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:
        
        return (self.is_dirichlet_boundary_dof_x,
                self.is_dirichlet_boundary_dof_y)

    # ----------------------------
    # Neumann：右边 x=1 与 上边 y=1
    # ----------------------------
    @cartesian
    def neumann_bc(self, points: TensorLike) -> TensorLike:
        """
        σ·n = t on Γ_N
        右边 x=1, n=(1, 0): t(1, y) = [0, (π/2)cos(πy) + π sin(πy/2)]^T
        上边 y=1, n=(0, 1): t(x, 1) = [-(π/2)sin(πx/2) - π cos(πx), 0]^T
        """
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi

        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)

        # 右边界：设置 t_y
        flag_right = bm.abs(x - domain[1]) < self._eps
        t_y_right = 0.5 * pi * bm.cos(pi * y) + pi * bm.sin(0.5 * pi * y)
        val = bm.set_at(val, (flag_right, 1), t_y_right[flag_right])

        # 上边界：设置 t_x
        flag_top = bm.abs(y - domain[3]) < self._eps
        t_x_top = -0.5 * pi * bm.sin(0.5 * pi * x) - pi * bm.cos(pi * x)
        val = bm.set_at(val, (flag_top, 0), t_x_top[flag_top])

        return val

    @cartesian
    def is_neumann_boundary_dof(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        flag_x1 = bm.abs(x - domain[1]) < self._eps  # 右边 x=1
        flag_y1 = bm.abs(y - domain[3]) < self._eps  # 上边 y=1
        
        return flag_x1 | flag_y1

    def is_neumann_boundary(self) -> Callable:
        
        return self.is_neumann_boundary_dof

    @cartesian
    def is_neumann_right_boundary_dof(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x = points[..., 0]
        flag_x1 = bm.abs(x - domain[1]) < self._eps  # 右边 x=1
        
        return flag_x1
    
    @cartesian
    def is_neumann_top_boundary_dof(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        y = points[..., 1]
        flag_y1 = bm.abs(y - domain[3]) < self._eps  # 上边 y=1
        
        return flag_y1


class BoxTriLagrange2dData(PDEBase):
    """
    -∇·σ = b    in Ω
       u = 0    on ∂Ω (homogeneous Dirichlet)
    where:
    - σ is the stress tensor
    - ε = (∇u + ∇u^T)/2 is the strain tensor
    
    Material parameters:
        E = 1, nu = 0.3

    For isotropic materials:
        σ = 2με + λtr(ε)I
    """
    def __init__(self, 
                domain: List[float] = [0, 1, 0, 1],
                mesh_type: str = 'uniform_tri',
                E: float = 1.0, nu: float = 0.3,
                enable_logging: bool = False, 
                logger_name: Optional[str] = None 
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                        enable_logging=enable_logging, logger_name=logger_name)

        self._E, self._nu = E, nu

        self._eps = 1e-12
        self._plane_type = 'plane_strain'
        self._load_type = None
        self._boundary_type = 'dirichlet'


    #######################################################################################################################
    # 访问器
    #######################################################################################################################
    
    @property
    def E(self) -> float:
        """获取杨氏模量"""
        return self._E
    
    @property
    def nu(self) -> float:
        """获取泊松比"""
        return self._nu
    

    #######################################################################################################################
    # 变体方法
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
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi

        f_x = 22.5 * pi**2 / 13 * bm.sin(pi * x) * bm.sin(pi * y)
        f_y = -12.5 * pi**2 / 13 * bm.cos(pi * x) * bm.cos(pi * y)

        f = bm.stack([f_x, f_y], axis=-1)
        
        return f

    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi

        u_x = bm.sin(pi * x) * bm.sin(pi * y)
        u_y = bm.zeros_like(x) 

        u = bm.stack([u_x, u_y], axis=-1)

        return u
    
    @cartesian
    def disp_solution_gradient(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        
        du_x_dx = pi * bm.cos(pi * x) * bm.sin(pi * y)
        du_x_dy = pi * bm.sin(pi * x) * bm.cos(pi * y)
        
        du_y_dx = bm.zeros_like(x)
        du_y_dy = bm.zeros_like(x)
        
        grad_u = bm.stack([
            bm.stack([du_x_dx, du_x_dy], axis=-1),  
            bm.stack([du_y_dx, du_y_dy], axis=-1)   
        ], axis=-2)
        
        return grad_u

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        val = self.disp_solution(points)
        
        return val
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps
        flag_y0 = bm.abs(y - domain[2]) < self._eps
        flag_y1 = bm.abs(y - domain[3]) < self._eps

        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1
        
        return flag

    @cartesian  
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps
        flag_y0 = bm.abs(y - domain[2]) < self._eps
        flag_y1 = bm.abs(y - domain[3]) < self._eps

        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1

        return flag

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    

class BoxTriMixedLagrange2dData(PDEBase):
    """
    混合边界条件的线弹性问题
    
    控制方程：
        -∇·σ = f    in Ω
    
    边界条件：
        u = 0       on Γ_D = {x=0} ∪ {y=0} (Dirichlet)
        σ·n = g     on Γ_N = {x=1} ∪ {y=1} (Neumann)
    
    其中：
        - σ is the stress tensor
        - ε = (∇u + ∇u^T)/2 is the strain tensor
    
    材料参数：
        E = 1, nu = 0.3 (平面应变)
    
    精确解：
        u(x, y) = [sin(πx)sin(πy), 0]^T
    """
    def __init__(self, 
                domain: List[float] = [0, 1, 0, 1],
                mesh_type: str = 'uniform_tri',
                E: float = 1.0, 
                nu: float = 0.3,
                enable_logging: bool = False, 
                logger_name: Optional[str] = None 
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                        enable_logging=enable_logging, logger_name=logger_name)

        self._E = E
        self._nu = nu

        self._eps = 1e-12
        self._plane_type = 'plane_strain'
        self._load_type = 'distributed'  # 分布载荷
        self._boundary_type = 'mixed'     # 混合边界条件

        # 计算 Lamé 常数 (平面应变)
        self._mu = E / (2 * (1 + nu))
        self._lambda = E * nu / ((1 + nu) * (1 - 2 * nu)) 

    #######################################################################################################################
    # 访问器
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
    def mu(self) -> float:
        """获取第一 Lamé 常数 (剪切模量)"""
        return self._mu
    
    @property
    def lam(self) -> float:
        """获取第二 Lamé 常数"""
        return self._lambda

    #######################################################################################################################
    # 变体方法
    #######################################################################################################################

    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
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
        """体力密度向量"""
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi

        f_x = 22.5 * pi**2 / 13 * bm.sin(pi * x) * bm.sin(pi * y)
        f_y = -12.5 * pi**2 / 13 * bm.cos(pi * x) * bm.cos(pi * y)

        f = bm.stack([f_x, f_y], axis=-1)
        
        return f

    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        """精确位移解"""
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi

        u_x = bm.sin(pi * x) * bm.sin(pi * y)
        u_y = bm.zeros_like(x) 

        u = bm.stack([u_x, u_y], axis=-1)

        return u
    
    @cartesian
    def disp_solution_gradient(self, points: TensorLike) -> TensorLike:
        """精确位移梯度"""
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        
        du_x_dx = pi * bm.cos(pi * x) * bm.sin(pi * y)
        du_x_dy = pi * bm.sin(pi * x) * bm.cos(pi * y)
        
        du_y_dx = bm.zeros_like(x)
        du_y_dy = bm.zeros_like(x)
        
        grad_u = bm.stack([
            bm.stack([du_x_dx, du_x_dy], axis=-1),  
            bm.stack([du_y_dx, du_y_dy], axis=-1)   
        ], axis=-2)
        
        return grad_u

    #######################################################################################################################
    # Dirichlet 边界条件（左边和底边）
    #######################################################################################################################

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        val = self.disp_solution(points)
        
        return val
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """判断 x 方向位移是否在 Dirichlet 边界上（左边和底边）"""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        flag_x0 = bm.abs(x - domain[0]) < self._eps  # 左边 x=0
        flag_y0 = bm.abs(y - domain[2]) < self._eps  # 底边 y=0

        flag = flag_x0 | flag_y0
        
        return flag

    @cartesian  
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """判断 y 方向位移是否在 Dirichlet 边界上（左边和底边）"""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        flag_x0 = bm.abs(x - domain[0]) < self._eps  # 左边 x=0
        flag_y0 = bm.abs(y - domain[2]) < self._eps  # 底边 y=0

        flag = flag_x0 | flag_y0

        return flag

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)

    #######################################################################################################################
    # Neumann 边界条件（右边和顶边）
    #######################################################################################################################

    @cartesian
    def neumann_bc(self, points: TensorLike) -> TensorLike:
        """
        Neumann 边界条件: σ·n = g on Γ_N
        
        右边界 x=1, n=(1,0):
            g(1,y) = [-(2μ+λ)π sin(πy), 0]^T
        
        顶边界 y=1, n=(0,1):
            g(x,1) = [-μπ sin(πx), 0]^T = [-(5/13)π sin(πx), 0]^T
        """
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        
        # 右边界 x=1
        flag_right = bm.abs(x - domain[1]) < self._eps
        coef_right = -(2 * self._mu + self._lambda) * pi
        g_x_right = coef_right * bm.sin(pi * y)
        val = bm.set_at(val, (flag_right, 0), g_x_right[flag_right])
        
        # 顶边界 y=1
        flag_top = bm.abs(y - domain[3]) < self._eps
        coef_top = -self._mu * pi  # -5/13 * π
        g_x_top = coef_top * bm.sin(pi * x)
        val = bm.set_at(val, (flag_top, 0), g_x_top[flag_top])
        
        return val

    @cartesian
    def is_neumann_boundary_dof(self, points: TensorLike) -> TensorLike:
        """判断是否在 Neumann 边界上（右边和顶边）"""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        flag_x1 = bm.abs(x - domain[1]) < self._eps  # 右边 x=1
        flag_y1 = bm.abs(y - domain[3]) < self._eps  # 顶边 y=1

        flag = flag_x1 | flag_y1
        
        return flag
    
    def is_neumann_boundary(self) -> Callable:
        
        return self.is_neumann_boundary_dof