from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, TriangleMesh
from fealpy.decorator import cartesian, variantmethod
from fealpy.typing import TensorLike, Callable

from .pde_base import PDEBase

class BoxTriHuZhangData2d():
    """来源论文: A simple conforming mixed finite element for linear elasticity on rectangular grids in any space dimension
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
    def __init__(self, lam: float = 1.0, mu: float = 0.5) -> None:
        self.lam = lam
        self.mu = mu
        self.eps = 1e-12
        self.plane_type = 'plane_strain'

    def domain(self) -> list:
        return [0, 1, 0, 1]
    
    def stress_matrix_coefficient(self) -> tuple[float, float]:
        """
        材料为均匀各向同性线弹性体时, 计算应力块矩阵的系数 lambda0 和 lambda1
        
        Returns:
        --------
        lambda0: 1/(2μ)
        lambda1: λ/(2μ(dλ+2μ)), 其中 d=2 为空间维数
        """
        d = 2 
        lambda0 = 1.0 / (2 * self.mu)
        lambda1 = self.lam / (2 * self.mu * (d * self.lam + 2 * self.mu))
        
        return lambda0, lambda1
    
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
    def displacement_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        exp_xy = bm.exp(x - y)

        u1 = exp_xy * x * (1 - x) * y * (1 - y)
        u2 = bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

        val = bm.stack([u1, u2], axis=-1)

        return val

    @cartesian
    def displacement_bc(self, points: TensorLike) -> TensorLike:
        return self.displacement_solution(points)
    
    @cartesian
    def is_dirichlet_boundary(self, points: TensorLike) -> TensorLike:
        domain = self.domain()
        x = points[..., 0]
        y = points[..., 1]

        flag_x0 = bm.abs(x - domain[0]) < self.eps
        flag_x1 = bm.abs(x - domain[1]) < self.eps
        flag_y0 = bm.abs(y - domain[2]) < self.eps
        flag_y1 = bm.abs(y - domain[3]) < self.eps
        
        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1
        
        return flag
    
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
    
class BoxTriLagrangeData2d(PDEBase):
    """
    模型来源:
    https://wnesm678i4.feishu.cn/wiki/JvPPwCD9niMSTZkztTpcIcLxnne#share-LLcgd9YBAoJlvhxags2cR74Tnnd

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
        self._force_type = 'continuous'
        self._boundary_type = 'dirichlet'

        self._log_info(f"Initialized BoxTriLagrangeData2d with domain={self._domain}, "
                       f"mesh_type='{mesh_type}', E={E}, nu={nu}, "
                       f"plane_type='{self._plane_type}', force_type='{self._force_type}', boundary_type='{self._boundary_type}'")
    @property
    def E(self) -> float:
        """获取杨氏模量"""
        return self._E
    
    @property
    def nu(self) -> float:
        """获取泊松比"""
        return self._nu

    @variantmethod('uniform_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)
        
        self._save_mesh(mesh, 'uniform_tri', nx=nx, ny=ny, threshold=threshold, device=device)

        return mesh
    
    @init_mesh.register('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)

        self._save_mesh(mesh, 'uniform_quad', nx=nx, ny=ny, threshold=threshold, device=device)

        return mesh

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
    

class ManufacturedSolutionData2d(PDEBase):
    """
    使用造解法构建的线弹性验证模型
    该模型用于验证有限元求解器在材料属性非均匀分布情况下的收敛阶。

    -∇·σ(u, E(ρ(x))) = b    in Ω
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
                E0: float = 1.0, 
                E_min: float = 1e-9, 
                nu: float = 0.3, 
                p: float = 3.0,
                density_type: str = 'smooth',
                enable_logging: bool = False, 
                logger_name: Optional[str] = None 
            ) -> None:

        super().__init__(domain=domain, mesh_type=mesh_type, 
                        enable_logging=enable_logging, logger_name=logger_name)

        self._E0, self._E_min, self._nu, self._p = E0, E_min, nu, p
        self._density_type = density_type

        self._eps = 1e-12
        self._plane_type = 'plane_strain'
        
        self._log_info(f"Initialized ManufacturedSolutionData2d with E0={E0}, nu={nu}, p={p}, "
                       f"density_type='{density_type}'")
        
    @variantmethod('uniform_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny, device=device)
        self._save_mesh(mesh, 'uniform_tri', nx=nx, ny=ny, device=device)

        return mesh
    
    @init_mesh.register('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny, device=device)
        self._save_mesh(mesh, 'uniform_quad', nx=nx, ny=ny, device=device)

        return mesh

    @variantmethod('element_density')
    def density_field(self, points: TensorLike) -> TensorLike:
        """
        定义密度场 rho(x, y)。
        默认实现 'smooth' 类型。
        """
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        # 一个平滑的、在[0, 1]范围内变化的密度场
        rho = 0.5 * (bm.sin(pi * x) * bm.cos(2 * pi * y) + 1)
        return self._E_min + (self._E0 - self._E_min) * rho

    @density_field.register('discontinuous_circle')
    def _density_discontinuous(self, points: TensorLike) -> TensorLike:
        """不连续的圆形密度场"""
        x, y = points[..., 0], points[..., 1]
        center_x, center_y, r = 0.5, 0.5, 0.25
        # 判断点是否在圆内
        is_in_circle = (x - center_x)**2 + (y - center_y)**2 < r**2
        rho = bm.where(is_in_circle, 1.0, 0.1) # 圆内密度为1，圆外为0.1
        return self._E_min + (self._E0 - self._E_min) * rho

    def youngs_modulus(self, points: TensorLike) -> TensorLike:
        """根据密度场计算杨氏模量 E(x, y)"""
        rho = self.density_field(points, type=self._density_type)
        E = self._E_min + rho**self._p * (self._E0 - self._E_min)
        return E
    
    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi

        u_x = bm.sin(pi * x)**2 * bm.sin(pi * y)**2
        u_y = bm.sin(pi * x)**2 * bm.sin(pi * y)**2

        u = bm.stack([u_x, u_y], axis=-1)

        return u

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        
        E0 = self._E0
        E_min = self._E_min
        nu = self._nu
        p = self._p
        
        sin_pix = bm.sin(pi * x)
        cos_pix = bm.cos(pi * x)
        sin_piy = bm.sin(pi * y)
        cos_piy = bm.cos(pi * y)
        sin_2piy = bm.sin(2 * pi * y)
        cos_2piy = bm.cos(2 * pi * y)

        rho = 0.5 * (sin_pix * cos_2piy + 1)
        rho_x = 0.5 * pi * cos_pix * cos_2piy
        rho_y = -pi * sin_pix * sin_2piy

        E_term = E0 - E_min
        E = E_min + E_term * rho**p
        E_x = E_term * p * rho**(p - 1) * rho_x
        E_y = E_term * p * rho**(p - 1) * rho_y

        c1 = 1 / (2 * (1 + nu))
        c2 = nu / ((1 + nu) * (1 - 2 * nu))
        mu = E * c1
        mu_x = E_x * c1
        mu_y = E_y * c1
        lmbda = E * c2
        lmbda_x = E_x * c2
        lmbda_y = E_y * c2

        u_x = sin_pix**2 * sin_piy**2
        u_x_x = 2 * pi * sin_pix * cos_pix * sin_piy**2
        u_x_y = 2 * pi * sin_pix**2 * sin_piy * cos_piy
        u_x_xx = 2 * pi**2 * (cos_pix**2 - sin_pix**2) * sin_piy**2
        u_x_xy = 4 * pi**2 * sin_pix * cos_pix * sin_piy * cos_piy
        u_x_yy = 2 * pi**2 * sin_pix**2 * (cos_piy**2 - sin_piy**2)

        u_y = u_x 
        u_y_x = u_x_x
        u_y_y = u_x_y
        u_y_xx = u_x_xx
        u_y_xy = u_x_xy
        u_y_yy = u_x_yy

        eps_xx = u_x_x
        eps_yy = u_y_y
        eps_xy = 0.5 * (u_x_y + u_y_x)
        
        tr_eps = eps_xx + eps_yy
        sigma_xx = (2 * mu + lmbda) * eps_xx + lmbda * eps_yy
        sigma_yy = (2 * mu + lmbda) * eps_yy + lmbda * eps_xx
        sigma_xy = 2 * mu * eps_xy

        sigma_xx_x = (2 * mu_x + lmbda_x) * eps_xx + (2 * mu + lmbda) * u_x_xx + \
                     lmbda_x * eps_yy + lmbda * u_y_xy
        sigma_yy_y = (2 * mu_y + lmbda_y) * eps_yy + (2 * mu + lmbda) * u_y_yy + \
                     lmbda_y * eps_xx + lmbda * u_x_xy
        sigma_xy_x = 2 * mu_x * eps_xy + 2 * mu * 0.5 * (u_x_xy + u_y_xx)
        sigma_xy_y = 2 * mu_y * eps_xy + 2 * mu * 0.5 * (u_x_yy + u_y_xy)

        f_x = -(sigma_xx_x + sigma_xy_y)
        f_y = -(sigma_xy_x + sigma_yy_y)

        return bm.stack([f_x, f_y], axis=-1)

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        return self.disp_solution(points)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        flag = (bm.abs(x - domain[0]) < self._eps) | (bm.abs(x - domain[1]) < self._eps) | \
               (bm.abs(y - domain[2]) < self._eps) | (bm.abs(y - domain[3]) < self._eps)
        return flag

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        flag = (bm.abs(x - domain[0]) < self._eps) | (bm.abs(x - domain[1]) < self._eps) | \
               (bm.abs(y - domain[2]) < self._eps) | (bm.abs(y - domain[3]) < self._eps)
        return flag

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)


    

class PolyDisp2dData:
    """多项式位移算例"""
    def __init__(self) -> None:
        pass

    def domain(self) -> list:
        return [0, 1, 0, 1]

    @cartesian
    def source(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        val = bm.zeros(points.shape, dtype=bm.float64)
        val[..., 0] = 35/13*y - 35/13*y**2 + 10/13*x - 10/13*x**2
        val[..., 1] = -25/26*(-1+2*y) * (-1+2*x)

        return val

    @cartesian
    def solution(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        val = bm.zeros(points.shape, dtype=bm.float64)
        val[..., 0] = x*(1-x)*y*(1-y)
        val[..., 1] = 0

        return val

    @cartesian
    def dirichlet(self, points: TensorLike) -> Callable[[TensorLike], TensorLike]:
        
        return self.solution(points)

    @cartesian
    def is_dirichlet_boundary(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        flag1 = bm.abs(x - self.domain()[0]) < 1e-13
        flag2 = bm.abs(x - self.domain()[1]) < 1e-13
        flagx = bm.logical_or(flag1, flag2)
        flag3 = bm.abs(y - self.domain()[2]) < 1e-13
        flag4 = bm.abs(y - self.domain()[3]) < 1e-13
        flagy = bm.logical_or(flag3, flag4)
        flag = bm.logical_or(flagx, flagy)

        return flag
    
