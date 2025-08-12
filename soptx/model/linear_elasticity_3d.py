from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh
from fealpy.decorator import cartesian, variantmethod
from fealpy.typing import TensorLike, Callable

from soptx.model.pde_base import PDEBase

class BoxPolyHuZhangData3d(PDEBase):
    """
    模型来源: 

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
                domain: List[float] = [0, 1, 0, 1, 0, 1],
                mesh_type: str = 'uniform_tet', 
                lam: float = 1.0, mu: float = 0.5,                
                enable_logging: bool = False, 
                logger_name: Optional[str] = None 
            ) -> None:
        
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)
        
        self._lam, self._mu = lam, mu
        self._eps = 1e-12
        self._plane_type = '3d'
        self._force_type = 'distribution'
        self._boundary_type = 'dirichlet'

        self._log_info(f"Initialized BoxPolyHuZhangData3d with domain={self._domain}, "
                f"mesh_type='{mesh_type}', lam={lam}, mu={mu}, "
                f"plane_type='{self._plane_type}', force_type='{self._force_type}', boundary_type='{self._boundary_type}'")


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
    
    @variantmethod('uniform_tet')
    def init_mesh(self, **kwargs) -> TetrahedronMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        nz = kwargs.get('nz', 10)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TetrahedronMesh.from_box(
                                    box=self._domain, 
                                    nx=nx, ny=ny, nz=nz,
                                    threshold=threshold, 
                                    device=device
                                )
        
        self._save_mesh(mesh, 'uniform_tet', nx=nx, ny=ny, nz=nz, threshold=threshold, device=device)

        return mesh

    def stress_matrix_coefficient(self) -> tuple[float, float]:
        """
        材料为均匀各向同性线弹性体时, 计算应力块矩阵的系数 lambda0 和 lambda1
        
        Returns:
        --------
        lambda0: 1/(2μ)
        lambda1: λ/(2μ(dλ+2μ)), 其中 d=3 为空间维数
        """
        d = 3
        lambda0 = 1.0 / (2 * self._mu)
        lambda1 = self._lam / (2 * self._mu * (d * self._lam + 2 * self._mu))
        
        return lambda0, lambda1
    
    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        
        # 位移解的系数
        c1, c2, c3 = 16, 32, 64
        
        # 计算常用项
        xy = x * (1 - x) * y * (1 - y)
        xz = x * (1 - x) * z * (1 - z)
        yz = y * (1 - y) * z * (1 - z)
        
        # 计算导数项
        dx2_xy = (1 - 2*x) * (1 - 2*y)
        dx2_xz = (1 - 2*x) * (1 - 2*z)
        dy2_yz = (1 - 2*y) * (1 - 2*z)
        
        # 体力分量 b1
        b1_term1 = c1 * (xz + xy + 4*yz)
        b1_term2 = -3 * c1 * dx2_xy * z * (1 - z)
        b1_term3 = -6 * c1 * dx2_xz * y * (1 - y)
        b1 = b1_term1 + b1_term2 + b1_term3
        
        # 体力分量 b2
        b2_term1 = c2 * (yz + 4*xz + xy)
        b2_term2 = -1.5 * c2 * dx2_xy * z * (1 - z)
        b2_term3 = -6 * c2 * dy2_yz * x * (1 - x)
        b2 = b2_term1 + b2_term2 + b2_term3
        
        # 体力分量 b3
        b3_term1 = c3 * (yz + xz + 4*xy)
        b3_term2 = -1.5 * c3 * dx2_xz * y * (1 - y)
        b3_term3 = -3 * c3 * dy2_yz * x * (1 - x)
        b3 = b3_term1 + b3_term2 + b3_term3
        
        val = bm.stack([b1, b2, b3], axis=-1)
        
        return val
    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        
        common = x * (1 - x) * y * (1 - y) * z * (1 - z)
        
        u1 = 16 * common
        u2 = 32 * common
        u3 = 64 * common
        
        val = bm.stack([u1, u2, u3], axis=-1)
        
        return val
    
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:

        val = self.disp_solution(points)
        
        return val
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:

        domain = self.domain
        x, y, z = points[..., 0], points[..., 1], points[..., 2]

        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps
        flag_y0 = bm.abs(y - domain[2]) < self._eps
        flag_y1 = bm.abs(y - domain[3]) < self._eps
        flag_z0 = bm.abs(z - domain[4]) < self._eps
        flag_z1 = bm.abs(z - domain[5]) < self._eps

        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1 | flag_z0 | flag_z1

        return flag

    @cartesian  
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:

        domain = self.domain
        x, y, z = points[..., 0], points[..., 1], points[..., 2]

        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps
        flag_y0 = bm.abs(y - domain[2]) < self._eps
        flag_y1 = bm.abs(y - domain[3]) < self._eps
        flag_z0 = bm.abs(z - domain[4]) < self._eps
        flag_z1 = bm.abs(z - domain[5]) < self._eps

        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1 | flag_z0 | flag_z1
        
        return flag
    
    @cartesian  
    def is_dirichlet_boundary_dof_z(self, points: TensorLike) -> TensorLike:

        domain = self.domain
        x, y, z = points[..., 0], points[..., 1], points[..., 2]

        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps
        flag_y0 = bm.abs(y - domain[2]) < self._eps
        flag_y1 = bm.abs(y - domain[3]) < self._eps
        flag_z0 = bm.abs(z - domain[4]) < self._eps
        flag_z1 = bm.abs(z - domain[5]) < self._eps

        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1 | flag_z0 | flag_z1
        
        return flag

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y,
                self.is_dirichlet_boundary_dof_z)
    
    @cartesian
    def stress_solution(self, points: TensorLike) -> TensorLike:

        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        
        # 位移解的系数
        c1, c2, c3 = 16, 32, 64
        
        # 计算位移梯度
        # u1 = c1 * x(1-x)y(1-y)z(1-z)
        du1_dx = c1 * (1 - 2*x) * y * (1 - y) * z * (1 - z)
        du1_dy = c1 * x * (1 - x) * (1 - 2*y) * z * (1 - z)
        du1_dz = c1 * x * (1 - x) * y * (1 - y) * (1 - 2*z)
        
        # u2 = c2 * x(1-x)y(1-y)z(1-z)
        du2_dx = c2 * (1 - 2*x) * y * (1 - y) * z * (1 - z)
        du2_dy = c2 * x * (1 - x) * (1 - 2*y) * z * (1 - z)
        du2_dz = c2 * x * (1 - x) * y * (1 - y) * (1 - 2*z)
        
        # u3 = c3 * x(1-x)y(1-y)z(1-z)
        du3_dx = c3 * (1 - 2*x) * y * (1 - y) * z * (1 - z)
        du3_dy = c3 * x * (1 - x) * (1 - 2*y) * z * (1 - z)
        du3_dz = c3 * x * (1 - x) * y * (1 - y) * (1 - 2*z)
        
        # 计算应变张量 ε = (∇u + ∇u^T)/2
        eps_xx = du1_dx
        eps_yy = du2_dy
        eps_zz = du3_dz
        eps_xy = 0.5 * (du1_dy + du2_dx)
        eps_yz = 0.5 * (du2_dz + du3_dy)
        eps_xz = 0.5 * (du1_dz + du3_dx)
        
        # 计算应力张量 σ = λ(tr(ε))I + 2με
        lam, mu = self.lam, self.mu
        trace_eps = eps_xx + eps_yy + eps_zz
        
        sigma_xx = lam * trace_eps + 2 * mu * eps_xx
        sigma_yy = lam * trace_eps + 2 * mu * eps_yy
        sigma_zz = lam * trace_eps + 2 * mu * eps_zz
        sigma_xy = 2 * mu * eps_xy
        sigma_yz = 2 * mu * eps_yz
        sigma_xz = 2 * mu * eps_xz
        
        # 按照 3D 对称张量的 Voigt 记号存储顺序返回
        # 常见顺序: (σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz)
        val = bm.stack([sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz], axis=-1)
        
        return val
    

class BoxPolyLagrange3dData(PDEBase):
    """    
    -∇·σ = b    in Ω
       u = 0    on ∂Ω (homogeneous Dirichlet)
    where:
    - σ is the stress tensor
    - ε = (∇u + ∇u^T)/2 is the strain tensor
    
    Material parameters:
        λ = 1.0, μ = 1.0  (Lamé constants)

    For isotropic materials:
        σ = 2μ ε + λ tr(ε) I
    """
    def __init__(self, 
                domain: List[float] = [0, 1, 0, 1, 0, 1],
                mesh_type: str = 'uniform_tet',
                lam: float = 1.0, mu: float = 1.0,
                enable_logging: bool = False, 
                logger_name: Optional[str] = None 
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                        enable_logging=enable_logging, logger_name=logger_name)

        self._lam, self._mu = lam, mu

        self._eps = 1e-12
        self._plane_type = '3d'
        self._boundary_type = 'dirichlet'
        self._force_type = 'distribution'

        self._log_info(f"Initialized BoxHexLagrange3dData with domain={self._domain}, "
                       f"mesh_type='{mesh_type}', λ={lam}, μ={mu}, "
                       f"force_type='{self._force_type}', "
                       f"boundary_type='{self._boundary_type}'")


    #######################################################################################################################
    # 访问器
    #######################################################################################################################
    
    @property
    def lam(self) -> float:
        """获取拉梅第一常数 λ"""
        return self._lam
    
    @property
    def mu(self) -> float:
        """获取剪切模量 μ"""
        return self._mu
    

    def stress_matrix_coefficient(self) -> tuple[float, float]:
        """
        材料为均匀各向同性线弹性体时, 计算应力块矩阵的系数 lambda0 和 lambda1
        
        Returns:
        --------
        lambda0: 1/(2μ)
        lambda1: λ/(2μ(dλ+2μ)), 其中 d=3 为空间维数
        """
        d = 3
        lambda0 = 1.0 / (2 * self._mu)
        lambda1 = self._lam / (2 * self._mu * (d * self._lam + 2 * self._mu))
        
        return lambda0, lambda1
    
    #######################################################################################################################
    # 变体方法
    #######################################################################################################################

    @variantmethod('uniform_tet')
    def init_mesh(self, **kwargs) -> TetrahedronMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        nz = kwargs.get('nz', 10)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TetrahedronMesh.from_box(
                                    box=self._domain, 
                                    nx=nx, ny=ny, nz=nz,
                                    threshold=threshold, 
                                    device=device
                                )
        
        self._save_mesh(mesh, 'uniform_tet', nx=nx, ny=ny, nz=nz, threshold=threshold, device=device)

        return mesh


    #######################################################################################################################
    # 核心方法
    #######################################################################################################################

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        mu = self._mu

        f_x = -400 * mu * (2*y - 1) * (2*z - 1) * (
                3*(x**2 - x)**2 * (y**2 - y + z**2 - z) +
                (1 - 6*x + 6*x**2) * (y**2 - y) * (z**2 - z)
        )

        f_y = 200 * mu * (2*x - 1) * (2*z - 1) * (
                3*(y**2 - y)**2 * (x**2 - x + z**2 - z) +
                (1 - 6*y + 6*y**2) * (x**2 - x) * (z**2 - z)
        )

        f_z = 200 * mu * (2*x - 1) * (2*y - 1) * (
                3*(z**2 - z)**2 * (x**2 - x + y**2 - y) +
                (1 - 6*z + 6*z**2) * (x**2 - x) * (y**2 - y)
        )

        f = bm.stack([f_x, f_y, f_z], axis=-1)
        return f

    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        mu = self._mu

        u_x = 200 * mu * (x - x**2)**2 * (2*y**3 - 3*y**2 + y) * (2*z**3 - 3*z**2 + z)
        u_y = -100 * mu * (y - y**2)**2 * (2*x**3 - 3*x**2 + x) * (2*z**3 - 3*z**2 + z)
        u_z = -100 * mu * (z - z**2)**2 * (2*x**3 - 3*x**2 + x) * (2*y**3 - 3*y**2 + y)

        u = bm.stack([u_x, u_y, u_z], axis=-1)

        return u

    @cartesian
    def disp_solution_gradient(self, points: TensorLike) -> TensorLike:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        mu = self._mu

        # 基函数与导数（一次定义，多处复用）
        phi  = lambda t: (t - t**2)**2
        dphi = lambda t: 2*(t - t**2)*(1 - 2*t)
        psi  = lambda t: (2*t**3 - 3*t**2 + t)
        dpsi = lambda t: (6*t**2 - 6*t + 1)

        phx, phy, phz = phi(x),  phi(y),  phi(z)
        dphx, dphy, dphz = dphi(x), dphi(y), dphi(z)
        psx, psy, psz = psi(x),  psi(y),  psi(z)
        dpsx, dpsy, dpsz = dpsi(x), dpsi(y), dpsi(z)

        # 行 0：∂u_x/∂(x,y,z)
        dux_dx = 200*mu * dphx * psy * psz
        dux_dy = 200*mu * phx  * dpsy * psz
        dux_dz = 200*mu * phx  * psy  * dpsz

        # 行 1：∂u_y/∂(x,y,z)
        duy_dx = -100*mu * phy * dpsx * psz
        duy_dy = -100*mu * dphy * psx  * psz
        duy_dz = -100*mu * phy * psx  * dpsz

        # 行 2：∂u_z/∂(x,y,z)
        duz_dx = -100*mu * phz * dpsx * psy
        duz_dy = -100*mu * phz * psx  * dpsy
        duz_dz = -100*mu * dphz * psx  * psy

        grad_u = bm.stack([
                            bm.stack([dux_dx, dux_dy, dux_dz], axis=-1),
                            bm.stack([duy_dx, duy_dy, duy_dz], axis=-1),
                            bm.stack([duz_dx, duz_dy, duz_dz], axis=-1),
                        ], axis=-2)

        return grad_u


    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        val = self.disp_solution(points)
        
        return val
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y, z = points[..., 0], points[..., 1], points[..., 2]

        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps
        flag_y0 = bm.abs(y - domain[2]) < self._eps
        flag_y1 = bm.abs(y - domain[3]) < self._eps
        flag_z0 = bm.abs(z - domain[4]) < self._eps
        flag_z1 = bm.abs(z - domain[5]) < self._eps

        return flag_x0 | flag_x1 | flag_y0 | flag_y1 | flag_z0 | flag_z1

    @cartesian  
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:

        return self.is_dirichlet_boundary_dof_x(points)

    @cartesian  
    def is_dirichlet_boundary_dof_z(self, points: TensorLike) -> TensorLike:

        return self.is_dirichlet_boundary_dof_x(points)

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable, Callable]:
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y,
                self.is_dirichlet_boundary_dof_z)
