from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, TriangleMesh
from fealpy.decorator import cartesian, variantmethod
from fealpy.typing import TensorLike, Callable

from soptx.model.pde_base import PDEBase

class BoxTriHuZhangData2d(PDEBase):
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
                domain: List[float] = [0, 1, 0, 1],
                mesh_type: str = 'uniform_tri', 
                lam: float = 1.0, mu: float = 0.5,                
                enable_logging: bool = False, 
                logger_name: Optional[str] = None 
            ) -> None:
        
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)
        
        self._lam, self._mu = lam, mu
        self._eps = 1e-12
        self._plane_type = 'plane_strain'
        self._force_type = 'distribution'
        self._boundary_type = 'dirichlet'

        self._log_info(f"Initialized BoxTriLagrangeData2d with domain={self._domain}, "
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
    

    def stress_matrix_coefficient(self) -> tuple[float, float]:
        """
        材料为均匀各向同性线弹性体时, 计算应力块矩阵的系数 lambda0 和 lambda1
        
        Returns
        -------
        lambda0: 1/(2μ)
        lambda1: λ/(2μ(dλ+2μ)), 其中 d=2 为空间维数
        """
        d = 2 
        lambda0 = 1.0 / (2 * self._mu)
        lambda1 = self._lam / (2 * self._mu * (d * self._lam + 2 * self._mu))
        
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
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        exp_xy = bm.exp(x - y)

        u1 = exp_xy * x * (1 - x) * y * (1 - y)
        u2 = bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

        val = bm.stack([u1, u2], axis=-1)

        return val

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
        self._force_type = 'distribution'
        self._boundary_type = 'dirichlet'

        self._log_info(f"Initialized BoxTriLagrangeData2d with domain={self._domain}, "
                       f"mesh_type='{mesh_type}', E={E}, nu={nu}, "
                       f"plane_type='{self._plane_type}', "
                       f"force_type='{self._force_type}', "
                       f"boundary_type='{self._boundary_type}'")


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

    @variantmethod('uniform_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_tri', nx=nx, ny=ny, threshold=threshold, device=device)

        return mesh

    @init_mesh.register('polygon_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        device = kwargs.get('device', 'cpu')
        vertices = bm.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], 
                            dtype=bm.float64, device=device)
        mesh = TriangleMesh.from_polygon_gmsh(vertices=vertices, h=0.07)

        return mesh
    
    @init_mesh.register('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 10)
        ny = kwargs.get('ny', 10)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny, threshold=threshold, device=device)

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