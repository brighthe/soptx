from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, TriangleMesh
from fealpy.decorator import cartesian, variantmethod
from fealpy.typing import TensorLike, Callable

from soptx.model.pde_base import PDEBase

class TriMixHomoDirNHomoNeu2d(PDEBase):
    r"""
    二维线弹性 (平面应变) —— 解析解 + 混合边界条件 (上/下 齐次 Dirichlet, 左/右 非齐次 Neumann) 模型

    材料参数：
        λ = 1.0, μ = 0.5 

    解析位移 (三角函数):
        u(x, y) = [ sin(πx/2) · sin(πy),
                   -2 sin(πx) · (sin(πy/2)-y) ]^T

    体力密度: 
        b(x, y) = [ π^2( sin(πx/2) sin(πy) + (3/2) cos(πy) cos(πy/2) ) - 2π(λ+μ) cos(πx),
                   -(3/4)π^2 cos(πx/2) cos(πy) - 2π^2 sin(πx) sin(πy/2) + 2μ π^2 y sin(πx) ]^T

    Dirichlet:
        下边界  y = 0:  u(x, 0) = [0, 0]^T
        上边界  y = 1:  u(x, 1) = [0, 0]^T
    Neumann:
        左边界  x = 0,  n = (-1, 0), t = (0, -1): g(0, y) = [ -πsin(πy),
                                                            πsin(πy/2) - 2μπy ]^T
        右边界  x = 1,  n = ( 1, 0), t = (0,  1): g(1, y) = [ 0,
                                                            (π/2)cos(πy) + πsin(πy/2) - 2μπy ]^T
    """
    def __init__(self, 
            domain: List[float] = [0, 1, 0, 1],
            mesh_type: str = 'uniform_quad', 
            lam: float = 1.0, 
            mu: float = 0.5,
            plane_type: str = 'plane_strain', # 'plane_stress' or 'plane_strain'                     
            enable_logging: bool = False, 
            logger_name: Optional[str] = None) -> None:
        
        super().__init__(domain=domain, mesh_type=mesh_type, 
                        enable_logging=enable_logging, logger_name=logger_name)
        self._lam, self._mu = lam, mu
        self._eps = 1e-8

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
        u2 = -2.0 * bm.sin(pi * x) * (bm.sin(0.5 * pi * y) - y)

        return bm.stack([u1, u2], axis=-1)

    @cartesian
    def grad_disp_solution(self, points: TensorLike) -> TensorLike:
        """解析解梯度 ∇u"""
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
        """σ = 2μ ε + λ tr(ε) I, 返回 (σ_xx, σ_xy, σ_yy)."""
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        lam, mu = self.lam, self.mu

        # gradients
        du1_dx = 0.5 * pi * bm.cos(0.5 * pi * x) * bm.sin(pi * y)
        du1_dy =        pi * bm.sin(0.5 * pi * x) * bm.cos(pi * y)
        du2_dx =  2.0 * pi * bm.cos(pi * x) * (y - bm.sin(0.5 * pi * y))
        du2_dy = -      pi * bm.sin(pi * x) * bm.cos(0.5 * pi * y) + 2.0 * bm.sin(pi * x)

        # strains
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
        """根据平衡方程: -∇·σ = b, 所以 ∇·σ = -b"""
        return -self.body_force(points)
    
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
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        flag_y0 = bm.abs(y - domain[2]) < self._eps  # y=0
        flag_y1 = bm.abs(y - domain[3]) < self._eps  # y=1

        return flag_y0 | flag_y1

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        flag_y0 = bm.abs(y - domain[2]) < self._eps  # y=0
        flag_y1 = bm.abs(y - domain[3]) < self._eps  # y=1

        return flag_y0 | flag_y1

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:
        
        return (self.is_dirichlet_boundary_dof_x,
                self.is_dirichlet_boundary_dof_y)
    
    @cartesian
    def neumann_bc(self, points: TensorLike) -> TensorLike:
        """
        σ·n = t on Γ_N
        Left  x=0, n=(-1,0): t(0,y) = [ -π sin(πy),
                                        π sin(πy/2) - 2μ π y ]
        Right x=1, n=(1,0):  t(1,y) = [ 0,
                                        (π/2) cos(πy) + π sin(πy/2) - 2μ π y ]
        """
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        mu = self.mu

        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)

        # left boundary x=0
        flag_left = bm.abs(x - domain[0]) < self._eps
        t_x_left = -pi * bm.sin(pi * y)
        t_y_left =  pi * bm.sin(0.5 * pi * y) - 2.0 * mu * pi * y
        val = bm.set_at(val, (flag_left, 0), t_x_left[flag_left])
        val = bm.set_at(val, (flag_left, 1), t_y_left[flag_left])

        # right boundary x=1
        flag_right = bm.abs(x - domain[1]) < self._eps
        t_x_right = bm.zeros_like(x) 
        t_y_right = 0.5 * pi * bm.cos(pi * y) + pi * bm.sin(0.5 * pi * y) - 2.0 * mu * pi * y
        val = bm.set_at(val, (flag_right, 0), t_x_right[flag_right])
        val = bm.set_at(val, (flag_right, 1), t_y_right[flag_right])

        return val

    @cartesian
    def neumann_bc_normal(self, points: TensorLike) -> TensorLike:
        """Unit outward normal vector on Γ_N."""
        domain = self.domain
        x = points[..., 0]

        kwargs = bm.context(points)
        normals = bm.zeros((points.shape[0], 2), **kwargs)

        # x = 0, n = (-1, 0)
        flag_left = bm.abs(x - domain[0]) < self._eps
        normals = bm.set_at(normals, (flag_left, 0), -1.0)

        # x = 1, n = (1, 0)
        flag_right = bm.abs(x - domain[1]) < self._eps
        normals = bm.set_at(normals, (flag_right, 0), 1.0)

        return normals
    
    @cartesian
    def neumann_bc_tangent(self, points: TensorLike) -> TensorLike:
        """Unit tangent vector on Γ_N."""
        domain = self.domain
        x = points[..., 0]

        kwargs = bm.context(points)
        tangents = bm.zeros((points.shape[0], 2), **kwargs)

        # x = 0, t = (0, -1)
        flag_left = bm.abs(x - domain[0]) < self._eps
        tangents = bm.set_at(tangents, (flag_left, 1), -1.0)

        # x = 1, t = (0, 1)
        flag_right = bm.abs(x - domain[1]) < self._eps
        tangents = bm.set_at(tangents, (flag_right, 1), 1.0)

        return tangents

    @cartesian
    def is_neumann_boundary_dof_xx(self, points: TensorLike) -> TensorLike:
        """Mark σ_xx-DOFs on x=0 or x=1 (vertical edges) as Neumann-traction DOFs."""
        domain = self.domain
        x = points[..., 0]
        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps

        return flag_x0 | flag_x1

    @cartesian
    def is_neumann_boundary_dof_xy(self, points: TensorLike) -> TensorLike:
        """Mark σ_xy-DOFs on x=0 or x=1 as Neumann-traction DOFs."""
        domain = self.domain
        x = points[..., 0]
        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps

        return flag_x0 | flag_x1

    @cartesian
    def is_neumann_boundary_dof_yy(self, points: TensorLike) -> TensorLike:
        """No σ_yy traction constraint on vertical edges ⇒ always False."""
        
        return bm.zeros(points.shape[:-1], dtype=bm.bool, device=points.device)
    
    def is_neumann_boundary(self) -> Tuple[Callable, Callable, Callable]:

        return (self.is_neumann_boundary_dof_xx,
                self.is_neumann_boundary_dof_xy,
                self.is_neumann_boundary_dof_yy)
    
    @cartesian
    def is_neumann_boundary_dof_nn(self, points: TensorLike) -> TensorLike:
        """Mark σ_nn-DOFs on x=0 or x=1 (vertical edges) as Neumann-traction DOFs."""
        domain = self.domain
        x = points[..., 0]
        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps

        return flag_x0 | flag_x1

    @cartesian
    def is_neumann_boundary_dof_nt(self, points: TensorLike) -> TensorLike:
        """Mark σ_nt-DOFs on x=0 or x=1 as Neumann-traction DOFs."""
        domain = self.domain
        x = points[..., 0]
        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps

        return flag_x0 | flag_x1

    def is_neumann_boundary_edge(self) -> Tuple[Callable, Callable]:

        return (self.is_neumann_boundary_dof_nn,
                self.is_neumann_boundary_dof_nt)
    
    @cartesian
    def sigma_nn_bc(self, points: TensorLike) -> TensorLike:
        """ σ_nn = g · n """
        g_values = self.neumann_bc(points)
        n_values = self.neumann_bc_normal(points)

        return bm.einsum('...i, ...i -> ...', g_values, n_values)

    @cartesian
    def sigma_nt_bc(self, points: TensorLike) -> TensorLike:
        """ σ_nt = g · t """
        g_values = self.neumann_bc(points)
        t_values = self.neumann_bc_tangent(points)

        return bm.einsum('...i, ...i -> ...', g_values, t_values)