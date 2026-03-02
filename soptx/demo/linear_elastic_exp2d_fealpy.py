"""
三维线弹性问题的有限元求解程序
本模块基于 FEALPy 框架，
用于求解纯Dirichlet边界条件下的三维线弹性问题。
问题描述：
    u = 0    on ∂Ω (齐次Dirichlet边界条件)
    其中：
    - σ 为应力张量
    - ε = (∇u + ∇u^T)/2 为应变张量
    - 对于各向同性材料：σ = 2μ ε + λ tr(ε) I
    - λ, μ 为Lamé常数
模块组成：
    - PolySolPureDirLagrange3d：线性弹性问题的求解器类
    - test_linear_elasticity_with_fem：有限元求解测试函数
主要功能：
    1. 网格初始化（四面体网格）
    2. 有限元空间离散化（Lagrange有限元）
    3. 刚度矩阵和载荷向量组装
    4. Dirichlet边界条件处理
    5. 线性方程组求解（MUMPS直接求解器）
    6. 误差计算与收敛性分析
"""



from typing import List, Callable, Optional, Tuple

import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Callable
from fealpy.mesh import TetrahedronMesh, HexahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.linear_elasticity_integrator import LinearElasticityIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.solver import cg, spsolve
from fealpy.decorator import cartesian, variantmethod
from soptx.utils.show import showmultirate, show_error_table

class PolySolPureDirLagrange3d():
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
                plane_type: str = '3D',            
            ) -> None:
        self._domain = domain
        self._lam, self._mu = lam, mu

        self._eps = 1e-8
        self._plane_type = plane_type
        self._load_type = None
        self._boundary_type = 'dirichlet'


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
    def grad_disp_solution(self, points: TensorLike) -> TensorLike:
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
        domain = self._domain
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

def test_linear_elasticity_with_fem(p, pde):
    mesh = pde.init_mesh(nx=2, ny=2, nz=2)

    maxit = 4
    errorType = ['$|| \\boldsymbol{u}  - \\boldsymbol{u}_h ||_{L_2}$']
    errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
    NDof = bm.zeros(maxit, dtype=bm.int32)
    h = bm.zeros(maxit, dtype=bm.float64)
    for i in range(maxit):
        N = 2**(i+1)
        space = LagrangeFESpace(mesh, p=p, ctype='C')
        tensor_space = TensorFunctionSpace(space, shape=(-1, mesh.geo_dimension()))
        NDof[i] = tensor_space.number_of_global_dofs()

        linear_elastic_material = LinearElasticMaterial(
                                        name='E1nu03', 
                                        lame_lambda=pde.lam, 
                                        shear_modulus=pde.mu,
                                        hypo=pde._plane_type, 
                                        device=bm.get_device(mesh)
                                    )

        integrator_K = LinearElasticityIntegrator(
                            material=linear_elastic_material, 
                            q=tensor_space.p+3, 
                            method=None)
        bform = BilinearForm(tensor_space)
        bform.add_integrator(integrator_K)
        K = bform.assembly(format='csr')
        integrator_F = VectorSourceIntegrator(
                            source=pde.body_force, 
                            q=tensor_space.p+3
                        )
        lform = LinearForm(tensor_space)    
        lform.add_integrator(integrator_F)
        F = lform.assembly()

        dbc = DirichletBC(space=tensor_space, 
                        gd=pde.dirichlet_bc, 
                        threshold=None, 
                        method='interp')
        K, F = dbc.apply(A=K, f=F, uh=None, gd=pde.dirichlet_bc, check=True)

        uh = tensor_space.function()

        uh[:] = spsolve(K, F, solver='mumps')

        # L2 误差
        e0 = mesh.error(uh, pde.disp_solution)
        errorMatrix[0, i] = e0

        h[i] = 1 / N

        u_exact = tensor_space.interpolate(pde.disp_solution)
        errorMatrix[0, i] = bm.sqrt(bm.sum(bm.abs(uh[:] - u_exact)**2 * (1 / NDof[i])))

        if i < maxit-1:
            mesh.uniform_refine()

    print("errorMatrix:\n", errorType, "\n", errorMatrix)
    print("NDof:", NDof)
    print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
    show_error_table(h, errorType, errorMatrix)
    showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)

if __name__ == "__main__":
    p = 2
    pde = PolySolPureDirLagrange3d()

    test_linear_elasticity_with_fem(p, pde)

