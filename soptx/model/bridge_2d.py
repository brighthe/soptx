from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.pde.pde_base import PDEBase  

class Bridge2dData(PDEBase):
    '''
    Example 1 from Bruggi & Venini (2007) paper
    Single-point load bridge structure
    
    -∇·σ = b    in Ω
       u = 0    on ∂Ω_D (左右两端固支)
    where:
    - σ is the stress tensor
    - ε = (∇u + ∇u^T)/2 is the strain tensor
    
    几何参数:
        矩形域，两端固支，底部中点施加向下集中载荷
        由于对称性，只计算半域
    
    Material parameters:
        E = 1, nu = 0.35 (compressible) or 0.5 (incompressible)

    For isotropic materials:
        σ = 2με + λtr(ε)I
    '''
    def __init__(self,
                domain: List[float] = [0, 4, 0, 2],  # 半域
                mesh_type: str = 'uniform_quad',
                T: float = -2.0,  # 向下的集中载荷
                E: float = 1.0, 
                nu: float = 0.35,  # 默认使用可压缩材料
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)
        
        self._T = T
        self._E, self._nu = E, nu

        self._eps = 1e-12
        self._plane_type = 'plane_strain'  # 平面应变
        self._force_type = 'concentrated'
        self._boundary_type = 'dirichlet'

        self._log_info(f"Initialized Bridge2dData (Example 1) with domain={self._domain}, "
                f"mesh_type='{mesh_type}', force={T}, E={E}, nu={nu}, "
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
    
    @property
    def T(self) -> float:
        """获取集中力"""
        return self._T
    
    #######################################################################################################################
    # 变体方法
    #######################################################################################################################
    
    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        # 论文中使用约4100个JM单元（对于半域）
        nx = kwargs.get('nx', 64)  # 可调整以匹配论文的网格密度
        ny = kwargs.get('ny', 32)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)

        return mesh
    
    @init_mesh.register('uniform_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 64)
        ny = kwargs.get('ny', 32)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)
        
        self._save_meshdata(mesh, 'uniform_tri', nx=nx, ny=ny)

        return mesh


    #######################################################################################################################
    # 核心方法
    #######################################################################################################################

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        """
        定义体力（集中载荷）
        在底部中点施加向下的集中载荷 F = -2
        由于对称性，载荷点在 x=0（对称轴）, y=0（底部）
        """
        domain = self.domain

        x, y = points[..., 0], points[..., 1]   

        # 底部中点（对称轴上）：x = 0, y = 0
        coord = (
            (bm.abs(x - domain[0]) < self._eps) & 
            (bm.abs(y - domain[2]) < self._eps)
        )
        
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        # 在y方向施加向下的力
        val = bm.set_at(val, (coord, 1), self._T)
        
        return val
    
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """Dirichlet边界条件：位移为0"""
        kwargs = bm.context(points)
        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """
        x方向位移约束：
        1. 右端边界（x = 4）完全固定
        2. 对称边界条件：左边界（x = 0）在对称轴上，x方向位移为0
        """
        domain = self.domain
        x = points[..., 0]

        # 左边界（对称轴，x = 0）和右边界（固支，x = 4）
        coord = (bm.abs(x - domain[0]) < self._eps) | (bm.abs(x - domain[1]) < self._eps)
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """
        y方向位移约束：
        只有右端边界（x = 4）的y方向固定
        对称轴上y方向可以自由移动
        """
        domain = self.domain
        x = points[..., 0]

        # 只有右边界（x = 4）的y方向固定
        coord = bm.abs(x - domain[1]) < self._eps
        
        return coord
    
    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:
        """返回Dirichlet边界条件函数"""
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)