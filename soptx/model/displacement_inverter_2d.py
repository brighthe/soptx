from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.model.pde_base import PDEBase  

class DisplacementInverter2d(PDEBase):
    '''
    位移反向器柔顺机械的 PDE 模型定义

    -∇·σ = 0      in Ω
      u = u_D    on ∂Ω_D (Dirichlet BC)
    σ·n = t      on ∂Ω_N (Neumann BC)

    其中:
    - σ 是应力张量
    - ε = (∇u + ∇u^T)/2 是应变张量
    - u_D 在底边为 0
    - t 是在输入点施加的集中力 f_in (发现向右)
    
    材料参数:
        E = 1, nu = 0.3

    对于各向同性材料:
        σ = 2με + λtr(ε)I
    '''
    def __init__(self,
                domain: List[float] = [0, 40, 0, 20],
                mesh_type: str = 'uniform_quad',
                f_in: float = 1.0,   # N, 输入力 (集中载荷)
                f_out: float = -1.0, # N, 伴随力 (集中载荷)
                k_in: float = 1.0,   # N/m, 输入弹簧刚度
                k_out: float = 1.0,  # N/m, 输出弹簧刚度
                E: float = 1.0,      # Pa (N/m^2)
                nu: float = 0.3,
                plane_type: str = 'plane_stress', # 'plane_stress' or 'plane_strain'
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
                ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                         enable_logging=enable_logging, logger_name=logger_name)

        self._f_in = f_in
        self._f_out = f_out
        self._k_in = k_in
        self._k_out = k_out
        self._E, self._nu = E, nu
        self._plane_type = plane_type


        self._eps = 1e-12
        self._load_type = 'concentrated'
        self._boundary_type = 'mixed'


    #######################################################################################################################
    # 属性访问器
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
    def f_in(self) -> float:
        """获取输入力"""
        return self._f_in

    @property
    def f_out(self) -> float:
        """获取伴随力"""
        return self._f_out

    @property
    def k_in(self) -> float:
        """获取输入弹簧刚度"""
        return self._k_in

    @property
    def k_out(self) -> float:
        """获取输出弹簧刚度"""
        return self._k_out
        

    #######################################################################################################################
    # 变体方法
    #######################################################################################################################
    
    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 100)
        ny = kwargs.get('ny', 50)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                       threshold=threshold, device=device)
        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)
        return mesh
    
    @init_mesh.register('uniform_aligned_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 100)
        ny = kwargs.get('ny', 50)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                     threshold=threshold, device=device)
        self._save_meshdata(mesh, 'uniform_aligned_tri', nx=nx, ny=ny)
        return mesh
    
    @init_mesh.register('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 80)
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
    
    #######################################################################################################################
    # 核心方法
    #######################################################################################################################

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
        """
        判断 x 方向的 Dirichlet 边界自由度
        底边完全固支 -> u_x = 0, u_y = 0
        """
        domain = self._domain
        y = points[..., 1]
        coord = bm.abs(y - domain[2]) < self._eps

        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """
        判断 y 方向的 Dirichlet 边界自由度
        底边完全固支 -> u_x = 0, u_y = 0
        """
        domain = self._domain
        y = points[..., 1]
        coord = bm.abs(y - domain[2]) < self._eps

        return coord
    
    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:
          
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    
    @cartesian
    def neumann_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        # 力作用在 x 方向
        val = bm.set_at(val, (..., 0), self._f_in) 
        
        return val
    
    @cartesian
    def is_neumann_boundary_dof(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        on_top_boundary = (bm.abs(y - domain[3]) < self._eps)
        on_left_boundary = (bm.abs(x - domain[0]) < self._eps)

        return on_top_boundary & on_left_boundary

    def is_neumann_boundary(self) -> Callable:
        
        return self.is_neumann_boundary_dof
    
    @cartesian
    def adjoint_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        # 力作用在 x 方向
        val = bm.set_at(val, (..., 0), self._f_out) 
        
        return val
    
    @cartesian
    def is_adjoint_boundary_dof(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        on_top_boundary = (bm.abs(y - domain[3]) < self._eps)
        on_right_boundary = (bm.abs(x - domain[1]) < self._eps)

        return on_top_boundary & on_right_boundary
    
    def is_adjoint_boundary(self) -> Callable:
        
        return self.is_adjoint_boundary_dof 
    
    @cartesian
    def is_dout_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """输出点的边界自由度的 x 分量"""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        # 输出点在右上角
        coord = (bm.abs(x - domain[1]) < self._eps) & (bm.abs(y - domain[3]) < self._eps)

        return coord

    @cartesian
    def is_dout_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """对于任何输入点都返回 False, 用于明确指定某个方向上没有边界条件"""
        # points.shape[:-1] 获取输入点的数量, 创建一个形状匹配的全 False 布尔数组
        return bm.zeros(points.shape[:-1], dtype=bm.bool, device=bm.get_device(points))
    
    @cartesian
    def is_dout_boundary(self) -> Callable:

        return (self.is_dout_boundary_dof_x, self.is_dout_boundary_dof_y)

    @cartesian
    def is_spring_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """"""
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        coord_kin = (bm.abs(x - domain[0]) < self._eps) & (bm.abs(y - domain[3]) < self._eps)
        coord_kout = (bm.abs(x - domain[1]) < self._eps) & (bm.abs(y - domain[3]) < self._eps)

        return coord_kin | coord_kout
    
    @cartesian
    def is_spring_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """对于任何输入点都返回 False, 用于明确指定某个方向上没有边界条件"""
        # points.shape[:-1] 获取输入点的数量, 创建一个形状匹配的全 False 布尔数组
        return bm.zeros(points.shape[:-1], dtype=bm.bool, device=bm.get_device(points))
    
    def is_spring_boundary(self) -> Callable:

        return (self.is_spring_boundary_dof_x, self.is_spring_boundary_dof_y)
