from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.model.pde_base import PDEBase  

class DisplacementInverterUpper2d(PDEBase):
    """
    位移反向器上半设计域的 PDE 模型

    控制方程:
        -∇·σ = 0      in Ω
            u = u_D    on ∂F_D (Dirichlet BC)
          σ·n = t      on ∂Ω_N (Neumann BC)

    设计域:
        - 全设计域: 40 mm × 40 mm
        - 上半设计域: 40 mm × 20 mm (利用对称性简化)

    边界条件:
        - 下边界 (y = 0): 对称轴, 施加 u_y = 0
        - 左边界上顶点 (0, 20): 全固支约束, u_x = u_y = 0
        - 其余边界: 自由表面 (零牵引)

    载荷与弹簧:
        - 输入端 (0, 0): 水平向右载荷 F_in = 1 N, 输入弹簧 k_in = 0.1
        - 输出端 (40, 0): 输出弹簧 k_out = 0.1
        
    材料参数:
        E = 1 MPa, ν = 0.3
    """
    def __init__(self,
                 domain: List[float] = [0, 40, 0, 20],
                 mesh_type: str = 'uniform_quad',
                 f_in: float = 1.0,    # N, 输入力 (集中载荷, 水平向右)
                 f_out: float = -1.0,  # N, 伴随力 (集中载荷, 水平向左)
                 k_in: float = 0.1,    # N/mm, 输入弹簧刚度
                 k_out: float = 0.1,   # N/mm, 输出弹簧刚度
                 E: float = 1.0,       # MPa (N/mm^2)
                 nu: float = 0.3,
                 plane_type: str = 'plane_stress',
                 support_height: Optional[float] = None,  # 固支区域高度
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
        # 固支区域高度: 若未指定则默认为设计域高度的 1/20 (对应 ny=20 时的一个单元)
        self._support_height = support_height if support_height is not None \
                               else (domain[3] - domain[2]) / 20.0

        self._eps = 1e-12
        self._load_type = 'concentrated'
        self._boundary_type = 'mixed'

    ###########################################################################
    # 属性访问器
    ###########################################################################

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

    ###########################################################################
    # 网格生成
    ###########################################################################

    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 40)
        ny = kwargs.get('ny', 20)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                       threshold=threshold, device=device)
        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)
        return mesh

    @init_mesh.register('uniform_aligned_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 40)
        ny = kwargs.get('ny', 20)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                     threshold=threshold, device=device)
        self._save_meshdata(mesh, 'uniform_aligned_tri', nx=nx, ny=ny)
        return mesh

    @init_mesh.register('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 40)
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

    ###########################################################################
    # 体力
    ###########################################################################

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        """体力"""
        kwargs = bm.context(points)
        
        return bm.zeros(points.shape, **kwargs)


    ###########################################################################
    # 位移边界条件
    ###########################################################################
    
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        """Dirichlet 边界条件值 (均为零位移)"""
        kwargs = bm.context(points)
        
        return bm.zeros(points.shape, **kwargs)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """
        判断 x 方向的 Dirichlet 边界自由度

        左边界上顶点处施加固支约束 (u_x = 0)
        位置: (0, y) 其中 y ∈ [y_max - support_height, y_max]
        """
        domain = self._domain
        x, y = points[..., 0], points[..., 1]

        # 固支区域: 左边界上部 (靠近顶点)
        y_min_support = domain[3] - self._support_height

        is_left = bm.abs(x - domain[0]) < self._eps
        is_above_support = (y >= y_min_support - self._eps)

        return is_left & is_above_support

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """
        判断 y 方向的 Dirichlet 边界自由度

        1. 左边界上顶点处施加固支约束 (u_y = 0)
        2. 下边界 (对称轴) 施加对称约束 (u_y = 0)
        """
        domain = self._domain
        x, y = points[..., 0], points[..., 1]

        # 固支区域: 左边界上部
        y_min_support = domain[3] - self._support_height

        is_left = bm.abs(x - domain[0]) < self._eps
        is_above_support = y >= y_min_support - self._eps
        is_top_left_segment = is_left & is_above_support

        # 对称轴: 下边界 y = 0
        is_bottom = bm.abs(y - domain[2]) < self._eps

        return is_top_left_segment | is_bottom

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:
        """返回 Dirichlet 边界判断函数"""

        return (self.is_dirichlet_boundary_dof_x,
                self.is_dirichlet_boundary_dof_y)

    ###########################################################################
    # 集中载荷边界条件
    ###########################################################################

    @cartesian
    def concentrate_load_bc(self, points: TensorLike) -> TensorLike:
        """
        集中载荷值

        输入端 (0, 0) 处施加水平向右的力 F_in
        """
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        # 力作用在 x 方向 (水平向右为正)
        val = bm.set_at(val, (..., 0), self._f_in)

        return val

    @cartesian
    def is_concentrate_load_boundary_dof(self, points: TensorLike) -> TensorLike:
        """
        判断集中载荷作用点

        输入端位置: 对称轴左端 (0, 0) - 左下角
        """
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        on_bottom_boundary = (bm.abs(y - domain[2]) < self._eps)  # y = 0
        on_left_boundary = (bm.abs(x - domain[0]) < self._eps)    # x = 0

        return on_bottom_boundary & on_left_boundary

    def is_concentrate_load_boundary(self) -> Callable:
        """返回集中载荷边界判断函数"""
        return self.is_concentrate_load_boundary_dof

    ###########################################################################
    # 伴随载荷边界条件 - 用于柔顺机构优化
    ###########################################################################

    @cartesian
    def adjoint_load_bc(self, points: TensorLike) -> TensorLike:
        """
        伴随载荷值

        输出端 (40, 0) 处施加水平向左的伴随力 F_out
        """
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        # 力作用在 x 方向 (水平向左为负)
        val = bm.set_at(val, (..., 0), self._f_out)

        return val

    @cartesian
    def is_adjoint_load_boundary_dof(self, points: TensorLike) -> TensorLike:
        """
        判断伴随载荷作用点

        输出端位置: 对称轴右端 (40, 0) - 右下角
        """
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        on_bottom_boundary = (bm.abs(y - domain[2]) < self._eps)  # y = 0
        on_right_boundary = (bm.abs(x - domain[1]) < self._eps)   # x = 40

        return on_bottom_boundary & on_right_boundary

    def is_adjoint_load_boundary(self) -> Callable:
        """返回伴随载荷边界判断函数"""
        return self.is_adjoint_load_boundary_dof

    ###########################################################################
    # 输出点边界条件 - 用于计算输出位移
    ###########################################################################

    @cartesian
    def is_dout_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """
        输出点的 x 方向边界自由度

        输出点位置: 对称轴右端 (40, 0) - 右下角
        """
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        coord = ((bm.abs(x - domain[1]) < self._eps) &
                 (bm.abs(y - domain[2]) < self._eps))

        return coord

    @cartesian
    def is_dout_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """
        输出点的 y 方向边界自由度

        本问题中只关心 x 方向输出, y 方向返回全 False
        """
        return bm.zeros(points.shape[:-1], dtype=bm.bool,
                        device=bm.get_device(points))

    def is_dout_boundary(self) -> Tuple[Callable, Callable]:
        """返回输出点边界判断函数"""
        return (self.is_dout_boundary_dof_x, self.is_dout_boundary_dof_y)

    ###########################################################################
    # 弹簧边界条件 - 输入弹簧与输出弹簧
    ###########################################################################

    @cartesian
    def is_spring_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """
        弹簧作用点的 x 方向边界自由度

        输入弹簧位置: 左下角
        输出弹簧位置: 右下角
        """
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        # 输入弹簧: 左下角
        coord_kin = ((bm.abs(x - domain[0]) < self._eps) &
                     (bm.abs(y - domain[2]) < self._eps))

        # 输出弹簧: 右下角
        coord_kout = ((bm.abs(x - domain[1]) < self._eps) &
                      (bm.abs(y - domain[2]) < self._eps))

        return coord_kin | coord_kout

    @cartesian
    def is_spring_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """
        弹簧作用点的 y 方向边界自由度

        本问题中弹簧只在 x 方向起作用, y 方向返回全 False
        """
        return bm.zeros(points.shape[:-1], dtype=bm.bool,
                        device=bm.get_device(points))

    def is_spring_boundary(self) -> Tuple[Callable, Callable]:
        """返回弹簧边界判断函数"""
        return (self.is_spring_boundary_dof_x, self.is_spring_boundary_dof_y)