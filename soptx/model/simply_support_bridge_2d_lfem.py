from typing import List, Optional, Tuple, Callable

from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh
from fealpy.backend import backend_manager as bm
from soptx.model.pde_base import PDEBase  

class SimplySupportedBridge2d(PDEBase):
    """
    二维简支桥梁的 PDE 模型

    设计域: 120 mm x 40 mm

    边界条件:
    - 左下角铰支座 (u_x = u_y = 0)
    - 右下角滑移支座 (u_y = 0)

    载荷条件:
    - 上边界施加竖直向下的均布边界牵引载荷 t = 1 N/mm
    """
    def __init__(self,
                domain: List[float] = [0.0, 120.0, 0.0, 40.0],
                mesh_type: str = 'uniform_quad',
                t: float = 1.0,   # N/mm
                E: float = 1.0,   # MPa
                nu: float = 0.3,
                plane_type: str = 'plane_stress',  # 'plane_stress' or 'plane_strain'
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:

        super().__init__(domain=domain, mesh_type=mesh_type,
                         enable_logging=enable_logging, logger_name=logger_name)

        self._t = t
        self._E, self._nu = E, nu
        self._plane_type = plane_type

        self._eps = 1e-8
        self._load_type = 'distributed'
        self._boundary_type = 'mixed'

    @property
    def E(self) -> float:
        return self._E

    @property
    def nu(self) -> float:
        return self._nu

    @property
    def t(self) -> float:
        return self._t

    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> 'QuadrangleMesh':
        nx = kwargs.get('nx', 120)
        ny = kwargs.get('ny', 60)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                       threshold=threshold, device=device)
        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)
        return mesh

    @init_mesh.register('uniform_aligned_tri')
    def init_mesh(self, **kwargs) -> 'TriangleMesh':
        nx = kwargs.get('nx', 120)
        ny = kwargs.get('ny', 60)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                     threshold=threshold, device=device)
        self._save_meshdata(mesh, 'uniform_aligned_tri', nx=nx, ny=ny)
        return mesh

    @init_mesh.register('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> 'TriangleMesh':
        nx = kwargs.get('nx', 120)
        ny = kwargs.get('ny', 60)
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
    def body_force(self, points: 'TensorLike') -> 'TensorLike':
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)

    @cartesian
    def dirichlet_bc(self, points: 'TensorLike') -> 'TensorLike':
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: 'TensorLike') -> 'TensorLike':
        domain = self._domain
        x, y = points[..., 0], points[..., 1]
        left_bottom = (bm.abs(x - domain[0]) < self._eps) & (bm.abs(y - domain[2]) < self._eps)
        right_bottom = (bm.abs(x - domain[1]) < self._eps) & (bm.abs(y - domain[2]) < self._eps)

        return left_bottom | right_bottom

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: 'TensorLike') -> 'TensorLike':
        domain = self._domain
        x, y = points[..., 0], points[..., 1]
        left_bottom = (bm.abs(x - domain[0]) < self._eps) & (bm.abs(y - domain[2]) < self._eps)
        right_bottom = (bm.abs(x - domain[1]) < self._eps) & (bm.abs(y - domain[2]) < self._eps)

        return left_bottom | right_bottom

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x,
                self.is_dirichlet_boundary_dof_y)
    
    @cartesian
    def neumann_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val = bm.set_at(val, (..., 1), -self._t) 

        return val
    
    @cartesian
    def is_neumann_boundary_dof(self, points: TensorLike) -> TensorLike:
        domain = self._domain
        x, y = points[..., 0], points[..., 1]
        on_top = bm.abs(y - domain[3]) < self._eps

        return on_top

    def is_neumann_boundary(self) -> Callable:
        
        return self.is_neumann_boundary_dof