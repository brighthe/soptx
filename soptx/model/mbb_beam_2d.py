from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.model.pde_base import PDEBase  

class HalfMBBBeam2dData(PDEBase):
    '''
    -∇·σ = b    in Ω
       u = 0    on ∂Ω (homogeneous Dirichlet)
    where:
    - σ is the stress tensor
    - ε = (∇u + ∇u^T)/2 is the strain tensor
    
    Material parameters:
        E = 1, nu = 0.3

    For isotropic materials:
        σ = 2με + λtr(ε)I
    '''
    def __init__(self,
                domain: List[float] = [0, 60, 0, 20],
                mesh_type: str = 'uniform_quad',
                T: float = -1.0, # 负值代表方向向下
                E: float = 1.0, nu: float = 0.3,
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)
        
        self._T = T
        self._E, self._nu = E, nu
        self._eps = 1e-12
        
        self._plane_type = 'plane_stress'
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
    def T(self) -> float:
        """获取集中力"""
        return self._T
    
    #######################################################################################################################
    # 变体方法
    #######################################################################################################################
    
    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 20)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)

        return mesh
    
    @init_mesh.register('uniform_aligned_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 60)
        ny = kwargs.get('ny', 20)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = TriangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_aligned_tri', nx=nx, ny=ny)

        return mesh
    
    @init_mesh.register('uniform_crisscross_tri')
    def init_mesh(self, **kwargs) -> TriangleMesh:
        nx = kwargs.get('nx', 60)
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
    
    def get_neumann_loads(self):
       
       if self._load_type == 'concentrated':
            
            @cartesian
            def concentrated_force(points: TensorLike) -> TensorLike:
                domain = self._domain
                x, y = points[..., 0], points[..., 1]   

                coord = (
                    (bm.abs(x - domain[0]) < self._eps) & 
                    (bm.abs(y - domain[3]) < self._eps)
                )
                kwargs = bm.context(points)
                val = bm.zeros(points.shape, **kwargs)

                val = bm.set_at(val, (coord, 1), self._T)
        
                return val
            
            return concentrated_force
       
       elif self._load_type == 'distributed':
           
           pass
       
       else:
                raise NotImplementedError(f"不支持的载荷类型: {self._load_type}")
    
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self._domain

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self._eps
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain

        x = points[..., 0]
        y = points[..., 1]

        coord = ((bm.abs(x - domain[1]) < self._eps) &
                 (bm.abs(y - domain[2]) < self._eps))
        
        return coord
    
    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)


class MBBBeam2dData(PDEBase):
    """
    全区域的 MBB 梁
    
    -∇·σ = b    in Ω
       u = 0    on ∂Ω (homogeneous Dirichlet)
    where:
    - σ is the stress tensor
    - ε = (∇u + ∇u^T)/2 is the strain tensor
    
    Material parameters:
        E = 1, nu = 0.3

    For isotropic materials:
        σ = 2με + λtr(ε)I
        Young's modulus and Poisson's ratio.
    """

    def __init__(self,
                domain: List[float] = [0.0, 60.0, 0.0, 10.0],
                mesh_type: str = 'uniform_quad',
                p: float = -2.0,
                E: float = 1.0,
                nu: float = 0.3,
                enable_logging: bool = False,
                logger_name: Optional[str] = None
            ) -> None:
        
        super().__init__(domain=domain, mesh_type=mesh_type,
                         enable_logging=enable_logging, logger_name=logger_name)

        self._p = p
        self._E, self._nu = E, nu

        self._eps = 1e-12

        self._plane_type = 'plane_stress'
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
    def p(self) -> float:
        """获取集中力"""
        return self._p


    #######################################################################################################################
    # 变体方法
    #######################################################################################################################

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


    #######################################################################################################################
    # 核心方法
    #######################################################################################################################

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    def get_neumann_loads(self):
        """返回集中载荷函数, 用于位移有限元方法中的 Neumann 边界条件 (弱形式施加)"""
        if self._load_type == 'concentrated':
            @cartesian
            def concentrated_force(points: TensorLike) -> TensorLike:
                domain = self._domain
                xm = 0.5 * (domain[0] + domain[1])
                x, y = points[..., 0], points[..., 1]   

                coord = (bm.abs(x - xm) < self._eps) & (bm.abs(y - domain[3]) < self._eps)

                kwargs = bm.context(points)
                val = bm.zeros(points.shape, **kwargs)
                val = bm.set_at(val, (coord, 1), self._p)
        
                return val
            
            return concentrated_force

        elif self._load_type == 'distributed':
            pass
        
        else:
            raise NotImplementedError(f"不支持的载荷类型: {self._load_type}")

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self._domain

        x, y = points[..., 0], points[..., 1]

        temp = (bm.abs(x - domain[0]) < self._eps) & (bm.abs(y - domain[2]) < self._eps)

        return temp

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self._domain

        x, y = points[..., 0], points[..., 1]

        left = (bm.abs(x - domain[0]) < self._eps) & (bm.abs(y - domain[2]) < self._eps)
        right = (bm.abs(x - domain[1]) < self._eps) & (bm.abs(y - domain[2]) < self._eps)

        temp = left | right
    
        return temp

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x,
                self.is_dirichlet_boundary_dof_y)
