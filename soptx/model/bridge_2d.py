from typing import List, Callable, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.model.pde_base import PDEBase  

class Bridge2dSingleLoadData(PDEBase):
    '''
    Example 1 from Bruggi & Venini (2007) paper
    Single-point load bridge structure
    
    -∇·σ = b    in Ω
       u = 0    on ∂Ω_D (左右两端下半部分固支)
    where:
    - σ is the stress tensor
    - ε = (∇u + ∇u^T)/2 is the strain tensor
    
    几何参数:
        矩形域, 左右两端下半部分固支, 底部中点施加向下集中载荷
        考虑完整域（不利用对称性）
    
    Material parameters:
        E = 1, nu = 0.35 (compressible) or 0.5 (incompressible)

    For isotropic materials:
        σ = 2με + λtr(ε)I
    '''
    def __init__(self,
                domain: List[float] = [0, 80, 0, 40], 
                mesh_type: str = 'uniform_quad',
                T: float = -2.0,  # 向下的集中载荷
                E: float = 1.0, 
                nu: float = 0.35,  # 默认使用可压缩材料
                support_height_ratio: float = 0.5,  # 支撑高度比例（0.5表示下半部分）
                plane_type: str = 'plane_strain', # 'plane_stress' or 'plane_strain'
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)
        
        self._T = T
        self._E, self._nu = E, nu
        self._support_height_ratio = support_height_ratio  # 支撑高度比例
        self._plane_type = plane_type

        self._eps = 1e-12
        self._load_type = 'concentrated'
        self._boundary_type = 'dirichlet'


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
    
    @property
    def support_height_ratio(self) -> float:
        """获取支撑高度比例"""
        return self._support_height_ratio
    
    #######################################################################################################################
    # 变体方法
    #######################################################################################################################
    
    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        # 完整域需要更多单元（约论文中的两倍）
        nx = kwargs.get('nx', 128)  # 完整域的x方向单元数
        ny = kwargs.get('ny', 32)   # y方向单元数保持不变
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')

        mesh = QuadrangleMesh.from_box(box=self._domain, nx=nx, ny=ny,
                                    threshold=threshold, device=device)

        self._save_meshdata(mesh, 'uniform_quad', nx=nx, ny=ny)

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
        """
        定义体力（集中载荷）
        在底部中点施加向下的集中载荷 F = -2
        """
        domain = self.domain

        x, y = points[..., 0], points[..., 1]   

        mid_x = (domain[0] + domain[1]) / 2  
        coord = (
            (bm.abs(x - mid_x) < self._eps) & 
            (bm.abs(y - domain[2]) < self._eps)
        )
        
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val = bm.set_at(val, (coord, 1), self._T)
        
        return val
    
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        
        # 计算支撑的最大高度
        height = domain[3] - domain[2]
        y_max_support = domain[2] + height * self._support_height_ratio
        
        coord = ((bm.abs(x - domain[0]) < self._eps) | (bm.abs(x - domain[1]) < self._eps)) & (y <= y_max_support + self._eps)
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        
        # 计算支撑的最大高度
        height = domain[3] - domain[2]
        y_max_support = domain[2] + height * self._support_height_ratio
        
        coord = ((bm.abs(x - domain[0]) < self._eps) | (bm.abs(x - domain[1]) < self._eps)) & (y <= y_max_support + self._eps)
        
        return coord
    
    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:
        
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    
class BridgeDoubleLoad2d(PDEBase):
    '''
    两点载荷桥梁设计域的 PDE 模型

    设计域:
        - 全设计域: 80 mm x 40 mm

    边界条件:
        - 左右两端下半部分固支 (u)

    载荷条件:
        - 底部和顶部中点各施加向下集中载荷 P = -2
    
    材料参数:
        E = 1 [MPa], nu = 0.3
    '''
    def __init__(self,
                domain: List[float] = [0, 80, 0, 40], 
                mesh_type: str = 'uniform_quad',
                p1: float = -2.0,  
                p2: float = -2.0,  
                E: float = 1.0, 
                nu: float = 0.35,  
                support_height_ratio: float = 0.5,  # 支撑高度比例（0.5 表示下半部分）
                plane_type: str = 'plane_stress', # 'plane_stress' or 'plane_strain'
                enable_logging: bool = False, 
                logger_name: Optional[str] = None
            ) -> None:
        super().__init__(domain=domain, mesh_type=mesh_type, 
                enable_logging=enable_logging, logger_name=logger_name)

        self._p1 = p1
        self._p2 = p2
        self._E, self._nu = E, nu
        self._support_height_ratio = support_height_ratio 
        self._plane_type = plane_type

        self._eps = 1e-12        
        self._load_type = 'concentrated'
        self._boundary_type = 'mixed'

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
    def p1(self) -> float:
        """获取集中力 1"""
        return self._p1

    @property
    def p2(self) -> float:
        """获取集中力 2"""
        return self._p2

    @property
    def support_height_ratio(self) -> float:
        """获取支撑高度比例"""
        return self._support_height_ratio
    
    #######################################################################################################################
    # 变体方法
    #######################################################################################################################
    
    @variantmethod('uniform_quad')
    def init_mesh(self, **kwargs) -> QuadrangleMesh:
        nx = kwargs.get('nx', 128) 
        ny = kwargs.get('ny', 32)   
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
    
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        
        # 计算支撑的最大高度
        height = domain[3] - domain[2]
        y_max_support = domain[2] + height * self._support_height_ratio
        coord = ((bm.abs(x - domain[0]) < self._eps) | (bm.abs(x - domain[1]) < self._eps)) & (y <= y_max_support + self._eps)
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]
        
        # 计算支撑的最大高度
        height = domain[3] - domain[2]
        y_max_support = domain[2] + height * self._support_height_ratio
        coord = ((bm.abs(x - domain[0]) < self._eps) | (bm.abs(x - domain[1]) < self._eps)) & (y <= y_max_support + self._eps)
        
        return coord
    
    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:
        
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    
    def get_neumann_loads(self):
       """返回集中载荷函数, 用于位移有限元方法中的 Neumann 边界条件 (弱形式施加)"""
       if self._load_type == 'concentrated':
            
            @cartesian
            def concentrated_force(points: TensorLike) -> TensorLike:
                """
                定义集中载荷 (两点载荷), 点力恰好在节点上
                底部中点和顶部中点各施加方向向下, 相同大小的集中载荷 F = -2
                """
                domain = self.domain
                x, y = points[..., 0], points[..., 1]   

                mid_x = (domain[0] + domain[1]) / 2  
                
                coord_bottom = (
                    (bm.abs(x - mid_x) < self._eps) & 
                    (bm.abs(y - domain[2]) < self._eps)
                )
                coord_top = (
                    (bm.abs(x - mid_x) < self._eps) & 
                    (bm.abs(y - domain[3]) < self._eps)
                )
                
                kwargs = bm.context(points)
                val = bm.zeros(points.shape, **kwargs)
                
                val = bm.set_at(val, (coord_bottom, 1), self._p1) 
                val = bm.set_at(val, (coord_top, 1), self._p2)    
                
                return val
            
            return concentrated_force
       
       elif self._load_type == 'distributed':
           
           pass
       
       else:
                raise NotImplementedError(f"不支持的载荷类型: {self._load_type}")
       
    @cartesian
    def neumann_bc(self, points: TensorLike) -> TensorLike:
        """
        Neumann 边界条件: 表面力向量 t = (t_x, t_y) = (0, -2)
        在底部和顶部中点都施加相同大小的向下的力
        """
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val = bm.set_at(val, (..., 1), self._p1) 
        
        return val
    
    @cartesian
    def is_neumann_boundary_dof(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y = points[..., 0], points[..., 1]

        mid_x = (domain[0] + domain[1]) / 2  
        
        coord_bottom = (
            (bm.abs(x - mid_x) < self._eps) & 
            (bm.abs(y - domain[2]) < self._eps)
        )
        
        coord_top = (
            (bm.abs(x - mid_x) < self._eps) & 
            (bm.abs(y - domain[3]) < self._eps)
        )
        
        coord = (coord_bottom | coord_top)
        
        return coord
    
    def is_neumann_boundary(self) -> Callable:
        
        return self.is_neumann_boundary_dof
 
