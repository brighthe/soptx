from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from fealpy.decorator import cartesian

from typing import Tuple, Callable, List

class Cantilever2dData1:
    '''
    模型来源论文: Efficient topology optimization in MATLAB using 88 lines of code
    '''
    def __init__(self, 
                xmin: float, xmax: float, 
                ymin: float, ymax: float,
                T: float = -1):
        """
        位移边界条件: 梁的左边界固定
        载荷: 梁的右边界的下点施加垂直向下的力 T = -1
        """
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.T = T 
        self.eps = 1e-12

    def domain(self) -> list:
        
        box = [self.xmin, self.xmax, self.ymin, self.ymax]

        return box
    
    @cartesian
    def force(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        coord = (
            (bm.abs(x - domain[1]) < self.eps) & 
            (bm.abs(y - domain[2]) < self.eps)
        )
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val[coord, 1] = self.T

        return val
    
    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)
        # 这里仍然是固定左边界的位移
        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord    
    
    def threshold(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    
class Cantilever2dMultiLoadData1:
    '''
    模型来源论文: Efficient topology optimization in MATLAB using 88 lines of code
    '''
    def __init__(self, 
                xmin: float, xmax: float, 
                ymin: float, ymax: float,
                T: List[float] = [-1, 1]):
        """
        位移边界条件: 梁的左边界固定
        载荷: 
        - T[0]: 梁的右下角施加的垂直力 (默认为 -1, 向下)
        - T[1]: 梁的右上角施加的垂直力 (默认为 1, 向上)
        """
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.T = T 
        self.eps = 1e-12

    def domain(self) -> list:
        
        box = [self.xmin, self.xmax, self.ymin, self.ymax]

        return box
    
    @cartesian
    def force(self, points: TensorLike) -> TensorLike:
        """返回所有载荷工况的力向量
        
        Parameters:
        -----------
        points: 空间点坐标 (npoints, ndim)
        
        Returns:
        --------
        力向量 (nloads, npoints, ndim)，第一维表示不同的载荷工况
        """
        domain = self.domain()
        x = points[..., 0]
        y = points[..., 1]
        
        # 确定关键点位置
        bottom_right_corner = (
            (bm.abs(x - domain[1]) < self.eps) & 
            (bm.abs(y - domain[2]) < self.eps)
        )
        
        top_right_corner = (
            (bm.abs(x - domain[1]) < self.eps) & 
            (bm.abs(y - domain[3]) < self.eps)
        )
        
        # 创建批处理的多载荷力向量
        kwargs = bm.context(points)
        npoints = points.shape[0]
        ndim = points.shape[1]
        nloads = len(self.T)
        
        # 初始化力向量张量：(nloads, npoints, ndim)
        val = bm.zeros((nloads, npoints, ndim), **kwargs)
        
        # 设置第一个载荷工况（右下角）
        mask = bm.zeros((npoints,), **kwargs)
        mask = bm.where(bottom_right_corner, 1.0, 0.0)
        val[0, :, 1] = mask * self.T[0]
        
        # 设置第二个载荷工况（右上角）
        mask = bm.zeros((npoints,), **kwargs)
        mask = bm.where(top_right_corner, 1.0, 0.0)
        val[1, :, 1] = mask * self.T[1]
        
        return val
    
    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord    
    
    def threshold(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)

class Cantilever2dData2:
    '''
    新模型，适应区域大小和载荷改变：
    载荷施加在右边界的中点位置，大小为 T = 2000
    区域尺寸: L 和 H
    '''
    def __init__(self, 
                 xmin: float = 0, xmax: float = 3.0, 
                 ymin: float = 0, ymax: float = 1.0, 
                 T: float = 2000):
        """
        位移边界条件：梁的左边界固定
        载荷：梁的右边界的中点施加垂直向下的力 T = 2000
        0 ------- 3 ------- 6 
        |    0    |    2    |
        1 ------- 4 ------- 7 
        |    1    |    3    |
        2 ------- 5 ------- 8 
        """
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.T = T  # 载荷大小
        self.eps = 1e-12

    def domain(self) -> list:
        box = [self.xmin, self.xmax, self.ymin, self.ymax]

        return box
    
    @cartesian
    def force(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        # 载荷施加在右边界的中点处
        coord = (
            (bm.abs(x - domain[1]) < self.eps) & 
            (bm.abs(y - (domain[2] + domain[3]) / 2) < self.eps)  # 位于右边界中点
        )
        
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val[coord, 1] = -self.T  # 施加单位力 T
        
        return val
    
    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)
        # 这里仍然是固定左边界的位移
        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps  # 左边界的 x 坐标
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps  # 左边界的 x 坐标
        
        return coord
    
    def threshold(self) -> Tuple[Callable, Callable]:
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)