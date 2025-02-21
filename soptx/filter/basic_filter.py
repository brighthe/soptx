from abc import ABC, abstractmethod
from typing import Optional, Literal, Tuple
from math import ceil, sqrt

from fealpy.backend import backend_manager as bm
from fealpy.mesh import StructuredMesh, UniformMesh2d, UniformMesh3d
from fealpy.typing import TensorLike
from fealpy.sparse import COOTensor, CSRTensor

from soptx.utils import timer

class BasicFilter(ABC):
    """基础滤波器抽象基类"""
    
    def __init__(self, mesh: StructuredMesh, rmin: float):
        """
        Parameters
        - mesh : 均匀网格
        - rmin : 滤波半径 (物理距离)
        """
        if rmin <= 0:
            raise ValueError("Filter radius must be positive")
        if not isinstance(mesh, StructuredMesh):
            raise TypeError("Mesh must be StructuredMesh")
        
        self.rmin = rmin
        self.mesh = mesh
        
        self._H, self._Hs = self._compute_filter_matrix()
        self._cell_measure = self.mesh.entity_measure('cell')
        self._normalize_factor = self._H.matmul(self._cell_measure)

    @property
    def H(self) -> COOTensor:
        """滤波矩阵"""
        return self._H

    @property
    def Hs(self) -> TensorLike:
        """滤波矩阵行和向量"""
        return self._Hs
    
    def _compute_filter_matrix(self) -> Tuple[COOTensor, TensorLike]:
        """计算线性衰减的滤波器内核"""
        if isinstance(self.mesh, UniformMesh2d):
            return self._compute_filter_2d(
                                self.mesh.nx, self.mesh.ny,
                                self.mesh.h[0], self.mesh.h[1], 
                                self.rmin
                            )
        elif isinstance(self.mesh, UniformMesh3d):
            return self._compute_filter_3d(
                                self.mesh.nx, self.mesh.ny, self.mesh.nz,
                                self.mesh.h[0], self.mesh.h[1], self.mesh.h[2],
                                self.rmin
                            )
        
    def _compute_filter_2d(self, 
                nx: int, ny: int, 
                hx: float, hy: float,
                rmin: float
            ) -> Tuple[COOTensor, TensorLike]:
        """计算 2D 滤波矩阵
        TODO 能否使用 kd-tree 优化邻域搜索 (query_ipoint 函数)
        """
        min_h = min(hx, hy)
        max_cells = ceil(rmin/min_h)
        nfilter = int(nx * ny * ((2 * (max_cells - 1) + 1) ** 2))
        
        iH = bm.zeros(nfilter, dtype=bm.int32)
        jH = bm.zeros(nfilter, dtype=bm.int32)
        sH = bm.zeros(nfilter, dtype=bm.float64)
        cc = 0

        for i in range(nx):
            for j in range(ny):
                # 单元的编号顺序: y->x 
                row = i * ny + j
                # 根据物理距离计算搜索范围
                kk1 = int(max(i - (ceil(rmin/hx) - 1), 0))
                kk2 = int(min(i + ceil(rmin/hx), nx))
                ll1 = int(max(j - (ceil(rmin/hy) - 1), 0))
                ll2 = int(min(j + ceil(rmin/hy), ny))
                
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        # 单元的编号顺序: y->x 
                        col = k * ny + l
                        # 计算实际物理距离
                        physical_dist = sqrt((i - k)**2 * hx**2 + (j - l)**2 * hy**2)
                        fac = rmin - physical_dist
                        if fac > 0:
                            iH[cc] = row
                            jH[cc] = col
                            sH[cc] = max(0.0, fac)
                            cc += 1

        H = COOTensor(
                indices=bm.astype(bm.stack((iH[:cc], jH[:cc]), axis=0), bm.int32),
                values=sH[:cc],
                spshape=(nx * ny, nx * ny)
            )
        H = H.tocsr()
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64)
        
        return H, Hs
    
    def _compute_filter_3d(self, 
                    nx: int, ny: int, nz: int, 
                    hx: float, hy: float, hz: float,
                    rmin: float
                ) -> Tuple[COOTensor, TensorLike]:
        """计算 3D 滤波矩阵"""
        min_h = min(hx, hy, hz)
        max_cells = ceil(rmin/min_h)
        nfilter = nx * ny * nz * ((2 * (max_cells - 1) + 1) ** 3)
        
        iH = bm.zeros(nfilter, dtype=bm.int32)
        jH = bm.zeros(nfilter, dtype=bm.int32)
        sH = bm.zeros(nfilter, dtype=bm.float64)
        cc = 0

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 单元的编号顺序: z -> y -> x
                    row = k + j * nz + i * ny * nz
                    ii1 = int(max(i - (ceil(rmin/hx) - 1), 0))
                    ii2 = int(min(i + ceil(rmin/hx), nx))
                    jj1 = int(max(j - (ceil(rmin/hy) - 1), 0))
                    jj2 = int(min(j + ceil(rmin/hy), ny))
                    kk1 = int(max(k - (ceil(rmin/hz) - 1), 0))
                    kk2 = int(min(k + ceil(rmin/hz), nz))
                    
                    for ii in range(ii1, ii2):
                        for jj in range(jj1, jj2):
                            for kk in range(kk1, kk2):
                                # 单元的编号顺序: z -> y -> x
                                col = kk + jj * nz + ii * ny * nz
                                # 计算实际物理距离 
                                physical_dist = sqrt(
                                                    (i - ii)**2 * hx**2 + 
                                                    (j - jj)**2 * hy**2 + 
                                                    (k - kk)**2 * hz**2
                                                )
                                fac = rmin - physical_dist
                                if fac > 0:
                                    iH[cc] = row
                                    jH[cc] = col
                                    sH[cc] = max(0.0, fac)
                                    cc += 1

        H = COOTensor(
            indices=bm.astype(bm.stack((iH[:cc], jH[:cc]), axis=0), bm.int32),
            values=sH[:cc],
            spshape=(nx * ny * nz, nx * ny * nz)
        )
        H = H.tocsr()
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64)

        return H, Hs
    
    @abstractmethod
    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> None:
        """
        获取初始的物理密度场
        
        Parameters
        - x : 初始设计变量
        - xPhys : 初始物理变量 (输出)
        """
        pass

    abstractmethod
    def filter_variables(self, x: TensorLike, xPhys: TensorLike) -> None:
        """
        对设计变量进行滤波得到物理变量
        
        Parameters
        - x : 原始设计变量
        - xPhys : 过滤后的物理变量 (输出)
        """
        pass

    @abstractmethod
    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> None:
        """
        过滤目标函数的灵敏度
        
        Parameters
        - xPhys : 过滤后的物理变量
        - dobj : 原始的目标函数灵敏度
        """
        pass

    @abstractmethod
    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> None:
        """
        过滤约束函数的灵敏度
        
        Parameters
        - xPhys : 过滤后的物理变量
        - dcos : 原始的约束函数灵敏度
        """
        pass

class SensitivityBasicFilter(BasicFilter):
    """灵敏度滤波器"""
    def __init__(self, mesh: StructuredMesh, rmin: float):
        super().__init__(mesh, rmin)
    
    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> None:
        """灵敏度滤波器的初始物理密度等于设计变量"""
        xPhys[:] = x
    
    def filter_variables(self, x: TensorLike, xPhys: TensorLike) -> None:
        xPhys[:] = x

    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> None:
        # 计算密度加权的目标函数灵敏度
        weighted_dobj = bm.einsum('c, c -> c', xPhys, dobj)
        # 应用滤波矩阵
        filtered_dobj = self._H.matmul(weighted_dobj)
        # 计算修正因子
        correction_factor = self._Hs * bm.maximum(bm.tensor(0.001, dtype=bm.float64), xPhys)
        # 过滤后的目标函数灵敏度
        dobj[:] = filtered_dobj / correction_factor

    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> None:
        return

class DensityBasicFilter(BasicFilter):
    """密度滤波器"""
    def __init__(self, mesh: StructuredMesh, rmin: float):
        super().__init__(mesh, rmin)
    
    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> None:
        """密度滤波器的初始物理密度等于设计变量"""
        xPhys[:] = x

    def filter_variables(self, x: TensorLike, xPhys: TensorLike) -> None:
        '''
        TODO 需要进一步优化 filtered_x = self._H.matmul(weigthed_x) 的计算效率,
        因为 OC 中二分法求解 Lagrange 乘子的时候会多次调用这个函数
        '''
        # 计算加权密度
        weigthed_x = x * self._cell_measure
        # 应用滤波矩阵
        filtered_x = self._H.matmul(weigthed_x)
        # 返回标准化后的密度
        xPhys[:] = filtered_x / self._normalize_factor

    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> None:
        # 计算单元测度加权的目标函数灵敏度
        weighted_dobj = self._cell_measure * dobj
        # 应用滤波矩阵
        dobj[:] = self._H.matmul(weighted_dobj / self._normalize_factor)

    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> None:
        # 计算单元测度加权的约束函数灵敏度
        weighted_dcons = self._cell_measure * dcons
        # 应用滤波矩阵
        dcons[:] = self._H.matmul(weighted_dcons / self._normalize_factor)

class HeavisideProjectionBasicFilter(BasicFilter):
    """Heaviside 投影滤波器"""
    def __init__(self, mesh: StructuredMesh, rmin: float, 
                beta: float = 1.0, max_beta: float = 512, continuation_iter: int = 50):
        """
        Parameters
        - mesh : 均匀网格
        - rmin : 滤波半径 (物理距离)
        - beta : Heaviside 投影参数
        """
        super().__init__(mesh, rmin)
        if beta <= 0:
            raise ValueError("Heaviside beta must be positive")
            
        self.beta = beta
        self.max_beta = max_beta
        self.continuation_iter = continuation_iter
        self._xTilde = None  # 存储中间密度场

        self._beta_iter = 0  # 用于追踪 continuation 的内部状态

    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> None:
        """Heaviside 投影滤波器的初始物理密度需要投影"""
        self._xTilde = x 
        xPhys[:] = (1 - bm.exp(-self.beta * self._xTilde) + 
                   self._xTilde * bm.exp(-self.beta))
    
    def filter_variables(self, x: TensorLike, xPhys: TensorLike) -> None:
        weighted_x = self._cell_measure * x
        filtered_x = self._H.matmul(weighted_x)
        self._xTilde = filtered_x / self._normalize_factor

        xPhys[:] = (1 - bm.exp(-self.beta * self._xTilde) + 
                        self._xTilde * bm.exp(-self.beta))

    def filter_objective_sensitivities(self, 
                                    xPhys: TensorLike, dobj: TensorLike) -> None:        
        # 计算 Heaviside 投影的导数
        dx = self.beta * bm.exp(-self.beta * self._xTilde) + bm.exp(-self.beta)
        # 修改灵敏度并应用密度滤波
        weighted_dobj = dobj * dx * self._cell_measure
        dobj[:] = self._H.matmul(weighted_dobj / self._normalize_factor)

    def filter_constraint_sensitivities(self, 
                                    xPhys: TensorLike, dcons: TensorLike) -> None:        
        # 计算 Heaviside 投影的导数
        dx = self.beta * bm.exp(-self.beta * self._xTilde) + bm.exp(-self.beta)
        # 修改灵敏度并应用密度滤波
        weighted_dcons = dcons * dx * self._cell_measure
        dcons[:] = self._H.matmul(weighted_dcons / self._normalize_factor)

    def continuation_step(self, change: float) -> Tuple[float, bool]:
        """
        执行一步 beta continuation
        
        Parameters
        - change : 当前的收敛变化量
        
        Returns
        - new_change : 更新后的收敛变化量
        - continued : 是否执行了 continuation

        """
        self._beta_iter += 1
        
        if (self.beta < self.max_beta and 
                (self._beta_iter >= self.continuation_iter or change <= 0.01)):
            # 增加 beta 值
            self.beta *= 2
            # 重置计数器
            self._beta_iter = 0
            print(f"Beta increased to {self.beta}")
            return 1.0, True
        
        # 如果没有执行 continuation，返回原始的 change 值和 False
        return change, False
            