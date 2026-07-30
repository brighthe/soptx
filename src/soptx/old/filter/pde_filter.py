from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import StructuredMesh
from fealpy.typing import TensorLike
from fealpy.sparse import COOTensor

class PDEBasedFilter(ABC):
    """基于 PDE 的滤波器抽象基类"""
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
        
        self.mesh = mesh
        # 计算 Helmholtz 方程的参数
        self.Rmin = rmin / 2 / np.sqrt(3)
        # 构建 PDE 滤波矩阵
        self._KF, self._TF, self._L = self._build_filter_matrix()

    def _build_filter_matrix(self):
        """构建 PDE 滤波矩阵"""
        nx, ny = self.mesh.nx, self.mesh.ny
        
        # 构建单元刚度矩阵
        KEF = self.Rmin**2 * bm.array([[4, -1, -2, -1],
                                       [-1, 4, -1, -2],
                                       [-2, -1, 4, -1],
                                       [-1, -2, -1, 4]]) / 6 \
            + bm.array([[4, 2, 1, 2],
                        [2, 4, 2, 1],
                        [1, 2, 4, 2],
                        [2, 1, 2, 4]]) / 36

        cell2node1 = self.mesh.cell_to_node()
        # 构建节点编号
        nn = (nx + 1) * (ny + 1)  # 总节点数
        nodenrs = bm.arange(nn).reshape(ny + 1, nx + 1)
        
        # 构建单元到节点的映射
        NC = nx * ny  # 总单元数
        cell2node = bm.zeros((NC, 4), dtype=bm.int32)
        
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                nodes = [
                    nodenrs[j, i],
                    nodenrs[j, i+1],
                    nodenrs[j+1, i+1],
                    nodenrs[j+1, i]
                ]
                cell2node[idx] = nodes

        # 构建 KF 矩阵
        NK = 16 * NC  # KF 矩阵的非零元素个数
        iKF = bm.zeros(NK, dtype=bm.int32)
        jKF = bm.zeros(NK, dtype=bm.int32)
        sKF = bm.zeros(NK, dtype=bm.float64)
        
        for i in range(NC):
            nodes = cell2node[i]
            for j in range(4):
                for k in range(4):
                    idx = i * 16 + j * 4 + k
                    iKF[idx] = nodes[j]
                    jKF[idx] = nodes[k]
                    sKF[idx] = KEF[j, k]

        KF = COOTensor(
                indices=bm.stack((iKF, jKF), axis=0),
                values=sKF,
                spshape=(nn, nn)
            )

        # 构建投影矩阵 TF
        NT = 4 * NC  # TF 矩阵的非零元素个数
        iTF = bm.zeros(NT, dtype=bm.int32)
        jTF = bm.zeros(NT, dtype=bm.int32)
        sTF = bm.ones(NT, dtype=bm.float64) / 4

        for i in range(NC):
            nodes = cell2node[i]
            for j in range(4):
                idx = i * 4 + j
                iTF[idx] = nodes[j]
                jTF[idx] = i

        TF = COOTensor(
            indices=bm.stack((iTF, jTF), axis=0),
            values=sTF,
            spshape=(nn, NC)
        )

        # Cholesky 分解
        from scipy.sparse.linalg import splu
        L = splu(KF.tocsr())
        
        return KF, TF, L

    def solve_filter_system(self, b: TensorLike) -> TensorLike:
        """求解滤波系统"""
        # 使用 LU 分解求解系统
        return self._L.solve(b)
        
    def filter_field(self, x: TensorLike) -> TensorLike:
        """滤波一个场"""
        # 投影到节点
        nodal_x = self._TF.matmul(x)
        # 求解系统
        filtered_nodal = self.solve_filter_system(nodal_x)
        # 投影回单元
        return self._TF.transpose().matmul(filtered_nodal)

    @abstractmethod
    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> None:
        """
        获取初始的物理密度场
        
        Parameters
        - x : 初始设计变量
        - xPhys : 初始物理变量 (输出)
        """
        pass

    @abstractmethod
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
        - dcons : 原始的约束函数灵敏度
        """
        pass

class SensitivityPDEBasedFilter(PDEBasedFilter):
    """基于 PDE 的灵敏度滤波器"""
    def __init__(self, mesh: StructuredMesh, rmin: float):
        super().__init__(mesh, rmin)
    
    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> None:
        """灵敏度滤波器的初始物理密度等于设计变量"""
        xPhys[:] = x
    
    def filter_variables(self, x: TensorLike, xPhys: TensorLike) -> None:
        """灵敏度滤波器不对设计变量进行滤波"""
        xPhys[:] = x

    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> None:
        """滤波目标函数的灵敏度"""
        weighted_dobj = xPhys * dobj
        filtered_dobj = self.filter_field(weighted_dobj)
        dobj[:] = filtered_dobj / bm.maximum(bm.tensor(1e-3), xPhys)

    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> None:
        """灵敏度滤波器不滤波约束灵敏度"""
        return

class DensityPDEBasedFilter(PDEBasedFilter):
    """基于 PDE 的密度滤波器"""
    def __init__(self, mesh: StructuredMesh, rmin: float):
        super().__init__(mesh, rmin)
    
    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> None:
        """计算初始的物理密度"""
        xPhys[:] = self.filter_field(x)

    def filter_variables(self, x: TensorLike, xPhys: TensorLike) -> None:
        """对设计变量进行滤波得到物理密度"""
        xPhys[:] = self.filter_field(x)

    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> None:
        """滤波目标函数的灵敏度"""
        dobj[:] = self.filter_field(dobj)

    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> None:
        """滤波约束函数的灵敏度"""
        dcons[:] = self.filter_field(dcons)