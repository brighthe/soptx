"""基于细网格算子与内部块分解的精确 Schur 作用, 不存储接口 CSR.

细网格 EA 核心与内部求解器用于 NumPy/CPU; 完整接口适配器同时支持 PyTorch. 内部分解在一次固定密度
分析内缓存, 改变密度后必须重建. 所有内部求解均为 Cholesky 直接求解.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, cho_solve

from soptx.backend import backend_manager as bm


class InteriorBlockSolver:
    """缓存相互独立的内部刚度块分解.

    Parameters
    ----------
    global_dofs : numpy.ndarray
        (n_blocks, n_internal) 的全局内部自由度编号, 各块不得重叠.
    full_size : int
        全场自由度数.
    """

    def __init__(self, global_dofs, full_size):
        self.indices = np.asarray(global_dofs, dtype=np.int64)
        self.full_size = int(full_size)
        if self.indices.ndim != 2 or min(self.indices.shape) < 1:
            raise ValueError("内部自由度映射须为非空二维数组.")
        if self.indices.min() < 0 or self.indices.max() >= self.full_size:
            raise ValueError("内部自由度编号越界.")
        if np.unique(self.indices).size != self.indices.size:
            raise ValueError("内部自由度在不同块间不能重叠.")
        count, ni = self.indices.shape
        self.factors = np.empty((count, ni, ni), dtype=np.float64)
        self.ready = np.zeros(count, dtype=bool)

    def factor_batch(self, start, matrices):
        """分解并保存一批内部刚度, 密度改变后须创建新实例.

        Parameters
        ----------
        start : int
            首块编号.
        matrices : numpy.ndarray
            对应的对称正定内部刚度块.

        Returns
        -------
        None
            分解写入缓存.
        """
        matrices = np.asarray(matrices)
        end = start + len(matrices)
        if start < 0 or end > len(self.ready) or matrices.shape[1:] != self.factors.shape[1:]:
            raise ValueError("内部刚度批次形状或编号错误.")
        for offset, matrix in enumerate(matrices):
            if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0.0):
                # 对近零非对角项采用整体量级复核, 仅容许装配舍入误差.
                scale = max(float(np.max(np.abs(matrix))), np.finfo(float).tiny)
                if np.max(np.abs(matrix - matrix.T)) > 1e-12 * scale:
                    raise ValueError("内部刚度不对称.")
            self.factors[start + offset] = cholesky(matrix, lower=True, check_finite=True)
        self.ready[start:end] = True

    def solve_block(self, index, rhs):
        """复用一个内部块的 Cholesky 因子求解向量或多列右端.

        Parameters
        ----------
        index : int
            子结构编号.
        rhs : numpy.ndarray
            第一维为内部自由度的向量或矩阵.

        Returns
        -------
        numpy.ndarray
            内部方程的解, 不重新分解.
        """
        if index < 0 or index >= len(self.ready) or not self.ready[index]:
            raise ValueError("内部块尚未分解或编号越界.")
        rhs = np.asarray(rhs)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.factors.shape[1]:
            raise ValueError("内部右端形状错误.")
        return cho_solve((self.factors[index], True), rhs, check_finite=False)

    def schur_diagonal(self, start, matrices, i_dofs, b_dofs, column_chunk=32):
        """利用已缓存分解按列计算局部 Schur 对角, 不形成完整缩聚块.

        Parameters
        ----------
        start : int
            首个子结构编号.
        matrices : numpy.ndarray
            对应批次的完整局部刚度.
        i_dofs, b_dofs : numpy.ndarray
            内部与边界局部编号.
        column_chunk : int, optional
            一次内部求解的边界列数.

        Returns
        -------
        numpy.ndarray
            每个子结构的边界 Schur 对角.
        """
        if column_chunk < 1 or not self.ready[start:start + len(matrices)].all():
            raise ValueError("须提供正的列批大小并先完成内部分解.")
        diagonal = np.diagonal(matrices, axis1=1, axis2=2)[:, b_dofs].copy()
        for offset, matrix in enumerate(matrices):
            for first in range(0, len(b_dofs), column_chunk):
                last = min(first + column_chunk, len(b_dofs))
                coupling = matrix[np.ix_(i_dofs, b_dofs[first:last])]
                solved = self.solve_block(start + offset, coupling)
                diagonal[offset, first:last] -= np.sum(coupling * solved, axis=0)
        return diagonal

    def solve_into(self, rhs, result):
        """把各内部块解写入全场向量的内部位置.

        Parameters
        ----------
        rhs : numpy.ndarray
            全场右端向量.
        result : numpy.ndarray
            全场输出向量, 接口位置保持不变.

        Returns
        -------
        None
            原位写入 result.
        """
        if not self.ready.all():
            raise RuntimeError("内部刚度尚未全部分解.")
        if rhs.shape != (self.full_size,) or result.shape != rhs.shape:
            raise ValueError("内部求解向量形状错误.")
        for index, indices in enumerate(self.indices):
            result[indices] = self.solve_block(index, rhs[indices])


class ImplicitSchurOperator:
    """由细网格刚度与内部消元构造自由接口 Schur 算子.

    Parameters
    ----------
    full_operator : object
        已对齐次支承作对称消元的细网格算子.
    interior : InteriorBlockSolver
        与 full_operator 使用相同密度、材料和编号的内部块分解.
    free_interface : numpy.ndarray
        自由接口全局编号, 不包含固定自由度.
    fixed_dofs : numpy.ndarray
        全场固定自由度编号. 三类编号须恰好划分全场.
    """

    def __init__(self, full_operator, interior, free_interface, fixed_dofs):
        self.full_operator = full_operator
        self.interior = interior
        self.free_interface = np.asarray(free_interface, dtype=np.int64)
        self.fixed = np.asarray(fixed_dofs, dtype=np.int64)
        n = interior.full_size
        if full_operator.shape != (n, n):
            raise ValueError("全场算子与内部块映射维数不符.")
        all_indices = np.concatenate((interior.indices.ravel(), self.free_interface, self.fixed))
        if (len(all_indices) != n or all_indices.min() != 0 or all_indices.max() != n - 1
                or np.unique(all_indices).size != n):
            raise ValueError("内部、自由接口和支承编号未构成全场的互斥完整划分.")
        self.shape = (len(self.free_interface),) * 2
        self.matvec_count = 0

    def recover(self, interface_displacement):
        """从自由接口位移恢复内部无载荷的全场位移.

        Parameters
        ----------
        interface_displacement : numpy.ndarray
            自由接口位移.

        Returns
        -------
        numpy.ndarray
            内部平衡的全场位移, 固定自由度为零.
        """
        if interface_displacement.shape != (self.shape[0],):
            raise ValueError("接口向量形状错误.")
        full = np.zeros(self.interior.full_size, dtype=np.float64)
        full[self.free_interface] = interface_displacement
        rhs = -(self.full_operator @ full)
        self.interior.solve_into(rhs, full)
        return full

    def __matmul__(self, vector):
        """按内部平衡恢复后提取界面力."""
        self.matvec_count += 1
        return (self.full_operator @ self.recover(vector))[self.free_interface]


class FullInterfaceSchurOperator:
    """将自由接口 Schur 算子适配为含固定分量的完整接口向量协议.

    Parameters
    ----------
    core : ImplicitSchurOperator or LocalSchurOperator
        提供自由接口作用与 recover 的算子.
    free_positions : numpy.ndarray
        自由分量在完整接口向量中的位置, 顺序与 core 一致.
    diagonal : numpy.ndarray
        完整接口 Schur 对角. 固定位置设为 1.
    """

    def __init__(self, core, free_positions, diagonal):
        self.core = core
        device = bm.get_device(diagonal)
        self.free_positions = bm.asarray(free_positions, dtype=bm.int64, device=device)
        self._diagonal = bm.asarray(diagonal, dtype=bm.float64, device=device)
        self.shape = (len(self._diagonal),) * 2
        if core.shape[0] != len(self.free_positions):
            raise ValueError("自由接口维数不一致.")

    def diagonal(self):
        """返回接口对角, 供 Jacobi 预条件使用."""
        return self._diagonal

    def __matmul__(self, vector):
        """自由位置作用 Schur 算子, 固定位置作用恒等算子."""
        if vector.shape != (self.shape[0],):
            raise ValueError("完整接口向量形状错误.")
        result = bm.copy(vector)
        result[self.free_positions] = self.core @ vector[self.free_positions]
        return result

    def recover(self, vector):
        """提取自由接口分量并复用内部分解恢复全场位移."""
        return self.core.recover(vector[self.free_positions])


class RestrictedPreconditioner:
    """通过零延拓与转置提取把完整细网格预条件子限制到自由接口.

    Parameters
    ----------
    full_preconditioner : object
        已绑定全场算子的固定线性对称正定预条件子, 例如对称 MG V 循环.
    free_interface : numpy.ndarray
        自由接口的全局自由度编号.
    full_size : int
        全场自由度数.

    Notes
    -----
    作用为 R B R.T. 对自由全场刚度 K 有 R K^{-1} R.T = S^{-1};
    将 K^{-1} 替换为 B 得到接口预条件子, 不改变 Schur 方程本身.
    """

    def __init__(self, full_preconditioner, free_interface, full_size):
        self.full_preconditioner = full_preconditioner
        self.indices = np.asarray(free_interface, dtype=np.int64)
        self.full_size = int(full_size)
        self.shape = (len(self.indices),) * 2

    def __matmul__(self, residual):
        """将接口残差补零, 执行固定全场预条件作用, 提取接口修正."""
        if residual.shape != (self.shape[0],):
            raise ValueError("接口残差形状错误.")
        full_rhs = np.zeros(self.full_size, dtype=np.float64)
        full_rhs[self.indices] = residual
        return (self.full_preconditioner @ full_rhs)[self.indices]
