"""基于已装配局部刚度的 NumPy Schur 算子与公共恢复缓存.

未缩聚块的 CSR 模式只由单元连接关系决定, 不按数值阈值删项.
显式与隐式 FA 共用内部 Cholesky 分解、稀疏耦合块及恢复算法.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix

from soptx.backend import backend_manager as bm

from .implicit import InteriorBlockSolver


@dataclass(frozen=True)
class LocalBlockPattern:
    """一个局部矩形块的共享 CSR 拓扑.

    Attributes
    ----------
    rows, cols : numpy.ndarray
        数值条目在完整局部刚度矩阵中的行列编号.
    indices, indptr : numpy.ndarray
        块内 CSR 列编号与行指针, 所有子结构共享.
    shape : tuple
        矩形块的维数.
    """

    rows: np.ndarray
    cols: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple

    def extract(self, matrix):
        """提取一个子结构的块值, 不复制完整局部矩阵.

        Parameters
        ----------
        matrix : numpy.ndarray
            已装配的完整局部刚度.

        Returns
        -------
        scipy.sparse.csr_matrix
            拓扑内的全部条目, 包括数值恰为零的条目.
        """
        return csr_matrix(
            (matrix[self.rows, self.cols], self.indices, self.indptr),
            shape=self.shape, copy=False,
        )


def build_local_block_patterns(assembly_map, internal, boundary):
    """复用局部装配映射建立耦合块与边界块的固定 CSR 模式.

    Parameters
    ----------
    assembly_map : LocalAssemblyMap
        同构子结构的单元散加映射.
    internal, boundary : numpy.ndarray
        完整且互斥的局部自由度划分.

    Returns
    -------
    tuple[LocalBlockPattern, LocalBlockPattern]
        K_ib 与 K_bb 的模式, 不依赖材料数值.
    """
    n = assembly_map.n_local_dofs
    internal = np.asarray(internal, dtype=np.int64)
    boundary = np.asarray(boundary, dtype=np.int64)
    if not np.array_equal(np.sort(np.concatenate((internal, boundary))), np.arange(n)):
        raise ValueError("内部和边界编号须完整且互斥.")
    slots = assembly_map.scatter_index
    if hasattr(slots, 'detach'):
        slots = slots.detach().cpu().numpy()
    slots = np.unique(np.asarray(slots, dtype=np.int64))
    topology = csr_matrix(
        (np.ones(len(slots), dtype=np.int8), (slots // n, slots % n)), shape=(n, n))
    patterns = []
    for row_ids, col_ids in ((internal, boundary), (boundary, boundary)):
        block = topology[row_ids][:, col_ids].tocsr()
        block.sort_indices()
        rows = np.repeat(np.arange(len(row_ids)), np.diff(block.indptr))
        patterns.append(LocalBlockPattern(
            row_ids[rows], col_ids[block.indices], block.indices, block.indptr, block.shape))
    return tuple(patterns)


class LocalSchurBlocks:
    """一轮材料状态下的局部刚度、内部分解和恢复数据.

    Parameters
    ----------
    interior : InteriorBlockSolver
        内部自由度映射和分解缓存.
    interface_dofs : numpy.ndarray
        全局接口自由度编号.
    local_dofs : numpy.ndarray
        各子结构边界在全局接口向量中的编号.
    patterns : tuple
        由 build_local_block_patterns 建立的 K_ib 与 K_bb 模式.
    keep_boundary : bool
        隐式路径保存稀疏 K_bb, 显式路径不保存.
    """

    def __init__(self, interior, interface_dofs, local_dofs, patterns, *, keep_boundary):
        self.interior = interior
        self.interface_dofs = np.asarray(interface_dofs, dtype=np.int64)
        self.local_dofs = np.asarray(local_dofs, dtype=np.int64)
        count = len(interior.indices)
        if self.local_dofs.shape[0] != count:
            raise ValueError("接口映射与内部块数量不符.")
        if (np.any(self.local_dofs < 0)
                or np.any(self.local_dofs >= len(self.interface_dofs))):
            raise ValueError("局部接口编号越界.")
        all_dofs = np.concatenate((self.interface_dofs, interior.indices.ravel()))
        if not np.array_equal(np.sort(all_dofs), np.arange(interior.full_size)):
            raise ValueError("内部与接口编号须完整且互斥.")
        self.patterns = patterns
        self.coupling = [None] * count
        self.boundary = [None] * count if keep_boundary else None
        self.diagonal = np.zeros(len(self.interface_dofs), dtype=np.float64)
        self.ready = np.zeros(count, dtype=bool)

    def prepare_batch(self, start, matrices, internal, boundary, *, explicit,
                      timings, column_chunk=32):
        """缓存公共数据, 按列求精确对角并可选形成局部 Schur 块.

        Parameters
        ----------
        start : int
            首个子结构编号.
        matrices : numpy.ndarray
            当前连续批次的完整局部刚度.
        internal, boundary : numpy.ndarray
            局部内部和边界编号.
        explicit : bool
            是否返回显式 Schur 块.
        timings : dict
            累加本轮互不重叠的准备阶段耗时.
        column_chunk : int, optional
            内部多右端求解的列批大小.

        Returns
        -------
        numpy.ndarray or None
            显式局部 Schur 批次, 隐式路径返回 None.
        """
        if column_chunk < 1 or (explicit and self.boundary is not None) or (not explicit and self.boundary is None):
            raise ValueError("列批大小或边界块缓存策略不合法.")
        end = start + len(matrices)
        if start < 0 or end > len(self.ready) or self.ready[start:end].any():
            raise ValueError("局部批次越界或重复准备.")
        started = perf_counter()
        self.interior.factor_batch(start, matrices[:, internal[:, None], internal])
        timings['internal_factorization'] += perf_counter() - started
        nb = len(boundary)
        schur = np.empty((len(matrices), nb, nb), dtype=np.float64) if explicit else None
        for offset, matrix in enumerate(matrices):
            index = start + offset
            started = perf_counter()
            coupling = self.patterns[0].extract(matrix)
            self.coupling[index] = coupling
            if self.boundary is not None:
                self.boundary[index] = self.patterns[1].extract(matrix)
            timings['local_blocks'] += perf_counter() - started
            started = perf_counter()
            local_diagonal = matrix[boundary, boundary].copy()
            if explicit:
                schur[offset] = matrix[np.ix_(boundary, boundary)]
            for first in range(0, nb, column_chunk):
                last = min(first + column_chunk, nb)
                rhs = coupling[:, first:last].toarray()
                solved = self.interior.solve_block(index, rhs)
                local_diagonal[first:last] -= np.sum(rhs * solved, axis=0)
                if explicit:
                    schur[offset, :, first:last] -= coupling.T @ solved
            np.add.at(self.diagonal, self.local_dofs[index], local_diagonal)
            key = 'schur_formation' if explicit else 'schur_diagonal'
            timings[key] += perf_counter() - started
        self.ready[start:end] = True
        return schur

    def recover(self, vector):
        """使用公共耦合块与因子恢复内部无载荷的全场位移.

        Parameters
        ----------
        vector : numpy.ndarray
            含固定分量的完整接口位移.

        Returns
        -------
        numpy.ndarray
            全场位移, 不重新装配或分解.
        """
        if not self.ready.all() or vector.shape != self.diagonal.shape:
            raise ValueError("局部块尚未完整准备或接口向量维数不符.")
        full = np.zeros(self.interior.full_size, dtype=np.float64)
        full[self.interface_dofs] = vector
        for index, coupling in enumerate(self.coupling):
            rhs = -(coupling @ vector[self.local_dofs[index]])
            full[self.interior.indices[index]] = self.interior.solve_block(index, rhs)
        return full

    def apply(self, vector):
        """由未缩聚块计算完整接口 Schur 作用.

        Parameters
        ----------
        vector : numpy.ndarray
            完整接口向量.

        Returns
        -------
        numpy.ndarray
            各子结构界面力的散加结果.
        """
        if self.boundary is None or not self.ready.all() or vector.shape != self.diagonal.shape:
            raise ValueError("须准备隐式边界块并提供完整接口向量.")
        result = np.zeros_like(vector, dtype=np.float64)
        for index, (coupling, boundary) in enumerate(zip(self.coupling, self.boundary)):
            ids = self.local_dofs[index]
            local = vector[ids]
            solved = self.interior.solve_block(index, coupling @ local)
            np.add.at(result, ids, boundary @ local - coupling.T @ solved)
        return result


class LocalSchurOperator:
    """把局部刚度 Schur 作用限制到自由接口, 兼容完整接口适配器.

    Parameters
    ----------
    blocks : LocalSchurBlocks or TorchLocalSchurBlocks
        完整准备的隐式局部块.
    free_positions : numpy.ndarray
        自由分量在完整接口向量中的位置.
    """

    def __init__(self, blocks, free_positions):
        self.blocks = blocks
        self.free_positions = bm.asarray(
            free_positions, dtype=bm.int64, device=bm.get_device(blocks.diagonal))
        self.shape = (len(self.free_positions),) * 2
        self.matvec_count = 0

    def _extend(self, vector):
        """将自由接口向量补零到完整接口.

        Parameters
        ----------
        vector : numpy.ndarray
            自由接口向量.

        Returns
        -------
        numpy.ndarray
            固定位置为零的完整接口向量.
        """
        if vector.shape != (self.shape[0],):
            raise ValueError("自由接口向量维数不符.")
        full = bm.zeros_like(self.blocks.diagonal)
        full[self.free_positions] = vector
        return full

    def __matmul__(self, vector):
        """计算自由接口 Schur 作用并累计调用次数.

        Parameters
        ----------
        vector : numpy.ndarray
            自由接口向量.

        Returns
        -------
        numpy.ndarray
            自由接口上的 Schur 作用结果.
        """
        self.matvec_count += 1
        return self.blocks.apply(self._extend(vector))[self.free_positions]

    def recover(self, vector):
        """补齐固定接口分量后调用公共恢复路径.

        Parameters
        ----------
        vector : numpy.ndarray
            自由接口位移.

        Returns
        -------
        numpy.ndarray
            满足内部平衡的全场位移.
        """
        return self.blocks.recover(self._extend(vector))