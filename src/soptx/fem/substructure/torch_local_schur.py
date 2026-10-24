"""PyTorch CPU/CUDA 的批量局部刚度 Schur 与公共恢复缓存.

稀疏块共享由连接关系生成的拓扑, 数值按子结构连续存储在目标设备.
作用按批次 gather/scatter-add, 不构造稠密的全量边界块或全局细网格矩阵.
"""

from time import perf_counter

import numpy as np
import torch


class TorchInteriorBlockSolver:
    """在目标设备缓存批量内部 Cholesky 因子.

    Parameters
    ----------
    global_dofs : torch.Tensor
        各子结构内部自由度的全局编号.
    full_size : int
        全场自由度数.
    """

    def __init__(self, global_dofs, full_size):
        self.indices = global_dofs.to(dtype=torch.int64)
        self.full_size = int(full_size)
        if self.indices.ndim != 2 or min(self.indices.shape) < 1:
            raise ValueError("内部自由度映射须为非空二维数组.")
        if bool((self.indices < 0).any()) or bool((self.indices >= full_size).any()):
            raise ValueError("内部编号越界.")
        if torch.unique(self.indices).numel() != self.indices.numel():
            raise ValueError("内部自由度不得重叠.")
        count, ni = self.indices.shape
        self.factors = torch.empty((count, ni, ni), dtype=torch.float64, device=self.indices.device)
        self.ready = np.zeros(count, dtype=bool)

    def factor_batch(self, start, matrices):
        """分解一批内部刚度, 保留因子在输入设备.

        Parameters
        ----------
        start : int
            起始子结构编号.
        matrices : torch.Tensor
            形状为 (batch, ni, ni) 的内部刚度.

        Returns
        -------
        None
            原位更新分解缓存.
        """
        end = start + len(matrices)
        if (start < 0 or end > len(self.ready)
                or matrices.shape[1:] != self.factors.shape[1:]
                or matrices.device != self.factors.device or matrices.dtype != torch.float64):
            raise ValueError("内部矩阵批次的形状、设备或类型不符.")
        if not bool(torch.isfinite(matrices).all()):
            raise ValueError("内部矩阵含非有限值.")
        error = (matrices - matrices.transpose(-1, -2)).abs().amax(dim=(-1, -2))
        scale = matrices.abs().amax(dim=(-1, -2)).clamp_min(torch.finfo(matrices.dtype).tiny)
        if bool((error > 1e-12 * scale).any()):
            raise ValueError("内部矩阵不对称.")
        self.factors[start:end] = torch.linalg.cholesky(matrices)
        self.ready[start:end] = True

    def solve_batch(self, start, rhs):
        """使用缓存因子求解批量向量或多右端系统.

        Parameters
        ----------
        start : int
            起始子结构编号.
        rhs : torch.Tensor
            形状为 (batch, ni) 或 (batch, ni, nrhs).

        Returns
        -------
        torch.Tensor
            与 rhs 同形状且同设备的解.
        """
        end = start + len(rhs)
        if (start < 0 or end > len(self.ready) or not self.ready[start:end].all()
                or rhs.ndim not in (2, 3) or rhs.shape[1] != self.factors.shape[1]
                or rhs.device != self.factors.device or rhs.dtype != self.factors.dtype):
            raise ValueError("内部右端形状、设备或分解状态不符.")
        vector = rhs.ndim == 2
        result = torch.cholesky_solve(rhs.unsqueeze(-1) if vector else rhs, self.factors[start:end])
        return result.squeeze(-1) if vector else result


class TorchBlockPattern:
    """单个拓扑模式的设备端索引, 供所有子结构共享.

    Parameters
    ----------
    pattern : LocalBlockPattern
        主机端固定 CSR 模式.
    device : torch.device
        数值所在设备.
    """

    def __init__(self, pattern, device):
        self.rows = torch.as_tensor(pattern.rows, dtype=torch.int64, device=device)
        self.cols = torch.as_tensor(pattern.cols, dtype=torch.int64, device=device)
        self.block_rows = torch.as_tensor(
            np.repeat(np.arange(pattern.shape[0]), np.diff(pattern.indptr)),
            dtype=torch.int64, device=device)
        self.block_cols = torch.as_tensor(pattern.indices, dtype=torch.int64, device=device)
        self.shape = pattern.shape
        self.nnz = len(pattern.indices)

    def extract(self, matrices):
        """按固定拓扑提取一批数值.

        Parameters
        ----------
        matrices : torch.Tensor
            完整局部刚度批次.

        Returns
        -------
        torch.Tensor
            形状为 (batch, nnz) 的稀疏块值.
        """
        return matrices[:, self.rows, self.cols]

    def apply(self, values, vector, *, transpose=False):
        """对一批稀疏块执行矩阵向量乘, 正确累加重复行编号.

        Parameters
        ----------
        values : torch.Tensor
            批量拓扑内数值, 包括零条目.
        vector : torch.Tensor
            批量输入向量.
        transpose : bool, optional
            是否使用转置块.

        Returns
        -------
        torch.Tensor
            批量输出向量.
        """
        rows, cols = (self.block_cols, self.block_rows) if transpose else (self.block_rows, self.block_cols)
        nrows = self.shape[1] if transpose else self.shape[0]
        result = values.new_zeros((len(values), nrows))
        result.scatter_add_(1, rows.expand(len(values), -1), values * vector[:, cols])
        return result


class TorchLocalSchurBlocks:
    """两条 FA 路径共用的设备端局部块与恢复数据.

    Parameters
    ----------
    interior : TorchInteriorBlockSolver
        设备端内部块求解器.
    interface_dofs, local_dofs : torch.Tensor
        全局接口及各子结构的接口编号.
    patterns : tuple[TorchBlockPattern, TorchBlockPattern]
        耦合与边界模式, 仅依赖拓扑.
    keep_boundary : bool
        是否为隐式作用缓存边界块.
    chunk_size : int
        每次作用与恢复的最大子结构批量.
    """

    def __init__(self, interior, interface_dofs, local_dofs, patterns, *, keep_boundary, chunk_size):
        self.interior = interior
        self.interface_dofs = interface_dofs.to(dtype=torch.int64, device=interior.indices.device)
        self.local_dofs = local_dofs.to(dtype=torch.int64, device=interior.indices.device)
        self.patterns = patterns
        self.chunk_size = int(chunk_size)
        count = len(interior.indices)
        if (self.chunk_size < 1 or self.local_dofs.shape != (count, patterns[0].shape[1])
                or bool((self.local_dofs < 0).any())
                or bool((self.local_dofs >= len(self.interface_dofs)).any())):
            raise ValueError("局部接口映射或批量大小错误.")
        all_dofs = torch.cat((self.interface_dofs, interior.indices.flatten()))
        if not torch.equal(
                torch.sort(all_dofs).values,
                torch.arange(interior.full_size, device=all_dofs.device)):
            raise ValueError("内部与接口编号须完整且互斥.")
        self.coupling = interior.factors.new_empty((count, patterns[0].nnz))
        self.boundary = interior.factors.new_empty((count, patterns[1].nnz)) if keep_boundary else None
        self.diagonal = interior.factors.new_zeros(len(self.interface_dofs))
        self.ready = np.zeros(count, dtype=bool)

    def _clock(self):
        """同步目标 CUDA 设备后返回分项计时点.

        Returns
        -------
        float
            已等待此前设备操作完成的主机时间.
        """
        if self.coupling.is_cuda:
            torch.cuda.synchronize(self.coupling.device)
        return perf_counter()

    def prepare_batch(self, start, matrices, internal, boundary, *, explicit,
                      timings, column_chunk=32):
        """批量准备公共因子和稀疏块, 可选形成显式 Schur 块.

        Parameters
        ----------
        start : int
            起始子结构编号.
        matrices : torch.Tensor
            稠密局部刚度批次.
        internal, boundary : torch.Tensor
            局部内部与边界编号.
        explicit : bool
            是否形成显式 Schur 批次.
        timings : dict
            累加同步后的分项耗时.
        column_chunk : int, optional
            多右端内部求解的列批大小.

        Returns
        -------
        torch.Tensor or None
            显式 Schur 批次, 隐式返回 None.
        """
        if column_chunk < 1 or explicit == (self.boundary is not None):
            raise ValueError("列批大小或缓存策略不合法.")
        end = start + len(matrices)
        if start < 0 or end > len(self.ready) or self.ready[start:end].any():
            raise ValueError("批次越界或重复准备.")
        started = self._clock()
        self.interior.factor_batch(start, matrices[:, internal[:, None], internal])
        timings['internal_factorization'] += self._clock() - started
        started = self._clock()
        self.coupling[start:end] = self.patterns[0].extract(matrices)
        if self.boundary is not None:
            self.boundary[start:end] = self.patterns[1].extract(matrices)
        timings['local_blocks'] += self._clock() - started
        started = self._clock()
        # 只在准备当前批次时使用稠密耦合块, 迭代与恢复使用紧凑稀疏值.
        coupling = matrices[:, internal[:, None], boundary]
        diagonal = matrices[:, boundary, boundary].clone()
        schur = matrices[:, boundary[:, None], boundary].clone() if explicit else None
        for first in range(0, len(boundary), column_chunk):
            last = min(first + column_chunk, len(boundary))
            rhs = coupling[:, :, first:last]
            solved = self.interior.solve_batch(start, rhs)
            diagonal[:, first:last] -= (rhs * solved).sum(dim=1)
            if explicit:
                schur[:, :, first:last] -= torch.bmm(coupling.transpose(1, 2), solved)
        self.diagonal.index_add_(0, self.local_dofs[start:end].flatten(), diagonal.flatten())
        timings['schur_formation' if explicit else 'schur_diagonal'] += self._clock() - started
        self.ready[start:end] = True
        return schur

    def _check_vector(self, vector):
        """检查完整接口向量与缓存是否可用.

        Parameters
        ----------
        vector : torch.Tensor
            同设备的完整接口向量.

        Returns
        -------
        None
            检查失败时抛出异常.
        """
        if (not self.ready.all() or vector.shape != self.diagonal.shape
                or vector.device != self.diagonal.device or vector.dtype != self.diagonal.dtype):
            raise ValueError("局部块状态或接口向量的形状、设备、类型不符.")

    def recover(self, vector):
        """按批复用内部因子和稀疏耦合块恢复全场位移.

        Parameters
        ----------
        vector : torch.Tensor
            完整接口位移.

        Returns
        -------
        torch.Tensor
            同设备的全场位移.
        """
        self._check_vector(vector)
        full = vector.new_zeros(self.interior.full_size)
        full[self.interface_dofs] = vector
        for first in range(0, len(self.ready), self.chunk_size):
            last = min(first + self.chunk_size, len(self.ready))
            rhs = -self.patterns[0].apply(
                self.coupling[first:last], vector[self.local_dofs[first:last]])
            full[self.interior.indices[first:last]] = self.interior.solve_batch(first, rhs)
        return full

    def apply(self, vector):
        """按批计算并散加局部 Schur 作用, 不在迭代中往返主机.

        Parameters
        ----------
        vector : torch.Tensor
            完整接口向量.

        Returns
        -------
        torch.Tensor
            同设备的完整接口力.
        """
        self._check_vector(vector)
        if self.boundary is None:
            raise ValueError("隐式作用需要缓存边界块.")
        result = torch.zeros_like(vector)
        for first in range(0, len(self.ready), self.chunk_size):
            last = min(first + self.chunk_size, len(self.ready))
            ids = self.local_dofs[first:last]
            local = vector[ids]
            values = self.coupling[first:last]
            solved = self.interior.solve_batch(first, self.patterns[0].apply(values, local))
            force = (self.patterns[1].apply(self.boundary[first:last], local)
                     - self.patterns[0].apply(values, solved, transpose=True))
            result.index_add_(0, ids.flatten(), force.flatten())
        return result