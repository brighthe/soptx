# -*- coding: utf-8 -*-
"""模式先行 (Pattern-First) CSR 稀疏矩阵装配核心模块.

本模块将有限元全局刚度矩阵的装配解耦为两个独立阶段:
1. 符号阶段 (Symbolic Phase, 一次性构建):
   - 基于有限元空间自由度拓扑提取全局 CSR 稀疏骨架 `(crow, col)`;
   - 预计算每个单元单刚元素在全局 CSR 非零元数组中的槽位映射 `slot_map`;
   - 将模式对象常驻目标后端与物理计算设备 (CPU/GPU).
2. 数值阶段 (Numeric Phase, 每轮迭代快速执行):
   - 利用槽位映射进行原地原子累加 (NumPy `np.add.at` / PyTorch `scatter_add_`);
   - 零临时三元组物化、零显存排序, 以 O(1) 零拷贝直接产出标准的 FEALPy `CSRTensor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor
from fealpy.typing import TensorLike


@dataclass
class CSRPattern:
    """CSR 稀疏矩阵静态拓扑骨架与单元槽位映射.

    Attributes:
        crow (TensorLike): CSR 行指针数组 (n_dof + 1,).
        col (TensorLike): CSR 列索引数组 (nnz,).
        slot_map (TensorLike): 单刚局部元素到全局 values 数组的槽位映射 (n_elements * ldof * ldof,).
        sparse_shape (Tuple[int, int]): 全局稀疏矩阵形状 (n_dof, n_dof).
        device: 所在的硬件计算设备.
        backend: 后端类型名称 ('numpy' | 'pytorch').
        buffer: 预分配的原地数值缓冲张量 (nnz,).
    """

    crow: TensorLike
    col: TensorLike
    slot_map: TensorLike
    sparse_shape: Tuple[int, int]
    device: Any = None
    backend: str = "numpy"
    buffer: Optional[TensorLike] = None

    @property
    def nnz(self) -> int:
        """非零元总数 (Number of Non-Zeros)."""
        return int(self.col.shape[0])

    @property
    def shape(self) -> Tuple[int, int]:
        """稀疏矩阵全局维度."""
        return self.sparse_shape

    def to(self, device: Any) -> CSRPattern:
        """将模式中的所有张量迁移至指定的计算设备 (如 GPU 或 CPU)."""
        if self.backend == "pytorch":
            import torch

            target_device = torch.device(device) if isinstance(device, str) else device
            new_crow = self.crow.to(device=target_device)
            new_col = self.col.to(device=target_device)
            new_slot_map = self.slot_map.to(device=target_device)
            new_buffer = (
                self.buffer.to(device=target_device)
                if self.buffer is not None
                else None
            )
            return CSRPattern(
                crow=new_crow,
                col=new_col,
                slot_map=new_slot_map,
                sparse_shape=self.sparse_shape,
                device=target_device,
                backend=self.backend,
                buffer=new_buffer,
            )
        self.device = device
        return self


def build_csr_pattern(
    space: Any,
    device: Any = None,
    dtype: Any = None,
) -> CSRPattern:
    """从有限元空间构建 CSR 拓扑骨架与单元槽位映射 (符号阶段).

    Parameters:
        space: 有限元空间对象 (如 `TensorFunctionSpace` 或 `LagrangeFESpace`).
        device: 目标设备. 若为 `None`, 则自动推导自空间所在设备或当前后端.
        dtype: 数值缓冲区的数据类型. 若为 `None`, 默认为 `bm.float64`.

    Returns:
        pattern (CSRPattern): 构建完成的静态 CSR 模式与槽位映射对象.
    """
    backend_name = bm.backend_name

    # 1. 提取单元自由度映射数组 cell_to_dof
    c2d_raw = space.cell_to_dof()
    c2d = bm.to_numpy(c2d_raw)
    if c2d.ndim != 2:
        c2d = c2d.reshape(c2d.shape[0], -1)

    NC, ldof = c2d.shape
    gdof = space.number_of_global_dofs()

    # 2. 构造单元局部自由度的 Cartesian 积笛卡尔对 (I, J)
    row_idx = np.broadcast_to(c2d[:, :, None], (NC, ldof, ldof)).ravel()
    col_idx = np.broadcast_to(c2d[:, None, :], (NC, ldof, ldof)).ravel()

    # 3. 在 CPU 端利用轻量 COO -> CSR 拓扑去重分析提取唯一骨架
    A_skel = sp.coo_matrix(
        (np.ones(row_idx.size, dtype=np.int8), (row_idx, col_idx)),
        shape=(gdof, gdof),
    ).tocsr()

    indptr_np = A_skel.indptr.astype(np.int64)
    indices_np = A_skel.indices.astype(np.int64)
    nnz = indices_np.size

    # 4. 构建每行非零元的全局唯一排序键, 矢量化查找各单元局部贡献的槽位映射 slot_map
    deg = np.diff(indptr_np)
    rows_np = np.repeat(np.arange(gdof, dtype=np.int64), deg)
    key_np = rows_np * gdof + indices_np
    del rows_np, deg, A_skel

    query_key = row_idx.astype(np.int64) * gdof + col_idx.astype(np.int64)
    slot_map_np = np.searchsorted(key_np, query_key).astype(np.int64)
    del row_idx, col_idx, key_np, query_key

    # 5. 根据当前后端将骨架与槽位映射张量装载到对应设备
    if dtype is None:
        dtype = bm.float64

    if backend_name == "pytorch":
        import torch

        if device is None:
            if hasattr(c2d_raw, "device"):
                device = c2d_raw.device
            else:
                device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        crow_t = torch.from_numpy(indptr_np).to(device=device, dtype=torch.int64)
        col_t = torch.from_numpy(indices_np).to(device=device, dtype=torch.int64)
        slot_map_t = torch.from_numpy(slot_map_np).to(device=device, dtype=torch.int64)
        buffer_t = torch.zeros(nnz, dtype=torch.float64, device=device)

        return CSRPattern(
            crow=crow_t,
            col=col_t,
            slot_map=slot_map_t,
            sparse_shape=(gdof, gdof),
            device=device,
            backend="pytorch",
            buffer=buffer_t,
        )
    else:
        crow_np = bm.from_numpy(indptr_np)
        col_np = bm.from_numpy(indices_np)
        slot_map = bm.from_numpy(slot_map_np)
        buffer_np = np.zeros(nnz, dtype=np.float64)

        return CSRPattern(
            crow=crow_np,
            col=col_np,
            slot_map=slot_map,
            sparse_shape=(gdof, gdof),
            device="cpu",
            backend="numpy",
            buffer=buffer_np,
        )


def assemble_csr(
    K_e: TensorLike,
    pattern: CSRPattern,
    buffer: Optional[TensorLike] = None,
) -> CSRTensor:
    """利用静态模式与槽位映射进行数值装配 (数值阶段).

    Parameters:
        K_e: 单元刚度张量, 形状为 `(NC, ldof, ldof)`.
        pattern: 符号阶段预建的 `CSRPattern` 对象.
        buffer: 可选的原地缓冲数组/张量. 若为 `None`, 则使用 `pattern.buffer`.

    Returns:
        matrix (CSRTensor): 装配完成的标准 FEALPy `CSRTensor` 稀疏矩阵.
    """
    if buffer is None:
        buffer = pattern.buffer
        if buffer is None:
            if pattern.backend == "pytorch":
                import torch

                buffer = torch.zeros(
                    pattern.nnz,
                    dtype=K_e.dtype,
                    device=pattern.device,
                )
            else:
                buffer = np.zeros(pattern.nnz, dtype=np.float64)

    if pattern.backend == "pytorch":
        import torch

        # PyTorch 端: 显存原地置零与原子累加 (CUDA atomicAdd)
        buffer.zero_()
        k_flat = K_e.reshape(-1)
        buffer.scatter_add_(0, pattern.slot_map, k_flat)

        return CSRTensor(
            crow=pattern.crow,
            col=pattern.col,
            values=buffer,
            spshape=pattern.sparse_shape,
        )
    else:
        # NumPy 端: 数组原地置零与 bincount/add.at 原地累加
        buffer.fill(0.0)
        k_flat = bm.to_numpy(K_e).ravel()
        slot_flat = bm.to_numpy(pattern.slot_map).ravel()
        np.add.at(buffer, slot_flat, k_flat)

        return CSRTensor(
            crow=pattern.crow,
            col=pattern.col,
            values=bm.from_numpy(buffer),
            spshape=pattern.sparse_shape,
        )