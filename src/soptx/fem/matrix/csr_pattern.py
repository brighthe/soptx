# -*- coding: utf-8 -*-
"""模式先行 (Pattern-First) CSR 稀疏矩阵装配核心模块.

本模块将有限元全局刚度矩阵的装配解耦为两个独立阶段:

1. 符号阶段 (Symbolic Phase, 一次性构建):
   - 对张量空间, CSR 骨架建在其标量空间层面: 只对 ``scalar_space.cell_to_dof()``
     做 ``ldof_s x ldof_s`` 笛卡尔积 (长度 ``NC * ldof_s ** 2``), 再按 ``GD x GD``
     分量块闭式展开为张量自由度级的 ``(crow, col)``;
   - 常驻映射只保存标量级槽位基址 ``slot_base (NC, ldof_s, ldof_s)`` 与标量行度数
     ``row_deg (NC, ldof_s)``, 不物化长度 ``NC * (GD * ldof_s) ** 2`` 的槽位数组;
   - 将模式对象常驻目标后端与物理计算设备 (CPU/GPU).
2. 数值阶段 (Numeric Phase, 每轮迭代快速执行):
   - 对 ``GD x GD`` 个分量对, 由基址与行度数闭式算出该块的全局槽位, 用后端无关的
     ``bm.add_at`` 原地原子累加 (numpy 端为 ``np.add.at``, pytorch 端为
     ``index_put_(accumulate=True)``);
   - 零全长三元组物化, 零排序, 以 O(1) 零拷贝直接产出标准的 FEALPy ``CSRTensor``.

自由度编号约定与 FEALPy ``to_tensor_dof`` / ``generate_tensor_basis`` 严格一致.
记标量全局自由度 ``s``, 分量 ``a``, 单元内标量局部自由度 ``i``, 标量全局自由度总数
``sgdof``, 单元内标量局部自由度数 ``ldof_s``:

- ``dof_priority=True``: 全局编号 ``a * sgdof + s``, 单元内局部编号 ``a * ldof_s + i``;
- ``dof_priority=False``: 全局编号 ``s * GD + a``, 单元内局部编号 ``i * GD + a``.

不提供 ``scalar_space`` 的空间 (如 Hu-Zhang 等混合空间) 自动回退到自由度级骨架,
此时 ``dof_numel == 1``, ``slot_base`` 退化为自由度级槽位映射, 数值阶段公式不变.

后端与设备: 符号阶段的骨架分析在 CPU 端用 NumPy/SciPy 完成, 结果一次性搬运到目标
后端与设备; 数值阶段只用广播算术与 ``bm`` 层原语, 无后端分支, 全程留在设备上,
与原实现一样支持 ``numpy`` 与 ``pytorch`` (含 CUDA) 后端.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor
from fealpy.typing import TensorLike

#: 符号阶段查询键分块长度, 用于限制 ``searchsorted`` 的瞬时内存 (查询键与其输出各 8 B/元素).
_QUERY_CHUNK = 1 << 20


def _slot_of(
    slot_base: TensorLike,
    row_deg: Optional[TensorLike],
    dof_numel: int,
    dof_priority: bool,
    scalar_nnz: int,
    a: int,
    b: int,
    cell_slice: Any = slice(None),
) -> TensorLike:
    """闭式给出分量对 ``(a, b)`` 的全局槽位, 形状 ``(NC, ldof_s, ldof_s)``.

    设标量条目位于标量 CSR 的第 ``s`` 行、行内偏移 ``off``, 该行度数 ``deg``:

    - ``dof_priority=True``: 张量行 ``a * sgdof + s`` 的行首为
      ``a * GD * snnz + GD * sindptr[s]``, 行内位置为 ``b * deg + off``;
    - ``dof_priority=False``: 张量行 ``s * GD + a`` 的行首为
      ``GD ** 2 * sindptr[s] + a * GD * deg``, 行内位置为 ``GD * off + b``.

    与 ``a``, ``b`` 无关的部分已在符号阶段预先合并进 ``slot_base``.
    """
    slot_base = slot_base[cell_slice]
    if dof_numel == 1:
        return slot_base

    deg = row_deg[cell_slice, :, None]
    if dof_priority:
        return slot_base + a * (dof_numel * scalar_nnz) + b * deg
    return slot_base + (a * dof_numel) * deg + b


def _iter_blocks(
    K_e: TensorLike,
    NC: int,
    ldof_s: int,
    dof_numel: int,
    dof_priority: bool,
) -> Iterator[Tuple[int, int, TensorLike]]:
    """把 ``(NC, ldof, ldof)`` 单刚按分量对切成 ``GD * GD`` 个 ``(NC, ldof_s, ldof_s)`` 视图.

    局部自由度编号与 ``generate_tensor_basis`` 一致: ``dof_priority=True`` 为
    ``a * ldof_s + i``, ``dof_priority=False`` 为 ``i * GD + a``.
    """
    if dof_numel == 1:
        yield 0, 0, K_e.reshape(NC, ldof_s, ldof_s)
        return

    if dof_priority:
        Kv = K_e.reshape(NC, dof_numel, ldof_s, dof_numel, ldof_s)
        for a in range(dof_numel):
            for b in range(dof_numel):
                yield a, b, Kv[:, a, :, b, :]
    else:
        Kv = K_e.reshape(NC, ldof_s, dof_numel, ldof_s, dof_numel)
        for a in range(dof_numel):
            for b in range(dof_numel):
                yield a, b, Kv[:, :, a, :, b]


@dataclass
class CSRPattern:
    """CSR 稀疏矩阵静态拓扑骨架与标量级槽位映射.

    Attributes:
        crow (TensorLike): CSR 行指针数组 (gdof + 1,).
        col (TensorLike): CSR 列索引数组 (nnz,).
        slot_base (TensorLike): 标量级槽位基址 (NC, ldof_s, ldof_s); 各分量对的全局
            槽位由它与 ``row_deg`` 闭式算出. ``dof_numel == 1`` 时它即自由度级槽位映射.
        sparse_shape (Tuple[int, int]): 全局稀疏矩阵形状 (gdof, gdof).
        dof_numel (int): 每个标量自由度携带的分量数 GD; 非张量空间为 1.
        dof_priority (bool): 张量自由度排布优先级, 与 ``TensorFunctionSpace`` 同义.
        row_deg (TensorLike): 每个 (单元, 局部标量行自由度) 所在标量 CSR 行的非零元
            个数 (NC, ldof_s); ``dof_numel == 1`` 时为 ``None``.
        scalar_nnz (int): 标量级骨架非零元个数 snnz, 满足 ``nnz == snnz * GD ** 2``.
        device: 所在的硬件计算设备.
        backend (str): 后端类型名称 ('numpy' | 'pytorch').
        buffer (TensorLike): 预分配的原地数值缓冲张量 (nnz,).
    """

    crow: TensorLike
    col: TensorLike
    slot_base: TensorLike
    sparse_shape: Tuple[int, int]
    dof_numel: int = 1
    dof_priority: bool = True
    row_deg: Optional[TensorLike] = None
    scalar_nnz: int = 0
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

    @property
    def n_cells(self) -> int:
        """单元数 NC."""
        return int(self.slot_base.shape[0])

    @property
    def scalar_ldof(self) -> int:
        """单元内标量局部自由度数 ldof_s."""
        return int(self.slot_base.shape[1])

    def slot_of(self, a: int, b: int, cell_slice: Any = slice(None)) -> TensorLike:
        """分量对 ``(a, b)`` 对应的全局槽位, 形状 ``(NC, ldof_s, ldof_s)``."""
        return _slot_of(
            self.slot_base, self.row_deg, self.dof_numel,
            self.dof_priority, self.scalar_nnz, a, b, cell_slice,
        )

    def to(self, device: Any) -> CSRPattern:
        """将模式中的所有张量迁移至指定的计算设备 (如 GPU 或 CPU)."""
        if self.backend == "pytorch":
            import torch

            target_device = torch.device(device) if isinstance(device, str) else device

            def move(t):
                return None if t is None else t.to(device=target_device)

            return CSRPattern(
                crow=move(self.crow),
                col=move(self.col),
                slot_base=move(self.slot_base),
                sparse_shape=self.sparse_shape,
                dof_numel=self.dof_numel,
                dof_priority=self.dof_priority,
                row_deg=move(self.row_deg),
                scalar_nnz=self.scalar_nnz,
                device=target_device,
                backend=self.backend,
                buffer=move(self.buffer),
            )
        self.device = device
        return self


def _as_2d_numpy(c2d_raw: Any) -> np.ndarray:
    """把 ``cell_to_dof()`` 的返回值规整为 CPU 端 ``(NC, ldof)`` 的 NumPy 数组."""
    c2d = bm.to_numpy(c2d_raw)
    if c2d.ndim != 2:
        c2d = c2d.reshape(c2d.shape[0], -1)
    return c2d


def _skeleton_and_slots(
    c2d: np.ndarray,
    gdof: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """由 ``(NC, ldof)`` 自由度映射构造 CSR 骨架与每个 ``(cell, i, j)`` 的槽位.

    查询键按 ``_QUERY_CHUNK`` 分块构造并 ``searchsorted``, 因此瞬时内存与
    ``NC * ldof ** 2`` 无关, 只与分块长度有关.

    Parameters:
        c2d: 自由度映射, 形状 (NC, ldof).
        gdof: 该层的全局自由度总数.

    Returns:
        indptr: CSR 行指针 (gdof + 1,) int64.
        indices: CSR 列索引 (nnz,) int64.
        slot: 每个 (cell, i, j) 在 ``indices`` 中的位置 (NC, ldof, ldof) int64.
        deg: 每行非零元个数 (gdof,) int64.
    """
    NC, ldof = c2d.shape
    n_pair = NC * ldof * ldof

    row = np.broadcast_to(c2d[:, :, None], (NC, ldof, ldof)).reshape(-1)
    col = np.broadcast_to(c2d[:, None, :], (NC, ldof, ldof)).reshape(-1)

    skeleton = sp.coo_matrix(
        (np.ones(n_pair, dtype=np.int8), (row, col)),
        shape=(gdof, gdof),
    ).tocsr()
    indptr = skeleton.indptr.astype(np.int64)
    indices = skeleton.indices.astype(np.int64)
    del skeleton

    deg = np.diff(indptr)
    # 每行非零元按列升序排列, 故 key 全局单调递增, 可直接二分查找
    key = np.repeat(np.arange(gdof, dtype=np.int64), deg)
    key *= gdof
    key += indices

    slot = np.empty(n_pair, dtype=np.int64)
    for start in range(0, n_pair, _QUERY_CHUNK):
        stop = min(start + _QUERY_CHUNK, n_pair)
        query = np.multiply(row[start:stop], gdof, dtype=np.int64)
        query += col[start:stop]
        slot[start:stop] = np.searchsorted(key, query)
        del query
    del row, col, key

    return indptr, indices, slot.reshape(NC, ldof, ldof), deg


def _tensor_csr_topology(
    sindptr: np.ndarray,
    sindices: np.ndarray,
    sdeg: np.ndarray,
    sgdof: int,
    dof_numel: int,
    dof_priority: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """由标量 CSR 骨架闭式展开张量自由度级 CSR 骨架.

    标量骨架的每个非零元 ``(s, c)`` 对应张量矩阵中一个稠密的 ``GD x GD`` 块, 因此
    张量骨架的 ``indptr`` / ``indices`` 可由标量骨架直接算出, 中间量长度不超过 snnz.
    """
    GD = dof_numel
    snnz = int(sindices.shape[0])
    nnz = snnz * GD * GD
    comp = np.arange(GD, dtype=np.int64)

    if dof_priority:
        # 张量行 a * sgdof + s, 行首 a * GD * snnz + GD * sindptr[s]
        head = comp[:, None] * (GD * snnz) + GD * sindptr[None, :sgdof]
    else:
        # 张量行 s * GD + a, 行首 GD ** 2 * sindptr[s] + a * GD * sdeg[s]
        head = (GD * GD) * sindptr[:sgdof, None] + comp[None, :] * (GD * sdeg[:, None])

    indptr = np.empty(GD * sgdof + 1, dtype=np.int64)
    indptr[:-1] = head.reshape(-1)
    indptr[-1] = nnz
    del head

    entry_row = np.repeat(np.arange(sgdof, dtype=np.int64), sdeg)
    entry_start = sindptr[entry_row]
    entry_off = np.arange(snnz, dtype=np.int64) - entry_start
    entry_deg = sdeg[entry_row]
    del entry_row

    if dof_priority:
        entry_base = GD * entry_start + entry_off
        inc_a, inc_b = GD * snnz, entry_deg
    else:
        entry_base = (GD * GD) * entry_start + GD * entry_off
        inc_a, inc_b = GD * entry_deg, 1
    del entry_start, entry_off

    indices = np.empty(nnz, dtype=np.int64)
    for b in range(GD):
        col_b = sindices + b * sgdof if dof_priority else sindices * GD + b
        base_b = entry_base + b * inc_b
        for a in range(GD):
            indices[base_b + a * inc_a] = col_b

    return indptr, indices


def _finalize(
    indptr_np: np.ndarray,
    indices_np: np.ndarray,
    slot_np: np.ndarray,
    deg_np: Optional[np.ndarray],
    gdof: int,
    dof_numel: int,
    dof_priority: bool,
    scalar_nnz: int,
    device: Any,
    dtype: Any,
    device_hint: Any,
    allocate_buffer: bool = True,
) -> CSRPattern:
    """把 CPU 端的骨架与槽位一次性搬运到当前后端与目标设备."""
    nnz = int(indices_np.shape[0])
    if dtype is None:
        dtype = bm.float64

    if bm.backend_name == "pytorch":
        import torch

        if device is None:
            device = getattr(device_hint, "device", None) or torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        def load(a):
            if a is None:
                return None
            return torch.from_numpy(np.ascontiguousarray(a)).to(
                device=device, dtype=torch.int64
            )

        return CSRPattern(
            crow=load(indptr_np),
            col=load(indices_np),
            slot_base=load(slot_np),
            sparse_shape=(gdof, gdof),
            dof_numel=dof_numel,
            dof_priority=dof_priority,
            row_deg=load(deg_np),
            scalar_nnz=scalar_nnz,
            device=device,
            backend="pytorch",
            buffer=(
                torch.zeros(nnz, dtype=dtype, device=device)
                if allocate_buffer else None
            ),
        )

    return CSRPattern(
        crow=bm.from_numpy(indptr_np),
        col=bm.from_numpy(indices_np),
        slot_base=bm.from_numpy(slot_np),
        sparse_shape=(gdof, gdof),
        dof_numel=dof_numel,
        dof_priority=dof_priority,
        row_deg=None if deg_np is None else bm.from_numpy(deg_np),
        scalar_nnz=scalar_nnz,
        device="cpu" if device is None else device,
        backend="numpy",
        buffer=np.zeros(nnz, dtype=dtype) if allocate_buffer else None,
    )


def _build_dof_level(space: Any, device: Any, dtype: Any) -> CSRPattern:
    """自由度级骨架 (回退路径): 直接对 ``space.cell_to_dof()`` 做笛卡尔积."""
    c2d_raw = space.cell_to_dof()
    c2d = _as_2d_numpy(c2d_raw)
    gdof = int(space.number_of_global_dofs())

    indptr, indices, slot, _ = _skeleton_and_slots(c2d, gdof)
    del c2d

    return _finalize(
        indptr, indices, slot, None, gdof,
        dof_numel=1, dof_priority=True, scalar_nnz=int(indices.shape[0]),
        device=device, dtype=dtype, device_hint=c2d_raw,
    )


def _build_tensor_level(
    space: Any,
    scalar_space: Any,
    dof_numel: int,
    dof_priority: bool,
    device: Any,
    dtype: Any,
) -> CSRPattern:
    """标量级骨架 (主路径): 骨架与常驻槽位都建在标量空间层面."""
    GD = dof_numel
    c2d_raw = scalar_space.cell_to_dof()
    c2d_s = _as_2d_numpy(c2d_raw)
    NC, ldof_s = c2d_s.shape

    sgdof = int(scalar_space.number_of_global_dofs())
    gdof = int(space.number_of_global_dofs())
    if gdof != GD * sgdof:
        raise ValueError(
            f"张量空间全局自由度 {gdof} 与 GD * sgdof = {GD} * {sgdof} 不一致, "
            "无法按分量块展开标量骨架."
        )

    sindptr, sindices, sslot, sdeg = _skeleton_and_slots(c2d_s, sgdof)
    scalar_nnz = int(sindices.shape[0])

    indptr, indices = _tensor_csr_topology(
        sindptr, sindices, sdeg, sgdof, GD, dof_priority
    )
    del sindices

    row_start = sindptr[c2d_s]          # (NC, ldof_s) 标量行首在 sindices 中的位置
    row_deg = sdeg[c2d_s]               # (NC, ldof_s) 标量行的非零元个数
    del c2d_s, sindptr, sdeg

    # sslot -> 行内偏移 off -> 与分量无关的槽位基址 (原地演化, 不额外物化)
    sslot -= row_start[:, :, None]
    if dof_priority:
        sslot += (GD * row_start)[:, :, None]
    else:
        sslot *= GD
        sslot += ((GD * GD) * row_start)[:, :, None]
    del row_start

    return _finalize(
        indptr, indices, sslot, row_deg, gdof,
        dof_numel=GD, dof_priority=dof_priority, scalar_nnz=scalar_nnz,
        device=device, dtype=dtype, device_hint=c2d_raw,
    )


def build_csr_pattern(
    space: Any,
    device: Any = None,
    dtype: Any = None,
) -> CSRPattern:
    """从有限元空间构建 CSR 拓扑骨架与槽位映射 (符号阶段).

    张量空间走标量级骨架, 符号阶段与常驻映射的规模都比自由度级低 ``GD ** 2`` 倍;
    其余空间回退到自由度级骨架.

    Parameters:
        space: 有限元空间对象 (如 ``TensorFunctionSpace`` 或 ``LagrangeFESpace``).
        device: 目标设备. 若为 ``None``, 则自动推导自空间所在设备或当前后端.
        dtype: 数值缓冲区的数据类型. 若为 ``None``, 默认为 ``bm.float64``.

    Returns:
        pattern (CSRPattern): 构建完成的静态 CSR 模式与槽位映射对象.
    """
    scalar_space = getattr(space, "scalar_space", None)
    dof_numel = int(getattr(space, "dof_numel", 1))

    # cell_to_dof 返回 tuple 的变阶空间无法按分量块展开, 一律走自由度级骨架
    if scalar_space is not None and dof_numel > 1:
        if not isinstance(scalar_space.cell_to_dof(), tuple):
            return _build_tensor_level(
                space,
                scalar_space,
                dof_numel,
                bool(getattr(space, "dof_priority", True)),
                device,
                dtype,
            )

    return _build_dof_level(space, device, dtype)


def build_csr_pattern_from_dofmap(
    local_to_global: Any,
    number_of_global_dofs: int,
    *,
    dof_numel: int = 1,
    dof_priority: bool = True,
    device: Any = None,
    dtype: Any = None,
    allocate_buffer: bool = True,
) -> CSRPattern:
    """从显式局部到全局映射构建可复用的 CSR 模式.

    dof_numel 为 1 时输入完整自由度映射; 大于 1 时输入标量实体映射,
    并按 dof_priority 展开各分量.
    """
    c2d_raw = local_to_global
    c2d = _as_2d_numpy(c2d_raw)
    if not np.issubdtype(c2d.dtype, np.integer):
        raise TypeError("local_to_global 必须使用整数类型.")
    gdof = int(number_of_global_dofs)
    GD = int(dof_numel)
    if gdof <= 0 or GD <= 0 or gdof % GD != 0:
        raise ValueError(
            "number_of_global_dofs 必须为正且能被 dof_numel 整除; "
            f"当前为 {gdof} 与 {GD}."
        )
    sgdof = gdof // GD
    if c2d.size == 0 or np.min(c2d) < 0 or np.max(c2d) >= sgdof:
        raise ValueError(f"local_to_global 必须落在 [0, {sgdof}) 且不能为空.")

    sindptr, sindices, slot, sdeg = _skeleton_and_slots(c2d, sgdof)
    scalar_nnz = int(sindices.shape[0])
    if GD == 1:
        return _finalize(
            sindptr, sindices, slot, None, gdof,
            dof_numel=1, dof_priority=True, scalar_nnz=scalar_nnz,
            device=device, dtype=dtype, device_hint=c2d_raw,
            allocate_buffer=allocate_buffer,
        )

    indptr, indices = _tensor_csr_topology(
        sindptr, sindices, sdeg, sgdof, GD, bool(dof_priority)
    )
    row_start = sindptr[c2d]
    row_deg = sdeg[c2d]
    slot -= row_start[:, :, None]
    if dof_priority:
        slot += (GD * row_start)[:, :, None]
    else:
        slot *= GD
        slot += ((GD * GD) * row_start)[:, :, None]
    return _finalize(
        indptr, indices, slot, row_deg, gdof,
        dof_numel=GD, dof_priority=bool(dof_priority),
        scalar_nnz=scalar_nnz, device=device, dtype=dtype,
        device_hint=c2d_raw, allocate_buffer=allocate_buffer,
    )


def _add_csr_chunk(
    K_e: TensorLike,
    pattern: CSRPattern,
    buffer: TensorLike,
    start: int,
) -> None:
    """把连续局部矩阵批次累加到已清零的 CSR 数值缓冲区."""
    NC = int(K_e.shape[0])
    stop = start + NC
    if start < 0 or stop > pattern.n_cells:
        raise ValueError(
            f"局部矩阵批次 [{start}, {stop}) 超出 [0, {pattern.n_cells})."
        )
    ldof = pattern.scalar_ldof * pattern.dof_numel
    if tuple(K_e.shape[1:]) != (ldof, ldof):
        raise ValueError(
            f"局部矩阵尾部形状必须为 ({ldof}, {ldof}); "
            f"当前为 {tuple(K_e.shape[1:])}."
        )
    cell_slice = slice(start, stop)
    for a, b, values in _iter_blocks(
        K_e, NC, pattern.scalar_ldof,
        pattern.dof_numel, pattern.dof_priority,
    ):
        bm.add_at(buffer, pattern.slot_of(a, b, cell_slice), values)


def assemble_csr(
    K_e: TensorLike,
    pattern: CSRPattern,
    buffer: Optional[TensorLike] = None,
) -> CSRTensor:
    """利用静态模式与槽位映射进行数值装配 (数值阶段).

    按 ``GD * GD`` 个分量对分块累加: 每块的槽位由 ``slot_base`` 与 ``row_deg`` 闭式
    算出, 瞬时索引量为 ``NC * ldof_s ** 2``, 不出现 ``NC * ldof ** 2`` 的全长索引.

    Parameters:
        K_e: 单元刚度张量, 形状为 ``(NC, ldof, ldof)``.
        pattern: 符号阶段预建的 ``CSRPattern`` 对象.
        buffer: 可选的原地缓冲数组/张量. 若为 ``None``, 则使用 ``pattern.buffer``.

    Returns:
        matrix (CSRTensor): 装配完成的标准 FEALPy ``CSRTensor`` 稀疏矩阵.
    """
    GD = pattern.dof_numel
    NC = pattern.n_cells
    ldof_s = pattern.scalar_ldof

    if buffer is None:
        buffer = pattern.buffer
        if buffer is None:
            buffer = bm.zeros(pattern.nnz, dtype=K_e.dtype, device=pattern.device)

    # 原地置零后逐分量块原子累加. ``bm.add_at`` 在 numpy 端落到 ``np.add.at``, 在 pytorch
    # 端落到 ``index_put_(accumulate=True)`` (CUDA 上即 atomicAdd), 全程留在当前后端与设备.
    # 索引与源都保持 (NC, ldof_s, ldof_s) 形状而不展平: 分量块是 ``K_e`` 的跨步视图,
    # ``reshape(-1)`` 会强制物化一份 ``NC * ldof_s ** 2`` 的连续副本.
    bm.set_at(buffer, slice(None), 0.0)
    for a, b, block in _iter_blocks(K_e, NC, ldof_s, GD, pattern.dof_priority):
        bm.add_at(buffer, pattern.slot_of(a, b), block)

    return CSRTensor(
        crow=pattern.crow,
        col=pattern.col,
        values=buffer,
        spshape=pattern.sparse_shape,
    )


def assemble_csr_chunks(
    chunks: Iterable[Tuple[int, TensorLike]],
    pattern: CSRPattern,
    buffer: Optional[TensorLike] = None,
) -> CSRTensor:
    """按连续批次把局部矩阵累加到同一个 CSR 数值缓冲区."""
    iterator = iter(chunks)
    try:
        first_start, first_values = next(iterator)
    except StopIteration as error:
        raise ValueError("chunks 不能为空.") from error

    if buffer is None:
        buffer = pattern.buffer
        if buffer is None:
            buffer = bm.zeros(
                (pattern.nnz,), dtype=first_values.dtype, device=pattern.device
            )
    if int(buffer.shape[0]) != pattern.nnz:
        raise ValueError(
            f"buffer 长度必须为 {pattern.nnz}; 当前为 {int(buffer.shape[0])}."
        )
    bm.set_at(buffer, slice(None), 0.0)

    if int(first_start) != 0:
        raise ValueError(f"chunks 必须从 0 开始; 当前为 {first_start}.")
    _add_csr_chunk(first_values, pattern, buffer, 0)
    expected = int(first_values.shape[0])
    del first_values
    for start, values in iterator:
        start = int(start)
        if start != expected:
            raise ValueError(
                f"chunks 必须连续覆盖; 期望 {expected}, 当前为 {start}."
            )
        _add_csr_chunk(values, pattern, buffer, start)
        expected += int(values.shape[0])
        del values
    if expected != pattern.n_cells:
        raise ValueError(
            f"chunks 只覆盖 {expected} 个局部实体; 需要 {pattern.n_cells} 个."
        )
    return CSRTensor(
        crow=pattern.crow, col=pattern.col, values=buffer,
        spshape=pattern.sparse_shape,
    )
