"""实体级 MPI 消息传递接口 (EntityMPI) 模块.

管理分布式网格/有限元空间中几何实体与自由度的跨进程通信拓扑、重叠引用计数与数据同步.
"""

from __future__ import annotations

__all__ = [
    "EntityMPI",
    "SharingPair",
    "SparseData1D",
    "dist_from_masks",
    "mapped_masks",
]

from collections.abc import Callable, Sequence
from typing import NamedTuple, TypeVar

from mpi4py import MPI
from mpi4py.MPI import Comm, COMM_WORLD

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

_T = TypeVar("_T")


class SharingPair(NamedTuple):
    """跨进程共享实体局部索引映射对."""

    index_self: TensorLike
    index_other: TensorLike


class SparseData1D(NamedTuple):
    """稀疏一维张量数据包装."""

    data: TensorLike
    indices: TensorLike


class EntityMPI:
    """FEALPy/SOPTX 实体级 MPI 通信器.

    管理特定维度几何实体或自由度在各 rank 间的共享关系、引用重数及同步操作.
    """

    _id: int
    _process_map: list[int]
    _global_indices: TensorLike | None
    _sharing_pairs: list[SharingPair | None]

    def __init__(
        self,
        indices: TensorLike | None = None,
        pairs: list[SharingPair | None] | None = None,
        id: int | None = None,
        *,
        comm: Comm | None = None,
    ) -> None:
        self._global_indices = indices
        self._sharing_pairs = pairs if pairs is not None else []
        self._comm = comm if comm is not None else COMM_WORLD

        if id is None:
            id = self._comm.Get_rank()
        self._id = id
        self._process_map = self._make_process_table()

    @property
    def mpi_rank(self) -> int:
        """当前进程的 MPI Rank ID."""
        return self._comm.Get_rank()

    @property
    def mpi_size(self) -> int:
        """通信子中的总进程数."""
        return self._comm.Get_size()

    @property
    def neighbors(self) -> list[int]:
        """与当前 rank 共享实体的邻居 rank 列表."""
        return [
            self._process_map[idx]
            for idx, pair in enumerate(self._sharing_pairs)
            if pair is not None
        ]

    def _make_process_table(self) -> list[int]:
        size = self.mpi_size
        send_buf = [self._id] * size
        recv_buf = self._comm.alltoall(send_buf)
        part2process = [0] * size

        for pro_id in range(size):
            par_id = recv_buf[pro_id]
            part2process[par_id] = pro_id

        return part2process

    def _alltoall(self, data: Sequence[_T]) -> list[_T]:
        """在所有分区之间执行数据全交换 (All-to-All)."""
        size = self.mpi_size
        send_buf = [data[self._process_map[par_id]] for par_id in range(size)]
        recv_buf = self._comm.alltoall(send_buf)
        return [recv_buf[self._process_map[par_id]] for par_id in range(size)]

    def _gather(self, data: _T, /, root: int = 0) -> list[_T] | None:
        recv_buf = self._comm.gather(data, root=root)
        if self._comm.Get_rank() == root:
            assert recv_buf is not None
            size = self.mpi_size
            return [recv_buf[self._process_map[par_id]] for par_id in range(size)]
        return None

    def _scatter(self, data: list[_T] | None, /, root: int = 0) -> _T:
        if data is None:
            send_buf = None
        else:
            send_buf = [
                data[self._process_map[par_id]] for par_id in range(self.mpi_size)
            ]
        return self._comm.scatter(send_buf, root=root)

    def refs(self, size: int) -> TensorLike:
        """返回局部实体/自由度的重叠引用计数向量 r (被多少个 rank 持有)."""
        count = bm.ones((size,), dtype=bm.int32)
        for pair in self._sharing_pairs:
            if pair is None:
                continue
            count[pair.index_self] += 1
        return count

    def dot(self, local_size: int) -> tuple[Callable[[TensorLike, TensorLike], float], Callable[[TensorLike], float]]:
        """返回重叠修正后的内积与范数函数对 (dot, norm).

        共享自由度在多个 rank 上各有一份副本, 普通内积会重复计数.
        此处用 refs() 除以引用计数, 再经 MPI.allreduce 求和, 得到正确的全局加权内积.
        """
        references = self.refs(local_size)
        comm = self._comm

        def _dot(x: TensorLike, y: TensorLike) -> float:
            local = bm.sum(bm.conj(x) * y / references)
            return float(comm.allreduce(float(bm.real(local)), op=MPI.SUM))

        def _norm(x: TensorLike) -> float:
            return max(_dot(x, x), 0.0) ** 0.5

        return _dot, _norm

    def sync(self, array: TensorLike) -> list[SparseData1D | None]:
        """从共享分区同步并交换数组分量.

        参数:
            array: 在局部实体上定义的数据张量.

        返回:
            以分区 ID 排序的邻居数据列表, 元素为 SparseData1D(received_data, local_indices) 或 None.
        """
        if not self._sharing_pairs:
            raise ValueError("需要共享对映射以交换实体数据.")

        data: list[SparseData1D | None] = []
        for pair in self._sharing_pairs:
            if pair is None:
                data.append(None)
                continue
            data.append(
                SparseData1D(
                    bm.asarray(array[pair.index_self], copy=True),
                    pair.index_other,
                )
            )
        return self._alltoall(data)

    def sync_add(self, array: TensorLike) -> TensorLike:
        """跨进程同步归约算子 S: 与相邻 rank 交换重叠分量并累加求和."""
        result = bm.asarray(array, copy=True)
        data_list = self.sync(array)

        for data in data_list:
            if data is None:
                continue
            result = bm.index_add(result, data.indices, data.data)

        return result

    def gather(self, array: TensorLike, /, root: int = 0) -> list[SparseData1D] | None:
        """将各分区的局部数据收集至 Root 进程."""
        if self._global_indices is None:
            raise ValueError("收集数据需要全局实体索引映射.")

        data = SparseData1D(array, self._global_indices)
        return self._gather(data, root=root)

    def gather_add(
        self,
        array: TensorLike,
        /,
        root: int = 0,
        out: TensorLike | None = None,
    ) -> TensorLike | None:
        """将各分区的局部加和表示收集至 Root 并依据全局索引累加还原全局向量."""
        data_list = self.gather(array, root=root)

        if data_list is None:  # 非 Root rank
            return None

        if out is None:
            max_size = int(max(int(bm.max(data.indices)) for data in data_list)) + 1
            result = bm.zeros((max_size,), dtype=array.dtype)
        else:
            result = out

        for data in data_list:
            result = bm.index_add(result, data.indices, data.data)

        return result

    def bcast(self, array: TensorLike | None, /, root: int = 0) -> TensorLike:
        """将 Root 上的全局数组切片广播并分发给各分区的局部实体."""
        if self._global_indices is None:
            raise ValueError("分发数据需要全局实体索引映射.")

        global_indices_list = self._gather(self._global_indices, root=root)

        if global_indices_list is not None:
            assert array is not None, "Root 进程必须提供待广播分发的全局数组."
            data_list = [
                bm.asarray(array[index], copy=True)
                for index in global_indices_list
            ]
        else:
            data_list = None

        return self._scatter(data_list, root=root)


def dist_from_masks(
    masks: Sequence[TensorLike],
    mapping: TensorLike | None = None,
    *,
    comm: Comm | None = None,
) -> EntityMPI:
    """根据各分区的实体掩码列表构建 EntityMPI 通信器.

    参数:
        masks: 各进程的实体布尔掩码序列.
        mapping: 可选的从掩码到划分实体的映射.
        comm: 使用的 MPI 通信子.

    返回:
        当前进程的 EntityMPI 通信器实例.
    """
    comm = comm if comm is not None else COMM_WORLD
    rank = comm.Get_rank()
    thismask = masks[rank]
    indices = bm.nonzero(thismask)[0]
    pairs: list[SharingPair | None] = []

    if mapping is not None:
        masks = mapped_masks(masks=masks, mapping=mapping)

    for i, mask in enumerate(masks):
        if i == rank:
            pairs.append(None)
            continue
        pairs.append(
            SharingPair(
                bm.nonzero(mask[thismask])[0],
                bm.nonzero(thismask[mask])[0],
            )
        )

    return EntityMPI(indices, pairs, comm=comm)


def mapped_masks(
    masks: Sequence[TensorLike],
    mapping: TensorLike,
) -> list[TensorLike]:
    """依据实体连接映射关系将掩码从一种实体类型投影映射至另一种实体类型."""
    num_entity = int(bm.max(mapping)) + 1
    mapped_masks_list: list[TensorLike] = []

    for mask in masks:
        mapped = bm.zeros((num_entity,), dtype=mask.dtype)
        mapped[mapping[mask]] = True
        mapped_masks_list.append(mapped)

    return mapped_masks_list
