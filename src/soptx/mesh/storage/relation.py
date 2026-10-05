# 移植自 brighthe/fealpy ``fealpy/mesh/storage/relation.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""实体分区之间的规范有向关系.

在一个 :class:`MeshBlock` 内, 有序对 ``(src_sector_id, tgt_sector_id)`` 恰好对应
一条规范的 :class:`EntityRelation`. ``Relation`` 是为便于迁移而保留的仓库内私有
兼容别名, 新代码须使用 :class:`EntityRelation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

from ...backend import bm, Tensor

__all__ = ["EntityRelation", "Relation"]


class UniqueResult(NamedTuple):
    """每个源实体对应的第一个与最后一个目标实体."""
    first: Tensor
    last: Tensor


class LocalIndexResult(NamedTuple):
    """源实体在第一个与最后一个目标实体中的局部序号."""
    floc: Tensor
    lloc: Tensor


@dataclass(frozen=True)
class EntityRelation:
    """两个实体分区之间规范的有向关联 (映射).

    在一个 ``MeshBlock`` 内, 有序对 ``(src_sector_id, tgt_sector_id)`` 恰好对应
    一条关系; 稠密、COO、CSR 等不同的实体化方式与构造路径都须归结为同一条规范关系.

    ``tgt_indices`` 把每个源实体映射到一个或多个目标实体. 同类关系存二维的目标
    索引张量, ``src_indices`` 为 None; 异类关系存一维的目标张量与一维的
    ``src_indices``.

    反向的有序对是另一个规范身份: ``(src, tgt) != (tgt, src)``. 两个方向都实体化
    时, 逆关系须表示同一个关联 (转置), 而不是另一个独立的关系对象.

    Raises
    ------
    TypeError
        ``src_sector_id`` 或 ``tgt_sector_id`` 不是非空字符串.
    """

    src_sector_id: str
    tgt_sector_id: str
    tgt_indices: Tensor
    src_indices: Tensor | None = None

    def __post_init__(self) -> None:
        if type(self.src_sector_id) is not str or not self.src_sector_id:
            raise TypeError(
                "EntityRelation.src_sector_id must be a non-empty string"
            )
        if type(self.tgt_sector_id) is not str or not self.tgt_sector_id:
            raise TypeError(
                "EntityRelation.tgt_sector_id must be a non-empty string"
            )

    def heterogeneous_indices(self, copy: bool = True) -> tuple[Tensor, Tensor]:
        """以异类形式 (一维源索引与一维目标索引) 返回关系.

        Parameters
        ----------
        copy : bool, optional
            已是异类关系时是否复制索引. 默认 True.

        Returns
        -------
        tuple of Tensor
            ``(源索引, 目标索引)``.
        """
        if self.src_indices is None:
            src = bm.arange(self.tgt_indices.shape[0], dtype=bm.int32)
            src = bm.repeat(src, self.tgt_indices.shape[1])
            tgt = bm.reshape(bm.copy(self.tgt_indices), (-1,))
        else:
            if copy:
                src = bm.copy(self.src_indices)
                tgt = bm.copy(self.tgt_indices)
            else:
                src = self.src_indices
                tgt = self.tgt_indices

        return src, tgt

    def as_array(self) -> Tensor:
        """同类关系返回二维目标索引张量.

        Raises
        ------
        ValueError
            异类关系无法表示为数组, 应改用 ``as_coo`` 或 ``as_csr``.
        """
        if self.src_indices is None:
            return self.tgt_indices
        raise ValueError("Cannot convert heterogeneous relation to array. "
                         "Use `as_coo` or `as_csr` instead.")

    def as_coo(self):
        """以布尔值 COO 稀疏矩阵实体化本关系, 形状由最大索引推断."""
        from ...sparse import coo_matrix
        src, tgt = self.heterogeneous_indices()
        data = bm.ones_like(src, dtype=bm.bool)
        return coo_matrix(
            (data, (src, tgt)),
            shape=(bm.max(src) + 1, bm.max(tgt) + 1) # type: ignore
        )

    def as_csr(self):
        """以 CSR 稀疏矩阵实体化本关系."""
        return self.as_coo().tocsr()

    def inverse(self) -> "EntityRelation":
        """构造逆向关系.

        逆关系交换 ``src_sector_id`` 与 ``tgt_sector_id``, 以异类的源/目标索引对
        描述同一个关联. 实体化逆关系并不产生新的语义关系, 只是暴露同一规范关系的
        反向有序对.
        """
        src, tgt = self.heterogeneous_indices()
        return EntityRelation(
            src_sector_id=self.tgt_sector_id,
            tgt_sector_id=self.src_sector_id,
            src_indices=tgt,
            tgt_indices=src,
        )

    @cached_property
    def _fl_mask(self):
        assert self.src_indices is not None
        arg = bm.argsort(self.src_indices)
        src = self.src_indices[arg]
        TRUE = bm.ones((1,), dtype=bm.bool, device=src.device)
        diff = src[1:] != src[:-1]
        diff0 = bm.concat([TRUE, diff])
        diff1 = bm.concat([diff, TRUE])
        return arg, diff0, diff1

    @cached_property
    def unique(self) -> UniqueResult:
        """每个源实体对应的第一个与最后一个目标实体.

        Returns
        -------
        UniqueResult
            ``first`` 为第一个目标实体, ``last`` 为最后一个目标实体.
        """
        if self.src_indices is None:
            return UniqueResult(first=self.tgt_indices[:, 0], last=self.tgt_indices[:, -1])

        arg, diff0, diff1 = self._fl_mask
        tgt = self.tgt_indices[arg]
        return UniqueResult(first=tgt[diff0], last=tgt[diff1])

    @cached_property
    def local_index(self) -> LocalIndexResult:
        """源实体在其第一个与最后一个目标实体中的局部序号.

        Returns
        -------
        LocalIndexResult
            ``floc`` 为在第一个目标中的局部序号, ``lloc`` 为在最后一个目标中的局部序号.
        """
        if self.src_indices is None:
            count = self.tgt_indices.shape[0]
            width = self.tgt_indices.shape[1]
            floc = bm.zeros((count,), dtype=bm.int32, device=self.tgt_indices.device)
            lloc = bm.full((count,), width - 1, dtype=bm.int32, device=self.tgt_indices.device)
            return LocalIndexResult(floc=floc, lloc=lloc)

        order = bm.argsort(self.tgt_indices)
        tgt = self.tgt_indices[order]
        idx = bm.arange(tgt.shape[0], dtype=bm.int32, device=tgt.device)
        TRUE = bm.ones((1,), dtype=bm.bool, device=tgt.device)
        diff = bm.concat([TRUE, tgt[1:] != tgt[:-1]])
        group = bm.cumsum(bm.astype(diff, bm.int32), axis=0) - 1
        group_start = idx[diff]
        loc = idx - group_start[group]
        loc = loc[bm.argsort(order)]

        arg, diff0, diff1 = self._fl_mask
        loc = loc[arg]

        return LocalIndexResult(floc=loc[diff0], lloc=loc[diff1])


# 私有兼容别名. 新代码须使用 ``EntityRelation``; 保留此名只为尚未完成迁移的仓库内调用方.
Relation = EntityRelation
