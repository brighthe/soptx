# 移植自 brighthe/fealpy ``fealpy/mesh/storage/relation.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Canonical directed relations between entity sectors.

An ordered ``(src_sector_id, tgt_sector_id)`` pair identifies exactly one
canonical :class:`EntityRelation` within a :class:`MeshBlock`.  ``Relation``
is a private in-repository compatibility alias for migration convenience; new
code must use :class:`EntityRelation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

from ...backend import bm, Tensor

__all__ = ["EntityRelation", "Relation"]


class UniqueResult(NamedTuple):
    first: Tensor
    last: Tensor


class LocalIndexResult(NamedTuple):
    floc: Tensor
    lloc: Tensor


@dataclass(frozen=True)
class EntityRelation:
    """A canonical directed incidence/mapping between two entity sectors.

    Within one ``MeshBlock``, the ordered pair
    ``(src_sector_id, tgt_sector_id)`` identifies exactly one relation.
    Different dense/COO/CSR materializations and construction paths must
    converge to the same canonical relation.

    ``tgt_indices`` maps each source entity to one or more target entities.  A
    homogeneous relation stores a rank-2 target-index tensor and leaves
    ``src_indices`` as ``None``; a heterogeneous relation stores a rank-1 target
    tensor together with a rank-1 ``src_indices`` tensor.

    The reverse ordered pair has a different canonical identity:
    ``(src, tgt) != (tgt, src)``.  If both directions are materialized, the
    inverse must represent the same incidence/transpose relation rather than a
    second independent relation object.

    Raises:
        TypeError: If ``src_sector_id`` or ``tgt_sector_id`` is not a non-empty
            string.
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
        """Return the source and target indices for heterogeneous relations.

        Parameters:
            copy (bool, optional): Whether to copy the indices if already heterogeneous.
                Defaults to True.

        Returns:
            tuple[Tensor, Tensor]: The source and target indices.
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
        if self.src_indices is None:
            return self.tgt_indices
        raise ValueError("Cannot convert heterogeneous relation to array. "
                         "Use `as_coo` or `as_csr` instead.")

    def as_coo(self):
        """Return a sparse COO materialization of this relation."""
        from ...sparse import coo_matrix
        src, tgt = self.heterogeneous_indices()
        data = bm.ones_like(src, dtype=bm.bool)
        return coo_matrix(
            (data, (src, tgt)),
            shape=(bm.max(src) + 1, bm.max(tgt) + 1) # type: ignore
        )

    def as_csr(self):
        """Return a sparse CSR materialization of this relation."""
        return self.as_coo().tocsr()

    def inverse(self) -> "EntityRelation":
        """Create the inverse directed relation.

        The inverse swaps ``src_sector_id`` and ``tgt_sector_id`` and stores a
        heterogeneous source/target index pair describing the same incidence.
        Materializing an inverse does not create a new semantic relation; it
        only exposes the reverse ordered pair of the same canonical relation.
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
        """The first and last target indices for each source index.

        Returns:
            NamedTuple:
            - first: The first target index
            - last: The last target index
        """
        if self.src_indices is None:
            return UniqueResult(first=self.tgt_indices[:, 0], last=self.tgt_indices[:, -1])

        arg, diff0, diff1 = self._fl_mask
        tgt = self.tgt_indices[arg]
        return UniqueResult(first=tgt[diff0], last=tgt[diff1])

    @cached_property
    def local_index(self) -> LocalIndexResult:
        """The local indices of the source indices for each target index.

        Returns:
            NamedTuple:
            - floc: Local index in the first target
            - lloc: Local index in the last target
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


# Private compatibility alias.  New code must use ``EntityRelation``; this name
# is retained only for in-repository callers that have not completed migration.
Relation = EntityRelation
