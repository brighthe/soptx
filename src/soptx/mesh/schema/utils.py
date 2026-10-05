# 移植自 brighthe/fealpy ``fealpy/mesh/schema/utils.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Schema 的数组工具."""

from ...backend import bm, Tensor, dtype


def argpermute(src: Tensor, tgt: Tensor, /, *, dtype: dtype | None = None) -> Tensor:
    """返回把源数组置换为目标数组的下标.

    源数组与目标数组须同形状、含相同的元素, 顺序可以不同. 多维时置换作用于源数组的
    最后一维.
    """
    # 两个数组都沿最后一维排序, 再把目标的每个位置映射到排序后同一名次的源下标.
    src_sorted_to_src = bm.argsort(src, axis=-1, stable=True)

    if dtype is not None:
        src_sorted_to_src = bm.asarray(src_sorted_to_src, dtype=dtype)

    tgt_sorted_to_tgt = bm.argsort(tgt, axis=-1, stable=True)
    tgt_to_tgt_sorted = bm.argsort(tgt_sorted_to_tgt, axis=-1, stable=True)

    del tgt_sorted_to_tgt

    if dtype is not None:
        tgt_to_tgt_sorted = bm.asarray(tgt_to_tgt_sorted, dtype=dtype)

    return bm.take_along_axis(src_sorted_to_src, tgt_to_tgt_sorted, axis=-1)
