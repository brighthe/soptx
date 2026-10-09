# -*- coding: utf-8 -*-
"""全局稀疏矩阵上的 Dirichlet 对称消元.

对称消元把受约束自由度的行与列置零、对角置 1, 右端项的修正由调用方负责. CSR 下取
保结构做法: 不删行列, 复制数值后改写槽位, 共用原骨架; 与删除行列定义同一矩阵, 只多出
显式零. 要改写的槽位只依赖 CSR 骨架与约束掩码, 与数值无关, 因此可跨密度更新复用.

矩阵自由层级下的同一个离散系统由 ``soptx.fem.operators.ConstrainedOperator`` 给出.
"""

from typing import Optional, Tuple, Union

from soptx.backend import backend_manager as bm
from soptx.sparse import COOTensor, CSRTensor
from soptx.typing import TensorLike


def elimination_slots(crow: TensorLike,
                      col: TensorLike,
                      is_dirichlet: TensorLike,
                    ) -> Tuple[TensorLike, TensorLike]:
    """对称消元要改写的 CSR 槽位: 受约束行或列的全部槽位, 及受约束行的对角槽位.

    Parameters
    ----------
    crow : CSR 行指针.
    col : CSR 列索引.
    is_dirichlet : (gdof, ) 的 Dirichlet 自由度布尔掩码.

    Returns
    -------
    zero_slots : 须置零的槽位 (可含重复).
    diag_slots : 受约束行的对角槽位, 须置 1.

    Raises
    ------
    RuntimeError
        某受约束行在稀疏结构中没有对角元.

    Notes
    -----
    受约束行的槽位按行区间展开, 规模只与受约束行数成正比; 受约束列的槽位需扫描一遍 col.
    """
    rows = bm.nonzero(is_dirichlet)[0]
    starts = crow[rows]
    lengths = crow[rows + 1] - starts
    offsets = bm.cumsum(lengths, axis=0) - lengths
    local = bm.arange(int(bm.sum(lengths)), **bm.context(col)) - bm.repeat(offsets, lengths)
    row_slots = bm.repeat(starts, lengths) + local
    row_ids = bm.repeat(rows, lengths)
    diag_slots = row_slots[col[row_slots] == row_ids]
    if int(diag_slots.shape[0]) != int(rows.shape[0]):
        raise RuntimeError("对称消元要求每个受约束行在稀疏结构中含对角元")
    col_slots = bm.nonzero(is_dirichlet[col])[0]
    zero_slots = bm.concat([row_slots, col_slots], axis=0)

    return zero_slots, diag_slots


class SymmetricElimination:
    """Dirichlet 对称消元, CSR 下保结构并缓存要改写的槽位.

    缓存以 CSR 骨架 (crow, col) 的对象身份与约束掩码为键: 骨架为同一对象且掩码不变时
    复用上次算好的槽位. 每个实例只记最近一份, 不同骨架的矩阵 (如主算子与另行装配的
    预条件矩阵) 应各用一个实例, 以免互相挤掉.
    """

    def __init__(self) -> None:
        self._cache: Optional[tuple] = None

    def apply(self,
              matrix: Union[CSRTensor, COOTensor],
              is_dirichlet: TensorLike,
            ) -> Union[CSRTensor, COOTensor]:
        """只对左端矩阵施加 Dirichlet 边界条件, 不改写原矩阵.

        Parameters
        ----------
        matrix : 线性系统原始的左端稀疏矩阵.
        is_dirichlet : (gdof, ) 的布尔掩码, 标记 Dirichlet 自由度.

        Returns
        -------
        A : 施加边界条件后的左端稀疏矩阵. CSR 下与原矩阵共用骨架; COO 下删去受约束行列
            的非零元后补上单位对角; 其他类型原样返回.
        """
        if isinstance(matrix, CSRTensor):
            zero_slots, diag_slots = self._slots(matrix, is_dirichlet)
            values = bm.copy(matrix.values)
            values = bm.set_at(values, zero_slots, 0.0)
            values = bm.set_at(values, diag_slots, 1.0)
            return CSRTensor(matrix.crow, matrix.col, values, matrix.sparse_shape)

        if isinstance(matrix, COOTensor):
            kwargs = matrix.values_context()
            indices = matrix.indices
            remove_flag = bm.logical_or(
                is_dirichlet[indices[0, :]], is_dirichlet[indices[1, :]]
            )
            retain_flag = bm.logical_not(remove_flag)
            new_indices = indices[:, retain_flag]
            new_values = matrix.values[..., retain_flag]
            A = COOTensor(new_indices, new_values, matrix.sparse_shape)

            index = bm.nonzero(is_dirichlet)[0]
            shape = new_values.shape[:-1] + (len(index), )
            one_values = bm.ones(shape, **kwargs)
            one_indices = bm.stack([index, index], axis=0)
            A1 = COOTensor(one_indices, one_values, matrix.sparse_shape)
            return A.add(A1).coalesce()

        return matrix

    def _slots(self, matrix: CSRTensor, is_dirichlet: TensorLike) -> Tuple[TensorLike, TensorLike]:
        """取缓存的槽位; 骨架或掩码变了就重算"""
        cache = self._cache
        if (cache is not None and cache[0] is matrix.crow and cache[1] is matrix.col
                and bool(bm.all(cache[2] == is_dirichlet))):
            return cache[3], cache[4]

        zero_slots, diag_slots = elimination_slots(matrix.crow, matrix.col, is_dirichlet)
        self._cache = (matrix.crow, matrix.col, bm.copy(is_dirichlet), zero_slots, diag_slots)

        return zero_slots, diag_slots
