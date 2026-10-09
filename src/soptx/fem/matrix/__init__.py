# -*- coding: utf-8 -*-
"""全局稀疏矩阵: 模式先行的 CSR 装配 (符号阶段建 ``CSRPattern``, 数值阶段按槽位累加),
以及 Dirichlet 对称消元."""

from .csr_pattern import (
    CSRChunkAccumulator,
    CSRPattern,
    assemble_csr,
    build_csr_pattern,
)
from .elimination import SymmetricElimination, elimination_slots

__all__ = [
    "CSRChunkAccumulator",
    "CSRPattern",
    "build_csr_pattern",
    "assemble_csr",
    "SymmetricElimination",
    "elimination_slots",
]