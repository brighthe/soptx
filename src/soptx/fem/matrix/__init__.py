# -*- coding: utf-8 -*-
"""模式先行的 CSR 全局矩阵装配: 符号阶段建 ``CSRPattern``, 数值阶段按槽位累加."""

from .csr_pattern import (
    CSRChunkAccumulator,
    CSRPattern,
    assemble_csr,
    build_csr_pattern,
)

__all__ = [
    "CSRChunkAccumulator",
    "CSRPattern",
    "build_csr_pattern",
    "assemble_csr",
]