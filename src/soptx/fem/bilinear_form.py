# -*- coding: utf-8 -*-
"""SOPTX 高性能双线性型 (BilinearForm) 模块.

本模块提供自主可控的有限元双线性型门面 ``BilinearForm``:
1. 继承并兼容 FEALPy ``BilinearForm`` 的全部数学形式与积分子容器接口;
2. 重构 ``assembly()`` 逻辑, 默认采用 SOPTX 模式先行 (``CSRPattern``) 装配内核:
   - 符号阶段: 自动惰性构建并复用 CSR 拓扑骨架与槽位映射;
   - 数值阶段: 利用 CPU ``np.add.at`` 或 GPU ``scatter_add_`` 原地原子累加;
   - 性能收益: 彻底消灭 FEALPy 传统 COO 排序瓶颈;
3. 保持 EA (无矩阵算子) 下 ``@`` 矩阵向量乘接口完全不变.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Union, overload

from fealpy.fem.bilinear_form import BilinearForm as FEALPyBilinearForm
from fealpy.sparse import COOTensor, CSRTensor
from fealpy.typing import TensorLike

from soptx.fem.matrix.csr_pattern import CSRPattern, assemble_csr, build_csr_pattern


class BilinearForm(FEALPyBilinearForm):
    """SOPTX 高性能有限元双线性型门面.

    Parameters:
        space: 试验/测试有限元空间 (如 ``TensorFunctionSpace`` 或 ``LagrangeFESpace``).
        integrators: 可选的初始积分子列表.
        pattern: 可选的预构建 ``CSRPattern`` 对象. 若为 ``None``, 首次装配时自动构建并缓存.
    """

    def __init__(
        self,
        space: Any,
        *args: Any,
        pattern: Optional[CSRPattern] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(space, *args, **kwargs)
        self._pattern: Optional[CSRPattern] = pattern

    @property
    def pattern(self) -> Optional[CSRPattern]:
        """获取当前绑定的 CSR 拓扑模式对象."""
        return self._pattern

    @pattern.setter
    def pattern(self, value: Optional[CSRPattern]) -> None:
        """设置或更新 CSR 拓扑模式对象."""
        self._pattern = value

    @overload
    def assembly(self) -> CSRTensor: ...
    @overload
    def assembly(self, *, format: Literal["csr"], method: str = "pattern") -> CSRTensor: ...
    @overload
    def assembly(self, *, format: Literal["coo"], method: str = "pattern") -> COOTensor: ...
    def assembly(
        self,
        *,
        format: Literal["csr", "coo"] = "csr",
        method: str = "pattern",
    ) -> Union[CSRTensor, COOTensor]:
        """装配全局刚度矩阵 (Full Assembly).

        Parameters:
            format: 产出矩阵格式 ('csr' | 'coo'). 默认为 'csr'.
            method: 装配路线 ('pattern' | 'coalesce'). 默认为高性能模式先行 'pattern'.

        Returns:
            global_matrix: 装配完成的 FEALPy ``CSRTensor`` 或 ``COOTensor`` 稀疏矩阵.
        """
        if method == "coalesce":
            return super().assembly(format=format)

        if method != "pattern":
            raise ValueError(f"Unsupported assembly method: {method!r}, must be 'pattern' or 'coalesce'.")

        # 1. 符号阶段: 惰性提取并缓存 CSR 模式
        if self._pattern is None:
            self._pattern = build_csr_pattern(self._spaces[0])

        # 2. 累加所有单元局部刚度张量
        K_e = None
        for group_tensor, _ in self.assembly_local_iterative():
            if K_e is None:
                K_e = group_tensor
            else:
                K_e = K_e + group_tensor

        if K_e is None:
            raise RuntimeError("BilinearForm 中未添加任何有效的积分子 (Integrator).")

        # 3. 数值阶段: 原地原子累加装配
        K_csr = assemble_csr(K_e, self._pattern)

        if getattr(self, "_transposed", False):
            K_csr = K_csr.T

        self._M = K_csr

        if format == "coo":
            return K_csr.tocoo()
        elif format == "csr":
            return K_csr
        else:
            raise ValueError(f"Unsupported format: {format!r}, must be 'csr' or 'coo'.")