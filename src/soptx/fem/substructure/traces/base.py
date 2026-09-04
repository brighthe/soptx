"""子结构接口迹空间的统一线性代数契约."""

from __future__ import annotations

from typing import Any

from fealpy.backend import backend_manager as bm


class TraceBasis:
    """由迹自由度到完整接口自由度的线性映射.

    设 ``T`` 的形状为 ``(n_boundary, n_trace)``, 迹自由度 ``q`` 与完整接口
    位移 ``u_b`` 满足 ``u_b = T q``. 完整接口缩聚刚度 ``K_s`` 在该迹空间上
    的表示为 ``K_q = T^T K_s T``.

    子类只负责构造 ``T`` 并声明稳定的 ``name``; 刚度降阶、恢复矩阵降阶和
    位移展开由本类统一实现, 避免各条 Exact/PIML 路径重复写矩阵公式.
    """

    name = "custom"

    def __init__(self, matrix: Any) -> None:
        """保存并校验接口迹矩阵.

        参数:
            matrix: 迹插值矩阵 ``T``, 形状 ``(n_boundary, n_trace)``.

        异常:
            ValueError: 当 ``matrix`` 不是非空二维矩阵时抛出.
        """
        value = bm.asarray(matrix)
        if value.ndim != 2:
            raise ValueError(
                f"TraceBasis.matrix 必须是二维矩阵; 当前 ndim={value.ndim}."
            )
        if value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError(
                "TraceBasis.matrix 的边界自由度数和迹自由度数必须均为正."
            )
        self._matrix = value

    @property
    def matrix(self) -> Any:
        """返回迹插值矩阵 ``T``."""
        return self._matrix

    @property
    def n_boundary_dofs(self) -> int:
        """完整接口自由度数 ``n_boundary``."""
        return int(self._matrix.shape[0])

    @property
    def n_trace_dofs(self) -> int:
        """迹自由度数 ``n_trace``."""
        return int(self._matrix.shape[1])

    def _check_boundary_axis(self, value: Any, label: str) -> None:
        """校验张量末维与完整接口自由度数一致."""
        if value.shape[-1] != self.n_boundary_dofs:
            raise ValueError(
                f"{label} 的接口自由度数必须为 {self.n_boundary_dofs}; "
                f"当前形状为 {tuple(value.shape)}."
            )

    def project_stiffness(self, stiffness: Any) -> Any:
        """计算 ``K_q = T^T K_s T``.

        参数:
            stiffness: 完整接口缩聚刚度, 形状
                ``(..., n_boundary, n_boundary)``.

        返回:
            迹空间刚度, 形状 ``(..., n_trace, n_trace)``.
        """
        value = bm.asarray(stiffness)
        if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
            raise ValueError(
                "stiffness 必须具有形状 (..., n_boundary, n_boundary)."
            )
        self._check_boundary_axis(value, "stiffness")
        return (
            bm.matrix_transpose(self._matrix)
            @ value
            @ self._matrix
        )

    def reduce_recovery(self, recovery: Any) -> Any:
        """把完整接口恢复矩阵 ``N`` 变换为 ``N T``.

        参数:
            recovery: 内部位移恢复矩阵, 形状
                ``(..., n_internal, n_boundary)``.
        """
        value = bm.asarray(recovery)
        if value.ndim < 2:
            raise ValueError(
                "recovery 必须具有形状 (..., n_internal, n_boundary)."
            )
        self._check_boundary_axis(value, "recovery")
        return value @ self._matrix

    def expand_displacement(self, trace_displacement: Any) -> Any:
        """计算完整接口位移 ``u_b = T q``.

        参数:
            trace_displacement: 迹自由度位移, 形状 ``(..., n_trace)``.
        """
        value = bm.asarray(trace_displacement)
        if value.ndim < 1 or value.shape[-1] != self.n_trace_dofs:
            raise ValueError(
                f"trace_displacement 的末维必须为 {self.n_trace_dofs}; "
                f"当前形状为 {tuple(value.shape)}."
            )
        return value @ bm.matrix_transpose(self._matrix)
