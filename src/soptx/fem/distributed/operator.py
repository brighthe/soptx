"""重叠副本分布式算子 (Overlap Operator) 模块.

将局部微分算子提升为重叠副本布局下的全局算子, 执行 S ∘ K_loc ∘ C 三步并行流水线.
"""

from __future__ import annotations

__all__ = [
    "OverlapOperator",
]

import time
from typing import Any

from fealpy.typing import TensorLike

from .entity_mpi import EntityMPI


class OverlapOperator:
    """把局部算子提升为重叠副本布局下的全局算子.

    ``__matmul__`` 的三步执行流水线为:
    :math:`\\mathcal S \\circ K_{\\mathrm{loc}} \\circ \\mathcal C`:
    1. 投影 C: 先把加和表示的输入投影成一致表示 (sync_add / refs);
    2. 局部作用 K_loc: 本地独立计算局部算子矩阵-向量乘;
    3. 同步 S: 再把结果同步归约回加和表示 (sync_add).

    单 rank 下 ``refs`` 恒为 1 且 ``sync_add`` 是恒等映射, 因此串行与并行复用同一套无分支代码.
    ``__getattr__`` 转发未定义属性, 使被包装的算子在调用方看来与未包装时一致.
    """

    def __init__(self, local_operator: Any, dof_comm: EntityMPI) -> None:
        self.local_operator = local_operator
        self.dof_comm = dof_comm
        self._profiling_enabled = False
        self._profile_calls = 0
        self._profile_input_sync_seconds = 0.0
        self._profile_local_kernel_seconds = 0.0
        self._profile_output_sync_seconds = 0.0

    def enable_profiling(self, enabled: bool = True) -> None:
        """启用或关闭 MatVec 内部阶段计时.

        默认关闭, 避免正确性验证和正常求解路径承担计时器开销. 启用后仅累计本实例
        ``__matmul__`` 中的本地 wall time; 跨 rank 的关键路径由调用方以 ``MPI.MAX``
        归约这些统计量.
        """
        self._profiling_enabled = enabled
        self.reset_profile()

    def reset_profile(self) -> None:
        """清空已累计的 MatVec 内部阶段计时."""
        self._profile_calls = 0
        self._profile_input_sync_seconds = 0.0
        self._profile_local_kernel_seconds = 0.0
        self._profile_output_sync_seconds = 0.0

    def profile(self) -> dict[str, float | int]:
        """返回自上次清空以来累计的 MatVec 内部阶段计时.

        返回:
            dict[str, float | int]: ``calls`` 为 MatVec 次数; 其余字段分别是输入一致化
            同步、本地单元核和输出求和同步的本地累计 wall time, 单位为秒.
        """
        return {
            "calls": self._profile_calls,
            "input_sync_seconds": self._profile_input_sync_seconds,
            "local_kernel_seconds": self._profile_local_kernel_seconds,
            "output_sync_seconds": self._profile_output_sync_seconds,
        }

    def __matmul__(self, vector: TensorLike) -> TensorLike:
        if not self._profiling_enabled:
            references = self.dof_comm.refs(vector.shape[-1])
            consistent_vector = self.dof_comm.sync_add(vector) / references
            local_result = self.local_operator @ consistent_vector
            return self.dof_comm.sync_add(local_result)

        start = time.perf_counter()
        references = self.dof_comm.refs(vector.shape[-1])
        consistent_vector = self.dof_comm.sync_add(vector) / references
        input_sync_seconds = time.perf_counter() - start

        start = time.perf_counter()
        local_result = self.local_operator @ consistent_vector
        local_kernel_seconds = time.perf_counter() - start

        start = time.perf_counter()
        result = self.dof_comm.sync_add(local_result)
        output_sync_seconds = time.perf_counter() - start

        self._profile_calls += 1
        self._profile_input_sync_seconds += input_sync_seconds
        self._profile_local_kernel_seconds += local_kernel_seconds
        self._profile_output_sync_seconds += output_sync_seconds
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.local_operator, name)
